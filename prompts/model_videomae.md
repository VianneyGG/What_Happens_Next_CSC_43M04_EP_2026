# Model 3 — VideoMAE ViT-L

> **Role:** Tier 3 — primary submission. Self-supervised pretraining + fine-tuning.  
> **Expected Top-1:** 74–77% (likely 80%+ on 50-class subset)  
> **Compute:** 100× A100 for pretraining (~16h), same for fine-tuning (~2h)

---

## Overview

VideoMAE trains a Video Vision Transformer in two completely separate phases:

1. **Pretraining (no labels):** mask 90% of video tokens using tube masking, encode the visible 10%, decode + reconstruct masked pixels. The encoder learns to represent video dynamics purely by solving this reconstruction puzzle.
2. **Fine-tuning (with labels):** discard the decoder. Add a classification head. Fine-tune the encoder end-to-end on labeled clips.

**Why this is legal in Track A (closed world):** we never use external data or pretrained weights. The self-supervised signal comes entirely from the competition's own training videos.

---

## Tokenisation

A clip of shape `(B, 3, T, H, W)` with T=16, H=W=224 is divided into 3D patches:

- Spatial patch size: 16×16 pixels  
- Temporal tube size: 2 frames  

This gives: `(T/2) × (H/16) × (W/16) = 8 × 14 × 14 = 1568 tokens` per clip.

Each token is a flat vector of `2 × 16 × 16 × 3 = 1536` raw pixel values, projected to the model dimension (1024 for ViT-L) by a linear embedding layer.

---

## Tube Masking — The Core Innovation

Standard random masking lets the model "cheat" on video: a masked patch at time t can be inferred by copying from time t-1 or t+1, since adjacent frames are nearly identical. This makes the task trivial and the learned representations weak.

**Tube masking** prevents this: if a spatial patch (h, w) is selected for masking, it is masked at **all** timesteps. The model can no longer copy across time — it must reason about motion to reconstruct what was there.

```python
def tube_mask(num_tokens_per_frame, num_frames, mask_ratio=0.90):
    """
    Returns mask indices for a single clip.
    Masks entire temporal tubes — same spatial location across all frames.
    """
    num_spatial = num_tokens_per_frame          # 14 × 14 = 196
    num_mask_spatial = int(num_spatial * mask_ratio)  # 176

    # Sample which spatial positions to mask (same for all frames)
    noise = torch.rand(num_spatial)
    ids_shuffle = noise.argsort()
    masked_spatial = ids_shuffle[:num_mask_spatial]  # (176,)

    # Expand to all frames → flat token indices
    masked_ids = []
    for t in range(num_frames):
        masked_ids.append(masked_spatial + t * num_spatial)
    return torch.cat(masked_ids)  # (176 × 8 = 1408 masked tokens)

# Corresponding visible ids (157 tokens)
def visible_ids(masked_ids, total_tokens=1568):
    all_ids = torch.arange(total_tokens)
    mask = torch.ones(total_tokens, dtype=torch.bool)
    mask[masked_ids] = False
    return all_ids[mask]
```

---

## Architecture

### ViT-L encoder

```python
import torch
import torch.nn as nn
import math

class PatchEmbed3D(nn.Module):
    """3D patch embedding: video → tokens."""
    def __init__(self, temporal_patch=2, spatial_patch=16, in_channels=3, embed_dim=1024):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(temporal_patch, spatial_patch, spatial_patch),
            stride=(temporal_patch, spatial_patch, spatial_patch),
        )

    def forward(self, x):
        # x: (B, 3, T, H, W)
        x = self.proj(x)             # (B, D, T/2, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop),
        )
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTEncoder(nn.Module):
    """ViT-L: 24 layers, dim=1024, heads=16."""
    def __init__(self, num_tokens=1568, embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., drop_path_rate=0.2):
        super().__init__()
        self.patch_embed = PatchEmbed3D(embed_dim=embed_dim)
        # Learned positional embeddings (one per token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, visible_ids):
        # x: (B, 3, T, H, W)
        tokens = self.patch_embed(x)                 # (B, 1568, 1024)
        tokens = tokens + self.pos_embed             # add positional embeddings
        # Keep only visible tokens (pretraining) or all tokens (fine-tuning)
        if visible_ids is not None:
            tokens = tokens[:, visible_ids, :]       # (B, ~157, 1024)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)
```

### Lightweight decoder (pretraining only)

The decoder is a shallow 4-block ViT with smaller dimension (512). It takes encoder output at visible positions + learnable mask tokens at masked positions, and reconstructs raw pixel values.

```python
class VideoMAEDecoder(nn.Module):
    def __init__(self, encoder_dim=1024, decoder_dim=512, depth=4, num_heads=16,
                 num_patches=1568, patch_size=16, temporal_patch=2, num_channels=3):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)

        self.encoder_proj = nn.Linear(encoder_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.blocks = nn.ModuleList([
            ViTBlock(decoder_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)

        # Output: flat pixel values per masked patch
        patch_dim = temporal_patch * patch_size * patch_size * num_channels  # 1536
        self.pred_head = nn.Linear(decoder_dim, patch_dim)

    def forward(self, encoder_out, visible_ids, masked_ids, total_tokens=1568):
        B = encoder_out.shape[0]
        D = encoder_out.shape[-1]

        # Project encoder tokens to decoder dim
        vis_tokens = self.encoder_proj(encoder_out)  # (B, vis, decoder_dim)

        # Build full token sequence: visible + mask tokens
        full_tokens = self.mask_token.expand(B, total_tokens, -1).clone()
        full_tokens[:, visible_ids, :] = vis_tokens

        # Add positional embeddings
        full_tokens = full_tokens + self.pos_embed

        for block in self.blocks:
            full_tokens = block(full_tokens)
        full_tokens = self.norm(full_tokens)

        # Predict only the masked positions
        pred = self.pred_head(full_tokens[:, masked_ids, :])  # (B, num_masked, 1536)
        return pred
```

### Full VideoMAE model (pretraining)

```python
class VideoMAE(nn.Module):
    def __init__(self, mask_ratio=0.90, num_frames=16):
        super().__init__()
        self.mask_ratio  = mask_ratio
        self.num_frames  = num_frames
        self.num_spatial = 14 * 14   # 196 patches per frame
        self.total       = (num_frames // 2) * self.num_spatial  # 1568

        self.encoder = ViTEncoder()
        self.decoder = VideoMAEDecoder()

    def patchify(self, x):
        """Extract raw pixel values for each 3D patch — used as reconstruction target."""
        # x: (B, 3, T, H, W)
        B, C, T, H, W = x.shape
        tp, sp = 2, 16  # temporal and spatial patch sizes
        x = x.reshape(B, C, T//tp, tp, H//sp, sp, W//sp, sp)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)  # (B, T', H', W', tp, sp, sp, C)
        x = x.reshape(B, -1, tp * sp * sp * C)  # (B, 1568, 1536)
        return x

    def forward(self, x):
        # x: (B, 3, T, H, W)
        B = x.shape[0]

        # Generate tube mask (same across the whole batch shape, but random per sample)
        num_frames_half = self.num_frames // 2  # 8
        mask_ids   = tube_mask(self.num_spatial, num_frames_half, self.mask_ratio)
        vis_ids    = visible_ids(mask_ids, self.total)

        # Encode visible tokens
        enc_out = self.encoder(x, vis_ids)   # (B, ~157, 1024)

        # Decode + reconstruct masked tokens
        pred = self.decoder(enc_out, vis_ids, mask_ids, self.total)  # (B, 1408, 1536)

        # Target: raw pixel values at masked positions
        target = self.patchify(x)            # (B, 1568, 1536)
        target_masked = target[:, mask_ids]  # (B, 1408, 1536)

        loss = F.mse_loss(pred, target_masked)
        return loss
```

---

## Phase 1 — Pretraining

```python
def pretrain_videomae(config):
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    model = VideoMAE(mask_ratio=0.90, num_frames=16).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Unlabeled dataset: same videos, no labels needed
    # Use the full training set (labels are simply ignored)
    train_dataset = SSv2UnlabeledDataset(config.video_dir, pretrain_transform,
                                         num_frames=16)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_per_gpu,
                               sampler=train_sampler, num_workers=8, pin_memory=True)

    # Scale LR linearly with effective batch size
    eff_batch = config.batch_per_gpu * dist.get_world_size()
    lr = config.base_lr * eff_batch / 256

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=0.05, betas=(0.9, 0.95))

    def lr_lambda(step):
        total_steps = config.epochs * len(train_loader)
        warmup_steps = config.warmup_epochs * len(train_loader)
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler()

    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        for step, clips in enumerate(train_loader):
            clips = clips.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model(clips)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.02)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        if local_rank == 0 and epoch % 100 == 0:
            # Save encoder weights only
            torch.save(model.module.encoder.state_dict(),
                       f"videomae_encoder_epoch{epoch}.pth")
```

### Pretraining augmentation

During pretraining, use heavier spatial augmentation but **no** temporal augmentation (temporal order is the signal we're trying to learn):

```python
pretrain_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),  # aggressive spatial crop
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

---

## Phase 2 — Fine-tuning

After pretraining, the decoder is discarded. The encoder weights are loaded and a classification head is added on top:

```python
class VideoMAEClassifier(nn.Module):
    def __init__(self, encoder_path, num_classes=50, num_frames=16):
        super().__init__()
        self.encoder = ViTEncoder()
        # Load pretrained encoder weights
        state = torch.load(encoder_path, map_location='cpu')
        self.encoder.load_state_dict(state)

        # Classification head: global average pool + dropout + linear
        self.head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)        # → (B, C, T, H, W)
        # No masking during fine-tuning: visible_ids=None → all tokens
        enc = self.encoder(x, visible_ids=None)  # (B, 1568, 1024)
        pooled = enc.mean(dim=1)             # global average pool → (B, 1024)
        return self.head(pooled)
```

### Fine-tuning training loop

```python
def finetune_videomae(config):
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    model = VideoMAEClassifier(
        encoder_path=config.pretrained_encoder,
        num_classes=50,
    ).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    train_dataset = SSv2Dataset(config.train_ann, config.video_dir,
                                train_transform, num_frames=16)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_per_gpu,
                               sampler=train_sampler, num_workers=8, pin_memory=True)

    # Layer-wise LR decay: deeper layers get smaller LR
    # Encoder layers: 0.75× LR per block (common in VideoMAE fine-tuning)
    def build_param_groups(model, base_lr, layer_decay=0.75, weight_decay=0.05):
        param_groups = []
        num_layers = 24  # ViT-L depth

        # Classification head — full LR
        param_groups.append({
            'params': list(model.module.head.parameters()),
            'lr': base_lr, 'weight_decay': weight_decay,
        })

        # Encoder blocks — decayed LR per layer
        for i, block in enumerate(model.module.encoder.blocks):
            layer_lr = base_lr * (layer_decay ** (num_layers - i))
            param_groups.append({
                'params': list(block.parameters()),
                'lr': layer_lr, 'weight_decay': weight_decay,
            })

        # Patch embedding and pos embed — smallest LR
        param_groups.append({
            'params': [model.module.encoder.pos_embed,
                       *model.module.encoder.patch_embed.parameters()],
            'lr': base_lr * (layer_decay ** num_layers),
            'weight_decay': 0.,  # no weight decay on embedding/bias
        })
        return param_groups

    param_groups = build_param_groups(model, base_lr=config.lr)
    optimizer  = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return epoch / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler     = torch.cuda.amp.GradScaler()

    best_acc = 0.
    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        for clips, labels in train_loader:
            clips  = clips.cuda(local_rank, non_blocking=True)
            labels = labels.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(clips)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        if local_rank == 0:
            acc = evaluate(model, val_loader, local_rank)
            print(f"Epoch {epoch+1}/{config.epochs} | Top-1: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.module.state_dict(), "videomae_best.pth")
```

---

## Launch Commands

### Pretraining (100 nodes, 1 GPU each)

```bash
# On each node, run:
torchrun --nproc_per_node=1 \
         --nnodes=100 \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=29502 \
         pretrain_videomae.py
```

### Fine-tuning (same setup)

```bash
torchrun --nproc_per_node=1 \
         --nnodes=100 \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=29503 \
         finetune_videomae.py --pretrained encoder_epoch800.pth
```

---

## Configs

```python
@dataclass
class PretrainConfig:
    video_dir:      str   = "data/videos"
    num_frames:     int   = 16
    mask_ratio:     float = 0.90

    epochs:         int   = 800
    batch_per_gpu:  int   = 32     # small batch fine for MAE
    base_lr:        float = 1.5e-4
    warmup_epochs:  int   = 40
    weight_decay:   float = 0.05

@dataclass
class FinetuneConfig:
    train_ann:         str   = "data/train.csv"
    val_ann:           str   = "data/val.csv"
    video_dir:         str   = "data/videos"
    pretrained_encoder: str  = "encoder_epoch800.pth"
    num_frames:        int   = 16

    epochs:            int   = 50
    batch_per_gpu:     int   = 8
    lr:                float = 1e-3
    layer_decay:       float = 0.75   # LR decay per encoder layer
    warmup_epochs:     int   = 5
    weight_decay:      float = 0.05
    label_smoothing:   float = 0.1
    dropout_head:      float = 0.5
```

---

## Test-Time Augmentation (TTA)

At inference, average predictions from multiple views:

```python
def predict_with_tta(model, video_path, num_spatial_crops=3, num_temporal_clips=2):
    """
    Standard VideoMAE inference: 3 spatial crops × 2 temporal clips = 6 views.
    """
    all_probs = []
    for t_clip in range(num_temporal_clips):
        frames = load_frames_clip(video_path, clip_idx=t_clip)  # different temporal offset
        for s_crop in ['left', 'center', 'right']:
            clip = apply_spatial_crop(frames, s_crop)
            with torch.no_grad(), torch.cuda.amp.autocast():
                logits = model(clip.unsqueeze(0).cuda())
                all_probs.append(torch.softmax(logits, dim=-1))

    return torch.stack(all_probs).mean(0).argmax(dim=-1)
```

TTA typically gives **+0.5–1.5%** on SSv2.

---

## Expected Results

| Phase | Checkpoint | Val Top-1 (est.) |
|-------|-----------|-----------------|
| Pretrain only (frozen encoder probe) | epoch 400 | ~55–60% |
| Pretrain only (frozen encoder probe) | epoch 800 | ~62–65% |
| Fine-tuned | epoch 25 | ~72–74% |
| Fine-tuned (best) | epoch 50 | **~75–80%** |
| Fine-tuned + TTA | — | **~77–82%** |

---

## Key Failure Modes

- **NaN loss during pretraining:** gradient explosion — reduce `clip_grad_norm` from 0.02 to 0.01, or lower the base LR.
- **Underfitting in fine-tuning:** if val accuracy plateaus early, reduce `layer_decay` to 0.65 (gives deeper layers more LR) or increase fine-tuning epochs to 100.
- **NCCL timeout across 100 nodes:** set `NCCL_TIMEOUT=1800` and ensure all nodes can reach the master address. Use `NCCL_IB_DISABLE=1` if InfiniBand is unstable.
- **Wrong positional embeddings at fine-tune:** the encoder's `pos_embed` was learned during pretraining with 1568 tokens (no masking at the positional level, masking happens after). During fine-tuning all 1568 tokens are used with the same pos_embed. No interpolation needed.
