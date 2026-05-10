# Model 2 — UniFormer-B (Unified Transformer)

> **Role:** Tier 2 — best purely supervised from-scratch model.  
> **Expected Top-1:** 68–72% on the 50-class SSv2 subset  
> **Compute:** 8× A100, ~24h

---

## Why UniFormer for This Task

Pure Vision Transformers (e.g. TimeSformer) are data-hungry — global self-attention between all pairs of tokens from the very first layer means the model has to learn spatial structure from scratch without any inductive bias. On limited data (a 50-class subset) this leads to slow convergence and overfitting.

UniFormer solves this with a **stage-wise hybrid design**:

- **Early stages (low resolution):** local MHSA (Multi-Head Self-Attention) where each token only attends to its spatial neighbourhood. This is equivalent to a dynamic convolution with learned kernels — cheap, inductive-biased, great for learning local textures.
- **Late stages (high semantic level):** global MHSA across all tokens. By this stage the tokens are semantically meaningful, so global attention is efficient and productive.

The result: UniFormer converges faster than pure transformers on small datasets, while still capturing the long-range temporal context that ResNets miss.

---

## Architecture

```
Input: (B, C, T, H, W) — video in PyTorch channel-first format
       16 frames × 3 × 224 × 224

Stage 1  (local MHSA):  spatial 56×56, temporal 16  →  tokens: (B, 64,  16, 56, 56)
Stage 2  (local MHSA):  spatial 28×28, temporal 16  →  tokens: (B, 128, 16, 28, 28)
Stage 3  (global MHSA): spatial 14×14, temporal 8   →  tokens: (B, 320,  8, 14, 14)
Stage 4  (global MHSA): spatial 7×7,   temporal 8   →  tokens: (B, 512,  8,  7,  7)
                                                         ↓
Global average pool                               →  (B, 512)
Linear(512, 50)                                   →  (B, 50)
```

UniFormer-B config: `[5, 8, 20, 7]` blocks per stage, embed dims `[64, 128, 320, 512]`.

---

## Implementation

### Local MHSA block (early stages)

In early stages, each token attends only to its local 3D neighbourhood (temporal window × spatial window), instead of all tokens globally. This is implemented with a depth-wise 3D convolution as the "relation" (key-query interaction):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LocalMHSA(nn.Module):
    """Local Multi-Head Self-Attention for early UniFormer stages."""
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 kernel_size=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm = nn.BatchNorm3d(dim)
        # DW conv computes local relation matrix
        self.local_conv = nn.Conv3d(dim, dim, kernel_size=(1, kernel_size, kernel_size),
                                    padding=(0, kernel_size//2, kernel_size//2),
                                    groups=dim, bias=False)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # Local relation (DW conv)
        x_norm = self.norm(x)
        x_local = self.local_conv(x_norm)

        # Flatten spatial+temporal → tokens
        x_flat = rearrange(x_norm, 'b c t h w -> b (t h w) c')
        x_loc_flat = rearrange(x_local, 'b c t h w -> b (t h w) c')

        q = self.q(x_flat)
        k = self.k(x_loc_flat)
        v = self.v(x_loc_flat)

        # Multi-head attention with local keys
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj_drop(self.proj(out))
        out = rearrange(out, 'b (t h w) c -> b c t h w', t=T, h=H, w=W)
        return out
```

### Global MHSA block (late stages)

Same as standard ViT attention but operating on the full flattened token sequence:

```python
class GlobalMHSA(nn.Module):
    """Global Multi-Head Self-Attention for late UniFormer stages."""
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        N = T * H * W

        x_flat = rearrange(x, 'b c t h w -> b (t h w) c')
        x_norm = self.norm(x_flat)

        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        out = rearrange(out, 'b (t h w) c -> b c t h w', t=T, h=H, w=W)
        return out
```

### UniFormer block

Each block = attention + FFN with residual connections:

```python
class UniFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., local=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) if not local else nn.BatchNorm3d(dim)
        self.attn = (LocalMHSA(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
                     if local else
                     GlobalMHSA(dim, num_heads, attn_drop=attn_drop, proj_drop=drop))
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop),
        )
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = x + self.drop_path(self.attn(x))
        # FFN on flattened tokens
        B, C, T, H, W = x.shape
        x_flat = rearrange(x, 'b c t h w -> b (t h w) c')
        x_flat = x_flat + self.drop_path(self.ffn(self.norm2(x_flat)))
        x = rearrange(x_flat, 'b (t h w) c -> b c t h w', t=T, h=H, w=W)
        return x
```

### Stochastic depth (drop path)

Essential regularizer for transformers trained on limited data:

```python
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep)
        return x * random_tensor / keep
```

### Full UniFormer-B model

```python
class UniFormerB(nn.Module):
    def __init__(self, num_classes=50, num_frames=16,
                 depths=[5, 8, 20, 7],
                 dims=[64, 128, 320, 512],
                 num_heads=[2, 4, 10, 16],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3):
        super().__init__()
        self.num_frames = num_frames

        # Stochastic depth schedule: linearly increases across all blocks
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        idx = 0

        # Patch embedding (stem) — 3D conv to create initial tokens
        self.patch_embed1 = nn.Sequential(
            nn.Conv3d(3, dims[0], kernel_size=(1,4,4), stride=(1,4,4)),
            nn.BatchNorm3d(dims[0]),
            nn.GELU(),
        )
        self.patch_embed2 = nn.Sequential(
            nn.Conv3d(dims[0], dims[1], kernel_size=(1,2,2), stride=(1,2,2)),
            nn.BatchNorm3d(dims[1]),
            nn.GELU(),
        )
        self.patch_embed3 = nn.Sequential(
            nn.Conv3d(dims[1], dims[2], kernel_size=(2,2,2), stride=(2,2,2)),
            nn.BatchNorm3d(dims[2]),
            nn.GELU(),
        )
        self.patch_embed4 = nn.Sequential(
            nn.Conv3d(dims[2], dims[3], kernel_size=(2,2,2), stride=(2,2,2)),
            nn.BatchNorm3d(dims[3]),
            nn.GELU(),
        )

        # Stages
        self.stage1 = nn.Sequential(*[
            UniFormerBlock(dims[0], num_heads[0], mlp_ratio, drop_rate,
                           attn_drop_rate, dpr[idx+i], local=True)
            for i in range(depths[0])])
        idx += depths[0]

        self.stage2 = nn.Sequential(*[
            UniFormerBlock(dims[1], num_heads[1], mlp_ratio, drop_rate,
                           attn_drop_rate, dpr[idx+i], local=True)
            for i in range(depths[1])])
        idx += depths[1]

        self.stage3 = nn.Sequential(*[
            UniFormerBlock(dims[2], num_heads[2], mlp_ratio, drop_rate,
                           attn_drop_rate, dpr[idx+i], local=False)
            for i in range(depths[2])])
        idx += depths[2]

        self.stage4 = nn.Sequential(*[
            UniFormerBlock(dims[3], num_heads[3], mlp_ratio, drop_rate,
                           attn_drop_rate, dpr[idx+i], local=False)
            for i in range(depths[3])])

        self.norm = nn.LayerNorm(dims[3])
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dims[3], num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x: (B, T, C, H, W) → rearrange to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.patch_embed1(x)  # (B, 64,  T,   56, 56)
        x = self.stage1(x)

        x = self.patch_embed2(x)  # (B, 128, T,   28, 28)
        x = self.stage2(x)

        x = self.patch_embed3(x)  # (B, 320, T/2, 14, 14)
        x = self.stage3(x)

        x = self.patch_embed4(x)  # (B, 512, T/4,  7,  7)
        x = self.stage4(x)

        # Global average pool → classify
        x = x.flatten(2).mean(dim=-1)      # (B, 512)
        x = self.norm(x)
        return self.head(x)
```

---

## Training Script

```python
def train_uniformer(config):
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    model = UniFormerB(num_classes=50, num_frames=config.num_frames).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Data — 16 frames for UniFormer
    train_dataset = SSv2Dataset(config.train_ann, config.video_dir,
                                train_transform, config.num_frames)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_per_gpu,
                               sampler=train_sampler, num_workers=8, pin_memory=True)

    # AdamW — better than SGD for transformers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # Cosine schedule with linear warmup
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return epoch / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler     = torch.cuda.amp.GradScaler()

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
            # Gradient clipping — important for transformers
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        if local_rank == 0:
            acc = evaluate(model, val_loader, local_rank)
            print(f"Epoch {epoch+1} | Top-1: {acc:.2f}%")
            torch.save(model.module.state_dict(), f"uniformer_epoch{epoch+1}.pth")
```

---

## Launch Command

```bash
torchrun --nproc_per_node=8 \
         --nnodes=1 \
         --master_port=29501 \
         train_uniformer.py
```

---

## Config

```python
@dataclass
class UniFormerConfig:
    # Data
    train_ann:  str = "data/train.csv"
    val_ann:    str = "data/val.csv"
    video_dir:  str = "data/videos"
    num_frames: int = 16

    # Model
    drop_path_rate: float = 0.3   # stochastic depth — key regularizer

    # Training
    epochs:        int   = 50
    batch_per_gpu: int   = 8      # UniFormer-B: ~10GB VRAM per sample at fp16
    lr:            float = 1e-3
    weight_decay:  float = 0.05
    warmup_epochs: int   = 5
```

---

## Ablations to Run

| Change | Expected effect |
|--------|----------------|
| `drop_path_rate` 0.1 → 0.3 | +1–2% (less overfitting) |
| `num_frames` 8 → 16 | +1–2% (more temporal context) |
| Add mixup (α=0.8) | +0.5–1% |
| Add cutmix (α=1.0) | +0.5% |
| `label_smoothing` 0.0 → 0.1 | +0.5% |

### Mixup augmentation

```python
def mixup_data(x, y, alpha=0.8):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

---

## Key Failure Modes

- **OOM during training:** UniFormer-B with 16 frames at 224² is memory-intensive. Reduce `batch_per_gpu` to 4 and use gradient accumulation over 2 steps to keep effective batch size at 8.
- **Slow convergence vs TSM:** normal — transformers need warmup. Do not reduce LR before warmup_epochs.
- **Global attention stages becoming bottleneck:** if GPU utilisation drops sharply, the T×H×W token sequence in global MHSA is too long. Reduce frames to 8 or spatial resolution to 112 for ablations.
