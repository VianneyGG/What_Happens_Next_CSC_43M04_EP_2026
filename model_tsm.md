# Model 1 — TSM (Temporal Shift Module)

> **Role:** Tier 1 baseline. Validate the full pipeline in <1 day and get a leaderboard score.  
> **Expected Top-1:** 60–63% on the 50-class SSv2 subset  
> **Compute:** 1–4× A100, ~6–12h

---

## What TSM Does

TSM adds temporal reasoning to a standard 2D CNN (ResNet-50) for **free** — zero extra parameters, negligible extra compute. The trick: before each residual block, shift a fraction of the channel dimension along the time axis.

```
Frame 1: [ch_0 ... ch_7 | ch_8 ... ch_15 | ch_16 ... ch_63]
Frame 2: [ch_0 ... ch_7 | ch_8 ... ch_15 | ch_16 ... ch_63]
                                                    ↓  shift(−1)
Frame 1 after shift: [ Frame2_ch0..7 | Frame0_ch8..15 | Frame1_ch16..63 ]
```

- 1/8 of channels get the previous frame's features (shift backward)
- 1/8 of channels get the next frame's features (shift forward)
- 6/8 of channels stay in place

The residual convolution then processes this temporally-mixed feature map as if it were a normal 2D convolution. Temporal context is implicit in the shifted channels.

---

## Architecture

```
Input: (B, T, C, H, W)  — batch × 8 frames × 3 × 224 × 224
      ↓ reshape to (B×T, C, H, W)
      ↓ TSM shift (before each residual block)
ResNet-50 backbone (all 2D conv)
      ↓ (B×T, 2048, 7, 7)
      ↓ global average pool → (B×T, 2048)
      ↓ reshape to (B, T, 2048)
      ↓ temporal average pool → (B, 2048)
Linear(2048, 50)
      ↓ (B, 50) — logits
```

---

## Implementation

### TSM shift module

```python
import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8, inplace=False):
        super().__init__()
        self.net = net          # the residual block to wrap
        self.n_segment = n_segment
        self.fold_div = n_div   # 1/fold_div channels shifted each direction
        self.inplace = inplace

    def forward(self, x):
        x = self.shift(x, self.n_segment, self.fold_div, self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div, inplace):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # In-place version: slightly more memory efficient
            out = x.clone()
        else:
            out = torch.zeros_like(x)

        # Shift backward: frame i gets features from frame i-1
        out[:, 1:, :fold] = x[:, :-1, :fold]
        # Shift forward: frame i gets features from frame i+1
        out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]
        # No shift for the remaining channels
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out.view(nt, c, h, w)
```

### Wrap ResNet-50 with TSM

```python
import torchvision.models as models

def make_tsm_resnet50(num_classes=50, n_segment=8):
    backbone = models.resnet50(weights=None)  # from scratch — no pretrained weights

    # Wrap the first conv of every residual block with TSM
    def make_block_temporal(stage, n_segment):
        blocks = list(stage.children())
        for i, b in enumerate(blocks):
            blocks[i] = TemporalShift(b, n_segment=n_segment, n_div=8)
        return nn.Sequential(*blocks)

    backbone.layer1 = make_block_temporal(backbone.layer1, n_segment)
    backbone.layer2 = make_block_temporal(backbone.layer2, n_segment)
    backbone.layer3 = make_block_temporal(backbone.layer3, n_segment)
    backbone.layer4 = make_block_temporal(backbone.layer4, n_segment)

    # Replace final FC
    backbone.fc = nn.Linear(2048, num_classes)
    return backbone
```

### Forward pass wrapper

```python
class TSMModel(nn.Module):
    def __init__(self, num_classes=50, n_segment=8):
        super().__init__()
        self.n_segment = n_segment
        self.backbone = make_tsm_resnet50(num_classes, n_segment)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        assert T == self.n_segment

        x = x.view(B * T, C, H, W)        # merge batch and time
        x = self.backbone(x)               # (B*T, num_classes)
        x = x.view(B, T, -1).mean(dim=1)  # temporal average pool
        return x
```

---

## Training Script

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train_tsm(config):
    # --- Distributed init ---
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # --- Model ---
    model = TSMModel(num_classes=50, n_segment=config.num_frames).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # --- Data ---
    train_dataset = SSv2Dataset(config.train_ann, config.video_dir,
                                train_transform, config.num_frames)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_per_gpu,
                              sampler=train_sampler, num_workers=8, pin_memory=True)

    val_dataset = SSv2Dataset(config.val_ann, config.video_dir,
                              val_transform, config.num_frames)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_per_gpu * 2,
                            shuffle=False, num_workers=8, pin_memory=True)

    # --- Optimizer ---
    # SGD works better than AdamW for ResNet from scratch
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        for clips, labels in train_loader:
            clips = clips.cuda(local_rank, non_blocking=True)
            labels = labels.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(clips)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        if local_rank == 0:
            acc = evaluate(model, val_loader, local_rank)
            print(f"Epoch {epoch+1}/{config.epochs} | Val Top-1: {acc:.2f}%")
            torch.save(model.module.state_dict(), f"tsm_epoch{epoch+1}.pth")


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.cuda(device), labels.cuda(device)
            with torch.cuda.amp.autocast():
                preds = model(clips).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total
```

---

## Launch Command

```bash
torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --master_port=29500 \
         train_tsm.py
```

For a single GPU:
```bash
python train_tsm.py
```

---

## Config

```python
@dataclass
class TSMConfig:
    # Data
    train_ann: str = "data/train.csv"
    val_ann:   str = "data/val.csv"
    video_dir: str = "data/videos"
    num_frames: int = 8

    # Training
    epochs:        int   = 50
    batch_per_gpu: int   = 16
    lr:            float = 0.01      # base LR for SGD
    weight_decay:  float = 1e-4
    warmup_epochs: int   = 5
```

---

## Inference / Submission

```python
def predict_tsm(model_path, test_loader):
    model = TSMModel(num_classes=50, n_segment=8)
    model.load_state_dict(torch.load(model_path))
    model.cuda().eval()

    rows = []
    with torch.no_grad():
        for clips, video_ids in test_loader:
            probs = torch.softmax(model(clips.cuda()), dim=-1)
            preds = probs.argmax(dim=-1)
            for vid, pred in zip(video_ids, preds.cpu()):
                rows.append({"video_id": vid, "label": pred.item()})
    return pd.DataFrame(rows)
```

---

## Expected Results & Ablations

| Variant | Frames T | Top-1 (est.) | Notes |
|---------|----------|-------------|-------|
| TSM ResNet-50 | 8 | ~60% | baseline |
| TSM ResNet-50 | 16 | ~62% | more temporal context |
| TSM ResNet-101 | 8 | ~63% | deeper backbone |
| TSM ResNet-50 + TTA | 8 | ~61% | 3-crop test-time aug |

**TTA (test-time augmentation):** average predictions from 3 spatial crops (left, center, right) and 2 temporal clips. Typically +0.5–1%.

```python
def predict_with_tta(model, clip):
    # clip: (1, T, C, H, W) — full resolution
    crops = [left_crop(clip), center_crop(clip), right_crop(clip)]
    probs = [torch.softmax(model(c.cuda()), dim=-1) for c in crops]
    return torch.stack(probs).mean(0)
```

---

## Key Failure Modes to Watch

- **Overfitting:** ResNet-50 from scratch on limited data will overfit. Add dropout before the FC layer (`nn.Dropout(p=0.5)`) and use data augmentation aggressively.
- **Temporal order not captured:** TSM only shifts by 1 frame at a time. If the temporal pattern spans many frames, TSM may miss it — this is the motivation for moving to UniFormer/VideoMAE.
- **Wrong number of frames at inference:** The shift module assumes exactly `n_segment` frames. Passing a different number will corrupt the shift pattern.
