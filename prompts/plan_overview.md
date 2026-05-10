# Implementation Plan — Track A Closed World
## From TSM Baseline to VideoMAE ViT-L

> **Competition:** What Happens Next? Track A — Closed World  
> **Task:** Action anticipation on 50-class SSv2 subset  
> **Constraint:** Train from scratch on provided data only — no pretrained weights, no external data  
> **Compute:** ~100 nodes × A100 GPUs (DDP via `torchrun`)

---

## Rationale & Strategy

SSv2 is a motion-heavy benchmark. Actions like *"Pretending to throw something"* vs *"Throwing something"* cannot be distinguished from static frames alone — the model must capture temporal dynamics. This drives every architecture choice below.

The plan is structured in three tiers of increasing complexity:

| Tier | Model | Expected Top-1 | Role |
|------|-------|---------------|------|
| 1 | TSM ResNet-50 | ~60–63% | Fast baseline, pipeline validation |
| 2 | UniFormer-B | ~68–72% | Best pure-supervised from-scratch |
| 3 | VideoMAE ViT-L | ~74–77% | Primary submission |
| — | Ensemble (all three) | **+1–3% over best single** | Final submission |

Because the dataset is a 50-class subset (easier than full SSv2's 174 classes), all models should exceed their full-SSv2 benchmarks. The ceiling for a well-trained VideoMAE ViT-L on this task is realistically **80%+**.

---

## Phase Overview

```
Week 1        Week 2        Week 3        Week 4
──────────────────────────────────────────────────
[TSM]         [UniFormer]   [VideoMAE]     [Ensemble]
pipeline  →   training  →   pretrain  →   + final
validation    + ablation    + finetune     submit
```

---

## Step 1 — Data Pipeline (shared across all models)

Before training any model, the data pipeline must be solid. All models share the same input format.

### Frame extraction

```python
# Extract T frames uniformly from the first 50% of each clip
# (anticipation task — we only see the beginning)
import decord
import numpy as np

def load_frames(video_path, num_frames=16, clip_fraction=0.5):
    vr = decord.VideoReader(video_path, num_threads=4)
    total = len(vr)
    end_idx = int(total * clip_fraction)
    indices = np.linspace(0, end_idx - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)
    return frames
```

### Augmentation (training)

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

### Dataset class

```python
class SSv2Dataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, video_dir, transform, num_frames=16):
        self.samples = pd.read_csv(annotation_file)  # columns: video_id, label
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        frames = load_frames(f"{self.video_dir}/{row.video_id}.mp4", self.num_frames)
        frames = [self.transform(Image.fromarray(f)) for f in frames]
        clip = torch.stack(frames)  # (T, C, H, W)
        return clip, row.label

    def __len__(self):
        return len(self.samples)
```

---

## Step 2 — TSM (Tier 1 baseline)

**Goal:** validate the full training pipeline in <1 day, get a score on the leaderboard.

**Architecture:** Temporal Shift Module wrapping ResNet-50. 1/8 of channels are shifted forward in time, 1/8 backward, the rest stay. This adds zero parameters to ResNet-50.

**Training:** ~8 epochs, ~6h on a single A100.

**Expected Top-1:** 60–63% on 50-class subset.

See [`model_tsm.md`](./model_tsm.md) for full implementation.

---

## Step 3 — UniFormer-B (Tier 2)

**Goal:** best purely supervised from-scratch model. Combines local convolution (cheap, handles fine-grained spatial patterns) with global attention (captures long-range temporal context). This hybrid is specifically designed to avoid the data-hunger of pure transformers.

**Training:** ~50 epochs, ~24h on 8× A100s.

**Expected Top-1:** 68–72%.

See [`model_uniformer.md`](./model_uniformer.md) for full implementation.

---

## Step 4 — VideoMAE ViT-L (Tier 3)

This is the primary bet. Two sub-phases:

### 4a — Self-supervised pretraining (no labels)

Train a masked video autoencoder on the competition training clips. The encoder sees only ~10% of tokens (tube masking at 90%). The decoder reconstructs the masked pixel values from those visible tokens. This pretrains the encoder to understand video motion without any labels.

**Why this is legal in Track A:** No external data. No external pretrained weights. Everything comes from the competition training set.

**Training:** ~800 epochs, ~16h on 100× A100s.

### 4b — Supervised fine-tuning (with labels)

Discard the decoder. Add a classification head (global average pool + linear layer). Fine-tune the encoder end-to-end on the labeled training clips.

**Training:** ~50 epochs, ~2h on 100× A100s.

**Expected Top-1:** 74–77% (likely higher on 50-class subset).

See [`model_videomae.md`](./model_videomae.md) for full implementation.

---

## Step 5 — Ensemble

Combine the three models by averaging their softmax output probabilities at inference time.

```python
def ensemble_predict(models, clip):
    probs = []
    for model in models:
        with torch.no_grad():
            logits = model(clip)
            probs.append(torch.softmax(logits, dim=-1))
    return torch.stack(probs).mean(0).argmax(dim=-1)
```

Weighted averaging (e.g. VideoMAE × 0.5, UniFormer × 0.3, TSM × 0.2) may help if the models have very different individual accuracies. Tune weights on the public leaderboard.

**Expected gain:** +1–3% over best single model.

---

## Distributed Training Setup

All models use PyTorch DDP. Launch with:

```bash
torchrun --nproc_per_node=NUM_GPUS_PER_NODE \
         --nnodes=NUM_NODES \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=29500 \
         train.py --config configs/videomae_pretrain.yaml
```

### DDP boilerplate

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def init_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = init_distributed()
model = model.cuda(local_rank)
model = DDP(model, device_ids=[local_rank])
```

### Effective batch size and LR scaling

With DDP, each GPU processes its own mini-batch. Gradients are averaged across all GPUs after each backward pass.

```
effective_batch = batch_per_gpu × num_gpus
LR = base_LR × (effective_batch / 256)
```

For VideoMAE pretraining with 100 GPUs and batch_per_gpu=4:
```
effective_batch = 4 × 100 = 400
LR = 1.5e-4 × (400 / 256) ≈ 2.34e-4
```

### Mixed precision (mandatory on A100)

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    loss = criterion(model(clips), labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Key Hyperparameters Summary

| Parameter | TSM | UniFormer-B | VideoMAE (pretrain) | VideoMAE (finetune) |
|-----------|-----|-------------|--------------------|--------------------|
| Frames T | 8 | 16 | 16 | 16 |
| Resolution | 224 | 224 | 224 | 224 |
| Epochs | 50 | 50 | 800 | 50 |
| Batch/GPU | 16 | 8 | 32 | 8 |
| Optimizer | SGD | AdamW | AdamW | AdamW |
| Base LR | 0.01 | 1e-3 | 1.5e-4 | 1e-3 |
| LR schedule | cosine | cosine | cosine | cosine |
| Warmup epochs | 5 | 5 | 40 | 5 |
| Weight decay | 1e-4 | 0.05 | 0.05 | 0.05 |
| Label smoothing | — | 0.1 | — | 0.1 |
| Dropout (head) | — | — | — | 0.5 |
| Mask ratio | — | — | 0.90 | — |

---

## Submission Format

All models output a CSV:

```csv
video_id,label
video_000001,12
video_000002,3
video_000003,45
```

Labels must match `class_to_idx.json` exactly. Generate with:

```python
def generate_submission(model, test_loader, output_path):
    model.eval()
    rows = []
    with torch.no_grad():
        for clips, video_ids in test_loader:
            preds = model(clips.cuda()).argmax(dim=-1)
            for vid_id, pred in zip(video_ids, preds.cpu()):
                rows.append({"video_id": vid_id, "label": pred.item()})
    pd.DataFrame(rows).to_csv(output_path, index=False)
```

---

## Expected Timeline

| Day | Task |
|-----|------|
| 1 | Set up data pipeline, verify frame extraction on all clips |
| 2 | Train TSM, submit first prediction, validate leaderboard setup |
| 3–4 | Train UniFormer-B, ablate number of frames and augmentation |
| 5–8 | VideoMAE pretraining (800 epochs, distributed across 100 A100s) |
| 9–10 | VideoMAE fine-tuning + ablation (frames, LR, dropout) |
| 11 | Ensemble TSM + UniFormer + VideoMAE, tune weights on public LB |
| 12 | Final submission |
