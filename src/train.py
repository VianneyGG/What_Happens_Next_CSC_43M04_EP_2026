"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=tsm_from_scratch
    python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``training.epochs=10``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations

import json
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).parent.parent

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.tsm import TSMResNet50
from utils import build_transforms, set_seed, split_train_val


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig) -> nn.Module:
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.get("pretrained", False)

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)

    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )

    if name == "tsm":
        return TSMResNet50(
            num_classes=int(num_classes),
            n_segment=int(cfg.model.n_segment),
            dropout=float(cfg.model.get("dropout", 0.5)),
        )

    if name == "uniformer":
        from models.uniformer import UniFormerB
        return UniFormerB(
            num_classes=int(num_classes),
            depths=list(cfg.model.get("depths", [5, 8, 20, 7])),
            dims=list(cfg.model.get("dims", [64, 128, 320, 512])),
            num_heads=list(cfg.model.get("num_heads", [2, 4, 10, 16])),
            mlp_ratio=float(cfg.model.get("mlp_ratio", 4.0)),
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.3)),
            window_size=list(cfg.model.get("window_size", [4, 7, 7])),
        )

    raise ValueError(f"Unknown model.name: {name!r}")


# ---------------------------------------------------------------------------
# Optimizer & scheduler factories
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    opt_name = cfg.training.get("optimizer", "adam").lower()
    lr = float(cfg.training.lr)

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.training.get("momentum", 0.9)),
            weight_decay=float(cfg.training.get("weight_decay", 1e-4)),
            nesterov=bool(cfg.training.get("nesterov", True)),
        )

    if opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=float(cfg.training.get("weight_decay", 0.05)),
        )

    # default: adam
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg.training.get("weight_decay", 0.0)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    sched_name = cfg.training.get("lr_scheduler", "none").lower()
    if sched_name == "none":
        return None

    epochs = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))

    if sched_name == "cosine":
        # Linear warmup then cosine decay, implemented as LambdaLR over epochs.
        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown lr_scheduler: {sched_name!r}. Choose 'cosine' or 'none'.")


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: float = 0.0,
    accum_steps: int = 1,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()
    for step, (video_batch, labels) in enumerate(data_loader):
        video_batch = video_batch.to(device)
        labels = labels.to(device)
        is_last = (step == len(data_loader) - 1)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(video_batch)
                loss = loss_fn(logits, labels) / accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(video_batch)
            loss = loss_fn(logits, labels) / accum_steps
            loss.backward()

        # Unscaled loss for logging
        running_loss += float(loss.item()) * accum_steps * labels.size(0)

        if (step + 1) % accum_steps == 0 or is_last:
            if scaler is not None:
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(video_batch)
                loss = loss_fn(logits, labels)
        else:
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # --- Data ---
    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
    )

    use_imagenet_norm = bool(cfg.model.get("pretrained", False))
    train_transform = build_transforms(is_training=True, use_imagenet_norm=use_imagenet_norm)
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=use_imagenet_norm)

    num_frames = int(cfg.dataset.num_frames)

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=train_transform,
        sample_list=train_samples,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    # --- Model / optimizer / scheduler ---
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    use_amp = bool(cfg.training.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    grad_clip = float(cfg.training.get("grad_clip", 0.0))
    accum_steps = int(cfg.training.get("accum_steps", 1))

    # --- Training loop ---
    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    patience = int(cfg.training.get("early_stopping_patience", 0))
    epochs_without_improvement = 0

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, grad_clip, accum_steps
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device, scaler
        )

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_without_improvement = 0
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "num_frames": num_frames,
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            # backward-compat fields for existing loaders
            if cfg.model.name in ("cnn_baseline", "cnn_lstm"):
                payload["pretrained"] = bool(cfg.model.get("pretrained", False))
            if cfg.model.name == "cnn_lstm":
                payload["lstm_hidden_size"] = int(cfg.model.get("lstm_hidden_size", 512))

            torch.save(payload, checkpoint_path)
            print(f"  Saved best model (val acc={val_acc:.4f}) -> {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"  Early stopping: no improvement for {patience} epochs.")
                break

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")

    results_dir = REPO_ROOT / ".claude" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result = {
        "timestamp": timestamp,
        "experiment": cfg.model.name,
        "best_val_accuracy": round(float(best_val_accuracy), 4),
        "epochs_trained": int(cfg.training.epochs),
        "lr": float(cfg.training.lr),
        "batch_size": int(cfg.training.batch_size),
        "num_frames": int(cfg.dataset.num_frames),
        "warning": "TOP1_LOW" if best_val_accuracy < 0.5 else None,
    }
    result_path = results_dir / f"{timestamp}-{cfg.model.name}.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Results → {result_path}")


if __name__ == "__main__":
    main()
