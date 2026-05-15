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

Multi-node distributed training:
    # Single-node multi-GPU (torchrun):
    torchrun --nproc_per_node=4 train.py experiment=baseline_from_scratch data=vianney
    # Multi-node via SLURM:
    sbatch scripts/slurm_train.sh experiment=baseline_from_scratch data=vianney
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="This DataLoader will create", category=UserWarning)
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")
from datetime import datetime, timedelta
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).parent.parent

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 0.0,
    accum_steps: int = 1,
    total_steps: Optional[int] = None,
    rank: int = 0,
    log_every: int = 10,
    local_sgd_period: int = 0,
) -> Tuple[float, float]:
    import time as _step_time
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    _step_t = _step_time.monotonic()
    _opt_steps = 0
    local_sgd_active = local_sgd_period > 0

    optimizer.zero_grad()
    for step, (video_batch, labels) in enumerate(data_loader):
        if log_every and step % log_every == 0 and rank == 0:
            elapsed = _step_time.monotonic() - _step_t
            print(f"[train step {step}  elapsed={elapsed:.1f}s  samples={total}]", flush=True)
        video_batch = video_batch.to(device)
        labels = labels.to(device)
        if total_steps is not None:
            is_last = (step + 1 == total_steps)
        else:
            try:
                is_last = (step == len(data_loader) - 1)
            except TypeError:
                is_last = False  # IterableDataset (streaming) has no len()

        is_accum_step = (step + 1) % accum_steps == 0 or is_last
        # Local SGD: bypass DDP gradient sync; average parameters periodically instead
        is_sync_step = is_accum_step and not local_sgd_active
        no_sync_ctx = (
            model.no_sync()
            if not is_sync_step and hasattr(model, "no_sync")
            else contextlib.nullcontext()
        )
        with no_sync_ctx:
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(video_batch)
                    loss = loss_fn(logits, labels) / accum_steps
                scaler.scale(loss).backward()
            else:
                logits = model(video_batch)
                loss = loss_fn(logits, labels) / accum_steps
                loss.backward()

        running_loss += float(loss.item()) * accum_steps * labels.size(0)

        if is_accum_step:
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
            if local_sgd_active:
                _opt_steps += 1
                if _opt_steps % local_sgd_period == 0:
                    for p in model.parameters():
                        dist.all_reduce(p.data, op=dist.ReduceOp.AVG)

        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
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
    # --- Distributed setup ---
    # Map SLURM per-task vars → PyTorch env:// vars when launched via plain srun (no torchrun).
    # No-op when torchrun already injected RANK/WORLD_SIZE/LOCAL_RANK.
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
        # MASTER_ADDR / MASTER_PORT must be exported by the batch script before srun

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1

    for key, val in (cfg.training.get("nccl_env") or {}).items():
        os.environ.setdefault(str(key), str(val))

    import socket, time as _time
    _t0 = _time.monotonic()
    _host = socket.gethostname().split(".")[0]
    def _log(msg: str) -> None:
        if rank == 0:
            print(f"[{_time.monotonic() - _t0:6.1f}s] {msg}", flush=True)

    _log(f"init_process_group start (node={socket.gethostname()})")
    if is_dist:
        dist.init_process_group(
            backend=cfg.training.get("dist_backend", "nccl"),
            timeout=timedelta(seconds=int(cfg.training.get("nccl_timeout_sec", 120))),
        )
    _log("init_process_group done")

    if rank == 0:
        print(
            f"model={cfg.model.name}  bs={cfg.training.batch_size}  "
            f"lr={cfg.training.lr}  epochs={cfg.training.epochs}  "
            f"workers={cfg.training.num_workers}  amp={cfg.training.get('use_amp', False)}  "
            f"ws={world_size}"
        )

    # Different seed per rank ensures independent data augmentation
    set_seed(int(cfg.dataset.seed) + rank)

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        if rank == 0:
            print("CUDA not available; falling back to CPU.")
        device_str = "cpu"

    if device_str == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device(device_str)

    # --- Data ---
    use_imagenet_norm = bool(cfg.model.get("pretrained", False))
    num_frames = int(cfg.dataset.num_frames)
    train_sampler = None  # overridden below for non-streaming DDP

    if cfg.dataset.get("streaming", False):
        from dataset.streaming_dataset import make_streaming_loader
        _log("train_loader create start")
        train_loader = make_streaming_loader(
            shard_pattern=cfg.dataset.train_shards,
            num_frames=num_frames,
            transform=build_transforms(is_training=True, use_imagenet_norm=use_imagenet_norm),
            batch_size=int(cfg.training.batch_size),
            num_workers=int(cfg.training.num_workers),
            is_train=True,
        )
        val_loader = make_streaming_loader(
            shard_pattern=cfg.dataset.val_shards,
            num_frames=num_frames,
            transform=build_transforms(is_training=False, use_imagenet_norm=use_imagenet_norm),
            batch_size=int(cfg.training.batch_size),
            num_workers=int(cfg.training.num_workers),
            is_train=False,
        )
        _log("loaders created")
    else:
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

        train_dataset = VideoFrameDataset(
            root_dir=train_dir,
            num_frames=num_frames,
            transform=build_transforms(is_training=True, use_imagenet_norm=use_imagenet_norm),
            sample_list=train_samples,
        )
        val_dataset = VideoFrameDataset(
            root_dir=train_dir,
            num_frames=num_frames,
            transform=build_transforms(is_training=False, use_imagenet_norm=use_imagenet_norm),
            sample_list=val_samples,
        )

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        ) if is_dist else None
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        ) if is_dist else None

        _nw = int(cfg.training.num_workers)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=_nw,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(_nw > 0),
            prefetch_factor=(2 if _nw > 0 else None),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            sampler=val_sampler,
            num_workers=_nw,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(_nw > 0),
            prefetch_factor=(2 if _nw > 0 else None),
        )

    # --- Model / optimizer / scheduler ---
    _log("build_model start")
    model = build_model(cfg).to(device)
    if cfg.training.get("channels_last", False) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if cfg.training.get("compile", False):
        compile_mode = cfg.training.get("compile_mode", None) or None
        model = torch.compile(model, mode=compile_mode)
        if rank == 0:
            print(f"torch.compile active (mode={compile_mode!r})", flush=True)
    if is_dist:
        # static_graph=True is incompatible with no_sync() (used when accum_steps > 1)
        use_static_graph = cfg.training.get("grad_compress", "") != "powersgd" and int(cfg.training.get("accum_steps", 1)) <= 1
        # gradient_as_bucket_view=True causes stride mismatch warnings with torch.compile
        use_bucket_view = not cfg.training.get("compile", False)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=use_static_graph,
            gradient_as_bucket_view=use_bucket_view,
            bucket_cap_mb=200,
        )
        _grad_compress = cfg.training.get("grad_compress", "")
        if _grad_compress == "fp16":
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
            model.register_comm_hook(None, fp16_compress_hook)
            if rank == 0:
                print("Gradient compression: fp16 hook active (2× bandwidth reduction)", flush=True)
        elif _grad_compress == "powersgd":
            from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook
            state = PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=100,
                min_compression_rate=2,
            )
            model.register_comm_hook(state, powerSGD_hook)
            if rank == 0:
                print("Gradient compression: PowerSGD rank-1 hook active", flush=True)
    _log("model ready (DDP wrapped)")

    optimizer = build_optimizer(model, cfg)
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        steps_per_epoch = 0  # streaming IterableDataset has no len()
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=steps_per_epoch)

    use_amp = bool(cfg.training.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    grad_clip = float(cfg.training.get("grad_clip", 0.0))
    accum_steps = int(cfg.training.get("accum_steps", 1))

    # --- Auto-resume (elastic restarts re-run the script; pick up from last checkpoint) ---
    best_val_accuracy = 0.0
    start_epoch = 0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    if cfg.training.get("resume", True) and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt_model = ckpt.get("model_name", "unknown")
        if ckpt_model != cfg.model.name:
            if rank == 0:
                print(f"Checkpoint model ({ckpt_model}) != current model ({cfg.model.name}); skipping resume.")
        else:
            raw_model = model.module if is_dist else model
            if hasattr(raw_model, "_orig_mod"):  # torch.compile wraps in OptimizedModule
                raw_model = raw_model._orig_mod
            state_dict = ckpt["model_state_dict"]
            # Strip _orig_mod. prefix from checkpoints saved before the compile/save fix
            if any(k.startswith("_orig_mod.") for k in state_dict):
                state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
            raw_model.load_state_dict(state_dict)
            if "optimizer_state_dict" in ckpt:
                ckpt_opt = ckpt.get("optimizer_name", None)
                cur_opt = cfg.training.get("optimizer", "adam").lower()
                if ckpt_opt is not None and ckpt_opt == cur_opt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except (ValueError, KeyError) as e:
                        if rank == 0:
                            print(f"Optimizer state incompatible ({e}); starting optimizer from scratch.")
                else:
                    if rank == 0:
                        reason = f"ckpt={ckpt_opt}" if ckpt_opt else "no optimizer_name in checkpoint"
                        print(f"Skipping optimizer state ({reason} vs current={cur_opt}).")
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_accuracy = float(ckpt.get("best_val_accuracy", ckpt.get("val_accuracy", 0.0)))
            if rank == 0:
                print(f"Resumed from checkpoint (epoch {start_epoch}, best_val_acc={best_val_accuracy:.4f})")

    # --- Training loop ---
    patience = int(cfg.training.get("early_stopping_patience", 0))
    epochs_without_improvement = 0
    epoch_step_cap = int(cfg.training.get("steps_per_epoch", 0)) or None

    if cfg.dataset.get("streaming", False) and epoch_step_cap is None:
        if is_dist:
            dist.destroy_process_group()
        sys.exit(
            "streaming=True requires training.steps_per_epoch > 0 "
            "(ResampledShards is infinite). Pass training.steps_per_epoch=N."
        )

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        _log(f"epoch {epoch} train_one_epoch start")
        local_sgd_period = int(cfg.training.get("local_sgd_period", 0))
        train_loss, train_acc = train_one_epoch(
            model, itertools.islice(train_loader, epoch_step_cap),
            loss_fn, optimizer, device, scaler, grad_clip, accum_steps,
            total_steps=epoch_step_cap, rank=rank, log_every=10,
            local_sgd_period=local_sgd_period,
        )
        _log(f"epoch {epoch} train_one_epoch done  loss={train_loss:.4f}")

        _log(f"epoch {epoch} evaluate_epoch start")
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device, scaler)
        _log(f"epoch {epoch} evaluate_epoch done  val_acc={val_acc:.4f}")

        # Average metrics across all ranks for consistent logging and checkpoint decisions
        if is_dist:
            metrics = torch.tensor([train_loss, train_acc, val_loss, val_acc], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size
            train_loss, train_acc, val_loss, val_acc = metrics.tolist()

        if scheduler is not None:
            scheduler.step()

        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{cfg.training.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_without_improvement = 0
            if rank == 0:
                raw_model = model.module if is_dist else model
                if hasattr(raw_model, "_orig_mod"):  # torch.compile wraps in OptimizedModule
                    raw_model = raw_model._orig_mod
                payload: Dict[str, Any] = {
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_accuracy": val_acc,
                    "model_name": cfg.model.name,
                    "num_classes": int(cfg.model.num_classes),
                    "num_frames": num_frames,
                    "val_accuracy": val_acc,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "optimizer_name": cfg.training.get("optimizer", "adam").lower(),
                }
                # backward-compat fields for existing loaders
                if cfg.model.name in ("cnn_baseline", "cnn_lstm"):
                    payload["pretrained"] = bool(cfg.model.get("pretrained", False))
                if cfg.model.name == "cnn_lstm":
                    payload["lstm_hidden_size"] = int(cfg.model.get("lstm_hidden_size", 512))

                tmp_path = checkpoint_path.with_suffix(".tmp")
                torch.save(payload, tmp_path)
                tmp_path.replace(checkpoint_path)
                print(f"  Saved best model (val acc={val_acc:.4f}) -> {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                if rank == 0:
                    print(f"  Early stopping: no improvement for {patience} epochs.")
                break

    if rank == 0:
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
            "world_size": world_size,
            "warning": "TOP1_LOW" if best_val_accuracy < 0.5 else None,
        }
        result_path = results_dir / f"{timestamp}-{cfg.model.name}.json"
        result_path.write_text(json.dumps(result, indent=2))
        print(f"Results → {result_path}")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
