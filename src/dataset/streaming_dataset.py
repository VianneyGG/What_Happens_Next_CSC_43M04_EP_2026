"""
WebDataset-based streaming loader for video frame data.

Reads tar shards produced by src/misc/convert_to_webdataset.py.
Shards can be local paths or HTTP URLs (served via the launcher's reverse tunnel).

Tensor contract: same as VideoFrameDataset — returns (B, T, C, H, W) video tensors
and (B,) int64 label tensors.

In distributed training, split_by_node + split_by_worker replace DistributedSampler:
each GPU node reads a disjoint subset of shards, and each DataLoader worker reads a
disjoint subset within the node's shards.
"""

from __future__ import annotations

import glob as _glob
import io
from typing import Callable

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader


def _resolve_shards(pattern: str | list[str]) -> str | list[str]:
    if isinstance(pattern, list):
        return pattern
    if pattern.startswith("http"):
        return pattern
    paths = sorted(_glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shards found: {pattern}")
    return paths


def make_streaming_loader(
    shard_pattern: str | list[str],
    num_frames: int,
    transform: Callable,
    batch_size: int,
    num_workers: int,
    is_train: bool,
) -> DataLoader:
    """
    Build a DataLoader backed by WebDataset tar shards.

    Args:
        shard_pattern: Shard source — one of:
            - A brace-expanded pattern: "path/shard-{000000..000089}.tar"
            - An HTTP URL pattern: "http://localhost:9888/train/shard-{000000..000089}.tar"
            - A list of file paths / URLs (e.g. from glob)
        num_frames: Number of frames T to load per video.
        transform: Per-frame PIL → Tensor transform (same as VideoFrameDataset).
        batch_size: Batch size B.
        num_workers: DataLoader worker count.
        is_train: If True, enables buffer shuffling of shards and samples.

    Returns:
        DataLoader yielding (video, label) with shapes (B, T, C, H, W) and (B,).
    """
    def decode_sample(sample: dict) -> tuple[torch.Tensor, torch.Tensor]:
        npz = np.load(io.BytesIO(sample["frames.npz"]))
        label = int(sample["cls"].decode().strip())
        frames = []
        for i in range(num_frames):
            key = f"frame_{i}"
            if key not in npz:
                key = f"frame_{len(npz.files) - 1}"  # repeat last if fewer than requested
            frame_bytes = bytes(npz[key])
            img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            frames.append(transform(img))
        video = torch.stack(frames, dim=0)  # (T, C, H, W)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return video, label_tensor

    if is_train:
        # ResampledShards resamples shards with replacement → infinite iterator.
        # Combined with islice(train_loader, steps_per_epoch) in train.py, all DDP
        # ranks run exactly steps_per_epoch backward passes regardless of shard count.
        pipeline_steps: list = [
            wds.ResampledShards(_resolve_shards(shard_pattern)),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(1000),
            wds.map(decode_sample),
            wds.batched(batch_size, collation_fn=torch.utils.data.default_collate, partial=False),
        ]
    else:
        pipeline_steps = [
            wds.SimpleShardList(_resolve_shards(shard_pattern)),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.map(decode_sample),
            wds.batched(batch_size, collation_fn=torch.utils.data.default_collate, partial=False),
        ]

    dataset = wds.DataPipeline(*pipeline_steps)

    return DataLoader(
        dataset,
        batch_size=None,  # batching handled by wds.batched()
        num_workers=num_workers,
        pin_memory=True,
    )
