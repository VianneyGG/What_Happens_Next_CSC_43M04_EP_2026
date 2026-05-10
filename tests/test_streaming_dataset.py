"""
Comprehensive tests for the WebDataset streaming pipeline.

All tests use synthetic data written to pytest's tmp_path — no real dataset needed.
Tests cover: shard conversion, loader shapes, label correctness,
node-level distribution, training integration, HTTP serving, and SSH command generation.
"""

from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
import tarfile
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset.streaming_dataset import make_streaming_loader
from misc.convert_to_webdataset import run_conversion
from scripts.launch_distributed import build_ssh_command
from train import build_model, train_one_epoch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 2
VIDEOS_PER_CLASS = 3
NUM_FRAMES = 4
FRAME_SIZE = (8, 8)  # tiny images for speed
BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_video_dir(tmp_path: Path) -> Path:
    """Write 6 synthetic videos (3 per class, 4 JPEG frames each) under tmp_path/train/."""
    train_dir = tmp_path / "train"
    for cls_idx in range(NUM_CLASSES):
        cls_dir = train_dir / f"{cls_idx:03d}_Class{cls_idx}"
        for vid in range(VIDEOS_PER_CLASS):
            vid_dir = cls_dir / f"video_{vid:05d}"
            vid_dir.mkdir(parents=True)
            for f in range(NUM_FRAMES):
                pixels = np.random.randint(0, 255, (*FRAME_SIZE, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(vid_dir / f"frame_{f:03d}.jpg")
    return tmp_path


@pytest.fixture
def tiny_wds_dir(tiny_video_dir: Path, tmp_path: Path) -> Path:
    """Convert tiny_video_dir/train to WebDataset shards under tmp_path/wds/train/."""
    out = tmp_path / "wds"
    run_conversion(
        input_dir=tiny_video_dir / "train",
        output_dir=out / "train",
        shard_size=3,
    )
    return out


@pytest.fixture
def simple_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _total_samples_in_shards(shard_dir: Path) -> int:
    """Count .cls entries across all tar shards in shard_dir."""
    count = 0
    for tar_path in sorted(shard_dir.glob("shard-*.tar")):
        with tarfile.open(tar_path) as tf:
            count += sum(1 for m in tf.getmembers() if m.name.endswith(".cls"))
    return count


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Conversion tests
# ---------------------------------------------------------------------------

def test_conversion_creates_shards(tiny_video_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    n_shards = run_conversion(
        input_dir=tiny_video_dir / "train",
        output_dir=out,
        shard_size=3,
    )
    assert n_shards >= 1, "Expected at least one shard"
    tar_files = list(out.glob("shard-*.tar"))
    assert len(tar_files) == n_shards


def test_shard_contains_npz_and_cls(tiny_video_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_conversion(input_dir=tiny_video_dir / "train", output_dir=out, shard_size=10)
    first_tar = sorted(out.glob("shard-*.tar"))[0]
    with tarfile.open(first_tar) as tf:
        names = [m.name for m in tf.getmembers()]
    assert any(n.endswith(".cls") for n in names), "Missing .cls entries"
    assert any(n.endswith(".frames.npz") for n in names), "Missing .frames.npz entries"


def test_shard_sample_count_matches_input(tiny_video_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_conversion(input_dir=tiny_video_dir / "train", output_dir=out, shard_size=4)
    total = _total_samples_in_shards(out)
    expected = NUM_CLASSES * VIDEOS_PER_CLASS
    assert total == expected, f"Expected {expected} samples, got {total}"


def test_npz_contains_expected_frames(tiny_video_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_conversion(input_dir=tiny_video_dir / "train", output_dir=out, shard_size=10)
    first_tar = sorted(out.glob("shard-*.tar"))[0]
    with tarfile.open(first_tar) as tf:
        npz_member = next(m for m in tf.getmembers() if m.name.endswith(".frames.npz"))
        raw = tf.extractfile(npz_member).read()
    npz = np.load(io.BytesIO(raw))
    for i in range(NUM_FRAMES):
        assert f"frame_{i}" in npz.files, f"Missing frame_{i} in npz"
        # Each entry is raw JPEG bytes stored as uint8 array
        assert npz[f"frame_{i}"].dtype == np.uint8


# ---------------------------------------------------------------------------
# Streaming loader tests
# ---------------------------------------------------------------------------

def test_streaming_loader_tensor_shape(tiny_wds_dir: Path, simple_transform) -> None:
    shard_pattern = str(tiny_wds_dir / "train" / "shard-{000000..000010}.tar")
    loader = make_streaming_loader(
        shard_pattern=shard_pattern,
        num_frames=NUM_FRAMES,
        transform=simple_transform,
        batch_size=BATCH_SIZE,
        num_workers=0,
        is_train=False,
    )
    video, label = next(iter(loader))
    assert video.shape == (BATCH_SIZE, NUM_FRAMES, 3, *FRAME_SIZE), (
        f"Expected (B={BATCH_SIZE}, T={NUM_FRAMES}, C=3, H={FRAME_SIZE[0]}, W={FRAME_SIZE[1]}), got {video.shape}"
    )
    assert label.dtype == torch.int64


def test_streaming_loader_label_range(tiny_wds_dir: Path, simple_transform) -> None:
    shards = sorted(str(p) for p in (tiny_wds_dir / "train").glob("shard-*.tar"))
    loader = make_streaming_loader(
        shard_pattern=shards,
        num_frames=NUM_FRAMES,
        transform=simple_transform,
        batch_size=BATCH_SIZE,
        num_workers=0,
        is_train=False,
    )
    all_labels: set[int] = set()
    for _, label in loader:
        all_labels.update(label.tolist())
    assert all_labels <= set(range(NUM_CLASSES)), f"Labels out of range: {all_labels}"


def test_streaming_loader_all_samples_seen(tiny_wds_dir: Path, simple_transform) -> None:
    shards = sorted(str(p) for p in (tiny_wds_dir / "train").glob("shard-*.tar"))
    loader = make_streaming_loader(
        shard_pattern=shards,
        num_frames=NUM_FRAMES,
        transform=simple_transform,
        batch_size=1,
        num_workers=0,
        is_train=False,
    )
    total = sum(1 for _ in loader)
    expected = NUM_CLASSES * VIDEOS_PER_CLASS
    assert total == expected, f"Expected {expected} samples, saw {total}"


# ---------------------------------------------------------------------------
# Distributed split test
# ---------------------------------------------------------------------------

def test_split_by_node_disjoint(tiny_video_dir: Path, simple_transform) -> None:
    """Simulate two-node split: rank 0 and rank 1 should see disjoint shards."""
    # shard_size=1 → 1 video per shard → 6 shards total
    out = tiny_video_dir.parent / "wds_split_test"
    run_conversion(
        input_dir=tiny_video_dir / "train",
        output_dir=out / "train",
        shard_size=1,
    )
    shards = sorted(str(p) for p in (out / "train").glob("shard-*.tar"))

    def collect_labels(rank: int, world_size: int) -> list[int]:
        env_backup = {k: os.environ.get(k) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        try:
            loader = make_streaming_loader(
                shard_pattern=shards,
                num_frames=NUM_FRAMES,
                transform=simple_transform,
                batch_size=1,
                num_workers=0,
                is_train=False,
            )
            return [int(lbl.item()) for _, lbl in loader]
        finally:
            for k, v in env_backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    labels_rank0 = collect_labels(rank=0, world_size=2)
    labels_rank1 = collect_labels(rank=1, world_size=2)

    # Combined they should cover all samples; individually they should not overlap in count
    assert len(labels_rank0) > 0, "Rank 0 got no samples"
    assert len(labels_rank1) > 0, "Rank 1 got no samples"
    assert len(labels_rank0) + len(labels_rank1) == NUM_CLASSES * VIDEOS_PER_CLASS


# ---------------------------------------------------------------------------
# Training integration test
# ---------------------------------------------------------------------------

def test_streaming_train_one_epoch_cpu(tiny_wds_dir: Path, simple_transform) -> None:
    """One epoch of training through the streaming path on CPU."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"model": {"name": "cnn_baseline", "num_classes": NUM_CLASSES, "pretrained": False}})
    model = build_model(cfg)

    shards = sorted(str(p) for p in (tiny_wds_dir / "train").glob("shard-*.tar"))
    loader = make_streaming_loader(
        shard_pattern=shards,
        num_frames=NUM_FRAMES,
        transform=simple_transform,
        batch_size=BATCH_SIZE,
        num_workers=0,
        is_train=True,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    loss, acc = train_one_epoch(model, loader, loss_fn, optimizer, device)

    assert isinstance(loss, float) and loss >= 0.0, f"Bad loss: {loss}"
    assert isinstance(acc, float) and 0.0 <= acc <= 1.0, f"Bad acc: {acc}"
    assert not (loss != loss), "Loss is NaN"


# ---------------------------------------------------------------------------
# HTTP serving test
# ---------------------------------------------------------------------------

def test_http_server_serves_shard(tiny_wds_dir: Path, simple_transform) -> None:
    """Start a real HTTP server, load a shard via URL, verify tensor shape."""
    shard_dir = tiny_wds_dir / "train"
    port = _find_free_port()

    class _QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args): pass
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(shard_dir), **kwargs)

    server = HTTPServer(("127.0.0.1", port), _QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)  # let the server start

    try:
        shard_pattern = f"http://127.0.0.1:{port}/shard-{{000000..000010}}.tar"
        loader = make_streaming_loader(
            shard_pattern=shard_pattern,
            num_frames=NUM_FRAMES,
            transform=simple_transform,
            batch_size=BATCH_SIZE,
            num_workers=0,
            is_train=False,
        )
        video, label = next(iter(loader))
        assert video.shape == (BATCH_SIZE, NUM_FRAMES, 3, *FRAME_SIZE)
        assert label.dtype == torch.int64
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Launcher SSH command test
# ---------------------------------------------------------------------------

def test_build_ssh_command_no_tunnel() -> None:
    cmd = build_ssh_command("myhost", "echo hi")
    assert "myhost" in cmd
    assert "echo hi" in cmd
    assert "-R" not in cmd


def test_build_ssh_command_with_tunnel() -> None:
    cmd = build_ssh_command(
        host="myhost",
        remote_cmd="torchrun ...",
        remote_http_port=9888,
        local_http_port=8888,
    )
    assert "-R" in cmd
    assert "9888:localhost:8888" in cmd
    assert "myhost" in cmd


def test_build_ssh_command_tunnel_args_order() -> None:
    """Tunnel flag must appear before the host argument."""
    cmd = build_ssh_command("myhost", "cmd", remote_http_port=9888, local_http_port=8888)
    r_idx = cmd.index("-R")
    host_idx = cmd.index("myhost")
    assert r_idx < host_idx, "Tunnel flag must come before the hostname"
