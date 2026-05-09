import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

from src.dataset.video_dataset import (
    _list_frame_paths,
    _parse_class_index,
    _pick_frame_indices,
    VideoFrameDataset,
    collect_video_samples,
)


def _make_fake_dataset(root: Path, class_count: int = 2, videos_per_class: int = 1, frames_per_video: int = 4) -> Path:
    """Create a minimal fake dataset tree with real JPEG files."""
    for cls_idx in range(class_count):
        for vid_idx in range(videos_per_class):
            video_dir = root / f"{cls_idx:03d}_Class{cls_idx}" / f"video_{vid_idx}"
            video_dir.mkdir(parents=True)
            for f in range(frames_per_video):
                img = Image.new("RGB", (8, 8), color=(cls_idx * 40, vid_idx * 20, f * 10))
                img.save(video_dir / f"frame_{f:03d}.jpg", "JPEG")
    return root


_TRANSFORM = T.ToTensor()


# --- pure function tests (no I/O) ---

def test_parse_class_index_valid():
    assert _parse_class_index("017_Foo_Bar") == 17


def test_parse_class_index_no_prefix():
    assert _parse_class_index("no_prefix") is None


def test_pick_frame_indices_exact():
    assert _pick_frame_indices(4, 4) == [0, 1, 2, 3]


def test_pick_frame_indices_upsample():
    indices = _pick_frame_indices(4, 8)
    assert len(indices) == 8
    assert all(0 <= i <= 3 for i in indices)


def test_pick_frame_indices_single_frame():
    assert _pick_frame_indices(1, 4) == [0, 0, 0, 0]


# --- filesystem tests (use tmp_path) ---

def test_list_frame_paths_sorted(tmp_path):
    for name in ("frame_002.jpg", "frame_000.jpg", "frame_001.jpg"):
        (tmp_path / name).write_bytes(b"")
    paths = _list_frame_paths(tmp_path)
    assert [p.name for p in paths] == ["frame_000.jpg", "frame_001.jpg", "frame_002.jpg"]


def test_collect_video_samples_labels(tmp_path):
    _make_fake_dataset(tmp_path, class_count=2, frames_per_video=1)
    samples = collect_video_samples(tmp_path)
    labels = {label for _, label in samples}
    assert labels == {0, 1}


def test_dataset_len(tmp_path):
    _make_fake_dataset(tmp_path, class_count=2, videos_per_class=3)
    ds = VideoFrameDataset(tmp_path, num_frames=4, transform=_TRANSFORM)
    assert len(ds) == 6  # 2 classes × 3 videos


def test_dataset_item_shape(tmp_path):
    _make_fake_dataset(tmp_path, frames_per_video=4)
    T_frames = 4
    ds = VideoFrameDataset(tmp_path, num_frames=T_frames, transform=_TRANSFORM)
    video, _ = ds[0]
    assert video.shape == (T_frames, 3, 8, 8), f"expected ({T_frames}, 3, 8, 8), got {video.shape}"


def test_dataset_item_label_dtype(tmp_path):
    _make_fake_dataset(tmp_path, frames_per_video=1)
    ds = VideoFrameDataset(tmp_path, num_frames=1, transform=_TRANSFORM)
    _, label = ds[0]
    assert label.dtype == torch.int64
