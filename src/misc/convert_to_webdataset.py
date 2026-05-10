#!/usr/bin/env python3
"""
Convert a processed_data/ folder tree to WebDataset tar shards.

Each shard is a .tar file containing one entry per video:
  sample_XXXXXX.frames.npz  — np.savez_compressed with keys frame_0..frame_3 (raw JPEG bytes)
  sample_XXXXXX.cls         — class index as a plain text string

Usage (from repo root):
    uv run python src/misc/convert_to_webdataset.py \\
        --input_dir processed_data \\
        --output_dir processed_data_wds \\
        --shard_size 500

After conversion, the output directory contains:
    processed_data_wds/
      train/shard-000000.tar ...
      val/shard-000000.tar ...
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import webdataset as wds

# Allow running from repo root or from src/
_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))

from dataset.video_dataset import _list_frame_paths, collect_video_samples


def run_conversion(input_dir: Path, output_dir: Path, shard_size: int) -> int:
    """Convert one split directory to shards. Returns number of shards written."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_video_samples(input_dir)
    print(f"  {input_dir.name}: {len(samples)} videos → ", end="", flush=True)

    shard_pattern = str(output_dir / "shard-%06d.tar")
    num_written = 0

    with wds.ShardWriter(shard_pattern, maxcount=shard_size) as sink:
        for i, (video_dir, label) in enumerate(samples):
            frame_paths = _list_frame_paths(video_dir)
            if not frame_paths:
                continue

            # Read raw JPEG bytes — store compressed, decode at load time
            frame_bytes: dict[str, np.ndarray] = {}
            for j, path in enumerate(frame_paths):
                raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
                frame_bytes[f"frame_{j}"] = raw

            npz_buf = io.BytesIO()
            np.savez_compressed(npz_buf, **frame_bytes)

            sink.write({
                "__key__": f"sample_{i:06d}",
                "frames.npz": npz_buf.getvalue(),
                "cls": str(label),
            })
            num_written += 1

    num_shards = len(list(output_dir.glob("shard-*.tar")))
    print(f"{num_shards} shards")
    return num_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert video folder dataset to WebDataset shards")
    parser.add_argument("--input_dir", required=True, help="Root of processed_data/ (contains train/, val/, test/)")
    parser.add_argument("--output_dir", required=True, help="Output root for shards")
    parser.add_argument("--shard_size", type=int, default=500, help="Videos per shard (default: 500)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to convert")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    print(f"Converting {input_root} → {output_root}  (shard_size={args.shard_size})")
    for split in args.splits:
        split_in = input_root / split
        if not split_in.is_dir():
            print(f"  {split}: skipped (not found at {split_in})")
            continue
        run_conversion(split_in, output_root / split, args.shard_size)

    print("Done.")


if __name__ == "__main__":
    main()
