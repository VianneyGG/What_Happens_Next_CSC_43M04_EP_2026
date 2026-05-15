#!/bin/bash
# Stage WebDataset shards from NFS to per-node local scratch before training.
#
# Run as a pre-step inside your SLURM batch script (one task per node):
#   STAGE_ROOT=${TMPDIR:-/tmp}/wds_shards
#   srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
#       bash scripts/stage_shards.sh /path/to/nfs/shards "$STAGE_ROOT"
#
# Then launch training with the staged paths:
#   srun uv run python src/train.py \
#       "dataset.train_shards=$STAGE_ROOT/train/*.tar" \
#       "dataset.val_shards=$STAGE_ROOT/val/*.tar" \
#       dataset.streaming=true

set -euo pipefail

SRC=${1:?Usage: stage_shards.sh <src_dir> [dst_dir]}
DST=${2:-${TMPDIR:-/tmp}/wds_shards}

mkdir -p "$DST"
rsync -a --checksum "$SRC/" "$DST/"
echo "[$(hostname)] staged $(find "$DST" -name '*.tar' | wc -l) shards -> $DST"
