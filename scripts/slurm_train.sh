#!/bin/bash
# SLURM batch script for multi-node distributed training.
# Each srun task = one Python process = one GPU (pure srun, no torchrun).
#
# Edit the SBATCH directives below, then submit:
#   sbatch scripts/slurm_train.sh experiment=baseline_from_scratch +data=vianney
#
# Logs go to slurm-<JOBID>.out by default.

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # 1 GPU per node on this cluster
#SBATCH --cpus-per-task=5          # 1 main + 4 DataLoader workers
#SBATCH --partition=SallesInfo
#SBATCH --job-name=train

set -euo pipefail

# Rendezvous: rank-0 node is the process group master.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR  WORLD_SIZE=$SLURM_NTASKS"

# Stage shards from NFS to per-node local /tmp to eliminate NFS read bottleneck.
# rsync runs once per node in parallel; ~2.4 GB per node, typically done in <60s.
NFS_SHARD_ROOT="$(pwd)/processed_data_wds"
STAGE_ROOT="/tmp/wds_shards_${SLURM_JOB_ID}"
echo "Staging shards to local /tmp on each node ($(date))..."
srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash "$(pwd)/scripts/stage_shards.sh" "$NFS_SHARD_ROOT" "$STAGE_ROOT"
echo "Staging done ($(date)). Launching training..."

# Override shard paths to point at local staged copies.
# Brace expansion passed as a literal string (no shell expansion inside double quotes).
TRAIN_SHARDS="$STAGE_ROOT/train/shard-{000000..000089}.tar"
VAL_SHARDS="$STAGE_ROOT/val/shard-{000000..000013}.tar"

srun --kill-on-bad-exit=1 \
    uv run python src/train.py "$@" \
    "dataset.train_shards='$TRAIN_SHARDS'" \
    "dataset.val_shards='$VAL_SHARDS'"
