#!/bin/bash
# Wrapper around sbatch that excludes nodes with insufficient free GPU memory
# or without NFS access to the training shards.
#
# Usage:
#   bash scripts/sbatch_filtered.sh --nodes=5 scripts/slurm_train.sh [hydra overrides...]

set -euo pipefail

MIN_FREE_MIB=10000   # TSM needs ~8 GB; excludes workstation nodes (e.g. epervier: 1.8 GB free)
PARTITION=SallesInfo
NFS_CHECK_PATH="/users/eleves-b/2024/vianney.gauthier/Cours/ModalDL/What_Happens_Next_CSC_43M04_EP_2026/processed_data_wds"
NFS_PROBE_FILE="$NFS_CHECK_PATH/train/shard-000000.tar"

# Expand all idle node hostnames, excluding test/dev nodes (different subnet, slow NCCL init)
IDLE=$(sinfo -p "$PARTITION" -h -o "%N" -t idle | xargs scontrol show hostnames 2>/dev/null | grep -v '^test-')
# Always exclude test nodes regardless of probe result
ALWAYS_EXCLUDE=$(sinfo -p "$PARTITION" -h -o "%N" -t idle | xargs scontrol show hostnames 2>/dev/null | grep '^test-' | tr '\n' ',' | sed 's/,$//')
TOTAL=$(echo "$IDLE" | wc -w)
echo "Probing $TOTAL idle nodes (GPU memory + NFS, parallel SSH up to 20 at a time)..."

probe_node() {
    local node=$1
    local result free nfs_ok
    result=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes "$node" \
        "free=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1); \
         nfs=\$(dd if=\"$NFS_PROBE_FILE\" bs=512 count=1 of=/dev/null 2>/dev/null && echo ok || echo no); \
         echo \$free \$nfs" \
        2>/dev/null)
    free=$(echo "$result" | awk '{print $1}')
    nfs_ok=$(echo "$result" | awk '{print $2}')
    if [ -z "$free" ] || [ "$free" -lt "$MIN_FREE_MIB" ] || [ "$nfs_ok" != "ok" ]; then
        echo "$node"
    fi
}
export -f probe_node
export MIN_FREE_MIB NFS_CHECK_PATH NFS_PROBE_FILE

BAD=$(echo "$IDLE" | xargs -P 20 -n 1 bash -c 'probe_node "$@"' _ \
    | tr '\n' ',' | sed 's/,$//')

# Merge always-excluded list with probe-failed list
# Extract any --exclude= value already present in "$@" so we can merge it with
# probe-detected bad nodes (SLURM uses only the last --exclude when there are duplicates).
CALLER_EXCLUDE=""
for arg in "$@"; do
    case "$arg" in
        --exclude=*) CALLER_EXCLUDE="${arg#--exclude=}" ;;
    esac
done

ALL_EXCLUDE=$(echo "${ALWAYS_EXCLUDE},${BAD},${CALLER_EXCLUDE}" | tr ',' '\n' | sort -u | grep -v '^$' | tr '\n' ',' | sed 's/,$//')

# Remove any --exclude= from "$@" since we pass it merged below.
FILTERED_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --exclude=*) ;;
        *) FILTERED_ARGS+=("$arg") ;;
    esac
done

if [ -n "$ALL_EXCLUDE" ]; then
    echo "Excluding nodes (test nodes + low GPU memory or no NFS): $ALL_EXCLUDE"
    sbatch --exclude="$ALL_EXCLUDE" "${FILTERED_ARGS[@]}"
else
    echo "All idle nodes passed GPU memory and NFS checks."
    sbatch "${FILTERED_ARGS[@]}"
fi
