#!/usr/bin/env python3
"""
Multi-node torchrun launcher via SSH.

Reads a hosts file, SSHes into each node in parallel, and launches torchrun
with the c10d rendezvous backend. By default the first host in the file is the
rendezvous endpoint; use --rdzv_host to point it at a stable login node instead.

Usage:
    # Standard launch from local machine (bugatti = login node = stable rdzv host):
    python scripts/launch_distributed.py \\
        --hosts scripts/hosts.txt \\
        --rdzv_host bugatti.polytechnique.fr \\
        -- experiment=baseline_from_scratch data=vianney_wds_nfs

    # Skip code sync (editing directly on the cluster):
    python scripts/launch_distributed.py \\
        --hosts scripts/hosts.txt \\
        --rdzv_host bugatti.polytechnique.fr \\
        --no_sync \\
        -- experiment=baseline_from_scratch data=vianney_wds_nfs

    # Evaluate distributed:
    python scripts/launch_distributed.py \\
        --hosts scripts/hosts.txt \\
        --rdzv_host bugatti.polytechnique.fr \\
        --script src/evaluate.py \\
        -- training.checkpoint_path=src/best_model.pt data=vianney_wds_nfs

Requires passwordless SSH from the machine running this script to every host
listed in hosts.txt. The torchrun command runs in --workdir on the remote node.

Rendezvous host (--rdzv_host):
  torchrun's c10d backend starts its own TCPStore on this host — no external
  coordinator is started or needed. rdzv_host MUST be one of the compute nodes
  (the torchrun process there acts as the store server). If the specified host
  is excluded from compute (e.g. insufficient GPU memory), the launcher falls
  back to the first eligible compute node automatically.

Code sync (default):
  rsyncs the local repo to master:{workdir}. NFS propagates to all nodes instantly.
  No git commit/push needed to iterate. Skips .git/, .venv/, checkpoints, and data.
  Use --no_sync when editing directly on the cluster.
"""

from __future__ import annotations

import argparse
import dataclasses
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ---------------------------------------------------------------------------
# Unified host probe
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NodeInfo:
    host: str
    gpu_count: int   # 0 = unreachable or no GPU
    free_mib: int    # min free GPU memory in MiB across all GPUs; 0 on error
    data_ok: bool    # True if data_dir accessible (or no data_dir specified)
    skip_reason: str # empty string = include this node


def probe_node(host: str, data_dir: str | None, min_free_mib: int, timeout: int = 12) -> NodeInfo:
    """Single SSH session that collects GPU count, free memory, and data dir in one round-trip."""
    data_check = f"test -d '{data_dir}' && echo DATA_OK || echo DATA_MISSING" if data_dir else "echo DATA_OK"
    remote_cmd = (
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null; "
        "echo '---SEP---'; "
        + data_check
    )
    try:
        r = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=no", host, remote_cmd],
            capture_output=True, text=True, timeout=timeout + 2,
        )
    except subprocess.TimeoutExpired:
        return NodeInfo(host=host, gpu_count=0, free_mib=0, data_ok=False,
                        skip_reason="SSH timeout")

    if r.returncode != 0:
        return NodeInfo(host=host, gpu_count=0, free_mib=0, data_ok=False,
                        skip_reason="SSH unreachable")

    parts = r.stdout.split("---SEP---")
    gpu_section = parts[0].strip() if parts else ""
    data_section = parts[1].strip() if len(parts) > 1 else "DATA_MISSING"

    gpu_values = [int(v.strip()) for v in gpu_section.splitlines() if v.strip().isdigit()]
    gpu_count = len(gpu_values)
    free_mib = min(gpu_values) if gpu_values else 0
    data_ok = "DATA_OK" in data_section

    if gpu_count == 0:
        return NodeInfo(host=host, gpu_count=0, free_mib=0, data_ok=data_ok,
                        skip_reason="no GPUs detected")
    if free_mib < min_free_mib:
        return NodeInfo(host=host, gpu_count=gpu_count, free_mib=free_mib, data_ok=data_ok,
                        skip_reason=f"only {free_mib} MiB free (< {min_free_mib})")
    if data_dir and not data_ok:
        return NodeInfo(host=host, gpu_count=gpu_count, free_mib=free_mib, data_ok=False,
                        skip_reason=f"data dir not accessible: {data_dir}")
    return NodeInfo(host=host, gpu_count=gpu_count, free_mib=free_mib, data_ok=data_ok,
                    skip_reason="")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_hosts(hosts_file: str) -> list[str]:
    lines = Path(hosts_file).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def build_ssh_command(
    host: str,
    remote_cmd: str,
    remote_http_port: int | None = None,
    local_http_port: int = 8888,
) -> list[str]:
    """Build the SSH command list for launching on a remote node.

    If remote_http_port is given, adds an SSH reverse tunnel so that
    remote localhost:remote_http_port → local localhost:local_http_port.
    """
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
    if remote_http_port is not None:
        cmd += ["-R", f"{remote_http_port}:localhost:{local_http_port}"]
    cmd += [host, remote_cmd]
    return cmd


def stream_output(proc: subprocess.Popen, prefix: str) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(f"{prefix} {line}", end="", flush=True)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-node torchrun SSH dispatcher")
    parser.add_argument("--hosts", required=True, help="Path to hosts file (one hostname/IP per line)")
    parser.add_argument(
        "--min_nodes",
        type=int,
        default=1,
        help="Minimum live nodes for elastic torchrun; job continues as long as running >= this (default: 1).",
    )
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=1,
        help="Number of GPUs per node (default: 1 — matches RTX 4000 Ada Generation nodes on this cluster)",
    )
    parser.add_argument(
        "--min_free_gpu_mib",
        type=int,
        default=14000,
        help="Min free GPU memory in MiB to include a node (default: 14000 for 20 GB RTX 4000 Ada).",
    )
    parser.add_argument("--port", type=int, default=29500, help="Rendezvous port on master node")
    parser.add_argument(
        "--rdzv_host",
        default=None,
        help="Hostname/IP for the c10d rendezvous endpoint (default: first host in hosts file). "
             "Set to the login node hostname when running from tmux there: --rdzv_host $(hostname)",
    )
    parser.add_argument(
        "--workdir",
        default="/users/eleves-b/2024/vianney.gauthier/Cours/ModalDL/What_Happens_Next_CSC_43M04_EP_2026",
        help="Remote repo root (absolute path on the remote nodes).",
    )
    parser.add_argument(
        "--torchrun",
        default=".venv/bin/torchrun",
        help="torchrun binary path relative to --workdir (default: .venv/bin/torchrun)",
    )
    parser.add_argument(
        "--script",
        default="src/train.py",
        help="Python script to run relative to --workdir (default: src/train.py)",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Remote data directory to probe on each host; nodes that cannot access it are skipped.",
    )
    parser.add_argument(
        "--wds_dir",
        default=None,
        help="Local directory of WebDataset shards to serve via HTTP (enables streaming mode).",
    )
    parser.add_argument("--http_port", type=int, default=8888, help="Local HTTP server port (default: 8888)")
    parser.add_argument(
        "--remote_http_port",
        type=int,
        default=9888,
        help="Port forwarded on each remote node back to local HTTP server (default: 9888)",
    )
    parser.add_argument(
        "--no_sync",
        action="store_true",
        help="Skip code sync (use when editing directly on NFS or the cluster).",
    )
    # --- Training hyperparameters (RTX 4000 Ada defaults) ---
    parser.add_argument("--num_workers", type=int, default=2,
        help="DataLoader workers per GPU (default: 2, tuned for RTX 4000 Ada nodes)")
    parser.add_argument("--epochs", type=int, default=20,
        help="Training epochs (default: 20)")
    parser.add_argument("--model", type=str, default="cnn_baseline",
        help="Model name — passed as Hydra override model=<name> (default: cnn_baseline)")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Hydra overrides passed to the script")
    args = parser.parse_args()

    hosts = parse_hosts(args.hosts)
    if not hosts:
        sys.exit("No hosts found in hosts file.")

    # --- Single-pass probe: SSH reachability + GPU count + free memory + data dir ---
    print(f"Probing {len(hosts)} host(s) (GPU count, free memory >= {args.min_free_gpu_mib} MiB"
          + (f", data dir {args.data_dir}" if args.data_dir else "") + ") ...")
    with ThreadPoolExecutor(max_workers=len(hosts)) as ex:
        infos: list[NodeInfo] = list(ex.map(
            lambda h: probe_node(h, args.data_dir, args.min_free_gpu_mib),
            hosts,
        ))

    for info in infos:
        if info.skip_reason:
            print(f"  SKIP  {info.host} — {info.skip_reason}", file=sys.stderr)

    good = [info for info in infos if not info.skip_reason]
    skipped = len(infos) - len(good)
    skip_note = f"  ({skipped} skipped)" if skipped else ""
    print(f"  {len(good)}/{len(infos)} nodes ready{skip_note}")
    if not good:
        sys.exit("No usable nodes after probing. Abort.")

    hosts = [info.host for info in good]
    nnodes = len(hosts)

    # Clamp gpus_per_node to what's actually available
    gpus_per_node = args.gpus_per_node
    effective_gpus = min(info.gpu_count for info in good)
    if gpus_per_node > effective_gpus:
        print(
            f"  WARNING: --gpus_per_node={gpus_per_node} > available {effective_gpus};"
            f" clamping to {effective_gpus}.",
            file=sys.stderr,
        )
        gpus_per_node = effective_gpus

    # Validate min_nodes against available node count
    min_nodes = args.min_nodes
    if min_nodes > nnodes:
        print(
            f"  WARNING: --min_nodes={min_nodes} > usable nodes ({nnodes});"
            f" clamping to {nnodes}.",
            file=sys.stderr,
        )
        min_nodes = nnodes

    # Strip leading '--' separator from remainder args
    train_args = args.train_args[1:] if args.train_args and args.train_args[0] == "--" else args.train_args

    # Prepend hyperparameter defaults as Hydra overrides; explicit train_args take precedence
    injected = [
        f"training.num_workers={args.num_workers}",
        f"training.epochs={args.epochs}",
        f"model={args.model}",
    ]
    train_args = injected + train_args

    master = hosts[0]
    master_host = master.split("@")[-1]  # strip user@ — SSH target only
    rdzv_host = args.rdzv_host if args.rdzv_host else master_host
    rdzv_id = str(uuid.uuid4())[:16]

    # --- Code sync: rsync local repo to master (NFS propagates to all nodes) ---
    if not args.no_sync:
        rsync_src = str(Path(__file__).parent.parent.resolve()) + "/"
        rsync_dst = f"{master}:{args.workdir}/"
        print(f"Syncing code to {master}:{args.workdir} ...")
        sync = subprocess.run(
            ["rsync", "-a",
             "--filter=:- .gitignore",  # respect .gitignore (covers processed_data/, *.pt, etc.)
             "--exclude=.git/",
             "--exclude=.venv/",
             rsync_src, rsync_dst],
            capture_output=True, text=True,
        )
        if sync.returncode != 0:
            print(f"  WARNING: rsync failed — {sync.stderr.strip()}", file=sys.stderr)
        else:
            print("  Code synced.")
    else:
        print("Skipping code sync (--no_sync).")

    # torchrun's c10d backend manages its own TCPStore on rdzv_host — no external
    # coordinator is needed or compatible. When --rdzv_host is not a compute node
    # (e.g. login node excluded due to low GPU memory), fall back to the first
    # compute node so that the torchrun process there serves the store.
    if args.rdzv_host:
        good_hostnames = {info.host.split("@")[-1] for info in good}
        if args.rdzv_host not in good_hostnames:
            fallback = good[0].host.split("@")[-1]
            print(
                f"  WARNING: --rdzv_host={args.rdzv_host} is not a compute node "
                f"(excluded or not in hosts file). Falling back to {fallback}.",
                file=sys.stderr,
            )
            rdzv_host = fallback

    torchrun_cmd = (
        f"cd {shlex.quote(args.workdir)} && "
        f"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600 "
        f"{args.torchrun} "
        f"--nnodes={min_nodes}:{nnodes} "
        f"--nproc_per_node={gpus_per_node} "
        f"--rdzv_backend=c10d "
        f"--rdzv_endpoint={rdzv_host}:{args.port} "
        f"--rdzv_id={rdzv_id} "
        f"{args.script} {' '.join(shlex.quote(a) for a in train_args)}"
    )

    # --- Optional: local HTTP server + SSH reverse tunnel for WebDataset streaming ---
    http_proc: subprocess.Popen | None = None
    if args.wds_dir:
        wds_path = Path(args.wds_dir).resolve()
        if not wds_path.is_dir():
            sys.exit(f"--wds_dir not found: {wds_path}")
        http_proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(args.http_port)],
            cwd=wds_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"HTTP server started on port {args.http_port} (serving {wds_path})")
        print(f"  → forwarded to remote localhost:{args.remote_http_port} via SSH tunnel")

    print(f"Launching {nnodes}×{gpus_per_node} GPU  rdzv={rdzv_host}:{args.port}  id={rdzv_id[:8]}  min={min_nodes}")

    procs: list[subprocess.Popen] = []
    threads: list[threading.Thread] = []

    def kill_all(sig=None, frame=None) -> None:
        print("\nInterrupted — killing all nodes...")
        for p in procs:
            try:
                p.kill()
            except ProcessLookupError:
                pass
        if http_proc is not None:
            try:
                http_proc.kill()
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, kill_all)
    signal.signal(signal.SIGTERM, kill_all)

    for i, host in enumerate(hosts):
        short = host.split("@")[-1].split(".")[0]
        prefix = f"[{i}|{short}]"
        ssh_cmd = build_ssh_command(
            host=host,
            remote_cmd=torchrun_cmd,
            remote_http_port=args.remote_http_port if args.wds_dir else None,
            local_http_port=args.http_port,
        )
        proc = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append(proc)
        t = threading.Thread(target=stream_output, args=(proc, prefix), daemon=True)
        t.start()
        threads.append(t)

    def _monitor() -> None:
        while True:
            snapshots = [(p, p.poll()) for p in procs]
            running = sum(1 for _, rc in snapshots if rc is None)
            any_failed = any(rc is not None and rc != 0 for _, rc in snapshots)
            if any_failed and running < min_nodes:
                print(
                    f"\nOnly {running} node(s) alive < min_nodes={min_nodes} — killing all.",
                    file=sys.stderr,
                )
                for p, _ in snapshots:
                    try:
                        p.kill()
                    except ProcessLookupError:
                        pass
                return  # main thread unblocks from p.wait() and handles sys.exit
            if running == 0:
                return
            time.sleep(2)

    threading.Thread(target=_monitor, daemon=True).start()

    exit_codes = [p.wait() for p in procs]
    for t in threads:
        t.join(timeout=5)

    if http_proc is not None:
        http_proc.kill()
        print("HTTP server stopped.")


    failed = [(hosts[i], code) for i, code in enumerate(exit_codes) if code != 0]
    succeeded = sum(1 for code in exit_codes if code == 0)
    if failed:
        if succeeded >= min_nodes:
            print(
                f"Training completed on {succeeded}/{nnodes} node(s) "
                f"({len(failed)} node(s) dropped but above min_nodes={min_nodes})."
            )
        else:
            for host, code in failed:
                print(f"FAILED [{host}] exit code {code}", file=sys.stderr)
            sys.exit(1)
    else:
        print("All nodes completed successfully.")


if __name__ == "__main__":
    main()
