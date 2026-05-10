#!/usr/bin/env python3
"""
Multi-node torchrun launcher via SSH.

Reads a hosts file, SSHes into each node in parallel, and launches torchrun
with the c10d rendezvous backend. The first host in the file becomes the
rendezvous master.

Usage:
    python scripts/launch_distributed.py \\
        --hosts scripts/hosts.txt \\
        --gpus_per_node 8 \\
        -- experiment=baseline_from_scratch data=vianney_wds

    # With WebDataset streaming (serves local shards via HTTP reverse tunnel):
    python scripts/launch_distributed.py \\
        --hosts scripts/hosts.txt \\
        --gpus_per_node 8 \\
        --wds_dir processed_data_wds \\
        -- experiment=baseline_from_scratch data=vianney_wds

    # Evaluate distributed:
    python scripts/launch_distributed.py \\
        --hosts scripts/hosts.txt \\
        --gpus_per_node 8 \\
        --script src/evaluate.py \\
        -- training.checkpoint_path=src/best_model.pt data=vianney_wds

Requires passwordless SSH from the machine running this script to every host
listed in hosts.txt. The torchrun command runs in --workdir on the remote node.

WebDataset streaming (--wds_dir):
  Starts a local HTTP server serving the shard directory, then adds an SSH
  reverse tunnel (-R remote_port:localhost:local_port) to each SSH connection.
  Each GPU node reads shards from http://localhost:<remote_port>/ which is
  transparently forwarded back to the local HTTP server. No data copy needed.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def probe_host(host: str, timeout: int = 5) -> bool:
    try:
        r = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=no", host, "exit", "0"],
            capture_output=True,
            timeout=timeout + 2,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def parse_hosts(hosts_file: str) -> list[str]:
    lines = Path(hosts_file).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def hostname_only(ssh_target: str) -> str:
    """Strip user@ prefix from an SSH target, leaving just the hostname/IP."""
    return ssh_target.split("@")[-1]


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
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"{prefix} {line}", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-node torchrun SSH dispatcher")
    parser.add_argument("--hosts", required=True, help="Path to hosts file (one hostname/IP per line)")
    parser.add_argument("--gpus_per_node", type=int, required=True, help="Number of GPUs per node")
    parser.add_argument("--port", type=int, default=29500, help="Rendezvous port on master node")
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
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Hydra overrides passed to the script")
    args = parser.parse_args()

    hosts = parse_hosts(args.hosts)
    if not hosts:
        sys.exit("No hosts found in hosts file.")

    print("Probing hosts...")
    with ThreadPoolExecutor(max_workers=len(hosts)) as ex:
        results = list(ex.map(lambda h: (h, probe_host(h)), hosts))
    for h, ok in results:
        if not ok:
            print(f"  SKIPPED {h} — SSH unreachable", file=sys.stderr)
    hosts = [h for h, ok in results if ok]
    if not hosts:
        sys.exit("No reachable hosts. Abort.")

    # Strip leading '--' separator from remainder args
    train_args = args.train_args[1:] if args.train_args and args.train_args[0] == "--" else args.train_args

    master = hosts[0]
    master_host = hostname_only(master)  # strip user@ for TCP endpoint
    nnodes = len(hosts)
    rdzv_id = str(uuid.uuid4())[:8]

    torchrun_cmd = (
        f"cd {args.workdir} && "
        f"git pull --ff-only && "
        f"{args.torchrun} "
        f"--nnodes={nnodes} "
        f"--nproc_per_node={args.gpus_per_node} "
        f"--rdzv_backend=c10d "
        f"--rdzv_endpoint={master_host}:{args.port} "
        f"--rdzv_id={rdzv_id} "
        f"{args.script} {' '.join(train_args)}"
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

    print(f"Launching on {nnodes} node(s), {args.gpus_per_node} GPU(s) each ({nnodes * args.gpus_per_node} total)")
    print(f"Master: {master_host}:{args.port}  rdzv_id: {rdzv_id}")
    print(f"Command: {torchrun_cmd}\n")

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
        sys.exit(1)

    signal.signal(signal.SIGINT, kill_all)
    signal.signal(signal.SIGTERM, kill_all)

    for i, host in enumerate(hosts):
        prefix = f"[node{i}|{host}]"
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
            for p in procs:
                if p.poll() is not None and p.returncode != 0:
                    print(f"\nNode exited with code {p.returncode} — killing all.", file=sys.stderr)
                    kill_all()
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
    if failed:
        for host, code in failed:
            print(f"FAILED [{host}] exit code {code}", file=sys.stderr)
        sys.exit(1)

    print("All nodes completed successfully.")


if __name__ == "__main__":
    main()
