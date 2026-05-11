#!/usr/bin/env python3
"""
Smoke-test the rdzv coordinator path against bugatti directly.

Run from the local machine (same machine you launch distributed training from):
    python scripts/test_rdzv_coordinator.py
    python scripts/test_rdzv_coordinator.py --rdzv_host bugatti.polytechnique.fr --port 29500

Each test prints PASS / FAIL and exits with code 0 only if all pass.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

WORKDIR = "/users/eleves-b/2024/vianney.gauthier/Cours/ModalDL/What_Happens_Next_CSC_43M04_EP_2026"
TORCHRUN = ".venv/bin/torchrun"


def ssh(host: str, cmd: str, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
         "-o", "StrictHostKeyChecking=no", host, cmd],
        capture_output=True, text=True, timeout=timeout + 2,
    )


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdzv_host", default="bugatti.polytechnique.fr")
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--workdir", default=WORKDIR)
    parser.add_argument("--torchrun", default=TORCHRUN)
    args = parser.parse_args()

    venv_python = str(Path(args.torchrun).parent / "python")
    results: list[bool] = []

    print(f"\nTesting rdzv coordinator on {args.rdzv_host}:{args.port}")
    print(f"  venv python: {args.workdir}/{venv_python}\n")

    # ── Test 1: SSH reachability ──────────────────────────────────────────────
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
         "-o", "StrictHostKeyChecking=no", args.rdzv_host, "echo OK"],
        capture_output=True, text=True, timeout=12,
    )
    results.append(check("SSH to rdzv_host", r.returncode == 0, r.stderr.strip()))

    # ── Test 2: venv python exists ───────────────────────────────────────────
    r = ssh(args.rdzv_host, f"test -x {args.workdir}/{venv_python} && echo OK || echo MISSING")
    results.append(check("venv python exists on rdzv_host", "OK" in r.stdout,
                         r.stdout.strip() or r.stderr.strip()))

    # ── Test 3: torch importable ─────────────────────────────────────────────
    r = ssh(args.rdzv_host,
            f"cd {shlex.quote(args.workdir)} && {venv_python} -c 'import torch; print(torch.__version__)'",
            timeout=20)
    results.append(check("torch importable", r.returncode == 0,
                         r.stdout.strip() or r.stderr.strip()[:80]))

    # ── Test 4: port not already in use ──────────────────────────────────────
    r = ssh(args.rdzv_host, f"ss -tlnp 2>/dev/null | grep ':{args.port} ' || echo FREE")
    already_bound = r.returncode == 0 and "FREE" not in r.stdout
    results.append(check(f"port {args.port} free on rdzv_host", not already_bound,
                         r.stdout.strip() if already_bound else ""))

    # ── Test 5: start TCPStore coordinator ───────────────────────────────────
    coord_py = (
        f"import torch.distributed as dist, signal; "
        f"dist.TCPStore('{args.rdzv_host}', {args.port}, -1, True); "
        f"signal.pause()"
    )
    coord_cmd = f"cd {shlex.quote(args.workdir)} && {venv_python} -c {shlex.quote(coord_py)}"
    proc = subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
         args.rdzv_host, coord_cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    time.sleep(2.0)
    still_running = proc.poll() is None
    if not still_running:
        err = proc.stderr.read().decode().strip()
    else:
        err = ""
    results.append(check("TCPStore coordinator starts and stays alive", still_running,
                         err[:120] if err else ""))

    # ── Test 6: port reachable from local machine ─────────────────────────────
    if still_running:
        r = subprocess.run(
            ["nc", "-z", "-w", "5", args.rdzv_host, str(args.port)],
            capture_output=True, timeout=8,
        )
        results.append(check(f"port {args.port} reachable from this machine (nc)", r.returncode == 0))
    else:
        results.append(check(f"port {args.port} reachable from this machine (nc)",
                              False, "skipped — coordinator not running"))

    # ── Test 7: connect as TCPStore client ────────────────────────────────────
    if still_running:
        client_py = (
            f"import torch.distributed as dist; "
            f"s = dist.TCPStore('{args.rdzv_host}', {args.port}, -1, False); "
            f"s.set('smoke_test', 'ok'); "
            f"assert s.get('smoke_test') == b'ok'; "
            f"print('CLIENT_OK')"
        )
        r = subprocess.run(
            [sys.executable, "-c", client_py],
            capture_output=True, text=True, timeout=10,
        )
        results.append(check("TCPStore client connect + set/get from local machine",
                              "CLIENT_OK" in r.stdout,
                              r.stderr.strip()[:120] if r.returncode != 0 else ""))
    else:
        results.append(check("TCPStore client connect + set/get from local machine",
                              False, "skipped — coordinator not running"))

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if proc.poll() is None:
        proc.kill()
        proc.wait()

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
