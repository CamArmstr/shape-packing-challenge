#!/usr/bin/env python3
"""
swarm.py — Unified orchestrator for shape-packing workers.

Reads a config file defining the worker mix, launches all workers as
subprocesses, monitors health, and restarts crashed workers.

Usage:
    python swarm.py                          # uses swarm_config.json
    python swarm.py --config my_config.json
    python swarm.py --dry-run                # show what would launch
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / ".swarm_pids"
MASTER_PID_FILE = ROOT / ".swarm_master.pid"


@dataclass
class WorkerSpec:
    tag: str
    script: str
    args: list[str] = field(default_factory=list)
    restart: bool = True


@dataclass
class WorkerProcess:
    spec: WorkerSpec
    proc: Optional[subprocess.Popen] = None
    restarts: int = 0
    last_start: float = 0.0


def load_config(path: Path) -> list[WorkerSpec]:
    with open(path) as f:
        raw = json.load(f)
    specs = []
    for entry in raw["workers"]:
        specs.append(WorkerSpec(
            tag=entry.get("tag", entry["script"]),
            script=entry["script"],
            args=[str(a) for a in entry.get("args", [])],
            restart=entry.get("restart", True),
        ))
    return specs


def launch_worker(spec: WorkerSpec) -> subprocess.Popen:
    cmd = [sys.executable, str(ROOT / spec.script)] + spec.args
    log_path = ROOT / f".swarm_{spec.tag}.log"
    log_file = open(log_path, "a")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=lambda: os.nice(5),
    )
    # Keep reference to log file so it doesn't get GC'd/closed
    proc._log_file = log_file  # type: ignore
    return proc


def write_pid_file(workers: list[WorkerProcess]):
    lines = []
    for w in workers:
        if w.proc and w.proc.poll() is None:
            lines.append(f"{w.spec.tag} {w.proc.pid}")
    PID_FILE.write_text("\n".join(lines) + "\n")


def tail_logs(workers: list[WorkerProcess], lines: int = 5):
    for w in workers:
        log_path = ROOT / f".swarm_{w.spec.tag}.log"
        if not log_path.exists():
            continue
        with open(log_path) as f:
            all_lines = f.readlines()
        recent = all_lines[-lines:]
        for line in recent:
            print(f"  [{w.spec.tag}] {line.rstrip()}")


def run(config_path: Path, dry_run: bool = False):
    specs = load_config(config_path)
    print(f"Swarm config: {config_path.name} ({len(specs)} workers)")
    for s in specs:
        print(f"  {s.tag}: {s.script} {' '.join(s.args)}")

    if dry_run:
        print("\n(dry run — nothing launched)")
        return

    workers: list[WorkerProcess] = []
    shutting_down = False

    def handle_signal(signum, frame):
        nonlocal shutting_down
        shutting_down = True
        print(f"\nReceived signal {signum}, shutting down...")

    # Write master PID for watchdog
    MASTER_PID_FILE.write_text(str(os.getpid()))

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Launch all workers
    for spec in specs:
        proc = launch_worker(spec)
        wp = WorkerProcess(spec=spec, proc=proc, last_start=time.time())
        workers.append(wp)
        print(f"  Started {spec.tag} (PID {proc.pid})")

    write_pid_file(workers)
    print(f"\nSwarm running. PIDs in {PID_FILE.name}. Ctrl-C to stop.\n")

    # Monitor loop
    check_interval = 10
    report_interval = 120
    last_report = time.time()

    while not shutting_down:
        time.sleep(check_interval)

        # Check for crashed workers and restart
        for w in workers:
            if w.proc is None:
                continue
            ret = w.proc.poll()
            if ret is not None and not shutting_down:
                if w.spec.restart:
                    # Rate-limit restarts: wait at least 30s between restarts
                    elapsed = time.time() - w.last_start
                    if elapsed < 30:
                        time.sleep(max(0, 30 - elapsed))
                    w.restarts += 1
                    print(f"  [{w.spec.tag}] exited ({ret}), restarting (attempt {w.restarts})")
                    w.proc = launch_worker(w.spec)
                    w.last_start = time.time()
                    write_pid_file(workers)
                else:
                    print(f"  [{w.spec.tag}] exited ({ret}), no restart configured")
                    w.proc = None

        # Periodic status report
        now = time.time()
        if now - last_report >= report_interval:
            last_report = now
            alive = sum(1 for w in workers if w.proc and w.proc.poll() is None)
            total_restarts = sum(w.restarts for w in workers)
            elapsed_min = (now - workers[0].last_start) / 60 if workers else 0
            print(f"\n--- Swarm status: {alive}/{len(workers)} alive, {total_restarts} total restarts ---")
            for w in workers:
                status = "running" if w.proc and w.proc.poll() is None else "stopped"
                pid = w.proc.pid if w.proc and w.proc.poll() is None else "-"
                print(f"  {w.spec.tag}: {status} (PID {pid}, restarts={w.restarts})")
            print("--- Recent log lines ---")
            tail_logs(workers, lines=2)
            print()

    # Shutdown: terminate all workers
    print("Terminating workers...")
    for w in workers:
        if w.proc and w.proc.poll() is None:
            w.proc.terminate()

    # Wait up to 10s for graceful exit
    deadline = time.time() + 10
    for w in workers:
        if w.proc and w.proc.poll() is None:
            remaining = max(0, deadline - time.time())
            try:
                w.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                w.proc.kill()

    # Clean up
    for w in workers:
        if w.proc and hasattr(w.proc, '_log_file'):
            w.proc._log_file.close()

    if PID_FILE.exists():
        PID_FILE.unlink()
    if MASTER_PID_FILE.exists():
        MASTER_PID_FILE.unlink()

    print("Swarm stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified swarm orchestrator")
    parser.add_argument("--config", default=str(ROOT / "swarm_config.json"), help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Show config without launching")
    args = parser.parse_args()
    run(Path(args.config), dry_run=args.dry_run)
