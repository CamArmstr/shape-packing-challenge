#!/usr/bin/env python3
"""
mcmc_exact.py

Exact single-walker MCMC / SA optimizer for the semicircle packing challenge.

Goals for v1:
- use the official scorer only for score updates
- use exact overlap checks before scoring proposals
- adapt Gaussian proposal scale toward a target acceptance rate
- support explorer and polisher modes
- save new official bests under a file lock
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
from src.semicircle_packing.scoring import validate_and_score

ROOT = Path(__file__).resolve().parent
BEST_FILE = ROOT / "best_solution.json"
LOCK_FILE = ROOT / "best_solution.json.lock"
SOLUTIONS_DIR = ROOT / "solutions"
TWO_PI = 2 * math.pi
N = 15


@dataclass
class WalkerState:
    xs: np.ndarray
    ys: np.ndarray
    ts: np.ndarray
    score: float
    mec: tuple[float, float, float]


@dataclass
class BatchStats:
    proposals: int = 0
    valid_proposals: int = 0
    accepted: int = 0
    improved: int = 0


def gaussian(rng: random.Random, sigma: float) -> float:
    return rng.gauss(0.0, sigma)


def load_solution(path: Path) -> WalkerState:
    with open(path) as f:
        raw = json.load(f)

    xs = np.array([round(float(s["x"]), 6) for s in raw], dtype=float)
    ys = np.array([round(float(s["y"]), 6) for s in raw], dtype=float)
    ts = np.array([round(float(s["theta"]), 6) for s in raw], dtype=float)

    state = build_state(xs, ys, ts)
    if state is None:
        raise ValueError(f"Seed solution at {path} is invalid")
    return state


def build_state(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> Optional[WalkerState]:
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    result = validate_and_score(sol)
    if not result.valid or result.score is None or result.mec is None:
        return None
    return WalkerState(xs=xs.copy(), ys=ys.copy(), ts=ts.copy(), score=float(result.score), mec=result.mec)


def moved_shape_valid(idx: int, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> bool:
    candidate = Semicircle(float(xs[idx]), float(ys[idx]), float(ts[idx]))
    for j in range(N):
        if j == idx:
            continue
        other = Semicircle(float(xs[j]), float(ys[j]), float(ts[j]))
        if semicircles_overlap(candidate, other):
            return False
    return True


def rounded_payload(state: WalkerState, *, center: bool) -> list[dict[str, float]]:
    cx, cy = (state.mec[0], state.mec[1]) if center else (0.0, 0.0)
    return [
        {
            "x": round(float(state.xs[i] - cx), 6),
            "y": round(float(state.ys[i] - cy), 6),
            "theta": round(float(state.ts[i] % TWO_PI), 6),
        }
        for i in range(N)
    ]


def payload_to_state(payload: list[dict[str, float]]) -> Optional[WalkerState]:
    xs = np.array([round(float(s["x"]), 6) for s in payload], dtype=float)
    ys = np.array([round(float(s["y"]), 6) for s in payload], dtype=float)
    ts = np.array([round(float(s["theta"]), 6) for s in payload], dtype=float)
    return build_state(xs, ys, ts)


def make_saveable_state(state: WalkerState) -> tuple[Optional[WalkerState], Optional[list[dict[str, float]]], str]:
    centered = rounded_payload(state, center=True)
    centered_state = payload_to_state(centered)
    if centered_state is not None:
        return centered_state, centered, "centered"

    uncentered = rounded_payload(state, center=False)
    uncentered_state = payload_to_state(uncentered)
    if uncentered_state is not None:
        return uncentered_state, uncentered, "uncentered"

    return None, None, "invalid-after-rounding"


def archive_solution(payload: list[dict[str, float]], saved_state: WalkerState) -> Path:
    SOLUTIONS_DIR.mkdir(exist_ok=True)
    out = SOLUTIONS_DIR / f"R{saved_state.score:.6f}.json"
    if not out.exists():
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
    return out


def save_if_global_best(state: WalkerState, tag: str) -> bool:
    with open(LOCK_FILE, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)

        current_best = load_solution(BEST_FILE)
        saveable_state, payload, save_mode = make_saveable_state(state)
        if saveable_state is None or payload is None:
            return False
        if saveable_state.score >= current_best.score - 1e-12:
            return False

        tmp_file = BEST_FILE.with_suffix(BEST_FILE.suffix + ".tmp")
        with open(tmp_file, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, BEST_FILE)

        archive_path = archive_solution(payload, saveable_state)

        try:
            import subprocess
            subprocess.run(["git", "add", "best_solution.json"], cwd=ROOT, capture_output=True)
            subprocess.run(["git", "add", str(archive_path)], cwd=ROOT, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"best: R={saveable_state.score:.6f} ({tag}, {save_mode})"],
                cwd=ROOT,
                capture_output=True,
            )
        except Exception:
            pass

        return True


def mode_defaults(mode: str) -> tuple[float, float, float, float, float]:
    if mode == "polisher":
        return 1e-7, 0.004, 1e-5, 0.015, 0.985
    return 5e-4, 0.08, 1e-4, 0.50, 0.995


def propose(state: WalkerState, idx: int, sigma_xy: float, sigma_theta: float, rng: random.Random) -> Optional[WalkerState]:
    xs = state.xs.copy()
    ys = state.ys.copy()
    ts = state.ts.copy()

    xs[idx] += gaussian(rng, sigma_xy)
    ys[idx] += gaussian(rng, sigma_xy)
    ts[idx] = (ts[idx] + gaussian(rng, sigma_theta)) % TWO_PI

    if not moved_shape_valid(idx, xs, ys, ts):
        return None

    next_state = build_state(xs, ys, ts)
    return next_state


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    temp, step_xy, min_step, max_step, cooling = mode_defaults(args.mode)
    temp = args.temp if args.temp is not None else temp
    step_xy = args.step if args.step is not None else step_xy
    sigma_theta_factor = args.theta_factor

    state = load_solution(Path(args.seed_file))
    best_local = WalkerState(state.xs.copy(), state.ys.copy(), state.ts.copy(), state.score, state.mec)

    print(f"[{args.tag}] seed score: {state.score:.6f}")
    print(f"[{args.tag}] mode={args.mode} temp={temp:.8f} step={step_xy:.6f} target_accept={args.target_accept:.3f}")

    started = time.time()
    last_report = started

    for batch_idx in range(1, args.batches + 1):
        stats = BatchStats()

        for _ in range(args.batch_size):
            stats.proposals += 1
            idx = rng.randrange(N)
            candidate = propose(state, idx, step_xy, step_xy * sigma_theta_factor, rng)
            if candidate is None:
                continue

            stats.valid_proposals += 1
            delta = candidate.score - state.score
            accept = delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12))
            if not accept:
                continue

            state = candidate
            stats.accepted += 1

            if state.score < best_local.score - 1e-12:
                best_local = WalkerState(state.xs.copy(), state.ys.copy(), state.ts.copy(), state.score, state.mec)
                stats.improved += 1
                was_global = save_if_global_best(best_local, args.tag)
                print(
                    f"[{args.tag}] improvement batch={batch_idx} score={best_local.score:.6f} "
                    f"global={'yes' if was_global else 'no'}"
                )

        acceptance = stats.accepted / stats.valid_proposals if stats.valid_proposals else 0.0
        if acceptance > args.target_accept:
            step_xy = min(step_xy * 1.02, max_step)
        else:
            step_xy = max(step_xy * 0.98, min_step)

        if args.mode == "explorer":
            temp = max(temp * cooling, args.min_temp)
        else:
            temp = max(temp * args.polisher_cooling, args.min_temp)

        now = time.time()
        if batch_idx == 1 or batch_idx % args.report_every == 0 or (now - last_report) > 30:
            elapsed = now - started
            evals = batch_idx * args.batch_size
            print(
                f"[{args.tag}] batch={batch_idx}/{args.batches} best={best_local.score:.6f} current={state.score:.6f} "
                f"accept={acceptance:.3f} valid={stats.valid_proposals}/{stats.proposals} "
                f"step={step_xy:.6f} temp={temp:.8f} evals={evals} elapsed={elapsed:.1f}s"
            )
            last_report = now

    print(f"[{args.tag}] done. best local score: {best_local.score:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact MCMC optimizer for semicircle packing")
    parser.add_argument("--seed-file", default=str(BEST_FILE), help="JSON seed solution file")
    parser.add_argument("--mode", choices=["explorer", "polisher"], default="explorer")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--step", type=float, default=None)
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--min-temp", type=float, default=1e-7)
    parser.add_argument("--theta-factor", type=float, default=math.pi)
    parser.add_argument("--target-accept", type=float, default=0.234)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--polisher-cooling", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="mcmc_exact")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
