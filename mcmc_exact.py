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
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.semicircle_packing.geometry import Semicircle, semicircles_overlap, farthest_boundary_point_from
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


@dataclass
class ArchiveEntry:
    state: WalkerState
    source: str
    signature: tuple


def gaussian(rng: random.Random, sigma: float) -> float:
    return rng.gauss(0.0, sigma)


def load_solution(path: Path) -> WalkerState:
    with open(path) as f:
        raw = json.load(f)

    xs = np.array([round(float(s["x"]), 8) for s in raw], dtype=float)
    ys = np.array([round(float(s["y"]), 8) for s in raw], dtype=float)
    ts = np.array([round(float(s["theta"]), 8) for s in raw], dtype=float)

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
        if (xs[idx] - xs[j]) ** 2 + (ys[idx] - ys[j]) ** 2 > 4.05:
            continue
        other = Semicircle(float(xs[j]), float(ys[j]), float(ts[j]))
        if semicircles_overlap(candidate, other):
            return False
    return True


def moved_cluster_valid(indices: list[int], xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> bool:
    moved = set(indices)
    for i in indices:
        candidate = Semicircle(float(xs[i]), float(ys[i]), float(ts[i]))
        for j in range(N):
            if j in moved:
                continue
            if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 > 4.05:
                continue
            other = Semicircle(float(xs[j]), float(ys[j]), float(ts[j]))
            if semicircles_overlap(candidate, other):
                return False
    return True


def quick_radius_estimate(state: WalkerState, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, changed_indices: list[int]) -> float:
    cx, cy, _ = state.mec
    est = state.score
    for i in changed_indices:
        sc = Semicircle(float(xs[i]), float(ys[i]), float(ts[i]))
        fx, fy = farthest_boundary_point_from(sc, cx, cy)
        est = max(est, math.hypot(fx - cx, fy - cy))
    return est


def cheap_prefilter(
    state: WalkerState,
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    changed_indices: list[int],
    *,
    score_slack: float,
    rescue_prob: float,
    rng: random.Random,
) -> bool:
    est = quick_radius_estimate(state, xs, ys, ts, changed_indices)
    if est <= state.score + score_slack:
        return True
    return rng.random() < rescue_prob


def rounded_payload(state: WalkerState, *, center: bool) -> list[dict[str, float]]:
    cx, cy = (state.mec[0], state.mec[1]) if center else (0.0, 0.0)
    return [
        {
            "x": round(float(state.xs[i] - cx), 8),
            "y": round(float(state.ys[i] - cy), 8),
            "theta": round(float(state.ts[i] % TWO_PI), 8),
        }
        for i in range(N)
    ]


def payload_to_state(payload: list[dict[str, float]]) -> Optional[WalkerState]:
    xs = np.array([round(float(s["x"]), 8) for s in payload], dtype=float)
    ys = np.array([round(float(s["y"]), 8) for s in payload], dtype=float)
    ts = np.array([round(float(s["theta"]), 8) for s in payload], dtype=float)
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


def state_signature(state: WalkerState) -> tuple:
    cx, cy, _ = state.mec
    rs = np.sqrt((state.xs - cx) ** 2 + (state.ys - cy) ** 2)
    radial = tuple(np.round(np.sort(rs), 3))

    nnd = []
    for i in range(N):
        d = np.hypot(state.xs[i] - state.xs, state.ys[i] - state.ys)
        d[i] = np.inf
        nnd.append(np.min(d))
    nn = tuple(np.round(np.sort(np.array(nnd)), 3))
    return radial + nn


def add_to_archive(archive: list[ArchiveEntry], state: WalkerState, source: str, max_size: int) -> bool:
    sig = state_signature(state)
    for i, entry in enumerate(archive):
        if entry.signature == sig:
            if state.score < entry.state.score - 1e-12:
                archive[i] = ArchiveEntry(
                    state=WalkerState(state.xs.copy(), state.ys.copy(), state.ts.copy(), state.score, state.mec),
                    source=source,
                    signature=sig,
                )
                archive.sort(key=lambda e: e.state.score)
                return True
            return False

    archive.append(
        ArchiveEntry(
            state=WalkerState(state.xs.copy(), state.ys.copy(), state.ts.copy(), state.score, state.mec),
            source=source,
            signature=sig,
        )
    )
    archive.sort(key=lambda e: e.state.score)
    del archive[max_size:]
    return True


def load_archive(max_size: int) -> list[ArchiveEntry]:
    archive: list[ArchiveEntry] = []
    candidates = [BEST_FILE] + [Path(p) for p in sorted(glob(str(SOLUTIONS_DIR / 'R*.json')))]
    for path in candidates:
        if not path.exists():
            continue
        try:
            state = load_solution(path)
        except Exception:
            continue
        add_to_archive(archive, state, path.name, max_size)
    return archive


def pick_archive_seed(archive: list[ArchiveEntry], rng: random.Random) -> ArchiveEntry:
    idx = min(int((rng.random() ** 2) * len(archive)), len(archive) - 1)
    return archive[idx]


def kicked_copy(state: WalkerState, rng: random.Random, kick_scale: float) -> Optional[WalkerState]:
    xs = state.xs.copy()
    ys = state.ys.copy()
    ts = state.ts.copy()
    for i in range(N):
        xs[i] += gaussian(rng, kick_scale)
        ys[i] += gaussian(rng, kick_scale)
        ts[i] = (ts[i] + gaussian(rng, kick_scale * math.pi)) % TWO_PI
    return build_state(xs, ys, ts)


def maybe_reload_best(current_best: WalkerState) -> WalkerState:
    latest = load_solution(BEST_FILE)
    return latest if latest.score < current_best.score - 1e-12 else current_best


def mode_defaults(mode: str) -> tuple[float, float, float, float, float]:
    if mode == "polisher":
        return 1e-7, 0.004, 1e-5, 0.015, 0.985
    return 5e-4, 0.08, 1e-4, 0.50, 0.995


def find_mec_boundary_pieces(state: WalkerState, top_k: int = 5) -> list[int]:
    """Identify pieces whose farthest boundary point is closest to the MEC radius."""
    cx, cy, r = state.mec
    dists = []
    for i in range(N):
        sc = Semicircle(float(state.xs[i]), float(state.ys[i]), float(state.ts[i]))
        fx, fy = farthest_boundary_point_from(sc, cx, cy)
        d = math.hypot(fx - cx, fy - cy)
        dists.append((r - d, i))  # smaller gap = closer to boundary
    dists.sort()
    return [idx for _, idx in dists[:top_k]]


def propose_mec_biased(
    state: WalkerState,
    sigma_xy: float,
    sigma_theta: float,
    rng: random.Random,
    mec_pieces: list[int],
    *,
    score_slack: float,
    rescue_prob: float,
) -> Optional[WalkerState]:
    """Propose a move biased toward MEC-boundary pieces, nudging them inward."""
    idx = rng.choice(mec_pieces)
    xs = state.xs.copy()
    ys = state.ys.copy()
    ts = state.ts.copy()

    cx, cy, _ = state.mec
    # Bias displacement toward MEC center
    dx_to_center = cx - xs[idx]
    dy_to_center = cy - ys[idx]
    dist = math.hypot(dx_to_center, dy_to_center)
    if dist > 1e-12:
        dx_to_center /= dist
        dy_to_center /= dist

    # 70% inward bias + 30% random
    inward_weight = 0.7
    rand_dx = gaussian(rng, sigma_xy)
    rand_dy = gaussian(rng, sigma_xy)
    inward_mag = abs(gaussian(rng, sigma_xy))
    xs[idx] += inward_weight * inward_mag * dx_to_center + (1 - inward_weight) * rand_dx
    ys[idx] += inward_weight * inward_mag * dy_to_center + (1 - inward_weight) * rand_dy
    ts[idx] = (ts[idx] + gaussian(rng, sigma_theta)) % TWO_PI

    if not moved_shape_valid(idx, xs, ys, ts):
        return None
    if not cheap_prefilter(state, xs, ys, ts, [idx], score_slack=score_slack, rescue_prob=rescue_prob, rng=rng):
        return None

    return build_state(xs, ys, ts)


def propose_single(
    state: WalkerState,
    idx: int,
    sigma_xy: float,
    sigma_theta: float,
    rng: random.Random,
    *,
    score_slack: float,
    rescue_prob: float,
) -> Optional[WalkerState]:
    xs = state.xs.copy()
    ys = state.ys.copy()
    ts = state.ts.copy()

    xs[idx] += gaussian(rng, sigma_xy)
    ys[idx] += gaussian(rng, sigma_xy)
    ts[idx] = (ts[idx] + gaussian(rng, sigma_theta)) % TWO_PI

    if not moved_shape_valid(idx, xs, ys, ts):
        return None
    if not cheap_prefilter(state, xs, ys, ts, [idx], score_slack=score_slack, rescue_prob=rescue_prob, rng=rng):
        return None

    return build_state(xs, ys, ts)


def choose_cluster(state: WalkerState, rng: random.Random, min_size: int, max_size: int) -> list[int]:
    center = rng.randrange(N)
    k = rng.randint(min_size, max_size)
    d = np.hypot(state.xs - state.xs[center], state.ys - state.ys[center])
    order = np.argsort(d)
    return [int(i) for i in order[:k]]


def propose_cluster(
    state: WalkerState,
    sigma_xy: float,
    sigma_theta: float,
    rng: random.Random,
    min_size: int,
    max_size: int,
    *,
    score_slack: float,
    rescue_prob: float,
) -> Optional[WalkerState]:
    indices = choose_cluster(state, rng, min_size, max_size)
    xs = state.xs.copy()
    ys = state.ys.copy()
    ts = state.ts.copy()

    cx = float(np.mean(xs[indices]))
    cy = float(np.mean(ys[indices]))
    dx = gaussian(rng, sigma_xy)
    dy = gaussian(rng, sigma_xy)
    dphi = gaussian(rng, sigma_theta * 0.35)
    cos_phi = math.cos(dphi)
    sin_phi = math.sin(dphi)

    for i in indices:
        rx = xs[i] - cx
        ry = ys[i] - cy
        xs[i] = cx + rx * cos_phi - ry * sin_phi + dx
        ys[i] = cy + rx * sin_phi + ry * cos_phi + dy
        ts[i] = (ts[i] + dphi) % TWO_PI

    if not moved_cluster_valid(indices, xs, ys, ts):
        return None
    if not cheap_prefilter(state, xs, ys, ts, indices, score_slack=score_slack, rescue_prob=rescue_prob, rng=rng):
        return None

    return build_state(xs, ys, ts)


def propose(
    state: WalkerState,
    idx: int,
    sigma_xy: float,
    sigma_theta: float,
    rng: random.Random,
    *,
    cluster_prob: float,
    cluster_min: int,
    cluster_max: int,
    score_slack: float,
    rescue_prob: float,
    mec_bias_prob: float = 0.0,
    mec_pieces: Optional[list[int]] = None,
) -> Optional[WalkerState]:
    r = rng.random()
    if mec_pieces and r < mec_bias_prob:
        return propose_mec_biased(
            state,
            sigma_xy,
            sigma_theta,
            rng,
            mec_pieces,
            score_slack=score_slack,
            rescue_prob=rescue_prob,
        )
    if r < mec_bias_prob + cluster_prob:
        return propose_cluster(
            state,
            sigma_xy,
            sigma_theta,
            rng,
            cluster_min,
            cluster_max,
            score_slack=score_slack,
            rescue_prob=rescue_prob,
        )
    return propose_single(
        state,
        idx,
        sigma_xy,
        sigma_theta,
        rng,
        score_slack=score_slack,
        rescue_prob=rescue_prob,
    )


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    temp, step_xy, min_step, max_step, cooling = mode_defaults(args.mode)
    temp = args.temp if args.temp is not None else temp
    step_xy = args.step if args.step is not None else step_xy
    if args.min_step is not None:
        min_step = args.min_step
    if args.max_step is not None:
        max_step = args.max_step
    sigma_theta_factor = args.theta_factor

    state = load_solution(Path(args.seed_file))
    best_local = WalkerState(state.xs.copy(), state.ys.copy(), state.ts.copy(), state.score, state.mec)
    global_best = maybe_reload_best(best_local)
    archive = load_archive(args.archive_size)
    add_to_archive(archive, state, 'seed', args.archive_size)

    print(f"[{args.tag}] seed score: {state.score:.6f}")
    print(f"[{args.tag}] mode={args.mode} temp={temp:.8f} step={step_xy:.6f} step_range=[{min_step},{max_step}] target_accept={args.target_accept:.3f} archive={len(archive)}")
    if args.mec_bias_prob > 0:
        print(f"[{args.tag}] MEC-biased proposals: {args.mec_bias_prob:.0%}")

    started = time.time()
    last_report = started
    last_improve_batch = 0
    mec_pieces: list[int] = []
    if args.mec_bias_prob > 0:
        mec_pieces = find_mec_boundary_pieces(state, top_k=args.mec_top_k)

    batch_idx = 0
    infinite = args.batches == 0
    while infinite or batch_idx < args.batches:
        batch_idx += 1
        stats = BatchStats()
        global_best = maybe_reload_best(global_best)
        add_to_archive(archive, global_best, 'global_best', args.archive_size)

        for _ in range(args.batch_size):
            stats.proposals += 1
            idx = rng.randrange(N)
            cluster_prob = args.cluster_prob if args.mode == 'explorer' else args.polisher_cluster_prob
            candidate = propose(
                state,
                idx,
                step_xy,
                step_xy * sigma_theta_factor,
                rng,
                cluster_prob=cluster_prob,
                cluster_min=args.cluster_min,
                cluster_max=args.cluster_max,
                score_slack=args.score_slack,
                rescue_prob=args.rescue_prob,
                mec_bias_prob=args.mec_bias_prob,
                mec_pieces=mec_pieces,
            )
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
                last_improve_batch = batch_idx
                add_to_archive(archive, best_local, args.tag, args.archive_size)
                was_global = save_if_global_best(best_local, args.tag)
                if was_global:
                    global_best = maybe_reload_best(global_best)
                    add_to_archive(archive, global_best, 'global_best', args.archive_size)
                # Recompute MEC pieces on improvement
                if args.mec_bias_prob > 0:
                    mec_pieces = find_mec_boundary_pieces(best_local, top_k=args.mec_top_k)
                print(
                    f"[{args.tag}] improvement batch={batch_idx} score={best_local.score:.6f} "
                    f"global={'yes' if was_global else 'no'} archive={len(archive)}"
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

        stale_batches = batch_idx - last_improve_batch
        if args.restart_every > 0 and stale_batches >= args.restart_every and archive:
            seed_entry = pick_archive_seed(archive, rng)
            restarted = kicked_copy(seed_entry.state, rng, args.kick_scale)
            if restarted is None:
                restarted = seed_entry.state
            state = WalkerState(restarted.xs.copy(), restarted.ys.copy(), restarted.ts.copy(), restarted.score, restarted.mec)
            if args.mode == 'explorer':
                temp = max(args.restart_temp, args.min_temp)
                step_xy = max(args.restart_step, min_step)
            else:
                state = WalkerState(global_best.xs.copy(), global_best.ys.copy(), global_best.ts.copy(), global_best.score, global_best.mec)
                temp = max(min(temp, args.min_temp * 10), args.min_temp)
                step_xy = min(max(args.restart_step * 0.25, min_step), max_step)
            last_improve_batch = batch_idx
            # Recompute MEC pieces on restart
            if args.mec_bias_prob > 0:
                mec_pieces = find_mec_boundary_pieces(state, top_k=args.mec_top_k)
            print(
                f"[{args.tag}] restart batch={batch_idx} from={seed_entry.source} seed_score={seed_entry.state.score:.6f} "
                f"new_score={state.score:.6f} archive={len(archive)}"
            )

        now = time.time()
        batch_label = f"{batch_idx}" if infinite else f"{batch_idx}/{args.batches}"
        if batch_idx == 1 or batch_idx % args.report_every == 0 or (now - last_report) > 30:
            elapsed = now - started
            evals = batch_idx * args.batch_size
            print(
                f"[{args.tag}] batch={batch_label} best={best_local.score:.6f} current={state.score:.6f} "
                f"accept={acceptance:.3f} valid={stats.valid_proposals}/{stats.proposals} "
                f"step={step_xy:.6f} temp={temp:.8f} evals={evals} elapsed={elapsed:.1f}s archive={len(archive)}"
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
    parser.add_argument("--min-step", type=float, default=None, help="Override min step size (default from mode)")
    parser.add_argument("--max-step", type=float, default=None, help="Override max step size (default from mode)")
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--min-temp", type=float, default=1e-7)
    parser.add_argument("--theta-factor", type=float, default=math.pi)
    parser.add_argument("--target-accept", type=float, default=0.234)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--polisher-cooling", type=float, default=0.995)
    parser.add_argument("--archive-size", type=int, default=24)
    parser.add_argument("--restart-every", type=int, default=25)
    parser.add_argument("--restart-temp", type=float, default=0.002)
    parser.add_argument("--restart-step", type=float, default=0.05)
    parser.add_argument("--kick-scale", type=float, default=0.03)
    parser.add_argument("--cluster-prob", type=float, default=0.25)
    parser.add_argument("--cluster-min", type=int, default=2)
    parser.add_argument("--cluster-max", type=int, default=4)
    parser.add_argument("--polisher-cluster-prob", type=float, default=0.0, help="Cluster move prob in polisher mode")
    parser.add_argument("--score-slack", type=float, default=0.01)
    parser.add_argument("--rescue-prob", type=float, default=0.05)
    parser.add_argument("--mec-bias-prob", type=float, default=0.0, help="Prob of MEC-boundary-biased proposal")
    parser.add_argument("--mec-top-k", type=int, default=5, help="Number of MEC-boundary pieces to target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="mcmc_exact")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
