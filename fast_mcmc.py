#!/usr/bin/env python3
"""
fast_mcmc.py

Approximate high-throughput MCMC search inspired by the browser optimizer.
Uses cheap overlap + cheap MEC in the inner loop, and exact official
validation only when a candidate looks save-worthy.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
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

from src.semicircle_packing.geometry import Semicircle
from src.semicircle_packing.scoring import minimum_enclosing_circle, validate_and_score

ROOT = Path(__file__).resolve().parent
BEST_FILE = ROOT / "best_solution.json"
LOCK_FILE = ROOT / "best_solution.json.lock"
SOLUTIONS_DIR = ROOT / "solutions"
SCRATCH_DIR = SOLUTIONS_DIR / "scratch_candidates"
RESTART_POOL_DIR = SOLUTIONS_DIR / "restart_pool"
N = 15
TWO_PI = 2 * math.pi
ARC_STEPS = 30


@dataclass
class FastState:
    xs: np.ndarray
    ys: np.ndarray
    ts: np.ndarray
    approx_score: float


def load_json_state(path: Path) -> FastState:
    with open(path) as f:
        raw = json.load(f)
    xs = np.array([float(s["x"]) for s in raw], dtype=float)
    ys = np.array([float(s["y"]) for s in raw], dtype=float)
    ts = np.array([float(s["theta"]) for s in raw], dtype=float)
    return FastState(xs, ys, ts, approx_score(xs, ys, ts))


def archive_exact_score(path: Path) -> float:
    if path.name == BEST_FILE.name:
        try:
            state = load_json_state(path)
            valid, exact_score, _ = exact_result_for_state(state)
            if valid and exact_score is not None:
                return float(exact_score)
        except Exception:
            return float('inf')
        return float('inf')

    stem = path.stem
    if stem.startswith('R'):
        try:
            return float(stem[1:].split('_', 1)[0])
        except ValueError:
            return float('inf')
    return float('inf')


def load_archive_paths(limit: int, restart_pool_limit: int) -> list[Path]:
    paths = [BEST_FILE]
    solution_paths = sorted(
        (Path(p) for p in glob.glob(str(SOLUTIONS_DIR / "R*.json"))),
        key=lambda p: (archive_exact_score(p), p.name),
    )
    if limit > 0 and len(solution_paths) > limit:
        solution_paths = solution_paths[:limit]
    paths.extend(solution_paths)

    restart_pool_paths = sorted(
        (Path(p) for p in glob.glob(str(RESTART_POOL_DIR / "R*.json"))),
        key=lambda p: (archive_exact_score(p), p.name),
    )
    if restart_pool_limit > 0 and len(restart_pool_paths) > restart_pool_limit:
        restart_pool_paths = restart_pool_paths[:restart_pool_limit]
    paths.extend(restart_pool_paths)

    deduped = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen or not p.exists():
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def filter_restart_paths(archive_paths: list[Path], score_slack: float) -> list[Path]:
    if not archive_paths:
        return [BEST_FILE]
    scored = sorted(((archive_exact_score(p), p) for p in archive_paths), key=lambda item: (item[0], item[1].name))
    best_score = scored[0][0]
    if not math.isfinite(best_score):
        return archive_paths
    filtered = [p for score, p in scored if score <= best_score + score_slack]
    return filtered or [p for _, p in scored[: min(8, len(scored))]]


def restart_diversity_bucket(path: Path, score_bucket_decimals: int) -> str:
    exact_score = archive_exact_score(path)
    if math.isfinite(exact_score):
        exact_bucket = f'{round(exact_score, score_bucket_decimals):.{score_bucket_decimals}f}'
    else:
        exact_bucket = 'inf'

    if path.name == BEST_FILE.name:
        return f'{exact_bucket}|best'

    stem = path.stem
    if stem.startswith('R'):
        tail = stem.split('_', 1)
        if len(tail) == 2:
            source = tail[1]
            source = source.rsplit('_b', 1)[0]
            if source:
                return f'{exact_bucket}|{source}'
    return f'{exact_bucket}|{stem}'


def pick_restart_path(
    archive_paths: list[Path],
    rng: random.Random,
    best_bias: float,
    recent_names: list[str],
    recent_window: int,
    score_slack: float,
    recent_score_buckets: list[str],
    score_bucket_decimals: int,
) -> Path:
    if not archive_paths:
        return BEST_FILE
    filtered = filter_restart_paths(archive_paths, score_slack)
    if rng.random() < best_bias and BEST_FILE in filtered:
        return BEST_FILE

    grouped: dict[str, list[tuple[int, Path]]] = {}
    for rank, path in enumerate(filtered):
        bucket = restart_diversity_bucket(path, score_bucket_decimals)
        grouped.setdefault(bucket, []).append((rank, path))

    bucket_weights: list[tuple[float, str]] = []
    recent_bucket_slice = recent_score_buckets[-recent_window:]
    for bucket, items in grouped.items():
        best_rank = min(rank for rank, _ in items)
        base = 1.0 / (1.0 + best_rank)
        recent_penalty = 0.4 if bucket in recent_bucket_slice else 1.0
        bucket_weights.append((base * recent_penalty, bucket))

    total = sum(weight for weight, _ in bucket_weights)
    if total <= 0:
        chosen_bucket = rng.choice(list(grouped))
    else:
        roll = rng.random() * total
        chosen_bucket = bucket_weights[-1][1]
        for weight, bucket in bucket_weights:
            roll -= weight
            if roll <= 0:
                chosen_bucket = bucket
                break

    weighted: list[tuple[float, Path]] = []
    for rank, path in grouped[chosen_bucket]:
        base = 1.0 / (1.0 + rank)
        recent_penalty = 0.35 if path.name in recent_names[-recent_window:] else 1.0
        weighted.append((base * recent_penalty, path))

    total = sum(weight for weight, _ in weighted)
    if total <= 0:
        return grouped[chosen_bucket][rng.randrange(len(grouped[chosen_bucket]))][1]

    roll = rng.random() * total
    for weight, path in weighted:
        roll -= weight
        if roll <= 0:
            return path
    return weighted[-1][1]


def approx_points(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> np.ndarray:
    pts = []
    for i in range(N):
        pts.append((xs[i], ys[i]))
        for k in range(ARC_STEPS + 1):
            a = ts[i] - math.pi / 2 + (math.pi * k) / ARC_STEPS
            pts.append((xs[i] + math.cos(a), ys[i] + math.sin(a)))
    return np.array(pts, dtype=float)


def approx_score(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> float:
    pts = approx_points(xs, ys, ts)
    _, _, r = minimum_enclosing_circle(pts)
    return float(r)


def dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def segment_circle_intersections(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, r: float):
    vx, vy = bx - ax, by - ay
    wx, wy = ax - cx, ay - cy
    A = vx * vx + vy * vy
    B = 2 * (vx * wx + vy * wy)
    C = wx * wx + wy * wy - r * r
    disc = B * B - 4 * A * C
    if disc < 0:
        return []
    root = math.sqrt(max(0.0, disc))
    out = []
    for t in ((-B - root) / (2 * A), (-B + root) / (2 * A)):
        if -1e-9 <= t <= 1 + 1e-9:
            out.append((ax + t * vx, ay + t * vy))
    return out


def circles_intersections(x1: float, y1: float, x2: float, y2: float, r: float):
    dx, dy = x2 - x1, y2 - y1
    d2 = dx * dx + dy * dy
    d = math.sqrt(d2)
    if d > 2 * r + 1e-9 or d < 1e-9:
        return []
    a = d / 2
    h2 = r * r - a * a
    if h2 < 0:
        return []
    h = math.sqrt(max(0.0, h2))
    mx, my = x1 + a * dx / d, y1 + a * dy / d
    px, py = -dy / d, dx / d
    return [(mx + h * px, my + h * py), (mx - h * px, my - h * py)]


def quick_overlap(i: int, j: int, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> bool:
    if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 > 4.000001:
        return False

    xi, yi, ti = xs[i], ys[i], ts[i]
    xj, yj, tj = xs[j], ys[j], ts[j]
    dxi, dyi = math.cos(ti), math.sin(ti)
    dxj, dyj = math.cos(tj), math.sin(tj)

    if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 < 1e-8 and (dxi * dxj + dyi * dyj) > -0.999:
        return True

    fi1 = (xi + dyi, yi - dxi)
    fi2 = (xi - dyi, yi + dxi)
    fj1 = (xj + dyj, yj - dxj)
    fj2 = (xj - dyj, yj + dxj)

    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    c1, c2 = ccw(fi1, fi2, fj1), ccw(fi1, fi2, fj2)
    c3, c4 = ccw(fj1, fj2, fi1), ccw(fj1, fj2, fi2)
    if ((c1 > 1e-6 and c2 < -1e-6) or (c1 < -1e-6 and c2 > 1e-6)) and ((c3 > 1e-6 and c4 < -1e-6) or (c3 < -1e-6 and c4 > 1e-6)):
        return True

    for px, py in segment_circle_intersections(fi1[0], fi1[1], fi2[0], fi2[1], xj, yj, 1.0):
        if (px - xi) ** 2 + (py - yi) ** 2 < 1 - 1e-6 and dot(px - xj, py - yj, dxj, dyj) > 1e-6:
            return True
    for px, py in segment_circle_intersections(fj1[0], fj1[1], fj2[0], fj2[1], xi, yi, 1.0):
        if (px - xj) ** 2 + (py - yj) ** 2 < 1 - 1e-6 and dot(px - xi, py - yi, dxi, dyi) > 1e-6:
            return True
    for px, py in circles_intersections(xi, yi, xj, yj, 1.0):
        if dot(px - xi, py - yi, dxi, dyi) > 1e-6 and dot(px - xj, py - yj, dxj, dyj) > 1e-6:
            return True
    return False


def moved_valid(indices: list[int], xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> bool:
    moved = set(indices)
    for i in indices:
        for j in range(N):
            if i == j:
                continue
            if j in moved and j < i:
                continue
            if quick_overlap(i, j, xs, ys, ts):
                return False
    return True


def choose_cluster(xs: np.ndarray, ys: np.ndarray, rng: random.Random, min_size: int, max_size: int) -> list[int]:
    center = rng.randrange(N)
    k = rng.randint(min_size, max_size)
    d = np.hypot(xs - xs[center], ys - ys[center])
    order = np.argsort(d)
    return [int(i) for i in order[:k]]


def propose(state: FastState, rng: random.Random, step_xy: float, step_theta: float, cluster_prob: float, cluster_min: int, cluster_max: int) -> tuple[Optional[FastState], int]:
    xs = state.xs.copy()
    ys = state.ys.copy()
    ts = state.ts.copy()
    if rng.random() < cluster_prob:
        indices = choose_cluster(xs, ys, rng, cluster_min, cluster_max)
        cx = float(np.mean(xs[indices]))
        cy = float(np.mean(ys[indices]))
        dx = rng.gauss(0.0, step_xy)
        dy = rng.gauss(0.0, step_xy)
        dphi = rng.gauss(0.0, step_theta * 0.35)
        c, s = math.cos(dphi), math.sin(dphi)
        for i in indices:
            rx, ry = xs[i] - cx, ys[i] - cy
            xs[i] = cx + rx * c - ry * s + dx
            ys[i] = cy + rx * s + ry * c + dy
            ts[i] = (ts[i] + dphi) % TWO_PI
    else:
        idx = rng.randrange(N)
        indices = [idx]
        xs[idx] += rng.gauss(0.0, step_xy)
        ys[idx] += rng.gauss(0.0, step_xy)
        ts[idx] = (ts[idx] + rng.gauss(0.0, step_theta)) % TWO_PI

    if not moved_valid(indices, xs, ys, ts):
        return None, 0
    return FastState(xs, ys, ts, approx_score(xs, ys, ts)), 1


def rounded_state_payload(state: FastState) -> list[dict[str, float]]:
    return [
        {"x": round(float(state.xs[i]), 6), "y": round(float(state.ys[i]), 6), "theta": round(float(state.ts[i] % TWO_PI), 6)}
        for i in range(N)
    ]


def state_signature(state: FastState, decimals: int) -> str:
    parts: list[str] = []
    for i in range(N):
        parts.append(
            f'{round(float(state.xs[i]), decimals):.{decimals}f},'
            f'{round(float(state.ys[i]), decimals):.{decimals}f},'
            f'{round(float(state.ts[i] % TWO_PI), decimals):.{decimals}f}'
        )
    return '|'.join(parts)


def exact_result_for_state(state: FastState) -> tuple[bool, Optional[float], Optional[list[dict[str, float]]]]:
    rounded = rounded_state_payload(state)
    sol = [Semicircle(d["x"], d["y"], d["theta"]) for d in rounded]
    result = validate_and_score(sol)
    if not result.valid or result.score is None or result.mec is None:
        return False, None, None

    cx, cy, _ = result.mec
    centered = [
        {"x": round(float(sol[i].x - cx), 6), "y": round(float(sol[i].y - cy), 6), "theta": round(float(sol[i].theta % TWO_PI), 6)}
        for i in range(N)
    ]
    centered_sol = [Semicircle(d["x"], d["y"], d["theta"]) for d in centered]
    centered_result = validate_and_score(centered_sol)
    if not centered_result.valid or centered_result.score is None:
        return False, None, None
    return True, centered_result.score, centered


def retain_scratch_candidate(state: FastState, tag: str, batch: int, reason: str, valid: bool, exact_score: Optional[float], limit: int) -> Optional[Path]:
    if limit <= 0:
        return None

    payload = rounded_state_payload(state)
    approx_text = f'{state.approx_score:.6f}'
    exact_text = f'{exact_score:.6f}' if exact_score is not None else 'invalid'
    stem = f'{reason}_{"valid" if valid else "invalid"}_A{approx_text}_E{exact_text}_{tag}_b{batch}'
    out = SCRATCH_DIR / f'{stem}.json'
    if out.exists():
        return None

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)

    def score_from_name(path: Path, marker: str) -> float:
        stem = path.stem
        if marker not in stem:
            return float('inf')
        try:
            token = stem.split(marker, 1)[1].split('_', 1)[0]
        except IndexError:
            return float('inf')
        if token == 'invalid':
            return float('inf')
        try:
            return float(token)
        except ValueError:
            return float('inf')

    valid_candidates: list[tuple[float, float, Path]] = []
    invalid_candidates: list[tuple[float, float, Path]] = []
    for path in SCRATCH_DIR.glob('*.json'):
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        exact_value = score_from_name(path, '_E')
        approx_value = score_from_name(path, '_A')
        if '_valid_' in path.name:
            valid_candidates.append((exact_value, -mtime, path))
        else:
            invalid_candidates.append((approx_value, -mtime, path))

    valid_candidates.sort(key=lambda item: (item[0], item[1], item[2].name))
    invalid_candidates.sort(key=lambda item: (item[0], item[1], item[2].name))

    valid_quota = min(limit, max(8, limit // 4))
    keep = {path for _, _, path in valid_candidates[:valid_quota]}
    remaining = max(0, limit - len(keep))
    keep.update(path for _, _, path in invalid_candidates[:remaining])
    for stale in list(SCRATCH_DIR.glob('*.json')):
        if stale not in keep:
            stale.unlink(missing_ok=True)
    return out


def retain_restart_pool_candidate(centered: list[dict[str, float]], exact_score: float, tag: str, batch: int, limit: int) -> Optional[Path]:
    if limit <= 0:
        return None

    RESTART_POOL_DIR.mkdir(parents=True, exist_ok=True)
    out = RESTART_POOL_DIR / f'R{exact_score:.6f}_{tag}_b{batch}.json'
    if not out.exists():
        with open(out, 'w') as f:
            json.dump(centered, f, indent=2)

    candidates = sorted(RESTART_POOL_DIR.glob('R*.json'), key=lambda p: (archive_exact_score(p), p.name))
    for stale in candidates[limit:]:
        stale.unlink(missing_ok=True)
    return out


def exact_save_gate(state: FastState, tag: str) -> tuple[bool, bool, Optional[float]]:
    valid, exact_score, centered = exact_result_for_state(state)
    if not valid or exact_score is None or centered is None:
        return False, False, None

    with open(LOCK_FILE, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        current = load_json_state(BEST_FILE)
        current_exact = validate_and_score([Semicircle(float(current.xs[i]), float(current.ys[i]), float(current.ts[i])) for i in range(N)])
        current_score = current_exact.score if current_exact.valid and current_exact.score is not None else float('inf')
        if exact_score >= current_score - 1e-12:
            return False, True, exact_score

        tmp = BEST_FILE.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(centered, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, BEST_FILE)

        SOLUTIONS_DIR.mkdir(exist_ok=True)
        out = SOLUTIONS_DIR / f'R{exact_score:.6f}.json'
        with open(out, 'w') as f:
            json.dump(centered, f, indent=2)

        try:
            import subprocess
            subprocess.run(['git', 'add', 'best_solution.json', str(out)], cwd=ROOT, capture_output=True)
            subprocess.run(['git', 'commit', '-m', f'best: R={exact_score:.6f} ({tag}, fast)'], cwd=ROOT, capture_output=True)
        except Exception:
            pass
        return True, True, exact_score


def build_restart_state(seed: FastState, rng: random.Random, kick_scale: float, attempts: int) -> FastState:
    base = FastState(seed.xs.copy(), seed.ys.copy(), seed.ts.copy(), seed.approx_score)
    if kick_scale <= 0:
        return base

    best_candidate: Optional[FastState] = None
    for _ in range(max(1, attempts)):
        xs = seed.xs.copy()
        ys = seed.ys.copy()
        ts = seed.ts.copy()
        for i in range(N):
            xs[i] += rng.gauss(0.0, kick_scale)
            ys[i] += rng.gauss(0.0, kick_scale)
            ts[i] = (ts[i] + rng.gauss(0.0, kick_scale * math.pi)) % TWO_PI
        if not moved_valid(list(range(N)), xs, ys, ts):
            continue
        candidate = FastState(xs, ys, ts, approx_score(xs, ys, ts))
        if best_candidate is None or candidate.approx_score < best_candidate.approx_score:
            best_candidate = candidate

    return best_candidate or base


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    archive_paths = load_archive_paths(args.archive_limit, args.restart_pool_limit)
    state = load_json_state(Path(args.seed_file))
    best_seen = FastState(state.xs.copy(), state.ys.copy(), state.ts.copy(), state.approx_score)
    best_exact_anchor = FastState(state.xs.copy(), state.ys.copy(), state.ts.copy(), state.approx_score)
    temp = args.temp
    step = args.step
    start = time.time()
    gate_checks = 0
    gate_skips = 0
    last_gate_batch = 0
    recent_restart_names: list[str] = []
    recent_restart_score_buckets: list[str] = []
    recent_invalid_gate_batches: dict[str, int] = {}

    print(f'[{args.tag}] seed approx={state.approx_score:.6f}')
    print(f'[{args.tag}] mode={args.mode} step={step:.6f} temp={temp:.6f}')

    if args.mode == 'polisher':
        cluster_prob = 0.0
        min_step, max_step = 1e-5, 0.01
    else:
        cluster_prob = args.cluster_prob
        min_step, max_step = 5e-4, 0.2

    accepted = valid = total = 0
    for batch in range(1, args.batches + 1):
        batch_valid = batch_accept = 0
        for _ in range(args.batch_size):
            total += 1
            candidate, ok = propose(state, rng, step, step * math.pi, cluster_prob, args.cluster_min, args.cluster_max)
            batch_valid += ok
            valid += ok
            if candidate is None:
                continue
            delta = candidate.approx_score - state.approx_score
            if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
                state = candidate
                batch_accept += 1
                accepted += 1
                if state.approx_score < best_seen.approx_score - 1e-12:
                    best_seen = FastState(state.xs.copy(), state.ys.copy(), state.ts.copy(), state.approx_score)
                    saved, exact_valid, exact_score = exact_save_gate(best_seen, args.tag)
                    if not saved:
                        scratch_path = retain_scratch_candidate(
                            best_seen,
                            args.tag,
                            batch,
                            reason='improvement',
                            valid=exact_valid,
                            exact_score=exact_score,
                            limit=args.scratch_retain_limit,
                        )
                        if exact_valid and exact_score is not None:
                            _, _, centered = exact_result_for_state(best_seen)
                            if centered is not None:
                                retain_restart_pool_candidate(centered, exact_score, args.tag, batch, args.restart_pool_limit)
                                best_exact_anchor = FastState(
                                    best_seen.xs.copy(),
                                    best_seen.ys.copy(),
                                    best_seen.ts.copy(),
                                    best_seen.approx_score,
                                )
                    else:
                        scratch_path = None
                        best_exact_anchor = FastState(
                            best_seen.xs.copy(),
                            best_seen.ys.copy(),
                            best_seen.ts.copy(),
                            best_seen.approx_score,
                        )
                    exact_text = f'{exact_score:.6f}' if exact_score is not None else 'invalid'
                    scratch_text = f' scratch={scratch_path.name}' if scratch_path is not None else ''
                    print(
                        f'[{args.tag}] improvement batch={batch} approx={best_seen.approx_score:.6f} '
                        f'anchor={best_exact_anchor.approx_score:.6f} exact={exact_text} '
                        f'exact_save={"yes" if saved else "no"}{scratch_text}'
                    )
                elif (
                    args.gate_window > 0.0
                    and state.approx_score <= best_exact_anchor.approx_score + args.gate_window
                    and batch - last_gate_batch >= args.gate_log_interval
                ):
                    sig = state_signature(state, args.gate_signature_decimals)
                    last_invalid_batch = recent_invalid_gate_batches.get(sig)
                    if last_invalid_batch is not None and batch - last_invalid_batch < args.invalid_gate_cooldown:
                        gate_skips += 1
                        last_gate_batch = batch
                        print(
                            f'[{args.tag}] gate_skip batch={batch} current_approx={state.approx_score:.6f} '
                            f'anchor_approx={best_exact_anchor.approx_score:.6f} since_invalid={batch - last_invalid_batch} '
                            f'gate_skips={gate_skips}'
                        )
                    else:
                        gate_checks += 1
                        last_gate_batch = batch
                        valid, exact_score, centered = exact_result_for_state(state)
                        if not valid:
                            recent_invalid_gate_batches[sig] = batch
                        elif sig in recent_invalid_gate_batches:
                            del recent_invalid_gate_batches[sig]
                        scratch_path = retain_scratch_candidate(
                            state,
                            args.tag,
                            batch,
                            reason='gate',
                            valid=valid,
                            exact_score=exact_score,
                            limit=args.scratch_retain_limit,
                        )
                        if valid and exact_score is not None and centered is not None:
                            retain_restart_pool_candidate(centered, exact_score, args.tag, batch, args.restart_pool_limit)
                        exact_text = f'{exact_score:.6f}' if exact_score is not None else 'invalid'
                        scratch_text = f' scratch={scratch_path.name}' if scratch_path is not None else ''
                        print(
                            f'[{args.tag}] gate_probe batch={batch} current_approx={state.approx_score:.6f} '
                            f'anchor_approx={best_exact_anchor.approx_score:.6f} exact={exact_text} valid={"yes" if valid else "no"} '
                            f'gate_checks={gate_checks}{scratch_text}'
                        )

        ar = batch_accept / batch_valid if batch_valid else 0.0
        step = min(step * 1.03, max_step) if ar > args.target_accept else max(step * 0.97, min_step)
        temp = max(temp * args.cooling, args.min_temp)

        if args.mode == 'explorer' and batch % args.restart_every == 0:
            archive_paths = load_archive_paths(args.archive_limit, args.restart_pool_limit)
            seed_path = pick_restart_path(
                archive_paths,
                rng,
                args.best_bias,
                recent_restart_names,
                args.restart_recent_window,
                args.restart_score_slack,
                recent_restart_score_buckets,
                args.restart_score_bucket_decimals,
            )
            recent_restart_names.append(seed_path.name)
            if len(recent_restart_names) > max(1, args.restart_recent_window):
                recent_restart_names = recent_restart_names[-args.restart_recent_window:]
            recent_restart_score_buckets.append(
                restart_diversity_bucket(seed_path, args.restart_score_bucket_decimals)
            )
            if len(recent_restart_score_buckets) > max(1, args.restart_recent_window):
                recent_restart_score_buckets = recent_restart_score_buckets[-args.restart_recent_window:]
            seed = load_json_state(seed_path)
            state = build_restart_state(seed, rng, args.kick_scale, args.restart_kick_attempts)
            temp = args.restart_temp
            step = args.restart_step
            print(
                f'[{args.tag}] restart batch={batch} seed={seed_path.name} approx={state.approx_score:.6f} '
                f'archive={len(archive_paths)} recent={len(set(recent_restart_names))}/{max(1, args.restart_recent_window)} '
                f'score_buckets={len(set(recent_restart_score_buckets))}/{max(1, args.restart_recent_window)} '
                f'kick_attempts={args.restart_kick_attempts} '
                f'slack={args.restart_score_slack:.6f}'
            )

        if batch == 1 or batch % args.report_every == 0:
            elapsed = time.time() - start
            print(
                f'[{args.tag}] batch={batch}/{args.batches} best_seen={best_seen.approx_score:.6f} '
                f'anchor={best_exact_anchor.approx_score:.6f} current={state.approx_score:.6f} '
                f'accept={ar:.3f} valid={batch_valid}/{args.batch_size} step={step:.6f} '
                f'temp={temp:.6f} elapsed={elapsed:.1f}s'
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--seed-file', default=str(BEST_FILE))
    p.add_argument('--mode', choices=['explorer', 'polisher'], default='explorer')
    p.add_argument('--batches', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=400)
    p.add_argument('--step', type=float, default=0.05)
    p.add_argument('--temp', type=float, default=0.003)
    p.add_argument('--min-temp', type=float, default=1e-7)
    p.add_argument('--cooling', type=float, default=0.995)
    p.add_argument('--target-accept', type=float, default=0.234)
    p.add_argument('--restart-every', type=int, default=30)
    p.add_argument('--restart-step', type=float, default=0.06)
    p.add_argument('--restart-temp', type=float, default=0.004)
    p.add_argument('--kick-scale', type=float, default=0.05)
    p.add_argument('--restart-kick-attempts', type=int, default=6)
    p.add_argument('--cluster-prob', type=float, default=0.35)
    p.add_argument('--cluster-min', type=int, default=2)
    p.add_argument('--cluster-max', type=int, default=4)
    p.add_argument('--report-every', type=int, default=10)
    p.add_argument('--gate-window', type=float, default=0.002)
    p.add_argument('--gate-log-interval', type=int, default=5)
    p.add_argument('--archive-limit', type=int, default=64)
    p.add_argument('--best-bias', type=float, default=0.35)
    p.add_argument('--restart-recent-window', type=int, default=6)
    p.add_argument('--restart-score-slack', type=float, default=0.0015)
    p.add_argument('--restart-score-bucket-decimals', type=int, default=4)
    p.add_argument('--invalid-gate-cooldown', type=int, default=48)
    p.add_argument('--gate-signature-decimals', type=int, default=4)
    p.add_argument('--scratch-retain-limit', type=int, default=64)
    p.add_argument('--restart-pool-limit', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tag', default='fast_mcmc')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
