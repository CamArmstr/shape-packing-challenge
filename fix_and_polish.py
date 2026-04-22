#!/usr/bin/env python3
"""
fix_and_polish.py

Load the competitor's exported solution, fix the overlap between shapes 9 & 13
by nudging them apart, re-center via MEC, then GJK polish.
Run many random nudge trials in parallel.
"""

import sys, os, json, time, math, fcntl
import numpy as np
import multiprocessing as mp

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from src.semicircle_packing.scoring import validate_and_score, minimum_enclosing_circle
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
from gjk_numba import semicircle_gjk_signed_dist
import numba as nb

N = 15
TWO_PI = 2 * math.pi
BEST_FILE = 'best_solution.json'
LOCK_PATH  = BEST_FILE + '.lock'
SEED_FILE  = 'imported_competitor.json'

# ── GJK helpers ──────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def gjk_overlap_full(xs, ys, ts):
    n = xs.shape[0]
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = semicircle_gjk_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < 0.0:
                e += d * d
    return e

@nb.njit(cache=True)
def r_single_nb(x, y, t):
    r = math.sqrt((x + math.cos(t))**2 + (y + math.sin(t))**2)
    px, py = -math.sin(t), math.cos(t)
    r1 = math.sqrt((x + px)**2 + (y + py)**2)
    r2 = math.sqrt((x - px)**2 + (y - py)**2)
    if r1 > r: r = r1
    if r2 > r: r = r2
    for k in range(24):
        a = t - math.pi / 2 + math.pi * k / 23
        rx_ = math.sqrt((x + math.cos(a))**2 + (y + math.sin(a))**2)
        if rx_ > r: r = rx_
    return r

@nb.njit(cache=True)
def r_max_nb(xs, ys, ts):
    rm = 0.0
    for i in range(xs.shape[0]):
        ri = r_single_nb(xs[i], ys[i], ts[i])
        if ri > rm: rm = ri
    return rm

@nb.njit(cache=True)
def gjk_polish(xs, ys, ts, n_steps, T_start, T_end, step_init, seed):
    np.random.seed(seed)
    n = xs.shape[0]
    cur_xs, cur_ys, cur_ts = xs.copy(), ys.copy(), ts.copy()
    cur_ovlp = gjk_overlap_full(cur_xs, cur_ys, cur_ts)
    cur_R    = r_max_nb(cur_xs, cur_ys, cur_ts)
    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
    best_R = 1e18
    if cur_ovlp < 1e-14:
        best_R = cur_R
    step = step_init
    acc = 0
    for s in range(n_steps):
        T = T_start * (T_end / T_start) ** (s / max(n_steps - 1, 1))
        idx = s % n
        ox, oy, ot = cur_xs[idx], cur_ys[idx], cur_ts[idx]
        cur_xs[idx] += np.random.randn() * step
        cur_ys[idx] += np.random.randn() * step
        cur_ts[idx] += np.random.randn() * step
        new_ovlp = gjk_overlap_full(cur_xs, cur_ys, cur_ts)
        new_R    = r_max_nb(cur_xs, cur_ys, cur_ts)
        if cur_ovlp > 1e-14:
            dE = new_ovlp - cur_ovlp
        else:
            dE = 1e9 if new_ovlp > 1e-14 else new_R - cur_R
        if dE < 0 or np.random.random() < math.exp(-dE / max(T, 1e-14)):
            cur_ovlp = new_ovlp
            cur_R    = new_R
            acc += 1
            if new_ovlp < 1e-14 and new_R < best_R:
                best_R = new_R
                best_xs[:] = cur_xs[:]
                best_ys[:] = cur_ys[:]
                best_ts[:] = cur_ts[:]
        else:
            cur_xs[idx] = ox; cur_ys[idx] = oy; cur_ts[idx] = ot
        if (s + 1) % 2000 == 0:
            rate = acc / 2000
            if rate > 0.3: step = min(step * 1.05, 0.05)
            else:          step = max(step * 0.95, 1e-5)
            acc = 0
    return best_xs, best_ys, best_ts, best_R


def load_best_score():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    shapes = [Semicircle(s['x'], s['y'], s['theta']) for s in raw]
    result = validate_and_score(shapes)
    return result.score if result.valid else float('inf')


def try_save_if_better(xs, ys, ts, wid, cycle):
    shapes = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    result = validate_and_score(shapes)
    if not result.valid or result.score is None:
        return False, None
    score = result.score
    with open(LOCK_PATH, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            disk = load_best_score()
            if score < disk:
                data = [{'x': float(xs[i]), 'y': float(ys[i]), 'theta': float(ts[i])} for i in range(N)]
                with open(BEST_FILE, 'w') as bf:
                    json.dump(data, bf, indent=2)
                print(f'[worker {wid}] *** NEW BEST: {score:.6f} (was {disk:.6f}) cycle={cycle} ***', flush=True)
                return True, score
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return False, score


def center_by_mec(xs, ys, ts):
    """Translate so MEC center is at origin."""
    # Sample boundary points
    pts = []
    for i in range(N):
        for k in range(64):
            a = ts[i] - math.pi/2 + math.pi * k / 63
            pts.append((xs[i] + math.cos(a), ys[i] + math.sin(a)))
        pts.append((xs[i] - math.sin(ts[i]), ys[i] + math.cos(ts[i])))
        pts.append((xs[i] + math.sin(ts[i]), ys[i] - math.cos(ts[i])))
    # Bounding-box center as MEC approximation
    px = [p[0] for p in pts]
    py = [p[1] for p in pts]
    cx = (min(px) + max(px)) / 2
    cy = (min(py) + max(py)) / 2
    return xs - cx, ys - cy


def worker(wid):
    os.nice(10)
    # JIT warmup
    _ = gjk_overlap_full(np.zeros(3), np.zeros(3), np.zeros(3))
    _ = gjk_polish(np.zeros(N), np.zeros(N), np.zeros(N), 200, 0.01, 0.001, 0.003, 42)

    with open(SEED_FILE) as f:
        raw = json.load(f)
    seed_xs = np.array([s['x'] for s in raw], dtype=np.float64)
    seed_ys = np.array([s['y'] for s in raw], dtype=np.float64)
    seed_ts = np.array([s['theta'] for s in raw], dtype=np.float64)

    # Center by bounding-box MEC approximation
    seed_xs, seed_ys = center_by_mec(seed_xs, seed_ys, seed_ts)
    print(f'[worker {wid}] Centered, approx R={r_max_nb(seed_xs, seed_ys, seed_ts):.6f}', flush=True)

    rng = np.random.RandomState(wid * 9973 + int(time.time()) % 100000)
    cycle = 0

    while True:
        cycle += 1
        xs = seed_xs.copy()
        ys = seed_ys.copy()
        ts = seed_ts.copy()

        # Nudge shapes 9 and 13 apart along their separation vector
        # plus random perturbation to explore the neighborhood
        dx = xs[9] - xs[13]
        dy = ys[9] - ys[13]
        dist = math.sqrt(dx*dx + dy*dy) + 1e-9
        nudge = rng.uniform(0.005, 0.05)
        # Push 9 and 13 apart
        xs[9]  += dx/dist * nudge + rng.randn() * 0.01
        ys[9]  += dy/dist * nudge + rng.randn() * 0.01
        ts[9]  += rng.randn() * 0.05
        xs[13] -= dx/dist * nudge + rng.randn() * 0.01
        ys[13] -= dy/dist * nudge + rng.randn() * 0.01
        ts[13] += rng.randn() * 0.05

        # Also randomly perturb all shapes slightly to explore
        if rng.random() < 0.5:
            perturb = rng.uniform(0.001, 0.02)
            xs += rng.randn(N) * perturb
            ys += rng.randn(N) * perturb
            ts += rng.randn(N) * perturb * 0.5

        # Quick validity check
        still_overlap = any(
            semicircles_overlap(Semicircle(xs[i], ys[i], ts[i]), Semicircle(xs[j], ys[j], ts[j]))
            for i, j in [(9,13)]
        )

        # GJK polish pass 1
        p_xs, p_ys, p_ts, gjk_R = gjk_polish(
            xs, ys, ts, 3_000_000, 0.004, 5e-6, 0.004,
            rng.randint(0, 99999)
        )

        if gjk_R > 1e17:
            print(f'[worker {wid}] cycle {cycle} no valid GJK result', flush=True)
            continue

        saved, score = try_save_if_better(p_xs, p_ys, p_ts, wid, cycle)
        print(f'[worker {wid}] cycle {cycle} gjk={gjk_R:.6f} exact={score} {"SAVED" if saved else ""}', flush=True)

        # If competitive, do a deeper polish pass
        if score is not None and score < load_best_score() + 0.005:
            p2_xs, p2_ys, p2_ts, gjk_R2 = gjk_polish(
                p_xs, p_ys, p_ts, 5_000_000, 0.001, 1e-6, 0.001,
                rng.randint(0, 99999)
            )
            saved2, score2 = try_save_if_better(p2_xs, p2_ys, p2_ts, wid, cycle)
            if score2:
                print(f'[worker {wid}] pass-2 gjk={gjk_R2:.6f} exact={score2:.6f} {"SAVED" if saved2 else ""}', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    print(f'Starting {args.workers} workers — nudge-and-polish from competitor topology', flush=True)
    print(f'Current best: {load_best_score():.6f}', flush=True)

    _ = gjk_overlap_full(np.zeros(3), np.zeros(3), np.zeros(3))

    procs = [mp.Process(target=worker, args=(i,), daemon=True) for i in range(args.workers)]
    for proc in procs:
        proc.start()
    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        print('Stopping.')
        for proc in procs:
            proc.terminate()
