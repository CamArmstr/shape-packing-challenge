#!/usr/bin/env python3
"""
seed_competitor.py

Load the competitor's exported solution (which has one overlap: shapes 9 & 13),
remove the overlapping pair, reinsert using lns3's contact-seeking logic,
GJK polish, validate with exact scorer. Run many trials in parallel.
If we find something better than best_solution.json, save it.
"""

import sys, os, json, time, random, math, fcntl
import numpy as np
import multiprocessing as mp

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
from gjk_numba import semicircle_gjk_signed_dist
import numba as nb

N = 15
TWO_PI = 2 * math.pi
BEST_FILE = 'best_solution.json'
LOCK_PATH  = BEST_FILE + '.lock'
SEED_FILE  = 'imported_competitor.json'

# ── GJK helpers (copied from lns3) ───────────────────────────────────────────

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
def gjk_polish(xs, ys, ts, n_steps, T_start, T_end, lam, seed):
    np.random.seed(seed)
    n = xs.shape[0]
    cur_xs, cur_ys, cur_ts = xs.copy(), ys.copy(), ts.copy()
    cur_ovlp = gjk_overlap_full(cur_xs, cur_ys, cur_ts)
    cur_R    = r_max_nb(cur_xs, cur_ys, cur_ts)
    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
    best_R = 1e18
    if cur_ovlp < 1e-14:
        best_R = cur_R
    step = 0.003
    for s in range(n_steps):
        T = T_start * (T_end / T_start) ** (s / max(n_steps - 1, 1))
        idx = s % n
        ox, oy, ot = cur_xs[idx], cur_ys[idx], cur_ts[idx]
        dx = np.random.randn() * step
        dy = np.random.randn() * step
        dt = np.random.randn() * step
        cur_xs[idx] += dx; cur_ys[idx] += dy; cur_ts[idx] += dt
        new_ovlp = gjk_overlap_full(cur_xs, cur_ys, cur_ts)
        new_R    = r_max_nb(cur_xs, cur_ys, cur_ts)
        # prefer overlap-free; then minimize R
        if cur_ovlp > 1e-14:
            dE = new_ovlp - cur_ovlp
        else:
            if new_ovlp > 1e-14:
                dE = 1e6
            else:
                dE = new_R - cur_R
        if dE < 0 or np.random.random() < math.exp(-dE / max(T, 1e-12)):
            cur_ovlp = new_ovlp
            cur_R    = new_R
            if new_ovlp < 1e-14 and new_R < best_R:
                best_R = new_R
                best_xs[:] = cur_xs[:]
                best_ys[:] = cur_ys[:]
                best_ts[:] = cur_ts[:]
            # adaptive step
            if s % 500 == 0:
                step = min(step * 1.05, 0.05)
        else:
            cur_xs[idx] = ox; cur_ys[idx] = oy; cur_ts[idx] = ot
            if s % 500 == 0:
                step = max(step * 0.95, 1e-5)
    return best_xs, best_ys, best_ts, best_R


def load_best_score():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    shapes = [Semicircle(s['x'], s['y'], s['theta']) for s in raw]
    result = validate_and_score(shapes)
    return result.score if result.valid else float('inf')


def try_save_if_better(xs, ys, ts, wid):
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
                print(f'[worker {wid}] *** NEW BEST: {score:.6f} (was {disk:.6f}) ***', flush=True)
                return True, score
            else:
                return False, score
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def try_reinsert(xs, ys, ts, removed_idxs, rng, n_cand=60):
    mask = np.ones(N, dtype=bool)
    for idx in removed_idxs:
        mask[idx] = False
    cur_xs = list(xs[mask])
    cur_ys = list(ys[mask])
    cur_ts = list(ts[mask])

    for removed in removed_idxs:
        n_cur = len(cur_xs)
        best_s, best_pos = float('inf'), None
        for _ in range(n_cand):
            if rng.random() < 0.65 and n_cur > 0:
                anchor = rng.randint(0, n_cur)
                angle  = rng.uniform(0, TWO_PI)
                dist   = rng.uniform(1.85, 2.35)
                x = float(cur_xs[anchor]) + dist * math.cos(angle)
                y = float(cur_ys[anchor]) + dist * math.sin(angle)
            else:
                curr_r = max(math.hypot(cur_xs[j], cur_ys[j]) + 1.2 for j in range(n_cur)) if n_cur > 0 else 2.0
                rpos   = rng.uniform(0, curr_r)
                angle  = rng.uniform(0, TWO_PI)
                x = rpos * math.cos(angle)
                y = rpos * math.sin(angle)
            for _ in range(10):
                t = rng.uniform(0, TWO_PI)
                sc = Semicircle(x, y, t)
                ok = all(not semicircles_overlap(sc, Semicircle(float(cur_xs[j]), float(cur_ys[j]), float(cur_ts[j]))) for j in range(n_cur))
                if ok:
                    trial_xs = np.array(cur_xs + [x], dtype=np.float64)
                    trial_ys = np.array(cur_ys + [y], dtype=np.float64)
                    trial_ts = np.array(cur_ts + [t], dtype=np.float64)
                    s_proxy = r_max_nb(trial_xs, trial_ys, trial_ts)
                    if s_proxy < best_s:
                        best_s = s_proxy
                        best_pos = (x, y, t)
                    break
        if best_pos is None:
            return None
        cur_xs.append(best_pos[0])
        cur_ys.append(best_pos[1])
        cur_ts.append(best_pos[2])

    return (np.array(cur_xs, dtype=np.float64),
            np.array(cur_ys, dtype=np.float64),
            np.array(cur_ts, dtype=np.float64))


def worker(wid):
    os.nice(10)
    # JIT warmup
    _w = gjk_overlap_full(np.zeros(3), np.zeros(3), np.zeros(3))
    _p = gjk_polish(np.zeros(N), np.zeros(N), np.zeros(N), 200, 0.01, 0.001, 1000.0, 42)

    with open(SEED_FILE) as f:
        raw = json.load(f)
    seed_xs = np.array([s['x'] for s in raw], dtype=np.float64)
    seed_ys = np.array([s['y'] for s in raw], dtype=np.float64)
    seed_ts = np.array([s['theta'] for s in raw], dtype=np.float64)

    # Center the solution at origin (MEC center)
    cx = seed_xs.mean()
    cy = seed_ys.mean()
    seed_xs -= cx
    seed_ys -= cy
    print(f'[worker {wid}] Centered seed by ({cx:.4f}, {cy:.4f}), approx R={r_max_nb(seed_xs, seed_ys, seed_ts):.6f}', flush=True)

    rng = np.random.RandomState(wid * 9973 + int(time.time()) % 100000)
    cycle = 0
    start = time.time()

    while True:
        cycle += 1
        elapsed = time.time() - start

        # Vary removal strategy: always remove the overlapping pair (9,13),
        # plus optionally 1-2 more random shapes
        fixed_remove = [9, 13]
        n_extra = rng.randint(0, 3)
        extras = list(rng.choice([i for i in range(N) if i not in fixed_remove], n_extra, replace=False))
        removed = fixed_remove + extras

        result = try_reinsert(seed_xs, seed_ys, seed_ts, removed, rng, n_cand=60)
        if result is None:
            print(f'[worker {wid}] cycle {cycle} reinsertion failed, retrying', flush=True)
            continue

        r_xs, r_ys, r_ts = result
        approx_R = r_max_nb(r_xs, r_ys, r_ts)

        # GJK polish pass 1 (3M steps)
        p_xs, p_ys, p_ts, gjk_R = gjk_polish(r_xs, r_ys, r_ts, 3_000_000, 0.003, 5e-6, 80000.0, rng.randint(0, 99999))

        if gjk_R < 1e17:
            saved, score = try_save_if_better(p_xs, p_ys, p_ts, wid)
            print(f'[worker {wid}] cycle {cycle} remove={removed} approx={approx_R:.6f} gjk={gjk_R:.6f} exact={score:.6f} {"SAVED" if saved else ""}', flush=True)

            # Polish pass 2 if competitive
            if score is not None and score < load_best_score() + 0.003:
                p2_xs, p2_ys, p2_ts, gjk_R2 = gjk_polish(p_xs, p_ys, p_ts, 5_000_000, 0.001, 1e-6, 200000.0, rng.randint(0, 99999))
                saved2, score2 = try_save_if_better(p2_xs, p2_ys, p2_ts, wid)
                if score2:
                    print(f'[worker {wid}] pass-2 gjk={gjk_R2:.6f} exact={score2:.6f} {"SAVED" if saved2 else ""}', flush=True)
        else:
            print(f'[worker {wid}] cycle {cycle} remove={removed} approx={approx_R:.6f} no valid GJK result', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    print(f'Starting {args.workers} workers seeded from competitor topology', flush=True)
    print(f'Current best: {load_best_score():.6f}', flush=True)

    # JIT warmup in main process
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
