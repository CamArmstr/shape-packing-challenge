#!/usr/bin/env python3
"""
coordinated_polish.py

Polisher that uses coordinated moves: rotate/translate clusters of 2-3
neighboring shapes together, swap shape positions, and translate pairs.
Always re-reads best_solution.json each cycle to stay current.
"""

import sys, os, json, time, math, fcntl
import numpy as np
import multiprocessing as mp

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from src.semicircle_packing.scoring import validate_and_score, minimum_enclosing_circle
from src.semicircle_packing.geometry import Semicircle
from gjk_numba import semicircle_gjk_signed_dist
import numba as nb

N = 15
BEST_FILE = 'best_solution.json'
LOCK_PATH = BEST_FILE + '.lock'


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
    """Standard single-shape SA polish."""
    np.random.seed(seed)
    n = xs.shape[0]
    cur_xs, cur_ys, cur_ts = xs.copy(), ys.copy(), ts.copy()
    cur_ovlp = gjk_overlap_full(cur_xs, cur_ys, cur_ts)
    cur_R = r_max_nb(cur_xs, cur_ys, cur_ts)
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
        new_R = r_max_nb(cur_xs, cur_ys, cur_ts)
        if cur_ovlp > 1e-14:
            dE = new_ovlp - cur_ovlp
        else:
            dE = 1e9 if new_ovlp > 1e-14 else new_R - cur_R
        if dE < 0 or np.random.random() < math.exp(-dE / max(T, 1e-14)):
            cur_ovlp = new_ovlp
            cur_R = new_R
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


def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    xs = np.array([s['x'] for s in raw], dtype=np.float64)
    ys = np.array([s['y'] for s in raw], dtype=np.float64)
    ts = np.array([s['theta'] for s in raw], dtype=np.float64)
    return xs, ys, ts


def load_best_score():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    shapes = [Semicircle(s['x'], s['y'], s['theta']) for s in raw]
    result = validate_and_score(shapes)
    return result.score if result.valid else float('inf')


def try_save_if_better(xs, ys, ts, tag, cycle):
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
                # Auto-commit
                os.system(f'cd /home/camcore/.openclaw/workspace/shape-packing-challenge && '
                          f'git add best_solution.json && '
                          f'git commit -m "best: R={score:.6f} ({tag}, coordinated)" --no-gpg-sign -q')
                print(f'[{tag}] *** NEW BEST: {score:.6f} (was {disk:.6f}) cycle={cycle} ***', flush=True)
                return True, score
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return False, score


def find_neighbors(xs, ys, k=3):
    """Find k nearest neighbor indices for each shape."""
    dists = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dists[i, j] = math.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
    neighbors = {}
    for i in range(N):
        order = np.argsort(dists[i])
        neighbors[i] = list(order[1:k+1])
    return neighbors


def apply_coordinated_move(xs, ys, ts, rng, neighbors):
    """Apply one of several coordinated move types. Returns modified copies."""
    mx, my, mt = xs.copy(), ys.copy(), ts.copy()
    move_type = rng.randint(0, 4)

    if move_type == 0:
        # Cluster translate: pick a shape and its neighbors, translate together
        anchor = rng.randint(0, N)
        group = [anchor] + neighbors[anchor][:rng.randint(1, 3)]
        dx = rng.randn() * 0.008
        dy = rng.randn() * 0.008
        for idx in group:
            mx[idx] += dx
            my[idx] += dy

    elif move_type == 1:
        # Cluster rotate: pick a shape and neighbors, rotate around their centroid
        anchor = rng.randint(0, N)
        group = [anchor] + neighbors[anchor][:rng.randint(1, 3)]
        cx = np.mean(mx[group])
        cy = np.mean(my[group])
        angle = rng.randn() * 0.015
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for idx in group:
            rx, ry = mx[idx] - cx, my[idx] - cy
            mx[idx] = cx + rx * cos_a - ry * sin_a
            my[idx] = cy + rx * sin_a + ry * cos_a
            mt[idx] += angle  # rotate shape orientation too

    elif move_type == 2:
        # Partial swap: nudge two neighboring shapes toward each other's positions
        i = rng.randint(0, N)
        j = neighbors[i][0]  # nearest neighbor
        frac = rng.uniform(0.02, 0.12)  # only move 2-12% of the way
        ix, iy = mx[i], my[i]
        jx, jy = mx[j], my[j]
        mx[i] += (jx - ix) * frac
        my[i] += (jy - iy) * frac
        mx[j] += (ix - jx) * frac
        my[j] += (iy - jy) * frac

    elif move_type == 3:
        # Pair translate toward center: pick the outermost shape, nudge it inward
        dists_from_origin = np.sqrt(mx**2 + my**2)
        outer = np.argmax(dists_from_origin)
        scale = rng.uniform(0.002, 0.015)
        mx[outer] -= mx[outer] * scale
        my[outer] -= my[outer] * scale

    else:
        # Global scale: contract all positions slightly toward origin
        scale = 1.0 - rng.uniform(0.0005, 0.003)
        mx *= scale
        my *= scale

    return mx, my, mt


def worker(wid):
    os.nice(10)
    tag = f'coord_w{wid}'

    # JIT warmup
    _ = gjk_overlap_full(np.zeros(3), np.zeros(3), np.zeros(3))
    _ = gjk_polish(np.zeros(N), np.zeros(N), np.zeros(N), 200, 0.01, 0.001, 0.003, 42)

    rng = np.random.RandomState(wid * 7919 + int(time.time()) % 100000)
    cycle = 0

    while True:
        cycle += 1

        # Always re-read best from disk
        xs, ys, ts = load_best()
        neighbors = find_neighbors(xs, ys, k=3)

        # Apply coordinated move
        mx, my, mt = apply_coordinated_move(xs, ys, ts, rng, neighbors)

        # Quick overlap check before expensive polish
        ovlp = gjk_overlap_full(mx, my, mt)

        # Polish: if overlapping, use overlap-resolving mode; if clean, polish R
        if ovlp > 1e-14:
            # Try to fix overlap with short SA
            p_xs, p_ys, p_ts, gjk_R = gjk_polish(
                mx, my, mt, 500_000, 0.01, 1e-5, 0.005,
                rng.randint(0, 99999)
            )
        else:
            # Already valid, fine polish for R
            p_xs, p_ys, p_ts, gjk_R = gjk_polish(
                mx, my, mt, 2_000_000, 0.002, 1e-6, 0.002,
                rng.randint(0, 99999)
            )

        if gjk_R > 1e17:
            if cycle % 50 == 0:
                print(f'[{tag}] cycle {cycle} no valid result', flush=True)
            continue

        saved, score = try_save_if_better(p_xs, p_ys, p_ts, tag, cycle)
        if cycle % 20 == 0 or saved:
            print(f'[{tag}] cycle {cycle} gjk={gjk_R:.6f} exact={score} {"SAVED" if saved else ""}', flush=True)

        # If close to best, do a deeper second pass
        if score is not None and score < load_best_score() + 0.002:
            p2_xs, p2_ys, p2_ts, gjk_R2 = gjk_polish(
                p_xs, p_ys, p_ts, 4_000_000, 0.001, 1e-7, 0.001,
                rng.randint(0, 99999)
            )
            saved2, score2 = try_save_if_better(p2_xs, p2_ys, p2_ts, tag, cycle)
            if saved2:
                print(f'[{tag}] pass-2 gjk={gjk_R2:.6f} exact={score2:.6f} SAVED', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()

    print(f'Starting {args.workers} coordinated-move workers', flush=True)
    print(f'Current best: {load_best_score():.6f}', flush=True)
    print(f'Move types: cluster-translate, cluster-rotate, swap, inward-nudge, global-contract', flush=True)

    # JIT warmup in main
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
