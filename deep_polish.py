#!/usr/bin/env python3
"""
deep_polish.py — Sustained GJK polish on the current best solution.

Runs long GJK SA polish passes on best_solution.json with progressively
tighter temperatures. No topology changes — pure local search within the basin.

Strategy:
  - 3 phases: warm → cold → ultra-cold
  - Each phase: 20M steps
  - Between phases: re-read from disk (accept external improvements too)
  - Save if strictly better

Single process, runs indefinitely until killed.
"""

import sys, os, json, time, subprocess, math
import numpy as np

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
from gjk_numba import semicircle_gjk_signed_dist
import numba as nb
import fcntl

BEST_FILE = 'best_solution.json'
LOCK_PATH  = BEST_FILE + '.lock'
N = 15; TWO_PI = 2 * math.pi
LOG = 'deep_polish.log'


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
def gjk_overlap_single(xs, ys, ts, idx):
    n = xs.shape[0]
    e = 0.0
    for j in range(n):
        if j == idx:
            continue
        d = semicircle_gjk_signed_dist(xs[idx], ys[idx], ts[idx], xs[j], ys[j], ts[j])
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
    for k in range(32):
        a = t - math.pi / 2 + math.pi * k / 31
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
        best_xs[:] = cur_xs[:]
        best_ys[:] = cur_ys[:]
        best_ts[:] = cur_ts[:]

    log_T = math.log(T_end / T_start)
    scale = 0.004  # tighter than lns3

    for step in range(n_steps):
        frac = step / n_steps
        T    = T_start * math.exp(log_T * frac)
        idx  = int(np.random.random() * n) % n

        old_x, old_y, old_t   = cur_xs[idx], cur_ys[idx], cur_ts[idx]
        old_ovlp_i = gjk_overlap_single(cur_xs, cur_ys, cur_ts, idx)
        old_ri     = r_single_nb(old_x, old_y, old_t)

        r = np.random.random()
        if r < 0.50:
            cur_xs[idx] += np.random.normal(0, scale)
            cur_ys[idx] += np.random.normal(0, scale)
        elif r < 0.75:
            cur_ts[idx] += np.random.normal(0, scale * 3)
        elif r < 0.88:
            cur_xs[idx] += np.random.normal(0, scale * 0.3)
            cur_ys[idx] += np.random.normal(0, scale * 0.3)
            cur_ts[idx] += np.random.normal(0, scale)
        else:
            # all three together, very small
            cur_xs[idx] += np.random.normal(0, scale * 0.2)
            cur_ys[idx] += np.random.normal(0, scale * 0.2)
            cur_ts[idx] += np.random.normal(0, scale * 0.5)

        new_ovlp_i = gjk_overlap_single(cur_xs, cur_ys, cur_ts, idx)
        new_ri     = r_single_nb(cur_xs[idx], cur_ys[idx], cur_ts[idx])
        new_ovlp   = cur_ovlp - old_ovlp_i + new_ovlp_i
        new_R      = cur_R
        if new_ri > cur_R:
            new_R = new_ri
        elif abs(old_ri - cur_R) < 1e-10:
            new_R = r_max_nb(cur_xs, cur_ys, cur_ts)

        old_obj = cur_R + lam * cur_ovlp
        new_obj = new_R + lam * new_ovlp
        delta   = new_obj - old_obj

        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-15)):
            cur_ovlp, cur_R = new_ovlp, new_R
            if new_ovlp < 1e-14 and new_R < best_R:
                best_R = new_R
                best_xs[:] = cur_xs[:]
                best_ys[:] = cur_ys[:]
                best_ts[:] = cur_ts[:]
        else:
            cur_xs[idx], cur_ys[idx], cur_ts[idx] = old_x, old_y, old_t

    return best_xs, best_ys, best_ts, best_R


def official_score(xs, ys, ts):
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    r   = validate_and_score(sol)
    return (r.score if r.valid else float('inf')), r

def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    return (np.array([s['x'] for s in raw]),
            np.array([s['y'] for s in raw]),
            np.array([s['theta'] for s in raw]))

def load_best_score():
    xs, ys, ts = load_best()
    s, _ = official_score(xs, ys, ts)
    return s

def save_if_better(xs, ys, ts, score, best_ref):
    if score >= best_ref[0]:
        return False
    with open(LOCK_PATH, 'w') as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX)
            disk_s = load_best_score()
            if disk_s < best_ref[0]:
                best_ref[0] = disk_s
            if score >= best_ref[0]:
                return False
            s, r = official_score(xs, ys, ts)
            if not r.valid or s >= best_ref[0]:
                return False
            cx, cy = r.mec[0], r.mec[1]
            out = [{'x': round(float(xs[i]-cx), 6),
                    'y': round(float(ys[i]-cy), 6),
                    'theta': round(float(ts[i]) % TWO_PI, 6)} for i in range(N)]
            with open(BEST_FILE, 'w') as f:
                json.dump(out, f, indent=2)
            best_ref[0] = s
            try:
                import shutil, os as _os
                _os.makedirs('solutions', exist_ok=True)
                shutil.copy(BEST_FILE, f'solutions/R{s:.6f}.json')
                subprocess.run(['git', 'add', 'best_solution.json',
                                f'solutions/R{s:.6f}.json'],
                               capture_output=True)
                subprocess.run(['git', 'commit', '-m', f'best: R={s:.6f} (deep_polish)'],
                               capture_output=True)
            except:
                pass
            return True
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def main():
    # JIT warmup
    _w = gjk_polish(np.zeros(N), np.zeros(N), np.zeros(N), 200, 0.01, 0.001, 1000.0, 0)

    log = open(LOG, 'a')
    def logp(msg):
        ts = time.strftime('%H:%M:%S')
        line = f'{ts} {msg}'
        print(line, flush=True)
        log.write(line + '\n'); log.flush()

    best_ref = [load_best_score()]
    logp(f'deep_polish start | best on disk: R={best_ref[0]:.6f}')

    # Phase schedule: (T_start, T_end, lam, n_steps, label)
    phases = [
        (0.002,  0.000010, 50_000.0,  20_000_000, 'warm'),
        (0.0008, 0.000003, 150_000.0, 20_000_000, 'cold'),
        (0.0002, 0.0000005, 500_000.0, 20_000_000, 'ultra'),
    ]

    rng = np.random.default_rng(int(time.time()) % 100000)
    run = 0

    while True:
        run += 1
        # Always reload best from disk (lns3 may have improved it)
        xs, ys, ts = load_best()
        disk_s = load_best_score()
        if disk_s < best_ref[0]:
            best_ref[0] = disk_s
            logp(f'[run {run}] reloaded disk best: R={disk_s:.6f}')

        cur_xs, cur_ys, cur_ts = xs.copy(), ys.copy(), ts.copy()
        cur_s = disk_s

        for phase_T_start, phase_T_end, phase_lam, phase_steps, phase_name in phases:
            seed = int(rng.integers(0, 2**31))
            px, py, pt, pr = gjk_polish(
                cur_xs, cur_ys, cur_ts,
                n_steps=phase_steps,
                T_start=phase_T_start,
                T_end=phase_T_end,
                lam=phase_lam,
                seed=seed,
            )
            pol_s, _ = official_score(px, py, pt)
            logp(f'[run {run}] {phase_name}: R={pol_s:.6f} (was {cur_s:.6f})')

            if pol_s < cur_s:
                cur_xs, cur_ys, cur_ts = px.copy(), py.copy(), pt.copy()
                cur_s = pol_s
                saved = save_if_better(cur_xs, cur_ys, cur_ts, cur_s, best_ref)
                if saved:
                    logp(f'[run {run}] ★ NEW BEST R={cur_s:.6f}')

        logp(f'[run {run}] done | best this run: R={cur_s:.6f} | global: R={best_ref[0]:.6f}')


if __name__ == '__main__':
    main()
