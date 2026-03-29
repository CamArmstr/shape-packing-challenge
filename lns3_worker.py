#!/usr/bin/env python3
"""
lns3_worker.py — Single worker process for lns3.
Run directly: python3 lns3_worker.py --wid 0 --runtime 21600
Launched by lns3_launch.sh for multi-worker runs.
"""
import sys, os, argparse
os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

import json, time, math, subprocess
import fcntl
import numpy as np

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
from gjk_numba import semicircle_gjk_signed_dist
import numba as nb

BEST_FILE = 'best_solution.json'
LOCK_PATH  = BEST_FILE + '.lock'
N = 15; TWO_PI = 2 * math.pi


# ── GJK polish (Numba) ────────────────────────────────────────────────────────

@nb.njit(cache=True)
def gjk_overlap_full(xs, ys, ts):
    n = xs.shape[0]; e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = semicircle_gjk_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < 0.0: e += d * d
    return e

@nb.njit(cache=True)
def gjk_overlap_single(xs, ys, ts, idx):
    n = xs.shape[0]; e = 0.0
    for j in range(n):
        if j == idx: continue
        d = semicircle_gjk_signed_dist(xs[idx], ys[idx], ts[idx], xs[j], ys[j], ts[j])
        if d < 0.0: e += d * d
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
        best_xs[:] = cur_xs[:]
        best_ys[:] = cur_ys[:]
        best_ts[:] = cur_ts[:]
    log_T = math.log(T_end / T_start)
    scale = 0.008
    for step in range(n_steps):
        frac = step / n_steps
        T    = T_start * math.exp(log_T * frac)
        idx  = int(np.random.random() * n) % n
        old_x, old_y, old_t   = cur_xs[idx], cur_ys[idx], cur_ts[idx]
        old_ovlp_i = gjk_overlap_single(cur_xs, cur_ys, cur_ts, idx)
        old_ri     = r_single_nb(old_x, old_y, old_t)
        r = np.random.random()
        if r < 0.55:
            cur_xs[idx] += np.random.normal(0, scale)
            cur_ys[idx] += np.random.normal(0, scale)
        elif r < 0.80:
            cur_ts[idx] += np.random.normal(0, scale * 2.5)
        else:
            cur_xs[idx] += np.random.normal(0, scale * 0.4)
            cur_ys[idx] += np.random.normal(0, scale * 0.4)
            cur_ts[idx] += np.random.normal(0, scale)
        new_ovlp_i = gjk_overlap_single(cur_xs, cur_ys, cur_ts, idx)
        new_ri     = r_single_nb(cur_xs[idx], cur_ys[idx], cur_ts[idx])
        new_ovlp   = cur_ovlp - old_ovlp_i + new_ovlp_i
        new_R      = cur_R
        if new_ri > cur_R: new_R = new_ri
        elif abs(old_ri - cur_R) < 1e-10: new_R = r_max_nb(cur_xs, cur_ys, cur_ts)
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


# ── Helpers ───────────────────────────────────────────────────────────────────

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

def save_if_better(xs, ys, ts, score):
    with open(LOCK_PATH, 'w') as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX)
            disk_s = load_best_score()
            if score >= disk_s:
                return False, disk_s
            s, r = official_score(xs, ys, ts)
            if not r.valid or s >= disk_s:
                return False, disk_s
            cx, cy = r.mec[0], r.mec[1]
            out = [{'x': round(float(xs[i]-cx), 6),
                    'y': round(float(ys[i]-cy), 6),
                    'theta': round(float(ts[i]) % TWO_PI, 6)} for i in range(N)]
            with open(BEST_FILE, 'w') as f:
                json.dump(out, f, indent=2)
            import shutil
            os.makedirs('solutions', exist_ok=True)
            shutil.copy(BEST_FILE, f'solutions/R{s:.6f}.json')
            try:
                subprocess.run(['git', 'add', 'best_solution.json',
                                f'solutions/R{s:.6f}.json'], capture_output=True)
                subprocess.run(['git', 'commit', '-m',
                                f'best: R={s:.6f} (lns3_worker w{WID})'],
                               capture_output=True)
            except: pass
            return True, s
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

def boundary_order(xs, ys, ts):
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    r   = validate_and_score(sol)
    if not r.valid: return list(range(N))
    cx, cy = r.mec[0], r.mec[1]
    pts = []
    for i in range(N):
        x, y, t = xs[i], ys[i], ts[i]
        rim = [(x+math.cos(t), y+math.sin(t)),
               (x-math.sin(t), y+math.cos(t)),
               (x+math.sin(t), y-math.cos(t))]
        ri = max(math.hypot(px-cx, py-cy) for px, py in rim)
        pts.append((ri, i))
    pts.sort(reverse=True)
    return [idx for _, idx in pts]

def try_reinsert(xs, ys, ts, removed_idxs, rng, n_cand=50):
    mask = np.ones(N, dtype=bool)
    for idx in removed_idxs: mask[idx] = False
    cur_xs = list(xs[mask]); cur_ys = list(ys[mask]); cur_ts = list(ts[mask])
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
                curr_r = max(math.hypot(cur_xs[j], cur_ys[j]) + 1.2
                             for j in range(n_cur)) if n_cur > 0 else 2.0
                rpos   = rng.uniform(0, curr_r)
                angle  = rng.uniform(0, TWO_PI)
                x = rpos * math.cos(angle); y = rpos * math.sin(angle)
            for _ in range(10):
                t = rng.uniform(0, TWO_PI)
                sc = Semicircle(x, y, t)
                ok = all(not semicircles_overlap(sc,
                         Semicircle(float(cur_xs[j]), float(cur_ys[j]), float(cur_ts[j])))
                         for j in range(n_cur))
                if ok:
                    trial_xs = np.array(cur_xs + [x], dtype=np.float64)
                    trial_ys = np.array(cur_ys + [y], dtype=np.float64)
                    trial_ts = np.array(cur_ts + [t], dtype=np.float64)
                    s_proxy = r_max_nb(trial_xs, trial_ys, trial_ts)
                    if s_proxy < best_s:
                        best_s = s_proxy; best_pos = (x, y, t)
                    break
        if best_pos is None: return None
        cur_xs.append(best_pos[0]); cur_ys.append(best_pos[1]); cur_ts.append(best_pos[2])
    return (np.array(cur_xs, dtype=np.float64),
            np.array(cur_ys, dtype=np.float64),
            np.array(cur_ts, dtype=np.float64))


# ── Main worker loop ──────────────────────────────────────────────────────────

def run(wid, runtime):
    global WID
    WID = wid

    # JIT warmup — do this BEFORE the main loop
    print(f'[w{wid}] warming up JIT...', flush=True)
    _w1 = gjk_overlap_full(np.zeros(3), np.zeros(3), np.zeros(3))
    _w2 = gjk_polish(np.zeros(N), np.zeros(N), np.zeros(N), 200, 0.01, 0.001, 1000.0, 42)
    print(f'[w{wid}] JIT ready', flush=True)

    rng   = np.random.RandomState(wid * 9973 + int(time.time()) % 100000)
    start = time.time()
    cycle = 0

    while time.time() - start < runtime:
        cycle += 1
        xs, ys, ts = load_best()
        cur_best   = load_best_score()

        k_choices  = [1, 1, 1, 2, 2, 3]
        k = int(rng.choice(k_choices))

        order = boundary_order(xs, ys, ts)
        roll  = rng.random()
        if roll < 0.30:
            removed = order[:k]
        elif roll < 0.55:
            removed = list(rng.choice(N, k, replace=False))
        elif roll < 0.70:
            alt = order[1] if len(order) > 1 else order[0]
            removed = [alt] + [i for i in rng.permutation(N) if i != alt][:k-1]
        elif roll < 0.85:
            removed = [order[0]]
            non_boundary = [i for i in rng.permutation(N) if i != order[0]]
            removed += list(non_boundary[:k-1])
        else:
            cx_ = float(np.mean(xs)); cy_ = float(np.mean(ys))
            angles = [(math.atan2(ys[i]-cy_, xs[i]-cx_), i) for i in range(N)]
            angles.sort()
            step_ = N // max(k, 1)
            start_ = rng.randint(0, step_)
            removed = [angles[(start_ + j*step_) % N][1] for j in range(k)]

        result = try_reinsert(xs, ys, ts, removed, rng, n_cand=50)
        if result is None:
            print(f'[w{wid}] c{cycle} rm={removed} no placement', flush=True)
            continue

        rx, ry, rt = result
        pre_s, _ = official_score(rx, ry, rt)
        if pre_s == float('inf'): continue

        seed = wid * 10007 + cycle
        px, py, pt, _ = gjk_polish(rx, ry, rt, 3_000_000,
                                    0.003, 0.000005, 80_000.0, seed)
        pol_s, _ = official_score(px, py, pt)
        if pol_s == float('inf'):
            pol_s = pre_s; px, py, pt = rx, ry, rt

        saved, saved_s = save_if_better(px, py, pt, pol_s)

        if pol_s < cur_best + 0.08:
            msg = f'[w{wid}] c{cycle} rm={removed} pre={pre_s:.4f} → pol={pol_s:.6f}'
            if saved: msg += f'  ★ NEW BEST R={saved_s:.6f}'
            print(msg, flush=True)
        elif saved:
            print(f'[w{wid}] c{cycle}  ★ NEW BEST R={saved_s:.6f}', flush=True)

        if pol_s < cur_best + 0.03:
            px2, py2, pt2, _ = gjk_polish(px, py, pt, 5_000_000,
                                            0.0008, 0.000001, 200_000.0, seed+1)
            pol2_s, _ = official_score(px2, py2, pt2)
            if pol2_s < pol_s:
                saved2, saved2_s = save_if_better(px2, py2, pt2, pol2_s)
                if saved2:
                    print(f'[w{wid}] c{cycle}  ★★ PASS-2 BEST R={saved2_s:.6f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wid',     type=int, default=0)
    parser.add_argument('--runtime', type=int, default=21600)
    args = parser.parse_args()
    import argparse as _ap  # already imported above
    run(args.wid, args.runtime)
