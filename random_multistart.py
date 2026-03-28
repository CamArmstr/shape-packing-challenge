#!/usr/bin/env python3
"""
Random multistart: seed from completely random valid configurations,
run deep SA from each. Targets unexplored basins.

Strategy:
- Generate random valid placements (no ring structure bias)
- 6 workers, each running independent random starts
- 50M steps per start (fast enough for many starts)
- Save any improvement to best_solution.json immediately
"""
import sys, os, json, time, random, multiprocessing as mp
sys.path.insert(0, '.')
import numpy as np
import math
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
from sa_v2 import sa_run_v2_wrapper
import fcntl

BEST_FILE = 'best_solution.json'
LOG_FILE  = 'random_multistart.log'
N = 15
TWO_PI = 2 * math.pi

def load_best_score():
    with open(BEST_FILE) as f: raw = json.load(f)
    sol = [Semicircle(raw[i]['x'], raw[i]['y'], raw[i]['theta']) for i in range(N)]
    return validate_and_score(sol).score

def save_if_better(xs, ys, ts):
    sol = [Semicircle(float(xs[i]),float(ys[i]),float(ts[i])) for i in range(N)]
    r = validate_and_score(sol)
    if not r.valid: return False, None
    lock_path = BEST_FILE + '.lock'
    with open(lock_path, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        current = load_best_score()
        if r.score >= current: return False, r.score
        cx, cy = r.mec[0], r.mec[1]
        raw = [{'x': round(float(xs[i]-cx),6), 'y': round(float(ys[i]-cy),6),
                'theta': round(float(ts[i])%TWO_PI,6)} for i in range(N)]
        with open(BEST_FILE,'w') as f: json.dump(raw, f, indent=2)
        return True, r.score

def random_valid_seed(rng, container_r=4.0, max_attempts=2000):
    """Place 15 semicircles randomly in a circle, no overlaps."""
    for _ in range(max_attempts):
        xs, ys, ts = [], [], []
        placed = []
        failed = False
        for k in range(N):
            ok = False
            for _ in range(300):
                # Random position within container
                r = rng.uniform(0, container_r - 1.0)
                angle = rng.uniform(0, TWO_PI)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                t = rng.uniform(0, TWO_PI)
                sc = Semicircle(x, y, t)
                if any(semicircles_overlap(sc, p) for p in placed):
                    continue
                xs.append(x); ys.append(y); ts.append(t)
                placed.append(sc)
                ok = True
                break
            if not ok:
                failed = True
                break
        if not failed:
            return np.array(xs), np.array(ys), np.array(ts)
    return None

def worker(wid, runtime_secs, seed_base):
    rng = np.random.RandomState(seed_base + wid * 7919)
    random.seed(seed_base + wid * 1337)
    t_start = time.time()
    run_id = 0
    while time.time() - t_start < runtime_secs:
        run_id += 1
        result = random_valid_seed(rng)
        if result is None:
            continue
        xs, ys, ts = result
        # Score starting point
        sol = [Semicircle(float(xs[i]),float(ys[i]),float(ts[i])) for i in range(N)]
        r0 = validate_and_score(sol)
        if not r0.valid: continue
        # Run SA from this random start
        seed = int(time.time() * 1000) % 1000000 + wid * 10000 + run_id
        bx, by, bt, br = sa_run_v2_wrapper(
            xs, ys, ts, n_steps=50_000_000,
            T_start=0.5, T_end=0.001,
            lam_start=500.0, lam_end=5000.0, seed=seed)
        saved, score = save_if_better(bx, by, bt)
        if score:
            msg = f'[w{wid} run{run_id:3d}] start={r0.score:.3f} → R={score:.6f}' + (' *** NEW BEST ***' if saved else '')
        else:
            msg = f'[w{wid} run{run_id:3d}] start={r0.score:.3f} → invalid'
        print(msg, flush=True)
        with open(LOG_FILE, 'a') as f: f.write(time.strftime('%H:%M:%S') + ' ' + msg + '\n')
    print(f'[w{wid}] done after {run_id} starts', flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=int, default=18000, help='seconds')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    start_score = load_best_score()
    print(f'Random multistart: {args.workers} workers, {args.runtime}s, starting from R={start_score:.6f}', flush=True)
    print(f'Target: R < 2.97275', flush=True)

    seed_base = int(time.time()) % 100000
    procs = [mp.Process(target=worker, args=(i, args.runtime, seed_base)) for i in range(args.workers)]
    for p in procs: p.start()
    for p in procs: p.join()

    final = load_best_score()
    print(f'\nFinal best: R={final:.6f}  (started: R={start_score:.6f})', flush=True)
