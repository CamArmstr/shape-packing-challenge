#!/usr/bin/env python3
"""
Two-phase SA focused on 4-11 topology with varied orientation strategies.
Replicates the original discovery method that found R=2.976.

6 workers × orientation modes:
  w0, w1: standard (seeds.py seed_4_11, as-is)
  w2, w3: random theta reassignment after seeding
  w4, w5: flip outer ring orientations (θ += π for outer semicircles)

Phase 1 (hot 20M) → Phase 2 (cold 200M) per run.
"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sa_v2 import sa_run_v2_wrapper
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
from seeds import seed_4_11
import fcntl

BEST_FILE = os.path.join(os.path.dirname(__file__), 'best_solution.json')
LOG_FILE  = os.path.join(os.path.dirname(__file__), 'twophase_4_11.log')
N = 15; TWO_PI = 2 * math.pi

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
        try:
            current = load_best_score()
            if r.score >= current: return False, r.score
            cx, cy = r.mec[0], r.mec[1]
            out = [{'x': round(float(xs[i]-cx),6), 'y': round(float(ys[i]-cy),6),
                    'theta': round(float(ts[i])%TWO_PI,6)} for i in range(N)]
            with open(BEST_FILE,'w') as f: json.dump(out, f, indent=2)
            return True, r.score
        finally:
            pass

def get_seed(rng_seed, orient_mode):
    """Get a 4-11 seed with orientation variant applied."""
    # Try a few seeds in case some fail
    result = None
    for attempt in range(10):
        result = seed_4_11(seed=rng_seed + attempt * 777)
        if result is not None:
            break
    if result is None:
        return None, None, None
    xs, ys, ts = result

    if orient_mode == 'standard':
        pass  # use as-is from seeds.py

    elif orient_mode == 'random':
        # Randomize all thetas completely
        rng = np.random.RandomState(rng_seed + 50000)
        ts = rng.uniform(0, TWO_PI, N)

    elif orient_mode == 'flip_outer':
        # Identify outer semicircles (farther from origin) and flip their theta
        dists = np.sqrt(xs**2 + ys**2)
        threshold = np.percentile(dists, 30)  # inner ~4/15
        for i in range(N):
            if dists[i] > threshold:
                ts[i] = (ts[i] + math.pi) % TWO_PI

    return xs, ys, ts

def run_worker(wid, n_runs, orient_mode):
    log = open(LOG_FILE, 'a')
    def logprint(msg):
        print(msg, flush=True)
        log.write(time.strftime('%H:%M:%S') + ' ' + msg + '\n')
        log.flush()

    logprint(f'[w{wid}] start | orient={orient_mode} | {n_runs} runs')

    for run_id in range(n_runs):
        seed_val = wid * 100000 + run_id + int(time.time()) % 10000
        xs, ys, ts = get_seed(seed_val, orient_mode)
        if xs is None:
            logprint(f'[w{wid}] run{run_id:2d} seed failed')
            continue

        # Phase 1: hot compress — higher lam to resolve overlaps from orientation changes
        bx, by, bt, _ = sa_run_v2_wrapper(
            xs, ys, ts, n_steps=20_000_000,
            T_start=0.5, T_end=0.01,
            lam_start=1000.0, lam_end=5000.0, seed=seed_val)
        if bx is None:
            logprint(f'[w{wid}] run{run_id:2d} P1 no feasible')
            continue

        sol1 = [Semicircle(float(bx[i]),float(by[i]),float(bt[i])) for i in range(N)]
        r1 = validate_and_score(sol1)
        if not r1.valid or r1.score > 3.5:
            logprint(f'[w{wid}] run{run_id:2d} P1={r1.score:.4f} too loose, skip')
            continue

        logprint(f'[w{wid}] run{run_id:2d} P1={r1.score:.6f} → P2')

        # Phase 2: cold descent
        bx2, by2, bt2, _ = sa_run_v2_wrapper(
            bx, by, bt, n_steps=200_000_000,
            T_start=0.01, T_end=0.000005,
            lam_start=3000.0, lam_end=30000.0, seed=seed_val+1)
        if bx2 is None:
            logprint(f'[w{wid}] run{run_id:2d} P2 no feasible')
            continue

        saved, score = save_if_better(bx2, by2, bt2)
        msg = f'[w{wid}] run{run_id:2d} P2={score:.6f}'
        if saved: msg += ' *** NEW BEST ***'
        logprint(msg)

    logprint(f'[w{wid}] done')
    log.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wid', type=int, required=True)
    parser.add_argument('--runs', type=int, default=9)
    parser.add_argument('--orient', type=str, default='standard',
                        choices=['standard','random','flip_outer'])
    args = parser.parse_args()
    run_worker(args.wid, args.runs, args.orient)
