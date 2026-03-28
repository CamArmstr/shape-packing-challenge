#!/usr/bin/env python3
"""
hillclimber2.py — Shapely-exact hill climber on best_solution.json.
Rebuilt from hillclimb.txt evidence: makes incremental improvements,
accepts only improvements (strictly monotonic).

Uses ONLY Shapely for overlap checks — no phi-function distortion.
Tries random small perturbations of individual shapes; keeps improvements.
"""
import sys, os, json, time, random, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
import fcntl

BEST_FILE = os.path.join(os.path.dirname(__file__), 'best_solution.json')
LOG_FILE  = os.path.join(os.path.dirname(__file__), 'hillclimb2.log')
N = 15; TWO_PI = 2 * math.pi

def load_best():
    with open(BEST_FILE) as f: raw = json.load(f)
    xs = np.array([s['x'] for s in raw])
    ys = np.array([s['y'] for s in raw])
    ts = np.array([s['theta'] for s in raw])
    return xs, ys, ts

def score(xs, ys, ts):
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    r = validate_and_score(sol)
    return r.score if r.valid else float('inf'), r

def save_if_better(xs, ys, ts, current_best):
    s, r = score(xs, ys, ts)
    if s >= current_best: return current_best, False
    lock_path = BEST_FILE + '.lock'
    with open(lock_path, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        # Re-check under lock
        xs2, ys2, ts2 = load_best()
        s2, r2 = score(xs2, ys2, ts2)
        if s >= s2: return s2, False
        cx, cy = r.mec[0], r.mec[1]
        out = [{'x': round(float(xs[i]-cx),6), 'y': round(float(ys[i]-cy),6),
                'theta': round(float(ts[i])%TWO_PI,6)} for i in range(N)]
        with open(BEST_FILE,'w') as f: json.dump(out, f, indent=2)
        return s, True

def run(max_trials=500000, log_interval=10000):
    xs, ys, ts = load_best()
    best, _ = score(xs, ys, ts)
    print(f'Starting: R={best:.6f}', flush=True)

    log = open(LOG_FILE, 'a')
    def logprint(msg):
        print(msg, flush=True)
        log.write(time.strftime('%H:%M:%S') + ' ' + msg + '\n')
        log.flush()

    improvements = 0
    rng = np.random.RandomState(int(time.time()) % 100000)
    t0 = time.time()

    # Perturbation schedule: cycle through scales
    scales = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
    scale_idx = 0
    no_improve_streak = 0

    for trial in range(max_trials):
        # Cycle scales, increase when stuck
        scale = scales[scale_idx % len(scales)]
        if no_improve_streak > 5000:
            scale_idx = (scale_idx + 1) % (len(scales) * 3)
            no_improve_streak = 0

        # Pick perturbation type
        r = rng.random()
        nxs, nys, nts = xs.copy(), ys.copy(), ts.copy()

        if r < 0.5:
            # Perturb one random shape (x, y, theta)
            i = rng.randint(0, N)
            nxs[i] += rng.uniform(-scale, scale)
            nys[i] += rng.uniform(-scale, scale)
            nts[i] = (nts[i] + rng.uniform(-scale*3, scale*3)) % TWO_PI
        elif r < 0.7:
            # Perturb x only
            i = rng.randint(0, N)
            nxs[i] += rng.uniform(-scale, scale)
        elif r < 0.85:
            # Perturb y only
            i = rng.randint(0, N)
            nys[i] += rng.uniform(-scale, scale)
        elif r < 0.95:
            # Perturb theta only
            i = rng.randint(0, N)
            nts[i] = (nts[i] + rng.uniform(-scale*5, scale*5)) % TWO_PI
        else:
            # Perturb 2 shapes together
            i, j = rng.choice(N, 2, replace=False)
            nxs[i] += rng.uniform(-scale, scale)
            nys[i] += rng.uniform(-scale, scale)
            nxs[j] += rng.uniform(-scale, scale)
            nys[j] += rng.uniform(-scale, scale)

        new_score, new_r = score(nxs, nys, nts)
        if new_score < best:
            best = new_score
            xs, ys, ts = nxs.copy(), nys.copy(), nts.copy()
            improvements += 1
            no_improve_streak = 0
            scale_idx = 0  # reset to fine scale after improvement
            save_if_better(xs, ys, ts, best + 0.0001)
            logprint(f'[{trial}] R={best:.6f} (improved #{improvements})')
        else:
            no_improve_streak += 1

        if trial % log_interval == 0 and trial > 0:
            elapsed = time.time() - t0
            logprint(f'[{elapsed:.0f}s] trial={trial}, best={best:.6f}, improvements={improvements}')

    logprint(f'Done. Final: R={best:.6f} ({improvements} improvements in {max_trials} trials)')
    log.close()

if __name__ == '__main__':
    run(max_trials=500000)
