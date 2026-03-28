#!/usr/bin/env python3
"""
Deep SA restarts: 200M steps per run, tight noise, 6 workers.
Target: push from 2.97468 to the 2.9727 cluster floor.
"""
import sys, os, json, time, random, multiprocessing as mp
sys.path.insert(0, '.')
import numpy as np
from sa_v2 import sa_run_v2_wrapper
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
import fcntl

BEST_FILE = 'best_solution.json'
LOG_FILE  = 'sa_deep.log'
N = 15
N_RUNS    = 30   # 30 deep runs × 6 workers = 5 runs each
N_WORKERS = 6

def load_best():
    with open(BEST_FILE) as f: raw = json.load(f)
    return (np.array([s['x'] for s in raw]),
            np.array([s['y'] for s in raw]),
            np.array([s['theta'] for s in raw]))

def load_best_score():
    xs, ys, ts = load_best()
    sol = [Semicircle(float(xs[i]),float(ys[i]),float(ts[i])) for i in range(N)]
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
        raw = [{'x': round(float(xs[i]-cx),6),'y': round(float(ys[i]-cy),6),
                'theta': round(float(ts[i])%(2*np.pi),6)} for i in range(N)]
        with open(BEST_FILE,'w') as f: json.dump(raw, f, indent=2)
        return True, r.score

def worker(run_ids, noise, T_start, T_end, n_steps):
    for run_id in run_ids:
        rng = np.random.RandomState(run_id * 9973 + int(time.time()) % 10000)
        xs, ys, ts = load_best()
        xs += rng.randn(N) * noise
        ys += rng.randn(N) * noise
        ts = (ts + rng.randn(N) * noise * 1.5) % (2*np.pi)
        t0 = time.time()
        bx, by, bt, br = sa_run_v2_wrapper(
            xs, ys, ts, n_steps=n_steps,
            T_start=T_start, T_end=T_end,
            lam_start=3000.0, lam_end=30000.0, seed=run_id)
        elapsed = time.time() - t0
        saved, score = save_if_better(bx, by, bt)
        msg = f'[run {run_id:2d}] R={score:.6f}  ({elapsed:.0f}s)' + (' *** NEW BEST ***' if saved else '') if score else f'[run {run_id:2d}] invalid'
        print(msg, flush=True)
        with open(LOG_FILE, 'a') as f: f.write(time.strftime('%H:%M:%S') + ' ' + msg + '\n')

if __name__ == '__main__':
    start = load_best_score()
    print(f'Deep SA: {N_RUNS} runs × 200M steps from R={start:.6f}', flush=True)
    print(f'Target: R < 2.97275 (cluster floor)', flush=True)
    runs = list(range(N_RUNS))
    chunks = [runs[i::N_WORKERS] for i in range(N_WORKERS)]
    # Vary noise and temperature across workers
    configs = [
        (0.03, 0.008, 0.00001),   # very tight, very cold
        (0.05, 0.012, 0.00001),   # tight, cold
        (0.03, 0.015, 0.00001),   # very tight, slightly warmer
        (0.07, 0.020, 0.00001),   # medium, warmer
        (0.05, 0.010, 0.000005),  # tight, coldest
        (0.10, 0.025, 0.00001),   # loose, warm
    ]
    procs = []
    for i in range(N_WORKERS):
        noise, T_start, T_end = configs[i]
        p = mp.Process(target=worker, args=(chunks[i], noise, T_start, T_end, 200_000_000))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    final = load_best_score()
    print(f'\nDone. Best: R={final:.6f}  (started: R={start:.6f})', flush=True)
