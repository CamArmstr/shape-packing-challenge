#!/usr/bin/env python3
"""
C: SA restarts from best solution — 50 independent runs, 6 parallel workers.
Each run: load best, add small perturbation, run cold SA (20M steps).
"""
import sys, os, json, time, random, multiprocessing as mp
sys.path.insert(0, '.')
import numpy as np
from sa_v2 import sa_run_v2_wrapper
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
import fcntl

BEST_FILE = 'best_solution.json'
LOG_FILE = 'sa_restarts.log'
N = 15
N_RUNS = 50
N_WORKERS = 6

def load_best():
    with open(BEST_FILE) as f: raw = json.load(f)
    return (np.array([s['x'] for s in raw]),
            np.array([s['y'] for s in raw]),
            np.array([s['theta'] for s in raw]))

def load_best_score():
    xs, ys, ts = load_best()
    sol = [Semicircle(float(xs[i]),float(ys[i]),float(ts[i])) for i in range(N)]
    r = validate_and_score(sol)
    return r.score

def save_if_better(xs, ys, ts, run_id):
    sol = [Semicircle(float(xs[i]),float(ys[i]),float(ts[i])) for i in range(N)]
    r = validate_and_score(sol)
    if not r.valid: return False, None
    lock_path = BEST_FILE + '.lock'
    with open(lock_path, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        current = load_best_score()
        if r.score >= current: return False, r.score
        cx, cy = r.mec[0], r.mec[1]
        raw = [{'x': round(float(xs[i]-cx),6),'y': round(float(ys[i]-cy),6),'theta': round(float(ts[i])%(2*np.pi),6)} for i in range(N)]
        with open(BEST_FILE,'w') as f: json.dump(raw, f, indent=2)
        return True, r.score

def worker(run_ids, noise, T_start, T_end):
    for run_id in run_ids:
        rng = np.random.RandomState(run_id * 1337 + int(time.time()) % 10000)
        xs, ys, ts = load_best()
        xs += rng.randn(N) * noise
        ys += rng.randn(N) * noise
        ts = (ts + rng.randn(N) * noise * 2) % (2*np.pi)
        bx, by, bt, br = sa_run_v2_wrapper(xs, ys, ts, n_steps=20_000_000,
            T_start=T_start, T_end=T_end, lam_start=3000.0, lam_end=30000.0, seed=run_id)
        saved, score = save_if_better(bx, by, bt, run_id)
        msg = f'[run {run_id:3d}] R={score:.6f}' + (' *** NEW BEST ***' if saved else '') if score else f'[run {run_id:3d}] invalid'
        print(msg, flush=True)
        with open(LOG_FILE, 'a') as f: f.write(time.strftime('%H:%M:%S') + ' ' + msg + '\n')

if __name__ == '__main__':
    start_score = load_best_score()
    print(f'Starting {N_RUNS} SA restarts from R={start_score:.6f}, {N_WORKERS} workers', flush=True)
    # 3 noise levels: tight (0.05), medium (0.15), loose (0.3)
    runs = list(range(N_RUNS))
    chunks = [runs[i::N_WORKERS] for i in range(N_WORKERS)]
    noises = [0.05, 0.05, 0.15, 0.15, 0.3, 0.3]
    T_starts = [0.01, 0.01, 0.05, 0.05, 0.15, 0.15]
    procs = []
    for i in range(N_WORKERS):
        p = mp.Process(target=worker, args=(chunks[i], noises[i], T_starts[i], 0.0001))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    final_score = load_best_score()
    print(f'Done. Final best: R={final_score:.6f}', flush=True)
