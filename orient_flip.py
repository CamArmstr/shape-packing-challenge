#!/usr/bin/env python3
"""
Priority 1: Orientation-flip combinatorial search within 4-11.
For each subset of {1,2,3} semicircles, flip θ → θ+π and run 5M SA.
575 variants total: 15 singles + 105 pairs + 455 triples.
"""
import sys, os, json, time, itertools, multiprocessing as mp
sys.path.insert(0, '.')
import numpy as np
from sa_v2 import sa_run_v2_wrapper
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
import fcntl

BEST_FILE = 'best_solution.json'
LOG_FILE  = 'orient_flip.log'
N = 15; TWO_PI = 2*np.pi

def load_best():
    with open(BEST_FILE) as f: raw = json.load(f)
    return (np.array([s['x'] for s in raw]),
            np.array([s['y'] for s in raw]),
            np.array([s['theta'] for s in raw]))

def load_best_score():
    xs, ys, ts = load_best()
    sol = [Semicircle(float(xs[i]),float(ys[i]),float(ts[i])) for i in range(N)]
    return validate_and_score(sol).score

def save_if_better(xs, ys, ts, label):
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

def run_variant(flip_indices, base_xs, base_ys, base_ts, seed):
    xs, ys, ts = base_xs.copy(), base_ys.copy(), base_ts.copy()
    for idx in flip_indices:
        ts[idx] = (ts[idx] + np.pi) % TWO_PI
    bx, by, bt, br = sa_run_v2_wrapper(
        xs, ys, ts, n_steps=5_000_000,
        T_start=0.1, T_end=0.0005,
        lam_start=1000.0, lam_end=10000.0, seed=seed)
    return bx, by, bt

def worker(variants, wid):
    base_xs, base_ys, base_ts = load_best()
    best_seen = load_best_score()
    for i, flip_idx in enumerate(variants):
        label = f'flip{list(flip_idx)}'
        seed = wid * 100000 + i
        bx, by, bt = run_variant(flip_idx, base_xs, base_ys, base_ts, seed)
        saved, score = save_if_better(bx, by, bt, label)
        if score and score < best_seen - 0.0005:  # print only meaningful results
            best_seen = score
            msg = f'[w{wid}] {label} → R={score:.6f}' + (' *** NEW BEST ***' if saved else ' (improved)')
            print(msg, flush=True)
            with open(LOG_FILE,'a') as f: f.write(time.strftime('%H:%M:%S')+' '+msg+'\n')
        elif saved:
            msg = f'[w{wid}] {label} → R={score:.6f} *** NEW BEST ***'
            print(msg, flush=True)
            with open(LOG_FILE,'a') as f: f.write(time.strftime('%H:%M:%S')+' '+msg+'\n')
    print(f'[w{wid}] done {len(variants)} variants', flush=True)

if __name__ == '__main__':
    N_WORKERS = 6
    # Generate all flip subsets: singles, pairs, triples
    all_variants = []
    for r in [1, 2, 3]:
        for combo in itertools.combinations(range(N), r):
            all_variants.append(combo)
    print(f'Total variants: {len(all_variants)} ({N} singles + {N*(N-1)//2} pairs + ... = 575 expected)', flush=True)
    print(f'Using {N_WORKERS} workers, 5M steps each', flush=True)
    print(f'Start: R={load_best_score():.6f}', flush=True)
    t0 = time.time()

    # Distribute variants across workers
    chunks = [all_variants[i::N_WORKERS] for i in range(N_WORKERS)]
    procs = [mp.Process(target=worker, args=(chunks[i], i)) for i in range(N_WORKERS)]
    for p in procs: p.start()
    for p in procs: p.join()

    elapsed = time.time() - t0
    final = load_best_score()
    print(f'\nDone in {elapsed:.0f}s. Best: R={final:.6f}', flush=True)
