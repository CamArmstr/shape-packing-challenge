#!/usr/bin/env python3
"""Single-process flip worker — called by orient_flip_launcher.sh"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sa_v2 import sa_run_v2_wrapper
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
import fcntl

BEST_FILE = os.path.join(os.path.dirname(__file__), 'best_solution.json')
LOG_FILE  = os.path.join(os.path.dirname(__file__), 'orient_flip.log')
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

wid = int(sys.argv[1])
chunk_file = sys.argv[2]
with open(chunk_file) as f:
    variants = json.load(f)

base_xs, base_ys, base_ts = load_best()
best_seen = load_best_score()
print(f'[w{wid}] start R={best_seen:.6f}, {len(variants)} variants', flush=True)

for i, flip_idx in enumerate(variants):
    xs, ys, ts = base_xs.copy(), base_ys.copy(), base_ts.copy()
    for idx in flip_idx:
        ts[idx] = (ts[idx] + np.pi) % TWO_PI
    seed = wid * 100000 + i
    bx, by, bt, br = sa_run_v2_wrapper(
        xs, ys, ts, n_steps=5_000_000,
        T_start=0.1, T_end=0.0005,
        lam_start=1000.0, lam_end=10000.0, seed=seed)
    if bx is None:
        continue
    result = save_if_better(bx, by, bt)
    saved, score = result if result else (False, None)
    if saved and score:
        msg = f'[w{wid}] flip{flip_idx} → R={score:.6f} *** NEW BEST ***'
        print(msg, flush=True)
        with open(LOG_FILE,'a') as f: f.write(time.strftime('%H:%M:%S')+' '+msg+'\n')
    elif score and score < best_seen - 0.001:
        best_seen = score
        msg = f'[w{wid}] flip{flip_idx} → R={score:.6f}'
        print(msg, flush=True)

print(f'[w{wid}] done {len(variants)} variants', flush=True)
