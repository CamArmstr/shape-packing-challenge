#!/usr/bin/env python3
"""
run_overnight.py — Final overnight optimizer. 

3 workers: polish from best (skip Phase 1, go straight to cold 200M SA)
3 workers: explore new topologies (Phase 1 → Phase 2)

Uses sa_v2 (phi-function) which is inaccurate but finds good basins.
All results validated with official Shapely scorer before saving.
"""

import sys, os, json, math, time, random, multiprocessing as mp, argparse
import numpy as np

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from sa_v2 import sa_run_v2_wrapper as run_sa
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
from seeds import seed_3_5_7, seed_2_5_8, seed_3_4_8, seed_1_5_9, seed_2_6_7, seed_5_10
import fcntl as _fcntl

BEST_FILE = 'best_solution.json'
LOCK_PATH = BEST_FILE + '.lock'
N = 15

SEED_FNS = [seed_3_5_7, seed_2_5_8, seed_3_4_8, seed_1_5_9, seed_2_6_7, seed_5_10]
SEED_NAMES = ['3-5-7', '2-5-8', '3-4-8', '1-5-9', '2-6-7', '5-10']


def official_validate(xs, ys, ts):
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    return validate_and_score(sol)

def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    return (np.array([s['x'] for s in raw]),
            np.array([s['y'] for s in raw]),
            np.array([s['theta'] for s in raw]))

def load_best_score():
    try:
        xs, ys, ts = load_best()
        r = official_validate(xs, ys, ts)
        return r.score if r.valid else float('inf')
    except:
        return float('inf')

def save_if_better(xs, ys, ts, global_best):
    r = official_validate(xs, ys, ts)
    if not r.valid or r.score >= global_best.value:
        return False, r.score if r.valid else float('inf')
    
    with open(LOCK_PATH, 'w') as lf:
        try:
            _fcntl.flock(lf, _fcntl.LOCK_EX)
            disk_score = load_best_score()
            if disk_score < global_best.value:
                global_best.value = disk_score
            if r.score >= global_best.value:
                return False, r.score
            
            cx, cy = r.mec[0], r.mec[1]
            raw = [{'x': round(float(xs[i]-cx), 6), 'y': round(float(ys[i]-cy), 6),
                     'theta': round(float(ts[i]), 6)} for i in range(N)]
            with open(BEST_FILE, 'w') as f:
                json.dump(raw, f, indent=2)
            global_best.value = r.score
            
            try:
                sol = [Semicircle(d['x'],d['y'],d['theta']) for d in raw]
                rv = validate_and_score(sol)
                from src.semicircle_packing.visualization import plot_packing
                plot_packing(sol, rv.mec, save_path='best_solution.png')
            except: pass
            
            return True, r.score
        finally:
            _fcntl.flock(lf, _fcntl.LOCK_UN)


def polish_worker(wid, global_best, stop_event):
    """Polish from best solution with small perturbations. Skip Phase 1."""
    os.nice(10)
    rng = np.random.default_rng(wid * 1000 + int(time.time()) % 10000)
    rnd = 0
    
    while not stop_event.is_set():
        rnd += 1
        xs, ys, ts = load_best()
        
        # Very small perturbation — we're polishing, not exploring
        noise = 0.002 + rng.uniform(0, 0.015)
        xs = xs + rng.normal(0, noise, N)
        ys = ys + rng.normal(0, noise, N)
        ts = ts + rng.normal(0, noise, N)
        
        seed_val = int(rng.integers(0, 2**31))
        
        # Very cold SA (200M steps) — polish, don't destroy
        rx, ry, rt, r_score = run_sa(
            xs, ys, ts,
            n_steps=200_000_000,
            T_start=0.05, T_end=0.00005,
            lam_start=2000, lam_end=50000,
            seed=seed_val,
            shapely_check_interval=5_000_000,
        )
        
        if rx is None:
            print(f"  [p{wid}_polish_{rnd}] no feasible", flush=True)
            continue
        
        val = official_validate(rx, ry, rt)
        if not val.valid:
            print(f"  [p{wid}_polish_{rnd}] invalid", flush=True)
            continue
        
        score = val.score
        print(f"  [p{wid}_polish_{rnd}] R={score:.6f} (best={global_best.value:.6f})", flush=True)
        
        saved, _ = save_if_better(rx, ry, rt, global_best)
        if saved:
            print(f"  [p{wid}_polish_{rnd}] ★ NEW BEST: R={score:.6f}", flush=True)


def explore_worker(wid, global_best, stop_event):
    """Explore new topologies with Phase 1 → Phase 2."""
    os.nice(10)
    np.random.seed(wid * 1000 + int(time.time()) % 10000)
    random.seed(wid * 7919 + int(time.time()) % 10000)
    
    seed_fn = SEED_FNS[wid % len(SEED_FNS)]
    seed_name = SEED_NAMES[wid % len(SEED_NAMES)]
    rnd = 0
    
    while not stop_event.is_set():
        rnd += 1
        
        init = seed_fn()
        if init is None:
            time.sleep(1)
            continue
        xs, ys, ts = init
        seed_val = rnd * 137 * (wid + 1) + int(time.time()) % 10000
        
        # Phase 1: hot compress
        rx, ry, rt, r1 = run_sa(
            xs, ys, ts,
            n_steps=20_000_000,
            T_start=3.0, T_end=0.01,
            lam_start=20, lam_end=2000,
            seed=seed_val,
            shapely_check_interval=5_000_000,
        )
        
        if rx is None:
            continue
        
        val1 = official_validate(rx, ry, rt)
        if not val1.valid or val1.score > 3.3:
            print(f"  [e{wid}_{seed_name}_{rnd}] P1: {val1.score if val1.valid else 'invalid'}", flush=True)
            continue
        
        print(f"  [e{wid}_{seed_name}_{rnd}] P1: {val1.score:.4f} → P2", flush=True)
        
        # Phase 2: cold squeeze
        best_rx, best_ry, best_rt = rx.copy(), ry.copy(), rt.copy()
        best_score = val1.score
        
        for cold in range(5):
            if stop_event.is_set():
                break
            
            noise = 0.015 + cold * 0.01
            px = best_rx + np.random.randn(N) * noise
            py = best_ry + np.random.randn(N) * noise
            pt = best_rt + np.random.randn(N) * noise * 2
            
            cx, cy, ct, r2 = run_sa(
                px, py, pt,
                n_steps=200_000_000,
                T_start=0.5, T_end=0.0002,
                lam_start=500, lam_end=20000,
                seed=seed_val * 100 + cold,
                shapely_check_interval=10_000_000,
            )
            
            if cx is None:
                continue
            
            val2 = official_validate(cx, cy, ct)
            if not val2.valid:
                continue
            
            score = val2.score
            print(f"  [e{wid}_{seed_name}_{rnd}] P2.{cold}: {score:.6f} (was {best_score:.6f})", flush=True)
            
            if score < best_score:
                best_rx, best_ry, best_rt = cx.copy(), cy.copy(), ct.copy()
                best_score = score
                
                saved, _ = save_if_better(best_rx, best_ry, best_rt, global_best)
                if saved:
                    print(f"  [e{wid}_{seed_name}_{rnd}] ★ NEW BEST: {score:.6f}", flush=True)
            else:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=int, default=21600)
    args = parser.parse_args()
    
    global_best = mp.Value('d', load_best_score())
    stop = mp.Event()
    
    print(f"run_overnight: 3 polish + 3 explore, {args.runtime}s")
    print(f"Best on disk: R={global_best.value:.6f}")
    sys.stdout.flush()
    
    procs = []
    # 3 polish workers
    for i in range(3):
        p = mp.Process(target=polish_worker, args=(i, global_best, stop))
        p.start()
        procs.append(p)
    
    # 3 explore workers
    for i in range(3, 6):
        p = mp.Process(target=explore_worker, args=(i, global_best, stop))
        p.start()
        procs.append(p)
    
    start = time.time()
    last_report = start
    
    try:
        while time.time() - start < args.runtime:
            time.sleep(5)
            now = time.time()
            if now - last_report > 120:
                elapsed = int(now - start)
                alive = sum(1 for p in procs if p.is_alive())
                print(f"\n[{elapsed}s] === Best={global_best.value:.6f} Workers={alive}/6 ===\n", flush=True)
                last_report = now
    except KeyboardInterrupt:
        pass
    
    stop.set()
    print("\nStopping workers...")
    for p in procs:
        p.join(timeout=30)
    
    print(f"Final: R={global_best.value:.6f}")


if __name__ == '__main__':
    main()
