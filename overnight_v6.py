#!/usr/bin/env python3
"""
overnight_v6.py — Hybrid phi+GJK optimizer.

Strategy:
- Phase 1: phi-SA (fast, forgiving) to find good regions (20M steps)
- Phase 2: phi-SA cold squeeze (200M steps) to compress
- Phase 3: GJK local polisher on any solution < global_best + 0.05

The phi-SA finds candidates. GJK polishes them exactly.

6 workers: 3 from best+perturb, 1 interleaved, 1 random, 1 from seeds.py
"""

import sys, os, json, math, time, random, multiprocessing as mp, argparse
import numpy as np

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

import importlib.util
spec = importlib.util.spec_from_file_location("overnight", "overnight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from sa_v2 import sa_run_v2_wrapper as phi_sa
from gjk_numba import semicircle_gjk_signed_dist
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
from seeds import seed_3_5_7, seed_2_5_8, seed_3_4_8
import fcntl as _fcntl
import numba as nb

BEST_FILE = 'best_solution.json'
LOCK_PATH = BEST_FILE + '.lock'
N = 15


# ── GJK polisher ─────────────────────────────────────────────────────────────

@nb.njit
def gjk_overlap_single(xs, ys, ts, idx):
    n = xs.shape[0]
    energy = 0.0
    for j in range(n):
        if j == idx:
            continue
        d = semicircle_gjk_signed_dist(xs[idx], ys[idx], ts[idx], xs[j], ys[j], ts[j])
        if d < 0.0:
            energy += d * d
    return energy

@nb.njit
def gjk_overlap_full(xs, ys, ts):
    n = xs.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = semicircle_gjk_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < 0.0:
                energy += d * d
    return energy

@nb.njit
def gjk_min_sep(xs, ys, ts):
    n = xs.shape[0]
    mn = 1e18
    for i in range(n):
        for j in range(i + 1, n):
            d = semicircle_gjk_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < mn:
                mn = d
    return mn

@nb.njit
def r_single_nb(x, y, t):
    r = math.sqrt((x + math.cos(t))**2 + (y + math.sin(t))**2)
    px, py = -math.sin(t), math.cos(t)
    r1 = math.sqrt((x + px)**2 + (y + py)**2)
    r2 = math.sqrt((x - px)**2 + (y - py)**2)
    if r1 > r: r = r1
    if r2 > r: r = r2
    for k in range(16):
        a = t - math.pi/2 + math.pi * k / 15
        rx = math.sqrt((x + math.cos(a))**2 + (y + math.sin(a))**2)
        if rx > r: r = rx
    return r

@nb.njit
def r_max_nb(xs, ys, ts):
    n = xs.shape[0]
    rm = 0.0
    for i in range(n):
        ri = r_single_nb(xs[i], ys[i], ts[i])
        if ri > rm: rm = ri
    return rm

@nb.njit
def gjk_polish(xs, ys, ts, n_steps, T_start, T_end, lam, seed):
    """
    GJK-exact local polisher. Very cold SA with tiny moves.
    Designed to push a near-optimal solution down by fractions.
    """
    np.random.seed(seed)
    n = xs.shape[0]
    
    cur_xs = xs.copy()
    cur_ys = ys.copy()
    cur_ts = ts.copy()
    
    cur_ovlp = gjk_overlap_full(cur_xs, cur_ys, cur_ts)
    cur_R = r_max_nb(cur_xs, cur_ys, cur_ts)
    
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_ts = ts.copy()
    best_R = 1e18
    if cur_ovlp < 1e-15:
        best_R = cur_R
        best_xs[:] = cur_xs[:]
        best_ys[:] = cur_ys[:]
        best_ts[:] = cur_ts[:]
    
    scale = 0.005  # very small moves
    log_T = math.log(T_end / T_start)
    
    for step in range(n_steps):
        frac = step / n_steps
        T = T_start * math.exp(log_T * frac)
        
        idx = int(np.random.random() * n) % n
        old_x, old_y, old_t = cur_xs[idx], cur_ys[idx], cur_ts[idx]
        old_ovlp_i = gjk_overlap_single(cur_xs, cur_ys, cur_ts, idx)
        old_ri = r_single_nb(old_x, old_y, old_t)
        
        r = np.random.random()
        if r < 0.6:
            cur_xs[idx] += np.random.normal(0, scale)
            cur_ys[idx] += np.random.normal(0, scale)
        elif r < 0.85:
            cur_ts[idx] += np.random.normal(0, scale * 2)
        else:
            cur_xs[idx] += np.random.normal(0, scale * 0.5)
            cur_ys[idx] += np.random.normal(0, scale * 0.5)
            cur_ts[idx] += np.random.normal(0, scale)
        
        new_ovlp_i = gjk_overlap_single(cur_xs, cur_ys, cur_ts, idx)
        new_ri = r_single_nb(cur_xs[idx], cur_ys[idx], cur_ts[idx])
        
        new_ovlp = cur_ovlp - old_ovlp_i + new_ovlp_i
        new_R = cur_R
        if new_ri > cur_R:
            new_R = new_ri
        elif abs(old_ri - cur_R) < 1e-10:
            new_R = r_max_nb(cur_xs, cur_ys, cur_ts)
        
        old_obj = cur_R + lam * cur_ovlp
        new_obj = new_R + lam * new_ovlp
        delta = new_obj - old_obj
        
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-15)):
            cur_ovlp = new_ovlp
            cur_R = new_R
            if new_ovlp < 1e-15 and new_R < best_R:
                best_R = new_R
                best_xs[:] = cur_xs[:]
                best_ys[:] = cur_ys[:]
                best_ts[:] = cur_ts[:]
        else:
            cur_xs[idx] = old_x
            cur_ys[idx] = old_y
            cur_ts[idx] = old_t
    
    return best_xs, best_ys, best_ts, best_R


# ── Helpers ───────────────────────────────────────────────────────────────────

def official_validate(xs, ys, ts):
    raw = [{'x': float(xs[i]), 'y': float(ys[i]), 'theta': float(ts[i])} for i in range(N)]
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    return validate_and_score(sol)

def load_best_score():
    try:
        with open(BEST_FILE) as f:
            raw = json.load(f)
        result = official_validate(
            np.array([s['x'] for s in raw]),
            np.array([s['y'] for s in raw]),
            np.array([s['theta'] for s in raw]),
        )
        return result.score if result.valid else float('inf')
    except:
        return float('inf')

def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    return (np.array([s['x'] for s in raw]),
            np.array([s['y'] for s in raw]),
            np.array([s['theta'] for s in raw]))

def save_if_better(xs, ys, ts, score, global_best):
    if score >= global_best.value:
        return False
    with open(LOCK_PATH, 'w') as lf:
        try:
            _fcntl.flock(lf, _fcntl.LOCK_EX)
            disk_score = load_best_score()
            if disk_score < global_best.value:
                global_best.value = disk_score
            if score >= global_best.value:
                return False
            
            result = official_validate(xs, ys, ts)
            if not result.valid or result.score >= global_best.value:
                return False
            
            cx, cy = result.mec[0], result.mec[1]
            raw = [{'x': round(float(xs[i]-cx), 6), 'y': round(float(ys[i]-cy), 6),
                     'theta': round(float(ts[i]), 6)} for i in range(N)]
            with open(BEST_FILE, 'w') as f:
                json.dump(raw, f, indent=2)
            
            global_best.value = result.score
            
            # Update visualization
            try:
                sol = [Semicircle(d['x'],d['y'],d['theta']) for d in raw]
                r = validate_and_score(sol)
                from src.semicircle_packing.visualization import plot_packing
                plot_packing(sol, r.mec, save_path='best_solution.png')
            except: pass
            
            return True
        finally:
            _fcntl.flock(lf, _fcntl.LOCK_UN)


# ── Seed generators ──────────────────────────────────────────────────────────

def seed_best_perturb(rng):
    xs, ys, ts = load_best()
    noise = 0.01 + rng.uniform(0, 0.08)
    return (xs + rng.normal(0, noise, N),
            ys + rng.normal(0, noise, N),
            ts + rng.normal(0, noise, N))

def seed_interleaved(rng):
    xs, ys, ts = np.zeros(N), np.zeros(N), np.zeros(N)
    idx = 0
    for row in range(3):
        y = (row - 1) * 1.4
        for col in range(5):
            x = (col - 2) * 1.1
            xs[idx] = x + rng.uniform(-0.2, 0.2)
            ys[idx] = y + rng.uniform(-0.2, 0.2)
            ts[idx] = (math.pi/2 if (row+col)%2==0 else -math.pi/2) + rng.uniform(-0.4, 0.4)
            idx += 1
    return xs, ys, ts

def seed_from_seeds_py(rng):
    fn = rng.choice([seed_3_5_7, seed_2_5_8, seed_3_4_8])
    result = fn()
    if result is None:
        return seed_interleaved(rng)
    return result


WORKERS = [
    ("best1", seed_best_perturb),
    ("best2", seed_best_perturb),
    ("best3", seed_best_perturb),
    ("inter", seed_interleaved),
    ("seeds", seed_from_seeds_py),
    ("random", lambda rng: (rng.uniform(-2.5,2.5,N), rng.uniform(-2.5,2.5,N), rng.uniform(0,2*math.pi,N))),
]


# ── Worker ───────────────────────────────────────────────────────────────────

def worker(wid, name, seed_fn, global_best, runtime, log_q):
    os.nice(10)
    rng = np.random.default_rng(wid * 1000 + int(time.time()) % 10000)
    start = time.time()
    rnd = 0
    
    # Warmup GJK JIT
    _dummy = gjk_overlap_full(np.zeros(3), np.zeros(3), np.zeros(3))
    _dummy2 = gjk_polish(np.zeros(N), np.zeros(N), np.zeros(N), 100, 0.01, 0.001, 1000.0, 42)
    
    while time.time() - start < runtime:
        rnd += 1
        
        xs, ys, ts = seed_fn(rng)
        seed_val = int(rng.integers(0, 2**31))
        
        # Phase 1: phi-SA hot compress (20M steps)
        rx, ry, rt, r1 = phi_sa(
            xs, ys, ts,
            n_steps=20_000_000,
            T_start=3.0, T_end=0.01,
            lam_start=20, lam_end=2000,
            seed=seed_val,
            shapely_check_interval=5_000_000,
        )
        
        if rx is None:
            continue
        
        val = official_validate(rx, ry, rt)
        if not val.valid:
            continue
        
        r1_score = val.score
        if r1_score > 3.2:
            log_q.put(f"[w{wid}_{name}_{rnd}] P1: {r1_score:.4f} (skip)")
            continue
        
        log_q.put(f"[w{wid}_{name}_{rnd}] P1: {r1_score:.4f} → P2")
        
        # Phase 2: phi-SA cold squeeze (200M steps)
        best_rx, best_ry, best_rt = rx.copy(), ry.copy(), rt.copy()
        best_score = r1_score
        
        for cold in range(3):
            noise = 0.015 + cold * 0.01
            px = best_rx + rng.normal(0, noise, N)
            py = best_ry + rng.normal(0, noise, N)
            pt = best_rt + rng.normal(0, noise * 2, N)
            
            cx, cy, ct, r2 = phi_sa(
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
            
            r2_score = val2.score
            log_q.put(f"[w{wid}_{name}_{rnd}] P2.{cold}: {r2_score:.6f} (was {best_score:.6f})")
            
            if r2_score < best_score:
                best_rx, best_ry, best_rt = cx.copy(), cy.copy(), ct.copy()
                best_score = r2_score
                
                saved = save_if_better(best_rx, best_ry, best_rt, best_score, global_best)
                if saved:
                    log_q.put(f"[w{wid}_{name}_{rnd}] ★ P2 BEST: {best_score:.6f}")
            else:
                break
        
        # Phase 3: GJK polish (if close to global best)
        if best_score < global_best.value + 0.05:
            log_q.put(f"[w{wid}_{name}_{rnd}] P3: GJK polish from {best_score:.6f}")
            
            for polish_round in range(3):
                px, py, pt, pr = gjk_polish(
                    best_rx, best_ry, best_rt,
                    n_steps=5_000_000,
                    T_start=0.002, T_end=0.00001,
                    lam=50000.0,
                    seed=int(rng.integers(0, 2**31)),
                )
                
                if pr < best_score:
                    val3 = official_validate(px, py, pt)
                    if val3.valid and val3.score < best_score:
                        best_rx, best_ry, best_rt = px.copy(), py.copy(), pt.copy()
                        best_score = val3.score
                        log_q.put(f"[w{wid}_{name}_{rnd}] P3.{polish_round}: {best_score:.6f}")
                        
                        saved = save_if_better(best_rx, best_ry, best_rt, best_score, global_best)
                        if saved:
                            log_q.put(f"[w{wid}_{name}_{rnd}] ★ P3 BEST: {best_score:.6f}")
                    else:
                        break
                else:
                    break


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--runtime', type=int, default=21600)
    args = parser.parse_args()
    
    global_best = mp.Value('d', load_best_score())
    log_q = mp.Queue()
    
    print(f"overnight_v6: {args.workers} workers, {args.runtime}s (until 5am ET)")
    print(f"Best on disk: R={global_best.value:.6f}")
    print(f"Strategy: phi-SA → phi-SA cold → GJK polish")
    sys.stdout.flush()
    
    procs = []
    for i in range(args.workers):
        name, fn = WORKERS[i % len(WORKERS)]
        p = mp.Process(target=worker, args=(i, name, fn, global_best, args.runtime, log_q))
        p.start()
        procs.append(p)
    
    start = time.time()
    last_report = start
    
    while any(p.is_alive() for p in procs):
        try:
            while not log_q.empty():
                msg = log_q.get_nowait()
                elapsed = int(time.time() - start)
                print(f"[{elapsed}s] {msg}")
                sys.stdout.flush()
        except:
            pass
        
        now = time.time()
        if now - last_report > 120:
            elapsed = int(now - start)
            alive = sum(1 for p in procs if p.is_alive())
            print(f"\n[{elapsed}s] === Best={global_best.value:.6f} Workers={alive}/{args.workers} ===\n")
            sys.stdout.flush()
            last_report = now
        
        time.sleep(0.5)
    
    while not log_q.empty():
        try:
            print(log_q.get_nowait())
        except:
            break
    
    print(f"\nFinal: R={global_best.value:.6f}")


if __name__ == '__main__':
    main()
