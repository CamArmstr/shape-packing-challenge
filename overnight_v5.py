#!/usr/bin/env python3
"""
overnight_v5.py — GJK-powered optimizer using sa_v3 (exact GJK distance).

Uses the full sa_v2 structure (Thompson Sampling, swap, cluster, squeeze, 
adaptive step, stagnation escape) but with GJK exact distance instead of 
the broken phi-function approximation.

6 workers with different seed topologies. Runs until killed or --runtime.
"""

import sys, os, json, math, time, random, multiprocessing as mp, argparse
import numpy as np

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from sa_v3 import sa_run_v2 as sa_run_v3, overlap_energy_nb, r_exact_nb
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
import fcntl as _fcntl

BEST_FILE = 'best_solution.json'
LOCK_PATH = BEST_FILE + '.lock'
N = 15


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


def save_if_better(xs, ys, ts, global_best):
    """Validate officially, center, and save if improved."""
    result = official_validate(xs, ys, ts)
    if not result.valid:
        return False, None, result.errors[:1] if result.errors else ["unknown"]
    
    score = result.score
    if score >= global_best.value:
        return False, score, None
    
    with open(LOCK_PATH, 'w') as lf:
        try:
            _fcntl.flock(lf, _fcntl.LOCK_EX)
            
            # Re-check disk
            disk_score = load_best_score()
            if disk_score < global_best.value:
                global_best.value = disk_score
            if score >= global_best.value:
                return False, score, None
            
            # Center
            cx, cy = result.mec[0], result.mec[1]
            cxs = xs - cx
            cys = ys - cy
            
            raw = [{'x': round(float(cxs[i]), 6), 'y': round(float(cys[i]), 6),
                     'theta': round(float(ts[i]), 6)} for i in range(N)]
            with open(BEST_FILE, 'w') as f:
                json.dump(raw, f, indent=2)
            
            global_best.value = score
            return True, score, None
        finally:
            _fcntl.flock(lf, _fcntl.LOCK_UN)


# ── Seed generators ──────────────────────────────────────────────────────────

def seed_from_best(rng=None):
    """Start from current best with perturbation."""
    try:
        with open(BEST_FILE) as f:
            raw = json.load(f)
        xs = np.array([s['x'] for s in raw])
        ys = np.array([s['y'] for s in raw])
        ts = np.array([s['theta'] for s in raw])
        return xs, ys, ts
    except:
        return seed_random(rng=rng)

def seed_random(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    xs = rng.uniform(-2, 2, N)
    ys = rng.uniform(-2, 2, N)
    ts = rng.uniform(0, 2 * math.pi, N)
    return xs, ys, ts

def seed_interleaved(rng=None):
    """Alternating up/down rows. Best topology from initial testing."""
    if rng is None:
        rng = np.random.default_rng()
    xs = np.zeros(N)
    ys = np.zeros(N)
    ts = np.zeros(N)
    idx = 0
    for row in range(3):
        y_base = (row - 1) * 1.4
        for col in range(5):
            x_base = (col - 2) * 1.1
            xs[idx] = x_base + rng.uniform(-0.15, 0.15)
            ys[idx] = y_base + rng.uniform(-0.15, 0.15)
            if (row + col) % 2 == 0:
                ts[idx] = math.pi / 2 + rng.uniform(-0.3, 0.3)
            else:
                ts[idx] = -math.pi / 2 + rng.uniform(-0.3, 0.3)
            idx += 1
    return xs, ys, ts

def seed_flower(rng=None):
    """Outer ring arcs-in, inner cluster. Second-best topology."""
    if rng is None:
        rng = np.random.default_rng()
    xs = np.zeros(N)
    ys = np.zeros(N)
    ts = np.zeros(N)
    idx = 0
    for k in range(4):
        a = k * math.pi / 2 + rng.uniform(-0.4, 0.4)
        r = 0.6 + rng.uniform(-0.2, 0.2)
        xs[idx] = r * math.cos(a)
        ys[idx] = r * math.sin(a)
        ts[idx] = rng.uniform(0, 2 * math.pi)
        idx += 1
    for k in range(11):
        a = k * 2 * math.pi / 11 + rng.uniform(-0.15, 0.15)
        r = 2.0 + rng.uniform(-0.15, 0.15)
        xs[idx] = r * math.cos(a)
        ys[idx] = r * math.sin(a)
        ts[idx] = a + math.pi + rng.uniform(-0.4, 0.4)
        idx += 1
    return xs, ys, ts

def seed_concentric(rng=None):
    """3-5-7 concentric rings."""
    if rng is None:
        rng = np.random.default_rng()
    xs = np.zeros(N)
    ys = np.zeros(N)
    ts = np.zeros(N)
    idx = 0
    # Inner ring: 3
    for k in range(3):
        a = k * 2 * math.pi / 3 + rng.uniform(-0.3, 0.3)
        r = 0.7 + rng.uniform(-0.1, 0.1)
        xs[idx] = r * math.cos(a)
        ys[idx] = r * math.sin(a)
        ts[idx] = a + math.pi + rng.uniform(-0.4, 0.4)
        idx += 1
    # Middle ring: 5
    for k in range(5):
        a = k * 2 * math.pi / 5 + rng.uniform(-0.2, 0.2)
        r = 1.5 + rng.uniform(-0.1, 0.1)
        xs[idx] = r * math.cos(a)
        ys[idx] = r * math.sin(a)
        ts[idx] = a + rng.choice(np.array([0.0, math.pi])) + rng.uniform(-0.3, 0.3)
        idx += 1
    # Outer ring: 7
    for k in range(7):
        a = k * 2 * math.pi / 7 + rng.uniform(-0.15, 0.15)
        r = 2.2 + rng.uniform(-0.1, 0.1)
        xs[idx] = r * math.cos(a)
        ys[idx] = r * math.sin(a)
        ts[idx] = a + math.pi + rng.uniform(-0.3, 0.3)
        idx += 1
    return xs, ys, ts


SEED_CONFIGS = [
    ("best_perturb", seed_from_best),
    ("best_perturb2", seed_from_best),
    ("interleaved", seed_interleaved),
    ("flower", seed_flower),
    ("concentric", seed_concentric),
    ("random", seed_random),
]


# ── SA wrapper (handles Shapely gating) ──────────────────────────────────────

def run_sa_phase(xs, ys, ts, n_steps, T_start, T_end, lam_start, lam_end, seed, 
                 shapely_interval=2_000_000):
    """Run SA with periodic Shapely validation."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    
    n_chunks = max(1, n_steps // shapely_interval)
    chunk_size = n_steps // n_chunks
    
    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
    best_score = float('inf')
    
    cur_xs, cur_ys, cur_ts = xs.copy(), ys.copy(), ts.copy()
    phi_best_r = float('inf')
    phi_best_xs = xs.copy()
    phi_best_ys = ys.copy()
    phi_best_ts = ts.copy()
    
    rng = np.random.default_rng(seed + 7777)
    
    for chunk in range(n_chunks):
        f0 = chunk / n_chunks
        f1 = (chunk + 1) / n_chunks
        
        T_s = T_start * (T_end / T_start) ** f0
        T_e = T_start * (T_end / T_start) ** f1
        lam_s = lam_start * (lam_end / lam_start) ** f0
        lam_e = lam_start * (lam_end / lam_start) ** f1
        
        bx, by, bt, br, found, _, _ = sa_run_v3(
            cur_xs, cur_ys, cur_ts,
            chunk_size, T_s, T_e, lam_s, lam_e,
            seed * 1000 + chunk
        )
        
        if found and br < phi_best_r:
            phi_best_r = br
            phi_best_xs[:] = bx
            phi_best_ys[:] = by
            phi_best_ts[:] = bt
        
        # Shapely check
        if found and phi_best_r < best_score - 1e-6:
            result = official_validate(phi_best_xs, phi_best_ys, phi_best_ts)
            if result.valid and result.score < best_score:
                best_score = result.score
                best_xs[:] = phi_best_xs
                best_ys[:] = phi_best_ys
                best_ts[:] = phi_best_ts
        
        cur_xs[:] = bx
        cur_ys[:] = by
        cur_ts[:] = bt
    
    return best_xs, best_ys, best_ts, best_score


# ── Worker ───────────────────────────────────────────────────────────────────

def worker(wid, seed_name, seed_fn, global_best, runtime, log_q):
    rng = np.random.default_rng(wid * 1000 + int(time.time()) % 10000)
    start = time.time()
    rnd = 0
    
    while time.time() - start < runtime:
        rnd += 1
        
        # Generate seed
        if "best" in seed_name:
            xs, ys, ts = seed_fn()
            noise = 0.01 + rng.uniform(0, 0.05)
            xs = xs + rng.normal(0, noise, N)
            ys = ys + rng.normal(0, noise, N)
            ts = ts + rng.normal(0, noise, N)
        else:
            xs, ys, ts = seed_fn(rng=rng)
        
        # Phase 1: Hot compress (10M steps, higher T for GJK strictness)
        seed_val = int(rng.integers(0, 2**31))
        bx, by, bt, br = run_sa_phase(
            xs, ys, ts,
            n_steps=10_000_000,
            T_start=0.8, T_end=0.005,
            lam_start=100.0, lam_end=5000.0,
            seed=seed_val,
            shapely_interval=2_500_000,
        )
        
        if br > 3.2 or br == float('inf'):
            log_q.put(f"[w{wid}_{seed_name}_{rnd}] P1: R={br:.4f} (skip)")
            continue
        
        log_q.put(f"[w{wid}_{seed_name}_{rnd}] P1: R={br:.4f} → P2")
        
        # Try to save Phase 1 result
        if br < global_best.value:
            saved, score, errs = save_if_better(bx, by, bt, global_best)
            if saved:
                log_q.put(f"[w{wid}_{seed_name}_{rnd}] ★ P1 BEST: R={score:.6f}")
        
        # Phase 2: Cold squeeze (50M steps, repeated)
        for sq in range(5):
            seed_val = int(rng.integers(0, 2**31))
            bx2, by2, bt2, br2 = run_sa_phase(
                bx, by, bt,
                n_steps=50_000_000,
                T_start=0.015, T_end=0.00005,
                lam_start=3000.0, lam_end=100000.0,
                seed=seed_val,
                shapely_interval=5_000_000,
            )
            
            if br2 < br:
                log_q.put(f"[w{wid}_{seed_name}_{rnd}] P2.{sq}: R={br2:.6f} (from {br:.6f})")
                bx, by, bt, br = bx2, by2, bt2, br2
                
                if br < global_best.value:
                    saved, score, errs = save_if_better(bx, by, bt, global_best)
                    if saved:
                        log_q.put(f"[w{wid}_{seed_name}_{rnd}] ★ NEW BEST: R={score:.6f}")
                    elif errs:
                        log_q.put(f"[w{wid}_{seed_name}_{rnd}] invalid: {errs}")
            else:
                brs = f"{br2:.6f}" if br2 < float('inf') else "inf"
                log_q.put(f"[w{wid}_{seed_name}_{rnd}] P2.{sq}: R={brs} (no imp)")
                break


# ── Main ───���─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--runtime', type=int, default=7200)
    args = parser.parse_args()
    
    global_best = mp.Value('d', load_best_score())
    log_q = mp.Queue()
    
    print(f"overnight_v5: {args.workers} workers, {args.runtime}s")
    print(f"Best on disk: R={global_best.value:.6f}")
    print(f"Seeds: {[s[0] for s in SEED_CONFIGS]}")
    sys.stdout.flush()
    
    procs = []
    for i in range(args.workers):
        sn, sf = SEED_CONFIGS[i % len(SEED_CONFIGS)]
        p = mp.Process(target=worker, args=(i, sn, sf, global_best, args.runtime, log_q))
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
        if now - last_report > 60:
            elapsed = int(now - start)
            alive = sum(1 for p in procs if p.is_alive())
            print(f"\n[{elapsed}s] === Best={global_best.value:.6f} Workers={alive}/{args.workers} ===\n")
            sys.stdout.flush()
            last_report = now
        
        time.sleep(0.5)
    
    # Drain
    while not log_q.empty():
        try:
            print(log_q.get_nowait())
        except:
            break
    
    print(f"\nFinal best: R={global_best.value:.6f}")


if __name__ == '__main__':
    main()
