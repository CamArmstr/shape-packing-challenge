#!/usr/bin/env python3
"""
overnight_gjk.py — GJK-powered overnight optimizer.

Uses exact GJK distance (685k single-pair checks/sec) instead of phi-function.
6 workers, each cycling different starting topologies.
Runs until killed or --runtime seconds elapsed.

Key insight: phi_pair_nb was wildly inaccurate on many pair types.
GJK gives exact convex distance for semicircles.
"""

import sys, os, json, math, time, random, multiprocessing as mp, argparse
import numpy as np
import numba as nb

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from gjk_numba import semicircle_gjk_signed_dist
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
import fcntl as _fcntl

BEST_FILE = 'best_solution.json'
LOCK_PATH = BEST_FILE + '.lock'
N = 15

# ── GJK-based SA kernel ─────────────────────────────────────────────────────

@nb.njit
def overlap_energy_single(xs, ys, ts, idx):
    """Overlap energy for semicircle idx vs all others. GJK exact."""
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
def overlap_energy_full(xs, ys, ts):
    """Total overlap energy. GJK exact."""
    n = xs.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = semicircle_gjk_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < 0.0:
                energy += d * d
    return energy

@nb.njit
def r_single(x, y, t):
    """Max distance from origin to any point of semicircle (x,y,t)."""
    # Arc apex
    r = math.sqrt((x + math.cos(t))**2 + (y + math.sin(t))**2)
    # Diameter endpoints
    px = -math.sin(t)
    py = math.cos(t)
    r1 = math.sqrt((x + px)**2 + (y + py)**2)
    r2 = math.sqrt((x - px)**2 + (y - py)**2)
    if r1 > r:
        r = r1
    if r2 > r:
        r = r2
    # Sample a few arc points for safety
    for k in range(8):
        a = t - math.pi/2 + math.pi * k / 7
        rx = math.sqrt((x + math.cos(a))**2 + (y + math.sin(a))**2)
        if rx > r:
            r = rx
    return r

@nb.njit
def r_max(xs, ys, ts):
    """Enclosing radius (origin-centered)."""
    n = xs.shape[0]
    rm = 0.0
    for i in range(n):
        ri = r_single(xs[i], ys[i], ts[i])
        if ri > rm:
            rm = ri
    return rm

@nb.njit
def sa_gjk(xs, ys, ts, n_steps, T_start, T_end, lam_start, lam_end, seed):
    """
    Penalty SA with GJK exact distance.
    Objective: R + lambda * overlap_energy
    Returns (best_xs, best_ys, best_ts, best_R, feasible_found)
    """
    np.random.seed(seed)
    n = xs.shape[0]
    
    cur_xs = xs.copy()
    cur_ys = ys.copy()
    cur_ts = ts.copy()
    
    cur_ovlp = overlap_energy_full(cur_xs, cur_ys, cur_ts)
    cur_R = r_max(cur_xs, cur_ys, cur_ts)
    
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_ts = ts.copy()
    best_R = 1e18
    feasible = False
    
    log_T = math.log(T_end / T_start)
    log_lam = math.log(lam_end / lam_start)
    
    scale = 0.15
    accept = 0
    total = 0
    
    for step in range(n_steps):
        frac = step / n_steps
        T = T_start * math.exp(log_T * frac)
        lam = lam_start * math.exp(log_lam * frac)
        
        # Pick random semicircle
        idx = int(np.random.random() * n) % n
        
        # Save old state
        old_x = cur_xs[idx]
        old_y = cur_ys[idx]
        old_t = cur_ts[idx]
        old_ovlp_i = overlap_energy_single(cur_xs, cur_ys, cur_ts, idx)
        old_r_i = r_single(old_x, old_y, old_t)
        
        # Move type: 70% translate, 20% rotate, 10% both
        r = np.random.random()
        if r < 0.7:
            cur_xs[idx] += np.random.normal(0, scale)
            cur_ys[idx] += np.random.normal(0, scale)
        elif r < 0.9:
            cur_ts[idx] += np.random.normal(0, scale * 2)
        else:
            cur_xs[idx] += np.random.normal(0, scale * 0.7)
            cur_ys[idx] += np.random.normal(0, scale * 0.7)
            cur_ts[idx] += np.random.normal(0, scale)
        
        # Compute new energy (incremental)
        new_ovlp_i = overlap_energy_single(cur_xs, cur_ys, cur_ts, idx)
        new_r_i = r_single(cur_xs[idx], cur_ys[idx], cur_ts[idx])
        
        new_ovlp = cur_ovlp - old_ovlp_i + new_ovlp_i
        
        # R change: only matters if this semicircle was the max or is now
        new_R = cur_R
        if new_r_i > cur_R:
            new_R = new_r_i
        elif abs(old_r_i - cur_R) < 1e-10:
            # Was the max, need full recompute
            new_R = r_max(cur_xs, cur_ys, cur_ts)
        
        old_obj = cur_R + lam * cur_ovlp
        new_obj = new_R + lam * new_ovlp
        delta = new_obj - old_obj
        
        # Accept?
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            cur_ovlp = new_ovlp
            cur_R = new_R
            accept += 1
            
            # Track best feasible
            if new_ovlp < 1e-12 and new_R < best_R:
                best_R = new_R
                best_xs[:] = cur_xs[:]
                best_ys[:] = cur_ys[:]
                best_ts[:] = cur_ts[:]
                feasible = True
        else:
            # Reject: restore
            cur_xs[idx] = old_x
            cur_ys[idx] = old_y
            cur_ts[idx] = old_t
        
        total += 1
        
        # Adaptive step size every 10k
        if total % 10000 == 0 and total > 0:
            rate = accept / total
            if rate > 0.4:
                scale *= 1.05
            elif rate < 0.2:
                scale *= 0.95
            accept = 0
            total = 0
    
    return best_xs, best_ys, best_ts, best_R, feasible


# ── Seed generators ──────────────────────────────────────────────────────────

def seed_random_scatter(R_init=3.5, rng=None):
    """Random scatter within circle of radius R_init."""
    if rng is None:
        rng = np.random.default_rng()
    xs = np.zeros(N)
    ys = np.zeros(N)
    ts = np.zeros(N)
    for i in range(N):
        while True:
            x = rng.uniform(-R_init + 1, R_init - 1)
            y = rng.uniform(-R_init + 1, R_init - 1)
            if x*x + y*y < (R_init - 1)**2:
                xs[i] = x
                ys[i] = y
                ts[i] = rng.uniform(0, 2 * math.pi)
                break
    return xs, ys, ts

def seed_hex_pairs(rng=None):
    """7 pairs + 1 singleton in hex arrangement (the classical approach)."""
    if rng is None:
        rng = np.random.default_rng()
    # Hex centers for 7 circles of radius 1
    centers = [(0, 0)]
    for k in range(6):
        a = k * math.pi / 3
        centers.append((2 * math.cos(a), 2 * math.sin(a)))
    
    xs = np.zeros(N)
    ys = np.zeros(N)
    ts = np.zeros(N)
    
    idx = 0
    for ci, (cx, cy) in enumerate(centers):
        if ci < 7:
            # Pair: two semicircles facing opposite directions
            t = rng.uniform(0, 2 * math.pi)
            xs[idx] = cx
            ys[idx] = cy
            ts[idx] = t
            idx += 1
            if idx < N:
                xs[idx] = cx
                ys[idx] = cy
                ts[idx] = t + math.pi
                idx += 1
    
    # Last singleton
    if idx < N:
        xs[idx] = 0
        ys[idx] = 0
        ts[idx] = rng.uniform(0, 2 * math.pi)
    
    return xs, ys, ts

def seed_pinwheel(rng=None):
    """Pinwheel: all semicircles arranged tangentially around concentric rings.
    No pairing. Flat edges face tangent direction."""
    if rng is None:
        rng = np.random.default_rng()
    xs = np.zeros(N)
    ys = np.zeros(N)
    ts = np.zeros(N)
    
    # Ring 1: 5 semicircles at radius ~1.0
    # Ring 2: 8 semicircles at radius ~2.2
    # Center: 2 semicircles
    
    idx = 0
    # Center: 2 semicircles, opposing
    for k in range(2):
        xs[idx] = 0
        ys[idx] = 0
        ts[idx] = k * math.pi + rng.uniform(-0.3, 0.3)
        idx += 1
    
    # Ring 1: 5 at r=1.2, tangential
    for k in range(5):
        a = k * 2 * math.pi / 5 + rng.uniform(-0.2, 0.2)
        r = 1.2 + rng.uniform(-0.1, 0.1)
        xs[idx] = r * math.cos(a)
        ys[idx] = r * math.sin(a)
        # Tangential: theta perpendicular to radial
        ts[idx] = a + math.pi / 2 + rng.uniform(-0.3, 0.3)
        idx += 1
    
    # Ring 2: 8 at r=2.3, tangential
    for k in range(8):
        a = k * 2 * math.pi / 8 + rng.uniform(-0.15, 0.15)
        r = 2.3 + rng.uniform(-0.1, 0.1)
        xs[idx] = r * math.cos(a)
        ys[idx] = r * math.sin(a)
        ts[idx] = a + math.pi / 2 + rng.uniform(-0.3, 0.3)
        idx += 1
    
    return xs, ys, ts

def seed_flower(rng=None):
    """Flower: outer ring faces inward (arcs toward center), inner cluster."""
    if rng is None:
        rng = np.random.default_rng()
    xs = np.zeros(N)
    ys = np.zeros(N)
    ts = np.zeros(N)
    
    idx = 0
    # Inner: 4 semicircles near center, random orientation
    for k in range(4):
        a = k * math.pi / 2 + rng.uniform(-0.3, 0.3)
        r = 0.6 + rng.uniform(-0.2, 0.2)
        xs[idx] = r * math.cos(a)
        ys[idx] = r * math.sin(a)
        ts[idx] = rng.uniform(0, 2 * math.pi)
        idx += 1
    
    # Outer: 11 semicircles, arcs facing inward
    for k in range(11):
        a = k * 2 * math.pi / 11 + rng.uniform(-0.1, 0.1)
        r = 2.1 + rng.uniform(-0.1, 0.1)
        xs[idx] = r * math.cos(a)
        ys[idx] = r * math.sin(a)
        # Arc faces inward: theta points toward origin
        ts[idx] = a + math.pi + rng.uniform(-0.3, 0.3)
        idx += 1
    
    return xs, ys, ts

def seed_interleaved(rng=None):
    """Interleaved: alternating orientations, flat edges nesting against arcs."""
    if rng is None:
        rng = np.random.default_rng()
    xs = np.zeros(N)
    ys = np.zeros(N)
    ts = np.zeros(N)
    
    idx = 0
    # 3 rows of 5, alternating up/down
    for row in range(3):
        y_base = (row - 1) * 1.4
        for col in range(5):
            x_base = (col - 2) * 1.1
            xs[idx] = x_base + rng.uniform(-0.1, 0.1)
            ys[idx] = y_base + rng.uniform(-0.1, 0.1)
            # Alternate: even=up, odd=down
            if (row + col) % 2 == 0:
                ts[idx] = math.pi / 2 + rng.uniform(-0.2, 0.2)
            else:
                ts[idx] = -math.pi / 2 + rng.uniform(-0.2, 0.2)
            idx += 1
    
    return xs, ys, ts

def seed_from_best():
    """Start from current best with perturbation."""
    try:
        with open(BEST_FILE) as f:
            raw = json.load(f)
        xs = np.array([s['x'] for s in raw])
        ys = np.array([s['y'] for s in raw])
        ts = np.array([s['theta'] for s in raw])
        return xs, ys, ts
    except:
        return seed_random_scatter()


SEED_GENERATORS = [
    ("best+perturb", seed_from_best),
    ("random", seed_random_scatter),
    ("hex_pairs", seed_hex_pairs),
    ("pinwheel", seed_pinwheel),
    ("flower", seed_flower),
    ("interleaved", seed_interleaved),
]


# ── MEC centering ────────────────────────────────────────────────────────────

def official_validate(xs, ys, ts):
    raw = [{'x': float(xs[i]), 'y': float(ys[i]), 'theta': float(ts[i])} for i in range(N)]
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    return validate_and_score(sol)

def center_solution(xs, ys, ts):
    """Center solution so MEC is at origin."""
    result = official_validate(xs, ys, ts)
    if result.valid:
        cx, cy = result.mec[0], result.mec[1]
        return xs - cx, ys - cy, ts, result.score
    return xs, ys, ts, None


# ── File I/O ─────────────────────────────────────────────────────────────────

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

def save_if_better(xs, ys, ts, score, global_best):
    if score >= global_best.value:
        return False
    with open(LOCK_PATH, 'w') as lf:
        try:
            _fcntl.flock(lf, _fcntl.LOCK_EX)
            # Re-check disk
            disk_score = load_best_score()
            if disk_score < global_best.value:
                global_best.value = disk_score
            if score >= global_best.value:
                return False
            
            # Center and save
            cxs, cys, cts, cscore = center_solution(xs, ys, ts)
            if cscore is None or cscore >= global_best.value:
                return False
            
            raw = [{'x': round(float(cxs[i]), 6), 'y': round(float(cys[i]), 6),
                     'theta': round(float(cts[i]), 6)} for i in range(N)]
            with open(BEST_FILE, 'w') as f:
                json.dump(raw, f, indent=2)
            
            global_best.value = cscore
            return True
        finally:
            _fcntl.flock(lf, _fcntl.LOCK_UN)


# ── Worker ───────────────────────────────────────────────────────────────────

def worker(worker_id, seed_name, seed_fn, global_best, runtime, log_queue):
    rng = np.random.default_rng(worker_id * 1000 + int(time.time()) % 10000)
    start = time.time()
    round_num = 0
    
    while time.time() - start < runtime:
        round_num += 1
        
        # Generate starting config
        if seed_name == "best+perturb":
            xs, ys, ts = seed_fn()
            noise = 0.05 + rng.uniform(0, 0.15)
            xs += rng.normal(0, noise, N)
            ys += rng.normal(0, noise, N)
            ts += rng.normal(0, noise * 2, N)
        else:
            xs, ys, ts = seed_fn(rng=rng)
        
        # Phase 1: Hot compress (2M steps)
        bx, by, bt, br, found = sa_gjk(
            xs, ys, ts,
            n_steps=2_000_000,
            T_start=0.5, T_end=0.01,
            lam_start=50.0, lam_end=2000.0,
            seed=rng.integers(0, 2**31),
        )
        
        if not found or br > 3.5:
            log_queue.put(f"[w{worker_id}_{seed_name}_{round_num}] Phase1: R={br:.4f} {'(no feasible)' if not found else '(too loose)'}")
            continue
        
        log_queue.put(f"[w{worker_id}_{seed_name}_{round_num}] Phase1: R={br:.4f} → Phase2")
        
        # Phase 2: Cold squeeze (20M steps, repeated if improving)
        for squeeze in range(5):
            bx2, by2, bt2, br2, found2 = sa_gjk(
                bx, by, bt,
                n_steps=20_000_000,
                T_start=0.02, T_end=0.0001,
                lam_start=2000.0, lam_end=50000.0,
                seed=rng.integers(0, 2**31),
            )
            
            if found2 and br2 < br:
                log_queue.put(f"[w{worker_id}_{seed_name}_{round_num}] Phase2.{squeeze}: R={br2:.6f} (improved from {br:.6f})")
                bx, by, bt, br = bx2, by2, bt2, br2
                
                # Try to save
                if br < global_best.value:
                    # Official validate
                    result = official_validate(bx, by, bt)
                    if result.valid:
                        saved = save_if_better(bx, by, bt, result.score, global_best)
                        if saved:
                            log_queue.put(f"[w{worker_id}_{seed_name}_{round_num}] ★ NEW BEST: R={result.score:.6f}")
                    else:
                        log_queue.put(f"[w{worker_id}_{seed_name}_{round_num}] GJK valid but official invalid: {result.errors[:1]}")
            else:
                r_str = f"{br2:.6f}" if found2 else "inf"
                log_queue.put(f"[w{worker_id}_{seed_name}_{round_num}] Phase2.{squeeze}: R={r_str} (no improvement)")
                break


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--runtime', type=int, default=7200, help='seconds')
    args = parser.parse_args()
    
    global_best = mp.Value('d', load_best_score())
    log_queue = mp.Queue()
    
    print(f"Starting {args.workers} workers for {args.runtime}s")
    print(f"Current best: R={global_best.value:.6f}")
    print(f"Seed types: {[s[0] for s in SEED_GENERATORS]}")
    print()
    
    workers = []
    for i in range(args.workers):
        seed_name, seed_fn = SEED_GENERATORS[i % len(SEED_GENERATORS)]
        p = mp.Process(target=worker, args=(i, seed_name, seed_fn, global_best, args.runtime, log_queue))
        p.start()
        workers.append(p)
    
    start = time.time()
    last_report = start
    
    while any(p.is_alive() for p in workers):
        try:
            while not log_queue.empty():
                msg = log_queue.get_nowait()
                elapsed = int(time.time() - start)
                print(f"[{elapsed}s] {msg}")
        except:
            pass
        
        now = time.time()
        if now - last_report > 60:
            elapsed = int(now - start)
            alive = sum(1 for p in workers if p.is_alive())
            print(f"\n[{elapsed}s] === Best={global_best.value:.6f} Workers={alive}/{args.workers} ===\n")
            last_report = now
        
        time.sleep(1)
    
    # Drain remaining logs
    while not log_queue.empty():
        try:
            msg = log_queue.get_nowait()
            print(msg)
        except:
            break
    
    print(f"\nDone. Final best: R={global_best.value:.6f}")


if __name__ == '__main__':
    main()
