"""
Semicircle packing optimizer v3.
Custom SA with fast analytical penalty.
Key insight: move ONE semicircle at a time, not all 45 dims.
"""

import json
import math
import time
import sys
import numpy as np

RADIUS = 1.0
N = 15


def overlap_penalty(xs, ys, ts):
    """Fast pairwise overlap penalty."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            if dist_sq >= 4.0:
                continue
            dist = math.sqrt(dist_sq) if dist_sq > 1e-24 else 1e-12
            
            # Direction from i to j and j to i
            dir_i2j = math.atan2(-dy, -dx)
            dir_j2i = math.atan2(dy, dx)
            
            # Angular difference from each semicircle's facing direction
            diff_i = abs(math.atan2(math.sin(dir_i2j - ts[i]), math.cos(dir_i2j - ts[i])))
            diff_j = abs(math.atan2(math.sin(dir_j2i - ts[j]), math.cos(dir_j2i - ts[j])))
            
            # Facing factors
            face_i = max(0.0, 1.0 - diff_i / (math.pi * 0.5))
            face_j = max(0.0, 1.0 - diff_j / (math.pi * 0.5))
            
            penetration = max(0.0, 2.0 - dist)
            penalty += penetration * penetration * face_i * face_j
    return penalty


def mec_radius(xs, ys, ts):
    """MEC radius from key boundary points (3 per semicircle)."""
    pts_x = []
    pts_y = []
    for i in range(N):
        x, y, t = xs[i], ys[i], ts[i]
        # Arc tip
        pts_x.append(x + math.cos(t))
        pts_y.append(y + math.sin(t))
        # Flat edge endpoints
        pts_x.append(x + math.cos(t + math.pi/2))
        pts_y.append(y + math.sin(t + math.pi/2))
        pts_x.append(x + math.cos(t - math.pi/2))
        pts_y.append(y + math.sin(t - math.pi/2))
    
    pts_x = np.array(pts_x)
    pts_y = np.array(pts_y)
    
    # 1-center iterative
    cx = np.mean(pts_x)
    cy = np.mean(pts_y)
    for it in range(200):
        dists = np.sqrt((pts_x - cx)**2 + (pts_y - cy)**2)
        idx = np.argmax(dists)
        step = 0.3 / (1 + it/5)
        cx += step * (pts_x[idx] - cx)
        cy += step * (pts_y[idx] - cy)
    
    dists = np.sqrt((pts_x - cx)**2 + (pts_y - cy)**2)
    return float(np.max(dists))


def objective(xs, ys, ts, lam=100.0):
    return mec_radius(xs, ys, ts) + lam * overlap_penalty(xs, ys, ts)


def validate_official(xs, ys, ts):
    """Run official scorer."""
    solution = []
    for i in range(N):
        solution.append({
            "x": round(xs[i], 6),
            "y": round(ys[i], 6),
            "theta": round(ts[i], 6),
        })
    
    with open("_tmp_sol.json", "w") as f:
        json.dump(solution, f)
    
    from semicircle_packing.scoring import validate_and_score
    from semicircle_packing.geometry import Semicircle
    
    scs = [Semicircle(x=s["x"], y=s["y"], theta=s["theta"]) for s in solution]
    return validate_and_score(scs)


# ── Starting configurations ──────────────────────────────────────

def make_config(desc):
    """Generate various starting configs."""
    xs = np.zeros(N)
    ys = np.zeros(N)
    ts = np.zeros(N)
    
    if desc == "hex7":
        # 7 disks hex + 1 loose
        idx = 0
        centers = [(0, 0)]
        for k in range(6):
            a = k * math.pi/3
            centers.append((2*math.cos(a), 2*math.sin(a)))
        for cx, cy in centers:
            xs[idx], ys[idx], ts[idx] = cx, cy, 0
            idx += 1
            xs[idx], ys[idx], ts[idx] = cx, cy, math.pi
            idx += 1
        xs[idx], ys[idx], ts[idx] = 0, 0, math.pi/2
        
    elif desc == "grid":
        # 3x2 grid of pairs + 3 on top (the baseline)
        idx = 0
        for row in range(2):
            for col in range(3):
                cx = (col - 1) * 2
                cy = row * 2
                xs[idx], ys[idx], ts[idx] = cx, cy, 0
                idx += 1
                xs[idx], ys[idx], ts[idx] = cx, cy, math.pi
                idx += 1
        for col in range(3):
            cx = (col - 1) * 2
            xs[idx], ys[idx], ts[idx] = cx, 3, math.pi/2
            idx += 1
            
    elif desc == "concentric":
        # Inner ring of 5 + outer ring of 10
        idx = 0
        for k in range(5):
            a = 2*math.pi*k/5
            xs[idx], ys[idx], ts[idx] = 0.8*math.cos(a), 0.8*math.sin(a), a+math.pi
            idx += 1
        for k in range(10):
            a = 2*math.pi*k/10
            xs[idx], ys[idx], ts[idx] = 2.2*math.cos(a), 2.2*math.sin(a), a
            idx += 1
    
    elif desc == "flower":
        # Center pair + 6 petals (each a pair facing outward) + 1 cap
        idx = 0
        xs[idx], ys[idx], ts[idx] = 0, 0, 0; idx += 1
        xs[idx], ys[idx], ts[idx] = 0, 0, math.pi; idx += 1
        for k in range(6):
            a = k * math.pi/3
            r = 2.0
            xs[idx], ys[idx], ts[idx] = r*math.cos(a), r*math.sin(a), a
            idx += 1
        # 1 spare
        xs[idx], ys[idx], ts[idx] = 0, 2.0, math.pi/2; idx += 1
        # Fill remaining with random nearby
        rng = np.random.default_rng(42)
        while idx < N:
            xs[idx] = rng.uniform(-1, 1)
            ys[idx] = rng.uniform(-1, 1)
            ts[idx] = rng.uniform(0, 2*math.pi)
            idx += 1
    
    elif desc.startswith("random"):
        seed = int(desc.split("_")[1]) if "_" in desc else 42
        rng = np.random.default_rng(seed)
        for i in range(N):
            xs[i] = rng.uniform(-2, 2)
            ys[i] = rng.uniform(-2, 2)
            ts[i] = rng.uniform(0, 2*math.pi)
    
    return xs, ys, ts


# ── Simulated Annealing ──────────────────────────────────────────

def sa_optimize(xs, ys, ts, n_steps=500000, T_start=1.0, T_end=0.001, lam=100.0, seed=42):
    """
    SA that moves one semicircle at a time.
    Much faster per step than DE over 45 dims.
    """
    rng = np.random.default_rng(seed)
    
    xs = xs.copy()
    ys = ys.copy()
    ts = ts.copy()
    
    current_obj = objective(xs, ys, ts, lam)
    best_obj = current_obj
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_ts = ts.copy()
    
    n_accepted = 0
    n_improved = 0
    
    for step in range(n_steps):
        # Temperature schedule (geometric)
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        
        # Pick a random semicircle
        idx = rng.integers(0, N)
        
        # Save old values
        old_x, old_y, old_t = xs[idx], ys[idx], ts[idx]
        
        # Step size scales with temperature
        scale = 0.5 * T / T_start + 0.01
        
        # Random move type
        move = rng.random()
        if move < 0.4:
            # Translate
            xs[idx] += rng.normal(0, scale)
            ys[idx] += rng.normal(0, scale)
        elif move < 0.7:
            # Rotate
            ts[idx] += rng.normal(0, scale * math.pi)
            ts[idx] = ts[idx] % (2 * math.pi)
        else:
            # Both
            xs[idx] += rng.normal(0, scale * 0.5)
            ys[idx] += rng.normal(0, scale * 0.5)
            ts[idx] += rng.normal(0, scale * math.pi * 0.5)
            ts[idx] = ts[idx] % (2 * math.pi)
        
        new_obj = objective(xs, ys, ts, lam)
        delta = new_obj - current_obj
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
            current_obj = new_obj
            n_accepted += 1
            if new_obj < best_obj:
                best_obj = new_obj
                best_xs[:] = xs
                best_ys[:] = ys
                best_ts[:] = ts
                n_improved += 1
        else:
            # Revert
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
        
        if step % 100000 == 0 and step > 0:
            accept_rate = n_accepted / step
            print(f"    step {step:>7d}/{n_steps}: obj={current_obj:.6f} best={best_obj:.6f} T={T:.6f} accept={accept_rate:.2%}")
    
    return best_xs, best_ys, best_ts, best_obj


def main():
    configs = ["grid", "hex7", "concentric", "flower"]
    configs += [f"random_{s}" for s in [42, 99, 777, 1234, 9999, 31415]]
    
    global_best_score = float('inf')
    global_best = None
    
    print("=" * 60)
    print("Semicircle Packing Optimizer v3 (SA)")
    print("=" * 60)
    
    # Phase 1: Quick SA from each config (100k steps ~15s each)
    print("\nPhase 1: Quick screen (100k steps each)")
    
    for name in configs:
        xs, ys, ts = make_config(name)
        t0 = time.time()
        print(f"\n  [{name}]")
        
        bx, by, bt, obj = sa_optimize(xs, ys, ts, n_steps=100000, T_start=2.0, T_end=0.001, lam=100.0, seed=42)
        
        # Validate
        val = validate_official(bx, by, bt)
        elapsed = time.time() - t0
        
        if val.valid:
            print(f"  [{name}] VALID score={val.score:.6f} obj={obj:.6f} ({elapsed:.1f}s)")
            if val.score < global_best_score:
                global_best_score = val.score
                global_best = (bx.copy(), by.copy(), bt.copy())
        else:
            mec = mec_radius(bx, by, bt)
            ovlp = overlap_penalty(bx, by, bt)
            n_err = len(val.errors)
            print(f"  [{name}] INVALID mec={mec:.4f} ovlp={ovlp:.6f} errors={n_err} ({elapsed:.1f}s)")
    
    print(f"\n{'=' * 60}")
    print(f"Phase 1 best: {global_best_score:.6f}")
    
    # Phase 2: Long SA from best + re-attempts with higher penalty
    if global_best is not None:
        print(f"\nPhase 2: Extended SA from best (2M steps)")
        bx, by, bt = global_best
        
        for attempt, (lam, seed) in enumerate([(200, 123), (500, 456), (1000, 789)]):
            print(f"\n  Attempt {attempt+1}: lam={lam}")
            t0 = time.time()
            bx2, by2, bt2, obj = sa_optimize(bx, by, bt, n_steps=2000000, T_start=0.5, T_end=0.0001, lam=lam, seed=seed)
            
            val = validate_official(bx2, by2, bt2)
            elapsed = time.time() - t0
            
            if val.valid:
                print(f"  VALID score={val.score:.6f} ({elapsed:.1f}s)")
                if val.score < global_best_score:
                    global_best_score = val.score
                    global_best = (bx2.copy(), by2.copy(), bt2.copy())
            else:
                n_err = len(val.errors)
                print(f"  INVALID errors={n_err} ({elapsed:.1f}s)")
    else:
        # No valid solution found: try with very high penalty
        print("\nNo valid solutions in Phase 1. Phase 2: high penalty from random starts")
        for seed in [42, 99, 777]:
            xs, ys, ts = make_config(f"random_{seed}")
            bx, by, bt, obj = sa_optimize(xs, ys, ts, n_steps=2000000, T_start=3.0, T_end=0.0001, lam=1000.0, seed=seed)
            val = validate_official(bx, by, bt)
            if val.valid and val.score < global_best_score:
                global_best_score = val.score
                global_best = (bx.copy(), by.copy(), bt.copy())
                print(f"  VALID score={val.score:.6f}")
    
    # Save
    print(f"\n{'=' * 60}")
    print(f"FINAL BEST: {global_best_score:.6f}")
    print("=" * 60)
    
    if global_best is not None:
        bx, by, bt = global_best
        solution = []
        for i in range(N):
            solution.append({
                "x": round(float(bx[i]), 6),
                "y": round(float(by[i]), 6),
                "theta": round(float(bt[i]), 6),
            })
        for path in ["solution.json", "best_solution.json"]:
            with open(path, "w") as f:
                json.dump(solution, f, indent=2)
        print(f"Saved to solution.json and best_solution.json")


if __name__ == "__main__":
    main()
