"""
Semicircle packing optimizer v6.
Fix: use proper Welzl MEC from the scoring module.
Fix: add compound moves (swap, pair-rotate, squeeze-toward-center).
"""

import json
import math
import time
import sys
import numpy as np
from shapely.geometry import Polygon

from semicircle_packing.geometry import Semicircle
from semicircle_packing.scoring import compute_mec, validate_and_score

RADIUS = 1.0
N = 15
ARC_PTS = 128
OVERLAP_TOL = 1e-4


def make_poly(x, y, theta):
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, ARC_PTS)
    return Polygon(list(zip(x + RADIUS * np.cos(angles), y + RADIUS * np.sin(angles))))


def official_mec(xs, ys, ts):
    """Use the official compute_mec which does Welzl + analytical refinement."""
    scs = [Semicircle(x=round(xs[i], 6), y=round(ys[i], 6), theta=round(ts[i], 6)) for i in range(N)]
    cx, cy, cr = compute_mec(scs)
    return cr, cx, cy


def check_overlap_one(xs, ys, ts, polys, idx):
    """Check if semicircle idx overlaps any other using cached polygons."""
    p = make_poly(xs[idx], ys[idx], ts[idx])
    for j in range(N):
        if j == idx:
            continue
        dx = xs[idx] - xs[j]
        dy = ys[idx] - ys[j]
        if dx*dx + dy*dy > 4.0:
            continue
        if p.intersection(polys[j]).area > OVERLAP_TOL:
            return True, p
    return False, p


def check_overlap_indices(xs, ys, ts, polys, indices):
    """Check if any of the given indices overlap with anything."""
    new_polys = {}
    for idx in indices:
        new_polys[idx] = make_poly(xs[idx], ys[idx], ts[idx])
    
    for idx in indices:
        for j in range(N):
            if j == idx:
                continue
            dx = xs[idx] - xs[j]
            dy = ys[idx] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue
            pj = new_polys.get(j, polys[j])
            if new_polys[idx].intersection(pj).area > OVERLAP_TOL:
                return True, new_polys
    return False, new_polys


def sa_optimize(xs, ys, ts, n_steps=300000, T_start=0.5, T_end=0.001, seed=42, label=""):
    """SA with compound moves and proper MEC."""
    rng = np.random.default_rng(seed)
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    
    polys = [make_poly(xs[i], ys[i], ts[i]) for i in range(N)]
    
    current_r, _, _ = official_mec(xs, ys, ts)
    best_r = current_r
    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
    
    n_improved = 0
    mec_calls = 0
    
    for step in range(n_steps):
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        scale = 0.15 * (T / T_start) + 0.001
        
        # Save state
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single semicircle move
            idx = rng.integers(0, N)
            old = (xs[idx], ys[idx], ts[idx])
            
            m = rng.random()
            if m < 0.3:
                xs[idx] += rng.normal(0, scale)
                ys[idx] += rng.normal(0, scale)
            elif m < 0.5:
                ts[idx] += rng.normal(0, scale * 3)
                ts[idx] %= 2 * math.pi
            elif m < 0.8:
                # Move toward center (squeeze)
                cx, cy = np.mean(xs), np.mean(ys)
                dx = cx - xs[idx]
                dy = cy - ys[idx]
                d = math.sqrt(dx*dx + dy*dy)
                if d > 0.01:
                    xs[idx] += scale * dx / d
                    ys[idx] += scale * dy / d
            else:
                xs[idx] += rng.normal(0, scale * 0.5)
                ys[idx] += rng.normal(0, scale * 0.5)
                ts[idx] += rng.normal(0, scale * 2)
                ts[idx] %= 2 * math.pi
            
            overlap, new_poly = check_overlap_one(xs, ys, ts, polys, idx)
            if overlap:
                xs[idx], ys[idx], ts[idx] = old
                continue
            
            new_r, _, _ = official_mec(xs, ys, ts)
            mec_calls += 1
            delta = new_r - current_r
            
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
                polys[idx] = new_poly
                current_r = new_r
                if new_r < best_r:
                    best_r = new_r
                    best_xs[:] = xs; best_ys[:] = ys; best_ts[:] = ts
                    n_improved += 1
            else:
                xs[idx], ys[idx], ts[idx] = old
        
        elif move_type < 0.75:
            # Swap two semicircles
            i, j = rng.choice(N, 2, replace=False)
            xs[i], xs[j] = xs[j], xs[i]
            ys[i], ys[j] = ys[j], ys[i]
            ts[i], ts[j] = ts[j], ts[i]
            
            overlap, new_polys = check_overlap_indices(xs, ys, ts, polys, [i, j])
            if overlap:
                xs[i], xs[j] = xs[j], xs[i]
                ys[i], ys[j] = ys[j], ys[i]
                ts[i], ts[j] = ts[j], ts[i]
                continue
            
            new_r, _, _ = official_mec(xs, ys, ts)
            mec_calls += 1
            delta = new_r - current_r
            
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
                for k, p in new_polys.items():
                    polys[k] = p
                current_r = new_r
                if new_r < best_r:
                    best_r = new_r
                    best_xs[:] = xs; best_ys[:] = ys; best_ts[:] = ts
                    n_improved += 1
            else:
                xs[i], xs[j] = xs[j], xs[i]
                ys[i], ys[j] = ys[j], ys[i]
                ts[i], ts[j] = ts[j], ts[i]
        
        else:
            # Rotate entire configuration
            angle = rng.normal(0, scale * 0.5)
            old_xs, old_ys = xs.copy(), ys.copy()
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for i in range(N):
                nx = xs[i] * cos_a - ys[i] * sin_a
                ny = xs[i] * sin_a + ys[i] * cos_a
                xs[i], ys[i] = nx, ny
                ts[i] = (ts[i] + angle) % (2 * math.pi)
            
            # No overlap change from rigid rotation, but MEC might change
            new_polys_all = [make_poly(xs[i], ys[i], ts[i]) for i in range(N)]
            new_r, _, _ = official_mec(xs, ys, ts)
            mec_calls += 1
            delta = new_r - current_r
            
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
                polys = new_polys_all
                current_r = new_r
                if new_r < best_r:
                    best_r = new_r
                    best_xs[:] = xs; best_ys[:] = ys; best_ts[:] = ts
                    n_improved += 1
            else:
                xs[:] = old_xs; ys[:] = old_ys
                ts[:] = best_ts  # revert thetas too
                # Actually need to revert properly...
                for i in range(N):
                    ts[i] = (ts[i] - angle) % (2 * math.pi)
        
        if step % 25000 == 0:
            print(f"    [{label}] step {step:>7d}: r={current_r:.6f} best={best_r:.6f} T={T:.4f} imp={n_improved}", flush=True)
    
    return best_xs, best_ys, best_ts, best_r


def config_grid():
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    idx = 0
    for row in range(2):
        for col in range(3):
            cx, cy = (col-1)*2.0, row*2.0
            xs[idx], ys[idx], ts[idx] = cx, cy, 0; idx += 1
            xs[idx], ys[idx], ts[idx] = cx, cy, math.pi; idx += 1
    for col in range(3):
        xs[idx], ys[idx], ts[idx] = (col-1)*2.0, 3.0, math.pi/2; idx += 1
    return xs, ys, ts


def main():
    print("=" * 60)
    print("Semicircle Packing Optimizer v6")
    print("=" * 60, flush=True)
    
    xs, ys, ts = config_grid()
    r0, _, _ = official_mec(xs, ys, ts)
    print(f"Grid baseline MEC: {r0:.6f}", flush=True)
    
    # Benchmark MEC speed
    t0 = time.time()
    for _ in range(100):
        official_mec(xs, ys, ts)
    elapsed = time.time() - t0
    print(f"MEC speed: {100/elapsed:.0f}/sec", flush=True)
    
    # Run SA
    print("\nRunning SA from grid (300k steps)...", flush=True)
    t0 = time.time()
    bx, by, bt, br = sa_optimize(xs, ys, ts, n_steps=300000, T_start=0.5, T_end=0.001, seed=42, label="grid")
    elapsed = time.time() - t0
    
    scs = [Semicircle(x=round(bx[i], 6), y=round(by[i], 6), theta=round(bt[i], 6)) for i in range(N)]
    val = validate_and_score(scs)
    
    if val.valid:
        print(f"\nVALID: score={val.score:.6f} ({elapsed:.1f}s)", flush=True)
    else:
        print(f"\nINVALID: errors={len(val.errors)} ({elapsed:.1f}s)", flush=True)
    
    # Save
    solution = [{"x": round(float(bx[i]), 6), "y": round(float(by[i]), 6), "theta": round(float(bt[i]), 6)} for i in range(N)]
    with open("solution.json", "w") as f:
        json.dump(solution, f, indent=2)
    print("Saved to solution.json", flush=True)


if __name__ == "__main__":
    main()
