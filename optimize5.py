"""
Semicircle packing optimizer v5.
Fast conservative overlap check + SA.
Key: use reduced arc points (128 instead of 4096) for speed,
with a tighter tolerance to compensate.
"""

import json
import math
import time
import numpy as np
from shapely.geometry import Polygon

RADIUS = 1.0
N = 15
ARC_POINTS = 128  # Much faster than 4096, still accurate to ~6e-4
OVERLAP_TOL = 1e-4  # Stricter than needed to compensate for lower resolution


def make_polygon(x, y, theta, n_arc=ARC_POINTS):
    """Build Shapely polygon for semicircle."""
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, n_arc)
    arc_x = x + RADIUS * np.cos(angles)
    arc_y = y + RADIUS * np.sin(angles)
    coords = list(zip(arc_x, arc_y))
    return Polygon(coords)


def check_overlap_pair(x1, y1, t1, x2, y2, t2):
    """Check if two semicircles overlap. Fast with distance pre-check."""
    dx = x1 - x2
    dy = y1 - y2
    if dx*dx + dy*dy > 4.0:
        return False
    p1 = make_polygon(x1, y1, t1)
    p2 = make_polygon(x2, y2, t2)
    return p1.intersection(p2).area > OVERLAP_TOL


def check_index_overlaps(xs, ys, ts, idx, polys=None):
    """Check if semicircle[idx] overlaps any other."""
    if polys is not None:
        p = make_polygon(xs[idx], ys[idx], ts[idx])
        for j in range(N):
            if j == idx:
                continue
            dx = xs[idx] - xs[j]
            dy = ys[idx] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue
            if p.intersection(polys[j]).area > OVERLAP_TOL:
                return True
        return False
    else:
        for j in range(N):
            if j == idx:
                continue
            if check_overlap_pair(xs[idx], ys[idx], ts[idx], xs[j], ys[j], ts[j]):
                return True
        return False


def fast_mec_radius(xs, ys, ts):
    """MEC from key boundary points."""
    pts = np.zeros((N * 3, 2))
    for i in range(N):
        pts[i*3, 0] = xs[i] + math.cos(ts[i])
        pts[i*3, 1] = ys[i] + math.sin(ts[i])
        pts[i*3+1, 0] = xs[i] + math.cos(ts[i] + math.pi/2)
        pts[i*3+1, 1] = ys[i] + math.sin(ts[i] + math.pi/2)
        pts[i*3+2, 0] = xs[i] + math.cos(ts[i] - math.pi/2)
        pts[i*3+2, 1] = ys[i] + math.sin(ts[i] - math.pi/2)
    
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    for it in range(200):
        dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        idx = np.argmax(dists)
        step = 0.3 / (1 + it/5)
        cx += step * (pts[idx, 0] - cx)
        cy += step * (pts[idx, 1] - cy)
    
    return float(np.max(np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)))


def sa_optimize(xs, ys, ts, n_steps=500000, T_start=1.0, T_end=0.001, seed=42):
    """SA: move one semicircle at a time, only accept feasible moves."""
    rng = np.random.default_rng(seed)
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    
    # Pre-build polygons for all semicircles
    polys = [make_polygon(xs[i], ys[i], ts[i]) for i in range(N)]
    
    current_r = fast_mec_radius(xs, ys, ts)
    best_r = current_r
    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
    
    n_accepted = 0
    n_improved = 0
    
    for step in range(n_steps):
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        
        idx = rng.integers(0, N)
        old_x, old_y, old_t = xs[idx], ys[idx], ts[idx]
        old_poly = polys[idx]
        
        scale = 0.2 * (T / T_start) + 0.002
        
        move = rng.random()
        if move < 0.4:
            xs[idx] += rng.normal(0, scale)
            ys[idx] += rng.normal(0, scale)
        elif move < 0.7:
            ts[idx] += rng.normal(0, scale * 2)
            ts[idx] = ts[idx] % (2 * math.pi)
        else:
            xs[idx] += rng.normal(0, scale * 0.5)
            ys[idx] += rng.normal(0, scale * 0.5)
            ts[idx] += rng.normal(0, scale)
            ts[idx] = ts[idx] % (2 * math.pi)
        
        # Build new polygon and check overlaps
        new_poly = make_polygon(xs[idx], ys[idx], ts[idx])
        
        overlap_found = False
        for j in range(N):
            if j == idx:
                continue
            dx = xs[idx] - xs[j]
            dy = ys[idx] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue
            if new_poly.intersection(polys[j]).area > OVERLAP_TOL:
                overlap_found = True
                break
        
        if overlap_found:
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
            continue
        
        new_r = fast_mec_radius(xs, ys, ts)
        delta = new_r - current_r
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
            polys[idx] = new_poly
            current_r = new_r
            n_accepted += 1
            if new_r < best_r:
                best_r = new_r
                best_xs[:] = xs
                best_ys[:] = ys
                best_ts[:] = ts
                n_improved += 1
        else:
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
        
        if step % 50000 == 0:
            feas_rate = n_accepted / max(step, 1)
            print(f"    step {step:>7d}: r={current_r:.6f} best={best_r:.6f} T={T:.4f} improved={n_improved}")
            sys.stdout.flush()
    
    return best_xs, best_ys, best_ts, best_r


import sys

# ── Starting configs (all must be overlap-free) ──────────────────

def config_grid():
    """3x2 paired grid + 3 up on top. Score 3.50."""
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


def config_grid_down():
    """Same but 3 extras point down."""
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    idx = 0
    for row in range(2):
        for col in range(3):
            cx, cy = (col-1)*2.0, row*2.0
            xs[idx], ys[idx], ts[idx] = cx, cy, 0; idx += 1
            xs[idx], ys[idx], ts[idx] = cx, cy, math.pi; idx += 1
    for col in range(3):
        xs[idx], ys[idx], ts[idx] = (col-1)*2.0, -1.0, -math.pi/2; idx += 1
    return xs, ys, ts


def config_2x4_loose():
    """2 rows of 4 pairs (16 > 15, so 7 pairs + 1 loose)."""
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    idx = 0
    # 7 pairs = 14 semicircles
    positions = [(-3, 0), (-1, 0), (1, 0), (3, 0), (-2, 2), (0, 2), (2, 2)]
    for cx, cy in positions:
        xs[idx], ys[idx], ts[idx] = cx, cy, 0; idx += 1
        xs[idx], ys[idx], ts[idx] = cx, cy, math.pi; idx += 1
    # 1 loose
    xs[idx], ys[idx], ts[idx] = 0, 4, math.pi/2
    return xs, ys, ts


def verify_no_overlap(xs, ys, ts, name):
    """Check starting config is valid."""
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue
            if check_overlap_pair(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j]):
                print(f"  WARNING: [{name}] overlap between {i} and {j}")
                return False
    return True


def main():
    print("=" * 60)
    print("Semicircle Packing Optimizer v5 (Feasible SA, 128-pt)")
    print("=" * 60)
    
    # Benchmark
    xs, ys, ts = config_grid()
    p1 = make_polygon(xs[0], ys[0], ts[0])
    p2 = make_polygon(xs[1], ys[1], ts[1])
    t0 = time.time()
    for _ in range(1000):
        p1.intersection(p2).area
    elapsed = time.time() - t0
    print(f"\n128-pt intersection: {1000/elapsed:.0f}/sec")
    
    # Per-step estimate: ~14 checks per step (avg nearby pairs)
    # But most pairs are far apart, so maybe 2-4 Shapely calls per step
    
    configs = {
        "grid": config_grid(),
        "grid_down": config_grid_down(),
    }
    
    best_score = float('inf')
    best_result = None
    
    print("\nPhase 1: SA from each config (200k steps)")
    sys.stdout.flush()
    
    for name, (xs, ys, ts) in configs.items():
        if not verify_no_overlap(xs, ys, ts, name):
            continue
        
        r0 = fast_mec_radius(xs, ys, ts)
        print(f"\n  [{name}] starting r={r0:.6f}")
        sys.stdout.flush()
        
        t0 = time.time()
        bx, by, bt, r = sa_optimize(xs, ys, ts, n_steps=200000, T_start=1.0, T_end=0.001, seed=42)
        elapsed = time.time() - t0
        
        # Official validate
        from semicircle_packing.scoring import validate_and_score
        from semicircle_packing.geometry import Semicircle
        scs = [Semicircle(x=round(bx[i], 6), y=round(by[i], 6), theta=round(bt[i], 6)) for i in range(N)]
        val = validate_and_score(scs)
        
        if val.valid:
            print(f"  [{name}] VALID score={val.score:.6f} ({elapsed:.1f}s, {200000/elapsed:.0f} steps/s)")
            if val.score < best_score:
                best_score = val.score
                best_result = (bx.copy(), by.copy(), bt.copy())
        else:
            print(f"  [{name}] INVALID r_fast={r:.6f} errors={len(val.errors)} ({elapsed:.1f}s)")
        sys.stdout.flush()
    
    # Phase 2
    if best_result is not None:
        print(f"\nPhase 1 best: {best_score:.6f}")
        print(f"\nPhase 2: Extended SA (1M steps) from best")
        sys.stdout.flush()
        
        bx, by, bt = best_result
        t0 = time.time()
        bx2, by2, bt2, r = sa_optimize(bx, by, bt, n_steps=1000000, T_start=0.2, T_end=0.00005, seed=123)
        elapsed = time.time() - t0
        
        scs = [Semicircle(x=round(bx2[i], 6), y=round(by2[i], 6), theta=round(bt2[i], 6)) for i in range(N)]
        val = validate_and_score(scs)
        
        if val.valid:
            print(f"  Phase 2 VALID score={val.score:.6f} ({elapsed:.1f}s)")
            if val.score < best_score:
                best_score = val.score
                best_result = (bx2.copy(), by2.copy(), bt2.copy())
        else:
            print(f"  Phase 2 INVALID errors={len(val.errors)} ({elapsed:.1f}s)")
        sys.stdout.flush()
    
    # Save
    print(f"\n{'=' * 60}")
    print(f"FINAL BEST: {best_score:.6f}")
    print("=" * 60)
    
    if best_result is not None:
        bx, by, bt = best_result
        solution = [{"x": round(float(bx[i]), 6), "y": round(float(by[i]), 6), "theta": round(float(bt[i]), 6)} for i in range(N)]
        for path in ["solution.json", "best_solution.json"]:
            with open(path, "w") as f:
                json.dump(solution, f, indent=2)
        print("Saved")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
