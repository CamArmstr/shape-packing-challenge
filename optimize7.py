"""
Semicircle packing optimizer v7.
Strategy change: penalty-based SA with official overlap area.
Start from SPREAD OUT config (not tight grid).
Anneal penalty weight upward to push toward feasibility.

Also: use fast MEC (3 points per sc) for SA inner loop,
validate with official MEC only at checkpoints.
"""

import json
import math
import time
import sys
import numpy as np
from shapely.geometry import Polygon

RADIUS = 1.0
N = 15
ARC_PTS = 64  # Fast enough for penalty estimation


def make_poly(x, y, theta):
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, ARC_PTS)
    return Polygon(list(zip(x + RADIUS * np.cos(angles), y + RADIUS * np.sin(angles))))


def total_overlap_area(xs, ys, ts, polys=None):
    """Sum of pairwise intersection areas."""
    if polys is None:
        polys = [make_poly(xs[i], ys[i], ts[i]) for i in range(N)]
    
    total = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue
            area = polys[i].intersection(polys[j]).area
            if area > 1e-6:
                total += area
    return total


def overlap_change_for_index(xs, ys, ts, polys, idx, new_poly):
    """Compute change in overlap area when semicircle[idx] changes."""
    old_ovlp = 0.0
    new_ovlp = 0.0
    
    for j in range(N):
        if j == idx:
            continue
        dx_old = xs[idx] - xs[j]  # xs still has old value conceptually... no, it's updated
        dy_old = ys[idx] - ys[j]
        # We need to check both old and new overlaps
        # Actually: just compute new overlap sum for this index
        if dx_old*dx_old + dy_old*dy_old <= 4.0:
            area = new_poly.intersection(polys[j]).area
            if area > 1e-8:
                new_ovlp += area
    
    return new_ovlp


def fast_mec(xs, ys, ts):
    """Fast approximate MEC using 3 key boundary points per semicircle + Welzl."""
    pts = np.zeros((N * 3, 2))
    for i in range(N):
        x, y, t = xs[i], ys[i], ts[i]
        pts[i*3] = [x + math.cos(t), y + math.sin(t)]
        pts[i*3+1] = [x + math.cos(t + math.pi/2), y + math.sin(t + math.pi/2)]
        pts[i*3+2] = [x + math.cos(t - math.pi/2), y + math.sin(t - math.pi/2)]
    
    # Simple minimax center
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    for _ in range(500):
        dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        idx = np.argmax(dists)
        r = dists[idx]
        # Subgradient step
        fx, fy = pts[idx]
        d = r
        if d > 0:
            cx += 0.01 * (fx - cx)
            cy += 0.01 * (fy - cy)
    
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    return float(np.max(dists))


def sa_penalty(xs, ys, ts, n_steps=500000, T_start=1.0, T_end=0.001, 
               lam_start=10.0, lam_end=1000.0, seed=42, label=""):
    """SA with penalty for overlaps. Lambda increases over time."""
    rng = np.random.default_rng(seed)
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    
    polys = [make_poly(xs[i], ys[i], ts[i]) for i in range(N)]
    
    current_ovlp = total_overlap_area(xs, ys, ts, polys)
    current_r = fast_mec(xs, ys, ts)
    lam = lam_start
    current_obj = current_r + lam * current_ovlp
    
    best_feasible_r = float('inf')
    best_xs, best_ys, best_ts = None, None, None
    best_obj = current_obj
    best_obj_xs = xs.copy()
    best_obj_ys = ys.copy()
    best_obj_ts = ts.copy()
    
    n_improved = 0
    
    for step in range(n_steps):
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        lam = lam_start * (lam_end / lam_start) ** frac
        
        scale = 0.3 * (1.0 - 0.9 * frac) + 0.003
        
        idx = rng.integers(0, N)
        old_x, old_y, old_t = xs[idx], ys[idx], ts[idx]
        old_poly = polys[idx]
        
        m = rng.random()
        if m < 0.35:
            xs[idx] += rng.normal(0, scale)
            ys[idx] += rng.normal(0, scale)
        elif m < 0.55:
            ts[idx] += rng.normal(0, scale * 3)
            ts[idx] %= 2 * math.pi
        elif m < 0.85:
            # Squeeze toward centroid
            cx = np.mean(xs)
            cy = np.mean(ys)
            dx = cx - xs[idx]
            dy = cy - ys[idx]
            d = math.sqrt(dx*dx + dy*dy)
            if d > 0.01:
                xs[idx] += scale * 0.5 * dx / d
                ys[idx] += scale * 0.5 * dy / d
            ts[idx] += rng.normal(0, scale)
            ts[idx] %= 2 * math.pi
        else:
            xs[idx] += rng.normal(0, scale * 0.5)
            ys[idx] += rng.normal(0, scale * 0.5)
            ts[idx] += rng.normal(0, scale * 2)
            ts[idx] %= 2 * math.pi
        
        new_poly = make_poly(xs[idx], ys[idx], ts[idx])
        
        # Compute new overlap for this index only (incremental)
        new_idx_ovlp = overlap_change_for_index(xs, ys, ts, polys, idx, new_poly)
        old_idx_ovlp = 0.0
        for j in range(N):
            if j == idx:
                continue
            dx = old_x - xs[j]
            dy = old_y - ys[j]
            if dx*dx + dy*dy <= 4.0:
                area = old_poly.intersection(polys[j]).area
                if area > 1e-8:
                    old_idx_ovlp += area
        
        new_ovlp = current_ovlp - old_idx_ovlp + new_idx_ovlp
        new_r = fast_mec(xs, ys, ts)
        new_obj = new_r + lam * new_ovlp
        
        delta = new_obj - current_obj
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
            polys[idx] = new_poly
            current_ovlp = new_ovlp
            current_r = new_r
            current_obj = new_obj
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_obj_xs[:] = xs
                best_obj_ys[:] = ys
                best_obj_ts[:] = ts
            
            # Track best feasible
            if new_ovlp < 1e-5 and new_r < best_feasible_r:
                best_feasible_r = new_r
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_ts = ts.copy()
                n_improved += 1
        else:
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
        
        if step % 50000 == 0:
            print(f"    [{label}] step {step:>7d}: r={current_r:.4f} ovlp={current_ovlp:.6f} lam={lam:.0f} obj={current_obj:.4f} best_feas={best_feasible_r:.6f} imp={n_improved}", flush=True)
    
    if best_xs is None:
        # No feasible solution found, return best obj
        return best_obj_xs, best_obj_ys, best_obj_ts, best_feasible_r
    return best_xs, best_ys, best_ts, best_feasible_r


# ── Configs (spread out, easy to improve) ─────────────────────

def config_spread_ring():
    """15 semicircles in a wide ring. Easy to compress."""
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    for i in range(N):
        a = 2 * math.pi * i / N
        r = 3.5
        xs[i] = r * math.cos(a)
        ys[i] = r * math.sin(a)
        ts[i] = a + math.pi  # pointing inward
    return xs, ys, ts


def config_spread_grid():
    """5x3 grid of semicircles, spread out."""
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    idx = 0
    for row in range(3):
        for col in range(5):
            if idx >= N:
                break
            xs[idx] = (col - 2) * 2.5
            ys[idx] = (row - 1) * 2.5
            ts[idx] = 0  # all pointing right
            idx += 1
    return xs, ys, ts


def config_jittered_pairs():
    """7 pairs + 1 loose, spread out with jitter."""
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    rng = np.random.default_rng(42)
    idx = 0
    positions = [(0, 0), (3, 0), (-3, 0), (0, 3), (0, -3), (2.1, 2.1), (-2.1, 2.1)]
    for cx, cy in positions:
        xs[idx] = cx + rng.normal(0, 0.1)
        ys[idx] = cy + rng.normal(0, 0.1)
        ts[idx] = 0; idx += 1
        xs[idx] = cx + rng.normal(0, 0.1)
        ys[idx] = cy + rng.normal(0, 0.1)
        ts[idx] = math.pi; idx += 1
    xs[idx] = -2.1 + rng.normal(0, 0.1)
    ys[idx] = -2.1 + rng.normal(0, 0.1)
    ts[idx] = math.pi/4
    return xs, ys, ts


def config_random(seed=42):
    rng = np.random.default_rng(seed)
    r = 3.0
    xs = rng.uniform(-r, r, N)
    ys = rng.uniform(-r, r, N)
    ts = rng.uniform(0, 2*math.pi, N)
    return xs, ys, ts


def main():
    print("=" * 60)
    print("Semicircle Packing Optimizer v7 (Penalty SA)")
    print("=" * 60, flush=True)
    
    # Benchmark
    xs, ys, ts = config_spread_ring()
    polys = [make_poly(xs[i], ys[i], ts[i]) for i in range(N)]
    t0 = time.time()
    for _ in range(100):
        total_overlap_area(xs, ys, ts, polys)
    elapsed = time.time() - t0
    print(f"Overlap area (full): {100/elapsed:.0f}/sec", flush=True)
    
    t0 = time.time()
    for _ in range(1000):
        fast_mec(xs, ys, ts)
    elapsed = time.time() - t0
    print(f"Fast MEC: {1000/elapsed:.0f}/sec", flush=True)
    
    configs = {
        "ring": config_spread_ring(),
        "sgrid": config_spread_grid(),
        "pairs": config_jittered_pairs(),
        "rand42": config_random(42),
        "rand99": config_random(99),
        "rand777": config_random(777),
    }
    
    best_score = float('inf')
    best_result = None
    
    print(f"\nPhase 1: Penalty SA from spread configs (500k steps)", flush=True)
    
    for name, (xs, ys, ts) in configs.items():
        print(f"\n  Config: {name}", flush=True)
        t0 = time.time()
        bx, by, bt, br = sa_penalty(xs, ys, ts, n_steps=500000, 
                                     T_start=2.0, T_end=0.005,
                                     lam_start=5.0, lam_end=500.0,
                                     seed=42, label=name)
        elapsed = time.time() - t0
        
        # Official validate
        from semicircle_packing.geometry import Semicircle
        from semicircle_packing.scoring import validate_and_score
        scs = [Semicircle(x=round(bx[i], 6), y=round(by[i], 6), theta=round(bt[i], 6)) for i in range(N)]
        val = validate_and_score(scs)
        
        if val.valid:
            print(f"  [{name}] VALID score={val.score:.6f} ({elapsed:.0f}s)", flush=True)
            if val.score < best_score:
                best_score = val.score
                best_result = (bx.copy(), by.copy(), bt.copy())
        else:
            print(f"  [{name}] INVALID r_fast={br:.4f} errors={len(val.errors)} ({elapsed:.0f}s)", flush=True)
    
    # Save
    print(f"\n{'=' * 60}")
    print(f"FINAL BEST: {best_score:.6f}")
    print("=" * 60, flush=True)
    
    if best_result is not None:
        bx, by, bt = best_result
        solution = [{"x": round(float(bx[i]), 6), "y": round(float(by[i]), 6), "theta": round(float(bt[i]), 6)} for i in range(N)]
        for path in ["solution.json", "best_solution.json"]:
            with open(path, "w") as f:
                json.dump(solution, f, indent=2)
        print("Saved", flush=True)


if __name__ == "__main__":
    main()
