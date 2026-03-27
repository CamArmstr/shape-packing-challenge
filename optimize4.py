"""
Semicircle packing optimizer v4.
Uses OFFICIAL Shapely overlap check for correctness.
Optimizes by SA, moving one semicircle at a time.
Accepts: only moves that don't create overlaps AND reduce MEC.
"""

import json
import math
import time
import numpy as np
from semicircle_packing.geometry import Semicircle, semicircles_overlap, semicircle_polygon
from semicircle_packing.scoring import validate_and_score, compute_mec

RADIUS = 1.0
N = 15


def check_overlaps_for_index(scs, idx):
    """Check if semicircle[idx] overlaps any other. Returns True if overlap found."""
    for j in range(N):
        if j == idx:
            continue
        # Quick distance check first
        dx = scs[idx].x - scs[j].x
        dy = scs[idx].y - scs[j].y
        if dx*dx + dy*dy > 4.0:
            continue
        if semicircles_overlap(scs[idx], scs[j]):
            return True
    return False


def has_any_overlap(scs):
    """Check all pairs for overlap."""
    for i in range(N):
        for j in range(i+1, N):
            dx = scs[i].x - scs[j].x
            dy = scs[i].y - scs[j].y
            if dx*dx + dy*dy > 4.0:
                continue
            if semicircles_overlap(scs[i], scs[j]):
                return True
    return False


def fast_mec_radius(scs):
    """Quick MEC estimate from 3 key points per semicircle."""
    pts = []
    for sc in scs:
        pts.append((sc.x + math.cos(sc.theta), sc.y + math.sin(sc.theta)))
        pts.append((sc.x + math.cos(sc.theta + math.pi/2), sc.y + math.sin(sc.theta + math.pi/2)))
        pts.append((sc.x + math.cos(sc.theta - math.pi/2), sc.y + math.sin(sc.theta - math.pi/2)))
    
    pts = np.array(pts)
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    
    for it in range(200):
        dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        idx = np.argmax(dists)
        step = 0.3 / (1 + it/5)
        cx += step * (pts[idx, 0] - cx)
        cy += step * (pts[idx, 1] - cy)
    
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    return float(np.max(dists))


def make_scs(xs, ys, ts):
    return [Semicircle(x=round(xs[i], 6), y=round(ys[i], 6), theta=round(ts[i], 6)) for i in range(N)]


def sa_feasible(xs, ys, ts, n_steps=200000, T_start=0.5, T_end=0.001, seed=42):
    """
    SA that only accepts feasible (non-overlapping) moves.
    Objective: minimize MEC radius.
    """
    rng = np.random.default_rng(seed)
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    
    scs = make_scs(xs, ys, ts)
    current_r = fast_mec_radius(scs)
    best_r = current_r
    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
    
    n_accepted = 0
    n_feasible = 0
    n_improved = 0
    
    for step in range(n_steps):
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        
        idx = rng.integers(0, N)
        old_x, old_y, old_t = xs[idx], ys[idx], ts[idx]
        
        scale = 0.3 * (T / T_start) + 0.005
        
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
        
        # Check feasibility (only the moved semicircle)
        new_sc = Semicircle(x=round(xs[idx], 6), y=round(ys[idx], 6), theta=round(ts[idx], 6))
        scs_new = list(scs)
        scs_new[idx] = new_sc
        
        if check_overlaps_for_index(scs_new, idx):
            # Infeasible: revert
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
            continue
        
        n_feasible += 1
        new_r = fast_mec_radius(scs_new)
        delta = new_r - current_r
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
            scs = scs_new
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
        
        if step % 50000 == 0 and step > 0:
            feas_rate = n_feasible / step if step > 0 else 0
            print(f"    step {step:>7d}: r={current_r:.6f} best={best_r:.6f} T={T:.6f} feas={feas_rate:.1%}")
    
    return best_xs, best_ys, best_ts, best_r


# ── Configs ──────────────────────────────────────────────────────

def config_grid():
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    idx = 0
    for row in range(2):
        for col in range(3):
            cx = (col - 1) * 2.0
            cy = row * 2.0
            xs[idx], ys[idx], ts[idx] = cx, cy, 0; idx += 1
            xs[idx], ys[idx], ts[idx] = cx, cy, math.pi; idx += 1
    for col in range(3):
        xs[idx], ys[idx], ts[idx] = (col-1)*2.0, 3.0, math.pi/2; idx += 1
    return xs, ys, ts


def config_tight_grid():
    """Tighter grid: pairs closer together."""
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    idx = 0
    # 3x2 grid but spacing = 2*R = 2.0 (touching)
    for row in range(2):
        for col in range(3):
            cx = (col - 1) * 2.0
            cy = row * 2.0
            xs[idx], ys[idx], ts[idx] = cx, cy, 0; idx += 1
            xs[idx], ys[idx], ts[idx] = cx, cy, math.pi; idx += 1
    # 3 on top, pushed closer
    for col in range(3):
        xs[idx], ys[idx], ts[idx] = (col-1)*2.0, 2.5, math.pi/2; idx += 1
    return xs, ys, ts


def config_2x3_plus_row():
    """2 rows of 3 pairs (12) + row of 3 singles on top, staggered."""
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    idx = 0
    for row in range(2):
        for col in range(3):
            cx = (col - 1) * 2.0
            cy = row * 2.0
            xs[idx], ys[idx], ts[idx] = cx, cy, 0; idx += 1
            xs[idx], ys[idx], ts[idx] = cx, cy, math.pi; idx += 1
    # Staggered row on top
    for k in range(3):
        cx = (k - 1) * 2.0
        xs[idx], ys[idx], ts[idx] = cx, 3.0, -math.pi/2; idx += 1  # pointing down
    return xs, ys, ts


def config_random(seed=42):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-2.5, 2.5, N)
    ys = rng.uniform(-2.5, 2.5, N)
    ts = rng.uniform(0, 2*math.pi, N)
    return xs, ys, ts


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Semicircle Packing Optimizer v4 (Feasible SA)")
    print("=" * 60)
    
    # Benchmark overlap check speed
    xs, ys, ts = config_grid()
    scs = make_scs(xs, ys, ts)
    t0 = time.time()
    for _ in range(100):
        check_overlaps_for_index(scs, 0)
    elapsed = time.time() - t0
    print(f"\nOverlap check: {100/elapsed:.0f} checks/sec (single index)")
    print(f"Expected SA speed: ~{100/elapsed/2:.0f} steps/sec")
    
    configs = {
        "grid": config_grid(),
        "tight_grid": config_tight_grid(),
        "down_row": config_2x3_plus_row(),
    }
    
    best_score = float('inf')
    best_result = None
    
    # Phase 1: 200k steps from structured configs
    print("\nPhase 1: SA from structured configs (200k steps)")
    
    for name, (xs, ys, ts) in configs.items():
        scs = make_scs(xs, ys, ts)
        if has_any_overlap(scs):
            print(f"\n  [{name}] Starting config has overlaps, skipping")
            continue
        
        print(f"\n  [{name}]")
        t0 = time.time()
        bx, by, bt, r = sa_feasible(xs, ys, ts, n_steps=200000, T_start=1.0, T_end=0.001, seed=42)
        elapsed = time.time() - t0
        
        # Official validate
        scs_best = make_scs(bx, by, bt)
        val = validate_and_score(scs_best)
        
        if val.valid:
            print(f"  [{name}] VALID score={val.score:.6f} ({elapsed:.1f}s)")
            if val.score < best_score:
                best_score = val.score
                best_result = (bx.copy(), by.copy(), bt.copy())
        else:
            print(f"  [{name}] INVALID fast_r={r:.6f} errors={len(val.errors)} ({elapsed:.1f}s)")
    
    # Phase 2: longer run from best
    if best_result is not None:
        print(f"\n{'=' * 60}")
        print(f"Phase 1 best: {best_score:.6f}")
        print(f"Phase 2: Extended SA (1M steps)")
        
        bx, by, bt = best_result
        t0 = time.time()
        bx2, by2, bt2, r = sa_feasible(bx, by, bt, n_steps=1000000, T_start=0.3, T_end=0.0001, seed=123)
        elapsed = time.time() - t0
        
        scs_best = make_scs(bx2, by2, bt2)
        val = validate_and_score(scs_best)
        
        if val.valid:
            print(f"  Phase 2 VALID score={val.score:.6f} ({elapsed:.1f}s)")
            if val.score < best_score:
                best_score = val.score
                best_result = (bx2.copy(), by2.copy(), bt2.copy())
        else:
            print(f"  Phase 2 INVALID errors={len(val.errors)} ({elapsed:.1f}s)")
    
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
        print("Saved to solution.json and best_solution.json")


if __name__ == "__main__":
    main()
