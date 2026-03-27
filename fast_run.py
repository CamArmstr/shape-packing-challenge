#!/usr/bin/env python3
"""
Fast targeted run: feasible SA from grid configs.
Uses 128-pt Shapely for overlap, fast MEC for SA, official MEC for scoring.
Goal: get a valid sub-3.2 solution ASAP.
"""

import json
import math
import time
import sys
import numpy as np
from shapely.geometry import Polygon

from semicircle_packing.geometry import Semicircle
from semicircle_packing.scoring import validate_and_score, compute_mec

RADIUS = 1.0
N = 15
ARC_PTS = 256  # Higher accuracy to match 4096-pt official scorer


def make_poly(x, y, theta):
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, ARC_PTS)
    return Polygon(list(zip(x + RADIUS * np.cos(angles), y + RADIUS * np.sin(angles))))


def check_overlap_one(xs, ys, ts, polys, idx, new_poly):
    """Returns True if overlap found."""
    for j in range(N):
        if j == idx:
            continue
        dx = xs[idx] - xs[j]
        dy = ys[idx] - ys[j]
        if dx*dx + dy*dy > 4.0:
            continue
        if new_poly.intersection(polys[j]).area > 5e-7:
            return True
    return False


def fast_mec(xs, ys, ts):
    """Fast MEC from boundary points."""
    pts = np.zeros((N * 3, 2))
    for i in range(N):
        x, y, t = xs[i], ys[i], ts[i]
        pts[i*3] = [x + math.cos(t), y + math.sin(t)]
        pts[i*3+1] = [x + math.cos(t + math.pi/2), y + math.sin(t + math.pi/2)]
        pts[i*3+2] = [x + math.cos(t - math.pi/2), y + math.sin(t - math.pi/2)]
    
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    for k in range(100):
        dists_sq = (pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2
        idx = np.argmax(dists_sq)
        step = max(0.008, 0.15 / (1 + k/5))
        cx += step * (pts[idx, 0] - cx)
        cy += step * (pts[idx, 1] - cy)
    
    return float(np.sqrt(np.max((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)))


def official_score(xs, ys, ts):
    scs = [Semicircle(x=round(xs[i], 6), y=round(ys[i], 6), theta=round(ts[i], 6)) for i in range(N)]
    return validate_and_score(scs)


def sa_feasible(xs, ys, ts, n_steps=1000000, T_start=0.5, T_end=0.001, seed=42, label=""):
    rng = np.random.default_rng(seed)
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    
    polys = [make_poly(xs[i], ys[i], ts[i]) for i in range(N)]
    cur_r = fast_mec(xs, ys, ts)
    best_r = cur_r
    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
    n_improved = 0
    n_feasible_moves = 0
    
    for step in range(n_steps):
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        scale = 0.2 * (T / T_start) + 0.002
        
        idx = rng.integers(0, N)
        old_x, old_y, old_t = xs[idx], ys[idx], ts[idx]
        
        m = rng.random()
        if m < 0.30:
            xs[idx] += rng.normal(0, scale)
            ys[idx] += rng.normal(0, scale)
        elif m < 0.50:
            ts[idx] += rng.normal(0, scale * 3)
            ts[idx] %= 2 * math.pi
        elif m < 0.80:
            # Squeeze toward centroid
            cx = np.mean(xs); cy = np.mean(ys)
            dx = cx - xs[idx]; dy = cy - ys[idx]
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
        
        if check_overlap_one(xs, ys, ts, polys, idx, new_poly):
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
            continue
        
        n_feasible_moves += 1
        new_r = fast_mec(xs, ys, ts)
        delta = new_r - cur_r
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
            polys[idx] = new_poly
            cur_r = new_r
            if new_r < best_r:
                best_r = new_r
                best_xs[:] = xs; best_ys[:] = ys; best_ts[:] = ts
                n_improved += 1
        else:
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
        
        if step % 100000 == 0:
            feas_pct = n_feasible_moves / max(step, 1) * 100
            print(f"  [{label}] {step:>7d}: r={cur_r:.6f} best={best_r:.6f} T={T:.4f} imp={n_improved} feas={feas_pct:.0f}%", flush=True)
        
        # Periodic official scoring
        if step > 0 and step % 500000 == 0:
            val = official_score(best_xs, best_ys, best_ts)
            if val.valid:
                print(f"  [{label}] CHECKPOINT: official={val.score:.6f} (fast={best_r:.6f})", flush=True)
    
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


def config_grid_down():
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


def config_grid_compact():
    """Grid with rows closer together."""
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    idx = 0
    for row in range(2):
        for col in range(3):
            cx = (col-1)*2.0
            cy = row * 1.8  # Tighter vertical spacing
            xs[idx], ys[idx], ts[idx] = cx, cy, 0; idx += 1
            xs[idx], ys[idx], ts[idx] = cx, cy, math.pi; idx += 1
    for col in range(3):
        xs[idx], ys[idx], ts[idx] = (col-1)*2.0, 2.6, math.pi/2; idx += 1
    return xs, ys, ts


def save_best(xs, ys, ts, path="best_solution.json"):
    sol = [{"x": round(float(xs[i]), 6), "y": round(float(ys[i]), 6), "theta": round(float(ts[i]), 6)} for i in range(N)]
    for p in [path, "solution.json"]:
        with open(p, "w") as f:
            json.dump(sol, f, indent=2)


def main():
    print("=" * 60)
    print("Fast Run: Get sub-3.2 ASAP")
    print(f"Started: {time.strftime('%H:%M:%S')}")
    print("=" * 60, flush=True)
    
    global_best = float('inf')
    
    # Load existing best
    try:
        val = official_score(*[np.array([s[k] for s in json.load(open("best_solution.json"))]) for k in ["x", "y", "theta"]])
        if val.valid:
            global_best = val.score
            print(f"Existing best: {global_best:.6f}", flush=True)
    except:
        pass
    
    configs = [
        ("grid_s42", config_grid(), 42),
        ("grid_down_s42", config_grid_down(), 42),
        ("grid_s99", config_grid(), 99),
        ("grid_down_s99", config_grid_down(), 99),
    ]
    
    for name, (xs, ys, ts), seed in configs:
        # Verify starting config is valid
        val0 = official_score(xs, ys, ts)
        if not val0.valid:
            print(f"\n  [{name}] Starting config INVALID, skipping", flush=True)
            continue
        
        print(f"\n{'─' * 50}")
        print(f"Config: {name} (seed={seed}, 2M steps)")
        print(f"Start score: {val0.score:.6f}")
        print(f"{'─' * 50}", flush=True)
        
        t0 = time.time()
        bx, by, bt, fast_r = sa_feasible(xs, ys, ts, 
                                          n_steps=2000000,
                                          T_start=1.0, T_end=0.0005,
                                          seed=seed, label=name)
        elapsed = time.time() - t0
        
        val = official_score(bx, by, bt)
        if val.valid:
            print(f"  [{name}] FINAL: official={val.score:.6f} fast={fast_r:.6f} ({elapsed:.0f}s)", flush=True)
            if val.score < global_best:
                global_best = val.score
                save_best(bx, by, bt)
                print(f"  *** NEW BEST: {global_best:.6f} ***", flush=True)
        else:
            print(f"  [{name}] INVALID fast={fast_r:.6f} errors={len(val.errors)} ({elapsed:.0f}s)", flush=True)
    
    # Phase 2: refine best
    if global_best < float('inf'):
        print(f"\n{'=' * 60}")
        print(f"Phase 2: Refine best ({global_best:.6f})")
        print("=" * 60, flush=True)
        
        sol = json.load(open("best_solution.json"))
        bx = np.array([s["x"] for s in sol])
        by = np.array([s["y"] for s in sol])
        bt = np.array([s["theta"] for s in sol])
        
        for attempt in range(10):
            print(f"\n  Refinement {attempt+1}/10 (5M steps)", flush=True)
            t0 = time.time()
            rx, ry, rt, fast_r = sa_feasible(bx, by, bt,
                                             n_steps=5000000,
                                             T_start=0.15,
                                             T_end=0.00005,
                                             seed=10000 + attempt * 137,
                                             label=f"ref{attempt}")
            elapsed = time.time() - t0
            
            val = official_score(rx, ry, rt)
            if val.valid and val.score < global_best:
                global_best = val.score
                save_best(rx, ry, rt)
                bx, by, bt = rx.copy(), ry.copy(), rt.copy()
                print(f"  *** NEW BEST: {global_best:.6f} *** ({elapsed:.0f}s)", flush=True)
            elif val.valid:
                print(f"  {val.score:.6f} (not better) ({elapsed:.0f}s)", flush=True)
            else:
                print(f"  INVALID ({elapsed:.0f}s)", flush=True)
    
    print(f"\n{'=' * 60}")
    print(f"FINAL BEST: {global_best:.6f}")
    print(f"Finished: {time.strftime('%H:%M:%S')}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
