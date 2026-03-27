#!/usr/bin/env python3
"""
Overnight semicircle packing optimizer.
Penalty-based SA with 128-pt Shapely overlap.
Runs multiple configs, keeps best valid solution.
Designed to run for hours unattended.
"""

import json
import math
import time
import sys
import os
import numpy as np
from shapely.geometry import Polygon

RADIUS = 1.0
N = 15
ARC_PTS = 128  # Higher than v7's 64 to avoid false negatives


def make_poly(x, y, theta):
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, ARC_PTS)
    return Polygon(list(zip(x + RADIUS * np.cos(angles), y + RADIUS * np.sin(angles))))


def overlap_for_index(xs, ys, ts, polys, idx, new_poly=None):
    """Compute overlap area for one semicircle against all others."""
    p = new_poly if new_poly is not None else polys[idx]
    total = 0.0
    for j in range(N):
        if j == idx:
            continue
        dx = xs[idx] - xs[j]
        dy = ys[idx] - ys[j]
        if dx*dx + dy*dy > 4.0:
            continue
        area = p.intersection(polys[j]).area
        if area > 1e-8:
            total += area
    return total


def total_overlap(xs, ys, ts, polys):
    """Total pairwise overlap area."""
    total = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue
            area = polys[i].intersection(polys[j]).area
            if area > 1e-8:
                total += area
    return total


ARC_PTS_MEC = 16  # arc sample points for fast_mec (was 3 — tripled accuracy)
_ARC_OFFSETS = np.linspace(-math.pi/2, math.pi/2, ARC_PTS_MEC)
PTS_PER_SC = ARC_PTS_MEC + 2  # arc + 2 flat endpoints


def compute_boundary_pts(xs, ys, ts):
    """Sample ARC_PTS_MEC arc points + 2 flat endpoints per semicircle."""
    pts = np.zeros((N * PTS_PER_SC, 2))
    for i in range(N):
        x, y, t = xs[i], ys[i], ts[i]
        base = i * PTS_PER_SC
        for k, off in enumerate(_ARC_OFFSETS):
            a = t + off
            pts[base + k] = [x + math.cos(a), y + math.sin(a)]
        pts[base + ARC_PTS_MEC] = [x + math.cos(t + math.pi/2), y + math.sin(t + math.pi/2)]
        pts[base + ARC_PTS_MEC + 1] = [x + math.cos(t - math.pi/2), y + math.sin(t - math.pi/2)]
    return pts


def update_boundary_pts(pts, xs, ys, ts, idx):
    """Update boundary points for a single semicircle."""
    x, y, t = xs[idx], ys[idx], ts[idx]
    base = idx * PTS_PER_SC
    for k, off in enumerate(_ARC_OFFSETS):
        a = t + off
        pts[base + k] = [x + math.cos(a), y + math.sin(a)]
    pts[base + ARC_PTS_MEC] = [x + math.cos(t + math.pi/2), y + math.sin(t + math.pi/2)]
    pts[base + ARC_PTS_MEC + 1] = [x + math.cos(t - math.pi/2), y + math.sin(t - math.pi/2)]


def fast_mec(pts):
    """MEC from precomputed boundary points + iterative minimax."""
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    for k in range(80):
        dists_sq = (pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2
        idx = np.argmax(dists_sq)
        step = max(0.01, 0.2 / (1 + k/5))
        cx += step * (pts[idx, 0] - cx)
        cy += step * (pts[idx, 1] - cy)
    
    return float(np.sqrt(np.max((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)))


def sa_run(xs, ys, ts, n_steps=1000000, T_start=2.0, T_end=0.002,
           lam_start=5.0, lam_end=2000.0, seed=42, label=""):
    """Single SA run. Returns best feasible solution or best overall."""
    rng = np.random.default_rng(seed)
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    
    polys = [make_poly(xs[i], ys[i], ts[i]) for i in range(N)]
    bpts = compute_boundary_pts(xs, ys, ts)
    
    cur_ovlp = total_overlap(xs, ys, ts, polys)
    cur_r = fast_mec(bpts)
    lam = lam_start
    cur_obj = cur_r + lam * cur_ovlp
    
    best_feas_r = float('inf')
    best_feas = None
    n_improved = 0
    
    for step in range(n_steps):
        frac = step / n_steps
        T = T_start * (T_end / T_start) ** frac
        lam = lam_start * (lam_end / lam_start) ** frac
        
        scale = 0.25 * (1.0 - 0.85 * frac) + 0.002
        
        idx = rng.integers(0, N)
        old_x, old_y, old_t = xs[idx], ys[idx], ts[idx]
        old_poly = polys[idx]
        
        m = rng.random()
        if m < 0.30:
            xs[idx] += rng.normal(0, scale)
            ys[idx] += rng.normal(0, scale)
        elif m < 0.50:
            ts[idx] += rng.normal(0, scale * 3)
            ts[idx] %= 2 * math.pi
        elif m < 0.80:
            # Squeeze toward centroid
            cx = np.mean(xs)
            cy = np.mean(ys)
            dx = cx - xs[idx]
            dy = cy - ys[idx]
            d = math.sqrt(dx*dx + dy*dy)
            if d > 0.01:
                xs[idx] += scale * 0.4 * dx / d
                ys[idx] += scale * 0.4 * dy / d
            ts[idx] += rng.normal(0, scale)
            ts[idx] %= 2 * math.pi
        else:
            xs[idx] += rng.normal(0, scale * 0.5)
            ys[idx] += rng.normal(0, scale * 0.5)
            ts[idx] += rng.normal(0, scale * 2)
            ts[idx] %= 2 * math.pi
        
        # Compute old overlap for this index BEFORE changing position
        # Need to temporarily revert to get correct distances
        new_x, new_y, new_t = xs[idx], ys[idx], ts[idx]
        xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
        old_idx_ovlp = overlap_for_index(xs, ys, ts, polys, idx, old_poly)
        
        # Now apply new position and compute new overlap
        xs[idx], ys[idx], ts[idx] = new_x, new_y, new_t
        new_poly = make_poly(xs[idx], ys[idx], ts[idx])
        new_idx_ovlp = overlap_for_index(xs, ys, ts, polys, idx, new_poly)
        
        new_ovlp = cur_ovlp - old_idx_ovlp + new_idx_ovlp
        
        # Update boundary points for MEC
        old_bpts = bpts[idx*3:idx*3+3].copy()
        update_boundary_pts(bpts, xs, ys, ts, idx)
        new_r = fast_mec(bpts)
        new_obj = new_r + lam * new_ovlp
        
        delta = new_obj - cur_obj
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
            polys[idx] = new_poly
            cur_ovlp = new_ovlp
            cur_r = new_r
            cur_obj = new_obj
            
            if new_ovlp < 1e-5 and new_r < best_feas_r:
                best_feas_r = new_r
                best_feas = (xs.copy(), ys.copy(), ts.copy())
                n_improved += 1
        else:
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
            bpts[idx*3:idx*3+3] = old_bpts
        
        if step % 100000 == 0:
            print(f"  [{label}] {step:>7d}/{n_steps}: r={cur_r:.4f} ovlp={cur_ovlp:.6f} lam={lam:.0f} best_feas={best_feas_r:.4f} imp={n_improved}", flush=True)
    
    return best_feas, best_feas_r


# ── Configs ──────────────────────────────────────────────────────

def ring_config(n=N, r=3.5, inward=True):
    xs = np.zeros(n); ys = np.zeros(n); ts = np.zeros(n)
    for i in range(n):
        a = 2 * math.pi * i / n
        xs[i] = r * math.cos(a)
        ys[i] = r * math.sin(a)
        ts[i] = (a + math.pi) if inward else a
    return xs, ys, ts


def double_ring():
    """Inner 5 + outer 10."""
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    for i in range(5):
        a = 2 * math.pi * i / 5
        xs[i] = 1.2 * math.cos(a)
        ys[i] = 1.2 * math.sin(a)
        ts[i] = a + math.pi
    for i in range(10):
        a = 2 * math.pi * i / 10
        xs[5+i] = 3.0 * math.cos(a)
        ys[5+i] = 3.0 * math.sin(a)
        ts[5+i] = a + math.pi
    return xs, ys, ts


def triple_ring():
    """3 + 6 + 6."""
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    idx = 0
    for i in range(3):
        a = 2 * math.pi * i / 3
        xs[idx] = 0.8 * math.cos(a); ys[idx] = 0.8 * math.sin(a); ts[idx] = a + math.pi; idx += 1
    for i in range(6):
        a = 2 * math.pi * i / 6
        xs[idx] = 2.0 * math.cos(a); ys[idx] = 2.0 * math.sin(a); ts[idx] = a + math.pi; idx += 1
    for i in range(6):
        a = 2 * math.pi * i / 6 + math.pi/6
        xs[idx] = 3.2 * math.cos(a); ys[idx] = 3.2 * math.sin(a); ts[idx] = a + math.pi; idx += 1
    return xs, ys, ts


def paired_triangle():
    """5 pairs in pentagon + 5 facing center."""
    xs = np.zeros(N); ys = np.zeros(N); ts = np.zeros(N)
    idx = 0
    for i in range(5):
        a = 2 * math.pi * i / 5
        r = 2.5
        xs[idx] = r * math.cos(a); ys[idx] = r * math.sin(a); ts[idx] = a; idx += 1
        xs[idx] = r * math.cos(a); ys[idx] = r * math.sin(a); ts[idx] = a + math.pi; idx += 1
    for i in range(5):
        a = 2 * math.pi * i / 5 + math.pi/5
        xs[idx] = 0.5 * math.cos(a); ys[idx] = 0.5 * math.sin(a); ts[idx] = a + math.pi; idx += 1
    return xs, ys, ts


def random_config(seed=42):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-3, 3, N)
    ys = rng.uniform(-3, 3, N)
    ts = rng.uniform(0, 2*math.pi, N)
    return xs, ys, ts


def official_validate(xs, ys, ts):
    from semicircle_packing.geometry import Semicircle
    from semicircle_packing.scoring import validate_and_score
    scs = [Semicircle(x=round(xs[i], 6), y=round(ys[i], 6), theta=round(ts[i], 6)) for i in range(N)]
    return validate_and_score(scs)


def save_solution(xs, ys, ts, path="solution.json"):
    sol = [{"x": round(float(xs[i]), 6), "y": round(float(ys[i]), 6), "theta": round(float(ts[i]), 6)} for i in range(N)]
    with open(path, "w") as f:
        json.dump(sol, f, indent=2)


def main():
    print("=" * 60)
    print("Overnight Semicircle Packing Optimizer")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60, flush=True)
    
    # Load best known solution if exists
    global_best = float('inf')
    if os.path.exists("best_solution.json"):
        with open("best_solution.json") as f:
            sol = json.load(f)
        bx = np.array([s["x"] for s in sol])
        by = np.array([s["y"] for s in sol])
        bt = np.array([s["theta"] for s in sol])
        val = official_validate(bx, by, bt)
        if val.valid:
            global_best = val.score
            print(f"Loaded existing best: {global_best:.6f}", flush=True)
    
    configs = [
        ("ring_3.5", ring_config(r=3.5)),
        ("ring_3.0", ring_config(r=3.0)),
        ("ring_4.0", ring_config(r=4.0)),
        ("ring_out", ring_config(r=3.5, inward=False)),
        ("double", double_ring()),
        ("triple", triple_ring()),
        ("pentagon", paired_triangle()),
    ]
    # Add random seeds
    for s in [42, 99, 777, 1234, 9999]:
        configs.append((f"rand_{s}", random_config(s)))
    
    run_number = 0
    
    # Phase 1: 1M steps per config
    for name, (xs, ys, ts) in configs:
        run_number += 1
        print(f"\n{'─' * 50}")
        print(f"Run {run_number}/{len(configs)}: {name} (1M steps)")
        print(f"{'─' * 50}", flush=True)
        
        t0 = time.time()
        result, best_r = sa_run(xs, ys, ts, 
                                n_steps=1000000,
                                T_start=2.0, T_end=0.002,
                                lam_start=5.0, lam_end=2000.0,
                                seed=run_number * 137,
                                label=name)
        elapsed = time.time() - t0
        
        if result is not None:
            bx, by, bt = result
            val = official_validate(bx, by, bt)
            if val.valid:
                print(f"  [{name}] VALID: {val.score:.6f} ({elapsed:.0f}s)", flush=True)
                if val.score < global_best:
                    global_best = val.score
                    save_solution(bx, by, bt, "best_solution.json")
                    save_solution(bx, by, bt, "solution.json")
                    print(f"  *** NEW BEST: {global_best:.6f} ***", flush=True)
            else:
                print(f"  [{name}] INVALID (fast_r={best_r:.4f}, errors={len(val.errors)}) ({elapsed:.0f}s)", flush=True)
        else:
            print(f"  [{name}] No feasible solution found ({elapsed:.0f}s)", flush=True)
    
    # Phase 2: Extended runs from best + top configs with different seeds
    if global_best < float('inf'):
        print(f"\n{'=' * 60}")
        print(f"Phase 2: Extended refinement from best ({global_best:.6f})")
        print(f"{'=' * 60}", flush=True)
        
        with open("best_solution.json") as f:
            sol = json.load(f)
        bx = np.array([s["x"] for s in sol])
        by = np.array([s["y"] for s in sol])
        bt = np.array([s["theta"] for s in sol])
        
        for attempt in range(5):
            print(f"\n  Refinement {attempt+1}/5 (3M steps)", flush=True)
            t0 = time.time()
            result, best_r = sa_run(bx, by, bt,
                                    n_steps=3000000,
                                    T_start=0.3, T_end=0.0005,
                                    lam_start=100.0, lam_end=5000.0,
                                    seed=10000 + attempt * 31,
                                    label=f"refine_{attempt}")
            elapsed = time.time() - t0
            
            if result is not None:
                rx, ry, rt = result
                val = official_validate(rx, ry, rt)
                if val.valid and val.score < global_best:
                    global_best = val.score
                    save_solution(rx, ry, rt, "best_solution.json")
                    save_solution(rx, ry, rt, "solution.json")
                    bx, by, bt = rx, ry, rt
                    print(f"  *** NEW BEST: {global_best:.6f} *** ({elapsed:.0f}s)", flush=True)
                elif val.valid:
                    print(f"  Valid but not better: {val.score:.6f} ({elapsed:.0f}s)", flush=True)
                else:
                    print(f"  Invalid ({elapsed:.0f}s)", flush=True)
            else:
                print(f"  No feasible ({elapsed:.0f}s)", flush=True)
    
    print(f"\n{'=' * 60}")
    print(f"FINAL BEST: {global_best:.6f}")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
