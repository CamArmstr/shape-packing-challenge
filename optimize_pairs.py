#!/usr/bin/env python3
"""
Conjugate-pairs optimizer.
Key insight: 15 semicircles = 7 pairs + 1 singleton.
Two semicircles flat-side-to-flat-side = a unit disk.
7 unit disks in a hex arrangement fit in radius ~3.0.
Start from this geometry and refine with penalty SA.
"""

import sys, os, json, math, time, random
import numpy as np
os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
from shapely.geometry import Polygon

import importlib.util
spec = importlib.util.spec_from_file_location("overnight", "overnight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

BEST_FILE = 'best_solution.json'
N = 15


def load_best_score():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    bx = np.array([s['x'] for s in raw])
    by = np.array([s['y'] for s in raw])
    bt = np.array([s['theta'] for s in raw])
    val = mod.official_validate(bx, by, bt)
    return bx, by, bt, val.score if val.valid else float('inf')


def center_solution(raw):
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    result = validate_and_score(sol)
    if not result.valid:
        return raw, None
    cx, cy, r = result.mec
    centered = [{'x': round(d['x']-cx, 6), 'y': round(d['y']-cy, 6),
                 'theta': round(d['theta'], 6)} for d in raw]
    return centered, r


def make_pair(cx, cy, pair_angle, pair_orientation):
    """
    Two semicircles flat-side-to-flat-side = unit disk centered at (cx, cy).
    pair_angle: direction the pair is "pointing" (rotation of the pair as a whole)
    pair_orientation: which semicircle is which (0 or 1 = flip)
    
    The flat side of a semicircle is perpendicular to theta.
    For two to join flat-side-to-flat-side:
      sc1: theta = pair_angle
      sc2: theta = pair_angle + pi
    Both centered at (cx, cy) — they share the same center, flat sides touching along
    the diameter perpendicular to pair_angle.
    """
    theta1 = pair_angle
    theta2 = pair_angle + math.pi
    if pair_orientation:
        theta1, theta2 = theta2, theta1
    return (
        {'x': cx, 'y': cy, 'theta': theta1},
        {'x': cx, 'y': cy, 'theta': theta2},
    )


def hex_7_centers(r_disk=1.0, scale=1.0):
    """
    7 disk centers in a hex arrangement.
    One in center, 6 around it at distance 2*r_disk (touching).
    Scale > 1 spreads them out to avoid initial overlaps.
    """
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        centers.append((
            scale * 2 * r_disk * math.cos(angle),
            scale * 2 * r_disk * math.sin(angle)
        ))
    return centers


def build_pairs_start(scale=1.05, singleton_gap_idx=0, pair_angles=None, noise=0.0):
    """
    Build a starting configuration using 7 conjugate pairs + 1 singleton.
    singleton_gap_idx: which gap between hex disks to put the singleton in.
    """
    centers = hex_7_centers(r_disk=1.0, scale=scale)
    
    if pair_angles is None:
        # Random orientations for each pair
        pair_angles = [random.uniform(0, math.pi) for _ in range(7)]
    
    raw = []
    for i, (cx, cy) in enumerate(centers):
        if noise > 0:
            cx += random.gauss(0, noise)
            cy += random.gauss(0, noise)
        angle = pair_angles[i % len(pair_angles)]
        sc1, sc2 = make_pair(cx, cy, angle, 0)
        raw.append(sc1)
        raw.append(sc2)
    
    # We now have 14 semicircles (7 pairs). Need 1 more singleton.
    # Place it in a gap between hex disks.
    # Gaps are at midpoints between adjacent outer disks.
    gap_centers = []
    for i in range(6):
        a1 = i * math.pi / 3
        a2 = (i + 1) * math.pi / 3
        cx1 = scale * 2 * math.cos(a1)
        cy1 = scale * 2 * math.sin(a1)
        cx2 = scale * 2 * math.cos(a2)
        cy2 = scale * 2 * math.sin(a2)
        # Also try midpoint between center pair and outer pair
        gap_centers.append(((cx1 + cx2) / 2, (cy1 + cy2) / 2))
    
    # Also try outside the ring
    for i in range(6):
        a = (i + 0.5) * math.pi / 3
        gap_centers.append((scale * 3.5 * math.cos(a), scale * 3.5 * math.sin(a)))
    
    # Pick the gap position and find a valid placement for the singleton
    sc_raw = raw[:14]  # the 14 paired semicircles
    
    best_singleton = None
    best_singleton_dist = float('inf')
    
    for gcx, gcy in gap_centers:
        for t in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            # Try various positions around the gap center
            for r_off in [0.0, 0.3, 0.6, 1.0]:
                for angle_off in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                    x = gcx + r_off * math.cos(angle_off)
                    y = gcy + r_off * math.sin(angle_off)
                    sc = Semicircle(x, y, t)
                    # Check no overlap with existing 14
                    ok = True
                    for d in sc_raw:
                        dx = x - d['x']
                        dy = y - d['y']
                        if dx*dx + dy*dy <= 4.01:
                            if semicircles_overlap(sc, Semicircle(d['x'], d['y'], d['theta'])):
                                ok = False
                                break
                    if ok:
                        # Score: prefer close to center
                        dist = math.sqrt(x*x + y*y)
                        if dist < best_singleton_dist:
                            best_singleton_dist = dist
                            best_singleton = {'x': x, 'y': y, 'theta': t}
    
    if best_singleton is None:
        return None
    
    raw.append(best_singleton)
    return raw


def validate_raw(raw):
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    result = validate_and_score(sol)
    return result


def run_pairs_search(n_attempts=50, sa_steps=2000000, label="pairs"):
    global_best_score = load_best_score()[3]
    print(f"Starting best: {global_best_score:.6f}")
    
    for attempt in range(n_attempts):
        # Vary scale and pair angles
        scale = random.uniform(1.02, 1.15)
        pair_angles = [random.uniform(0, math.pi) for _ in range(7)]
        noise = random.uniform(0, 0.05)
        
        print(f"\n[{label}] Attempt {attempt+1}/{n_attempts}: scale={scale:.3f} noise={noise:.3f}")
        
        raw = build_pairs_start(scale=scale, pair_angles=pair_angles, noise=noise)
        if raw is None:
            print(f"  Could not place singleton, skipping")
            continue
        
        result = validate_raw(raw)
        if not result.valid:
            print(f"  Initial config invalid, skipping")
            continue
        
        print(f"  Initial score: {result.score:.4f}")
        
        # Convert to arrays for overnight.py's SA
        xs = np.array([d['x'] for d in raw])
        ys = np.array([d['y'] for d in raw])
        ts = np.array([d['theta'] for d in raw])
        
        # Run penalty SA from this start
        # Tighter schedule for pairs: already near feasible, don't need to explore far
        t0 = time.time()
        result_sa, best_r = mod.sa_run(
            xs, ys, ts,
            n_steps=sa_steps,
            T_start=0.4, T_end=0.0005,
            lam_start=50.0, lam_end=5000.0,
            seed=attempt * 137 + 42,
            label=f"{label}_{attempt}"
        )
        elapsed = time.time() - t0
        
        if result_sa is None:
            print(f"  SA returned no result ({elapsed:.0f}s)")
            continue
        
        rx, ry, rt = result_sa
        val = mod.official_validate(rx, ry, rt)
        
        if val.valid:
            print(f"  VALID: {val.score:.6f} ({elapsed:.0f}s)")
            if val.score < global_best_score:
                global_best_score = val.score
                # Center and save
                raw_result = [{'x': float(rx[i]), 'y': float(ry[i]), 'theta': float(rt[i])} for i in range(N)]
                centered, score = center_solution(raw_result)
                if score is not None:
                    with open(BEST_FILE, 'w') as f:
                        json.dump(centered, f, indent=2)
                    # Save viz
                    sol_v = [Semicircle(d['x'], d['y'], d['theta']) for d in centered]
                    r_v = validate_and_score(sol_v)
                    from src.semicircle_packing.visualization import plot_packing
                    plot_packing(sol_v, r_v.mec, save_path='best_solution.png')
                    print(f"  *** NEW BEST: {global_best_score:.6f} ***")
        else:
            print(f"  INVALID ({elapsed:.0f}s)")
    
    return global_best_score


if __name__ == '__main__':
    # First: show what the pairs starting geometry looks like
    import sys
    
    print("Building conjugate-pairs starting geometry...")
    raw = build_pairs_start(scale=1.05, noise=0.0)
    
    if raw:
        result = validate_raw(raw)
        print(f"Pairs start: valid={result.valid}, score={result.score:.4f}")
        
        # Save viz of starting geometry
        if result.valid:
            from src.semicircle_packing.visualization import plot_packing
            sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
            plot_packing(sol, result.mec, save_path='pairs_start.png')
            print("Saved pairs_start.png")
            
            # Center it
            centered, r = center_solution(raw)
            print(f"Centered score: {r:.4f}")
    else:
        print("Failed to build pairs start")
        sys.exit(1)
    
    print("\nRunning pairs-based search...")
    run_pairs_search(n_attempts=100, sa_steps=1500000, label="pairs")
