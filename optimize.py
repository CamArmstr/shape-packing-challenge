"""
Semicircle packing optimizer.
Penalty-based continuous optimization using scipy.

Strategy: minimize MEC radius + penalty for overlaps.
Uses analytical overlap detection (fast) for the inner loop,
falls back to Shapely for final validation.

Variables: 15 semicircles × 3 DOF (x, y, theta) = 45 variables.
"""

import json
import math
import time
import sys
import numpy as np
from scipy.optimize import differential_evolution, minimize

# ── Fast analytical overlap (no Shapely) ──────────────────────────────

RADIUS = 1.0
N = 15

def semicircle_points_fast(x, y, theta, n=64):
    """Sample boundary points of a semicircle. Returns (n, 2) array."""
    # Arc points
    n_arc = n // 2
    n_flat = n - n_arc
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, n_arc)
    arc = np.column_stack([x + RADIUS * np.cos(angles), y + RADIUS * np.sin(angles)])
    
    # Flat edge
    perp = theta + math.pi/2
    ex1 = np.array([x + RADIUS * math.cos(perp), y + RADIUS * math.sin(perp)])
    ex2 = np.array([x - RADIUS * math.cos(perp), y - RADIUS * math.sin(perp)])
    t = np.linspace(0, 1, n_flat).reshape(-1, 1)
    flat = ex1 * (1 - t) + ex2 * t
    
    return np.vstack([arc, flat])


def point_in_semicircle(px, py, sc_x, sc_y, sc_theta):
    """Check if point is inside semicircle (vectorized-friendly)."""
    dx = px - sc_x
    dy = py - sc_y
    # Must be within unit circle
    dist_sq = dx*dx + dy*dy
    in_circle = dist_sq < RADIUS*RADIUS - 1e-8
    # Must be on the correct side (dot product with theta direction > 0)
    dot = dx * math.cos(sc_theta) + dy * math.sin(sc_theta)
    return in_circle & (dot > -1e-8)


def overlap_penalty_fast(params):
    """Compute total overlap penalty. Higher = more overlap."""
    xs = params[0::3]
    ys = params[1::3]
    thetas = params[2::3]
    
    penalty = 0.0
    
    for i in range(N):
        for j in range(i+1, N):
            # Quick distance check
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 2 * RADIUS:
                continue
            
            # Sample points of i, check how many are inside j (and vice versa)
            pts_i = semicircle_points_fast(xs[i], ys[i], thetas[i], n=32)
            pts_j = semicircle_points_fast(xs[j], ys[j], thetas[j], n=32)
            
            # Points of i inside j
            for p in pts_i:
                dpx = p[0] - xs[j]
                dpy = p[1] - ys[j]
                if dpx*dpx + dpy*dpy < RADIUS*RADIUS:
                    dot = dpx * math.cos(thetas[j]) + dpy * math.sin(thetas[j])
                    if dot > -1e-8:
                        # Penetration depth approximation
                        d = math.sqrt(dpx*dpx + dpy*dpy)
                        penalty += (RADIUS - d) if d < RADIUS else 0
            
            # Points of j inside i
            for p in pts_j:
                dpx = p[0] - xs[i]
                dpy = p[1] - ys[i]
                if dpx*dpx + dpy*dpy < RADIUS*RADIUS:
                    dot = dpx * math.cos(thetas[i]) + dpy * math.sin(thetas[i])
                    if dot > -1e-8:
                        d = math.sqrt(dpx*dpx + dpy*dpy)
                        penalty += (RADIUS - d) if d < RADIUS else 0
    
    return penalty


def mec_radius_fast(params):
    """Fast MEC radius estimate from boundary points."""
    xs = params[0::3]
    ys = params[1::3]
    thetas = params[2::3]
    
    # Collect all boundary points
    all_pts = []
    for i in range(N):
        pts = semicircle_points_fast(xs[i], ys[i], thetas[i], n=32)
        all_pts.append(pts)
    all_pts = np.vstack(all_pts)
    
    # Approximate MEC: use centroid as center, find max distance
    cx = np.mean(all_pts[:, 0])
    cy = np.mean(all_pts[:, 1])
    dists = np.sqrt((all_pts[:, 0] - cx)**2 + (all_pts[:, 1] - cy)**2)
    
    # Iterative 1-center: move center toward farthest point
    for _ in range(50):
        idx = np.argmax(dists)
        fx, fy = all_pts[idx]
        # Move center 10% toward farthest point
        cx = cx + 0.1 * (fx - cx)
        cy = cy + 0.1 * (fy - cy)
        dists = np.sqrt((all_pts[:, 0] - cx)**2 + (all_pts[:, 1] - cy)**2)
    
    return np.max(dists)


def objective(params, lam=100.0):
    """Combined objective: MEC radius + lambda * overlap penalty."""
    r = mec_radius_fast(params)
    p = overlap_penalty_fast(params)
    return r + lam * p


# ── Initial configurations ────────────────────────────────────────────

def config_paired_hex():
    """7 full disks (hex-ish arrangement) + 1 extra semicircle."""
    params = []
    # Hexagonal-ish: center + ring of 6
    centers = [(0, 0)]
    for k in range(6):
        angle = k * math.pi / 3
        centers.append((2.0 * math.cos(angle), 2.0 * math.sin(angle)))
    
    # 7 disks = 14 semicircles (pairs facing opposite directions)
    for cx, cy in centers:
        params.extend([cx, cy, 0.0])        # right-facing
        params.extend([cx, cy, math.pi])     # left-facing
    
    # 15th semicircle: tuck it somewhere
    params.extend([0.0, 0.0, math.pi/2])  # pointing up at center
    
    return np.array(params)


def config_ring():
    """All 15 on a ring, flat edges inward."""
    params = []
    ring_r = 2.5
    for i in range(N):
        angle = 2 * math.pi * i / N
        x = ring_r * math.cos(angle)
        y = ring_r * math.sin(angle)
        theta = angle + math.pi  # flat edge outward, curve inward
        params.extend([x, y, theta])
    return np.array(params)


def config_grid_baseline():
    """The repo's grid baseline (3.50)."""
    params = []
    for row in range(2):
        for col in range(3):
            cx = (col - 1) * 2
            cy = row * 2
            params.extend([cx, cy, 0.0])
            params.extend([cx, cy, math.pi])
    for col in range(3):
        cx = (col - 1) * 2
        params.extend([cx, cy + 1, math.pi/2])
    return np.array(params)


def config_compact_pairs():
    """5 full disks in tight cross + 5 loose semicircles."""
    params = []
    # Cross pattern: center + 4 cardinal
    disk_centers = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
    for cx, cy in disk_centers:
        params.extend([cx, cy, 0.0])
        params.extend([cx, cy, math.pi])
    
    # 5 loose semicircles filling gaps
    loose = [
        (1.0, 1.0, math.pi/4),
        (-1.0, 1.0, 3*math.pi/4),
        (1.0, -1.0, -math.pi/4),
        (-1.0, -1.0, -3*math.pi/4),
        (0.0, 2.5, math.pi/2),
    ]
    for x, y, t in loose:
        params.extend([x, y, t])
    
    return np.array(params)


def config_random_seed(seed=42):
    """Random initial config within a bounding box."""
    rng = np.random.default_rng(seed)
    params = []
    for _ in range(N):
        x = rng.uniform(-3, 3)
        y = rng.uniform(-3, 3)
        theta = rng.uniform(0, 2*math.pi)
        params.extend([x, y, theta])
    return np.array(params)


# ── Optimization ──────────────────────────────────────────────────────

def run_differential_evolution(init_params=None, maxiter=1000, seed=42, lam=100.0, popsize=30):
    """Run scipy differential_evolution with penalty method."""
    bounds = []
    for i in range(N):
        bounds.append((-4, 4))      # x
        bounds.append((-4, 4))      # y
        bounds.append((0, 2*math.pi))  # theta
    
    def obj(params):
        return objective(params, lam=lam)
    
    print(f"  Running DE: popsize={popsize}, maxiter={maxiter}, lambda={lam}")
    
    result = differential_evolution(
        obj,
        bounds,
        maxiter=maxiter,
        seed=seed,
        popsize=popsize,
        tol=1e-12,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True,
        x0=init_params,
        disp=False,
        workers=1,
    )
    
    return result


def params_to_solution(params):
    """Convert flat parameter array to solution.json format."""
    solution = []
    for i in range(N):
        solution.append({
            "x": round(float(params[i*3]), 6),
            "y": round(float(params[i*3 + 1]), 6),
            "theta": round(float(params[i*3 + 2]), 6),
        })
    return solution


def save_solution(params, path="solution.json"):
    """Save parameters as solution.json."""
    solution = params_to_solution(params)
    with open(path, "w") as f:
        json.dump(solution, f, indent=2)
    return path


def validate_with_scorer(path="solution.json"):
    """Run the official scorer and return the result."""
    from semicircle_packing.scoring import validate_and_score
    from semicircle_packing.geometry import Semicircle
    
    with open(path) as f:
        data = json.load(f)
    
    semicircles = []
    for item in data:
        semicircles.append(Semicircle(
            x=round(float(item["x"]), 6),
            y=round(float(item["y"]), 6),
            theta=round(float(item["theta"]), 6),
        ))
    
    return validate_and_score(semicircles)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    configs = {
        "grid_baseline": config_grid_baseline(),
        "paired_hex": config_paired_hex(),
        "compact_pairs": config_compact_pairs(),
        "ring": config_ring(),
        "random_42": config_random_seed(42),
        "random_99": config_random_seed(99),
        "random_777": config_random_seed(777),
    }
    
    best_score = float('inf')
    best_name = None
    best_params = None
    
    # Phase 1: Quick runs from each starting config
    print("=" * 60)
    print("Phase 1: Quick DE from multiple starting configs")
    print("=" * 60)
    
    for name, init in configs.items():
        print(f"\n--- Config: {name} ---")
        t0 = time.time()
        
        try:
            result = run_differential_evolution(
                init_params=init, 
                maxiter=200,
                seed=42,
                lam=200.0,
                popsize=20,
            )
            
            # Save and validate
            save_solution(result.x, "solution.json")
            val = validate_with_scorer("solution.json")
            elapsed = time.time() - t0
            
            if val.valid and val.score is not None:
                print(f"  VALID: score={val.score:.6f} (time={elapsed:.1f}s)")
                if val.score < best_score:
                    best_score = val.score
                    best_name = name
                    best_params = result.x.copy()
            else:
                # Report overlap penalty
                ovlp = overlap_penalty_fast(result.x)
                mec = mec_radius_fast(result.x)
                print(f"  INVALID: mec_est={mec:.4f}, overlap={ovlp:.4f} (time={elapsed:.1f}s)")
                if val.errors:
                    print(f"  Errors: {val.errors[:3]}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Phase 1 best: {best_name} = {best_score:.6f}")
    print(f"{'=' * 60}")
    
    if best_params is not None:
        # Phase 2: Longer run from best config
        print(f"\nPhase 2: Extended DE from best ({best_name})")
        
        result = run_differential_evolution(
            init_params=best_params,
            maxiter=2000,
            seed=123,
            lam=500.0,
            popsize=40,
        )
        
        save_solution(result.x, "solution.json")
        val = validate_with_scorer("solution.json")
        
        if val.valid and val.score is not None:
            print(f"  Phase 2 VALID: score={val.score:.6f}")
            if val.score < best_score:
                best_score = val.score
                best_params = result.x.copy()
                save_solution(best_params, "best_solution.json")
                print(f"  New best saved to best_solution.json")
        else:
            print(f"  Phase 2 INVALID")
            if val.errors:
                print(f"  Errors: {val.errors[:3]}")
        
        # Always save the phase 1 best
        save_solution(best_params, "best_solution.json")
    
    print(f"\n{'=' * 60}")
    print(f"FINAL BEST: {best_score:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
