"""
Fast semicircle packing optimizer v2.
Uses analytical overlap penalty (no point sampling).
Overlap between two semicircles approximated by penetration depth.
"""

import json
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

RADIUS = 1.0
N = 15


def overlap_penalty_analytical(params):
    """
    Fast overlap penalty using analytical geometry.
    For each pair: if centers within 2R, compute approximate penetration.
    A semicircle occupies the half-disk on the theta side of its center.
    """
    xs = params[0::3]
    ys = params[1::3]
    ts = params[2::3]
    
    penalty = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            
            # Two semicircles can only overlap if centers within 2R
            if dist_sq >= 4.0:  # (2*RADIUS)^2
                continue
            
            dist = math.sqrt(dist_sq)
            
            # For full circles, overlap starts when dist < 2R.
            # For semicircles, it depends on orientation.
            # Approximate: check if each center is "visible" from the other's
            # half-plane, then use circle-circle penetration scaled down.
            
            # Direction from i to j
            if dist < 1e-12:
                # Same center: always overlap unless facing away
                angle_ij = 0.0
            else:
                angle_ij = math.atan2(dy, dx)  # from j toward i
                
            # How much of semicircle i faces toward j?
            # Semicircle i occupies angles [ti - pi/2, ti + pi/2]
            # j is at angle (pi + angle_ij) from i
            dir_i_to_j = math.atan2(-dy, -dx)  # from i toward j
            diff_i = abs(math.atan2(math.sin(dir_i_to_j - ts[i]), math.cos(dir_i_to_j - ts[i])))
            
            dir_j_to_i = math.atan2(dy, dx)  # from j toward i
            diff_j = abs(math.atan2(math.sin(dir_j_to_i - ts[j]), math.cos(dir_j_to_i - ts[j])))
            
            # Both semicircles need to "face" each other for significant overlap
            # Scale penalty by how much they face each other
            face_i = max(0, 1.0 - diff_i / (math.pi/2))  # 1 = directly facing, 0 = perpendicular
            face_j = max(0, 1.0 - diff_j / (math.pi/2))
            
            # Circle-circle penetration
            penetration = max(0, 2*RADIUS - dist)
            
            # Scale by facing factors (both need to face each other)
            penalty += penetration * penetration * face_i * face_j
    
    return penalty


def mec_radius_analytical(params):
    """
    Compute MEC radius analytically.
    For each semicircle, the farthest point from origin-center is either:
    - The arc tip (center + R in theta direction)
    - One of the two flat-edge endpoints
    We compute these 3 points per semicircle = 45 points total,
    then find the 1-center (minimax) of all points.
    """
    points = []
    for i in range(N):
        x, y, t = params[i*3], params[i*3+1], params[i*3+2]
        # Arc tip
        points.append((x + RADIUS * math.cos(t), y + RADIUS * math.sin(t)))
        # Flat edge endpoints
        points.append((x + RADIUS * math.cos(t + math.pi/2), y + RADIUS * math.sin(t + math.pi/2)))
        points.append((x + RADIUS * math.cos(t - math.pi/2), y + RADIUS * math.sin(t - math.pi/2)))
    
    pts = np.array(points)
    
    # Iterative 1-center (Elzinga-Hearn style)
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    
    for _ in range(100):
        dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        idx = np.argmax(dists)
        r = dists[idx]
        # Move center toward farthest point
        fx, fy = pts[idx]
        step = 0.5 / (1 + _/10)
        cx = cx + step * (fx - cx)
        cy = cy + step * (fy - cy)
    
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    return np.max(dists)


def objective(params, lam=50.0):
    """MEC radius + penalty * lambda."""
    r = mec_radius_analytical(params)
    p = overlap_penalty_analytical(params)
    return r + lam * p


# ── Starting configs ──────────────────────────────────────────────

def config_7pairs_plus_1():
    """7 full disks hexagonal + 1 loose semicircle."""
    params = []
    # Center disk
    params.extend([0, 0, 0])
    params.extend([0, 0, math.pi])
    # Ring of 6
    for k in range(6):
        a = k * math.pi / 3
        x = 2.0 * math.cos(a)
        y = 2.0 * math.sin(a)
        params.extend([x, y, 0])
        params.extend([x, y, math.pi])
    # 15th: tuck into a gap
    params.extend([1.0, 1.73, math.pi/2])
    return np.array(params[:45])


def config_5pairs_5loose():
    """5 full disks + 5 loose semicircles in gaps."""
    params = []
    # Pentagon of 5 disks
    for k in range(5):
        a = 2 * math.pi * k / 5
        r = 1.8
        x = r * math.cos(a)
        y = r * math.sin(a)
        params.extend([x, y, a])         # outward facing
        params.extend([x, y, a + math.pi])  # inward facing
    # 5 loose semicircles filling center + gaps
    for k in range(5):
        a = 2 * math.pi * k / 5 + math.pi/5
        x = 0.5 * math.cos(a)
        y = 0.5 * math.sin(a)
        params.extend([x, y, a + math.pi])  # pointing inward
    return np.array(params[:45])


def config_tight_rows():
    """3 rows of semicircles, alternating orientations."""
    params = []
    # Row 1: 5 semicircles pointing up
    for i in range(5):
        x = (i - 2) * 1.1
        params.extend([x, -1.5, math.pi/2])
    # Row 2: 5 semicircles pointing down  
    for i in range(5):
        x = (i - 2) * 1.1
        params.extend([x, 0.0, -math.pi/2])
    # Row 3: 5 semicircles pointing up
    for i in range(5):
        x = (i - 2) * 1.1
        params.extend([x, 1.5, math.pi/2])
    return np.array(params[:45])


def config_random(seed):
    rng = np.random.default_rng(seed)
    params = []
    for _ in range(N):
        params.extend([rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5), rng.uniform(0, 2*math.pi)])
    return np.array(params)


# ── Run ───────────────────────────────────────────────────────────

def save_and_validate(params, path="solution.json"):
    solution = []
    for i in range(N):
        solution.append({
            "x": round(float(params[i*3]), 6),
            "y": round(float(params[i*3 + 1]), 6),
            "theta": round(float(params[i*3 + 2]), 6),
        })
    with open(path, "w") as f:
        json.dump(solution, f, indent=2)
    
    # Official validation
    from semicircle_packing.scoring import validate_and_score
    from semicircle_packing.geometry import Semicircle
    
    scs = [Semicircle(x=s["x"], y=s["y"], theta=s["theta"]) for s in solution]
    return validate_and_score(scs)


def run_one(name, init, maxiter=500, lam=50.0, popsize=25, seed=42):
    bounds = [(-4, 4), (-4, 4), (0, 2*math.pi)] * N
    
    t0 = time.time()
    print(f"\n  [{name}] DE maxiter={maxiter} pop={popsize} lam={lam}")
    
    result = differential_evolution(
        lambda p: objective(p, lam=lam),
        bounds,
        maxiter=maxiter,
        seed=seed,
        popsize=popsize,
        tol=1e-12,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=False,  # skip polish for speed
        x0=init,
        disp=False,
    )
    
    elapsed = time.time() - t0
    val = save_and_validate(result.x)
    
    if val.valid:
        print(f"  [{name}] VALID score={val.score:.6f} ({elapsed:.0f}s)")
    else:
        mec_est = mec_radius_analytical(result.x)
        ovlp = overlap_penalty_analytical(result.x)
        print(f"  [{name}] INVALID mec~{mec_est:.4f} ovlp={ovlp:.4f} ({elapsed:.0f}s)")
        n_errors = len(val.errors)
        print(f"  [{name}] {n_errors} overlap errors")
    
    return result.x, val, elapsed


def main():
    configs = {
        "7pairs": config_7pairs_plus_1(),
        "5pairs": config_5pairs_5loose(),
        "rows": config_tight_rows(),
    }
    # Add random seeds
    for s in [42, 99, 777, 1234, 9999]:
        configs[f"rand_{s}"] = config_random(s)
    
    best_score = float('inf')
    best_params = None
    best_name = None
    
    print("=" * 60)
    print("Phase 1: Quick screen (200 iters each)")
    print("=" * 60)
    
    for name, init in configs.items():
        try:
            params, val, elapsed = run_one(name, init, maxiter=200, lam=50.0, popsize=20)
            if val.valid and val.score < best_score:
                best_score = val.score
                best_params = params.copy()
                best_name = name
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
    
    print(f"\n{'=' * 60}")
    if best_params is not None:
        print(f"Phase 1 best: {best_name} = {best_score:.6f}")
    else:
        print("Phase 1: No valid solutions found!")
        # Use the last result anyway as a starting point
        print("Proceeding with increased penalty...")
    
    # Phase 2: longer run with higher penalty from best (or all if none valid)
    print(f"\nPhase 2: Extended runs")
    print("=" * 60)
    
    phase2_configs = {}
    if best_params is not None:
        phase2_configs["best_p1"] = best_params
    # Also try with much higher lambda from structured configs
    for name in ["7pairs", "5pairs", "rows"]:
        phase2_configs[f"{name}_high_lam"] = configs[name]
    
    for name, init in phase2_configs.items():
        try:
            params, val, elapsed = run_one(name, init, maxiter=1000, lam=500.0, popsize=30, seed=123)
            if val.valid and val.score < best_score:
                best_score = val.score
                best_params = params.copy()
                best_name = name
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"FINAL BEST: {best_score:.6f} ({best_name})")
    print("=" * 60)
    
    if best_params is not None:
        save_and_validate(best_params, "best_solution.json")
        save_and_validate(best_params, "solution.json")
        print("Saved to best_solution.json and solution.json")


if __name__ == "__main__":
    main()
