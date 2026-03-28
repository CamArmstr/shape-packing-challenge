"""
gjk_numba.py — GJK distance + EPA penetration depth for unit semicircles.

Replaces the phi-function approximation with exact convex distance.
Semicircles are convex (intersection of disk and half-plane), so GJK applies directly.

Key functions:
  - gjk_distance(xi, yi, ti, xj, yj, tj) → distance (positive=separated, 0=touching/overlapping)
  - epa_penetration(xi, yi, ti, xj, yj, tj) → penetration depth (positive=overlapping)
  - semicircle_gjk_signed_dist(...) → signed distance (positive=separated, negative=overlapping)
"""

import math
import numba as nb
import numpy as np


# ── Semicircle support mapping ───────────────────────────────────────────────

@nb.njit(cache=True)
def support_semicircle(cx, cy, ux, uy, dx, dy):
    """
    Support point of unit semicircle S(c, u) in direction d.
    
    S = {c + v : ||v|| <= 1, v · u >= 0}
    support(d) = c + argmax_{v in S} d · v
    
    If d · u >= 0: disk maximizer v = d_hat is in the arc half → support = c + d_hat
    Else: maximizer is on the diameter → support = c + t or c - t, whichever has larger d·v
    where t = (-uy, ux) is the tangent direction.
    """
    d_len = math.sqrt(dx * dx + dy * dy)
    if d_len < 1e-15:
        return cx + ux, cy + uy  # arbitrary: return arc apex
    
    d_hat_x = dx / d_len
    d_hat_y = dy / d_len
    
    # Check if disk maximizer is in the arc half
    dot_du = d_hat_x * ux + d_hat_y * uy
    
    if dot_du >= 0.0:
        # Maximizer v = d_hat is feasible (in arc half)
        return cx + d_hat_x, cy + d_hat_y
    else:
        # Maximizer must be a diameter endpoint
        tx = -uy
        ty = ux
        dot_dt = dx * tx + dy * ty  # unnormalized is fine for comparison
        if dot_dt >= 0.0:
            return cx + tx, cy + ty
        else:
            return cx - tx, cy - ty


@nb.njit(cache=True)
def support_minkowski_diff(cx1, cy1, ux1, uy1, cx2, cy2, ux2, uy2, dx, dy):
    """Support point of Minkowski difference A - B in direction d.
    = support_A(d) - support_B(-d)"""
    ax, ay = support_semicircle(cx1, cy1, ux1, uy1, dx, dy)
    bx, by = support_semicircle(cx2, cy2, ux2, uy2, -dx, -dy)
    return ax - bx, ay - by


# ── 2D GJK ───────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def triple_product_2d(ax, ay, bx, by, cx, cy):
    """(A × B) × C in 2D, returned as 2D vector.
    = B(A·C) - A(B·C) but in 2D with z-cross."""
    # A × B = ax*by - ay*bx (scalar z-component)
    z = ax * by - ay * bx
    # result = z × C = (-z*cy, z*cx)
    return -z * cy, z * cx


@nb.njit(cache=True)
def gjk_test(cx1, cy1, ux1, uy1, cx2, cy2, ux2, uy2):
    """
    2D GJK: determine if two semicircles overlap.
    
    Returns:
      (overlapping, dist_sq, closest_on_A_x, closest_on_A_y, closest_on_B_x, closest_on_B_y)
    
    If overlapping=True: dist_sq is meaningless; use EPA for penetration depth.
    If overlapping=False: dist_sq is the squared distance between closest points.
    """
    MAX_ITER = 32
    
    # Initial direction: center to center
    dx = cx2 - cx1
    dy = cy2 - cy1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        dx = 1.0  # arbitrary
    
    # First support point
    sx, sy = support_minkowski_diff(cx1, cy1, ux1, uy1, cx2, cy2, ux2, uy2, dx, dy)
    
    # Check if origin is beyond first support in search direction
    if sx * dx + sy * dy < 0.0:
        # Origin is not past the first support → no intersection
        # Distance to origin from this point
        d2 = sx * sx + sy * sy
        return False, d2, 0.0, 0.0, 0.0, 0.0
    
    # Search in opposite direction
    dx = -sx
    dy = -sy
    
    # Second support point
    s2x, s2y = support_minkowski_diff(cx1, cy1, ux1, uy1, cx2, cy2, ux2, uy2, dx, dy)
    
    if s2x * dx + s2y * dy < 0.0:
        # Origin not past second support → closest point is on the line segment
        # Project origin onto segment (sx,sy)-(s2x,s2y)
        ex = s2x - sx
        ey = s2y - sy
        e_len2 = ex * ex + ey * ey
        if e_len2 < 1e-20:
            d2 = sx * sx + sy * sy
            return False, d2, 0.0, 0.0, 0.0, 0.0
        t = -(sx * ex + sy * ey) / e_len2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        px = sx + t * ex
        py = sy + t * ey
        d2 = px * px + py * py
        return False, d2, 0.0, 0.0, 0.0, 0.0
    
    # We have a line segment. Now iteratively build a simplex.
    # Simplex vertices
    s0x, s0y = sx, sy
    s1x, s1y = s2x, s2y
    
    for _iter in range(MAX_ITER):
        # Find which Voronoi region of the line segment the origin is in
        # and determine new search direction
        
        # Edge from s0 to s1
        e01x = s1x - s0x
        e01y = s1y - s0y
        
        # Normal toward origin: perpendicular to edge, pointing toward origin
        # Use triple product: (s1-s0) × (0-s0) gives z, then cross with edge
        # Simpler: two candidates for normal
        # n = (-e01y, e01x) or (e01y, -e01x)
        # Pick the one that points toward origin (dot with -s0 should be positive)
        n1x = -e01y
        n1y = e01x
        if n1x * (-s0x) + n1y * (-s0y) >= 0:
            dx = n1x
            dy = n1y
        else:
            dx = e01y
            dy = -e01x
        
        d_len2 = dx * dx + dy * dy
        if d_len2 < 1e-20:
            # Origin is on the edge → touching
            return True, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # New support point
        s2x, s2y = support_minkowski_diff(cx1, cy1, ux1, uy1, cx2, cy2, ux2, uy2, dx, dy)
        
        if s2x * dx + s2y * dy < 0.0:
            # Origin not past new support in search direction
            # Closest point is on edge s0-s1
            e_len2 = e01x * e01x + e01y * e01y
            if e_len2 < 1e-20:
                d2 = s0x * s0x + s0y * s0y
                return False, d2, 0.0, 0.0, 0.0, 0.0
            t = -(s0x * e01x + s0y * e01y) / e_len2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            px = s0x + t * e01x
            py = s0y + t * e01y
            d2 = px * px + py * py
            return False, d2, 0.0, 0.0, 0.0, 0.0
        
        # Check if origin is inside the triangle (s0, s1, s2)
        # Using cross products
        cross_01_02 = (s1x - s0x) * (s2y - s0y) - (s1y - s0y) * (s2x - s0x)
        cross_01_0O = (s1x - s0x) * (-s0y) - (s1y - s0y) * (-s0x)
        cross_02_0O = (s2x - s0x) * (-s0y) - (s2y - s0y) * (-s0x)
        
        # Same-sign checks for barycentric coordinates
        if abs(cross_01_02) < 1e-15:
            # Degenerate triangle — origin is on/near the line
            return True, 0.0, 0.0, 0.0, 0.0, 0.0
        
        u = cross_02_0O / cross_01_02
        v = -cross_01_0O / cross_01_02
        
        if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
            # Origin is inside triangle → intersection
            return True, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Origin is outside triangle. Determine which edge to keep.
        # The new search direction should be toward the origin from the closest edge.
        # Keep the edge closest to origin among: (s0,s2) and (s1,s2)
        
        # Check edge s0-s2
        e02x = s2x - s0x
        e02y = s2y - s0y
        dot_s0_e02 = -(s0x * e02x + s0y * e02y)
        len2_e02 = e02x * e02x + e02y * e02y
        
        # Check edge s1-s2
        e12x = s2x - s1x
        e12y = s2y - s1y
        dot_s1_e12 = -(s1x * e12x + s1y * e12y)
        len2_e12 = e12x * e12x + e12y * e12y
        
        # Choose edge whose normal points more toward origin
        # Test: is origin on the s2 side of edge s0-s1?
        # If v < 0: origin is on the s0-s2 side
        # If u < 0: origin is on the s1-s2 side
        if v < 0.0:
            # Keep edge s0-s2
            s1x, s1y = s2x, s2y
        else:
            # Keep edge s1-s2
            s0x, s0y = s2x, s2y
    
    # Max iterations: assume overlapping
    return True, 0.0, 0.0, 0.0, 0.0, 0.0


# ── 2D EPA for penetration depth ─────────────────────────────────────────────

@nb.njit(cache=True)
def epa_penetration(cx1, cy1, ux1, uy1, cx2, cy2, ux2, uy2):
    """
    EPA: find penetration depth after GJK detects overlap.
    Returns penetration depth (positive value).
    
    Starts from a triangle containing the origin (constructed here),
    then iteratively expands toward the Minkowski difference boundary.
    """
    MAX_ITER = 32
    MAX_POLY = 64
    EPSILON = 1e-6
    
    # Build initial simplex containing origin
    # Use 3 directions spread 120° apart
    dirs = np.empty((3, 2))
    for k in range(3):
        a = k * 2.0 * math.pi / 3.0
        dirs[k, 0] = math.cos(a)
        dirs[k, 1] = math.sin(a)
    
    poly_x = np.empty(MAX_POLY)
    poly_y = np.empty(MAX_POLY)
    n_pts = 0
    
    for k in range(3):
        sx, sy = support_minkowski_diff(
            cx1, cy1, ux1, uy1, cx2, cy2, ux2, uy2, dirs[k, 0], dirs[k, 1])
        poly_x[n_pts] = sx
        poly_y[n_pts] = sy
        n_pts += 1
    
    # Ensure CCW winding
    cross = (poly_x[1] - poly_x[0]) * (poly_y[2] - poly_y[0]) - \
            (poly_y[1] - poly_y[0]) * (poly_x[2] - poly_x[0])
    if cross < 0:
        poly_x[1], poly_x[2] = poly_x[2], poly_x[1]
        poly_y[1], poly_y[2] = poly_y[2], poly_y[1]
    
    for _iter in range(MAX_ITER):
        # Find closest edge to origin
        min_dist = 1e18
        min_idx = 0
        min_nx = 0.0
        min_ny = 0.0
        
        for i in range(n_pts):
            j = (i + 1) % n_pts
            ex = poly_x[j] - poly_x[i]
            ey = poly_y[j] - poly_y[i]
            
            # Outward normal (CCW winding → left normal is outward)
            nx = -ey
            ny = ex
            n_len = math.sqrt(nx * nx + ny * ny)
            if n_len < 1e-15:
                continue
            nx /= n_len
            ny /= n_len
            
            # Distance from origin to this edge along normal
            dist = nx * poly_x[i] + ny * poly_y[i]
            
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                min_nx = nx
                min_ny = ny
        
        # Get new support point in the direction of closest edge normal
        sx, sy = support_minkowski_diff(
            cx1, cy1, ux1, uy1, cx2, cy2, ux2, uy2, min_nx, min_ny)
        
        d_new = min_nx * sx + min_ny * sy
        
        if d_new - min_dist < EPSILON or n_pts >= MAX_POLY - 1:
            # Converged: penetration depth is min_dist
            return max(0.0, min_dist)
        
        # Insert new point between min_idx and min_idx+1
        insert_at = min_idx + 1
        # Shift everything after insert_at
        for i in range(n_pts, insert_at, -1):
            poly_x[i] = poly_x[i - 1]
            poly_y[i] = poly_y[i - 1]
        poly_x[insert_at] = sx
        poly_y[insert_at] = sy
        n_pts += 1
    
    return max(0.0, min_dist)


# ── Combined signed distance ─────────────────────────────────────────────────

@nb.njit(cache=True)
def semicircle_gjk_signed_dist(xi, yi, ti, xj, yj, tj):
    """
    Exact signed distance between two unit semicircles using GJK + EPA.
    
    Returns:
      > 0: separated (distance between closest points)
      = 0: touching
      < 0: overlapping (negative penetration depth)
    """
    # Quick pre-filter: if disk centers are >= 2 apart, definitely separated
    dx = xj - xi
    dy = yj - yi
    dc = math.sqrt(dx * dx + dy * dy)
    if dc >= 2.0:
        return dc - 2.0  # conservative lower bound on separation
    
    uxi = math.cos(ti)
    uyi = math.sin(ti)
    uxj = math.cos(tj)
    uyj = math.sin(tj)
    
    overlapping, dist_sq, _, _, _, _ = gjk_test(xi, yi, uxi, uyi, xj, yj, uxj, uyj)
    
    if not overlapping:
        return math.sqrt(max(0.0, dist_sq))
    else:
        depth = epa_penetration(xi, yi, uxi, uyi, xj, yj, uxj, uyj)
        return -depth


# ── Overlap energy using GJK (drop-in replacement for phi-based) ─────────────

@nb.njit(cache=True)
def overlap_energy_gjk(xs, ys, ts):
    """Sum of max(0, penetration_depth)^2 over all pairs."""
    n = xs.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = semicircle_gjk_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < 0.0:
                energy += d * d
    return energy


@nb.njit(cache=True)
def overlap_energy_gjk_for_idx(xs, ys, ts, idx):
    """Overlap energy contribution from semicircle idx against all others (GJK)."""
    n = xs.shape[0]
    energy = 0.0
    for j in range(n):
        if j == idx:
            continue
        d = semicircle_gjk_signed_dist(xs[idx], ys[idx], ts[idx], xs[j], ys[j], ts[j])
        if d < 0.0:
            energy += d * d
    return energy


# ── Benchmark and validation ─────────────────────────────────────────────────

def benchmark():
    import time
    import json
    
    sol_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_solution.json")
    with open(sol_path) as f:
        sol = json.load(f)
    xs = np.array([s["x"] for s in sol], dtype=np.float64)
    ys = np.array([s["y"] for s in sol], dtype=np.float64)
    ts = np.array([s["theta"] for s in sol], dtype=np.float64)
    
    # Warm up
    semicircle_gjk_signed_dist(xs[0], ys[0], ts[0], xs[1], ys[1], ts[1])
    overlap_energy_gjk(xs, ys, ts)
    
    # Validate: all pairs should be non-negative for a valid solution
    print("Pair distances (should all be >= 0):")
    neg_count = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            d = semicircle_gjk_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < -1e-6:
                neg_count += 1
                print(f"  NEGATIVE: ({i},{j}) d={d:.8f}")
    print(f"  Negative pairs: {neg_count}/105")
    print(f"  Overlap energy: {overlap_energy_gjk(xs, ys, ts):.2e}")
    
    # Cross-validate against phi
    from sa_v2 import phi_pair_nb
    disagree = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            gjk_d = semicircle_gjk_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            phi_d = phi_pair_nb(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            gjk_sign = 1 if gjk_d >= 0 else -1
            phi_sign = 1 if phi_d >= 0 else -1
            if gjk_sign != phi_sign:
                disagree += 1
                print(f"  DISAGREE ({i},{j}): gjk={gjk_d:.6f} phi={phi_d:.6f}")
    print(f"  Sign disagreements with phi: {disagree}/105")
    
    # Speed benchmark
    n_iter = 10000
    t0 = time.perf_counter()
    for _ in range(n_iter):
        overlap_energy_gjk(xs, ys, ts)
    elapsed = time.perf_counter() - t0
    rate = n_iter / elapsed
    print(f"\nBenchmark: {n_iter} full evals in {elapsed:.2f}s = {rate:,.0f} evals/sec")
    print(f"  = {rate * 105:,.0f} pair-checks/sec")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    benchmark()
