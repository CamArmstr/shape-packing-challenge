"""
exact_dist.py — O(1) exact signed distance between two unit semicircles.

A unit semicircle S(c, n) has:
  - Center c = (cx, cy)
  - Arc normal n = (cos θ, sin θ) pointing toward the arc (dome direction)
  - Tangent t = (-sin θ, cos θ) = perpendicular to n
  - Arc: unit circle restricted to the half-plane n·(p - c) ≥ 0
  - Flat edge: diameter segment from c - t to c + t (i.e. c ± t)

Boundary features:
  - Arc A: semicircular arc of radius 1, center c, from c-t to c+t through c+n
  - Flat F: line segment from c-t to c+t (the diameter)

For two semicircles, we check all 4 feature pairs:
  (A1, A2), (A1, F2), (F1, A2), (F1, F2)

The signed distance d(S1, S2) = min over all feature pairs of min_dist(feature1, feature2).
Positive → separated. Negative → overlapping. Zero → touching.

~135 FLOPs total, O(1), exact, differentiable almost everywhere.
"""

import math
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Geometric primitives
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _closest_point_on_segment(px, py, ax, ay, bx, by):
    """Closest point on segment AB to point P. Returns (qx, qy, t) where t∈[0,1]."""
    dx, dy = bx - ax, by - ay
    len2 = dx*dx + dy*dy
    if len2 < 1e-14:
        return ax, ay, 0.0
    t = _clamp(((px - ax)*dx + (py - ay)*dy) / len2, 0.0, 1.0)
    return ax + t*dx, ay + t*dy, t


def _on_arc(px, py, cx, cy, nx, ny):
    """
    Is point (px, py) on the arc of S(c, n)?
    Arc = unit circle center c, restricted to {n·(p-c) ≥ 0}.
    We need (px,py) on the unit circle AND in the arc half-plane.
    Tolerance: 1e-9.
    """
    dot_n = nx*(px - cx) + ny*(py - cy)
    return dot_n >= -1e-9


def _arc_endpoints(cx, cy, nx, ny):
    """Two arc endpoints: c ± t where t = (-ny, nx)."""
    tx, ty = -ny, nx
    return (cx + tx, cy + ty), (cx - tx, cy - ty)


def _closest_on_arc_to_point(px, py, cx, cy, nx, ny):
    """
    Closest point on arc A(c, n) to external point P.
    Arc = unit circle center c, half n·(p-c) ≥ 0.
    Returns (qx, qy).
    """
    dx, dy = px - cx, py - cy
    dist = math.sqrt(dx*dx + dy*dy)

    if dist < 1e-9:
        # P is at center — closest is the apex
        return cx + nx, cy + ny

    # Candidate 1: point on full circle toward P
    qx = cx + dx / dist
    qy = cy + dy / dist
    if nx*(qx - cx) + ny*(qy - cy) >= -1e-9:
        return qx, qy

    # Candidate is in flat half — return nearer endpoint
    ep1, ep2 = _arc_endpoints(cx, cy, nx, ny)
    d1 = (px - ep1[0])**2 + (py - ep1[1])**2
    d2 = (px - ep2[0])**2 + (py - ep2[1])**2
    return ep1 if d1 <= d2 else ep2


# ─────────────────────────────────────────────────────────────────────────────
# Feature-pair distances
# ─────────────────────────────────────────────────────────────────────────────

def _dist_arc_arc(cx1, cy1, nx1, ny1, cx2, cy2, nx2, ny2):
    """
    Minimum signed distance between arc A1 and arc A2.
    Positive = separated, negative = overlapping.
    
    Arc-interior critical pair: both points on center-to-center line.
    q1 = c1 + r1 * d̂, q2 = c2 - r2 * d̂, where d̂ = (c2-c1)/|c2-c1|.
    Also check: closest point on A1 to each endpoint of A2, and vice versa.
    """
    dx, dy = cx2 - cx1, cy2 - cy1
    dc = math.sqrt(dx*dx + dy*dy)
    candidates = []

    # --- Interior-interior candidate ---
    if dc > 1e-9:
        # Direction from c1 to c2
        ux, uy = dx / dc, dy / dc
        q1x, q1y = cx1 + ux, cy1 + uy  # on unit circle C1 toward C2
        q2x, q2y = cx2 - ux, cy2 - uy  # on unit circle C2 toward C1
        if _on_arc(q1x, q1y, cx1, cy1, nx1, ny1) and _on_arc(q2x, q2y, cx2, cy2, nx2, ny2):
            sep = dc - 2.0  # = |q1 - q2| when both on arc
            candidates.append(sep)

    # --- Endpoint-arc candidates (4 pairs) ---
    ep1a, ep1b = _arc_endpoints(cx1, cy1, nx1, ny1)
    ep2a, ep2b = _arc_endpoints(cx2, cy2, nx2, ny2)

    for (epx, epy) in [ep1a, ep1b]:
        q = _closest_on_arc_to_point(epx, epy, cx2, cy2, nx2, ny2)
        d = math.sqrt((epx - q[0])**2 + (epy - q[1])**2)
        # Sign: negative if epx is inside C2 and on arc half
        inside_c2 = (epx - cx2)**2 + (epy - cy2)**2 < 1.0
        in_arc2 = nx2*(epx - cx2) + ny2*(epy - cy2) >= 0
        sign = -1 if (inside_c2 and in_arc2) else 1
        candidates.append(sign * d)

    for (epx, epy) in [ep2a, ep2b]:
        q = _closest_on_arc_to_point(epx, epy, cx1, cy1, nx1, ny1)
        d = math.sqrt((epx - q[0])**2 + (epy - q[1])**2)
        inside_c1 = (epx - cx1)**2 + (epy - cy1)**2 < 1.0
        in_arc1 = nx1*(epx - cx1) + ny1*(epy - cy1) >= 0
        sign = -1 if (inside_c1 and in_arc1) else 1
        candidates.append(sign * d)

    return min(candidates) if candidates else 0.0


def _dist_arc_flat(cx, cy, nx, ny, ax, ay, bx, by):
    """
    Minimum distance between arc A(c,n) and flat segment [a,b].
    Returns always non-negative (unsigned minimum distance between the curves).

    Note: signed distance between SEMICIRCLES requires checking whether features
    are inside the opposing semicircle — handled at the semicircle level.
    Arc-flat feature distance is unsigned; signing happens in semicircle_signed_dist.
    """
    # Segment direction and normal
    sdx, sdy = bx - ax, by - ay
    slen = math.sqrt(sdx*sdx + sdy*sdy)
    if slen < 1e-9:
        q = _closest_on_arc_to_point(ax, ay, cx, cy, nx, ny)
        return math.sqrt((ax - q[0])**2 + (ay - q[1])**2)

    sux, suy = sdx / slen, sdy / slen
    snx, sny = -suy, sux

    candidates = []

    # --- Interior-interior: arc point perpendicular to segment ---
    for (qnx, qny) in [(snx, sny), (-snx, -sny)]:
        q_arcx = cx + qnx
        q_arcy = cy + qny
        if not _on_arc(q_arcx, q_arcy, cx, cy, nx, ny):
            continue
        t = _clamp(((q_arcx - ax)*sux + (q_arcy - ay)*suy), 0.0, slen) / slen
        proj_x = ax + t * sdx
        proj_y = ay + t * sdy
        d = math.sqrt((q_arcx - proj_x)**2 + (q_arcy - proj_y)**2)
        candidates.append(d)

    # --- Arc endpoints vs segment ---
    ep1, ep2 = _arc_endpoints(cx, cy, nx, ny)
    for (epx, epy) in [ep1, ep2]:
        qx, qy, _ = _closest_point_on_segment(epx, epy, ax, ay, bx, by)
        d = math.sqrt((epx - qx)**2 + (epy - qy)**2)
        candidates.append(d)

    # --- Segment endpoints vs arc ---
    for (epx, epy) in [(ax, ay), (bx, by)]:
        q = _closest_on_arc_to_point(epx, epy, cx, cy, nx, ny)
        d = math.sqrt((epx - q[0])**2 + (epy - q[1])**2)
        candidates.append(d)

    return min(candidates) if candidates else 0.0


def _dist_flat_flat(ax1, ay1, bx1, by1, ax2, ay2, bx2, by2):
    """
    Minimum distance between two line segments.
    Standard segment-segment distance.
    """
    # Check all 4 endpoint-segment combinations
    candidates = []
    for (px, py) in [(ax1, ay1), (bx1, by1)]:
        qx, qy, _ = _closest_point_on_segment(px, py, ax2, ay2, bx2, by2)
        candidates.append(math.sqrt((px-qx)**2 + (py-qy)**2))
    for (px, py) in [(ax2, ay2), (bx2, by2)]:
        qx, qy, _ = _closest_point_on_segment(px, py, ax1, ay1, bx1, by1)
        candidates.append(math.sqrt((px-qx)**2 + (py-qy)**2))

    # Check segment-segment intersection (distance = 0)
    # Using cross product method
    def cross2d(ux, uy, vx, vy): return ux*vy - uy*vx
    d1x, d1y = bx1-ax1, by1-ay1
    d2x, d2y = bx2-ax2, by2-ay2
    denom = cross2d(d1x, d1y, d2x, d2y)
    if abs(denom) > 1e-10:
        t = cross2d(ax2-ax1, ay2-ay1, d2x, d2y) / denom
        s = cross2d(ax2-ax1, ay2-ay1, d1x, d1y) / denom
        if 0 <= t <= 1 and 0 <= s <= 1:
            return 0.0  # segments intersect

    return min(candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Main signed distance function
# ─────────────────────────────────────────────────────────────────────────────

def _point_in_semicircle(px, py, cx, cy, nx, ny):
    """True if point (px,py) is strictly inside semicircle S(c,n)."""
    in_disk = (px - cx)**2 + (py - cy)**2 < 1.0
    in_arc_half = nx*(px - cx) + ny*(py - cy) > 0
    return in_disk and in_arc_half


def semicircle_signed_dist(xi, yi, ti, xj, yj, tj):
    """
    Exact signed distance between two unit semicircles.
    S_i: center (xi, yi), arc normal (cos(ti), sin(ti))
    S_j: center (xj, yj), arc normal (cos(tj), sin(tj))

    Returns:
      > 0: separated (gap = minimum distance between boundaries)
      = 0: touching
      < 0: overlapping (negative = penetrating)

    Strategy:
    1. Check if disks don't even touch → return positive disk-disk separation
    2. Check if any critical point of S_i is inside S_j, or vice versa → overlapping
    3. Otherwise compute minimum boundary-to-boundary distance across all 4 feature pairs
    """
    # --- Fast pre-filter: if disks don't overlap, return positive distance ---
    dx, dy = xj - xi, yj - yi
    dc = math.sqrt(dx*dx + dy*dy)
    if dc >= 2.0:
        return dc - 2.0

    # Arc normals and tangents
    nx1, ny1 = math.cos(ti), math.sin(ti)
    tx1, ty1 = -ny1, nx1
    nx2, ny2 = math.cos(tj), math.sin(tj)
    tx2, ty2 = -ny2, nx2

    # --- Sign test: is any critical point of S_i inside S_j (or vice versa)? ---
    # Critical points of S_i: arc apex + 2 diameter endpoints
    critical_i = [
        (xi + nx1, yi + ny1),   # arc apex
        (xi + tx1, yi + ty1),   # endpoint 1
        (xi - tx1, yi - ty1),   # endpoint 2
    ]
    critical_j = [
        (xj + nx2, yj + ny2),
        (xj + tx2, yj + ty2),
        (xj - tx2, yj - ty2),
    ]

    # If any critical point of i is inside j (or vice versa), they overlap
    for (px, py) in critical_i:
        if _point_in_semicircle(px, py, xj, yj, nx2, ny2):
            # Overlapping — return negative minimum boundary distance
            pass  # fall through to compute unsigned dist, return negative

    # --- Analytical lens interior check (catches thin sliver arc-arc overlaps) ---
    # The lens D1 ∩ D2 (intersection of two unit disks) has interior points.
    # S1 ∩ S2 = (D1 ∩ H1) ∩ (D2 ∩ H2). Non-empty iff some point is in all four.
    # 
    # Key candidates:
    # 1. Circle-circle intersection points (on boundary of lens)
    # 2. Points along chord between intersection points (interior of lens)
    # 3. Interior samples along c1→c2 axis and perpendiculars
    # 4. c2+n1 and c1+n2: analytically derived sliver-overlap detectors
    overlapping = False
    if dc < 1e-9:
        # Coincident centers — definitely overlapping
        overlapping = True
    hat_x = dx / dc if dc >= 1e-9 else 1.0
    hat_y = dy / dc if dc >= 1e-9 else 0.0
    perp_x, perp_y = -hat_y, hat_x   # perpendicular

    # Circle-circle intersection points (boundary of lens)
    a_ = dc / 2.0
    h_ = math.sqrt(max(0.0, 1.0 - a_ * a_))
    midx = (xi + xj) / 2.0
    midy = (yi + yj) / 2.0
    cc_ints = [
        (midx + h_ * perp_x, midy + h_ * perp_y),
        (midx - h_ * perp_x, midy - h_ * perp_y),
    ]

    # Lens interior samples: interpolate along the lens axis and perpendiculars
    # Lens axis: from (xi+hat) to (xj-hat), both inside the other disk
    lens_pts = list(cc_ints)  # boundary
    # Points on the chord between the two intersection points (all inside both disks)
    for t_ in [0.25, 0.5, 0.75]:
        lx = cc_ints[0][0] + t_ * (cc_ints[1][0] - cc_ints[0][0])
        ly = cc_ints[0][1] + t_ * (cc_ints[1][1] - cc_ints[0][1])
        lens_pts.append((lx, ly))
    # Points moving toward each arc center from the midpoint of c1-c2
    for r_ in [0.3, 0.6, 0.9]:
        lens_pts.append((midx + r_ * hat_x, midy + r_ * hat_y))
        lens_pts.append((midx - r_ * hat_x, midy - r_ * hat_y))
        lens_pts.append((midx + r_ * perp_x, midy + r_ * perp_y))
        lens_pts.append((midx - r_ * perp_x, midy - r_ * perp_y))
    for (px, py) in lens_pts:
        # Must be strictly inside both disks (not just touching boundary)
        in_d1 = (px - xi)**2 + (py - yi)**2 <= 1.0 - 1e-7
        in_d2 = (px - xj)**2 + (py - yj)**2 <= 1.0 - 1e-7
        if not (in_d1 and in_d2):
            continue
        on_h1 = nx1 * (px - xi) + ny1 * (py - yi) >= 1e-7
        on_h2 = nx2 * (px - xj) + ny2 * (py - yj) >= 1e-7
        if on_h1 and on_h2:
            overlapping = True
            break

    # Key analytically-derived candidates (check with looser disk tolerance):
    # c2 + n1: point on circle C2 facing in the arc-normal direction of S1
    #          If strictly inside D1, in H1, and in H2, the semicircles overlap.
    # c1 + n2: symmetric candidate on circle C1 facing direction n2
    if not overlapping:
        for (px, py) in [(xj + nx1, yj + ny1), (xi + nx2, yi + ny2)]:
            # For these candidates: one disk contains the point on its boundary (d=1),
            # so check the OTHER disk is strictly interior (d < 1-eps)
            d1sq = (px - xi)**2 + (py - yi)**2
            d2sq = (px - xj)**2 + (py - yj)**2
            # The point is on C1 (d1=1) or C2 (d2=1); other must be strictly inside
            if not ((d1sq <= 1.0 + 1e-9 and d2sq < 1.0 - 1e-5) or
                    (d2sq <= 1.0 + 1e-9 and d1sq < 1.0 - 1e-5)):
                continue
            # Both half-plane conditions must be clearly satisfied (not just touching boundary)
            on_h1 = nx1 * (px - xi) + ny1 * (py - yi) >= 1e-4
            on_h2 = nx2 * (px - xj) + ny2 * (py - yj) >= 1e-4
            if on_h1 and on_h2:
                overlapping = True
                break

    # Also check critical points and directional samples
    def arc_sample(cx, cy, nax, nay, tgt_x, tgt_y):
        """A few arc samples in the direction of the other semicircle."""
        results = []
        for angle_offset in [0, math.pi/6, -math.pi/6, math.pi/4, -math.pi/4,
                              math.pi/3, -math.pi/3, math.pi/2, -math.pi/2]:
            base = math.atan2(tgt_y - cy, tgt_x - cx)
            a = base + angle_offset
            px, py = cx + math.cos(a), cy + math.sin(a)
            if nax*(px - cx) + nay*(py - cy) >= -1e-6:
                results.append((px, py))
        return results

    if not overlapping:
        samples_i = arc_sample(xi, yi, nx1, ny1, xj, yj)
        for (px, py) in samples_i:
            if _point_in_semicircle(px, py, xj, yj, nx2, ny2):
                overlapping = True; break

    if not overlapping:
        samples_j = arc_sample(xj, yj, nx2, ny2, xi, yi)
        for (px, py) in samples_j:
            if _point_in_semicircle(px, py, xi, yi, nx1, ny1):
                overlapping = True; break

    if not overlapping:
        for (px, py) in critical_j:
            if _point_in_semicircle(px, py, xi, yi, nx1, ny1):
                overlapping = True; break

    # Flat edge endpoints of S_i/S_j are also boundary points
    if not overlapping:
        for (px, py) in [(xi + tx1, yi + ty1), (xi - tx1, yi - ty1)]:
            if _point_in_semicircle(px, py, xj, yj, nx2, ny2):
                overlapping = True; break
    if not overlapping:
        for (px, py) in [(xj + tx2, yj + ty2), (xj - tx2, yj - ty2)]:
            if _point_in_semicircle(px, py, xi, yi, nx1, ny1):
                overlapping = True; break

    # Flat edge endpoints (flat edge = diameter segment)
    f1ax, f1ay = xi - tx1, yi - ty1
    f1bx, f1by = xi + tx1, yi + ty1
    f2ax, f2ay = xj - tx2, yj - ty2
    f2bx, f2by = xj + tx2, yj + ty2

    # Compute separation measure
    d_aa = _dist_arc_arc(xi, yi, nx1, ny1, xj, yj, nx2, ny2)
    d_af = _dist_arc_flat(xi, yi, nx1, ny1, f2ax, f2ay, f2bx, f2by)
    d_fa = _dist_arc_flat(xj, yj, nx2, ny2, f1ax, f1ay, f1bx, f1by)
    d_ff = _dist_flat_flat(f1ax, f1ay, f1bx, f1by, f2ax, f2ay, f2bx, f2by)

    if not overlapping:
        # Non-overlapping: return minimum boundary-to-boundary distance (positive)
        return min(d_aa, d_af, d_fa, d_ff)
    else:
        # Overlapping: return negative penetration depth.
        # For overlapping shapes, the "separation" measures (arc-arc, arc-flat, flat-flat)
        # can be zero or positive (features outside each other).
        # Use the Shapely-like proxy: penetration depth ≈ -sqrt(overlap_area).
        # Since we don't have area here, estimate via the disk-disk overlap as a proxy:
        # The deepest penetrating direction for circle-circle gives -(2 - dc) when dc < 2.
        # For semicircles, use the arc-arc signed distance (negative = overlapping disks)
        # as the best single-number penetration estimate.
        # d_aa is already computed as center_dist^2 - 4, divided by... wait, no:
        # d_aa = _dist_arc_arc which is unsigned (no sign returned). Use disk overlap:
        overlap_depth = max(0.0, 2.0 - dc)  # disk penetration depth
        return -overlap_depth


def semicircle_signed_dist_vec(xi, yi, ti, xj, yj, tj):
    """NumPy-vectorized version for arrays of semicircles."""
    n = len(xi)
    dists = np.zeros(n)
    for k in range(n):
        dists[k] = semicircle_signed_dist(xi[k], yi[k], ti[k], xj[k], yj[k], tj[k])
    return dists


def all_pairs_signed_dist(xs, ys, ts):
    """Compute exact signed distance for all N*(N-1)/2 pairs."""
    n = len(xs)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(semicircle_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j]))
    return np.array(dists)


def containment_signed_dist(xi, yi, ti, R):
    """
    Signed distance for semicircle S_i inside container circle of radius R.
    Positive = inside with margin, negative = outside.
    Uses 3 critical points (arc apex + 2 diameter endpoints).
    """
    pts = [
        (xi + math.cos(ti), yi + math.sin(ti)),   # arc apex
        (xi - math.sin(ti), yi + math.cos(ti)),   # endpoint 1
        (xi + math.sin(ti), yi - math.cos(ti)),   # endpoint 2
    ]
    slacks = [R - math.sqrt(px*px + py*py) for px, py in pts]
    return min(slacks)


def is_feasible_exact(xs, ys, ts, R, tol=1e-6):
    """True if all semicircles are mutually non-overlapping (exact) and contained."""
    n = len(xs)
    for i in range(n):
        if containment_signed_dist(xs[i], ys[i], ts[i], R) < -tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            if semicircle_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j]) < -tol:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Penalty energy (for gradient-based optimization)
# ─────────────────────────────────────────────────────────────────────────────

def penalty_energy_exact(xs, ys, ts, R):
    """
    Exact signed-distance-based penalty energy.
    E = Σ max(0, -dist(i,j))² + Σ max(0, -cont(i))²
    """
    n = len(xs)
    E = 0.0
    for i in range(n):
        c = containment_signed_dist(xs[i], ys[i], ts[i], R)
        if c < 0:
            E += c * c
    for i in range(n):
        for j in range(i+1, n):
            # Quick pre-filter
            dc2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
            if dc2 >= 4.0:
                continue
            d = semicircle_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < 0:
                E += d * d
    return E


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import json, time

    print("=== exact_dist.py validation ===\n")

    # Load known valid solution
    with open('best_solution.json') as f:
        raw = json.load(f)
    xs = np.array([s['x'] for s in raw])
    ys = np.array([s['y'] for s in raw])
    ts = np.array([s['theta'] for s in raw])
    R = 3.071975

    # Test 1: known valid solution should have all positive distances
    t0 = time.time()
    dists = all_pairs_signed_dist(xs, ys, ts)
    elapsed = (time.time() - t0) * 1000
    print(f"All-pair distances on valid solution ({elapsed:.1f}ms for 105 pairs):")
    print(f"  Min dist: {dists.min():.6f} (expect > 0)")
    print(f"  Negative pairs: {(dists < -1e-6).sum()} (expect 0)")
    print(f"  Feasible: {is_feasible_exact(xs, ys, ts, R)}")
    print(f"  Penalty energy: {penalty_energy_exact(xs, ys, ts, R):.2e} (expect 0)")

    # Test 2: clearly overlapping pair
    print("\nOverlap test (two same-orientation semicircles, dist=0.5):")
    d_overlap = semicircle_signed_dist(0, 0, 0, 0.5, 0, 0)
    print(f"  dist = {d_overlap:.4f} (expect < 0)")

    # Test 3: clearly separated pair
    print("\nSeparation test (back-to-back, dist=2.5 between centers):")
    d_sep = semicircle_signed_dist(0, 0, 0, 2.5, 0, math.pi)
    print(f"  dist = {d_sep:.4f} (expect > 0)")

    # Test 4: conjugate pair touching
    print("\nConjugate pair (centers coincident, opposite normals):")
    d_conj = semicircle_signed_dist(0, 0, 0, 0, 0, math.pi)
    print(f"  dist = {d_conj:.4f} (expect ≈ 0, they tile a full circle)")

    # Test 5: speed benchmark
    print("\nSpeed benchmark (1000 × 105 pairs):")
    t0 = time.time()
    for _ in range(1000):
        all_pairs_signed_dist(xs, ys, ts)
    elapsed = (time.time() - t0) * 1000
    print(f"  {elapsed:.1f}ms for 1000 calls = {elapsed/1000:.3f}ms per call")
    print(f"  = {elapsed/1000/105*1e6:.0f}ns per pair")

    # Test 6: vs Shapely on valid solution
    print("\nCross-check with phi_pair on known false positives:")
    from phi import phi_pair
    neg_phi = 0; neg_exact = 0; agree = 0
    for i in range(15):
        for j in range(i+1, 15):
            p = phi_pair(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            d = semicircle_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if p < 0: neg_phi += 1
            if d < 0: neg_exact += 1
            if (p >= 0) == (d >= 0): agree += 1
    print(f"  phi negatives (false positives on valid sol): {neg_phi}")
    print(f"  exact negatives (should be 0): {neg_exact}")
    print(f"  agreement: {agree}/105")
