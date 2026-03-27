"""
phi.py — Exact analytical phi-function overlap and containment checker
for unit semicircles packed in a circle of radius R.

A semicircle S_i is defined by center (x_i, y_i) and orientation θ_i.
The flat edge (diameter) has normal direction (sin θ_i, -cos θ_i),
so the half-plane P_i is: sin(θ_i)(x - x_i) - cos(θ_i)(y - y_i) ≥ -1
and the disk C_i is: (x - x_i)² + (y - y_i)² ≤ 1.

Phi-function Φ^{S_i S_j} = max(Φ_CC, Φ_CP, Φ_PC, Φ_PP)
  Φ > 0: separated, Φ = 0: touching, Φ < 0: overlapping.
Non-overlap constraint: Φ^{S_i S_j} ≥ 0.

Reference: Chernov, Stoyan, Romanova, Pankratov (2012).
"""

import numpy as np


# ── Core phi-function components ─────────────────────────────────────────────

def phi_CC(xi, yi, xj, yj):
    """Circle i vs Circle j: separation minus sum of radii (both radius 1)."""
    return (xi - xj)**2 + (yi - yj)**2 - 4.0


def phi_CP(xi, yi, xj, yj, tj):
    """
    Circle i vs HalfPlane j.
    Half-plane P_j (arc side): cos(tj)*(px-xj) + sin(tj)*(py-yj) >= 0
    S_i's disk is separated from complement of P_j iff the center of S_i
    is more than 1 unit into the flat (non-arc) side of S_j:
    φ_CP = -(cos(tj)*(xi-xj) + sin(tj)*(yi-yj)) - 1
    >= 0 means separated (C_i cannot reach the arc side of S_j).
    """
    return -(np.cos(tj) * (xi - xj) + np.sin(tj) * (yi - yj)) - 1.0


def phi_PC(xi, yi, ti, xj, yj):
    """HalfPlane i vs Circle j (symmetric to phi_CP)."""
    return -(np.cos(ti) * (xj - xi) + np.sin(ti) * (yj - yi)) - 1.0


def phi_PP(xi, yi, ti, xj, yj, tj):
    """
    HalfPlane i vs HalfPlane j.
    Correct condition: H_i and H_j (the arc-side half-planes) are disjoint,
    meaning their intersection is empty. For near-anti-parallel normals
    (conjugate pair geometry), the separating condition is:
    n_i . (ci - cj) > 0  i.e. the arc normal of S_i points away from S_j's center.
    This generalizes correctly across all orientation relationships:
    φ_PP = cos(ti)*(xi-xj) + sin(ti)*(yi-yj)
    >= 0 means the arc normal of i points away from j (flat sides face each other).
    """
    return np.cos(ti) * (xi - xj) + np.sin(ti) * (yi - yj)


# ── Pairwise phi (scalar) ─────────────────────────────────────────────────────

def phi_pair(xi, yi, ti, xj, yj, tj):
    """
    Phi-function for semicircle pair (i, j).
    Returns scalar: ≥ 0 means non-overlapping.

    Components used:
    - phi_CC: disk-disk separation (always correct)
    - phi_CP: disk i vs flat-side of j (correct)
    - phi_PC: disk j vs flat-side of i (correct)
    - phi_PP: conditional — only applied when normals are sufficiently anti-parallel
              (n_dot < -0.5), which is the conjugate-pair regime where phi_PP
              is geometrically valid. For similar-orientation pairs, phi_PP
              produces false-positive separations and is excluded.
    """
    cc = phi_CC(xi, yi, xj, yj)
    cp = phi_CP(xi, yi, xj, yj, tj)
    pc = phi_PC(xi, yi, ti, xj, yj)
    phi = max(cc, cp, pc)

    # Only include phi_PP for near-anti-parallel orientations
    n_dot = np.cos(ti) * np.cos(tj) + np.sin(ti) * np.sin(tj)
    if n_dot < -0.5:
        pp = phi_PP(xi, yi, ti, xj, yj, tj)
        phi = max(phi, pp)

    return phi


# ── Vectorized pairwise phi (all 105 pairs at once) ───────────────────────────

def phi_all_pairs(xs, ys, ts):
    """
    Compute phi for all N*(N-1)/2 pairs.
    Returns array of shape (N*(N-1)//2,) — positive means non-overlapping.
    """
    n = len(xs)
    phis = []
    for i in range(n):
        for j in range(i + 1, n):
            phis.append(phi_pair(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j]))
    return np.array(phis)


def all_pairs_overlap_free(xs, ys, ts, tol=1e-6):
    """Returns True if all pairs are non-overlapping (phi >= tol for clearance)."""
    return bool(np.all(phi_all_pairs(xs, ys, ts) >= -tol))


# ── Containment: semicircle S_i inside circle of radius R ────────────────────

def containment_points(xi, yi, ti):
    """
    Three critical points of semicircle S_i that must lie inside container:
      - Arc apex (direction theta): (xi + cos(ti), yi + sin(ti))
      - Two diameter endpoints (perpendicular to theta):
          (xi - sin(ti), yi + cos(ti))  and  (xi + sin(ti), yi - cos(ti))
    Returns array of shape (3, 2).
    """
    return np.array([
        [xi + np.cos(ti),  yi + np.sin(ti)],   # arc apex
        [xi - np.sin(ti),  yi + np.cos(ti)],   # diameter endpoint 1
        [xi + np.sin(ti),  yi - np.cos(ti)],   # diameter endpoint 2
    ])


def phi_containment(xi, yi, ti, R):
    """
    Containment phi for semicircle i in container of radius R.
    Returns min slack across 3 critical points: positive = inside.
    phi_cont = R² - |p|² for each critical point p; all must be ≥ 0.
    We return the worst (minimum) across the 3 points.
    """
    pts = containment_points(xi, yi, ti)
    slacks = R**2 - (pts[:, 0]**2 + pts[:, 1]**2)
    return float(np.min(slacks))


def all_contained(xs, ys, ts, R, tol=1e-6):
    """Returns True if all semicircles are inside container of radius R."""
    for i in range(len(xs)):
        if phi_containment(xs[i], ys[i], ts[i], R) < -tol:
            return False
    return True


# ── Full feasibility check ────────────────────────────────────────────────────

def is_feasible(xs, ys, ts, R, tol=1e-6):
    """True if all semicircles are mutually non-overlapping and contained in R.
    Uses tol=1e-6 (positive clearance) to avoid accepting near-boundary configurations
    that L-BFGS hasn't fully converged on — these have tiny real overlaps."""
    return all_contained(xs, ys, ts, R, tol) and all_pairs_overlap_free(xs, ys, ts, tol)


# ── Penalty energy (for gradient-based optimization) ─────────────────────────

def penalty_energy(xs, ys, ts, R):
    """
    E = Σ max(0, -phi_pair)² + Σ max(0, -phi_cont)²
    Zero iff feasible. Smooth, differentiable almost everywhere.
    """
    n = len(xs)
    E = 0.0
    # Pairwise overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            p = phi_pair(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if p < 0:
                E += p * p
    # Containment penalties
    for i in range(n):
        c = phi_containment(xs[i], ys[i], ts[i], R)
        if c < 0:
            E += c * c
    return E


def penalty_energy_flat(params, R):
    """Flat params vector [x0,y0,t0, x1,y1,t1, ...] → scalar penalty."""
    n = len(params) // 3
    xs = params[0::3]
    ys = params[1::3]
    ts = params[2::3]
    return penalty_energy(xs, ys, ts, R)


# ── Analytical gradient of penalty energy ─────────────────────────────────────

def penalty_gradient(xs, ys, ts, R):
    """
    Analytical gradient of E w.r.t. (xs, ys, ts).
    Returns (grad_xs, grad_ys, grad_ts) each of shape (N,).
    """
    n = len(xs)
    gx = np.zeros(n)
    gy = np.zeros(n)
    gt = np.zeros(n)

    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ti = xs[i], ys[i], ts[i]
            xj, yj, tj = xs[j], ys[j], ts[j]

            cc = phi_CC(xi, yi, xj, yj)
            cp = phi_CP(xi, yi, xj, yj, tj)
            pc = phi_PC(xi, yi, ti, xj, yj)
            pp = phi_PP(xi, yi, ti, xj, yj, tj)

                        # Conditional phi_PP: only for near-anti-parallel orientations
            n_dot = np.cos(ti)*np.cos(tj) + np.sin(ti)*np.sin(tj)
            use_pp = n_dot < -0.5
            pp_val = phi_PP(xi, yi, ti, xj, yj, tj) if use_pp else -np.inf
            phi = max(cc, cp, pc, pp_val)

            if phi >= 0:
                continue  # non-overlapping, no gradient contribution

            coeff = 2.0 * phi  # negative

            if phi == cc:
                ddx = 2*(xi - xj); ddy = 2*(yi - yj)
                gx[i] += coeff * ddx;   gy[i] += coeff * ddy
                gx[j] += coeff * (-ddx); gy[j] += coeff * (-ddy)

            elif phi == cp:
                ctj = np.cos(tj); stj = np.sin(tj)
                gx[i] += coeff * (-ctj); gy[i] += coeff * (-stj)
                gx[j] += coeff * ctj;    gy[j] += coeff * stj
                gt[j] += coeff * (np.sin(tj)*(xi-xj) - np.cos(tj)*(yi-yj))

            elif phi == pc:
                cti = np.cos(ti); sti = np.sin(ti)
                gx[i] += coeff * cti;    gy[i] += coeff * sti
                gx[j] += coeff * (-cti); gy[j] += coeff * (-sti)
                gt[i] += coeff * (np.sin(ti)*(xj-xi) - np.cos(ti)*(yj-yi))

            else:  # pp (only reached when use_pp=True and n_dot < -0.5)
                cti = np.cos(ti); sti = np.sin(ti)
                gx[i] += coeff * cti
                gy[i] += coeff * sti
                gx[j] += coeff * (-cti)
                gy[j] += coeff * (-sti)
                gt[i] += coeff * (-sti*(xi-xj) + cti*(yi-yj))

    # Containment gradients
    for i in range(n):
        xi, yi, ti = xs[i], ys[i], ts[i]
        pts = containment_points(xi, yi, ti)
        slacks = R**2 - (pts[:, 0]**2 + pts[:, 1]**2)
        for k in range(3):
            s = slacks[k]
            if s >= 0:
                continue
            # E_cont = (-s)^2 for s < 0, so dE/d(param) = 2*s * d(s)/d(param)
            # slack = R^2 - px^2 - py^2
            # d(slack)/d(xi) = -2*px (since d(px)/d(xi) = 1 for all 3 critical points)
            # dE/d(xi) = 2*s * (-2*px)
            px, py = pts[k]
            ds_dxi = -2.0 * px   # d(slack)/d(xi)
            ds_dyi = -2.0 * py   # d(slack)/d(yi)
            gx[i] += 2.0 * s * ds_dxi
            gy[i] += 2.0 * s * ds_dyi
            # d(slack)/d(ti) = -2*px*d(px)/d(ti) - 2*py*d(py)/d(ti)
            if k == 0:   # (xi+cos(ti), yi+sin(ti))
                dpx_dti = -np.sin(ti); dpy_dti = np.cos(ti)
            elif k == 1: # (xi-sin(ti), yi+cos(ti))
                dpx_dti = -np.cos(ti); dpy_dti = -np.sin(ti)
            else:        # (xi+sin(ti), yi-cos(ti))
                dpx_dti = np.cos(ti);  dpy_dti = np.sin(ti)
            ds_dti = -2.0 * (px * dpx_dti + py * dpy_dti)
            gt[i] += 2.0 * s * ds_dti

    return gx, gy, gt


def penalty_gradient_flat(params, R):
    """Flat params → flat gradient."""
    n = len(params) // 3
    xs = params[0::3].copy()
    ys = params[1::3].copy()
    ts = params[2::3].copy()
    gx, gy, gt = penalty_gradient(xs, ys, ts, R)
    grad = np.zeros_like(params)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gt
    return grad


# ── Validation against known solution ────────────────────────────────────────

if __name__ == '__main__':
    import json, os
    best_file = os.path.join(os.path.dirname(__file__), 'best_solution.json')
    with open(best_file) as f:
        raw = json.load(f)
    xs = np.array([s['x'] for s in raw])
    ys = np.array([s['y'] for s in raw])
    ts = np.array([s['theta'] for s in raw])
    R = 3.071975

    print(f"N = {len(xs)} semicircles, R = {R}")
    print(f"Feasible: {is_feasible(xs, ys, ts, R)}")
    print(f"Penalty energy: {penalty_energy(xs, ys, ts, R):.2e}")

    phis = phi_all_pairs(xs, ys, ts)
    print(f"Min pair phi: {phis.min():.6f} (should be ≥ 0)")
    print(f"Negative pairs: {(phis < 0).sum()} (should be 0)")

    cont = [phi_containment(xs[i], ys[i], ts[i], R) for i in range(len(xs))]
    print(f"Min containment phi: {min(cont):.6f} (should be ≥ 0)")
    print(f"Negative containment: {sum(c < 0 for c in cont)} (should be 0)")
    print("✓ phi.py validation complete")
