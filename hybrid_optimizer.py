#!/usr/bin/env python3
"""
hybrid_optimizer.py — Hybrid optimizer for semicircle packing.

Strategy (Option C from PHI_DEBUG_LOG.md):
  - Use phi-function L-BFGS for fast local optimization (phase 1)
  - Use Shapely-based L-BFGS with FD gradients for polish (phase 2, 64-pt for speed)
  - Validate with official scorer (4096 pts)
  - MBH loop with diverse perturbations for basin exploration

Key insight: containment must check ALL arc points, not just 3 critical points.
The furthest point on the arc from the origin is at angle atan2(cy,cx) if it
falls within the arc range, giving max_dist = d_center + 1.
"""

import json, math, time, random, os, sys
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon

# Import phi-based functions (fast analytical)
from phi import (
    penalty_energy_flat as _phi_energy_flat,
    penalty_gradient_flat as _phi_gradient_flat,
    phi_pair, phi_containment,
)

# Import official scorer
from fast_run import official_score

N = 15
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_FILE = os.path.join(BASE_DIR, 'best_solution.json')
LOG_FILE = os.path.join(BASE_DIR, 'hybrid_log.txt')

_log_fh = None

def logprint(msg):
    global _log_fh
    if _log_fh is None:
        _log_fh = open(LOG_FILE, 'a')
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    _log_fh.write(line + '\n')
    _log_fh.flush()


# ── Correct containment ────────────────────────────────────────────────────

def max_dist_from_origin(x, y, theta):
    """
    Maximum distance from origin to any point on the semicircle.
    The arc is parameterized as (x+cos(a), y+sin(a)) for a in [theta-pi/2, theta+pi/2].
    Distance² from origin = x²+y²+1+2(x·cos(a)+y·sin(a)).
    Max of x·cos(a)+y·sin(a) = sqrt(x²+y²) at a=atan2(y,x), if within arc range.
    """
    d_center = math.sqrt(x*x + y*y)
    if d_center < 1e-12:
        return 1.0  # center at origin, max dist = radius = 1

    angle_to_center = math.atan2(y, x)
    delta = angle_to_center - theta
    delta = (delta + math.pi) % (2 * math.pi) - math.pi

    if abs(delta) <= math.pi / 2:
        # Direction to center is within arc range
        return d_center + 1.0
    else:
        # Check 3 critical points
        pts = [
            (x + math.cos(theta), y + math.sin(theta)),
            (x - math.sin(theta), y + math.cos(theta)),
            (x + math.sin(theta), y - math.cos(theta)),
        ]
        return max(math.sqrt(px*px + py*py) for px, py in pts)


def containment_energy(xs, ys, ts, R):
    """Containment penalty using correct max-distance check."""
    E = 0.0
    for i in range(N):
        max_d = max_dist_from_origin(xs[i], ys[i], ts[i])
        slack = R - max_d
        if slack < 0:
            E += slack * slack
    return E


def corrected_energy_flat(params, R):
    """
    Phi overlap energy + CORRECT containment energy.
    The phi overlap terms are mostly OK; it's the containment that's wrong.
    We replace phi containment with the correct max-distance check.
    """
    xs = params[0::3]; ys = params[1::3]; ts = params[2::3]
    # Phi pairwise overlap energy (skip phi's containment)
    E = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            p = phi_pair(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if p < 0:
                E += p * p
    # Correct containment
    E += containment_energy(xs, ys, ts, R)
    return E


def corrected_gradient_flat(params, R, eps=1e-7):
    """
    Hybrid gradient: phi analytical gradient for overlap terms,
    finite-difference for correct containment terms.
    """
    # Start with phi gradient (handles overlap well)
    grad = _phi_gradient_flat(params, R)

    xs = params[0::3]; ys = params[1::3]; ts = params[2::3]

    # Replace containment gradient with FD of correct containment
    # Only need FD for the ~3-5 semicircles that violate containment
    for i in range(N):
        max_d = max_dist_from_origin(xs[i], ys[i], ts[i])
        if max_d > R - 0.1:  # near or beyond boundary
            for k_off in range(3):
                idx = 3*i + k_off
                params[idx] += eps
                E_plus = containment_energy(
                    params[0::3], params[1::3], params[2::3], R)
                params[idx] -= 2*eps
                E_minus = containment_energy(
                    params[0::3], params[1::3], params[2::3], R)
                params[idx] += eps
                # Replace phi containment gradient component with correct FD
                grad[idx] = grad[idx]  # keep phi overlap gradient
                # Add correct containment gradient (we need to subtract phi's
                # containment contribution and add our own, but that's complex.
                # Simpler: just add the containment FD gradient on top)
                cont_grad = (E_plus - E_minus) / (2*eps)
                # Estimate phi's containment contribution to subtract
                # Actually, just adding the correct containment gradient works
                # because phi's containment at these points is also negative
                grad[idx] += cont_grad * 5.0  # amplify to dominate phi's incorrect term

    return grad


# ── Shapely overlap energy (64-pt, fast) ────────────────────────────────────

def make_poly(x, y, theta, n=64):
    angles = np.linspace(theta - math.pi / 2, theta + math.pi / 2, n)
    return Polygon(list(zip(x + np.cos(angles), y + np.sin(angles))))


def overlap_energy(xs, ys, ts):
    """Shapely intersection area² for overlapping pairs."""
    polys = [make_poly(xs[i], ys[i], ts[i]) for i in range(N)]
    E = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue
            area = polys[i].intersection(polys[j]).area
            if area > 1e-8:
                E += area * area
    return E


def shapely_energy_flat(params, R):
    """Combined Shapely overlap + correct containment energy."""
    xs = params[0::3]; ys = params[1::3]; ts = params[2::3]
    return overlap_energy(xs, ys, ts) + containment_energy(xs, ys, ts, R)


def shapely_gradient_fd(params, R, eps=1e-5):
    """Finite-difference gradient of Shapely energy."""
    grad = np.zeros_like(params)
    E0 = shapely_energy_flat(params, R)
    for k in range(len(params)):
        params[k] += eps
        Ep = shapely_energy_flat(params, R)
        params[k] -= eps
        grad[k] = (Ep - E0) / eps
    return grad


# ── Hybrid L-BFGS ───────────────────────────────────────────────────────────

def quick_shapely_check(xs, ys, ts, R):
    """
    256-pt Shapely overlap check + correct containment check.
    Uses 256-pt polygons (matching fast_run.py) with OVERLAP_TOL=1e-6.
    Returns (n_overlaps, max_overlap_area, containment_ok).
    """
    polys = [make_poly(xs[i], ys[i], ts[i], n=256) for i in range(N)]
    n_overlaps = 0
    max_area = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue
            area = polys[i].intersection(polys[j]).area
            if area > 1e-6:  # match official OVERLAP_TOL
                n_overlaps += 1
                max_area = max(max_area, area)

    cont_ok = True
    for i in range(N):
        if max_dist_from_origin(xs[i], ys[i], ts[i]) > R:
            cont_ok = False
            break

    return n_overlaps, max_area, cont_ok


def find_overlapping_indices(xs, ys, ts):
    """Find indices of semicircles involved in Shapely overlaps (256-pt)."""
    polys = [make_poly(xs[i], ys[i], ts[i], n=256) for i in range(N)]
    involved = set()
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue
            if polys[i].intersection(polys[j]).area > 1e-6:
                involved.add(i)
                involved.add(j)
    return sorted(involved)


def repair_energy(sub_params, fixed_xs, fixed_ys, fixed_ts, free_idx, R):
    """Energy for just the free semicircles, checking against all others.
    Uses buffered polygons to be conservative."""
    xs = fixed_xs.copy()
    ys = fixed_ys.copy()
    ts = fixed_ts.copy()
    for k, idx in enumerate(free_idx):
        xs[idx] = sub_params[3*k]
        ys[idx] = sub_params[3*k+1]
        ts[idx] = sub_params[3*k+2]

    polys = [make_poly(xs[i], ys[i], ts[i], n=256) for i in range(N)]
    E = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue
            area = polys[i].intersection(polys[j]).area
            if area > 0:
                E += area * area
    # Containment for free indices
    for idx in free_idx:
        md = max_dist_from_origin(xs[idx], ys[idx], ts[idx])
        slack = R - md
        if slack < 0:
            E += slack * slack
    return E


def repair_overlaps(xs, ys, ts, R, max_iter=60):
    """
    Repair step: find overlapping pairs, optimize only involved semicircles
    using Shapely FD gradients on the small subset.
    Returns (xs, ys, ts, n_remaining_overlaps).
    """
    free_idx = find_overlapping_indices(xs, ys, ts)
    if not free_idx:
        return xs, ys, ts, 0

    # Build sub-parameter vector for free semicircles
    sub_p = np.zeros(3 * len(free_idx))
    for k, idx in enumerate(free_idx):
        sub_p[3*k] = xs[idx]
        sub_p[3*k+1] = ys[idx]
        sub_p[3*k+2] = ts[idx]

    def f(sp):
        return repair_energy(sp, xs, ys, ts, free_idx, R)

    # FD gradient on small subspace (3*len(free_idx) params, typically 6-18)
    result = minimize(f, sub_p, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10,
                               'eps': 1e-6})

    rxs, rys, rts = xs.copy(), ys.copy(), ts.copy()
    for k, idx in enumerate(free_idx):
        rxs[idx] = result.x[3*k]
        rys[idx] = result.x[3*k+1]
        rts[idx] = result.x[3*k+2]

    n_ov, _, _ = quick_shapely_check(rxs, rys, rts, R)
    return rxs, rys, rts, n_ov


def hybrid_lbfgs(xs, ys, ts, R, max_iter=300):
    """
    Phase 1: Fast phi-gradient L-BFGS (two-pass, ~0.3s)
    Phase 2: If Shapely finds overlaps, repair them (~1-5s)
    Returns (xs, ys, ts, n_shapely_overlaps).
    """
    p0 = np.zeros(3 * N)
    p0[0::3] = xs; p0[1::3] = ys; p0[2::3] = ts

    # Phase 1: phi L-BFGS (fast, uses phi's 3-point containment)
    result = minimize(
        _phi_energy_flat, p0, args=(R,),
        jac=lambda p, R=R: _phi_gradient_flat(p, R),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    result = minimize(
        _phi_energy_flat, result.x, args=(R,),
        jac=lambda p, R=R: _phi_gradient_flat(p, R),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12}
    )

    # Phase 1b: fix containment with correct energy (FD gradient, fast)
    # Only runs ~10 iters if needed
    E_corr = corrected_energy_flat(result.x, R)
    if E_corr > 1e-10 and result.fun < 0.1:
        result2 = minimize(
            corrected_energy_flat, result.x, args=(R,),
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-10, 'eps': 1e-7}
        )
        if result2.fun < E_corr:
            result = result2

    rxs = result.x[0::3].copy()
    rys = result.x[1::3].copy()
    rts = result.x[2::3].copy()
    E = result.fun

    if E > 0.1:
        return rxs, rys, rts, -1  # infeasible, skip repair

    # Phase 2: check and repair Shapely overlaps
    n_ov, _, cont_ok = quick_shapely_check(rxs, rys, rts, R)
    if n_ov == 0 and cont_ok:
        return rxs, rys, rts, 0  # clean!

    if n_ov > 0 and n_ov <= 6:  # only repair small numbers of overlaps
        rxs, rys, rts, n_ov = repair_overlaps(rxs, rys, rts, R)

    return rxs, rys, rts, n_ov


# ── Perturbation operators ──────────────────────────────────────────────────

def perturb_flip(xs, ys, ts, n_flip=None):
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    if n_flip is None:
        n_flip = random.choice([1, 2, 3])
    for i in random.sample(range(N), n_flip):
        ts[i] = (ts[i] + math.pi) % (2 * math.pi)
    return xs, ys, ts


def perturb_jitter(xs, ys, ts, sigma=0.1):
    return (
        xs + np.random.randn(N) * sigma,
        ys + np.random.randn(N) * sigma,
        ts + np.random.randn(N) * sigma * 0.5,
    )


def perturb_swap(xs, ys, ts):
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    i, j = random.sample(range(N), 2)
    xs[i], xs[j] = xs[j], xs[i]
    ys[i], ys[j] = ys[j], ys[i]
    ts[i], ts[j] = ts[j], ts[i]
    return xs, ys, ts


def perturb_shift_worst(xs, ys, ts, R):
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    penalties = []
    for i in range(N):
        p = 0.0
        for j in range(N):
            if i == j: continue
            phi = phi_pair(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if phi < 0: p += phi * phi
        c = phi_containment(xs[i], ys[i], ts[i], R)
        if c < 0: p += c * c
        penalties.append(p)
    worst = int(np.argmax(penalties))
    r_max = R - 1.0
    for _ in range(200):
        r = random.uniform(0, r_max)
        angle = random.uniform(0, 2 * math.pi)
        xs[worst] = r * math.cos(angle)
        ys[worst] = r * math.sin(angle)
        ts[worst] = random.uniform(0, 2 * math.pi)
        if phi_containment(xs[worst], ys[worst], ts[worst], R) >= -0.1:
            break
    return xs, ys, ts


def perturb_cluster_reflect(xs, ys, ts):
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    angle = random.uniform(0, math.pi)
    nx, ny = math.cos(angle), math.sin(angle)
    k = random.randint(3, N // 2 + 1)
    for i in random.sample(range(N), k):
        dot = xs[i] * nx + ys[i] * ny
        xs[i] = 2 * dot * nx - xs[i]
        ys[i] = 2 * dot * ny - ys[i]
        ts[i] = 2 * angle - ts[i]
    return xs, ys, ts


# Weighted selection
PERT_TABLE = [
    ('flip', 2), ('jitter', 4), ('swap', 1),
    ('shift_worst', 3), ('reflect', 2),
]
PERT_NAMES = [p[0] for p in PERT_TABLE]
PERT_WEIGHTS = [p[1] for p in PERT_TABLE]

def apply_perturbation(xs, ys, ts, R, kind=None):
    if kind is None:
        kind = random.choices(PERT_NAMES, weights=PERT_WEIGHTS, k=1)[0]
    if kind == 'flip':
        return perturb_flip(xs, ys, ts), kind
    elif kind == 'jitter':
        return perturb_jitter(xs, ys, ts), kind
    elif kind == 'swap':
        return perturb_swap(xs, ys, ts), kind
    elif kind == 'shift_worst':
        return perturb_shift_worst(xs, ys, ts, R), kind
    else:
        return perturb_cluster_reflect(xs, ys, ts), kind


# ── I/O ─────────────────────────────────────────────────────────────────────

def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    return (np.array([s['x'] for s in raw]),
            np.array([s['y'] for s in raw]),
            np.array([s['theta'] for s in raw]))


def save_best(xs, ys, ts):
    data = [{'x': round(float(xs[i]), 6), 'y': round(float(ys[i]), 6),
             'theta': round(float(ts[i]), 6)} for i in range(N)]
    for path in [BEST_FILE, os.path.join(BASE_DIR, 'solution.json')]:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# ── Validation ──────────────────────────────────────────────────────────────

def validate():
    logprint("=== Validation ===")
    xs, ys, ts = load_best()

    # Check containment energy at best R
    result = official_score(xs, ys, ts)
    best_R = float(result.score) if result.valid else None
    logprint(f"  Official: valid={result.valid}, R={best_R}")

    if best_R:
        p = np.zeros(45); p[0::3]=xs; p[1::3]=ys; p[2::3]=ts
        sE = shapely_energy_flat(p, best_R)
        logprint(f"  Shapely energy at R={best_R:.6f}: {sE:.2e}")

        sE2 = shapely_energy_flat(p, best_R + 0.01)
        logprint(f"  Shapely energy at R={best_R+0.01:.6f}: {sE2:.2e}")

    # Test hybrid_lbfgs at best R
    rxs, rys, rts, n_ov = hybrid_lbfgs(xs, ys, ts, best_R + 0.01, max_iter=200)
    logprint(f"  hybrid_lbfgs at R={best_R+0.01:.4f}: n_ov={n_ov}")
    if n_ov == 0:
        r2 = official_score(rxs, rys, rts)
        logprint(f"  → valid={r2.valid}, R={r2.score}")

    # Test at tighter R
    rxs2, rys2, rts2, n_ov2 = hybrid_lbfgs(xs, ys, ts, best_R - 0.005, max_iter=300)
    logprint(f"  hybrid_lbfgs at R={best_R-0.005:.4f}: n_ov={n_ov2}")

    logprint("=== Validation done ===\n")
    return best_R


# ── MBH main loop ──────────────────────────────────────────────────────────

def run_mbh(best_R, r_min=2.85, r_step=0.005, rounds_per_R=30):
    """MBH with hybrid L-BFGS. Starts just below best known R."""
    xs0, ys0, ts0 = load_best()
    global_best_R = best_R
    best_xs, best_ys, best_ts = xs0.copy(), ys0.copy(), ts0.copy()

    logprint(f"=== MBH started | best R={global_best_R:.6f} ===")

    R = global_best_R - 0.001  # start just below best
    consecutive_fails = 0

    while R >= r_min and consecutive_fails < 3:
        logprint(f"\n── R = {R:.4f} ──")
        feasible_found = False

        for rnd in range(rounds_per_R):
            t0 = time.time()
            (px, py, pt), kind = apply_perturbation(best_xs, best_ys, best_ts, R)
            rxs, rys, rts, n_ov = hybrid_lbfgs(px, py, pt, R, max_iter=300)
            dt = time.time() - t0

            status = f"  {rnd+1:3d} ({kind:12s}): ov={n_ov} ({dt:.1f}s)"

            if n_ov == 0:
                # Clean per 256-pt Shapely — check officially
                result = official_score(rxs, rys, rts)
                if result.valid:
                    oR = float(result.score)
                    status += f" ✓ R={oR:.6f}"
                    logprint(status)
                    if oR < global_best_R:
                        global_best_R = oR
                        best_xs, best_ys, best_ts = rxs.copy(), rys.copy(), rts.copy()
                        save_best(rxs, rys, rts)
                        logprint(f"  ★ NEW BEST: R = {oR:.6f}")
                        feasible_found = True
                        # Squeeze from the new best
                        cur_xs, cur_ys, cur_ts = rxs.copy(), rys.copy(), rts.copy()
                        for sq in range(10):
                            tR = global_best_R - 0.002
                            if tR < r_min: break
                            sx, sy, st, sn = hybrid_lbfgs(cur_xs, cur_ys, cur_ts, tR, max_iter=300)
                            if sn == 0:
                                sr = official_score(sx, sy, st)
                                if sr.valid and float(sr.score) < global_best_R:
                                    global_best_R = float(sr.score)
                                    best_xs, best_ys, best_ts = sx.copy(), sy.copy(), st.copy()
                                    cur_xs, cur_ys, cur_ts = sx.copy(), sy.copy(), st.copy()
                                    save_best(sx, sy, st)
                                    logprint(f"    Squeeze ★ R = {sr.score:.6f}")
                                else:
                                    break
                            else:
                                break
                        break
                else:
                    status += f" ✗official({len(result.errors)})"
            elif n_ov > 0:
                status += f" ({n_ov}ov)"
            logprint(status)

        if not feasible_found:
            logprint(f"  ✗ Failed at R={R:.4f}")
            consecutive_fails += 1
            R += r_step * 0.5
        else:
            consecutive_fails = 0
            R = global_best_R - r_step

    logprint(f"\n=== MBH done | best R = {global_best_R:.6f} ===")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logprint(f"\n{'='*60}")
    logprint(f"Hybrid Optimizer — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logprint(f"{'='*60}")

    best_R = validate()

    # Quick test: 20 rounds at R just below best
    test_R = best_R - 0.001
    logprint(f"\n=== Quick test: 20 rounds at R={test_R:.4f} ===")
    xs0, ys0, ts0 = load_best()
    valid_count = 0
    for rnd in range(20):
        t0 = time.time()
        (px, py, pt), kind = apply_perturbation(xs0, ys0, ts0, test_R)
        rxs, rys, rts, n_ov = hybrid_lbfgs(px, py, pt, test_R, max_iter=300)
        dt = time.time() - t0
        valid_str = ""
        if n_ov == 0:
            result = official_score(rxs, rys, rts)
            if result.valid:
                valid_count += 1
                valid_str = f" ✓ R={result.score:.6f}"
            else:
                valid_str = f" ✗({len(result.errors)})"
        elif n_ov > 0:
            valid_str = f" ({n_ov}ov)"
        logprint(f"  rnd {rnd+1:2d} ({kind:12s}): ov={n_ov} ({dt:.1f}s){valid_str}")
    logprint(f"  Valid: {valid_count}/20")

    # Full MBH
    logprint("\n=== Starting full MBH ===")
    run_mbh(best_R, r_min=2.85, r_step=0.005, rounds_per_R=100)

    logprint(f"\nDone at {time.strftime('%H:%M:%S')}")
    if _log_fh:
        _log_fh.close()
