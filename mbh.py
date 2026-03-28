"""
mbh.py — Monotonic Basin Hopping optimizer for semicircle packing.

Strictly monotonic acceptance: NEVER accept worse solutions.
No temperature, no Metropolis criterion.

Architecture:
  - Maintain current best (xs, ys, ts) + score
  - Each iteration: perturb → L-BFGS-B local minimize → validate → accept if better
  - Perturbation variants: standard, swap, rattler, flat_face
  - FSS escape every 10 consecutive non-improving iterations

Usage:
    python mbh.py [--iters 500] [--max-no-imp 200]
"""

import json, math, time, random, argparse, os, sys, fcntl
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from phi import (
    phi_pair, penalty_energy_flat, penalty_gradient_flat,
    penalty_energy, phi_containment, containment_points
)
from gjk_numba import semicircle_gjk_signed_dist
from src.semicircle_packing.geometry import Semicircle
from src.semicircle_packing.scoring import validate_and_score

BEST_FILE = os.path.join(os.path.dirname(__file__), 'best_solution.json')
LOG_FILE  = os.path.join(os.path.dirname(__file__), 'mbh_log.txt')
N = 15
TWO_PI = 2.0 * math.pi


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    xs = np.array([s['x'] for s in raw])
    ys = np.array([s['y'] for s in raw])
    ts = np.array([s['theta'] for s in raw])
    return xs, ys, ts


def save_best(xs, ys, ts, R, log=True):
    """Write to BEST_FILE only if R is strictly better than disk. File-locked."""
    lock_path = BEST_FILE + '.lock'
    with open(lock_path, 'w') as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with open(BEST_FILE) as f:
                    disk_raw = json.load(f)
                disk_xs = [s['x'] for s in disk_raw]
                disk_ys = [s['y'] for s in disk_raw]
                cx = sum(disk_xs)/len(disk_xs); cy = sum(disk_ys)/len(disk_ys)
                disk_R_approx = max(math.hypot(x-cx, y-cy) for x, y in zip(disk_xs, disk_ys)) + 1.0
                if disk_R_approx < R - 1e-6:
                    msg = f"[SKIP WRITE] disk R≈{disk_R_approx:.6f} < candidate R={R:.6f}"
                    print(msg, flush=True)
                    if log:
                        with open(LOG_FILE, 'a') as f:
                            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
                    return
            except Exception:
                pass

            data = [{'x': round(float(xs[i]), 6), 'y': round(float(ys[i]), 6),
                     'theta': round(float(ts[i]) % TWO_PI, 6)}
                    for i in range(N)]
            with open(BEST_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            msg = f"[NEW BEST] R = {R:.6f}"
            print(msg, flush=True)
            if log:
                with open(LOG_FILE, 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
            try:
                import subprocess
                subprocess.Popen([
                    'openclaw', 'message', 'send',
                    '--channel', 'telegram',
                    '--target', '1602537663',
                    '--message', f'🎯 Packing new best (MBH): R={R:.6f}'
                ])
            except Exception:
                pass
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def pack(xs, ys, ts):
    p = np.zeros(3 * N)
    p[0::3] = xs; p[1::3] = ys; p[2::3] = ts
    return p


def unpack(p):
    return p[0::3].copy(), p[1::3].copy(), p[2::3].copy()


def official_validate(xs, ys, ts):
    scs = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    return validate_and_score(scs)


def center_solution(xs, ys, ts):
    r = official_validate(xs, ys, ts)
    if r.mec:
        cx, cy, _ = r.mec
        return xs - cx, ys - cy, ts
    return xs, ys, ts


# ── L-BFGS-B local minimizer ────────────────────────────────────────────────

def lbfgs_minimize(xs, ys, ts, R, lam=500.0, max_iter=2000):
    """
    Minimize penalty energy (overlap + containment) at fixed R.
    Returns (xs, ys, ts, energy).
    """
    p0 = pack(xs, ys, ts)

    def f(p):
        return lam * penalty_energy_flat(p, R)

    def g(p):
        return lam * penalty_gradient_flat(p, R)

    result = minimize(f, p0, jac=g, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
    rxs, rys, rts = unpack(result.x)
    energy = penalty_energy(rxs, rys, rts, R)
    return rxs, rys, rts, energy


# ── GJK-based L-BFGS-B local minimizer ─────────────────────────────────────

def gjk_lbfgs_minimize(xs, ys, ts, R, lam=1e6, max_iter=3000):
    """
    Minimize GJK-based penalty energy (overlap + containment) at fixed R.
    Uses finite-difference gradients since GJK has no analytical gradient.
    Returns (xs, ys, ts, energy).
    """
    p0 = pack(xs, ys, ts)
    n_vars = len(p0)
    eps = 1e-5

    def gjk_energy(p):
        _xs = p[0::3]
        _ys = p[1::3]
        _ts = p[2::3]
        E = 0.0
        # Pairwise overlap via GJK
        for i in range(N):
            for j in range(i + 1, N):
                d = semicircle_gjk_signed_dist(
                    _xs[i], _ys[i], _ts[i], _xs[j], _ys[j], _ts[j])
                if d < 0.0:
                    E += d * d * lam
        # Containment penalties
        for i in range(N):
            c = phi_containment(_xs[i], _ys[i], _ts[i], R)
            if c < 0.0:
                E += c * c * lam
        return E

    def fg(p):
        """Return (energy, gradient) for L-BFGS-B with jac=True."""
        E0 = gjk_energy(p)
        grad = np.empty(n_vars)
        for k in range(n_vars):
            p[k] += eps
            Ep = gjk_energy(p)
            p[k] -= 2.0 * eps
            Em = gjk_energy(p)
            p[k] += eps  # restore
            grad[k] = (Ep - Em) / (2.0 * eps)
        return E0, grad

    result = minimize(fg, p0, jac=True, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    rxs, rys, rts = unpack(result.x)
    energy = gjk_energy(result.x)
    return rxs, rys, rts, energy


# ── Perturbation operators ───────────────────────────────────────────────────

def perturb_standard(xs, ys, ts, delta=0.8):
    """Perturb all 15 positions + orientations."""
    xs = xs + np.random.uniform(-delta, delta, N)
    ys = ys + np.random.uniform(-delta, delta, N)
    ts = (ts + np.random.uniform(-delta, delta, N)) % TWO_PI
    return xs, ys, ts


def perturb_swap(xs, ys, ts):
    """Swap two random semicircles' full state, then perturb all."""
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    i, j = random.sample(range(N), 2)
    xs[i], xs[j] = xs[j], xs[i]
    ys[i], ys[j] = ys[j], ys[i]
    ts[i], ts[j] = ts[j], ts[i]
    # Then small perturbation on all
    xs += np.random.uniform(-0.3, 0.3, N)
    ys += np.random.uniform(-0.3, 0.3, N)
    ts = (ts + np.random.uniform(-0.3, 0.3, N)) % TWO_PI
    return xs, ys, ts


def perturb_rattler(xs, ys, ts, R):
    """Find semicircles with <3 contacts, reinsert at random valid positions."""
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    contact_counts = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            p = phi_pair(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if p < 0.1:  # close enough to count as near-contact
                contact_counts[i] += 1

    rattlers = [i for i in range(N) if contact_counts[i] < 3]
    if not rattlers:
        rattlers = [random.randrange(N)]

    for idx in rattlers[:3]:  # reinsert at most 3
        r_max = max(R - 1.0, 0.5)
        for _ in range(100):
            r = random.uniform(0, r_max)
            angle = random.uniform(0, TWO_PI)
            xs[idx] = r * math.cos(angle)
            ys[idx] = r * math.sin(angle)
            ts[idx] = random.uniform(0, TWO_PI)
            if phi_containment(xs[idx], ys[idx], ts[idx], R) >= -0.1:
                break
    return xs, ys, ts


def perturb_flat_face(xs, ys, ts):
    """Pick two semicircles, set antiparallel thetas, slide into contact."""
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    i, j = random.sample(range(N), 2)
    # Set j antiparallel to i
    ts[j] = (ts[i] + math.pi) % TWO_PI
    # Slide j so flat edge is adjacent to i's flat edge
    # Tangent direction along flat edge of i
    tx = -math.sin(ts[i])
    ty = math.cos(ts[i])
    offset = random.uniform(-1.2, 1.2)
    xs[j] = xs[i] + tx * offset
    ys[j] = ys[i] + ty * offset
    return xs, ys, ts


def apply_perturbation(xs, ys, ts, R, delta=0.8):
    """Mixed perturbation: 50% standard, 20% swap, 15% rattler, 15% flat_face."""
    r = random.random()
    if r < 0.50:
        return perturb_standard(xs, ys, ts, delta=delta), 'standard'
    elif r < 0.70:
        return perturb_swap(xs, ys, ts), 'swap'
    elif r < 0.85:
        return perturb_rattler(xs, ys, ts, R), 'rattler'
    else:
        return perturb_flat_face(xs, ys, ts), 'flat_face'


# ── FSS escape (polar perturbation) ─────────────────────────────────────────

def fss_escape(xs, ys, ts, R, best_score):
    """
    Convert to polar, perturb, convert back, minimize, accept if better.
    Returns (xs, ys, ts, score, improved).
    """
    # Convert to polar
    rs = np.sqrt(xs**2 + ys**2)
    phis = np.arctan2(ys, xs)
    thetas = ts.copy()

    # Perturb in polar space
    rs = rs + np.random.uniform(-0.5, 0.5, N)
    rs = np.clip(rs, 0.0, R - 1.0)
    phis = phis + np.random.uniform(-0.3, 0.3, N)
    thetas = (thetas + np.random.uniform(-0.8, 0.8, N)) % TWO_PI

    # Convert back to Cartesian
    pxs = rs * np.cos(phis)
    pys = rs * np.sin(phis)

    # Local minimize
    rxs, rys, rts, energy = lbfgs_minimize(pxs, pys, thetas, R)

    if energy > 1e-4:
        return xs, ys, ts, best_score, False

    # Validate
    result = official_validate(rxs, rys, rts)
    if not result.valid:
        return xs, ys, ts, best_score, False

    score = float(result.score)
    if score < best_score:
        return rxs, rys, rts, score, True

    return xs, ys, ts, best_score, False


# ── Main MBH loop ───────────────────────────────────────────────────────────

def mbh_iteration(xs, ys, ts, R, best_score, delta=0.8):
    """
    Single MBH iteration: perturb → minimize → validate → accept if better.
    Returns (xs, ys, ts, score, improved, kind).
    """
    (pxs, pys, pts), kind = apply_perturbation(xs, ys, ts, R, delta=delta)

    # Local minimize
    rxs, rys, rts, energy = lbfgs_minimize(pxs, pys, pts, R)

    if energy > 1e-4:
        return xs, ys, ts, best_score, False, kind

    # Validate with official Shapely scorer
    result = official_validate(rxs, rys, rts)
    if not result.valid:
        return xs, ys, ts, best_score, False, kind

    score = float(result.score)

    # STRICTLY MONOTONIC: only accept if better
    if score < best_score:
        return rxs, rys, rts, score, True, kind

    return xs, ys, ts, best_score, False, kind


def run_mbh(max_iters=500, max_no_imp=200, verbose=True):
    """
    Monotonic Basin Hopping main loop.
    Returns best (xs, ys, ts, score).
    """
    xs, ys, ts = load_best()
    result = official_validate(xs, ys, ts)
    if not result.valid:
        print("WARNING: loaded solution is not valid!", flush=True)
        best_score = float('inf')
    else:
        best_score = float(result.score)

    R = best_score + 0.01  # search radius slightly above best

    log = open(LOG_FILE, 'a')
    def logprint(msg):
        print(msg, flush=True)
        log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        log.flush()

    logprint(f"=== MBH started | R={best_score:.6f} | max_iters={max_iters} ===")

    no_imp = 0
    fss_no_imp = 0

    for it in range(max_iters):
        xs, ys, ts, best_score, improved, kind = mbh_iteration(xs, ys, ts, R, best_score)

        if improved:
            no_imp = 0
            fss_no_imp = 0
            R = best_score + 0.01
            # Center and save
            cxs, cys, cts = center_solution(xs, ys, ts)
            save_best(cxs, cys, cts, best_score)
            logprint(f"  [{it+1:4d}] ★ {kind:10s} → R={best_score:.6f}")
            xs, ys, ts = cxs, cys, cts
        else:
            no_imp += 1
            fss_no_imp += 1
            if (it + 1) % 20 == 0 or no_imp == max_no_imp:
                logprint(f"  [{it+1:4d}] {kind:10s} no_imp={no_imp}/{max_no_imp}")

        # FSS escape every 10 consecutive non-improving iterations
        if fss_no_imp > 0 and fss_no_imp % 10 == 0:
            xs, ys, ts, best_score, fss_improved = fss_escape(xs, ys, ts, R, best_score)
            if fss_improved:
                no_imp = 0
                fss_no_imp = 0
                R = best_score + 0.01
                cxs, cys, cts = center_solution(xs, ys, ts)
                save_best(cxs, cys, cts, best_score)
                logprint(f"  [{it+1:4d}] ★ FSS_escape → R={best_score:.6f}")
                xs, ys, ts = cxs, cys, cts

        if no_imp >= max_no_imp:
            logprint(f"  [{it+1:4d}] MaxNoImp={max_no_imp} reached")
            # Don't restart — let caller handle
            no_imp = 0  # reset and continue
            # Reload best from disk (another worker may have found better)
            try:
                dxs, dys, dts = load_best()
                dr = official_validate(dxs, dys, dts)
                if dr.valid and dr.score < best_score:
                    xs, ys, ts = dxs, dys, dts
                    best_score = float(dr.score)
                    R = best_score + 0.01
                    logprint(f"  [{it+1:4d}] Reloaded from disk: R={best_score:.6f}")
            except Exception:
                pass

    logprint(f"=== MBH done | best R={best_score:.6f} ===")
    log.close()
    return xs, ys, ts, best_score


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--max-no-imp', type=int, default=200)
    args = parser.parse_args()

    np.random.seed(int(time.time()) % 100000)
    random.seed(int(time.time()) % 100000)

    run_mbh(max_iters=args.iters, max_no_imp=args.max_no_imp)
