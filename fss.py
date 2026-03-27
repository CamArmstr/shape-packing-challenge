"""
fss.py — Formulation Space Search for semicircle packing.

Based on Mladenović, Plastria, Urošević (2005) and López & Beasley (2011).

Three coordinate encodings per semicircle:
  0: (x, y, θ)         Cartesian, absolute angle
  1: (r, φ, θ)         Polar position, absolute angle
  2: (r, φ, θ_rel)     Polar position, relative angle θ_rel = θ - φ
                        → sliding along wall keeps orientation fixed

Core idea: a stationary point in encoding A is generally NOT stationary
in encoding B (nonlinear coord change breaks stationarity).
Switching encoding and re-minimizing escapes the basin.

Reformulation Descent (RD):
  Solve to optimum in encoding A → switch to B → solve → switch to C → solve
  Repeat until no encoding improves the solution.

Full FSS (VNS over 3^15 formulations, practical subset):
  For k = 1, 2, ..., K:
    Pick k random semicircles, flip their encoding
    Solve to local optimum
    Accept if improved, else backtrack and increase k

Usage:
    python fss.py [--rounds 50]
"""

import json, math, time, random, os, sys, argparse
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from phi import penalty_energy_flat, penalty_gradient_flat
from exact_dist import penalty_energy_exact
from overnight import official_validate
from src.semicircle_packing.geometry import Semicircle
from src.semicircle_packing.scoring import validate_and_score

N = 15
BEST_FILE = os.path.join(os.path.dirname(__file__), 'best_solution.json')
LOG_FILE  = os.path.join(os.path.dirname(__file__), 'fss_log.txt')

ENC_CART      = 0   # (x, y, θ)
ENC_POLAR     = 1   # (r, φ, θ)
ENC_POLAR_REL = 2   # (r, φ, θ_rel=θ-φ)


# ─── Encoding conversions ──────────────────────────────────────────────────────

def to_polar(x, y, theta):
    r   = math.sqrt(x*x + y*y)
    phi = math.atan2(y, x)
    return r, phi, theta

def from_polar(r, phi, theta):
    return r * math.cos(phi), r * math.sin(phi), theta

def to_polar_rel(x, y, theta):
    r   = math.sqrt(x*x + y*y)
    phi = math.atan2(y, x)
    return r, phi, theta - phi  # θ_rel = θ - φ

def from_polar_rel(r, phi, theta_rel):
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    return x, y, theta_rel + phi  # θ = θ_rel + φ


# ─── Pack / unpack with mixed encodings ───────────────────────────────────────

def encode_all(xs, ys, ts, encs):
    """Pack N semicircles using given encoding per piece → flat vector."""
    params = []
    for i in range(N):
        e = encs[i]
        if e == ENC_CART:
            params += [xs[i], ys[i], ts[i]]
        elif e == ENC_POLAR:
            r, phi, theta = to_polar(xs[i], ys[i], ts[i])
            params += [r, phi, theta]
        else:  # ENC_POLAR_REL
            r, phi, trel = to_polar_rel(xs[i], ys[i], ts[i])
            params += [r, phi, trel]
    return np.array(params)


def decode_all(p, encs):
    """Unpack flat vector → xs, ys, ts using given encoding per piece."""
    xs = np.empty(N); ys = np.empty(N); ts = np.empty(N)
    for i in range(N):
        a, b, c = p[3*i], p[3*i+1], p[3*i+2]
        e = encs[i]
        if e == ENC_CART:
            xs[i], ys[i], ts[i] = a, b, c
        elif e == ENC_POLAR:
            xs[i], ys[i], ts[i] = from_polar(a, b, c)
        else:
            xs[i], ys[i], ts[i] = from_polar_rel(a, b, c)
    return xs, ys, ts


# ─── Penalty (phi) in arbitrary encoding ──────────────────────────────────────

def penalty_in_enc(p, encs, R, lam=500.0):
    """Exact penalty evaluated after decoding from mixed encoding."""
    xs, ys, ts = decode_all(p, encs)
    return lam * penalty_energy_exact(xs, ys, ts, R)


_FD_EPS = 1e-5

def penalty_grad_in_enc(p, encs, R, lam=500.0):
    """
    Gradient of exact_dist penalty in mixed encoding space.
    Uses chain rule on the phi gradient (fast) plus FD correction for
    thin-sliver pairs that exact_dist catches but phi misses.
    """
    from phi import penalty_gradient_flat
    xs_, ys_, ts_ = decode_all(p, encs)
    p_cart = np.concatenate([xs_, ys_, ts_])

    # Phi gradient in Cartesian
    g_phi_cart = penalty_gradient_flat(p_cart, R)  # (3N,)

    # Exact penalty correction: FD only for variables of thin-sliver pairs
    E_exact = penalty_energy_exact(xs_, ys_, ts_, R)
    g_exact_cart = np.zeros(3*N)
    if E_exact > 1e-9:
        # Find which pairs have exact_dist < 0 (thin slivers phi misses)
        from exact_dist import semicircle_signed_dist
        sliver_pieces = set()
        for ii in range(N):
            for jj in range(ii+1, N):
                dc2 = (xs_[ii]-xs_[jj])**2+(ys_[ii]-ys_[jj])**2
                if dc2 >= 4.0:
                    continue
                d = semicircle_signed_dist(xs_[ii], ys_[ii], ts_[ii], xs_[jj], ys_[jj], ts_[jj])
                if d < -1e-6:
                    sliver_pieces.add(ii)
                    sliver_pieces.add(jj)
        # Only compute FD for pieces involved in thin slivers
        for ii in sliver_pieces:
            for dim in range(3):
                k = dim * N + ii if dim < 2 else 2*N + ii
                p_cart[k] += _FD_EPS
                xs2, ys2, ts2 = p_cart[:N], p_cart[N:2*N], p_cart[2*N:]
                g_exact_cart[k] = (penalty_energy_exact(xs2, ys2, ts2, R) - E_exact) / _FD_EPS
                p_cart[k] -= _FD_EPS

    g_cart = lam * (g_phi_cart + g_exact_cart)

    # Chain rule: transform to encoding space
    g = np.zeros_like(p)
    for i in range(N):
        gx = g_cart[i]
        gy = g_cart[N + i]
        gt = g_cart[2*N + i]
        e = encs[i]
        ri, phi_i = p[3*i], p[3*i+1]
        if e == ENC_CART:
            g[3*i] = gx; g[3*i+1] = gy; g[3*i+2] = gt
        elif e == ENC_POLAR:
            g[3*i]   = gx * math.cos(phi_i) + gy * math.sin(phi_i)
            g[3*i+1] = gx * (-ri * math.sin(phi_i)) + gy * (ri * math.cos(phi_i))
            g[3*i+2] = gt
        else:  # ENC_POLAR_REL
            g[3*i]   = gx * math.cos(phi_i) + gy * math.sin(phi_i)
            g[3*i+1] = gx * (-ri * math.sin(phi_i)) + gy * (ri * math.cos(phi_i)) + gt
            g[3*i+2] = gt
    return g


# ─── Single-encoding local optimizer ─────────────────────────────────────────

def local_opt(xs, ys, ts, encs, R, max_iter=150):
    """
    Minimize exact penalty in the given mixed encoding.
    Returns (xs, ys, ts, exact_E).
    """
    p0 = encode_all(xs, ys, ts, encs)
    lam = 500.0

    result = minimize(
        lambda p: penalty_in_enc(p, encs, R, lam),
        p0,
        jac=lambda p: penalty_grad_in_enc(p, encs, R, lam),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    xs_, ys_, ts_ = decode_all(result.x, encs)
    exact_E = penalty_energy_exact(xs_, ys_, ts_, R)
    return xs_, ys_, ts_, exact_E


# ─── Reformulation Descent (RD) ───────────────────────────────────────────────

def reformulation_descent(xs, ys, ts, R, encodings_sequence=None):
    """
    Cycle through encoding sequences until no improvement.
    Default sequence: all-Cartesian → all-Polar → all-PolarRel → repeat.
    Returns (xs, ys, ts, exact_E).
    """
    if encodings_sequence is None:
        encodings_sequence = [
            [ENC_CART] * N,
            [ENC_POLAR] * N,
            [ENC_POLAR_REL] * N,
        ]

    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
    best_E = penalty_energy_exact(xs, ys, ts, R)

    improved = True
    while improved:
        improved = False
        for encs in encodings_sequence:
            xs_, ys_, ts_, E = local_opt(best_xs, best_ys, best_ts, encs, R)
            if E < best_E - 1e-9:
                best_xs, best_ys, best_ts = xs_, ys_, ts_
                best_E = E
                improved = True

    return best_xs, best_ys, best_ts, best_E


# ─── FSS (Variable Neighborhood Search over formulation space) ────────────────

def fss(xs, ys, ts, R, max_rounds=50, k_max=8):
    """
    Full FSS: VNS over 3^N formulation space.
    Start from (xs, ys, ts), return best found.
    """
    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
    # Start in Cartesian (Mladenović recommends Cartesian as start)
    encs_curr = [ENC_CART] * N
    best_xs, best_ys, best_ts, best_E = local_opt(xs, ys, ts, encs_curr, R)

    for rnd in range(max_rounds):
        improved = False
        for k in range(1, k_max + 1):
            # Flip k random pieces to a different encoding
            idxs = random.sample(range(N), k)
            encs_new = encs_curr[:]
            for i in idxs:
                # Pick a different encoding from current
                others = [e for e in [0, 1, 2] if e != encs_new[i]]
                encs_new[i] = random.choice(others)

            xs_, ys_, ts_, E = local_opt(best_xs, best_ys, best_ts, encs_new, R)

            if E < best_E - 1e-9:
                best_xs, best_ys, best_ts = xs_, ys_, ts_
                best_E = E
                encs_curr = encs_new[:]
                improved = True
                break  # restart with k=1

        if not improved:
            break

    return best_xs, best_ys, best_ts, best_E


# ─── Population Basin Hopping with FSS ────────────────────────────────────────

def hungarian_distance(xs1, ys1, xs2, ys2):
    """
    Optimal assignment distance between two packings.
    O(n^3) Hungarian, handles S_N permutation ambiguity of identical pieces.
    Uses scipy linear_sum_assignment.
    """
    from scipy.optimize import linear_sum_assignment
    cost = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cost[i, j] = (xs1[i]-xs2[j])**2 + (ys1[i]-ys2[j])**2
    row_ind, col_ind = linear_sum_assignment(cost)
    return math.sqrt(cost[row_ind, col_ind].sum())


def center_solution(xs, ys, ts):
    """Shift so MEC center is at origin."""
    scs = [Semicircle(x=round(float(xs[i]),6), y=round(float(ys[i]),6),
                      theta=round(float(ts[i]),6)) for i in range(N)]
    result = validate_and_score(scs)
    if result.mec:
        cx, cy, _ = result.mec
        return xs - cx, ys - cy, ts
    return xs, ys, ts


def score_R(xs, ys, ts):
    scs = [Semicircle(x=round(float(xs[i]),6), y=round(float(ys[i]),6),
                      theta=round(float(ts[i]),6)) for i in range(N)]
    result = validate_and_score(scs)
    return float(result.score) if result.valid else float('inf')


def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    xs = np.array([s['x'] for s in raw])
    ys = np.array([s['y'] for s in raw])
    ts = np.array([s['theta'] for s in raw])
    return xs, ys, ts


def save_best(xs, ys, ts, R):
    sol = [{"x": round(float(xs[i]),6), "y": round(float(ys[i]),6),
            "theta": round(float(ts[i]),6)} for i in range(N)]
    with open(BEST_FILE, 'w') as f:
        json.dump(sol, f, indent=2)


def random_perturbation(xs, ys, ts, sigma=0.15):
    """Random jitter for population diversification."""
    xs2 = xs + np.random.normal(0, sigma, N)
    ys2 = ys + np.random.normal(0, sigma, N)
    ts2 = ts + np.random.normal(0, sigma * 2, N)
    return xs2, ys2, ts2


def pop_basin_hopping_fss(rounds=100, pop_size=12, d_cut=0.3):
    """
    Population Basin Hopping with FSS inner solver.
    Maintains a diverse population of local minima.
    """
    log = open(LOG_FILE, 'w')
    def logprint(msg):
        print(msg, flush=True)
        log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        log.flush()

    xs0, ys0, ts0 = load_best()
    result0 = official_validate(xs0, ys0, ts0)
    global_best_R = float(result0.score)
    logprint(f"=== FSS Population BH | start R={global_best_R:.6f} ===")

    # Initialize population from best with perturbations
    population = []  # list of (xs, ys, ts, R)
    # Seed: best solution
    xs_, ys_, ts_, E = reformulation_descent(xs0, ys0, ts0, global_best_R)
    if E < 1e-4:
        r = score_R(xs_, ys_, ts_)
        population.append((xs_, ys_, ts_, r))
    else:
        population.append((xs0, ys0, ts0, global_best_R))

    # Fill with perturbed variants
    for _ in range(pop_size - 1):
        pxs, pys, pts = random_perturbation(xs0, ys0, ts0, sigma=0.1)
        xs_, ys_, ts_, E = reformulation_descent(pxs, pys, pts, global_best_R)
        if E < 1e-4:
            r = score_R(xs_, ys_, ts_)
            if r < float('inf'):
                # Check dissimilarity from existing population
                min_dist = min(hungarian_distance(xs_, ys_, q[0], q[1]) for q in population)
                if min_dist > d_cut:
                    population.append((xs_, ys_, ts_, r))
                    if len(population) >= pop_size:
                        break

    logprint(f"Population initialized: {len(population)} members")

    for rnd in range(rounds):
        # Pick a random population member to perturb
        base = random.choice(population)
        bxs, bys, bts, bR = base

        # Perturb and apply FSS
        sigma = 0.08 + 0.12 * random.random()
        pxs, pys, pts = random_perturbation(bxs, bys, bts, sigma=sigma)
        xs_, ys_, ts_, E = fss(pxs, pys, pts, global_best_R, max_rounds=10, k_max=5)

        r_fast_label = f"E={E:.3e}"
        if E < 1e-4:
            result = official_validate(xs_, ys_, ts_)
            if result.valid:
                actual_R = float(result.score)
                logprint(f"  [{rnd+1:3d}] VALID R={actual_R:.6f}")

                if actual_R < global_best_R:
                    global_best_R = actual_R
                    xs_, ys_, ts_ = center_solution(xs_, ys_, ts_)
                    save_best(xs_, ys_, ts_, actual_R)
                    logprint(f"  ★ NEW GLOBAL BEST: R = {actual_R:.6f}")

                # Add to population if sufficiently diverse
                min_dist = min(hungarian_distance(xs_, ys_, q[0], q[1]) for q in population)
                if min_dist > d_cut:
                    # Replace worst member if full
                    if len(population) >= pop_size:
                        worst_idx = max(range(len(population)), key=lambda i: population[i][3])
                        if actual_R < population[worst_idx][3]:
                            population[worst_idx] = (xs_, ys_, ts_, actual_R)
                    else:
                        population.append((xs_, ys_, ts_, actual_R))
            else:
                logprint(f"  [{rnd+1:3d}] Shapely reject {r_fast_label}")
        else:
            logprint(f"  [{rnd+1:3d}] {r_fast_label}")

    logprint(f"\n=== Done | best R = {global_best_R:.6f} ===")
    log.close()
    return global_best_R


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--pop', type=int, default=12)
    parser.add_argument('--mode', choices=['rd', 'fss', 'pbh'], default='pbh')
    args = parser.parse_args()

    np.random.seed(int(time.time()) % 10000)
    random.seed(int(time.time()) % 10000)

    if args.mode == 'rd':
        xs0, ys0, ts0 = load_best()
        result0 = official_validate(xs0, ys0, ts0)
        R = float(result0.score)
        print(f"RD from R={R:.6f}")
        t0 = time.time()
        xs_, ys_, ts_, E = reformulation_descent(xs0, ys0, ts0, R)
        t1 = time.time()
        print(f"E={E:.3e}, time={t1-t0:.1f}s")
        if E < 1e-4:
            result = official_validate(xs_, ys_, ts_)
            print(f"Shapely valid: {result.valid}, R={result.score:.6f}")
    elif args.mode == 'fss':
        xs0, ys0, ts0 = load_best()
        result0 = official_validate(xs0, ys0, ts0)
        R = float(result0.score)
        print(f"FSS from R={R:.6f}")
        t0 = time.time()
        xs_, ys_, ts_, E = fss(xs0, ys0, ts0, R, max_rounds=args.rounds, k_max=6)
        t1 = time.time()
        print(f"E={E:.3e}, time={t1-t0:.1f}s")
        if E < 1e-4:
            result = official_validate(xs_, ys_, ts_)
            print(f"Shapely valid: {result.valid}, R={result.score:.6f}")
    else:
        pop_basin_hopping_fss(rounds=args.rounds, pop_size=args.pop)
