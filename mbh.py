"""
mbh.py — Monotonic Basin Hopping optimizer for semicircle packing.

Architecture (per the research report):
  - Bilevel: outer binary search on R, inner feasibility minimization
  - Inner loop: L-BFGS-B with analytical phi-function gradients
  - Perturbations: orientation-flip, swap, shift-worst, cluster-reflect
  - Population: maintains K diverse solutions simultaneously

Usage:
    python mbh.py [--r-start 3.07] [--r-min 2.85] [--rounds 50] [--pop 8]
"""

import json, math, time, random, argparse, os, sys, importlib.util
import numpy as np
from scipy.optimize import minimize
from phi import (
    penalty_energy_flat, penalty_gradient_flat,
    is_feasible, penalty_energy, phi_all_pairs, phi_containment
)
from exact_dist import (
    semicircle_signed_dist, all_pairs_signed_dist,
    is_feasible_exact, penalty_energy_exact, containment_signed_dist
)

# Load Shapely-based official scorer for final validation
def _load_official_scorer():
    try:
        spec = importlib.util.spec_from_file_location('fr', os.path.join(os.path.dirname(__file__), 'fast_run.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.official_score
    except Exception:
        return None

_official_score = _load_official_scorer()

def shapely_valid(xs, ys, ts):
    """Cross-check with Shapely before accepting a solution."""
    if _official_score is None:
        return True  # can't check, assume ok
    result = _official_score(xs, ys, ts)
    return result.valid

BEST_FILE  = os.path.join(os.path.dirname(__file__), 'best_solution.json')
LOG_FILE   = os.path.join(os.path.dirname(__file__), 'mbh_log.txt')
N = 15


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    xs = np.array([s['x'] for s in raw])
    ys = np.array([s['y'] for s in raw])
    ts = np.array([s['theta'] for s in raw])
    R  = float(np.max(np.sqrt(xs**2 + ys**2)) + 1.0)  # conservative estimate
    return xs, ys, ts


def save_best(xs, ys, ts, R, log=True):
    data = [{'x': float(xs[i]), 'y': float(ys[i]), 'theta': float(ts[i])}
            for i in range(N)]
    with open(BEST_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    msg = f"[NEW BEST] R = {R:.6f}"
    print(msg, flush=True)
    if log:
        with open(LOG_FILE, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")


def pack(xs, ys, ts):
    p = np.zeros(3 * N)
    p[0::3] = xs; p[1::3] = ys; p[2::3] = ts
    return p

def unpack(p):
    return p[0::3], p[1::3], p[2::3]

def score_R_fast(xs, ys, ts):
    """Fast R estimate: max distance from origin to any critical containment point."""
    R = 0.0
    for i in range(N):
        pts = np.array([
            [xs[i] + np.cos(ts[i]), ys[i] + np.sin(ts[i])],
            [xs[i] - np.cos(ts[i]), ys[i] - np.sin(ts[i])],
            [xs[i] - np.sin(ts[i]), ys[i] + np.cos(ts[i])],
        ])
        R = max(R, float(np.max(np.sqrt(pts[:,0]**2 + pts[:,1]**2))))
    return R

def score_R(xs, ys, ts):
    """Official Shapely MEC score. Only call when we believe solution is valid."""
    if _official_score is None:
        return score_R_fast(xs, ys, ts)
    result = _official_score(xs, ys, ts)
    if result.valid and result.score is not None:
        return float(result.score)
    return float('inf')


# ── L-BFGS local minimization ─────────────────────────────────────────────────

def lbfgs_refine(xs, ys, ts, R, max_iter=500, ftol=1e-15, gtol=1e-10):
    """
    Minimize phi penalty only at fixed R (feasibility problem).
    The score gradient is NOT included — it causes thin-sliver traps.
    Instead, the MBH outer loop controls R via binary search.
    Returns (xs, ys, ts, exact_energy).
    """
    p0 = pack(xs, ys, ts)
    lam = 500.0

    def f(p):
        return lam * penalty_energy_flat(p, R)

    def g(p):
        return lam * penalty_gradient_flat(p, R)

    result = minimize(f, p0, jac=g, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    rxs, rys, rts = unpack(result.x)
    exact_E = penalty_energy_exact(rxs, rys, rts, R)
    return rxs, rys, rts, exact_E


def repair_thin_slivers(xs, ys, ts, R, n_passes=10, push=0.003):
    """
    Identify pairs where exact_dist < 0 (thin-sliver overlap)
    and push them apart slightly along the center-to-center direction.
    """
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    for _ in range(n_passes):
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                dc2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                if dc2 >= 4.0:
                    continue
                d = semicircle_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
                if d < -1e-6:
                    # Push apart along c_i → c_j direction
                    dc = math.sqrt(dc2) + 1e-12
                    dx_n = (xs[j]-xs[i])/dc
                    dy_n = (ys[j]-ys[i])/dc
                    xs[i] -= push * dx_n
                    ys[i] -= push * dy_n
                    xs[j] += push * dx_n
                    ys[j] += push * dy_n
                    moved = True
        if not moved:
            break
    return xs, ys, ts


# ── Perturbation operators ────────────────────────────────────────────────────

def perturb_orientation_flip(xs, ys, ts, n_flip=1):
    """Flip orientation of n_flip randomly chosen semicircles by π."""
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    idxs = random.sample(range(N), n_flip)
    for i in idxs:
        ts[i] = (ts[i] + math.pi) % (2 * math.pi)
    return xs, ys, ts


def perturb_swap(xs, ys, ts):
    """Swap full placement (x, y, θ) of two random semicircles."""
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    i, j = random.sample(range(N), 2)
    xs[i], xs[j] = xs[j], xs[i]
    ys[i], ys[j] = ys[j], ys[i]
    ts[i], ts[j] = ts[j], ts[i]
    return xs, ys, ts


def perturb_shift_worst(xs, ys, ts, R):
    """
    CEGP: relocate the most-overlapping semicircle to a random position.
    """
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    # Find worst: highest overlap penalty contribution
    penalties = []
    for i in range(N):
        p = 0.0
        for j in range(N):
            if i == j: continue
            from phi import phi_pair
            phi = phi_pair(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if phi < 0: p += phi * phi
        c = phi_containment(xs[i], ys[i], ts[i], R)
        if c < 0: p += c * c
        penalties.append(p)
    worst = int(np.argmax(penalties))

    # Relocate to random feasible-ish position
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
    """Reflect a random subset (~half) of semicircles about a random axis."""
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    angle = random.uniform(0, math.pi)  # reflection axis angle
    nx, ny = math.cos(angle), math.sin(angle)  # axis normal

    k = random.randint(3, N // 2 + 1)
    idxs = random.sample(range(N), k)
    for i in idxs:
        # Reflect position about line through origin with direction (nx, ny)
        dot = xs[i] * nx + ys[i] * ny
        xs[i] = 2 * dot * nx - xs[i]
        ys[i] = 2 * dot * ny - ys[i]
        # Reflect orientation
        ts[i] = 2 * angle - ts[i]
    return xs, ys, ts


def perturb_random_jitter(xs, ys, ts, sigma=0.15):
    """Small random jitter on all positions and orientations."""
    xs = xs + np.random.randn(N) * sigma
    ys = ys + np.random.randn(N) * sigma
    ts = ts + np.random.randn(N) * sigma * 0.5
    return xs, ys, ts


def perturb_lns(xs, ys, ts, R, n_remove=3):
    """
    Large Neighborhood Search (Destroy-and-Repair):
    Remove the n_remove worst-placed semicircles (highest overlap penalty),
    then greedily re-insert them at the best available gap positions.
    This creates genuinely large topological jumps while preserving
    the good structure of the remaining 12-13 semicircles.
    """
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()

    # Score each semicircle by its contribution to overlap/containment penalty
    scores = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            d = semicircle_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < 0: scores[i] += d * d
        c = containment_signed_dist(xs[i], ys[i], ts[i], R)
        if c < 0: scores[i] += c * c

    # Remove worst n_remove
    worst_idxs = np.argsort(scores)[-n_remove:]

    # Greedily re-insert at positions that maximize minimum clearance
    for idx in worst_idxs:
        best_pos = None; best_score = -np.inf
        for _ in range(120):
            r = random.uniform(0, R - 1.0)
            angle = random.uniform(0, 2 * math.pi)
            nx, ny = r * math.cos(angle), r * math.sin(angle)
            nt = random.uniform(0, 2 * math.pi)

            # Check containment
            if containment_signed_dist(nx, ny, nt, R) < 0:
                continue

            # Score = minimum clearance from all other semicircles
            min_clearance = float('inf')
            for j in range(N):
                if j == idx: continue
                d = semicircle_signed_dist(nx, ny, nt, xs[j], ys[j], ts[j])
                if d < min_clearance:
                    min_clearance = d

            if min_clearance > best_score:
                best_score = min_clearance
                best_pos = (nx, ny, nt)

        if best_pos:
            xs[idx], ys[idx], ts[idx] = best_pos

    return xs, ys, ts


def perturb_scale_expand(xs, ys, ts, factor=1.08):
    """
    Scale all positions outward by factor, then let L-BFGS compress back.
    Creates a large hop that can escape tight local minima by first
    relaxing all overlap constraints, then re-optimizing.
    """
    xs = xs * factor
    ys = ys * factor
    # Keep orientations — they encode topology, not scale
    return xs, ys, ts.copy()


def perturb_wall_slide(xs, ys, ts, R, n_slide=2, dphi=0.3):
    """
    FSS-inspired perturbation: slide n_slide pieces along the container wall.
    Changes φ (angular position) while holding θ_rel = θ - φ fixed.
    This moves a piece around the container wall without changing its
    orientation relative to the radial direction — a move that requires
    coordinated (x,y,θ) changes in Cartesian space.
    """
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    idxs = random.sample(range(N), min(n_slide, N))
    for i in idxs:
        r = math.sqrt(xs[i]**2 + ys[i]**2)
        if r < 0.1:
            continue  # can't slide something at the center
        phi = math.atan2(ys[i], xs[i])
        theta_rel = ts[i] - phi  # relative orientation
        # Slide: change phi by dphi (signed random)
        dphi_sign = random.choice([-1, 1]) * dphi * (0.5 + random.random())
        new_phi = phi + dphi_sign
        xs[i] = r * math.cos(new_phi)
        ys[i] = r * math.sin(new_phi)
        ts[i] = theta_rel + new_phi  # maintain θ_rel
    return xs, ys, ts


def perturb_herringbone_seed(xs, ys, ts, R):
    """
    Attempt to create flat-edge-aligned clusters (herringbone/brick pattern).
    Picks a random piece and places a copy rotated by π nearby with flat edges touching.
    Based on Fejes Tóth's insight: flat edges pack more densely than curved.
    """
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    i = random.randrange(N)
    j = random.randrange(N)
    if i == j:
        return xs, ys, ts

    xi, yi, ti = xs[i], ys[i], ts[i]
    # Place j anti-parallel to i with flat edges touching
    # Anti-parallel: tj = ti + π
    # Flat edge contact: center-to-center distance = 0 (flat edges coincide, offset perpendicular)
    # Place j at: xi + sin(ti)*offset, yi - cos(ti)*offset, ti + π
    # with offset along the flat edge direction
    tx, ty = -math.sin(ti), math.cos(ti)  # tangent (along flat edge)
    offset = (random.random() - 0.5) * 1.5
    xs[j] = xi + tx * offset
    ys[j] = yi + ty * offset
    ts[j] = ti + math.pi
    return xs, ys, ts


PERTURBATIONS = ['flip', 'swap', 'shift_worst', 'reflect', 'jitter', 'lns', 'expand',
                 'wall_slide', 'herringbone']

def apply_perturbation(xs, ys, ts, R, kind=None):
    if kind is None:
        kind = random.choice(PERTURBATIONS)
    if kind == 'flip':
        n = random.choice([1, 2, 3])
        return perturb_orientation_flip(xs, ys, ts, n)
    elif kind == 'swap':
        return perturb_swap(xs, ys, ts)
    elif kind == 'shift_worst':
        return perturb_shift_worst(xs, ys, ts, R)
    elif kind == 'reflect':
        return perturb_cluster_reflect(xs, ys, ts)
    elif kind == 'lns':
        n = random.choice([2, 3, 4])
        return perturb_lns(xs, ys, ts, R, n_remove=n)
    elif kind == 'expand':
        factor = random.uniform(1.04, 1.15)
        return perturb_scale_expand(xs, ys, ts, factor)
    elif kind == 'wall_slide':
        n = random.choice([1, 2, 3])
        return perturb_wall_slide(xs, ys, ts, R, n_slide=n)
    elif kind == 'herringbone':
        return perturb_herringbone_seed(xs, ys, ts, R)
    else:
        return perturb_random_jitter(xs, ys, ts)


# ── Feasibility check at fixed R ─────────────────────────────────────────────

def check_feasibility(xs0, ys0, ts0, R, n_restarts=20, verbose=False):
    """
    Minimize penalty energy at fixed R from multiple starting points.
    Returns (best_xs, best_ys, best_ts, best_energy).
    Energy = 0 means feasible.
    """
    best_E = float('inf')
    best_sol = (xs0.copy(), ys0.copy(), ts0.copy())

    # Start from given solution
    xs, ys, ts, E = lbfgs_refine(xs0, ys0, ts0, R)
    if E < best_E:
        best_E = E
        best_sol = (xs, ys, ts)
    if verbose:
        print(f"  Start 0: E={E:.4e}", flush=True)

    for restart in range(n_restarts - 1):
        kind = random.choice(PERTURBATIONS)
        px, py, pt = apply_perturbation(*best_sol, R, kind=kind)
        xs, ys, ts, E = lbfgs_refine(px, py, pt, R)
        if verbose:
            print(f"  Restart {restart+1} ({kind}): E={E:.4e}", flush=True)
        if E < best_E:
            best_E = E
            best_sol = (xs, ys, ts)
            if best_E < 1e-10:
                break  # feasible — done

    return best_sol[0], best_sol[1], best_sol[2], best_E


# ── Population diversity ──────────────────────────────────────────────────────

def dissimilarity(xs1, ys1, xs2, ys2):
    """Sum of squared position differences (min over permutations is hard; use sorted by angle)."""
    return float(np.sum((xs1 - xs2)**2 + (ys1 - ys2)**2))


def most_similar_idx(pop, xs, ys):
    sims = [dissimilarity(p[0], p[1], xs, ys) for p in pop]
    return int(np.argmin(sims))


# ── Main MBH loop ─────────────────────────────────────────────────────────────

def run_mbh(r_start=3.07, r_min=2.85, r_step=0.01, rounds_per_R=30,
            pop_size=8, n_restarts=15, verbose=True):
    """
    Bilevel monotonic basin hopping.
    Outer: binary-search R downward from r_start.
    Inner: MBH feasibility search at fixed R.
    """
    xs0, ys0, ts0 = load_best()
    global_best_R = score_R(xs0, ys0, ts0)

    log = open(LOG_FILE, 'a')
    def logprint(msg):
        print(msg, flush=True)
        log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        log.flush()

    logprint(f"=== MBH started | initial R={global_best_R:.6f} | target R≤{r_min} ===")

    # Initialize population from best solution with perturbations
    population = [(xs0.copy(), ys0.copy(), ts0.copy(), global_best_R)]
    for _ in range(pop_size - 1):
        px, py, pt = apply_perturbation(xs0, ys0, ts0, global_best_R)
        population.append((px, py, pt, global_best_R))

    # Outer R loop: try decreasing R values
    R = r_start
    while R >= r_min:
        logprint(f"\n── Testing R = {R:.4f} ──────────────────────")
        feasible_found = False

        # MBH rounds at this R
        for rnd in range(rounds_per_R):
            # Pick a population member to perturb
            base_idx = random.randrange(len(population))
            bxs, bys, bts, _ = population[base_idx]

            # Perturb and refine
            kind = random.choice(PERTURBATIONS)
            px, py, pt = apply_perturbation(bxs, bys, bts, R, kind=kind)
            rxs, rys, rts, E = lbfgs_refine(px, py, pt, R)

            actual_R = score_R_fast(rxs, rys, rts)
            logprint(f"  Round {rnd+1:3d} ({kind:12s}): E={E:.3e}, R_fast={actual_R:.5f}")

            # Shapely is the final ground truth
            if E < 1e-4:
                shapely_result = _official_score(rxs, rys, rts) if _official_score else None
                is_valid = shapely_result is not None and shapely_result.valid
                official_R = float(shapely_result.score) if (is_valid and shapely_result.score) else float('inf')
                if not is_valid:
                    errs = shapely_result.errors[:2] if shapely_result else []
                    logprint(f"  ✗ Shapely reject R={R:.4f} E={E:.2e} R_fast={actual_R:.4f}: {errs}")
                    continue  # Shapely says still overlapping, keep trying
                logprint(f"  ✓ FEASIBLE at R={R:.4f}! Official R={official_R:.6f}")
                feasible_found = True
                actual_R = official_R

                if actual_R < global_best_R:
                    global_best_R = actual_R
                    save_best(rxs, rys, rts, actual_R)
                    logprint(f"  ★ NEW GLOBAL BEST: R = {actual_R:.6f}")

                # Update population (monotonic: only accept improvements)
                sim_idx = most_similar_idx(population, rxs, rys)
                if actual_R < population[sim_idx][3]:
                    population[sim_idx] = (rxs, rys, rts, actual_R)

                # Squeeze: binary-search R downward from actual_R
                lo, hi = r_min, actual_R
                for squeeze in range(15):
                    mid_R = (lo + hi) / 2
                    if hi - lo < 0.002: break
                    # Multiple L-BFGS passes with increasing iterations to settle complex configs
                    sx, sy, st = rxs.copy(), rys.copy(), rts.copy()
                    for n_iter in [300, 500, 1000]:
                        sx, sy, st, sE = lbfgs_refine(sx, sy, st, mid_R, max_iter=n_iter)
                        if sE < 1e-6: break
                    if sE < 1e-4 and shapely_valid(sx, sy, st):
                        s_R = score_R(sx, sy, st)
                        logprint(f"    Squeeze R={mid_R:.4f} → feasible (R={s_R:.6f})")
                        if s_R < global_best_R:
                            global_best_R = s_R
                            save_best(sx, sy, st, s_R)
                            logprint(f"  ★ NEW GLOBAL BEST: R = {s_R:.6f}")
                        rxs, rys, rts = sx, sy, st
                        actual_R = s_R
                        hi = mid_R  # can go tighter
                    else:
                        lo = mid_R  # too tight, relax
                break  # found feasible, move to smaller R

            elif E < penalty_energy(bxs, bys, bts, R) + 0.01:
                # Monotonic accept: only if energy improved
                sim_idx = most_similar_idx(population, rxs, rys)
                if E < 1e4:  # don't add garbage
                    population[sim_idx] = (rxs, rys, rts, R)

        if not feasible_found:
            logprint(f"  ✗ Not feasible at R={R:.4f} after {rounds_per_R} rounds")
            break  # can't go tighter from this population state
        else:
            R = min(actual_R - r_step, R - r_step)

    logprint(f"\n=== MBH done | best R = {global_best_R:.6f} ===")
    log.close()
    return global_best_R


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r-start',  type=float, default=3.07)
    parser.add_argument('--r-min',    type=float, default=2.85)
    parser.add_argument('--r-step',   type=float, default=0.02)
    parser.add_argument('--rounds',   type=int,   default=30)
    parser.add_argument('--pop',      type=int,   default=8)
    parser.add_argument('--restarts', type=int,   default=15)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()

    if args.validate_only:
        # Just validate phi.py on the best known solution
        import subprocess, sys
        result = subprocess.run([sys.executable, 'phi.py'], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
    else:
        run_mbh(
            r_start=args.r_start,
            r_min=args.r_min,
            r_step=args.r_step,
            rounds_per_R=args.rounds,
            pop_size=args.pop,
            n_restarts=args.restarts,
            verbose=True,
        )
