#!/usr/bin/env python3
"""
CMA-ES + Numba pipeline for semicircle packing.
Phases:
  1. Pairs warm-start (7 conjugate pairs + singleton) → R ≈ 3.0
  2. CMA-ES with Numba overlap (~100x faster) → break pairs, find R < 3.0
  3. Local refinement from best found

Key insights from Fejes Tóth (1971):
  - Pairing is suboptimal: semicircles can exploit flat edges more efficiently
  - Boundary semicircles should face outward (flat edge tangent to enclosing circle)
  - Interior pairs should break when it reduces R
"""

import sys, os, json, math, time, random
import numpy as np
os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from numba import njit, prange
import cma

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle

BEST_FILE = 'best_solution.json'
N = 15


# ─────────────────────────────────────────────────────────────────
# Numba-accelerated overlap penalty
# ─────────────────────────────────────────────────────────────────

@njit(fastmath=True, cache=True)
def semicircle_overlap_penalty(xi, yi, ti, xj, yj, tj):
    """
    Fast analytical overlap proxy.
    Key insight: two semicircles at the same center (d~0) with theta_j = theta_i + pi
    are a valid conjugate pair (flat sides touching) — zero overlap.
    """
    dx = xj - xi
    dy = yj - yi
    d = math.sqrt(dx*dx + dy*dy)
    if d >= 2.0:
        return 0.0

    # Orientation normals
    nix = math.cos(ti); niy = math.sin(ti)
    njx = math.cos(tj); njy = math.sin(tj)

    # Check for conjugate pair: same center + opposite orientations
    # dot(ni, nj) = cos(ti-tj); conjugate pair has ti-tj = pi, so dot = -1
    dot_nn = nix*njx + niy*njy  # -1 for conjugate pair
    if d < 0.15 and dot_nn < -0.95:
        # Conjugate pair: flat sides touching, zero actual overlap
        return 0.0

    # Base penetration
    penetration = (2.0 - d) ** 2

    if d < 1e-10:
        facing_i = 0.0; facing_j = 0.0
    else:
        facing_i = (dx * nix + dy * niy) / d
        facing_j = -(dx * njx + dy * njy) / d

    # Sigmoid gates
    gate_i = 1.0 / (1.0 + math.exp(-12.0 * facing_i))
    gate_j = 1.0 / (1.0 + math.exp(-12.0 * facing_j))

    return penetration * gate_i * gate_j


@njit(fastmath=True, cache=True)
def total_overlap_numba(xs, ys, ts):
    """Sum of pairwise overlap penalties over all 105 pairs."""
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            total += semicircle_overlap_penalty(xs[i], ys[i], ts[i],
                                                 xs[j], ys[j], ts[j])
    return total


@njit(fastmath=True, cache=True)
def fast_mec_radius(xs, ys, ts, n_arc=32):
    """Approximate MEC from sampled boundary points via iterative minimax."""
    n_pts = N * (n_arc + 2)
    pts_x = np.empty(n_pts)
    pts_y = np.empty(n_pts)
    k = 0
    half_pi = math.pi / 2.0
    for i in range(N):
        x, y, t = xs[i], ys[i], ts[i]
        for j in range(n_arc):
            angle = t - half_pi + j * math.pi / (n_arc - 1)
            pts_x[k] = x + math.cos(angle)
            pts_y[k] = y + math.sin(angle)
            k += 1
        pts_x[k] = x + math.cos(t + half_pi); pts_y[k] = y + math.sin(t + half_pi); k += 1
        pts_x[k] = x + math.cos(t - half_pi); pts_y[k] = y + math.sin(t - half_pi); k += 1
    # Iterative minimax
    cx = 0.0; cy = 0.0
    for i in range(n_pts):
        cx += pts_x[i]; cy += pts_y[i]
    cx /= n_pts; cy /= n_pts
    for _ in range(40):
        max_d2 = 0.0; far = 0
        for i in range(n_pts):
            d2 = (pts_x[i]-cx)**2 + (pts_y[i]-cy)**2
            if d2 > max_d2:
                max_d2 = d2; far = i
        cx = 0.7*cx + 0.3*pts_x[far]
        cy = 0.7*cy + 0.3*pts_y[far]
    max_r = 0.0
    for i in range(n_pts):
        r = math.sqrt((pts_x[i]-cx)**2 + (pts_y[i]-cy)**2)
        if r > max_r: max_r = r
    return max_r


@njit(fastmath=True, cache=True)
def objective_numba(v, lam):
    """Full objective: fast_MEC + lam * total_overlap."""
    xs = v[0:N]; ys = v[N:2*N]; ts = v[2*N:3*N]
    mec = fast_mec_radius(xs, ys, ts)
    ovlp = total_overlap_numba(xs, ys, ts)
    return mec + lam * ovlp


def pack_v(raw):
    """JSON list → flat numpy vector [x0..x14, y0..y14, t0..t14]."""
    xs = np.array([d['x'] for d in raw])
    ys = np.array([d['y'] for d in raw])
    ts = np.array([d['theta'] for d in raw])
    return np.concatenate([xs, ys, ts])


def unpack_v(v):
    return [{'x': float(v[i]), 'y': float(v[N+i]), 'theta': float(v[2*N+i])} for i in range(N)]


def exact_score(v):
    raw = unpack_v(v)
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    return validate_and_score(sol)


def center_v(v):
    result = exact_score(v)
    if not result.valid:
        return v, None
    cx, cy = result.mec[0], result.mec[1]
    v2 = v.copy()
    v2[0:N] -= cx; v2[N:2*N] -= cy
    return v2, result.score


def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    return pack_v(raw)


def save_best(v):
    raw = unpack_v(v)
    centered, score = center_v(pack_v(raw))
    if score is None:
        return None
    centered_raw = unpack_v(centered)
    with open(BEST_FILE, 'w') as f:
        json.dump(centered_raw, f, indent=2)
    # Visualization
    try:
        sol = [Semicircle(d['x'], d['y'], d['theta']) for d in centered_raw]
        r = validate_and_score(sol)
        from src.semicircle_packing.visualization import plot_packing
        plot_packing(sol, r.mec, save_path='best_solution.png')
    except:
        pass
    return score


# ─────────────────────────────────────────────────────────────────
# Starting geometries
# ─────────────────────────────────────────────────────────────────

def pairs_warm_start(scale=1.02, noise=0.0):
    """7 conjugate pairs in hex + singleton."""
    centers = [(0.0, 0.0)]
    for i in range(6):
        a = i * math.pi / 3
        centers.append((scale * 2.0 * math.cos(a), scale * 2.0 * math.sin(a)))
    
    raw = []
    for cx, cy in centers:
        if noise > 0:
            cx += random.gauss(0, noise); cy += random.gauss(0, noise)
        angle = random.uniform(0, math.pi)
        raw.append({'x': cx, 'y': cy, 'theta': angle})
        raw.append({'x': cx, 'y': cy, 'theta': angle + math.pi})
    
    # Singleton in a gap
    from src.semicircle_packing.geometry import semicircles_overlap
    placed = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    best_s = None; best_d = float('inf')
    for r_try in np.linspace(0.5, 3.5, 12):
        for a_try in np.linspace(0, 2*math.pi, 36, endpoint=False):
            x = r_try * math.cos(a_try); y = r_try * math.sin(a_try)
            for t_try in np.linspace(0, 2*math.pi, 12, endpoint=False):
                sc = Semicircle(x, y, t_try)
                ok = all(
                    not ((x-p.x)**2+(y-p.y)**2 <= 4.01 and semicircles_overlap(sc, p))
                    for p in placed
                )
                if ok and r_try < best_d:
                    best_d = r_try; best_s = {'x': x, 'y': y, 'theta': t_try}
    if best_s is None:
        return None
    raw.append(best_s)
    return pack_v(raw)


def boundary_start(scale=1.02):
    """
    Alternative: orient some semicircles flat-edge-outward (boundary fillers).
    Mix of interior pairs and boundary-oriented singletons.
    """
    raw = []
    # 5 pairs in interior hex (10 semicircles)
    interior_centers = [(0.0, 0.0)]
    for i in range(4):
        a = i * math.pi / 2
        interior_centers.append((scale * 1.8 * math.cos(a), scale * 1.8 * math.sin(a)))
    for cx, cy in interior_centers:
        angle = random.uniform(0, math.pi)
        raw.append({'x': cx, 'y': cy, 'theta': angle})
        raw.append({'x': cx, 'y': cy, 'theta': angle + math.pi})
    # 5 singletons around boundary, flat edge outward
    r_boundary = scale * 2.5
    for i in range(5):
        a = i * 2 * math.pi / 5
        x = r_boundary * math.cos(a); y = r_boundary * math.sin(a)
        theta = a + math.pi  # flat edge facing outward (curved side inward)
        raw.append({'x': x, 'y': y, 'theta': theta})
    return pack_v(raw)


# ─────────────────────────────────────────────────────────────────
# CMA-ES optimizer
# ─────────────────────────────────────────────────────────────────

def run_cmaes(v0, lam=100.0, sigma0=0.3, maxfevals=300000,
              popsize=100, label="cmaes", global_best=float('inf')):
    """
    CMA-ES with fixed penalty lambda.
    Returns best valid solution found.
    """
    best_exact = global_best
    best_v = None
    eval_count = [0]
    last_exact_check = [0]

    def objective(x):
        v = np.array(x)
        return objective_numba(v, lam)

    opts = cma.CMAOptions()
    opts['maxfevals'] = maxfevals
    opts['popsize'] = popsize
    opts['verbose'] = -9
    opts['tolx'] = 1e-8
    opts['tolfun'] = 1e-8
    opts['CMA_active'] = True

    es = cma.CMAEvolutionStrategy(v0.tolist(), sigma0, opts)

    t0 = time.time()
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        eval_count[0] += len(solutions)

        # Periodically exact-score the best candidate
        if eval_count[0] - last_exact_check[0] >= 2000:
            last_exact_check[0] = eval_count[0]
            best_x = np.array(es.result.xbest)
            result = exact_score(best_x)
            if result.valid and result.score < best_exact:
                best_exact = result.score
                best_v = best_x.copy()
                elapsed = time.time() - t0
                print(f"  [{label}] evals={eval_count[0]} exact={best_exact:.6f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    # Final check
    if es.result.xbest is not None:
        best_x = np.array(es.result.xbest)
        result = exact_score(best_x)
        if result.valid and result.score < best_exact:
            best_exact = result.score
            best_v = best_x.copy()

    approx = es.result.fbest
    print(f"  [{label}] Done {elapsed:.0f}s | evals={eval_count[0]} approx={approx:.5f} exact={'N/A' if best_v is None else f'{best_exact:.6f}'}", flush=True)
    return best_v, best_exact


def run_cmaes_auglag(v0, sigma0=0.3, maxfevals=400000,
                     popsize=150, label="cmaes_aug", global_best=float('inf')):
    """
    CMA-ES with augmented Lagrangian: adaptively adjusts lambda.
    More principled than fixed lambda.
    """
    lam = 10.0
    best_exact = global_best
    best_v = None
    t0 = time.time()

    # Phase 1: explore (low lambda, large sigma, many fevals)
    # Phase 2: tighten (high lambda, small sigma)
    schedule = [
        (lam,           sigma0,        maxfevals * 3 // 10),
        (lam * 50,      sigma0 * 0.5,  maxfevals * 3 // 10),
        (lam * 2000,    sigma0 * 0.2,  maxfevals * 2 // 10),
        (lam * 100000,  sigma0 * 0.05, maxfevals * 2 // 10),
    ]

    for phase_num, (lam_phase, sigma_phase, phase_fevals) in enumerate(schedule):
        def objective(x, _lam=lam_phase):
            v = np.array(x)
            return objective_numba(v, _lam)

        opts = cma.CMAOptions()
        opts['maxfevals'] = phase_fevals
        opts['popsize'] = popsize
        opts['verbose'] = -9
        opts['CMA_active'] = True
        opts['tolx'] = 1e-9

        es = cma.CMAEvolutionStrategy(v0.tolist(), sigma_phase, opts)
        while not es.stop():
            sols = es.ask()
            fits = [objective(x) for x in sols]
            es.tell(sols, fits)

        if es.result.xbest is not None:
            v0 = np.array(es.result.xbest)
            result = exact_score(v0)
            ovlp = total_overlap_numba(v0[0:N], v0[N:2*N], v0[2*N:3*N])
            elapsed = time.time() - t0
            print(f"  [{label}] phase={phase_num+1}/4 lam={lam_phase:.0f} ovlp={ovlp:.6f} "
                  f"exact={'INVALID' if not result.valid else f'{result.score:.6f}'} ({elapsed:.0f}s)", flush=True)
            if result.valid and result.score < best_exact:
                best_exact = result.score
                best_v = v0.copy()

    return best_v, best_exact


# ─────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Warm up Numba JIT
    print("Warming up Numba JIT...", flush=True)
    dummy = np.zeros(3*N)
    dummy[0:N] = np.linspace(-2, 2, N)
    dummy[N:2*N] = np.linspace(-2, 2, N)
    _ = objective_numba(dummy, 100.0)
    _ = total_overlap_numba(dummy[0:N], dummy[N:2*N], dummy[2*N:3*N])

    # Benchmark
    t0 = time.time()
    for _ in range(10000):
        objective_numba(dummy, 100.0)
    elapsed = time.time() - t0
    print(f"Numba speed: {10000/elapsed:.0f} evals/sec", flush=True)

    global_best_v = load_best()
    result0 = exact_score(global_best_v)
    global_best = result0.score
    print(f"Starting best: {global_best:.6f}", flush=True)

    run_num = 0
    t_total = time.time()
    max_runtime = 3600

    starts = [
        ('from_best_s',  lambda: global_best_v + np.random.randn(3*N) * 0.03),
        ('from_best_m',  lambda: global_best_v + np.random.randn(3*N) * 0.08),
        ('pairs_tight',  lambda: pairs_warm_start(scale=1.02, noise=0.0)),
        ('from_best_l',  lambda: global_best_v + np.random.randn(3*N) * 0.20),
        ('pairs_noisy',  lambda: pairs_warm_start(scale=1.05, noise=0.05)),
        ('from_best_m2', lambda: global_best_v + np.random.randn(3*N) * 0.08),
        ('boundary',     lambda: boundary_start(scale=1.02)),
        ('pairs_loose',  lambda: pairs_warm_start(scale=1.10, noise=0.03)),
    ]

    while time.time() - t_total < max_runtime:
        run_num += 1
        label, init_fn = starts[(run_num - 1) % len(starts)]
        label = f"{label}_{run_num}"

        print(f"\n{'='*60}\nRun {run_num}: {label}\n{'='*60}", flush=True)

        try:
            v0 = init_fn()
            if v0 is None:
                print("  Failed to build start, skipping", flush=True)
                continue

            # Quick validity check
            result_init = exact_score(v0)
            ovlp0 = total_overlap_numba(v0[0:N], v0[N:2*N], v0[2*N:3*N])
            approx0 = objective_numba(v0, 0.0)
            print(f"  Init: approx_mec={approx0:.4f} overlap={ovlp0:.4f} exact={'valid '+str(round(result_init.score,4)) if result_init.valid else 'invalid'}", flush=True)

            # Run augmented Lagrangian CMA-ES
            best_v, best_score = run_cmaes_auglag(
                v0, sigma0=0.25, maxfevals=500000,
                popsize=120, label=label,
                global_best=global_best
            )

            if best_v is not None and best_score < global_best:
                centered_v, centered_score = center_v(best_v)
                if centered_score is not None:
                    global_best = centered_score
                    global_best_v = centered_v
                    saved = save_best(best_v)
                    print(f"\n*** NEW BEST: {global_best:.6f} ***\n", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()

    print(f"\nFinal best: {global_best:.6f}", flush=True)
