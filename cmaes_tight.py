"""
cmaes_tight.py — CMA-ES refinement starting from best solution.

Uses penalty_energy_exact directly (no phi function lies).
Gradient-free, so no false signal from phi conservatism.

Targets: beat current best of 3.010195.
"""

import json, math, time, os, sys, random
import numpy as np
import cma

sys.path.insert(0, os.path.dirname(__file__))
from exact_dist import penalty_energy_exact
from overnight import official_validate
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle

N = 15
BEST_FILE = os.path.join(os.path.dirname(__file__), 'best_solution.json')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'cmaes_tight_log.txt')


def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    xs = np.array([s['x'] for s in raw])
    ys = np.array([s['y'] for s in raw])
    ts = np.array([s['theta'] for s in raw])
    return xs, ys, ts


def save_best(xs, ys, ts, R):
    sol = [{"x": round(float(xs[i]), 6), "y": round(float(ys[i]), 6),
            "theta": round(float(ts[i]), 6)} for i in range(N)]
    with open(BEST_FILE, 'w') as f:
        json.dump(sol, f, indent=2)
    print(f"  ★ Saved new best: R = {R:.6f}", flush=True)


def official_score(xs, ys, ts):
    scs = [Semicircle(x=round(float(xs[i]), 6), y=round(float(ys[i]), 6),
                      theta=round(float(ts[i]), 6)) for i in range(N)]
    result = validate_and_score(scs)
    return result


def pack(xs, ys, ts):
    return np.concatenate([xs, ys, ts])


def unpack(p):
    return p[:N], p[N:2*N], p[2*N:]


def score_fast(xs, ys, ts):
    """Quick R estimate: max distance from origin of critical points."""
    best = 0.0
    for i in range(N):
        for px, py in [
            (xs[i] + math.cos(ts[i]), ys[i] + math.sin(ts[i])),
            (xs[i] - math.cos(ts[i]), ys[i] - math.sin(ts[i])),
            (xs[i] - math.sin(ts[i]), ys[i] + math.cos(ts[i])),
        ]:
            d = math.sqrt(px*px + py*py)
            if d > best:
                best = d
    return best


def center_solution(xs, ys, ts):
    """Shift solution so MEC center is at origin."""
    result = official_score(xs, ys, ts)
    if result.mec:
        cx, cy, _ = result.mec
        return xs - cx, ys - cy, ts
    return xs, ys, ts


def objective(p, R, lam=1000.0):
    xs, ys, ts = unpack(p)
    r = score_fast(xs, ys, ts)
    pen = penalty_energy_exact(xs, ys, ts, R)
    return r + lam * pen


def run_cmaes_round(xs0, ys0, ts0, R_target, sigma0=0.02, maxiter=2000):
    """Run CMA-ES from given start, targeting R_target."""
    p0 = pack(xs0, ys0, ts0)

    opts = cma.CMAOptions()
    opts['maxiter'] = maxiter
    opts['tolx'] = 1e-10
    opts['tolfun'] = 1e-10
    opts['verbose'] = -9  # silent
    # No bounds — let it explore freely (penalty enforces containment)
    # opts['bounds'] = [[-5.0] * len(p0), [5.0] * len(p0)]

    es = cma.CMAEvolutionStrategy(p0, sigma0, opts)

    best_E = float('inf')
    best_p = p0.copy()

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(s, R_target) for s in solutions]
        es.tell(solutions, fitnesses)

        best_idx = int(np.argmin(fitnesses))
        if fitnesses[best_idx] < best_E:
            best_E = fitnesses[best_idx]
            best_p = solutions[best_idx].copy()

    xs, ys, ts = unpack(best_p)
    return xs, ys, ts, best_E


def main():
    log = open(LOG_FILE, 'w')
    def logprint(msg):
        print(msg, flush=True)
        log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        log.flush()

    logprint("=== CMA-ES tight refinement ===")

    xs0, ys0, ts0 = load_best()
    result0 = official_score(xs0, ys0, ts0)
    global_best_R = float(result0.score)
    logprint(f"Starting R: {global_best_R:.6f}")

    R_targets = [global_best_R - 0.01, global_best_R - 0.02, global_best_R - 0.05, 2.95, 2.90]

    for R_target in R_targets:
        logprint(f"\n── CMA-ES targeting R = {R_target:.4f} ──")

        for trial in range(5):
            sigma = 0.02 * (1 + trial * 0.5)
            logprint(f"  Trial {trial+1}/5 sigma={sigma:.3f}")

            # Random perturbation of best
            pxs = xs0 + np.random.normal(0, sigma * 0.5, N)
            pys = ys0 + np.random.normal(0, sigma * 0.5, N)
            pts = ts0 + np.random.normal(0, sigma, N)

            xs, ys, ts, E = run_cmaes_round(pxs, pys, pts, R_target,
                                             sigma0=sigma, maxiter=500)

            R_fast = score_fast(xs, ys, ts)
            logprint(f"  E={E:.3e}, R_fast={R_fast:.4f}")

            if E < 1e-4:
                result = official_score(xs, ys, ts)
                if result.valid:
                    actual_R = float(result.score)
                    logprint(f"  ✓ FEASIBLE! R={actual_R:.6f}")
                    if actual_R < global_best_R:
                        global_best_R = actual_R
                        xs0, ys0, ts0 = center_solution(xs, ys, ts)
                        save_best(xs0, ys0, ts0, actual_R)
                else:
                    logprint(f"  ✗ Shapely reject: {result.errors[:2]}")

    logprint(f"\n=== Done | best R = {global_best_R:.6f} ===")
    log.close()


if __name__ == '__main__':
    np.random.seed(int(time.time()) % 10000)
    main()
