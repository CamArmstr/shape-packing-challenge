"""
gjk_optimize.py — Run GJK-based L-BFGS-B local minimizer on best_solution.json.

Loads the current best, optimizes with GJK overlap detection (replacing phi-function),
validates with official Shapely scorer, saves if improved.
"""

import json, time, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mbh import (
    gjk_lbfgs_minimize, load_best, save_best, official_validate,
    center_solution, N, pack, unpack
)
from gjk_numba import semicircle_gjk_signed_dist

BEST_FILE = os.path.join(os.path.dirname(__file__), 'best_solution.json')


def print_pair_report(xs, ys, ts):
    """Show closest/most-overlapping pairs."""
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            d = semicircle_gjk_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            pairs.append((d, i, j))
    pairs.sort()
    print(f"  Closest 5 pairs (GJK signed dist):")
    for d, i, j in pairs[:5]:
        status = "OVERLAP" if d < -1e-8 else "OK"
        print(f"    ({i:2d},{j:2d}) d={d:+.8f}  {status}")
    neg = sum(1 for d, _, _ in pairs if d < -1e-8)
    print(f"  Overlapping pairs: {neg}/105")


def run_optimization():
    print("=" * 60)
    print("GJK-based L-BFGS-B Optimizer")
    print("=" * 60)

    # Load current best
    xs, ys, ts = load_best()
    result = official_validate(xs, ys, ts)
    if not result.valid:
        print(f"WARNING: loaded solution invalid: {result.errors}")
        current_score = float('inf')
    else:
        current_score = float(result.score)
    print(f"\nCurrent best score: {current_score:.6f}")
    print_pair_report(xs, ys, ts)

    # Try multiple R values around current best
    r_values = [
        current_score + 0.005,
        current_score + 0.002,
        current_score + 0.001,
        current_score,
        current_score - 0.001,
        current_score - 0.002,
        current_score - 0.005,
    ]

    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
    best_score = current_score
    improved = False

    for R in r_values:
        print(f"\n--- Trying R = {R:.6f} ---")
        t0 = time.time()
        rxs, rys, rts, energy = gjk_lbfgs_minimize(xs, ys, ts, R, lam=1e6, max_iter=3000)
        elapsed = time.time() - t0
        print(f"  Energy: {energy:.2e}  ({elapsed:.1f}s)")

        if energy > 1e-4:
            print(f"  Skipping — energy too high (not feasible)")
            continue

        # Validate with official scorer
        vr = official_validate(rxs, rys, rts)
        if not vr.valid:
            print(f"  Invalid: {vr.errors[:3]}")
            continue

        score = float(vr.score)
        print(f"  Valid! Score = {score:.6f}")
        print_pair_report(rxs, rys, rts)

        if score < best_score:
            best_score = score
            best_xs, best_ys, best_ts = rxs.copy(), rys.copy(), rts.copy()
            improved = True
            print(f"  *** NEW BEST: {score:.6f} ***")

    print(f"\n{'=' * 60}")
    print(f"Final result: {best_score:.6f} (was {current_score:.6f})")

    if improved and best_score < current_score:
        # Center and save
        cxs, cys, cts = center_solution(best_xs, best_ys, best_ts)
        save_best(cxs, cys, cts, best_score)
        print(f"Saved new best: {best_score:.6f}")
    else:
        print("No improvement found.")

    # Second pass: perturb slightly and re-optimize (shake loose from local min)
    print(f"\n{'=' * 60}")
    print("Phase 2: Small perturbations + GJK optimization")
    print("=" * 60)

    xs, ys, ts = best_xs.copy(), best_ys.copy(), best_ts.copy()
    R = best_score + 0.002

    for trial in range(20):
        # Small random perturbation
        pxs = xs + np.random.uniform(-0.02, 0.02, N)
        pys = ys + np.random.uniform(-0.02, 0.02, N)
        pts = ts + np.random.uniform(-0.05, 0.05, N)

        t0 = time.time()
        rxs, rys, rts, energy = gjk_lbfgs_minimize(pxs, pys, pts, R, lam=1e6, max_iter=3000)
        elapsed = time.time() - t0

        if energy > 1e-4:
            print(f"  Trial {trial+1:2d}: energy={energy:.2e} (skip) [{elapsed:.1f}s]")
            continue

        vr = official_validate(rxs, rys, rts)
        if not vr.valid:
            print(f"  Trial {trial+1:2d}: invalid [{elapsed:.1f}s]")
            continue

        score = float(vr.score)
        tag = ""
        if score < best_score:
            best_score = score
            best_xs, best_ys, best_ts = rxs.copy(), rys.copy(), rts.copy()
            improved = True
            tag = " *** NEW BEST ***"
            cxs, cys, cts = center_solution(best_xs, best_ys, best_ts)
            save_best(cxs, cys, cts, best_score)
            xs, ys, ts = cxs, cys, cts
            R = best_score + 0.002
        print(f"  Trial {trial+1:2d}: score={score:.6f} [{elapsed:.1f}s]{tag}")

    print(f"\n{'=' * 60}")
    print(f"FINAL: {best_score:.6f} (started at {current_score:.6f})")
    delta = current_score - best_score
    if delta > 0:
        print(f"Improvement: {delta:.6f}")
    else:
        print("No improvement.")


if __name__ == '__main__':
    np.random.seed(42)
    run_optimization()
