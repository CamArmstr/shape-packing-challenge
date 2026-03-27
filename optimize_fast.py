#!/usr/bin/env python3
"""
Fast Constrained SA:
- Pre-built 64-pt Shapely polys, only rebuild for moved piece (~12k steps/sec)
- Approx MEC (farthest-point from cached center) for SA decisions
- Exact validate_and_score only when fast score beats best
"""

import json, math, time, random, sys
import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, '/home/camcore/.openclaw/workspace/shape-packing-challenge')
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, farthest_boundary_point_from

N = 15
OVERLAP_TOL = 1e-6
BEST_FILE = '/home/camcore/.openclaw/workspace/shape-packing-challenge/best_solution.json'


def make_poly(x, y, theta, n=64):
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, n)
    pts = list(zip(float(x) + np.cos(angles), float(y) + np.sin(angles)))
    pts.append((float(x), float(y)))
    return Polygon(pts)


def load_best():
    with open(BEST_FILE) as f:
        return json.load(f)


def save_best(raw):
    with open(BEST_FILE, 'w') as f:
        json.dump(raw, f, indent=2)


def center_and_score(raw):
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    result = validate_and_score(sol)
    cx, cy, mec_r = result.mec
    centered = [{'x': round(d['x'] - cx, 6), 'y': round(d['y'] - cy, 6),
                 'theta': round(d['theta'], 6)} for d in raw]
    return centered, result.score, result.valid


def run_csa(init_raw, T_start=0.15, T_end=1e-7, steps=1000000, label="run"):
    xs = np.array([d['x'] for d in init_raw], dtype=float)
    ys = np.array([d['y'] for d in init_raw], dtype=float)
    ts = np.array([d['theta'] for d in init_raw], dtype=float)

    # Exact initial score + MEC center
    sol0 = [Semicircle(xs[i], ys[i], ts[i]) for i in range(N)]
    r0 = validate_and_score(sol0)
    assert r0.valid, "Initial must be valid"
    current_exact = r0.score
    mec_cx, mec_cy = r0.mec[0], r0.mec[1]
    best_score = current_exact
    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()

    # Pre-build polys
    polys = [make_poly(xs[i], ys[i], ts[i]) for i in range(N)]

    # Per-piece farthest distances from MEC center (approx current MEC radius)
    dists = np.array([
        math.hypot(*farthest_boundary_point_from(sol0[i], mec_cx, mec_cy)) -
        math.hypot(0, 0)  # just want distance from mec center
        for i in range(N)
    ])
    # Recompute correctly
    dists = np.array([
        math.hypot(
            farthest_boundary_point_from(sol0[i], mec_cx, mec_cy)[0] - mec_cx,
            farthest_boundary_point_from(sol0[i], mec_cx, mec_cy)[1] - mec_cy
        ) for i in range(N)
    ])
    approx_score = float(np.max(dists))

    T = T_start
    alpha = (T_end / T_start) ** (1.0 / steps)
    step_xy = 0.07
    step_t = 0.18
    accepted = 0
    rejected = 0
    improvements = 0
    exact_evals = 0
    window_acc = 0
    t0 = time.time()

    for step in range(steps):
        T *= alpha
        idx = random.randint(0, N - 1)
        old_x, old_y, old_t = xs[idx], ys[idx], ts[idx]
        old_poly = polys[idx]
        old_dist = dists[idx]

        r = random.random()
        if r < 0.35:
            xs[idx] += random.gauss(0, step_xy)
            ys[idx] += random.gauss(0, step_xy)
        elif r < 0.65:
            ts[idx] += random.gauss(0, step_t)
        else:
            xs[idx] += random.gauss(0, step_xy)
            ys[idx] += random.gauss(0, step_xy)
            ts[idx] += random.gauss(0, step_t)

        new_poly = make_poly(xs[idx], ys[idx], ts[idx])

        # Overlap check
        overlap = False
        for j in range(N):
            if j == idx:
                continue
            dx = xs[idx] - xs[j]
            dy = ys[idx] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue
            if new_poly.intersection(polys[j]).area > OVERLAP_TOL:
                overlap = True
                break

        if overlap:
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
            rejected += 1
            continue

        # Compute new piece's farthest distance from cached MEC center
        new_sc = Semicircle(float(xs[idx]), float(ys[idx]), float(ts[idx]))
        fx, fy = farthest_boundary_point_from(new_sc, mec_cx, mec_cy)
        new_dist = math.hypot(fx - mec_cx, fy - mec_cy)
        dists[idx] = new_dist
        new_approx = float(np.max(dists))

        delta = new_approx - approx_score

        if delta >= 0 and random.random() >= math.exp(-delta / T):
            # Reject
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
            dists[idx] = old_dist
            rejected += 1
            continue

        # Accept by approx score
        polys[idx] = new_poly
        approx_score = new_approx
        accepted += 1
        window_acc += 1

        # Exact eval only when approx suggests we might beat best
        if new_approx < best_score + 0.01:
            sol_new = [Semicircle(xs[i], ys[i], ts[i]) for i in range(N)]
            r_new = validate_and_score(sol_new)
            exact_evals += 1
            if r_new.valid:
                current_exact = r_new.score
                mec_cx, mec_cy = r_new.mec[0], r_new.mec[1]
                # Update all distances from new MEC center
                for i in range(N):
                    sc_i = Semicircle(xs[i], ys[i], ts[i])
                    fx_i, fy_i = farthest_boundary_point_from(sc_i, mec_cx, mec_cy)
                    dists[i] = math.hypot(fx_i - mec_cx, fy_i - mec_cy)
                approx_score = float(np.max(dists))
                if current_exact < best_score:
                    best_score = current_exact
                    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
                    improvements += 1

        # Adaptive step
        if step % 2000 == 1999:
            rate = window_acc / 2000
            if rate < 0.05:
                step_xy = max(step_xy * 0.92, 0.003)
                step_t = max(step_t * 0.92, 0.008)
            elif rate > 0.30:
                step_xy = min(step_xy * 1.05, 0.25)
                step_t = min(step_t * 1.05, 0.6)
            window_acc = 0

        if step % 200000 == 0 and step > 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {step:>8}/{steps}: exact={current_exact:.5f} best={best_score:.5f} T={T:.6f} acc={accepted/(accepted+rejected):.2%} imp={improvements} exact_evals={exact_evals} sxy={step_xy:.4f} ({step/elapsed:.0f}/s)")

    elapsed = time.time() - t0
    sol_final = [Semicircle(float(best_xs[i]), float(best_ys[i]), float(best_ts[i])) for i in range(N)]
    r_final = validate_and_score(sol_final)
    final_score = r_final.score if r_final.valid else float('inf')
    print(f"  [{label}] DONE {elapsed:.1f}s | exact={final_score:.6f} | imp={improvements} exact_evals={exact_evals} | {steps/elapsed:.0f}/s")
    best_raw = [{'x': float(best_xs[i]), 'y': float(best_ys[i]), 'theta': float(best_ts[i])} for i in range(N)]
    return best_raw, final_score, r_final.valid


def make_random_valid_start(radius=2.8):
    for _ in range(500):
        raw = []
        polys_placed = []
        success = True
        for i in range(N):
            placed = False
            for _ in range(200):
                r_val = random.uniform(0.5, radius)
                angle = random.uniform(0, 2 * math.pi)
                x = r_val * math.cos(angle)
                y = r_val * math.sin(angle)
                t = random.uniform(0, 2 * math.pi)
                p = make_poly(x, y, t)
                if all(p.intersection(ep).area <= OVERLAP_TOL for ep in polys_placed):
                    raw.append({'x': x, 'y': y, 'theta': t})
                    polys_placed.append(p)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success:
            sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
            result = validate_and_score(sol)
            if result.valid:
                return raw, result.score
    return None, None


if __name__ == '__main__':
    best_raw = load_best()
    best_raw, overall_best, _ = center_and_score(best_raw)
    save_best(best_raw)
    print(f"Starting best: {overall_best:.6f}")

    run_num = 0
    t_total = time.time()
    max_runtime = 3600

    while time.time() - t_total < max_runtime:
        run_num += 1
        if run_num % 4 != 0:
            label = f"best_{run_num}"
            init_raw, init_score, _ = center_and_score(load_best())
            print(f"\n{'='*50}\nRun {run_num}: from_best (score={init_score:.5f})\n{'='*50}")
        else:
            label = f"rand_{run_num}"
            print(f"\n{'='*50}\nRun {run_num}: random\n{'='*50}")
            init_raw, init_score = make_random_valid_start()
            if init_raw is None:
                print("  Couldn't make valid random start, skipping")
                continue
            print(f"  Random start: {init_score:.5f}")

        try:
            result_raw, result_score, result_valid = run_csa(
                init_raw, T_start=0.12, T_end=1e-7, steps=800000, label=label
            )
            if not result_valid:
                print("  Final invalid, skipping")
                continue
            centered_raw, centered_score, cv = center_and_score(result_raw)
            print(f"  Centered: {centered_score:.6f} (best: {overall_best:.6f})")
            if cv and centered_score < overall_best:
                overall_best = centered_score
                save_best(centered_raw)
                sol_vis = [Semicircle(d['x'], d['y'], d['theta']) for d in centered_raw]
                r_vis = validate_and_score(sol_vis)
                from src.semicircle_packing.visualization import plot_packing
                plot_packing(sol_vis, r_vis.mec,
                             save_path='/home/camcore/.openclaw/workspace/shape-packing-challenge/best_solution.png')
                print(f"  *** NEW BEST: {centered_score:.6f} ***")
        except Exception as e:
            import traceback
            traceback.print_exc()

    print(f"\nFinal best: {overall_best:.6f}")
