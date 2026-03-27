#!/usr/bin/env python3
"""
Constrained Simulated Annealing for semicircle packing.
Only accepts fully valid moves. Centers before saving.
"""

import json, math, time, random, sys
import numpy as np

sys.path.insert(0, '/home/camcore/.openclaw/workspace/shape-packing-challenge')
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap

RADIUS = 1.0
N = 15
BEST_FILE = '/home/camcore/.openclaw/workspace/shape-packing-challenge/best_solution.json'


def load_best():
    with open(BEST_FILE) as f:
        return json.load(f)


def save_best(raw):
    with open(BEST_FILE, 'w') as f:
        json.dump(raw, f, indent=2)


def center_solution(raw):
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    result = validate_and_score(sol)
    cx, cy, r = result.mec
    centered = [{'x': round(d['x'] - cx, 6), 'y': round(d['y'] - cy, 6), 'theta': round(d['theta'], 6)} for d in raw]
    return centered, r


def full_score(raw):
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    result = validate_and_score(sol)
    return result.score if result.valid else None


def check_overlaps(semicircles, skip_i=-1, skip_j=-1):
    for i in range(N):
        for j in range(i + 1, N):
            dx = semicircles[i].x - semicircles[j].x
            dy = semicircles[i].y - semicircles[j].y
            if dx*dx + dy*dy > 4.01:
                continue
            if semicircles_overlap(semicircles[i], semicircles[j]):
                return True
    return False


def make_semicircles(xs, ys, ts):
    return [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]


def run_csa(init_raw, T_start=0.2, T_end=1e-6, steps=500000, label="run"):
    xs = np.array([d['x'] for d in init_raw], dtype=float)
    ys = np.array([d['y'] for d in init_raw], dtype=float)
    ts = np.array([d['theta'] for d in init_raw], dtype=float)

    sol = make_semicircles(xs, ys, ts)
    result = validate_and_score(sol)
    assert result.valid, "Initial solution must be valid"

    current_score = result.score
    best_score = current_score
    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()

    T = T_start
    alpha = (T_end / T_start) ** (1.0 / steps)

    step_xy = 0.08
    step_t = 0.2
    accepted = 0
    total = 0
    improvements = 0
    window_accepts = 0
    t0 = time.time()

    for step in range(steps):
        T *= alpha
        total += 1

        idx = random.randint(0, N - 1)
        old_x, old_y, old_t = xs[idx], ys[idx], ts[idx]

        r = random.random()
        if r < 0.35:
            new_x = old_x + random.gauss(0, step_xy)
            new_y = old_y + random.gauss(0, step_xy)
            new_t = old_t
        elif r < 0.65:
            new_x, new_y = old_x, old_y
            new_t = old_t + random.gauss(0, step_t)
        else:
            new_x = old_x + random.gauss(0, step_xy)
            new_y = old_y + random.gauss(0, step_xy)
            new_t = old_t + random.gauss(0, step_t)

        xs[idx], ys[idx], ts[idx] = new_x, new_y, new_t

        # Fast overlap check for moved piece only
        new_sc = Semicircle(new_x, new_y, new_t)
        overlap = False
        for j in range(N):
            if j == idx:
                continue
            dx = new_x - xs[j]
            dy = new_y - ys[j]
            if dx*dx + dy*dy > 4.01:
                continue
            if semicircles_overlap(new_sc, Semicircle(xs[j], ys[j], ts[j])):
                overlap = True
                break

        if overlap:
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
            continue

        result_new = validate_and_score(make_semicircles(xs, ys, ts))
        if not result_new.valid:
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
            continue

        new_score = result_new.score
        delta = new_score - current_score

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            accepted += 1
            window_accepts += 1
            if new_score < best_score:
                best_score = new_score
                best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
                improvements += 1
        else:
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t

        # Adaptive step size
        if step % 500 == 499:
            rate = window_accepts / 500
            if rate < 0.05:
                step_xy = max(step_xy * 0.9, 0.005)
                step_t = max(step_t * 0.9, 0.01)
            elif rate > 0.25:
                step_xy = min(step_xy * 1.05, 0.3)
                step_t = min(step_t * 1.05, 0.8)
            window_accepts = 0

        if step % 50000 == 0 and step > 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {step:>7}/{steps}: score={current_score:.5f} best={best_score:.5f} T={T:.5f} acc={accepted/total:.2%} imp={improvements} sxy={step_xy:.4f} ({step/elapsed:.0f}/s)")

    elapsed = time.time() - t0
    print(f"  [{label}] DONE {elapsed:.1f}s | best={best_score:.6f} | imp={improvements}")

    best_raw = [{'x': float(best_xs[i]), 'y': float(best_ys[i]), 'theta': float(best_ts[i])} for i in range(N)]
    return best_raw, best_score


def make_random_valid_start(max_attempts=1000):
    """Place semicircles one at a time, each non-overlapping with previous."""
    for _ in range(max_attempts):
        raw = []
        placed = []
        success = True
        for i in range(N):
            for attempt in range(500):
                r = random.uniform(0, 2.8)
                angle = random.uniform(0, 2 * math.pi)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                t = random.uniform(0, 2 * math.pi)
                sc = Semicircle(x, y, t)
                ok = True
                for p in placed:
                    dx = x - p.x
                    dy = y - p.y
                    if dx*dx + dy*dy <= 4.01 and semicircles_overlap(sc, p):
                        ok = False
                        break
                if ok:
                    raw.append({'x': x, 'y': y, 'theta': t})
                    placed.append(sc)
                    break
            else:
                success = False
                break
        if success:
            result = validate_and_score([Semicircle(d['x'], d['y'], d['theta']) for d in raw])
            if result.valid:
                return raw
    return None


if __name__ == '__main__':
    overall_best_raw = load_best()
    overall_best_raw, overall_best_score = center_solution(overall_best_raw)
    save_best(overall_best_raw)
    print(f"Starting best: {overall_best_score:.6f}")

    run_num = 0
    t_total = time.time()
    max_runtime = 3600

    while time.time() - t_total < max_runtime:
        run_num += 1
        use_best = run_num % 3 != 0  # 2/3 restarts from best, 1/3 random

        if use_best:
            label = f"from_best_{run_num}"
            init_raw, init_score = center_solution(load_best())
            print(f"\n{'='*50}\nRun {run_num}: {label} (score={init_score:.5f})\n{'='*50}")
        else:
            label = f"random_{run_num}"
            print(f"\n{'='*50}\nRun {run_num}: {label}\n{'='*50}")
            init_raw = make_random_valid_start()
            if init_raw is None:
                print("  Could not generate valid random start, skipping")
                continue
            result = validate_and_score([Semicircle(d['x'], d['y'], d['theta']) for d in init_raw])
            init_score = result.score
            print(f"  Random start score: {init_score:.5f}")

        try:
            best_raw, best_score = run_csa(
                init_raw,
                T_start=0.15,
                T_end=1e-7,
                steps=400000,
                label=label
            )

            centered_raw, centered_score = center_solution(best_raw)
            print(f"  Centered: {centered_score:.6f} (global best: {overall_best_score:.6f})")

            if centered_score < overall_best_score:
                overall_best_score = centered_score
                overall_best_raw = centered_raw
                save_best(centered_raw)
                # Save visualization
                sol_vis = [Semicircle(d['x'], d['y'], d['theta']) for d in centered_raw]
                r_vis = validate_and_score(sol_vis)
                from src.semicircle_packing.visualization import plot_packing
                plot_packing(sol_vis, r_vis.mec, save_path='/home/camcore/.openclaw/workspace/shape-packing-challenge/best_solution.png')
                print(f"  *** NEW BEST: {centered_score:.6f} ***")

        except Exception as e:
            import traceback
            print(f"  Error: {e}")
            traceback.print_exc()

    print(f"\nFinal best: {overall_best_score:.6f}")
