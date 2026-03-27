#!/usr/bin/env python3
"""
Fast Constrained SA: 
- Overlap check only for most steps (fast)
- Approximate MEC (Welzl on arc sample points) for score estimate
- Full validate_and_score only when we think we've improved
"""

import json, math, time, random, sys
import numpy as np

sys.path.insert(0, '/home/camcore/.openclaw/workspace/shape-packing-challenge')
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap

N = 15
RADIUS = 1.0
BEST_FILE = '/home/camcore/.openclaw/workspace/shape-packing-challenge/best_solution.json'
ARC_PTS = 16  # points per semicircle for fast MEC — enough for ~1% accuracy


def load_best():
    with open(BEST_FILE) as f:
        return json.load(f)


def save_best(raw):
    with open(BEST_FILE, 'w') as f:
        json.dump(raw, f, indent=2)


def center_and_score(raw):
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    result = validate_and_score(sol)
    cx, cy, r = result.mec
    centered = [{'x': round(d['x'] - cx, 6), 'y': round(d['y'] - cy, 6), 'theta': round(d['theta'], 6)} for d in raw]
    return centered, r, result.valid


def build_boundary_pts(xs, ys, ts):
    """ARC_PTS arc points + 2 flat endpoints per semicircle."""
    pts = np.empty((N * (ARC_PTS + 2), 2))
    k = 0
    half = math.pi / 2
    for i in range(N):
        x, y, t = xs[i], ys[i], ts[i]
        for angle in np.linspace(t - half, t + half, ARC_PTS):
            pts[k, 0] = x + math.cos(angle)
            pts[k, 1] = y + math.sin(angle)
            k += 1
        pts[k, 0] = x + math.cos(t + half)
        pts[k, 1] = y + math.sin(t + half)
        k += 1
        pts[k, 0] = x + math.cos(t - half)
        pts[k, 1] = y + math.sin(t - half)
        k += 1
    return pts


def update_boundary_pts(pts, xs, ys, ts, idx):
    """Update boundary points for just one semicircle."""
    start = idx * (ARC_PTS + 2)
    x, y, t = xs[idx], ys[idx], ts[idx]
    half = math.pi / 2
    k = start
    for angle in np.linspace(t - half, t + half, ARC_PTS):
        pts[k, 0] = x + math.cos(angle)
        pts[k, 1] = y + math.sin(angle)
        k += 1
    pts[k, 0] = x + math.cos(t + half)
    pts[k, 1] = y + math.sin(t + half)
    k += 1
    pts[k, 0] = x + math.cos(t - half)
    pts[k, 1] = y + math.sin(t - half)


def fast_mec_radius(pts):
    """MEC radius via iterative minimax (fast, ~1% accurate)."""
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    for _ in range(20):
        dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        far_idx = np.argmax(dists)
        cx = 0.7 * cx + 0.3 * pts[far_idx, 0]
        cy = 0.7 * cy + 0.3 * pts[far_idx, 1]
    return float(np.max(np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)))


def check_overlap_fast(xs, ys, ts, idx):
    """Check if semicircle idx overlaps any other."""
    new_sc = Semicircle(float(xs[idx]), float(ys[idx]), float(ts[idx]))
    for j in range(N):
        if j == idx:
            continue
        dx = xs[idx] - xs[j]
        dy = ys[idx] - ys[j]
        if dx*dx + dy*dy > 4.01:
            continue
        if semicircles_overlap(new_sc, Semicircle(float(xs[j]), float(ys[j]), float(ts[j]))):
            return True
    return False


def run_csa(init_raw, T_start=0.15, T_end=1e-7, steps=600000, label="run"):
    xs = np.array([d['x'] for d in init_raw], dtype=float)
    ys = np.array([d['y'] for d in init_raw], dtype=float)
    ts = np.array([d['theta'] for d in init_raw], dtype=float)

    # Initial exact score
    sol0 = [Semicircle(xs[i], ys[i], ts[i]) for i in range(N)]
    r0 = validate_and_score(sol0)
    assert r0.valid, "Initial must be valid"
    current_score = r0.score
    best_score = current_score
    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()

    pts = build_boundary_pts(xs, ys, ts)
    fast_score = fast_mec_radius(pts)

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

        # Fast: check overlaps only
        if check_overlap_fast(xs, ys, ts, idx):
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
            rejected += 1
            continue

        # Update boundary pts for this piece
        old_pts = pts[idx * (ARC_PTS + 2):(idx + 1) * (ARC_PTS + 2)].copy()
        update_boundary_pts(pts, xs, ys, ts, idx)

        # Fast MEC estimate
        new_fast = fast_mec_radius(pts)
        delta_fast = new_fast - fast_score

        # SA decision on fast score
        if delta_fast >= 0 and random.random() >= math.exp(-delta_fast / T):
            # Reject
            xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
            pts[idx * (ARC_PTS + 2):(idx + 1) * (ARC_PTS + 2)] = old_pts
            rejected += 1
            continue

        # Accepted by fast score — now check exact score if it might be an improvement
        fast_score = new_fast

        # Only do exact eval when fast score suggests improvement over best
        if new_fast < best_score * 1.005:  # within 0.5% of best
            sol_new = [Semicircle(xs[i], ys[i], ts[i]) for i in range(N)]
            r_new = validate_and_score(sol_new)
            exact_evals += 1
            if not r_new.valid:
                xs[idx], ys[idx], ts[idx] = old_x, old_y, old_t
                pts[idx * (ARC_PTS + 2):(idx + 1) * (ARC_PTS + 2)] = old_pts
                fast_score = fast_mec_radius(pts)
                rejected += 1
                continue
            new_score = r_new.score
        else:
            new_score = new_fast  # use fast approx for non-best-candidate moves

        current_score = new_score
        accepted += 1
        window_acc += 1

        if new_fast < best_score and 'r_new' in dir() and new_fast < best_score * 1.005:
            # We did an exact eval — use that
            exact_score = r_new.score if r_new.valid else None
            if exact_score and exact_score < best_score:
                best_score = exact_score
                best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
                improvements += 1

        # Adaptive step
        if step % 1000 == 999:
            rate = window_acc / 1000
            if rate < 0.05:
                step_xy = max(step_xy * 0.92, 0.003)
                step_t = max(step_t * 0.92, 0.008)
            elif rate > 0.25:
                step_xy = min(step_xy * 1.05, 0.25)
                step_t = min(step_t * 1.05, 0.6)
            window_acc = 0

        if step % 100000 == 0 and step > 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {step:>7}/{steps}: best={best_score:.5f} fast={fast_score:.5f} T={T:.5f} acc={accepted/(accepted+rejected):.2%} imp={improvements} exact={exact_evals} sxy={step_xy:.4f} ({step/elapsed:.0f}/s)")

    elapsed = time.time() - t0
    # Final exact eval of best
    sol_final = [Semicircle(best_xs[i], best_ys[i], best_ts[i]) for i in range(N)]
    r_final = validate_and_score(sol_final)
    final_score = r_final.score if r_final.valid else float('inf')
    print(f"  [{label}] DONE {elapsed:.1f}s | exact_best={final_score:.6f} | imp={improvements} | {steps/elapsed:.0f}/s")

    best_raw = [{'x': float(best_xs[i]), 'y': float(best_ys[i]), 'theta': float(best_ts[i])} for i in range(N)]
    return best_raw, final_score


def make_random_valid_start(radius=2.8, max_attempts=2000):
    for _ in range(max_attempts):
        raw = []
        placed = []
        success = True
        for i in range(N):
            placed_sc = False
            for attempt in range(300):
                r = random.uniform(0, radius)
                angle = random.uniform(0, 2 * math.pi)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                t = random.uniform(0, 2 * math.pi)
                sc = Semicircle(x, y, t)
                ok = all(
                    not (
                        (x - p.x)**2 + (y - p.y)**2 <= 4.01
                        and semicircles_overlap(sc, p)
                    )
                    for p in placed
                )
                if ok:
                    raw.append({'x': x, 'y': y, 'theta': t})
                    placed.append(sc)
                    placed_sc = True
                    break
            if not placed_sc:
                success = False
                break
        if success:
            result = validate_and_score([Semicircle(d['x'], d['y'], d['theta']) for d in raw])
            if result.valid:
                return raw, result.score
    return None, None


if __name__ == '__main__':
    best_raw = load_best()
    best_raw, overall_best_score, valid = center_and_score(best_raw)
    save_best(best_raw)
    print(f"Starting best: {overall_best_score:.6f}")

    run_num = 0
    t_total = time.time()
    max_runtime = 3600

    while time.time() - t_total < max_runtime:
        run_num += 1
        use_best = (run_num % 4 != 0)

        if use_best:
            label = f"best_{run_num}"
            init_raw, init_score, _ = center_and_score(load_best())
            print(f"\n{'='*50}\nRun {run_num}: from_best (score={init_score:.5f})\n{'='*50}")
        else:
            label = f"rand_{run_num}"
            print(f"\n{'='*50}\nRun {run_num}: random start\n{'='*50}")
            init_raw, init_score = make_random_valid_start()
            if init_raw is None:
                print("  Failed to generate valid start, skipping")
                continue
            print(f"  Random start: {init_score:.5f}")

        try:
            result_raw, result_score = run_csa(
                init_raw, T_start=0.12, T_end=1e-7, steps=500000, label=label
            )
            centered_raw, centered_score, centered_valid = center_and_score(result_raw)
            print(f"  Run result: {centered_score:.6f} (best: {overall_best_score:.6f})")

            if centered_valid and centered_score < overall_best_score:
                overall_best_score = centered_score
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

    print(f"\nFinal best: {overall_best_score:.6f}")
