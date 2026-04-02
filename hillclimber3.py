#!/usr/bin/env python3
"""
hillclimber3.py — Shapely-exact hill climber with:
  1. Contact graph fingerprinting (zero overhead — only on improvements)
  2. Collective rigid-body cluster moves (25% of single-shape budget)
  3. Edge-break + repair move (every 2000 trials)

Uses ONLY Shapely for overlap checks — no phi-function distortion.
"""
import sys, os, json, time, random, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircle_polygon
import fcntl

BEST_FILE = os.path.join(os.path.dirname(__file__), 'best_solution.json')
LOG_FILE  = os.path.join(os.path.dirname(__file__), 'hillclimb3.log')
N = 15; TWO_PI = 2 * math.pi

def load_best():
    with open(BEST_FILE) as f: raw = json.load(f)
    xs = np.array([s['x'] for s in raw])
    ys = np.array([s['y'] for s in raw])
    ts = np.array([s['theta'] for s in raw])
    return xs, ys, ts

def score(xs, ys, ts):
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    r = validate_and_score(sol)
    return r.score if r.valid else float('inf'), r

def save_if_better(xs, ys, ts, current_best):
    s, r = score(xs, ys, ts)
    if s >= current_best: return current_best, False
    lock_path = BEST_FILE + '.lock'
    with open(lock_path, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        # Re-check under lock
        xs2, ys2, ts2 = load_best()
        s2, r2 = score(xs2, ys2, ts2)
        if s >= s2: return s2, False
        cx, cy = r.mec[0], r.mec[1]
        out = [{'x': round(float(xs[i]-cx),6), 'y': round(float(ys[i]-cy),6),
                'theta': round(float(ts[i])%TWO_PI,6)} for i in range(N)]
        with open(BEST_FILE,'w') as f: json.dump(out, f, indent=2)
        # Auto-commit to git so we never lose a best solution
        try:
            import subprocess
            subprocess.run(['git', 'add', 'best_solution.json'], cwd=os.path.dirname(BEST_FILE), capture_output=True)
            subprocess.run(['git', 'commit', '-m', f'best: R={s:.6f} (auto-commit)'],
                          cwd=os.path.dirname(BEST_FILE), capture_output=True)
        except Exception:
            pass
        return s, True

# ─── Addition 1: Contact graph fingerprint ───────────────────────────
def fingerprint(xs, ys, ts):
    """Compute contact graph fingerprint using Shapely distances.
    Returns (contacts_frozenset, boundary_active_set).
    contacts_frozenset: frozenset of (i, j, type) tuples
    boundary_active_set: set of indices whose R_i is within 0.001 of max R
    """
    polys = [semicircle_polygon(Semicircle(float(xs[i]), float(ys[i]), float(ts[i])))
             for i in range(N)]

    # Pairwise near-contacts
    contacts = []
    for i in range(N):
        for j in range(i+1, N):
            d = polys[i].distance(polys[j])
            if d < 0.05:
                # Classify contact type by relative theta
                dtheta = abs(ts[i] - ts[j]) % math.pi
                if dtheta < 0.15 or dtheta > math.pi - 0.15:
                    ctype = "flat_flat"
                elif abs(dtheta - math.pi/2) < 0.15:
                    ctype = "arc_flat"
                else:
                    ctype = "arc_arc"
                contacts.append((i, j, ctype))

    # Boundary-active semicircles
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    from src.semicircle_packing.scoring import compute_mec
    from src.semicircle_packing.geometry import farthest_boundary_point_from
    cx, cy, cr = compute_mec(sol)
    boundary_active = set()
    for i in range(N):
        fx, fy = farthest_boundary_point_from(sol[i], cx, cy)
        ri = math.hypot(fx - cx, fy - cy)
        if ri >= cr - 0.001:
            boundary_active.add(i)

    return frozenset(contacts), boundary_active

# ─── Addition 2: Cluster move ────────────────────────────────────────
def perturb_cluster(xs, ys, ts, scale, rng, min_k=2, max_k=4):
    """Pick adjacent semicircles and apply a rigid transform to the cluster."""
    nxs, nys, nts = xs.copy(), ys.copy(), ts.copy()

    # Pick random center shape
    center = rng.randint(0, N)

    # Find k nearest neighbors
    k = rng.randint(min_k, max_k + 1)
    dists = np.hypot(xs - xs[center], ys - ys[center])
    dists[center] = -1  # ensure center is first
    order = np.argsort(dists)
    cluster = order[:k+1]  # center + k neighbors

    # Rigid transform: translate + rotate
    dx = rng.uniform(-scale, scale)
    dy = rng.uniform(-scale, scale)
    dphi = rng.uniform(-scale, scale)

    # Rotate around cluster centroid
    cx = np.mean(xs[cluster])
    cy = np.mean(ys[cluster])
    cos_d, sin_d = math.cos(dphi), math.sin(dphi)

    for i in cluster:
        rx, ry = nxs[i] - cx, nys[i] - cy
        nxs[i] = cx + rx * cos_d - ry * sin_d + dx
        nys[i] = cy + rx * sin_d + ry * cos_d + dy
        nts[i] = (nts[i] + dphi) % TWO_PI

    return nxs, nys, nts

# ─── Addition 3: Edge-break + repair move ────────────────────────────
def edge_break_move(xs, ys, ts, scale, rng):
    """Find boundary-active semicircle, separate it from closest neighbor."""
    nxs, nys, nts = xs.copy(), ys.copy(), ts.copy()

    # Find which semicircle contributes most to R (boundary-active max)
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    from src.semicircle_packing.scoring import compute_mec
    from src.semicircle_packing.geometry import farthest_boundary_point_from
    cx, cy, cr = compute_mec(sol)

    max_ri = -1
    max_i = 0
    for i in range(N):
        fx, fy = farthest_boundary_point_from(sol[i], cx, cy)
        ri = math.hypot(fx - cx, fy - cy)
        if ri > max_ri:
            max_ri = ri
            max_i = i

    # Find closest neighbor to max_i
    dists = np.hypot(xs - xs[max_i], ys - ys[max_i])
    dists[max_i] = float('inf')
    closest = np.argmin(dists)

    # Push apart along connecting line
    dx = xs[max_i] - xs[closest]
    dy = ys[max_i] - ys[closest]
    d = math.hypot(dx, dy)
    if d > 1e-10:
        ux, uy = dx/d, dy/d
    else:
        ux, uy = 1.0, 0.0

    push = scale * 0.5
    nxs[max_i] += ux * push
    nys[max_i] += uy * push
    nxs[closest] -= ux * push
    nys[closest] -= uy * push

    # Small random noise to both
    for i in [max_i, closest]:
        nxs[i] += rng.uniform(-scale*0.3, scale*0.3)
        nys[i] += rng.uniform(-scale*0.3, scale*0.3)
        nts[i] = (nts[i] + rng.uniform(-scale, scale)) % TWO_PI

    return nxs, nys, nts

# ─── Main loop ───────────────────────────────────────────────────────
def run(max_trials=500000, log_interval=10000):
    xs, ys, ts = load_best()
    best, _ = score(xs, ys, ts)
    print(f'Starting: R={best:.6f}', flush=True)

    log = open(LOG_FILE, 'a')
    def logprint(msg):
        print(msg, flush=True)
        log.write(time.strftime('%H:%M:%S') + ' ' + msg + '\n')
        log.flush()

    improvements = 0
    rng = np.random.RandomState(int(time.time()) % 100000)
    t0 = time.time()

    # Perturbation schedule: cycle through scales
    scales = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
    scale_idx = 0
    no_improve_streak = 0

    for trial in range(max_trials):
        # Cycle scales, increase when stuck
        scale = scales[scale_idx % len(scales)]
        if no_improve_streak > 5000:
            scale_idx = (scale_idx + 1) % (len(scales) * 3)
            no_improve_streak = 0

        # ── Edge-break move every 2000 trials ──
        if trial > 0 and trial % 2000 == 0:
            eb_xs, eb_ys, eb_ts = edge_break_move(xs, ys, ts, scale, rng)
            eb_score, eb_r = score(eb_xs, eb_ys, eb_ts)
            if eb_score < best:
                best = eb_score
                xs, ys, ts = eb_xs.copy(), eb_ys.copy(), eb_ts.copy()
                improvements += 1
                no_improve_streak = 0
                scale_idx = 0
                save_if_better(xs, ys, ts, best + 0.0001)
                fp_contacts, fp_boundary = fingerprint(xs, ys, ts)
                logprint(f'[{trial}] R={best:.6f} (edge-break #{improvements}) '
                         f'contacts={len(fp_contacts)} boundary={len(fp_boundary)}')
                continue

        # Pick perturbation type
        r = rng.random()
        nxs, nys, nts = xs.copy(), ys.copy(), ts.copy()

        if r < 0.10:
            # Large cluster move (6-8 shapes) at coarse scale — contact-graph escape
            coarse = max(scale, 0.02)
            nxs, nys, nts = perturb_cluster(xs, ys, ts, coarse, rng, min_k=5, max_k=7)
        elif r < 0.2625:
            # Standard cluster move (3-5 shapes) at 2x scale
            nxs, nys, nts = perturb_cluster(xs, ys, ts, scale * 2, rng, min_k=2, max_k=4)
        elif r < 0.5:
            # Perturb one random shape (x, y, theta)
            i = rng.randint(0, N)
            nxs[i] += rng.uniform(-scale, scale)
            nys[i] += rng.uniform(-scale, scale)
            nts[i] = (nts[i] + rng.uniform(-scale*3, scale*3)) % TWO_PI
        elif r < 0.7:
            # Perturb x only
            i = rng.randint(0, N)
            nxs[i] += rng.uniform(-scale, scale)
        elif r < 0.85:
            # Perturb y only
            i = rng.randint(0, N)
            nys[i] += rng.uniform(-scale, scale)
        elif r < 0.95:
            # Perturb theta only
            i = rng.randint(0, N)
            nts[i] = (nts[i] + rng.uniform(-scale*5, scale*5)) % TWO_PI
        else:
            # Perturb 2 shapes together
            i, j = rng.choice(N, 2, replace=False)
            nxs[i] += rng.uniform(-scale, scale)
            nys[i] += rng.uniform(-scale, scale)
            nxs[j] += rng.uniform(-scale, scale)
            nys[j] += rng.uniform(-scale, scale)

        new_score, new_r = score(nxs, nys, nts)
        if new_score < best:
            best = new_score
            xs, ys, ts = nxs.copy(), nys.copy(), nts.copy()
            improvements += 1
            no_improve_streak = 0
            scale_idx = 0  # reset to fine scale after improvement
            save_if_better(xs, ys, ts, best + 0.0001)
            move_type = "cluster" if r < 0.1625 else "single"
            fp_contacts, fp_boundary = fingerprint(xs, ys, ts)
            logprint(f'[{trial}] R={best:.6f} ({move_type} #{improvements}) '
                     f'contacts={len(fp_contacts)} boundary={len(fp_boundary)}')
        else:
            no_improve_streak += 1

        if trial % log_interval == 0 and trial > 0:
            elapsed = time.time() - t0
            logprint(f'[{elapsed:.0f}s] trial={trial}, best={best:.6f}, improvements={improvements}')

    logprint(f'Done. Final: R={best:.6f} ({improvements} improvements in {max_trials} trials)')
    log.close()

if __name__ == '__main__':
    run(max_trials=500000)
