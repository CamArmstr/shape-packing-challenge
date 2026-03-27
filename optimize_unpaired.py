#!/usr/bin/env python3
"""
Non-paired starting configurations only.
Per Fejes Tóth (1971): the optimal is NOT paired semicircles.
Tries configurations where semicircles are independently placed,
oriented to exploit boundary effects and flat-edge interlocking.
"""

import sys, os, json, math, time, random
import numpy as np
os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

import importlib.util
spec = importlib.util.spec_from_file_location("overnight", "overnight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
from shapely.geometry import Polygon

BEST_FILE = 'best_solution.json'
N = 15
ARC = 64

def make_poly(x, y, theta):
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, ARC)
    pts = list(zip(x + np.cos(angles), y + np.sin(angles)))
    pts.append((x, y))
    return Polygon(pts)

def load_best_score():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    bx = np.array([s['x'] for s in raw])
    by = np.array([s['y'] for s in raw])
    bt = np.array([s['theta'] for s in raw])
    val = mod.official_validate(bx, by, bt)
    return bx, by, bt, val.score if val.valid else float('inf')

def center_solution(raw):
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    result = validate_and_score(sol)
    if not result.valid:
        return raw, None
    cx, cy, r = result.mec
    centered = [{'x': round(d['x']-cx, 6), 'y': round(d['y']-cy, 6),
                 'theta': round(d['theta'], 6)} for d in raw]
    return centered, r

def random_valid_start(r_max=2.8):
    """Place semicircles one at a time, greedy, no pairs."""
    for _ in range(200):
        raw = []; polys = []
        success = True
        for i in range(N):
            placed = False
            for _ in range(400):
                r = random.uniform(0.3, r_max)
                a = random.uniform(0, 2*math.pi)
                x = r * math.cos(a); y = r * math.sin(a)
                t = random.uniform(0, 2*math.pi)
                p = make_poly(x, y, t)
                if all(p.intersection(ep).area < 1e-6 for ep in polys):
                    raw.append({'x': x, 'y': y, 'theta': t})
                    polys.append(p)
                    placed = True; break
            if not placed:
                success = False; break
        if success:
            sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
            r = validate_and_score(sol)
            if r.valid:
                return raw, r.score
    return None, None

def boundary_biased_start():
    """
    Fejes Tóth insight: boundary semicircles should face outward.
    Place ~6 semicircles around the boundary flat-edge-out,
    rest fill interior freely.
    """
    for _ in range(100):
        raw = []; polys = []
        r_enc = 2.6

        # 6 boundary semicircles: flat edge tangent to enclosing circle
        for i in range(6):
            a = i * math.pi / 3 + random.gauss(0, 0.1)
            r = r_enc - 1.0 + random.gauss(0, 0.05)
            x = r * math.cos(a); y = r * math.sin(a)
            # theta: curved side faces inward = theta points toward center
            t = a + math.pi + random.gauss(0, 0.15)
            p = make_poly(x, y, t)
            if all(p.intersection(ep).area < 1e-6 for ep in polys):
                raw.append({'x': x, 'y': y, 'theta': t})
                polys.append(p)

        if len(raw) < 6:
            continue

        # Fill interior
        success = True
        for i in range(N - len(raw)):
            placed = False
            for _ in range(300):
                r = random.uniform(0.2, r_enc - 1.2)
                a = random.uniform(0, 2*math.pi)
                x = r*math.cos(a); y = r*math.sin(a)
                t = random.uniform(0, 2*math.pi)
                p = make_poly(x, y, t)
                if all(p.intersection(ep).area < 1e-6 for ep in polys):
                    raw.append({'x': x, 'y': y, 'theta': t})
                    polys.append(p)
                    placed = True; break
            if not placed:
                success = False; break

        if success and len(raw) == N:
            sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
            r = validate_and_score(sol)
            if r.valid:
                return raw, r.score

    return None, None

def hex_unpaired_start():
    """
    Hex grid of semicircles with alternating orientations —
    no pairs, flat edges interlock with neighbors.
    """
    for _ in range(50):
        positions = []
        for row in range(-2, 3):
            for col in range(-3, 4):
                x = col * 1.6 + (0.8 if row % 2 else 0) + random.gauss(0, 0.05)
                y = row * 1.38 + random.gauss(0, 0.05)
                if x*x + y*y < 14:
                    positions.append((x, y))
        random.shuffle(positions)

        raw = []; polys = []
        for i, (x, y) in enumerate(positions[:N]):
            # Alternate orientation: point flat edge toward neighbor
            t = random.uniform(0, 2*math.pi)
            p = make_poly(x, y, t)
            if all(p.intersection(ep).area < 1e-6 for ep in polys):
                raw.append({'x': x, 'y': y, 'theta': t})
                polys.append(p)

        if len(raw) == N:
            sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
            r = validate_and_score(sol)
            if r.valid:
                return raw, r.score

    return None, None

if __name__ == '__main__':
    _, _, _, global_best = load_best_score()
    print(f"Starting best: {global_best:.6f}", flush=True)

    run_num = 0
    t_total = time.time()
    max_runtime = 3600 * 6  # 6 hours

    starts = [
        ('boundary_biased', boundary_biased_start),
        ('random_valid',    lambda: random_valid_start(r_max=2.8)),
        ('hex_unpaired',    hex_unpaired_start),
        ('random_tight',    lambda: random_valid_start(r_max=2.5)),
        ('boundary_biased', boundary_biased_start),
        ('random_valid',    lambda: random_valid_start(r_max=3.0)),
    ]

    while time.time() - t_total < max_runtime:
        run_num += 1
        label, init_fn = starts[(run_num - 1) % len(starts)]
        label = f"{label}_{run_num}"

        print(f"\n{'='*50}\nRun {run_num}: {label}", flush=True)

        raw, init_score = init_fn()
        if raw is None:
            print(f"  Could not build start, skipping", flush=True)
            continue
        print(f"  Init score: {init_score:.4f}", flush=True)

        xs = np.array([d['x'] for d in raw])
        ys = np.array([d['y'] for d in raw])
        ts = np.array([d['theta'] for d in raw])

        t0 = time.time()
        result, _ = mod.sa_run(
            xs, ys, ts,
            n_steps=2000000,
            T_start=1.5, T_end=0.001,
            lam_start=5.0, lam_end=3000.0,
            seed=run_num * 1337,
            label=label
        )
        elapsed = time.time() - t0

        if result is not None:
            rx, ry, rt = result
            val = mod.official_validate(rx, ry, rt)
            if val.valid:
                print(f"  VALID: {val.score:.6f} ({elapsed:.0f}s)", flush=True)
                if val.score < global_best:
                    global_best = val.score
                    raw_r = [{'x': float(rx[i]), 'y': float(ry[i]), 'theta': float(rt[i])} for i in range(N)]
                    centered, score = center_solution(raw_r)
                    if score is not None:
                        with open(BEST_FILE, 'w') as f:
                            json.dump(centered, f, indent=2)
                        sol_v = [Semicircle(d['x'],d['y'],d['theta']) for d in centered]
                        r_v = validate_and_score(sol_v)
                        from src.semicircle_packing.visualization import plot_packing
                        plot_packing(sol_v, r_v.mec, save_path='best_solution.png')
                    print(f"  *** NEW BEST: {global_best:.6f} ***", flush=True)
            else:
                print(f"  INVALID ({elapsed:.0f}s)", flush=True)

    print(f"\nFinal best: {global_best:.6f}")
