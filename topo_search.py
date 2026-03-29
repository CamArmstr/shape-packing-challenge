#!/usr/bin/env python3
"""
topo_search.py — Topology grid search seeded from best_solution.json.

Tests unexplored ring topologies by constructing seeds and running
phi-SA + GJK polish. Each topology attempt gets 30M SA steps.

Topologies to test (inner-mid-outer ring counts summing to 15):
  Current: 2-11-2  ← already found
  New:
    1-13-1, 1-12-2, 1-11-3, 1-10-4
    3-9-3,  3-10-2, 3-11-1
    4-7-4,  4-9-2,  4-8-3
    5-5-5,  5-8-2,  5-7-3
    0-13-2, 0-12-3, 0-11-4

Each is initialized with semicircles placed on rings at specific radii,
orientations varied randomly, then compressed with phi-SA.

Best results saved to best_solution.json if they beat the current best.
"""

import sys, os, json, math, time, subprocess, shutil, argparse
import numpy as np
import fcntl

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
from sa_v2 import sa_run_v2_wrapper as phi_sa
from gjk_numba import semicircle_gjk_signed_dist
from lns4_worker import (gjk_overlap_full, gjk_polish, official_score,
                         load_best, load_best_score, save_if_better,
                         N, TWO_PI)

BEST_FILE = 'best_solution.json'
LOG = 'topo_search.log'

# Ring radii for placement (unit semicircles, tuned for tight packing)
RING_RADII = {
    'inner':  0.9,
    'mid':    2.0,
    'outer':  2.85,
}

TOPOLOGIES = [
    # (n_inner, n_mid, n_outer, label)
    (1, 13, 1, '1-13-1'),
    (1, 12, 2, '1-12-2'),
    (1, 11, 3, '1-11-3'),
    (1, 10, 4, '1-10-4'),
    (3,  9, 3, '3-9-3'),
    (3, 10, 2, '3-10-2'),
    (3, 11, 1, '3-11-1'),
    (4,  7, 4, '4-7-4'),
    (4,  9, 2, '4-9-2'),
    (4,  8, 3, '4-8-3'),
    (5,  5, 5, '5-5-5'),
    (5,  8, 2, '5-8-2'),
    (5,  7, 3, '5-7-3'),
    (0, 13, 2, '0-13-2'),
    (0, 12, 3, '0-12-3'),
    (0, 11, 4, '0-11-4'),
    (2, 11, 2, '2-11-2-alt'),  # our known topology, different orientation
    (0, 15, 0, '0-15-0'),      # all on one ring
]


def make_seed(n_inner, n_mid, n_outer, rng, noise=0.15):
    """Place shapes on concentric rings with random orientation."""
    xs, ys, ts = [], [], []

    def place_ring(n, r, noise_scale):
        for i in range(n):
            angle = 2*math.pi*i/n + rng.uniform(-noise_scale, noise_scale)
            x = r * math.cos(angle) + rng.normal(0, noise_scale*0.5)
            y = r * math.sin(angle) + rng.normal(0, noise_scale*0.5)
            t = rng.uniform(0, TWO_PI)
            xs.append(x); ys.append(y); ts.append(t)

    if n_inner > 0: place_ring(n_inner, RING_RADII['inner'], noise)
    if n_mid   > 0: place_ring(n_mid,   RING_RADII['mid'],   noise)
    if n_outer > 0: place_ring(n_outer, RING_RADII['outer'], noise)

    return (np.array(xs, dtype=np.float64),
            np.array(ys, dtype=np.float64),
            np.array(ts, dtype=np.float64))


def run_topology(n_inner, n_mid, n_outer, label, best_ref, log, rng, n_seeds=5):
    """Try n_seeds random seeds for this topology."""
    log.write(f'\n=== {label} ({n_inner}-{n_mid}-{n_outer}) ===\n'); log.flush()
    best_this = float('inf')

    for s in range(n_seeds):
        seed_val = int(rng.integers(0, 2**31))
        xs, ys, ts = make_seed(n_inner, n_mid, n_outer, np.random.RandomState(seed_val))

        # Phase 1: hot phi-SA compress (30M steps)
        rx, ry, rt, _ = phi_sa(
            xs, ys, ts,
            n_steps=30_000_000,
            T_start=4.0, T_end=0.02,
            lam_start=10, lam_end=1000,
            seed=seed_val,
        )
        if rx is None:
            log.write(f'  seed {s}: phi-SA returned None\n'); log.flush()
            continue

        val = official_score(rx, ry, rt)
        r1 = val[0]
        log.write(f'  seed {s} P1: R={r1:.4f}\n'); log.flush()

        if r1 > 3.5:
            continue  # not promising

        # Phase 2: cold phi-SA squeeze (100M steps)
        cx, cy, ct, _ = phi_sa(
            rx, ry, rt,
            n_steps=100_000_000,
            T_start=0.5, T_end=0.0005,
            lam_start=200, lam_end=10000,
            seed=seed_val + 1,
        )
        if cx is None: cx, cy, ct = rx, ry, rt

        val2 = official_score(cx, cy, ct)
        r2 = val2[0]
        log.write(f'  seed {s} P2: R={r2:.6f}\n'); log.flush()

        if r2 > 3.1: continue

        # Phase 3: GJK polish
        px, py, pt, _ = gjk_polish(
            cx, cy, ct,
            n_steps=5_000_000,
            T_start=0.003, T_end=0.000005,
            lam=80_000.0,
            seed=seed_val + 2,
        )
        pol_s, _ = official_score(px, py, pt)
        log.write(f'  seed {s} P3: R={pol_s:.6f}'); log.flush()

        if pol_s < best_this: best_this = pol_s

        saved = save_if_better(px, py, pt, pol_s, best_ref, f'topo_{label}_s{s}')
        if saved:
            log.write(f'  ★ NEW BEST R={pol_s:.6f} [{label}]\n')
            print(f'★ NEW BEST R={pol_s:.6f} [{label} seed {s}]', flush=True)
        else:
            log.write('\n')
        log.flush()

    log.write(f'  best for {label}: R={best_this:.6f}\n'); log.flush()
    return best_this


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=5, help='Seeds per topology')
    parser.add_argument('--topologies', type=str, default='all',
                        help='Comma-separated labels or "all"')
    args = parser.parse_args()

    # JIT warmup
    gjk_overlap_full(np.zeros(3), np.zeros(3), np.zeros(3))
    gjk_polish(np.zeros(N), np.zeros(N), np.zeros(N), 200, 0.01, 0.001, 1000.0, 0)

    best_ref = [load_best_score()]
    rng = np.random.default_rng(int(time.time()) % 100000)

    log = open(LOG, 'a')
    log.write(f'\n\n=== topo_search start {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
    log.write(f'Current best: R={best_ref[0]:.6f}\n')
    log.write(f'Seeds per topology: {args.seeds}\n')
    log.flush()

    print(f'topo_search: {len(TOPOLOGIES)} topologies × {args.seeds} seeds each')
    print(f'Current best: R={best_ref[0]:.6f}')

    topo_list = TOPOLOGIES
    if args.topologies != 'all':
        labels = set(args.topologies.split(','))
        topo_list = [t for t in TOPOLOGIES if t[3] in labels]

    results = []
    for n_inner, n_mid, n_outer, label in topo_list:
        if n_inner + n_mid + n_outer != N:
            print(f'  skip {label}: counts sum to {n_inner+n_mid+n_outer} != {N}')
            continue
        t0 = time.time()
        best_r = run_topology(n_inner, n_mid, n_outer, label, best_ref, log, rng, args.seeds)
        elapsed = int(time.time() - t0)
        results.append((best_r, label, elapsed))
        print(f'  {label}: best={best_r:.4f} in {elapsed}s | global={best_ref[0]:.6f}')

    results.sort()
    log.write('\n=== Summary ===\n')
    for r, label, elapsed in results:
        log.write(f'  {label}: R={r:.6f} ({elapsed}s)\n')
    log.write(f'Final best: R={best_ref[0]:.6f}\n')
    log.close()

    print('\n=== Summary ===')
    for r, label, elapsed in results:
        print(f'  {label}: R={r:.4f} ({elapsed}s)')
    print(f'Final best on disk: R={best_ref[0]:.6f}')


if __name__ == '__main__':
    main()
