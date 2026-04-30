#!/usr/bin/env python3
"""
deliberate_starts.py — Deliberate topology starts for shape packing.

Three strategies for creating controlled starting configurations:

Strategy 3: Structured perturbations of best_solution.json
  - swap two shapes' positions
  - flip a shape's orientation (θ += π)
  - rotate only inner/outer ring shapes
  - radial compress/expand a ring

Strategy 4: Analytical placement for promising topologies
  - compute ring radii and angular spacing geometrically
  - place shapes at exact positions with orientations matching
    the observed best-solution pattern

Strategy 5: Extract & vary the best solution's topology
  - classify shapes into rings by radius clustering
  - extract angular spacing pattern
  - generate variations: ring radius sweep, angular offset sweep,
    orientation pattern variations

Usage:
    python3 deliberate_starts.py --strategy 3 --variants 20
    python3 deliberate_starts.py --strategy 4 --topology 3-11-1
    python3 deliberate_starts.py --strategy 5 --variants 20
    python3 deliberate_starts.py --all --variants 10
"""

import sys, os, json, math, time, argparse
import numpy as np

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
from lns4_worker import (gjk_overlap_full, gjk_polish, official_score,
                         load_best, load_best_score, save_if_better,
                         N, TWO_PI)

BEST_FILE = 'best_solution.json'


def load_best_solution():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    xs = np.array([s['x'] for s in raw], dtype=np.float64)
    ys = np.array([s['y'] for s in raw], dtype=np.float64)
    ts = np.array([s['theta'] for s in raw], dtype=np.float64)
    return xs, ys, ts


def classify_rings(xs, ys, ts=None, gap=0.3):
    """Cluster shapes into rings by radius from centroid."""
    cx, cy = xs.mean(), ys.mean()
    rs = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    order = np.argsort(rs)
    sorted_rs = rs[order]

    clusters = [[order[0]]]
    for k in range(1, len(order)):
        if sorted_rs[k] - sorted_rs[k-1] > gap:
            clusters.append([])
        clusters[-1].append(order[k])

    rings = []
    for cluster in clusters:
        idxs = np.array(cluster)
        r_mean = rs[idxs].mean()
        angles = np.arctan2(ys[idxs] - cy, xs[idxs] - cx)
        ring = {
            'indices': idxs,
            'r_mean': r_mean,
            'n': len(idxs),
            'angles': angles,
        }
        if ts is not None:
            ring['thetas'] = ts[idxs]
        rings.append(ring)
    return rings, cx, cy


def is_valid(xs, ys, ts):
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    result = validate_and_score(sol)
    return result.valid


def optimize_start(xs, ys, ts, label, best_ref, sa_steps=30_000_000, polish_steps=5_000_000):
    """Run SA compress + GJK polish on a starting config."""
    from sa_v2 import sa_run_v2_wrapper as phi_sa

    seed_val = int(time.time() * 1000) % (2**31)

    # Phase 1: hot SA compress
    rx, ry, rt, _ = phi_sa(
        xs, ys, ts,
        n_steps=sa_steps,
        T_start=4.0, T_end=0.02,
        lam_start=10, lam_end=1000,
        seed=seed_val,
    )
    if rx is None:
        print(f'  {label}: SA phase 1 failed')
        return float('inf')

    r1 = official_score(rx, ry, rt)[0]
    print(f'  {label} P1: R={r1:.4f}', flush=True)

    if r1 > 3.5:
        return r1

    # Phase 2: cold SA squeeze
    cx, cy, ct, _ = phi_sa(
        rx, ry, rt,
        n_steps=sa_steps * 3,
        T_start=0.5, T_end=0.0005,
        lam_start=200, lam_end=10000,
        seed=seed_val + 1,
    )
    if cx is None:
        cx, cy, ct = rx, ry, rt

    r2 = official_score(cx, cy, ct)[0]
    print(f'  {label} P2: R={r2:.6f}', flush=True)

    if r2 > 3.1:
        return r2

    # Phase 3: GJK polish
    px, py, pt, _ = gjk_polish(
        cx, cy, ct,
        n_steps=polish_steps,
        T_start=0.003, T_end=0.000005,
        lam=80_000.0,
        seed=seed_val + 2,
    )
    r3 = official_score(px, py, pt)[0]
    print(f'  {label} P3: R={r3:.6f}', flush=True)

    saved = save_if_better(px, py, pt, r3, best_ref, f'deliberate_{label}')
    if saved:
        print(f'  ★ NEW BEST R={r3:.6f} [{label}]', flush=True)

    return r3


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 3: Structured perturbations
# ═══════════════════════════════════════════════════════════════════════════════

def perturb_swap_positions(xs, ys, ts, i, j):
    """Swap positions of shapes i and j, keep orientations."""
    nx, ny, nt = xs.copy(), ys.copy(), ts.copy()
    nx[i], nx[j] = nx[j], nx[i]
    ny[i], ny[j] = ny[j], ny[i]
    return nx, ny, nt


def perturb_flip_orientation(xs, ys, ts, i):
    """Flip shape i's orientation by π."""
    nx, ny, nt = xs.copy(), ys.copy(), ts.copy()
    nt[i] = (nt[i] + math.pi) % TWO_PI
    return nx, ny, nt


def perturb_rotate_ring(xs, ys, ts, ring_indices, delta_angle):
    """Rotate all shapes in a ring by delta_angle around centroid."""
    nx, ny, nt = xs.copy(), ys.copy(), ts.copy()
    cx, cy = xs.mean(), ys.mean()
    cos_d, sin_d = math.cos(delta_angle), math.sin(delta_angle)
    for i in ring_indices:
        dx, dy = nx[i] - cx, ny[i] - cy
        nx[i] = cx + dx * cos_d - dy * sin_d
        ny[i] = cy + dx * sin_d + dy * cos_d
        nt[i] = (nt[i] + delta_angle) % TWO_PI
    return nx, ny, nt


def perturb_scale_ring(xs, ys, ts, ring_indices, scale):
    """Scale ring radius by factor (positions only, not orientations)."""
    nx, ny, nt = xs.copy(), ys.copy(), ts.copy()
    cx, cy = xs.mean(), ys.mean()
    for i in ring_indices:
        dx, dy = nx[i] - cx, ny[i] - cy
        nx[i] = cx + dx * scale
        ny[i] = cy + dy * scale
    return nx, ny, nt


def perturb_swap_rings(xs, ys, ts, ring_a_indices, ring_b_indices):
    """Swap radial positions of two rings (shapes move to each other's radii)."""
    nx, ny, nt = xs.copy(), ys.copy(), ts.copy()
    cx, cy = xs.mean(), ys.mean()

    # Compute mean radii
    r_a = np.mean(np.sqrt((xs[ring_a_indices] - cx)**2 + (ys[ring_a_indices] - cy)**2))
    r_b = np.mean(np.sqrt((xs[ring_b_indices] - cx)**2 + (ys[ring_b_indices] - cy)**2))

    for i in ring_a_indices:
        r_i = math.sqrt((nx[i] - cx)**2 + (ny[i] - cy)**2)
        if r_i > 1e-6:
            scale = r_b / r_i
            nx[i] = cx + (nx[i] - cx) * scale
            ny[i] = cy + (ny[i] - cy) * scale

    for i in ring_b_indices:
        r_i = math.sqrt((nx[i] - cx)**2 + (ny[i] - cy)**2)
        if r_i > 1e-6:
            scale = r_a / r_i
            nx[i] = cx + (nx[i] - cx) * scale
            ny[i] = cy + (ny[i] - cy) * scale

    return nx, ny, nt


def generate_strategy3(n_variants):
    """Generate structured perturbations of best solution."""
    xs, ys, ts = load_best_solution()
    rings, cx, cy = classify_rings(xs, ys, ts)
    rng = np.random.default_rng(42)

    starts = []

    # Swap pairs of shapes from different rings
    if len(rings) >= 2:
        for _ in range(min(n_variants // 4, 10)):
            r0, r1 = rng.choice(len(rings), 2, replace=False)
            i = rng.choice(rings[r0]['indices'])
            j = rng.choice(rings[r1]['indices'])
            nx, ny, nt = perturb_swap_positions(xs, ys, ts, i, j)
            starts.append((nx, ny, nt, f's3_swap_{i}_{j}'))

    # Flip orientations of individual shapes
    for idx in rng.choice(N, min(n_variants // 4, 5), replace=False):
        nx, ny, nt = perturb_flip_orientation(xs, ys, ts, idx)
        starts.append((nx, ny, nt, f's3_flip_{idx}'))

    # Rotate outer ring by small angles
    if len(rings) >= 2:
        outer = rings[-1]['indices'] if len(rings[-1]['indices']) > 1 else rings[-2]['indices']
        for delta in np.linspace(-0.3, 0.3, min(n_variants // 4, 7)):
            if abs(delta) < 0.01:
                continue
            nx, ny, nt = perturb_rotate_ring(xs, ys, ts, outer, delta)
            starts.append((nx, ny, nt, f's3_rot_{delta:.3f}'))

    # Scale inner ring
    inner = rings[0]['indices']
    for scale in [0.8, 0.9, 1.1, 1.2]:
        nx, ny, nt = perturb_scale_ring(xs, ys, ts, inner, scale)
        starts.append((nx, ny, nt, f's3_scale_inner_{scale:.1f}'))

    # Multi-flip: flip all shapes in inner ring
    nx, ny, nt = xs.copy(), ys.copy(), ts.copy()
    for i in inner:
        nt[i] = (nt[i] + math.pi) % TWO_PI
    starts.append((nx, ny, nt, 's3_flip_inner_all'))

    return starts[:n_variants]


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 4: Analytical placement
# ═══════════════════════════════════════════════════════════════════════════════

def analytical_ring(n, r, orient_offset=0.0, orient_mode='tangential'):
    """Place n semicircles analytically on a ring of radius r.

    orient_mode:
      'tangential'  — flat edge tangent to ring (θ = angle + π/2 + offset)
      'radial_out'  — flat edge pointing outward (θ = angle + offset)
      'radial_in'   — flat edge pointing inward (θ = angle + π + offset)
      'alternating' — alternate radial_out and radial_in
    """
    xs, ys, ts = [], [], []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = r * math.cos(angle)
        y = r * math.sin(angle)

        if orient_mode == 'tangential':
            t = angle + math.pi / 2 + orient_offset
        elif orient_mode == 'radial_out':
            t = angle + orient_offset
        elif orient_mode == 'radial_in':
            t = angle + math.pi + orient_offset
        elif orient_mode == 'alternating':
            t = angle + (0 if i % 2 == 0 else math.pi) + orient_offset
        else:
            t = angle + orient_offset

        xs.append(x)
        ys.append(y)
        ts.append(t % TWO_PI)
    return xs, ys, ts


def analytical_seed(topology, inner_r=None, mid_r=None, outer_r=None,
                    inner_orient='tangential', mid_orient='tangential',
                    outer_orient='tangential', orient_offset=0.0):
    """Create an analytical seed for a given topology like '3-11-1'.

    Auto-computes ring radii if not specified, based on semicircle radius = 1.
    """
    parts = [int(x) for x in topology.split('-')]
    assert sum(parts) == N, f"Topology {topology} doesn't sum to {N}"

    xs, ys, ts = [], [], []

    if len(parts) == 2:
        n_inner, n_outer = parts
        if inner_r is None:
            # For n_inner shapes on inner ring: circumradius ~ n_inner / (2π) * spacing
            # Minimum spacing ~2.05 (slightly > 2 for unit semicircles)
            inner_r = max(0.3, n_inner * 2.05 / (2 * math.pi)) if n_inner > 1 else 0.0
        if outer_r is None:
            outer_r = max(inner_r + 2.1, n_outer * 2.05 / (2 * math.pi))

        ix, iy, it = analytical_ring(n_inner, inner_r, orient_offset, inner_orient)
        ox, oy, ot = analytical_ring(n_outer, outer_r, orient_offset, outer_orient)
        xs = ix + ox
        ys = iy + oy
        ts = it + ot

    elif len(parts) == 3:
        n_inner, n_mid, n_outer = parts
        if inner_r is None:
            inner_r = max(0.3, n_inner * 2.05 / (2 * math.pi)) if n_inner > 1 else 0.0
        if mid_r is None:
            mid_r = max(inner_r + 2.1, n_mid * 2.05 / (2 * math.pi))
        if outer_r is None:
            outer_r = max(mid_r + 2.1, n_outer * 2.05 / (2 * math.pi)) if n_outer > 0 else 0.0

        ix, iy, it = analytical_ring(n_inner, inner_r, orient_offset, inner_orient)
        mx, my, mt = analytical_ring(n_mid, mid_r, orient_offset, mid_orient)
        ox, oy, ot = analytical_ring(n_outer, outer_r, orient_offset, outer_orient)
        xs = ix + mx + ox
        ys = iy + my + oy
        ts = it + mt + ot

    return (np.array(xs, dtype=np.float64),
            np.array(ys, dtype=np.float64),
            np.array(ts, dtype=np.float64))


def generate_strategy4(topology='3-11-1'):
    """Generate analytical placements for a given topology with parameter sweeps."""
    starts = []

    # Sweep ring radii
    parts = [int(x) for x in topology.split('-')]
    n_rings = len([p for p in parts if p > 0])

    orient_modes = ['tangential', 'radial_out', 'alternating']

    if len(parts) == 3:
        n_inner, n_mid, n_outer = parts

        # Base analytical radii
        base_inner = max(0.3, n_inner * 2.05 / (2 * math.pi)) if n_inner > 1 else 0.0
        base_mid = max(base_inner + 2.1, n_mid * 2.05 / (2 * math.pi))

        # Sweep mid ring radius (this is where most shapes are for 3-11-1)
        for mid_r in np.arange(max(1.5, base_mid - 0.5), base_mid + 0.8, 0.2):
            for orient in orient_modes:
                xs, ys, ts = analytical_seed(
                    topology, inner_r=base_inner, mid_r=mid_r,
                    mid_orient=orient
                )
                starts.append((xs, ys, ts, f's4_{topology}_mr{mid_r:.1f}_{orient[:3]}'))

        # Sweep inner ring radius
        for inner_r in np.arange(0.3, 1.5, 0.3):
            xs, ys, ts = analytical_seed(
                topology, inner_r=inner_r, mid_r=base_mid,
                inner_orient='tangential', mid_orient='tangential'
            )
            starts.append((xs, ys, ts, f's4_{topology}_ir{inner_r:.1f}'))

        # Sweep angular offset
        for offset in np.linspace(0, math.pi / n_mid, 5):
            xs, ys, ts = analytical_seed(
                topology, orient_offset=offset
            )
            starts.append((xs, ys, ts, f's4_{topology}_off{offset:.3f}'))

    elif len(parts) == 2:
        n_inner, n_outer = parts
        base_inner = max(0.3, n_inner * 2.05 / (2 * math.pi)) if n_inner > 1 else 0.0
        base_outer = max(base_inner + 2.1, n_outer * 2.05 / (2 * math.pi))

        for outer_r in np.arange(max(2.0, base_outer - 0.5), base_outer + 0.8, 0.2):
            for orient in orient_modes:
                xs, ys, ts = analytical_seed(
                    topology, inner_r=base_inner, outer_r=outer_r,
                    outer_orient=orient
                )
                starts.append((xs, ys, ts, f's4_{topology}_or{outer_r:.1f}_{orient[:3]}'))

    return starts


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 5: Extract & vary best solution topology
# ═══════════════════════════════════════════════════════════════════════════════

def generate_strategy5(n_variants):
    """Extract topology from best solution and generate systematic variations."""
    xs, ys, ts = load_best_solution()
    rings, cx, cy = classify_rings(xs, ys, ts)
    rng = np.random.default_rng(123)

    starts = []

    # Get the actual topology string
    topo_str = '-'.join(str(r['n']) for r in rings)
    print(f'  Extracted topology: {topo_str}')
    print(f'  Ring radii: {[f"{r["r_mean"]:.3f}" for r in rings]}')

    # Variation A: radius sweep — scale each ring's radius independently
    for ring_idx, ring in enumerate(rings):
        if ring['n'] <= 1:
            continue
        for factor in [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]:
            nx, ny, nt = perturb_scale_ring(xs, ys, ts, ring['indices'], factor)
            starts.append((nx, ny, nt, f's5_ring{ring_idx}_scale{factor:.2f}'))

    # Variation B: angular offset sweep — rotate each ring relative to others
    for ring_idx, ring in enumerate(rings):
        if ring['n'] <= 1:
            continue
        n = ring['n']
        # One inter-shape angular spacing is 2π/n
        step = 2 * math.pi / n
        for frac in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
            delta = step * frac
            nx, ny, nt = perturb_rotate_ring(xs, ys, ts, ring['indices'], delta)
            starts.append((nx, ny, nt, f's5_ring{ring_idx}_rot{frac:+.2f}'))

    # Variation C: orientation pattern variations
    # Compute relative orientations (θ - angular_position) for each ring
    for ring_idx, ring in enumerate(rings):
        if ring['n'] <= 1:
            continue
        idxs = ring['indices']
        angles_pos = np.arctan2(ys[idxs] - cy, xs[idxs] - cx)
        rel_thetas = (ts[idxs] - angles_pos) % TWO_PI

        # Mirror the relative orientations
        nx, ny, nt = xs.copy(), ys.copy(), ts.copy()
        for k, i in enumerate(idxs):
            new_rel = (TWO_PI - rel_thetas[k]) % TWO_PI
            nt[i] = (angles_pos[k] + new_rel) % TWO_PI
        starts.append((nx, ny, nt, f's5_ring{ring_idx}_mirror_orient'))

        # Uniform relative orientation (set all to mean)
        mean_rel = np.mean(rel_thetas)
        nx, ny, nt = xs.copy(), ys.copy(), ts.copy()
        for k, i in enumerate(idxs):
            nt[i] = (angles_pos[k] + mean_rel) % TWO_PI
        starts.append((nx, ny, nt, f's5_ring{ring_idx}_uniform_orient'))

    # Variation D: reconstruct analytically from extracted parameters
    # Use the extracted ring structure but with exact angular spacing
    all_xs, all_ys, all_ts = [], [], []
    for ring in rings:
        n = ring['n']
        r = ring['r_mean']
        # Average relative theta for this ring
        idxs = ring['indices']
        angles_pos = np.arctan2(ys[idxs] - cy, xs[idxs] - cx)
        rel_thetas = (ts[idxs] - angles_pos) % TWO_PI
        mean_rel = float(np.mean(rel_thetas))

        rx, ry, rt = analytical_ring(n, r, orient_offset=mean_rel, orient_mode='radial_out')
        all_xs.extend(rx)
        all_ys.extend(ry)
        all_ts.extend(rt)

    starts.append((
        np.array(all_xs), np.array(all_ys), np.array(all_ts),
        's5_analytical_reconstruction'
    ))

    # Variation E: jitter the analytical reconstruction
    for jitter in [0.05, 0.1, 0.15]:
        for trial in range(3):
            jx = np.array(all_xs) + rng.normal(0, jitter, N)
            jy = np.array(all_ys) + rng.normal(0, jitter, N)
            jt = np.array(all_ts) + rng.normal(0, jitter * 2, N)
            starts.append((jx, jy, jt, f's5_analytical_jitter{jitter:.2f}_t{trial}'))

    return starts[:n_variants]


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Deliberate topology starts')
    parser.add_argument('--strategy', type=int, choices=[3, 4, 5],
                        help='Which strategy to run (3, 4, or 5)')
    parser.add_argument('--all', action='store_true', help='Run all strategies')
    parser.add_argument('--topology', type=str, default='3-11-1',
                        help='Topology for strategy 4 (e.g. 3-11-1, 4-11, 5-10)')
    parser.add_argument('--variants', type=int, default=20,
                        help='Max variants per strategy')
    parser.add_argument('--sa-steps', type=int, default=30_000_000,
                        help='SA steps per phase')
    parser.add_argument('--polish-steps', type=int, default=5_000_000,
                        help='GJK polish steps')
    parser.add_argument('--dry-run', action='store_true',
                        help='List starts without optimizing')
    args = parser.parse_args()

    # JIT warmup
    print('JIT warmup...', flush=True)
    gjk_overlap_full(np.zeros(3), np.zeros(3), np.zeros(3))
    gjk_polish(np.zeros(N), np.zeros(N), np.zeros(N), 200, 0.01, 0.001, 1000.0, 0)

    strategies = []
    if args.all or args.strategy == 3:
        strategies.append(('Strategy 3: Structured perturbations', generate_strategy3(args.variants)))
    if args.all or args.strategy == 4:
        strategies.append((f'Strategy 4: Analytical ({args.topology})', generate_strategy4(args.topology)))
    if args.all or args.strategy == 5:
        strategies.append(('Strategy 5: Extract & vary', generate_strategy5(args.variants)))

    if not strategies:
        print('No strategy selected. Use --strategy 3/4/5 or --all')
        return

    best_ref = [load_best_score()]
    print(f'Current best: R={best_ref[0]:.6f}')

    results = []
    for strat_name, starts in strategies:
        print(f'\n{"="*60}')
        print(f'  {strat_name} ({len(starts)} starts)')
        print(f'{"="*60}')

        for xs, ys, ts, label in starts:
            if args.dry_run:
                # Quick quality check
                overlap = gjk_overlap_full(xs, ys, ts)
                r_est = max(math.sqrt(x**2 + y**2) + 1 for x, y in zip(xs, ys))
                print(f'  {label}: R_est={r_est:.2f}, overlap={overlap:.6f}')
                continue

            t0 = time.time()
            r = optimize_start(xs, ys, ts, label, best_ref, args.sa_steps, args.polish_steps)
            elapsed = int(time.time() - t0)
            results.append((r, label, elapsed))
            print(f'  → {label}: R={r:.4f} ({elapsed}s) | global={best_ref[0]:.6f}')

    if results:
        results.sort()
        print(f'\n{"="*60}')
        print(f'  RESULTS (sorted)')
        print(f'{"="*60}')
        for r, label, elapsed in results[:20]:
            marker = ' ★' if r <= best_ref[0] else ''
            print(f'  {label}: R={r:.6f} ({elapsed}s){marker}')
        print(f'\nFinal best on disk: R={best_ref[0]:.6f}')


if __name__ == '__main__':
    main()
