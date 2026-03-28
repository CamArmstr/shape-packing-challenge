"""
seeds.py — Seed topology generators for semicircle packing optimization.

Eight distinct shell distributions, each generating valid initial configurations:
  1. 4-11  (current best topology)
  2. 5-10  (circle-packing analog)
  3. 3-5-7 (balanced 3-shell)
  4. 2-5-8 (tight outer ring, solid middle)
  5. 3-4-8 (variant 3-shell)
  6. 1-5-9 (minimal center, heavy outer)
  7. 2-6-7 (heavy middle ring)
  8. 6-9   (2-shell, large inner cluster)

All generators use boundary-first greedy initialization:
  - Outer ring: flat edges facing boundary (maximizing boundary nesting)
  - Middle ring: tangential orientation
  - Inner ring: random/radial orientation
"""

import math
import random
import os
import sys
import json

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(__file__))
from src.semicircle_packing.geometry import Semicircle
from src.semicircle_packing.scoring import validate_and_score

N = 15
ARC = 64


def make_poly(x, y, theta):
    angles = np.linspace(theta - math.pi / 2, theta + math.pi / 2, ARC)
    pts = list(zip(x + np.cos(angles), y + np.sin(angles)))
    pts.append((x, y))
    return Polygon(pts)


def _validate(xs, ys, ts):
    """Quick Shapely validation."""
    if len(xs) != N:
        return False
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    return validate_and_score(sol).valid


def _try_place(x, y, t, polys):
    """Try to place a semicircle, return True if no overlap with existing."""
    p = make_poly(x, y, t)
    for ep in polys:
        if p.intersection(ep).area > 1e-6:
            return False
    return True


def _place_ring(n_ring, r_center, r_spread, orient_mode, polys, xs, ys, ts, max_attempts=800):
    """
    Place n_ring semicircles on a ring at radius r_center ± r_spread.
    orient_mode: 'outward-tangential' (arc 30-75° from radial, matching observed best),
                 'tangential', 'random'
    Returns True if all placed successfully.
    """
    placed = 0
    base_angles = np.linspace(0, 2 * math.pi, n_ring, endpoint=False)
    base_angles += random.uniform(0, 2 * math.pi / max(n_ring, 1))

    for i in range(n_ring):
        success = False
        a = base_angles[i] + random.gauss(0, 0.15)
        for attempt in range(max_attempts):
            if attempt < 10:
                r = r_center + random.gauss(0, 0.05)
                angle = a + random.gauss(0, 0.05 * attempt)
            elif attempt < 100:
                r = r_center + random.uniform(-r_spread, r_spread)
                angle = a + random.gauss(0, 0.4)
            else:
                # Wide random search
                r = r_center + random.uniform(-r_spread * 2, r_spread * 2)
                angle = random.uniform(0, 2 * math.pi)

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            if orient_mode == 'outward-tangential':
                # Match observed pattern: θ_rel ∈ [20°, 75°] from radial
                rel_angle = math.radians(random.uniform(20, 75))
                t = angle + rel_angle
            elif orient_mode == 'tangential':
                t = angle + math.pi / 2 + random.gauss(0, 0.3)
            elif orient_mode == 'inward':
                t = angle + math.pi + random.gauss(0, 0.3)
            else:  # random
                t = random.uniform(0, 2 * math.pi)

            # After many attempts, try any orientation
            if attempt > 200:
                t = random.uniform(0, 2 * math.pi)

            if _try_place(x, y, t, polys):
                xs.append(x)
                ys.append(y)
                ts.append(t)
                polys.append(make_poly(x, y, t))
                placed += 1
                success = True
                break

        if not success:
            return False
    return True


def seed_from_best(noise=0.03, seed=None):
    """Start from current best with small noise."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    best_path = os.path.join(os.path.dirname(__file__), 'best_solution.json')
    with open(best_path) as f:
        raw = json.load(f)
    xs = np.array([s['x'] for s in raw]) + np.random.randn(N) * noise
    ys = np.array([s['y'] for s in raw]) + np.random.randn(N) * noise
    ts = np.array([s['theta'] for s in raw]) + np.random.randn(N) * noise * 2
    return xs, ys, ts


def _shell_seed(inner_n, mid_n, outer_n, inner_r, mid_r, outer_r, seed=None):
    """
    Generic shell-based seed generator.
    Strategy: greedy sequential placement with collision avoidance.
    Places all semicircles one at a time, each at the best available position
    on its target ring. Uses wider search if ideal position is blocked.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    assert inner_n + mid_n + outer_n == N

    # Build placement plan: (ring_r, orient_mode, ring_n)
    plan = []
    # Outer first (boundary-first greedy)
    for _ in range(outer_n):
        plan.append((outer_r, 'outward-tangential'))
    for _ in range(mid_n):
        plan.append((mid_r, 'tangential'))
    for _ in range(inner_n):
        plan.append((inner_r, 'random'))

    for attempt in range(200):
        xs, ys, ts, polys = [], [], [], []
        success = True

        for pi, (target_r, orient_mode) in enumerate(plan):
            placed = False
            for try_i in range(600):
                if try_i < 20:
                    # Try on the target ring with some angular spread
                    r = target_r + random.gauss(0, 0.1)
                    a = random.uniform(0, 2 * math.pi)
                elif try_i < 100:
                    r = target_r + random.uniform(-0.6, 0.6)
                    a = random.uniform(0, 2 * math.pi)
                else:
                    # Desperate: anywhere in a wider range
                    r = random.uniform(max(0.1, target_r - 1.2), target_r + 1.2)
                    a = random.uniform(0, 2 * math.pi)

                x = r * math.cos(a)
                y = r * math.sin(a)

                if orient_mode == 'outward-tangential':
                    t = a + math.radians(random.uniform(20, 75))
                elif orient_mode == 'tangential':
                    t = a + math.pi / 2 + random.gauss(0, 0.3)
                else:
                    t = random.uniform(0, 2 * math.pi)

                if try_i > 200:
                    t = random.uniform(0, 2 * math.pi)

                if _try_place(x, y, t, polys):
                    xs.append(x)
                    ys.append(y)
                    ts.append(t)
                    polys.append(make_poly(x, y, t))
                    placed = True
                    break

            if not placed:
                success = False
                break

        if success and len(xs) == N and _validate(xs, ys, ts):
            return np.array(xs), np.array(ys), np.array(ts)

    return None


def seed_4_11(seed=None):
    """4 inner + 11 outer (current best topology)."""
    return _shell_seed(4, 0, 11, 1.0, 0, 3.0, seed=seed)


def seed_5_10(seed=None):
    """5 inner + 10 outer (circle-packing analog for n=15)."""
    return _shell_seed(5, 0, 10, 1.2, 0, 3.0, seed=seed)


def seed_3_5_7(seed=None):
    """3 inner + 5 middle + 7 outer (balanced 3-shell)."""
    return _shell_seed(3, 5, 7, 0.6, 1.8, 3.0, seed=seed)


def seed_2_5_8(seed=None):
    """2 inner + 5 middle + 8 outer (tight outer ring, solid middle)."""
    return _shell_seed(2, 5, 8, 0.5, 1.7, 3.0, seed=seed)


def seed_3_4_8(seed=None):
    """3 inner + 4 middle + 8 outer (variant 3-shell)."""
    return _shell_seed(3, 4, 8, 0.6, 1.8, 3.0, seed=seed)


def seed_1_5_9(seed=None):
    """1 center + 5 middle + 9 outer (minimal center, heavy outer)."""
    return _shell_seed(1, 5, 9, 0.0, 1.7, 3.0, seed=seed)


def seed_2_6_7(seed=None):
    """2 inner + 6 middle + 7 outer (heavy middle ring)."""
    return _shell_seed(2, 6, 7, 0.5, 1.8, 3.0, seed=seed)


def seed_6_9(seed=None):
    """6 inner + 9 outer (2-shell, large inner cluster)."""
    return _shell_seed(6, 0, 9, 1.3, 0, 3.0, seed=seed)


# ── Topology A: Conjugate-pair seeding ────────────────────────────────────────

def _circle_pack_pairs(n_pairs, n_singles, pair_rings, single_rings, seed=None):
    """
    Circle-packing approach to conjugate pairs:
    1. Place n_pairs "full disks" (center + radius 1) using distance > 2 constraint
    2. Split each disk into 2 antiparallel semicircles at the same center
    3. Place n_singles individual semicircles in remaining gaps
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for attempt in range(80):
        # Phase 1: place disk centers (simple distance check, no Shapely)
        disk_xs, disk_ys = [], []
        ok = True

        for ring_r, ring_n in pair_rings:
            base_a = random.uniform(0, 2 * math.pi / max(ring_n, 1))
            angles = np.linspace(0, 2 * math.pi, ring_n, endpoint=False) + base_a
            for a in angles:
                placed = False
                for tri in range(80):
                    if tri < 20:
                        r = ring_r + random.gauss(0, 0.1)
                        ang = a + random.gauss(0, 0.1)
                    elif tri < 60:
                        r = ring_r + random.uniform(-0.3, 0.3)
                        ang = a + random.uniform(-0.3, 0.3)
                    else:
                        r = ring_r + random.uniform(-0.6, 0.6)
                        ang = random.uniform(0, 2 * math.pi)
                    x = r * math.cos(ang)
                    y = r * math.sin(ang)
                    # Check distance > 2.02 from all existing disks
                    collision = False
                    for dx, dy in zip(disk_xs, disk_ys):
                        if (x - dx) ** 2 + (y - dy) ** 2 < 2.02 ** 2:
                            collision = True
                            break
                    if not collision:
                        disk_xs.append(x)
                        disk_ys.append(y)
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if not ok:
                break

        if not ok or len(disk_xs) != n_pairs:
            continue

        # Phase 2: split each disk into antiparallel pair
        xs, ys, ts = [], [], []
        polys = []
        for dx, dy in zip(disk_xs, disk_ys):
            theta = random.uniform(0, 2 * math.pi)
            xs.extend([dx, dx])
            ys.extend([dy, dy])
            ts.extend([theta, (theta + math.pi) % (2 * math.pi)])
            polys.append(make_poly(dx, dy, theta))
            polys.append(make_poly(dx, dy, (theta + math.pi) % (2 * math.pi)))

        # Phase 3: place singles in gaps using Shapely
        for ring_r, ring_n in single_rings:
            base_a = random.uniform(0, 2 * math.pi / max(ring_n, 1))
            angles = np.linspace(0, 2 * math.pi, ring_n, endpoint=False) + base_a
            for a in angles:
                placed = False
                for tri in range(100):
                    if tri < 20:
                        r = ring_r + random.gauss(0, 0.15)
                        ang = a + random.gauss(0, 0.1)
                    elif tri < 60:
                        r = ring_r + random.uniform(-0.5, 0.5)
                        ang = a + random.uniform(-0.4, 0.4)
                    else:
                        r = random.uniform(0.2, ring_r + 1.0)
                        ang = random.uniform(0, 2 * math.pi)
                    x = r * math.cos(ang)
                    y = r * math.sin(ang)
                    t = random.uniform(0, 2 * math.pi)
                    if _try_place(x, y, t, polys):
                        xs.append(x); ys.append(y); ts.append(t)
                        polys.append(make_poly(x, y, t))
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if not ok:
                break

        if not ok or len(xs) != N:
            continue

        xsa = np.array(xs)
        ysa = np.array(ys)
        tsa = np.array(ts)

        if _validate(xsa, ysa, tsa):
            return xsa, ysa, tsa

    return None


def seed_conjugate(seed=None):
    """
    Conjugate-pair seeding: 6 pairs + 3 singles = 15.
    Place 6 full disks via circle packing, split into antiparallel pairs,
    then place 3 singles in gaps.
    """
    return _circle_pack_pairs(
        n_pairs=6, n_singles=3,
        pair_rings=[(1.0, 2), (2.5, 4)],
        single_rings=[(1.8, 3)],
        seed=seed)


def seed_conjugate_7pairs(seed=None):
    """
    Conjugate-pair seeding: 7 pairs + 1 singleton = 15.
    Place 7 full disks, split into antiparallel pairs, place 1 single.
    1 center + 6 ring at r≈2.5.
    """
    return _circle_pack_pairs(
        n_pairs=7, n_singles=1,
        pair_rings=[(0.0, 1), (2.5, 6)],
        single_rings=[(1.5, 1)],
        seed=seed)


# ── Topology B: C5 pentagonal symmetry ────────────────────────────────────────

def seed_c5(seed=None):
    """
    C5 pentagonal symmetry: 5 conjugate pairs at 72° intervals + 5 outward singles.
    Uses circle packing for pairs, then Shapely placement for singles.
    """
    return _circle_pack_pairs(
        n_pairs=5, n_singles=5,
        pair_rings=[(1.8, 5)],
        single_rings=[(2.8, 5)],
        seed=seed)


def seed_c5_loose(seed=None):
    """
    4 conjugate pairs (1 center + 3 at r≈2.5) + 7 singles at r≈1.8.
    Different pair topology from seed_conjugate and seed_c5.
    """
    return _circle_pack_pairs(
        n_pairs=4, n_singles=7,
        pair_rings=[(0.0, 1), (2.5, 3)],
        single_rings=[(1.5, 4), (2.8, 3)],
        seed=seed)


# ── Topology D: Brick-wall strip seeding ───────────────────────────────────────

def seed_brickwall(seed=None):
    """
    3×5 brick-wall grid: even rows at x=0,2.1,...  odd row offset by half-spacing.
    Even rows arc up (θ=π/2), odd rows arc down (θ=-π/2).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for attempt in range(50):
        xs, ys, ts = [], [], []
        for row in range(3):
            for col in range(5):
                if row % 2 == 0:
                    x = col * 2.1 + random.uniform(-0.1, 0.1)
                    y = row * 2.0 + random.uniform(-0.1, 0.1)
                    t = math.pi / 2 + random.uniform(-0.2, 0.2)
                else:
                    x = col * 2.1 + 1.05 + random.uniform(-0.1, 0.1)
                    y = row * 2.0 + random.uniform(-0.1, 0.1)
                    t = -math.pi / 2 + random.uniform(-0.2, 0.2)
                xs.append(x)
                ys.append(y)
                ts.append(t)

        # Center the configuration
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        xs = [x - cx for x in xs]
        ys = [y - cy for y in ys]

        xsa, ysa, tsa = np.array(xs), np.array(ys), np.array(ts)
        if _validate(xsa, ysa, tsa):
            return xsa, ysa, tsa

    return None


def seed_brickwall_tight(seed=None):
    """
    Tighter brick-wall: spacing 2.05x, 1.8y. Nearly touching semicircles.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for attempt in range(50):
        xs, ys, ts = [], [], []
        for row in range(3):
            for col in range(5):
                if row % 2 == 0:
                    x = col * 2.05 + random.uniform(-0.05, 0.05)
                    y = row * 1.8 + random.uniform(-0.05, 0.05)
                    t = math.pi / 2 + random.uniform(-0.1, 0.1)
                else:
                    x = col * 2.05 + 1.025 + random.uniform(-0.05, 0.05)
                    y = row * 1.8 + random.uniform(-0.05, 0.05)
                    t = -math.pi / 2 + random.uniform(-0.1, 0.1)
                xs.append(x)
                ys.append(y)
                ts.append(t)

        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        xs = [x - cx for x in xs]
        ys = [y - cy for y in ys]

        xsa, ysa, tsa = np.array(xs), np.array(ys), np.array(ts)
        if _validate(xsa, ysa, tsa):
            return xsa, ysa, tsa

    return None


def seed_brickwall_random(seed=None):
    """
    Brick-wall tight layout but random theta per semicircle (π/2 or -π/2)
    and larger perturbation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for attempt in range(50):
        xs, ys, ts = [], [], []
        for row in range(3):
            for col in range(5):
                if row % 2 == 0:
                    x = col * 2.05 + random.uniform(-0.15, 0.15)
                    y = row * 1.8 + random.uniform(-0.15, 0.15)
                else:
                    x = col * 2.05 + 1.025 + random.uniform(-0.15, 0.15)
                    y = row * 1.8 + random.uniform(-0.15, 0.15)
                t = random.choice([math.pi / 2, -math.pi / 2]) + random.uniform(-0.3, 0.3)
                xs.append(x)
                ys.append(y)
                ts.append(t)

        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        xs = [x - cx for x in xs]
        ys = [y - cy for y in ys]

        xsa, ysa, tsa = np.array(xs), np.array(ys), np.array(ts)
        if _validate(xsa, ysa, tsa):
            return xsa, ysa, tsa

    return None


# All seed generators in priority order
SEED_GENERATORS = [
    ('4-11', seed_4_11),
    ('5-10', seed_5_10),
    ('3-5-7', seed_3_5_7),
    ('2-5-8', seed_2_5_8),
    ('3-4-8', seed_3_4_8),
    ('1-5-9', seed_1_5_9),
    # ('2-6-7', seed_2_6_7),  # reserve — swap in if needed
    # ('6-9', seed_6_9),       # reserve
]


def test_all_seeds():
    """Test that all seed generators can produce valid configurations."""
    all_gens = [
        ('4-11', seed_4_11), ('5-10', seed_5_10), ('3-5-7', seed_3_5_7),
        ('2-5-8', seed_2_5_8), ('3-4-8', seed_3_4_8), ('1-5-9', seed_1_5_9),
        ('2-6-7', seed_2_6_7), ('6-9', seed_6_9),
        ('conjugate', seed_conjugate), ('conjugate_7', seed_conjugate_7pairs),
        ('c5', seed_c5), ('c5_loose', seed_c5_loose),
        ('brickwall', seed_brickwall), ('brickwall_tight', seed_brickwall_tight),
        ('brickwall_random', seed_brickwall_random),
    ]
    for name, gen in all_gens:
        result = gen(seed=42)
        if result is not None:
            xs, ys, ts = result
            # Quick R estimate
            r_est = max(math.sqrt(x**2 + y**2) + 1 for x, y in zip(xs, ys))
            print(f"  {name:6s}: ✓ valid (R_est ≈ {r_est:.2f})")
        else:
            print(f"  {name:6s}: ✗ failed to generate")


if __name__ == '__main__':
    print("Testing all seed topology generators...")
    test_all_seeds()
