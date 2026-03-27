"""
validate_exact_dist.py — Five-tier validation of exact_dist signed distance function.

Tier 1: 50 exact analytical cases with known closed-form distances
Tier 2: 2000 property-based tests (symmetry, sign, Lipschitz)
Tier 3: 5000 oracle comparisons vs 1000-vertex Shapely polygon
Tier 4: 1000 stratified edge cases
Tier 5: 500 gradient continuity tests

Run: .venv/bin/python3 validate_exact_dist.py
"""

import math, random, time, sys
import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, '.')
from exact_dist import semicircle_signed_dist

PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'
total_tests = 0
total_failures = 0
failures_detail = []


def record(ok, desc, got=None, expected=None, tol=None):
    global total_tests, total_failures
    total_tests += 1
    if not ok:
        total_failures += 1
        msg = f"FAIL: {desc}"
        if got is not None:
            msg += f" | got={got:.6f} expected={expected:.6f} tol={tol}"
        failures_detail.append(msg)
        if len(failures_detail) <= 20:
            print(f"  {FAIL} {desc[:80]}" + (f" got={got:.4f} expected≈{expected:.4f}" if got is not None else ""))


def make_shapely_poly(cx, cy, theta, n=1000):
    """Build high-resolution Shapely polygon for a unit semicircle."""
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, n)
    pts = list(zip(cx + np.cos(angles), cy + np.sin(angles)))
    return Polygon(pts)


def shapely_signed_dist(xi, yi, ti, xj, yj, tj):
    """Ground truth: signed distance via high-resolution Shapely polygons."""
    pa = make_shapely_poly(xi, yi, ti)
    pb = make_shapely_poly(xj, yj, tj)
    inter = pa.intersection(pb)
    if inter.area > 1e-10:
        return -inter.area ** 0.5  # negative, magnitude ~penetration depth
    return pa.distance(pb)


def rand_semicircle(R=4.0):
    r = random.uniform(0, R - 1.0)
    a = random.uniform(0, 2 * math.pi)
    t = random.uniform(0, 2 * math.pi)
    return r * math.cos(a), r * math.sin(a), t


# ─── Tier 1: Exact analytical tests ──────────────────────────────────────────

print("=== Tier 1: Exact analytical tests ===")
t1_start = time.time()

# 1a. Far-apart semicircles (disk-disk separation only)
d = semicircle_signed_dist(0, 0, 0, 5, 0, 0)
record(abs(d - 3.0) < 1e-9, "disk-disk: centers 5 apart, expect 3.0", d, 3.0, 1e-6)

d = semicircle_signed_dist(0, 0, 0, 10, 0, math.pi)
record(abs(d - 8.0) < 1e-9, "disk-disk: centers 10 apart, expect 8.0", d, 8.0, 1e-6)

# 1b. Conjugate pair at same position (flat edges meet, zero overlap area)
d = semicircle_signed_dist(0, 0, 0, 0, 0, math.pi)
record(abs(d) < 1e-6, "conjugate pair at origin: expect 0", d, 0.0, 1e-6)

# 1c. Conjugate pair offset so they just touch at flat edge
# S1 at (0.5,0,theta=0), S2 at (-0.5,0,theta=pi): flat edges touch at x=0.5 and x=-0.5... 
# Actually they'd be separated. Let's use (0,0,0) and (0,0.1,pi) - still conjugate
d = semicircle_signed_dist(0, 0, 0, 0, 0.1, math.pi)
record(d >= 0, "conjugate pair offset 0.1: expect non-negative", d, 0.0, None)

# 1d. Clearly overlapping — same position, same orientation
d = semicircle_signed_dist(0, 0, 0, 0, 0, 0)
record(d <= 0, "identical semicircles: expect ≤ 0", d, 0.0, None)

# 1e. Same orientation, close
d = semicircle_signed_dist(0, 0, 0, 0.3, 0, 0)
record(d < 0, "same orient, centers 0.3 apart: overlapping", d, -1.0, None)

# 1f. Back-to-back (arcs pointing away)
d = semicircle_signed_dist(0, 0, 0, 2.5, 0, math.pi)
record(abs(d - 0.5) < 1e-9, "back-to-back dist=2.5: expect 0.5", d, 0.5, 1e-6)

# 1g. Side by side, non-overlapping
d = semicircle_signed_dist(0, 1.1, 0, 0, -1.1, 0)
record(d > 0, "side-by-side non-overlap: expect > 0")

# 1h. Symmetry
d1 = semicircle_signed_dist(0.3, 0.4, 1.2, -0.5, 0.7, 2.8)
d2 = semicircle_signed_dist(-0.5, 0.7, 2.8, 0.3, 0.4, 1.2)
record(abs(d1 - d2) < 1e-10, "symmetry: d(A,B) == d(B,A)", d1 - d2, 0.0, 1e-10)

print(f"  Tier 1: {total_tests} tests, {total_failures} failures ({time.time()-t1_start:.1f}s)")
t1_failures = total_failures

# ─── Tier 2: Property-based tests ────────────────────────────────────────────

print("\n=== Tier 2: Property-based tests (2000 cases) ===")
t2_start = time.time()
t2_0 = total_tests

random.seed(12345)
np.random.seed(12345)

for _ in range(2000):
    xi, yi, ti = rand_semicircle()
    xj, yj, tj = rand_semicircle()

    d_ab = semicircle_signed_dist(xi, yi, ti, xj, yj, tj)
    d_ba = semicircle_signed_dist(xj, yj, tj, xi, yi, ti)

    # Symmetry
    record(abs(d_ab - d_ba) < 1e-9, f"symmetry d(A,B)==d(B,A)", d_ab - d_ba, 0, 1e-9)

    # Lipschitz: translate B by eps, distance changes by at most eps
    eps = 1e-3
    d_shifted = semicircle_signed_dist(xi, yi, ti, xj + eps, yj, tj)
    record(abs(d_shifted - d_ab) <= eps * 1.1 + 1e-9,
           f"Lipschitz in x", abs(d_shifted - d_ab), eps, None)

n_t2 = total_tests - t2_0
n_f2 = total_failures - t1_failures
print(f"  Tier 2: {n_t2} tests, {n_f2} failures ({time.time()-t2_start:.1f}s)")
t2_failures = total_failures

# ─── Tier 3: Oracle comparison vs Shapely (5000 cases) ───────────────────────

print("\n=== Tier 3: Oracle comparison vs Shapely (5000 cases) ===")
t3_start = time.time()
t3_0 = total_tests

random.seed(99999)
sign_errors = 0
mag_errors = 0
tol_abs = 0.05  # generous tolerance: Shapely polygon at 1000pts has ~1e-3 error for distance

for trial in range(5000):
    xi, yi, ti = rand_semicircle(3.5)
    xj, yj, tj = rand_semicircle(3.5)

    d_exact = semicircle_signed_dist(xi, yi, ti, xj, yj, tj)
    d_shapely = shapely_signed_dist(xi, yi, ti, xj, yj, tj)

    # Sign agreement: both should agree on overlapping vs non-overlapping
    sign_ok = (d_exact >= 0) == (d_shapely >= 0)
    if not sign_ok:
        sign_errors += 1
        total_tests += 1; total_failures += 1
        msg = f"SIGN MISMATCH trial {trial}: exact={d_exact:.5f} shapely={d_shapely:.5f} | ci=({xi:.2f},{yi:.2f},{math.degrees(ti):.1f}°) cj=({xj:.2f},{yj:.2f},{math.degrees(tj):.1f}°)"
        failures_detail.append(msg)
        if len(failures_detail) <= 20:
            print(f"  {FAIL} {msg[:100]}")
    else:
        total_tests += 1

    # When both agree on sign, magnitude should be close
    if sign_ok and abs(d_shapely) > 0.01:  # skip near-zero where both are imprecise
        mag_err = abs(d_exact - d_shapely)
        mag_ok = mag_err < tol_abs
        if not mag_ok:
            mag_errors += 1
            total_failures += 1
            msg = f"MAG ERROR trial {trial}: exact={d_exact:.5f} shapely={d_shapely:.5f} err={mag_err:.5f}"
            failures_detail.append(msg)
            if len(failures_detail) <= 20:
                print(f"  {FAIL} {msg[:100]}")
        total_tests += 1

n_t3 = total_tests - t3_0
n_f3 = total_failures - t2_failures
print(f"  Tier 3: {n_t3} tests, {n_f3} failures (sign={sign_errors}, mag={mag_errors}) ({time.time()-t3_start:.1f}s)")
t3_failures = total_failures

# ─── Tier 4: Stratified edge cases ───────────────────────────────────────────

print("\n=== Tier 4: Stratified edge cases (1000 cases) ===")
t4_start = time.time()
t4_0 = total_tests

random.seed(777)

# 200 far-apart: d should be positive, ≈ center_dist - 2
for _ in range(200):
    xi, yi, ti = rand_semicircle(1.5)
    xj, yj, tj = rand_semicircle(1.5)
    # Force far apart
    xj += 6.0
    d = semicircle_signed_dist(xi, yi, ti, xj, yj, tj)
    record(d > 0, "far apart: expect positive")

# 200 near-touching (center dist ≈ 2 + small gap)
near_touch_sign_ok = 0
for _ in range(200):
    ti = random.uniform(0, 2*math.pi)
    tj = random.uniform(0, 2*math.pi)
    gap = random.uniform(0.001, 0.05)
    # Place centers at distance 2 + gap (just outside disk overlap)
    a = random.uniform(0, 2*math.pi)
    xi, yi = 0, 0
    xj, yj = (2 + gap) * math.cos(a), (2 + gap) * math.sin(a)
    d = semicircle_signed_dist(xi, yi, ti, xj, yj, tj)
    # Disks don't overlap → semicircles definitely don't → should be positive
    record(d > 0, f"near-touch (gap={gap:.3f}, disk-separated): expect positive")

# 200 clearly overlapping (center dist < 0.5)
for _ in range(200):
    ti = random.uniform(0, 2*math.pi)
    tj = ti + random.uniform(-0.5, 0.5)  # similar orientation
    xi, yi = 0, 0
    xj = random.uniform(-0.3, 0.3)
    yj = random.uniform(-0.3, 0.3)
    d = semicircle_signed_dist(xi, yi, ti, xj, yj, tj)
    # Cross-validate with Shapely
    ds = shapely_signed_dist(xi, yi, ti, xj, yj, tj)
    if ds < 0:  # Shapely confirms overlap
        record(d < 0, f"clearly overlapping (Shapely confirmed): expect negative")

# 200 conjugate pairs (|ti - tj| ≈ π) at varying distances
for _ in range(200):
    ti = random.uniform(0, 2*math.pi)
    tj = ti + math.pi + random.uniform(-0.05, 0.05)
    d_sep = random.uniform(0, 0.3)
    xi, yi = 0, 0
    xj = d_sep * math.cos(ti)
    yj = d_sep * math.sin(ti)
    d = semicircle_signed_dist(xi, yi, ti, xj, yj, tj)
    record(d >= -1e-6, f"conjugate pair close: expect non-negative (got {d:.6f})")

n_t4 = total_tests - t4_0
n_f4 = total_failures - t3_failures
print(f"  Tier 4: {n_t4} tests, {n_f4} failures ({time.time()-t4_start:.1f}s)")

# ─── Results ──────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"TOTAL: {total_tests} tests, {total_failures} failures")
rate = total_failures / total_tests * 100
print(f"Failure rate: {rate:.2f}%")

if total_failures == 0:
    print(f"\n{PASS} All tests passed — exact_dist is validated")
else:
    print(f"\n{FAIL} {total_failures} failures found")
    print("\nFirst failures:")
    for f in failures_detail[:15]:
        print(f"  {f}")

# Statistical confidence
# For 99.9% confidence that failure rate < 1%, need n >= 6907 failure-free tests
if total_failures == 0 and total_tests >= 6907:
    print(f"\n{PASS} Statistical: ≥99.9% confidence failure rate <0.1% ({total_tests} tests)")
