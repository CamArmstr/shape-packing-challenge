"""Benchmark GJK for SA feasibility."""
import time, numpy as np, numba as nb
from gjk_numba import semicircle_gjk_signed_dist

@nb.njit
def overlap_energy_gjk(xs, ys, ts):
    n = xs.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = semicircle_gjk_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < 0.0:
                energy += d * d
    return energy

@nb.njit
def overlap_single_gjk(xs, ys, ts, idx):
    n = xs.shape[0]
    energy = 0.0
    for j in range(n):
        if j == idx:
            continue
        d = semicircle_gjk_signed_dist(xs[idx], ys[idx], ts[idx], xs[j], ys[j], ts[j])
        if d < 0.0:
            energy += d * d
    return energy

@nb.njit
def min_sep_gjk(xs, ys, ts):
    n = xs.shape[0]
    mn = 1e18
    for i in range(n):
        for j in range(i + 1, n):
            d = semicircle_gjk_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < mn:
                mn = d
    return mn

xs = np.random.uniform(-2, 2, 15)
ys = np.random.uniform(-2, 2, 15)
ts = np.random.uniform(0, 6.28, 15)

# Warmup
overlap_energy_gjk(xs, ys, ts)
overlap_single_gjk(xs, ys, ts, 0)
min_sep_gjk(xs, ys, ts)

# Benchmark full
t0 = time.perf_counter()
N = 100000
for _ in range(N):
    overlap_energy_gjk(xs, ys, ts)
t1 = time.perf_counter()
print(f'Full overlap check (GJK): {N/(t1-t0):.0f}/sec')

# Benchmark single
t0 = time.perf_counter()
for _ in range(N):
    overlap_single_gjk(xs, ys, ts, 7)
t1 = time.perf_counter()
print(f'Single-semicircle check (GJK): {N/(t1-t0):.0f}/sec')
