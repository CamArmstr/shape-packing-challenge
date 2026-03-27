"""
Numba-JIT compiled SA for semicircle packing.
Target: 50k+ steps/sec (vs ~900 with Shapely).

15 unit semicircles (radius=1) into smallest enclosing circle.
Uses analytic phi-function overlap detection instead of Shapely polygons.
"""

import json
import math
import os
import time

import numba as nb
import numpy as np

N = 15
TWO_PI = 2.0 * math.pi


# ── Core phi-function (analytic overlap check) ──────────────────────────────

@nb.njit(cache=True)
def phi_pair_nb(xi, yi, ti, xj, yj, tj):
    """
    Phi-function for a pair of unit semicircles.
    phi >= 0 means no overlap; phi < 0 means overlap.
    """
    # Circle-circle separation (radii both 1, sum=2, sum²=4)
    cc = (xi - xj) ** 2 + (yi - yj) ** 2 - 4.0
    # Circle i vs flat side of j
    cp = -(math.cos(tj) * (xi - xj) + math.sin(tj) * (yi - yj)) - 1.0
    # Circle j vs flat side of i
    pc = -(math.cos(ti) * (xj - xi) + math.sin(ti) * (yj - yi)) - 1.0

    phi = cc
    if cp > phi:
        phi = cp
    if pc > phi:
        phi = pc

    # Half-plane vs half-plane: only valid when normals near antiparallel
    n_dot = math.cos(ti) * math.cos(tj) + math.sin(ti) * math.sin(tj)
    if n_dot <= -(1.0 - 1e-4):
        pp = math.cos(ti) * (xi - xj) + math.sin(ti) * (yi - yj)
        if pp > phi:
            phi = pp

    return phi


# ── Overlap energy (sum over all 105 pairs) ─────────────────────────────────

@nb.njit(cache=True)
def overlap_energy_nb(xs, ys, ts):
    """Sum of max(0, -phi)^2 over all pairs. Zero means no overlap."""
    n = xs.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            phi = phi_pair_nb(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if phi < 0.0:
                energy += phi * phi
    return energy


# ── Fast MEC proxy ───────────────────────────────────────────────────────────

@nb.njit(cache=True)
def r_fast_nb(xs, ys):
    """
    Fast MEC radius proxy: max over all semicircles of (dist_from_origin + 1).
    Slight overestimate but O(N) and good enough for SA guidance.
    """
    n = xs.shape[0]
    r_max = 0.0
    for i in range(n):
        d = math.sqrt(xs[i] * xs[i] + ys[i] * ys[i]) + 1.0
        if d > r_max:
            r_max = d
    return r_max


# ── Delta overlap energy (only recompute pairs involving idx) ────────────────

@nb.njit(cache=True)
def overlap_energy_for_idx(xs, ys, ts, idx):
    """Overlap energy contribution from semicircle idx against all others."""
    n = xs.shape[0]
    energy = 0.0
    for j in range(n):
        if j == idx:
            continue
        phi = phi_pair_nb(xs[idx], ys[idx], ts[idx], xs[j], ys[j], ts[j])
        if phi < 0.0:
            energy += phi * phi
    return energy


# ── Full SA loop ─────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def sa_run_numba(xs, ys, ts, n_steps, T_start, T_end, lam_start, lam_end, seed):
    """
    Simulated annealing loop for semicircle packing — pure Numba.

    Returns (best_xs, best_ys, best_ts, best_r, found_feasible).
    """
    np.random.seed(seed)
    n = xs.shape[0]

    # Work on copies
    cur_xs = xs.copy()
    cur_ys = ys.copy()
    cur_ts = ts.copy()

    # Best feasible tracking
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_ts = ts.copy()
    best_r = 1e18
    found_feasible = False

    # Current state
    cur_ovlp = overlap_energy_nb(cur_xs, cur_ys, cur_ts)
    cur_rf = r_fast_nb(cur_xs, cur_ys)
    lam = lam_start
    cur_obj = cur_rf + lam * cur_ovlp

    log_ratio_T = math.log(T_end / T_start)
    log_ratio_lam = math.log(lam_end / lam_start)

    for step in range(n_steps):
        frac = step / n_steps
        T = T_start * math.exp(log_ratio_T * frac)
        lam = lam_start * math.exp(log_ratio_lam * frac)

        scale = 0.25 * (1.0 - 0.85 * frac) + 0.002

        idx = np.random.randint(0, n)
        old_x = cur_xs[idx]
        old_y = cur_ys[idx]
        old_t = cur_ts[idx]

        # Old overlap contribution for this index
        old_idx_ovlp = overlap_energy_for_idx(cur_xs, cur_ys, cur_ts, idx)

        # Propose move
        m = np.random.random()
        if m < 0.30:
            # Translate
            cur_xs[idx] += np.random.randn() * scale
            cur_ys[idx] += np.random.randn() * scale
        elif m < 0.50:
            # Rotate
            cur_ts[idx] += np.random.randn() * scale * 3.0
            cur_ts[idx] = cur_ts[idx] % TWO_PI
        elif m < 0.80:
            # Squeeze toward centroid + small rotate
            cx = 0.0
            cy = 0.0
            for k in range(n):
                cx += cur_xs[k]
                cy += cur_ys[k]
            cx /= n
            cy /= n
            dx = cx - cur_xs[idx]
            dy = cy - cur_ys[idx]
            d = math.sqrt(dx * dx + dy * dy)
            if d > 0.01:
                cur_xs[idx] += scale * 0.4 * dx / d
                cur_ys[idx] += scale * 0.4 * dy / d
            cur_ts[idx] += np.random.randn() * scale
            cur_ts[idx] = cur_ts[idx] % TWO_PI
        else:
            # Translate + rotate
            cur_xs[idx] += np.random.randn() * scale * 0.5
            cur_ys[idx] += np.random.randn() * scale * 0.5
            cur_ts[idx] += np.random.randn() * scale * 2.0
            cur_ts[idx] = cur_ts[idx] % TWO_PI

        # New overlap contribution
        new_idx_ovlp = overlap_energy_for_idx(cur_xs, cur_ys, cur_ts, idx)
        new_ovlp = cur_ovlp - old_idx_ovlp + new_idx_ovlp

        new_rf = r_fast_nb(cur_xs, cur_ys)
        new_obj = new_rf + lam * new_ovlp

        delta = new_obj - cur_obj

        if delta < 0.0 or np.random.random() < math.exp(-delta / max(T, 1e-15)):
            # Accept
            cur_ovlp = new_ovlp
            cur_rf = new_rf
            cur_obj = new_obj

            # Track best feasible
            if new_ovlp < 1e-5 and new_rf < best_r:
                best_r = new_rf
                for k in range(n):
                    best_xs[k] = cur_xs[k]
                    best_ys[k] = cur_ys[k]
                    best_ts[k] = cur_ts[k]
                found_feasible = True
        else:
            # Reject — revert
            cur_xs[idx] = old_x
            cur_ys[idx] = old_y
            cur_ts[idx] = old_t

    return best_xs, best_ys, best_ts, best_r, found_feasible


# ── Python wrapper ───────────────────────────────────────────────────────────

def sa_run_numba_wrapper(xs, ys, ts, n_steps=2000000, T_start=0.25, T_end=0.0005,
                         lam_start=500.0, lam_end=5000.0, seed=42):
    """
    Python wrapper around the Numba SA loop.
    Returns (best_xs, best_ys, best_ts, best_r) or (None, None, None, inf).
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)

    bx, by, bt, br, found = sa_run_numba(
        xs, ys, ts, n_steps, T_start, T_end, lam_start, lam_end, seed
    )

    if found:
        return bx, by, bt, br
    else:
        return None, None, None, float('inf')


# ── Benchmark ────────────────────────────────────────────────────────────────

def benchmark():
    """Time 100k steps and print steps/sec."""
    # Warm up JIT
    xs = np.random.randn(N) * 2.0
    ys = np.random.randn(N) * 2.0
    ts = np.random.rand(N) * TWO_PI
    sa_run_numba(xs, ys, ts, 100, 0.25, 0.0005, 500.0, 5000.0, 1)

    # Benchmark
    n_bench = 100_000
    t0 = time.perf_counter()
    sa_run_numba(xs.copy(), ys.copy(), ts.copy(), n_bench, 0.25, 0.0005, 500.0, 5000.0, 42)
    elapsed = time.perf_counter() - t0
    rate = n_bench / elapsed
    print(f"Benchmark: {n_bench} steps in {elapsed:.2f}s = {rate:,.0f} steps/sec")
    return rate


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("sa_numba.py — Numba-JIT SA for semicircle packing")
    print("=" * 50)

    # 1. Benchmark
    print("\n--- Benchmark (100k steps) ---")
    rate = benchmark()

    # 2. Load best_solution.json and run 200k steps
    sol_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_solution.json")
    if os.path.exists(sol_path):
        print(f"\n--- Test run (200k steps from best_solution.json) ---")
        with open(sol_path) as f:
            sol = json.load(f)
        xs = np.array([s["x"] for s in sol], dtype=np.float64)
        ys = np.array([s["y"] for s in sol], dtype=np.float64)
        ts = np.array([s["theta"] for s in sol], dtype=np.float64)

        t0 = time.perf_counter()
        bx, by, bt, br = sa_run_numba_wrapper(
            xs, ys, ts,
            n_steps=200_000,
            T_start=0.25,
            T_end=0.0005,
            lam_start=500.0,
            lam_end=5000.0,
            seed=42,
        )
        elapsed = time.perf_counter() - t0

        if bx is not None:
            print(f"Result R = {br:.6f} ({elapsed:.2f}s)")
            # 3. Sanity check
            assert isinstance(br, float), "best_r should be float"
            assert br < 4.0, f"R={br} >= 4.0 — sanity check failed"
            print("Sanity checks passed (R is float and < 4.0)")
        else:
            print(f"No feasible solution found in 200k steps ({elapsed:.2f}s)")
    else:
        print(f"\nbest_solution.json not found at {sol_path}, skipping test run")

    print("\nDone.")
