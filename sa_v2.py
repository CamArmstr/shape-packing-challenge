"""
sa_v2.py — Numba-JIT SA with Thompson Sampling operator selection,
swap moves, adaptive step size, and stagnation escape.

Operators:
  0: translate (single particle)
  1: rotate (single particle orientation)
  2: swap (exchange two particles' full state)
  3: squeeze (translate toward centroid + rotate)
  4: cluster (rigid body 2-4 neighbors)

Thompson Sampling: Beta(alpha, beta) posterior per operator.
  - Improvement → alpha += 1
  - No improvement → beta += 1
  - Decay all by 0.95 every DECAY_INTERVAL steps

Adaptive step size: target 35-45% acceptance, adjust every 200 steps.
Stagnation escape: after 500 non-improving steps, force 3 cluster moves at 3x scale.
"""

import math
import os
import time
import json

import numba as nb
import numpy as np

N = 15
TWO_PI = 2.0 * math.pi
N_OPS = 5  # number of operators
DECAY_INTERVAL = 5000
ADAPT_INTERVAL = 200
STAGNATION_THRESHOLD = 500


# ── Core phi-function (same as sa_numba.py) ──────────────────────────────────

@nb.njit(cache=True)
def phi_pair_nb(xi, yi, ti, xj, yj, tj):
    cc = (xi - xj) ** 2 + (yi - yj) ** 2 - 4.0
    cp = -(math.cos(tj) * (xi - xj) + math.sin(tj) * (yi - yj)) - 1.0
    pc = -(math.cos(ti) * (xj - xi) + math.sin(ti) * (yj - yi)) - 1.0
    phi = cc
    if cp > phi:
        phi = cp
    if pc > phi:
        phi = pc
    n_dot = math.cos(ti) * math.cos(tj) + math.sin(ti) * math.sin(tj)
    if n_dot <= -(1.0 - 1e-4):
        pp = math.cos(ti) * (xi - xj) + math.sin(ti) * (yi - yj)
        if pp > phi:
            phi = pp
    return phi


@nb.njit(cache=True)
def overlap_energy_nb(xs, ys, ts):
    n = xs.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            phi = phi_pair_nb(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if phi < 0.0:
                energy += phi * phi
    return energy


@nb.njit(cache=True)
def r_single(x, y, t):
    """Exact origin-centered enclosing radius contribution from one semicircle."""
    ux = math.cos(t)
    uy = math.sin(t)
    a_i = x * ux + y * uy
    if a_i >= 0.0:
        return math.sqrt(x * x + y * y) + 1.0
    else:
        tx = -uy
        ty = ux
        b_i = x * tx + y * ty
        return math.sqrt(x * x + y * y + 1.0 + 2.0 * abs(b_i))


@nb.njit(cache=True)
def r_exact_nb(xs, ys, ts):
    """Exact origin-centered enclosing radius for all semicircles."""
    n = xs.shape[0]
    r_max = 0.0
    for i in range(n):
        ri = r_single(xs[i], ys[i], ts[i])
        if ri > r_max:
            r_max = ri
    return r_max


@nb.njit(cache=True)
def r_update_nb(xs, ys, ts, old_r, changed_idx):
    """
    Incremental R update: only recompute R for the changed particle.
    If the changed particle's new R_i >= old_r, return the new max.
    If old_r was set by the changed particle and new R_i < old_r,
    we need a full recompute (returns -1 as signal).
    Otherwise return old_r.
    """
    new_ri = r_single(xs[changed_idx], ys[changed_idx], ts[changed_idx])
    if new_ri >= old_r:
        return new_ri  # new particle sets the max
    # We don't know if old_r was set by this particle — full recompute needed
    # This is still faster on average because most moves don't change the max
    return r_exact_nb(xs, ys, ts)


@nb.njit(cache=True)
def overlap_energy_for_idx(xs, ys, ts, idx):
    n = xs.shape[0]
    energy = 0.0
    for j in range(n):
        if j == idx:
            continue
        phi = phi_pair_nb(xs[idx], ys[idx], ts[idx], xs[j], ys[j], ts[j])
        if phi < 0.0:
            energy += phi * phi
    return energy


@nb.njit(cache=True)
def min_clearance_for_idx(xs, ys, ts, idx):
    """Minimum phi (clearance) from semicircle idx to all others.
    Positive = separated, negative = overlapping."""
    n = xs.shape[0]
    min_phi = 1e18
    for j in range(n):
        if j == idx:
            continue
        phi = phi_pair_nb(xs[idx], ys[idx], ts[idx], xs[j], ys[j], ts[j])
        if phi < min_phi:
            min_phi = phi
    return min_phi


@nb.njit(cache=True)
def overlap_energy_for_pair(xs, ys, ts, idx1, idx2):
    """Overlap energy involving idx1 or idx2 (avoiding double-count of their mutual pair)."""
    n = xs.shape[0]
    energy = 0.0
    for j in range(n):
        if j == idx1 or j == idx2:
            continue
        phi1 = phi_pair_nb(xs[idx1], ys[idx1], ts[idx1], xs[j], ys[j], ts[j])
        if phi1 < 0.0:
            energy += phi1 * phi1
        phi2 = phi_pair_nb(xs[idx2], ys[idx2], ts[idx2], xs[j], ys[j], ts[j])
        if phi2 < 0.0:
            energy += phi2 * phi2
    # Add the mutual pair
    phi_m = phi_pair_nb(xs[idx1], ys[idx1], ts[idx1], xs[idx2], ys[idx2], ts[idx2])
    if phi_m < 0.0:
        energy += phi_m * phi_m
    return energy


# ── Thompson Sampling helper ─────────────────────────────────────────────────

@nb.njit(cache=True)
def sample_beta(alpha, beta):
    """Sample from Beta(alpha, beta) using the Gamma trick.
    Beta(a,b) = Ga/(Ga+Gb) where Ga~Gamma(a,1), Gb~Gamma(b,1)."""
    # np.random.gamma not available in numba, use standard_gamma
    ga = 0.0
    gb = 0.0
    # Numba supports np.random.gamma(shape) as of recent versions
    # Fallback: use the fact that Gamma(1,1) = Exponential(1) for alpha/beta near 1
    # For general alpha, use np.random.standard_gamma
    if alpha <= 0.01:
        alpha = 0.01
    if beta <= 0.01:
        beta = 0.01
    ga = np.random.standard_gamma(alpha)
    gb = np.random.standard_gamma(beta)
    if ga + gb < 1e-15:
        return 0.5
    return ga / (ga + gb)


@nb.njit(cache=True)
def select_operator(alphas, betas):
    """Thompson Sampling with epsilon-greedy exploration floor.
    10% of the time, pick a random operator (ensures all get explored).
    90% of the time, use Thompson Sampling."""
    if np.random.random() < 0.10:
        return np.random.randint(0, N_OPS)
    best_op = 0
    best_val = -1.0
    for i in range(N_OPS):
        val = sample_beta(alphas[i], betas[i])
        if val > best_val:
            best_val = val
            best_op = i
    return best_op


# ── Main SA loop ─────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def sa_run_v2(xs, ys, ts, n_steps, T_start, T_end, lam_start, lam_end, seed):
    """
    SA with Thompson Sampling operator selection, swap moves,
    adaptive step size, and stagnation escape.

    Returns (best_xs, best_ys, best_ts, best_r, found_feasible,
             final_alphas, final_betas).
    """
    np.random.seed(seed)
    n = xs.shape[0]

    cur_xs = xs.copy()
    cur_ys = ys.copy()
    cur_ts = ts.copy()

    best_xs = xs.copy()
    best_ys = ys.copy()
    best_ts = ts.copy()
    best_r = 1e18
    found_feasible = False

    cur_ovlp = overlap_energy_nb(cur_xs, cur_ys, cur_ts)
    cur_rf = r_exact_nb(cur_xs, cur_ys, cur_ts)
    lam = lam_start
    cur_obj = cur_rf + lam * cur_ovlp

    log_ratio_T = math.log(T_end / T_start)
    log_ratio_lam = math.log(lam_end / lam_start)

    # Thompson Sampling state: Beta(alpha, beta) per operator
    alphas = np.ones(N_OPS)  # prior: Beta(1,1) = uniform
    betas = np.ones(N_OPS)

    # Adaptive step size
    scale = 0.25
    accept_count = 0
    total_count = 0

    # Stagnation tracking
    steps_since_improve = 0

    for step in range(n_steps):
        frac = step / n_steps
        T = T_start * math.exp(log_ratio_T * frac)
        lam = lam_start * math.exp(log_ratio_lam * frac)

        # Stagnation escape: force cluster moves at 3x scale
        force_cluster = False
        if steps_since_improve >= STAGNATION_THRESHOLD:
            force_cluster = True

        # Select operator
        if force_cluster:
            op = 4  # cluster
            eff_scale = scale * 3.0
            steps_since_improve = 0  # reset after forcing
        else:
            op = select_operator(alphas, betas)
            eff_scale = scale

        old_obj = cur_obj
        accepted = False

        if op == 0:
            # ── Translate: single particle position ──
            idx = np.random.randint(0, n)
            old_x = cur_xs[idx]
            old_y = cur_ys[idx]
            old_ovlp_idx = overlap_energy_for_idx(cur_xs, cur_ys, cur_ts, idx)

            cur_xs[idx] += np.random.randn() * eff_scale
            cur_ys[idx] += np.random.randn() * eff_scale

            new_ovlp_idx = overlap_energy_for_idx(cur_xs, cur_ys, cur_ts, idx)
            new_ovlp = cur_ovlp - old_ovlp_idx + new_ovlp_idx
            new_rf = r_update_nb(cur_xs, cur_ys, cur_ts, cur_rf, idx)
            new_obj = new_rf + lam * new_ovlp
            delta = new_obj - cur_obj

            if delta < 0.0 or np.random.random() < math.exp(-delta / max(T, 1e-15)):
                cur_ovlp = new_ovlp
                cur_rf = new_rf
                cur_obj = new_obj
                accepted = True
            else:
                cur_xs[idx] = old_x
                cur_ys[idx] = old_y

        elif op == 1:
            # ── Rotate: single particle orientation ──
            idx = np.random.randint(0, n)
            old_t = cur_ts[idx]
            old_ovlp_idx = overlap_energy_for_idx(cur_xs, cur_ys, cur_ts, idx)

            cur_ts[idx] = (cur_ts[idx] + np.random.randn() * eff_scale * 3.0) % TWO_PI

            new_ovlp_idx = overlap_energy_for_idx(cur_xs, cur_ys, cur_ts, idx)
            new_ovlp = cur_ovlp - old_ovlp_idx + new_ovlp_idx
            new_rf = r_update_nb(cur_xs, cur_ys, cur_ts, cur_rf, idx)
            new_obj = new_rf + lam * new_ovlp
            delta = new_obj - cur_obj

            if delta < 0.0 or np.random.random() < math.exp(-delta / max(T, 1e-15)):
                cur_ovlp = new_ovlp
                cur_rf = new_rf
                cur_obj = new_obj
                accepted = True
            else:
                cur_ts[idx] = old_t

        elif op == 2:
            # ── Swap: exchange full state of two particles ──
            # Pick two particles that are actually in different shells (different radii)
            # to avoid trivial identity swaps
            idx1 = np.random.randint(0, n)
            idx2 = np.random.randint(0, n)
            for _sw in range(5):  # try a few times to find a non-trivial swap
                idx2 = np.random.randint(0, n)
                if idx2 != idx1:
                    r1 = math.sqrt(cur_xs[idx1]**2 + cur_ys[idx1]**2)
                    r2 = math.sqrt(cur_xs[idx2]**2 + cur_ys[idx2]**2)
                    if abs(r1 - r2) > 0.3:  # different shells
                        break
            if idx2 == idx1:
                idx2 = (idx1 + 1) % n

            old_ovlp_pair = overlap_energy_for_pair(cur_xs, cur_ys, cur_ts, idx1, idx2)

            # Position-only swap: exchange (x,y) but KEEP original orientations.
            # For identical semicircles, swapping full (x,y,θ) is a no-op (just relabeling).
            # Position-only swap moves θ₁ to position (x₂,y₂) — a genuine topological change.
            cur_xs[idx1], cur_xs[idx2] = cur_xs[idx2], cur_xs[idx1]
            cur_ys[idx1], cur_ys[idx2] = cur_ys[idx2], cur_ys[idx1]
            # θ stays: cur_ts[idx1] keeps its old value at the new position

            new_ovlp_pair = overlap_energy_for_pair(cur_xs, cur_ys, cur_ts, idx1, idx2)
            new_ovlp = cur_ovlp - old_ovlp_pair + new_ovlp_pair
            new_rf = r_exact_nb(cur_xs, cur_ys, cur_ts)
            new_obj = new_rf + lam * new_ovlp
            delta = new_obj - cur_obj

            if delta < 0.0 or np.random.random() < math.exp(-delta / max(T, 1e-15)):
                cur_ovlp = new_ovlp
                cur_rf = new_rf
                cur_obj = new_obj
                accepted = True
            else:
                # Swap positions back
                cur_xs[idx1], cur_xs[idx2] = cur_xs[idx2], cur_xs[idx1]
                cur_ys[idx1], cur_ys[idx2] = cur_ys[idx2], cur_ys[idx1]

        elif op == 3:
            # ── Squeeze: translate toward centroid + rotate ──
            idx = np.random.randint(0, n)
            old_x = cur_xs[idx]
            old_y = cur_ys[idx]
            old_t = cur_ts[idx]
            old_ovlp_idx = overlap_energy_for_idx(cur_xs, cur_ys, cur_ts, idx)

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
                cur_xs[idx] += eff_scale * 0.4 * dx / d
                cur_ys[idx] += eff_scale * 0.4 * dy / d
            cur_ts[idx] = (cur_ts[idx] + np.random.randn() * eff_scale) % TWO_PI

            new_ovlp_idx = overlap_energy_for_idx(cur_xs, cur_ys, cur_ts, idx)
            new_ovlp = cur_ovlp - old_ovlp_idx + new_ovlp_idx
            new_rf = r_update_nb(cur_xs, cur_ys, cur_ts, cur_rf, idx)
            new_obj = new_rf + lam * new_ovlp
            delta = new_obj - cur_obj

            if delta < 0.0 or np.random.random() < math.exp(-delta / max(T, 1e-15)):
                cur_ovlp = new_ovlp
                cur_rf = new_rf
                cur_obj = new_obj
                accepted = True
            else:
                cur_xs[idx] = old_x
                cur_ys[idx] = old_y
                cur_ts[idx] = old_t

        elif op == 4:
            # ── Cluster move: rigid body translate+rotate of 2-4 nearest neighbors ──
            anchor = np.random.randint(0, n)
            cluster_size = 2 + np.random.randint(0, 3)  # 2, 3, or 4

            dists_sq = np.empty(n)
            for k in range(n):
                dists_sq[k] = (cur_xs[k] - cur_xs[anchor])**2 + (cur_ys[k] - cur_ys[anchor])**2
            dists_sq[anchor] = 1e18

            cluster = np.empty(cluster_size + 1, dtype=nb.int64)
            cluster[0] = anchor
            taken = np.zeros(n, dtype=nb.boolean)
            taken[anchor] = True
            actual_extra = 0
            for ci in range(cluster_size):
                best_k = -1
                best_d = 1e18
                for k in range(n):
                    if not taken[k] and dists_sq[k] < best_d:
                        best_d = dists_sq[k]
                        best_k = k
                if best_k >= 0:
                    cluster[ci + 1] = best_k
                    taken[best_k] = True
                    actual_extra += 1
                else:
                    break
            actual_size = actual_extra + 1

            # Save old state
            old_xs_c = np.empty(actual_size)
            old_ys_c = np.empty(actual_size)
            old_ts_c = np.empty(actual_size)
            for ci in range(actual_size):
                k = cluster[ci]
                old_xs_c[ci] = cur_xs[k]
                old_ys_c[ci] = cur_ys[k]
                old_ts_c[ci] = cur_ts[k]

            # Old overlap for cluster members
            old_cluster_ovlp = 0.0
            for ci in range(actual_size):
                old_cluster_ovlp += overlap_energy_for_idx(cur_xs, cur_ys, cur_ts, cluster[ci])
            for ci in range(actual_size):
                for cj in range(ci + 1, actual_size):
                    ii = cluster[ci]
                    jj = cluster[cj]
                    p = phi_pair_nb(cur_xs[ii], cur_ys[ii], cur_ts[ii],
                                    cur_xs[jj], cur_ys[jj], cur_ts[jj])
                    v = -p if p < 0.0 else 0.0
                    old_cluster_ovlp -= v * v

            # Rigid body move
            dxm = np.random.randn() * eff_scale * 0.7
            dym = np.random.randn() * eff_scale * 0.7
            dtheta = np.random.randn() * eff_scale * 1.5

            # Centroid
            ccx = 0.0
            ccy = 0.0
            for ci in range(actual_size):
                ccx += cur_xs[cluster[ci]]
                ccy += cur_ys[cluster[ci]]
            ccx /= actual_size
            ccy /= actual_size

            cos_dt = math.cos(dtheta)
            sin_dt = math.sin(dtheta)
            for ci in range(actual_size):
                k = cluster[ci]
                rx = cur_xs[k] - ccx
                ry = cur_ys[k] - ccy
                cur_xs[k] = ccx + cos_dt * rx - sin_dt * ry + dxm
                cur_ys[k] = ccy + sin_dt * rx + cos_dt * ry + dym
                cur_ts[k] = (cur_ts[k] + dtheta) % TWO_PI

            # New overlap
            new_cluster_ovlp = 0.0
            for ci in range(actual_size):
                new_cluster_ovlp += overlap_energy_for_idx(cur_xs, cur_ys, cur_ts, cluster[ci])
            for ci in range(actual_size):
                for cj in range(ci + 1, actual_size):
                    ii = cluster[ci]
                    jj = cluster[cj]
                    p = phi_pair_nb(cur_xs[ii], cur_ys[ii], cur_ts[ii],
                                    cur_xs[jj], cur_ys[jj], cur_ts[jj])
                    v = -p if p < 0.0 else 0.0
                    new_cluster_ovlp -= v * v

            new_ovlp = cur_ovlp - old_cluster_ovlp + new_cluster_ovlp
            new_rf = r_exact_nb(cur_xs, cur_ys, cur_ts)
            new_obj = new_rf + lam * new_ovlp
            delta = new_obj - cur_obj

            if delta < 0.0 or np.random.random() < math.exp(-delta / max(T, 1e-15)):
                cur_ovlp = new_ovlp
                cur_rf = new_rf
                cur_obj = new_obj
                accepted = True
            else:
                for ci in range(actual_size):
                    k = cluster[ci]
                    cur_xs[k] = old_xs_c[ci]
                    cur_ys[k] = old_ys_c[ci]
                    cur_ts[k] = old_ts_c[ci]

        # ── Update Thompson Sampling ──
        # Credit = actual objective improvement (not just acceptance)
        # Use extreme-value credit: reward proportional to improvement magnitude
        # Threshold 1e-6 avoids float noise from swaps of similar-position particles
        improvement = old_obj - cur_obj  # positive if improved
        if improvement > 1e-6:
            # Scale credit by improvement magnitude (extreme-value style)
            credit = min(1.0 + improvement * 100.0, 5.0)
            alphas[op] += credit
            steps_since_improve = 0
        else:
            betas[op] += 1.0
            if not accepted:
                steps_since_improve += 1
            # Accepted but no improvement: still counts as stagnation
            elif improvement < 1e-10:
                steps_since_improve += 1
            else:
                steps_since_improve = 0

        # ── Decay posteriors periodically ──
        if (step + 1) % DECAY_INTERVAL == 0:
            for i in range(N_OPS):
                alphas[i] *= 0.95
                betas[i] *= 0.95
                if alphas[i] < 0.5:
                    alphas[i] = 0.5
                if betas[i] < 0.5:
                    betas[i] = 0.5

        # ── Adaptive step size ──
        total_count += 1
        if accepted:
            accept_count += 1
        if total_count % ADAPT_INTERVAL == 0:
            rate = accept_count / ADAPT_INTERVAL
            if rate > 0.45:
                scale *= 1.1
            elif rate < 0.35:
                scale *= 0.9
            # Clamp scale
            if scale < 0.005:
                scale = 0.005
            if scale > 1.0:
                scale = 1.0
            accept_count = 0

        # ── Track best feasible ──
        if cur_ovlp < 1e-5 and cur_rf < best_r:
            best_r = cur_rf
            for k in range(n):
                best_xs[k] = cur_xs[k]
                best_ys[k] = cur_ys[k]
                best_ts[k] = cur_ts[k]
            found_feasible = True

    return best_xs, best_ys, best_ts, best_r, found_feasible, alphas, betas


# ── Python wrapper with Shapely gating ───────────────────────────────────────

def _gjk_validate(xs, ys, ts):
    """Fast GJK-based feasibility check before expensive Shapely validation.
    Returns (valid, R) or (False, inf)."""
    try:
        from gjk_numba import overlap_energy_gjk
        energy = overlap_energy_gjk(xs, ys, ts)
        if energy > 1e-8:
            return False, float('inf')  # GJK says overlapping, skip Shapely
        # GJK says feasible — now confirm with Shapely (authoritative)
        return _shapely_validate_full(xs, ys, ts)
    except Exception:
        return _shapely_validate_full(xs, ys, ts)


def _shapely_validate_full(xs, ys, ts):
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from src.semicircle_packing.geometry import Semicircle
        from src.semicircle_packing.scoring import validate_and_score
        scs = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(len(xs))]
        r = validate_and_score(scs)
        return r.valid, (r.score if r.valid else float('inf'))
    except Exception:
        return False, float('inf')


def sa_run_v2_wrapper(xs, ys, ts, n_steps=50_000_000, T_start=0.25, T_end=0.0005,
                      lam_start=500.0, lam_end=5000.0, seed=42,
                      shapely_check_interval=1_000_000):
    """
    Chunked SA runner with Shapely feasibility gating.
    Same structure as sa_numba.py wrapper but using the v2 kernel.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)

    n_chunks = max(1, n_steps // shapely_check_interval)
    chunk_size = n_steps // n_chunks

    best_xs, best_ys, best_ts = xs.copy(), ys.copy(), ts.copy()
    best_r = float('inf')

    cur_xs, cur_ys, cur_ts = xs.copy(), ys.copy(), ts.copy()
    phi_best_r = float('inf')
    phi_best_xs = xs.copy()
    phi_best_ys = ys.copy()
    phi_best_ts = ts.copy()
    last_shapely_checked_r = float('inf')
    rng = np.random.default_rng(seed + 9999)

    for chunk_idx in range(n_chunks):
        frac_start = chunk_idx / n_chunks
        frac_end = (chunk_idx + 1) / n_chunks

        T_s = T_start * (T_end / T_start) ** frac_start
        T_e = T_start * (T_end / T_start) ** frac_end
        lam_s = lam_start * (lam_end / lam_start) ** frac_start
        lam_e = lam_start * (lam_end / lam_start) ** frac_end

        chunk_seed = seed * 1000 + chunk_idx

        bx, by, bt, br, found, alphas, betas = sa_run_v2(
            cur_xs, cur_ys, cur_ts,
            chunk_size, T_s, T_e, lam_s, lam_e, chunk_seed
        )

        if found and br < phi_best_r:
            phi_best_r = br
            phi_best_xs, phi_best_ys, phi_best_ts = bx.copy(), by.copy(), bt.copy()

        if found and phi_best_r < last_shapely_checked_r - 1e-6:
            last_shapely_checked_r = phi_best_r
            valid, shapely_r = _gjk_validate(phi_best_xs, phi_best_ys, phi_best_ts)
            if valid and shapely_r < best_r:
                best_r = shapely_r
                best_xs = phi_best_xs.copy()
                best_ys = phi_best_ys.copy()
                best_ts = phi_best_ts.copy()
                cur_xs, cur_ys, cur_ts = phi_best_xs.copy(), phi_best_ys.copy(), phi_best_ts.copy()
            elif not valid:
                cur_xs = phi_best_xs + rng.standard_normal(len(phi_best_xs)) * 0.15
                cur_ys = phi_best_ys + rng.standard_normal(len(phi_best_ys)) * 0.15
                cur_ts = phi_best_ts + rng.standard_normal(len(phi_best_ts)) * 0.3
                phi_best_r = float('inf')
                last_shapely_checked_r = float('inf')
            else:
                cur_xs, cur_ys, cur_ts = phi_best_xs.copy(), phi_best_ys.copy(), phi_best_ts.copy()
        elif found:
            cur_xs, cur_ys, cur_ts = bx.copy(), by.copy(), bt.copy()
        else:
            cur_xs, cur_ys, cur_ts = bx.copy(), by.copy(), bt.copy()

    if best_r < float('inf'):
        return best_xs, best_ys, best_ts, best_r
    else:
        return None, None, None, float('inf')


# ── Benchmark ────────────────────────────────────────────────────────────────

def benchmark():
    xs = np.random.randn(N) * 2.0
    ys = np.random.randn(N) * 2.0
    ts = np.random.rand(N) * TWO_PI
    # Warm up
    sa_run_v2(xs.copy(), ys.copy(), ts.copy(), 100, 0.25, 0.0005, 500.0, 5000.0, 1)

    n_bench = 100_000
    t0 = time.perf_counter()
    sa_run_v2(xs.copy(), ys.copy(), ts.copy(), n_bench, 0.25, 0.0005, 500.0, 5000.0, 42)
    elapsed = time.perf_counter() - t0
    rate = n_bench / elapsed
    print(f"sa_v2 benchmark: {n_bench} steps in {elapsed:.2f}s = {rate:,.0f} steps/sec")
    return rate


if __name__ == "__main__":
    print("sa_v2.py — benchmarking...")
    benchmark()

    sol_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_solution.json")
    if os.path.exists(sol_path):
        print("\nTest run (500k steps from best_solution.json)...")
        with open(sol_path) as f:
            sol = json.load(f)
        xs = np.array([s["x"] for s in sol], dtype=np.float64)
        ys = np.array([s["y"] for s in sol], dtype=np.float64)
        ts = np.array([s["theta"] for s in sol], dtype=np.float64)

        t0 = time.perf_counter()
        bx, by, bt, br, found, alphas, betas = sa_run_v2(
            xs, ys, ts, 500_000, 0.25, 0.0005, 500.0, 5000.0, 42
        )
        elapsed = time.perf_counter() - t0

        if found:
            print(f"Result R_fast = {br:.6f} ({elapsed:.2f}s)")
        else:
            print(f"No feasible found in 500k steps ({elapsed:.2f}s)")

        # Report Thompson Sampling state
        op_names = ['translate', 'rotate', 'swap', 'squeeze', 'cluster']
        print("\nThompson Sampling final state:")
        for i in range(N_OPS):
            total = alphas[i] + betas[i]
            rate = alphas[i] / total if total > 0 else 0
            print(f"  {op_names[i]:12s}: α={alphas[i]:.1f} β={betas[i]:.1f} success_rate={rate:.3f}")
