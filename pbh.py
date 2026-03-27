"""
pbh.py — Population Basin Hopping for semicircle packing.

Based on Grosso, Locatelli, Schoen (2010) "A population-based approach
for hard global optimization problems based on dissimilarity measures".

Key features vs the current MBH:
1. Population of K diverse local minima (not just 1 best)
2. Hungarian assignment distance for permutation-invariant fingerprinting
3. d_cut dissimilarity threshold — only diverse solutions enter population
4. Tracks restart diversity: how many distinct basins found
5. CEGP-inspired destroy+repair: remove worst-placed pieces first

Usage:
    python pbh.py [--rounds 200] [--pop 15] [--d-cut 0.3]
"""

import json, math, time, random, os, sys, argparse
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from phi import penalty_energy_flat, penalty_gradient_flat
from exact_dist import penalty_energy_exact
from overnight import official_validate
from src.semicircle_packing.geometry import Semicircle
from src.semicircle_packing.scoring import validate_and_score
from mbh import (pack, unpack, score_R_fast, N, lbfgs_refine,
                 apply_perturbation, BEST_FILE, PERTURBATIONS)

LOG_FILE = os.path.join(os.path.dirname(__file__), 'pbh_log.txt')


# ─── Hungarian assignment distance ────────────────────────────────────────────

def hungarian_distance(xs1, ys1, xs2, ys2):
    """
    Optimal-assignment distance between two packing configurations.
    Handles the S_N permutation ambiguity of identical pieces.
    Uses scipy linear_sum_assignment (O(N^3), negligible for N=15).
    Returns sqrt of minimum sum of squared center distances.
    """
    cost = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cost[i, j] = (xs1[i] - xs2[j])**2 + (ys1[i] - ys2[j])**2
    row_ind, col_ind = linear_sum_assignment(cost)
    return math.sqrt(cost[row_ind, col_ind].sum())


# ─── Approximate symmetry detection ──────────────────────────────────────────

def detect_symmetry(xs, ys, ts, k_candidates=(2, 3, 5, 7)):
    """
    Check for approximate C_k rotational symmetry.
    For each k, rotate configuration by 2π/k and compute Hungarian distance.
    Returns dict {k: distance} — small distance means approximate C_k symmetry.
    """
    results = {}
    for k in k_candidates:
        angle = 2 * math.pi / k
        xs_rot = xs * math.cos(angle) - ys * math.sin(angle)
        ys_rot = xs * math.sin(angle) + ys * math.cos(angle)
        ts_rot = ts + angle  # rotate orientation too
        dist = hungarian_distance(xs, ys, xs_rot, ys_rot)
        results[k] = dist
    return results


# ─── Configuration fingerprint ────────────────────────────────────────────────

def fingerprint(xs, ys, ts):
    """
    Composite fingerprint for basin identification:
    - Sorted pairwise center distances (105 values, permutation-invariant)
    - Sorted radial distances from container center (15 values)
    Returns a hashable tuple for dictionary keying.
    Rounded to 2 decimal places for near-duplicate detection.
    """
    # Pairwise distances
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            d = math.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            dists.append(round(d, 2))
    dists.sort()

    # Radial distances
    radii = sorted(round(math.sqrt(x**2 + y**2), 2) for x, y in zip(xs, ys))

    return tuple(dists + radii)


# ─── CEGP-inspired worst-piece selection ──────────────────────────────────────

def worst_pieces(xs, ys, ts, R_target, n=3):
    """
    Find the n pieces contributing most to infeasibility at R_target.
    Uses the containment violation + pairwise overlap energy per piece.
    Higher score = worse placed = should be removed first (CEGP-style).
    """
    from exact_dist import semicircle_signed_dist, containment_signed_dist
    scores = np.zeros(N)

    # Containment violation
    for i in range(N):
        c = containment_signed_dist(xs[i], ys[i], ts[i], R_target)
        if c < 0:
            scores[i] += c * c * 10  # weighted more heavily

    # Pairwise overlap
    for i in range(N):
        for j in range(i + 1, N):
            dc2 = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
            if dc2 >= 4.0:
                continue
            d = semicircle_signed_dist(xs[i], ys[i], ts[i], xs[j], ys[j], ts[j])
            if d < 0:
                scores[i] += d * d
                scores[j] += d * d

    return np.argsort(scores)[-n:][::-1]  # indices of n worst pieces


def cegp_repair(xs, ys, ts, R_target, n_remove=3):
    """
    CEGP-inspired: remove the n worst-placed pieces, then re-insert
    them at random positions and re-optimize.
    """
    xs, ys, ts = xs.copy(), ys.copy(), ts.copy()
    remove_idxs = worst_pieces(xs, ys, ts, R_target, n=n_remove)

    for i in remove_idxs:
        # Re-place at random position within container
        for _ in range(20):
            r = R_target * math.sqrt(random.random()) * 0.8
            phi = random.uniform(0, 2 * math.pi)
            xs[i] = r * math.cos(phi)
            ys[i] = r * math.sin(phi)
            ts[i] = random.uniform(0, 2 * math.pi)
            # Quick check: not too close to existing pieces
            ok = True
            for j in range(N):
                if j == i:
                    continue
                if (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 < 0.25:
                    ok = False
                    break
            if ok:
                break

    return xs, ys, ts


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    xs = np.array([s['x'] for s in raw])
    ys = np.array([s['y'] for s in raw])
    ts = np.array([s['theta'] for s in raw])
    return xs, ys, ts


def save_best(xs, ys, ts, R):
    sol = [{"x": round(float(xs[i]), 6), "y": round(float(ys[i]), 6),
            "theta": round(float(ts[i]), 6)} for i in range(N)]
    with open(BEST_FILE, 'w') as f:
        json.dump(sol, f, indent=2)


def official_score(xs, ys, ts):
    scs = [Semicircle(x=round(float(xs[i]),6), y=round(float(ys[i]),6),
                      theta=round(float(ts[i]),6)) for i in range(N)]
    return validate_and_score(scs)


def center_solution(xs, ys, ts):
    result = official_score(xs, ys, ts)
    if result.mec:
        cx, cy, _ = result.mec
        return xs - cx, ys - cy, ts
    return xs, ys, ts


# ─── Population management ────────────────────────────────────────────────────

class Population:
    """
    Maintains K diverse local minima.
    New solutions only admitted if Hungarian distance > d_cut from all existing.
    Tracks basin diversity statistics.
    """
    def __init__(self, d_cut=0.3, max_size=15):
        self.d_cut = d_cut
        self.max_size = max_size
        self.members = []   # list of (xs, ys, ts, R)
        self.fingerprints = {}   # fingerprint → count
        self.n_distinct_basins = 0
        self.n_total_found = 0

    def add(self, xs, ys, ts, R):
        """Try to add solution. Returns True if admitted."""
        self.n_total_found += 1
        fp = fingerprint(xs, ys, ts)

        # Track basin diversity
        if fp not in self.fingerprints:
            self.fingerprints[fp] = 0
            self.n_distinct_basins += 1
        self.fingerprints[fp] += 1

        # Check dissimilarity from existing population
        if self.members:
            min_dist = min(hungarian_distance(xs, ys, m[0], m[1])
                          for m in self.members)
            if min_dist < self.d_cut:
                return False  # too similar, reject

        if len(self.members) < self.max_size:
            self.members.append((xs, ys, ts, R))
            return True
        else:
            # Replace worst member if new solution improves it
            worst_idx = max(range(len(self.members)), key=lambda i: self.members[i][3])
            if R < self.members[worst_idx][3]:
                self.members[worst_idx] = (xs, ys, ts, R)
                return True
        return False

    def best_R(self):
        if not self.members:
            return float('inf')
        return min(m[3] for m in self.members)

    def sample(self):
        """Sample a random member, weighted toward better solutions."""
        Rs = np.array([m[3] for m in self.members])
        weights = 1.0 / (Rs - Rs.min() + 0.01)
        weights /= weights.sum()
        idx = np.random.choice(len(self.members), p=weights)
        return self.members[idx]

    def confidence_level(self):
        """
        Fraction of best-basin visits out of total (convergence indicator).
        If the best fingerprint appears ≥3 times, flag near-optimal confidence.
        """
        if not self.fingerprints:
            return 0.0
        best_count = max(self.fingerprints.values())
        return best_count / max(self.n_total_found, 1)


# ─── Main PBH loop ────────────────────────────────────────────────────────────

def run_pbh(rounds=200, pop_size=15, d_cut=0.3, r_start=None, r_step=0.005):
    log = open(LOG_FILE, 'w')
    def logprint(msg):
        print(msg, flush=True)
        log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        log.flush()

    xs0, ys0, ts0 = load_best()
    result0 = official_validate(xs0, ys0, ts0)
    global_best_R = float(result0.score)
    if r_start is None:
        r_start = global_best_R

    logprint(f"=== PBH | start R={global_best_R:.6f} | pop={pop_size} | d_cut={d_cut} ===")

    pop = Population(d_cut=d_cut, max_size=pop_size)

    # Seed population from best solution
    xs_, ys_, ts_, E = lbfgs_refine(xs0, ys0, ts0, r_start)
    if E < 1e-4:
        result = official_validate(xs_, ys_, ts_)
        if result.valid:
            pop.add(xs_, ys_, ts_, float(result.score))

    # Fill with diverse perturbations of best
    logprint(f"Seeding population...")
    seed_kinds = ['wall_slide', 'expand', 'jitter', 'shift_worst', 'swap',
                  'reflect', 'flip', 'lns', 'herringbone', 'wall_slide',
                  'expand', 'jitter', 'shift_worst', 'swap', 'reflect']
    for kind in seed_kinds:
        if len(pop.members) >= pop_size:
            break
        pxs, pys, pts = apply_perturbation(xs0, ys0, ts0, r_start, kind=kind)
        xs_, ys_, ts_, E = lbfgs_refine(pxs, pys, pts, r_start)
        if E < 1e-4:
            result = official_validate(xs_, ys_, ts_)
            if result.valid:
                admitted = pop.add(xs_, ys_, ts_, float(result.score))
                if admitted:
                    logprint(f"  Seed via {kind}: R={result.score:.4f} (pop={len(pop.members)})")

    logprint(f"Initial population: {len(pop.members)} members, best R={pop.best_R():.6f}")

    R_target = r_start - r_step

    for rnd in range(rounds):
        if not pop.members:
            logprint("Population empty, reinitializing...")
            pxs, pys, pts = apply_perturbation(xs0, ys0, ts0, r_start)
            xs_, ys_, ts_, E = lbfgs_refine(pxs, pys, pts, r_start)
            if E < 1e-4:
                result = official_validate(xs_, ys_, ts_)
                if result.valid:
                    pop.add(xs_, ys_, ts_, float(result.score))
            continue

        # Sample base + apply perturbation
        bxs, bys, bts, bR = pop.sample()
        kind = random.choice(list(PERTURBATIONS) + ['wall_slide', 'herringbone',
                                                       'cegp', 'cegp'])
        if kind == 'cegp':
            n_rem = random.choice([2, 3, 4])
            pxs, pys, pts = cegp_repair(bxs, bys, bts, R_target, n_remove=n_rem)
        else:
            pxs, pys, pts = apply_perturbation(bxs, bys, bts, R_target, kind=kind)

        xs_, ys_, ts_, E = lbfgs_refine(pxs, pys, pts, R_target)

        label = f"[{rnd+1:4d}] {kind:12s} E={E:.2e}"
        if E < 1e-4:
            result = official_validate(xs_, ys_, ts_)
            if result.valid:
                actual_R = float(result.score)
                admitted = pop.add(xs_, ys_, ts_, actual_R)

                if actual_R < global_best_R:
                    global_best_R = actual_R
                    xs_c, ys_c, ts_c = center_solution(xs_, ys_, ts_)
                    save_best(xs_c, ys_c, ts_c, actual_R)
                    logprint(f"  ★ NEW GLOBAL BEST: R = {actual_R:.6f}")
                    # Update baseline for future perturbations
                    xs0, ys0, ts0 = xs_c, ys_c, ts_c
                    R_target = actual_R - r_step

                if admitted:
                    logprint(f"{label} → R={actual_R:.4f} (pop={len(pop.members)} basins={pop.n_distinct_basins})")
                else:
                    logprint(f"{label} → R={actual_R:.4f} dup")
            else:
                logprint(f"{label} → Shapely reject")
        else:
            if (rnd + 1) % 20 == 0:
                logprint(f"{label}")

        # Periodic diversity report
        if (rnd + 1) % 50 == 0:
            conf = pop.confidence_level()
            logprint(f"  --- Round {rnd+1}: basins={pop.n_distinct_basins} "
                     f"found={pop.n_total_found} conf={conf:.2f} best_R={global_best_R:.6f} ---")
            # Check for convergence signal
            if conf >= 0.03 and pop.n_total_found >= 100:
                logprint(f"  Convergence signal: best basin visited {conf*100:.0f}% of time")

        # Widen search if stuck
        if (rnd + 1) % 30 == 0 and global_best_R >= r_start - 0.005:
            R_target = r_start + random.uniform(-0.01, 0.02)  # explore nearby

    logprint(f"\n=== PBH done | best R={global_best_R:.6f} | "
             f"distinct basins={pop.n_distinct_basins} | total found={pop.n_total_found} ===")

    # Symmetry check on best
    syms = detect_symmetry(xs0, ys0, ts0)
    logprint(f"Symmetry of best: " + ", ".join(f"C{k}={v:.3f}" for k, v in syms.items()))

    log.close()
    return global_best_R


if __name__ == '__main__':
    from mbh import PERTURBATIONS
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--pop', type=int, default=15)
    parser.add_argument('--d-cut', type=float, default=0.3)
    args = parser.parse_args()

    np.random.seed(int(time.time()) % 10000)
    random.seed(int(time.time()) % 10000)

    run_pbh(rounds=args.rounds, pop_size=args.pop, d_cut=args.d_cut)
