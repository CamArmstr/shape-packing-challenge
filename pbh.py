"""
pbh.py — Population Basin Hopping for semicircle packing.

Wraps mbh.py: maintains a population of N_pop diverse solutions,
each improved via MBH iterations.

Usage:
    python pbh.py [--pop 8] [--rounds 1000]
"""

import json, math, time, random, os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mbh import (
    mbh_iteration, fss_escape, load_best, save_best, center_solution,
    official_validate, N, TWO_PI, BEST_FILE
)

LOG_FILE = os.path.join(os.path.dirname(__file__), 'pbh_log.txt')


# ── Dissimilarity measure ────────────────────────────────────────────────────

def sorted_distances(xs, ys):
    """Sorted pairwise center distances (105 values for N=15)."""
    dists = []
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(math.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2))
    dists.sort()
    return np.array(dists)


def dissimilarity(xs1, ys1, xs2, ys2):
    """L2 norm of sorted pairwise distance vectors."""
    d1 = sorted_distances(xs1, ys1)
    d2 = sorted_distances(xs2, ys2)
    return float(np.sqrt(np.sum((d1 - d2)**2)))


def is_diverse(xs, ys, population, threshold=0.3):
    """Check if solution is diverse enough from all population members."""
    for member in population:
        d1 = sorted_distances(xs, ys)
        d2 = sorted_distances(member[0], member[1])
        diff = d1 - d2
        if np.var(diff) < threshold:
            return False
    return True


def most_similar_idx(xs, ys, population):
    """Index of population member most similar to (xs, ys)."""
    d_new = sorted_distances(xs, ys)
    best_idx = 0
    best_dist = float('inf')
    for i, member in enumerate(population):
        d_m = sorted_distances(member[0], member[1])
        dist = float(np.sqrt(np.sum((d_new - d_m)**2)))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


# ── Population Basin Hopping ─────────────────────────────────────────────────

def run_pbh(n_pop=8, rounds=1000, verbose=True, start=None, save_to_disk=True):
    """
    Population Basin Hopping main loop.

    start: optional (xs, ys, ts) tuple to use as initial solution
           instead of loading from best_solution.json.
    save_to_disk: if False, do not save improvements to best_solution.json
                  (useful for topology testing where start solutions are worse).
    """
    log = open(LOG_FILE, 'a')
    def logprint(msg):
        print(msg, flush=True)
        log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        log.flush()

    # Load current best or use provided start
    if start is not None:
        xs0, ys0, ts0 = start
    else:
        xs0, ys0, ts0 = load_best()
    r0 = official_validate(xs0, ys0, ts0)
    if not r0.valid:
        logprint("WARNING: loaded solution invalid!")
        global_best = float('inf')
    else:
        global_best = float(r0.score)

    logprint(f"=== PBH | pop={n_pop} | rounds={rounds} | start R={global_best:.6f} ===")

    # Initialize population by perturbing best solution
    # Each member: (xs, ys, ts, score)
    population = [(xs0.copy(), ys0.copy(), ts0.copy(), global_best)]

    for i in range(n_pop - 1):
        pxs = xs0 + np.random.uniform(-0.5, 0.5, N)
        pys = ys0 + np.random.uniform(-0.5, 0.5, N)
        pts = (ts0 + np.random.uniform(-0.5, 0.5, N)) % TWO_PI
        result = official_validate(pxs, pys, pts)
        if result.valid:
            population.append((pxs, pys, pts, float(result.score)))
        else:
            # Use with inf score as placeholder
            population.append((pxs, pys, pts, float('inf')))

    logprint(f"Population initialized: {len(population)} members")

    R = global_best + 0.01
    fss_counter = 0

    for rnd in range(rounds):
        # Adaptive perturbation schedule: large jumps early, fine-tune late
        frac = rnd / max(rounds - 1, 1)
        if frac < 0.15:
            delta = 2.5   # rounds 0-15%: big kicks, escape starting basin fast
        elif frac < 0.40:
            delta = 1.2   # rounds 15-40%: medium kicks, explore basin region
        else:
            delta = 0.4   # rounds 40-100%: fine-tune basin floor

        # For each population member, run 1 MBH iteration
        for mi in range(len(population)):
            mxs, mys, mts, mscore = population[mi]
            member_R = mscore + 0.01 if mscore < float('inf') else R

            nxs, nys, nts, nscore, improved, kind = mbh_iteration(
                mxs, mys, mts, member_R, mscore, delta=delta
            )

            if improved:
                fss_counter = 0
                # Check diversity and update population
                sim_idx = most_similar_idx(nxs, nys, population)

                if nscore < population[sim_idx][3]:
                    population[sim_idx] = (nxs.copy(), nys.copy(), nts.copy(), nscore)

                # Update global best
                if nscore < global_best:
                    global_best = nscore
                    R = global_best + 0.01
                    cxs, cys, cts = center_solution(nxs, nys, nts)
                    if save_to_disk:
                        save_best(cxs, cys, cts, global_best)
                    logprint(f"  [{rnd+1:4d}.{mi}] ★ {kind:10s} → R={global_best:.6f}")
            else:
                # Update member in-place even without global improvement
                # (member may have improved internally via MBH's monotonic acceptance)
                population[mi] = (nxs.copy(), nys.copy(), nts.copy(), nscore)
                fss_counter += 1

        # FSS escape for best member every 10 non-improving rounds
        if fss_counter > 0 and fss_counter % (10 * len(population)) == 0:
            best_mi = min(range(len(population)), key=lambda i: population[i][3])
            bxs, bys, bts, bscore = population[best_mi]
            bR = bscore + 0.01 if bscore < float('inf') else R
            nxs, nys, nts, nscore, fss_imp = fss_escape(bxs, bys, bts, bR, bscore)
            if fss_imp:
                population[best_mi] = (nxs.copy(), nys.copy(), nts.copy(), nscore)
                if nscore < global_best:
                    global_best = nscore
                    R = global_best + 0.01
                    cxs, cys, cts = center_solution(nxs, nys, nts)
                    if save_to_disk:
                        save_best(cxs, cys, cts, global_best)
                    logprint(f"  [{rnd+1:4d}] ★ FSS_escape → R={global_best:.6f}")
                fss_counter = 0

        # Status report every 20 rounds
        if (rnd + 1) % 20 == 0:
            scores = [m[3] for m in population if m[3] < float('inf')]
            n_valid = len(scores)
            best_pop = min(scores) if scores else float('inf')
            logprint(f"  --- Round {rnd+1}: global_best={global_best:.6f} "
                     f"pop_best={best_pop:.6f} valid={n_valid}/{len(population)} ---")

        # Reload from disk periodically (coordination with other workers)
        if save_to_disk and (rnd + 1) % 50 == 0:
            try:
                dxs, dys, dts = load_best()
                dr = official_validate(dxs, dys, dts)
                if dr.valid and dr.score < global_best:
                    global_best = float(dr.score)
                    R = global_best + 0.01
                    # Inject into population replacing worst
                    worst_idx = max(range(len(population)), key=lambda i: population[i][3])
                    population[worst_idx] = (dxs.copy(), dys.copy(), dts.copy(), global_best)
                    logprint(f"  [{rnd+1:4d}] Reloaded disk best: R={global_best:.6f}")
            except Exception:
                pass

    logprint(f"\n=== PBH done | best R={global_best:.6f} ===")
    log.close()
    return global_best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=1000)
    args = parser.parse_args()

    np.random.seed(int(time.time()) % 100000)
    random.seed(int(time.time()) % 100000)

    run_pbh(n_pop=args.pop, rounds=args.rounds)
