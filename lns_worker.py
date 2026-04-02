#!/usr/bin/env python3
"""
lns_worker.py — Large Neighborhood Search via remove-and-reinsert.

Each LNS cycle:
1. Load best solution
2. Remove 1-3 shapes (prioritize boundary-active + high-contact)
3. Try 80 greedy reinsertion positions for each removed shape
4. Run 500-trial hill-climb on the best valid reinsertion
5. If better than current best: save

Designed to change contacts=29 by forcing shape(s) into new positions.
"""
import sys, os, json, time, random, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
import fcntl

BEST_FILE = os.path.join(os.path.dirname(__file__), 'best_solution.json')
LOG_FILE  = os.path.join(os.path.dirname(__file__), 'lns.log')
N = 15; TWO_PI = 2 * math.pi

def load_best():
    with open(BEST_FILE) as f: raw = json.load(f)
    xs = np.array([s['x'] for s in raw])
    ys = np.array([s['y'] for s in raw])
    ts = np.array([s['theta'] for s in raw])
    return xs, ys, ts

def score(xs, ys, ts):
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    r = validate_and_score(sol)
    return (r.score if r.valid else float('inf')), r

def save_if_better(xs, ys, ts, current_best):
    s, r = score(xs, ys, ts)
    if s >= current_best: return current_best, False
    lock_path = BEST_FILE + '.lock'
    with open(lock_path, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        xs2, ys2, ts2 = load_best()
        s2, _ = score(xs2, ys2, ts2)
        if s >= s2: return s2, False
        r2s, r2 = score(xs, ys, ts)
        cx, cy = r2.mec[0], r2.mec[1]
        out = [{'x': round(float(xs[i]-cx),6), 'y': round(float(ys[i]-cy),6),
                'theta': round(float(ts[i])%TWO_PI,6)} for i in range(N)]
        with open(BEST_FILE,'w') as f: json.dump(out, f, indent=2)
        return s, True

def quick_hillclimb(xs, ys, ts, n_trials=500):
    """Short hill-climb to refine a candidate."""
    rng = np.random.RandomState(int(time.time()*1000)%100000)
    best_s, _ = score(xs, ys, ts)
    if best_s == float('inf'): return xs, ys, ts, best_s
    for _ in range(n_trials):
        i = rng.randint(0, N)
        scale = 0.005
        nxs, nys, nts = xs.copy(), ys.copy(), ts.copy()
        nxs[i] += rng.uniform(-scale, scale)
        nys[i] += rng.uniform(-scale, scale)
        nts[i] = (nts[i] + rng.uniform(-scale*3, scale*3)) % TWO_PI
        ns, _ = score(nxs, nys, nts)
        if ns < best_s:
            best_s = ns
            xs, ys, ts = nxs, nys, nts
    return xs, ys, ts, best_s

def get_boundary_active(xs, ys, ts):
    """Return indices sorted by how much each shape contributes to R."""
    sol = [Semicircle(float(xs[i]), float(ys[i]), float(ts[i])) for i in range(N)]
    r = validate_and_score(sol)
    if not r.valid: return list(range(N))
    cx, cy = r.mec[0], r.mec[1]
    # Approximate farthest point per shape
    contributions = []
    for i in range(N):
        x, y, t = xs[i], ys[i], ts[i]
        # Check arc tip and edges
        pts = [
            (x + math.cos(t), y + math.sin(t)),
            (x + math.cos(t+math.pi/2), y + math.sin(t+math.pi/2)),
            (x + math.cos(t-math.pi/2), y + math.sin(t-math.pi/2)),
            (x, y),
        ]
        ri = max(math.hypot(px-cx, py-cy) for px,py in pts)
        contributions.append((ri, i))
    contributions.sort(reverse=True)
    return [idx for _, idx in contributions]

def try_reinsert(xs, ys, ts, removed_indices, rng, n_candidates=15):
    """Try reinserting removed shapes at candidate positions."""
    # Build base configuration without removed shapes
    base_xs = np.delete(xs, removed_indices)
    base_ys = np.delete(ys, removed_indices)
    base_ts = np.delete(ts, removed_indices)
    n_base = N - len(removed_indices)

    # Estimate current enclosing radius to set search bounds
    current_r = max(math.hypot(base_xs[i], base_ys[i]) + 1 for i in range(n_base))
    container_r = current_r * 0.95  # try to fit inside slightly smaller

    best_score = float('inf')
    best_config = None

    for candidate in range(n_candidates):
        # Try inserting removed shapes sequentially
        cur_xs = list(base_xs)
        cur_ys = list(base_ys)
        cur_ts = list(base_ts)
        valid = True

        for rem_idx in removed_indices:
            placed = False
            for attempt in range(150):
                # Sample position in container
                r = rng.uniform(0, container_r - 0.5)
                angle = rng.uniform(0, TWO_PI)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                t = rng.uniform(0, TWO_PI)

                # Check overlap with existing shapes
                sc = Semicircle(x, y, t)
                ok = True
                for j in range(len(cur_xs)):
                    sc2 = Semicircle(float(cur_xs[j]), float(cur_ys[j]), float(cur_ts[j]))
                    if semicircles_overlap(sc, sc2):
                        ok = False
                        break
                if ok:
                    cur_xs.append(x)
                    cur_ys.append(y)
                    cur_ts.append(t)
                    placed = True
                    break
            if not placed:
                valid = False
                break

        if not valid or len(cur_xs) != N:
            continue

        cxs = np.array(cur_xs)
        cys = np.array(cur_ys)
        cts = np.array(cur_ts)
        s, r = score(cxs, cys, cts)
        if s < best_score:
            best_score = s
            best_config = (cxs.copy(), cys.copy(), cts.copy())

    return best_config, best_score

def run(wid, max_cycles=500):
    rng = np.random.RandomState(wid * 9973 + int(time.time()) % 100000)
    log = open(LOG_FILE, 'a')
    def logprint(msg):
        print(msg, flush=True)
        log.write(time.strftime('%H:%M:%S') + f' [w{wid}] ' + msg + '\n')
        log.flush()

    xs, ys, ts = load_best()
    best_s, _ = score(xs, ys, ts)
    logprint(f'start R={best_s:.6f} | max_cycles={max_cycles}')

    for cycle in range(max_cycles):
        xs, ys, ts = load_best()
        best_s, _ = score(xs, ys, ts)

        # Choose how many shapes to remove (1-3)
        n_remove = 1  # single removal for dense packing

        # Prioritize boundary-active shapes
        order = get_boundary_active(xs, ys, ts)
        # Mix: sometimes take boundary-active, sometimes random
        if rng.random() < 0.6:
            removed_indices = order[:n_remove]
        else:
            removed_indices = list(rng.choice(N, n_remove, replace=False))

        # Try reinsertion
        best_config, reinsert_score = try_reinsert(xs, ys, ts, removed_indices, rng, n_candidates=15)

        if best_config is None:
            logprint(f'cycle {cycle:3d} remove={removed_indices} no valid reinsertion')
            continue

        rx, ry, rt = best_config
        if reinsert_score < float('inf'):
            # Short hill-climb to refine
            rx, ry, rt, refined_score = quick_hillclimb(rx, ry, rt, n_trials=500)
            saved_score, saved = save_if_better(rx, ry, rt, best_s)
            msg = f'cycle {cycle:3d} remove={removed_indices} reinsert={reinsert_score:.5f} refined={refined_score:.5f}'
            if saved: msg += ' *** NEW BEST ***'
            logprint(msg)
        else:
            logprint(f'cycle {cycle:3d} remove={removed_indices} invalid')

    logprint('done')
    log.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wid', type=int, default=0)
    parser.add_argument('--cycles', type=int, default=500)
    args = parser.parse_args()
    run(args.wid, args.cycles)
