#!/usr/bin/env python3
"""
lns2.py — Fixed Large Neighborhood Search.

Strategy: remove boundary-active shapes, reinsert in contact-seeking positions,
run short hill-climb, save only if strictly better than disk (re-read under lock).

Goal: change the contact graph (contacts=30 → 31+) to find new basins.
"""
import sys, os, json, time, random, math, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
import fcntl

BEST_FILE = os.path.join(os.path.dirname(__file__), 'best_solution.json')
LOG_FILE  = os.path.join(os.path.dirname(__file__), 'lns2.log')
N = 15; TWO_PI = 2 * math.pi

def load_best():
    with open(BEST_FILE) as f: raw = json.load(f)
    return (np.array([s['x'] for s in raw]),
            np.array([s['y'] for s in raw]),
            np.array([s['theta'] for s in raw]))

def score_sol(xs, ys, ts):
    sol = [Semicircle(float(xs[i]),float(ys[i]),float(ts[i])) for i in range(N)]
    r = validate_and_score(sol)
    return (r.score if r.valid else float('inf')), r

def load_best_score():
    xs, ys, ts = load_best()
    s, _ = score_sol(xs, ys, ts)
    return s

def save_if_better(xs, ys, ts):
    """Always re-read from disk under lock. Never trust a cached value."""
    s, r = score_sol(xs, ys, ts)
    if s == float('inf'): return False, s

    lock_path = BEST_FILE + '.lock'
    with open(lock_path, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            # ALWAYS re-read from disk under lock
            xs_disk, ys_disk, ts_disk = load_best()
            s_disk, _ = score_sol(xs_disk, ys_disk, ts_disk)
            if s >= s_disk:
                return False, s_disk  # disk is better, don't overwrite

            cx, cy = r.mec[0], r.mec[1]
            out = [{'x': round(float(xs[i]-cx),6), 'y': round(float(ys[i]-cy),6),
                    'theta': round(float(ts[i])%TWO_PI,6)} for i in range(N)]
            with open(BEST_FILE,'w') as f:
                json.dump(out, f, indent=2)

            # Auto-commit
            try:
                d = os.path.dirname(BEST_FILE)
                subprocess.run(['git','add','best_solution.json'], cwd=d, capture_output=True)
                subprocess.run(['git','commit','-m',f'best: R={s:.6f} (lns2 auto-commit)'], cwd=d, capture_output=True)
            except: pass

            return True, s
        finally:
            pass

def get_contacts(xs, ys, ts, threshold=0.02):
    """Return contact graph: set of (i,j) pairs within threshold distance."""
    contacts = set()
    for i in range(N):
        for j in range(i+1, N):
            sc_i = Semicircle(float(xs[i]),float(ys[i]),float(ts[i]))
            sc_j = Semicircle(float(xs[j]),float(ys[j]),float(ts[j]))
            # Approximate distance via center distance - 2 (for unit semicircles)
            d = math.hypot(xs[i]-xs[j], ys[i]-ys[j]) - 2.0
            if d < threshold:
                contacts.add((i,j))
    return contacts

def get_boundary_active_order(xs, ys, ts):
    """Return shape indices sorted by MEC contribution (highest first)."""
    sol = [Semicircle(float(xs[i]),float(ys[i]),float(ts[i])) for i in range(N)]
    r = validate_and_score(sol)
    if not r.valid: return list(range(N))
    cx, cy = r.mec[0], r.mec[1]
    contribs = []
    for i in range(N):
        x, y, t = xs[i], ys[i], ts[i]
        pts = [(x+math.cos(t), y+math.sin(t)),
               (x+math.cos(t+math.pi/2), y+math.sin(t+math.pi/2)),
               (x+math.cos(t-math.pi/2), y+math.sin(t-math.pi/2))]
        ri = max(math.hypot(px-cx, py-cy) for px,py in pts)
        contribs.append((ri, i))
    contribs.sort(reverse=True)
    return [idx for _,idx in contribs]

def quick_hillclimb(xs, ys, ts, n_trials=200, seed=None):
    """Fast refinement using Numba SA (10M steps) — much faster than Shapely per-step."""
    from sa_v2 import sa_run_v2_wrapper
    seed = seed or int(time.time()*1000) % 100000
    bx, by, bt, br = sa_run_v2_wrapper(
        xs, ys, ts, n_steps=20_000_000,
        T_start=0.5, T_end=0.001,
        lam_start=500.0, lam_end=5000.0, seed=seed)
    if bx is None:
        return xs, ys, ts, float('inf')
    s, _ = score_sol(bx, by, bt)
    return bx, by, bt, s
    return xs, ys, ts, best_s

def try_contact_seeking_reinsert(xs, ys, ts, removed_idx, rng, n_candidates=25):
    """
    Reinsert removed shape in positions that maximise new contacts.
    Strategy: try positions adjacent to existing shapes (likely to form contacts)
    + some random positions for diversity.
    """
    # Base configuration without removed shape
    base_xs = np.delete(xs, removed_idx)
    base_ys = np.delete(ys, removed_idx)
    base_ts = np.delete(ts, removed_idx)
    n_base = N - 1

    best_s = float('inf')
    best_config = None

    for cand in range(n_candidates):
        # 60%: try position adjacent to an existing shape (contact-seeking)
        # 40%: random position in container
        if rng.random() < 0.6 and n_base > 0:
            # Pick a random existing shape and place near it
            anchor = rng.randint(0, n_base)
            angle = rng.uniform(0, TWO_PI)
            dist = rng.uniform(1.8, 2.3)  # just touching distance for unit semicircles
            x = float(base_xs[anchor]) + dist * math.cos(angle)
            y = float(base_ys[anchor]) + dist * math.sin(angle)
        else:
            # Random within container
            current_r = max(math.hypot(base_xs[i], base_ys[i]) + 1 for i in range(n_base))
            r_pos = rng.uniform(0, current_r - 0.5)
            angle = rng.uniform(0, TWO_PI)
            x = r_pos * math.cos(angle)
            y = r_pos * math.sin(angle)

        # Try multiple orientations
        for t_attempt in range(8):
            t = rng.uniform(0, TWO_PI)
            sc = Semicircle(x, y, t)
            # Check overlap with base
            ok = True
            for j in range(n_base):
                sc2 = Semicircle(float(base_xs[j]),float(base_ys[j]),float(base_ts[j]))
                if semicircles_overlap(sc, sc2):
                    ok = False; break
            if ok:
                # Valid placement — build full config and score
                cur_xs = np.append(base_xs, x)
                cur_ys = np.append(base_ys, y)
                cur_ts = np.append(base_ts, t)
                s, _ = score_sol(cur_xs, cur_ys, cur_ts)
                if s < best_s:
                    best_s = s
                    best_config = (cur_xs.copy(), cur_ys.copy(), cur_ts.copy())
                break

    return best_config, best_s

def run(wid, max_cycles=2000):
    rng = np.random.RandomState(wid * 9973 + int(time.time()) % 100000)
    log = open(LOG_FILE, 'a')
    def logprint(msg):
        ts = time.strftime('%H:%M:%S')
        line = f'{ts} [w{wid}] {msg}'
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()

    logprint(f'start | max_cycles={max_cycles}')
    cycles_no_improve = 0

    for cycle in range(max_cycles):
        # Always reload best from disk at cycle start
        xs, ys, ts = load_best()
        global_best = load_best_score()
        contacts_before = len(get_contacts(xs, ys, ts))

        # Choose removal strategy
        order = get_boundary_active_order(xs, ys, ts)
        if rng.random() < 0.7:
            remove_idx = order[0]  # boundary-active (most likely to help)
        else:
            remove_idx = int(rng.randint(0, N))  # random for diversity

        # Try contact-seeking reinsertion
        best_config, reinsert_s = try_contact_seeking_reinsert(xs, ys, ts, remove_idx, rng, n_candidates=25)

        if best_config is None:
            logprint(f'cycle {cycle:4d} remove={remove_idx} no valid placement')
            continue

        rx, ry, rt = best_config
        # Short hill-climb to refine
        rx, ry, rt, refined_s = quick_hillclimb(rx, ry, rt, n_trials=200, seed=wid*1000+cycle)

        contacts_after = len(get_contacts(rx, ry, rt))
        saved, saved_score = save_if_better(rx, ry, rt)

        if refined_s < global_best + 0.05:  # only log near-competitive results
            msg = f'cycle {cycle:4d} remove={remove_idx} contacts {contacts_before}→{contacts_after} refined={refined_s:.6f}'
            if saved: msg += f' *** NEW BEST R={saved_score:.6f} ***'
            logprint(msg)
        elif saved:
            logprint(f'cycle {cycle:4d} *** NEW BEST R={saved_score:.6f} *** contacts {contacts_before}→{contacts_after}')

        if saved:
            cycles_no_improve = 0
        else:
            cycles_no_improve += 1

        if cycles_no_improve % 100 == 0 and cycles_no_improve > 0:
            logprint(f'[{cycles_no_improve} cycles without improvement, global_best={global_best:.6f}]')

    logprint('done')
    log.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wid', type=int, default=0)
    parser.add_argument('--cycles', type=int, default=2000)
    args = parser.parse_args()
    run(args.wid, args.cycles)