"""
topology_run.py — Test new topology seed generators and optimize via PBH.

Tests conjugate-pair and C5-pentagonal seed topologies, runs PBH on each
valid start, tracks best R per topology.

Usage:
    python3 topology_run.py [--trials 20] [--rounds 500]
"""

import math
import time
import random
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from seeds import (seed_conjugate, seed_conjugate_7pairs, seed_c5, seed_c5_loose,
                   seed_brickwall, seed_brickwall_tight, seed_brickwall_random)
from pbh import run_pbh
from mbh import official_validate, save_best, center_solution, BEST_FILE

CURRENT_BEST_R = 2.974  # threshold for "new best" alert


def test_topology(name, gen_func, n_trials, pbh_rounds):
    """Test a topology: generate seeds, run PBH on valid ones."""
    print(f"\n{'='*60}")
    print(f"  Topology: {name}")
    print(f"{'='*60}")

    valid_starts = 0
    best_r = float('inf')

    for trial in range(n_trials):
        seed_val = int(time.time() * 1000) % 1000000 + trial * 137
        result = gen_func(seed=seed_val)

        if result is None:
            print(f"  Trial {trial+1:2d}/{n_trials}: seed failed")
            continue

        xs, ys, ts = result
        r0 = official_validate(xs, ys, ts)
        if not r0.valid:
            print(f"  Trial {trial+1:2d}/{n_trials}: seed invalid (overlap)")
            continue

        valid_starts += 1
        start_r = float(r0.score)
        print(f"  Trial {trial+1:2d}/{n_trials}: valid start R={start_r:.4f}, running PBH...",
              flush=True)

        # Run PBH from this start — do NOT save to disk (seeds start worse than current best)
        np.random.seed(seed_val)
        random.seed(seed_val)
        final_r = run_pbh(n_pop=8, rounds=pbh_rounds, verbose=False,
                          start=(xs, ys, ts), save_to_disk=False)

        print(f"    PBH result: R={final_r:.6f}")

        if final_r < best_r:
            best_r = final_r

        # Check for new global best — save only if it beats current best
        if final_r < CURRENT_BEST_R:
            print(f"\n  !!!! NEW GLOBAL BEST: R={final_r:.6f} (was {CURRENT_BEST_R}) !!!!")
            print(f"  Topology: {name}, Trial: {trial+1}")
            # Now save to disk
            # Re-validate to get the best solution from PBH's internal state
            # Since PBH returned the score, we need the actual solution
            # The best solution was already centered by PBH internally
            # Save it properly by writing to best_solution.json
            with open(BEST_FILE) as f:
                disk_data = json.load(f)
            disk_xs = np.array([s['x'] for s in disk_data])
            disk_ys = np.array([s['y'] for s in disk_data])
            disk_ts = np.array([s['theta'] for s in disk_data])
            disk_r = official_validate(disk_xs, disk_ys, disk_ts)
            if disk_r.valid and disk_r.score > final_r:
                # PBH didn't save (save_to_disk=False), but we found a record
                print(f"  NOTE: PBH found R={final_r:.6f} but didn't save to disk.")
                print(f"  Re-running PBH with save_to_disk=True to capture the result...")
                np.random.seed(seed_val)
                random.seed(seed_val)
                run_pbh(n_pop=8, rounds=pbh_rounds, verbose=False,
                        start=(xs, ys, ts), save_to_disk=True)

    return valid_starts, best_r, n_trials


def main():
    parser = argparse.ArgumentParser(description='Topology seed testing + PBH optimization')
    parser.add_argument('--trials', type=int, default=20, help='Number of seed trials per topology')
    parser.add_argument('--rounds', type=int, default=500, help='PBH rounds per valid start')
    parser.add_argument('--topology', type=str, default=None, help='Run only this topology (conjugate/conjugate_7/c5/c5_loose)')
    args = parser.parse_args()

    topologies = [
        ('seed_conjugate',        seed_conjugate),
        ('seed_conjugate_7',      seed_conjugate_7pairs),
        ('seed_c5',               seed_c5),
        ('seed_c5_loose',         seed_c5_loose),
        ('seed_brickwall',        seed_brickwall),
        ('seed_brickwall_tight',  seed_brickwall_tight),
        ('seed_brickwall_random', seed_brickwall_random),
    ]

    if args.topology:
        topologies = [(n, f) for n, f in topologies if n == args.topology]
        if not topologies:
            print(f"Unknown topology: {args.topology}. Choose from: conjugate, conjugate_7, c5, c5_loose")
            sys.exit(1)

    results = []
    for name, gen_func in topologies:
        valid, best_r, total = test_topology(name, gen_func, args.trials, args.rounds)
        results.append((name, valid, total, best_r, args.rounds))

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Topology':<20s} | {'Valid starts':>12s} | {'Best R':>8s} | {'Rounds':>6s}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*8}-+-{'-'*6}")
    for name, valid, total, best_r, rounds in results:
        r_str = f"{best_r:.4f}" if best_r < float('inf') else "N/A"
        print(f"  {name:<20s} | {valid:>5d}/{total:<5d}  | {r_str:>8s} | {rounds:>6d}")
    print()

    # Check if any beat current best
    global_best = min(r[3] for r in results)
    if global_best < CURRENT_BEST_R:
        print(f"  *** NEW RECORD: R={global_best:.6f} (beat {CURRENT_BEST_R}) ***")
    else:
        print(f"  Best found: R={global_best:.6f} (current best: {CURRENT_BEST_R})")


if __name__ == '__main__':
    main()
