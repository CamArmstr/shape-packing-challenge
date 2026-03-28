#!/usr/bin/env python3
"""
optimize_v4.py — Two-phase chained optimization targeting R ≈ 2.96.

Phase 1 (HOT): Quick compress from seed (R≈4.2) to R≈3.1
  - 20M steps, high T, low lambda
  - Goal: find the right contact topology, don't polish

Phase 2 (COLD): Long exploitation from Phase 1 result
  - 200M steps, lower T, high lambda
  - Goal: squeeze R from ~3.1 toward 2.96
  - Repeated: if R > 3.0, restart Phase 1 with different seed

Each worker cycles: Phase1 → Phase2 → (if improved) more Phase2 → else new Phase1

6 workers with different 3-shell topologies.
"""

import sys, os, json, math, time, random, multiprocessing as mp, argparse
import numpy as np

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from sa_v2 import sa_run_v2_wrapper as _sa_v2

import importlib.util
spec = importlib.util.spec_from_file_location("overnight", "overnight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
from seeds import (seed_3_5_7, seed_2_5_8, seed_3_4_8, seed_1_5_9,
                   seed_2_6_7, seed_5_10)

import fcntl as _fcntl

BEST_FILE = 'best_solution.json'
BEST_FILE_LOCK_PATH = BEST_FILE + '.lock'
N = 15
TELEGRAM_TARGET = '1602537663'


def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    bx = np.array([s['x'] for s in raw])
    by = np.array([s['y'] for s in raw])
    bt = np.array([s['theta'] for s in raw])
    val = mod.official_validate(bx, by, bt)
    return bx, by, bt, val.score if val.valid else float('inf')


def save_if_better(rx, ry, rt, score, global_best_ref):
    if score >= global_best_ref.value:
        return False
    with open(BEST_FILE_LOCK_PATH, 'w') as lf:
        try:
            _fcntl.flock(lf, _fcntl.LOCK_EX)
            try:
                with open(BEST_FILE) as f:
                    disk_raw = json.load(f)
                disk_val = mod.official_validate(
                    np.array([s['x'] for s in disk_raw]),
                    np.array([s['y'] for s in disk_raw]),
                    np.array([s['theta'] for s in disk_raw])
                )
                if disk_val.valid and disk_val.score < global_best_ref.value:
                    global_best_ref.value = disk_val.score
            except:
                pass
            if score >= global_best_ref.value:
                return False
            raw = [{'x': float(rx[i]), 'y': float(ry[i]), 'theta': float(rt[i])} for i in range(N)]
            sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
            result = validate_and_score(sol)
            if not result.valid or result.score >= global_best_ref.value:
                return False
            cx, cy = result.mec[0], result.mec[1]
            centered = [{'x': round(d['x']-cx,6), 'y': round(d['y']-cy,6),
                         'theta': round(d['theta'],6)} for d in raw]
            global_best_ref.value = result.score
            with open(BEST_FILE, 'w') as f:
                json.dump(centered, f, indent=2)
            try:
                sol_v = [Semicircle(d['x'],d['y'],d['theta']) for d in centered]
                r_v = validate_and_score(sol_v)
                from src.semicircle_packing.visualization import plot_packing
                plot_packing(sol_v, r_v.mec, save_path='best_solution.png')
            except: pass
            return True
        finally:
            _fcntl.flock(lf, _fcntl.LOCK_UN)


SEED_FNS = [seed_3_5_7, seed_2_5_8, seed_3_4_8, seed_1_5_9, seed_2_6_7, seed_5_10]
SEED_NAMES = ['3-5-7', '2-5-8', '3-4-8', '1-5-9', '2-6-7', '5-10']


def worker(worker_id, global_best, result_queue, stop_event):
    os.nice(10)
    np.random.seed(worker_id * 1000 + int(time.time()) % 10000)
    random.seed(worker_id * 7919 + int(time.time()) % 10000)

    seed_fn = SEED_FNS[worker_id % len(SEED_FNS)]
    seed_name = SEED_NAMES[worker_id % len(SEED_NAMES)]
    run = 0

    while not stop_event.is_set():
        run += 1
        label = f"w{worker_id}_{seed_name}_{run}"

        # ── Phase 1: HOT compress (seed → R≈3.1) ──
        print(f"  [{label}] Phase1: hot compress", flush=True)

        init = seed_fn()
        if init is None:
            print(f"  [{label}] seed failed, retrying", flush=True)
            time.sleep(2)
            continue
        xs, ys, ts = init

        seed_val = run * 137 * (worker_id + 1) + int(time.time()) % 10000

        # Hot SA: high T, low lambda, 20M steps
        rx, ry, rt, r1 = _sa_v2(
            xs, ys, ts,
            n_steps=20_000_000,
            T_start=3.0, T_end=0.01,
            lam_start=20, lam_end=2000,
            seed=seed_val
        )

        if rx is None:
            print(f"  [{label}] Phase1: no feasible", flush=True)
            continue

        val1 = mod.official_validate(rx, ry, rt)
        if not val1.valid:
            print(f"  [{label}] Phase1: invalid", flush=True)
            continue

        r1_score = float(val1.score)
        print(f"  [{label}] Phase1: R={r1_score:.4f}", flush=True)

        if r1_score > 3.3:
            # Too loose, skip phase 2
            print(f"  [{label}] Phase1 too loose ({r1_score:.3f}), restarting", flush=True)
            continue

        # ── Phase 2: COLD exploit (R≈3.1 → target 2.96) ──
        # Run multiple cold passes from the Phase 1 result
        best_rx, best_ry, best_rt = rx.copy(), ry.copy(), rt.copy()
        best_score = r1_score

        for cold_pass in range(5):  # up to 5 cold passes
            if stop_event.is_set():
                break

            print(f"  [{label}] Phase2 pass {cold_pass+1}: from R={best_score:.4f}", flush=True)

            # Add small noise to escape local minimum
            noise = 0.02 + cold_pass * 0.01
            px = best_rx + np.random.randn(N) * noise
            py = best_ry + np.random.randn(N) * noise
            pt = best_rt + np.random.randn(N) * noise * 2

            cold_seed = seed_val * 100 + cold_pass

            # Cold SA: lower T, higher lambda, 200M steps
            cx, cy, ct, r2 = _sa_v2(
                px, py, pt,
                n_steps=200_000_000,
                T_start=0.5, T_end=0.0002,
                lam_start=500, lam_end=20000,
                seed=cold_seed
            )

            if cx is None:
                print(f"  [{label}] Phase2 pass {cold_pass+1}: no feasible", flush=True)
                continue

            val2 = mod.official_validate(cx, cy, ct)
            if not val2.valid:
                print(f"  [{label}] Phase2 pass {cold_pass+1}: invalid", flush=True)
                continue

            r2_score = float(val2.score)
            print(f"  [{label}] Phase2 pass {cold_pass+1}: R={r2_score:.6f} (best this chain: {best_score:.6f})", flush=True)

            result_queue.put((r2_score, cx.tolist(), cy.tolist(), ct.tolist(), f"{label}_p2_{cold_pass+1}"))

            if r2_score < best_score:
                best_rx, best_ry, best_rt = cx.copy(), cy.copy(), ct.copy()
                best_score = r2_score
                print(f"  [{label}] Chain improved: R={best_score:.6f}", flush=True)

                # If we improved, keep going
                if best_score < global_best.value:
                    improved = save_if_better(best_rx, best_ry, best_rt, best_score, global_best)
                    if improved:
                        result_queue.put(('NEW_BEST', best_score, 0, 0, label))
            else:
                # No improvement — try a bigger perturbation or break
                if cold_pass >= 2:
                    break  # diminishing returns, start fresh


def send_telegram(msg):
    try:
        import subprocess
        subprocess.Popen(['openclaw', 'message', 'send',
                          '--channel', 'telegram', '--target', TELEGRAM_TARGET,
                          '--message', msg])
    except:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--runtime', type=int, default=7200)
    args = parser.parse_args()

    print("Warming up Numba JIT...", flush=True)
    from sa_v2 import sa_run_v2
    _xs = np.random.randn(N) * 2.0
    _ys = np.random.randn(N) * 2.0
    _ts = np.random.rand(N) * 2 * math.pi
    sa_run_v2(_xs, _ys, _ts, 1000, 0.1, 0.001, 500, 5000, 0)
    print("JIT ready.", flush=True)

    _, _, _, initial_best = load_best()
    print(f"Starting best: {initial_best:.6f}")
    print(f"TARGET: R ≈ 2.96")
    print(f"Strategy: Phase1 (hot 20M) → Phase2 (cold 200M) × 5 passes")
    print(f"Launching {args.workers} workers for {args.runtime}s")

    global_best = mp.Value('d', initial_best)
    result_queue = mp.Queue()
    stop_event = mp.Event()

    workers = []
    for i in range(args.workers):
        p = mp.Process(target=worker, args=(i, global_best, result_queue, stop_event), daemon=True)
        p.start()
        workers.append(p)
        time.sleep(0.5)

    t_start = time.time()

    try:
        while time.time() - t_start < args.runtime:
            time.sleep(15)

            while not result_queue.empty():
                item = result_queue.get_nowait()
                if item[0] == 'NEW_BEST':
                    _, score, _, _, label = item
                    msg = f"*** NEW BEST: {score:.6f} ({label}) ***"
                    print(f"\n{msg}\n", flush=True)
                    send_telegram(f"🎯 v4 new best: R={score:.6f} ({label})")
                else:
                    score, rx, ry, rt, label = item
                    if score < global_best.value:
                        rx_a = np.array(rx); ry_a = np.array(ry); rt_a = np.array(rt)
                        improved = save_if_better(rx_a, ry_a, rt_a, score, global_best)
                        if improved:
                            msg = f"*** NEW BEST: {global_best.value:.6f} ({label}) ***"
                            print(f"\n{msg}\n", flush=True)
                            send_telegram(f"🎯 v4 new best: R={global_best.value:.6f} ({label})")

            elapsed = time.time() - t_start
            alive = sum(1 for p in workers if p.is_alive())
            print(f"[{elapsed:.0f}s] Best={global_best.value:.6f} Workers={alive}/{args.workers}", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for p in workers:
            p.terminate()
            p.join(timeout=3)

    final = global_best.value
    print(f"\nv4 done. Final: {final:.6f} (started: {initial_best:.6f})")
    if final < initial_best:
        send_telegram(f"✅ v4 improved: {initial_best:.6f} → {final:.6f}")
    else:
        send_telegram(f"📊 v4 done. Best: {final:.6f}")


if __name__ == '__main__':
    mp.set_start_method('fork')
    main()
