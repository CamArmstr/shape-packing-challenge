#!/usr/bin/env python3
"""
optimize_v3.py — Targeting R ≈ 2.96 (leaderboard competitive).

Key insight: top solutions are R ≈ 2.961-2.973. Our 4-11 structure at 3.010
is a wrong local minimum. The winning topology likely uses 3 shells.

Strategy:
- 6 workers, each with a different 3-shell topology
- Much longer SA runs (200M steps) to fully compress
- Higher lambda to force feasibility at small R
- SA temperature schedule tuned for R ≈ 2.96 target
- L-BFGS refinement on promising candidates
"""

import sys, os, json, math, time, random, multiprocessing as mp, argparse
import numpy as np

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from sa_v2 import sa_run_v2_wrapper as _sa_v2
from gjk_numba import overlap_energy_gjk

import importlib.util
spec = importlib.util.spec_from_file_location("overnight", "overnight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
from seeds import (seed_3_5_7, seed_2_5_8, seed_3_4_8, seed_1_5_9,
                   seed_5_10, seed_from_best, seed_2_6_7, seed_1_5_9)

import fcntl as _fcntl

BEST_FILE = 'best_solution.json'
BEST_FILE_LOCK_PATH = BEST_FILE + '.lock'
LOG_FILE = 'v3_log.txt'
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


# Worker strategies: all 3-shell topologies, tuned for R ≈ 2.96
# Higher T_start (more exploration), higher lam (stronger feasibility pressure)
# 200M steps per run instead of 50M
STRATEGIES = [
    # (name, seed_fn, T_start, T_end, lam_start, lam_end, n_steps)
    # Worker 0: 3-5-7 balanced 3-shell (most promising from v2)
    ('3_5_7_a',   lambda: seed_3_5_7(),  2.0, 0.0005, 50, 10000, 200_000_000),
    # Worker 1: 2-5-8 tight outer + solid middle
    ('2_5_8_a',   lambda: seed_2_5_8(),  2.0, 0.0005, 50, 10000, 200_000_000),
    # Worker 2: 3-4-8 variant 3-shell
    ('3_4_8_a',   lambda: seed_3_4_8(),  2.0, 0.0005, 50, 10000, 200_000_000),
    # Worker 3: 1-5-9 minimal center
    ('1_5_9_a',   lambda: seed_1_5_9(),  2.0, 0.0005, 50, 10000, 200_000_000),
    # Worker 4: 2-6-7 heavy middle
    ('2_6_7_a',   lambda: seed_2_6_7(),  2.0, 0.0005, 50, 10000, 200_000_000),
    # Worker 5: 3-5-7 with different SA schedule (more aggressive)
    ('3_5_7_b',   lambda: seed_3_5_7(),  3.0, 0.0003, 20, 15000, 200_000_000),
]


def worker(worker_id, global_best, result_queue, stop_event):
    os.nice(10)
    np.random.seed(worker_id * 1000 + int(time.time()) % 10000)
    random.seed(worker_id * 7919 + int(time.time()) % 10000)

    name, builder, T_start, T_end, lam_start, lam_end, n_steps = STRATEGIES[worker_id]
    run = 0

    while not stop_event.is_set():
        run += 1
        label = f"w{worker_id}_{name}_{run}"
        print(f"  [{label}] starting (200M steps)", flush=True)

        init = builder()
        if init is None:
            print(f"  [{label}] seed failed, retrying", flush=True)
            time.sleep(2)
            continue
        xs, ys, ts = init

        run_seed = run * 137 * (worker_id + 1) + int(time.time()) % 10000

        rx, ry, rt, r_fast = _sa_v2(
            xs, ys, ts,
            n_steps=n_steps, T_start=T_start, T_end=T_end,
            lam_start=lam_start, lam_end=lam_end,
            seed=run_seed
        )

        if rx is None:
            print(f"  [{label}] no feasible result", flush=True)
            continue

        val = mod.official_validate(rx, ry, rt)
        if val.valid:
            score = float(val.score)
            print(f"  [{label}] VALID: {score:.6f} (best: {global_best.value:.6f})", flush=True)
            result_queue.put((score, rx.tolist(), ry.tolist(), rt.tolist(), label))
        else:
            print(f"  [{label}] INVALID (official)", flush=True)


def send_telegram(msg):
    try:
        import subprocess
        subprocess.Popen([
            'openclaw', 'message', 'send',
            '--channel', 'telegram',
            '--target', TELEGRAM_TARGET,
            '--message', msg
        ])
    except:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--runtime', type=int, default=7200, help='Default 2 hours')
    args = parser.parse_args()

    n_workers = min(args.workers, len(STRATEGIES))

    print("Warming up Numba JIT...", flush=True)
    try:
        from sa_v2 import sa_run_v2
        _xs = np.random.randn(N) * 2.0
        _ys = np.random.randn(N) * 2.0
        _ts = np.random.rand(N) * 2 * math.pi
        sa_run_v2(_xs, _ys, _ts, 1000, 0.1, 0.001, 500, 5000, 0)
        overlap_energy_gjk(_xs, _ys, _ts)
        print("JIT ready.", flush=True)
    except Exception as e:
        print(f"JIT warm-up failed: {e}", flush=True)

    _, _, _, initial_best = load_best()
    print(f"Starting best: {initial_best:.6f}")
    print(f"TARGET: R ≈ 2.96 (leaderboard competitive)")
    print(f"Launching {n_workers} workers for {args.runtime}s")

    log = open(LOG_FILE, 'a')
    log.write(f"\n{'='*60}\n{time.strftime('%Y-%m-%d %H:%M:%S')} v3 starting (target R≈2.96)\n")
    log.flush()

    global_best = mp.Value('d', initial_best)
    result_queue = mp.Queue()
    stop_event = mp.Event()

    workers = []
    for i in range(n_workers):
        p = mp.Process(target=worker, args=(i, global_best, result_queue, stop_event), daemon=True)
        p.start()
        workers.append(p)
        time.sleep(0.5)

    t_start = time.time()

    try:
        while time.time() - t_start < args.runtime:
            time.sleep(15)

            while not result_queue.empty():
                score, rx, ry, rt, label = result_queue.get_nowait()
                if score < global_best.value:
                    rx_a = np.array(rx); ry_a = np.array(ry); rt_a = np.array(rt)
                    improved = save_if_better(rx_a, ry_a, rt_a, score, global_best)
                    if improved:
                        msg = f"*** NEW BEST: {global_best.value:.6f} (from {label}) ***"
                        print(f"\n{msg}\n", flush=True)
                        log.write(f"{time.strftime('%H:%M:%S')} {msg}\n"); log.flush()
                        send_telegram(f"🎯 v3 new best: R={global_best.value:.6f} ({label})")

            elapsed = time.time() - t_start
            alive = sum(1 for p in workers if p.is_alive())
            status = f"[{elapsed:.0f}s] Best={global_best.value:.6f} Workers={alive}/{n_workers}"
            print(status, flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for p in workers:
            p.terminate()
            p.join(timeout=3)

    final = global_best.value
    msg = f"v3 done. Final: {final:.6f} (started: {initial_best:.6f})"
    print(f"\n{msg}")
    log.write(f"\n{time.strftime('%H:%M:%S')} {msg}\n")
    log.close()

    if final < initial_best:
        send_telegram(f"✅ v3 finished. Improved: {initial_best:.6f} → {final:.6f}")
    else:
        send_telegram(f"📊 v3 finished. Best: {final:.6f}")


if __name__ == '__main__':
    mp.set_start_method('fork')
    main()
