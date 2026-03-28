#!/usr/bin/env python3
"""
optimize_v2.py — 6-worker parallel optimizer with Thompson Sampling SA,
diverse seed topologies, and thermal-aware scheduling.

6 workers on 16 cores (~37% sustained utilization, targeting <80°C).
Each worker starts from a distinct topology seed.
Workers share best_solution.json via file locking.

Usage:
    python optimize_v2.py [--workers 6] [--runtime 3600]
"""

import sys, os, json, math, time, random, multiprocessing as mp, argparse
import numpy as np

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

# Import SA v2 kernel — warm up Numba JIT before forking
from sa_v2 import sa_run_v2_wrapper as _sa_v2

import importlib.util
spec = importlib.util.spec_from_file_location("overnight", "overnight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle

from seeds import (
    seed_from_best, seed_4_11, seed_5_10, seed_3_5_7,
    seed_2_5_8, seed_3_4_8, seed_1_5_9
)

import fcntl as _fcntl

BEST_FILE = 'best_solution.json'
BEST_FILE_LOCK_PATH = BEST_FILE + '.lock'
LOG_FILE = 'v2_log.txt'
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
    """Save centered solution if better than global best. File-locked."""
    if score >= global_best_ref.value:
        return False

    with open(BEST_FILE_LOCK_PATH, 'w') as lf:
        try:
            _fcntl.flock(lf, _fcntl.LOCK_EX)

            # Re-read disk under lock
            try:
                with open(BEST_FILE) as f:
                    disk_raw = json.load(f)
                disk_val = mod.official_validate(
                    np.array([s['x'] for s in disk_raw]),
                    np.array([s['y'] for s in disk_raw]),
                    np.array([s['theta'] for s in disk_raw])
                )
                disk_R = disk_val.score if disk_val.valid else float('inf')
                if disk_R < global_best_ref.value:
                    global_best_ref.value = disk_R
            except Exception:
                disk_R = float('inf')

            if score >= global_best_ref.value:
                return False

            raw = [{'x': float(rx[i]), 'y': float(ry[i]), 'theta': float(rt[i])} for i in range(N)]
            sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
            result = validate_and_score(sol)
            if not result.valid or result.score >= global_best_ref.value:
                return False

            cx, cy = result.mec[0], result.mec[1]
            centered = [{'x': round(d['x']-cx,6), 'y': round(d['y']-cy,6), 'theta': round(d['theta'],6)} for d in raw]
            global_best_ref.value = result.score
            with open(BEST_FILE, 'w') as f:
                json.dump(centered, f, indent=2)

            # Update plot
            try:
                sol_v = [Semicircle(d['x'],d['y'],d['theta']) for d in centered]
                r_v = validate_and_score(sol_v)
                from src.semicircle_packing.visualization import plot_packing
                plot_packing(sol_v, r_v.mec, save_path='best_solution.png')
            except: pass

            return True
        finally:
            _fcntl.flock(lf, _fcntl.LOCK_UN)


# ─── Worker strategies ──────────────────────────────────────────────

STRATEGIES = [
    # (name, seed_fn, T_start, T_end, lam_start, lam_end, n_steps)
    # Worker 0: exploit current best
    ('best_tight',    lambda: seed_from_best(0.02), 0.15, 0.0003, 1000, 15000, 50_000_000),
    # Worker 1: 5-10 topology (new)
    ('topo_5_10',     lambda: seed_5_10(),          1.5,  0.001,  10,   3000,  50_000_000),
    # Worker 2: 3-5-7 three-shell (highest-priority new topology)
    ('topo_3_5_7',    lambda: seed_3_5_7(),         1.5,  0.001,  10,   3000,  50_000_000),
    # Worker 3: 2-5-8 tight outer ring
    ('topo_2_5_8',    lambda: seed_2_5_8(),         1.5,  0.001,  10,   3000,  50_000_000),
    # Worker 4: 3-4-8 variant three-shell
    ('topo_3_4_8',    lambda: seed_3_4_8(),         1.5,  0.001,  10,   3000,  50_000_000),
    # Worker 5: best with large perturbation (basin escape)
    ('best_large',    lambda: seed_from_best(0.20), 0.8,  0.001,  50,   3000,  50_000_000),
]


def worker(worker_id, global_best, result_queue, stop_event):
    os.nice(10)  # be gentle on thermals
    np.random.seed(worker_id * 1000 + int(time.time()) % 10000)
    random.seed(worker_id * 7919 + int(time.time()) % 10000)

    name, builder, T_start, T_end, lam_start, lam_end, n_steps = STRATEGIES[worker_id]
    run = 0

    while not stop_event.is_set():
        run += 1
        label = f"w{worker_id}_{name}_{run}"
        print(f"  [{label}] starting", flush=True)

        # Build starting configuration
        init = builder()
        if init is None:
            print(f"  [{label}] seed generation failed, retrying", flush=True)
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
            print(f"  [{label}] VALID: {val.score:.6f} (best: {global_best.value:.6f})", flush=True)
            result_queue.put((val.score, rx.tolist(), ry.tolist(), rt.tolist(), label))
        else:
            print(f"  [{label}] INVALID (official)", flush=True)

        # After exploitation strategies, re-read best to start from latest
        if 'best' in name:
            noise = 0.02 if 'tight' in name else 0.20
            STRATEGIES[worker_id] = (name, lambda n=noise: seed_from_best(n),
                                     T_start, T_end, lam_start, lam_end, n_steps)


def send_telegram(msg):
    try:
        import subprocess
        subprocess.Popen([
            'openclaw', 'message', 'send',
            '--channel', 'telegram',
            '--target', TELEGRAM_TARGET,
            '--message', msg
        ])
    except Exception as e:
        print(f"[telegram] failed: {e}", flush=True)


def sync_from_disk(global_best):
    try:
        with open(BEST_FILE_LOCK_PATH, 'w') as lf:
            _fcntl.flock(lf, _fcntl.LOCK_SH)
            try:
                with open(BEST_FILE) as f:
                    disk_raw = json.load(f)
            finally:
                _fcntl.flock(lf, _fcntl.LOCK_UN)
        disk_val = mod.official_validate(
            np.array([s['x'] for s in disk_raw]),
            np.array([s['y'] for s in disk_raw]),
            np.array([s['theta'] for s in disk_raw])
        )
        if disk_val.valid and disk_val.score < global_best.value:
            old = global_best.value
            global_best.value = disk_val.score
            print(f"[sync] disk improved: {old:.6f} → {disk_val.score:.6f}", flush=True)
    except Exception as e:
        print(f"[sync] error: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--runtime', type=int, default=3600, help='Max runtime in seconds')
    args = parser.parse_args()

    n_workers = min(args.workers, len(STRATEGIES))

    # Warm up Numba JIT before forking
    print("Warming up Numba JIT...", flush=True)
    try:
        from sa_v2 import sa_run_v2
        _xs = np.random.randn(N) * 2.0
        _ys = np.random.randn(N) * 2.0
        _ts = np.random.rand(N) * 2 * math.pi
        sa_run_v2(_xs, _ys, _ts, 1000, 0.1, 0.001, 500, 5000, 0)
        print("Numba JIT ready.", flush=True)
    except Exception as e:
        print(f"Numba warm-up failed: {e}", flush=True)

    _, _, _, initial_best = load_best()
    print(f"Starting best: {initial_best:.6f}")
    print(f"Launching {n_workers} workers for up to {args.runtime}s")

    # Log file
    log = open(LOG_FILE, 'a')
    log.write(f"\n{'='*60}\n{time.strftime('%Y-%m-%d %H:%M:%S')} optimize_v2 starting\n")
    log.write(f"Workers: {n_workers}, Runtime: {args.runtime}s, Initial best: {initial_best:.6f}\n")
    log.flush()

    global_best = mp.Value('d', initial_best)
    result_queue = mp.Queue()
    stop_event = mp.Event()

    workers = []
    for i in range(n_workers):
        p = mp.Process(target=worker, args=(i, global_best, result_queue, stop_event), daemon=True)
        p.start()
        workers.append(p)
        time.sleep(0.5)  # stagger starts

    t_start = time.time()
    last_disk_sync = time.time()

    try:
        while time.time() - t_start < args.runtime:
            time.sleep(10)

            # Drain result queue
            while not result_queue.empty():
                score, rx, ry, rt, label = result_queue.get_nowait()
                if score < global_best.value:
                    rx_a = np.array(rx); ry_a = np.array(ry); rt_a = np.array(rt)
                    improved = save_if_better(rx_a, ry_a, rt_a, score, global_best)
                    if improved:
                        msg = f"*** NEW BEST: {global_best.value:.6f} (from {label}) ***"
                        print(f"\n{msg}\n", flush=True)
                        log.write(f"{time.strftime('%H:%M:%S')} {msg}\n"); log.flush()
                        send_telegram(f"🎯 v2 new best: R={global_best.value:.6f} ({label})")

            # Periodic disk sync
            now = time.time()
            if now - last_disk_sync >= 30:
                sync_from_disk(global_best)
                last_disk_sync = now

            elapsed = now - t_start
            alive = sum(1 for p in workers if p.is_alive())
            status = f"[{elapsed:.0f}s] Best={global_best.value:.6f} Workers={alive}/{n_workers}"
            print(status, flush=True)
            if int(elapsed) % 300 < 15:  # log every ~5 min
                log.write(f"{time.strftime('%H:%M:%S')} {status}\n"); log.flush()

    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        stop_event.set()
        for p in workers:
            p.terminate()
            p.join(timeout=3)

    final = global_best.value
    msg = f"optimize_v2 done. Final best: {final:.6f} (started: {initial_best:.6f})"
    print(f"\n{msg}")
    log.write(f"\n{time.strftime('%H:%M:%S')} {msg}\n")
    log.close()

    if final < initial_best:
        send_telegram(f"✅ v2 optimizer finished. Improved: {initial_best:.6f} → {final:.6f}")
    else:
        send_telegram(f"📊 v2 optimizer finished. No improvement (best: {final:.6f})")


if __name__ == '__main__':
    mp.set_start_method('fork')
    main()
