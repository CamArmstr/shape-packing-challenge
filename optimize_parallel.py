#!/usr/bin/env python3
"""
Multi-process parallel SA: spawn N_WORKERS independent overnight.py SA runs.
Each worker uses a different random seed and starting config.
Workers write to shared best_solution.json via file lock.
"""

import sys, os, json, time, math, multiprocessing as mp
import numpy as np

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

import importlib.util
spec = importlib.util.spec_from_file_location("overnight", "overnight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

BEST_FILE = 'best_solution.json'
N_WORKERS = 6
STEPS_PER_RUN = 2000000

def load_best_score():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    bx = np.array([s['x'] for s in raw])
    by = np.array([s['y'] for s in raw])
    bt = np.array([s['theta'] for s in raw])
    val = mod.official_validate(bx, by, bt)
    return bx, by, bt, val.score if val.valid else float('inf')


def worker(worker_id, result_queue, stop_event):
    """Independent SA worker. Runs continuously until stop_event set."""
    np.random.seed(worker_id * 1000 + int(time.time()) % 1000)
    
    # Starting configs to cycle through
    starts = [
        ('best', None),
        ('ring_3.5', mod.ring_config(r=3.5)),
        ('ring_3.0', mod.ring_config(r=3.0)),
        ('double', mod.double_ring()),
        ('random', mod.random_config(seed=worker_id * 37)),
        ('best', None),
        ('ring_4.0', mod.ring_config(r=4.0)),
        ('random', mod.random_config(seed=worker_id * 73 + 100)),
    ]
    
    run_num = 0
    local_best = float('inf')
    
    while not stop_event.is_set():
        label, config = starts[run_num % len(starts)]
        run_num += 1
        
        if label == 'best' or config is None:
            bx, by, bt, score = load_best_score()
            xs, ys, ts = bx.copy(), by.copy(), bt.copy()
            # Add small noise to escape local minimum
            xs += np.random.randn(len(xs)) * 0.05
            ys += np.random.randn(len(ys)) * 0.05
            ts += np.random.randn(len(ts)) * 0.1
            label = f'worker{worker_id}_from_best'
        else:
            xs, ys, ts = config
            label = f'worker{worker_id}_{label}'
        
        T_start = 0.8 if 'best' in label else 2.0
        
        result, best_r = mod.sa_run(
            xs, ys, ts,
            n_steps=STEPS_PER_RUN,
            T_start=T_start, T_end=0.001,
            lam_start=50.0, lam_end=3000.0,
            seed=worker_id * 1337 + run_num * 31,
            label=label
        )
        
        if result is not None:
            rx, ry, rt = result
            val = mod.official_validate(rx, ry, rt)
            if val.valid:
                result_queue.put((val.score, rx.tolist(), ry.tolist(), rt.tolist(), label))
                if val.score < local_best:
                    local_best = val.score
                    print(f"  [Worker {worker_id}] Found: {val.score:.6f}", flush=True)


def main():
    bx, by, bt, initial_best = load_best_score()
    print(f"Starting best: {initial_best:.6f}")
    print(f"Launching {N_WORKERS} parallel workers on {mp.cpu_count()} cores")
    
    result_queue = mp.Queue()
    stop_event = mp.Event()
    global_best = initial_best
    
    workers = []
    for i in range(N_WORKERS):
        p = mp.Process(target=worker, args=(i, result_queue, stop_event), daemon=True)
        p.start()
        workers.append(p)
    
    t_start = time.time()
    max_runtime = 3600  # 1 hour
    
    try:
        while time.time() - t_start < max_runtime:
            time.sleep(5)
            # Drain result queue
            while not result_queue.empty():
                score, rx, ry, rt, label = result_queue.get_nowait()
                if score < global_best:
                    global_best = score
                    # Center and save
                    rx_a = np.array(rx); ry_a = np.array(ry); rt_a = np.array(rt)
                    mod.save_solution(rx_a, ry_a, rt_a, BEST_FILE)
                    mod.save_solution(rx_a, ry_a, rt_a, 'solution.json')
                    # Also save visualization
                    try:
                        from src.semicircle_packing.geometry import Semicircle
                        from src.semicircle_packing.scoring import validate_and_score
                        from src.semicircle_packing.visualization import plot_packing
                        sol = [Semicircle(rx[j], ry[j], rt[j]) for j in range(len(rx))]
                        r_v = validate_and_score(sol)
                        plot_packing(sol, r_v.mec, save_path='best_solution.png')
                    except:
                        pass
                    print(f"\n*** GLOBAL NEW BEST: {global_best:.6f} (from {label}) ***\n", flush=True)
            
            elapsed = time.time() - t_start
            alive = sum(1 for p in workers if p.is_alive())
            print(f"[{elapsed:.0f}s] Best={global_best:.6f} Workers={alive}/{N_WORKERS}", flush=True)
    
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for p in workers:
            p.terminate()
            p.join(timeout=5)
    
    print(f"\nFinal best: {global_best:.6f}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
