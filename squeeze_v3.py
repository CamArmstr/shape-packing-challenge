#!/usr/bin/env python3
"""
squeeze_v3.py — Focused exploitation: squeeze R below current best.

Strategy: MBH bilevel binary search.
1. Start from best known solution
2. Try R_target = best_R - delta
3. Minimize overlap penalty at fixed R_target using multiple restarts
4. If feasible: new best! Reduce R_target further.
5. If infeasible after many restarts: increase R_target (binary search)

Uses sa_v2 kernel with Thompson Sampling for the inner minimization.
6 workers, each independently binary-searching R from the best solution.

Key insight from the data: new topologies find R ≈ 3.06-3.13 but can't
reach the 3.010 basin. So we exploit the known basin harder instead.
"""

import sys, os, json, math, time, random, multiprocessing as mp, argparse
import numpy as np

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from sa_v2 import sa_run_v2, r_exact_nb, N_OPS
from gjk_numba import overlap_energy_gjk

import importlib.util
spec = importlib.util.spec_from_file_location("overnight", "overnight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle
from scipy.optimize import minimize as scipy_minimize
from phi import penalty_energy_flat, penalty_gradient_flat

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


def save_if_better(rx, ry, rt, global_best_ref):
    """Validate and save if better than current best."""
    sol = [Semicircle(float(rx[i]), float(ry[i]), float(rt[i])) for i in range(N)]
    result = validate_and_score(sol)
    if not result.valid:
        return False, float('inf')

    score = float(result.score)
    if score >= global_best_ref.value:
        return False, score

    with open(BEST_FILE_LOCK_PATH, 'w') as lf:
        try:
            _fcntl.flock(lf, _fcntl.LOCK_EX)

            # Re-check under lock
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
                return False, score

            cx, cy = result.mec[0], result.mec[1]
            centered = [{'x': round(float(rx[i])-cx, 6), 'y': round(float(ry[i])-cy, 6),
                         'theta': round(float(rt[i]), 6)} for i in range(N)]
            global_best_ref.value = score
            with open(BEST_FILE, 'w') as f:
                json.dump(centered, f, indent=2)

            try:
                sol_v = [Semicircle(d['x'], d['y'], d['theta']) for d in centered]
                r_v = validate_and_score(sol_v)
                from src.semicircle_packing.visualization import plot_packing
                plot_packing(sol_v, r_v.mec, save_path='best_solution.png')
            except:
                pass

            return True, score
        finally:
            _fcntl.flock(lf, _fcntl.LOCK_UN)


def pack(xs, ys, ts):
    p = np.zeros(3 * N)
    p[0::3] = xs; p[1::3] = ys; p[2::3] = ts
    return p


def unpack(p):
    return p[0::3].copy(), p[1::3].copy(), p[2::3].copy()


def lbfgs_refine(xs, ys, ts, R_target, max_iter=500, lam=1000.0):
    """L-BFGS-B: minimize overlap penalty at fixed R_target."""
    p0 = pack(xs, ys, ts)

    def f(p):
        return lam * penalty_energy_flat(p, R_target)

    def g(p):
        return lam * penalty_gradient_flat(p, R_target)

    result = scipy_minimize(f, p0, jac=g, method='L-BFGS-B',
                            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
    rxs, rys, rts = unpack(result.x)
    energy = overlap_energy_gjk(rxs, rys, rts)
    return rxs, rys, rts, energy


def perturb_small(xs, ys, ts, sigma=0.05):
    """Small perturbation for MBH restarts."""
    return (xs + np.random.randn(N) * sigma,
            ys + np.random.randn(N) * sigma,
            ts + np.random.randn(N) * sigma * 2)


def perturb_medium(xs, ys, ts, sigma=0.15):
    """Medium perturbation: also flip 1-2 orientations."""
    xs = xs + np.random.randn(N) * sigma
    ys = ys + np.random.randn(N) * sigma
    ts = ts + np.random.randn(N) * sigma * 2
    # Flip 1-2 random orientations by π
    n_flip = random.choice([1, 2])
    for idx in random.sample(range(N), n_flip):
        ts[idx] = (ts[idx] + math.pi) % (2 * math.pi)
    return xs, ys, ts


def sa_feasibility_solve(xs, ys, ts, R_target, n_steps=5_000_000, seed=42):
    """
    SA targeting feasibility at R_target.
    Uses high lambda (strong overlap penalty) and moderate temperature.
    """
    # The trick: set lam very high so SA focuses on eliminating overlaps
    bx, by, bt, br, found, _, _ = sa_run_v2(
        xs, ys, ts, n_steps,
        T_start=0.3, T_end=0.0001,
        lam_start=5000.0, lam_end=50000.0,
        seed=seed
    )
    if not found:
        return None, None, None, float('inf')

    # Check if it fits in R_target
    r = r_exact_nb(bx, by, bt)
    if r <= R_target + 0.001:
        # Refine with L-BFGS
        rx, ry, rt, energy = lbfgs_refine(bx, by, bt, R_target, max_iter=1000, lam=5000)
        return rx, ry, rt, energy
    return None, None, None, float('inf')


def worker(worker_id, global_best, result_queue, stop_event):
    os.nice(10)
    np.random.seed(worker_id * 1000 + int(time.time()) % 10000)
    random.seed(worker_id * 7919 + int(time.time()) % 10000)

    # Load best
    bx, by, bt, best_R = load_best()

    # Binary search parameters
    R_lo = 2.90  # theoretical lower bound ~2.74, but 2.90 is realistic
    R_hi = best_R
    R_target = best_R - 0.005  # start slightly below best

    round_num = 0
    n_restarts_per_R = 15  # how many perturbations to try at each R

    while not stop_event.is_set():
        round_num += 1

        # Re-read best periodically
        if round_num % 5 == 0:
            try:
                bx, by, bt, current_best = load_best()
                if current_best < R_hi:
                    R_hi = current_best
                    R_target = R_hi - 0.005
            except:
                pass

        label = f"w{worker_id}_r{round_num}"
        print(f"  [{label}] R_target={R_target:.6f} (lo={R_lo:.4f} hi={R_hi:.6f})", flush=True)

        feasible_found = False

        for restart in range(n_restarts_per_R):
            if stop_event.is_set():
                break

            # Perturb from best
            if restart < 5:
                px, py, pt = perturb_small(bx, by, bt, sigma=0.03 + restart * 0.01)
            elif restart < 10:
                px, py, pt = perturb_medium(bx, by, bt, sigma=0.08)
            else:
                px, py, pt = perturb_medium(bx, by, bt, sigma=0.15)

            # L-BFGS at fixed R_target
            rx, ry, rt, energy = lbfgs_refine(px, py, pt, R_target, max_iter=800, lam=2000)

            if energy < 1e-6:
                # GJK confirms feasible at R_target
                gjk_energy = overlap_energy_gjk(rx, ry, rt)
                r_actual = r_exact_nb(rx, ry, rt)

                if gjk_energy < 1e-8 and r_actual <= R_target + 0.001:
                    # Official validation
                    improved, score = save_if_better(rx, ry, rt, global_best)
                    if improved:
                        result_queue.put(('NEW_BEST', score, label))
                        bx, by, bt = rx.copy(), ry.copy(), rt.copy()
                        R_hi = score
                        feasible_found = True
                        print(f"  [{label}] ★ NEW BEST: R={score:.6f}", flush=True)
                        break
                    elif score < float('inf'):
                        # Valid but not better — still useful, narrows search
                        R_hi = min(R_hi, score)
                        feasible_found = True
                        print(f"  [{label}] valid R={score:.6f} (not improvement)", flush=True)

            elif energy < 1e-3:
                # Nearly feasible — try SA to resolve remaining overlaps
                seed = round_num * 100 + restart
                sx, sy, st, se = sa_feasibility_solve(rx, ry, rt, R_target,
                                                       n_steps=2_000_000, seed=seed)
                if sx is not None and se < 1e-8:
                    improved, score = save_if_better(sx, sy, st, global_best)
                    if improved:
                        result_queue.put(('NEW_BEST', score, label))
                        bx, by, bt = sx.copy(), sy.copy(), st.copy()
                        R_hi = score
                        feasible_found = True
                        print(f"  [{label}] ★ SA-repaired NEW BEST: R={score:.6f}", flush=True)
                        break

        # Binary search update
        if feasible_found:
            R_target = (R_lo + R_hi) / 2  # bisect
        else:
            R_lo = R_target  # can't go this low
            R_target = (R_lo + R_hi) / 2

        if R_hi - R_lo < 0.001:
            print(f"  [{label}] converged: R ∈ [{R_lo:.6f}, {R_hi:.6f}]", flush=True)
            # Reset and try again with wider bounds
            R_lo = max(2.90, R_hi - 0.05)
            R_target = (R_lo + R_hi) / 2


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
    parser.add_argument('--runtime', type=int, default=3600)
    args = parser.parse_args()

    # Warm up Numba
    print("Warming up JIT...", flush=True)
    from sa_v2 import sa_run_v2 as _warmup
    _xs = np.random.randn(N) * 2.0
    _ys = np.random.randn(N) * 2.0
    _ts = np.random.rand(N) * 2 * math.pi
    _warmup(_xs, _ys, _ts, 1000, 0.1, 0.001, 500, 5000, 0)
    overlap_energy_gjk(_xs, _ys, _ts)
    print("JIT ready.", flush=True)

    _, _, _, initial_best = load_best()
    print(f"Starting best: {initial_best:.6f}")
    print(f"Launching {args.workers} squeeze workers for {args.runtime}s")

    global_best = mp.Value('d', initial_best)
    result_queue = mp.Queue()
    stop_event = mp.Event()

    workers = []
    for i in range(args.workers):
        p = mp.Process(target=worker, args=(i, global_best, result_queue, stop_event), daemon=True)
        p.start()
        workers.append(p)
        time.sleep(0.3)

    t_start = time.time()

    try:
        while time.time() - t_start < args.runtime:
            time.sleep(10)

            while not result_queue.empty():
                msg_type, score, label = result_queue.get_nowait()
                if msg_type == 'NEW_BEST':
                    print(f"\n*** SQUEEZE NEW BEST: {score:.6f} ({label}) ***\n", flush=True)
                    send_telegram(f"🎯 Squeeze new best: R={score:.6f} ({label})")

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
    print(f"\nSqueeze done. Final: {final:.6f} (started: {initial_best:.6f})")
    if final < initial_best:
        send_telegram(f"✅ Squeeze improved: {initial_best:.6f} → {final:.6f}")


if __name__ == '__main__':
    mp.set_start_method('fork')
    main()
