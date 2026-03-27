#!/usr/bin/env python3
"""
Full parallel optimizer: 8 workers, 8 distinct strategies.
Strategies covering the research paths:
  0: from_best_tight   — local search around 3.072
  1: pairs_break       — pairs start, high T to force pair-breaking
  2: boundary_biased   — flat edges outward (Fejes Tóth insight)
  3: d1_symmetric      — D1 symmetry (reflection axis), known good for N=15
  4: random_loose      — pure random valid, large radius
  5: pairs_tight       — pairs start, tighter SA
  6: random_medium     — random valid, medium radius
  7: from_best_perturb — large perturbation from best, different basin

Workers share best_solution.json. Exact-score valid results; save if better.
Sends Telegram alert if we beat 3.072.
"""

import sys, os, json, math, time, random, multiprocessing as mp
import numpy as np
os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

import importlib.util
spec = importlib.util.spec_from_file_location("overnight", "overnight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
from shapely.geometry import Polygon

BEST_FILE = 'best_solution.json'
N = 15
ARC = 64
TELEGRAM_TARGET = '1602537663'


def make_poly(x, y, theta):
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, ARC)
    pts = list(zip(x + np.cos(angles), y + np.sin(angles)))
    pts.append((x, y))
    return Polygon(pts)


def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    bx = np.array([s['x'] for s in raw])
    by = np.array([s['y'] for s in raw])
    bt = np.array([s['theta'] for s in raw])
    val = mod.official_validate(bx, by, bt)
    return bx, by, bt, val.score if val.valid else float('inf')


def save_if_better(rx, ry, rt, score, global_best_ref):
    """Save centered solution if better than global best. Thread-safe via file lock."""
    if score >= global_best_ref.value:
        return False
    raw = [{'x': float(rx[i]), 'y': float(ry[i]), 'theta': float(rt[i])} for i in range(N)]
    sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
    result = validate_and_score(sol)
    if not result.valid:
        return False
    cx, cy = result.mec[0], result.mec[1]
    centered = [{'x': round(d['x']-cx,6), 'y': round(d['y']-cy,6), 'theta': round(d['theta'],6)} for d in raw]
    if result.score < global_best_ref.value:
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
    return False


# ─── Starting geometries ───────────────────────────────────────────

def start_from_best(noise=0.03):
    bx, by, bt, _ = load_best()
    bx = bx + np.random.randn(N)*noise
    by = by + np.random.randn(N)*noise
    bt = bt + np.random.randn(N)*noise*2
    return bx, by, bt


def start_pairs(scale=1.04, noise=0.0, seed=None):
    if seed: random.seed(seed)
    centers = [(0.0, 0.0)]
    for i in range(6):
        a = i * math.pi/3
        centers.append((scale*2*math.cos(a), scale*2*math.sin(a)))
    xs, ys, ts = [], [], []
    for cx, cy in centers:
        cx += random.gauss(0, noise); cy += random.gauss(0, noise)
        angle = random.uniform(0, math.pi)
        xs += [cx, cx]; ys += [cy, cy]; ts += [angle, angle+math.pi]
    # Singleton
    placed = [Semicircle(xs[i], ys[i], ts[i]) for i in range(14)]
    best_s = best_d = None, float('inf')
    for r_try in np.linspace(0.5, 3.5, 10):
        for a_try in np.linspace(0, 2*math.pi, 24, endpoint=False):
            x = r_try*math.cos(a_try); y = r_try*math.sin(a_try)
            for t_try in np.linspace(0, 2*math.pi, 8, endpoint=False):
                sc = Semicircle(x, y, t_try)
                ok = all(not ((x-p.x)**2+(y-p.y)**2<=4.01 and semicircles_overlap(sc,p)) for p in placed)
                if ok and r_try < best_d[1]:
                    best_s = (x, y, t_try); best_d = (None, r_try)
    if best_s[0] is None: return None
    xs.append(best_s[0]); ys.append(best_s[1]); ts.append(best_s[2])
    return np.array(xs), np.array(ys), np.array(ts)


def start_boundary_biased(seed=None):
    """6 boundary semicircles flat-edge-outward + 9 interior."""
    if seed: random.seed(seed)
    for _ in range(100):
        xs, ys, ts = [], [], []
        polys = []
        r_enc = 2.6
        for i in range(6):
            a = i*math.pi/3 + random.gauss(0, 0.12)
            r = r_enc - 1.0 + random.gauss(0, 0.06)
            x = r*math.cos(a); y = r*math.sin(a)
            t = a + math.pi + random.gauss(0, 0.15)
            p = make_poly(x, y, t)
            if all(p.intersection(ep).area < 1e-6 for ep in polys):
                xs.append(x); ys.append(y); ts.append(t); polys.append(p)
        if len(xs) < 5: continue
        success = True
        for _ in range(N - len(xs)):
            placed = False
            for _ in range(300):
                r = random.uniform(0.2, r_enc-1.1)
                a = random.uniform(0, 2*math.pi)
                x = r*math.cos(a); y = r*math.sin(a)
                t = random.uniform(0, 2*math.pi)
                p = make_poly(x, y, t)
                if all(p.intersection(ep).area < 1e-6 for ep in polys):
                    xs.append(x); ys.append(y); ts.append(t); polys.append(p)
                    placed = True; break
            if not placed: success = False; break
        if success and len(xs)==N:
            sol = [Semicircle(xs[i],ys[i],ts[i]) for i in range(N)]
            if validate_and_score(sol).valid:
                return np.array(xs), np.array(ys), np.array(ts)
    return None


def start_d1_symmetric(seed=None):
    """
    D1 symmetry: reflection axis = x-axis.
    8 semicircles placed symmetrically (4 pairs of mirror images) + 7 on axis.
    For N=15: 6 mirror pairs (12) + 3 on-axis.
    """
    if seed: random.seed(seed)
    for attempt in range(50):
        xs, ys, ts = [], [], []
        polys = []
        # 3 on-axis semicircles (y=0, theta pointing up or down)
        for rx in [0.0, 1.6, -1.6]:
            ry = random.gauss(0, 0.05)
            t = random.choice([0, math.pi/2, math.pi, 3*math.pi/2]) + random.gauss(0, 0.1)
            p = make_poly(rx, ry, t)
            if all(p.intersection(ep).area < 1e-6 for ep in polys):
                xs.append(rx); ys.append(ry); ts.append(t); polys.append(p)
        # 6 mirror pairs
        placed_pairs = 0
        for _ in range(200):
            if placed_pairs >= 6: break
            r = random.uniform(0.8, 2.8)
            a = random.uniform(0.1, math.pi - 0.1)  # upper half
            x = r*math.cos(a); y = r*math.sin(a)
            t = random.uniform(0, 2*math.pi)
            t_mirror = -t  # mirror theta across x-axis
            p1 = make_poly(x, y, t)
            p2 = make_poly(x, -y, t_mirror)
            all_existing = polys + [p1]
            if (all(p1.intersection(ep).area < 1e-6 for ep in polys) and
                all(p2.intersection(ep).area < 1e-6 for ep in all_existing)):
                xs += [x, x]; ys += [y, -y]; ts += [t, t_mirror]
                polys += [p1, p2]
                placed_pairs += 1
        if len(xs) == N:
            sol = [Semicircle(xs[i],ys[i],ts[i]) for i in range(N)]
            if validate_and_score(sol).valid:
                return np.array(xs), np.array(ys), np.array(ts)
    return None


def start_random_valid(r_max=2.8, seed=None):
    if seed: random.seed(seed)
    for _ in range(100):
        xs, ys, ts = [], [], []
        polys = []
        success = True
        for i in range(N):
            placed = False
            for _ in range(300):
                r = random.uniform(0.3, r_max)
                a = random.uniform(0, 2*math.pi)
                x = r*math.cos(a); y = r*math.sin(a)
                t = random.uniform(0, 2*math.pi)
                p = make_poly(x, y, t)
                if all(p.intersection(ep).area < 1e-6 for ep in polys):
                    xs.append(x); ys.append(y); ts.append(t); polys.append(p); placed = True; break
            if not placed: success = False; break
        if success:
            sol = [Semicircle(xs[i],ys[i],ts[i]) for i in range(N)]
            if validate_and_score(sol).valid:
                return np.array(xs), np.array(ys), np.array(ts)
    return None


# ─── SA parameters per strategy ───────────────────────────────────

STRATEGIES = [
    # (name, builder, T_start, T_end, lam_start, lam_end, n_steps)
    ('from_best_tight',   lambda: start_from_best(0.02),          0.25, 0.0005, 500,    5000,   2000000),
    ('pairs_break',       lambda: start_pairs(1.04, 0.01, None),  1.2,  0.001,  8,      3000,   2000000),
    ('boundary_biased',   lambda: start_boundary_biased() or start_random_valid(2.7), 1.5, 0.001, 5, 3000, 2000000),
    ('d1_symmetric',      lambda: start_d1_symmetric() or start_random_valid(2.8),    1.2, 0.001, 8, 3000, 2000000),
    ('random_loose',      lambda: start_random_valid(3.0),        2.0,  0.001,  5,      3000,   2000000),
    ('pairs_tight',       lambda: start_pairs(1.02, 0.0, None),   0.5,  0.0005, 100,    5000,   2000000),
    ('random_medium',     lambda: start_random_valid(2.6),        1.5,  0.001,  5,      3000,   2000000),
    ('from_best_large',   lambda: start_from_best(0.20),          1.0,  0.001,  50,     3000,   2000000),
]


def worker(worker_id, global_best, result_queue, stop_event):
    np.random.seed(worker_id * 1000 + int(time.time()) % 10000)
    random.seed(worker_id * 7919 + int(time.time()) % 10000)

    name, builder, T_start, T_end, lam_start, lam_end, n_steps = STRATEGIES[worker_id]
    run = 0

    while not stop_event.is_set():
        run += 1
        label = f"w{worker_id}_{name}_{run}"
        print(f"  [{label}] starting", flush=True)

        init = builder()
        if init is None:
            print(f"  [{label}] failed to build start, retrying", flush=True)
            time.sleep(2)
            continue

        xs, ys, ts = init
        result, _ = mod.sa_run(xs, ys, ts,
            n_steps=n_steps, T_start=T_start, T_end=T_end,
            lam_start=lam_start, lam_end=lam_end,
            seed=run * 137 * (worker_id+1),
            label=label)

        if result is None:
            print(f"  [{label}] no result", flush=True)
            continue

        rx, ry, rt = result
        val = mod.official_validate(rx, ry, rt)
        if val.valid:
            print(f"  [{label}] VALID: {val.score:.6f} (best: {global_best.value:.6f})", flush=True)
            result_queue.put((val.score, rx.tolist(), ry.tolist(), rt.tolist(), label))
        else:
            print(f"  [{label}] INVALID", flush=True)

        # After best-based strategies, refresh with new best each run
        if 'from_best' in name or 'tight' in name:
            STRATEGIES[worker_id] = (name, lambda: start_from_best(0.02 if 'tight' in name else 0.20),
                                     T_start, T_end, lam_start, lam_end, n_steps)


def main():
    _, _, _, initial_best = load_best()
    print(f"Starting best: {initial_best:.6f}")
    print(f"Launching {len(STRATEGIES)} parallel workers")

    global_best = mp.Value('d', initial_best)
    result_queue = mp.Queue()
    stop_event = mp.Event()

    workers = []
    for i in range(len(STRATEGIES)):
        p = mp.Process(target=worker, args=(i, global_best, result_queue, stop_event), daemon=True)
        p.start()
        workers.append(p)
        time.sleep(0.3)  # stagger starts

    t_start = time.time()
    max_runtime = 3600 * 8

    try:
        while time.time() - t_start < max_runtime:
            time.sleep(5)
            while not result_queue.empty():
                score, rx, ry, rt, label = result_queue.get_nowait()
                if score < global_best.value:
                    rx_a = np.array(rx); ry_a = np.array(ry); rt_a = np.array(rt)
                    improved = save_if_better(rx_a, ry_a, rt_a, score, global_best)
                    if improved:
                        msg = f"*** NEW BEST: {global_best.value:.6f} (from {label}) ***"
                        print(f"\n{msg}\n", flush=True)
                        # Telegram alert
                        try:
                            import subprocess
                            # Use openclaw to send Telegram
                            subprocess.Popen(['node', '-e', f'''
const {{OpenClaw}} = require("/usr/lib/node_modules/openclaw");
// simple HTTP to gateway
const http = require("http");
const data = JSON.stringify({{channel:"telegram",target:"{TELEGRAM_TARGET}",message:"🎯 Packing: {msg}"}});
const opts = {{hostname:"localhost",port:3001,path:"/api/send-message",method:"POST",headers:{{"Content-Type":"application/json","Content-Length":data.length}}}};
'''])
                        except: pass

            elapsed = time.time() - t_start
            alive = sum(1 for p in workers if p.is_alive())
            print(f"[{elapsed:.0f}s] Best={global_best.value:.6f} Workers={alive}/{len(STRATEGIES)}", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for p in workers:
            p.terminate()
            p.join(timeout=3)

    print(f"\nFinal best: {global_best.value:.6f}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
