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

# Numba SA — import and warm up JIT at module level so workers inherit compiled cache
try:
    from sa_numba import sa_run_numba_wrapper as _sa_numba
    USE_NUMBA = True
except Exception as _e:
    USE_NUMBA = False
    print(f"[warn] Numba SA unavailable: {_e}; falling back to Shapely SA", flush=True)

import importlib.util
spec = importlib.util.spec_from_file_location("overnight", "overnight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
from shapely.geometry import Polygon

BEST_FILE = 'best_solution.json'
POOL_FILE = 'start_pool.jsonl'
N = 15
ARC = 64
TELEGRAM_TARGET = '1602537663'


def make_poly(x, y, theta):
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, ARC)
    pts = list(zip(x + np.cos(angles), y + np.sin(angles)))
    pts.append((x, y))
    return Polygon(pts)


def pop_from_pool(preferred_strategy=None):
    """Pop one entry from the start pool. Returns (xs, ys, ts) or None."""
    if not os.path.exists(POOL_FILE):
        return None
    try:
        with open(POOL_FILE, 'r+') as f:
            lines = f.readlines()
            if not lines:
                return None
            # Pick matching strategy if possible, else any
            candidates = [i for i, l in enumerate(lines)
                          if preferred_strategy and preferred_strategy in l]
            idx = candidates[0] if candidates else 0
            entry = json.loads(lines[idx])
            remaining = [l for i, l in enumerate(lines) if i != idx]
            f.seek(0); f.writelines(remaining); f.truncate()
        return np.array(entry['xs']), np.array(entry['ys']), np.array(entry['ts'])
    except:
        return None


def load_best():
    with open(BEST_FILE) as f:
        raw = json.load(f)
    bx = np.array([s['x'] for s in raw])
    by = np.array([s['y'] for s in raw])
    bt = np.array([s['theta'] for s in raw])
    val = mod.official_validate(bx, by, bt)
    return bx, by, bt, val.score if val.valid else float('inf')


import fcntl as _fcntl
BEST_FILE_LOCK_PATH = BEST_FILE + '.lock'

def save_if_better(rx, ry, rt, score, global_best_ref):
    """Save centered solution if better than global best.
    
    Uses fcntl file locking so this coordinates with mbh.py/pbh.py (separate processes).
    Re-reads disk value under lock so external writes are respected before deciding to save.
    """
    if score >= global_best_ref.value:
        return False

    with open(BEST_FILE_LOCK_PATH, 'w') as lf:
        try:
            _fcntl.flock(lf, _fcntl.LOCK_EX)

            # Re-read disk under lock — catch improvements from external processes
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
                    print(f"[sync] external best R={disk_R:.6f} loaded from disk", flush=True)
            except Exception:
                disk_R = float('inf')

            if score >= global_best_ref.value:
                return False  # worse than disk after sync

            raw = [{'x': float(rx[i]), 'y': float(ry[i]), 'theta': float(rt[i])} for i in range(N)]
            sol = [Semicircle(d['x'], d['y'], d['theta']) for d in raw]
            result = validate_and_score(sol)
            if not result.valid:
                return False
            if result.score >= global_best_ref.value:
                return False

            cx, cy = result.mec[0], result.mec[1]
            centered = [{'x': round(d['x']-cx,6), 'y': round(d['y']-cy,6), 'theta': round(d['theta'],6)} for d in raw]
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


def start_double_lattice(seed=None):
    """
    Double-lattice start: two semicircles per cell, rotated ~60-90° relative to each other
    (NOT flat-to-flat conjugates). Based on Kallus pessimal packing theory —
    optimal non-centrally-symmetric packing uses non-conjugate double-lattice cells.
    Try hexagonal arrangement of 7-8 double-lattice cells.
    """
    if seed: random.seed(seed)
    # Cell relative angle: try 60°, 90°, 120° (not 180° = conjugate)
    cell_angle = random.choice([math.pi/3, math.pi/2, 2*math.pi/3, math.pi*0.4])
    scale = random.uniform(1.05, 1.15)

    centers = [(0.0, 0.0)]
    for i in range(6):
        a = i * math.pi/3
        centers.append((scale*2*math.cos(a), scale*2*math.sin(a)))

    xs, ys, ts = [], [], []
    placed = []

    for cx, cy in centers[:7]:  # 7 cells × 2 = 14
        cx += random.gauss(0, 0.05); cy += random.gauss(0, 0.05)
        base_angle = random.uniform(0, 2*math.pi)
        t1 = base_angle
        t2 = base_angle + cell_angle  # NOT +pi, that's conjugate
        if not overlaps_any(cx, cy, t1, placed):
            xs.append(cx); ys.append(cy); ts.append(t1)
            placed.append((cx, cy, t1))
        if not overlaps_any(cx, cy, t2, placed):
            xs.append(cx); ys.append(cy); ts.append(t2)
            placed.append((cx, cy, t2))

    # Fill to 15
    for _ in range(N - len(xs)):
        for _ in range(200):
            r = random.uniform(0.3, 3.0)
            a = random.uniform(0, 2*math.pi)
            x, y = r*math.cos(a), r*math.sin(a)
            t = random.uniform(0, 2*math.pi)
            if not overlaps_any(x, y, t, placed):
                xs.append(x); ys.append(y); ts.append(t)
                placed.append((x, y, t)); break

    if len(xs) == N:
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


def start_three_shell(seed=None):
    """
    3-shell start: ~3 inner (r~0.5-0.9), ~5 mid (r~1.2-1.6), ~7 outer (r~1.9-2.2).
    Motivated by: current best has 0 mid-shell semicircles — a different topology
    may have a lower enclosing radius.
    Orientations: inner arcs point outward, mid arcs tangential, outer arcs inward.
    """
    if seed is not None:
        random.seed(seed)
    for attempt in range(200):
        xs, ys, ts = [], [], []
        polys = []

        # Inner shell: 3 semicircles, arcs pointing outward (toward boundary)
        inner_n = 3
        for i in range(inner_n):
            placed = False
            for _ in range(200):
                r = random.uniform(0.3, 0.9)
                a = random.uniform(0, 2 * math.pi)
                x, y = r * math.cos(a), r * math.sin(a)
                t = a + random.gauss(0, 0.3)  # arc mostly outward
                p = make_poly(x, y, t)
                if all(p.intersection(ep).area < 1e-6 for ep in polys):
                    xs.append(x); ys.append(y); ts.append(t)
                    polys.append(p); placed = True; break
            if not placed:
                break
        if len(xs) < inner_n:
            continue

        # Mid shell: 5 semicircles, arcs tangential (~90° from radial)
        mid_n = 5
        for i in range(mid_n):
            placed = False
            for _ in range(300):
                r = random.uniform(1.1, 1.6)
                a = random.uniform(0, 2 * math.pi)
                x, y = r * math.cos(a), r * math.sin(a)
                t = a + math.pi / 2 + random.gauss(0, 0.4)  # tangential
                p = make_poly(x, y, t)
                if all(p.intersection(ep).area < 1e-6 for ep in polys):
                    xs.append(x); ys.append(y); ts.append(t)
                    polys.append(p); placed = True; break
            if not placed:
                break
        if len(xs) < inner_n + mid_n:
            continue

        # Outer shell: 7 semicircles, arcs pointing inward
        outer_n = N - inner_n - mid_n  # 7
        for i in range(outer_n):
            placed = False
            for _ in range(400):
                r = random.uniform(1.8, 2.3)
                a = random.uniform(0, 2 * math.pi)
                x, y = r * math.cos(a), r * math.sin(a)
                t = a + math.pi + random.gauss(0, 0.3)  # arc inward
                p = make_poly(x, y, t)
                if all(p.intersection(ep).area < 1e-6 for ep in polys):
                    xs.append(x); ys.append(y); ts.append(t)
                    polys.append(p); placed = True; break
            if not placed:
                break
        if len(xs) == N:
            sol = [Semicircle(xs[i], ys[i], ts[i]) for i in range(N)]
            if validate_and_score(sol).valid:
                return np.array(xs), np.array(ys), np.array(ts)
    return None


def start_with_conjugate_pairs(seed=None):
    """
    Seed with explicit conjugate pairs: two semicircles with antiparallel normals,
    nested flat-to-flat (d < 2.0, n_dot ≈ -1). Theory suggests optimal packing
    for non-centrally-symmetric shapes uses conjugate pairs.
    Try 5 conjugate pairs (10 semicircles) + 5 singles.
    """
    if seed is not None:
        random.seed(seed)
    for attempt in range(200):
        xs, ys, ts = [], [], []
        polys = []

        # Place 5 conjugate pairs
        n_pairs = 5
        placed_pairs = 0
        for _ in range(500):
            if placed_pairs >= n_pairs:
                break
            r = random.uniform(0.5, 2.0)
            a = random.uniform(0, 2 * math.pi)
            cx, cy = r * math.cos(a), r * math.sin(a)
            t1 = random.uniform(0, 2 * math.pi)
            t2 = (t1 + math.pi) % (2 * math.pi)  # antiparallel = conjugate
            # Offset the pair slightly so they don't exactly overlap
            offset = random.uniform(0.05, 0.3)
            oa = random.uniform(0, 2 * math.pi)
            x1 = cx + offset * math.cos(oa)
            y1 = cy + offset * math.sin(oa)
            x2 = cx - offset * math.cos(oa)
            y2 = cy - offset * math.sin(oa)
            p1 = make_poly(x1, y1, t1)
            p2 = make_poly(x2, y2, t2)
            if (all(p1.intersection(ep).area < 1e-6 for ep in polys) and
                    p1.intersection(p2).area < 1e-6 and
                    all(p2.intersection(ep).area < 1e-6 for ep in polys + [p1])):
                xs += [x1, x2]; ys += [y1, y2]; ts += [t1, t2]
                polys += [p1, p2]
                placed_pairs += 1

        if placed_pairs < 3:
            continue

        # Fill remaining with random valid
        for _ in range(N - len(xs)):
            placed = False
            for _ in range(300):
                r = random.uniform(0.3, 2.5)
                a = random.uniform(0, 2 * math.pi)
                x, y = r * math.cos(a), r * math.sin(a)
                t = random.uniform(0, 2 * math.pi)
                p = make_poly(x, y, t)
                if all(p.intersection(ep).area < 1e-6 for ep in polys):
                    xs.append(x); ys.append(y); ts.append(t)
                    polys.append(p); placed = True; break
            if not placed:
                break

        if len(xs) == N:
            sol = [Semicircle(xs[i], ys[i], ts[i]) for i in range(N)]
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
    # 12 workers on 16-core machine.
    # Mix: tight exploitation, cluster-move exploitation, 3-shell, conjugate-pair basin.

    # --- Tight exploitation near best (standard single-particle moves) ---
    ('from_best_tight_0', lambda: start_from_best(0.01),  0.10, 0.0002, 2000, 20000, 50_000_000),
    ('from_best_tight_1', lambda: start_from_best(0.02),  0.15, 0.0003, 1000, 15000, 50_000_000),
    ('from_best_tight_2', lambda: start_from_best(0.03),  0.20, 0.0004,  800, 12000, 50_000_000),

    # --- Cluster-move workers: same starts, higher cluster_prob ---
    # Flagged with 'cluster' in name so worker can pass cluster_prob=0.30
    ('from_best_cluster_0', lambda: start_from_best(0.02), 0.15, 0.0003, 1000, 15000, 50_000_000),
    ('from_best_cluster_1', lambda: start_from_best(0.05), 0.25, 0.0005,  600, 10000, 50_000_000),
    ('from_best_cluster_2', lambda: start_from_best(0.10), 0.35, 0.0006,  400,  8000, 50_000_000),

    # --- Medium perturbation ---
    ('from_best_med_0',   lambda: start_from_best(0.08),  0.40, 0.0008,  300,  8000, 50_000_000),
    ('from_best_med_1',   lambda: start_from_best(0.15),  0.55, 0.001,   200,  6000, 50_000_000),

    # --- New topologies ---
    ('three_shell_0',      lambda: start_three_shell() or start_random_valid(2.8),          1.5, 0.001, 10, 3000, 50_000_000),
    ('three_shell_1',      lambda: start_three_shell() or start_random_valid(2.8),          1.5, 0.001, 10, 3000, 50_000_000),
    ('conjugate_pairs_0',  lambda: start_with_conjugate_pairs() or start_random_valid(2.8), 1.5, 0.001, 10, 3000, 50_000_000),

    # --- Large perturbation for basin escape ---
    ('from_best_large_0', lambda: start_from_best(0.30),  1.0,  0.001,   50,  3000, 50_000_000),
]


def worker(worker_id, global_best, result_queue, stop_event):
    os.nice(5)  # yield to higher-priority system tasks
    np.random.seed(worker_id * 1000 + int(time.time()) % 10000)
    random.seed(worker_id * 7919 + int(time.time()) % 10000)

    name, builder, T_start, T_end, lam_start, lam_end, n_steps = STRATEGIES[worker_id]
    run = 0

    while not stop_event.is_set():
        run += 1
        label = f"w{worker_id}_{name}_{run}"
        print(f"  [{label}] starting", flush=True)

        # Try pool first, fall back to inline builder
        pool_result = pop_from_pool(name)
        if pool_result is not None:
            xs, ys, ts = pool_result
            print(f"  [{label}] using pool start", flush=True)
        else:
            init = builder()
            if init is None:
                print(f"  [{label}] failed to build start, retrying", flush=True)
                time.sleep(1)
                continue
            xs, ys, ts = init
        run_seed = run * 137 * (worker_id + 1)
        cluster_prob = 0.30 if 'cluster' in name else 0.15
        if USE_NUMBA:
            rx, ry, rt, r_fast = _sa_numba(
                xs, ys, ts,
                n_steps=n_steps, T_start=T_start, T_end=T_end,
                lam_start=lam_start, lam_end=lam_end,
                seed=run_seed, cluster_prob=cluster_prob)
            if rx is None:
                print(f"  [{label}] no feasible result", flush=True)
                continue
            # Official exact validation (Shapely) — only when numba found something
            val = mod.official_validate(rx, ry, rt)
        else:
            result, _ = mod.sa_run(xs, ys, ts,
                n_steps=n_steps, T_start=T_start, T_end=T_end,
                lam_start=lam_start, lam_end=lam_end,
                seed=run_seed, label=label)
            if result is None:
                print(f"  [{label}] no result", flush=True)
                continue
            rx, ry, rt = result
            val = mod.official_validate(rx, ry, rt)
        mode = "numba" if USE_NUMBA else "shapely"
        if val.valid:
            print(f"  [{label}|{mode}] VALID: {val.score:.6f} (best: {global_best.value:.6f})", flush=True)
            result_queue.put((val.score, rx.tolist(), ry.tolist(), rt.tolist(), label))
        else:
            print(f"  [{label}|{mode}] INVALID (official)", flush=True)

        # After best-based strategies, rebuild the lambda so it re-reads best_solution.json
        # each run rather than capturing the stale array from startup
        if 'from_best' in name:
            noise = (0.01 if 'tight' in name else
                     0.10 if 'med' in name else 0.30)
            STRATEGIES[worker_id] = (name, lambda n=noise: start_from_best(n),
                                     T_start, T_end, lam_start, lam_end, n_steps)


def send_telegram(msg):
    """Send Telegram alert via openclaw CLI."""
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
    """Check best_solution.json for improvements from external processes (e.g. pbh.py)."""
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
            msg = f"[sync] disk improved: {old:.6f} → {disk_val.score:.6f}"
            print(f"\n{msg}\n", flush=True)
            send_telegram(f"🎯 Packing external improvement: R={disk_val.score:.6f}")
    except Exception as e:
        print(f"[sync] error: {e}", flush=True)


def main():
    # Warm up Numba JIT before forking — workers inherit compiled functions
    if USE_NUMBA:
        import json as _json
        print("Warming up Numba JIT...", flush=True)
        try:
            _raw = _json.load(open(BEST_FILE))
            _xs = np.array([s['x'] for s in _raw]) + np.random.randn(N)*0.05
            _ys = np.array([s['y'] for s in _raw]) + np.random.randn(N)*0.05
            _ts = np.array([s['theta'] for s in _raw]) + np.random.randn(N)*0.05
            _sa_numba(_xs, _ys, _ts, n_steps=10_000, T_start=0.1, T_end=0.001,
                      lam_start=500, lam_end=5000, seed=0, shapely_check_interval=5000)
            print("Numba JIT ready.", flush=True)
        except Exception as e:
            print(f"Numba warm-up failed: {e}", flush=True)

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
    last_disk_sync = time.time()
    DISK_SYNC_INTERVAL = 30  # check disk every 30s for external improvements

    try:
        while time.time() - t_start < max_runtime:
            time.sleep(5)

            # Drain result queue
            while not result_queue.empty():
                score, rx, ry, rt, label = result_queue.get_nowait()
                if score < global_best.value:
                    rx_a = np.array(rx); ry_a = np.array(ry); rt_a = np.array(rt)
                    improved = save_if_better(rx_a, ry_a, rt_a, score, global_best)
                    if improved:
                        msg = f"*** NEW BEST: {global_best.value:.6f} (from {label}) ***"
                        print(f"\n{msg}\n", flush=True)
                        send_telegram(f"🎯 Packing new best: R={global_best.value:.6f} (worker: {label})")

            # Periodic disk sync — catch improvements from pbh.py and other external processes
            now = time.time()
            if now - last_disk_sync >= DISK_SYNC_INTERVAL:
                sync_from_disk(global_best)
                last_disk_sync = now

            elapsed = now - t_start
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
    # Use 'fork' so worker processes inherit the parent's JIT-compiled Numba
    # functions. 'spawn' causes Numba cache misses and __mp_main__ errors.
    mp.set_start_method('fork')
    main()
