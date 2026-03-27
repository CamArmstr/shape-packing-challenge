#!/usr/bin/env python3
"""
Background start-pool generator.
Runs continuously, pre-generating valid starting configurations
and writing them to start_pool.jsonl (one JSON per line).
Workers read from the pool instead of building starts inline.

Each entry: {"strategy": "...", "xs": [...], "ys": [...], "ts": [...], "score": 3.xx}
Pool capped at MAX_POOL entries; old entries cycled out.
"""

import sys, os, json, math, time, random
import numpy as np
from collections import deque
import fcntl

os.chdir('/home/camcore/.openclaw/workspace/shape-packing-challenge')
sys.path.insert(0, '.')

from src.semicircle_packing.scoring import validate_and_score
from src.semicircle_packing.geometry import Semicircle, semicircles_overlap
from shapely.geometry import Polygon

POOL_FILE = 'start_pool.jsonl'
MAX_POOL = 200
N = 15
ARC = 16  # fast for start generation; final validate_and_score uses exact


def make_poly_fast(x, y, theta):
    """16-pt Shapely polygon — fast enough for start generation."""
    angles = np.linspace(theta - math.pi/2, theta + math.pi/2, ARC)
    pts = list(zip(float(x) + np.cos(angles), float(y) + np.sin(angles)))
    pts.append((float(x), float(y)))
    return Polygon(pts)


def overlaps_any(x, y, t, placed):
    """
    Fast check using Shapely 16-pt polygons.
    placed = list of (x, y, t) tuples.
    """
    if not placed:
        return False
    p = make_poly_fast(x, y, t)
    for px, py, pt in placed:
        dx, dy = x - px, y - py
        if dx*dx + dy*dy >= 4.0:
            continue
        ep = make_poly_fast(px, py, pt)
        if p.intersection(ep).area > 1e-6:
            return True
    return False


def write_entry(entry):
    """Append entry to pool file, atomically."""
    line = json.dumps(entry) + '\n'
    with open(POOL_FILE, 'a') as f:
        f.write(line)


def count_pool():
    try:
        with open(POOL_FILE) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def trim_pool():
    """Keep only the last MAX_POOL entries."""
    try:
        with open(POOL_FILE) as f:
            lines = f.readlines()
        if len(lines) > MAX_POOL:
            with open(POOL_FILE, 'w') as f:
                f.writelines(lines[-MAX_POOL:])
    except: pass


def build_random_valid(r_max=2.8):
    for _ in range(30):
        xs, ys, ts = [], [], []
        placed = []  # list of (x, y, t) tuples
        ok = True
        for _ in range(N):
            p = False
            for _ in range(500):
                r = random.uniform(0.3, r_max)
                a = random.uniform(0, 2*math.pi)
                x, y = r*math.cos(a), r*math.sin(a)
                t = random.uniform(0, 2*math.pi)
                if not overlaps_any(x, y, t, placed):
                    xs.append(x); ys.append(y); ts.append(t)
                    placed.append((x, y, t)); p = True; break
            if not p: ok = False; break
        if ok:
            sol = [Semicircle(xs[i],ys[i],ts[i]) for i in range(N)]
            r = validate_and_score(sol)
            if r.valid:
                return xs, ys, ts, r.score
    return None


def build_boundary_biased():
    for _ in range(50):
        xs, ys, ts, placed = [], [], [], []
        r_enc = 2.6
        n_boundary = random.randint(5, 7)
        for i in range(n_boundary):
            a = i*(2*math.pi/n_boundary) + random.gauss(0, 0.15)
            r = r_enc - 1.0 + random.gauss(0, 0.08)
            x, y = r*math.cos(a), r*math.sin(a)
            t = a + math.pi + random.gauss(0, 0.2)
            if not overlaps_any(x, y, t, placed):
                xs.append(x); ys.append(y); ts.append(t); placed.append((x,y,t))
        if len(xs) < 4: continue
        ok = True
        for _ in range(N - len(xs)):
            p = False
            for _ in range(400):
                r = random.uniform(0.2, r_enc-1.0)
                a = random.uniform(0, 2*math.pi)
                x, y = r*math.cos(a), r*math.sin(a)
                t = random.uniform(0, 2*math.pi)
                if not overlaps_any(x, y, t, placed):
                    xs.append(x); ys.append(y); ts.append(t); placed.append((x,y,t)); p=True; break
            if not p: ok = False; break
        if ok and len(xs)==N:
            sol = [Semicircle(xs[i],ys[i],ts[i]) for i in range(N)]
            r = validate_and_score(sol)
            if r.valid:
                return xs, ys, ts, r.score
    return None


def build_d1_symmetric():
    for _ in range(80):
        xs, ys, ts, placed = [], [], [], []
        on_axis_x = [0.0, random.uniform(1.2, 2.0), random.uniform(-2.0, -1.2)]
        for ax in on_axis_x:
            ay = random.gauss(0, 0.04)
            t = random.choice([0, math.pi/2, math.pi, 3*math.pi/2]) + random.gauss(0, 0.12)
            if not overlaps_any(ax, ay, t, placed):
                xs.append(ax); ys.append(ay); ts.append(t); placed.append((ax,ay,t))
        pairs = 0
        for _ in range(400):
            if pairs >= 6: break
            r = random.uniform(0.8, 2.8)
            a = random.uniform(0.1, math.pi-0.1)
            x, y = r*math.cos(a), r*math.sin(a)
            t = random.uniform(0, 2*math.pi)
            t_m = (2*math.pi - t) % (2*math.pi)
            if (not overlaps_any(x, y, t, placed) and
                    not overlaps_any(x, -y, t_m, placed + [(x,y,t)])):
                xs += [x, x]; ys += [y, -y]; ts += [t, t_m]
                placed += [(x,y,t),(x,-y,t_m)]; pairs += 1
        if len(xs) == N:
            sol = [Semicircle(xs[i],ys[i],ts[i]) for i in range(N)]
            r = validate_and_score(sol)
            if r.valid:
                return xs, ys, ts, r.score
    return None


def build_pairs(scale=1.04, noise=0.02):
    centers = [(0.0, 0.0)]
    for i in range(6):
        a = i*math.pi/3
        centers.append((scale*2*math.cos(a)+random.gauss(0,noise),
                        scale*2*math.sin(a)+random.gauss(0,noise)))
    xs, ys, ts = [], [], []
    for cx, cy in centers:
        a = random.uniform(0, math.pi)
        xs += [cx, cx]; ys += [cy, cy]; ts += [a, a+math.pi]
    placed = [Semicircle(xs[i],ys[i],ts[i]) for i in range(14)]
    best = None; best_d = float('inf')
    for rr in np.linspace(0.3, 3.8, 12):
        for aa in np.linspace(0, 2*math.pi, 32, endpoint=False):
            x, y = rr*math.cos(aa), rr*math.sin(aa)
            for tt in np.linspace(0, 2*math.pi, 8, endpoint=False):
                sc = Semicircle(x, y, tt)
                if all(not((x-p.x)**2+(y-p.y)**2<=4.01 and semicircles_overlap(sc,p)) for p in placed):
                    if rr < best_d:
                        best = (x, y, tt); best_d = rr
    if best is None: return None
    xs.append(best[0]); ys.append(best[1]); ts.append(best[2])
    sol = [Semicircle(xs[i],ys[i],ts[i]) for i in range(N)]
    r = validate_and_score(sol)
    if r.valid: return xs, ys, ts, r.score
    return None


BUILDERS = [
    ('random_2.6',    lambda: build_random_valid(2.6)),
    ('random_2.8',    lambda: build_random_valid(2.8)),
    ('random_3.0',    lambda: build_random_valid(3.0)),
    ('boundary',      build_boundary_biased),
    ('d1_symmetric',  build_d1_symmetric),
    ('pairs_tight',   lambda: build_pairs(1.02, 0.01)),
    ('pairs_loose',   lambda: build_pairs(1.08, 0.04)),
    ('random_2.4',    lambda: build_random_valid(2.4)),
]

if __name__ == '__main__':
    print(f"Start generator running. Pool file: {POOL_FILE}", flush=True)
    # Clear stale pool
    if os.path.exists(POOL_FILE):
        os.remove(POOL_FILE)

    t0 = time.time()
    generated = 0
    failures = 0

    while True:
        current_count = count_pool()
        if current_count >= MAX_POOL:
            time.sleep(2)
            continue

        strategy, builder = random.choice(BUILDERS)
        t_build = time.time()
        result = builder()
        elapsed = time.time() - t_build

        if result:
            xs, ys, ts, score = result
            entry = {
                'strategy': strategy,
                'xs': [round(x, 6) for x in xs],
                'ys': [round(y, 6) for y in ys],
                'ts': [round(t, 6) for t in ts],
                'score': round(score, 6),
            }
            write_entry(entry)
            generated += 1
            rate = generated / (time.time() - t0)
            print(f"  [{strategy}] score={score:.4f} ({elapsed:.1f}s) pool={current_count+1} rate={rate:.2f}/s", flush=True)
            if generated % 20 == 0:
                trim_pool()
        else:
            failures += 1
            if failures % 10 == 0:
                print(f"  failures={failures}", flush=True)
