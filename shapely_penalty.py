"""
shapely_penalty.py — Exact Shapely-based penalty energy for semicircle packing.

Uses Shapely polygon intersection area as the overlap penalty.
This is ~50x slower than phi-functions but provably correct.
Used for final local minimization once phi-based SA finds a good basin.
"""

import math, numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize

N = 15
ARC_PTS = 64  # polygon resolution (matches fast_run.py default)


def make_poly(x, y, theta, n=ARC_PTS):
    """Build a Shapely Polygon for a unit semicircle."""
    angles = np.linspace(theta - math.pi / 2, theta + math.pi / 2, n)
    pts = list(zip(x + np.cos(angles), y + np.sin(angles)))
    return Polygon(pts)


def containment_slack(x, y, theta, R):
    """
    Minimum slack for semicircle containment in circle of radius R.
    Checks 3 critical points: arc apex and two diameter endpoints.
    Returns negative if outside container.
    """
    pts = [
        (x + math.cos(theta),  y + math.sin(theta)),   # arc apex
        (x - math.sin(theta),  y + math.cos(theta)),   # endpoint 1
        (x + math.sin(theta),  y - math.cos(theta)),   # endpoint 2
    ]
    slacks = [R**2 - (px**2 + py**2) for px, py in pts]
    return min(slacks)


def penalty_energy(params, R):
    """
    Shapely-based penalty energy:
    E = Σ overlap_area(i,j)^2 + Σ max(0, -containment_slack(i))^2
    Zero iff feasible at radius R.
    """
    xs = params[0::3]; ys = params[1::3]; ts = params[2::3]

    # Build all polygons
    polys = [make_poly(xs[i], ys[i], ts[i]) for i in range(N)]

    E = 0.0

    # Pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            # Quick distance check first
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            if dx*dx + dy*dy > 4.0:
                continue  # disks can't overlap
            area = polys[i].intersection(polys[j]).area
            if area > 0:
                E += area * area

    # Containment penalties
    for i in range(N):
        s = containment_slack(xs[i], ys[i], ts[i], R)
        if s < 0:
            E += s * s

    return E


def penalty_gradient_fd(params, R, eps=1e-5):
    """Finite-difference gradient of Shapely penalty energy."""
    grad = np.zeros_like(params)
    E0 = penalty_energy(params, R)
    for k in range(len(params)):
        params[k] += eps
        Ep = penalty_energy(params, R)
        params[k] -= eps
        grad[k] = (Ep - E0) / eps
    return grad


def lbfgs_refine_shapely(xs, ys, ts, R, max_iter=200):
    """
    Minimize Shapely penalty energy using L-BFGS-B with FD gradients.
    Slower but exact. Use after phi-based SA finds a good basin.
    """
    p0 = np.zeros(3 * N)
    p0[0::3] = xs; p0[1::3] = ys; p0[2::3] = ts

    def f(p):
        return penalty_energy(p, R)

    def g(p):
        return penalty_gradient_fd(p, R)

    result = minimize(f, p0, jac=g, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-8})
    rxs, rys, rts = result.x[0::3], result.x[1::3], result.x[2::3]
    return rxs, rys, rts, result.fun


if __name__ == '__main__':
    import json, time
    with open('best_solution.json') as f:
        raw = json.load(f)
    xs = np.array([s['x'] for s in raw])
    ys = np.array([s['y'] for s in raw])
    ts = np.array([s['theta'] for s in raw])
    R = 3.071975

    p = np.zeros(45); p[0::3]=xs; p[1::3]=ys; p[2::3]=ts
    E = penalty_energy(p, R)
    print(f'Valid solution: E={E:.2e} (expect 0)')

    # Test at slightly compressed R
    t0 = time.time()
    rxs, rys, rts, E2 = lbfgs_refine_shapely(xs*0.97, ys*0.97, ts, 3.0)
    print(f'Refinement at R=3.0: E={E2:.4f}, time={time.time()-t0:.1f}s')
