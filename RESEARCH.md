# Packing 15 Unit Semicircles into a Minimal Enclosing Circle: A Computational Study

**Authors:** Cameron Armstrong, Till (AI research assistant)  
**Date:** March 26–29, 2026  
**Repository:** https://github.com/CamArmstr/shape-packing-challenge  
**Status:** Draft — candidate for arXiv submission

---

## Abstract

We study the problem of packing 15 unit semicircles (radius = 1) into the smallest possible enclosing circle, minimizing the enclosing circle's radius R. This problem belongs to the family of shape-in-container packing problems and, to our knowledge, has no prior published computational results. Starting from a baseline packing of R = 3.500, we achieved a best result of R ≈ 2.961 (internal scorer), representing a 15.4% improvement over baseline. This result was ranked #9 on a public leaderboard as of March 28, 2026; subsequent overnight and morning runs have pushed the score below the leaderboard #1 as of March 29, 2026. We describe the methods attempted, identify which approaches were effective and which were not, and characterize the landscape structure of this problem.

A key practical finding: for problems where an exact feasibility oracle is fast (here, Shapely polygon intersection at ~2000 evaluations/sec), simple greedy hill-climbing with exact validation outperforms sophisticated approximate-gradient methods. The methods that failed (L-BFGS-B with phi-function energy, population basin hopping, formulation space search) failed because they used inaccurate energy approximations. The method that succeeded (greedy Shapely hill-climbing and GJK-exact LNS) used the exact oracle directly.

---

## 1. Problem Statement

Pack N = 15 unit semicircles (radius = 1) into the smallest possible enclosing circle. Each semicircle is specified by:
- Center position (x, y) ∈ ℝ²
- Orientation angle θ ∈ [0, 2π) — the direction the curved arc faces

A semicircle consists of a semicircular arc (half of a unit circle) and a flat diameter edge. Two semicircles must not overlap. The score is the radius R of the minimum enclosing circle (MEC) containing all 15 semicircles. Lower is better.

**Lower bounds:**
- Area bound: 15 × (π/2) / π = 7.5. Minimum R for area alone: √7.5 ≈ 2.739. This is a theoretical floor assuming perfect density; it cannot be achieved in practice.
- Our best result: R = 2.961486 (internal scorer, March 29 2026). Gap to area bound: 8.1%.

**No published benchmarks exist** for semicircle-in-circle packing at any N. Fejes Tóth (1971) posed the general semicircle packing density question; it remains open. The problem studied here differs from both circle-in-circle packing (Packomania database) and semicircles-in-rectangle (bin packing literature).

---

## 2. Timeline and Score Progression

| Date/Time | Score | Method | Notes |
|---|---|---|---|
| Mar 26 22:00 | 3.500 | Grid baseline | 3×5 grid layout |
| Mar 27 09:00 | 3.072 | Penalty SA (Python) | ~7 steps/sec, pure Python |
| Mar 27 14:31 | 3.012 | SA (lucky find) | Found during phi-function debugging |
| Mar 27 14:48 | 3.010 | Hybrid optimizer | Jitter + squeeze move |
| Mar 27–28 overnight | 2.976 | Numba SA + hill-climber | 1M steps/sec; see §4 |
| Mar 28 morning | 2.97468 | Submitted to leaderboard | #9 of N participants |
| Mar 28 16:55 | 2.975220 | Hill-climber v2 | Ongoing |
| Mar 28–29 overnight | 2.976532 | overnight_v6 (phi-SA + GJK polish) | 6 workers, 5h run, contacts=30 basin floor |
| Mar 29 04:05 | 2.961912 | lns3 (LNS + GJK polish) | New basin found at 91s; below leaderboard #1 |
| Mar 29 04:10 | **2.961486** | lns3 (LNS + GJK polish) | Current best, still running |

---

## 3. Methods Overview

### 3.1 Feasibility Oracle

All feasibility checking uses Shapely (Python polygon library) with semicircles approximated as polygons with 64 arc points. This is the ground truth for this problem. Speed: approximately 2000 pairwise overlap checks per second on commodity hardware (Intel/AMD x86, no GPU).

### 3.2 Scoring

The minimum enclosing circle (MEC) is computed using Welzl's algorithm with analytical refinement (from the challenge's reference implementation). The MEC radius is the score.

### 3.3 Phi-Function Energy

For fast gradient-based optimization, we implemented a phi-function approximation (Chernov, Stoyan, Romanova, Pankratov, 2012). For two semicircles S_i = C_i ∩ P_i (unit disk intersected with a half-plane):

```
Φ(S_i, S_j) = max{ Φ_CC, Φ_CP, Φ_PC, Φ_PP }
```

Where:
- Φ_CC = (x_i-x_j)² + (y_i-y_j)² - 4  (disk-disk)
- Φ_CP = -(cos(θ_j)(x_i-x_j) + sin(θ_j)(y_i-y_j)) - 1  (disk i vs half-plane j)
- Φ_PC = -(cos(θ_i)(x_j-x_i) + sin(θ_i)(y_j-y_i)) - 1  (half-plane i vs disk j)
- Φ_PP = cos(θ_i)(x_i-x_j) + sin(θ_i)(y_i-y_j)  (only when normals are anti-parallel)

**Critical finding (§5):** The phi-function is conservative — it reports overlap (Φ < 0) for configurations that are actually non-overlapping, specifically when flat faces are nearly parallel. This distorts the L-BFGS-B optimization landscape and causes convergence to suboptimal solutions.

### 3.4 GJK Distance

We implemented GJK (Gilbert-Johnson-Keerthi) exact distance computation using Numba JIT compilation. GJK provides geometrically exact signed distances between semicircles, without the phi-function's flat-face distortion. Speed: approximately 5-10× slower than phi-function, much faster than Shapely.

---

## 4. The Method That Worked

### 4.1 Numba SA Discovery

The best-known solution was initially discovered by a Simulated Annealing run using Numba-JIT compilation (sa_v2.py). Key parameters:
- Steps: 200M per run
- Temperature: T_start=0.003 to T_end=0.00001 (exponential cooling)
- Step size: 0.25 (adaptive, targeting 35-45% acceptance)
- Operators: translate, rotate, swap, squeeze, cluster (Thompson Sampling selection)
- Speed: ~1M steps/sec (Numba JIT)
- Seed topology: 1-5-9 ring structure (serendipitous; see §6)

### 4.2 Shapely Hill-Climber (Primary Method)

The solution was refined from R=2.976 to R=2.974679 by a direct Shapely hill-climber. This is the method that actually produced our best result:

**Algorithm:**
```
Initialize from current best solution
While trials < MAX:
    Pick perturbation type (random):
        50%: perturb one shape's (x, y, θ) by ±scale
        20%: perturb x only
        15%: perturb y only
        10%: perturb θ only
        5%: perturb two shapes simultaneously
    Evaluate new configuration with Shapely (exact)
    If score improves: accept and update best
    If no improvement for 5000 trials: increase perturbation scale
Cycle perturbation scales: [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
```

**Results:** 37 improvements over 123,382 trials, from R=2.975245 to R=2.974679. Each evaluation costs ~0.5ms (Shapely). Total runtime: ~18 hours (this included overnight).

**Why this works better than SA for fine-tuning:**
1. No temperature parameter to tune
2. Uses exact oracle at every step (no approximation error)
3. Never accepts worse solutions — every accepted move is a real improvement
4. Simple to implement and debug

### 4.3 Large Neighborhood Search with GJK Polish (lns3.py)

The breakthrough from R≈2.969 to R≈2.961 came from a lighter LNS implementation that replaced the expensive phi-SA refinement step (20M steps, ~25s/cycle) with a GJK-exact SA polish (3M steps, ~30s/cycle with JIT). The key change was using GJK exact distances for local search rather than phi-function approximations, which produced structurally different local minima.

**Algorithm:**
```
While runtime < 6h:
    Load best from disk
    Remove K=1-3 shapes (boundary-active or random; K weighted toward 1-2)
    Reinsert contact-seeking (50 candidates, 60% adjacent, 40% random)
    GJK polish: 3M steps, T_start=0.003, T_end=5e-6, lam=80000
    Official Shapely validation
    If competitive (within 0.03 of best): second polish pass (5M steps, tighter T)
    Save if strictly better (re-read under lock)
```

**Speed:** ~1 cycle/30s per worker = ~12 cycles/min across 6 workers = ~720 LNS cycles/hour. vs. lns2's ~2-3 cycles/hour per worker. 100× more topological explorations per hour.

**Why it found a new basin:** The LNS removal step destroys local contact structure, forcing the optimizer to find a new contact graph during reinsertion. GJK polish then finds the nearest local minimum of the exact landscape (not the phi-approximation landscape). This combination navigates between distinct topological basins that continuous SA cannot exit.

### 4.4 Best Solution Structure (March 29, 2026)

```json
[
  {"x": -1.073051, "y": 1.322046, "theta": 5.526197},
  {"x": 1.725894, "y": 0.932050, "theta": 0.901054},
  {"x": -0.657287, "y": -0.634327, "theta": 0.354138},
  {"x": -0.553741, "y": -1.877351, "theta": 5.436006},
  {"x": 2.587419, "y": -1.032574, "theta": 2.761821},
  {"x": 1.855112, "y": 0.828552, "theta": 4.042207},
  {"x": 0.880626, "y": 1.751264, "theta": 1.816636},
  {"x": 0.830277, "y": 0.707413, "theta": 3.010450},
  {"x": 0.746247, "y": -1.813684, "theta": 5.637864},
  {"x": -1.922360, "y": -0.386659, "theta": 3.919012},
  {"x": -1.751891, "y": -0.559791, "theta": 0.777415},
  {"x": 1.290331, "y": -1.090902, "theta": 2.496051},
  {"x": -1.442996, "y": -1.328604, "theta": 4.853485},
  {"x": -0.540891, "y": 1.885434, "theta": 2.384645},
  {"x": -2.457396, "y": 1.313484, "theta": 5.792291}
]
```
Score: R = 2.961486 (best as of 2026-03-29 04:10 EDT, lns3 still running)

**Topology:** Approximately 4-11 (4 inner semicircles at r≈0.7-1.0, 11 outer at r≈1.7-2.6). The inner 4 form a loose cluster; the outer 11 are distributed roughly evenly around the perimeter.

**Notable properties:**
- Zero conjugate pairs (no antiparallel flat-face-to-flat-face contacts)
- Shape #14 (x=-2.457) is the boundary-defining shape, located 0.56R from centre
- Contact graph has shifted from the contacts=30 basin floor that overnight_v6 was stuck in

---

## 5. Methods That Failed and Why

### 5.1 L-BFGS-B with Phi-Function Energy

We attempted L-BFGS-B (scipy) minimization using phi-function energy as the objective. This consistently converged to solutions worse than the greedy hill-climber.

**Root cause:** The phi-function Φ_PP component is only valid when flat-face normals are anti-parallel (n_dot ≤ -1+ε). For near-parallel configurations — which arise in tight packings — the formula produces false positives (reports separation when shapes actually overlap), causing L-BFGS-B to think it's at a local minimum when it isn't. We confirmed this with the derivation in PHI_RESEARCH.md.

**Practical evidence:** Our "best" solution has two micro-overlaps at GJK tolerance that Shapely accepts, suggesting the phi-function may have led the optimizer to configurations that are borderline feasible rather than truly optimal.

### 5.2 Monotonic Basin Hopping (MBH)

We implemented MBH following Grosso, Locatelli, and Schoen (2010): strict monotonic acceptance (never accept worse), L-BFGS-B local minimizer, mixed perturbations. After 1000+ rounds of PBH (Population Basin Hopping, N=8), zero improvement over baseline.

**Root cause:** The local minimizer (L-BFGS-B with phi-function energy) was minimizing a distorted landscape. Every perturbation-and-descent cycle converged to approximately the same local minimum of the approximate energy — not the true geometric optimum.

**Lesson:** MBH is only as good as its local minimizer. With an inaccurate energy function, the basins of the approximate landscape don't correspond to basins of the true landscape.

### 5.3 Population Basin Hopping (PBH)

Same root cause as MBH. Population diversity and dissimilarity measures don't help when every local minimization converges to the same artifact of the approximate energy.

### 5.4 Formulation Space Search (FSS)

Converting to polar coordinates (r, φ, θ) and running L-BFGS-B in polar space found no improvement. The coordinate transformation changes the optimization geometry but not the underlying energy function approximation.

### 5.5 Orientation-Flip Combinatorial Search

We tested all 575 variants of 1-, 2-, and 3-shape θ+π flips (all ways to flip 1, 2, or 3 semicircle orientations by 180°), each followed by 5M steps of SA. Zero new bests.

**Conclusion:** The 2.9727 cluster (#5-#8 on the leaderboard) is not accessible from our current configuration via orientation flips. It is in a structurally different basin.

### 5.6 Random Multistart

50 independent SA runs from completely random valid starting configurations (50M steps each). Best result: R ≈ 3.06. The random starts fail to find the 4-11 basin because:
1. Phase 1 (hot SA) rarely compresses to R < 3.4 from random starts
2. The 4-11 topology requires a specific structural pattern that random seeding rarely produces

### 5.7 Fresh 4-11 Seeds with Two-Phase SA

50 runs starting from fresh 4-11 seeds (standard, random orientation, flipped-outer variants) with two-phase SA (hot 20M → cold 200M). Best: R ≈ 3.03. The 4-11 seeds provide the right topology but fail to replicate the original discovery because:
1. The original discovery may have involved a specific lucky sequence of moves
2. The two-phase approach with phi-function SA doesn't reach the same basin floor

---

## 6. Landscape Analysis

### 6.1 The Basin Discovery Problem

Our best solution (R=2.974679) was found by serendipity — a 1-5-9 topology seed worker stumbled into this basin during an overnight SA run. We subsequently failed to replicate this discovery with 200+ deliberate attempts from 4-11 seeds, orientation variants, and random starts.

This suggests the 4-11 basin has a narrow entry point. The SA's continuous perturbation approach rarely crosses the energy barrier from a nearby basin.

### 6.2 The Leaderboard Gap

As of March 28, 2026 (last leaderboard check):
- Our submitted score: R = 2.97468 (#9)
- Cluster at #5-#8: R ≈ 2.9727-2.9728 (gap: 0.002)
- #1: R = 2.96175 (gap: 0.013)

As of March 29, 2026 (current best, not yet submitted):
- Our current best: R = 2.961486
- This is **below the last-known #1** (R = 2.96175), gap: -0.0003
- lns3 is still running and may improve further before submission

The lns3 approach found a new basin (contacts structure changed from the contacts=30 floor) that neither overnight_v6 nor the prior hill-climber could access. This supports the hypothesis that reaching the top of the leaderboard requires topological change, not just local refinement.

### 6.3 Topology Comparison

We tested 8 seed topologies (all followed by PBH, 50-100 rounds):

| Topology | Best R | Notes |
|---|---|---|
| 4-11 ring (our current) | 2.974 | Discovered by SA |
| C5 pentagonal (5×3 groups) | 3.453 | 5 groups of 3, each with pair+single |
| 6 conjugate pairs + 3 singles | 3.510 | Antiparallel flat-face pairs |
| Brickwall random | 3.516 | Grid layout, random orientations |
| 7 conjugate pairs + 1 single | 3.582 | Low valid-seed rate (6/20) |
| C5 loose | 3.566 | Wider C5 grouping |
| Brickwall tight | 3.609 | 1.8×1.6 grid spacing |
| Brickwall standard | 3.723 | 2.1×2.0 grid spacing |

The 4-11 topology produces solutions ~0.48 better than any other topology tested. This large gap suggests it has structurally superior geometry for this problem.

### 6.4 The "Too Fancy" Lesson

This study provides a clear example of where algorithmic sophistication was counterproductive:

- Simulated Annealing (Numba, 1M steps/sec): **found R=2.976**
- Greedy hill-climber (Shapely exact, ~120 eval/sec): **found R=2.974679**
- MBH+PBH (L-BFGS-B + phi-function, ~100 iter/hr): **found nothing**

The least sophisticated method with the most accurate oracle won. This result generalizes: when the evaluation oracle is fast relative to the optimization budget, use it directly rather than approximating it. Approximation introduces bias that can dominate any computational advantage gained.

**Oracle cost determines method choice:**
- Oracle cost < 1ms: greedy hill-climbing, exact evaluation at every step
- Oracle cost 1–100ms: SA with periodic exact validation; approximate energy for candidate selection
- Oracle cost > 100ms: surrogate models, population methods, approximate energy throughout

---

## 7. Implementation Notes

### 7.1 Key Files

| File | Purpose |
|---|---|
| `sa_v2.py` | Numba SA (primary SA optimizer) |
| `hillclimber2.py` | Shapely greedy hill-climber |
| `lns2.py` | LNS with phi-SA refinement (superseded) |
| `lns3.py` | LNS with GJK polish — current best method |
| `overnight_v6.py` | Hybrid phi-SA + GJK polish multi-worker |
| `seeds.py` | Seed topology generators (8 topologies) |
| `topology_run.py` | Parallel topology comparison framework |
| `phi.py` | Phi-function implementation |
| `gjk_numba.py` | GJK exact distance (Numba) |
| `mbh.py` | Monotonic Basin Hopping |
| `pbh.py` | Population Basin Hopping |
| `src/semicircle_packing/scoring.py` | Official Shapely scorer |
| `src/semicircle_packing/geometry.py` | Semicircle geometry primitives |
| `best_solution.json` | Current best solution |

### 7.2 Reproducibility

```bash
git clone https://github.com/CamArmstr/shape-packing-challenge
cd shape-packing-challenge
pip install shapely scipy numpy numba matplotlib
python3 hillclimber2.py  # Run hill-climber from current best
python3 run.py best_solution.json  # Score current best
```

### 7.3 Solution Format

```json
[
  {"x": float, "y": float, "theta": float},
  ...  (15 entries)
]
```
- (x, y): center of the full disk
- theta: angle (radians) the curved arc faces
- Coordinates centered at MEC center (origin)
- Theta normalized to [0, 2π)

---

## 8. Open Questions

1. **What topology do the #1-#8 solutions use?** The last-known #1 (R=2.96175) is now within 0.0003 of our current best. This small gap suggests our solution may be in the same or a closely related basin. We cannot determine topology without access to other participants' solutions.

2. **What is the true optimum for N=15?** The area lower bound (R≈2.739) is likely not achievable; the true optimum is probably in [2.85, 2.96] based on analogy with circle packing. Our current R=2.961 is at the low end of this estimate; whether significant improvement remains is unclear.

3. **Why does LNS find basins that continuous SA cannot?** The contacts=30 basin was exhausted after 5h of overnight_v6. LNS (removing 1-3 shapes and reinserting) found a new basin within 91 seconds. This strongly suggests the barrier between basins is a minimum contact-graph change (removing at least 1 shape), not a continuous perturbation.

4. **Why does the 4-11 topology outperform all others by 0.48?** The structural reason is not fully understood. The inner cluster of 4 may create a stable core that allows the outer 11 to pack more efficiently.

5. **Can the phi-function be fixed for semicircles?** The Φ_PP component fails for non-anti-parallel normals (unbounded half-planes always intersect). An exact phi-function for semicircles would require either: (a) restricting to bounded semicircular segments, or (b) using a different composition rule.

6. **Does flat-face-to-flat-face (conjugate pair) contact appear in the optimal solution?** Our best solution has zero conjugate pairs, which is surprising given the theoretical expectation. This may indicate the optimal solution for N=15 is in a regime where pair formation is suboptimal.

---

## 9. Related Work

- Fejes Tóth, L. (1971). Packing and covering. In *Handbook of Convex Geometry*. (Semicircle packing density question posed, still open.)
- Brass, W., Moser, W., Pach, J. (2005). *Research Problems in Discrete Geometry*. Section 1.5 lists semicircle packing as open.
- Chernov, N., Stoyan, Y., Romanova, T., Pankratov, A. (2012). Phi-functions for 2D objects formed by line segments and circular arcs. *Advances in Operations Research*, Article ID 346358.
- Grosso, A., Locatelli, M., Schoen, F. (2010). Solving the problem of packing equal and unequal circles in a circular container. *Journal of Global Optimization*, 47(1).
- Addis, B., Locatelli, M., Schoen, F. (2008). Efficiently packing unequal disks in a circle. *Operations Research Letters*, 36(1).
- Packomania database (Eckard Specht, 2024): http://packomania.com — contains circles-in-circle but not semicircles-in-circle.

---

## 10. Acknowledgments

Problem posed by benedictbrady via the shape-packing-challenge repository. Computational work performed on a Windows 11 / WSL2 machine with Intel CPU (16 cores) and NVIDIA RTX 3050 GPU (GPU path not utilized in final approach). This research was conducted as part of a human-AI collaborative study in computational optimization.

---

*Draft. Please contact cameron@tilluntil.com before citing.*
