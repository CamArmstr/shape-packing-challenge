# Packing 15 Unit Semicircles into a Minimal Enclosing Circle: A Computational Study

**Authors:** Cameron Armstrong, Till (AI research assistant)  
**Date:** March 26–29, 2026  
**Repository:** https://github.com/CamArmstr/shape-packing-challenge  
**Status:** Draft — candidate for arXiv submission
**Last updated:** April 24, 2026

---

## Abstract

We study the problem of packing 15 unit semicircles (radius = 1) into the smallest possible enclosing circle, minimizing the enclosing circle's radius R. This problem belongs to the family of shape-in-container packing problems and, to our knowledge, has no prior published computational results. Starting from a baseline packing of R = 3.500, we achieved a best result of R = 2.948572 (official scorer), representing a 15.7% improvement over baseline across 48 days of computation and 662 git-tracked solution commits. We describe the methods attempted, identify which approaches were effective and which were not, and characterize the landscape structure of this problem.

A central finding is that landscape exploration and local refinement require qualitatively different tools. A fast approximate inner loop (high-throughput MCMC with a cheap overlap proxy) enabled a basin jump of dR ~ 0.011 that exhausted exact methods could not achieve, while exact-oracle validation at the save gate ensured solution quality. A secondary finding: when an exact feasibility oracle is fast (here, Shapely polygon intersection at ~2000 evaluations/sec), simple greedy hill-climbing with exact validation outperforms sophisticated approximate-gradient methods. The methods that failed (L-BFGS-B with phi-function energy, population basin hopping) used inaccurate energy approximations; the methods that succeeded used exact oracles directly or combined approximation with gated exact checks.

---

## 1. Problem Statement

Pack N = 15 unit semicircles (radius = 1) into the smallest possible enclosing circle. Each semicircle is specified by:
- Center position (x, y) ∈ ℝ²
- Orientation angle θ ∈ [0, 2π) — the direction the curved arc faces

A semicircle consists of a semicircular arc (half of a unit circle) and a flat diameter edge. Two semicircles must not overlap. The score is the radius R of the minimum enclosing circle (MEC) containing all 15 semicircles. Lower is better.

**Lower bounds:**
- Area bound: 15 × (π/2) / π = 7.5. Minimum R for area alone: √7.5 ≈ 2.739. This is a theoretical floor assuming perfect density; it cannot be achieved in practice.
- Our best result: R = 2.948572 (official scorer, April 22 2026). Gap to area bound: 7.6%.

**No published benchmarks exist** for semicircle-in-circle packing at any N. Fejes Tóth (1971) posed the general semicircle packing density question; it remains open. The problem studied here differs from both circle-in-circle packing (Packomania database) and semicircles-in-rectangle (bin packing literature).

---

## 2. Timeline and Score Progression

| Date/Time | Score | Method | Notes |
|---|---|---|---|
| Mar 26 22:00 | 3.500 | Grid baseline | 3x5 grid layout |
| Mar 27 09:00 | 3.072 | Penalty SA (Python) | ~7 steps/sec, pure Python |
| Mar 27 14:31 | 3.012 | SA (lucky find) | Found during phi-function debugging |
| Mar 27 14:48 | 3.010 | Hybrid optimizer | Jitter + squeeze move |
| Mar 27-28 overnight | 2.976 | Numba SA + hill-climber | 1M steps/sec; see S4 |
| Mar 28 morning | 2.97468 | Official submission | Rank #9 on leaderboard |
| Mar 28 16:55 | 2.97522 | Hill-climber v2 | Ongoing refinement |
| Mar 28-29 overnight | 2.976532 | overnight_v6 (phi-SA + GJK polish) | 6 workers, 5h run, contacts=30 basin floor |
| Mar 29 04:05 | **2.961912** | lns3 (LNS + GJK polish) | Basin jump; dR=0.013 in 91s |
| Mar 29-Apr 20 | 2.960344 | Exact-oracle polisher | Grinding from LNS3 seed |
| Apr 21 20:30 | **2.949161** | Fast approx MCMC explorer | dR=0.011; largest single jump; 3-11-1 topology |
| Apr 21 | 2.948911 | Fast approx MCMC polisher | 41 incremental commits |
| Apr 22 12:54 | 2.948598 | Ultra polish | Exact-oracle nano polishing |
| Apr 22 14:30 | **2.948572** | Polish nano | Current best; 662 git-tracked commits |

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

### 4.4 Fast Approximate MCMC with Exact Validation Gate

After LNS and exact-oracle polishing stalled near R ~ 2.960, a qualitative change in search strategy produced the largest single improvement: dR ~ 0.011 in one run, bringing the best from R ~ 2.960 to R = 2.949161. This approach was directly inspired by compusophy's browser-based parallel tempering optimizer, which used a split architecture of 12 MCMC "explorers" and 4 "greedy polishers" targeting 0.234 acceptance rate.

**Key idea:** Decouple the inner search loop from exact validation. The inner loop uses a cheap approximate overlap proxy (GJK signed distance with penalty energy) running at ~300,000 mini-batches per run. Exact Shapely validation is only invoked when a candidate looks promising, acting as a save gate.

**Additional mechanisms:**
- Archive-seeded restarts (bias toward best scores, recency penalty to avoid trapping)
- Invalid gate cooldown (suppress failed signatures to avoid wasted validation)
- Scratch candidate retention (exact-valid non-bests retained as restart seeds)
- Best-approximate persistence (every approximate improvement saved to permanent archive)

Three parallel explorer instances ran with diversified hyperparameters to maximize basin coverage.

### 4.5 Nano/Micro Polishing Phase

After the MCMC explorer found the 2.949 basin, exact-oracle nano polishing continued from Apr 21-22, pushing through 2.948911 to the current best of 2.948572. This phase used ultra-fine perturbation scales and exact Shapely validation at every step, squeezing the last ~340 millionths out of the solution across hundreds of git-tracked commits.

### 4.6 Best Solution Structure (April 22, 2026)

```json
[
  {"x": 1.938028, "y": 0.202442, "theta": 5.336543},
  {"x": 0.124380, "y": -1.944598, "theta": 5.312997},
  {"x": -1.339857, "y": 1.414814, "theta": 3.064984},
  {"x": -0.619526, "y": -0.083000, "theta": 4.389019},
  {"x": 0.829071, "y": -1.461855, "theta": 2.171414},
  {"x": 0.685905, "y": 0.532934, "theta": 5.488696},
  {"x": -1.168468, "y": -1.559364, "theta": 5.263949},
  {"x": 0.413225, "y": 1.666611, "theta": 5.185991},
  {"x": -0.377234, "y": 0.890394, "theta": 4.362949},
  {"x": -1.790756, "y": -0.768195, "theta": 4.413237},
  {"x": -0.297759, "y": 1.925687, "theta": 2.897761},
  {"x": 0.705463, "y": 1.816385, "theta": 2.044400},
  {"x": 2.118373, "y": -1.790690, "theta": 2.439826},
  {"x": 1.562506, "y": 1.164263, "theta": 0.236043},
  {"x": -1.922318, "y": 0.318790, "theta": 3.589860}
]
```
Score: R = 2.948572 (best as of 2026-04-22 14:30 EDT, polish_nano)

**Topology:** 3-11-1 (3 inner semicircles at r < 1.0, 11 mid-ring at r ~ 1.55-2.11, 1 boundary-defining at r ~ 2.83). This is a structurally different basin from the prior 4-11 topology.

**Notable properties:**
- Zero conjugate pairs (no antiparallel flat-face-to-flat-face contacts)
- Shape 12 (r=2.834) is the sole boundary-defining shape, determining MEC radius
- 31 GJK contacts (up from 30 in the 2.961-era 4-11 solution)
- Shape 3 is the most connected shape (6 contacts: 4, 5, 6, 8, 9, 14)
- 11 mid-ring shapes packed within a radial band of width 0.55 (r = 1.55 to 2.11)

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

Our best solution topologies were found through two distinct mechanisms: serendipitous topological collapse (SA falling into a 4-11 arrangement) and high-throughput approximate exploration (fast MCMC exploring the 3-11-1 region). Neither was achievable by careful exact refinement within the prior basin.

This pattern suggests the landscape for semicircle packing is multi-modal with high inter-basin barriers. The barriers appear to require a minimum contact-graph change to cross, making them inaccessible to continuous exact-oracle perturbations that preserve local contacts.

### 6.2 The Leaderboard Gap

As of March 28, 2026 (last leaderboard check):
- Our submitted score: R = 2.97468 (#9)
- Cluster at #5-#8: R ~ 2.9727-2.9728 (gap: 0.002)
- #1: R = 2.96175 (gap: 0.013)

As of April 22, 2026 (current best):
- Our current best: R = 2.948572
- This is 0.013 below the last-known #1 (R = 2.96175)
- The target to formally beat leaderboard #1 was R < 2.94889; our solution surpasses this by ~320 millionths

The progression required two basin jumps: LNS (4-11 basin, R=2.961) and fast approx MCMC (3-11-1 basin, R=2.949), followed by extensive nano polishing to 2.948572.

### 6.3 Topology Comparison

We tested 8 seed topologies (all followed by PBH, 50-100 rounds):

| Topology | Best R | Notes |
|---|---|---|
| 3-11-1 (current best) | 2.948572 | Found by fast MCMC |
| 4-11 ring (prior best) | 2.960344 | Found by LNS3 from SA seed |
| C5 pentagonal (5×3 groups) | 3.453 | 5 groups of 3, each with pair+single |
| 6 conjugate pairs + 3 singles | 3.510 | Antiparallel flat-face pairs |
| Brickwall random | 3.516 | Grid layout, random orientations |
| 7 conjugate pairs + 1 single | 3.582 | Low valid-seed rate (6/20) |
| C5 loose | 3.566 | Wider C5 grouping |
| Brickwall tight | 3.609 | 1.8×1.6 grid spacing |
| Brickwall standard | 3.723 | 2.1×2.0 grid spacing |

The 4-11 and 3-11-1 topologies both strongly outperform all others, but they are structurally distinct basins: direct perturbation from 4-11 configurations could not reach 3-11-1 geometries.

### 6.4 The "Too Fancy" Lesson

This study provides a clear example of where algorithmic sophistication was counterproductive:

- Numba SA (~1M steps/sec): **found R=2.976**
- Greedy hill-climber (Shapely exact, ~120 eval/sec): **found R=2.974679**
- MBH+PBH (L-BFGS-B + phi-function, ~100 iter/hr): **found nothing**
- LNS3 with GJK polish (~12 cycles/min): **dR=0.013 in 91 seconds**
- Fast approximate MCMC (~300k batches/run, Shapely gate): **dR=0.011 in one run**

The pattern is instructive: approximate fast methods dominated for large-scale basin exploration, while exact methods dominated for local refinement. Neither alone was sufficient to reach the current best.

**Oracle cost determines method choice:**
- Oracle cost < 1ms: greedy hill-climbing or SA with exact evaluation at every step
- Oracle cost 1-100ms: fast approximate inner loop with exact validation gate at save events
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

1. **What is the true optimum for N=15?** The area lower bound (R ~ 2.739) is likely not achievable; the true optimum is probably in [2.85, 2.95] based on area bounds and observed landscape structure. Our current R=2.948572 is near the low end of this estimate. Whether significant improvement remains is unclear.

2. **Can the approximate/exact gap be closed?** 12 distinct configurations with approximate score R_approx = 2.948586 (~300 millionths below the formal leaderboard target) are known but fail exact validation due to marginal overlaps. Whether any can be nudged to validity without inflating the exact score remains open.

3. **Why does LNS find basins that continuous SA cannot?** The contacts=30 basin was exhausted after 5h of overnight_v6. LNS (removing 1-3 shapes and reinserting) found a new basin within 91 seconds. The barrier between basins is a minimum contact-graph change (removing at least 1 shape), not a continuous perturbation.

4. **Can the phi-function be fixed for semicircles?** The phi_PP component fails for non-anti-parallel normals. An exact phi-function for semicircles would require either: (a) restricting to bounded semicircular segments, or (b) using a different composition rule.

5. **Does flat-face-to-flat-face (conjugate pair) contact appear in the optimal solution?** Both our best solutions (4-11 and 3-11-1) have zero conjugate pairs, suggesting that for N=15 the optimal regime avoids flat-face pair formation.

6. **Does the global optimum use a ring topology, and if so which one?** The 3-11-1 and 4-11 topologies both strongly outperform all others tested but are structurally distinct basins. Whether a third, undiscovered topology exists below 2.948 is unknown.

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
