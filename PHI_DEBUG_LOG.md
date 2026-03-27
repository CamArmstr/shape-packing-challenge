# Phi-Function Debug Log
*Last updated: 2026-03-27*

## Problem Statement
Pack 15 unit semicircles in the smallest enclosing circle (radius R).
Current best (SA + Shapely): **R = 3.071975** (saved in best_solution.json).

## Geometry Convention
From `src/semicircle_packing/geometry.py`:
- Semicircle S_i has center (xi, yi), orientation θ_i
- Arc spans from θ_i - π/2 to θ_i + π/2 (apex at angle θ_i, i.e. the dome points in direction θ_i)
- Arc-side half-plane P_i: `cos(θi)*(p.x - xi) + sin(θi)*(p.y - yi) >= 0`
- Flat edge (diameter) is perpendicular to θ_i
- Critical containment points: arc apex `(xi+cos(ti), yi+sin(ti))`, endpoints `(xi±sin(ti), yi∓cos(ti))`

## Phi-Function Components

### phi_CC (Disk-Disk) ✅ CORRECT
`phi_CC = (xi-xj)² + (yi-yj)² - 4`
- >= 0 iff disks are separated by >= 0 gap

### phi_CP (Disk i vs Flat-side of j) ✅ CORRECT FORMULA, INCOMPLETE CONDITION
`phi_CP = -(cos(tj)*(xi-xj) + sin(tj)*(yi-yj)) - 1`
- >= 0 iff center of C_i is >= 1 unit into the flat (non-arc) side of S_j
- This means C_i's closest point to S_j's arc region is on or past the flat edge
- **LIMITATION**: phi_CP >= 0 does NOT guarantee S_i ∩ S_j = empty. It only guarantees C_i ∩ P_j (arc half-plane) has zero interior.

### phi_PC (Flat-side of i vs Disk j) ✅ CORRECT FORMULA, INCOMPLETE CONDITION
`phi_PC = -(cos(ti)*(xj-xi) + sin(ti)*(yj-yi)) - 1`
- Symmetric to phi_CP

### phi_PP (HalfPlane-HalfPlane) ❌ PROBLEMATIC
`phi_PP = cos(ti)*(xi-xj) + sin(ti)*(yi-yj)`
- Derived for the near-conjugate pair case (n_dot ≈ -1)
- **FALSE NEGATIVES**: For same/similar orientation pairs (n_dot > 0), phi_PP can be large positive while shapes genuinely overlap. This caused the bogus R=2.71 result.
- **NEEDED FOR**: Conjugate pairs (n_dot ≈ -1) where phi_CP/phi_PC both say "overlapping" but shapes don't actually overlap.

## Error Taxonomy

### False Positives (phi says overlap, Shapely says fine)
Observed in: valid best_solution.json (8 pairs)
Cause: phi_CP or phi_PC returns negative, but the two semicircles don't actually overlap because even though C_i partially enters the arc half of S_j, the arc-restricted regions S_i = C_i ∩ P_i and S_j = C_j ∩ P_j don't intersect.
Example pair (0,5): n_dot=0.800, phi_CC=-0.765, phi_CP=-0.174, phi_PC=-0.703, phi_PP=0.298

### False Negatives (phi says fine, Shapely says overlap)
Observed in: jitter+L-BFGS perturbed configurations
Cause: phi_PP returns positive for same-orientation overlapping pairs, which drives max(phi_CC, phi_CP, phi_PC, phi_PP) >= 0 incorrectly.
Example: pairs with n_dot=0.7-0.9 where phi_PP ≈ 0.7 overrides the true phi=-0.3

## What Actually Works

1. **phi_CC only**: Never gives false negatives for disk overlap. Safe to use as a pre-filter (skip pair if phi_CC >= 0).
2. **Shapely intersection area**: Ground truth. ~1ms per 105-pair config evaluation.
3. **SA + Shapely**: The original optimizer. Correct but slow (CPU-bound SA, 2M steps per run).

## Gradient Quality
- phi_CC, phi_CP, phi_PC gradients: ✅ verified correct via finite difference (rel error < 1e-10)
- phi_PP gradient: ✅ formula is correct but the phi_PP value itself is wrong for general cases
- Gradient check with compressed config (has overlaps): ✅ PASS

## Attempted Fixes (All Failed)
1. **Remove phi_PP entirely**: Fixes false negatives, causes false positives (conjugate pairs)
2. **Conditional phi_PP (n_dot < -0.5)**: Partially better but still ~8 false positives
3. **Shapely L-BFGS with FD gradients**: Correct but 20s per call (45 Shapely evals per gradient)
4. **Two-pass L-BFGS tighter convergence**: Same local min, doesn't help

## Root Cause (Conceptual)
The phi-function decomposition `Φ^{S_i,S_j} = max(Φ_CC, Φ_CP, Φ_PC, Φ_PP)` is not the correct Stoyan formulation for semicircle pairs. The components check {disk vs disk}, {disk vs flat half-plane}, etc. — but these are NOT the same as checking {semicircle vs semicircle} because S_i = C_i ∩ P_i restricts the shape. The correct Stoyan phi for intersection shapes requires a more careful decomposition.

## Paths Forward
1. **Derive correct phi**: The proper Stoyan phi for S_i = C_i ∩ P_i vs S_j = C_j ∩ P_j. Need Chernov et al. 2012 paper for exact formula. This could be a few more hours of math.
2. **Use Shapely area + phi gradient direction**: Use phi gradient to steer L-BFGS, but evaluate Shapely area for energy. Would need ~90ms per gradient step (FD with Shapely). Feasible for small runs.
3. **Hybrid SA + phi polish**: Run SA (correct) for global search, then use phi-gradient L-BFGS only as post-processor on SA-valid solutions. This is likely the fastest path to improvement.
4. **Overnight SA run**: Just restart the SA (optimize_all.py) which we know is correct. Slow but guaranteed valid.

## Status
- phi.py: gradient correct, phi values incorrect for general semicircle pairs
- mbh.py: Shapely gate working correctly, phi energy driving optimizer to wrong basins
- best_solution.json: R=3.071975 (SA result, Shapely-validated)
- Next: decide between path 3 (hybrid) or path 4 (overnight SA)

---

## Report 3: Optimization Strategies for Sub-R=3.0 (partial — awaiting continuation)

### Key strategies identified:

**1. Gap-targeted LNS (most impactful)**
- CEGP: score pieces by overlap energy, remove worst 2-3 (destroy ratio 10-30%)
- Re-insertion via Maximum Hole Degree (tangent-to-two-objects positions)
- Alternative: grid-based vacant-degree scan (100×100 grid, ~15k distance evals)
- Voronoi gap detection: Delaunay triangulation of remaining centers, circumcenters = gap candidates
- Orientation scan: 12-16 discrete θ values + L-BFGS refinement at best
- Full cycle: score → remove worst 2-3 → find gaps (Voronoi/grid) → scan orientations → L-BFGS → accept

**3. Formulation space switching (mentioned as second most impactful)**
- Alternating between Cartesian and polar coordinates
- Local min in one formulation = saddle in another

---

## Report 3 (continued): Full optimization strategy report

### 2. Dense planar semicircle packing hints
- Fejes Tóth 1971 open problem — exact optimal density unknown but > π/(2√3) ≈ 0.9069
- Key structural motifs: flat edges aligned against each other, alternating orientations (up/down in rows), brick-like row offsets with curved edges nesting into concavity
- Double-lattice packing: one sublattice = K, other = -K (180° rotated). Kuperberg 1990: every convex body achieves ≥ √3/2 ≈ 0.866 via double lattice
- For finite packing: don't assume paired circles. Herringbone / flat-edge alignment patterns + curved edges at container boundary may be denser
- Current density at R=3.010: 15 × π/2 / π×3.010² ≈ 83.0% — high but achievable

### 3. Formulation Space Switching (FSS) — Specific implementation
- Key insight: a KKT point in Cartesian is NOT stationary in polar — coordinate transformation breaks first-order conditions
- Reformulation Descent (RD): solve Cartesian → convert to polar → solve again → convert back → repeat
- Full FSS: each of 15 semicircles independently Cartesian or polar → 2¹⁵ = 32,768 formulations
- Third encoding: (r, φ, θ_rel) where θ_rel = θ - φ (orientation relative to radial direction) — natural for boundary pieces
- Total formulation space: 3¹⁵ ≈ 14.3M
- Implementation: scipy.minimize L-BFGS-B, solve to full convergence before switching
- RD alone: ~40% of cases finds near-optimal in isolation, ~150× faster than Newton methods
- Start in Cartesian (Mladenović found this generally better)

### 4. Basin diversity — Population Basin Hopping (PBH)
- Maintain 10-20 diverse local minima
- Dissimilarity threshold d_cut: new solution only enters if sufficiently different from all existing
- Measure: Hungarian-algorithm optimal assignment distance (Σ‖pᵢ - q_π(i)‖²) — handles S₁₅ permutation
- Fingerprinting: sorted pairwise distances (105) + sorted radial distances (15) + contact graph hash
- Rigidity check: rigid packing needs ~3n-3 = 42 contacts. Fewer contacts = zero-energy deformation modes = optimization opportunity
- Symmetry detection: rotate by 2π/k, compute assignment distance — check C₁, C₂, C₃, C₅
- Confidence threshold: if best found by ≥3 of ~100 independent restarts, reasonably confident it's near-optimal
- Scale: 200-3,000 restarts for comprehensive basin coverage

### 5. Exact distance function validation plan (priority — do before more optimization)
Six distinct distance cases: arc-arc, arc-flat, arc-endpoint, flat-flat, flat-endpoint, endpoint-endpoint
Dangerous cases: near-tangent arcs, near-parallel flat edges (subtractive cancellation)

**Tier 1** — 100 exact analytical tests (known closed-form distances)
**Tier 2** — 10,000 property-based tests via Hypothesis (symmetry, sign correctness, Lipschitz, triangle inequality)
**Tier 3** — 5,000 oracle comparisons vs 1000-vertex Shapely polygon (strongest validation)
**Tier 4** — 1,000 stratified edge cases (far, near-touching |d|<0.01, overlapping, contained, coincident)
**Tier 5** — 1,000 perturbation/gradient tests (continuity, no jumps across case boundaries)

Statistical: 17,100 total tests → 99.9% confidence failure rate < 0.1%

Numerical robustness: clamp acos/asin inputs to [-1,1], use atan2, translate to origin before computing

### Concrete upgrade path from R=3.010 → sub-3.0
Priority order:
1. **Validate exact_dist first** (Tier 3 oracle comparison vs Shapely, 5000 cases)
2. **CEGP + MHD/grid repair** (replaces random perturbation with gap-targeted destroy-repair)
3. **Reformulation Descent** (Cartesian ↔ polar, escapes first-order stationary points)
4. **PBH with fingerprinting** (population diversity, duplicate rejection)
5. **Herringbone seed configurations** (flat-edge aligned starts, not just random)
