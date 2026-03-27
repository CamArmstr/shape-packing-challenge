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
