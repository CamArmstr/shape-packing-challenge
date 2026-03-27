# Phi-Function Research: Correct Formula for Two Semicircles

## Reference Paper

**"Phi-functions for 2D objects formed by line segments and circular arcs"**
N. Chernov, Yu. Stoyan, T. Romanova, A. Pankratov (2012)
*Advances in Operations Research*, Article ID 346358.

## General Composition Rule

The paper establishes that for shapes defined as **intersections** of simpler convex objects:

> If **S = A ∩ B** and **S' = C ∩ D**, then:
> **Φ(S, S') = max{ Φ(A,C), Φ(A,D), Φ(B,C), Φ(B,D) }**

**Set-theoretic justification:** (A∩B) ∩ (C∩D) ⊆ A∩C, ⊆ A∩D, ⊆ B∩C, ⊆ B∩D.
So if ANY cross-pair has empty intersection, the shapes are separated.
This is a **sufficient** condition for non-overlap: Φ ≥ 0 ⟹ non-overlapping.

For **unions**, the dual rule applies: Φ(A∪B, C∪D) = min over all pairs.

## Application to Semicircles

A semicircle: **S_i = C_i ∩ P_i** (unit disk intersected with arc-side half-plane).

So: **Φ(S_i, S_j) = max{ Φ_CC, Φ_CP, Φ_PC, Φ_PP }**

### The Four Components

| Component | Formula | Status |
|-----------|---------|--------|
| Φ_CC (disk–disk) | `(xi-xj)² + (yi-yj)² - 4` | ✅ Correct |
| Φ_CP (disk i vs half-plane j) | `-(cos(tj)·(xi-xj) + sin(tj)·(yi-yj)) - 1` | ✅ Correct |
| Φ_PC (half-plane i vs disk j) | `-(cos(ti)·(xj-xi) + sin(ti)·(yj-yi)) - 1` | ✅ Correct |
| Φ_PP (half-plane i vs half-plane j) | `cos(ti)·(xi-xj) + sin(ti)·(yi-yj)` | ❌ **Only valid for anti-parallel normals** |

## The Φ_PP Problem — Critical Finding

### Why the current formula fails

Two **unbounded** half-planes P_i, P_j can only have empty intersection when their normals are **exactly anti-parallel** (n_i = -n_j). When normals are NOT anti-parallel, two half-planes ALWAYS overlap (their intersection is an infinite wedge).

**Proof:** For non-anti-parallel normals, the boundary lines are not parallel, so they intersect at some point p*. The interior of both half-planes extends from p* in some direction, forming a non-empty wedge.

The current formula `cos(ti)·(xi-xj) + sin(ti)·(yi-yj)` measures the signed projection of the center-to-center vector along n_i. This can be **positive for non-anti-parallel normals**, producing a **false positive** (claims separation when the half-planes actually overlap).

### Concrete counterexample

```
S_1: center=(0, 0), θ₁=3π/4   (arc faces upper-left)
S_2: center=(0.5, 0), θ₂=π/4  (arc faces upper-right)
n_dot = n₁·n₂ = 0  (perpendicular normals)
```

**Current Φ_PP** = cos(3π/4)·(0-0.5) + sin(3π/4)·0 = (−√2/2)(−0.5) = **+0.354** (false positive!)

The half-planes are: P₁ = {y ≥ x} and P₂ = {x+y ≥ 0.5}. Point (0, 0.5) is in both — they overlap.

**Actual semicircle overlap check:** Point (0, 0.5) is in S₁ (0²+0.25² ≤ 1 ✓, 0.5 ≥ 0 ✓) AND in S₂ ((−0.5)²+0.25²=0.5 ≤ 1 ✓, 0+0.5 ≥ 0.5 ✓). **The semicircles DO overlap.**

With current code (n_dot=0, threshold is -0.5, so phi_PP excluded): max(cc, cp, pc) = max(−3.75, −0.646, −0.646) = −0.646 < 0. ✅ Correctly reports overlap.

**If phi_PP were always included:** max(−3.75, −0.646, −0.646, +0.354) = +0.354. ❌ False positive!

### Why the n_dot < -0.5 threshold is also wrong

Even at n_dot = −0.7 (angle ≈134°), the half-planes are not anti-parallel and always overlap. The formula can still produce false positives. The threshold -0.5 is too permissive.

**However:** the current code's approach of EXCLUDING phi_PP when n_dot ≥ -0.5 is **conservative** (it may report overlap when shapes are actually separated). This is safe for packing validation but causes the optimizer to push shapes apart more than necessary, potentially missing tight back-to-back configurations.

## The Fundamental Issue

The Chernov paper's composition formula works for shapes decomposed into **bounded** convex objects (disks and triangles/polygons). Half-planes are **unbounded** — there is no valid phi-function between two non-anti-parallel half-planes that produces correct signs, because non-anti-parallel half-planes always overlap.

This means the simple 4-way max formula does **not** yield a valid phi-function for semicircles (= disk ∩ half-plane), unless phi_PP is restricted to anti-parallel configurations.

## Correct Approaches

### Option A: Conservative (current code, refined)

Use only 3 components by default:
```
Φ = max(Φ_CC, Φ_CP, Φ_PC)
```

Add Φ_PP only when normals are very close to anti-parallel (n_dot < -0.99 instead of -0.5):
```python
if n_dot < -0.99:
    phi = max(phi, phi_PP(xi, yi, ti, xj, yj, tj))
```

**Pro:** Simple, safe, never produces false positives.
**Con:** Still conservative — may miss valid back-to-back separation for normals not quite anti-parallel.

### Option B: Smooth damped Φ_PP

Scale Φ_PP by a factor that goes to 0 for non-anti-parallel normals:
```python
damping = max(0.0, -n_dot)  # 1 when anti-parallel, 0 when perpendicular/parallel
phi_PP_damped = phi_PP_raw * damping
```

**Behavior:**
- Anti-parallel (n_dot = −1): damping=1, Φ_PP = full value ✅
- Perpendicular (n_dot = 0): damping=0, Φ_PP = 0 (neutral, not falsely positive)
- 135° (n_dot = −0.707): damping=0.707, Φ_PP reduced

**Caution:** When Φ_PP_damped = 0 and all others < 0, max = 0 ("touching" when shapes may be separated). This is conservative, not unsafe.

**Pro:** Smooth, differentiable, no discontinuous threshold.
**Con:** Still can be slightly positive for intermediate angles when half-planes actually overlap.

### Option C: Bounded representation (follows the Chernov paper exactly)

Replace the half-plane P_i with the **diameter line segment** L_i of the semicircle:
- L_i from (x_i − sin θ_i, y_i + cos θ_i) to (x_i + sin θ_i, y_i − cos θ_i)

Then redefine the PP component as a **segment-segment** phi-function (Chernov Eq. 5-6):
```
S_i = C_i ∩ T_i   where T_i is a triangle/polygon containing the semicircle
Φ(S_i, S_j) = max(Φ_CC, Φ_CT', Φ_TC', Φ_TT')
```

For two line segments, the phi-function is based on the signed separation distance:
```
Φ_LL = min over vertex-edge pairs of signed distances
```

**Pro:** Theoretically correct, bounded objects, matches the paper's framework.
**Con:** More complex implementation; need to derive segment-disk and segment-segment phi-functions with gradients.

### Option D: Symmetric Φ_PP (partial fix)

Use the maximum of BOTH directional projections:
```python
phi_PP = max(
    n_i · (c_i - c_j),   # i separates from j
    n_j · (c_j - c_i)    # j separates from i
)
```

This is still wrong for non-anti-parallel normals (both can be positive), but it's symmetric and slightly more robust.

### Option E: Use Shapely as ground truth, phi only for gradients

Since the current hybrid optimizer already uses Shapely for the true overlap check:
1. Keep the current phi-function (with its imperfections) for **gradient computation only**
2. Use Shapely intersection area as the **energy function**
3. The phi gradients just need to push shapes in approximately the right direction

**Pro:** Already implemented, correct overlap detection guaranteed.
**Con:** Incorrect gradients can mislead the optimizer, causing slow convergence or oscillation near configurations where phi disagrees with Shapely.

## Recommendations (Priority Order)

1. **Immediate fix (Option A):** Tighten the n_dot threshold from -0.5 to -0.95 or -0.99. This reduces false positives while keeping some phi_PP benefit for near-conjugate pairs. Minimal code change.

2. **Better fix (Option B):** Replace the threshold with smooth damping: `phi_PP *= max(0, -n_dot)`. This is differentiable everywhere and automatically handles the transition. The gradients flow smoothly. Requires updating the gradient code too.

3. **Correct fix (Option C):** Implement the bounded segment-based approach from the Chernov paper. This is the theoretically sound solution but requires significant implementation work (segment-disk phi-function, segment-segment phi-function, and all their gradients).

4. **Pragmatic fix (Option E):** Accept the current phi-function's imperfections and rely on Shapely for correctness. Focus optimization effort on better perturbation strategies rather than phi-function accuracy.

## Summary

| What | Status |
|------|--------|
| Φ_CC formula | ✅ Correct |
| Φ_CP formula | ✅ Correct |
| Φ_PC formula | ✅ Correct |
| Φ_PP formula | ❌ Only valid for anti-parallel normals |
| max composition | ✅ Correct principle, but requires valid component phi-functions |
| Current n_dot < -0.5 threshold | ⚠️ Too permissive (should be < -0.95 or use smooth damping) |
| Root cause | Half-planes are unbounded; the Chernov paper uses bounded shapes (triangles) |

## Key Equations Reference

Convention: n_i = (cos θ_i, sin θ_i), half-plane P_i: n_i · (p − c_i) ≥ 0.

```
Φ_CC = |c_i − c_j|² − 4

Φ_CP = −n_j · (c_i − c_j) − 1

Φ_PC = −n_i · (c_j − c_i) − 1    [= −n_i · (c_j − c_i) − 1]

Φ_PP = n_i · (c_i − c_j)          [ONLY for n_i · n_j ≈ −1]

Φ(S_i, S_j) = max(Φ_CC, Φ_CP, Φ_PC, [Φ_PP if applicable])
```
