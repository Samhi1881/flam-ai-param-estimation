# Parameter Estimation for a Parametric Curve — Methodology & References

This document explains the approach used to recover the unknown parameters
`θ`, `M`, `X` of the parametric curve

```
x(t) = t·cos(θ) − e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```

from an unordered, unlabeled set of `(x, y)` points sampled for `6 < t < 60`.

---

## 1. Why this is harder than ordinary curve fitting

`xy_data.csv` gives points on the curve but **no `t` labels and no guarantee
of ordering**. Standard least-squares curve fitting assumes you know which
data point corresponds to which parameter value, or at least that the points
are sequentially ordered along the curve. Neither is true here, so the
problem falls in the "curve reconstruction / fitting from unorganized
points" family rather than ordinary regression.

Foundational treatments of this exact issue:

- Fang & Gossard, *Multidimensional curve fitting to unorganized data
  points by nonlinear minimization* [3]
- Lee, *Curve reconstruction from unorganized points* [5]
- Ma & Kruth, *Parameterization of randomly measured points for least
  squares fitting of B-spline curves and surfaces* [7]

These establish that when point order/parameterization is unknown, the
correspondence between data points and curve parameter must itself be
estimated (or marginalized out) as part of the fit — which motivates Step 2
below.

A second difficulty is that the model is **non-convex** in the unknowns:
`θ` enters through `sin`/`cos` (periodic, so multiple angles can produce
locally similar-looking curves within the given bounds), and the term
`e^{M|t|}·sin(0.3t)` oscillates with a period-dependent envelope controlled
by `M`, so nearby `(θ,M,X)` triples can correspond to visually different
curves. This rules out simple gradient descent / closed-form least squares
as a reliable primary method (it's used later only as a fast local
polish — §2, Step 4) and motivates the global-search stage (§2, Step 3).

---

## 2. Complete process — step-by-step

### Step 0 — Load & inspect data
Read `xy_data.csv` into an `(N, 2)` array of `(x, y)` pairs. No ordering or
`t` value is assumed. Basic sanity checks: point count, coordinate ranges,
and a quick scatter plot to visually confirm the data traces a single
open curve (not multiple disconnected branches, which would indicate a
data issue rather than a fitting issue).

### Step 1 — Closed-form initial guess (PCA)
Before any iterative search, a fast initial estimate of `θ` (and hence
`X`) is derived analytically from the data's principal direction of
variance — see §5.7 for the derivation. This is not required for
correctness (DE in Step 3 explores the full bounded space regardless) but
substantially reduces the number of generations needed to converge.

### Step 2 — Candidate curve generation
For a candidate `(θ, M, X)`, generate a dense set of points by evaluating
`x(t), y(t)` on a fine grid over `6 < t < 60` (`K ≈ 5000` samples, chosen so
consecutive curve samples are closer together than the typical spacing
between adjacent data points — this keeps the nearest-neighbor
approximation in Step 3 accurate).

### Step 3 — Correspondence via nearest-neighbor matching
Since the data points are unordered, each observed point is matched to its
closest point on the current candidate curve using a nearest-neighbor
search, rather than assuming a fixed index-to-`t` mapping. This turns an
"unordered point set vs. curve" comparison into a well-defined distance that
can be minimized, following the scalable nearest-neighbor approach of
Muja & Lowe [8] and, for larger grids, the improved query-cost bound of
Wang et al. (2025) [12]. Full derivation in §5.1–5.2.

### Step 4 — Global search with Differential Evolution
The residual from Step 3 is non-convex in `(θ, M, X)` (§1), so
Differential Evolution (DE) [10] is used to explore the bounded search
space `0° < θ < 50°, −0.05 < M < 0.05, 0 < X < 100` broadly and avoid
getting trapped in a poor local optimum, following the practical parameter
guidance in the DE survey by Das & Suganthan [2]. The Step 1 PCA estimate
seeds one population member; the rest are drawn uniformly at random from
the bounds so the global search isn't biased if the PCA approximation is
poor. Full update equations in §5.4.

### Step 5 — Local refinement with Nelder–Mead
The best DE candidate is polished with the Nelder–Mead simplex method [4],
which converges quickly once near the optimum, giving a tight final fit.
Full update equations in §5.5. Running DE then Nelder–Mead in sequence
(rather than either alone) follows the two-stage hybrid strategy validated
by Wang, Xu & Li [11] and Lin [6] — rationale in §5.6.

### Step 6 — Validation against the grading metric
The fitted `(θ, M, X)` is validated using the **same** L1-distance
procedure described in §4 that is used for grading — uniformly resample
both the fitted curve and (as a proxy for the unknown ground truth) the
original data points, and report the resulting L1 value as a sanity check
before finalizing the submitted numbers. This is a self-check, not the
official score (only the grader has the true `(θ*, M*, X*)` to compare
against).

### Step 7 — Report unknowns
The final `(θ, M, X)` is written out in the required LaTeX/Desmos format
per the assignment's submission instructions.

### Pseudocode — full pipeline (high-level)

```
INPUT:  P = unordered (x, y) points from xy_data.csv
        bounds on θ, M, X ; t ∈ [6, 60]
OUTPUT: (θ̂, M̂, X̂)

θ0, M0, X0 ← PCA_initial_guess(P)              # §5.7

FUNCTION J(θ, M, X):                            # §5.1–5.3
    Q ← curve points C(t; θ,M,X) on a fine t-grid
    RETURN mean squared nearest-neighbor distance from P to Q

v_DE  ← DifferentialEvolution(J, bounds, seed = (θ0,M0,X0))   # §5.4
θ̂,M̂,X̂ ← NelderMead(J, start = v_DE)                          # §5.5

L1_proxy ← self_validate(θ̂, M̂, X̂, P)          # §4.3, sanity check
RETURN (θ̂, M̂, X̂)
```

---

## 3. Recent literature (2025–2026)

The following recent papers were reviewed while designing/justifying the
pipeline.

1. **Ugail, H., & Howard, N. (2026).** Neural adaptive tension for
   multi-geometry curve subdivision: a unified approach. [1]
   *Introduces a neural-network-controlled tension parameter for
   subdivision-curve schemes that generate smooth curves from discrete
   control points across different curvature geometries. Relevant as a
   modern alternative for generating a smooth curve representation from
   sparse/discrete points — a different sub-problem (subdivision-based
   smoothing) from the explicit parametric model fit used in this
   assignment, but useful context on current curve-reconstruction
   techniques.*

2. **Zou, Q., Zhu, L., Wu, J., & Yang, Z. (2025).** SplineGen:
   approximating unorganized points through generative AI. [13]
   *Uses a learned generative model to fit a spline to unorganized points,
   illustrating a data-driven alternative to the optimization-based
   approach used here, and confirming "unorganized points → curve" is an
   active 2025 research area.*

3. **Pavelkin, R., Zavala-Mondragon, L. A., Ekin, A., & van der Sommen, F.
   (2025).** Spline-based shape compression for interventional device
   tracking. [9]
   *A 2025 application of parametric curve fitting to noisy, sparsely
   sampled real-world point data, relevant to how measurement noise in
   xy_data.csv is handled during fitting.*

4. **Wang, P., Song, J., Xin, S., Chen, S., Tu, C., Wang, W., & Wang, J.
   (2025).** Efficient nearest neighbor search using dynamic programming.
   [12]
   *A 2025 IEEE improvement to nearest-neighbor query cost, applicable to
   speeding up the correspondence step (Step 3) for larger point sets.*

---

## 4. Evaluation metric: L1 distance between uniformly sampled points

This is the assignment's primary scoring criterion (max score 100), so it
is treated as a first-class part of the methodology rather than an
afterthought.

### 4.1 Definition

Given the **expected** curve `C_true(t) = C(t; θ*, M*, X*)` (the unknown
ground-truth parameters the grader holds) and the **predicted** curve
`C_pred(t) = C(t; θ̂, M̂, X̂)` (our fitted parameters), both curves are
resampled at a common set of `M` uniformly spaced parameter values

```
t_m = 6 + (m − 1)·(60 − 6)/(M − 1),   m = 1, …, M
```

so that both curves are compared **at the same `t`**, not via any
re-matching step. Because `t` is uniform and identical for both curves,
this is a direct pointwise comparison, unlike the nearest-neighbor
correspondence used internally during fitting (§5.2) — there, the data has
no known `t`, so correspondence has to be found; here, both curves are
generated with an explicit, shared `t`, so no correspondence step is
needed or appropriate.

The L1 distance is then

```
L1 = (1/M) Σ_{m=1}^{M} ( |x_pred(t_m) − x_true(t_m)| + |y_pred(t_m) − y_true(t_m)| )
```

i.e. the mean of the per-sample Manhattan (L1 / taxicab) distance between
the two curves' `(x, y)` coordinates, averaged over the `M` uniform
samples. Lower is better; `L1 = 0` means the predicted curve coincides
exactly with the expected curve at every sampled `t`.

### 4.2 Why minimizing our internal (L2, unordered) objective also minimizes this (L1, ordered) metric

The optimizer described in §5.3 minimizes a **different but closely
related** quantity: mean squared nearest-neighbor distance between the
*unordered* data points and the candidate curve. Two things justify why
driving that objective to near-zero also drives the final grading metric
(§4.1) to near-zero:

1. **Correct correspondence ⇒ pointwise agreement.** Once `(θ,M,X)` is
   close enough to `(θ*,M*,X*)` that the nearest-neighbor match for each
   data point `p_i` is genuinely its corresponding true point on the
   curve (i.e. `d_i ≈ 0`), then for *any* `t`, `C_pred(t) ≈ C_true(t)` on
   the region of the curve the data actually spans — because both curves
   are smooth (`C∞`) functions of `t`, and agreeing on a dense sample of
   points forces the entire curve segment between them to agree too
   (bounded derivative from the `t·cosθ`, `e^{M|t|}` terms rules out the
   two curves diverging sharply between samples).

2. **Norm equivalence in finite dimensions.** For any finite-dimensional
   real vector `v = (v_x, v_y)`, the L1 and L2 norms satisfy
   `‖v‖₂ ≤ ‖v‖₁ ≤ √2·‖v‖₂`. So a fit that drives the per-point *squared
   Euclidean* residual toward 0 necessarily drives the *L1* residual
   toward 0 as well (and vice versa) — the two metrics can only disagree
   by a bounded constant factor, never in direction. This is why using an
   L2-based objective internally (chosen because it is smooth/
   differentiable-in-correspondence and behaves better inside DE and
   Nelder–Mead — see §5.3) is a valid proxy for the L1 metric actually
   used for grading.

### 4.3 Self-validation procedure used before submitting

Because the true `(θ*, M*, X*)` is unknown to us, we cannot compute the
*official* L1 score directly. As a proxy sanity check (Step 6, §2), we:

1. Resample the fitted curve `C_pred(t)` at `M` uniform `t_m` as in §4.1.
2. For each `t_m`, find the nearest original data point (reusing the
   Step 3 correspondence, §5.2) as a stand-in for `C_true(t_m)`.
3. Compute the same L1 average as in §4.1 between `C_pred(t_m)` and its
   matched data point.

This proxy is expected to be an **upper bound** on the true grading L1 in
the noise-free case (since the nearest matched data point is not
necessarily exactly `C_true(t_m)` when `M` is large relative to `N`), and
gives a concrete, reportable number confirming the fit quality before
finalizing the submitted `(θ, M, X)`.

---

## 5. Mathematics used to estimate/extract θ, M, X

This section derives the objective function and the update equations
actually used to solve for the unknowns, and ties each piece to the
specific cited paper it comes from.

### 5.1 Setup

Let the data be an **unordered** point set
`P = {p_i = (x_i, y_i)}, i = 1..N` (from `xy_data.csv`), and let the
candidate curve be

```
C(t; θ, M, X) = ( x(t), y(t) )
x(t) = t·cosθ − e^{M|t|}·sin(0.3t)·sinθ + X
y(t) = 42 + t·sinθ + e^{M|t|}·sin(0.3t)·cosθ ,   t ∈ [6, 60]
```

Because point order is unknown, we cannot write a direct residual
`p_i − C(t_i)` for a known `t_i`. This is precisely the situation formalized
by Fang & Gossard [3] and Lee [5]: when the parameterization
(the mapping from data index to curve parameter) is unknown, it must be
estimated jointly with the curve's own parameters. Following their
formulation, define the point-to-curve distance as a **projection distance**:

```
d_i(θ, M, X) = min_{t ∈ [6,60]}  ‖ p_i − C(t; θ, M, X) ‖₂
```

i.e. each data point is matched to the *closest* point achievable on the
candidate curve, rather than to a fixed index. This is the continuous
analogue of the discrete correspondence step in Fang & Gossard's
"nonlinear minimization" formulation [3].

### 5.2 Discretized correspondence via nearest-neighbor search

`t` is discretized into a fine grid `{t_1, …, t_K}` (e.g. `K = 5000`) and the
curve is evaluated at each, giving a reference point set
`Q = {C(t_k)}_{k=1}^K`. The projection distance becomes a nearest-neighbor
query:

```
d_i(θ,M,X) = min_{k=1..K} ‖ p_i − C(t_k; θ,M,X) ‖₂
           = ‖ p_i − NN(p_i, Q) ‖₂
```

`NN(p_i, Q)` is computed with a spatial search structure, following the
scalable nearest-neighbor methods surveyed by Muja & Lowe [8]
(build cost `O(K log K)`, query cost `O(log K)` on average). Since this
search is re-run for **every** candidate `(θ,M,X)` evaluated by the
optimizer (thousands of times), query efficiency directly matters; the 2025
result of Wang, Song, Xin, Chen, Tu, Wang & Wang [12] gives an
improved dynamic-programming bound on nearest-neighbor query cost that
applies directly here if `K` is scaled up for higher precision.

### 5.3 Objective function (fitting criterion)

The fitting objective minimized during optimization is the mean squared
projection distance (standard least-squares curve fitting to unorganized
points, per Ma & Kruth [7]):

```
J(θ, M, X) = (1/N) Σ_{i=1}^N d_i(θ, M, X)²
```

This is an **alternating minimization** (an Iterated-Closest-Point-style
scheme): for fixed `(θ,M,X)`, the correspondences `d_i` are recomputed by
nearest-neighbor search (§5.2, the "E-step"); then a global/local optimizer
adjusts `(θ,M,X)` to reduce `J` given those correspondences (the "M-step",
§5.4–5.5). The two steps are repeated as part of each candidate evaluation
inside the optimizer's loop. Why this L2-based `J` is a valid proxy for the
assignment's L1 grading metric is derived in §4.2.

### 5.4 Global search — Differential Evolution update equations

Following Storn & Price [10], encode each candidate solution as a
3-vector `v = (θ, M, X)`. Maintain a population `{v_1, …, v_{NP}}` sampled
uniformly from the given bounds
`θ∈(0°,50°), M∈(−0.05,0.05), X∈(0,100)`. For each generation and each
population member `v_j`:

**Mutation** (choose distinct random indices `r1, r2, r3 ≠ j`):
```
u_j = v_{r1} + F · (v_{r2} − v_{r3})        F ∈ [0, 2]  (differential weight)
```

**Crossover** (binomial), with a randomly chosen index `k_rand` guaranteed
to come from the donor vector:
```
        ⎧ u_j[k]   if rand_k(0,1) ≤ CR  or  k = k_rand
w_j[k] = ⎨
        ⎩ v_j[k]   otherwise                              CR ∈ [0,1]
```

**Selection**:
```
v_j ← w_j   if J(w_j) < J(v_j)   else   v_j unchanged
```

`F` and `CR` are set following the practical guidance surveyed in
Das & Suganthan [2] (typically `F ≈ 0.5–0.8`, `CR ≈ 0.7–0.9` for
smooth low-dimensional problems like this 3-parameter case). DE is used at
this stage specifically because `J(θ,M,X)` is **non-convex**: the
`e^{M|t|}·sin(0.3t)` term creates many local ripples, and `θ` enters through
`sin`/`cos`, so gradient-based or purely local methods risk converging to
the wrong basin.

### 5.5 Local refinement — Nelder–Mead update equations

The best DE candidate seeds a simplex of `n+1 = 4` vertices in the 3D
`(θ,M,X)` space (the seed plus 3 small perturbations). Following
Gao & Han [4], at each iteration:

1. Order vertices so `J(v_1) ≤ J(v_2) ≤ J(v_3) ≤ J(v_4)`.
2. Compute centroid of all but the worst: `v̄ = (1/n) Σ_{i=1}^{n} v_i`.
3. **Reflect**: `v_r = v̄ + α(v̄ − v_4)`
4. If `J(v_1) ≤ J(v_r) < J(v_3)`: accept `v_r`.
5. If `J(v_r) < J(v_1)`, **expand**: `v_e = v̄ + γ(v_r − v̄)`; keep whichever
   of `v_e, v_r` is better.
6. Else **contract**: `v_c = v̄ + ρ(v_4 − v̄)`; accept if better than `v_4`.
7. Otherwise **shrink** all vertices toward `v_1`:
   `v_i ← v_1 + σ(v_i − v_1)`, `i = 2..4`.

Gao & Han's adaptive-parameter recommendation [4] (dimension-dependent, here
`n = 3`) is used instead of the classical fixed `(1, 2, 0.5, 0.5)`:
```
α = 1,   γ = 1 + 2/n,   ρ = 0.75 − 1/(2n),   σ = 1 − 1/n
```
Iteration stops when the simplex's function-value spread
`max_i J(v_i) − min_i J(v_i)` falls below a tolerance `ε` (e.g. `1e‑8`).

### 5.6 Why chain DE → Nelder–Mead specifically

Running DE to convergence and then switching to Nelder–Mead — rather than
using either alone — follows the two-stage hybrid strategy validated by
Wang, Xu & Li [11] for nonlinear parameter identification (they fit
parameters of chaotic systems the same way: DE globally, NM locally) and by
Lin [6], who shows this ordering (global-explore-then-local-exploit)
consistently outperforms single-method optimization on smooth
multi-modal objectives of this size. DE alone converges slowly near the
optimum; NM alone is fast but only locally convergent and highly sensitive
to its starting simplex — DE supplies NM with a good starting point.

### 5.7 Closed-form initial guess (accelerates convergence)

For large `|t|`, note the linear term `t·cosθ` (resp. `t·sinθ`) grows without
bound while the oscillatory term `e^{M|t|}·sin(0.3t)` is bounded in
magnitude by `e^{0.05·60} ≈ 20.1` (using the upper bound of `M`'s given
range). So for `t` near the upper end of `[6,60]`, the data approximately
lie along a straight line through direction `(cosθ, sinθ)`:

```
x(t) ≈ t·cosθ + X ,     y(t) ≈ 42 + t·sinθ
```

This means the **dominant direction of variance** in the point cloud
`{p_i}` approximates `(cosθ, sinθ)`. A standard Principal Component Analysis
on the (mean-centered) data gives this direction directly — the top
eigenvector of the point cloud's covariance matrix — yielding a fast initial
estimate `θ₀ = atan2(v_y, v_x)` where `v = (v_x, v_y)` is the leading
eigenvector. `X₀` follows from matching the mean of `x_i` to the mean of
`t·cosθ₀` over the sampled `t` range, and `M₀` can be seeded at `0` (its
range is narrow and centered near zero). This PCA-based estimate is used
only to **initialize** the DE population's first member (the rest of the
population is still drawn uniformly from the bounds, per §5.4) — the same
role that a good initial parameterization plays in the classical
unorganized-point fitting schemes of Fang & Gossard [3] and Ma & Kruth [7].
*(This initialization step is an original derivation for this assignment,
not itself drawn from a cited source.)*

---

## 6. Result

The final fitted parametric equation (see repo code / commit history for
the exact fitted values and the Desmos link in the assignment submission)
minimizes the L1 distance between uniformly sampled points on the candidate
curve and the given data points, using the DE → Nelder–Mead pipeline
described above, validated per §4.3 before submission.

---

## 7. Full reference list

[1] Ugail, Hassan, and Newton Howard. "Neural Adaptive Tension for Multi-Geometry Curve Subdivision: A Unified Approach." (2026).

[2] Das, Swagatam, and Ponnuthurai Nagaratnam Suganthan. "Differential evolution: A survey of the state-of-the-art." *IEEE Transactions on Evolutionary Computation* 15, no. 1 (2010): 4-31.

[3] Fang, Lian, and David C. Gossard. "Multidimensional curve fitting to unorganized data points by nonlinear minimization." *Computer-Aided Design* 27, no. 1 (1995): 48-58.

[4] Gao, Fuchang, and Lixing Han. "Implementing the Nelder-Mead simplex algorithm with adaptive parameters." *Computational Optimization and Applications* 51, no. 1 (2012): 259-277.

[5] Lee, In-Kwon. "Curve reconstruction from unorganized points." *Computer Aided Geometric Design* 17, no. 2 (2000): 161-177.

[6] Lin, Hongwei. "Hybridizing differential evolution and Nelder-Mead simplex algorithm for global optimization." In *2016 12th International Conference on Computational Intelligence and Security (CIS)*, pp. 198-202. IEEE, 2016.

[7] Ma, Weiyin, and Jean-Pierre Kruth. "Parameterization of randomly measured points for least squares fitting of B-spline curves and surfaces." *Computer-Aided Design* 27, no. 9 (1995): 663-675.

[8] Muja, Marius, and David G. Lowe. "Scalable nearest neighbor algorithms for high dimensional data." *IEEE Transactions on Pattern Analysis and Machine Intelligence* 36, no. 11 (2014): 2227-2240.

[9] Pavelkin, Roman, Luis A. Zavala-Mondragon, Ahmet Ekin, and Fons van der Sommen. "Spline-Based Shape Compression for Interventional Device Tracking." In *International Workshop on Shape in Medical Imaging*, pp. 179-192. Cham: Springer Nature Switzerland, 2025.

[10] Storn, Rainer, and Kenneth Price. "Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces." *Journal of Global Optimization* 11, no. 4 (1997): 341-359.

[11] Wang, Ling, Ye Xu, and Lingpo Li. "Parameter identification of chaotic systems by hybrid Nelder–Mead simplex search and differential evolution algorithm." *Expert Systems with Applications* 38, no. 4 (2011): 3238-3245.

[12] Wang, Pengfei, Jiantao Song, Shiqing Xin, Shuangmin Chen, Changhe Tu, Wenping Wang, and Jiaye Wang. "Efficient nearest neighbor search using dynamic programming." *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2025).

[13] Zou, Qiang, Lizhen Zhu, Jiayu Wu, and Zhijie Yang. "SplineGen: Approximating unorganized points through generative AI." *Computer-Aided Design* 178 (2025): 103809.
