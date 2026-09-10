---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Project 2: Ill-Conditioned Optimization

## Introduction

Project 1 asked you to *formulate* a real-world optimization problem. Project 2 asks
you to *solve* one — and to confront the single most common reason a well-posed problem
is hard to solve in practice: an **ill-conditioned landscape** around the minimizer.

Near a minimizer $x^\star$, a smooth objective looks like a quadratic bowl,

$$
f(x) \approx f(x^\star) + \tfrac{1}{2}(x-x^\star)^\top \nabla^2 f(x^\star)\,(x-x^\star).
$$

The shape of that bowl is set by the eigenvalues of the Hessian
$H = \nabla^2 f(x^\star)$. The **condition number**

$$
\kappa = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}
$$

measures how elongated the bowl is. When $\kappa \approx 1$ the level sets are round and
gradient descent walks straight to the bottom. When $\kappa \gg 1$ the level sets are
long, thin valleys, and gradient descent zig-zags across the valley while creeping along
it. This is exactly the behavior shown in
[part 1 of the gradient-descent notes](gradient_descent_pt1_2025.md): under
$\mu$-strong convexity and $L$-smoothness, gradient descent converges at the rate

$$
f(x_k)-f^\star \le \left(1-\kappa^{-1}\right)^k \left(f(x_0)-f^\star\right),
\qquad \kappa = L/\mu,
$$

so the number of iterations to a fixed accuracy grows *linearly in $\kappa$*.

Ill-conditioning is not an exotic corner case. It is the mechanism behind two of the
most important computational problems today:

- **Deep learning.** Neural-network training is governed by the neural tangent kernel
  (NTK), whose eigenvalues span many orders of magnitude — large curvature for
  low-frequency components of the target, tiny curvature for high-frequency detail. The
  result is the **frequency principle** (networks learn coarse structure first, fine
  detail last), covered in [part 1](gradient_descent_pt1_2025.md).
- **Solving PDEs.** Discretizing a differential operator produces a matrix whose
  condition number grows as the mesh is refined — for the Laplacian, $\kappa \sim
  \mathcal{O}(1/h^2)$, where $h$ is the grid spacing. This makes naive solvers
  impractically slow on fine grids.

The good news: the [gradient-descent notes](gradient_descent_pt2_2025.md) already give
you the full toolkit for fighting ill-conditioning — preconditioning, Newton and
quasi-Newton methods, conjugate gradient, trust regions, and momentum/adaptive methods
(NAG, Adam). This project is where you put that toolkit to work on a problem *you*
choose.

---

## What your team will do

Teams of **1 to 5 students** (same as Project 1). Your team will pick a real-world
inspired optimization problem whose landscape is ill-conditioned around the minimizer,
and deliver **four things**:

1. **Formulate** the problem clearly — decision variables, objective, and constraints as
   explicit mathematical expressions (the Project 1 standard).
2. **Explain the mechanism** — *why* is this landscape ill-conditioned? Trace the large
   condition number to a specific structural source, and prove it is **intrinsic** (see
   [the intrinsic-$\kappa$ rule](#anchor-the-intrinsic-rule)).
3. **Show the effect** — demonstrate, with the [standard diagnostic
   kit](#anchor-standard-diagnostic-kit), how ill-conditioning slows a baseline optimizer.
4. **Propose and demonstrate a solution** — apply an appropriate remedy, explain *why*
   it addresses your specific mechanism, and show the before/after improvement.

You may **reuse your Project 1 problem** if you can make its landscape ill-conditioned
(for example, by moving its constraints into the objective as a penalty — see the menu).

### Timeline

| Date | Event |
|------|-------|
| Sep 16 | Presenting team announced |
| Sep 18 | Zoom rehearsal |
| Sep 28 | In-class presentation |

---

## Mechanism primer: where does ill-conditioning come from?

Before choosing a problem, understand the *sources* of large $\kappa$. Most real cases
fall into one of these families. (The full menu at the end is organized by these
letters.)

| | Family | One-line mechanism |
|---|---|---|
| **A** | Multiscale physical parameters | Stiff and soft components in one system → Hessian eigenvalues span the stiffness ratio |
| **B** | Discretized differential operators | The discrete Laplacian/biharmonic has $\kappa \sim 1/h^2$ (or $1/h^4$) as the mesh refines |
| **C** | Correlated / multiscale features | In least squares the Hessian is $X^\top X$; collinear or unscaled features make it near-singular |
| **D** | NTK / spectral bias | Neural-net training curvature spans decades across frequency → high-frequency detail learned slowly |
| **E** | Nonlinear least squares / "sloppy" models | Near-collinear parameter effects → near-singular Gauss–Newton Hessian |
| **F** | Unit / scale mismatch | Variables in wildly different units → large $\kappa$ (the *trivial*, curable case) |
| **G** | Penalty / barrier reformulations | Enforcing constraints via a penalty of weight $\rho$ drives $\kappa \sim \rho$ |
| **H** | Rank deficiency / gauge freedom | Symmetry or over-parameterization creates flat/near-flat directions |
| **I** | Long-horizon dynamics & control | Sensitivities compound through time (vanishing/exploding), like RNN training |
| **J** | Registration / alignment | Rotation, translation, and scale have very different sensitivities |

(anchor-the-intrinsic-rule)=
## The intrinsic-$\kappa$ rule (required)

Not all large condition numbers are interesting. If you measure one variable in microns
and another in kilometers, $\kappa$ will be enormous but a *single diagonal rescaling*
fixes it completely. That is **trivial** ill-conditioning (family F), and it is not
enough for this project.

Your problem must exhibit **intrinsic** ill-conditioning, defined by a two-part test that
you must report:

1. **Growth under refinement/scaling.** $\kappa$ increases as a structural knob grows —
   grid resolution $N$, polynomial degree, horizon length, feature correlation, or
   penalty weight $\rho$. Show a table or plot of $\kappa$ versus the knob.
2. **Survival under diagonal rescaling.** Apply the best per-coordinate (Jacobi)
   rescaling $H \mapsto D^{-1/2} H D^{-1/2}$ with $D=\operatorname{diag}(H)$. If this
   collapses $\kappa$ to $\mathcal{O}(1)$, your ill-conditioning was trivial. It must
   *survive* — $\kappa$ should stay large.

Family F is acceptable only as a **contrast** subsection (to show what trivial looks
like), never as your core problem.

(anchor-standard-diagnostic-kit)=
## Standard diagnostic kit (required)

To keep results comparable across teams, use this standard protocol to produce evidence
for deliverables 3 and 4. The four measurements are:

- **D1 — Spectrum.** Report $\kappa$ and plot the eigenvalue spectrum of $H$ (or its
  Gauss–Newton / normal-equations approximation) on a log scale.
- **D2 — Intrinsic test.** The two plots from the [intrinsic-$\kappa$
  rule](#anchor-the-intrinsic-rule): $\kappa$ vs. the knob, and $\kappa$ before/after
  diagonal rescaling.
- **D3 — Effect.** Convergence curves ($f(x_k)-f^\star$ and/or $\lVert\nabla
  f(x_k)\rVert$ on a semilog axis) for a **baseline** first-order method. For
  two-variable problems, also plot the iterate path over the contours (the zig-zag).
  Report iterations (or wall-clock) to a fixed tolerance.
- **D4 — Fix.** The same curves *before vs. after* your remedy, plus the change in
  $\kappa$ (for preconditioning/reparameterization) or the changed effective rate (for
  Newton/momentum).

The helper below is generic — it takes any `f`, `grad`, and (optionally) a Hessian, and
implements D1–D4. Adapt it to your problem rather than reinventing it. **You are
encouraged to use coding agents** to build and extend these diagnostics; just verify the
$\kappa$ values by hand on a small case before trusting the plots.

```{code-cell} ipython3
:tags: [thebe-init, hide-input]

import numpy as np
import matplotlib.pyplot as plt


def spectrum(H):
    """D1: eigenvalues (ascending) and condition number of a symmetric H."""
    ev = np.linalg.eigvalsh(H)
    return ev, ev[-1] / ev[0]


def cond_after_diagonal_rescale(H):
    """D2 survival test: kappa after symmetric-Jacobi (diagonal) rescaling."""
    d = np.sqrt(np.abs(np.diag(H)))
    return np.linalg.cond(H / np.outer(d, d))


def gradient_descent(grad, x0, step, tol, maxit, f=None, x_star=None):
    """D3 baseline first-order method. Returns iterate/objective history."""
    x = np.array(x0, dtype=float)
    xs, fs = [x.copy()], []
    ref = np.linalg.norm(grad(x0)) if np.linalg.norm(grad(x0)) > 0 else 1.0
    for _ in range(maxit):
        g = grad(x)
        if f is not None:
            fs.append(f(x))
        if np.linalg.norm(g) <= tol * ref:
            break
        x = x - step * g
        xs.append(x.copy())
    return np.array(xs), np.array(fs)


def conjugate_gradient(A, b, tol=1e-8, maxit=None):
    """A remedy for SPD-quadratic problems: CG needs ~sqrt(kappa) iters, not kappa."""
    n = b.shape[0]
    maxit = maxit or n
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rs = r @ r
    ref = np.linalg.norm(b)
    hist = []
    for k in range(1, maxit + 1):
        Ap = A @ p
        alpha = rs / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        hist.append(np.linalg.norm(r) / ref)
        if np.linalg.norm(r) <= tol * ref:
            break
        rs_new = r @ r
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, np.array(hist)
```

---

## Fully worked example: the Poisson equation as energy minimization

This is a complete example of all four deliverables on the canonical PDE case (family
**B**). It is intentionally simple so you can see the whole pipeline; **do not** submit
this problem — build your own.

**Problem.** Steady-state heat in a 1-D rod on $(0,1)$ with fixed ends and a heat source
$b(x)$ satisfies $-u''(x)=b(x)$, $u(0)=u(1)=0$. Discretizing on $N$ interior points with
spacing $h=1/(N+1)$ turns the second derivative into the tridiagonal matrix $A$ (the
discrete Laplacian), and the physics is equivalent to **minimizing the discrete
energy**

$$
\min_{u\in\mathbb{R}^N}\; f(u) = \tfrac{1}{2}\,u^\top A\,u - b^\top u,
\qquad
A = \frac{1}{h^2}\operatorname{tridiag}(-1,\,2,\,-1).
$$

This is unconstrained, convex, and quadratic — its minimizer solves $Au=b$.

**Mechanism.** The eigenvalues of $A$ are
$\lambda_k = \frac{4}{h^2}\sin^2\!\big(\tfrac{k\pi h}{2}\big)$, $k=1,\dots,N$. The
smallest is $\lambda_1\approx\pi^2$ and the largest is $\lambda_N\approx 4/h^2$, so

$$
\kappa(A)\;\approx\;\frac{4}{\pi^2 h^2}\;=\;\mathcal{O}(N^2).
$$

Refining the mesh (larger $N$) makes the problem *more* ill-conditioned — an intrinsic
mechanism, not a units artifact.

```{code-cell} ipython3
:tags: [thebe-init, hide-input]

def poisson_1d(N):
    """Discrete 1-D Laplacian energy: f(u)=1/2 u'A u - b'u, with two heat sources."""
    h = 1.0 / (N + 1)
    main = 2.0 * np.ones(N) / h**2
    off = -1.0 * np.ones(N - 1) / h**2
    A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    x = np.linspace(h, 1 - h, N)
    b = np.exp(-((x - 0.3) / 0.05)**2) - 0.7 * np.exp(-((x - 0.7) / 0.05)**2)
    return A, b, x
```

### D1 — Spectrum

```{code-cell} ipython3
N = 127
A, b, grid = poisson_1d(N)
ev, kappa = spectrum(A)
print(f"N = {N}:  lambda_min = {ev[0]:.2f},  lambda_max = {ev[-1]:.1f},  kappa = {kappa:.1f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(np.arange(1, N + 1), ev, ".")
ax.set_xlabel("index $k$")
ax.set_ylabel(r"eigenvalue $\lambda_k$")
ax.set_title(f"Spectrum of the discrete Laplacian ($N={N}$, $\\kappa\\approx{kappa:.0f}$)")
ax.grid(True, alpha=0.25)
plt.show()
```

### D2 — Intrinsic test

$\kappa$ grows like $N^2$ (the ratio $\kappa/N^2$ approaches the constant $4/\pi^2\approx
0.405$), and symmetric-Jacobi rescaling leaves it **unchanged** — because $A$ already has
a constant diagonal, so per-coordinate scaling does nothing. The ill-conditioning is
intrinsic.

```{code-cell} ipython3
print(" N   |   kappa    | kappa / N^2 | kappa after diagonal rescale")
print("-----+------------+-------------+------------------------------")
for Ni in [15, 31, 63, 127, 255]:
    Ai, bi, _ = poisson_1d(Ni)
    _, ki = spectrum(Ai)
    kj = cond_after_diagonal_rescale(Ai)
    print(f"{Ni:4d} | {ki:10.1f} |   {ki / Ni**2:.4f}    |  {kj:10.1f}")
```

### D3 — Effect, and D4 — Fix

Gradient descent needs $\mathcal{O}(\kappa)\sim\mathcal{O}(N^2)$ iterations (tens of
thousands here). Conjugate gradient — a remedy you already have from
[part 2](gradient_descent_pt2_2025.md) — needs only
$\mathcal{O}(\sqrt{\kappa})\le N$, because it builds $A$-conjugate directions instead of
fighting the valley. Same problem, two orders of magnitude fewer iterations.

```{code-cell} ipython3
L, mu = ev[-1], ev[0]
u_star = np.linalg.solve(A, b)
f_star = 0.5 * u_star @ A @ u_star - b @ u_star

# D3: baseline gradient descent with the optimal fixed step 2/(L+mu)
f = lambda u: 0.5 * u @ A @ u - b @ u
grad = lambda u: A @ u - b
xs_gd, fs_gd = gradient_descent(
    grad, np.zeros(N), step=2.0 / (L + mu), tol=1e-6, maxit=60000, f=f
)

# D4: conjugate gradient on the same system
_, res_cg = conjugate_gradient(A, b, tol=1e-6)

print(f"kappa = {kappa:.0f},  sqrt(kappa) = {np.sqrt(kappa):.0f}")
print(f"GD iterations to 1e-6: {len(fs_gd)}   (~ order kappa)")
print(f"CG iterations to 1e-6: {len(res_cg)}   (~ order sqrt(kappa), <= N)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(np.maximum(fs_gd - f_star, 1e-16), label="GD (baseline)")
ax.semilogy(res_cg, label="CG (remedy)")
ax.set_xlabel("iteration")
ax.set_ylabel("relative error")
ax.set_title("Ill-conditioning slows GD; CG largely sidesteps it")
ax.legend()
ax.grid(True, alpha=0.25)
plt.show()
```

**Going further.** For this problem the ultimate remedy is a **multigrid** or
incomplete-Cholesky preconditioner, which drives the iteration count to $\mathcal{O}(1)$
independent of $N$. Multigrid works precisely because of the frequency principle from
[part 1](gradient_descent_pt1_2025.md): simple
smoothers kill high-frequency error fast but low-frequency error slowly, so multigrid
solves the low-frequency part on coarse grids where it *looks* high-frequency. A strong
submission would demonstrate this.

---

## Sketched examples

Three more problems, each from a different family, to show the range. These are
deliberately incomplete — enough to see the mechanism and the fix, not a solution.

**Vandermonde polynomial fitting (family C).** Fit a degree-$d$ polynomial to data by
least squares in the *monomial* basis $\{1,x,x^2,\dots\}$. The Hessian is $V^\top V$
where $V$ is the Vandermonde matrix, whose condition number grows *exponentially* with
$d$ because high powers of $x$ become nearly linearly dependent. **Fix:** switch to an
orthogonal basis (Chebyshev/Legendre) — a change of coordinates that is exactly a
preconditioner, dropping $\kappa$ by many orders of magnitude for the same fit.

**Sum-of-exponentials / relaxation-rate fitting (family E).** Fit
$y(t)=\sum_i a_i e^{-\lambda_i t}$ to decay data (spectroscopy, pharmacokinetics, circuit
transients). When two rates $\lambda_i$ are close, their contributions are nearly
collinear, so the Gauss–Newton Hessian is nearly singular — the classic "sloppy model."
**Fix:** Gauss–Newton / Levenberg–Marquardt (trust-region flavored, from
[part 2](gradient_descent_pt2_2025.md)) plus a log-rate reparameterization
and/or mild regularization.

**Penalty reformulation of a constrained problem (family G).** Take a constrained problem
(possibly your Project 1 problem) $\min f(x)$ s.t. $g(x)=0$, and move the constraint into
the objective: $\min f(x)+\tfrac{\rho}{2}\lVert g(x)\rVert^2$. As you increase $\rho$ to
enforce the constraint tightly, the penalty Hessian splits into $\mathcal{O}(1)$
directions (along the constraint) and $\mathcal{O}(\rho)$ directions (normal to it), so
$\kappa\sim\rho$. **Fix:** the **augmented Lagrangian** method, which enforces the
constraint *without* sending $\rho\to\infty$, keeping $\kappa$ bounded. This example
directly explains *why* naive penalty methods are hard.

---

## The full menu

Any of these is a fair starting point. Pick one, adapt it to a concrete real-world
setting, and make it your own — or propose something else with the same intrinsic
structure. Difficulty is ★ (accessible) to ★★★ (ambitious); compute is L/M/H;
"overlap" flags how close it is to the lecture demos. **★ marks recommended
sweet-spot picks.**

| # | Problem | $\kappa$ mechanism | Constr.? | $\kappa$ knob | Effective fix | Diff. | Compute | Overlap |
|---|---|---|---|---|---|---|---|---|
| 1 | Spring/truss min. potential energy, mixed stiffnesses ★ | stiffness ratio in graph-Laplacian Hessian (A) | native uncon. | $k_{\max}/k_{\min}$ | diagonal precond.; Newton (1 step) | ★ | L | Med |
| 2 | 1-D/2-D Poisson (steady heat) as energy min. ★ | discrete Laplacian, $\kappa\sim1/h^2$ (B) | native uncon. | grid resolution $N$ | PCG (incomplete-Chol.); **multigrid** | ★★ | L–M | Low–Med |
| 3 | Beam/plate bending (biharmonic) | $\kappa\sim1/h^4$ (B) | native uncon. | resolution | preconditioned CG, multigrid | ★★★ | M | Low |
| 4 | Image deblurring / deconvolution ★ | blur operator singular values decay → ill-posed (B) | native uncon. | blur width, noise, $\lambda$ | Tikhonov reg.; FFT/circulant precond.; CG + early stop | ★★ | M | Low |
| 5 | Polynomial fit, monomial basis (Vandermonde) ★ | basis Gram matrix $V^\top V$ explodes with degree (C) | native uncon. | polynomial degree | orthogonal basis (Chebyshev) = change of coords | ★ | L | Low |
| 6 | Multicollinear regression / kernel-ridge (GP) | correlated features / Gram matrix near-singular (C) | native uncon. | feature correlation; GP lengthscale | ridge/nugget (Tikhonov); PCA; PCG | ★★ | L–M | Med |
| 7 | Coordinate-MLP fitting a signal/image ★ | NTK spectrum spans decades → spectral bias (D) | native uncon. | target high-freq. content | **Fourier features** as input preconditioner; SIREN | ★★ | M | Med–High |
| 8 | Physics-informed NN (PINN) for a simple PDE | operator + boundary-loss curvatures mismatch (D) | penalty form | frequency / loss weights | loss reweighting; domain scaling; NTK reweighting | ★★★ | M–H | Med |
| 9 | Sum-of-exponentials / relaxation-rate fit ★ | near-collinear components → near-singular Hessian (E) | native uncon. | number/spacing of rates; noise | Gauss–Newton/**Levenberg–Marquardt**; log-reparam. | ★★ | L–M | Low |
| 10 | ODE parameter estimation / "sloppy" models (SIR, PK, kinetics) ★ | eigenvalues span many decades (E) | native uncon. | model size; data sparsity | reparameterize; LM; geodesic accel.; regularize | ★★★ | M | Low |
| 11 | Nondimensionalization / unit-scale demo | disparate units → diagonal $\kappa$ (F) | native uncon. | unit choice | diagonal scaling (cures it) | ★ | L | Low (trivial) |
| 12 | Penalty/barrier reformulation of a constrained problem ★ | penalty Hessian eigen-split, $\kappa\sim\rho$ (G) | reformulated | penalty weight $\rho$ / barrier $t$ | Newton on penalty; **augmented Lagrangian**; continuation | ★★ | L–M | Low |
| 13 | Low-rank matrix factorization / recommender | factor scale ambiguity → flat/unbalanced directions (H) | native uncon. | rank; factor imbalance | balancing reg.; scaled-GD precond.; Riemannian | ★★★ | M | Low |
| 14 | Trajectory optimization / long-horizon control | sensitivity through time (vanish/explode) (I) | native uncon. | horizon length | multiple shooting; Gauss–Newton (iLQR); precond. | ★★★ | M–H | Low |

Notes and existing scaffolding in this repo:

- **Family A (1):** the most physically intuitive $\kappa$ — a chain of stiff and soft
  springs, minimizing potential energy. Quadratic, so Newton solves it in one step. Good
  entry point.
- **Family B (2–4):** the canonical PDE cases. `Demo/topopt_heat_2d.ipynb` and
  `Demo/topology_optimization.ipynb` are related starting points. Deblurring (4) is the
  most visually compelling and introduces regularization of ill-posed inverse problems.
- **Family C (5–6):** ill-conditioning from *data* geometry. The Vandermonde case (5) is
  the cheapest complete project with an elegant fix; multicollinearity/GP (6) connects to
  regularization.
- **Family D (7–8):** the deep-learning cases you saw in
  [part 1](gradient_descent_pt1_2025.md). Fitting a coordinate-MLP to a mixed-frequency
  signal and curing spectral bias with Fourier features (7) is the best "AI" option;
  PINNs (8) are notoriously ill-conditioned and best for ambitious teams.
- **Family E (9–10):** the classic *real* nonlinear ill-conditioning. Sum-of-exponentials
  (9) is tractable; full ODE-parameter "sloppy models" (10) are research-grade but very
  real (systems biology, epidemiology).
- **Family G (12):** the highest-value bridge — reuse a constrained problem, watch it
  become ill-conditioned as $\rho$ grows, and motivate the augmented Lagrangian.
- **Families H, I (13–14):** advanced. `Demo/MPC.ipynb` is a starting point for
  trajectory optimization (14).

**Banned as core problems** (they appear directly in the lecture demos, so they cannot be
your submission — you may use them only as a baseline sanity check): the plain rotated
quadratic with a dialed-in $\kappa$, the Rosenbrock function, and vanilla logistic
regression.

---

## Report requirements

Your report is a **single Markdown file or notebook** in your team's **public** GitHub
repository, with runnable code. It must contain the following sections.

### 1. Problem identification and motivation
Describe a real-world decision or modeling problem, who faces it, and why it matters.

### 2. Formulation
- Decision variables with notation, units, dimensions, bounds, and type
  (continuous/integer/binary).
- The objective as an **explicit** mathematical expression (not "minimize error").
- All constraints as explicit expressions, if any (constrained problems are allowed).
- Problem classification (as in Project 1).

### 3. Ill-conditioning mechanism
- Name the family (A–J) and derive or argue *where* the large $\kappa$ comes from.
- Pass the [intrinsic-$\kappa$ test](#anchor-the-intrinsic-rule) (**D2**): show $\kappa$
  growing with a structural knob, and surviving diagonal rescaling.

### 4. Effect of ill-conditioning
- **D1** (spectrum + $\kappa$) and **D3** (baseline convergence curves, and the iterate
  path for 2-variable problems) using the [standard kit](#anchor-standard-diagnostic-kit).
- Report iterations (or wall-clock) to a fixed tolerance.

### 5. Proposed solution and demonstration
- State your remedy and explain *why* it addresses **your** mechanism (e.g., "a Chebyshev
  basis orthogonalizes the columns that made $V^\top V$ near-singular").
- **D4**: before/after convergence evidence, and the change in $\kappa$ or effective rate.

### 6. Assumptions and simplifications
State modeling assumptions and what your formulation does not capture.

---

## Rubric (100 points)

Computation is **required** this time (it was the bonus in Project 1).

| Category | Points | Full marks |
|---|---|---|
| Problem motivation and real-world relevance | 10 | Clear, well-scoped, a non-expert can follow |
| Formulation (variables / objective / constraints / classification) | 20 | Complete, explicit math; correct classification |
| Ill-conditioning mechanism + intrinsic-$\kappa$ justification | 20 | Correct family; $\kappa$ source derived/argued; passes the D2 intrinsic test |
| Effect demonstration (D1, D3 via the standard kit) | 20 | Spectrum + $\kappa$ reported; baseline slowdown clearly shown and quantified |
| Solution: proposal + before/after demonstration (D4) | 20 | Appropriate remedy; *why* it fits the mechanism is explained; before/after evidence convincing |
| Reproducibility (runnable code, fixed seeds, instructions) | 5 | Another person can rerun and get your numbers |
| Presentation and clarity (organized; math renders on GitHub) | 5 | Professional and readable |

---

## Submission

1. Create a **public** GitHub repository for your team.
2. Write your report as a single Markdown file or notebook, with runnable code.
3. Submit the repository link on **Canvas**.

### Tips

- Use GitHub math: inline `$...$` and display `$$...$$`; test that it renders before
  submitting.
- **Fix all random seeds** so your numbers are reproducible.
- **Use coding agents** to draft your diagnostics and solvers — but verify the $\kappa$
  values and convergence counts by hand on a small case before trusting the plots.
- Start from the [standard diagnostic kit](#anchor-standard-diagnostic-kit) above rather
  than writing everything from scratch.
- Reuse the methods from the gradient-descent notes: preconditioning and CG
  ([part 2](gradient_descent_pt2_2025.md)), Newton and quasi-Newton
  ([part 2](gradient_descent_pt2_2025.md)), and momentum/NAG/Adam
  ([part 3](gradient_descent_pt3_2025.md)).
