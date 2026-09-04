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

# Gradient-based methods for unconstrained optimization, part 2

Source deck: `Slides/Lecture_notes_2025/optimization_gradient_descent_pt2_2025.pptx`
Related notebooks: `Demo/Gradient_Optimization_Walkthrough.ipynb`,
`Demo/logreg_optimizers_2d.ipynb`, and
`Demo/portfolio_simplex_optimizers.ipynb`.

This page covers methods that use curvature information to improve on gradient
descent: steepest descent with preconditioning, Newton's method, conjugate
gradient, trust-region methods, and quasi-Newton (BFGS / L-BFGS).

## 1. Steepest descent

Standard gradient descent uses the direction $-\nabla f(x)$, which is the
steepest descent direction under the Euclidean norm. More generally, the
steepest descent direction under a norm induced by a symmetric positive
definite matrix $M$ is

$$
d_k = -M^{-1}\nabla f(x_k).
$$

Using $M = I$ recovers standard GD. Using $M = \nabla^2 f(x_k)$ recovers a
Newton step. The quality of the preconditioner $M$ determines how well the
algorithm handles ill-conditioning.

```{image} _static/gd_gifs/sd_identity.gif
:alt: Steepest descent with M=I showing zig-zagging on ill-conditioned quadratic
:width: 80%
:align: center
```

With $M = I$ on an ill-conditioned quadratic ($\kappa = 10^4$), steepest
descent zig-zags along the narrow valley.

```{image} _static/gd_gifs/sd_full_hessian.gif
:alt: Steepest descent with M=A converging in one step
:width: 80%
:align: center
```

With $M = A$ (the exact Hessian), steepest descent solves the quadratic in a
single step — it becomes equivalent to Newton's method.

## 2. Newton's method

GD is affected by the local Hessian geometry. Newton's method normalizes the
search direction by the Hessian:

$$
x_k = x_{k-1}
- \alpha [\nabla^2 f(x_{k-1})]^{-1}\nabla f(x_{k-1}).
$$

**Remark 1.** Newton's method solves strictly convex quadratic problems in one
step.

```{image} _static/gd_gifs/newton_quadratic.gif
:alt: Newton's method converging in one step on a quadratic
:width: 80%
:align: center
```

**Remark 2.** Newton's method does not enjoy global convergence for non-convex
problems. Without globalization (line search or trust region), the unit step
$\alpha=1$ can diverge:

```{image} _static/gd_gifs/newton_pure_rosenbrock.gif
:alt: Pure Newton on Rosenbrock — erratic behavior without line search
:width: 80%
:align: center
```

Adding Armijo backtracking (damped Newton) restores robust convergence:

```{image} _static/gd_gifs/newton_damped_rosenbrock.gif
:alt: Damped Newton with Armijo on Rosenbrock — robust convergence
:width: 80%
:align: center
```

**Remark 3.** Newton's method requires computing the Hessian, which costs
$O(d_x^2)$ storage and $O(d_x^3)$ per solve.

Under A1, A3, and A4, Newton's method has quadratic local convergence:

$$
f(x_{k+1})-f^\star
\le
\frac{2LM^2}{\mu^3}
\left(f(x_k)-f^\star\right)^2.
$$

**Proposition 4** (quadratic convergence). Under A1, A3, and A4, if $x_0$ is
sufficiently close to $x^\star$, then Newton's method with unit step size
satisfies

$$
\|x_{k+1}-x^\star\| \le \frac{M}{2\mu}\|x_k-x^\star\|^2.
$$

### Live code: Newton convergence comparison

```{code-cell} ipython3
:tags: [thebe-init, hide-input]

import numpy as np
import matplotlib.pyplot as plt

def rosenbrock_f(xy):
    x, y = xy[0], xy[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_g(xy):
    x, y = xy[0], xy[1]
    return np.array([-2*(1-x) - 400*x*(y-x**2), 200*(y-x**2)])

def rosenbrock_H(xy):
    x, y = xy[0], xy[1]
    return np.array([[2 - 400*(y - 3*x**2), -400*x],
                     [-400*x, 200.0]])

def armijo_bt(f, g, x, p, alpha0=1.0, c=1e-4, beta=0.5, max_bt=60):
    slope = g(x) @ p
    if slope >= 0: return 0.0
    fx = f(x); alpha = alpha0
    for _ in range(max_bt):
        if f(x + alpha * p) <= fx + c * alpha * slope: return alpha
        alpha *= beta
    return 0.0

def newton_run(f, g, H, x0, variant="damped", maxit=80, tol=1e-10):
    x = x0.copy(); xs = [x.copy()]; fs = [f(x)]; gn = [np.linalg.norm(g(x))]
    for _ in range(maxit):
        gx = g(x); Hx = H(x)
        if np.linalg.norm(gx) < tol: break
        try: p = -np.linalg.solve(Hx, gx)
        except: p = -gx
        if variant == "pure":
            alpha = 1.0
        else:
            alpha = armijo_bt(f, g, x, p)
            if alpha == 0: break
        x = x + alpha * p
        if not np.all(np.isfinite(x)): break
        xs.append(x.copy()); fs.append(f(x)); gn.append(np.linalg.norm(g(x)))
    return np.array(fs), np.array(gn)
```

```{code-cell} ipython3
x0 = np.array([-1.2, 1.0])
fs_pure, gn_pure = newton_run(rosenbrock_f, rosenbrock_g, rosenbrock_H, x0, "pure")
fs_damp, gn_damp = newton_run(rosenbrock_f, rosenbrock_g, rosenbrock_H, x0, "damped")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].semilogy(fs_pure, label="Pure Newton"); axes[0].semilogy(fs_damp, label="Damped Newton")
axes[0].set_xlabel("iteration"); axes[0].set_ylabel(r"$f(x_k)$")
axes[0].set_title("Newton on Rosenbrock: objective"); axes[0].legend(); axes[0].grid(True, alpha=0.25)
axes[1].semilogy(gn_pure, label="Pure Newton"); axes[1].semilogy(gn_damp, label="Damped Newton")
axes[1].set_xlabel("iteration"); axes[1].set_ylabel(r"$\|\nabla f(x_k)\|$")
axes[1].set_title("Newton on Rosenbrock: gradient norm"); axes[1].legend(); axes[1].grid(True, alpha=0.25)
plt.tight_layout(); plt.show()
```

## 3. Conjugate gradient

Conjugate gradient addresses the slow convergence of GD on quadratic problems:

$$
\min_x f(x) = \frac{1}{2}x^T A x - b^T x,
$$

where $A$ is symmetric positive definite. This optimization problem is
equivalent to solving the linear system

$$
Ax=b.
$$

GD convergence depends strongly on the condition number $\kappa(A)$.
Conjugate gradient builds $A$-conjugate directions and can solve the exact
quadratic problem in at most $d_x$ iterations in exact arithmetic.

GD with exact line search takes orthogonal steps that zig-zag inefficiently:

```{image} _static/gd_gifs/gd_exact_ls.gif
:alt: GD with exact line search showing orthogonal zig-zag steps
:width: 80%
:align: center
```

Conjugate gradient solves the same 2D problem in exactly 2 steps using
$A$-conjugate directions:

```{image} _static/gd_gifs/cg_conjugate.gif
:alt: CG solving 2D quadratic in exactly 2 A-conjugate steps
:width: 80%
:align: center
```

**Remark.** CG has a linear convergence rate but is less affected by
ill-conditioning than gradient descent. On a quadratic with $d_x$ variables,
CG terminates in at most $d_x$ steps (in exact arithmetic).

## 4. Summary of gradient-based algorithms

| Method | Convergence rate | Global convergence? | Cost per iteration |
|---|---:|---:|---:|
| GD | Linear under A1+A3 | Yes with line search under A1 | $O(d_x)$ |
| CG (linear) | Linear, at most $d_x$ steps | Yes for SPD quadratics | $O(d_x)$ |
| Newton | Quadratic under A3+A4 | Only under stronger assumptions | $O(d_x^3)$ |

## 5. Trust region

At $x_k$, trust-region methods solve a local quadratic model inside a radius
$\Delta_k$:

$$
\begin{aligned}
\min_s\quad & \nabla f(x_k)^T s
+ \frac{1}{2}s^T\nabla^2 f(x_k)s \\
\text{s.t.}\quad & \|s\| \le \Delta_k .
\end{aligned}
$$

The solution can be written as

$$
s_k =
-[\nabla^2 f(x_k)+\mu I]^{-1}\nabla f(x_k),
$$

where $\mu \ge 0$ enforces $\|s_k\|\le \Delta_k$. Large trust regions recover a
Newton-like step. Small trust regions recover a gradient-descent-like step.

```{image} _static/gd_gifs/dogleg_quadratic.gif
:alt: Dogleg trust region on ill-conditioned quadratic
:width: 80%
:align: center
```

### Live code: trust-region subsolver comparison

```{code-cell} ipython3
:tags: [thebe-init, hide-input]

import math

def cauchy_pt(g, H, Delta):
    gnorm = np.linalg.norm(g)
    if gnorm == 0: return np.zeros_like(g)
    Hgg = float(g @ (H @ g))
    tau = min(gnorm**3 / (Delta * Hgg), 1.0) if Hgg > 0 else 1.0
    return -(tau * Delta / gnorm) * g

def dogleg_pt(g, H, Delta):
    try: pN = -np.linalg.solve(H, g)
    except: return cauchy_pt(g, H, Delta)
    gnorm = np.linalg.norm(g)
    if gnorm == 0: return np.zeros_like(g)
    alpha_sd = (g @ g) / max(g @ (H @ g), 1e-16)
    pU = -alpha_sd * g
    if np.linalg.norm(pN) <= Delta: return pN
    if np.linalg.norm(pU) >= Delta: return -(Delta / gnorm) * g
    d = pN - pU; a = d@d; b = 2*(pU@d); c = pU@pU - Delta**2
    disc = b*b - 4*a*c
    tau = (-b + math.sqrt(max(0, disc))) / (2*a) if a > 0 else 0
    return pU + tau * d

def cg_steihaug(g, H, Delta, tol=1e-10, maxit=None):
    n = g.shape[0]
    if maxit is None: maxit = n
    x = np.zeros_like(g); r = g.copy(); d = -r
    if np.linalg.norm(r) < tol: return x
    for _ in range(maxit):
        Hd = H @ d; dHd = float(d @ Hd)
        if dHd <= 0:
            a = d@d; b = 2*(x@d); c = x@x - Delta**2
            disc = b*b - 4*a*c
            tau = (-b + math.sqrt(max(0, disc))) / (2*a) if a > 0 else 0
            return x + tau * d
        alpha = (r @ r) / dHd
        x_new = x + alpha * d
        if np.linalg.norm(x_new) >= Delta:
            a = d@d; b = 2*(x@d); c = x@x - Delta**2
            disc = b*b - 4*a*c
            tau = (-b + math.sqrt(max(0, disc))) / (2*a) if a > 0 else 0
            return x + tau * d
        r_new = r + alpha * Hd
        if np.linalg.norm(r_new) < tol: return x_new
        beta = (r_new @ r_new) / (r @ r)
        d = -r_new + beta * d; x, r = x_new, r_new
    return x

def trust_region_run(f, g, H, x0, method="dogleg", Delta0=1.0, eta=0.1, maxit=200, tol=1e-10):
    x = x0.copy(); fs = [f(x)]; gn = [np.linalg.norm(g(x))]; Delta = Delta0
    subsolver = {"cauchy": cauchy_pt, "dogleg": dogleg_pt, "cg": cg_steihaug}[method]
    for _ in range(maxit):
        gx = g(x); Hx = H(x)
        if np.linalg.norm(gx) < tol: break
        p = subsolver(gx, Hx, Delta)
        pred = -float(gx @ p) - 0.5 * float(p @ (Hx @ p))
        if pred <= 0: Delta = max(1e-12, 0.25 * Delta); continue
        f_new = f(x + p); ared = fs[-1] - f_new; rho = ared / pred
        if rho < 0.25: Delta = max(1e-12, 0.25 * Delta)
        elif rho > 0.75 and abs(np.linalg.norm(p) - Delta) < 1e-12: Delta = min(1e3, 2 * Delta)
        if rho > eta and np.isfinite(f_new):
            x = x + p; fs.append(f_new); gn.append(np.linalg.norm(g(x)))
    return np.array(fs), np.array(gn)
```

```{code-cell} ipython3
x0 = np.array([-1.2, 1.0])
fs_c, gn_c = trust_region_run(rosenbrock_f, rosenbrock_g, rosenbrock_H, x0, "cauchy")
fs_d, gn_d = trust_region_run(rosenbrock_f, rosenbrock_g, rosenbrock_H, x0, "dogleg")
fs_s, gn_s = trust_region_run(rosenbrock_f, rosenbrock_g, rosenbrock_H, x0, "cg")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for fs, lab in [(fs_c, "Cauchy"), (fs_d, "Dogleg"), (fs_s, "CG-Steihaug")]:
    axes[0].semilogy(fs, label=lab)
axes[0].set_xlabel("iteration"); axes[0].set_ylabel(r"$f(x_k)$")
axes[0].set_title("Trust region on Rosenbrock: objective"); axes[0].legend(); axes[0].grid(True, alpha=0.25)
for gn, lab in [(gn_c, "Cauchy"), (gn_d, "Dogleg"), (gn_s, "CG-Steihaug")]:
    axes[1].semilogy(gn, label=lab)
axes[1].set_xlabel("iteration"); axes[1].set_ylabel(r"$\|\nabla f(x_k)\|$")
axes[1].set_title("Trust region on Rosenbrock: gradient norm"); axes[1].legend(); axes[1].grid(True, alpha=0.25)
plt.tight_layout(); plt.show()
```

## 6. BFGS

BFGS avoids computing the true Hessian by updating a positive definite
Hessian approximation from gradient history. Define

$$
s_k = x_{k+1}-x_k,\qquad
y_k = \nabla f(x_{k+1})-\nabla f(x_k),\qquad
\rho_k = \frac{1}{y_k^T s_k}.
$$

The Hessian-form BFGS update is

$$
B_{k+1}
= B_k
- \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}
+ \frac{y_k y_k^T}{y_k^T s_k}.
$$

The inverse-Hessian form avoids the linear solve:

$$
H_{k+1}
= (I-\rho_k s_k y_k^T)H_k(I-\rho_k y_k s_k^T)
+ \rho_k s_k s_k^T.
$$

```{image} _static/gd_gifs/bfgs_rosenbrock.gif
:alt: BFGS trajectory on Rosenbrock showing quasi-Newton convergence
:width: 80%
:align: center
```

## 7. L-BFGS

L-BFGS stores only the most recent $m$ pairs $(s_k,y_k)$, typically
$m\in[3,20]$, so it is practical when $d_x$ is very large. The two-loop
recursion computes $H_k \nabla f(x_k)$ in $O(md_x)$ time without forming
$H_k$ explicitly.

### Live code: quasi-Newton comparison

```{code-cell} ipython3
:tags: [thebe-init, hide-input]

def bfgs_run(f, g, x0, maxit=300, tol=1e-10):
    x = x0.copy(); n = x.size; B = np.eye(n)
    fs = [f(x)]; gn = [np.linalg.norm(g(x))]
    for _ in range(maxit):
        gx = g(x)
        if np.linalg.norm(gx) < tol: break
        try: p = -np.linalg.solve(B, gx)
        except: p = -gx
        alpha = armijo_bt(f, g, x, p)
        if alpha == 0: break
        x_new = x + alpha * p; s = x_new - x; y = g(x_new) - gx
        Bs = B @ s; sty = float(s @ y); sBs = float(s @ Bs)
        if sty < 0.2 * sBs:
            theta = (0.8 * sBs) / max(1e-16, sBs - sty)
            y = theta * y + (1 - theta) * Bs; sty = float(s @ y)
        if sty > 1e-16:
            B = B - np.outer(Bs, Bs) / max(1e-16, sBs) + np.outer(y, y) / sty
        x = x_new; fs.append(f(x)); gn.append(np.linalg.norm(g(x)))
    return np.array(fs), np.array(gn)

def lbfgs_run(f, g, x0, m=10, maxit=300, tol=1e-10):
    x = x0.copy(); n = x.size
    fs = [f(x)]; gn = [np.linalg.norm(g(x))]
    ss = []; ys = []; rhos = []
    for _ in range(maxit):
        gx = g(x)
        if np.linalg.norm(gx) < tol: break
        q = gx.copy(); alphas_l = []
        for s, y, rho in zip(reversed(ss), reversed(ys), reversed(rhos)):
            a = rho * (s @ q); alphas_l.append(a); q = q - a * y
        alphas_l.reverse()
        gamma = (ss[-1] @ ys[-1]) / (ys[-1] @ ys[-1]) if ss else 1.0
        r = gamma * q
        for i, (s, y, rho) in enumerate(zip(ss, ys, rhos)):
            b = rho * (y @ r); r = r + (alphas_l[i] - b) * s
        p = -r
        alpha = armijo_bt(f, g, x, p)
        if alpha == 0: break
        x_new = x + alpha * p; s = x_new - x; y = g(x_new) - gx
        sty = float(s @ y)
        if sty > 1e-16:
            if len(ss) >= m: ss.pop(0); ys.pop(0); rhos.pop(0)
            ss.append(s); ys.append(y); rhos.append(1.0 / sty)
        x = x_new; fs.append(f(x)); gn.append(np.linalg.norm(g(x)))
    return np.array(fs), np.array(gn)

def sr1_run(f, g, H_func, x0, maxit=300, tol=1e-10, Delta0=1.0):
    x = x0.copy(); n = x.size; B = np.eye(n)
    fs = [f(x)]; gn = [np.linalg.norm(g(x))]; Delta = Delta0
    for _ in range(maxit):
        gx = g(x)
        if np.linalg.norm(gx) < tol: break
        p = cg_steihaug(gx, B, Delta)
        pred = -float(gx @ p) - 0.5 * float(p @ (B @ p))
        if pred <= 0: Delta = max(1e-12, 0.25 * Delta); continue
        f_new = f(x + p); ared = fs[-1] - f_new; rho = ared / pred
        if rho < 0.25: Delta = max(1e-12, 0.25 * Delta)
        elif rho > 0.75 and abs(np.linalg.norm(p) - Delta) < 1e-12: Delta = min(1e3, 2 * Delta)
        if rho > 0.1 and np.isfinite(f_new):
            x_new = x + p; s = x_new - x; y = g(x_new) - gx
            ymBs = y - B @ s; denom = float(ymBs @ s)
            if abs(denom) > 1e-8 * np.linalg.norm(s) * np.linalg.norm(ymBs):
                B = B + np.outer(ymBs, ymBs) / denom
            x = x_new; fs.append(f(x)); gn.append(np.linalg.norm(g(x)))
    return np.array(fs), np.array(gn)
```

```{code-cell} ipython3
x0 = np.array([-1.2, 1.0])
fs_bfgs, gn_bfgs = bfgs_run(rosenbrock_f, rosenbrock_g, x0)
fs_lbfgs, gn_lbfgs = lbfgs_run(rosenbrock_f, rosenbrock_g, x0)
fs_sr1, gn_sr1 = sr1_run(rosenbrock_f, rosenbrock_g, rosenbrock_H, x0)
fs_newt, gn_newt = newton_run(rosenbrock_f, rosenbrock_g, rosenbrock_H, x0, "damped")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for fs, lab in [(fs_bfgs, "BFGS"), (fs_lbfgs, "L-BFGS"), (fs_sr1, "SR1"), (fs_newt, "Newton")]:
    axes[0].semilogy(fs, label=lab)
axes[0].set_xlabel("iteration"); axes[0].set_ylabel(r"$f(x_k)$")
axes[0].set_title("Quasi-Newton comparison on Rosenbrock"); axes[0].legend(); axes[0].grid(True, alpha=0.25)
for gn, lab in [(gn_bfgs, "BFGS"), (gn_lbfgs, "L-BFGS"), (gn_sr1, "SR1"), (gn_newt, "Newton")]:
    axes[1].semilogy(gn, label=lab)
axes[1].set_xlabel("iteration"); axes[1].set_ylabel(r"$\|\nabla f(x_k)\|$")
axes[1].set_title("Gradient norm"); axes[1].legend(); axes[1].grid(True, alpha=0.25)
plt.tight_layout(); plt.show()
```

## 8. Summary of gradient-based algorithms

| Method | Convergence rate | Global convergence? | Cost per iteration |
|---|---:|---:|---:|
| GD | Linear under A1+A3 | Yes with line search under A1 | $O(d_x)$ |
| CG (linear) | Linear, at most $d_x$ steps | Yes for SPD quadratics | $O(d_x)$ |
| Newton | Quadratic under A3+A4 | Only under stronger assumptions | $O(d_x^3)$ |
| Trust region | Superlinear or quadratic | Yes for broad smooth cases | $O(kd_x)$ with CG |
| BFGS | Superlinear under A3 | Yes with line search under A1 | $O(d_x^2)$ |
| L-BFGS | Superlinear under A3 | Yes with line search under A1 | $O(md_x)$ |

## 9. Exercises

1. Derive the exact line-search step size $\alpha_k$ for the quadratic
   $f(x)=\tfrac{1}{2}x^TAx - b^Tx$ along the direction $d_k=-\nabla f(x_k)$.

2. Show that, for the quadratic case, consecutive GD search directions with
   exact line search are orthogonal: $d_{k+1}^T d_k = 0$.

3. Let $f(x)=\tfrac{1}{2}x^TAx - b^Tx$ with $A$ SPD. Show that the CG
   directions satisfy $A$-conjugacy: $d_i^T A d_j = 0$ for $i\ne j$.

4. Verify that Newton's method applied to a strictly convex quadratic
   converges in one step regardless of the starting point.

5. Consider the BFGS update $B_{k+1} = B_k - \frac{B_k s_k s_k^T B_k}{s_k^T
   B_k s_k} + \frac{y_k y_k^T}{y_k^T s_k}$. Show that $B_{k+1}$ is
   symmetric positive definite whenever $B_k$ is symmetric positive definite
   and $s_k^T y_k > 0$.

## Q&A

```{raw} html
<div class="qa-widget" data-context-file="_static/gradient_descent_pt2_context.json" data-lecture-id="gradient_descent_pt2_2025" data-engagement-endpoint="https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/qa">
  <div class="qa-identity">
    <input class="qa-first-name" autocomplete="given-name" placeholder="First name" />
    <input class="qa-last-name" autocomplete="family-name" placeholder="Last name" />
    <input class="qa-university-id" autocomplete="username" placeholder="University ID" />
  </div>
  <div class="qa-input-area">
    <textarea class="qa-question" rows="4" placeholder="Ask about Newton, CG, trust region, BFGS, or the demo notebooks."></textarea>
    <div class="qa-controls">
      <select class="qa-model">
        <option value="gemini-3.5-flash-lite" selected>Gemini 3.5 Flash Lite</option>
        <option value="gemini-3.5-flash">Gemini 3.5 Flash</option>
      </select>
      <button class="qa-submit" type="button" aria-label="Ask"><svg viewBox="0 0 16 16" fill="none"><path d="M3 13L8 3L13 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></button>
    </div>
  </div>
  <div class="qa-answer" aria-live="polite"></div>
</div>
```
