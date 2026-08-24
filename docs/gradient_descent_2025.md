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

# Gradient descent for unconstrained optimization

Source deck: `Slides/gradient_descent_2025.pptx`  
Related notebooks: `Demo/Gradient_Optimization_Walkthrough.ipynb`,
`Demo/logreg_optimizers_2d.ipynb`, and
`Demo/portfolio_simplex_optimizers.ipynb`.

## 1. Gradient descent

For a scalar objective $f(x)$, the gradient $\nabla f(x_0)$ gives the local
steepest ascent direction. The negative gradient $-\nabla f(x_0)$ is therefore
the steepest local descent direction.

The basic gradient descent update is

$$
x_k = x_{k-1} - \alpha \nabla f(x_{k-1}), \qquad k = 1,2,\ldots
$$

where $\alpha > 0$ is the step size or learning rate.

## 2. Termination and optimality

Gradient descent usually terminates when

$$
\|\nabla f(x)\| \le \epsilon,
$$

where $\epsilon > 0$ controls solution accuracy.

Necessary condition:

$$
x^\star \text{ local minimizer} \Rightarrow
\nabla f(x^\star)=0,\qquad \nabla^2 f(x^\star)\succeq 0.
$$

Sufficient condition:

$$
\nabla f(x^\star)=0,\qquad \nabla^2 f(x^\star)\succ 0
\Rightarrow x^\star \text{ local minimizer}.
$$

If $f$ is convex and $\nabla f(x^\star)=0$, then $x^\star$ is a global
minimizer. If $f$ is strictly convex, the minimizer is unique.

## 3. Line search

A fixed arbitrary step size is not guaranteed to converge. Line search chooses
$\alpha_k > 0$ so that the step gives enough decrease without becoming
unnecessarily small.

For a descent direction $d_k$, the Armijo sufficient decrease condition is

$$
f(x_k+\alpha_k d_k)
\le f(x_k) + c_1 \alpha_k \nabla f(x_k)^T d_k,
\qquad 0<c_1<1.
$$

The backtracking version starts with $\alpha_0=1$ and repeatedly shrinks
$\alpha \leftarrow \beta \alpha$ until the condition is satisfied.

## 4. Live code: Armijo line search

The hidden initialization cell below defines a quadratic test problem and a
small plotting helper. Use the Live Code button to edit and run the cells in the
browser.

```{code-cell} ipython3
:tags: [thebe-init, hide-input]

import numpy as np
import matplotlib.pyplot as plt

def make_spd_quadratic(cond=25.0, theta=0.7):
    Q = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])
    A = Q @ np.diag([1.0, cond]) @ Q.T
    b = np.array([1.0, -0.5])
    x_star = np.linalg.solve(A, b)
    f_star = 0.5 * x_star @ A @ x_star - b @ x_star

    def f(x):
        return 0.5 * x @ A @ x - b @ x

    def grad(x):
        return A @ x - b

    return A, b, x_star, f_star, f, grad

def armijo_step(f, grad, x, direction, alpha0=1.0, c1=1e-4, beta=0.5):
    alpha = alpha0
    fx = f(x)
    gtd = grad(x) @ direction
    backtracks = 0
    while f(x + alpha * direction) > fx + c1 * alpha * gtd:
        alpha *= beta
        backtracks += 1
    return alpha, backtracks

def gradient_descent(f, grad, x0, steps=40, alpha_mode="armijo", fixed_alpha=0.04):
    x = x0.astype(float).copy()
    xs = [x.copy()]
    fvals = [f(x)]
    alphas = []
    backtracks = []

    for _ in range(steps):
        g = grad(x)
        if np.linalg.norm(g) < 1e-9:
            break
        d = -g
        if alpha_mode == "armijo":
            alpha, bt = armijo_step(f, grad, x, d)
        else:
            alpha, bt = fixed_alpha, 0
        x = x + alpha * d
        xs.append(x.copy())
        fvals.append(f(x))
        alphas.append(alpha)
        backtracks.append(bt)

    return np.array(xs), np.array(fvals), np.array(alphas), np.array(backtracks)
```

```{code-cell} ipython3
cond = 50.0
x0 = np.array([-1.8, 1.8])
A, b, x_star, f_star, f, grad = make_spd_quadratic(cond=cond)

xs_armijo, vals_armijo, alphas, bts = gradient_descent(
    f, grad, x0, steps=35, alpha_mode="armijo"
)
xs_fixed, vals_fixed, _, _ = gradient_descent(
    f, grad, x0, steps=35, alpha_mode="fixed", fixed_alpha=1.0 / cond
)

print(f"condition number: {np.linalg.cond(A):.1f}")
print(f"Armijo final error: {vals_armijo[-1] - f_star:.2e}")
print(f"fixed-step final error: {vals_fixed[-1] - f_star:.2e}")
print(f"mean Armijo backtracks: {bts.mean():.2f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(vals_armijo - f_star, label="GD + Armijo")
ax.semilogy(vals_fixed - f_star, label="GD fixed step")
ax.set_xlabel("iteration")
ax.set_ylabel(r"$f(x_k)-f^\star$")
ax.set_title("Line search stabilizes progress")
ax.grid(True, alpha=0.25)
ax.legend()
plt.show()
```

## 5. Convergence assumptions

The deck states four common assumptions.

A1, L-smoothness:

$$
\|\nabla f(x)-\nabla f(y)\| \le L\|x-y\|,\qquad \forall x,y.
$$

A2, convexity:

$$
f(y) \ge f(x) + \nabla f(x)^T(y-x),\qquad \forall x,y.
$$

A3, $\mu$-strong convexity:

$$
f(y) \ge f(x) + \nabla f(x)^T(y-x)
+ \frac{\mu}{2}\|x-y\|^2,\qquad \mu>0.
$$

A4, Hessian Lipschitz:

$$
\|\nabla^2 f(x)-\nabla^2 f(y)\| \le M\|x-y\|,\qquad M>0.
$$

## 6. Convergence properties

Nonconvex smooth case:

Under A1 and $\alpha_k \in (0,2/L)$, GD decreases $f$ monotonically and

$$
\min_{i \le k}\|\nabla f(x_i)\| = O(k^{-1/2}).
$$

Convex smooth case:

Under A1 and A2 with $\alpha_k = 1/L$,

$$
f(x_k)-f^\star \le \frac{L\|x_0-x^\star\|^2}{2k}.
$$

Strongly convex smooth case:

Under A1 and A3, using the optimal fixed step size
$\alpha_k = 2/(\mu+L)$ and condition number $\kappa=L/\mu$,

$$
f(x_k)-f^\star
\le
\left(1-\kappa^{-1}\right)^k
\left(f(x_0)-f^\star\right).
$$

Ill-conditioned Hessians, where $\kappa \to \infty$, make GD slow.

## 7. Live code: condition number and zig-zagging

```{code-cell} ipython3
def run_condition_demo(condition_numbers=(5, 50, 500), steps=80):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cond in condition_numbers:
        A, b, x_star, f_star, f, grad = make_spd_quadratic(cond=cond)
        xs, vals, _, _ = gradient_descent(
            f, grad, np.array([-1.8, 1.8]), steps=steps, alpha_mode="armijo"
        )
        axes[0].semilogy(vals - f_star, label=f"kappa={cond:g}")
        axes[1].plot(xs[:, 0], xs[:, 1], marker=".", markersize=3, label=f"kappa={cond:g}")

    axes[0].set_title("Objective error")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(r"$f(x_k)-f^\star$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Optimization path")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    plt.tight_layout()
    plt.show()

run_condition_demo()
```

## 8. Conjugate gradient

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

## 9. Newton's method

GD is affected by the local Hessian geometry. Newton's method normalizes the
search direction by the Hessian:

$$
x_k = x_{k-1}
- \alpha [\nabla^2 f(x_{k-1})]^{-1}\nabla f(x_{k-1}).
$$

Under A1, A3, and A4, Newton's method has quadratic local convergence:

$$
f(x_{k+1})-f^\star
\le
\frac{2LM^2}{\mu^3}
\left(f(x_k)-f^\star\right)^2.
$$

## 10. Trust region

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

## 11. BFGS and L-BFGS

BFGS avoids computing the true Hessian by updating a positive definite inverse
Hessian approximation from gradient history. Define

$$
s_k = x_{k+1}-x_k,\qquad
y_k = \nabla f(x_{k+1})-\nabla f(x_k),\qquad
\rho_k = \frac{1}{y_k^T s_k}.
$$

The inverse-Hessian BFGS update is

$$
B_{k+1}
= (I-\rho_k s_k y_k^T)B_k(I-\rho_k y_k s_k^T)
+ \rho_k s_k s_k^T.
$$

L-BFGS stores only the most recent $m$ pairs $(s_k,y_k)$, typically
$m\in[3,20]$, so it is practical when $d_x$ is very large.

## 12. Summary

| Method | Convergence rate | Global convergence? | Cost per iteration |
|---|---:|---:|---:|
| GD | Linear under A1+A3 | Yes with line search under A1 | $O(d_x)$ |
| Newton | Quadratic under A3+A4 | Only under stronger assumptions | $O(d_x^3)$ |
| Trust region | Superlinear or quadratic | Yes for broad smooth cases | $O(kd_x)$ with CG |
| L-BFGS | Superlinear under A3 | Yes with line search under A1 | $O(md_x)$ |

## Q&A

```{raw} html
<div class="qa-widget" data-context-file="_static/gradient_descent_context.json" data-lecture-id="gradient_descent_2025" data-engagement-endpoint="https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/qa">
  <div class="qa-identity">
    <input class="qa-first-name" autocomplete="given-name" placeholder="First name" />
    <input class="qa-last-name" autocomplete="family-name" placeholder="Last name" />
    <input class="qa-university-id" autocomplete="username" placeholder="University ID" />
  </div>
  <textarea class="qa-question" rows="4" placeholder="Ask about Armijo, convergence rates, Newton, BFGS, or the demo notebooks."></textarea>
  <div class="qa-controls">
    <select class="qa-model">
      <option value="openai-fast" selected>openai-fast</option>
      <option value="openai">openai</option>
    </select>
    <button class="qa-submit" type="button">Ask</button>
  </div>
  <div class="qa-answer" aria-live="polite"></div>
</div>
```
