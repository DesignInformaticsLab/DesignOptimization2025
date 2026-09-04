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

# Gradient-based methods for unconstrained optimization, part 1

Source deck: `Slides/Lecture_notes_2025/optimization_gradient_descent_pt1_2025.pptx`
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

```{image} _static/gd_figs/surface_landscape.gif
:alt: Non-convex optimization landscape
:width: 50%
:align: center
```

The step size $\alpha$ is critical: too small and the algorithm is slow; too
large and it can diverge.

```{image} _static/gd_gifs/gd_good_step.gif
:alt: GD trajectory with good step size showing zig-zagging
:width: 80%
:align: center
```

With a step size $\alpha = 1.9/L$ (just under the stability limit), GD
converges but zig-zags in the narrow valley of an ill-conditioned quadratic.

```{image} _static/gd_gifs/gd_diverge.gif
:alt: GD trajectory with too-large step size diverging
:width: 80%
:align: center
```

When $\alpha = 2.1/L$ exceeds the stability bound $2/L$, the iterates
diverge.

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

**Exact line search.** For a descent direction $d_k$, the exact step solves

$$
\alpha_k = \arg\min_{\alpha > 0} f(x_k + \alpha d_k).
$$

On quadratics this has a closed form. On general objectives it is expensive.

**Armijo line search.** A cheaper alternative requires only sufficient decrease.
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

On the Rosenbrock function, Armijo backtracking navigates the narrow curved
valley:

```{image} _static/gd_gifs/armijo_rosenbrock.gif
:alt: GD with Armijo line search on the Rosenbrock function
:width: 80%
:align: center
```

## 5. Summary so far

| Component | Purpose |
|---|---|
| Gradient descent | Follow $-\nabla f$ to decrease the objective |
| Termination | Stop when $\|\nabla f\| \le \epsilon$ |
| Exact line search | Minimize along the ray (closed form on quadratics) |
| Armijo backtracking | Cheap sufficient decrease guarantee |

## 6. Convergence assumptions

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

## 7. Convergence properties

**Nonconvex smooth case:**

Under A1 and $\alpha_k \in (0,2/L)$, GD decreases $f$ monotonically and

$$
\min_{i \le k}\|\nabla f(x_i)\| = O(k^{-1/2}).
$$

**Convex smooth case:**

Under A1 and A2 with $\alpha_k = 1/L$,

$$
f(x_k)-f^\star \le \frac{L\|x_0-x^\star\|^2}{2k}.
$$

**Strongly convex smooth case:**

Under A1 and A3, using the optimal fixed step size
$\alpha_k = 2/(\mu+L)$ and condition number $\kappa=L/\mu$,

$$
f(x_k)-f^\star
\le
\left(1-\kappa^{-1}\right)^k
\left(f(x_0)-f^\star\right).
$$

Ill-conditioned Hessians, where $\kappa \to \infty$, make GD slow.

## 8. Live code: condition number and zig-zagging

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

## 9. Frequency principle and neural networks

The convergence rate $\left(1-\kappa^{-1}\right)^k$ implies that components of
the gradient along eigenvectors with large eigenvalues are resolved first,
while low-curvature directions converge slowly. This phenomenon has an
important analogy in deep learning.

Rahaman et al. (2019) showed that neural networks trained with GD learn
low-frequency components of the target function before high-frequency
components — the **frequency principle**. This is because the loss landscape
has larger curvatures (eigenvalues) for low-frequency basis functions.

```{image} _static/gd_figs/frequency_principle_iterations.png
:alt: NN training progressively fits low to high frequency (Rahaman et al. 2019)
:width: 100%
:align: center
```

```{image} _static/gd_figs/frequency_principle_fft.gif
:alt: FFT of DNN output vs target — low frequencies fit first
:width: 50%
:align: center
```

Xu et al. (2020) demonstrated the same effect on image reconstruction: the
network first resolves the coarse structure (low frequencies) and only
gradually sharpens fine detail.

```{image} _static/gd_figs/frequency_principle_cameraman.png
:alt: NN image reconstruction showing low-frequency bias (Xu et al. 2020)
:width: 100%
:align: center
```

> Xu, Z.-Q.J., Zhang, Y., Luo, T., Xiao, Y., Ma, Z.: Frequency principle:
> Fourier analysis sheds light on deep neural networks. *Communications in
> Computational Physics* 28(5), 1746–1767 (2020).

**Remark.** Neural networks have large curvatures for low-frequency basis and
resolve these residuals first, leading to a low-frequency bias. Diffusion
methods alleviate this bias by forcing high-frequency learning (analogous to
preconditioning). Classical PDE solvers exhibit the opposite behavior: the
physics generates large curvatures for high-frequency residuals.

## 10. Convergence proofs

**Proposition 1** (nonconvex smooth). Under A1, if $\alpha \in (0, 2/L)$, then
GD produces iterates satisfying

$$
\min_{0 \le i \le k} \|\nabla f(x_i)\|^2
\le \frac{f(x_0)-f^\star}{(k+1)\alpha(1-\alpha L/2)}.
$$

**Proposition 2** (convex smooth). Under A1 and A2, with $\alpha = 1/L$,

$$
f(x_k)-f^\star \le \frac{L\|x_0-x^\star\|^2}{2k}.
$$

**Proposition 3** (strongly convex smooth). Under A1 and A3, with
$\alpha = 2/(\mu+L)$,

$$
f(x_k)-f^\star
\le \frac{L}{2}
\left(\frac{\kappa-1}{\kappa+1}\right)^{2k}
\|x_0-x^\star\|^2.
$$

## Q&A

```{raw} html
<div class="qa-widget" data-context-file="_static/gradient_descent_pt1_context.json" data-lecture-id="gradient_descent_pt1_2025" data-engagement-endpoint="https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/qa">
  <div class="qa-identity">
    <input class="qa-first-name" autocomplete="given-name" placeholder="First name" />
    <input class="qa-last-name" autocomplete="family-name" placeholder="Last name" />
    <input class="qa-university-id" autocomplete="username" placeholder="University ID" />
  </div>
  <div class="qa-input-area">
    <textarea class="qa-question" rows="4" placeholder="Ask about gradient descent, Armijo, convergence rates, or the demo notebooks."></textarea>
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
