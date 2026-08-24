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

# Gradient-based methods for unconstrained optimization, part 3

Source deck: `Slides/optimization_gradient_descent_advanced_2025.pptx`  
Related page: `gradient_descent_2025.md`

This page covers momentum-based first-order methods: heavy-ball momentum,
Nesterov acceleration, and Adam. These methods are designed for settings where
second-order or quasi-Newton methods are too expensive per iteration.

## 1. Motivation

Trust-region methods and L-BFGS use curvature information to improve search
directions.

- L-BFGS stores recent gradient-displacement pairs to build a low-rank inverse
  Hessian approximation.
- Trust-region methods solve a local quadratic subproblem, often using
  conjugate gradient.

These methods can be effective, but for extremely large design variables
$d_x$ their memory or per-iteration cost may be too high. Momentum-based
methods keep first-order memory cost while trying to reduce oscillation in
ill-conditioned problems.

The representative methods here are:

- heavy-ball momentum
- Nesterov accelerated gradient
- Adam

## 2. Heavy-ball momentum

Gradient descent can oscillate in narrow valleys when the Hessian is
ill-conditioned. Heavy-ball momentum adds inertia by carrying a velocity across
iterations.

One common form is

$$
v_{k+1}=\beta v_k-\alpha \nabla f(x_k),
\qquad
x_{k+1}=x_k+v_{k+1},
$$

where $\alpha>0$ is the step size and $\beta\in[0,1)$ is the momentum
parameter.

Equivalently,

$$
x_{k+1}=x_k-\alpha \nabla f(x_k)+\beta(x_k-x_{k-1}).
$$

For strongly convex quadratics, optimally tuned heavy-ball momentum can achieve
a faster linear rate than plain gradient descent. The catch is that the best
parameters depend on curvature information such as the smallest and largest
eigenvalues of the Hessian.

## 3. Physics interpretation

Heavy-ball momentum is related to a damped mechanical system

$$
\ddot{x}(t)+\gamma \dot{x}(t)+\nabla f(x(t))=0.
$$

The objective $f$ acts like a potential energy, $\dot{x}$ is velocity, and
$\gamma\dot{x}$ is viscous damping. Discretizing this system gives an update
that resembles heavy-ball momentum: keep moving in the previous direction, but
reduce velocity through damping and correct using the gradient force.

## 4. Nesterov accelerated gradient

Nesterov accelerated gradient evaluates the gradient at a look-ahead point
rather than the current iterate. A common form is

$$
y_k=x_k+\beta_k(x_k-x_{k-1}),
$$

$$
x_{k+1}=y_k-\alpha \nabla f(y_k).
$$

The look-ahead gradient gives a correction before the method commits fully to
the momentum direction. For smooth convex objectives, Nesterov acceleration
achieves the optimal first-order convergence rate

$$
f(x_k)-f^\star=O(k^{-2}).
$$

This is faster than the $O(k^{-1})$ rate of basic gradient descent for smooth
convex objectives.

## 5. Adam

Adam is designed for large-scale stochastic optimization, especially neural
network training. It combines momentum with coordinate-wise adaptive scaling.

Given a stochastic gradient $g_k$, Adam maintains exponential moving averages

$$
m_k=\beta_1 m_{k-1}+(1-\beta_1)g_k,
$$

$$
s_k=\beta_2 s_{k-1}+(1-\beta_2)(g_k\odot g_k),
$$

with bias corrections

$$
\hat{m}_k=\frac{m_k}{1-\beta_1^k},
\qquad
\hat{s}_k=\frac{s_k}{1-\beta_2^k}.
$$

The update is

$$
x_{k+1}=x_k-\alpha\frac{\hat{m}_k}{\sqrt{\hat{s}_k}+\epsilon}.
$$

Adam is robust to stochastic gradients and coordinate scaling. It is not always
the best choice for deterministic smooth optimization, especially when
gradients change rapidly or when high-accuracy convergence is required.

## 6. Live code: GD, heavy-ball, NAG, and Adam

The hidden initialization cell defines an ill-conditioned quadratic. Run the
comparison cell to see how momentum changes convergence behavior.

```{code-cell} ipython3
:tags: [thebe-init, hide-input]

import numpy as np
import matplotlib.pyplot as plt

def make_quadratic(cond=100.0, theta=0.6):
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

    return A, x_star, f_star, f, grad

def run_gd(grad, x0, alpha, steps):
    x = x0.astype(float).copy()
    xs = [x.copy()]
    for _ in range(steps):
        x = x - alpha * grad(x)
        xs.append(x.copy())
    return np.array(xs)

def run_heavy_ball(grad, x0, alpha, beta, steps):
    x = x0.astype(float).copy()
    v = np.zeros_like(x)
    xs = [x.copy()]
    for _ in range(steps):
        v = beta * v - alpha * grad(x)
        x = x + v
        xs.append(x.copy())
    return np.array(xs)

def run_nag(grad, x0, alpha, beta, steps):
    x_prev = x0.astype(float).copy()
    x = x0.astype(float).copy()
    xs = [x.copy()]
    for _ in range(steps):
        y = x + beta * (x - x_prev)
        x_next = y - alpha * grad(y)
        x_prev, x = x, x_next
        xs.append(x.copy())
    return np.array(xs)

def run_adam(grad, x0, alpha, steps, beta1=0.9, beta2=0.999, eps=1e-8):
    x = x0.astype(float).copy()
    m = np.zeros_like(x)
    s = np.zeros_like(x)
    xs = [x.copy()]
    for k in range(1, steps + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1**k)
        s_hat = s / (1 - beta2**k)
        x = x - alpha * m_hat / (np.sqrt(s_hat) + eps)
        xs.append(x.copy())
    return np.array(xs)
```

```{code-cell} ipython3
cond = 100.0
steps = 90
x0 = np.array([-1.7, 1.6])
A, x_star, f_star, f, grad = make_quadratic(cond=cond)
L = np.linalg.eigvalsh(A).max()
mu = np.linalg.eigvalsh(A).min()

paths = {
    "GD": run_gd(grad, x0, alpha=1.0/L, steps=steps),
    "Heavy-ball": run_heavy_ball(
        grad, x0,
        alpha=4.0 / (np.sqrt(L) + np.sqrt(mu))**2,
        beta=((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu)))**2,
        steps=steps,
    ),
    "NAG": run_nag(
        grad, x0,
        alpha=1.0/L,
        beta=(np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu)),
        steps=steps,
    ),
    "Adam": run_adam(grad, x0, alpha=0.08, steps=steps),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, xs in paths.items():
    vals = np.array([f(x) - f_star for x in xs])
    axes[0].semilogy(vals, label=name)
    axes[1].plot(xs[:, 0], xs[:, 1], marker=".", markersize=3, label=name)

axes[0].set_xlabel("iteration")
axes[0].set_ylabel(r"$f(x_k)-f^\star$")
axes[0].set_title("Objective error")
axes[0].grid(True, alpha=0.25)
axes[0].legend()

axes[1].scatter([x_star[0]], [x_star[1]], marker="*", s=140, c="black", label="optimum")
axes[1].set_xlabel("$x_1$")
axes[1].set_ylabel("$x_2$")
axes[1].set_title("Optimization paths")
axes[1].axis("equal")
axes[1].grid(True, alpha=0.25)
axes[1].legend()
plt.tight_layout()
plt.show()
```

## 7. Method comparison

| Method | Strength | Main limitation | Typical use |
|---|---|---|---|
| Trust region | Strong local models and globalization | More work per iteration | Medium-scale nonconvex problems |
| L-BFGS | Curvature approximation with limited memory | Not ideal for noisy stochastic gradients | Medium-scale smooth problems |
| Heavy-ball | Reduces zig-zagging in narrow valleys | Sensitive to tuning | Deterministic ill-conditioned problems |
| NAG | Optimal first-order rate for smooth convex objectives | Still sensitive to step size and assumptions | Smooth convex large-scale problems |
| Adam | Robust stochastic optimizer with coordinate scaling | Can be weak for high-accuracy deterministic convergence | Neural-network training |

## 8. Open questions

Open research directions include:

- curvature information that is cheap enough for very large problems
- stable first-order methods for stochastic nonconvex engineering systems
- optimization for nonlinear model predictive control, PDE-constrained design,
  physics-informed neural networks, and neural operators
- better practical bridges between deterministic optimization theory and noisy
  data-driven training

## 9. Stochastic gradient descent

For objectives of the form

$$
f(x)=\frac{1}{N}\sum_{i=1}^N f_i(x),
$$

full gradients can be expensive. Stochastic gradient descent approximates the
gradient using a mini-batch $B_k$:

$$
g_k=\frac{1}{|B_k|}\sum_{i\in B_k}\nabla f_i(x_k).
$$

The update is

$$
x_{k+1}=x_k-\alpha_k g_k.
$$

For nonconvex smooth objectives, SGD is often analyzed through stationarity:
how quickly it reaches a point where $\|\nabla f(x)\|$ is small in expectation.

## 10. Summary

- Momentum methods try to reduce oscillation while keeping first-order memory
  cost.
- Heavy-ball momentum carries velocity from previous steps.
- Nesterov acceleration evaluates the gradient at a look-ahead point and
  achieves $O(k^{-2})$ convergence for smooth convex problems.
- Adam combines momentum and adaptive coordinate scaling for stochastic
  optimization.
- Faster practical methods often trade off robustness, memory, per-iteration
  cost, and theoretical guarantees.

## Ask the lecture notes

```{raw} html
<div class="qa-widget" data-context-file="_static/optimization_gradient_descent_advanced_context.json" data-lecture-id="optimization_gradient_descent_advanced_2025" data-engagement-endpoint="https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/qa">
  <div class="qa-identity">
    <label>
      First name
      <input class="qa-first-name" autocomplete="given-name" />
    </label>
    <label>
      Last name
      <input class="qa-last-name" autocomplete="family-name" />
    </label>
    <label>
      University ID
      <input class="qa-university-id" autocomplete="username" />
    </label>
  </div>
  <label>
    Question
    <textarea class="qa-question" rows="4" placeholder="Ask about heavy-ball, Nesterov acceleration, Adam, or stochastic gradients."></textarea>
  </label>
  <div class="qa-controls">
    <label>
      Model
      <select class="qa-model">
        <option value="openai-fast" selected>openai-fast</option>
        <option value="openai">openai</option>
      </select>
    </label>
    <button class="qa-submit" type="button">Ask</button>
  </div>
  <div class="qa-answer" aria-live="polite"></div>
</div>
```
