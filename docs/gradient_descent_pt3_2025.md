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

Source deck: `Slides/Lecture_notes_2025/optimization_gradient_descent_pt3_2025.pptx`
Related page: `gradient_descent_pt1_2025.md`, `gradient_descent_pt2_2025.md`

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
ill-conditioned. Heavy-ball momentum (Polyak, 1964) adds inertia by carrying
a velocity across iterations.

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

### Physics interpretation

Heavy-ball momentum is related to a damped mechanical system

$$
\ddot{x}(t)+\gamma \dot{x}(t)+\nabla f(x(t))=0.
$$

The objective $f$ acts like a potential energy, $\dot{x}$ is velocity, and
$\gamma\dot{x}$ is viscous damping. Discretizing this system gives an update
that resembles heavy-ball momentum: keep moving in the previous direction, but
reduce velocity through damping and correct using the gradient force.

## 3. Nesterov accelerated gradient

Nesterov accelerated gradient (NAG, 1983) evaluates the gradient at a
look-ahead point rather than the current iterate. A common form is

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

## 4. Adam

Adam (Kingma & Ba, 2015) is designed for large-scale stochastic optimization,
especially neural network training. It combines momentum with coordinate-wise
adaptive scaling.

**Motivation.** Deep learning needs optimizers robust to:

- Nonconvex objectives
- Stochastic gradients (due to random sampling)
- Different scaling per coordinate

Adam combines momentum and adaptive step size.

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

## 5. Live code: optimizer comparison

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

### Deterministic quadratic comparison

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

### Convex full-batch comparison: GD vs Heavy-ball vs NAG

On a larger convex logistic-regression problem ($n=5000$, $d=50$), NAG
achieves the fastest convergence rate among the three first-order methods.

```{code-cell} ipython3
:tags: [thebe-init, hide-input]

def make_logreg_convex(n=5000, d=50, lam=1e-2, seed=42):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=d) * 0.5
    X = rng.normal(size=(n, d)) * 0.3
    logits = X @ w_true + 0.3 * rng.normal(size=n)
    y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(float)

    def f(w):
        z = X @ w
        loss = np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)
        return float(np.mean(loss) + 0.5 * lam * (w @ w))

    def g(w):
        z = X @ w; p = 1.0 / (1.0 + np.exp(-z))
        return X.T @ (p - y) / n + lam * w

    x0 = np.zeros(d)
    return f, g, x0
```

```{code-cell} ipython3
f_lr, g_lr, x0_lr = make_logreg_convex(n=5000, d=50)

steps_lr = 300
alpha_lr = 0.05

fs_gd = []; x = x0_lr.copy()
for _ in range(steps_lr):
    fs_gd.append(f_lr(x)); x = x - alpha_lr * g_lr(x)

fs_hb = []; x = x0_lr.copy(); v = np.zeros_like(x)
for _ in range(steps_lr):
    fs_hb.append(f_lr(x)); v = 0.9 * v - alpha_lr * g_lr(x); x = x + v

fs_nag = []; x = x0_lr.copy(); x_prev = x.copy()
for _ in range(steps_lr):
    fs_nag.append(f_lr(x))
    y = x + 0.9 * (x - x_prev)
    x_prev = x.copy(); x = y - alpha_lr * g_lr(y)

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(fs_gd, label="GD"); ax.semilogy(fs_hb, label="Heavy-ball")
ax.semilogy(fs_nag, label="NAG")
ax.set_xlabel("iteration"); ax.set_ylabel(r"$f(x_k)$")
ax.set_title("Convex logistic regression: GD vs HB vs NAG")
ax.legend(); ax.grid(True, alpha=0.25); plt.tight_layout(); plt.show()
```

### Stochastic comparison: SGD vs Momentum vs Adam

On a stochastic logistic-regression problem, Adam and momentum-SGD
outperform plain SGD. Adam's adaptive scaling is especially effective when
gradient magnitudes vary across coordinates.

```{code-cell} ipython3
:tags: [thebe-init, hide-input]

def make_logreg_stochastic(n=10000, d=100, lam=1e-3, seed=42):
    rng = np.random.default_rng(seed)
    scales = np.exp(rng.uniform(-1, 2, size=d))
    w_true = rng.normal(size=d) * 0.3
    X = rng.normal(size=(n, d)) * scales
    logits = X @ w_true + 0.5 * rng.normal(size=n)
    y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(float)

    def f(w):
        z = X @ w
        loss = np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)
        return float(np.mean(loss) + 0.5 * lam * (w @ w))

    def g_batch(w, idx):
        Xi = X[idx]; yi = y[idx]; z = Xi @ w
        p = 1.0 / (1.0 + np.exp(-z))
        return Xi.T @ (p - yi) / len(idx) + lam * w

    x0 = np.zeros(d)
    return f, g_batch, x0, n
```

```{code-cell} ipython3
f_s, g_s, x0_s, n_s = make_logreg_stochastic(n=10000, d=100)
rng = np.random.default_rng(123)
epochs = 20; batch = 64; steps_per_epoch = n_s // batch
lr = 0.01

def train_sgd(lr):
    x = x0_s.copy(); fs = []
    for ep in range(epochs):
        fs.append(f_s(x))
        perm = rng.permutation(n_s)
        for j in range(steps_per_epoch):
            idx = perm[j*batch:(j+1)*batch]; x = x - lr * g_s(x, idx)
    return fs

def train_momentum(lr, beta=0.9):
    x = x0_s.copy(); v = np.zeros_like(x); fs = []
    for ep in range(epochs):
        fs.append(f_s(x))
        perm = rng.permutation(n_s)
        for j in range(steps_per_epoch):
            idx = perm[j*batch:(j+1)*batch]
            v = beta * v - lr * g_s(x, idx); x = x + v
    return fs

def train_adam(lr, beta1=0.9, beta2=0.999, eps=1e-8):
    x = x0_s.copy(); m = np.zeros_like(x); s = np.zeros_like(x); fs = []; t = 0
    for ep in range(epochs):
        fs.append(f_s(x))
        perm = rng.permutation(n_s)
        for j in range(steps_per_epoch):
            t += 1; idx = perm[j*batch:(j+1)*batch]; gg = g_s(x, idx)
            m = beta1 * m + (1 - beta1) * gg
            s = beta2 * s + (1 - beta2) * (gg * gg)
            mh = m / (1 - beta1**t); sh = s / (1 - beta2**t)
            x = x - lr * mh / (np.sqrt(sh) + eps)
    return fs

rng = np.random.default_rng(123); fs_sgd = train_sgd(lr)
rng = np.random.default_rng(123); fs_mom = train_momentum(lr)
rng = np.random.default_rng(123); fs_adam = train_adam(lr * 0.3)

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(fs_sgd, label="SGD"); ax.semilogy(fs_mom, label="Momentum-SGD")
ax.semilogy(fs_adam, label="Adam")
ax.set_xlabel("epoch"); ax.set_ylabel(r"$f(x)$")
ax.set_title("Stochastic logistic regression: SGD vs Momentum vs Adam")
ax.legend(); ax.grid(True, alpha=0.25); plt.tight_layout(); plt.show()
```

## 6. Summary

| Method | Strength | Main limitation | Typical use |
|---|---|---|---|
| Trust region | Strong local models and globalization | More work per iteration | Medium-scale nonconvex problems |
| L-BFGS | Curvature approximation with limited memory | Not ideal for noisy stochastic gradients | Medium-scale smooth problems |
| Heavy-ball | Reduces zig-zagging in narrow valleys | Sensitive to tuning | Deterministic ill-conditioned problems |
| NAG | Optimal first-order rate for smooth convex objectives | Still sensitive to step size and assumptions | Smooth convex large-scale problems |
| Adam | Robust stochastic optimizer with coordinate scaling | Can be weak for high-accuracy deterministic convergence | Neural-network training |

## 7. Open questions

Open research directions include:

**Curvature on the cheap.** A recipe that (i) costs first-order per step and
memory, (ii) uses curvature enough to beat momentum/Adam on hard problems,
and (iii) stays stable in nonconvex regimes.

Applications include MPC with nonlinear dynamics (e.g., soft robotics),
PDE-constrained optimization (e.g., engineering design), and PINN/NO (e.g.,
climate prediction).

```{image} _static/gd_figs/topology_opt_aircraft.png
:alt: Aircraft fuselage topology optimization (Van den Brink et al. 2024)
:width: 80%
:align: center
```

> Van den Brink et al. (2024)

**Stochastic nonconvex optimization and games.** Complexity results are lacking
for noisy, structured, nonconvex engineering problems.

Applications include reinforcement learning and learning dynamics from
real-world data.

```{image} _static/gd_figs/rl_path_type1.gif
:alt: RL path planning — type 1 agent
:width: 45%
:align: center
```

```{image} _static/gd_figs/rl_path_type0.gif
:alt: RL path planning — type 0 agent with learned dynamics
:width: 45%
:align: center
```

## 8. Stochastic gradient descent

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

## 9. Summary

- Momentum methods try to reduce oscillation while keeping first-order memory
  cost.
- Heavy-ball momentum carries velocity from previous steps.
- Nesterov acceleration evaluates the gradient at a look-ahead point and
  achieves $O(k^{-2})$ convergence for smooth convex problems.
- Adam combines momentum and adaptive coordinate scaling for stochastic
  optimization.
- Faster practical methods often trade off robustness, memory, per-iteration
  cost, and theoretical guarantees.

## Q&A

```{raw} html
<div class="qa-widget" data-context-file="_static/gradient_descent_pt3_context.json" data-lecture-id="gradient_descent_pt3_2025" data-engagement-endpoint="https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/qa">
  <div class="qa-identity">
    <input class="qa-first-name" autocomplete="given-name" placeholder="First name" />
    <input class="qa-last-name" autocomplete="family-name" placeholder="Last name" />
    <input class="qa-university-id" autocomplete="username" placeholder="University ID" />
  </div>
  <div class="qa-input-area">
    <textarea class="qa-question" rows="4" placeholder="Ask about heavy-ball, Nesterov acceleration, Adam, or stochastic gradients."></textarea>
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
