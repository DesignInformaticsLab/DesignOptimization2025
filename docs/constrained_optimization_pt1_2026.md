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

# Constrained optimization, part 1: KKT conditions

Source deck: `Slides/optimization_kkt_pt1_2026.pptx`

This page covers first- and second-order optimality conditions for constrained
problems: the Karush-Kuhn-Tucker (KKT) conditions, their geometric meaning,
a derivation from the unconstrained case, and the interpretation of Lagrange
multipliers as sensitivities. It ends with exercises drawn from optimal
control, PDE-constrained design, neural network training, communications,
and economics.

## 1. The constrained problem

We consider the constrained problem

$$
\textbf{(P)}\qquad
\min_{x \in \mathcal{X}} \; f(x)
\quad \text{s.t.} \quad
g(x) \le 0, \quad h(x) = 0,
$$

under the standing assumptions

1. $f:\mathbb{R}^n \to \mathbb{R}$, $g:\mathbb{R}^n \to \mathbb{R}^p$,
   $h:\mathbb{R}^n \to \mathbb{R}^m$;
2. $f \in C^2$ (Hessian exists) and is bounded; $g, h \in C^1$ (gradients
   exist);
3. the support $\mathcal{X}$ is compact (closed and bounded);
4. a feasible solution exists: $\exists\, x \in \mathcal{X}$ with
   $g(x) \le 0$ and $h(x) = 0$.

**Notation.** As in the derivations below, we treat $\nabla_x f$ (or
$\partial f / \partial x$) as a $1 \times n$ row vector, and $\nabla_x g$,
$\nabla_x h$ as the $p \times n$ and $m \times n$ Jacobians. Then
$\mu^\top \nabla_x g = \sum_i \mu_i \nabla_x g_i$ is again a row vector.

## 2. KKT points

A **KKT point** $(x^*, \mu^*, \lambda^*)$ satisfies:

1. **Stationarity:**
   $\nabla_x f(x^*) + {\mu^*}^\top \nabla_x g(x^*) + {\lambda^*}^\top \nabla_x h(x^*) = 0$;
2. **Primal feasibility:** $g(x^*) \le 0$, $h(x^*) = 0$;
3. **Dual feasibility:** $\mu^* \ge 0$;
4. **Complementary slackness:** ${\mu^*}^\top g(x^*) = 0$.

Because every term $\mu_i^* g_i(x^*)$ is $\le 0$ (by 2 and 3), condition 4
is equivalent to $\mu_i^* g_i(x^*) = 0$ for every $i$.

The vectors $\mu \in \mathbb{R}^p$ and $\lambda \in \mathbb{R}^m$ are the
**Lagrange multipliers**, and

$$
L(x, \mu, \lambda) = f(x) + \mu^\top g(x) + \lambda^\top h(x)
$$

is the **Lagrangian**. Stationarity reads $\nabla_x L(x^*, \mu^*, \lambda^*) = 0$.

**History.** William Karush first derived these conditions in his 1939
master's thesis at the University of Chicago. Harold Kuhn and Albert Tucker
rediscovered them independently in 1951. Tucker was John Nash's PhD adviser
(on non-cooperative games) and coined the name "prisoner's dilemma"; Kuhn was
Nash's colleague and one of his main advocates.

## 3. Geometric interpretation: force balance

Rewrite stationarity as

$$
\underbrace{-\nabla_x f}_{\text{"gravity"}}
\;+\; \sum_i \underbrace{\left(-\mu_i \nabla_x g_i\right)}_{\text{wall } i}
\;+\; \sum_j \underbrace{\left(-\lambda_j \nabla_x h_j\right)}_{\text{rail } j}
\;=\; 0 .
$$

- $-\nabla_x f$ pulls $x$ downhill, like gravity. At a constrained minimum
  this pull is balanced by **reaction forces** from the constraints.
- $\nabla_x g_i$ points out of the feasible set, so $-\mu_i \nabla_x g_i$ is
  a force pushing back into it. The multipliers $\mu_i$ and $\lambda_j$ set
  the magnitudes of these forces (in units of $\|\nabla_x g_i\|$ and
  $\|\nabla_x h_j\|$).
- **Complementary slackness:** if $g_i(x^*) < 0$, the point does not touch
  wall $i$, so there is no reaction force: $\mu_i = 0$. If $g_i(x^*) = 0$,
  then $\mu_i \ge 0$. Usually $\mu_i > 0$ (the point presses against the
  wall), but $\mu_i = 0$ is possible: the point touches the wall without
  pressing on it (see weakly active constraints below).
- **Dual feasibility** $\mu_i \ge 0$: a wall can only push. The reaction
  force never pulls $x^*$ into the wall.
- An equality constraint is a two-sided rail, so $\lambda_j$ can have either
  sign.

The figure below shows a toy problem,
$\min\,(x_1-2)^2+(x_2-2)^2$ s.t. $x_1^2+x_2^2 \le 2$, whose solution
$x^* = (1,1)$ has $\mu^* = 1$.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

x1, x2 = np.meshgrid(np.linspace(-0.5, 3, 300), np.linspace(-0.5, 3, 300))
F = (x1 - 2) ** 2 + (x2 - 2) ** 2
G = x1 ** 2 + x2 ** 2 - 2

x_star = np.array([1.0, 1.0])
grad_f = 2 * (x_star - np.array([2.0, 2.0]))   # (-2, -2)
grad_g = 2 * x_star                             # ( 2,  2)
mu = -(grad_f @ grad_g) / (grad_g @ grad_g)     # stationarity => mu = 1

fig, ax = plt.subplots(figsize=(5, 5))
ax.contour(x1, x2, F, levels=12, cmap="viridis", linewidths=0.8)
ax.contourf(x1, x2, G, levels=[0, G.max()], colors=["0.85"])
ax.contour(x1, x2, G, levels=[0], colors="k", linewidths=1.5)
ax.plot(*x_star, "ko")
ax.quiver(*x_star, *(-grad_f), angles="xy", scale_units="xy", scale=3,
          color="tab:blue", label=r"gravity $-\nabla f$")
ax.quiver(*x_star, *(-mu * grad_g), angles="xy", scale_units="xy", scale=3,
          color="tab:red", label=r"reaction $-\mu\nabla g$")
ax.text(2.05, 2.05, "unconstrained\nminimum", fontsize=9)
ax.plot(2, 2, "x", color="tab:blue")
ax.text(1.6, -0.3, "infeasible (shaded)", fontsize=9)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_aspect("equal")
ax.legend(loc="upper left")
ax.set_title(rf"Force balance at $x^*$ ($\mu^*={mu:.0f}$)")
plt.show()
```

## 4. Optimality conditions

**Active constraints.** At a solution $x^*$ with multipliers $\mu^*$, an
inequality constraint $g_i$ is

- **strongly active** if $g_i(x^*) = 0$ and $\mu_i^* > 0$ (touching, with
  contact force);
- **weakly active** (degenerate) if $g_i(x^*) = 0$ and $\mu_i^* = 0$
  (touching, no force);
- **inactive** if $g_i(x^*) < 0$ (and therefore $\mu_i^* = 0$; no touch).

Let $\mathcal{A}$, $\mathcal{A}_1$, $\mathcal{A}_0$ be the index sets of
strongly active, weakly active, and inactive constraints, and
$\bar{\mathcal{A}} = \mathcal{A}_1 \cup \mathcal{A}_0$, so that
$\mathcal{A} \cup \bar{\mathcal{A}} = [p]$. The **active set**
$I(x^*) = \{i : g_i(x^*) = 0\} = \mathcal{A} \cup \mathcal{A}_1$ depends only
on $x^*$, not on the multipliers.

**First-order necessary conditions.** Under a constraint qualification
(Section 5), if $x^*$ is a local solution of (P), then there exist
multipliers $\mu^*, \lambda^*$ such that $(x^*, \mu^*, \lambda^*)$ is a KKT
point.

**Critical directions.** A direction $w$ is *first-order feasible* at $x^*$
if $w^\top \nabla_x g_i \le 0$ for all $i \in I(x^*)$ and
$w^\top \nabla_x h_j = 0$ for all $j \in [m]$. Among these, the second-order
test only needs the **critical directions** $w \ne 0$ with

$$
w^\top \nabla_x g_i = 0 \;\; (i \in \mathcal{A}), \qquad
w^\top \nabla_x g_i \le 0 \;\; (i \in \mathcal{A}_1), \qquad
w^\top \nabla_x h_j = 0 \;\; (j \in [m]).
$$

*Question:* why does the critical set require equality for strongly active
constraints instead of $w^\top \nabla_x g_i \le 0$ for $i \in \mathcal{A}$?
(Hint: take a first-order feasible $w$ with $w^\top \nabla_x g_i < 0$ for some
$i \in \mathcal{A}$ and use stationarity to compute $\nabla_x f\, w$.)

**Second-order sufficient conditions.** If $(x^*, \mu^*, \lambda^*)$ is a KKT
point and

$$
w^\top \nabla_x^2 L(x^*, \mu^*, \lambda^*)\, w > 0
\quad \text{for every critical direction } w,
$$

then $x^*$ is a strict local solution of (P). Unlike the necessary
conditions, this statement does not need a constraint qualification.

## 5. Constraint qualifications and convex problems

**Linear independence constraint qualification (LICQ).** The gradients of all
active constraints,

$$
\{\nabla_x g_i(x^*) \mid i \in I(x^*)\} \cup
\{\nabla_x h_j(x^*) \mid j \in [m]\},
$$

are linearly independent. Under LICQ the multipliers $(\mu^*, \lambda^*)$ are
also unique.

**Slater's condition** (for convex (P), where $f$ and $g$ are convex and $h$
is affine, $h(x) = Ax - b$). There exists a point $\tilde{x}$ with
$A\tilde{x} = b$ and $g(\tilde{x}) < 0$, i.e., a strictly feasible point.

**First-order sufficient condition.** If (P) is convex and
$(x^*, \mu^*, \lambda^*)$ is a KKT point, then $x^*$ is a global solution of
(P). If, in addition, $f$ is strictly convex, then the solution $x^*$ is
unique. The multipliers need not be unique without LICQ.

**First-order necessary condition for convex (P).** If (P) is convex, $x^*$
is a global solution, and Slater's condition holds, then there exist
$\mu^*, \lambda^*$ such that $(x^*, \mu^*, \lambda^*)$ is a KKT point.

## 6. Exercises: is the solution a KKT point?

**Exercise 1 (solution is a KKT point).** Consider

$$
\min_{x_1 \in [0,1],\; x_2 \in [0,1]} \; (x_1 + 2)^2 + (x_2 - 2)^2 .
$$

Visualize to determine the solution. Write the box as four inequality
constraints and verify that the solution is a KKT point.

**Exercise 2 (solution exists but no KKT point).** Consider

$$
\min_{x_1, x_2} \; -x_1
\quad \text{s.t.} \quad
x_2 - (1 - x_1)^3 \le 0, \quad x_2 \ge 0 .
$$

Visualize to determine the solution. Verify that the solution is **not** a
KKT point. Which hypothesis of the first-order necessary conditions fails?

The helper below plots objective contours and shades the infeasible region.
Edit the problem definitions to explore.

```{code-cell} ipython3
:tags: [thebe-init, hide-input]

import numpy as np
import matplotlib.pyplot as plt

def plot_problem(f, constraints, xlim, ylim, title="", n=400):
    """Contours of f; shade points where any g(x) > 0 (infeasible)."""
    x1, x2 = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    infeasible = np.zeros_like(x1, dtype=bool)
    for g in constraints:
        infeasible |= g(x1, x2) > 0
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cs = ax.contour(x1, x2, f(x1, x2), levels=15, cmap="viridis",
                    linewidths=0.8)
    ax.clabel(cs, fontsize=7)
    ax.contourf(x1, x2, infeasible.astype(float), levels=[0.5, 1.5],
                colors=["0.85"])
    for g in constraints:  # constraint boundaries g(x) = 0
        ax.contour(x1, x2, g(x1, x2), levels=[0], colors="k", linewidths=1.2)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title + " (shaded = infeasible)")
    ax.set_aspect("equal")
    plt.show()
```

```{code-cell} ipython3
# Exercise 1: box constraints written as g(x) <= 0
plot_problem(
    f=lambda x1, x2: (x1 + 2) ** 2 + (x2 - 2) ** 2,
    constraints=[
        lambda x1, x2: -x1,
        lambda x1, x2: x1 - 1,
        lambda x1, x2: -x2,
        lambda x1, x2: x2 - 1,
    ],
    xlim=(-2.5, 1.5), ylim=(-0.5, 2.5), title="Exercise 1",
)

# Exercise 2
plot_problem(
    f=lambda x1, x2: -x1,
    constraints=[
        lambda x1, x2: x2 - (1 - x1) ** 3,
        lambda x1, x2: -x2,
    ],
    xlim=(-0.5, 2.0), ylim=(-0.5, 1.5), title="Exercise 2",
)
```

## 7. Derivation with equality constraints only

We start with the equality-constrained problem

$$
\textbf{(P0)}\qquad
\min_{x \in \mathcal{X}} \; f(x)
\quad \text{s.t.} \quad
h(x) = 0 .
$$

For $h:\mathbb{R}^n \to \mathbb{R}^m$ with a full-row-rank Jacobian
$\nabla_x h(x) \in \mathbb{R}^{m \times n}$, the set $h = 0$ is locally an
$(n-m)$-dimensional surface. If $h(x) = 0$, a perturbation $\Delta x$ keeps
feasibility to first order when

$$
h(x + \Delta x) \approx h(x) + \nabla_x h(x)\, \Delta x = 0
\;\;\Rightarrow\;\;
\nabla_x h(x)\, \Delta x = 0 .
$$

Split the variables as $x = [d; s]$, where $d \in \mathbb{R}^{n-m}$ are
**decision** (free) variables and $s \in \mathbb{R}^m$ are **state**
variables, chosen so that $\nabla_s h$ ($m \times m$) is invertible. Then

$$
\nabla_d h\, \Delta d + \nabla_s h\, \Delta s = 0
\;\;\Rightarrow\;\;
\Delta s = -\left(\nabla_s h\right)^{-1} \nabla_d h\, \Delta d .
$$

By the implicit function theorem, $h = 0$ defines $s = s(d)$ locally, with
$\partial s / \partial d = -(\partial h / \partial s)^{-1}\, \partial h / \partial d$.
(P0) therefore becomes an **unconstrained** problem in $d$, with **total
derivative** (also called the reduced gradient)

$$
\frac{\mathrm{d} f(d, s(d))}{\mathrm{d} d}
= \frac{\partial f}{\partial d} + \frac{\partial f}{\partial s}\frac{\partial s}{\partial d}
= \frac{\partial f}{\partial d}
- \frac{\partial f}{\partial s}
\left(\frac{\partial h}{\partial s}\right)^{-1}
\frac{\partial h}{\partial d}.
$$

From the first-order condition for unconstrained problems, if
$(d^*, s^*)$ is a local solution, then

$$
\frac{\partial f}{\partial d^*}
- \frac{\partial f}{\partial s^*}
\left(\frac{\partial h}{\partial s^*}\right)^{-1}
\frac{\partial h}{\partial d^*} = 0 .
$$

Differentiating once more to get $\mathrm{d}^2 f / \mathrm{d} d^2$ is
possible, but the second-order condition is easier to state once we introduce
the Lagrangian.

**Exercise 3.** Consider

$$
\max_{x_1, x_2, x_3} \; x_1 x_2 + x_2 x_3 + x_1 x_3
\quad \text{s.t.} \quad
x_1 + x_2 + x_3 = 0 .
$$

Solve by introducing decision and state variables and deriving the total
derivative. (A maximization is the minimization of $-f$.)

## 8. From the total derivative to the Lagrangian

Starting from the first-order condition above, define

$$
\lambda^\top = -\frac{\partial f}{\partial s^*}
\left(\frac{\partial h}{\partial s^*}\right)^{-1}.
$$

Then the condition and the definition of $\lambda$ give, respectively,

$$
\frac{\partial f}{\partial d^*} + \lambda^\top \frac{\partial h}{\partial d^*} = 0,
\qquad
\frac{\partial f}{\partial s^*} + \lambda^\top \frac{\partial h}{\partial s^*} = 0 .
$$

Since $x = [d; s]$, the two combine into

$$
\frac{\partial f}{\partial x^*} + \lambda^\top \frac{\partial h}{\partial x^*} = 0 .
$$

With the Lagrangian $L(x, \lambda) = f + \lambda^\top h$, the first-order
optimality conditions of (P0) become

$$
\nabla_x L = 0, \qquad \nabla_\lambda L = 0 \;\; (\text{i.e., } h = 0).
$$

This $\lambda$ is also called the **adjoint** (or costate) variable. It
reappears in the PDE-constrained and optimal-control exercises below.

## 9. Second-order conditions

For a decision perturbation $\Delta d$, let
$w = [\Delta d;\, \Delta s]$ with
$\Delta s = -(\partial h / \partial s)^{-1} (\partial h / \partial d)\, \Delta d$,
so that $\nabla_x h\, w = 0$. At a point satisfying the first-order
conditions, one can show that the curvature of $f$ along the feasible surface
is the curvature of the Lagrangian along $w$:

$$
\Delta d^\top \,
\frac{\mathrm{d}^2 f(d, s(d))}{\mathrm{d} d^2}\,
\Delta d
= w^\top \nabla_x^2 L(x^*, \lambda^*)\, w .
$$

The total derivative already contains the matrix
$(\partial h / \partial s)^{-1}$. Differentiating it again brings in the
second derivatives of $h$, a third-order tensor. The Lagrangian packages that
term as $\lambda^\top \nabla_x^2 h$, so it is easier to compute the curvature
as $w^\top \nabla_x^2 L\, w$. The second-order sufficient condition for (P0)
is $w^\top \nabla_x^2 L\, w > 0$ for every $w \ne 0$ with
$\nabla_x h\, w = 0$. With inequality constraints, $w$ ranges over the
critical directions of Section 4.

**Exercise 4.** Consider

$$
\max_{x_1, x_2, x_3} \; x_1 x_2 + x_2 x_3 + x_1 x_3
\quad \text{s.t.} \quad
x_1 + x_2 + x_3 = 3 .
$$

Solve by introducing the Lagrangian and verify the solution using
second-order conditions.

## 10. Incorporating inequality constraints

Let $x^*$ be a local solution of (P) that satisfies LICQ. Near $x^*$, the
inactive constraints stay satisfied, so $x^*$ is also a local solution of the
equality-constrained problem in which every active constraint
$i \in I(x^*)$ is imposed as $g_i(x) = 0$. Applying Section 8 to that problem
gives multipliers with

$$
\frac{\partial f}{\partial x^*}
+ \lambda^\top \frac{\partial h}{\partial x^*}
+ \sum_{i \in I(x^*)} \mu_i \frac{\partial g_i}{\partial x^*} = 0 .
$$

Setting $\mu_i = 0$ for the inactive constraints, this is

$$
\frac{\partial f}{\partial x^*}
+ \lambda^\top \frac{\partial h}{\partial x^*}
+ \mu^\top \frac{\partial g}{\partial x^*} = 0,
$$

and by construction $\mu^\top g = 0$ (complementary slackness), along with
$h = 0$ and $g \le 0$ (primal feasibility). Any active constraint whose
multiplier turns out to be zero is weakly active.

**Why $\mu \ge 0$ (dual feasibility).** Suppose $\mu_j < 0$ for some
$j \in I(x^*)$. By LICQ, the rows $\partial g_i / \partial x^*$
($i \in I(x^*)$) and $\partial h / \partial x^*$ are linearly independent, so
we can find $w$ with

$$
\frac{\partial g_j}{\partial x^*} w = -1, \qquad
\frac{\partial g_i}{\partial x^*} w = 0 \;\; (i \in I(x^*),\, i \ne j), \qquad
\frac{\partial h}{\partial x^*} w = 0 \quad \text{(why?)}.
$$

This $w$ moves strictly into the feasible side of constraint $j$ and keeps
all other active constraints satisfied, so it is a feasible direction. Since
$x^*$ is a local minimum, $\frac{\partial f}{\partial x^*} w \ge 0$ (why?).
But stationarity gives

$$
\frac{\partial f}{\partial x^*} w
= -\lambda^\top \frac{\partial h}{\partial x^*} w
- \mu^\top \frac{\partial g}{\partial x^*} w
= -\mu_j \cdot (-1) = \mu_j < 0 .
$$

Contradiction. Hence $\mu \ge 0$.

## 11. Sensitivity analysis

Perturb the constraints of (P) by small amounts $\Delta g$ and $\Delta h$:

$$
\textbf{(P}'\textbf{)}\qquad
\min_{x \in \mathcal{X}} \; f(x)
\quad \text{s.t.} \quad
g(x) \le \Delta g, \quad h(x) = \Delta h .
$$

Let the optimum of (P) be $f(x^*)$ and that of (P$'$) be
$f(x^* + \Delta x)$. What is $f(x^* + \Delta x) - f(x^*)$?

Assume LICQ, the second-order sufficient conditions, and that the set of
active constraints does not change. Then, to first order,

$$
f(x^* + \Delta x) - f(x^*) \approx -\lambda^\top \Delta h - \mu^\top \Delta g .
$$

**Lagrange multipliers measure the sensitivity of the optimal objective to
changes in the constraints.** In economics they are called *shadow prices*.

Relaxing a single strongly active inequality, $g_i(x) \le \Delta g_i$ with
$\Delta g_i > 0$, gives

$$
f(x^* + \Delta x) - f(x^*) \approx -\mu_i \Delta g_i < 0 .
$$

This is another way to see why multipliers of active inequality constraints
are non-negative. Relaxing a constraint enlarges the feasible set, so it can
never make the optimal objective worse.

**Parametric problems.** More generally, let (P) depend on parameters
$\theta$:

$$
\min_{x \in \mathcal{X}} \; f(x; \theta)
\quad \text{s.t.} \quad
g(x; \theta) \le 0, \quad h(x; \theta) = 0 .
$$

At a local solution $x^*(\theta)$ with multipliers $(\lambda, \mu)$, one can
show (the *envelope theorem*)

$$
\frac{\mathrm{d} f(x^*(\theta); \theta)}{\mathrm{d}\theta}
= \frac{\partial L}{\partial \theta}
= \nabla_\theta f(x^*; \theta)
+ \lambda^\top \nabla_\theta h(x^*; \theta)
+ \mu^\top \nabla_\theta g(x^*; \theta).
$$

(P$'$) is the special case $g(x;\theta) = g(x) - \Delta g$,
$h(x;\theta) = h(x) - \Delta h$, which gives back the formula above.

The cell below checks the sensitivity formula numerically on the toy problem
from Section 3, relaxing the constraint to $x_1^2 + x_2^2 \le 2 + \Delta g$.

```{code-cell} ipython3
import numpy as np
from scipy.optimize import minimize

def solve_toy(dg=0.0):
    f = lambda x: (x[0] - 2) ** 2 + (x[1] - 2) ** 2
    g = lambda x: x[0] ** 2 + x[1] ** 2 - 2 - dg      # g(x) <= 0
    res = minimize(f, x0=[0.0, 0.0], method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda x: -g(x)}],
                   options={"ftol": 1e-12})
    x = res.x
    grad_f = 2 * (x - 2)
    grad_g = 2 * x
    mu = -(grad_f @ grad_g) / (grad_g @ grad_g)        # from stationarity
    return x, res.fun, mu

x0, f0, mu0 = solve_toy(0.0)
print(f"x* = {x0.round(6)},  f* = {f0:.6f},  mu* = {mu0:.6f}")
for dg in [0.1, 0.01, -0.01]:
    _, f1, _ = solve_toy(dg)
    print(f"dg = {dg:+.2f}:  actual change = {f1 - f0:+.6f},  "
          f"-mu*dg = {-mu0 * dg:+.6f}")
```

## 12. Exercises

**Exercise 5.** Solve Exercise 1 analytically using the KKT conditions.

**Exercise 6 (principal component analysis).** For a symmetric matrix $A$
(e.g., a covariance matrix), consider

$$
\max_x \; x^\top A x \quad \text{s.t.} \quad x^\top x = 1 .
$$

Solve by KKT conditions and verify using second-order conditions.

**Exercise 7 (linearly constrained problem).** Consider

$$
\min_x \; f(x) \quad \text{s.t.} \quad Ax = b,
$$

with $A \in \mathbb{R}^{m \times n}$ of full row rank.

1. Derive the total derivative.
2. Derive the KKT conditions when $f(x) = \tfrac{1}{2} x^\top Q x$.
3. Derive the analytical solution when $Q \succ 0$.

**Exercise 8 (discrete-time Pontryagin's maximum principle).** Consider

$$
\min_{\{x_i, u_i\}} \; \sum_{i=1}^{K-1} f_i(x_i, u_i) + f_K(x_K)
\quad \text{s.t.} \quad
x_{i+1} = A x_i + B u_i, \quad x_1 = \text{const}.
$$

Derive the KKT conditions.

*Lev Pontryagin* was a Soviet mathematician who lost his sight in an accident
at age 14. With his students he developed the maximum principle of optimal
control in the mid-1950s (first announced in 1956).

**Exercise 9 (linear PDE-constrained optimization).** Consider

$$
\min_{x, u} \; F(x, u) := \tfrac{1}{2} x^\top K(u)\, x
\quad \text{s.t.} \quad
K(u)\, x = f .
$$

Here $u$ are design variables (e.g., structure topology), $x$ are system
states (e.g., structural displacements), $f$ is a forcing term, and $K(u)$ is
the system stiffness. Derive the reduced gradient $\mathrm{d}F/\mathrm{d}u$.

*Martin Bendsøe* (mathematician, DTU) and *Noboru Kikuchi* (later president of
Toyota Central R&D Labs) introduced the homogenization method for topology
optimization in their 1988 paper. The approach was commercialized in
structural-design software (Altair's OptiStruct/HyperWorks). Siemens
announced its acquisition of Altair for about US\$10.6B in 2024.

**Exercise 10 (nonlinear PDE-constrained optimization).** Now consider

$$
\min_{u, \{x_i\}} \; F(u, x_K)
\quad \text{s.t.} \quad
x_{i+1} = f(x_i, u), \;\; i = 1, \ldots, K-1, \quad x_1 \text{ given}.
$$

The constraints represent the iterations of a nonlinear PDE solver: $x_i$ is
the evolving system state, $x_K$ is the approximately converged solution, and
$u$ are the design variables. Derive the reduced gradient
$\mathrm{d}F/\mathrm{d}u$ and explain its computational complexity.

**Exercise 11 (neural network training).** Now consider

$$
\min_{\{u_i\}, \{x_i\}} \; \sum_{i=1}^{K-1} l_i(x_i, u_i) + l_K(x_K)
\quad \text{s.t.} \quad
x_{i+1} = f_i(x_i, u_i), \;\; i = 1, \ldots, K-1, \quad x_1 \text{ given}.
$$

Here $x_1$ is the input, $x_i$ are the activations of layer $i$, $u_i$ are the
weights of layer $i$, $f_i$ is the layer map, and $l_i$ is the loss. Explain how neural
networks are trained.

*Geoffrey Hinton* co-authored the 1986 paper that popularized backpropagation
for training neural networks. He shared the 2018 Turing Award (with Yoshua
Bengio and Yann LeCun) for deep learning, and the 2024 Nobel Prize in Physics
(with John Hopfield), cited in part for the Boltzmann machine.

**Exercise 12 (wireless communication: water-filling).** Consider

$$
\min_{\{x_i\}} \; -\sum_{i=1}^{N} \log(a_i + x_i)
\quad \text{s.t.} \quad
\sum_i x_i = 1, \quad x_i \ge 0, \;\; i = 1, \ldots, N .
$$

Here $x_i$ is the power allocated to channel $i$ and $a_i > 0$ is that
channel's noise level (normalized by its gain), so a smaller $a_i$ means a
better channel. The negated objective is the total information rate (up to a
constant). Explain the solution structure of this problem.

**Exercise 13 (manufacturing).** A company manufactures two products, A and B.
At prices $x_A, x_B$ it sells quantities

$$
q_A = 400 - 2x_A + x_B, \qquad q_B = 200 + x_A - x_B .
$$

One unit of A requires 2 hours of labor and 1 unit of raw material; one unit
of B requires 3 hours of labor and 2 units of raw material. There are 1000
hours of labor and 200 units of raw material available. The company chooses
prices to maximize revenue $x_A q_A + x_B q_B$, and quantities must be
non-negative.

1. What are the optimal prices?
2. How much would the company be willing to pay for another hour of labor? For
   another unit of raw material? (Use the sensitivity results of Section 11.)

**Exercise 14 (sensitivity).** In 2D, find the point on the curve
$y = 2px^2$ nearest to the point $(0, p)$, assuming $p \ge \tfrac{1}{2}$. If
the curve shifts to $y = 2px^2 - \epsilon$ for some small $\epsilon$, how will
the minimal distance change? Answer using the Lagrange multiplier rather than
by re-solving the problem.

## Q&A

```{raw} html
<div class="qa-widget" data-context-file="_static/constrained_optimization_pt1_context.json" data-lecture-id="constrained_optimization_pt1_2026" data-engagement-endpoint="https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/qa">
  <div class="qa-identity">
    <input class="qa-first-name" autocomplete="given-name" placeholder="First name" />
    <input class="qa-last-name" autocomplete="family-name" placeholder="Last name" />
    <input class="qa-university-id" autocomplete="username" placeholder="University ID" />
  </div>
  <div class="qa-input-area">
    <textarea class="qa-question" rows="4" placeholder="Ask about KKT conditions, constraint qualifications, Lagrange multipliers, or sensitivity."></textarea>
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
