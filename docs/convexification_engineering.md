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

# Convexification Examples on Engineering Problems

Most engineering design and control problems are **nonconvex** as first written.
Yet a surprising number of them can be rewritten, sometimes *exactly*, as convex
programs: linear programs (LP), second-order cone programs (SOCP), or
semidefinite programs (SDP). Once that is done, we get global optimality,
reliable solvers, and certificates. This note collects such examples from
optimal control, robotics, structural optimization, and PDE-constrained
optimization.

Every example follows the same template:

1. **Context.** Where the problem comes from.
2. **Original problem.** The full formulation, as an engineer would first write it.
3. **Why it is nonconvex.**
4. **Convexification.** The reformulation and why it is exact (or how close it is).
5. **Takeaways and limits.** What the trick buys, and when it breaks.

```{admonition} Prerequisites
:class: tip
Convex sets and functions, Lagrange duality and KKT conditions, linear
state-space systems $\dot x = Ax + Bu$, Lyapunov stability, and basic linear
algebra (eigenvalues, positive semidefinite matrices). Everything else is
introduced in Section 0.
```

## Contents

| # | Example | Field | Original (nonconvex) problem | Convexification | Payoff |
|---|---|---|---|---|---|
| **Part I** | {ref}`Optimal control and robotics <cvx-part1>` | | | | |
| 1 | {ref}`LQR by change of variables <cvx-ex1>` | control | optimize feedback gain $K$ | $Y = KX$ → SDP; hidden convexity | global LQR design; why policy gradient works |
| 2 | {ref}`Rocket powered descent <cvx-ex2>` | aerospace | fuel-optimal landing with thrust annulus | lossless relaxation → SOCP | real-time, guaranteed onboard guidance |
| 3 | {ref}`Decentralized control <cvx-ex3>` | networked control | controller with information constraints | Youla parameter + quadratic invariance | convex design for networks |
| 4 | {ref}`Occupation measures and Koopman <cvx-ex4>` | nonlinear control | nonlinear optimal control | LP over measures → moment-SOS SDPs | certified lower bounds; Koopman's limits |
| 5 | {ref}`Graphs of Convex Sets <cvx-ex5>` | robotics | collision-free shortest path | perspective formulation → tight SOCP | near-global motion plans |
| 6 | {ref}`Certifiable rotation averaging <cvx-ex6>` | robotics / SLAM | orientations on $SO(3)$ | Gram-matrix lift → SDP | maps with optimality certificates |
| **Part II** | {ref}`PDE-constrained optimization <cvx-part2>` | | {ref}`overview: when is it convex? <cvx-pde-overview>` | | |
| 7 | {ref}`Truss topology <cvx-ex7>` | structures | minimum-compliance truss | energy principles → LP / SDP | globally optimal trusses |
| 8 | {ref}`Continuum topology optimization <cvx-ex8>` | structures / heat | 0/1 material layout | relaxation by homogenization (convex) | well-posed design, microstructures |
| 9 | {ref}`Frequency maximization <cvx-ex9>` | structural dynamics | maximize lowest eigenvalue | quasiconvexity → bisection on SDPs | resonance-avoiding designs |
| 10 | {ref}`Frames and shells <cvx-ex10>` | structures | polynomial stiffness | moment-SOS hierarchy | certified global optimum |
| 11 | {ref}`Dynamic optimal transport <cvx-ex11>` | fluids / swarms | steer a density | $m = \rho v$ → SOCP | optimal distribution steering |
| 12 | {ref}`Impedance tomography <cvx-ex12>` | imaging / NDT | recover conductivity | Loewner monotonicity → convex SDP | inversion without local minima |
| 13 | {ref}`Carleman convexification <cvx-ex13>` | inverse scattering | recover a PDE coefficient | Carleman-weighted functional | global convergence |
| 14 | {ref}`Auxiliary functions <cvx-ex14>` | turbulence / chaos | extreme long-time averages | invariant measures → SOS | certified bounds |

## Overview: the most engineering-relevant highlights

Not all fourteen ideas are equally mature. These are the ones that already
matter in engineering practice, or are closest to it.

| Application | Where | Status | Why it is attractive |
|---|---|---|---|
| **Rocket and lander guidance** | {ref}`Ex. 2 <cvx-ex2>` | flight-tested (JPL/Masten G-FOLD); convex guidance in reusable-booster landing | a nonconvex trajectory problem solved onboard, in milliseconds, with a convergence guarantee |
| **Controller synthesis by LMIs** | {ref}`Ex. 1 <cvx-ex1>`, {ref}`3 <cvx-ex3>` | standard in robust control | $\mathcal H_2/\mathcal H_\infty$ and multi-objective design become SDPs; quadratic invariance and system-level synthesis scale to networks such as power grids |
| **Structural and topology optimization** | {ref}`Ex. 7 <cvx-ex7>`, {ref}`8 <cvx-ex8>` | truss LPs and SDPs are classical; homogenization + de-homogenization for lattice and additive-manufacturing parts | compliance is convex by an energy principle, so the global optimum is computable, and relaxation explains *why* microstructures appear |
| **Certifiable SLAM and perception** | {ref}`Ex. 6 <cvx-ex6>` | open-source solvers (SE-Sync, TEASER) used in robotics | a fast local solver plus a cheap eigenvalue test either *proves* the map is globally optimal or flags failure |
| **Motion planning (Graphs of Convex Sets)** | {ref}`Ex. 5 <cvx-ex5>` | available in the open-source Drake toolbox; used for robot arms and quadrotors | discrete route choice and continuous trajectory optimized together, with a tight convex relaxation |
| **Learning-based control** | {ref}`Ex. 1 <cvx-ex1>` (hidden convexity) | active research with mature theory | explains when model-free policy gradient finds the *global* optimum |
| **Imaging and nondestructive testing** | {ref}`Ex. 12 <cvx-ex12>`, {ref}`13 <cvx-ex13>` | research, validated on experimental data | coefficient inverse problems without the local-minimum trap |
| **Certified bounds in fluids and heat transfer** | {ref}`Ex. 14 <cvx-ex14>` | research | provable limits on drag, dissipation or heat transport that simulation alone cannot give |

**Recurring theme.** Engineering convexifications rarely come from abstract
theory alone. They come from **physics**: energy principles (trusses,
topology, EIT), conservation laws written in the right variables (optimal
transport, occupation measures), and the structure of information
(decentralized control). Section 0 collects the few mathematical tools behind
all of them.

---

## 0. The toolkit

Almost every example below reuses a handful of facts. We collect them here.

### 0.1 Conic programs

We call a problem *tractable* if it can be written as

$$
\min_{x}\; c^\top x \quad \text{s.t.}\quad Ax = b,\; x \in \mathcal{K},
$$

where $\mathcal K$ is a product of "nice" cones:

- the **nonnegative orthant** $\mathbb R^n_+$, which gives LPs;
- the **second-order cone** $\{(x,t): \|x\|_2 \le t\}$, which gives SOCPs;
- the **positive semidefinite (PSD) cone** $\mathbb S^n_+ = \{X = X^\top : X \succeq 0\}$, which gives SDPs.

Interior-point methods solve these to accuracy $\varepsilon$ in polynomial time.
A constraint of the form $F_0 + \sum_i x_i F_i \succeq 0$ with symmetric $F_i$ is
called a **linear matrix inequality (LMI)**.

### 0.2 The Schur complement lemma

For symmetric blocks with $C \succ 0$,

$$
\begin{bmatrix} A & B \\ B^\top & C \end{bmatrix} \succeq 0
\quad\Longleftrightarrow\quad
A - B C^{-1} B^\top \succeq 0 .
$$

This lemma turns *inverses* into *LMIs*. For example,
$\tau \ge f^\top K^{-1} f$ (with $K \succ 0$) is equivalent to
$\begin{bmatrix}\tau & f^\top \\ f & K\end{bmatrix} \succeq 0$, which is
**linear** in $(\tau, K)$ jointly.

### 0.3 Perspective functions

If $g:\mathbb R^n\to\mathbb R$ is convex, its **perspective**

$$
\tilde g(x, t) = t\, g(x/t), \qquad t>0,
$$

is *jointly* convex in $(x,t)$. Three perspectives appear repeatedly below:

| Base function $g$ | Perspective | Appears in |
|---|---|---|
| $\tfrac12 \lVert m \rVert^2$ | $\dfrac{\lVert m\rVert^2}{2\rho}$ | optimal transport (Ex. 11), EIT (Ex. 12), truss (Ex. 7) |
| $\lVert x \rVert$ (already homogeneous) | $\lVert x\rVert$ | graphs of convex sets (Ex. 5) |
| $y^\top y$, matrix version | $Y X^{-1} Y^\top$ (matrix fractional) | LQR (Ex. 1) |

The mental model is: **the product of a "scale" variable and a "shape"
variable is nonconvex, but if we take the product itself as the new variable,
the problem often becomes convex.**

### 0.3.1 Geometric picture: "scale × shape"

Many engineering variables come in pairs: a nonnegative **scale** $t$ that says
*how much* there is, and a **shape** $x$ that says something *per unit*. Their
product $z = t\,x$ is a **total**:

| Scale $t \ge 0$ ("how much") | Shape $x$ ("per unit") | Total $z = t\,x$ | Example |
|---|---|---|---|
| mass density $\rho$ | velocity $v$ | momentum $m=\rho v$ | 11 |
| bar area $a$ | stress $s$ | bar force $q = a s$ | 7 |
| conductivity $\sigma$ | electric field $e$ | current density $J = \sigma e$ | 12 |
| rocket mass $m$ | acceleration $u$ | thrust $T = m u$ | 2 |
| state covariance $X$ | feedback gain $K$ | $Y = KX$ | 1 |
| "is this used?" $y\in[0,1]$ | a location $x$ | $z = y\,x$ | 5 |

There are three ways to see why the totals are the right variables.

**(a) Mixing is linear in totals, not in per-unit quantities.** Convexity is a
statement about *averaging*. Mix two parcels of fluid, $(\rho_1,v_1)$ and
$(\rho_2,v_2)$. Mass and momentum simply add, so the mixture is the *sum* in
$(\rho, m)$ coordinates. Its velocity, however, is the mass-weighted average
$(\rho_1v_1+\rho_2v_2)/(\rho_1+\rho_2)$, **not** the plain average
$(v_1+v_2)/2$. A convex combination in $(\rho,m)$ is a physical mixture. A
convex combination in $(\rho, v)$ is not physically meaningful. Problems become
convex in the coordinates where averaging is physical.

**(b) The cone picture.** Suppose the shape is restricted to a convex set $C$.
In $(t, z)$ coordinates, the feasible set $\{(t,z) : t\ge0,\ z\in tC\}$ is the
**cone over $C$**: its cross-section at height $t=1$ is $C$, it shrinks
linearly to the apex at $t=0$, and it is convex. What breaks convexity in the
$(t,x)$ coordinates is any cost or constraint of the form $t\cdot g(x)$
("amount times per-unit cost"), such as kinetic energy $\rho|v|^2/2$, fuel, or
dissipation. In $(t,z)$ coordinates, $t\,g(x) = t\,g(z/t)$ is exactly the
perspective of $g$, which is convex.

**(c) A two-variable example.** Let $h(t,x) = t\,x^2$, i.e. mass times
velocity squared. Its Hessian $\begin{bmatrix}0 & 2x\\ 2x & 2t\end{bmatrix}$
has determinant $-4x^2<0$, so $h$ is a saddle and is **not** convex. In the
coordinates $(t, z) = (t, tx)$, the same function is $z^2/t$, which **is**
convex.

The map $(t,x)\mapsto(t,tx)$ is a smooth, invertible change of coordinates for
$t>0$, with inverse $x = z/t$. It bends the coordinate grid so that curved
sublevel sets become convex. The figure shows the sublevel set $h\le 1$ in both
coordinate systems.

- *Left, $(t,x)$ coordinates.* The straight segment between two feasible points
  leaves the set.
- *Right, $(t,z)$ coordinates.* The straight segment between the same two
  points stays inside.
- *Dashed curve, left panel.* The right-hand segment mapped back to $(t,x)$. It
  is the *physical mixing path*, and it stays feasible.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

P1, P2 = np.array([0.04, 5.0]), np.array([1.0, 1.0])     # both satisfy t x^2 = 1
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))

t = np.linspace(1e-3, 1.3, 400)
ax = axs[0]
ax.fill_between(t, -1 / np.sqrt(t), 1 / np.sqrt(t), color="C0", alpha=0.15)
ax.plot(*np.c_[P1, P2], "C3-o", label="straight segment in $(t, x)$")
s = np.linspace(0, 1, 100)
tz = (1 - s)[:, None] * [P1[0], P1[0] * P1[1]] + s[:, None] * [P2[0], P2[0] * P2[1]]
ax.plot(tz[:, 0], tz[:, 1] / tz[:, 0], "C2--", label="segment in $(t, z)$, mapped back")
ax.set_xlim(0, 1.3); ax.set_ylim(-1, 6)
ax.set_xlabel("scale $t$"); ax.set_ylabel("shape $x$")
ax.set_title(r"$\{t\,x^2 \leq 1\}$: nonconvex", fontsize=10)
ax.legend(fontsize=8, loc="upper right")

ax = axs[1]
ax.fill_between(t, -np.sqrt(t), np.sqrt(t), color="C0", alpha=0.15)
Q1, Q2 = [P1[0], P1[0] * P1[1]], [P2[0], P2[0] * P2[1]]
ax.plot(*np.c_[Q1, Q2], "C2-o", label="straight segment in $(t, z)$")
ax.set_xlim(0, 1.3); ax.set_ylim(-1.2, 1.4)
ax.set_xlabel("scale $t$"); ax.set_ylabel("total $z = t\\,x$")
ax.set_title(r"$\{z^2/t \leq 1\}$: convex", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.show()
```

**(d) Recoverability.** We must be able to get the shape back: $x = z/t$
whenever $t>0$. When $t=0$ (no material, no mass, unused vertex), the shape is
meaningless anyway, and the perspective convention ($0$ if $z=0$, $+\infty$
otherwise) handles it automatically. When recoverability fails, the trick
fails; see static output feedback in Example 1.

### 0.4 Supremum of affine functions

If $h(a) = \sup_{u} \{\alpha(u) + \beta(u)^\top a\}$, then $h$ is convex in $a$,
whatever the dependence on $u$. Physics provides such suprema for free through
**energy principles**. For example, the compliance of a linear elastic structure
is a supremum of functions that are affine in the material distribution
(Ex. 7 and 8, and the overview of Part II).

### 0.5 Lifting to measures

For any continuous $f$ on a compact set $X$,

$$
\min_{x\in X} f(x) \;=\; \min_{\mu \in \mathcal P(X)} \int_X f\, d\mu,
$$

where $\mathcal P(X)$ is the set of probability measures on $X$. The right-hand
side is a **linear program** over measures. It is infinite-dimensional, but it
can be approximated by finite SDPs through moments and sums of squares (SOS).
Examples 4, 10 and 14 use this idea.

### 0.6 Five convexification mechanisms

| Mechanism | Idea | Examples |
|---|---|---|
| **M1. Change of variables** | rename products/ratios so the problem becomes convex; recover the original variables afterwards | 1, 2, 3, 5, 11 |
| **M2. Lossless relaxation** | enlarge the feasible set to a convex one, then *prove* the optimum lies in the original set | 2, 6, 9 |
| **M3. Energy (variational) principles** | write the objective as a sup/inf of affine or jointly convex terms | 7, 8, 12 |
| **M4. Lifting to measures + SOS** | exact LP over measures, approximated by SDP hierarchies with certificates | 4, 10, 14 |
| **M5. Weights and monotonicity** | reweight a nonconvex functional, or exploit order structure, so that convexity emerges | 12, 13 |

---

(cvx-part1)=
# Part I: Optimal Control and Robotics

(cvx-ex1)=
## Example 1: Linear-quadratic regulator by change of variables

### Context

The linear-quadratic regulator (LQR) is the workhorse of control. With a known
model, the optimal gain comes from a Riccati equation. That solution, however,
does not extend to added constraints (structured gains, robustness margins,
multiple objectives). It also does not explain why model-free reinforcement
learning, which runs gradient descent directly on the gain $K$, works so well
on LQR. Both questions are answered by a convex reformulation.

### Original problem

Consider $\dot x = Ax + Bu$ with $x \in \mathbb R^n$ and $u\in\mathbb R^m$. Let the initial state
be random with $\mathbb E[x_0x_0^\top] = \Sigma_0 \succ 0$. We look for a static
state-feedback gain $u = Kx$ that solves

$$
\min_{K \in \mathcal S}\; J(K) = \mathbb E \int_0^\infty \big(x^\top Q x + u^\top R u\big)\,dt,
\qquad Q \succeq 0,\; R \succ 0,
$$

where $\mathcal S = \{K : A+BK \text{ is Hurwitz}\}$ is the set of stabilizing gains.

**What "Hurwitz" means.** With $u = Kx$, the closed loop is
$\dot x = (A+BK)x =: A_Kx$, so $x(t) = e^{A_Kt}x_0$. If $A_K$ is
diagonalizable with eigenvalues $\lambda_i$, then $x(t)$ is a combination of the
modes $e^{\lambda_i t}$, and $|e^{\lambda_i t}| = e^{\operatorname{Re}(\lambda_i)t}$.
A matrix is **Hurwitz** if *every* eigenvalue has $\operatorname{Re}\lambda_i<0$.
Then every mode, and hence every trajectory, decays exponentially to zero. The
same holds for non-diagonalizable $A_K$, where the modes are
$t^k e^{\lambda_i t}$. If some eigenvalue has $\operatorname{Re}\lambda_i\ge0$,
trajectories starting along that mode do not decay, and the cost $J(K)$ is
infinite. So "$K\in\mathcal S$" is exactly the condition for $J(K)$ to be
finite.

**Writing the cost with a covariance matrix.** For any matrix $M$,
$x^\top Mx = \operatorname{tr}(M\,xx^\top)$. With $u=Kx$, the integrand is
$x^\top(Q+K^\top RK)x$, so

$$
J(K) = \operatorname{tr}\!\Big((Q+K^\top RK)\ \underbrace{\mathbb E\!\int_0^\infty x(t)x(t)^\top dt}_{=:X_K}\Big),
\qquad
X_K = \int_0^\infty e^{A_Kt}\,\Sigma_0\,e^{A_K^\top t}\,dt .
$$

$X_K\succeq0$ measures how much state energy, in which directions, the closed
loop accumulates over time. It is the (time-integrated) state covariance.

**Why $X_K$ satisfies a Lyapunov equation.** Differentiate the integrand:
$\tfrac{d}{dt}\big(e^{A_Kt}\Sigma_0e^{A_K^\top t}\big) = A_K\big(e^{A_Kt}\Sigma_0e^{A_K^\top t}\big) + \big(e^{A_Kt}\Sigma_0e^{A_K^\top t}\big)A_K^\top$.
Integrate from $0$ to $\infty$. The right side gives $A_KX_K + X_KA_K^\top$.
The left side gives $[\,e^{A_Kt}\Sigma_0e^{A_K^\top t}\,]_0^\infty = 0 - \Sigma_0$,
because the Hurwitz property makes the integrand vanish at infinity. Hence

$$
J(K) = \operatorname{tr}\!\big((Q + K^\top R K)\, X_K\big),
\qquad
A_K X_K + X_K A_K^\top + \Sigma_0 = 0 .
$$

The Lyapunov equation is a linear equation in $X_K$ that replaces the integral.

### Why it is nonconvex

- **The feasible set $\mathcal S$ is nonconvex.** Take $A = 0$, $B = I_2$, so
  $A_K = K$. Then
  $K_1 = \begin{bmatrix}-1 & 10\\ 0 & -1\end{bmatrix}$ and
  $K_2 = \begin{bmatrix}-1 & 0\\ 10 & -1\end{bmatrix}$ are both Hurwitz
  (eigenvalues $-1, -1$). Their average
  $\begin{bmatrix}-1 & 5\\ 5 & -1\end{bmatrix}$ has eigenvalue $+4$.
- **The cost is nonconvex as well,** because $X_K$ depends on $K$ through the
  inverse of a Lyapunov operator, and that is multiplied by $K^\top R K$.

### Convexification (M1: $Y = KX$)

Treat the covariance $X \succ 0$ and the product $Y = KX$ as the decision
variables. The Lyapunov equation becomes **linear** in $(X,Y)$:

$$
AX + XA^\top + BY + Y^\top B^\top + \Sigma_0 = 0 .
$$

The cost becomes $\operatorname{tr}(QX) + \operatorname{tr}(R^{1/2} Y X^{-1}
Y^\top R^{1/2})$. The second term is a matrix-fractional function, jointly
convex in $(X, Y)$. Introduce an epigraph variable $Z$ and apply the Schur
complement:

$$
\boxed{
\begin{aligned}
\min_{X, Y, Z}\quad & \operatorname{tr}(QX) + \operatorname{tr}(Z)\\
\text{s.t.}\quad & AX + XA^\top + BY + Y^\top B^\top + \Sigma_0 \preceq 0,\\
& \begin{bmatrix} Z & R^{1/2}Y \\ Y^\top R^{1/2} & X\end{bmatrix} \succeq 0,
\qquad X \succ 0 .
\end{aligned}}
$$

This is an SDP. The gain is recovered as $K^\star = Y^\star (X^\star)^{-1}$.

**Why it is exact.**

- *SDP value ≤ LQR value.* Every stabilizing $K$ gives a feasible point
  $(X_K, KX_K, R^{1/2}KX_KK^\top R^{1/2})$ with the same cost.
- *SDP value ≥ LQR value.* Take any feasible $(X,Y,Z)$ and set $K = YX^{-1}$,
  so $Y = KX$. Substituting into the first constraint gives
  $A_KX + XA_K^\top \preceq -\Sigma_0 \prec 0$. There are three steps.

  1. **$K$ is stabilizing (Lyapunov's theorem).** Let $\lambda$ be any
     eigenvalue of $A_K$, with left eigenvector $w\neq0$ ($w^HA_K = \lambda w^H$,
     hence $A_K^\top w = \bar\lambda w$). Multiply the inequality by $w^H$ on
     the left and $w$ on the right:
     $w^H(A_KX+XA_K^\top)w = (\lambda+\bar\lambda)\,w^HXw = 2\operatorname{Re}(\lambda)\,w^HXw < 0$.
     Since $X\succ0$, we have $w^HXw>0$, so $\operatorname{Re}\lambda<0$.
  2. **$X$ over-estimates the true covariance: $X\succeq X_K$.** Name the slack
     $W := -(A_KX+XA_K^\top+\Sigma_0)\succeq0$. Subtracting the exact equation
     $A_KX_K+X_KA_K^\top+\Sigma_0=0$ gives a Lyapunov equation for the
     difference $D = X - X_K$:
     $A_KD + DA_K^\top = -W$.
     Because $A_K$ is Hurwitz, its unique solution is
     $D = \int_0^\infty e^{A_Kt}\,W\,e^{A_K^\top t}\,dt$, which is an integral
     of PSD matrices, hence PSD.

     Physically, $X$ is the covariance of the same closed loop with *extra*
     noise $W$ injected, so it can only be larger.
  3. **The SDP cost upper-bounds $J(K)$.** The Schur constraint gives
     $Z\succeq R^{1/2}YX^{-1}Y^\top R^{1/2} = R^{1/2}KXK^\top R^{1/2}$, so the
     SDP cost is at least
     $\operatorname{tr}(QX)+\operatorname{tr}(RKXK^\top) = \operatorname{tr}((Q+K^\top RK)X)$.
     Since $M := Q+K^\top RK\succeq0$ and $D\succeq0$, we have
     $\operatorname{tr}(MD)\ge0$, i.e. $\operatorname{tr}(MX)\ge\operatorname{tr}(MX_K) = J(K)$.

  Relaxing "$=$" to "$\preceq$" therefore costs nothing: the inequality is
  tight at the optimum.

The figure below checks the two-gain example numerically. Interpolating the
gains directly leaves the stable region. Interpolating in $(X, Y)$ and mapping
back with $K = YX^{-1}$ stays stable along the whole path.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov

K1 = np.array([[-1.0, 10.0], [0.0, -1.0]])
K2 = np.array([[-1.0, 0.0], [10.0, -1.0]])
I2 = np.eye(2)                       # A = 0, B = I, Sigma_0 = I

s = np.linspace(0, 1, 201)
naive = [np.linalg.eigvals((1 - t) * K1 + t * K2).real.max() for t in s]

X = [solve_continuous_lyapunov(K, -I2) for K in (K1, K2)]   # K X + X K^T + I = 0
Y = [K @ Xi for K, Xi in zip((K1, K2), X)]
lifted = []
for t in s:
    Xt = (1 - t) * X[0] + t * X[1]
    Yt = (1 - t) * Y[0] + t * Y[1]
    lifted.append(np.linalg.eigvals(Yt @ np.linalg.inv(Xt)).real.max())

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(s, naive, label=r"interpolate $K$ directly")
ax.plot(s, lifted, label=r"interpolate $(X, Y)$, then $K = YX^{-1}$")
ax.axhline(0, color="k", lw=0.8)
ax.fill_between(s, 0, 5, color="C3", alpha=0.08)
ax.text(0.02, 3.6, "unstable", color="C3")
ax.set_xlabel("interpolation parameter $s$")
ax.set_ylabel(r"max Re$\,\lambda(A+BK)$")
ax.set_ylim(-1.5, 4.5)
ax.legend(loc="center right", fontsize=9)
plt.tight_layout()
plt.show()
```

### Takeaways and limits

- **Recoverability is essential.** The substitution works because $K$ can be
  recovered from $(X, Y)$ as $K = YX^{-1}$. Two variants show where this
  breaks.
  - **Static output feedback.** In practice we rarely measure the full state.
    We measure $y = Cx$ with $C\in\mathbb R^{p\times n}$ and usually $p<n$, for
    example positions but not velocities. So yes, $Cx$ is a *partial
    observation* of $x$. *Static* output feedback is the memoryless law
    $u = Ky = KCx$. The analogous substitution is $Y = KCX$, and $K$ can no
    longer be recovered: $KC$ is constrained to a subspace (rows in the row
    space of $C$), and $\{(X, KCX)\}$ is not a convex set. Static output
    feedback with gain bounds is NP-hard in general (Blondel & Tsitsiklis,
    1997).
  - **Dynamic output feedback.** The controller has its own internal state
    $x_c$:
    $\dot x_c = A_cx_c + B_cy$, $u = C_cx_c + D_cy$. A familiar example is
    "Kalman filter + LQR", where $x_c$ is the state estimate $\hat x$.
    **Full-order** means $\dim x_c = n$, the same as the plant. In that case,
    a clever nonlinear but *invertible* change of variables on
    $(A_c,B_c,C_c,D_c)$ and the Lyapunov matrix makes $\mathcal H_2$ and
    $\mathcal H_\infty$ synthesis LMIs (Scherer, Gahinet & Chilali, 1997).
    **Reduced-order** controllers ($\dim x_c < n$; static feedback is the
    extreme case $\dim x_c = 0$) lose invertibility and are nonconvex.
- **Hidden convexity explains policy gradient.** The map
  $K \mapsto (X_K, KX_K)$ is a smooth bijection from the nonconvex set
  $\mathcal S$ onto a convex set. As a result, $J(K)$ has no spurious local
  minima, and on every sublevel set it satisfies a *gradient-dominance*
  inequality $J(K) - J^\star \le c\,\|\nabla J(K)\|_F^2$. So plain gradient descent on
  $K$ converges globally (Fazel et al., 2018; Mohammadi et al., 2022). The next
  subsection explains why, and what generalizes.

### A general mechanism: hidden convexity

The LQR story is an instance of a general pattern (Fatkhullin, He & Hu, 2023;
Zheng, Pai & Tang, 2023/24):

$$
J(\theta) = F(\Phi(\theta)),\qquad F \text{ convex on a convex set } \mathcal C,\qquad
\Phi:\Theta\to\mathcal C \text{ smooth and invertible, with invertible Jacobian}.
$$

Here $\theta$ is the parameter we optimize (the gain $K$), $\Phi$ is a "hidden"
reparametrization (here $K\mapsto(X_K,KX_K)$), and $F$ is the convex cost in the
hidden coordinates.

**1. No spurious stationary points.** By the chain rule,
$\nabla J(\theta) = D\Phi(\theta)^\top\,\nabla F(\Phi(\theta))$. If $D\Phi$ is
invertible, then $\nabla J(\theta) = 0$ exactly when
$\nabla F(\Phi(\theta)) = 0$. For a convex $F$, that means $\Phi(\theta)$ is a
global minimizer. Every stationary point of the nonconvex $J$ is therefore
global.

**2. Gradient dominance, hence fast convergence.** Assume $F$ satisfies a
gradient-dominance inequality $F(\xi)-F^\star\le c_F\|\nabla F(\xi)\|^2$. Strongly
convex functions do, for example. Assume also that $D\Phi$ is uniformly well
conditioned on a sublevel set, $\sigma_{\min}(D\Phi)\ge s>0$. Then

$$
J(\theta) - J^\star = F(\Phi(\theta)) - F^\star \le c_F\|\nabla F\|^2 \le \frac{c_F}{s^2}\,\|\nabla J(\theta)\|^2 ,
$$

which is the Polyak–Łojasiewicz inequality. Gradient descent on $\theta$ then
converges linearly to the global optimum.

**3. You never need to know $\Phi$.** The algorithm runs in the original
coordinates $\theta$, for instance model-free policy gradient estimated from
rollouts. The hidden convexity is only used in the *analysis*. This is why
reinforcement learning on LQR works without knowing $A$ and $B$.

**Examples of the same mechanism:**

- **Tabular reinforcement learning.** The expected return is *linear* in the
  state-action occupation measure, and the occupation measures form a polytope
  (the LP formulation of MDPs). The map from policy to occupation measure plays
  the role of $\Phi$. Policy gradient therefore enjoys gradient-domination-type
  global convergence (Agarwal, Kakade, Lee & Mahajan, 2021).
- **Inventory and revenue management, and convex RL.** The decision rule
  determines a distribution or occupancy through an invertible map, in which
  the cost is convex (Fatkhullin, He & Hu, 2023).
- **LQG, $\mathcal H_\infty$, filtering and distributed control.** The
  Extended Convex Lifting (ECL) framework (Zheng, Pai & Tang, 2023/24) weakens
  "$\Phi$ invertible" to "a convex lifted problem exists whose partial
  minimization reproduces $J$". Its conclusion: every *non-degenerate*
  stationary point is globally optimal.

**Where the mechanism fails.** The failure is always at points where $D\Phi$
is singular or $\Phi$ is not onto. In LQG, non-minimal controllers (with
uncontrollable or unobservable internal modes) are such degenerate points.
They can be suboptimal saddle points (Zheng, Tang & Li, 2023). The Burer–Monteiro
factorization $X = VV^\top$ is another example: the map $V\mapsto VV^\top$ is
not invertible, so extra rank conditions are needed to rule out spurious
points.

---

(cvx-ex2)=
## Example 2: Lossless convexification of rocket powered descent

### Context

A reusable rocket or planetary lander must divert to a target and touch down
softly using as little propellant as possible. The guidance computer must solve
this trajectory optimization problem **onboard, in real time, with a guarantee
of convergence**, so local solvers with uncertain convergence are unacceptable.
Lossless convexification (Açıkmeşe & Ploen, 2007; Açıkmeşe & Blackmore, 2011)
made this possible: the problem becomes an SOCP. It was flight-tested in the
JPL/Masten G-FOLD experiments, and convex optimization is used in SpaceX
booster landing guidance (Blackmore, 2016).

### Original problem

The states are position $r\in\mathbb R^3$, velocity $v$, and mass $m$. The control is
the thrust vector $T\in\mathbb R^3$. The model is

$$
\begin{aligned}
\min_{T(\cdot)}\quad & \int_0^{t_f} \|T(t)\|\,dt
&& \text{(fuel; equivalently maximize } m(t_f))\\
\text{s.t.}\quad & \dot r = v,\quad \dot v = g + \frac{T}{m},\quad \dot m = -\alpha \|T\|,\\
& 0 < \rho_1 \le \|T(t)\| \le \rho_2 && \text{(throttle limits)}\\
& \hat e_z^\top T(t) \ge \|T(t)\|\cos\theta && \text{(thrust pointing)}\\
& \|(r_x, r_y)\| \tan\gamma \le r_z && \text{(glide slope)}\\
& r(0)=r_0,\; v(0)=v_0,\; m(0)=m_0,\quad r(t_f)=0,\; v(t_f)=0,\; m(t_f)\ge m_{\text{dry}} .
\end{aligned}
$$

Here $\alpha = 1/(I_{sp} g_0)$ is the fuel-consumption rate. The lower throttle
bound $\rho_1 > 0$ is physical: rocket engines cannot be throttled arbitrarily
low once lit.

### Why it is nonconvex

1. **The thrust set** $\{\rho_1\le \|T\|\le\rho_2\}$ is an annulus (a spherical
   shell), which is not convex.
2. **The dynamics** contain the ratio $T/m$ and the nonlinear mass equation
   $\dot m = -\alpha\|T\|$.

### Convexification

**Step 1 (M2: slack variable / lifting).** Introduce a scalar $\Gamma(t)$ and replace
the annulus by

$$
\|T\| \le \Gamma,\qquad \rho_1 \le \Gamma \le \rho_2,\qquad \dot m = -\alpha\Gamma,
\qquad \hat e_z^\top T \ge \Gamma\cos\theta,
\qquad \text{cost } \int \Gamma\,dt .
$$

Geometrically, the nonconvex annulus in $T$-space is the projection of the
*boundary* $\|T\| = \Gamma$ of a convex truncated cone in $(T,\Gamma)$-space.
The relaxed problem lets $T$ go into the interior of the cone. The theorem
below says the optimizer never does.

**Step 2 (M1: change of variables).** Define

$$
u = \frac{T}{m},\qquad \sigma = \frac{\Gamma}{m},\qquad z = \ln m .
$$

The dynamics become linear, $\dot v = g + u$ and $\dot z = -\alpha\sigma$, with
$\|u\|\le\sigma$. The throttle bounds become
$\rho_1 e^{-z}\le \sigma \le \rho_2 e^{-z}$. The lower bound is convex. The upper
bound is not, so it is replaced by its tangent line around the reference
$z_0(t) = \ln(m_0 - \alpha\rho_2 t)$, which is conservative. The lower bound is
replaced by a second-order Taylor expansion, which is a second-order cone
constraint:

$$
\rho_1 e^{-z_0}\Big[1-(z-z_0)+\tfrac12(z-z_0)^2\Big] \le \sigma \le \rho_2 e^{-z_0}\big[1-(z-z_0)\big].
$$

After time discretization, this is an **SOCP** for each fixed $t_f$. In
practice $t_f$ is found by a one-dimensional line search.

**Theorem (lossless convexification, informal).** Suppose the linear system is
controllable and some mild technical conditions hold (Açıkmeşe & Blackmore,
2011). Then every optimal solution of the relaxed problem satisfies
$\|T^\star(t)\| = \Gamma^\star(t)$ for almost every $t$. Hence it is feasible,
and therefore optimal, for the original nonconvex problem.

**Proof sketch via the Maximum Principle.** Consider the generic form
$\dot x = Ax + Bu$, cost $\int\Gamma$, constraints $\|u\|\le\Gamma$ and
$\rho_1\le\Gamma\le\rho_2$. The Hamiltonian is
$H = \Gamma + \lambda^\top(Ax + Bu)$, and Pontryagin's principle minimizes it
pointwise in $(u,\Gamma)$.

- Minimizing $\lambda^\top B u$ over $\|u\| \le \Gamma$ gives
  $u^\star = -\Gamma\, B^\top\lambda/\|B^\top\lambda\|$, so $\|u^\star\| = \Gamma$,
  **whenever $B^\top \lambda(t) \neq 0$**.
- The remaining Hamiltonian $\Gamma(1 - \|B^\top\lambda\|)$ is linear in
  $\Gamma$. So $\Gamma^\star$ is bang-bang between $\rho_1$ and $\rho_2$, which
  is the familiar max-min-max thrust profile.
- The costate obeys $\dot\lambda = -A^\top\lambda$, so $B^\top\lambda(t)$ is
  real-analytic in $t$. If it vanished on an interval, all its derivatives
  would vanish: $B^\top (A^\top)^k\lambda(0) = 0$ for all $k$. That means
  $\lambda(0) \perp \operatorname{range}[B, AB, \dots, A^{n-1}B]$, and
  controllability forces $\lambda \equiv 0$. That degenerate case is excluded
  by the technical conditions.
- Therefore $B^\top\lambda$ vanishes only at isolated instants, and
  $\|u^\star\| = \Gamma^\star$ almost everywhere. $\square$

### Takeaways and limits

- **The sufficient condition is checkable in polynomial time.** It is a rank
  test on the controllability matrix plus simple geometric checks.
- **The proof pattern is "relax, then prove the optimum lands on the boundary".**
  It reappears for semi-continuous inputs, pointing constraints and some state
  constraints; see Malyuta et al. (2022) for a tutorial.
- **Aerodynamics, attitude coupling and free final time are not covered.**
  These are handled by *sequential* convex programming (SCvx, GuSTO), which
  works well but gives only local guarantees.

---

(cvx-ex3)=
## Example 3: Decentralized control and quadratic invariance

### Context

Power grids, vehicle platoons and process plants are controlled by many local
controllers, each of which sees only part of the measurements, possibly with
delays. Designing the best controller under such *information constraints* is
notoriously hard. Witsenhausen (1968) showed that even a two-stage LQG problem
with a nonclassical information pattern has a nonlinear optimal controller.
**Quadratic invariance** (Rotkowitz & Lall, 2006) identifies exactly when the
problem becomes convex.

### Original problem

Take the generalized plant

$$
\begin{bmatrix} z \\ y \end{bmatrix} =
\begin{bmatrix} P_{11} & P_{12} \\ P_{21} & G \end{bmatrix}
\begin{bmatrix} w \\ u \end{bmatrix},
\qquad u = K y,
$$

The four signals are:

| Signal | Meaning | Examples |
|---|---|---|
| $w$ | **exogenous inputs** we do not control | wind gusts, load changes, sensor noise, reference commands |
| $z$ | **performance outputs** we want small | tracking errors, weighted states $Q^{1/2}x$, weighted control effort $R^{1/2}u$ |
| $y$ | **measurements** available to the controller | sensor readings |
| $u$ | **control inputs** | actuator commands |

The $P_{ij}$ and $G$ are transfer matrices, i.e. linear dynamical systems
written in the Laplace domain. For example, $G$ maps $u$ to $y$. Packaging
everything into one "generalized plant" $P$ is standard in robust control: any
design specification is encoded by choosing what goes into $w$ and $z$.

**Deriving the closed loop.** Substitute $u = Ky$ into $y = P_{21}w + Gu$:
$y = P_{21}w + GKy$, so $y = (I-GK)^{-1}P_{21}w$. Then
$u = K(I-GK)^{-1}P_{21}w$, and $z = P_{11}w + P_{12}u$. Hence

$$
z = T_{zw}(K)\,w,\qquad
T_{zw}(K) = P_{11} + P_{12} K (I - GK)^{-1} P_{21}.
$$

**$T_{zw}$ is the closed-loop transfer function from disturbances to
performance outputs.** It summarizes everything the controller achieves.

**The objective.** The $\mathcal H_2$ norm is

$$
\|T_{zw}\|_{\mathcal H_2}^2 = \frac{1}{2\pi}\int_{-\infty}^{\infty}\operatorname{tr}\big(T_{zw}(j\omega)^HT_{zw}(j\omega)\big)\,d\omega
= \sum_{k}\int_0^\infty \|z_k(t)\|^2\,dt ,
$$

where $z_k$ is the response to a unit impulse in the $k$-th disturbance
channel. Equivalently, it is the steady-state variance of $z$ when $w$ is unit
white noise.

With $z = (Q^{1/2}x,\ R^{1/2}u)$ and $w$ entering as $\dot x = Ax + Bu + B_ww$,
this is exactly the LQR cost of Example 1 with $\Sigma_0 = B_wB_w^\top$. So
Example 3 is "LQR or LQG, but with information constraints on the controller".
Replacing $\mathcal H_2$ with $\mathcal H_\infty$, the worst-case energy gain
from $w$ to $z$, gives robust control. Both norms are convex functions of
$T_{zw}$.

The design problem is

$$
\min_{K}\; \|T_{zw}(K)\|_{\mathcal H_2}
\quad \text{s.t.}\quad K \text{ internally stabilizes } G,\quad K \in S,
$$

where $S$ is a subspace that encodes the information structure. For example,
$K_{ij} = 0$ if controller $i$ cannot see measurement $j$.

### Why it is nonconvex

$T_{zw}$ depends on $K$ through $K(I-GK)^{-1}$, which is a nonlinear
fractional map. For a general $S$ the problem is intractable.

### Convexification (M1: Youla parameter)

For simplicity, assume $G$ is stable. Define

$$
Q = h(K) := K(I-GK)^{-1}
\quad\Longleftrightarrow\quad
K = Q(I+GQ)^{-1}.
$$

Then $T_{zw} = P_{11} + P_{12}QP_{21}$ is **affine** in $Q$, and internal
stability is equivalent to $Q$ being stable. The only remaining difficulty is
the constraint: $K\in S$ becomes $Q \in h(S)$, which is in general a
*nonconvex* set.

**Definition.** $S$ is *quadratically invariant* (QI) under $G$ if
$KGK \in S$ for all $K\in S$.

**Theorem (Rotkowitz & Lall, 2006).** $S$ is QI under $G$ if and only if
$h(S) = S$.

The intuition is the expansion
$h(K) = K + KGK + KGKGK + \cdots$. If $S$ is QI, every term stays in $S$.

So under QI the problem is

$$
\min_{Q\ \text{stable}}\; \|P_{11} + P_{12} Q P_{21}\|_{\mathcal H_2}
\quad\text{s.t.}\quad Q\in S,
$$

which is a **convex** program in $Q$. It is infinite-dimensional, and is solved
in practice via finite-impulse-response truncation or state-space methods.
Lessard & Lall (2011) showed that, under mild assumptions, QI is also
*necessary* for this convexity.

**Example.** Take two subsystems in a chain, where subsystem 1 influences
subsystem 2 but not vice versa:
$G = \begin{bmatrix} G_{11} & 0 \\ G_{21} & G_{22}\end{bmatrix}$.

- **Lower-triangular $S$** (controller 2 sees both measurements, controller 1
  sees only its own): $KGK$ is a product of lower-triangular matrices, hence
  lower-triangular. **QI holds, so the problem is convex.**
- **Diagonal $S$** (fully decentralized): $(KGK)_{21} = K_2 G_{21} K_1 \ne 0$.
  **QI fails.**

Physically, QI means *information must propagate at least as fast as the
dynamics*.

**A polynomial-time test.** For sparsity constraints, QI reduces to a
combinatorial check on binary patterns. If $K_{ij}$ is allowed, $G_{jk}\neq0$,
and $K_{kl}$ is allowed, then $K_{il}$ must be allowed. This takes $O(m^2p^2)$
operations.

### Takeaways and limits

- **QI is a clean necessary-and-sufficient, polynomial-time-checkable condition**
  for a convexifiable class. It is rare to have one.
- **System Level Synthesis** (Wang, Matni & Doyle, 2019) parametrizes the
  closed-loop responses $(\Phi_x,\Phi_u)$ directly, subject to the affine
  constraint $[zI-A,\ -B]\,[\Phi_x;\Phi_u] = I$. Locality constraints on $\Phi$
  are convex, which scales to very large networks. When QI holds, it agrees
  with the Youla approach.

---

(cvx-ex4)=
## Example 4: Nonlinear optimal control via occupation measures (and Koopman)

### Context

Nonlinear optimal control usually relies on local methods: direct collocation,
shooting, or iLQR. These need a good initial guess and give no certificate of
global optimality. Two "linearizing" ideas address this. The **Koopman
operator** linearizes dynamics on functions. **Occupation measures** linearize
the whole optimal control problem on measures. The two are adjoint to each
other, and only the second is an exact convexification.

### Original problem

$$
\begin{aligned}
\min_{u(\cdot)}\quad & \int_0^T \ell(x(t),u(t))\,dt + \phi(x(T))\\
\text{s.t.}\quad & \dot x = f(x,u),\quad x(0) = x_0,\quad x(t)\in X,\quad u(t)\in U,
\end{aligned}
$$

where $f$, $\ell$ and $\phi$ are polynomials, and $X$, $U$ are compact sets
described by polynomial inequalities.

### Why it is nonconvex

The map from $u(\cdot)$ to $x(\cdot)$ is nonlinear. There can be many locally
optimal trajectories, for example going around an obstacle on the left versus
on the right.

### Convexification (M4: occupation measures)

**Measures are described by what they do to functions.** A nonnegative measure
$\mu$ assigns a "mass" to sets. Equivalently, it is completely determined by
the numbers $\int g\,d\mu$ for *all* continuous functions $g$. In the
definitions below, $g$ is such a generic **test function**, a probe used to
define the measure. Two familiar cases:

- a Dirac mass $\delta_a$ satisfies $\int g\,d\delta_a = g(a)$;
- a probability density $p$ gives $\int g\,p\,dx$.

Associate to a trajectory its **occupation measure** $\mu$ on
$[0,T]\times X\times U$ and its **terminal measure** $\mu_T$ on $X$:

$$
\int g\,d\mu := \int_0^T g(t, x(t), u(t))\,dt,\qquad
\int g\,d\mu_T := g(x(T))\qquad\text{for every continuous } g .
$$

**How to read these definitions:**

- Choosing $g = \mathbf 1_A$, the indicator of a set $A$, gives
  $\mu(A)$ = *the amount of time the trajectory spends with $(t,x,u)\in A$*.
  Hence the name "occupation".
- $\mu_T = \delta_{x(T)}$ is a Dirac mass at the final state.
- Choosing $g=\ell$ gives the running cost, and $g = \phi$ the terminal cost.
  The cost of *any* trajectory is therefore **linear** in $(\mu,\mu_T)$.

For any smooth test function $v(t,x)$, the chain rule gives

$$
v(T,x(T)) - v(0,x_0) = \int_0^T \big(\partial_t v + \nabla_x v\cdot f\big)\,dt,
$$

which in measure language reads

$$
\int v(T,\cdot)\,d\mu_T - v(0,x_0) = \int \big(\partial_t v + \nabla_x v\cdot f\big)\, d\mu
\qquad \forall v \in C^1 .
$$

This is the weak form of the **Liouville equation**, and it is **linear in
$(\mu,\mu_T)$**. Note that $f$ can be arbitrarily nonlinear: it only appears
*inside* the integrand. The cost $\int\ell\,d\mu + \int\phi\,d\mu_T$ is also
linear. So we obtain an LP over nonnegative measures:

$$
\boxed{p^\star_{\text{LP}} = \inf_{\mu,\mu_T\ \ge\ 0}\ \int\ell\,d\mu + \int\phi\,d\mu_T
\quad\text{s.t. Liouville holds for all } v .}
$$

Every trajectory yields a feasible pair, so $p^\star_{\text{LP}}\le p^\star$. The
question is what *else* is feasible.

### What the LP adds: relaxed controls (chattering)

**An example.** Take $\dot x = u$ with $U = \{-1,+1\}$ (e.g. an on/off
thruster), $x(0) = 0$, $T = 1$, and the cost $\int_0^1 x^2\,dt$.

- The ideal is to stay at $x\equiv0$. That needs $u = 0\notin U$.
- Every admissible control has a cost strictly greater than $0$, because
  $\dot x = \pm1$ never lets $x$ stay at $0$.
- Switching $u$ between $+1$ and $-1$ every $\varepsilon$ seconds makes $x$
  zigzag within $[0,\varepsilon]$, with cost $\le\varepsilon^2\to0$. The
  infimum $0$ is approached but **never attained**.
- The limit object uses $+1$ and $-1$ each half of the time *at every instant*.
  It is a probability distribution over $U$,
  $\nu_t = \tfrac12\delta_{+1}+\tfrac12\delta_{-1}$, called a **relaxed
  control**, with average velocity $\int f\,d\nu_t = 0$.
- The zigzag's occupation measures converge to
  $\mu = dt\otimes\delta_{x=0}\otimes(\tfrac12\delta_{+1}+\tfrac12\delta_{-1})$.
  This measure satisfies Liouville and has cost $0$. The LP *attains* the
  infimum that no ordinary control attains.

**In general.** A relaxed control picks a distribution $\nu_t$ over $U$ at each
time. Then $\dot x = \int_U f(x,u)\,d\nu_t(u)$, and the running cost is
$\int_U\ell(x,u)\,d\nu_t(u)$. At a given state $x$, the achievable
(velocity, cost) pairs form the **convex hull** of
$\{(f(x,u),\ell(x,u)) : u\in U\}$. Three facts hold:

1. **Relaxed controls can be approximated by fast switching** (the "chattering
   lemma"). The relaxed optimal value therefore equals the infimum over
   ordinary controls; nothing is gained or lost in value.
2. **The LP value equals the relaxed value** under mild compactness conditions
   (Vinter, 1993; Lasserre et al., 2008). Every LP-feasible measure is a
   mixture of relaxed trajectories, so the LP optimum is attained by one of
   them.
3. **When chattering is unnecessary.** Suppose that, for every $x$, the set
   $\{(f(x,u),\ \ell(x,u)+r) : u\in U,\ r\ge0\}$ is already convex. A typical
   case is $f$ affine in $u$, $\ell$ convex in $u$, and $U$ convex. Then every
   averaged (velocity, cost) pair can be matched, or beaten in cost, by a
   *single* control value. The "$+\,r$" (the *epigraph extension*) means we may
   waste cost but must match velocity exactly. In this case the relaxed
   solution is an ordinary control, and there is no gap.

**Physical meaning.** Chattering is **pulse-width modulation**. Power
electronics switch transistors on and off at kHz rates to synthesize average
voltages, and spacecraft fire on/off thrusters in pulses to produce an average
thrust. The relaxed solution is the idealized limit of fast switching. The
lossless convexification of Example 2 can be read as a proof that, for that
problem, the optimum does not need chattering.

### The dual LP is the Hamilton–Jacobi–Bellman (HJB) inequality

$$
d^\star = \sup_{v\in C^1} v(0,x_0)
\quad\text{s.t.}\quad
\partial_t v + \nabla_x v\cdot f(x,u) + \ell(x,u) \ge 0\ \ \forall (t,x,u),\qquad
v(T,x)\le\phi(x)\ \ \forall x .
$$

Any feasible $v$ is a smooth *subsolution* of HJB, and it lower-bounds the cost
of **every** trajectory. The proof takes one line, using first the terminal
constraint and then the HJB inequality:

$$
\phi(x(T)) \ge v(T,x(T)) = v(0,x_0) + \int_0^T (\partial_t v + \nabla v\cdot f)\,dt \ge v(0,x_0) - \int_0^T \ell\,dt .
$$

### Finite-dimensional approximation: the moment-SOS hierarchy

The LP lives in an infinite-dimensional space of measures. To compute with it,
we need finitely many numbers. We explain the idea on the static problem of
Section 0.5 first, and then return to control.

**Setting.** Minimize a polynomial $p(x)$ over a compact set
$K = \{x\in\mathbb R^N : g_j(x)\ge0,\ j=1,\dots,J\}$, where the $g_j$ are
polynomials. Equivalently, minimize $\int p\,d\mu$ over probability measures
$\mu$ on $K$.

**Step 1: from measures to moments.** Write monomials as
$x^\alpha = x_1^{\alpha_1}\cdots x_N^{\alpha_N}$, and define the **moments**
$y_\alpha = \int x^\alpha\,d\mu$. Since $p = \sum_\alpha p_\alpha x^\alpha$,

$$
\int p\,d\mu = \sum_\alpha p_\alpha\,y_\alpha ,
$$

which is **linear in $y$**. A measure on a compact set is determined by its
moments, so we may optimize over $y$ instead of $\mu$. The catch is that we
must restrict $y$ to sequences that actually come from a nonnegative measure
on $K$.

**Step 2: necessary conditions on $y$ are PSD constraints.** Let $m_d(x)$ be
the vector of all monomials of degree $\le d$, and let $q(x) = c^\top m_d(x)$
be any polynomial of degree $\le d$. A nonnegative measure must satisfy
$\int q^2\,d\mu\ge0$. Expanding,

$$
\int q^2\,d\mu = c^\top\Big(\int m_d\,m_d^\top\,d\mu\Big)c = c^\top M_d(y)\,c,
\qquad [M_d(y)]_{\alpha\beta} = y_{\alpha+\beta}.
$$

Since this holds for every $c$, the **moment matrix** must satisfy
$M_d(y)\succeq0$. It is a matrix whose entries are *linear* in $y$.

Likewise, $g_j\ge0$ on the support gives $\int g_j\,q^2\,d\mu\ge0$, which is
the **localizing matrix** condition $M_{d-d_j}(g_j\,y)\succeq0$. Here
$d_j = \lceil\deg g_j/2\rceil$, and the matrix entries are again linear in $y$.

*Sanity check (1D, $d=1$).* With $m_1 = (1,x)$,
$M_1(y) = \begin{bmatrix}y_0&y_1\\y_1&y_2\end{bmatrix}\succeq0$. With $y_0 = 1$,
this says $y_2 - y_1^2\ge0$: **the variance is nonnegative**. Higher-order
moment matrices encode subtler facts of the same kind.

**Step 3: the relaxation of order $d$ is an SDP.**

$$
\rho_d = \min_y\ \sum_\alpha p_\alpha y_\alpha
\quad\text{s.t.}\quad y_0 = 1,\quad M_d(y)\succeq0,\quad M_{d-d_j}(g_j\,y)\succeq0\ \ \forall j .
$$

The constraints are necessary but not sufficient, so this is a
**relaxation**, and $\rho_d\le p^\star$. Raising $d$ adds constraints, so
$\rho_1\le\rho_2\le\dots\le p^\star$. By Putinar's Positivstellensatz, under
an "Archimedean" condition (satisfied, for example, if a ball constraint
$R^2-\|x\|^2\ge0$ is among the $g_j$), $\rho_d\to p^\star$ (Lasserre, 2001).

**Step 4: the dual SDP is a sum-of-squares (SOS) certificate.**

$$
\max_{\lambda,\ \sigma_0,\dots,\sigma_J}\ \lambda
\quad\text{s.t.}\quad
p(x) - \lambda = \sigma_0(x) + \sum_j\sigma_j(x)\,g_j(x),\qquad \sigma_j\ \text{SOS}.
$$

The right-hand side is visibly $\ge0$ on $K$: SOS polynomials are nonnegative
everywhere, and $g_j\ge0$ on $K$. So any feasible $\lambda$ is a *proven*
lower bound.

Why is this an SDP? A polynomial $\sigma$ is SOS if and only if
$\sigma(x) = m_d(x)^\top G\,m_d(x)$ for some $G\succeq0$. (Factor
$G = L^\top L$; then $\sigma = \|L\,m_d(x)\|^2$.) Matching coefficients of
$x^\alpha$ on both sides gives linear equations in $G$.

Deciding whether a polynomial is nonnegative is NP-hard already at degree 4.
SOS is a tractable *sufficient* condition, and the hierarchy tightens it step
by step.

**Step 5: certifying exactness and extracting the minimizers.** If the optimal
moment matrices satisfy the **flat extension** condition
$\operatorname{rank}M_d(y^\star) = \operatorname{rank}M_{d-d_K}(y^\star)$, where
$d_K = \max_j d_j$, then $\rho_d = p^\star$. The optimal measure is then a sum
of $r = \operatorname{rank}M_d$ Dirac masses at global minimizers, which can be
recovered by linear algebra (Curto & Fialkow).

**A worked example.** Minimize $p(x) = x^4 - x^2$ on $[-1,1]$, i.e.
$g = 1-x^2$. The true minimum is $-\tfrac14$, at $x = \pm1/\sqrt2$.

- *Order-2 relaxation.* Use the variables $(y_1,y_2,y_3,y_4)$ with $y_0 = 1$:

  $$
  \min\ y_4 - y_2\quad\text{s.t.}\quad
  \begin{bmatrix}1&y_1&y_2\\y_1&y_2&y_3\\y_2&y_3&y_4\end{bmatrix}\succeq0,\qquad
  \begin{bmatrix}1-y_2&y_1-y_3\\y_1-y_3&y_2-y_4\end{bmatrix}\succeq0 .
  $$

- *Dual certificate.* $x^4 - x^2 + \tfrac14 = (x^2-\tfrac12)^2$ is SOS, so
  $\rho_2\ge-\tfrac14$.
- *Primal certificate.* The measure
  $\tfrac12\delta_{1/\sqrt2}+\tfrac12\delta_{-1/\sqrt2}$ has moments
  $y = (0,\tfrac12,0,\tfrac14)$ and objective $-\tfrac14$. So
  $\rho_2 = -\tfrac14 = p^\star$: the relaxation is exact.
- *Flatness.* The moment matrix is
  $\begin{bmatrix}1&0&\tfrac12\\0&\tfrac12&0\\\tfrac12&0&\tfrac14\end{bmatrix}$,
  which has rank 2, the same as its top-left $2\times2$ block. Two Dirac
  masses means **two global minimizers**, read off from the moments.

**Back to optimal control.** Now the variables are $(t,x,u)$, so
$N = 1+n+m$.

- *Primal.* Both $\mu$ and $\mu_T$ get moment vectors. Liouville, imposed for
  every monomial test function $v = t^ax^b$ up to degree $2d$, becomes a set of
  linear equations in the moments. Moment and localizing matrices enforce
  nonnegativity and support on $[0,T]\times X\times U$ and on $X$.
- *Dual.* $v$ is a polynomial, and the HJB inequalities are enforced by SOS
  certificates.
- *Output.* Each order $d$ gives an SDP whose value is a **certified lower
  bound** on the best achievable cost. Combined with the cost of any feasible
  trajectory, for example one from a local solver, this brackets the true
  optimum. The bounds increase monotonically to $p^\star_{\text{LP}}$
  (Lasserre et al., 2008; Henrion, Korda & Lasserre, 2020).

The same machinery computes regions of attraction (Henrion & Korda, 2014) and
reachable sets.

### The Koopman connection

For autonomous dynamics $\dot x = F(x)$, the **Koopman semigroup**
$(\mathcal K^t g)(x) = g(\varphi^t(x))$ acts linearly on observables $g$. Its
generator is $\mathcal L g = \nabla g\cdot F$. Its adjoint, the
**Perron–Frobenius/Liouville operator**, acts on densities:
$\partial_t\rho = -\nabla\cdot(\rho F)$. The Liouville constraint above is
exactly this adjoint. In other words:

> **Koopman (functions) and occupation measures (measures) are two sides of the
> same linear duality.** Your intuition that Koopman "linearizes" control is
> correct, and the rigorous convex version is the occupation-measure LP.

**When a finite Koopman embedding exists.** Consider

$$
\dot x_1 = \mu x_1,\qquad \dot x_2 = \lambda(x_2 - x_1^2).
$$

With the lifted state $z = (x_1,\ x_2,\ x_1^2)$, the dynamics are *exactly*
linear:

$$
\dot z = \begin{bmatrix} \mu & 0 & 0\\ 0&\lambda&-\lambda\\ 0&0&2\mu\end{bmatrix} z
$$

(Brunton et al., 2016). The code below verifies this.

```{code-cell} ipython3
:tags: [hide-input]

from scipy.integrate import solve_ivp
from scipy.linalg import expm

mu, lam = -0.1, -1.0
f = lambda t, x: [mu * x[0], lam * (x[1] - x[0] ** 2)]
x0 = np.array([2.0, -1.0])
t = np.linspace(0, 10, 200)
sol = solve_ivp(f, (0, 10), x0, t_eval=t, rtol=1e-10, atol=1e-12)

A_lift = np.array([[mu, 0, 0], [0, lam, -lam], [0, 0, 2 * mu]])
z0 = np.array([x0[0], x0[1], x0[0] ** 2])
Z = np.array([expm(A_lift * ti) @ z0 for ti in t])

fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.plot(t, sol.y[0], "C0", lw=3, alpha=0.4, label="$x_1$ (nonlinear ODE)")
ax.plot(t, sol.y[1], "C1", lw=3, alpha=0.4, label="$x_2$ (nonlinear ODE)")
ax.plot(t, Z[:, 0], "C0--", label="$z_1$ (3-dim linear lift)")
ax.plot(t, Z[:, 1], "C1--", label="$z_2$ (3-dim linear lift)")
ax.set_xlabel("$t$")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()
print("max error:", np.abs(Z[:, :2].T - sol.y).max())
```

**Koopman MPC** (Korda & Mezić, 2018) picks a dictionary of observables
$z = \psi(x)\in\mathbb R^{N_\psi}$, fits a lifted model $z^+ \approx Az + Bu$,
$x \approx Cz$, by least squares on data, and then runs a convex QP-based MPC
in $z$. It is useful in practice, but it is an *approximation*, for three
reasons.

**1. Truncation.** An exact finite model requires $\operatorname{span}(\psi)$
to be *invariant*: $\mathcal L\psi_k$ must lie in the span for every $k$, as in
the example above. Such subspaces that also contain the state are rare.
Otherwise, least squares returns the best fit within the span, with no
guarantee that the error is small, and the error compounds over the prediction
horizon.

**2. Control enters bilinearly.** Take a control-affine system
$\dot x = f_0(x)+\sum_i f_i(x)u_i$. For any observable $g$,

$$
\frac{d}{dt}g(x(t)) = \nabla g\cdot\Big(f_0 + \sum_i f_iu_i\Big)
= (\mathcal L_0 g)(x) + \sum_i u_i\,(\mathcal L_i g)(x),
\qquad \mathcal L_i g := \nabla g\cdot f_i .
$$

Suppose the dictionary is invariant under all these operators:
$\mathcal L_0\psi = A\psi$ and $\mathcal L_i\psi = B_i\psi$. Then the exact
lifted dynamics are

$$
\dot z = Az + \sum_i u_i\,B_iz ,
$$

which contain products $u_i z_k$: they are **bilinear**, and MPC over $(z,u)$
is nonconvex again. A linear term "$Bu$" would require each $\mathcal L_i\psi$
to be a *constant* vector, i.e. $\nabla\psi\cdot f_i$ independent of $x$. That
fails for essentially any nonlinear observable.

This happens even for a **linear** plant. With $\dot x = -x+u$ and
$\psi = (x, x^2)$,
$\tfrac{d}{dt}x^2 = 2x(-x+u) = -2x^2 + 2\,xu$, and the $xu$ term is bilinear.
Linear-predictor Koopman models fit an "average" of this bilinear term. That is
accurate near the training data and degrades away from it.

**3. A topological obstruction.** Take $\dot x = x - x^3$. It has equilibria at
$-1$, $0$ and $+1$. Every $x_0>0$ flows to $+1$, and every $x_0<0$ flows to
$-1$. Suppose a continuous, injective $\psi:\mathbb R\to\mathbb R^{N}$ turned it
into a linear system, $\psi(x(t)) = e^{At}\psi(x_0)$.

- **In a linear system, the destination depends linearly on the start.**
  Whenever $e^{At}z$ converges, the limit is $Pz$ for a fixed matrix $P$ (the
  projection onto $\ker A$ along the decaying modes). So the final state is a
  *continuous* function of the initial state.
- **The nonlinear system jumps.** Starting at $x_0 = \pm\varepsilon$, the
  trajectory goes to $\pm1$, so
  $\psi(\pm1) = \lim_t e^{At}\psi(\pm\varepsilon) = P\psi(\pm\varepsilon)$.
- **Let $\varepsilon\to0$.** Continuity of $\psi$ and $P$ gives
  $P\psi(\pm\varepsilon)\to P\psi(0) = \psi(0)$, since $0$ is an equilibrium.
  Hence $\psi(+1) = \psi(0) = \psi(-1)$, which contradicts injectivity.

Linear systems cannot have **basin boundaries**, i.e. places where an
infinitesimal change in the initial condition changes the destination. Liu,
Ozay & Sontag (2025) prove the general version: any continuous immersion of a
system with several $\omega$-limit sets (equilibria, limit cycles, ...) into a
linear system collapses those limit sets. Multistable engineering systems
include snap-through (bistable) structures, power grids with several operating
points, and legged robots that either recover or fall. For such systems, an
exact Koopman model must be discontinuous, non-injective, or
infinite-dimensional. Learned models are accurate only near one attractor.

### Why the occupation-measure LP avoids these issues, and what it costs

1. **No hand-picked dictionary, and controlled truncation.** The LP imposes
   Liouville for *all* test functions. Truncation enters only through the
   moment degree $d$, and every truncation level returns a **valid bound**
   that converges monotonically. Koopman truncation, in contrast, returns a
   model with unquantified error.
2. **The control is just another coordinate.** $\mu$ lives on $(t,x,u)$
   jointly, and Liouville is linear in $\mu$ however $u$ enters $f$. We never
   propagate a lifted state forward with $u$ multiplying it. We constrain the
   *joint distribution* of $(t,x,u)$, so no bilinearity appears.
3. **Measures can split.** A measure can put weight $\tfrac12$ on a trajectory
   that goes to $+1$ and $\tfrac12$ on one that goes to $-1$. It represents
   distributions *over* trajectories, so no injective embedding is needed.

**The price is size.** The moment matrix $M_d$ is indexed by all monomials of
degree $\le d$ in $N = 1+n+m$ variables. There are $\binom{N+d}{d}$ of them:

| $N$ (e.g. $n$ states, $m$ inputs, plus time) | $d=2$ | $d=4$ | $d=6$ |
|---|---|---|---|
| 3 ($n=1, m=1$) | 10 | 35 | 84 |
| 5 ($n=3, m=1$) | 21 | 126 | 462 |
| 9 ($n=6, m=2$) | 55 | 715 | 5005 |

Interior-point SDP solvers handle PSD blocks up to roughly a few thousand rows,
with cost growing steeply with block size. In practice, dense hierarchies are
limited to about $n\lesssim 6$–$10$ states at moderate degree, unless one
exploits sparsity (variables that interact only locally) or symmetry.

---

(cvx-ex5)=
## Example 5: Motion planning in Graphs of Convex Sets

### Context

A robot must move from a start to a goal configuration without hitting
obstacles. Free space is nonconvex. Sampling-based planners (RRT, PRM) give no
optimality guarantees, and trajectory optimization gets stuck in local minima.
The **Graphs of Convex Sets** (GCS) framework (Marcucci et al., 2023, 2024)
gives a mixed-integer *convex* formulation whose convex relaxation is
remarkably tight.

### Original problem

Cover the free configuration space by convex regions $X_1,\dots,X_N$, for
example polytopes $X_v = \{x: A_vx\le b_v\}$ computed with IRIS (Deits &
Tedrake, 2015). Build a graph $G=(V,E)$ with an edge $(u,v)$ whenever
$X_u\cap X_v\ne\emptyset$.

Inside each region $v$, the robot travels on a straight segment from an entry
point $p_v$ to an exit point $q_v$, with $p_v, q_v \in X_v$. By convexity, the
segment stays in free space. The planning problem is

$$
\min_{\text{path } s\to t \text{ in } G,\ \{p_v,q_v\}}\
\sum_{v\in\text{path}} \|q_v - p_v\|
\quad\text{s.t.}\quad p_v,q_v\in X_v,\quad q_u = p_v \text{ for consecutive } (u,v).
$$

### Why it is nonconvex

The choice of path is discrete (combinatorial), and the continuous variables
exist only for the vertices on the chosen path.

### Convexification (M1 + M2: perspective formulation)

**Step 1: discrete variables (which regions are used).** Give each edge a
binary flow $y_e\in\{0,1\}$, equal to 1 if the path traverses edge $e$. Impose
flow conservation: net outflow $1$ at $s$, $-1$ at $t$, and $0$ elsewhere.
This is the standard shortest-path LP. The vertex indicator
$y_v = \sum_{e\in\text{in}(v)}y_e\in\{0,1\}$ says whether region $v$ is on the
path.

**Step 2: the product variable.** For each vertex we have:

- a scalar $y_v\in\{0,1\}$ ("is region $v$ used?"), and
- a vector $(p_v,q_v)\in\mathbb R^{2d}$ (entry and exit points stacked; $d$ is
  the configuration-space dimension).

Define the new vector

$$
z_v := y_v\,(p_v, q_v) = (z_v^p,\ z_v^q)\in\mathbb R^{2d},
$$

a **scalar times a vector**. Concretely, $z_v = (p_v,q_v)$ if region $v$ is
used, and $z_v = 0$ if it is not. This is exactly the "scale × shape" pattern
of Section 0.3.1: $y_v$ is the scale ("how much this region is used") and
$(p_v,q_v)$ is the shape ("where").

**Why bother?** In the original variables, both the cost
$\sum_v y_v\|q_v-p_v\|$ and the rule "$(p_v,q_v)\in X_v^2$ only if $v$ is used"
couple a binary scale to a continuous shape. Even after relaxing $y_v$ to
$[0,1]$, the product $y_v\|q_v-p_v\|$ is nonconvex (it has the form
$t\cdot g(x)$). In the variables $(z_v, y_v)$:

- **Cost.** $y_v\|q_v - p_v\| = \|y_vq_v - y_vp_v\| = \|z_v^q - z_v^p\|$
  (norms are positively homogeneous). This is **convex**.
- **Constraint.** $(p_v,q_v)\in X_v^2$ becomes the perspective constraint
  $A_vz_v^p\le b_v\,y_v$ and $A_vz_v^q\le b_v\,y_v$. This is **linear**.
  - If $y_v = 1$, it says $p_v,q_v\in X_v$, as desired.
  - If $y_v = 0$, it says $A_vz\le0$. Because $X_v$ is bounded, the only such
    $z$ is $0$ (a nonzero $z$ with $A_vz\le0$ would be a direction along which
    $X_v$ extends forever). So "unused" correctly forces $z_v = 0$.
  - If $0<y_v<1$, it says $z_v\in y_vX_v$, a copy of $X_v$ shrunk toward the
    origin by the factor $y_v$.
- **Recovery.** $(p_v,q_v) = z_v/y_v$ whenever $y_v>0$.

**Geometric picture.** The set $\{(z,y): z\in yX,\ 0\le y\le1\}$ is the cone
with apex at the origin and cross-section $X$ at height $y=1$. This cone is
precisely the **convex hull of the two pure options**: "unused" (the point
$(z,y) = (0,0)$) and "used" ($X\times\{1\}$). No convex set containing both
options can be smaller. That is why this relaxation is as tight as possible,
vertex by vertex. In the original $(x,y)$ coordinates, the two options are
disjoint pieces with nothing in between. The figure shows a 1D region
$X = [2,3]$.

```{code-cell} ipython3
:tags: [hide-input]

fig, axs = plt.subplots(1, 2, figsize=(8.5, 3))
ax = axs[0]
ax.axhline(0, color="C0", lw=4, alpha=0.6, label="unused: $y=0$, $x$ irrelevant")
ax.plot([2, 3], [1, 1], color="C3", lw=4, label="used: $y=1$, $x\\in X=[2,3]$")
ax.set_xlim(-0.5, 4); ax.set_ylim(-0.3, 1.4)
ax.set_xlabel("location $x$"); ax.set_ylabel("indicator $y$")
ax.set_title("original variables: two disjoint options", fontsize=10)
ax.legend(fontsize=8, loc="center left")

ax = axs[1]
ax.fill([0, 2, 3], [0, 1, 1], color="C2", alpha=0.25, label="$\\{(z,y): z \\in yX\\}$")
ax.plot(0, 0, "o", color="C0", ms=8, label="unused: $(z,y)=(0,0)$")
ax.plot([2, 3], [1, 1], color="C3", lw=4, label="used: $z \\in X$, $y=1$")
ax.set_xlim(-0.5, 4); ax.set_ylim(-0.3, 1.4)
ax.set_xlabel("product $z = y\\,x$"); ax.set_ylabel("indicator $y$")
ax.set_title("product variables: the convex hull of both", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.show()
```

**Step 3: edges.** The same construction is applied per edge. For edge
$e=(u,v)$, introduce copies $z_e^u = y_e(p_u,q_u)$ and $z_e^v = y_e(p_v,q_v)$
with their own perspective constraints. Impose:

- **continuity**, the linear equation $z_e^{u,q} = z_e^{v,p}$ (i.e.
  $y_eq_u = y_ep_v$: the exit of $u$ is the entry of $v$ when the edge is
  used); and
- **consistency**, $z_v = \sum_{e\in\text{in}(v)}z_e^v$ (exactly one incoming
  edge is active when $v$ is used).

The resulting problem is a mixed-integer *convex* program. Relaxing
$y_e\in[0,1]$ gives a convex program, an SOCP.

### Takeaways and limits

- **The relaxation is tight because of the perspective construction.** For each
  edge, it is the convex hull of a two-way disjunction ("edge used" or "edge not
  used"), and for disjunctive convex sets the perspective formulation is the
  tightest possible (Ceria & Soares, 1999).
- **In practice the relaxation is often nearly tight.** Rounding the relaxed
  flows yields paths within a few percent of the global optimum, and the
  relaxed value certifies the gap.
- **It extends beyond straight segments.** Bézier curves, velocity limits and
  time scaling fit the same framework (Marcucci et al., 2023).

---

(cvx-ex6)=
## Example 6: Certifiable rotation averaging (SLAM)

### Context

A mobile robot (or a drone, or a handheld camera) moves through an unknown
environment. At times $i = 1,\dots,n$ it has a **pose**: an orientation
$R_i\in SO(3)$ and a position $t_i\in\mathbb R^3$. Here $SO(3)$ is the set of
$3\times3$ rotation matrices ($R^\top R = I$, $\det R = 1$), and $R_i$ maps
body-frame vectors to world-frame vectors: its columns are the robot's axes
expressed in world coordinates.

The robot never observes its poses directly. It observes **relative
motions**:

- **Odometry** (wheel encoders, IMU, or matching consecutive camera or lidar
  frames) compares poses $i$ and $i+1$.
- **Loop closures** happen when the robot recognizes a place it has visited
  before. Matching the current scan to an old one gives a measurement between
  two poses that are far apart in time.

Each such comparison yields a noisy **relative rotation**
$\tilde R_{ij}\approx R_i^\top R_j$, the orientation of frame $j$ as seen from
frame $i$. Together these form a **pose graph**: nodes are poses, edges are
measurements.

Three features of the problem matter:

- **Only relative information is available.** Replacing every $R_i$ by $QR_i$
  for a common rotation $Q$ leaves every $R_i^\top R_j$ unchanged. This is a
  global "gauge" freedom, usually removed by fixing $R_1 = I$.
- **Measurements are redundant and inconsistent.** Composing the measured
  relative rotations around a loop should give $I$, but with noise it does not.
  Estimation finds the best compromise. Hence the name *rotation averaging*.
- **Translations are the easy part.** Full SLAM also estimates the $t_i$ from
  relative translation measurements. For the standard noise model, the cost is
  quadratic in the $t_i$, so they can be eliminated in closed form. What
  remains is a rotation-only problem with the same structure as below
  (SE-Sync, Rosen et al., 2019). Rotation averaging is the hard core of SLAM.

The industry-standard solvers (Gauss–Newton or Levenberg–Marquardt, as in g2o
and GTSAM, initialized from odometry) are fast but local. With large drift or
noise they can converge to a wrong local minimum, producing a twisted or folded
map with no warning. **Certifiable algorithms** solve an SDP relaxation that is
provably exact when the noise is moderate, and they *certify* global
optimality after the fact (Rosen et al., 2019).

### Original problem: deriving the cost

**Noise model.** Write $\tilde R_{ij} = R_i^\top R_j\,E_{ij}$, where $E_{ij}$ is
a random rotation close to $I$. A standard choice is the isotropic
**Langevin** distribution, with density $p(E)\propto\exp(\kappa\operatorname{tr}E)$.

For a rotation by angle $\theta$, $\operatorname{tr}E = 1+2\cos\theta$, so
$p\propto\exp(2\kappa\cos\theta)\approx \text{const}\cdot\exp(-\kappa\theta^2)$
for small $\theta$. This is the rotation analogue of a Gaussian, and the
concentration $\kappa_{ij}$ plays the role of an inverse variance: a confident
measurement has a large $\kappa$.

**Maximum likelihood.** Assuming independent noise, maximize
$\prod p(E_{ij})$, i.e. $\sum\kappa_{ij}\operatorname{tr}E_{ij}$. Solving the
noise model for $E_{ij}$ gives $E_{ij} = R_j^\top R_i\tilde R_{ij}$, so

$$
\operatorname{tr}E_{ij} = \operatorname{tr}(R_j^\top R_i\tilde R_{ij}) = \langle R_j,\ R_i\tilde R_{ij}\rangle_F ,
$$

where $\langle A,B\rangle_F = \operatorname{tr}(A^\top B)$. For rotations,
$\|R_j\|_F^2 = \|R_i\tilde R_{ij}\|_F^2 = 3$, so

$$
\|R_j - R_i\tilde R_{ij}\|_F^2 = 3 + 3 - 2\langle R_j, R_i\tilde R_{ij}\rangle_F = 6 - 2\operatorname{tr}E_{ij} .
$$

Maximizing the likelihood is therefore the same as solving

$$
\min_{R_1,\dots,R_n\in SO(3)}\ \sum_{(i,j)\in E}\kappa_{ij}\,\|R_j - R_i\tilde R_{ij}\|_F^2 .
$$

**Geometric meaning.** $R_i\tilde R_{ij}$ is where pose $i$ and the measurement
*predict* orientation $j$ to be. Each term is the squared **chordal distance**
between that prediction and the estimate $R_j$. For two rotations that differ
by angle $\theta$, it equals $8\sin^2(\theta/2)$, which is increasing in
$\theta$. The cost is a sum of weighted squared prediction errors, just like
least squares.

### Why it is nonconvex

$SO(3)$ is not a convex set. For example, the average of $I$ and a $180°$
rotation about $z$ is $\operatorname{diag}(0,0,1)$, which is not a rotation.
On a graph with loops, the cost has spurious local minima (wrong "wrap-around"
configurations), and local solvers find them when the initialization is poor.

### Convexification (lifting + rank relaxation)

**The cost depends only on pairwise products.** Since
$\langle R_j, R_i\tilde R_{ij}\rangle_F = \langle\tilde R_{ij}, R_i^\top R_j\rangle_F$
(by cyclicity of the trace), the cost depends on the rotations only through
the products $R_i^\top R_j$.

**Collect them in one Gram matrix.** Stack $R = [R_1,\dots,R_n]\in\mathbb R^{3\times 3n}$
and lift to $X = R^\top R\in\mathbb S^{3n}$, whose $(i,j)$ block is
$X_{ij} = R_i^\top R_j$. Define the block matrix $C$ with
$C_{ij} = \kappa_{ij}\tilde R_{ij}$ and $C_{ji} = C_{ij}^\top$ for edges, and
zero elsewhere. Then

$$
\sum_{(i,j)\in E}\kappa_{ij}\|R_j - R_i\tilde R_{ij}\|_F^2 = 6\sum_{(i,j)\in E}\kappa_{ij} - \operatorname{tr}(CX),
$$

which is **linear in $X$**. Minimizing the cost means maximizing
$\operatorname{tr}(CX)$.

**Which $X$ are allowed?** Exactly the matrices with

- $X\succeq0$ (it is a Gram matrix);
- $\operatorname{rank}X\le3$ (since $R$ has 3 rows);
- $X_{ii} = I_3$ (since $R_i^\top R_i = I$);
- $\det R_i = +1$ (proper rotations rather than reflections).

Conversely, any $X\succeq0$ with rank 3 and $X_{ii} = I_3$ factors as
$X = R^\top R$ with every $R_i$ orthogonal. So

$$
\max_X\ \operatorname{tr}(CX)\quad\text{s.t.}\quad X_{ii} = I_3,\quad X\succeq0,\quad
\operatorname{rank}X = 3,\quad \det R_i = +1 .
$$

All the nonconvexity now sits in the rank and determinant constraints.
Dropping them gives an **SDP**.

This is the matrix version of the Goemans–Williamson MaxCut SDP. There,
the unknowns are signs $x_i\in\{\pm1\} = O(1)$, $X_{ii} = 1$, and the
relaxation drops $\operatorname{rank}X = 1$.

**Exactness.** If the SDP solution has rank 3, factor $X^\star = R^{\star\top}R^\star$.
The result is a global optimum (after fixing the determinant signs). Rosen et
al. (2019) prove that this happens whenever the noise is below a threshold set
by the graph's connectivity.

**A posteriori certificate.** Take any candidate $\hat R$ produced by a fast
local solver. Set $\Lambda_i = \sum_j C_{ij}\hat R_j^\top\hat R_i$ and
$\Lambda = \operatorname{blkdiag}(\Lambda_i)$. If

$$
S = \Lambda - C\succeq 0 ,
$$

then $\hat R$ is **globally optimal**.

Why: $\Lambda$ is dual feasible, and complementary slackness
($S\hat R^\top = 0$) makes the primal and dual values equal. Checking the
certificate costs one minimum-eigenvalue computation.

### Takeaways

- **Certifiable perception** (SE-Sync; TEASER for registration with outliers,
  Yang, Shi & Carlone, 2021) turns "convexification" into a practical workflow:
  solve fast, then certify cheaply.
- **When certification fails** (at high noise), you know it, rather than
  silently returning a wrong map.

---

(cvx-part2)=
# Part II: PDE-Constrained Optimization

Structural design *is* PDE-constrained optimization: the state equation is
linear elasticity. A truss (Example 7) is its discrete analogue, in which the
"PDE" is the finite system $K(a)u = f$. Part II therefore treats structures,
transport, inverse problems and nonlinear dynamics together. We begin with the
question that organizes all of them.

(cvx-pde-overview)=
## Overview: when is PDE-constrained optimization convex?

Heating a plate, controlling a reactor, designing a bridge, and identifying
material parameters from measurements are all PDE-constrained optimization
problems. It is often said that such problems are hopelessly nonconvex. The
truth is more nuanced.

### A warm-up problem that is convex: heating a plate

Choose a heat source $u(x)$ (e.g. heating elements under a plate $\Omega$) so
that the temperature $y(x)$ matches a desired profile $y_d(x)$, using little
energy, with actuator limits $u_a\le u\le u_b$:

$$
\min_{u}\ \frac12\|y - y_d\|_{L^2(\Omega)}^2 + \frac{\alpha}{2}\|u\|_{L^2(\Omega)}^2
\quad\text{s.t.}\quad -\Delta y = u \text{ in } \Omega,\quad y = 0 \text{ on }\partial\Omega,\quad u_a\le u\le u_b .
$$

The solution operator $S: u\mapsto y$ of the heat equation is **linear**. The
reduced objective $\tfrac12\|Su-y_d\|^2 + \tfrac\alpha2\|u\|^2$ is therefore a
strictly convex quadratic, and the unique optimum is characterized by the
**adjoint equation**:

$$
-\Delta p = y - y_d,\ \ p|_{\partial\Omega}=0,\qquad
u = \operatorname{Proj}_{[u_a,u_b]}\!\big(-p/\alpha\big)
$$

(Tröltzsch, 2010).

**General rule.** A PDE-constrained problem is convex when the state equation
is affine in (state, control) jointly, the cost is convex, and the constraints
are convex.

### Where nonconvexity comes from

1. **Nonlinear state equations**, e.g. $-\Delta y + y^3 = u$, or Navier–Stokes.
2. **Coefficient (design or parameter) controls**, e.g.
   $-\nabla\cdot(a\nabla y) = f$ with $a$ the control. The product
   $a\nabla y$ is bilinear. **All structural design problems are of this
   type:** the design variable is a stiffness coefficient.

Within case 2, the **objective** decides everything:

- **Compliance-type objectives** $\int f y$ are convex in $a$, by the energy
  principle $\int fy = \sup_v\{2\int fv - \int a|\nabla v|^2\}$. This is why
  Examples 7–8 work.
- **Tracking-type objectives** $\tfrac12\|y(a) - y_{\text{obs}}\|^2$, which
  define inverse problems, are generally nonconvex.

**A one-dimensional illustration.** With constant $a$, $y(a) = y_1/a$, where
$y_1$ solves the problem for $a=1$. The tracking cost

$$
g(a) = \frac{\|y_1\|^2}{2a^2} - \frac{\langle y_1, y_{\text{obs}}\rangle}{a} + \text{const}
$$

is *nonconvex in $a$*, since $g''(a) < 0$ for large $a$, but *convex in $s = 1/a$*,
where it is a quadratic. The choice of parametrization (conductivity versus
resistivity) matters. For spatially varying $a$, no such simple fix exists,
which motivates Examples 12–13.

### Roadmap of Part II

| Example | State equation | Unknown | Objective | Source of convexity |
|---|---|---|---|---|
| 7. Truss | discrete elasticity $K(a)u=f$ | bar areas | compliance | energy principles |
| 8. Continuum topology | elasticity / conduction | material layout | compliance | energy principle + relaxation |
| 9. Frequency | elastic eigenproblem | bar areas | lowest eigenvalue | quasiconvexity (LMI) |
| 10. Frames and shells | elasticity, polynomial stiffness | section sizes | weight, compliance bound | moment-SOS hierarchy |
| 11. Optimal transport | continuity equation | density, velocity | kinetic energy | change of variables $m=\rho v$ |
| 12. EIT | conduction | conductivity | data fit (inverse) | monotonicity + matrix convexity |
| 13. Carleman | Helmholtz / wave | coefficient | data fit (inverse) | Carleman weights |
| 14. Auxiliary functions | nonlinear ODE / PDE | none (analysis) | long-time average | invariant measures |

---

(cvx-ex7)=
## Example 7: Truss topology design for minimum compliance

### Context

Truss topology design was one of the first structural problems to be solved to
global optimality. We decide which bars to keep, and how thick, to make a
structure as stiff as possible for a given amount of material. Here stiffness
means minimum *compliance*, i.e. the work done by the load.

### Original problem

Start from a **ground structure**: nodes on a grid, some of them supported
(fixed), and $m$ candidate bars connecting pairs of nodes. The optimizer
decides how thick each bar is. Bars with zero area disappear, and that is how
the topology is chosen.

**The displacement vector $u$.** When the load is applied, every free node
moves by a small displacement $(u_x, u_y)$ (in 2D). Stacking these for all free
nodes gives the **nodal displacement vector**

$$
u = (u_{1x}, u_{1y}, u_{2x}, u_{2y}, \dots)\in\mathbb R^d,
$$

where $d$ is the number of free degrees of freedom (two per free node in 2D).
Supported nodes do not move, so they are left out. The external load is
stacked the same way into $f\in\mathbb R^d$.

**Kinematics: bar elongation.** Consider bar $i$ from node $p$ to node $q$,
with length $\ell_i$ and unit direction vector $e_i$ (pointing from $p$ to
$q$). For small displacements, its elongation is the relative displacement of
its ends projected onto the bar axis:

$$
\delta_i = e_i^\top(u_q - u_p) = b_i^\top u ,
$$

where $b_i\in\mathbb R^d$ has $+e_i$ in node $q$'s two slots, $-e_i$ in node
$p$'s two slots, and zeros elsewhere (slots of supported nodes are dropped).
Collect these as the columns of $B = [b_1,\dots,b_m]\in\mathbb R^{d\times m}$,
so all elongations are $\delta = B^\top u$.

**Constitutive law (Hooke).** Bar $i$ has strain $\delta_i/\ell_i$, stress
$E\delta_i/\ell_i$, and axial force

$$
q_i = \frac{E a_i}{\ell_i}\,\delta_i = \frac{Ea_i}{\ell_i}\,b_i^\top u ,
$$

with $q_i>0$ in tension and $q_i<0$ in compression. Here $a_i\ge0$ is the
cross-sectional area, which is the design variable.

**Equilibrium.** At every free node, the bar forces balance the external load:
$\sum_i q_ib_i = f$, i.e.

$$
Bq = f .
$$

(The same matrix $B$ appears transposed in kinematics, $\delta = B^\top u$.
This duality between equilibrium and compatibility is the principle of
virtual work.)

**Putting them together.**
$f = \sum_i b_iq_i = \sum_i a_i\frac{E}{\ell_i}b_ib_i^\top u = K(a)\,u$, with

$$
K(a) = \sum_i a_i \frac{E}{\ell_i}\, b_ib_i^\top \qquad(\text{linear in } a).
$$

**The design problem.** The **compliance** $c = f^\top u$ is the work done by
the load. It equals twice the stored strain energy, and smaller means stiffer.
With a material budget $V$:

$$
\min_{a\ge0}\quad c(a) = f^\top u(a)
\quad\text{s.t.}\quad K(a)\,u(a) = f,\qquad \sum_i \ell_i a_i\le V .
$$

### Why it looks nonconvex

$c(a) = f^\top K(a)^{-1}f$ involves a matrix inverse. Also, $K(a)$ becomes
singular as bars are removed ($a_i = 0$), which is exactly what topology
design does.

### Convexification (M3: energy principles)

**Form 1: compliance is convex (minimum potential energy).** For fixed $a$,
consider the concave quadratic $\pi(u) = 2f^\top u - u^\top K(a)u$. Its
maximizer solves $2f - 2K(a)u = 0$, i.e. the equilibrium equation $K(a)u = f$.
The maximum value is $2f^\top K^{-1}f - f^\top K^{-1}f = f^\top K^{-1}f$. So

$$
c(a) = f^\top K(a)^{-1} f = \sup_{u}\ \big\{ 2f^\top u - u^\top K(a) u\big\}
= \sup_u\ \Big\{ 2f^\top u - \sum_i a_i \tfrac{E}{\ell_i}(b_i^\top u)^2\Big\} .
$$

For each fixed $u$, the expression inside the sup is affine in $a$. Hence
$c(a)$ is **convex** (Section 0.4).

The sup form also handles singular $K(a)$ gracefully. If the remaining bars
form a *mechanism* that cannot carry $f$, there is a $u$ with $K(a)u = 0$ and
$f^\top u>0$. Scaling it up sends the sup to $+\infty$, so $c(a) = +\infty$,
which is the physically right answer.

**Form 2: an SDP (derived from Form 1).** We want the epigraph constraint
$\tau\ge c(a)$ as an LMI. By Form 1,

$$
\tau\ge c(a)\iff \tau - 2f^\top u + u^\top K(a)u\ \ge 0\quad\forall u\in\mathbb R^d .
$$

The left side is the quadratic form of a block matrix evaluated at $(1,-u)$:

$$
\begin{bmatrix}s\\-u\end{bmatrix}^{\!\top}
\underbrace{\begin{bmatrix}\tau & f^\top\\ f & K(a)\end{bmatrix}}_{=:M(\tau,a)}
\begin{bmatrix}s\\-u\end{bmatrix}
= \tau s^2 - 2s\,f^\top u + u^\top K(a)u ,\qquad\text{at } s=1 .
$$

- **Vectors with $s\neq0$.** Divide by $s^2$ and rename $u/s$ as $u$. The form
  is nonnegative for all such vectors exactly when the condition holds at
  $s=1$.
- **Vectors with $s=0$.** The form is $u^\top K(a)u\ge0$, which is automatic
  because $a\ge0$ makes $K(a)\succeq0$.

So $\tau\ge c(a)$ if and only if $M(\tau,a)\succeq0$. This is the Schur
complement lemma of Section 0.2, proved directly, and the proof also covers
singular $K$. Since $M$ is **linear in $(\tau,a)$**, the design problem is the
SDP

$$
\min_{a\ge0,\ \tau}\ \tau
\quad\text{s.t.}\quad
\begin{bmatrix}\tau & f^\top\\ f & K(a)\end{bmatrix}\succeq0,\qquad \sum_i\ell_ia_i\le V .
$$

The SDP form matters because it extends to multiple loads, robust loads and
eigenvalue constraints (see Extensions and Example 9), where the LP below
does not.

**Form 3: an LP via bar forces (minimum complementary energy).** Instead of
displacements, optimize over force distributions $q$ that balance the load:

$$
c(a) = \min_{q:\ Bq = f}\ \sum_i \frac{\ell_i q_i^2}{E a_i}.
$$

*Why this holds.* Attach a multiplier $2u$ to $Bq = f$. Stationarity in $q_i$
gives $2\ell_iq_i/(Ea_i) = 2b_i^\top u$, i.e. $q_i = (Ea_i/\ell_i)b_i^\top u$.
This is **Hooke's law, appearing as an optimality condition**. Together with
$Bq=f$, it gives $K(a)u = f$, and the optimal value is
$\sum_iq_i\,b_i^\top u = (Bq)^\top u = f^\top u = c(a)$.

Each term $q_i^2/a_i$ is a perspective (quadratic-over-linear) function, so
the problem is jointly convex in $(q,a)$. Even better, we can minimize over $a$
in closed form. Cauchy–Schwarz on the vectors $(\sqrt{\ell_ia_i})_i$ and
$(\sqrt{\ell_i}\,|q_i|/\sqrt{a_i})_i$ gives

$$
\Big(\sum_i\ell_i|q_i|\Big)^2 \le \Big(\sum_i\ell_ia_i\Big)\Big(\sum_i\frac{\ell_iq_i^2}{a_i}\Big),
$$

with equality iff $\sqrt{\ell_i a_i}\propto\sqrt{\ell_i}|q_i|/\sqrt{a_i}$, i.e.
$a_i\propto|q_i|$. With $\sum_i\ell_ia_i = V$,

$$
\sum_i \frac{\ell_iq_i^2}{Ea_i}\ \ge\ \frac{\big(\sum_i \ell_i|q_i|\big)^2}{E\,V},
\qquad\text{with equality iff } a_i = \frac{V\,|q_i|}{\sum_j\ell_j|q_j|}.
$$

Therefore

$$
\boxed{c^\star = \frac{1}{EV}\Big(\min_{q}\ \sum_i\ell_i|q_i|\ \ \text{s.t.}\ \ Bq = f\Big)^2 .}
$$

**Why the inner problem is an LP**, even though $q_i$ can have either sign.
The objective $\sum_i\ell_i|q_i|$ is not linear, but it is piecewise linear
and convex, and the standard trick turns it into an LP. Split each force into
tension and compression parts, $q_i = q_i^+ - q_i^-$ with $q_i^+,q_i^-\ge0$, and
solve

$$
\min_{q^+,q^-\ge0}\ \sum_i\ell_i\,(q_i^+ + q_i^-)\quad\text{s.t.}\quad B(q^+ - q^-) = f .
$$

At an optimum, at most one of $q_i^+$ and $q_i^-$ is nonzero. If both were
positive, subtracting $\min(q_i^+,q_i^-)$ from both would keep $B(q^+-q^-)$
unchanged and lower the cost. So $q_i^++q_i^- = |q_i|$, and the LP solves the
original problem. (Equivalently, introduce $t_i$ with
$-t_i\le q_i\le t_i$ and minimize $\sum_i\ell_it_i$.)

$\sum_i\ell_i|q_i|$ is the "force × length" volume of a structure in which every
bar works at unit stress. This is the classic *plastic design* LP (Dorn,
Gomory & Greenberg, 1964; Ben-Tal & Bendsøe, 1993).

A bonus interpretation: at the optimum $a_i\propto|q_i|$, so $|q_i|/a_i$, the
stress magnitude, is the same in every bar. **Every bar is equally stressed**,
which is Michell's fully stressed design.

The cell below solves the LP for a cantilever ground structure with scipy.

```{code-cell} ipython3
:tags: [hide-input]

from math import gcd
from scipy.optimize import linprog

nx, ny, h = 9, 5, 0.5                     # 9 x 5 nodes, 4.0 x 2.0 domain
xy = np.array([(i * h, j * h) for i in range(nx) for j in range(ny)])
node = lambda i, j: i * ny + j
fixed = {node(0, j) for j in range(ny)}   # clamp the left edge
free = [d for n in range(len(xy)) if n not in fixed for d in (2 * n, 2 * n + 1)]

bars = []                                  # ground structure: all non-overlapping bars
for p in range(len(xy)):
    for q in range(p + 1, len(xy)):
        di = round(abs(xy[q, 0] - xy[p, 0]) / h)
        dj = round(abs(xy[q, 1] - xy[p, 1]) / h)
        if gcd(di, dj) == 1 and not (p in fixed and q in fixed):
            bars.append((p, q))
m = len(bars)

Bfull = np.zeros((2 * len(xy), m))
L = np.zeros(m)
for k, (p, q) in enumerate(bars):
    d = xy[q] - xy[p]
    L[k] = np.linalg.norm(d)
    e = d / L[k]
    Bfull[2 * p:2 * p + 2, k] = -e
    Bfull[2 * q:2 * q + 2, k] = e
B = Bfull[free]

f = np.zeros(2 * len(xy))
load_node = node(nx - 1, ny // 2)
f[2 * load_node + 1] = -1.0               # unit downward load at mid-right
f = f[free]

# min sum L|q|  s.t. B q = f,   with q = q_plus - q_minus
res = linprog(np.concatenate([L, L]), A_eq=np.hstack([B, -B]), b_eq=f,
              bounds=(0, None), method="highs")
qf = res.x[:m] - res.x[m:]
E, V = 1.0, 1.0
print(f"{m} candidate bars, {np.sum(np.abs(qf) > 1e-6)} bars in the optimal design")
print(f"optimal compliance c* = (sum l|q|)^2 / (E V) = {res.fun**2 / (E * V):.4f}")

fig, ax = plt.subplots(figsize=(7, 3.6))
qmax = np.abs(qf).max()
for k, (p, q) in enumerate(bars):
    if abs(qf[k]) > 1e-4 * qmax:
        ax.plot(*xy[[p, q]].T, color="C3" if qf[k] > 0 else "C0",
                lw=0.6 + 6 * abs(qf[k]) / qmax, solid_capstyle="round")
ax.plot(*xy.T, "k.", ms=2)
ax.plot(*xy[sorted(fixed)].T, "ks", ms=6)
ax.annotate("", xy=xy[load_node] + [0, -0.45], xytext=xy[load_node],
            arrowprops=dict(arrowstyle="->", lw=2))
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Optimal truss from the LP (red: tension, blue: compression)", fontsize=10)
plt.tight_layout()
plt.show()
```

### Extensions

- **Multiple load cases.** Use one LMI per load case.
- **Robust design** (Ben-Tal & Nemirovski, 1997). For all loads in an ellipsoid
  $\{F\zeta:\|\zeta\|\le1\}$,
  $\max_\zeta \zeta^\top F^\top K(a)^{-1}F\zeta\le\tau$ is equivalent to
  $\begin{bmatrix}\tau I & F^\top\\ F & K(a)\end{bmatrix}\succeq0$, which is
  still an SDP.
- **Discrete bar sizes** make the problem a mixed-integer SOCP, solvable to
  global optimality for moderate sizes.

---

(cvx-ex8)=
## Example 8: Continuum topology optimization: convex VTS, nonconvex SIMP, and homogenization

### Context

This is the continuum version of Example 7 and the basis of commercial tools
such as Altair OptiStruct and Ansys. It is also the most important example of
**deliberately introduced nonconvexity**.

### Original problem

On a design domain $\Omega$, choose a density $\rho(x)\in[0,1]$. The material
tensor is $\mathbb C(\rho) = \rho^p\,\mathbb C_0$ with $p \ge 1$ (SIMP:
Solid Isotropic Material with Penalization). The problem is

$$
\min_{\rho}\ c(\rho) = \ell(u_\rho)
\quad\text{s.t.}\quad
\int_\Omega \varepsilon(v):\mathbb C(\rho):\varepsilon(u_\rho)\,dx = \ell(v)\ \ \forall v,\qquad
\int_\Omega\rho\,dx \le V,\quad 0\le\rho\le1 .
$$

After finite element discretization, $K(\rho) = \sum_e \rho_e^p K_e$ and
$c = f^\top K(\rho)^{-1}f$.

### Convexity analysis

**Case $p=1$ (variable-thickness sheet, VTS).** The energy principle gives

$$
c(\rho) = \sup_u\ \Big\{2\ell(u) - \int_\Omega \rho\, \varepsilon(u):\mathbb C_0:\varepsilon(u)\,dx\Big\},
$$

which is convex in $\rho$, exactly as in Example 7. The catch is that optimal
designs have large *gray* regions with $0<\rho<1$. That is fine for a sheet of
varying thickness, but meaningless for a part that must be solid or void.

**Case $p>1$ (SIMP).** Substitute $x_e = \rho_e^p$. The compliance is convex in
$x$ (same argument), but the volume constraint becomes
$\sum_e x_e^{1/p}\le V$. Since $x^{1/p}$ is *concave*, this constraint set is
**nonconvex**. It makes intermediate densities uneconomical, because they give
stiffness $\rho^p < \rho$ for volume $\rho$, and pushes the design toward
$0/1$. The nonconvexity is the price of manufacturability.

**A two-spring illustration.** Two parallel springs with stiffness
$k\rho_i^p$ carry a load $F$, under the constraint $\rho_1+\rho_2 = 1$. Writing
$\rho_1 = t$:

$$
c(t) = \frac{F^2}{k\,\big(t^p + (1-t)^p\big)} .
$$

For $p=1$, every split is optimal. For $p>1$, the two "pure" designs are
**separate global minima**, and the even split becomes a *maximum*: a textbook
nonconvex landscape.

```{code-cell} ipython3
:tags: [hide-input]

t = np.linspace(0, 1, 401)
fig, ax = plt.subplots(figsize=(6, 3))
for p in (1, 2, 3):
    ax.plot(t, 1.0 / (t ** p + (1 - t) ** p), label=f"$p = {p}$")
ax.set_xlabel(r"$\rho_1 = t$   ($\rho_2 = 1 - t$)")
ax.set_ylabel(r"compliance $c(t)\,k/F^2$")
ax.legend()
ax.set_title("SIMP penalization creates multiple minima", fontsize=10)
plt.tight_layout()
plt.show()
```

### Why an optimal 0/1 design may not exist (ill-posedness)

Set SIMP aside and consider the problem we really want to solve: every point
is either solid or void, or more generally one of two materials. **Ill-posed**
here means that the infimum of the compliance over $0/1$ designs is *not
attained by any $0/1$ design*. There is a sequence of designs whose compliance
keeps decreasing, but no limit design.

The phenomenon is easiest to see in the scalar analogue of elasticity, **heat
(or electrical) conduction**. The mathematics is the same, without tensors.

**Setting.** A plate $\Omega$ is heated by a source $f$ (for example,
electronics generating heat). Place a good conductor $\beta$ on a fraction $V$
of $\Omega$, and a poor conductor $\alpha<\beta$ elsewhere:

$$
a(x) = \alpha + (\beta-\alpha)\chi(x),\qquad \chi(x)\in\{0,1\},\qquad \int_\Omega\chi\le V .
$$

The temperature $u$ solves $-\nabla\cdot(a\nabla u) = f$ (with boundary
conditions, e.g. $u = 0$ on a heat sink), and the heat flux is
$J = -a\nabla u$, so $\nabla\cdot J = f$. The **thermal compliance**
$c = \int_\Omega f u\,dx$ is the source-weighted average temperature, which we
want small. By the complementary energy (Thomson) principle, the same one used
for trusses in Example 7,

$$
c(\chi) = \min_{J:\ \nabla\cdot J = f}\ \int_\Omega \frac{|J|^2}{a(x)}\,dx .
$$

The original design problem is

$$
\min_{\chi\in\{0,1\},\ \int\chi\le V}\ \ \min_{J:\ \nabla\cdot J = f}\ \int_\Omega \frac{|J|^2}{\alpha + (\beta-\alpha)\chi}\,dx .
$$

**Why finer is better: laminates.** Take a striped composite of the two
materials, with a fraction $\theta$ of the good one, and stripes much thinner
than the scale over which the heat flow changes. Its effective conductivity
depends on direction:

- **Flux along the stripes.** The phases conduct in parallel, and the effective
  conductivity is the *arithmetic* mean $a_\parallel(\theta) = \theta\beta+(1-\theta)\alpha$.
- **Flux across the stripes.** The phases conduct in series, and the effective
  conductivity is the *harmonic* mean
  $a_\perp(\theta) = \big(\theta/\beta+(1-\theta)/\alpha\big)^{-1}$.

No mixture of the two materials, in any arrangement, conducts better than
$a_\parallel$ or worse than $a_\perp$ in any direction (the Wiener bounds). So
the best possible use of a local fraction $\theta$ of good conductor is
**stripes aligned with the heat flux**.

In a real problem the optimal heat flow *curves* through the domain, so the
stripes must curve with it. They must also be thin compared with the radius of
curvature, or the flux is forced to cross them. Every refinement, with thinner
stripes that follow the flux more faithfully, lowers the compliance. The
limiting object, "infinitely thin stripes", is not a $0/1$ function. The
designs $\chi_k$ oscillate faster and faster and converge only *on average*
(weakly) to a gray fraction $\theta(x)\in(0,1)$. The infimum is approached but
not attained (Murat & Tartar, 1985; Kohn & Strang, 1986). For example, the
optimal two-material torsion bar has a composite region (Goodman, Kohn &
Reyna, 1986).

**What this looks like numerically: mesh dependence.** Refine the finite
element mesh, and a $0/1$ optimizer finds finer and finer features with
ever-lower compliance, without converging. This is why practical topology
optimization always adds a **density filter or minimum length scale**. It
forbids fine features and thereby restores existence, at the price of keeping
the problem nonconvex.

```{code-cell} ipython3
:tags: [hide-input]

alpha, beta = 0.05, 1.0
th = np.linspace(0, 1, 201)
a_par = th * beta + (1 - th) * alpha
a_perp = 1.0 / (th / beta + (1 - th) / alpha)

fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axs[0]
for k in range(6):                       # stripes: good conductor dark, theta = 1/2
    ax.add_patch(plt.Rectangle((0, k / 6), 1, 1 / 12, color="0.25"))
ax.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5),
            arrowprops=dict(arrowstyle="->", color="C3", lw=2.5))
ax.text(0.5, 1.03, r"flux along stripes: $a_\parallel$ (parallel)", ha="center",
        color="C3", fontsize=9)
ax.annotate("", xy=(1.08, 1.0), xytext=(1.08, 0.0),
            arrowprops=dict(arrowstyle="->", color="C0", lw=2.5))
ax.text(1.12, 0.5, r"across: $a_\perp$ (series)", rotation=90, va="center",
        color="C0", fontsize=9)
ax.set_xlim(-0.05, 1.25); ax.set_ylim(-0.05, 1.12)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("a laminate (fine stripes) is anisotropic", fontsize=10)

ax = axs[1]
ax.fill_between(th, a_perp, a_par, color="C7", alpha=0.15,
                label="achievable by some mixture (Wiener bounds)")
ax.plot(th, a_par, "C3", label=r"$a_\parallel$: stripes along the flux (best)")
ax.plot(th, a_perp, "C0", label=r"$a_\perp$: stripes across the flux (worst)")
ax.plot(th, alpha + (beta - alpha) * th ** 3, "k--", lw=1,
        label=r"SIMP $\alpha+(\beta-\alpha)\theta^3$")
ax.set_xlabel(r"local fraction of good conductor $\theta$")
ax.set_ylabel("effective conductivity")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.show()
```

### Relaxation by homogenization: where the convexification happens

There are two remedies for ill-posedness:

- **restrict** the design space (filters, perimeter or length-scale
  constraints), which keeps $0/1$ designs and nonconvexity; or
- **relax** it, by accepting the limits of fine mixtures as legitimate
  designs.

Relaxation is the convexification. Here it is, step by step, for conduction.

**Step 1: enlarge the design space.** At each point, choose a local volume
fraction $\theta(x)\in[0,1]$ and a microstructure. With a single load, the best
microstructure is a laminate aligned with the local flux, so the conductivity
"seen" by the flux is $a_\parallel(\theta) = \alpha+(\beta-\alpha)\theta$.

**Step 2: write the relaxed problem.**

$$
\boxed{\min_{\theta,\ J}\ \int_\Omega\frac{|J|^2}{\alpha+(\beta-\alpha)\theta}\,dx
\quad\text{s.t.}\quad \nabla\cdot J = f,\qquad \int_\Omega\theta\,dx\le V,\qquad 0\le\theta\le1 .}
$$

**Step 3: see that it is convex.** The integrand $|J|^2/(\alpha+(\beta-\alpha)\theta)$
is quadratic-over-linear, a perspective function (Section 0.3), so it is
**jointly convex in $(J,\theta)$**. All constraints are linear. After
discretization, each element contributes a rotated second-order cone
constraint $|J_e|^2\le s_e\,(\alpha+(\beta-\alpha)\theta_e)$, so the problem is
an **SOCP**.

**Compare it with the original problem.** It has the *same integrand*, but
there $\chi$ was restricted to the nonconvex set $\{0,1\}$. The relaxation
simply replaces $\{0,1\}$ by its convex hull $[0,1]$. This is "flux $J$
(total) and fraction $\theta$ (scale)" in the language of Section 0.3.1.
Homogenization theory says the replacement is **not a mathematical trick**:
every gray value $\theta$ is physically realized, in the limit, by fine stripes
aligned with $J$.

**Step 4: what we gain.**

1. **Nothing is lost.** The relaxed minimum equals the infimum over $0/1$
   designs. No mixture beats $a_\parallel$, and aligned laminates achieve it.
2. **The minimum is attained.** The relaxed problem is convex and well-posed,
   so global optimization is easy.
3. **The solution is a recipe.** Where $\theta^\star\in\{0,1\}$, the design is
   classical. Where $0<\theta^\star<1$, it says: build a fine laminate of
   fraction $\theta^\star$ with stripes along $J^\star$.

**A useful coincidence.** This relaxed problem is exactly the $p=1$ ("VTS")
problem, written in flux variables. For conduction with a single load, the
gray regions of the convex $p=1$ problem are therefore **not meaningless**:
they are laminates.

**Step 5: elasticity.** The structure is the same, with tensors instead of
scalars.

- **Single-load compliance.** In 2D, the optimal microstructures are *rank-2
  laminates* (laminates of laminates) aligned with the principal stresses. A
  single family of stripes cannot be stiff in two directions at once, so the
  relaxed integrand is no longer the $p=1$ interpolation. It is still known
  explicitly in terms of the stress $\sigma$ and $\theta$ (Allaire, 2002).
- **The low-volume-fraction limit** ($V\to0$, very light structures). The
  relaxed problem reduces to **Michell's problem**:

  $$
  \min_\sigma\ \int_\Omega\big(|\sigma_1|+|\sigma_2|\big)\,dx\quad\text{s.t.}\quad\nabla\cdot\sigma + f = 0 ,
  $$

  where $\sigma_1,\sigma_2$ are the principal stresses. This is a **convex**
  problem, and it is the continuum analogue of the truss LP of Example 7,
  $\min\sum_i\ell_i|q_i|$ s.t. $Bq = f$. Its solutions are the Michell trusses
  (Michell, 1904).
- **Multiple loads and general objectives.** The full relaxation needs the
  *G-closure* $G_\theta$, the set of all effective tensors achievable by
  mixing at fraction $\theta$. It is generally not known explicitly.
- **Free material optimization (FMO)** goes one step further and lets the
  tensor field itself be the design: $\mathbb C(x)\succeq0$ with
  $\int\operatorname{tr}\mathbb C\le V$. Compliance is again a supremum of
  functions affine in $\mathbb C$, so FMO is **convex**, and an SDP after
  discretization (Ben-Tal, Kočvara, Nemirovski & Zowe, 1999). It ignores
  whether $\mathbb C$ is realizable, and its solutions describe ideal
  material layouts.

**Step 6: back to manufacturable designs.**

- **De-homogenization** projects the optimal laminate fields onto fine,
  manufacturable lattices whose orientation follows the principal directions
  (Pantz & Trabelsi, 2008; Groen & Sigmund, 2018). This gives near-optimal,
  high-resolution designs at low cost.
- **SIMP in this light.** For $p$ large enough ($p\ge3$ for Poisson's ratio
  $1/3$ in 2D), SIMP's gray stiffness lies within the Hashin–Shtrikman bounds,
  so a gray SIMP element corresponds to *some* realizable, but suboptimal,
  microstructure (Bendsøe & Sigmund, 1999). In the conduction figure above,
  the SIMP curve sits far below the aligned-laminate curve $a_\parallel$: gray
  SIMP material is deliberately *inefficient*. (Whether it is realizable at
  all depends on $p$ and the material contrast; here it dips slightly below
  the lower bound at small $\theta$.) The penalty steers the optimizer away
  from inefficient gray material toward $0/1$ designs, and the filter supplies
  the length scale.

| Approach | Design space | Convex? | Optimum exists? |
|---|---|---|---|
| $0/1$ design | $\chi\in\{0,1\}$ | no | not in general |
| $0/1$ + filter / length scale | $\chi\in\{0,1\}$, minimum feature size | no | yes |
| SIMP ($p>1$) + filter | $\rho\in[0,1]$, penalized | no | yes |
| VTS ($p=1$) | $\rho\in[0,1]$ | yes | yes |
| Homogenization (relaxation) | $\theta$ + microstructure | yes for conduction (single load); explicit for elasticity, convex Michell limit | yes |
| Free material optimization | $\mathbb C(x)\succeq0$ | yes (SDP) | yes |

```{admonition} Convexification can be physical
:class: note
In design with PDEs, the "convexified" problem is not a mathematical artifact:
it is the problem of designing **microstructured materials**. The relaxed
optimum is the limit of finer and finer real composites. This answers a common
objection that convexification "doesn't make physical sense" for PDE-constrained
problems.
```

---

(cvx-ex9)=
## Example 9: Maximizing the fundamental frequency (quasiconvexity)

### Context

Structures should have a high fundamental frequency so that they stay away
from resonance with machinery, wind or road excitation. Examples are aircraft
panels, machine frames and bridges.

### Original problem

$$
\max_{a\ge0}\ \lambda_{\min}(a)
\quad\text{where}\quad K(a)\phi = \lambda M(a)\phi,
\qquad \sum_i\ell_ia_i\le V,
$$

with $K(a) = \sum_i a_iK_i$ and $M(a) = M_0 + \sum_i a_iM_i$. Here $M_0\succ0$ is
the non-structural mass, and the fundamental frequency is
$\omega_1 = \sqrt{\lambda_{\min}}$.

### Why it is nonconvex

$\lambda_{\min}(a)$ is a generalized eigenvalue. It is not concave, and it is
**nonsmooth** wherever the lowest eigenvalue is repeated, which typically
happens *at* the optimum.

### Convexification (M2: quasiconvexity + LMI)

By the Rayleigh quotient $\lambda_{\min} = \min_\phi \phi^\top K\phi/\phi^\top M\phi$,

$$
\lambda_{\min}(a)\ \ge\ \lambda
\quad\Longleftrightarrow\quad
K(a) - \lambda M(a)\ \succeq\ 0 .
$$

For fixed $\lambda$, this is an **LMI in $a$**, so every superlevel set
$\{a:\lambda_{\min}(a)\ge\lambda\}$ is convex. In other words, $\lambda_{\min}$
is *quasiconcave*. Bisection on $\lambda$, with one SDP feasibility problem per
step, finds the global optimum (Ohsaki et al., 1999; Achtziger & Kočvara,
2008). Repeated eigenvalues cause no difficulty for the SDP.

### Limits: buckling

Linear buckling replaces $M$ with the geometric stiffness $G(q(a))$, which
depends on the member forces $q(a)$, which are determined by the
displacements $u(a) = K(a)^{-1}f$. The
constraint $K(a) + \lambda G(q(a))\succeq0$ is then **not** an LMI. It is a
bilinear matrix inequality, and the problem is genuinely nonconvex (Kočvara,
2002). It becomes an LMI again only for statically determinate structures,
where $q$ does not depend on $a$.

---

(cvx-ex10)=
## Example 10: Frames and shells: global optimality by the moment-SOS hierarchy

### Context

Unlike truss bars, beams and shells also resist *bending*. Bending stiffness
grows *nonlinearly* with the design variable, which destroys the convexity of
Example 7. Tyburec, Zeman, Kružík & Henrion (2021) showed that such problems
can still be solved to **certified global optimality**.

### Original problem

For Euler–Bernoulli frame elements with geometrically similar cross-sections,
the second moment of area scales as $I_i\propto a_i^2$. For shells of thickness
$t_i$, membrane stiffness is $\propto t_i$ and bending stiffness
$\propto t_i^3$. So

$$
K(a) = \sum_i \big(a_iK_i^{(1)} + a_i^2K_i^{(2)}\big)\quad\text{(frames)},\qquad
K(t) = \sum_i \big(t_iK_i^{(1)} + t_i^3K_i^{(3)}\big)\quad\text{(shells)} .
$$

The weight-minimization problem is

$$
\min_{a\ge0}\ \sum_i\ell_ia_i
\quad\text{s.t.}\quad
\begin{bmatrix}\bar c & f^\top\\ f & K(a)\end{bmatrix}\succeq0 .
$$

### Why it is nonconvex

The constraint is a **polynomial matrix inequality** (PMI). Its feasible set is
not convex because of the $a_i^2$ terms.

### Convexification (M4: moment-SOS)

Treat $a$ as random with an unknown probability measure supported on the
feasible set, and use its moments $y_\alpha = \mathbb E[a^\alpha]$ as variables.

- **Lowest-order relaxation (Shor-type).** Replace $a_i a_j$ by a new variable
  $Y_{ij}$:

  $$
  \begin{bmatrix}1 & a^\top\\ a & Y\end{bmatrix}\succeq0,\qquad
  \begin{bmatrix}\bar c & f^\top\\ f & \sum_i (a_iK_i^{(1)} + Y_{ii}K_i^{(2)})\end{bmatrix}\succeq0 .
  $$

  This is an SDP whose value is a **lower bound** on the minimum weight.
- **Higher orders.** These add higher moments, larger moment matrices and
  *localizing* matrices. The bounds increase monotonically toward the global
  optimum.
- **Certificate (flat extension).** If
  $\operatorname{rank}M_r(y^\star) = \operatorname{rank}M_{r-1}(y^\star)$, the
  relaxation is exact, and the global minimizers can be extracted by linear
  algebra (Curto–Fialkow flat extension).

In Tyburec et al.'s frame and shell examples, the hierarchy converged at a
*small* relaxation order. The output is then a design with a **proof of global
optimality**, something local SIMP-type methods never provide.

### Takeaways

This is the structural counterpart of Example 4: the same moment machinery,
applied to a static design problem. Its cost grows quickly with the number of
design variables, so it is best suited to moderate-size problems or to
certifying designs found by other methods.

---

(cvx-ex11)=
## Example 11: Dynamic optimal transport (Benamou–Brenier)

### Context

How do we steer a *distribution* rather than a single state? Examples include
moving a robot swarm from one formation to another, shaping a particle beam,
or redistributing mass in a fluid. The cheapest way to move a density
$\rho_0$ to $\rho_1$ through a velocity field is a PDE-constrained problem with
an exact convexification (Benamou & Brenier, 2000).

### Original problem

$$
\min_{\rho,\,v}\ \int_0^1\!\!\int_{\mathbb R^d} \tfrac12\rho\,|v|^2\,dx\,dt
\quad\text{s.t.}\quad
\partial_t\rho + \nabla\cdot(\rho v) = 0,\qquad \rho(0,\cdot) = \rho_0,\quad \rho(1,\cdot) = \rho_1 .
$$

### Why it is nonconvex

Both the PDE constraint (via $\rho v$) and the cost (via $\rho|v|^2$) are
*bilinear or cubic* in $(\rho,v)$.

### Convexification (M1: momentum variable + perspective)

Introduce the **momentum** $m = \rho v$:

$$
\boxed{\min_{\rho\ge0,\,m}\ \int_0^1\!\!\int \frac{|m|^2}{2\rho}\,dx\,dt
\quad\text{s.t.}\quad \partial_t\rho + \nabla\cdot m = 0,\quad \rho(0)=\rho_0,\ \rho(1)=\rho_1 .}
$$

- The **PDE is now linear**.
- The **integrand** $|m|^2/(2\rho)$ is the perspective of $\tfrac12|m|^2$, hence
  jointly convex. By convention it equals $0$ when $(\rho,m)=(0,0)$ and $+\infty$
  when $\rho=0$, $m\ne0$.
- After discretization, each cell contributes a **rotated second-order cone**
  constraint $|m|^2\le 2\rho s$.
- The optimal value equals $\tfrac12 W_2^2(\rho_0,\rho_1)$, half the squared
  Wasserstein distance.

### Takeaways

The trick $m = \rho v$ is the same as $Y = KX$ in Example 1, $u = T/m$ in
Example 2, and $z = y\,x$ in Example 5: **multiply the "shape" variable by the
"scale" variable**. The same idea convexifies potential mean-field games and
density steering in control (Chen, Georgiou & Pavon, 2021).

---

(cvx-ex12)=
## Example 12: Electrical impedance tomography as a convex SDP

### Context

**Electrical impedance tomography (EIT)** images the conductivity inside a body
(medical imaging, non-destructive testing of concrete or composites) from
voltage-current measurements on its surface. It is the prototypical *nonlinear,
ill-posed* inverse problem, known as the Calderón problem. Standard
least-squares fitting suffers from local minima. Harrach (2022, 2023) showed
that, with finitely many unknowns, it is **equivalent to a convex nonlinear
semidefinite program**.

### Original problem

The potential $u$ solves

$$
\nabla\cdot(\sigma\nabla u) = 0\ \text{in }\Omega,\qquad \sigma\partial_\nu u = g\ \text{on }\partial\Omega .
$$

The Neumann-to-Dirichlet map is $\Lambda(\sigma): g\mapsto u|_{\partial\Omega}$.

- **Measurements.** Apply boundary currents $g_1,\dots,g_m$ and collect
  $Y_{ij} = \int_{\partial\Omega} g_i\,\Lambda(\sigma^\dagger) g_j$. Define the
  forward map $F(\sigma)\in\mathbb S^m$ by
  $F(\sigma)_{ij} = \int g_i\Lambda(\sigma)g_j$.
- **Unknowns.** The conductivity is piecewise constant on $n$ pixels,
  $\sigma = \sum_{j=1}^n\sigma_j\chi_{P_j}$, with known bounds
  $a\le\sigma_j\le b$.
- **Standard approach.** $\min_\sigma\|F(\sigma) - Y\|^2$, which is nonconvex.

### The key structure (M3 + M5)

The convexification rests on **Thomson's principle**: the dissipated power for
a boundary current $g$ equals the minimum over all divergence-free current
fields $J$ with $J\cdot\nu = g$,

$$
\langle g,\Lambda(\sigma)g\rangle = \min_{J:\ \nabla\cdot J = 0,\ J\cdot\nu = g}\ \int_\Omega \frac{|J|^2}{\sigma}\,dx .
$$

Two consequences follow:

1. **Monotonicity.** $|J|^2/\sigma$ decreases in $\sigma$, so
   $\sigma\le\tau$ pointwise implies $F(\sigma)\succeq F(\tau)$ in the Loewner
   order.
2. **Matrix convexity.** $|J|^2/\sigma$ is a **perspective**, hence jointly
   convex in $(J,\sigma)$, and minimizing over $J$ subject to a linear
   constraint preserves convexity. So $c^\top F(\sigma)c$ is convex in
   $\sigma$ for every $c\in\mathbb R^m$:
   $F(\text{mix})\preceq\text{mix of }F$.

Therefore the set $\{\sigma: F(\sigma)\preceq Y\}$ is **convex**.

### Convexification

Harrach proves that, given enough measurements (an explicitly estimable number
$m$), there is a linear cost $c^\top\sigma$ with $c>0$ such that the true
conductivity is the **unique** solution of

$$
\boxed{\min_{\sigma\in[a,b]^n}\ c^\top\sigma\quad\text{s.t.}\quad F(\sigma)\preceq Y .}
$$

**Intuition.** Any $\sigma\ge\sigma^\dagger$ is feasible by monotonicity.
Minimizing a positive linear cost pushes $\sigma$ downward. Enough measurements
prevent any pixel from dropping below its true value. The precise weights and
the noisy-data version, with error estimates, are in Harrach's papers.

### Takeaways

The same perspective structure $|J|^2/\sigma$ appears here as in trusses
($q^2/a$) and optimal transport ($|m|^2/\rho$). This is not a coincidence: all
three are **complementary-energy** expressions of linear, self-adjoint physics.

---

(cvx-ex13)=
## Example 13: Klibanov's convexification for coefficient inverse problems

### Context

In coefficient inverse problems (CIPs), we recover a spatially varying
coefficient of a PDE from boundary data. Examples include the refractive index
in Helmholtz or wave equations (detecting buried objects with radar,
ultrasound imaging) and diffusion coefficients. These are the PDE problems
where nonconvexity seems most intrinsic: full-waveform inversion suffers from
"cycle skipping" local minima unless the initial guess is very good. Since 1997,
Klibanov and coauthors have developed a method literally called
**convexification**. It builds a cost functional that is *strictly convex on
an arbitrarily large bounded set* (Klibanov, 1997; Klibanov & Li, 2021).

### Original problem (schematic)

Find a coefficient $c(x)$ in, e.g., $\Delta u + k^2 c(x)u = 0$ from boundary
measurements of $u$ for many frequencies $k$ or many source positions. The
usual Tikhonov functional

$$
J(c) = \|\mathcal F(c) - g\|^2 + \beta\|c\|^2
$$

is nonconvex and has many local minima.

### Convexification (M5: Carleman weights)

1. **Eliminate the unknown coefficient.** Change variables (e.g.
   $w=\log u$) and differentiate with respect to a parameter the coefficient
   does not depend on (frequency, source position, or time). This yields a
   nonlinear PDE system $\mathcal L(v) = 0$ for a new unknown function $v$,
   with known boundary data. The coefficient is then recovered from $v$
   explicitly.
2. **Weight the residual with a Carleman weight function**
   $\varphi_\lambda(x) = e^{\lambda\psi(x)}$:

   $$
   J_\lambda(v) = \int_\Omega |\mathcal L(v)|^2\,\varphi_\lambda^2\,dx + \beta\|v\|^2 .
   $$

**Why this convexifies.** Expand $\mathcal L(v+h) = \mathcal L(v) + \mathcal L'_v h + N(v,h)$,
where $N$ is quadratic in $h$. The second variation of $J_\lambda$ contains

$$
\underbrace{\int|\mathcal L'_vh|^2\varphi_\lambda^2}_{\text{good, }\ge0}
\;+\;
\underbrace{2\int\mathcal L(v)\,N(v,h)\,\varphi_\lambda^2}_{\text{indefinite: source of nonconvexity}} .
$$

A **Carleman estimate** for the principal part of the operator gives

$$
\int|\mathcal L'_vh|^2\varphi_\lambda^2\ \ge\ C\lambda\int\big(|\nabla h|^2+\lambda^2|h|^2\big)\varphi_\lambda^2 .
$$

For $\lambda$ large enough, the good term grows like $\lambda^3$ and dominates
the indefinite term, which is bounded independently of $\lambda$ on the ball.

**Theorem (informal).** For every $R>0$, there is a $\lambda_0(R)$ such that
for $\lambda\ge\lambda_0$, $J_\lambda$ is strictly (indeed strongly) convex on
the ball $B(R)$. It has a unique minimizer there, and gradient projection
converges to it **from any starting point in $B(R)$**. With noisy data, the
reconstruction error is Hölder-stable.

### Takeaways and limits

- **"Global" here means no good initial guess is needed.** $R$ can be chosen
  as large as desired.
- **The weight $e^{2\lambda\psi}$ varies over many orders of magnitude,** which
  makes numerics delicate. In practice, moderate $\lambda$ works.
- **The method requires suitable data structure** (multi-frequency or
  multi-source data, Cauchy data on part of the boundary). It has been
  validated on experimental microwave data for detecting buried objects.

---

(cvx-ex14)=
## Example 14: Certified bounds for nonlinear dynamics and PDEs (auxiliary functions)

### Context

For turbulent or chaotic systems, engineers often care about **long-time
averages** rather than individual trajectories: mean heat transport (Nusselt
number), mean drag, mean energy dissipation. Simulating long trajectories
cannot prove an upper bound. A convex dual formulation can.

### Original problem

For $\dot x = f(x)$ (an ODE, or a Galerkin truncation of a PDE) with a compact
absorbing set $B$, and a quantity of interest $\Phi(x)$, compute

$$
\overline\Phi^\star = \sup_{x_0\in B}\ \limsup_{T\to\infty}\frac1T\int_0^T\Phi(x(t))\,dt .
$$

### Why it is nonconvex and hard

The supremum is over chaotic trajectories, and the extremal behavior is often
an unstable periodic orbit that simulations never find.

### Convexification (M4): from trajectories to invariant measures

The key step replaces "search over trajectories" by "search over probability
measures". Here is why that is exact.

**Step 1: a time average is an integral against a measure.** For a trajectory
$x(t)$ and a horizon $T$, define the *time-average measure* $\mu_T$ by

$$
\int g\,d\mu_T := \frac1T\int_0^T g(x(t))\,dt\qquad\text{for every continuous } g .
$$

$\mu_T(A)$ is the *fraction of time* the trajectory spends in the set $A$. It
is the normalized histogram of a long simulation, i.e. the occupation measure
of Example 4 divided by $T$ (and without a control). The time average of
$\Phi$ is then $\int\Phi\,d\mu_T$, which is **linear** in $\mu_T$.

**Step 2: take the long-time limit.** Because the trajectory stays in the
compact set $B$, the probability measures $\mu_T$ have convergent subsequences
as $T\to\infty$. Choose the subsequence that realizes the $\limsup$ of the
time average, and call its limit $\mu$. Then $\int\Phi\,d\mu$ equals that
$\limsup$.

**Step 3: the limit is invariant.** For any $C^1$ function $v$, the chain rule
gives

$$
\int\nabla v\cdot f\,d\mu_T = \frac1T\int_0^T\frac{d}{dt}v(x(t))\,dt = \frac{v(x(T)) - v(x_0)}{T}\ \xrightarrow{T\to\infty}\ 0 ,
$$

because $v$ is bounded on $B$. Hence

$$
\int\nabla v\cdot f\,d\mu = 0\qquad\text{for all } v\in C^1 .
$$

This is the weak form of the *stationary* Liouville equation
$\nabla\cdot(\mu f) = 0$. In words, **the flow does not change $\mu$**. If you
start an ensemble of initial conditions distributed according to $\mu$, it
stays distributed according to $\mu$ forever. Such a $\mu$ is called an
**invariant probability measure**.

**Examples of invariant measures.**

- **Equilibrium** $x^*$ (where $f(x^*) = 0$): $\mu = \delta_{x^*}$, since
  $\int\nabla v\cdot f\,d\delta_{x^*} = \nabla v(x^*)\cdot 0 = 0$.
- **Periodic orbit** of period $P$: $\mu$ spreads mass along the orbit in
  proportion to the time spent there, $\int g\,d\mu = \tfrac1P\int_0^Pg(x(t))\,dt$.
  The check gives $(v(x(P)) - v(x(0)))/P = 0$. Fast parts of the orbit carry
  little mass.
- **Chaotic attractor**: the histogram of a long simulation, shown below for
  the Lorenz system.
- **Mixtures**: any convex combination of invariant measures is invariant. The
  invariant measures form a **convex set** cut out by *linear* equations.

**Step 4: the converse, and the LP.** The extreme points of this convex set
are the *ergodic* measures. By Birkhoff's ergodic theorem, for an ergodic
$\mu$, the time average along $\mu$-almost every trajectory equals
$\int\Phi\,d\mu$. So every extreme invariant measure is the statistics of an
actual trajectory. A linear objective over a convex set is maximized at an
extreme point. Combining both directions,

$$
\boxed{\overline\Phi^\star = \max_{\mu\ \text{invariant}}\int\Phi\,d\mu
\quad\text{s.t.}\quad \int\nabla v\cdot f\,d\mu = 0\ \ \forall v,\quad \mu\ge0,\quad \int d\mu = 1 ,}
$$

an LP over measures. The hard search over initial conditions and infinite
horizons has disappeared.

*Example.* For $\dot x = x - x^3$ there are no periodic orbits (it is 1D), so
the invariant measures are the mixtures of $\delta_{-1}$, $\delta_0$ and
$\delta_{1}$. The largest value of $\int x^2\,d\mu$ is $1$, attained at
$\delta_{\pm1}$. The auxiliary-function bound below recovers this number from
the dual side.

**Why simulation is not enough.** The figure shows a long Lorenz trajectory.
Its histogram (left) is the "physical" invariant measure, and running time
averages from different initial conditions converge to the same value (right).
The attractor, however, also contains infinitely many *unstable periodic
orbits*, each with its own invariant measure and its own average of $\Phi$.
The maximum over *all* invariant measures is often attained by one of these
orbits, which a simulation essentially never follows (Tobasco, Goluskin &
Doering, 2018). The LP, and its dual below, accounts for all of them.

```{code-cell} ipython3
:tags: [hide-input]

def lorenz(t, s, sig=10.0, rho=28.0, b=8.0 / 3.0):
    x, y, z = s
    return [sig * (y - x), x * (rho - z) - y, x * y - b * z]

T, dt = 300.0, 0.01
tt = np.arange(0, T, dt)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
for k, s0 in enumerate([[1.0, 1.0, 1.0], [-8.0, 7.0, 30.0]]):
    sol = solve_ivp(lorenz, (0, T), s0, t_eval=tt, rtol=1e-8, atol=1e-8)
    keep = tt > 10                                  # discard the transient
    x, z = sol.y[0][keep], sol.y[2][keep]
    if k == 0:
        axs[0].hist2d(x, z, bins=120, cmap="Greys", cmin=1)
    run = np.cumsum(z) / np.arange(1, z.size + 1)
    axs[1].plot(tt[keep] - 10, run, label=f"initial condition {k + 1}")
axs[0].set_xlabel("$x$"); axs[0].set_ylabel("$z$")
axs[0].set_title("histogram of one long trajectory\n(an invariant measure)", fontsize=10)
axs[1].set_xlabel("averaging time $T$"); axs[1].set_ylabel(r"$\frac{1}{T}\int_0^T z\,dt$")
axs[1].set_ylim(18, 30)
axs[1].set_title("time averages converge to $\\int z\\,d\\mu$", fontsize=10)
axs[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

**Dual: auxiliary functions.**

$$
\overline\Phi^\star = \inf_{v\in C^1}\ \max_{x\in B}\ \big[\Phi(x) + \nabla v(x)\cdot f(x)\big] .
$$

*Upper bound in one line.* For any $v$,

$$
\frac1T\int_0^T\Phi\,dt = \frac1T\int_0^T(\Phi+\nabla v\cdot f)\,dt - \frac{v(x(T))-v(x_0)}{T}
\le \max_B(\Phi + \nabla v\cdot f) + O(1/T).
$$

Tobasco, Goluskin & Doering (2018) proved that the infimum is **exactly**
$\overline\Phi^\star$: there is no duality gap.

**A worked example.** Take $\dot x = x - x^3$ and $\Phi = x^2$. Try
$v = \tfrac c2x^2$, so $\nabla v\cdot f = c(x^2 - x^4)$ and

$$
\Phi + \nabla v\cdot f = (1+c)x^2 - cx^4\ \le\ \frac{(1+c)^2}{4c}\qquad (c>0).
$$

Choosing $c=1$ gives the bound $\overline{x^2}\le1$. This is sharp: it is
attained at the equilibria $x=\pm1$.

**Computation.** With polynomial $f$, $\Phi$ and $v$, the condition
"$U - \Phi - \nabla v\cdot f\ge0$ on $B$" is relaxed to an SOS condition, giving
an **SDP** for the best bound $U$. This has produced rigorous bounds for the
Lorenz system (Goluskin, 2018) and in fluid dynamics (Chernyshenko et al.,
2014). The classical *background method* for bounding energy dissipation in
Navier–Stokes flows (Doering & Constantin, 1994) is the special case of
quadratic auxiliary functionals. Extensions to PDEs work directly with
occupation measures on function spaces (Marx et al., 2020; Korda, Henrion &
Lasserre, 2022).

---

# Summary

## Patterns

| Pattern | Key object | Examples |
|---|---|---|
| **"Scale × shape" substitution** | $Y=KX$, $u=T/m$, $z = y\,x$, $m=\rho v$, $J/\sigma$ | 1, 2, 5, 11, 12 |
| **Energy principles** | compliance $= \sup_u(\ldots)$ affine in design | 7, 8 |
| **Complementary energy / perspective** | $q^2/a$, $\lvert m\rvert^2/\rho$, $\lvert J\rvert^2/\sigma$ | 7, 11, 12 |
| **Relax, then prove the optimum is on the boundary** | Maximum principle, rank-1/rank-3 conditions | 2, 6 |
| **Quasiconvexity** | superlevel sets are LMIs | 9 |
| **Information structure** | quadratic invariance | 3 |
| **Lift to measures, truncate with SOS** | occupation / invariant measures, moments | 4, 10, 14 |
| **Reweighting and monotonicity** | Carleman weights, Loewner order | 12, 13 |

## Where convexification fails (or is not known)

- **Static output feedback** and decentralized control without QI. These are
  NP-hard in general.
- **SIMP** with $p>1$. Nonconvex *by design*, to obtain $0/1$ layouts.
- **Buckling-constrained design.** The geometric stiffness depends on the
  stresses, which gives a bilinear matrix inequality.
- **Tracking-type inverse problems** without structure such as monotonicity,
  Carleman estimates or a good parametrization (e.g. full-waveform inversion
  with poor initial models).
- **Koopman lifts of systems with multiple attractors.** No finite linear
  embedding exists.
- **Moment-SOS hierarchies in high dimension.** Exact in principle, but the SDP
  size grows combinatorially.

## Exercises

1. **(Example 1)** Prove that if $X\succ0$ and $AX+XA^\top+BY+Y^\top B^\top\prec0$,
   then $K = YX^{-1}$ is stabilizing. Then verify numerically that the average
   of the $(X,Y)$ pairs from two stabilizing gains always yields a stabilizing
   gain.
2. **(Section 0.3)** Show that the perspective of a convex function is convex.
   Conclude that $(m,\rho)\mapsto|m|^2/\rho$ is jointly convex on $\rho>0$, and
   write it as a rotated second-order cone constraint.
3. **(Example 7)** Derive the truss LP from the compliance problem using the
   complementary energy principle and Cauchy–Schwarz. Then show that all bars
   in the optimal design have the same stress magnitude.
4. **(Example 2)** Explain why controllability of $(A,B)$ implies that
   $B^\top\lambda(t)$, with $\dot\lambda = -A^\top\lambda$ and $\lambda\not\equiv0$,
   can vanish only at isolated times.
5. **(Example 3)** For three subsystems in a chain $1\to2\to3$ with
   lower-triangular $G$, decide which of the following controller sparsity
   patterns are QI: (a) lower-triangular; (b) diagonal; (c) a pattern where
   controller 3 sees only subsystems 2 and 3.
6. **(Example 14)** For $\dot x = x - x^3$, find the sharpest bound on
   $\overline{x^4}$ using $v = \tfrac c2 x^2$. Is it sharp?

---

## References

1. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
2. Scherer, C., Gahinet, P., & Chilali, M. (1997). Multiobjective output-feedback control via LMI optimization. *IEEE Transactions on Automatic Control*, 42(7), 896–911.
3. Blondel, V., & Tsitsiklis, J. N. (1997). NP-hardness of some linear control design problems. *SIAM Journal on Control and Optimization*, 35(6), 2118–2127.
4. Fazel, M., Ge, R., Kakade, S., & Mesbahi, M. (2018). Global convergence of policy gradient methods for the linear quadratic regulator. *ICML 2018*.
5. Agarwal, A., Kakade, S. M., Lee, J. D., & Mahajan, G. (2021). On the theory of policy gradient methods: Optimality, approximation, and distribution shift. *Journal of Machine Learning Research*, 22(98), 1–76.
6. Fatkhullin, I., He, N., & Hu, Y. (2023). Stochastic optimization under hidden convexity. ([arXiv:2401.00108](https://arxiv.org/abs/2401.00108))
7. Lasserre, J. B. (2001). Global optimization with polynomials and the problem of moments. *SIAM Journal on Optimization*, 11(3), 796–817.
8. Mohammadi, H., Zare, A., Soltanolkotabi, M., & Jovanović, M. R. (2022). Convergence and sample complexity of gradient methods for the model-free linear-quadratic regulator problem. *IEEE Transactions on Automatic Control*, 67(5).
9. Zheng, Y., Pai, C.-F., & Tang, Y. (2023/2024). Benign nonconvex landscapes in optimal and robust control, Part I: Global optimality ([arXiv:2312.15332](https://arxiv.org/abs/2312.15332)); Part II: Extended convex lifting ([arXiv:2406.04001](https://arxiv.org/abs/2406.04001)).
10. Zheng, Y., Tang, Y., & Li, N. (2023). Analysis of the optimization landscape of linear quadratic Gaussian (LQG) control. *Mathematical Programming*.
11. Açıkmeşe, B., & Ploen, S. R. (2007). Convex programming approach to powered descent guidance for Mars landing. *Journal of Guidance, Control, and Dynamics*, 30(5), 1353–1366.
12. Açıkmeşe, B., & Blackmore, L. (2011). Lossless convexification of a class of optimal control problems with non-convex control constraints. *Automatica*, 47(2), 341–347.
13. Malyuta, D., Reynolds, T. P., Szmuk, M., Lew, T., Bonalli, R., Pavone, M., & Açıkmeşe, B. (2022). Convex optimization for trajectory generation. *IEEE Control Systems Magazine*, 42(5), 40–113. ([arXiv:2106.09125](https://arxiv.org/abs/2106.09125))
14. Blackmore, L. (2016). Autonomous precision landing of space rockets. *The Bridge* (National Academy of Engineering), 46(4).
15. Witsenhausen, H. S. (1968). A counterexample in stochastic optimum control. *SIAM Journal on Control*, 6(1), 131–147.
16. Rotkowitz, M., & Lall, S. (2006). A characterization of convex problems in decentralized control. *IEEE Transactions on Automatic Control*, 51(2), 274–286.
17. Lessard, L., & Lall, S. (2011). Quadratic invariance is necessary and sufficient for convexity. *American Control Conference 2011*.
18. Wang, Y.-S., Matni, N., & Doyle, J. C. (2019). A system-level approach to controller synthesis. *IEEE Transactions on Automatic Control*, 64(10), 4079–4093.
19. Vinter, R. (1993). Convex duality and nonlinear optimal control. *SIAM Journal on Control and Optimization*, 31(2), 518–538.
20. Lasserre, J. B., Henrion, D., Prieur, C., & Trélat, E. (2008). Nonlinear optimal control via occupation measures and LMI-relaxations. *SIAM Journal on Control and Optimization*, 47(4), 1643–1666.
21. Henrion, D., & Korda, M. (2014). Convex computation of the region of attraction of polynomial control systems. *IEEE Transactions on Automatic Control*, 59(2), 297–312.
22. Henrion, D., Korda, M., & Lasserre, J. B. (2020). *The Moment-SOS Hierarchy*. World Scientific.
23. Brunton, S. L., Brunton, B. W., Proctor, J. L., & Kutz, J. N. (2016). Koopman invariant subspaces and finite linear representations of nonlinear dynamical systems for control. *PLoS ONE*, 11(2), e0150171.
24. Korda, M., & Mezić, I. (2018). Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control. *Automatica*, 93, 149–160.
25. Liu, Z., Ozay, N., & Sontag, E. D. (2025). Properties of immersions for systems with multiple limit sets with implications to learning Koopman embeddings. *Automatica*. ([arXiv:2312.17045](https://arxiv.org/abs/2312.17045))
26. Deits, R., & Tedrake, R. (2015). Computing large convex regions of obstacle-free space through semidefinite programming. *Algorithmic Foundations of Robotics XI*, Springer.
27. Marcucci, T., Petersen, M., von Wrangel, D., & Tedrake, R. (2023). Motion planning around obstacles with convex optimization. *Science Robotics*, 8(84), eadf7843.
28. Marcucci, T., Umenhofer, J., Parrilo, P. A., & Tedrake, R. (2024). Shortest paths in graphs of convex sets. *SIAM Journal on Optimization*, 34(1), 507–532.
29. Ceria, S., & Soares, J. (1999). Convex programming for disjunctive convex optimization. *Mathematical Programming*, 86, 595–614.
30. Rosen, D. M., Carlone, L., Bandeira, A. S., & Leonard, J. J. (2019). SE-Sync: A certifiably correct algorithm for synchronization over the special Euclidean group. *International Journal of Robotics Research*, 38(2–3), 95–125.
31. Yang, H., Shi, J., & Carlone, L. (2021). TEASER: Fast and certifiable point cloud registration. *IEEE Transactions on Robotics*, 37(2), 314–333.
32. Dorn, W. S., Gomory, R. E., & Greenberg, H. J. (1964). Automatic design of optimal structures. *Journal de Mécanique*, 3, 25–52.
33. Ben-Tal, A., & Bendsøe, M. P. (1993). A new method for optimal truss topology design. *SIAM Journal on Optimization*, 3(2), 322–358.
34. Ben-Tal, A., & Nemirovski, A. (1997). Robust truss topology design via semidefinite programming. *SIAM Journal on Optimization*, 7(4), 991–1016.
35. Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods and Applications*. Springer.
36. Kohn, R. V., & Strang, G. (1986). Optimal design and relaxation of variational problems, I–III. *Communications on Pure and Applied Mathematics*, 39.
37. Murat, F., & Tartar, L. (1985). Calcul des variations et homogénéisation. In *Les Méthodes de l'Homogénéisation: Théorie et Applications en Physique*, Eyrolles, 319–369.
38. Allaire, G. (2002). *Shape Optimization by the Homogenization Method*. Springer.
39. Ben-Tal, A., Kočvara, M., Nemirovski, A., & Zowe, J. (1999). Free material design via semidefinite programming: The multiload case with contact conditions. *SIAM Journal on Optimization*, 9(4), 813–832.
40. Pantz, O., & Trabelsi, K. (2008). A post-treatment of the homogenization method for shape optimization. *SIAM Journal on Control and Optimization*, 47(3), 1380–1398.
41. Groen, J. P., & Sigmund, O. (2018). Homogenization-based topology optimization for high-resolution manufacturable microstructures. *International Journal for Numerical Methods in Engineering*, 113(8), 1148–1163.
42. Ohsaki, M., Fujisawa, K., Katoh, N., & Kanno, Y. (1999). Semi-definite programming for topology optimization of trusses under multiple eigenvalue constraints. *Computer Methods in Applied Mechanics and Engineering*, 180, 203–217.
43. Achtziger, W., & Kočvara, M. (2008). Structural topology optimization with eigenvalues. *SIAM Journal on Optimization*, 18(4), 1129–1164.
44. Kočvara, M. (2002). On the modelling and solving of the truss design problem with global stability constraints. *Structural and Multidisciplinary Optimization*, 23, 189–203.
45. Tyburec, M., Zeman, J., Kružík, M., & Henrion, D. (2021). Global optimality in minimum compliance topology optimization of frames and shells by moment-sum-of-squares hierarchy. *Structural and Multidisciplinary Optimization*. ([arXiv:2009.12560](https://arxiv.org/abs/2009.12560))
46. Tröltzsch, F. (2010). *Optimal Control of Partial Differential Equations*. AMS Graduate Studies in Mathematics, 112.
47. Benamou, J.-D., & Brenier, Y. (2000). A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem. *Numerische Mathematik*, 84(3), 375–393.
48. Chen, Y., Georgiou, T. T., & Pavon, M. (2021). Optimal transport in systems and control. *Annual Review of Control, Robotics, and Autonomous Systems*, 4, 89–113.
49. Harrach, B. (2022). Solving an inverse elliptic coefficient problem by convex non-linear semidefinite programming. *Optimization Letters*. ([arXiv:2105.11440](https://arxiv.org/abs/2105.11440))
50. Harrach, B. (2023). The Calderón problem with finitely many unknowns is equivalent to convex semidefinite optimization. ([arXiv:2203.16779](https://arxiv.org/abs/2203.16779))
51. Klibanov, M. V. (1997). Global convexity in a three-dimensional inverse acoustic problem. *SIAM Journal on Mathematical Analysis*, 28(6), 1371–1388.
52. Klibanov, M. V., & Li, J. (2021). *Inverse Problems and Carleman Estimates: Global Uniqueness, Global Convergence and Experimental Data*. De Gruyter.
53. Tobasco, I., Goluskin, D., & Doering, C. R. (2018). Optimal bounds and extremal trajectories for time averages in nonlinear dynamical systems. *Physics Letters A*, 382(6), 382–386.
54. Goluskin, D. (2018). Bounding averages rigorously using semidefinite programming: Mean moments of the Lorenz system. *Journal of Nonlinear Science*, 28, 621–651.
55. Chernyshenko, S. I., Goulart, P., Huang, D., & Papachristodoulou, A. (2014). Polynomial sum of squares in fluid dynamics: A review with a look ahead. *Philosophical Transactions of the Royal Society A*, 372, 20130350.
56. Doering, C. R., & Constantin, P. (1994). Variational bounds on energy dissipation in incompressible flows: Shear flow. *Physical Review E*, 49(5), 4087–4099.
57. Marx, S., Weisser, T., Henrion, D., & Lasserre, J. B. (2020). A moment approach for entropy solutions to nonlinear hyperbolic PDEs. *Mathematical Control and Related Fields*, 10(1), 113–140.
58. Korda, M., Henrion, D., & Lasserre, J. B. (2022). Moments and convex optimization for analysis and control of nonlinear PDEs. *Handbook of Numerical Analysis*, 23, 339–366.
