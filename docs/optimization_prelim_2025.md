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

# Preliminaries: calculus, convexity, and Taylor expansion

Source deck: `Slides/optimization_prelim_2025.pptx`

This page collects the mathematical language used throughout the optimization
course: gradients, Hessians, positive definiteness, convex sets, convex
functions, and Taylor expansion.

## 1. Matrix calculus

For a scalar function $f:\mathbb{R}^n\to\mathbb{R}$, the gradient is the vector
of first derivatives

$$
\nabla f(x)=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}(x)\\
\vdots\\
\frac{\partial f}{\partial x_n}(x)
\end{bmatrix}.
$$

The gradient points in the direction of steepest local ascent. The negative
gradient $-\nabla f(x)$ points in the direction of steepest local descent.

The Hessian is the matrix of second derivatives

$$
\nabla^2 f(x)=
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1\partial x_1}(x) & \cdots &
\frac{\partial^2 f}{\partial x_1\partial x_n}(x)\\
\vdots & \ddots & \vdots\\
\frac{\partial^2 f}{\partial x_n\partial x_1}(x) & \cdots &
\frac{\partial^2 f}{\partial x_n\partial x_n}(x)
\end{bmatrix}.
$$

When $f$ is twice continuously differentiable, $\nabla^2 f(x)$ is symmetric.
If $v$ is a unit vector, then

$$
v^T \nabla^2 f(x) v
$$

is the local curvature of $f$ at $x$ in direction $v$.

## 2. Positive definiteness

For a symmetric matrix $A$:

| Name | Notation | Meaning |
|---|---:|---|
| Positive definite | $A\succ 0$ | $x^T A x > 0$ for all $x\ne 0$ |
| Positive semidefinite | $A\succeq 0$ | $x^T A x \ge 0$ for all $x$ |
| Negative definite | $A\prec 0$ | $x^T A x < 0$ for all $x\ne 0$ |
| Negative semidefinite | $A\preceq 0$ | $x^T A x \le 0$ for all $x$ |
| Indefinite | - | $x^T A x$ is positive for some directions and negative for others |

For a symmetric matrix, these conditions can be checked using eigenvalues:
$A\succeq 0$ iff every eigenvalue is nonnegative, and $A\succ 0$ iff every
eigenvalue is positive.

## 3. Live code: gradient, Hessian, and curvature

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([-1.0, 0.5])

def f(x):
    return 0.5 * x @ A @ x + b @ x

def grad(x):
    return A @ x + b

def hess(x):
    return A

x0 = np.array([1.0, -0.5])
eigvals, eigvecs = np.linalg.eigh(hess(x0))
print("gradient at x0:", grad(x0))
print("Hessian eigenvalues:", eigvals)
print("positive definite?", np.all(eigvals > 0))

theta = np.linspace(0, 2*np.pi, 200)
dirs = np.c_[np.cos(theta), np.sin(theta)]
curvature = np.einsum("ij,jk,ik->i", dirs, A, dirs)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(theta, curvature)
ax.set_xlabel("direction angle")
ax.set_ylabel(r"$v^T \nabla^2 f(x_0)v$")
ax.set_title("Directional curvature of a quadratic")
ax.grid(True, alpha=0.25)
plt.show()
```

## 4. Convex sets

A set $C\subseteq\mathbb{R}^n$ is convex if, for any two points
$x,y\in C$ and any $\theta\in[0,1]$,

$$
\theta x+(1-\theta)y \in C.
$$

Geometrically, every line segment connecting two points in the set remains
inside the set.

Useful facts:

- Intersections of convex sets are convex.
- Affine images of convex sets are convex.
- The convex hull of a set is the smallest convex set containing it.

## 5. Convex functions

A function $f:C\to\mathbb{R}$ is convex if $C$ is convex and

$$
f(\theta x+(1-\theta)y)
\le
\theta f(x)+(1-\theta)f(y),
\qquad \forall x,y\in C,\ \theta\in[0,1].
$$

Strict convexity replaces $\le$ with $<$ for $x\ne y$ and
$\theta\in(0,1)$.

A function is $\mu$-strongly convex if, for $\mu>0$,

$$
f(y)\ge f(x)+\nabla f(x)^T(y-x)+\frac{\mu}{2}\|y-x\|^2.
$$

Convex functions are important because local minimizers are global minimizers.
If $f$ is strictly convex, the minimizer is unique when it exists.

## 6. First- and second-order convexity tests

For differentiable $f$, convexity is equivalent to the tangent-plane
inequality

$$
f(y)\ge f(x)+\nabla f(x)^T(y-x),
\qquad \forall x,y.
$$

For twice differentiable $f$ on a convex domain,

$$
f \text{ is convex}
\quad \Longleftrightarrow \quad
\nabla^2 f(x)\succeq 0,\qquad \forall x.
$$

Similarly, if $\nabla^2 f(x)\succeq \mu I$ for all $x$, then $f$ is
$\mu$-strongly convex.

## 7. Taylor expansion

For a smooth scalar function, the first-order Taylor expansion around $x$ is

$$
f(x+p)=f(x)+\nabla f(x)^T p + o(\|p\|).
$$

The second-order Taylor expansion is

$$
f(x+p)
=f(x)+\nabla f(x)^T p
+\frac{1}{2}p^T\nabla^2 f(x)p
+o(\|p\|^2).
$$

Optimization algorithms use this idea constantly: gradient descent uses a
first-order local model, while Newton and trust-region methods use a quadratic
local model.

## 8. Live code: convexity check for a quadratic

```{code-cell} ipython3
def classify_quadratic(A):
    eigvals = np.linalg.eigvalsh(A)
    if np.all(eigvals > 1e-10):
        return "strictly convex"
    if np.all(eigvals >= -1e-10):
        return "convex"
    if np.all(eigvals < -1e-10):
        return "strictly concave"
    return "indefinite / nonconvex"

matrices = {
    "positive definite": np.array([[2.0, 0.3], [0.3, 1.0]]),
    "positive semidefinite": np.array([[1.0, 1.0], [1.0, 1.0]]),
    "indefinite": np.array([[1.0, 0.0], [0.0, -1.0]]),
}

for name, M in matrices.items():
    print(f"{name:22s}", np.linalg.eigvalsh(M), "->", classify_quadratic(M))
```

## 9. Summary

- $\nabla f(x)$ gives the local slope and steepest ascent direction.
- $\nabla^2 f(x)$ gives local curvature.
- Positive semidefinite Hessians characterize twice differentiable convex
  functions.
- Convex sets contain every line segment between their points.
- Convex functions lie below chords and above tangent hyperplanes.
- Taylor expansion connects local calculus to optimization algorithms.

## Q&A

```{raw} html
<div class="qa-widget" data-context-file="_static/optimization_prelim_context.json" data-lecture-id="optimization_prelim_2025" data-engagement-endpoint="https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/qa">
  <div class="qa-identity">
    <input class="qa-first-name" autocomplete="given-name" placeholder="First name" />
    <input class="qa-last-name" autocomplete="family-name" placeholder="Last name" />
    <input class="qa-university-id" autocomplete="username" placeholder="University ID" />
  </div>
  <div class="qa-input-area">
    <textarea class="qa-question" rows="4" placeholder="Ask about gradients, Hessians, convexity, or Taylor expansion."></textarea>
    <div class="qa-controls">
      <select class="qa-model">
        <option value="openai-fast" selected>openai-fast</option>
        <option value="openai">openai</option>
      </select>
      <button class="qa-submit" type="button" aria-label="Ask"><svg viewBox="0 0 16 16" fill="none"><path d="M3 13L8 3L13 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></button>
    </div>
  </div>
  <div class="qa-answer" aria-live="polite"></div>
</div>
```
