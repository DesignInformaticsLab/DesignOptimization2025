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

# Introduction to design optimization

Source deck: `Slides/optimization_intro_2026.pptx`

This page introduces the optimization framework used throughout the course and
surveys applications in engineering design, optimal control, operations
research, machine learning, and game theory.

## 1. The optimization problem

Optimization means finding the best solution under constraints. The standard
form is

$$
\min_{x \in \mathcal{X}} f(x)
\quad \text{s.t.} \quad
g(x) \le 0, \quad h(x) = 0,
$$

where

- $f(x)$ is the **objective** to minimize,
- $x \in \mathcal{X}$ are the **design variables**,
- $g(x) \le 0$ are **inequality constraints**, and
- $h(x) = 0$ are **equality constraints**.

## 2. Standing assumptions

To simplify the discussion throughout this course we assume:

1. $f : \mathbb{R}^{d_x} \to \mathbb{R}$,
   $g : \mathbb{R}^{d_x} \to \mathbb{R}^{d_g}$,
   $h : \mathbb{R}^{d_x} \to \mathbb{R}^{d_h}$.
2. $f \in C^2$ (Hessian exists) and is bounded.
3. $g, h \in C^1$ (gradients exist).
4. The support $\mathcal{X}$ is compact (closed and bounded).
5. A feasible solution exists: $\exists\, x \in \mathcal{X}$ such that
   $g(x) \le 0$ and $h(x) = 0$.

## 3. Example 1 — engineering design

Design a structure to minimize its weight ($f$) with respect to its topology,
shape, sizes, material, and manufacturing processes ($x$), subject to
maximum stress, fatigue, natural frequency, and cost constraints ($g$, $h$).

```{image} _static/intro_gifs/shape_optimization.gif
:alt: Shape optimization of a structural component
:width: 480px
:align: center
```

**Remarks:**

- Real-world design problems are always multi-objective. Objectives can be
  translated into constraints. Exploration of the *Pareto frontier* is
  necessary in such cases.
- Understanding what problems to solve is more important than knowing how to
  solve these problems.

## 4. Example 2 — optimal control

Design a controller (e.g., a neural network parameterized by $x$) to minimize
a control loss ($f$, e.g., target reaching, stability, energy consumption),
subject to system dynamics and safety constraints ($g$, $h$).

```{image} _static/intro_gifs/robot_control.gif
:alt: Robot balancing via optimal control
:width: 400px
:align: center
```

```{image} _static/intro_gifs/power_grid_control.gif
:alt: Power grid control optimization
:width: 400px
:align: center
```

**Remark:** *Reinforcement learning* solves optimal control problems with
unknown or non-differentiable dynamics (e.g., video/board games, large language
models, contact-rich environments).

## 5. Example 3 — operations research

Optimize a schedule (e.g., which cargos go to which ports, $x$) to minimize
the risk (e.g., due to adversarial weather or attacks), subject to capacity,
demand, and time constraints ($g$, $h$).

```{image} _static/intro_gifs/shipping_routes.gif
:alt: Shipping route optimization
:width: 480px
:align: center
```

## 6. Example 4 — machine learning and AI

Optimize a neural network (parameterized by $x$) to minimize an empirical
loss ($f$, e.g., classification, regression, or distributional errors),
subject to capacity, data, and computational constraints ($g$, $h$).

```{image} _static/intro_gifs/neural_network_training.gif
:alt: Neural network training optimization
:width: 480px
:align: center
```

**Remark:** *Generative AI* = optimization where a neural network is optimized
so that its output distribution matches a data distribution.

```{image} _static/intro_gifs/generative_ai.gif
:alt: Generative AI as distribution matching
:width: 480px
:align: center
```

## 7. Example 5 — zero-sum games

Find the Nash equilibrium ($x$, $y$) of a payoff ($f$, e.g., risk in robust
control or reward in Poker), subject to game dynamics and constraints
($g$, $h$):

$$
\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} f(x, y)
\quad \text{s.t.} \quad
g(x, y) \le 0, \quad h(x, y) = 0.
$$

```{image} _static/intro_gifs/game_theory.gif
:alt: Zero-sum game simulation
:width: 480px
:align: center
```

**Remark:** Games are becoming increasingly valuable to solve in an agentic
world.

## 8. Class contents

This course covers two central questions:

- **Optimality conditions:** How do we determine whether $x^*$ is an optimal
  solution?
- **Algorithms:** How do we search for an optimal solution from any initial
  guess? Does the search converge? How fast does it converge?

**Value of this class in the AI era:**

- A systematic way to formulate, analyze, and solve real-world problems.
- A necessary foundation for careers in AI4Science and AI4Engineering.

## 9. Class rules

- Exploit AI to maximize your efficiency.
- Exams follow practice problems, cheat-sheet only.
- Participation matters:
  - All technical questions and answers during class count.
  - Submit your technical questions any time to the instructor.
  - Record your questions on the course website.

## Q&A

```{raw} html
<div class="qa-widget" data-context-file="_static/optimization_intro_context.json" data-lecture-id="optimization_intro_2026" data-engagement-endpoint="https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/qa">
  <div class="qa-identity">
    <input class="qa-first-name" autocomplete="given-name" placeholder="First name" />
    <input class="qa-last-name" autocomplete="family-name" placeholder="Last name" />
    <input class="qa-university-id" autocomplete="username" placeholder="University ID" />
  </div>
  <div class="qa-input-area">
    <textarea class="qa-question" rows="4" placeholder="Ask about optimization formulation, engineering design, optimal control, ML/AI, or game theory."></textarea>
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
