# Project 1: Optimization Problem Formulation

## Introduction

Optimization is the backbone of engineering decision-making. Before any solver can be applied, a real-world problem must be translated into a precise mathematical formulation — with clearly defined decision variables, an explicit objective function, and a complete set of constraints. A vague statement like "minimize cost" is not a formulation; a formulation writes out exactly what cost means in terms of the decision variables.

In this project, your team will:

1. **Identify** a real-world decision-making problem that reflects a genuine need.
2. **Formulate** it as a standard optimization problem with explicit mathematical expressions.
3. **Classify** the problem type (e.g., LP, QP, convex, nonconvex, MILP, combinatorial, nondifferentiable, multi-objective, stochastic) and justify the classification.
4. **Present** your formulation to the class.

Teams of **1 to 5 students** are allowed. Students are encouraged to check with the instructor on whether the chosen problem is sufficiently complex. The bottom line is that the problem should reflect some real-world need.

### Timeline

| Date | Event |
|------|-------|
| Sep 2 | Presenting team announced |
| Sep 4 | Zoom rehearsal |
| Sep 9 | In-class presentation |

---

## Sample Problems

Below are fully formulated examples spanning different optimization problem types. These illustrate the level of detail expected in your report.

### 1. Linear Programming (LP) — Diet Problem

**Context.** A nutrition program needs to design a minimum-cost daily meal plan that meets dietary requirements using available foods.

**Decision variables.** Let $x_i \geq 0$ be the number of servings of food $i$, for $i = 1, \dots, n$.

**Objective.** Minimize total cost:

$$\min_{x} \quad \sum_{i=1}^{n} c_i \, x_i$$

where $c_i$ is the cost per serving of food $i$.

**Constraints.**
- Nutrient requirements: for each nutrient $j = 1, \dots, m$,

$$\sum_{i=1}^{n} a_{ji} \, x_i \geq b_j$$

where $a_{ji}$ is the amount of nutrient $j$ in one serving of food $i$, and $b_j$ is the minimum daily requirement for nutrient $j$.

- Upper bounds on intake (e.g., sodium limits): for selected nutrients $j$,

$$\sum_{i=1}^{n} a_{ji} \, x_i \leq u_j$$

- Non-negativity: $x_i \geq 0$ for all $i$.

**Classification.** This is a **linear program (LP)**: the objective and all constraints are linear in the decision variables. The feasible set is a convex polytope, and the problem is convex.

---

### 2. Quadratic Programming (QP) / Convex — Portfolio Optimization

**Context.** An investor allocates capital across $n$ assets to minimize portfolio risk while achieving a target return (Markowitz model).

**Decision variables.** Let $w_i$ be the fraction of capital invested in asset $i$, for $i = 1, \dots, n$.

**Objective.** Minimize portfolio variance:

$$\min_{w} \quad w^\top \Sigma \, w$$

where $\Sigma \in \mathbb{R}^{n \times n}$ is the covariance matrix of asset returns (symmetric positive semidefinite).

**Constraints.**
- Target return:

$$\mu^\top w \geq r_{\text{target}}$$

where $\mu_i$ is the expected return of asset $i$ and $r_{\text{target}}$ is the minimum acceptable return.

- Budget:

$$\sum_{i=1}^{n} w_i = 1$$

- No short selling: $w_i \geq 0$ for all $i$.

**Classification.** This is a **convex quadratic program (QP)**: the objective is quadratic with a positive semidefinite Hessian $2\Sigma$, and all constraints are linear. The problem is convex.

---

### 3. Nonlinear Programming (NLP) / Nonconvex — Chemical Reactor Design

**Context.** A chemical engineer designs a continuous stirred-tank reactor (CSTR) to maximize product yield by selecting operating conditions.

**Decision variables.**
- $T$: reactor temperature (K), $T \in [300, 600]$
- $P$: reactor pressure (atm), $P \in [1, 50]$
- $F$: feed flow rate (mol/s), $F \in [0.1, 10]$

**Objective.** Maximize yield (conversion $\times$ selectivity):

$$\max_{T, P, F} \quad F \cdot X(T, P) \cdot S(T, P)$$

where the conversion $X$ and selectivity $S$ are nonlinear functions given by Arrhenius kinetics:

$$X(T, P) = 1 - \exp\!\left(-\frac{k_1(T) \, P \, V}{F}\right), \quad k_1(T) = A_1 \exp\!\left(-\frac{E_1}{RT}\right)$$

$$S(T, P) = \frac{k_1(T)}{k_1(T) + k_2(T) \, P}, \quad k_2(T) = A_2 \exp\!\left(-\frac{E_2}{RT}\right)$$

Here $A_1, A_2$ are pre-exponential factors, $E_1, E_2$ are activation energies, $R$ is the gas constant, and $V$ is reactor volume (fixed).

**Constraints.**
- Energy balance:

$$F \, c_p \, (T - T_{\text{in}}) = (-\Delta H) \, F \, X(T, P) + Q$$

where $c_p$ is heat capacity, $T_{\text{in}}$ is feed temperature, $\Delta H$ is heat of reaction, and $Q$ is heat removal rate.

- Safety: $Q \leq Q_{\max}$, and $P \leq P_{\max}$.

- Variable bounds: $T \in [300, 600]$, $P \in [1, 50]$, $F \in [0.1, 10]$.

**Classification.** This is a **nonconvex nonlinear program (NLP)**: the objective involves products and compositions of exponentials, and the energy balance constraint is nonlinear. Multiple local optima are expected.

---

### 4. Mixed-Integer Linear Programming (MILP) — Facility Location

**Context.** A logistics company decides which warehouses to open and how to assign customer demand to minimize total cost (fixed costs of opening + transportation costs).

**Decision variables.**
- $y_j \in \{0, 1\}$: whether to open warehouse $j$, for $j = 1, \dots, p$
- $x_{ij} \geq 0$: fraction of customer $i$'s demand served by warehouse $j$

**Objective.** Minimize total cost:

$$\min_{x, y} \quad \sum_{j=1}^{p} f_j \, y_j + \sum_{i=1}^{n} \sum_{j=1}^{p} c_{ij} \, d_i \, x_{ij}$$

where $f_j$ is the fixed cost of opening warehouse $j$, $c_{ij}$ is the unit transportation cost from $j$ to $i$, and $d_i$ is customer $i$'s demand.

**Constraints.**
- Demand satisfaction: each customer's demand is fully met:

$$\sum_{j=1}^{p} x_{ij} = 1 \quad \forall \; i$$

- Capacity: warehouse $j$ can serve at most $K_j$ units if open:

$$\sum_{i=1}^{n} d_i \, x_{ij} \leq K_j \, y_j \quad \forall \; j$$

- Linking: a customer can only be served by an open warehouse:

$$x_{ij} \leq y_j \quad \forall \; i, j$$

- Integrality: $y_j \in \{0, 1\}$; non-negativity: $x_{ij} \geq 0$.

**Classification.** This is a **mixed-integer linear program (MILP)**: the objective and constraints are linear, but some variables are binary. The problem is nonconvex due to the integrality constraints.

---

### 5. Combinatorial Optimization — University Course Scheduling

**Context.** A university registrar assigns courses to rooms and time slots, minimizing scheduling conflicts and unmet preferences.

**Decision variables.** Let $x_{crt} \in \{0, 1\}$ indicate whether course $c$ is assigned to room $r$ during time slot $t$, for $c = 1, \dots, C$, $r = 1, \dots, R$, $t = 1, \dots, T$.

**Objective.** Minimize total penalty (unmet preferences + room-capacity waste):

$$\min_{x} \quad \sum_{c,r,t} \left( \alpha \, p_{crt} + \beta \, \max(0, \, s_c - q_r) \right) x_{crt}$$

where $p_{crt}$ is the preference penalty for assigning course $c$ to room $r$ at time $t$, $s_c$ is the enrollment of course $c$, $q_r$ is the capacity of room $r$, and $\alpha, \beta$ are weights.

**Constraints.**
- Each course is assigned exactly once:

$$\sum_{r=1}^{R} \sum_{t=1}^{T} x_{crt} = 1 \quad \forall \; c$$

- No room conflict (at most one course per room per time slot):

$$\sum_{c=1}^{C} x_{crt} \leq 1 \quad \forall \; r, t$$

- Instructor conflict (an instructor teaching multiple courses cannot be double-booked): for each instructor $k$ and time slot $t$,

$$\sum_{c \in \mathcal{C}_k} \sum_{r=1}^{R} x_{crt} \leq 1$$

where $\mathcal{C}_k$ is the set of courses taught by instructor $k$.

- Room capacity: $x_{crt} = 0$ if $s_c > q_r$ (or penalize in objective as above).

- Binary: $x_{crt} \in \{0, 1\}$.

**Classification.** This is a **combinatorial optimization** problem (specifically, a binary integer program). The number of feasible assignments grows combinatorially with the number of courses, rooms, and time slots. The problem is NP-hard in general.

---

### 6. Nondifferentiable Optimization — Minimax Regression

**Context.** A data analyst fits a linear model to noisy data, minimizing the worst-case absolute error rather than the sum of squared errors, for robustness to outliers.

**Decision variables.** Let $\beta \in \mathbb{R}^{p}$ be the regression coefficient vector.

**Objective.** Minimize the maximum absolute residual:

$$\min_{\beta} \quad \max_{i=1,\dots,n} \; \left| y_i - x_i^\top \beta \right|$$

where $(x_i, y_i)$ are the $n$ data points, $x_i \in \mathbb{R}^p$, $y_i \in \mathbb{R}$.

**Equivalent reformulation.** Introduce auxiliary variable $z$:

$$\min_{\beta, z} \quad z$$

subject to:

$$y_i - x_i^\top \beta \leq z \quad \forall \; i$$

$$-(y_i - x_i^\top \beta) \leq z \quad \forall \; i$$

This reformulation is an LP in $(\beta, z)$.

**Classification.** In its original form, this is a **nondifferentiable optimization** problem: the max of absolute values is convex but not differentiable everywhere. The epigraph reformulation reveals it as an LP. This example illustrates that problem classification can depend on the formulation.

---

### 7. Multi-Objective Optimization — Structural Truss Design

**Context.** A structural engineer designs a planar truss by choosing member cross-sectional areas to simultaneously minimize weight and minimize maximum deflection.

**Decision variables.** Let $A_i$ be the cross-sectional area of truss member $i$, for $i = 1, \dots, M$, with $A_i \in [A_{\min}, A_{\max}]$.

**Objectives.**

1. Minimize total weight:

$$f_1(A) = \sum_{i=1}^{M} \rho_i \, L_i \, A_i$$

where $\rho_i$ is the material density and $L_i$ is the length of member $i$.

2. Minimize maximum nodal deflection:

$$f_2(A) = \max_{k} \; |u_k(A)|$$

where $u(A)$ is the displacement vector obtained from the finite element equilibrium $K(A) \, u = F$, with $K(A)$ being the global stiffness matrix (depends linearly on $A_i$) and $F$ the applied load vector.

**Constraints.**
- Stress limits: for each member $i$,

$$|\sigma_i(A)| = \left|\frac{F_i^{\text{int}}(A)}{A_i}\right| \leq \sigma_{\text{allow}}$$

- Euler buckling (compression members):

$$|F_i^{\text{int}}(A)| \leq \frac{\pi^2 E_i I_i(A_i)}{L_i^2}$$

where $I_i(A_i)$ is the moment of inertia (depends on cross-section shape and $A_i$).

- Bounds: $A_i \in [A_{\min}, A_{\max}]$.

**Scalarization (weighted sum).** A common approach converts this to a single objective:

$$\min_A \quad \lambda \, f_1(A) + (1 - \lambda) \, f_2(A), \quad \lambda \in [0, 1]$$

Varying $\lambda$ traces out the Pareto front.

**Classification.** This is a **multi-objective optimization** problem. Each single-objective subproblem is a nonconvex NLP due to the implicit dependence of $u(A)$ on $A$ through the stiffness equations and the buckling constraints. The Pareto front reveals the trade-off between structural weight and stiffness.

---

### 8. Stochastic Optimization — Power Grid Generation Scheduling

**Context.** A grid operator schedules power generators for the next day to minimize expected generation cost, accounting for uncertainty in renewable (wind/solar) output.

**Decision variables.**
- $p_g$: planned power output of generator $g$ (MW), for $g = 1, \dots, G$
- $r_g \geq 0$: reserve capacity allocated from generator $g$ (MW)
- $\delta_g^{(s)} \geq 0$: upward adjustment of generator $g$ under scenario $s$

**Scenarios.** Renewable output is modeled by $S$ discrete scenarios, each with probability $\pi_s$ and realized renewable generation $W^{(s)}$.

**Objective.** Minimize expected total cost:

$$\min_{p, r, \delta} \quad \sum_{g=1}^{G} \left( c_g \, p_g + c_g^r \, r_g \right) + \sum_{s=1}^{S} \pi_s \sum_{g=1}^{G} c_g^+ \, \delta_g^{(s)}$$

where $c_g$ is the generation cost, $c_g^r$ is the reserve cost, and $c_g^+$ is the real-time adjustment cost for generator $g$.

**Constraints.**
- Demand balance (planned):

$$\sum_{g=1}^{G} p_g + W^{\text{forecast}} = D$$

where $D$ is the forecasted demand and $W^{\text{forecast}}$ is the expected renewable output.

- Scenario balance: for each scenario $s$,

$$\sum_{g=1}^{G} \delta_g^{(s)} = W^{\text{forecast}} - W^{(s)} \quad \text{(shortfall from renewables)}$$

assuming $W^{(s)} \leq W^{\text{forecast}}$ for simplicity (otherwise, curtailment variables are added).

- Reserve limits:

$$\delta_g^{(s)} \leq r_g \quad \forall \; g, s$$

- Generator capacity:

$$p_g + r_g \leq P_g^{\max}, \quad p_g \geq P_g^{\min} \quad \forall \; g$$

- Ramp rate limits:

$$|p_g + \delta_g^{(s)} - p_g^{\text{prev}}| \leq R_g \quad \forall \; g, s$$

- Non-negativity: $p_g, r_g, \delta_g^{(s)} \geq 0$.

**Classification.** This is a **two-stage stochastic linear program**: the first stage decides $p_g$ and $r_g$ before uncertainty is revealed, and the second stage adjusts $\delta_g^{(s)}$ after each scenario is realized. All functions are linear, so each scenario subproblem is an LP. The challenge is the potentially large number of scenarios.

---

## Requirements

Your team report must be a **single Markdown file** (`report.md` or similar) in your team's collaborative GitHub repository. Only publicly accessible reports will be graded.

Your report must include:

### 1. Problem Identification and Motivation

- Describe a real-world decision-making problem.
- Explain why this problem matters — who faces it, what decisions are at stake, and what makes it nontrivial.

### 2. Decision Variables

- List all decision variables with clear notation.
- Specify units, dimensions, and any bounds (e.g., $x_i \geq 0$, $T \in [300, 600]$ K).
- State whether variables are continuous, integer, or binary.

### 3. Objective Function

- Write the objective function as an **explicit mathematical expression** in terms of the decision variables.
- "Minimize cost" is **not** a formulation. Write what cost equals.
- State whether it is a minimization or maximization problem.

### 4. Constraints

- Write every constraint as an explicit mathematical expression.
- Distinguish equality constraints from inequality constraints.
- Explain the physical or practical meaning of each constraint.

### 5. Problem Classification

- Classify the problem (e.g., LP, QP, NLP, MILP, combinatorial, nondifferentiable, multi-objective, stochastic).
- **Justify** the classification: what makes the problem convex or nonconvex? What introduces nonlinearity or integrality?

### 6. Assumptions and Simplifications

- State any assumptions you made in translating the real-world problem to the mathematical formulation.
- Discuss what aspects of the real problem are not captured by your formulation.

---

## Rubric

### Base Score (100 points)

| Category | Points | Full marks |
|---|---|---|
| Problem motivation and context | 15 | Clear real-world relevance; well-scoped; a reader unfamiliar with the domain can follow |
| Decision variables | 15 | Complete set; well-defined with notation, units, and bounds; variable types specified |
| Objective function | 25 | Explicit mathematical expression (not a placeholder); correct use of variables; min/max stated |
| Constraints | 25 | Complete set of equality and inequality constraints; correctly written; each constraint's meaning explained |
| Problem classification | 15 | Correct type identified; classification justified with reference to problem structure |
| Presentation and clarity | 5 | Well-organized Markdown; math renders correctly on GitHub; readable and professional |

### Bonus (up to +20 points)

If your team solves the optimization problem computationally and presents the results:

| Category | Points |
|---|---|
| Solution methodology | 8 | Appropriate solver/algorithm chosen; implementation described |
| Results and interpretation | 8 | Solution values reported; results interpreted in the context of the original problem |
| Code and reproducibility | 4 | Code is included or linked; another person could reproduce the results |

---

## Submission

1. Create a **public** GitHub repository for your team.
2. Write your report as a single Markdown file in the repository.
3. Submit the repository link on **Canvas**.

### Tips

- Use GitHub's math support: inline math with `$...$` and display math with `$$...$$`.
- Test that your math renders correctly on GitHub before submitting.
- Include figures or diagrams if they help explain the problem.
- Cite any data sources or references you use.
