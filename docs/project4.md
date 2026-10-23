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

# Project 4: PDE-Constrained Design Optimization

## Introduction

In MEE342 you designed a pedestrian bridge by hand and checked it in ANSYS: draw a
structure, run the analysis, read the stresses, change a few beams, run it again. That
loop works for a handful of design variables. It breaks down when a bridge has
hundreds of members, each with its own size, and the analysis has to tell you **which way
to change every one of them at once**.

Project 4 closes that loop with optimization, for an engineering structure of your
team's choice: a bridge, a bracket, a heat sink, a compliant gripper, a cold plate, and
so on (see [the menu](#anchor-structures)). Your team will:

1. **vibe code** (with a coding agent) its own finite element analysis of the structure,
   and **verify** it against exact solutions;
2. make that analysis **differentiable**, so that one extra linear solve returns the
   gradient of each performance metric with respect to *all* design variables;
3. optimize the design subject to several performance requirements, at least one of
   which involves **nonlinear** physics or stability;
4. publish the result as a **website** whose report contains a live 3D (WebGL)
   comparison of the initial and the optimized design.

In every one of these problems, each performance metric is a **functional of the solution
of a PDE**: stresses and displacements come from elasticity, temperatures from the heat
equation, pressure drops from the Stokes or Navier–Stokes equations, frequencies and
buckling loads from eigenproblems of the same operators. That is what makes this
**PDE-constrained optimization**. It is the same structure as the reduced-gradient derivation in the
[KKT notes](constrained_optimization_pt1_2026.md) (sections 7–8): the PDE solution
(displacements, temperatures, velocities) is the **state**, the geometry is the
**decision**, and the adjoint variable is the Lagrange multiplier on the state equation. Projects 2 and 3 also
connect: the stiffness matrix is ill-conditioned ([Project 2](project2.md)), and a
trained surrogate is an alternative to the solver you will write here
([Project 3](project3.md)).

These notes use one structure as a **worked example** throughout: the MEE342 steel
footbridge. The interactive figure below is the instructor's prototype of it. It uses one
set of modeling choices (Pratt truss, 187 tube members, deck loads carried by stringers,
linear buckling). It shows what a finished 3D comparison looks like; it is not a target
for your numbers. Drag to orbit, switch the coloring and the deformed shape, and hover
over members.

(anchor-p4-demo)=

```{raw} html
<iframe class="p4-demo" src="_static/project4/bridge_demo.html" title="Initial vs. optimized footbridge, interactive 3D" loading="lazy"></iframe>
<script>
  window.addEventListener("message", (e) => {
    const h = e.data && e.data.p4demoHeight;
    const f = document.querySelector("iframe.p4-demo");
    if (h && f && e.source === f.contentWindow) f.style.height = h + "px";
  });
</script>
```

| Prototype result | Initial (all tubes $r = 50$ mm, depth 2 m) | Optimized |
|---|---|---|
| Total mass (steel + deck) | 6,716 kg | 6,363 kg |
| Steel mass | 3,890 kg | 3,537 kg |
| Truss depth $H$ | 2.00 m | 3.05 m |
| Max von Mises stress | 173.0 MPa | 50.3 MPa |
| Fatigue factor of safety (Goodman) | 0.62 ✗ | 2.12 ✓ |
| Max deck deflection | 22.2 mm | 12.9 mm |
| Lowest natural frequency | 8.04 Hz | 4.47 Hz |
| Linear buckling load factor $\lambda_{cr}$ | 20.9 | 3.07 |

The initial design is heavier *and* infeasible (it fails fatigue). The optimizer (SLSQP
on 188 variables, 266 iterations, 5 minutes on a laptop) made it lighter and feasible at
the same time. Fatigue and buckling are the active constraints at the optimum.

---

## What your team will do

Teams of **1 to 5 students** (same as Projects 1–3).

1. **Choose a structure** from [the menu](#anchor-structures), or propose your own.
2. **Build and verify the analysis.** Write a finite element code for its governing
   PDE(s) and pass your [verification ladder](#anchor-verification-ladder) before you
   optimize anything.
3. **Differentiate it.** Implement adjoint (implicit-differentiation) gradients of every
   objective and constraint, and check them against finite differences.
4. **Optimize.** Solve the design problem with a gradient-based optimizer and interpret
   the result through its KKT conditions.
5. **Include a nonlinear component** ([N1 or N2](#anchor-nonlinear-component)), and show
   how it changed the design.
6. **Publish.** Your report is a Markdown page on GitHub Pages, with the 3D
   initial-vs-optimized comparison embedded in it. See
   [the deliverable](#anchor-deliverable).

Bonus work (up to 15 points) is listed in the [rubric](#anchor-rubric).

### Timeline

| Date | Event |
|------|-------|
| Oct 28 | Project released |
| Nov 4 | Structure chosen and site URL added to the course participation sheet (column "Project 4 URL"). Teams proposing their own structure need the instructor's approval by this date |
| Nov 18 | Presenting teams announced |
| Nov 25 | In-class presentations; final site due (we grade the commit at the deadline) |

---

(anchor-structures)=
## Choosing your structure

Pick one structure below, or propose your own. ★★ is the effort of the worked example;
★★★★ is ambitious for four weeks, so plan a reduced version (2D, or coarse 3D) first.

| Structure | Design variables | Performance metrics (functionals of the PDE solution) | Governing PDE and discretization | Nonlinear component | Effort |
|---|---|---|---|---|---|
| **Footbridge** ([worked example](#anchor-worked-example)) | Member radii; truss depth, chord profile | Mass; stress and fatigue, deflection, lowest frequency | 3D elasticity reduced to a 3D frame (beam elements); eigenproblems | N1: linear buckling. N2: corotational frame with imperfections | ★★ |
| **Drone arm or bicycle frame** | Tube sizes; joint positions | Mass; stiffness at the load points, lowest frequency above the rotor (or pedaling) excitation, stress | 3D frame; eigenproblem | N1: buckling of thin tubes. N2: large deflection of a long arm | ★★ |
| **Engine-mount bracket** (in the style of the GE jet-engine bracket challenge) | Element densities on a voxel grid (SIMP), or shape parameters | Mass; compliance and peak stress ($p$-norm) under several load cases | 3D linear elasticity, 8-node hexahedral elements | N2: large deformation under the extreme load case. N1: buckling of thin webs | ★★★ |
| **Heat sink** for a power chip | Fin heights, thicknesses, and spacing (or a density field) | Peak chip temperature (thermal resistance); mass | Steady heat conduction in the solid with convective (Robin) boundaries | N2: radiation $\varepsilon\sigma(T^4 - T_\infty^4)$, natural convection $h \propto \Delta T^{1/4}$, or conductivity $k(T)$ | ★★ |
| **Busbar or resistive heater** | Thickness or width distribution | Peak temperature or temperature uniformity; electrical resistance; mass | Coupled $\nabla\cdot(\sigma\nabla\phi) = 0$ and $\nabla\cdot(k\nabla T) + \sigma\lVert\nabla\phi\rVert^2 = 0$ | N2: temperature-dependent $\sigma(T)$ makes the coupling two-way | ★★★ |
| **Compliant gripper or flexure hinge** | Material layout (2D, extruded for display) or flexure widths | Output displacement or mechanical advantage; peak strain | 2D or 3D elasticity | N2: large deformation is how the device works | ★★★ |
| **Cold plate or flow manifold** | Channel widths, or a Brinkman density field | Pressure drop; flow uniformity across outlets; peak temperature | Stokes or Navier–Stokes with Brinkman penalization; optionally heat advection–diffusion | N2: the inertial term at $Re \approx 10$–$200$ | ★★★★ |
| **Stiffened panel or thin cylindrical shell** | Stiffener heights and spacing; wall thickness | Mass; buckling load | Plate or shell elements (or plate plus beam stiffeners) | N1: linear buckling. N2: imperfection sensitivity | ★★★★ |
| **Lattice energy absorber** | Strut radii | Energy absorbed per unit mass up to a crush distance; peak force | 3D frame with plasticity and large deformation | N2, path dependent (adjoint through the load history) | ★★★★ |
| **Soft pneumatic actuator** | Chamber wall thicknesses | Bending angle at a given pressure; peak strain | Hyperelastic (neo-Hookean) solid with a follower pressure load | N2: large-strain hyperelasticity | ★★★★ |

**Proposing your own structure.** It qualifies if:

1. the objective and at least two constraints are functionals of a PDE solution that
   **your** code computes;
2. there are enough design variables for the adjoint method to matter (at least 20);
3. it has a nonlinear component (N1 or N2, below);
4. it can be shown in 3D (a 2D design may be extruded);
5. a reduced version optimizes on a laptop in minutes, so you can iterate.

Write your own requirements with **explicit formulas and limits**, in the style of the
[worked example](#anchor-requirements). Where a limit comes from a standard or a data
sheet, cite it.

(anchor-nonlinear-component)=
### The nonlinear component

At least one of your requirements must go beyond a linear PDE solve. Two kinds count:

- **N1: linearized about a computed state.** An eigenvalue problem whose operator depends
  on the state, such as linear buckling ($K_G$ built from the static stresses) or
  stress-stiffened vibration. The gradient needs an eigenvalue sensitivity **plus** an
  adjoint solve through the state.
- **N2: a nonlinear state equation.** The PDE itself is nonlinear (large deformation,
  radiation, temperature-dependent properties, Navier–Stokes inertia, hyperelasticity,
  plasticity), so you solve it with Newton's method and differentiate it implicitly at
  the converged solution ([implementation iii](#anchor-implementations)).

Either way, report the optimized design **with and without** the nonlinear component, and
explain what changed. If nonlinearity does not change your design, say so and show the
evidence; that is a valid finding (it is what happened for the bridge at service loads).

---

(anchor-worked-example)=
## Worked example: the MEE342 footbridge

The problem is adapted from the
[MEE342 bridge activity](https://github.com/DesignInformaticsLab/DesignInformaticsLab.github.io/blob/master/mee342_bridge_design.md).
Two things change: you write the analysis instead of using ANSYS, and every requirement is
written as an explicit formula, because your code (not ANSYS) has to evaluate it. Teams
that choose the footbridge use this specification as written. Other teams should use it
as the model for their own.

### Geometry, loads, and material

| Item | Specification |
|---|---|
| Span $L$ | 15.0 m, simply supported |
| Deck | 3.0 m wide, 8 mm steel plate covering the full 15 m × 3 m footprint (not a design variable) |
| Supports | Along the two deck end lines. Left end pinned: $u_x = u_y = u_z = 0$. Right end: $u_y = u_z = 0$ ($u_x$ free). Apply them to the end nodes on each end line, not to a single node |
| Load case | Self-weight of all steel (members and deck) + pedestrian pressure $q = 5.0$ kPa downward on the whole deck. Axes: $+z$ up, $x$ along the span |
| Material | Structural steel: $E = 200$ GPa, $\nu = 0.3$, $\rho = 7850$ kg/m³, $S_y = 250$ MPa, $S_{ut} = 460$ MPa, fatigue strength $S_f(10^6) = 86.2$ MPa (the ANSYS structural-steel S–N curve at $10^6$ cycles) |
| Members | Circular hollow sections (CHS). Outer radius $r$; wall $t = r/10$ (so the section has one size variable); $10 \le r \le 150$ mm. Other section families are allowed if you document the formulas |
| Joints | Rigid (frame members share nodes; no joint flexibility) |

The structural concept is yours: Pratt, Warren, or Howe trusses, an arch, a Vierendeel
frame, an underslung truss, and so on. The deck must have **interior support lines**
(stringers or cross-beams), as in MEE342.

(anchor-requirements)=
### Requirements

**Objective.** Minimize the total mass (members + deck).

| # | Requirement | Formula (all under the single load case above) |
|---|---|---|
| R1 | Static strength | $\max \sigma_{vM} \le 160$ MPa |
| R2 | Fatigue | Goodman factor of safety $n_f \ge 2$ (see below) |
| R3 | Deflection | Max vertical deck deflection $\le L/400 = 37.5$ mm |
| R4 | Vibration | **Lowest** natural frequency (any mode shape) $\ge 3.0$ Hz. Mass = steel + deck; pedestrians are not included |
| R5 | Stability (new) | Route S1: linear buckling load factor $\lambda_{cr} \ge 3.0$; or route S2 (nonlinear). See [the bridge's nonlinear component](#anchor-stability-routes) |

Two of these differ from MEE342 on purpose:

- **R4 uses the lowest frequency of any mode**, not "the first vertical mode." Deciding
  which mode is "vertical" is a discrete classification that jumps when two modes swap
  order, which breaks gradients. The lowest-frequency rule is conservative and smooth
  (after aggregation; see [eigenvalue sensitivities](#anchor-eigen-sensitivity)).
- **R5 is new.** Minimum-mass optimization pushes compression members to be slender.
  In an early version of our prototype without a buckling requirement, the optimized
  bridge had compression members loaded to 2.3 times their allowable Euler load (with a
  safety factor of 2); it would buckle.

**Fatigue (R2), Goodman form.** The loading cycles between zero and the full design
load ("zero-based"), so at every point the stress amplitude and the mean stress are equal:
$\sigma_a = \sigma_m = \sigma_{\max}/2$, where $\sigma_{\max}$ is the von Mises stress
under the full load. The endurance strength with the fatigue strength factor
$K_f = 0.7$ is

$$
S_e = K_f\, S_f(10^6) = 0.7 \times 86.2 = 60.3\ \text{MPa}.
$$

The Goodman factor of safety and the requirement are

$$
n_f = \left(\frac{\sigma_a}{S_e} + \frac{\sigma_m}{S_{ut}}\right)^{-1} \ge 2
\quad\Longleftrightarrow\quad
\sigma_{\max} \le \frac{1}{n_f\left(\frac{1}{2 S_e} + \frac{1}{2 S_{ut}}\right)}
= 53.3\ \text{MPa}.
$$

So R2 is a stress limit of **53.3 MPa**, which is much tighter than the 160 MPa of R1.
Fatigue governs; R1 can never be active once R2 holds. Report both anyway. (Using von
Mises stress, which is always positive, as both amplitude and mean follows the MEE342
instruction to use equivalent stress for fatigue. It is conservative for members in
compression.)

### Design variables

- **Required:** a size for every member (or for symmetric member groups, if you justify
  the grouping), plus **at least two shape parameters**: for example, the truss depth and
  a top-chord profile $z(x)$ (camber or arch), or the free coordinates of selected nodes.
- **Optional:** topology, i.e., letting members disappear (a bonus; see the
  [rubric](#anchor-rubric)).

(anchor-deck-modeling)=
### Deck modeling

Pick one option:

- **D1: shell deck.** Model the 8 mm plate with shell (or plate + membrane) elements
  that share nodes with the members, as in the ANSYS activity. Add a shell verification
  test to your ladder (a simply supported square plate under uniform pressure, against
  the Navier series solution).
- **D2: deck as loads and mass.** Do not model the plate's stiffness. Apply the deck's
  pressure and self-weight as **line loads on the members that support the deck**, using
  tributary widths (a one-way plate spanning between stringers). Add the deck mass to
  those members' nodes. Then check the plate itself by hand: a simply supported strip of
  span $s$ (the stringer spacing) under pressure $q_{tot}$ has a peak bending stress

  $$
  \sigma_{\text{plate}} = \frac{6}{t^2}\cdot\frac{q_{tot} s^2}{8} = 0.75\,\frac{q_{tot} s^2}{t^2}.
  $$

  With $q_{tot} = 5.0 + 0.62$ kPa (pedestrians plus plate weight) and $t = 8$ mm, the
  53.3 MPa fatigue limit requires $s \le 0.90$ m. Report this check.

```{admonition} Do not lump the deck load at floor-beam nodes only
:class: warning
In our first prototype the deck load went straight to the nodes where stringers meet
floor beams. The stringers then carried no load, and the optimizer shrank them to the
minimum size: a perfectly "optimal" answer to the wrong model. Subdivide the stringers
(at least two elements per member) and spread the load along them.
```

### Nonlinearity in the footbridge

We tested this before writing the project; you should not have to rediscover it.

1. **Geometric nonlinearity barely matters at the design load.** On a pin-jointed version
   of the bridge, the nonlinear midspan deflection differed from the linear one by
   **0.08%** at the design load, 0.39% at 5 times the design load, and 1.4% at 20 times.
   Steel bridges at service loads are linear. Nonlinearity changes the *design* only
   through **stability** and **ultimate strength**.
2. **The model must be able to show the failure you are checking.** The pin-jointed model
   has no member buckling, so its "limit load" search ran to 190 times the design load at
   3.6 m of deflection, which is meaningless. Beam elements (at least two per member) are
   needed.
3. **Plasticity is path dependent and fragile.** With 1% strain hardening, Newton's method
   diverged right after first yield, because a nearly statically determinate truss turns
   into a mechanism once one member yields. A line search (and, for true collapse loads,
   arc-length continuation) fixed it. The path-dependent adjoint through 23 loading and
   unloading steps then matched finite differences to 7 digits.

(anchor-stability-routes)=
#### The bridge's nonlinear component (R5)

Choose one route.

- **S1: linear buckling (an N1 component).** Compute the axial forces $N$ from the linear static solution
  under the full design load, assemble $K_G(N)$, and find the smallest positive
  $\lambda$ with $(K + \lambda K_G)\phi = 0$. Require $\lambda_{cr} \ge 3.0$. The
  factor 3 covers what the eigenvalue analysis ignores: imperfections, residual stresses,
  and plasticity. Gradients: [eigenvalue sensitivities](#anchor-eigen-sensitivity), with
  the adjoint term through $N(u)$.
- **S2: geometrically nonlinear (an N2 component).** Use a corotational (or total Lagrangian)
  3D frame formulation. Add an initial imperfection shaped like the first buckling mode,
  scaled to a peak of 10 mm. Solve the nonlinear equilibrium $R(u; d, \bar\lambda) = 0$
  up to load factor $\bar\lambda = 2.0$ with load stepping. At that state, compute the
  smallest eigenvalue $\mu_{\min}$ of $K_T\varphi = \mu K_0 \varphi$ (where $K_0$ is the
  small-displacement stiffness), and require $\mu_{\min} \ge 1/3$. For a perfectly linear
  pre-buckling response, $K_T \approx K + 2K_G$, and this reduces to $\lambda_{cr} \ge 3$,
  so S1 and S2 agree in the linear limit; S2 adds imperfections and large rotations.
  Gradients: the custom rule around Newton (implementation iii) combined with the
  eigenvalue sensitivity of $K_T(u(d), d)$. Add V7 to your ladder.

In both routes, report the design **with and without** R5 and explain the difference.
Which members grew? Did the truss get deeper?

---

## Background: PDE-constrained design optimization

### 1. The problem in one page

Every problem in this project has the form

$$
\min_{d}\; f\big(d, u\big)
\quad \text{s.t.} \quad
h(d, u) = 0, \qquad g_i\big(d, u\big) \le 0,\; i = 1,\dots,p,
$$

where $d \in \mathbb{R}^n$ are the **design variables** (sizes, node coordinates,
densities), $u \in \mathbb{R}^m$ is the **state** (in the worked example, nodal
displacements and rotations, six per node; for a heat sink, nodal temperatures), and
$h(d,u) = 0$ is the **discretized PDE**. For linear statics (or linear heat conduction),

$$
h(d, u) = K(d)\,u - F(d) = 0,
$$

where $K$ is the stiffness matrix and $F$ is the load vector (which depends on $d$
through self-weight). In the bridge prototype, $n = 188$ and $m = 1{,}574$.

**Where is the PDE in the bridge?** The members obey 3D linear elasticity. Beam theory reduces that PDE
to an ODE along each member's axis (Euler–Bernoulli or Timoshenko), and the deck plate is
a 2D PDE (Kirchhoff–Love or Mindlin plate theory). The finite element method turns all of
it into $K(d)u = F(d)$. The other responses come from the same operator: natural
frequencies from $K\phi = \omega^2 M \phi$, and buckling from
$(K + \lambda K_G)\phi = 0$.

There are two ways to solve such a problem:

- **Nested (reduced-space, "NAND").** Treat $u = u(d)$ as an implicit function of $d$:
  every time the optimizer proposes a $d$, solve the PDE, evaluate $f$ and $g$, and
  compute their gradients by the adjoint method. This is what structural optimization
  almost always does, and what you will do.
- **Simultaneous (full-space, "SAND", all-at-once).** Make $u$ an optimization variable
  too and let the optimizer satisfy $h = 0$ only at convergence, typically with
  Newton–Krylov methods on the full KKT system (Biros & Ghattas, 2005). This pays off
  for very large or time-dependent problems (CFD, inverse problems), where a full PDE
  solve per iteration is too expensive.

### 2. Building the analysis

This is what your coding agent will write. Keep it small and testable. For any structure:

- **Element level first.** Write one function that maps an element's geometry, design
  variables, and state to its matrices, load vector, and outputs (stress, heat flux).
  Everything else is assembly. This is also the function you will differentiate.
- **Sparse linear algebra.** Assemble global matrices as sparse matrices and use a sparse
  factorization (`scipy.sparse.linalg.splu`) and, for eigenproblems, a shift-invert
  solver (`scipy.sparse.linalg.eigsh` with `sigma=0`). In the bridge prototype, one
  analysis with all four constraint gradients took **4.7 s with dense matrices and 0.6 s
  with sparse ones**. Over 266 optimizer iterations, that is roughly 20 minutes versus 3.
- **Units and precision.** SI units throughout, double precision everywhere.
- **Libraries.** You may use a mesher and plotting libraries. You may use FE libraries
  such as scikit-fem or FEniCSx to *check* your solver, but the graded solver, including
  its gradients, must be your own.

For a 3D frame (the worked example):

- **3D frame element.** 2 nodes, 6 DOFs per node (3 translations, 3 rotations), so a
  12 × 12 element stiffness matrix: axial ($EA/L$), torsion ($GJ/L$), and bending in two
  planes ($EI$), transformed to global coordinates by a rotation matrix built from the
  member axis and a reference vector. Vertical members need a different reference vector;
  this is the most common bug in vibe-coded frame codes.
- **Mass and geometric stiffness.** The consistent mass matrix $M_e$ for frequencies and
  the geometric stiffness $K_{G,e}(N)$ (proportional to the member's axial force $N$) for
  buckling.
- **Discretization.** Use **at least two elements per member**. One cubic element cannot
  represent a member's own buckling mode or the bending of a loaded stringer.
- **Stress recovery.** For a CHS member with end forces $(N, T, M_y, M_z)$,

  $$
  \sigma = \frac{|N|}{A} + \frac{\sqrt{M_y^2 + M_z^2}}{I/r}, \qquad
  \tau = \frac{|T|\, r}{J}, \qquad
  \sigma_{vM} = \sqrt{\sigma^2 + 3\tau^2},
  $$

  evaluated at both ends of every element (transverse shear is neglected for slender
  members; say so). With $r_i = 0.9r$:
  $A = \pi(r^2 - r_i^2)$, $I = \tfrac{\pi}{4}(r^4 - r_i^4)$, $J = 2I$.

(anchor-verification-ladder)=
#### The verification ladder

A vibe-coded solver is wrong until proven otherwise. Before optimizing, build a ladder of
at least six tests and report each one in a table: your number, the exact number, and the
relative error. Every ladder needs these kinds of rungs:

| Kind of test | What it catches | Examples |
|---|---|---|
| **Exact solutions** | Wrong element matrices, signs, units | Textbook beam, plate, and fin solutions; the **method of manufactured solutions** (pick $u$, compute the source term that makes it exact, and check that your solver recovers it) |
| **Convergence rate** | Wrong discretization, wrong boundary conditions | Error vs. mesh size on a log–log plot, with the slope your element should give |
| **Global balance** | Lost loads, missing terms | Reactions equal the applied load; heat in equals heat out; mass flow in equals mass flow out |
| **Invariance and rigid-body modes** | Wrong transformations | Rotating the whole model changes nothing; a free–free model has the right number of zero eigenvalues |
| **Nonlinear benchmark** (for N2) | Wrong tangent, wrong residual | A problem with a known nonlinear response; Newton should converge quadratically |
| **Gradient check** | Wrong adjoint | Adjoint vs. finite differences for every constraint and every kind of design variable, with a Taylor test |

The worked example's ladder:

| Test | Setup | Exact answer |
|---|---|---|
| **V1** Cantilever | Tip load $P$ in each bending plane; tip torque $T$ | $\delta = PL^3/(3EI)$, $\theta = PL^2/(2EI)$, $\phi = TL/(GJ)$ |
| **V2** Simply supported beam | Uniform load $w$; then rotate the whole model by an arbitrary 3D rotation | Midspan $\delta = 5wL^4/(384EI)$; results invariant under the rotation |
| **V3** Beam frequency | Simply supported beam, mass per length $\bar m$; refine the mesh | $f_1 = \dfrac{\pi}{2L^2}\sqrt{EI/\bar m}$ |
| **V4** Euler column | Pinned–pinned column with 1, 2, 4 elements | $P_{cr} = \pi^2 EI/L^2$ |
| **V5** Global checks | Free–free bridge model; supported bridge | Exactly 6 zero-frequency modes; support reactions sum to the total load |
| **V6** Gradient check | Every constraint, several random directions $v$ | Adjoint $\nabla g \cdot v$ vs. central difference; plot the error against the step size $\epsilon$ (it should fall as $\epsilon^2$ until round-off) |

For N2 and plasticity, the bridge ladder adds **V7** (the von Mises two-bar truss: compare your
nonlinear path with the exact
$P(w) = 2EA\,\dfrac{L_0 - L(w)}{L_0}\cdot\dfrac{h - w}{L(w)}$ with
$L(w) = \sqrt{a^2 + (h-w)^2}$) and **V8** (the symmetric three-bar truss: first yield at
$P_y = \sigma_y A(1 + 2\cos^3\alpha)$ and collapse at $P_c = \sigma_y A(1 + 2\cos\alpha)$).

### 3. How to describe a design

| Parameterization | Variables | Notes |
|---|---|---|
| **Sizing** | Cross-section of each member | Smooth, cheap. This project's core |
| **Shape** | Node coordinates, or a few profile parameters that move many nodes (e.g., truss depth, chord camber) | Changes element lengths and orientations, so $K$ depends on $d$ through geometry too. Watch for element inversion and very short members |
| **Topology: ground structure** | Areas of every member in a dense candidate grid; members may vanish | Classical truss layout optimization; becomes a linear program for plastic design (Dorn, Gomory & Greenberg, 1964; Gilbert & Tyas, 2003; Zegard & Paulino, 2014) |
| **Topology: density (SIMP)** | One density per finite element of a continuum | The workhorse of continuum topology optimization (Bendsøe, 1989; Sigmund, 2001; Andreassen et al., 2011), up to billions of elements (Aage et al., 2017). See `Demo/topology_optimization.ipynb` |
| **Topology: level set / explicit components** | A level-set function, or moving bars and plates | Crisp boundaries (Allaire et al., 2004; Guo et al., 2014) |
| **Neural reparameterization** | Weights of a network that outputs the design | Can find different local optima (Hoyer et al., 2019); recent neural-field methods are judged mainly by comparison with SIMP, with mixed results |

### 4. Sensitivity analysis and implicit differentiation

#### Why gradients, and why not finite differences

A gradient by forward differences costs $n + 1$ analyses (central differences cost
$2n$). In the prototype, one analysis took 0.29 s, so one finite-difference Jacobian of the
four constraints would take **55–110 s**. The adjoint method gave all four gradients
**exactly** in 0.63 s total, including the analysis itself. The cost of the adjoint
method grows with the number of *outputs*, not the number of design variables. That is
why we aggregate many stress constraints into one (see below).

#### The implicit function theorem gives two formulas

This is the derivation in section 7 of the [KKT notes](constrained_optimization_pt1_2026.md)
with $s \to u$. If $h(d, u) = 0$ and $\partial h/\partial u$ is invertible, then
$u = u(d)$ locally and

$$
\frac{\mathrm{d} u}{\mathrm{d} d} = -\left(\frac{\partial h}{\partial u}\right)^{-1}
\frac{\partial h}{\partial d},
\qquad
\frac{\mathrm{d} f}{\mathrm{d} d} = \frac{\partial f}{\partial d}
- \frac{\partial f}{\partial u}\left(\frac{\partial h}{\partial u}\right)^{-1}
\frac{\partial h}{\partial d}.
$$

The product on the right can be evaluated in either order:

- **Direct (forward) method:** solve $\dfrac{\partial h}{\partial u} S =
  -\dfrac{\partial h}{\partial d}$ for $S = \mathrm{d}u/\mathrm{d}d$. That is $n$ linear
  solves, one per design variable, after which any number of outputs is cheap.
- **Adjoint (reverse) method:** solve one linear system per output $f$,

  $$
  \left(\frac{\partial h}{\partial u}\right)^{\!\top} \lambda = -\left(\frac{\partial f}{\partial u}\right)^{\!\top},
  \qquad
  \frac{\mathrm{d} f}{\mathrm{d} d} = \frac{\partial f}{\partial d} + \lambda^\top \frac{\partial h}{\partial d}.
  $$

The $\lambda$ here is exactly the $\lambda$ of section 8 of the KKT notes: the Lagrange
multiplier of the state equation. **Implicit differentiation is the adjoint method.**
Reverse-mode automatic differentiation (AD) frameworks implement the same formula
whenever they differentiate a linear solve, and you can install it by hand around any
solver.

#### Specialized to linear statics

With $h = K(d)u - F(d)$, we have $\partial h/\partial u = K$ (symmetric), and

$$
K\lambda = -\left(\frac{\partial f}{\partial u}\right)^{\!\top},
\qquad
\frac{\mathrm{d} f}{\mathrm{d} d_j} = \frac{\partial f}{\partial d_j}
+ \lambda^\top\left(\frac{\partial K}{\partial d_j}u - \frac{\partial F}{\partial d_j}\right).
$$

The global matrix derivative is never formed: $\lambda^\top (\partial K/\partial d_j)\,u
= \sum_e \lambda_e^\top (\partial k_e/\partial d_j)\, u_e$ is a sum over the elements
that $d_j$ touches. **Example (compliance).** For $f = F^\top u$, the adjoint is
$\lambda = -u$ (the problem is *self-adjoint*), and
$\mathrm{d}f/\mathrm{d}d_j = 2u^\top \partial F/\partial d_j - u^\top (\partial K/\partial d_j)u$.
The first term exists only because self-weight depends on the design, which is why
compliance with self-weight is not monotone in member size (Bruyneel & Duysinx, 2005).

(anchor-implementations)=
#### Three ways to implement it

**(i) Let AD differentiate through the solve.** In JAX (or PyTorch), a differentiable
assembly plus a linear solve is enough; the framework's rule for `solve` *is* the adjoint
formula. Good for a first version and for checking a hand-written adjoint.

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)    # double precision for FE work

def compliance(d):
    K, F = assemble(d)               # your differentiable (dense) assembly
    u = jnp.linalg.solve(K, F)       # reverse mode = one adjoint solve with K^T
    return F @ u

grad_C = jax.grad(compliance)(d0)
```

**(ii) Element-level AD + hand-assembled adjoint (what scales).** Assemble sparse
matrices and solve with SciPy. Use AD only on the *element* function
$(x_a, x_b, r, u_e) \mapsto (k_e, m_e, k_{G,e}, f_e, \sigma_{vM,e})$, and apply its
vector–Jacobian product with cotangents built from $\lambda_e$ and $u_e$. The prototype
does this; it is how production FE codes compute sensitivities.

**(iii) A custom rule around a nonlinear solver.** For a nonlinear state equation
$h(d,u) = R(u; d) = 0$ solved by Newton's method, the gradient needs only the tangent
matrix $K_T = \partial R/\partial u$ *at the converged solution*:

```python
from functools import partial

def newton(resid, d, u0, tol=1e-10, maxit=30):
    u = u0
    for _ in range(maxit):
        r = resid(u, d)
        if jnp.linalg.norm(r) < tol:
            return u
        u = u - jnp.linalg.solve(jax.jacfwd(resid)(u, d), r)
    raise RuntimeError("Newton did not converge")

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def solve(resid, d):
    return newton(resid, d, jnp.zeros(NDOF))

def solve_fwd(resid, d):
    u = solve(resid, d)
    return u, (u, d)

def solve_bwd(resid, saved, u_bar):          # u_bar = (df/du)^T
    u, d = saved
    K_T = jax.jacfwd(resid)(u, d)            # tangent at the solution
    adj = jnp.linalg.solve(K_T.T, u_bar)     # one adjoint solve (adj = -lambda)
    _, vjp_d = jax.vjp(lambda dd: resid(u, dd), d)
    return (-vjp_d(adj)[0],)                 # df/dd = lambda^T dR/dd

solve.defvjp(solve_fwd, solve_bwd)
```

(This simple version runs outside `jax.jit`, because Newton's stopping test is a Python
`if`. Use `jax.lax.while_loop` if you need `jit`.)

How the alternatives compare, measured on a geometrically nonlinear truss version of the
bridge (122 DOFs, load factor 20):

| Gradient through a nonlinear solve | Cost | Accuracy (vs. central differences) |
|---|---|---|
| Implicit (custom rule above) | 0.8 s | Exact (6+ matching digits) |
| Unrolled: AD through all 8 Newton iterations | 3.0 s, memory grows with iterations | Exact *if* Newton converged; wrong if stopped early |
| "Jacobian-free" backprop (replace $K_T^{-\top}$ by $I$) | Cheapest | **Wrong**: cosine similarity with the true gradient 0.09, magnitude off by $3\times10^6$ |

Jacobian-free and "phantom" gradients (Fung et al., 2022; Geng et al., 2021) are
popular for training implicit neural networks, where the Jacobian is close to the
identity. A stiffness matrix is nothing like the identity (and has units), so the
approximation fails. Do not let a coding agent "speed up" your adjoint this way.

(anchor-eigen-sensitivity)=
#### Eigenvalue sensitivities

For $K\phi = \omega^2 M\phi$ with $\phi^\top M \phi = 1$, differentiate, multiply by
$\phi^\top$, and use symmetry:

$$
\frac{\mathrm{d}(\omega^2)}{\mathrm{d} d_j} = \phi^\top\left(\frac{\partial K}{\partial d_j}
- \omega^2 \frac{\partial M}{\partial d_j}\right)\phi.
$$

This is implicit differentiation of the eigenproblem, and it needs no extra solve. For
linear buckling, write the eigenproblem as $(-K_G)\psi = \mu K\psi$ with $\mu =
1/\lambda_{cr}$ and $\psi^\top K\psi = 1$:

$$
\frac{\mathrm{d}\mu}{\mathrm{d} d_j} = \psi^\top\left(-\frac{\mathrm{d} K_G}{\mathrm{d} d_j}
- \mu\frac{\partial K}{\partial d_j}\right)\psi.
$$

$K_G$ depends on the axial forces, which depend on $u(d)$. So
$\mathrm{d}K_G/\mathrm{d}d_j$ has a term through $u$ that needs **one adjoint solve**,
exactly as for a stress constraint.

**Repeated eigenvalues.** A symmetric bridge has nearly equal pairs of eigenvalues (the
two sides buckle or sway together or in opposition), and an individual eigenvalue is not
differentiable where two cross (Seyranian et al., 1994). Constrain a smooth symmetric
function of several of them instead, such as the KS aggregate below over the lowest 6–10
frequencies (or highest 6–10 values of $\mu$). The gradient of such a function stays
well defined when eigenvalues swap order.

#### Aggregating constraints

The stress constraint is really one constraint per evaluation point (748 in the
prototype). Their maximum is not differentiable, and 748 separate constraints would need
748 adjoint solves. The Kreisselmeier–Steinhauser (KS) function replaces the maximum
with a smooth upper bound:

$$
\mathrm{KS}_\rho(a) = a_{\max} + \frac{1}{\rho}\ln\sum_i e^{\rho(a_i - a_{\max})},
\qquad
a_{\max} \le \mathrm{KS}_\rho(a) \le a_{\max} + \frac{\ln N}{\rho}.
$$

With $a_i = \sigma_{vM,i}/53.3$ MPa, $N = 748$ and $\rho = 60$, the bound is conservative
by at most 11%. In the prototype, the actual peak stress at the optimum was 50.3 MPa,
6% below the limit. The $p$-norm is a common alternative (Le et al., 2010). Report the
true maximum alongside the aggregate.

#### Verifying gradients

- Central differences with a step-size sweep: the error should decrease as $\epsilon^2$
  until round-off takes over (the **Taylor test**). A flat error curve means a bug, not
  a bad step.
- Use **double precision**. In single precision, finite differences of an FE model are
  mostly noise.
- Check *every* constraint and every kind of design variable (sizes and shape, for
  example). Shape gradients are where bugs hide.

### 5. Optimizers and practical issues

- **Algorithms.** SciPy's `SLSQP` handled 188 variables and 4 constraints well.
  `trust-constr` also returns Lagrange multipliers. For larger problems, use the method of
  moving asymptotes (MMA; Svanberg, 1987), the standard in structural optimization, or
  IPOPT (`cyipopt`).
- **Scale everything.** Optimize $x = d/d_0$ and write constraints as $g/g_{\text{limit}} -
  1 \le 0$.
- **Read the KKT conditions at your optimum.** Which constraints are active? What are
  their multipliers? By section 11 of the KKT notes, a multiplier is the improvement in
  the objective per unit relaxation of that limit. For the bridge: how many kilograms is
  the $\lambda_{cr} \ge 3$ rule costing? Report these numbers.
- **Singular optima.** If members can shrink to zero, a vanishing member can keep a
  finite stress limit and trap the optimizer (Kirsch, 1990; Cheng & Guo, 1997). Lower
  bounds on size avoid this in sizing problems; topology optimization needs
  $\varepsilon$-relaxation or similar.
- **Local optima.** Shape and topology variables make the problem nonconvex. Run several
  starts and report the spread.

### 6. Nonlinear PDEs

Most real design problems are governed by nonlinear PDEs. The adjoint idea survives, but
both the forward solve and the gradient change.

| Nonlinearity | What changes in the forward solve | What changes in the gradient | Representative work |
|---|---|---|---|
| **Geometric** (large displacements and rotations) | Newton's method with load stepping and line search; arc-length continuation near limit points | Path-independent, so **one** adjoint solve with $K_T$ at the final equilibrium. At a limit point $K_T$ is singular and the implicit function theorem fails | Buhl, Pedersen & Sigmund (2000); Bruns & Tortorelli (2001); Wang et al. (2014) |
| **Stability** (buckling) | Linearized eigenvalue problem, or a nonlinear analysis with imperfections | Eigenvalue sensitivities; aggregation for repeated eigenvalues. Adaptive arc-length step control can make responses jump between designs, which misleads gradient-based optimizers | Ferrari & Sigmund (2019); Hrinda & Nguyen (2008) |
| **Material** (plasticity) | Incremental return mapping at every integration point; the answer depends on the load *history* | Adjoint swept **backward through the load steps**; the internal-variable history must be stored or recomputed (checkpointing) | Michaleris, Tortorelli & Vidal (1994); Wallin, Jönsson & Wingren (2016); Jia & Zhang (2025) |
| **Fluids** (Navier–Stokes) | Brinkman penalization for topology; turbulence models (RANS) | The adjoint must include the turbulence model ("frozen turbulence" gives wrong gradients). For chaotic unsteady flow, adjoints of time averages grow exponentially; shadowing methods address this | Borrvall & Petersson (2003); Gersborg-Hansen, Sigmund & Haber (2005); Dilgen et al. (2018); Wang, Hu & Blonigan (2014) |
| **Contact, fracture** | Non-smooth complementarity conditions | Gradients exist only piecewise; smoothing or non-smooth methods | — |

Differentiable FE libraries now implement these rules for you. Examples are
dolfin-adjoint/pyadjoint (Farrell et al., 2013; Mitusch et al., 2019), JAX-FEM (Xue et
al., 2023), which covers hyperelasticity, plasticity and 3D topology optimization, and
torch-fem. You may use them for verification. Your graded solver must be your own, so
that you understand every term in it.

### 7. Vibe coding a solver responsibly

Coding agents write FE codes quickly, and they write them **plausibly wrong**. The
rules:

- **You** derive the expected values for your ladder by hand (or from a textbook) *before* the
  agent sees the test. An agent that writes both the code and the expected answer can
  make them agree with each other and not with physics.
- Typical agent bugs in 3D frames (the worked example): the rotation matrix for vertical members, the sign of
  the $\theta_y$ coupling terms, $J$ versus $I_p$, mm versus m, missing self-weight,
  deck mass counted twice, wrong DOF order in assembly, and supports applied to the
  wrong DOFs.
- Keep **one** results file (`results.json`) written by the solver. Your report's tables
  and the 3D viewer both read from it, so no number is copied by hand.
- Record what the agent wrote and how you checked it (see the
  [AI-use statement](#anchor-report-contents)).

---

(anchor-deliverable)=
## The deliverable: a report website

Your report is a Markdown page served by **GitHub Pages** from your team's public
repository. GitHub's ordinary Markdown view does not run JavaScript, so a 3D viewer cannot
live inside a `.md` file viewed on github.com. GitHub Pages builds the same Markdown into
a real web page, where it can. The report, its math, and its 3D comparison then live in
one place.

### A minimal setup (tested)

We built and tested this with Jekyll 3.10, the version GitHub Pages runs. Four files:

````text
your-repo/
├── _config.yml
├── _layouts/default.html
├── index.md                  ← your report
└── assets/results.json       ← written by your solver
````

`_config.yml`:

```yaml
title: "Team X: Heat Sink Optimization"
markdown: kramdown
kramdown:
  input: GFM
  math_engine: mathjax
```

`_layouts/default.html` (loads MathJax for math and declares where three.js comes from):

```html
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ page.title | default: site.title }}</title>
<script>
  window.MathJax = { tex: { inlineMath: [['$','$'], ['\\(','\\)']] } };
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
<script type="importmap">
{ "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
               "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/" } }
</script>
</head><body>{{ content }}</body></html>
```

`index.md` starts with front matter and can contain math, tables, and the viewer:

```html
---
layout: default
title: Heat sink report
---

# Heat sink optimization

The peak chip temperature is $T_{\max} \le 85\,^\circ\mathrm{C}$.

<div id="viewer" style="height:480px"></div>
<script type="module">
  import * as THREE from "three";
  import { OrbitControls } from "three/addons/controls/OrbitControls.js";
  const data = await (await fetch("assets/results.json")).json();
  // ... build the scene from data ...
</script>
```

Then in the repository settings, set **Pages → Source: Deploy from a branch** (`main`,
root). The site appears at `https://<user>.github.io/<repo>/` about a minute after each
push.

Pitfalls we hit while testing:

- **Math.** Display math in `$$ … $$` is converted by kramdown and rendered by MathJax.
  Inline `$ … $` passes through untouched and is rendered by MathJax only because of the
  `inlineMath` setting above. Check every equation on the live site, not on github.com.
- **Liquid.** Jekyll treats `{{` and `{%` as template code, which can break LaTeX such as
  `\frac{{a}}{b}`. Wrap such passages in `{% raw %}` … `{% endraw %}`.
- **Paths.** A project site lives under `/<repo>/`. Use relative paths
  (`assets/results.json`), not paths starting with `/`.
- **Pin versions** of three.js and MathJax (as above), so that the page does not break
  when a library updates.

Other static-site tools (MyST/Jupyter Book or Quarto, deployed by GitHub Actions) are
fine if the result is the same: one public page with the report and the live 3D
comparison.

### The 3D comparison: requirements

The [demo above](#anchor-p4-demo) meets all of these for the bridge. Yours must meet them
for your structure, and you must write your own.

1. **Initial and optimized designs** side by side (or a toggle), with **linked cameras**.
2. **True geometry** in 3D (a 2D design extruded is fine). State any exaggeration, such
   as thickened members.
3. **Color by the field behind your most important constraint** (stress utilization,
   temperature, pressure, strain) with a **color bar**. Use a single-hue sequential scale
   for magnitudes and a two-hue diverging scale with a neutral midpoint for signed
   quantities. Never use a rainbow ("jet") map: it creates false boundaries and fails
   for color-blind readers. Regions over a limit must be identifiable without relying on
   color alone (a distinct mark, a label, or the tooltip).
4. **The state on the geometry**: the deformed shape with an exaggeration slider for
   structures, or the field itself for thermal and flow problems. Interpolate with the
   element's own shape functions. In the bridge, members are drawn as **cubic (Hermite)
   curves** using both end rotations, not straight lines between displaced nodes.
5. **One animation** that explains the design: a vibration or buckling mode, the
   nonlinear load–response path, or the optimization history.
6. **Inspect on hover or click**: identify the element or member and show its key
   values.
7. A **metrics panel** for each design: objective and every constraint value, each
   marked pass or fail.
8. Loads from `results.json`, works at phone width, and renders in a few seconds.

Ideas for going further (not graded separately): an **optimization-history scrubber**
that morphs the design through the iterations; field values varying *along* each member
or over each face; cut planes or iso-surfaces for 3D fields; dark mode.

(anchor-report-contents)=
### What the report must contain

1. **Summary.** Headline numbers and the 3D comparison at the top.
2. **Structure and formulation.** What the structure is and why its performance matters;
   design variables (with units and bounds), objective, and every constraint as an
   explicit formula with the source of each limit.
3. **Analysis model.** Governing PDE(s), element formulation, mesh, loads, boundary
   conditions, and modeling choices (for the bridge, deck option D1 or D2). List your
   assumptions.
4. **Verification.** Your ladder, as a table.
5. **Sensitivities.** Derive the adjoint for each kind of constraint; show the
   gradient-check table and Taylor-test plot; compare the cost of the adjoint method with
   finite differences on your model.
6. **Optimization.** Algorithm, scaling, starting designs, convergence history, active
   constraints and their multipliers, with an engineering interpretation.
7. **Nonlinear component.** N1 or N2; the design with and without it.
8. **Results table**: objective and every constraint for the initial and optimized
   designs.
9. **Bonus sections**, if any.
10. **Limitations** and an **AI-use statement**: what the coding agent wrote, which
    parts you derived or checked yourselves, and one bug the verification caught.

---

(anchor-rubric)=
## Rubric (100 points + up to 15 bonus)

| Category | Points | Full marks |
|---|---|---|
| Structure and formulation | 10 | A real engineering motivation; explicit variables, objective, and constraint formulas with justified limits |
| Analysis code and verification | 20 | A ladder covering every kind of test, all passing with errors reported; assumptions stated |
| Sensitivity analysis | 20 | Correct adjoint derivations; gradient checks for every constraint and every kind of variable; cost comparison with finite differences |
| Optimization | 15 | Sound setup and scaling; converged, feasible design; active constraints and multipliers interpreted |
| Nonlinear component | 15 | N1 or N2 done correctly; effect on the design shown and explained |
| Website and 3D comparison | 10 | Meets the eight viewer requirements; report and viewer in one place; math renders |
| Reproducibility | 5 | One command regenerates `results.json` and every number on the site |
| Presentation and clarity | 5 | Organized and readable by an engineer who has not taken this course |
| **Bonus: fully nonlinear** | +5 | The nonlinear component is N2 (a nonlinear state equation solved by Newton's method, with implicit-differentiation gradients), checked against a nonlinear benchmark |
| **Bonus: path or time dependence** | +5 | A history-dependent response (plasticity, transient heating, a load sequence) with an adjoint swept backward through the history, checked against finite differences |
| **Bonus: topology** | +5 | Material or members allowed to vanish (density, level set, or ground structure); how singular optima or checkerboards were handled; comparison with the fixed-topology design |
| **Bonus: design under uncertainty** | +5 | Loads, material properties, or geometry treated as uncertain (several sampled scenarios or a worst case), with gradients of the robust objective or constraint |

The bonus is capped at +15.

---

## Submission

1. Create a **public** GitHub repository for your team, with the solver code, the
   optimization scripts, and the website.
2. Enable GitHub Pages and check that the site works in a private browser window.
3. Submit **both links** (repository and site) on **Canvas**.

### Tips

- **Verification first.** Most of the time in this project goes into V1–V6. A wrong
  gradient of a wrong model optimizes beautifully.
- **Start small.** For the bridge: a single 2D truss panel, then a 3D box, then the
  bridge. For a continuum: one element, then a coarse 2D mesh, then 3D. Each step should
  pass the ladder.
- **Gradient-check early and often**, every time you add a response or a variable type.
- **Time your code.** If one analysis takes more than a few seconds, switch to sparse
  matrices before you optimize.
- **Save the optimization history** (design and constraint values per iteration). It
  makes the convergence plot and the history scrubber easy.
- **Coding agents** are encouraged. They are good at writing element matrices and
  three.js scenes. Check every agent-written formula against the verification ladder, not
  against the agent's own explanation.

---

## References

1. Aage, N., Andreassen, E., Lazarov, B. S., & Sigmund, O. (2017). Giga-voxel computational morphogenesis for structural design. *Nature*, 550, 84–86.
2. Allaire, G., Jouve, F., & Toader, A.-M. (2004). Structural optimization using sensitivity analysis and a level-set method. *Journal of Computational Physics*, 194(1), 363–393.
3. Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O. (2011). Efficient topology optimization in MATLAB using 88 lines of code. *Structural and Multidisciplinary Optimization*, 43, 1–16.
4. Bendsøe, M. P. (1989). Optimal shape design as a material distribution problem. *Structural Optimization*, 1, 193–202.
5. Biros, G., & Ghattas, O. (2005). Parallel Lagrange–Newton–Krylov–Schur methods for PDE-constrained optimization. Part I: The Krylov–Schur solver. *SIAM Journal on Scientific Computing*, 27(2), 687–713.
6. Borrvall, T., & Petersson, J. (2003). Topology optimization of fluids in Stokes flow. *International Journal for Numerical Methods in Fluids*, 41(1), 77–107.
7. Bruns, T. E., & Tortorelli, D. A. (2001). Topology optimization of non-linear elastic structures and compliant mechanisms. *Computer Methods in Applied Mechanics and Engineering*, 190(26–27), 3443–3459.
8. Bruyneel, M., & Duysinx, P. (2005). Note on topology optimization of continuum structures including self-weight. *Structural and Multidisciplinary Optimization*, 29, 245–256.
9. Buhl, T., Pedersen, C. B. W., & Sigmund, O. (2000). Stiffness design of geometrically nonlinear structures using topology optimization. *Structural and Multidisciplinary Optimization*, 19, 93–104.
10. Cheng, G. D., & Guo, X. (1997). ε-relaxed approach in structural topology optimization. *Structural Optimization*, 13, 258–266.
11. Dilgen, C. B., Dilgen, S. B., Fuhrman, D. R., Sigmund, O., & Lazarov, B. S. (2018). Topology optimization of turbulent flows. *Computer Methods in Applied Mechanics and Engineering*, 331, 363–393.
12. Dorn, W. S., Gomory, R. E., & Greenberg, H. J. (1964). Automatic design of optimal structures. *Journal de Mécanique*, 3, 25–52.
13. Farrell, P. E., Ham, D. A., Funke, S. W., & Rognes, M. E. (2013). Automated derivation of the adjoint of high-level transient finite element programs. *SIAM Journal on Scientific Computing*, 35(4), C369–C393.
14. Ferrari, F., & Sigmund, O. (2019). Revisiting topology optimization with buckling constraints. *Structural and Multidisciplinary Optimization*, 59, 1401–1415.
15. Fung, S. W., Heaton, H., Li, Q., McKenzie, D., Osher, S., & Yin, W. (2022). JFB: Jacobian-free backpropagation for implicit networks. *AAAI 2022*.
16. Geng, Z., Zhang, X.-Y., Bai, S., Wang, Y., & Lin, Z. (2021). On training implicit models. *NeurIPS 2021*.
17. Gersborg-Hansen, A., Sigmund, O., & Haber, R. B. (2005). Topology optimization of channel flow problems. *Structural and Multidisciplinary Optimization*, 30, 181–192.
18. Gilbert, M., & Tyas, A. (2003). Layout optimization of large-scale pin-jointed frames. *Engineering Computations*, 20(8), 1044–1064.
19. Guo, X., Zhang, W., & Zhong, W. (2014). Doing topology optimization explicitly and geometrically: A new moving morphable components based framework. *Journal of Applied Mechanics*, 81(8), 081009.
20. Hoyer, S., Sohl-Dickstein, J., & Greydanus, S. (2019). Neural reparameterization improves structural optimization. *arXiv:1909.04240*.
21. Hrinda, G. A., & Nguyen, D. T. (2008). Optimization of stability constrained geometrically nonlinear shallow trusses using an arc length sparse method with a strain energy density approach. *Finite Elements in Analysis and Design*, 44(15), 933–950.
22. Jia, Y., & Zhang, X. S. (2025). Multimaterial topology optimization for finite strain elastoplasticity: Theory, methods, and applications. *arXiv:2502.02052*.
23. Kirsch, U. (1990). On singular topologies in optimum structural design. *Structural Optimization*, 2, 133–142.
24. Le, C., Norato, J., Bruns, T., Ha, C., & Tortorelli, D. (2010). Stress-based topology optimization for continua. *Structural and Multidisciplinary Optimization*, 41, 605–620.
25. Michaleris, P., Tortorelli, D. A., & Vidal, C. A. (1994). Tangent operators and design sensitivity formulations for transient non-linear coupled problems with applications to elastoplasticity. *International Journal for Numerical Methods in Engineering*, 37(14), 2471–2499.
26. Mitusch, S. K., Funke, S. W., & Dokken, J. S. (2019). dolfin-adjoint 2018.1: Automated adjoints for FEniCS and Firedrake. *Journal of Open Source Software*, 4(38), 1292.
27. Seyranian, A. P., Lund, E., & Olhoff, N. (1994). Multiple eigenvalues in structural optimization problems. *Structural Optimization*, 8, 207–227.
28. Sigmund, O. (2001). A 99 line topology optimization code written in Matlab. *Structural and Multidisciplinary Optimization*, 21, 120–127.
29. Svanberg, K. (1987). The method of moving asymptotes—a new method for structural optimization. *International Journal for Numerical Methods in Engineering*, 24(2), 359–373.
30. Wallin, M., Jönsson, V., & Wingren, E. (2016). Topology optimization based on finite strain plasticity. *Structural and Multidisciplinary Optimization*, 54, 783–793.
31. Wang, F., Lazarov, B. S., Sigmund, O., & Jensen, J. S. (2014). Interpolation scheme for fictitious domain techniques and topology optimization of finite strain elastic problems. *Computer Methods in Applied Mechanics and Engineering*, 276, 453–472.
32. Wang, Q., Hu, R., & Blonigan, P. (2014). Least squares shadowing sensitivity analysis of chaotic limit cycle oscillations. *Journal of Computational Physics*, 267, 210–224.
33. Xue, T., Liao, S., Gan, Z., Park, C., Xie, X., Liu, W. K., & Cao, J. (2023). JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science. *Computer Physics Communications*, 291, 108802.
34. Zegard, T., & Paulino, G. H. (2014). GRAND—Ground structure based topology optimization for arbitrary 2D domains using MATLAB. *Structural and Multidisciplinary Optimization*, 50, 861–882.
