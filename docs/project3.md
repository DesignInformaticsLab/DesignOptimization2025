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

# Project 3: PINN/PINO in the Field

## Introduction

Projects 1 and 2 started from a problem *you* picked. Project 3 starts from someone
else's problem. Your team will find a researcher at ASU whose work runs on partial
differential equations, learn what they are trying to do and what is holding them back,
and then build the **smallest experiment that tells them something useful** about
whether physics-informed neural networks (PINNs) or physics-informed neural operators
(PINOs) can help.

This is how a **forward deployed engineer (FDE)** works. An FDE is an engineer who sits
with a customer, learns their domain well enough to speak their language, works out
where a technology actually helps (and where it does not), and builds a prototype fast
enough to change the customer's next decision. The skills are less about writing the
cleverest code and more about:

- asking questions that surface the real bottleneck,
- turning a messy domain problem into a precise mathematical one,
- shrinking that problem to a **minimal case** that still contains the hard part,
- reporting results honestly, including "this tool is the wrong choice for you."

Everything you need on the technical side is in the
[AI for PDE notes](ai_for_pde.md): the PINN loss, neural operators, the NTK view of
training, and the catalog of failure modes and fixes. The optimization tools from
[Project 2](project2.md) matter too: PINN training is an ill-conditioned optimization
problem, and many PINN failures are optimization failures.

---

## What your team will do

Teams of **1 to 5 students** (same as Projects 1 and 2).

1. **Find a researcher.** Identify a PhD student, postdoc, or faculty member at ASU whose
   work involves PDEs and could plausibly benefit from PINNs or PINOs. Reserve them on
   the shared course participation sheet under column "Project 3 URL" (see
   [the outreach rules](#anchor-outreach-rules)).
2. **Interview them.** Learn the scientific problem, its real-world value, the specific
   computational challenges, and whether they have tried PINN/PINO or similar methods.
   If they have, find out what went wrong.
3. **Build a minimal case study** that does *one* of the following:
   - **Path A: Viability.** Show that PINN/PINO solves a minimal version of their
     problem well enough to matter, and quantify the value against a conventional
     baseline.
   - **Path B: Diagnosis.** Reproduce a limitation the researcher reported (or one you
     expect for their problem), and identify its **root cause** with controlled
     experiments.
   - **Path C: Fix.** Do Path B, then propose and demonstrate a remedy that targets the
     root cause.
4. **Report back.** Write a one-page recommendation memo addressed to your researcher,
   and send it to them.

All three paths can earn full marks. A careful negative result (Path B) is worth as
much as a positive one (Path A). Path C is eligible for bonus points.

### Timeline

| Date | Event |
|------|-------|
| Oct 7 | Researcher identified and reserved on the course participation sheet |
| Oct 12 | Interview completed; presenting team announced |
| Oct 16 | Zoom rehearsal |
| Oct 21 | In-class presentation; final report due |

---

## Is PINN/PINO the right tool? A triage guide

Before you pitch anything, know where these methods tend to help and where they tend to
struggle. A good FDE knows this *before* the meeting.

| Usually a good fit | Usually a poor fit (or a good Path B) |
|---|---|
| **Inverse problems**: infer an unknown coefficient, source, or full field from sparse, noisy measurements (PINNs blend data and physics in one loss) | A **single forward solve** of a well-understood 2-D/3-D PDE. A mature FEM/FV solver is usually faster and more accurate |
| **Many-query problems**: design sweeps, optimization loops, uncertainty quantification, real-time control. A PINO/FNO trained once can be evaluated in milliseconds | **Shocks and discontinuities**: sharp fronts, interfaces, cracks |
| **High-dimensional PDEs**: HJB, Fokker–Planck, Black–Scholes, where grids blow up | **High-frequency or multiscale solutions**: Helmholtz at high wavenumber, thin boundary layers (spectral bias) |
| **Missing physics**: a known PDE with an unknown closure or constitutive term that can be learned from data | **Long-time chaotic or turbulent dynamics**: errors compound; causality violations |
| Smooth solutions on irregular domains where meshing is painful | Strongly **convection- or reaction-dominated** problems (Krishnapriyan et al., 2021) |

PINN vs. PINO in one sentence: a **PINN** solves *one* instance (fixed coefficients,
boundary conditions, and geometry); a **PINO** learns the *map* from problem inputs to
solutions, so it pays off when the researcher needs many solves.

### The break-even rule for surrogates

If a conventional solve costs $T_{\text{sol}}$, a trained surrogate costs
$T_{\text{inf}}$ per query, and training costs $T_{\text{train}}$ (including generating
any training data), the surrogate pays for itself after

$$
N^\star = \frac{T_{\text{train}}}{T_{\text{sol}} - T_{\text{inf}}}
$$

queries. Ask your researcher how many solves they actually need. If they need 10, a
surrogate that takes a week to train is a bad deal. If they need $10^5$, it may be
transformative. Report $N^\star$ in Path A.

---

## Step 1: Find a researcher

### Candidate researchers

The table below is a **starting point, not a complete list**. It was compiled in
September 2026 from public ASU profiles, news releases, and publications. Roles change
and summaries are short, so **read the researcher's own pages and two recent papers
before contacting anyone**. The "minimal-case idea" column is a conversation starter;
your interview should decide the actual case.

**SciML signal** means:
- **Active**: publicly works on scientific machine learning, PINNs, or neural
  surrogates. Expect an expert audience; a good fit for Paths B and C.
- **Adjacent**: uses machine learning or physics-informed ideas near their PDE work.
- **None found**: PDE-heavy work with no public ML-for-PDE work we could find. This
  is fresh territory where your interview may be the first conversation about PINNs.

#### Thermal-fluid sciences and energy (SEMTE, Fulton)

| Researcher | Problem and PDEs | SciML signal | Minimal-case idea |
|---|---|---|---|
| [Beomjin Kwon](https://search.asu.edu/profile/3322893) (faculty) | Boiling heat transfer, additively manufactured heat exchangers, thermoelectrics. Energy equation (advection–diffusion), Navier–Stokes, two-phase flow | **Active.** NSF CAREER (2024) on [inferring temperature fields in boiling fluids](https://news.asu.edu/20240412-science-and-technology-mapping-new-field); co-authored a review of ML in heat transfer that covers PINNs | Reconstruct a 2-D temperature field from sparse sensor points given a known velocity field (an inverse PINN) |
| [Yulia Peet](https://forge.engineering.asu.edu/faculty_mentor/yulia-peet/) (faculty) | Turbulence, high-order spectral-element methods, wind energy, biological fluid mechanics. Incompressible Navier–Stokes | **None found** | Flow past a cylinder or a channel flow at increasing Reynolds number. Where does the PINN break? |
| [Marcus Herrmann](https://search.asu.edu/profile/1096980) (faculty) | Atomization and turbulent multiphase flow, interface-capturing numerics. Navier–Stokes with moving interfaces, level-set equations | **None found** | 1-D/2-D level-set advection of a sharp interface (tests PINNs on near-discontinuities) |
| [Jeonglae Kim](https://search.asu.edu/profile/3167494) (faculty) | Large-eddy simulation of high-speed turbulent flows, aeroacoustics, flow control and optimization. Compressible Navier–Stokes | **Adjacent.** Data-driven modeling of low-frequency turbulent dynamics (APS DFD 2022) | 1-D viscous Burgers with decreasing viscosity (shock formation); or PDE-constrained flow control |
| [Mohamed Kasbaoui](https://search.asu.edu/profile/3323106) (faculty) | Particle-laden turbulence, immersed-boundary methods, massively parallel CFD (NSF CAREER 2025). Navier–Stokes coupled to Lagrangian particles | **None found** | Advection–diffusion of a particle concentration field in a prescribed vortex flow |
| [Huei-Ping Huang](https://search.asu.edu/profile/1272055) (faculty) | Geophysical fluid dynamics, atmospheric and ocean simulation, urban climate. Shallow-water and primitive equations | **None found** | 1-D/2-D shallow-water equations (connects to the weather-forecasting example in the notes) |
| [Konrad Rykaczewski](https://news.asu.edu/20230525-solutions-meet-andi-worlds-first-outdoor-sweating-breathing-and-walking-manikin) (faculty) | Human heat stress, thermal manikin "ANDI," personalized thermoregulation. Bioheat (Pennes) equation, heat and mass transfer with sweating | **None found** | 1-D layered-tissue Pennes equation; infer blood perfusion from surface heat-flux data |
| [Spring Berman](https://search.asu.edu/profile/1943720) (faculty) | Swarm robotics; controlling a robot swarm through its mean-field density. Advection–diffusion PDE with the control in the advection term | **None found** (PDE-based control, not ML) | PINO mapping a velocity field to the steady swarm density, for fast density control |
| [Yongming Liu](https://labs.engineering.asu.edu/paralab/person/yongming-liu) (faculty) | Fatigue, fracture, probabilistic computational mechanics, diagnostics and prognostics. Elasticity, elastic-wave propagation (structural health monitoring), crack growth | **Adjacent.** Physics-based prognostics and uncertainty quantification; check recent papers for physics-informed ML | 1-D elastic wave in a bar with a stiffness defect; infer the defect location from sensor signals |

#### Materials and manufacturing (SEMTE and Polytechnic, Fulton)

| Researcher | Problem and PDEs | SciML signal | Minimal-case idea |
|---|---|---|---|
| [Yang Jiao](https://search.asu.edu/profile/1970397) (faculty) | Microstructure of heterogeneous materials, microstructure evolution. Phase-field (Allen–Cahn, Cahn–Hilliard) | **Active.** Trained recurrent neural networks to emulate PDE-based microstructure evolution (*Patterns*, 2021) | 1-D/2-D Allen–Cahn, the same failure-and-fix example from the notes (causal training) |
| [Jay Oswald](https://search.asu.edu/profile/1759035) (faculty) | Fracture, plasticity, laser welding and materials processing; combines physics-based simulation with data-driven methods on industry projects. Heat conduction with moving sources, elastoplasticity | **Adjacent.** Physics plus data-driven modeling for industry | Moving point heat source in a plate (Rosenthal-type welding problem); PINO over laser power and speed |
| [Houlong Zhuang](https://search.asu.edu/profile/3152899) (faculty) | Quantum-mechanical materials simulation, machine learning, quantum computing. Schrödinger / Kohn–Sham eigenproblems | **Adjacent.** ML for materials | 1-D Schrödinger eigenproblem (PINN for eigenvalues and eigenfunctions) |
| [Dhruv Bhate](https://news.engineering.asu.edu/asu_person/dhruv-bhate/) (faculty) | Metal additive manufacturing, lattice and cellular materials, bio-inspired design. Linear elasticity, transient heat conduction | **None found** | Thermal history of a layer-by-layer deposited wall (1-D/2-D transient conduction with a growing domain) |

#### Computing, electrical engineering, and power (SCAI and ECEE, Fulton)

| Researcher | Problem and PDEs | SciML signal | Minimal-case idea |
|---|---|---|---|
| [Kookjin Lee](https://search.asu.edu/profile/3957975) (faculty) | Scientific machine learning for dynamical systems and spatiotemporal processes; NSF CAREER (2024); [Applied Materials project on plasma-chamber physics for chip manufacturing](https://news.asu.edu/20250428-science-and-technology-applying-ai-microelectronics-manufacturing) | **Active.** Core SciML researcher | 1-D drift–diffusion (plasma or semiconductor) model: stiff, multiscale, a natural Path B |
| [Ying-Cheng Lai](https://news.asu.edu/20240726-science-and-technology-human-brains-teach-ai-new-skills) (faculty) | Machine learning for nonlinear and chaotic dynamical systems, reservoir computing | **Active.** ML for dynamics (reservoir computing rather than PINNs) | Kuramoto–Sivashinsky equation: PINN vs. reservoir computing for short-term prediction |
| [Yang Weng](https://search.asu.edu/profile/917711) (faculty) | Power-system monitoring and control; "assured" machine learning that pairs AI models with physics-guided models | **Active.** Physics-guided ML (mostly algebraic and ODE models, not PDEs) | Generator swing equation (an ODE) with a PINN, or a PDE model of transmission-line transients |
| [Georgios Trichopoulos](https://faculty.engineering.asu.edu/trichopoulos/person/george-trichopoulos) (faculty) | Millimeter-wave and terahertz imaging, antennas, reconfigurable metasurfaces. Maxwell's equations, Helmholtz equation | **None found** | 1-D/2-D Helmholtz with increasing wavenumber, a textbook spectral-bias failure (Path B/C) |

#### Water and environment (SSEBE, Fulton; SHaDE Lab)

| Researcher | Problem and PDEs | SciML signal | Minimal-case idea |
|---|---|---|---|
| [Enrique Vivoni](https://search.asu.edu/profile/1346273) (faculty) | Watershed hydrology in arid regions; distributed hydrologic model tRIBS. Richards equation (soil water), kinematic-wave / shallow-water routing | **Adjacent.** Deep-learning surrogate of a hydrologic model for the Upper Colorado River | 1-D Richards infiltration with a sharp wetting front (a known hard case) |
| [Ariane Middel](https://shadelab.asu.edu/projects/) (faculty, SHaDE Lab) | Urban heat and outdoor thermal comfort. Radiative and convective heat transfer | **Active.** Published a multimodal PINN for mean radiant temperature (ICCV Workshops, 2025) | Ask what limited their PINN; reproduce it (Path B) |

#### Mathematics and Earth sciences (outside Fulton, but excellent PDE partners)

| Researcher | Problem and PDEs | SciML signal | Minimal-case idea |
|---|---|---|---|
| [Jimmie Adriazola](https://search.asu.edu/profile/jadriazo) (postdoc → asst. prof., SMSS) | Optimal control of dispersive waves (optics, fluids, quantum materials), adjoint methods, scientific ML | **Active.** SciML and optimal control | PDE-constrained optimal control of the 1-D heat equation: PINN vs. adjoint method (ties directly to this course) |
| Guangting Yu (PhD student, SMSS) | [Presented on PINNs, Deep Ritz, and ensemble Kalman inversion](https://math.asu.edu/node/8907) (Jan. 2024); may have graduated, so check | **Active** | Deep Ritz vs. PINN on a 2-D Poisson problem |
| [Rodrigo Platte](https://search.asu.edu/profile/913016) (faculty, SMSS) | Numerical analysis, approximation theory, PDEs on complex domains | **None found** (a classical-numerics expert, ideal for a skeptical baseline) | PINN vs. radial-basis-function collocation on an irregular 2-D domain |
| [Malena Español](https://search.asu.edu/profile/3488212) (faculty, SMSS) | Inverse problems, regularization, imaging, materials science; numerical analysis meets ML | **Adjacent** | Inverse source problem: PINN vs. Tikhonov regularization |
| Yang Kuang (faculty, SMSS) | Mathematical oncology; reaction–diffusion models of tumor growth (for example, glioblastoma with density-dependent diffusion) | **None found** | 1-D Fisher–KPP: infer diffusivity and growth rate from two tumor snapshots (inverse PINN) |
| [Wenbo Tang](https://search.asu.edu/profile/wtang17) (faculty, SMSS) | Mixing and transport in geophysical flows, Lagrangian coherent structures. Advection–diffusion–reaction | **None found** | Scalar advection–diffusion in a chaotic double-gyre flow |
| [Mingming Li](https://search.asu.edu/profile/mingming) (faculty, SESE) | Mantle convection and planetary interiors. Stokes flow coupled with heat transport at high Rayleigh number | **None found** | 2-D Rayleigh–Bénard convection with increasing Rayleigh number |

**Beyond this list.** Plenty of PDE-heavy researchers are not listed here, including
in biomedical engineering, civil and structural engineering, chemical engineering,
astrophysics, and battery research. If you find one, reserve them the same way. The
best candidates are often **PhD students and postdocs** in these labs: they do the
day-to-day computing, know exactly where the solver hurts, and usually have more time
than their advisors.

(anchor-outreach-rules)=
### Outreach rules

- **One team per researcher.** Reserve your researcher on the shared course
  participation sheet under column "Project 3 URL" *before* you email them. If your
  first choice is taken, pick another. Faculty are busy, and five emails about the same
  class project from five teams will hurt everyone's chances.
- **No mass emails.** Write to one person at a time, by name, about their work.
- **Follow up once.** The schedule is tight. If there is no reply after 2 business days,
  send one short, polite follow-up. If there is still no reply after another 2 days,
  release the reservation and move to your backup. Have a backup researcher in mind
  from the start.
- **Respect their time.** Ask for 30 minutes, offer Zoom or a visit to their lab, and
  end on time.

### How to find and reach people

- **ASU Search** ([search.asu.edu](https://search.asu.edu)) profiles list research
  areas, publications, and email addresses. Lab websites list current students and
  postdocs.
- **Google Scholar**: search for a PDE topic plus "Arizona State University" and sort
  by date.
- **Seminars**: SEMTE seminars, SMSS applied-math and computational-math seminars,
  SCAI seminars, and the LIONS seminar. Introducing yourself after a talk is the easiest
  cold contact there is.
- **Your own network**: your research advisor, TA, labmates, and classmates in other
  departments.
- **Snowball**: end every conversation with "Who else should I talk to?"

### Email template

Keep it short and specific. Change it so it is clearly about *their* work.

> **Subject:** 30-minute chat about [their topic] for an ASU design-optimization project
>
> Dear Dr. [Name] / Dear [First name],
>
> I'm a [year/program] student in MAE 598 Design Optimization. Our team project asks us
> to learn how researchers at ASU use PDE models and whether newer methods such as
> physics-informed neural networks could help with their computational challenges.
>
> I read your [paper/news story] on [specific topic], and I'm curious about [one
> specific question, e.g., "how long a single simulation of X takes and how many you
> need"].
>
> Would you (or a student in your group) have 30 minutes this week for a
> short conversation, on Zoom or in person? We'll share our final results with you, and
> we will not publish anything from our conversation without your permission.
>
> Thank you,
> [Name, team members, course]

---

## Step 2: The interview

### Before the meeting

- Read their profile, their lab page, and **two recent papers** (abstract, figures,
  conclusions).
- Write a **one-paragraph summary** of what you think their problem is. Your first
  question can be "Here's my understanding; what did I get wrong?" That shows you did
  the homework, and their corrections are the most valuable part of the interview.
- Ask whether you may **take notes or record**, and what you may **share publicly**.

### Interview guide

You won't get through all of these. Prioritize the bold ones.

**The science and its value**
- **What question are you trying to answer, and who cares about the answer?** (an
  industry, a patient population, a policy decision, a scientific debate)
- What would change in the world if this problem were solved 10× faster or more
  accurately?

**The PDE model**
- **What equations do you solve?** On what domain, with what boundary and initial
  conditions? What parameters are uncertain or unknown?
- Is any part of the physics unknown or approximated (closures, constitutive laws,
  source terms)?
- What measurement data do you have? How sparse and how noisy?

**The computational pain**
- **What solver do you use, and how long does one run take?** On what hardware?
- **How many runs do you need** (for a design study, calibration, uncertainty
  quantification, or control)? What do you give up because runs are expensive?
- What makes the problem hard: geometry, meshing, multiple scales, stiffness, high
  dimension, sharp fronts, or missing data?

**Experience with ML / PINN / PINO**
- **Have you or your group tried PINNs, neural operators, or other ML surrogates?**
- If yes: **what happened? Where did it fail, and what do you think the cause was?**
  (This is the raw material for Path B.)
- If no: why not? (Skepticism, unfamiliarity, and "our problem is too hard" are all
  useful answers.)

**Success and next steps**
- **What accuracy would actually be useful?** (A 5% error may be fine for screening and
  useless for certification.)
- What is the smallest result that would make you want to look further?
- Who else should we talk to?

### After the meeting

Within a week, send them a **problem brief** (half a page to one page) summarizing what
you learned, and ask them to correct mistakes. This is standard FDE practice: it catches
misunderstandings early and shows the researcher you were listening.

### Confidentiality and consent

- Unpublished data, results, and code belong to the researcher. Use them only with
  permission, and **keep them out of your public repository** unless they say otherwise.
- Industry-sponsored work may be under a nondisclosure agreement. In that case, build
  your minimal case on **public or synthetic data** that captures the same difficulty.
- Ask before quoting or naming the researcher in your public report. Offer to anonymize.
- Do not record without explicit consent.

---

## Step 3: The minimal case study

### What "minimal" means

Your case should be the **smallest problem that still contains the difficulty the
researcher cares about**. Usually this means going down in dimension (3-D → 1-D or
2-D), simplifying geometry, and replacing real data with a manufactured or synthetic
solution, while **keeping** the feature that makes the real problem hard (a sharp front,
a high Péclet number, a stiff reaction, an unknown coefficient, a large parameter
range). If a solver reaches the same accuracy on the simplified problem as on a trivial
one, you simplified away the point.

Every case study must have a **difficulty knob**: a parameter that makes the problem
harder as it grows (convection speed $\beta$, reaction rate $\rho$, wavenumber $k$,
Reynolds or Péclet number, time horizon, sensor sparsity, number of parameters in a
PINO family). This plays the same role the condition-number knob played in Project 2.

(anchor-pinn-diagnostic-kit)=
### PINN/PINO diagnostic kit (required)

Use these measurements so results are comparable across teams:

- **P1 – Baseline.** Solve the same minimal case with a conventional method (finite
  differences, finite elements, spectral, or `scipy.integrate.solve_ivp` with the
  method of lines). Report its accuracy and wall-clock time. This is also your
  reference solution.
- **P2 – Loss components.** Log each loss term (PDE residual, boundary, initial, data)
  *separately* over training, on a log scale. Imbalanced loss terms are the first
  symptom of NTK-type pathologies (see the notes).
- **P3 – Error vs. knob.** Relative $L^2$ error against the reference, plotted against
  the difficulty knob. Where does accuracy collapse?
- **P4 – Where is the error?** Plot the pointwise error field in space–time, and the
  Fourier spectrum of the error. Error concentrated at late times suggests a causality
  problem; error concentrated at high frequencies suggests spectral bias.
- **P5 – Cost.** Training time, inference time, and hardware. For PINOs and other
  surrogates, report the break-even number of queries $N^\star$.

### Path-specific requirements

**Path A: Viability.** Show that the PINN/PINO reaches the accuracy your researcher
called useful, across a meaningful range of the knob. Then make an honest value
argument using P1 and P5: a wall-clock speedup that includes training cost, a
favorable $N^\star$, or a capability the baseline lacks (for example, inferring an
unknown coefficient from sparse data). "It works on an easy case" is not a value
argument.

**Path B: Diagnosis.** Reproduce the failure in your minimal case, show how it worsens
along the knob (P3), and identify the root cause with **at least one controlled
experiment that separates two candidate explanations**. Classic examples:

- *Expressivity vs. optimization.* Fit the network directly to the reference solution
  by supervised regression. If that works, the network *can* represent the answer and
  the failure is in the physics-informed loss landscape (the Krishnapriyan et al. test).
- *Loss imbalance vs. spectral bias.* Rebalance the loss terms (fixed weights or NTK
  weights). If the error drops, imbalance was the culprit. If high-frequency error
  remains, suspect spectral bias.
- *Causality.* Train on short time windows (time-marching). If late-time error
  disappears, the full-domain PINN was violating causality.
- *Sampling.* Increase or adapt collocation points near the sharp feature. If the error
  drops, resolution, not optimization, was the bottleneck.

**Path C: Fix.** Everything in Path B, plus a remedy that targets *your* diagnosed cause
(for example: NTK-based loss weighting, curriculum regularization, causal training,
Fourier features, adaptive sampling, or a better optimizer from the gradient-descent
notes). Explain *why* it addresses the mechanism, and show before/after results with
P2–P5.

### Starter code

The following minimal PINN solves the 1-D convection equation from the notes,

$$
u_t + \beta u_x = 0,\quad x\in[0,2\pi],\ t\in[0,1],\quad u(x,0)=\sin x,
\quad u(0,t)=u(2\pi,t),
$$

whose exact solution is $u=\sin(x-\beta t)$. With $\beta=1$ it should succeed; with
$\beta=30$ it should reproduce the failure from Krishnapriyan et al. (2021). Use it to
check your setup, then replace the PDE with your own. It needs PyTorch; run it in Google
Colab or locally (a CPU is enough).

```python
import math
import torch

torch.manual_seed(0)
beta = 30.0  # difficulty knob: try 1, 10, 30

net = torch.nn.Sequential(
    torch.nn.Linear(2, 50), torch.nn.Tanh(),
    torch.nn.Linear(50, 50), torch.nn.Tanh(),
    torch.nn.Linear(50, 50), torch.nn.Tanh(),
    torch.nn.Linear(50, 1),
)

def u(x, t):
    return net(torch.cat([x, t], dim=1))

# Collocation (interior), initial-condition, and boundary points
x_r = (2 * math.pi * torch.rand(2000, 1)).requires_grad_()
t_r = torch.rand(2000, 1).requires_grad_()
x_0 = 2 * math.pi * torch.rand(256, 1)
t_b = torch.rand(256, 1)

def losses():
    u_r = u(x_r, t_r)
    u_t, u_x = torch.autograd.grad(u_r.sum(), (t_r, x_r), create_graph=True)
    L_pde = ((u_t + beta * u_x) ** 2).mean()
    L_ic = ((u(x_0, torch.zeros_like(x_0)) - torch.sin(x_0)) ** 2).mean()
    L_bc = ((u(torch.zeros_like(t_b), t_b)
             - u(2 * math.pi * torch.ones_like(t_b), t_b)) ** 2).mean()
    return L_pde, L_ic, L_bc

opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for it in range(10001):
    opt.zero_grad()
    L_pde, L_ic, L_bc = losses()
    (L_pde + L_ic + L_bc).backward()
    opt.step()
    if it % 1000 == 0:  # P2: log each loss term separately
        print(f"{it:6d}  pde={L_pde.item():.2e}  ic={L_ic.item():.2e}  bc={L_bc.item():.2e}")

# P3: relative L2 error against the exact solution
X, T = torch.meshgrid(torch.linspace(0, 2 * math.pi, 256),
                      torch.linspace(0, 1, 100), indexing="ij")
with torch.no_grad():
    U = u(X.reshape(-1, 1), T.reshape(-1, 1)).reshape(X.shape)
U_exact = torch.sin(X - beta * T)
print("relative L2 error:", (torch.norm(U - U_exact) / torch.norm(U_exact)).item())
```

On a laptop CPU this takes a few minutes. Expect a relative error of about **0.6%** at
$\beta=1$ and about **74%** at $\beta=30$. Notice that at $\beta=30$ every loss term
still drops to around $10^{-3}$ or below while the solution is badly wrong. **A small
training loss does not mean a correct solution**, which is why P1 (a reference solution)
is required.

### Useful tools

- **PINN libraries**: [DeepXDE](https://github.com/lululxvi/deepxde) (beginner-friendly,
  many examples), [jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi) (causal
  training and NTK weighting from the papers in the notes), NVIDIA PhysicsNeMo
  (formerly Modulus).
- **Neural operators**: [neuraloperator](https://github.com/neuraloperator/neuraloperator)
  (FNO and related models).
- **Conventional baselines**: `scipy` (method of lines with `solve_ivp`),
  [scikit-fem](https://github.com/kinnala/scikit-fem), [FEniCSx](https://fenicsproject.org),
  [py-pde](https://github.com/zwicker-group/py-pde).
- **Benchmarks** to compare against: PDEBench and PINNacle.
- **Compute**: Google Colab is enough for most minimal cases. ASU Research Computing
  (the Sol supercomputer) is available if you need a GPU for longer runs.
- **Coding agents** are encouraged, as in Project 2. Verify the reference solution and
  at least one PINN error value by hand before trusting any plot.

---

## Report requirements

Your report is a **single Markdown file or notebook** in your team's **public** GitHub
repository, with runnable code. (Keep anything the researcher asked you not to share
out of it.) It must contain:

### 1. Outreach log
Who you contacted, how you found them, and the outcome (including declines and
non-responses, which are normal and not penalized).

### 2. Problem brief
The researcher's scientific problem, its real-world value, the governing PDEs, their
current solver and its cost, the specific challenges, and any prior PINN/PINO attempts
and the limitations they saw. Note whether the researcher reviewed the brief.

### 3. Minimal case formulation
The PDE, domain, boundary and initial conditions, parameters (nondimensionalized where
sensible), and the difficulty knob. Explain **what you simplified and why the
simplification keeps the hard part**. State your path (A, B, or C).

### 4. Experiments
The [diagnostic kit](#anchor-pinn-diagnostic-kit) (P1–P5) plus your path-specific
requirements.

### 5. Recommendation memo
A one-page memo addressed to your researcher: what you found, how confident you are,
whether you would recommend they invest further in PINN/PINO for this problem, and what
the next experiment should be. Write it for *them*, not for the grader. Send it to the
researcher, and include their response if they give one.

### 6. Assumptions and limitations
What your minimal case does not capture, and how that could change the conclusion.

---

## Rubric (100 points + up to 10 bonus)

| Category | Points | Full marks |
|---|---|---|
| Outreach and interview | 10 | A real researcher interviewed; well-prepared questions; outreach rules followed |
| Problem brief | 15 | Accurate, specific, and clear to a non-expert; real-world value and the computational bottleneck are concrete; researcher-checked |
| Minimal case formulation | 15 | Complete PDE statement; well-chosen knob; simplification justified so the hard part is kept |
| Baseline and diagnostics (P1–P5) | 20 | Credible conventional baseline; all five measurements reported clearly |
| Path findings | 25 | **A**: accuracy and an honest value argument (including training cost / $N^\star$). **B**: failure reproduced along the knob, plus a controlled experiment that pins down the root cause. **C**: as B. |
| Recommendation memo | 5 | Actionable, honest, written for the researcher |
| Reproducibility | 5 | Runnable code, fixed seeds, instructions |
| Presentation and clarity | 5 | Organized; math renders on GitHub |
| **Bonus (Path C)** | +10 | A remedy that targets the diagnosed cause, with convincing before/after evidence |

---

## Submission

1. Create a **public** GitHub repository for your team.
2. Write your report as a single Markdown file or notebook, with runnable code.
3. Submit the repository link on **Canvas**.

### Tips

- **Start outreach immediately.** Scheduling is the slowest step, and the interview
  must happen by Oct 12.
- **Listen more than you pitch.** Spend the first two-thirds of the interview
  understanding their problem. Your goal is to find out whether PINNs help, not to sell
  them.
- **Do not overclaim.** A speedup that ignores training time, or an accuracy number on
  an easier problem than the one they care about, will cost you credibility with the
  researcher and points on the rubric.
- **Get the baseline right first.** Many apparent PINN successes vanish against a
  properly configured conventional solver (McGreivy & Hakim, 2024).
- **Close the loop.** Send the memo and thank them. Researchers who get useful results
  back are the ones who say yes to next year's students.

---

## References

1. Krishnapriyan, A. S., Gholami, A., Zhe, S., Kirby, R. M., & Mahoney, M. W. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS 2021*.

2. Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. *Journal of Computational Physics*, 449, 110768.

3. Wang, S., Sankaran, S., & Perdikaris, P. (2024). Respecting causality for training physics-informed neural networks. *Computer Methods in Applied Mechanics and Engineering*, 421, 116813.

4. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.

5. Li, Z., Kovachki, N., Azizzadenesheli, K., et al. (2021). Fourier neural operator for parametric partial differential equations. *ICLR 2021*.

6. Li, Z., et al. (2024). Physics-informed neural operator for learning partial differential equations. *ACM/JMS Journal of Data Science*, 1(3).

7. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208–228.

8. McGreivy, N., & Hakim, A. (2024). Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations. *Nature Machine Intelligence*, 6, 1256–1269.
