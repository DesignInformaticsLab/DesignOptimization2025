"""Generate trajectory-animation GIFs for the gradient descent slide decks.

Extracts core functions from Demo/Gradient_Optimization_Walkthrough.ipynb and
saves 12 GIF files to docs/_static/gd_gifs/.

Usage:
    python scripts/export_gd_gifs.py
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass, field
from typing import Optional, List
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "_static", "gd_gifs")
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 5
DPI = 90
MAX_FRAMES = 60

# ---------------------------------------------------------------------------
# Shared utilities (from notebook Module 0)
# ---------------------------------------------------------------------------

def norm(x):
    return float(np.linalg.norm(x))

def spd_from_cond(n, cond, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.exp(np.linspace(0, np.log(cond), n))
    A = (Q * lam) @ Q.T
    A = 0.5 * (A + A.T) + 1e-12 * np.eye(n)
    return A

@dataclass
class RunLog:
    xs: List[np.ndarray] = field(default_factory=list)
    fs: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)

def record(log, x, fval, gnorm):
    log.xs.append(x.copy())
    log.fs.append(float(fval))
    log.grad_norms.append(float(gnorm))

@dataclass
class QuadProblem:
    A: np.ndarray
    b: np.ndarray
    x_star: np.ndarray
    f_star: float
    def f(self, x): return 0.5 * x @ (self.A @ x) - self.b @ x
    def g(self, x): return self.A @ x - self.b
    def H(self, x): return self.A

def make_quadratic(n=2, cond=1e3, seed=42):
    rng = np.random.default_rng(seed)
    A = spd_from_cond(n, cond, rng)
    x_star = rng.normal(size=n)
    b = A @ x_star
    prob = QuadProblem(A=A, b=b, x_star=x_star,
                       f_star=0.5 * x_star @ (A @ x_star) - b @ x_star)
    x0 = rng.normal(size=n) * 3.0
    return prob, x0

# Rosenbrock
def rosenbrock_f(xy):
    x, y = xy[0], xy[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_g(xy):
    x, y = xy[0], xy[1]
    return np.array([-2*(1-x) - 400*x*(y-x**2), 200*(y-x**2)], dtype=float)

def rosenbrock_H(xy):
    x, y = xy[0], xy[1]
    return np.array([[2 - 400*(y - 3*x**2), -400*x],
                     [-400*x, 200]], dtype=float)

@dataclass
class RosenbrockProblem:
    x_star: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    f_star: float = 0.0
    def f(self, x): return rosenbrock_f(x)
    def g(self, x): return rosenbrock_g(x)
    def H(self, x): return rosenbrock_H(x)

def make_rosenbrock(seed=0):
    rng = np.random.default_rng(seed)
    x0 = np.array([-1.2, 1.0]) + 0.1 * rng.normal(size=2)
    return RosenbrockProblem(), x0

# Line search
def armijo_backtracking(f, g, x, p, alpha0=1.0, c=1e-4, beta=0.5, max_bt=60):
    gx = g(x)
    slope = float(gx @ p)
    if not np.isfinite(slope) or slope >= 0:
        return 0.0
    fx = f(x)
    alpha = float(alpha0)
    for _ in range(max_bt):
        x_trial = x + alpha * p
        if np.all(np.isfinite(x_trial)):
            ft = f(x_trial)
            if np.isfinite(ft) and ft <= fx + c * alpha * slope:
                return alpha
        alpha *= beta
    return 0.0

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def gd_fixed(prob, x0, step, maxit=500, tol=1e-8, clip=1e8):
    x = x0.copy(); log = RunLog()
    for _ in range(maxit):
        fx = prob.f(x); gx = prob.g(x)
        record(log, x, fx, norm(gx))
        if norm(gx) < tol: break
        x_next = x - step * gx
        if not np.all(np.isfinite(x_next)) or np.linalg.norm(x_next) > clip: break
        x = x_next
    return log

def gd_armijo(prob, x0, maxit=500, tol=1e-8, armijo_beta=0.5):
    x = x0.copy(); log = RunLog()
    for _ in range(maxit):
        fx = prob.f(x); gx = prob.g(x)
        record(log, x, fx, norm(gx))
        if norm(gx) < tol: break
        p = -gx
        alpha = armijo_backtracking(prob.f, prob.g, x, p, beta=armijo_beta)
        if alpha == 0.0: break
        x = x + alpha * p
    return log

def steepest_descent_quad(prob, x0, M_kind="I", maxit=500, tol=1e-8):
    A = prob.A; x = x0.copy(); log = RunLog()
    for _ in range(maxit):
        fx = prob.f(x); gx = prob.g(x)
        record(log, x, fx, norm(gx))
        if norm(gx) < tol: break
        if M_kind == "I":
            p = -gx
        elif M_kind == "A":
            p = -np.linalg.solve(A, gx)
        else:
            raise ValueError
        denom = p @ (A @ p)
        if denom <= 0: break
        alpha = -(gx @ p) / denom
        x = x + alpha * p
        if M_kind == "A": break
    record(log, x, prob.f(x), norm(prob.g(x)))
    return log

def gd_exact_ls_quad(prob, x0, maxit=100, tol=1e-14):
    A = prob.A; x = x0.copy(); log = RunLog()
    for _ in range(maxit):
        fx = prob.f(x); gx = prob.g(x)
        record(log, x, fx, norm(gx))
        if norm(gx) < tol: break
        p = -gx
        denom = p @ (A @ p)
        if denom <= 0: break
        alpha = (gx @ gx) / denom
        x = x + alpha * p
    record(log, x, prob.f(x), norm(prob.g(x)))
    return log

def linear_cg_2d(A, b, x0, tol=1e-14, maxit=2):
    x = x0.copy(); r = b - A @ x; p = r.copy()
    xs = [x.copy()]
    for _ in range(maxit):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        xs.append(x.copy())
        if np.linalg.norm(r_new) < tol: break
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
    return xs

def newton_method(prob, x0, variant="damped", maxit=50, tol=1e-8):
    x = x0.copy(); log = RunLog()
    for _ in range(maxit):
        fx = prob.f(x); gx = prob.g(x)
        record(log, x, fx, norm(gx))
        if not np.isfinite(fx) or norm(gx) < tol: break
        Hx = prob.H(x)
        try:
            p = -np.linalg.solve(Hx, gx)
        except np.linalg.LinAlgError:
            p = -gx
        if variant == "pure":
            alpha = 1.0
        else:
            alpha = armijo_backtracking(prob.f, prob.g, x, p)
            if alpha == 0.0: break
        x_next = x + alpha * p
        if not np.all(np.isfinite(x_next)): break
        x = x_next
    record(log, x, prob.f(x), norm(prob.g(x)))
    return log

def cauchy_point(g, H, Delta):
    gnorm = norm(g)
    if gnorm == 0: return np.zeros_like(g)
    Hgg = float(g @ (H @ g))
    if Hgg <= 0:
        tau = 1.0
    else:
        tau = min(gnorm**3 / (Delta * Hgg), 1.0)
    return -(tau * Delta / gnorm) * g

def dogleg_step(g, H, Delta):
    try:
        pN = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        return cauchy_point(g, H, Delta)
    gnorm = norm(g)
    if gnorm == 0: return np.zeros_like(g)
    alpha_sd = (g @ g) / max(g @ (H @ g), 1e-16)
    pU = -alpha_sd * g
    if np.linalg.norm(pN) <= Delta: return pN
    if np.linalg.norm(pU) >= Delta: return -(Delta / gnorm) * g
    d = pN - pU
    a = d @ d; b = 2 * (pU @ d); c = pU @ pU - Delta**2
    disc = b*b - 4*a*c
    tau = (-b + math.sqrt(max(0, disc))) / (2*a) if a > 0 else 0
    return pU + tau * d

def trust_region(prob, x0, method="dogleg", Delta0=1.0, eta=0.1, maxit=200, tol=1e-8):
    x = x0.copy(); log = RunLog(); Delta = float(Delta0)
    for _ in range(maxit):
        fx = prob.f(x); gx = prob.g(x); Hx = prob.H(x)
        record(log, x, fx, norm(gx))
        if not np.isfinite(fx) or norm(gx) < tol: break
        if method == "cauchy":
            p = cauchy_point(gx, Hx, Delta)
        elif method == "dogleg":
            p = dogleg_step(gx, Hx, Delta)
        else:
            raise ValueError
        pred = -float(gx @ p) - 0.5 * float(p @ (Hx @ p))
        if pred <= 0:
            Delta = max(1e-12, 0.25 * Delta); continue
        f_new = prob.f(x + p)
        ared = float(fx - f_new)
        rho = ared / pred
        if rho < 0.25:
            Delta = max(1e-12, 0.25 * Delta)
        elif rho > 0.75 and abs(norm(p) - Delta) < 1e-12:
            Delta = min(1e3, 2.0 * Delta)
        if rho > eta and np.isfinite(f_new):
            x = x + p
    record(log, x, prob.f(x), norm(prob.g(x)))
    return log

def bfgs(prob, x0, maxit=300, tol=1e-8):
    x = x0.copy(); n = x.size; B = np.eye(n); log = RunLog()
    for _ in range(maxit):
        fx = prob.f(x); gx = prob.g(x)
        record(log, x, fx, norm(gx))
        if not np.isfinite(fx) or norm(gx) < tol: break
        try:
            p = -np.linalg.solve(B, gx)
        except np.linalg.LinAlgError:
            p = -gx
        alpha = armijo_backtracking(prob.f, prob.g, x, p)
        if alpha == 0.0: break
        x_new = x + alpha * p
        s = x_new - x; y = prob.g(x_new) - gx
        Bs = B @ s; sty = float(s @ y); sBs = float(s @ Bs)
        if sty < 0.2 * sBs:
            theta = (0.8 * sBs) / max(1e-16, sBs - sty)
            y = theta * y + (1 - theta) * Bs
            sty = float(s @ y)
        if sty > 1e-16:
            B = B - np.outer(Bs, Bs) / max(1e-16, sBs) + np.outer(y, y) / sty
        x = x_new
    record(log, x, prob.f(x), norm(prob.g(x)))
    return log

# ---------------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------------

def _compute_bounds(traj, x_star=None, pad=0.25):
    xs_all = [traj]
    if x_star is not None and np.all(np.isfinite(x_star)):
        xs_all.append(x_star.reshape(1, 2))
    X = np.vstack(xs_all)
    minx, maxx = float(X[:,0].min()), float(X[:,0].max())
    miny, maxy = float(X[:,1].min()), float(X[:,1].max())
    dx = max(1e-6, maxx - minx); dy = max(1e-6, maxy - miny)
    return (minx - pad*dx, maxx + pad*dx), (miny - pad*dy, maxy + pad*dy)

def save_trajectory_gif(f_eval, xs, x_star, title, filename,
                        xlim=None, ylim=None, levels=30, wide=False):
    traj = np.asarray(xs, dtype=float)
    if traj.shape[0] > MAX_FRAMES:
        idx = np.linspace(0, traj.shape[0]-1, MAX_FRAMES, dtype=int)
        traj = traj[idx]
    if xlim is None or ylim is None:
        xlim, ylim = _compute_bounds(traj, x_star)

    grid_x = np.linspace(xlim[0], xlim[1], 200)
    grid_y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f_eval(np.array([X[i,j], Y[i,j]]))

    if wide:
        Z = np.log10(np.maximum(Z, 1e-16))
        levels = np.linspace(Z.min(), Z.max(), levels)

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=7)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")

    ax.plot(traj[0, 0], traj[0, 1], marker="^", ms=8, color="C1", zorder=5,
            label=r"$x_0$")
    if x_star is not None and np.all(np.isfinite(x_star)):
        ax.plot(x_star[0], x_star[1], marker="*", ms=10, color="gold",
                markeredgecolor="k", zorder=5, label=r"$x^\star$")
    ax.legend(loc="best", fontsize=8)

    point, = ax.plot([], [], "o", color="C3", ms=5, zorder=6)
    path,  = ax.plot([], [], "-", color="C3", lw=1.2, zorder=4)

    def init():
        point.set_data([], []); path.set_data([], [])
        return point, path

    def update(frame):
        i = int(frame)
        point.set_data([traj[i, 0]], [traj[i, 1]])
        path.set_data(traj[:i+1, 0], traj[:i+1, 1])
        return point, path

    ani = FuncAnimation(fig, update, frames=len(traj),
                        init_func=init, interval=200, blit=True)
    fpath = os.path.join(OUT_DIR, filename)
    ani.save(fpath, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    fsize = os.path.getsize(fpath) / 1024
    print(f"  Saved {filename} ({len(traj)} frames, {fsize:.0f} KB)")

# ---------------------------------------------------------------------------
# Generate all GIFs
# ---------------------------------------------------------------------------

def main():
    print("Generating trajectory GIFs for gradient descent decks...\n")

    # ---- M1.1 & M1.2: Vanilla GD good/bad step ----
    print("[1/12] M1.1 — GD good step (κ=1e3)")
    prob2d, x0_2d = make_quadratic(n=2, cond=1e3, seed=42)
    L = np.linalg.eigvalsh(prob2d.A)[-1]
    log_good = gd_fixed(prob2d, x0_2d, step=1.9/L, maxit=200)
    save_trajectory_gif(prob2d.f, log_good.xs, prob2d.x_star,
                        r"Vanilla GD — good step ($\alpha=1.9/L$)",
                        "gd_good_step.gif")

    print("[2/12] M1.2 — GD divergent step")
    log_bad = gd_fixed(prob2d, x0_2d, step=2.1/L, maxit=50)
    save_trajectory_gif(prob2d.f, log_bad.xs[:min(20, len(log_bad.xs))],
                        prob2d.x_star,
                        r"Vanilla GD — divergent ($\alpha=2.1/L$)",
                        "gd_diverge.gif")

    # ---- M3.3: Armijo on Rosenbrock ----
    print("[3/12] M3.3 — Armijo on Rosenbrock")
    prob_ros, x0_ros = make_rosenbrock(seed=0)
    log_armijo = gd_armijo(prob_ros, x0_ros, maxit=2000, tol=1e-6, armijo_beta=0.5)
    save_trajectory_gif(prob_ros.f, log_armijo.xs, prob_ros.x_star,
                        "GD + Armijo on Rosenbrock",
                        "armijo_rosenbrock.gif",
                        xlim=(-2, 2), ylim=(-1, 3), wide=True)

    # ---- M2.1 & M2.3: Steepest descent M=I and M=A ----
    print("[4/12] M2.1 — Steepest descent M=I (κ=1e4)")
    prob_sd, x0_sd = make_quadratic(n=2, cond=1e4, seed=7)
    log_I = steepest_descent_quad(prob_sd, x0_sd, M_kind="I", maxit=200)
    save_trajectory_gif(prob_sd.f, log_I.xs, prob_sd.x_star,
                        r"Steepest descent ($M=I$, $\kappa=10^4$)",
                        "sd_identity.gif")

    print("[5/12] M2.3 — Steepest descent M=A")
    log_A = steepest_descent_quad(prob_sd, x0_sd, M_kind="A", maxit=5)
    save_trajectory_gif(prob_sd.f, log_A.xs, prob_sd.x_star,
                        r"Steepest descent ($M=A$) — one step",
                        "sd_full_hessian.gif")

    # ---- M5.1: Newton on quadratic ----
    print("[6/12] M5.1 — Newton on quadratic")
    prob_nq, x0_nq = make_quadratic(n=2, cond=1e4, seed=7)
    log_nq = newton_method(prob_nq, x0_nq, variant="pure", maxit=5)
    save_trajectory_gif(prob_nq.f, log_nq.xs, prob_nq.x_star,
                        "Newton on quadratic — one step",
                        "newton_quadratic.gif")

    # ---- M5.2 & M5.3: Newton on Rosenbrock ----
    print("[7/12] M5.2 — Pure Newton on Rosenbrock")
    prob_nr, x0_nr = make_rosenbrock(seed=0)
    log_pure = newton_method(prob_nr, x0_nr, variant="pure", maxit=50)
    save_trajectory_gif(prob_nr.f, log_pure.xs, prob_nr.x_star,
                        "Pure Newton on Rosenbrock (no line search)",
                        "newton_pure_rosenbrock.gif",
                        xlim=(-2, 2), ylim=(-1, 3), wide=True)

    print("[8/12] M5.3 — Damped Newton on Rosenbrock")
    log_damped = newton_method(prob_nr, x0_nr, variant="damped", maxit=100)
    save_trajectory_gif(prob_nr.f, log_damped.xs, prob_nr.x_star,
                        "Damped Newton on Rosenbrock (Armijo)",
                        "newton_damped_rosenbrock.gif",
                        xlim=(-2, 2), ylim=(-1, 3), wide=True)

    # ---- M4.0a & M4.0b: GD exact LS vs CG ----
    print("[9/12] M4.0a — GD exact line search on quadratic")
    rng = np.random.default_rng(2025)
    theta = np.deg2rad(32)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    A_cg = R @ np.diag([1.0, 10.0]) @ R.T
    x_star_cg = np.array([0.7, -1.2])
    b_cg = A_cg @ x_star_cg
    prob_cg = QuadProblem(A=A_cg, b=b_cg, x_star=x_star_cg,
                          f_star=0.5*x_star_cg@(A_cg@x_star_cg) - b_cg@x_star_cg)
    # Choose x0 with both eigen components
    for _ in range(200):
        cand = x_star_cg + rng.normal(size=2) * 5.0
        V = np.linalg.eigh(A_cg)[1]
        coeffs = V.T @ (cand - x_star_cg)
        if min(abs(coeffs)) > 1e-2:
            x0_cg = cand; break
    else:
        x0_cg = x_star_cg + np.array([3.0, -2.0])

    log_gd_els = gd_exact_ls_quad(prob_cg, x0_cg, maxit=50)
    save_trajectory_gif(prob_cg.f, log_gd_els.xs, prob_cg.x_star,
                        "GD + exact line search — orthogonal steps",
                        "gd_exact_ls.gif")

    print("[10/12] M4.0b — CG on quadratic")
    xs_cg = linear_cg_2d(A_cg, b_cg, x0_cg)
    save_trajectory_gif(prob_cg.f, xs_cg, x_star_cg,
                        r"Conjugate gradient — $A$-conjugate steps",
                        "cg_conjugate.gif")

    # ---- M6.1: Dogleg on quadratic ----
    print("[11/12] M6.1 — Dogleg trust region on quadratic")
    prob_tr, x0_tr = make_quadratic(n=2, cond=1e4, seed=7)
    log_dl = trust_region(prob_tr, x0_tr, method="dogleg", Delta0=1.0, maxit=200)
    save_trajectory_gif(prob_tr.f, log_dl.xs, prob_tr.x_star,
                        r"Trust region (dogleg, $\kappa=10^4$)",
                        "dogleg_quadratic.gif")

    # ---- M8.1: BFGS on Rosenbrock ----
    print("[12/12] M8.1 — BFGS on Rosenbrock")
    prob_bfgs, x0_bfgs = make_rosenbrock(seed=0)
    log_bfgs = bfgs(prob_bfgs, x0_bfgs, maxit=300)
    save_trajectory_gif(prob_bfgs.f, log_bfgs.xs, prob_bfgs.x_star,
                        "BFGS on Rosenbrock",
                        "bfgs_rosenbrock.gif",
                        xlim=(-2, 2), ylim=(-1, 3), wide=True)

    print(f"\nDone! All GIFs saved to {OUT_DIR}/")
    for fname in sorted(os.listdir(OUT_DIR)):
        if fname.endswith('.gif'):
            fsize = os.path.getsize(os.path.join(OUT_DIR, fname)) / 1024
            print(f"  {fname}: {fsize:.0f} KB")

if __name__ == "__main__":
    main()
