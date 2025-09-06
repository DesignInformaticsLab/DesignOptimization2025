# === Module 0: Core MPC utilities (NumPy/SciPy/OSQP) ===

from __future__ import annotations

# --- Matplotlib backend selection for script runs (must be before any pyplot import) ---
try:
    import sys, matplotlib
    # If running as a plain script (not in IPython/Jupyter), pick an interactive backend if needed.
    if __name__ == "__main__" and "ipykernel" not in sys.modules:
        current = matplotlib.get_backend().lower()
        non_interactive = ("agg", "pdf", "svg", "ps", "cairo")
        if any(k in current for k in non_interactive):
            for bk in ("MacOSX", "QtAgg", "Qt5Agg", "TkAgg"):
                try:
                    matplotlib.use(bk, force=True)
                    break
                except Exception:
                    continue
except Exception:
    pass

import numpy as np
import scipy.sparse as sp

# ----------------------------
# Global handles (easy to tune)
# ----------------------------
DT = 0.02          # sampling time [s]
N_HORIZON = 30     # prediction steps

# Weights (defaults; cart-pole-specific values will be set in Module 2)
def default_weights(n_x, n_u):
    Q  = np.diag([1.0]*n_x)           # state stage cost
    R  = 1e-2 * np.eye(n_u)           # input stage cost
    Qf = 10.0 * Q                     # terminal cost
    return Q, R, Qf

# ----------------------------
# Prediction matrices Sx, Su
# ----------------------------
def build_prediction_matrices(A, B, N):
    """Return (Sx, Su) for X = Sx x0 + Su U with horizon N.
    X stacks x_1..x_N, U stacks u_0..u_{N-1}.
    Shapes:
      Sx: (N*n_x, n_x), Su: (N*n_x, N*n_u)
    """
    A = np.asarray(A)
    B = np.asarray(B)
    n_x, n_u = A.shape[0], B.shape[1]

    # Powers of A
    A_powers = [np.eye(n_x)]
    for k in range(1, N+1):
        A_powers.append(A_powers[-1] @ A)

    # Build Sx
    blocks_Sx = [A_powers[k] for k in range(1, N+1)]
    Sx = np.vstack(blocks_Sx)

    # Build Su as block lower-triangular with A^{k-1-j} B in (k,j) for j < k
    Su = np.zeros((N*n_x, N*n_u))
    for k in range(1, N+1):
        for j in range(0, k):
            Akj = A_powers[k-1-j]
            Su[(k-1)*n_x:k*n_x, j*n_u:(j+1)*n_u] = Akj @ B

    return Sx, Su

# ----------------------------
# Block-diagonal helpers
# ----------------------------
def blkdiag_repeat(M, times):
    """Return sparse blockdiag of M repeated `times` times."""
    return sp.block_diag([sp.csr_matrix(M) for _ in range(times)], format="csr")

def stack_refs(refs):
    """Stack a list/array of per-step references into a single column vector."""
    return np.vstack([r.reshape(-1,1) for r in refs])

# ----------------------------
# Condensed cost (H, f)
# ----------------------------
def build_condensed_cost(A, B, Q, R, Qf, N, x0, x_refs=None, u_refs=None):
    """
    Return H (sparse), f (dense), plus Sx, Su for later reuse.
    x_refs: list/array of length N with n_x each (x_1..x_N references)
    u_refs: list/array of length N with n_u each (u_0..u_{N-1} references)
    """
    n_x, n_u = A.shape[0], B.shape[1]
    Sx, Su = build_prediction_matrices(A, B, N)

    # Big Q and R
    Qbar = blkdiag_repeat(Q, N-1)
    Qbar = sp.block_diag([Qbar, sp.csr_matrix(Qf)], format="csr")  # add terminal
    Rbar = blkdiag_repeat(R, N)

    # References
    if x_refs is None:
        x_refs = [np.zeros(n_x) for _ in range(N)]
    if u_refs is None:
        u_refs = [np.zeros(n_u) for _ in range(N)]
    Xref = stack_refs(x_refs)    # (N*n_x, 1)
    Uref = stack_refs(u_refs)    # (N*n_u, 1)

    # Hessian and gradient
    H = (Su.T @ Qbar @ Su) + Rbar
    # f(U) = U^T [ Su^T Qbar (Sx x0 - Xref)  - Rbar Uref ]  (no 1/2 on linear term)
    f = (Su.T @ (Qbar @ (Sx @ x0.reshape(-1,1) - Xref)) - Rbar @ Uref).ravel()

    return H.tocsc(), f, Sx, Su, Qbar, Rbar

# ----------------------------
# Condensed constraints G U <= h(x0)
# ----------------------------
def build_condensed_inequalities(A, B, N, Cx=None, dx=None, Cu=None, du=None, Sx=None, Su=None, x0=None):
    """
    Build (G, h) with state and input linear inequalities.
    If Cx,dx provided: apply to each x_k (k=1..N).
    If Cu,du provided: apply to each u_k (k=0..N-1).
    """
    n_x, n_u = A.shape[0], B.shape[1]
    if Sx is None or Su is None:
        Sx, Su = build_prediction_matrices(A, B, N)

    G_list, h_list = [], []

    # State constraints
    if Cx is not None and dx is not None:
        Cx = sp.csr_matrix(Cx)
        Cx_bar = blkdiag_repeat(Cx, N)                           # (N*n_cx, N*n_x)
        Gx = Cx_bar @ Su                                         # (N*n_cx, N*n_u)
        if x0 is None:
            raise ValueError("x0 required to build state-constraint offsets h(x0).")
        hx = np.repeat(dx.reshape(-1,1), N, axis=0) - (Cx_bar @ (Sx @ x0.reshape(-1,1)))
        G_list.append(Gx)
        h_list.append(hx)

    # Input constraints
    if Cu is not None and du is not None:
        Cu = sp.csr_matrix(Cu)
        Cu_bar = blkdiag_repeat(Cu, N)                           # (N*n_cu, N*n_u)
        Gu = Cu_bar                                              # directly on U
        hu = np.repeat(du.reshape(-1,1), N, axis=0)
        G_list.append(Gu)
        h_list.append(hu)

    if G_list:
        G = sp.vstack(G_list, format="csr")
        h = np.vstack(h_list).ravel()
    else:
        G = sp.csr_matrix((0, N*n_u))
        h = np.zeros((0,))
    return G.tocsc(), h


# === Module 1: Inverted pendulum model (frictionless) ===
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm

# --------------------------------------------------------
# Parameters (all SI). Defaults match our earlier plan.
# --------------------------------------------------------
@dataclass
class CartPoleParams:
    M: float = 0.5     # cart mass [kg]
    m: float = 0.2     # pole mass [kg]
    l: float = 0.3     # COM distance from pivot [m]
    I: float = None    # pole inertia about COM [kg·m^2]; default = (1/3) m l^2 (slender rod of length 2l)
    g: float = 9.81    # gravity [m/s^2]

    def __post_init__(self):
        if self.I is None:
            # Slender rod of full length 2l: I_com = (1/12) m (2l)^2 = (1/3) m l^2
            self.I = (1.0/3.0) * self.m * (self.l**2)

# --------------------------------------------------------
# Nonlinear continuous-time dynamics: xdot = f(x,u)
# State x = [p, pdot, theta, thetadot]; input u = horizontal force
# --------------------------------------------------------
def cartpole_rhs(x: ArrayLike, u: float, par: CartPoleParams) -> np.ndarray:
    p, pdot, th, thdot = np.asarray(x, dtype=float).ravel()
    M, m, l, I, g = par.M, par.m, par.l, par.I, par.g

    s, c = np.sin(th), np.cos(th)

    # Mass matrix
    M11 = M + m
    M12 = m * l * c
    M21 = M12
    M22 = I + m * l**2
    det = M11 * M22 - M12 * M21

    # Bias terms (centripetal + gravity) written as C + G
    C1 = - m * l * s * thdot**2
    C2 = 0.0
    G1 = 0.0
    G2 = - m * g * l * s  # chosen so that theta=0 is an unstable equilibrium

    # Right-hand side: Bu - (C + G)
    r1 = u - (C1 + G1)
    r2 = 0.0 - (C2 + G2)

    # qddot = M^{-1} r
    inv11 =  M22 / det
    inv12 = -M12 / det
    inv21 = -M21 / det
    inv22 =  M11 / det

    pddot   = inv11 * r1 + inv12 * r2
    thddot  = inv21 * r1 + inv22 * r2

    return np.array([pdot, pddot, thdot, thddot], dtype=float)

# --------------------------------------------------------
# Linearization at the upright equilibrium (analytic closed form)
# --------------------------------------------------------
def linearize_cartpole_upright(par: CartPoleParams):
    M, m, l, I, g = par.M, par.m, par.l, par.I, par.g
    Delta = (M + m) * (I + m * l**2) - (m * l)**2

    a23 = - (m**2) * (l**2) * g / Delta
    a43 =   (M + m) * m * g * l     / Delta
    b2  =   (I + m * l**2)          / Delta
    b4  = - (m * l)                 / Delta

    Ac = np.array([[0, 1, 0, 0],
                   [0, 0, a23, 0],
                   [0, 0, 0, 1],
                   [0, 0, a43, 0]], dtype=float)
    Bc = np.array([[0.0],
                   [b2],
                   [0.0],
                   [b4]], dtype=float)
    return Ac, Bc

# --------------------------------------------------------
# Generic numeric linearization (central differences)
# Useful as a sanity check and for other equilibria if needed.
# --------------------------------------------------------
def numeric_linearize(rhs_fun, x_eq, u_eq, par: CartPoleParams, eps: float = 1e-6):
    x_eq = np.asarray(x_eq, dtype=float).ravel()
    n = x_eq.size
    # A = df/dx
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = np.zeros_like(x_eq)
        dx[i] = eps
        f_plus  = rhs_fun(x_eq + dx, u_eq, par)
        f_minus = rhs_fun(x_eq - dx, u_eq, par)
        A[:, i] = (f_plus - f_minus) / (2 * eps)

    # B = df/du
    du = eps
    f_plus  = rhs_fun(x_eq, u_eq + du, par)
    f_minus = rhs_fun(x_eq, u_eq - du, par)
    B = ((f_plus - f_minus) / (2 * du)).reshape(n, 1)
    return A, B

# --------------------------------------------------------
# Continuous-to-discrete (exact ZOH via augmented matrix exponential)
# --------------------------------------------------------
def c2d_exact(Ac: ArrayLike, Bc: ArrayLike, dt: float):
    Ac = np.asarray(Ac, dtype=float)
    Bc = np.asarray(Bc, dtype=float)
    n  = Ac.shape[0]
    aug = np.zeros((n + Bc.shape[1], n + Bc.shape[1]), dtype=float)
    aug[:n, :n] = Ac
    aug[:n, n:] = Bc
    # expm of block matrix
    E = expm(aug * dt)
    Ad = E[:n, :n]
    Bd = E[:n, n:]
    return Ad, Bd

# --------------------------------------------------------
# Convenience: build (A, B) for MPC at the chosen dt (DT from Module 0)
# --------------------------------------------------------
def build_discrete_linear_cartpole(par: CartPoleParams, dt: float):
    Ac, Bc = linearize_cartpole_upright(par)
    A, B   = c2d_exact(Ac, Bc, dt)
    return A, B, Ac, Bc


# === Module 2: Cart-pole loss & constraints (OSQP-ready structure) ===
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

# Imports from Module 0 utilities
# - DT, N_HORIZON
# - build_prediction_matrices, blkdiag_repeat
# Provide fallbacks if this cell is run standalone (can be removed if Module 0 is guaranteed to run first).
try:
    DT
except NameError:
    DT = 0.02
try:
    N_HORIZON
except NameError:
    N_HORIZON = 30

def _blkdiag_repeat(M, times):
    # Fallback helper if blkdiag_repeat isn't in scope
    return sp.block_diag([sp.csr_matrix(M) for _ in range(times)], format="csr")

# ----------------------------
# Default cart-pole weights
# ----------------------------
def cartpole_default_weights():
    Q  = np.diag([1.0, 0.1, 50.0, 5.0])  # [p, p_dot, theta, theta_dot]
    R  = 1e-2 * np.eye(1)                # single-input force
    Qf = 10.0 * Q
    return Q, R, Qf

# ----------------------------
# Constraint configs
# ----------------------------
@dataclass
class SoftAngleConfig:
    enabled: bool = True
    theta_max: float = 0.25   # [rad]
    lambda_s: float = 100.0   # penalty on slack^2

@dataclass
class ConstraintConfig:
    u_max: float = 10.0       # |u| <= u_max  [N]
    x_max: float = 2.4        # |p| <= x_max  [m]
    soft_angle: SoftAngleConfig = SoftAngleConfig()

# ----------------------------
# Problem structure container
# ----------------------------
@dataclass
class CondensedQPStructure:
    # Constant across time (factorizable once by OSQP)
    P: sp.csc_matrix          # Hessian on [U; s] (or on U if no slacks)
    A: sp.csc_matrix          # Constraint matrix on [U; s]
    N: int                    # horizon
    n_x: int                  # state dimension (4)
    n_u: int                  # input dimension (1)
    nU: int                   # N * n_u
    nS: int                   # N (if soft angle enabled) else 0
    use_soft_angle: bool

    # Precomputed prediction/cost pieces
    Sx: np.ndarray
    Su: np.ndarray
    Qbar: sp.csr_matrix
    Rbar: sp.csr_matrix
    H: sp.csc_matrix          # Hessian on U only

    # Hard-constraint ingredients for vector updates
    Cx: np.ndarray | None
    dx: np.ndarray | None
    Cx_bar: sp.csr_matrix | None
    G_hard: sp.csc_matrix     # rows affecting U only
    hard_row_count: int

    # Angle selection helpers
    theta_row_idx: np.ndarray | None  # indices of theta rows in Sx/Su (length N)

    def build_vectors(self, x0, x_refs=None, u_refs=None, uref_scalar=0.0):
        """
        Build q, l, u for OSQP given current x0 and optional references.
        Returns (q, l, u), where:
          - q has size nU (+ nS if soft angle)
          - l, u have size equal to A.shape[0]
        """
        n, m, N = self.n_x, self.n_u, self.N
        x0 = np.asarray(x0).reshape(-1, 1)

        # References (defaults to zeros)
        if x_refs is None:
            Xref = np.zeros((N * n, 1))
        else:
            Xref = np.vstack([np.asarray(r).reshape(-1,1) for r in x_refs])

        if u_refs is None:
            Uref = np.full((N * m, 1), float(uref_scalar))
        else:
            Uref = np.vstack([np.asarray(r).reshape(-1,1) for r in u_refs])

        # Linear term for condensed cost on U:
        # f = Su^T Qbar (Sx x0 - Xref) - Rbar Uref
        f = (self.Su.T @ (self.Qbar @ (self.Sx @ x0 - Xref)) - self.Rbar @ Uref).ravel()

        if self.use_soft_angle:
            # q = [f; 0], since slacks only enter through P (quadratic)
            q = np.concatenate([f, np.zeros(self.nS)])
        else:
            q = f

        # Hard constraints: G_hard U <= h_hard(x0)
        h_hard_list = []
        if self.Cx_bar is not None:
            # State-position (track) bounds
            hx = np.repeat(self.dx.reshape(-1,1), N, axis=0) - (self.Cx_bar @ (self.Sx @ x0))
            h_hard_list.append(hx)
        # Input bounds (already stacked inside G_hard with Cu_bar)
        # upper bound is just du repeated
        # We constructed G_hard with both state and input rows, so we need both parts of h:
        # To keep it simple, we appended state first (if any), then input.
        if h_hard_list:
            h_state = np.vstack(h_hard_list)
            # Input piece:
            # For inputs, rows count is 2*N (for +/-), with RHS = [u_max,...,u_max]^T
            hu = np.full((2*N, 1), self.du_scalar)  # self.du_scalar set during construction
            h_hard = np.vstack([h_state, hu])
        else:
            hu = np.full((2*N, 1), self.du_scalar)
            h_hard = hu

        # OSQP uses l <= A z <= u. Our hard inequalities are of the form G z <= h.
        # So l = -inf, u = h for those rows.
        inf = np.inf
        l_list = [ -inf * np.ones((self.hard_row_count,)) ]
        u_list = [ h_hard.ravel() ]

        if self.use_soft_angle:
            # Soft angle rows:
            # Build the RHS using theta rows of Sx
            Theta_Sx_x0 = (self.Sx[self.theta_row_idx, :] @ x0).ravel()  # length N

            # Upper bounds for +theta and -theta constraints
            u_theta_pos = self.theta_max * np.ones(N) - Theta_Sx_x0
            u_theta_neg = self.theta_max * np.ones(N) + Theta_Sx_x0
            u_theta = np.concatenate([u_theta_pos, u_theta_neg])

            # Their lower bounds are -inf
            l_theta = -inf * np.ones(2 * N)

            # Nonnegativity of slacks: s >= 0
            l_s = np.zeros(N)
            u_s = inf * np.ones(N)

            l_list += [l_theta, l_s]
            u_list += [u_theta, u_s]

        l = np.concatenate(l_list)
        u = np.concatenate(u_list)
        return q, l, u

# ----------------------------
# Builder for OSQP-ready structure
# ----------------------------
def prepare_cartpole_qp_structure(A, B, Q, R, Qf, N, constraints: ConstraintConfig):
    """
    Precompute constant (P, A) and store everything needed to build (q, l, u)
    at runtime given x0 and references.
    """
    A = np.asarray(A); B = np.asarray(B)
    n_x, n_u = A.shape[0], B.shape[1]
    assert n_x == 4 and n_u == 1, "This helper is specialized for cart-pole (4 states, 1 input)."

    # Prediction matrices and cost blocks
    Sx, Su = build_prediction_matrices(A, B, N)
    Qbar = _blkdiag_repeat(Q, N-1)
    Qbar = sp.block_diag([Qbar, sp.csr_matrix(Qf)], format="csr")
    Rbar = _blkdiag_repeat(R, N)
    H = (sp.csr_matrix(Su).T @ Qbar @ sp.csr_matrix(Su)) + Rbar  # keep sparse all the way
    H = H.tocsc()

    # ---- Hard constraints on position and input ----
    e_p = np.array([[1.0, 0.0, 0.0, 0.0]])
    Cx = np.vstack([ e_p, -e_p ])
    dx = np.array([constraints.x_max, constraints.x_max], dtype=float)

    Cu = np.array([[1.0], [-1.0]])
    du = np.array([constraints.u_max, constraints.u_max], dtype=float)

    Cx_bar = _blkdiag_repeat(Cx, N)            # sparse
    Su_sp  = sp.csr_matrix(Su)                 # <-- NEW: sparse Su
    Gx     = (Cx_bar @ Su_sp).tocsc()          # stays sparse
    Cu_bar = _blkdiag_repeat(Cu, N).tocsc()

    G_hard = sp.vstack([Gx, Cu_bar], format="csc")
    hard_row_count = G_hard.shape[0]

    # ---- Optional soft angle constraints with slacks ----
    use_soft = constraints.soft_angle.enabled
    if use_soft:
        theta_max = constraints.soft_angle.theta_max
        lambda_s  = constraints.soft_angle.lambda_s
        nS = N
        theta_row_idx = np.arange(N) * n_x + 2  # theta row in stacked states

        Theta_Su = Su_sp[theta_row_idx, :]      # (N, N*n_u) sparse

        A_top       = sp.hstack([G_hard, sp.csc_matrix((hard_row_count, nS))], format="csc")
        A_theta_pos = sp.hstack([ Theta_Su, -sp.eye(nS, format="csc") ], format="csc")
        A_theta_neg = sp.hstack([-Theta_Su, -sp.eye(nS, format="csc") ], format="csc")
        A_spos      = sp.hstack([ sp.csc_matrix((nS, Theta_Su.shape[1])), sp.eye(nS, format="csc") ], format="csc")
        A = sp.vstack([A_top, A_theta_pos, A_theta_neg, A_spos], format="csc")

        P = sp.block_diag([H, lambda_s * sp.eye(nS, format="csc")], format="csc")
    else:
        theta_max = None
        lambda_s  = None
        nS = 0
        theta_row_idx = None
        A = G_hard.copy()
        P = H.copy()

    spec = CondensedQPStructure(
        P=P.tocsc(), A=A.tocsc(), N=N, n_x=n_x, n_u=n_u,
        nU=N*n_u, nS=nS, use_soft_angle=use_soft,
        Sx=Sx, Su=Su, Qbar=Qbar, Rbar=Rbar, H=H,
        Cx=Cx, dx=dx, Cx_bar=Cx_bar, G_hard=G_hard, hard_row_count=hard_row_count,
        theta_row_idx=theta_row_idx
    )
    # set attributes post-init
    spec.theta_max = theta_max
    spec.lambda_s  = lambda_s
    spec.du_scalar = float(constraints.u_max)
    return spec


# === Module 3: OSQP-based MPC solver ===
import time
import numpy as np
import scipy.sparse as sp
import osqp  # pip install osqp

# Uses:
# - CondensedQPStructure, prepare_cartpole_qp_structure (Module 2)
# - DT, N_HORIZON (Module 0)

class LinearMPC:
    """
    Real-time linear MPC solver using OSQP on condensed QP.
    """
    def __init__(self, spec, osqp_settings=None):
        """
        spec: CondensedQPStructure from prepare_cartpole_qp_structure(...)
        osqp_settings: dict of OSQP options (eps_abs, eps_rel, max_iter, polish, etc.)
        """
        self.spec = spec
        self.nz = spec.nU + spec.nS
        self._model = osqp.OSQP()
        # Initial dummy vectors (will be updated before the first solve)
        q0 = np.zeros(self.nz)
        l0 = -np.inf * np.ones(spec.A.shape[0])
        u0 =  np.inf * np.ones(spec.A.shape[0])

        settings = dict(
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=4000,
            polish=False,
            verbose=False,
            adaptive_rho=True,
        )
        if osqp_settings:
            settings.update(osqp_settings)

        self._model.setup(P=spec.P, q=q0, A=spec.A, l=l0, u=u0, **settings)

        # Warm-start cache
        self._z_prev = np.zeros(self.nz)
        self._solve_time = 0.0
        self._last_status = None

    @staticmethod
    def _shift_sequence(vec, block=1):
        """Shift a stacked sequence by one block, repeating the last block."""
        vec = vec.copy()
        if vec.size % block != 0:
            raise ValueError("Vector size must be multiple of block size")
        T = vec.size // block
        if T <= 1:
            return vec
        head = vec[block: T*block]
        tail = vec[(T-1)*block: T*block]
        return np.concatenate([head, tail])

    def _shift_warm_start(self, z):
        """Shift U (and s if present) by one step for warm-start."""
        if self.spec.use_soft_angle:
            U = z[:self.spec.nU]
            S = z[self.spec.nU:]
            U2 = self._shift_sequence(U, block=self.spec.n_u)  # n_u=1 here, but keep general
            S2 = self._shift_sequence(S, block=1)
            return np.concatenate([U2, S2])
        else:
            return self._shift_sequence(z, block=self.spec.n_u)

    def solve(self, x0, x_refs=None, u_refs=None, uref_scalar=0.0):
        """
        One MPC step: update vectors with x0, warm-start, solve, and return u0 and diagnostic info.
        Returns:
          u0: scalar control to apply
          z_star: optimal decision vector [U; s?]
          info: dict with keys {'solve_time', 'status', 'obj_val'}
        """
        q, l, u = self.spec.build_vectors(x0, x_refs=x_refs, u_refs=u_refs, uref_scalar=uref_scalar)

        # Update vectors in OSQP and warm-start
        self._model.update(q=q, l=l, u=u)
        z_warm = self._shift_warm_start(self._z_prev)
        self._model.warm_start(x=z_warm)

        t0 = time.perf_counter()
        res = self._model.solve()
        t1 = time.perf_counter()
        self._solve_time = (t1 - t0)
        self._last_status = res.info.status

        if res.x is None:
            # Fallback if solver fails (keep last input)
            z_star = self._z_prev
            u0 = float(z_star[0]) if z_star.size > 0 else 0.0
            status = f"fail({self._last_status})"
            obj = np.nan
        else:
            z_star = res.x.copy()
            u0 = float(z_star[0])
            status = res.info.status
            obj = res.info.obj_val

        # Cache for next warm start
        self._z_prev = z_star.copy()

        info = dict(solve_time=self._solve_time, status=status, obj_val=obj)
        return u0, z_star, info

def rk4_step(f, x, u, dt, par):
    """One RK4 step for xdot = f(x,u)."""
    k1 = f(x, u, par)
    k2 = f(x + 0.5*dt*k1, u, par)
    k3 = f(x + 0.5*dt*k2, u, par)
    k4 = f(x + dt*k3, u, par)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# --- CartPoleRenderer (final) ---
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

class CartPoleRenderer:
    """Update artists + draw_idle; no blitting, no display handles, no auto-show."""
    def __init__(self, params, xlim=(-2.6, 2.6), dpi=110, u_max=10.0):
        self.par = params
        self.u_max = float(u_max)

        self.fig, self.ax = plt.subplots(figsize=(6.8, 2.6), dpi=dpi)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(-params.l*2.2, params.l*2.2)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for s in ("top","right","left","bottom"):
            self.ax.spines[s].set_visible(False)

        # Track
        self.track = Line2D([xlim[0], xlim[1]], [0, 0], lw=1.0, alpha=0.8)
        self.ax.add_line(self.track)

        # Cart
        self.cart_width = 0.30
        self.cart_height = 0.12
        self.cart = Rectangle((0 - self.cart_width/2, -self.cart_height/2),
                              self.cart_width, self.cart_height,
                              linewidth=1.0, edgecolor='k', facecolor='white')
        self.ax.add_patch(self.cart)

        # Pole
        self.pole = Line2D([0, 0], [0, 0], lw=1.8, color='k', alpha=0.9)
        self.ax.add_line(self.pole)

        # Control arrow (u)
        self.force_arrow = FancyArrowPatch((0, 0), (0, 0),
                                           arrowstyle='-|>', mutation_scale=16,
                                           lw=1.8, color='C0', alpha=0.9)
        self.ax.add_patch(self.force_arrow)

    def _cart_anchor(self, p):
        return (p, self.cart_height/2.0)

    def _update_force_arrow(self, p, u):
        base_x, base_y = p, 0.0
        Lnorm = np.clip(u / self.u_max, -1.0, 1.0) if self.u_max > 0 else 0.0
        dx = 0.75 * Lnorm
        self.force_arrow.set_positions((base_x, base_y), (base_x + dx, base_y))

    def draw_state(self, x, u=0.0):
        p, _, th, _ = x
        # Cart
        self.cart.set_x(p - self.cart_width/2)
        # Pole
        hx, hy = self._cart_anchor(p)
        L = 2.0 * self.par.l
        px = hx + L * math.sin(th)
        py = hy + L * math.cos(th)
        self.pole.set_data([hx, px], [hy, py])
        # Arrow
        self._update_force_arrow(p, u)

    def draw(self):
        import sys
        c = self.fig.canvas
        # In IPython/Jupyter, draw_idle is efficient; in plain scripts, force a full draw.
        if "ipykernel" in sys.modules:
            c.draw_idle()
        else:
            try:
                c.draw()  # explicit, synchronous draw for MacOSX/Tk/Qt in scripts
            except Exception:
                c.draw_idle()
        try:
            c.flush_events()
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt
            plt.pause(0.001)  # yield to GUI event loop
        except Exception:
            pass

    

def build_mpc_for_cartpole(par: CartPoleParams, dt=DT, N=N_HORIZON, soft_angle=True):
    # Linearize & discretize
    A, B, _, _ = build_discrete_linear_cartpole(par, dt)
    # Weights and constraints
    Q, R, Qf = cartpole_default_weights()

    constraints = ConstraintConfig()
    constraints.u_max = 10.0
    constraints.x_max = 2.4
    constraints.soft_angle.enabled = bool(soft_angle)
    if constraints.soft_angle.enabled:
        constraints.soft_angle.theta_max = 0.25
        constraints.soft_angle.lambda_s  = 100.0

    spec = prepare_cartpole_qp_structure(A, B, Q, R, Qf, N, constraints)
    # mpc = LinearMPC(spec, osqp_settings=dict(eps_abs=1e-4, eps_rel=1e-4, max_iter=4000, verbose=False))
    mpc = LinearMPC(spec, osqp_settings=dict(eps_abs=3e-4, eps_rel=3e-4, max_iter=2000, verbose=False, adaptive_rho=True))
    return mpc, (A, B, Q, R, Qf, constraints)


# --- RealtimeMPCSimulator (final) ---
import time, threading
import numpy as np
import matplotlib.pyplot as plt

class RealtimeMPCSimulator:
    def __init__(self, par, mpc, dt=DT, T=6.0,
                 disturb_time=0.5, disturb=(0.035, 0.3),
                 x0=None, u_max=10.0,
                 render_fps=24,         # 24–30 is smooth & light
                 phys_dt=None,          # physics substep, e.g. 0.005
                 use_threads=True,
                 close_old_figs=True):
        self.par = par
        self.mpc = mpc
        self.control_dt = float(dt)   # solver rate
        self.T = float(T)
        self.disturb_time = float(disturb_time)
        self.disturb = disturb
        self.x = np.zeros(4) if x0 is None else np.asarray(x0, dtype=float).ravel()
        self.u_latest = 0.0
        self.u_max = float(u_max)

        self.render_fps = int(render_fps)
        self.frame_dt = 1.0 / self.render_fps
        self.phys_dt = float(phys_dt) if phys_dt is not None else min(0.005, self.control_dt/2.0)
        self.use_threads = bool(use_threads)

        # dynamic logs (no index errors)
        self.t_log = []
        self.x_log = []
        self.u_log = []
        self.solve_log = []

        # one figure only
        if close_old_figs:
            try: plt.close('all')
            except Exception: pass
        self.renderer = CartPoleRenderer(par, xlim=(-2.6, 2.6), dpi=110, u_max=self.u_max)

        # shared for solver thread
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_solve_time = 0.0

    def _rk4(self, x, u, dt):
        return rk4_step(cartpole_rhs, x, u, dt, self.par)

    def _solver_loop(self):
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
            next_tick += self.control_dt

            with self._lock:
                x_meas = self.x.copy()

            t0 = time.perf_counter()
            u0, _, info = self.mpc.solve(x_meas)
            t1 = time.perf_counter()

            with self._lock:
                self.u_latest = float(u0)
                # prefer OSQP's reported time; fallback to wall time
                self._last_solve_time = float(info.get('solve_time', t1 - t0))

    def run(self, show=True):
        # Ensure interactive updates for all backends
        try:
            import matplotlib.pyplot as plt
            plt.ion()
        except Exception:
            pass

        # start solver in background
        th = None
        if self.use_threads:
            th = threading.Thread(target=self._solver_loop, daemon=True)
            th.start()

        # show figure once (non-blocking)
        try:
            self.renderer.fig.canvas.draw_idle()
            plt.show(block=False)
        except Exception:
            pass

        # real-time loop: fixed FPS render; physics substeps; T seconds total
        sim_t = 0.0
        disturbed = False
        n_frames = int(np.ceil(self.T / self.frame_dt))
        deadline = time.monotonic()

        for _ in range(n_frames):
            # integrate physics from sim_t to sim_t + frame_dt in fixed substeps
            remaining = self.frame_dt
            while remaining > 1e-12:
                dt_sub = min(self.phys_dt, remaining)
                with self._lock:
                    u = self.u_latest  # ZOH
                self.x = self._rk4(self.x, u, dt_sub)
                sim_t += dt_sub
                remaining -= dt_sub
                if (not disturbed) and (sim_t >= self.disturb_time):
                    self.x[2] += self.disturb[0]
                    self.x[3] += self.disturb[1]
                    disturbed = True

            # log (append; no fixed-size arrays)
            self.t_log.append(sim_t)
            with self._lock:
                self.u_log.append(self.u_latest)
                self.solve_log.append(self._last_solve_time)
            self.x_log.append(self.x.copy())

            # draw & give UI a timeslice
            self.renderer.draw_state(self.x, u=self.u_log[-1])
            self.renderer.draw()
            try:
                import matplotlib.pyplot as plt
                plt.pause(self.frame_dt * 0.5)
            except Exception:
                pass
            # sleep to maintain FPS (best-effort in notebooks)
            deadline += self.frame_dt
            sleep_t = deadline - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                # overrun; reset deadline to now to avoid drift
                deadline = time.monotonic()

        # stop solver thread cleanly
        self._stop.set()
        if th is not None:
            th.join(timeout=0.2)

        # return logs as arrays
        return dict(
            t=np.array(self.t_log, dtype=float),
            x=np.vstack(self.x_log).astype(float),
            u=np.array(self.u_log, dtype=float),
            solve_time=np.array(self.solve_log, dtype=float),
        )



if __name__ == "__main__":
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    # Show backends and interactive status (before/after ion())
    print("Matplotlib backend (matplotlib):", matplotlib.get_backend())
    print("Matplotlib backend (pyplot):", plt.get_backend())
    print("Interactive mode BEFORE ion():", plt.isinteractive())
    plt.ion()  # enable interactive draws for script runs
    print("Interactive mode AFTER ion():", plt.isinteractive())

    par = CartPoleParams()
    mpc, (_A,_B,_Q,_R,_Qf,constraints) = build_mpc_for_cartpole(par, dt=DT, N=N_HORIZON, soft_angle=True)

    # Try 24–30 FPS for smoother UI; physics substep 5 ms; solver at DT=0.02 s
    sim = RealtimeMPCSimulator(
        par, mpc,
        dt=DT, T=15.0,
        disturb_time=0.5, disturb=(0.135, 0.5),
        x0=np.zeros(4), u_max=constraints.u_max,
        render_fps=24, phys_dt=0.005, use_threads=True
    )
    logs = sim.run(show=True)
    print("Mean OSQP solve time [ms]:", 1e3*np.mean(logs["solve_time"]))

    # Keep the window open after the loop completes (useful in plain Python)
    plt.ioff()
    plt.show()

