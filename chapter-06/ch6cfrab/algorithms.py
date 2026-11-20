"""
D-CF-RAB and CF-RAB: distributed and centralized multi-constraint robust
beamforming for cell-free massive MIMO (Chapter 6).

Contents:
  - cell-free channel generator at 7.125 GHz (Rician model of Section 6.2.2,
    optional path-loss model of Section 6.7.1)
  - multi-constraint robust LCMV solver (Algorithm 6.1): nominal MVDR step plus
    working-set correction onto the binding worst-case constraints
  - exact single-ball solution (Section 6.3.6) used by the D-CF-RAB local stage
  - D-CF-RAB: per-AP local solves, CPU Wiener combiner and cross-AP rescaling
    check (Algorithm 6.2), with bounded fronthaul delay tau
  - comparison combiners B1..B5 and the MR / LP-MMSE / P-MMSE, local MVDR and
    INCM-reconstruction baselines
  - SINR, worst-user, timing and fronthaul metrics

Requires NumPy and SciPy; CVXPY with SCS for the primary-ball SOCP benchmark
(the same-constraint audit in scenarios.py uses Clarabel).
Wall-clock times are single-thread measurements and depend on the machine.
"""

from __future__ import annotations

import time
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.linalg import inv, norm, solve, eigh, pinv, cholesky, LinAlgError

# CVXPY is only needed for the primary-ball SOCP benchmark (B1) and the conic audits.
try:
    import cvxpy as cp
    HAVE_CVXPY = True
except Exception:
    HAVE_CVXPY = False


# =============================================================================
# 1.  Channel model
# =============================================================================

@dataclass
class CFConfig:
    """Cell-free network configuration."""
    L: int = 16                 # number of APs
    M_l: int = 8                # antennas per AP
    K: int = 8                  # single-antenna users
    area_m: float = 500.0       # square service area side length (m)
    fc_GHz: float = 7.125       # carrier frequency
    bw_MHz: float = 100.0
    scenario: str = "UMa-NLOS"  # or "UMi-LOS"
    # Per-AP CSI uncertainty radius (epsilon) tiers.
    eps_uniform: float = 0.10
    eps_tier_low: float = 0.05
    eps_tier_high: float = 0.30
    tier_fraction: float = 0.25  # fraction of "bad" fronthaul APs
    rng_seed: int = 0

    @property
    def M_total(self) -> int:
        return self.L * self.M_l


def pathloss_db(d_m: np.ndarray, fc_GHz: float, scenario: str) -> np.ndarray:
    """3GPP TR 38.901 simplified pathloss at fc_GHz (one term, NLOS UMa or LOS UMi)."""
    d_m = np.maximum(d_m, 10.0)
    if scenario == "UMa-NLOS":
        # Simplified UMa-NLOS at 7 GHz from TR 38.901 (Table 7.4.1-1, NLOS)
        return 13.54 + 39.08 * np.log10(d_m) + 20.0 * np.log10(fc_GHz) - 0.6 * (1.5 - 1.5)
    else:  # UMi-LOS
        return 32.4 + 21.0 * np.log10(d_m) + 20.0 * np.log10(fc_GHz)


def deploy_aps_users(cfg: CFConfig, rng: np.random.Generator):
    """Uniform AP and user drop in a square; UPA orientations at random azimuth."""
    ap_xy = rng.uniform(0, cfg.area_m, size=(cfg.L, 2))
    ue_xy = rng.uniform(0, cfg.area_m, size=(cfg.K, 2))
    return ap_xy, ue_xy


ERR_MODEL = 'boundary'     # 'boundary' (on the declared ball), 'interior' (radial law: relative radius fraction U^(1/(2M))), 'radius' (radius uniform in [0, eps]), 'gaussian' (unbounded, E||e||^2 = eps^2 ||h_true||^2)
USER_SYMBOLS = 'gaussian'  # symbol law of the users in the sample covariance: 'gaussian', 'qpsk', '16qam'
EXT_SYMBOLS = 'gaussian'   # symbol law of the external interferers: 'gaussian', 'qpsk', 'impulsive'
IMPULSE_P = 0.1            # burst probability of an impulsive interferer (power scaled by 1/p)
JOINT_LOG = []             # (snapshot, scored, mode) of every D-CF-RAB joint check
RESPONSE_LOG = []          # realized response Re{w^H h_k} of D-CF-RAB on the true channel at scored snapshots
_ERR_RNG = [None]

def _ball_radius(h, u, e):
    if e > 0 and ERR_MODEL != 'boundary':
        g = _ERR_RNG[0]; M_ = h.size
        if ERR_MODEL == 'interior':          # interior radial law: fraction U^(1/(2M)) of the declared relative radius
            return _ball_radius_b(h, u, e * float(g.random()) ** (1.0 / (2 * M_)))
        if ERR_MODEL == 'radius':            # radius uniform in [0, declared radius]
            return _ball_radius_b(h, u, e * float(g.random()))
        if ERR_MODEL == 'gaussian':          # CN(0, eps^2 ||h||^2 / M I): unbounded, E||e||^2 = eps^2 ||h||^2
            return e * float(np.linalg.norm(h)) * math.sqrt(float(g.gamma(M_)) / M_)
        raise ValueError('ERR_MODEL')
    return _ball_radius_b(h, u, e)

def _symbols(rng, shape, law):
    """Unit-power complex symbols: 'gaussian', 'qpsk', '16qam' or 'impulsive' (Bernoulli-Gaussian bursts)."""
    if law == 'gaussian':
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    if law == 'qpsk':
        return ((2 * rng.integers(0, 2, shape) - 1) + 1j * (2 * rng.integers(0, 2, shape) - 1)) / np.sqrt(2)
    if law == '16qam':
        return ((2 * rng.integers(0, 4, shape) - 3) + 1j * (2 * rng.integers(0, 4, shape) - 3)) / np.sqrt(10)
    if law == 'impulsive':
        g = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
        return g * (rng.random(shape) < IMPULSE_P) / np.sqrt(IMPULSE_P)
    raise ValueError('symbol law')

def _ball_radius_b(h, u, e):
    """Error norm rho along unit direction u such that ||h_hat - h|| = e ||h_hat||
    with h_hat = h + rho u: every generated error lies on the boundary of the
    declared ball of eq. (6.2), which is defined relative to the presumed channel."""
    if e <= 0:
        return 0.0
    a = float(np.real(np.vdot(h, u))); hn2 = float(np.real(np.vdot(h, h)))
    return (e * e * a + math.sqrt(e ** 4 * a * a + (1.0 - e * e) * e * e * hn2)) / (1.0 - e * e)


# ---------------------------------------------------------------------------
# Experiment switches. The defaults give the main study of Section 6.7.
COV_MODEL = 'population'   # 'population' (exact) or 'sample'
COV_SAMPLES = 256          # snapshots in the sample covariance
COV_LOADING = 1e-3         # sample mode only: diagonal loading, relative to tr(R)/(LM)
TIER_MODE = 'random'       # 'random' (0.05/0.30, random draw) or 'equal_mean' (0.05/0.25, mean 0.10)
CPU_ETA = {}               # method -> error loading eta of the CPU combiner (default 0)
PMMSE_C_SCALE = 1.0        # scale of the error covariance in P-MMSE
LP_C_SCALE = 1.0           # scale of the error covariance in LP-MMSE
LOCAL_C_OVERRIDE = None    # local bound level of D-CF-RAB (None: RABConfig default)
N_EXT_INTERF = 0           # external interferers without channel estimates (0: none)
ACPU_LOADING = 1e-3        # loading of the adaptive CPU combiner, relative to tr(R_y)/L
LMVDR_LOAD = 1e-6          # loading of local MVDR + CPU, relative to tr(R_l)/M (default of the main study)
_EXT_AT = [None]           # snapshot -> stacked interferer channels (J, LM), set per run
def _ext(s, H):
    """Append the external interferers at snapshot s to the stacked user channels."""
    return H if _EXT_AT[0] is None else np.vstack([H, _EXT_AT[0](s)])
# ---------------------------------------------------------------------------

def per_ap_eps(cfg: CFConfig, heterogeneous: bool, rng: np.random.Generator) -> np.ndarray:
    """Per-AP CSI uncertainty radius epsilon_l (one value per AP)."""
    if not heterogeneous:
        return np.full(cfg.L, cfg.eps_uniform)
    if TIER_MODE == 'equal_mean':
        eps = np.full(cfg.L, 0.05)
        eps[rng.permutation(cfg.L)[:cfg.L // 4]] = 0.25
        return eps
    bad = rng.random(cfg.L) < cfg.tier_fraction
    eps = np.where(bad, cfg.eps_tier_high, cfg.eps_tier_low)
    return eps


def gen_channel_block(cfg: CFConfig, ap_xy: np.ndarray, ue_xy: np.ndarray,
                      rng: np.random.Generator):
    """Generate (h_true, h_hat, eps_lk, large_scale) tensors of shape (L, K, M_l).

    Channel model: Rician with K-factor 7 dB.
      - dominant LOS ray at the geometric AoA from AP to user
      - 3 weak NLOS clusters, 4 rays each, ~5 deg spread
    This is the upper mid-band 7 GHz cell-free model of Section 6.2.2 and
    yields channels with a clear principal direction at each AP.
    """
    L, K, M = cfg.L, cfg.K, cfg.M_l
    h_true = np.zeros((L, K, M), dtype=np.complex128)
    large_scale = np.zeros((L, K))
    K_rice = 10 ** (7 / 10.0)
    rician_los = math.sqrt(K_rice / (K_rice + 1))
    rician_nlos = math.sqrt(1 / (K_rice + 1))

    eps_l = per_ap_eps(cfg, heterogeneous=False, rng=rng)

    m_idx = np.arange(M)
    for ell in range(L):
        for k in range(K):
            d_vec = ue_xy[k] - ap_xy[ell]
            d = max(np.linalg.norm(d_vec), 5.0)
            theta_geo = math.atan2(d_vec[1], d_vec[0])
            # Wrap into broadside-referenced AoA: theta in (-pi/2, pi/2)
            # for ULA boresight along +x
            theta_geo = math.atan2(math.sin(theta_geo), math.cos(theta_geo))
            theta_geo = max(min(theta_geo, math.pi / 2 - 0.05),
                             -math.pi / 2 + 0.05)

            PL_dB = pathloss_db(np.array([d]), cfg.fc_GHz, cfg.scenario)[0]
            beta = 10 ** (-PL_dB / 10.0)
            large_scale[ell, k] = beta

            # LOS component
            a_los = np.exp(1j * np.pi * m_idx * np.sin(theta_geo))
            h_los = rician_los * a_los

            # NLOS: 3 weak clusters, 4 rays each, near-LOS angular spread
            h_nlos = np.zeros(M, dtype=np.complex128)
            for c in range(3):
                theta_c = theta_geo + rng.normal(0, np.deg2rad(15.0))
                theta_c = max(min(theta_c, math.pi / 2 - 0.05),
                              -math.pi / 2 + 0.05)
                for r in range(4):
                    phi = theta_c + rng.normal(0, np.deg2rad(5.0))
                    phi = max(min(phi, math.pi / 2 - 0.05),
                              -math.pi / 2 + 0.05)
                    amp = (rng.normal() + 1j * rng.normal()) / math.sqrt(2 * 12)
                    h_nlos = h_nlos + amp * np.exp(1j * np.pi * m_idx * np.sin(phi))
            h_nlos = rician_nlos * h_nlos
            h_total = h_los + h_nlos
            h_true[ell, k] = math.sqrt(beta) * h_total

    # CSI error: bounded radius epsilon_l per AP (on the *un-normalized* channel)
    h_hat = np.empty_like(h_true)
    for ell in range(L):
        for k in range(K):
            noise = (rng.normal(size=M) + 1j * rng.normal(size=M)) / math.sqrt(2)
            nrm = norm(noise)
            unit_err = noise / max(nrm, 1e-12)
            # eps is a *relative* uncertainty radius
            err_norm = _ball_radius(h_true[ell, k], unit_err, eps_l[ell])
            h_hat[ell, k] = h_true[ell, k] + err_norm * unit_err
    return h_true, h_hat, eps_l, large_scale


# =============================================================================
# 2.  Core multi-constraint RAB solver (with active-set handling on the binding WC constraints)
# =============================================================================

@dataclass
class RABConfig:
    """Hyperparameters of the per-snapshot dual-step."""
    rho0: float = 1.0
    rho_min: float = 1e-3
    rho_max: float = 10.0
    eta_acc: float = 0.25
    eta_great: float = 0.75
    max_inner: int = 6
    eps_kkt: float = 1e-6
    # Diagonal regularization to keep the projection well-conditioned.
    proj_reg: float = 1e-6
    # Worst-case bound level, expressed as a fraction of the presumed-channel
    # norm:  nu_i = nu_c * ||h_hat||.  Setting nu_i to the declared uncertainty
    # radius instead makes the bound vanish as that radius shrinks, at which
    # point the constraint is inactive and the solver returns the
    # minimum-variance weight. The bound that maximises output SINR is instead a
    # fixed fraction of the presumed-channel norm, independent of the declared
    # radius.
    # remains feasible: the grid columns have norm sqrt(2)||h_hat||, so the
    # Feasibility requires nu_c < 1: for the primary column l_0 = h_hat,
    # Cauchy-Schwarz gives Re{w^H h_hat} <= ||h_hat|| ||w||, so the constraint
    # Re{w^H h_hat} >= nu_c ||h_hat|| ||w|| + 1 has no solution when nu_c >= 1.
    nu_c: float = 0.98
    local_solver: str = 'exact'   # 'exact' (closed form) or 'active_set' (Algorithm 6.1)


def rab_step(R: np.ndarray, w: np.ndarray, l_pres: np.ndarray,
                 nu: np.ndarray, cfg: RABConfig, rho_state: float = 1.0):
    """
    One RAB iteration on a single covariance R and a single weight vector.

    R       : M x M sample covariance (already loaded if needed by caller)
    w       : current M-vector (warm start; if None caller passes zeros + MV)
    l_pres  : M x N matrix of presumed constraint steering vectors
    nu      : length-N worst-case bound vector
    Returns : (w_new, lam, A_active, rho_state, info)
    """
    M, N = l_pres.shape

    # ---- 1) Unconstrained MV update direction (gradient step on Rw)
    # We use a stable normalized step: solve (R+gamma I) g = R w -> g
    gamma = cfg.proj_reg * np.trace(R).real / max(M, 1)
    Rload = R + gamma * np.eye(M)
    try:
        g = solve(Rload, R @ w) if norm(w) > 0 else np.zeros(M, dtype=np.complex128)
    except LinAlgError:
        g = R @ w
    mu = 1.0 / max(np.real(np.trace(R)) / M, 1e-6)
    w_hat = w - mu * (R @ w) * 0.0 + (np.zeros_like(w) if norm(w) == 0 else 0 * w)
    # In adaptive RLS-style step we instead just start from MV solution every snapshot:
    # w_MV = R^{-1} l_0 / (l_0^H R^{-1} l_0).  Use the primary constraint l_pres[:,0].
    l0 = l_pres[:, 0]
    try:
        Rinv_l0 = solve(Rload, l0)
        w_hat = Rinv_l0 / (l0.conj() @ Rinv_l0).real
    except LinAlgError:
        w_hat = l0 / max(norm(l0) ** 2, 1e-9)

    # ---- 2) Identify active set: violated worst-case constraints
    # Constraint i:  Re{ w^H l_i } >= nu_i * ||w|| + 1
    def margins(wv):
        proj = np.real(wv.conj() @ l_pres)              # (N,)
        rhs = nu * norm(wv) + 1.0
        return proj - rhs                                # negative -> violated

    m_hat = margins(w_hat)
    A = np.where(m_hat < 0)[0].tolist()
    if len(A) == 0:
        return w_hat, np.zeros(0), A, rho_state, {"iters": 0, "kkt_resid": 0.0}

    # ---- 2a) Scale before rotating.
    # The constraint Re{w^H l_i} >= nu_i ||w|| + 1 is homogeneous except for the
    # trailing +1, so rescaling w -> c w satisfies it for large enough c whenever
    # Re{w^H l_i}/||w|| > nu_i.  In that case the minimum-norm feasible point is a
    # rescaling of w_hat and the beam direction must not be rotated: rotating it
    # would move away from the (already optimal) minimum-variance direction.
    ratio = np.real(w_hat.conj() @ l_pres) / max(norm(w_hat), 1e-300)
    if np.all(ratio > nu + 1e-12):
        denom = np.real(w_hat.conj() @ l_pres) - nu * norm(w_hat)
        c = float(np.max(1.0 / denom))
        return (w_hat * max(c, 1.0), np.zeros(len(A)), A, rho_state,
                {"iters": 0, "kkt_resid": 0.0, "mode": "scaled"})

    # ---- 3) Direct projection onto active set:
    #         Delta w = LA tau,  Re{LA^H LA} tau = d,  d_i = nu_i*||w_hat|| + 1 - Re{w_hat^H l_i}
    LA = l_pres[:, A]
    G = np.real(LA.conj().T @ LA)
    # Scale-aware Tikhonov: the grid columns are near-parallel, so a fixed
    # absolute regulariser either vanishes or dominates depending on ||h||.
    G = G + 1e-8 * np.trace(G) / max(len(A), 1) * np.eye(len(A))
    d = (nu[A] * norm(w_hat) + 1.0) - np.real(w_hat.conj() @ LA)
    try:
        from scipy.optimize import nnls
        tau, _ = nnls(G, d)          # true NNLS, not solve-then-clip
    except Exception:
        try:
            tau = np.maximum(solve(G, d), 0.0)
        except LinAlgError:
            tau = np.maximum(pinv(G) @ d, 0.0)
    w_proj = w_hat + LA @ tau

    # ---- 4) Dual-space TR refinement on the active multipliers
    rho = rho_state
    lam = tau.copy()                  # initialize dual from projection
    kkt = 0.0
    for it in range(cfg.max_inner):
        # KKT residual: complementary slackness  lam_i * g_i(w(lam))
        # Complementarity via Fischer-Burmeister:  phi(a,b) = a + b - sqrt(a^2+b^2)
        # phi = 0  <=>  a >= 0, b >= 0, a b = 0.  With a = lam_i and b = -g_i this
        # cannot be satisfied by lam = 0 at an infeasible point, unlike the product
        # form lam_i g_i, whose trivial root lam = 0 certifies infeasible points.
        m_cur = margins(w_proj)
        gi = -m_cur[A]                # g_i = nu_i ||w|| + 1 - Re{w^H l_i}
        F = lam + (-gi) - np.sqrt(lam * lam + gi * gi + 1e-300)
        kkt = norm(F)
        if kkt < cfg.eps_kkt:
            break

        # Jacobian J = d F / d lam approximated by finite differences along the
        # subspace spanned by LA (cheap and stable; full analytic Jacobian is
        # available but the FD form is fine at |A|<=4 in our regime).
        delta = 1e-4 * (1.0 + np.abs(lam))
        J = np.zeros((len(A), len(A)))
        for j in range(len(A)):
            e = np.zeros(len(A)); e[j] = delta[j]
            lam_pert = np.maximum(lam + e, 0)
            tau_pert = lam_pert
            w_pert = w_hat + LA @ tau_pert
            m_pert = margins(w_pert)
            gi_pert = -m_pert[A]
            F_pert = (lam_pert + (-gi_pert)
                      - np.sqrt(lam_pert * lam_pert + gi_pert * gi_pert + 1e-300))
            J[:, j] = (F_pert - F) / delta[j]

        # TR subproblem: min || F + J dlam ||  s.t. ||dlam|| <= rho, lam+dlam >= 0
        # Cauchy-style step (gradient descent within TR):
        gJF = J.T @ F
        if norm(gJF) < 1e-12:
            break
        step = -gJF / max(norm(gJF), 1e-12) * rho
        # Project to non-negative orthant
        lam_new = np.maximum(lam + step, 0)
        step = lam_new - lam

        # Actual vs predicted reduction
        _l = lam + step
        _g = -margins(w_hat + LA @ _l)[A]
        F_new = _l + (-_g) - np.sqrt(_l * _l + _g * _g + 1e-300)
        ared = 0.5 * (norm(F) ** 2 - norm(F_new) ** 2)
        pred = 0.5 * (norm(F) ** 2 - norm(F + J @ step) ** 2)
        ratio = ared / max(pred, 1e-12)

        if ratio > cfg.eta_great:
            rho = min(2 * rho, cfg.rho_max)
            lam = lam + step
        elif ratio > cfg.eta_acc:
            lam = lam + step
        else:
            rho = max(rho / 2, cfg.rho_min)

        w_proj = w_hat + LA @ lam
        if norm(step) < 1e-8:
            break

    # ---- Feasibility repair, by scaling first. --------------------------
    # If the dual iteration leaves any constraint violated, a common rescale
    # of w_proj restores feasibility whenever Re{w^H l_i}/||w|| > nu_i for all
    # i, and it is free: SINR is invariant to the scale of the weight.
    _m = np.real(w_proj.conj() @ l_pres) - (nu * norm(w_proj) + 1.0)
    if (_m < -1e-9).any() and norm(w_proj) > 1e-300:
        _r = np.real(w_proj.conj() @ l_pres) / norm(w_proj)
        if np.all(_r > nu + 1e-12):
            _den = np.real(w_proj.conj() @ l_pres) - nu * norm(w_proj)
            _c = float(np.max(1.0 / _den))
            if _c > 1.0:
                w_proj = w_proj * _c

    return w_proj, lam, A, rho, {"iters": it + 1, "kkt_resid": float(kkt)}


# =============================================================================
# 3.  D-CF-RAB (per-AP + CPU consensus, bounded fronthaul delay)
# =============================================================================

@dataclass
class DistConfig:
    fronthaul_delay: int = 0      # tau in snapshots
    consensus_iters: int = 1
    cap_active: int = 4           # |A_x| cap


def estimate_aoa(h_hat_lk: np.ndarray) -> float:
    """Crude dominant-AoA estimate via grid search on a half-wavelength ULA."""
    M = h_hat_lk.size
    grid = np.linspace(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, 181)
    m_idx = np.arange(M)
    best_th, best_p = 0.0, -np.inf
    for th in grid:
        sv = np.exp(1j * np.pi * m_idx * np.sin(th)) / np.sqrt(M)
        p = np.abs(sv.conj() @ h_hat_lk) ** 2
        if p > best_p:
            best_p, best_th = float(p), float(th)
    return best_th


def build_constraint_grid(h_hat_lk: np.ndarray, eps: float, N: int = 4) -> np.ndarray:
    """
    Per-AP per-user constraint pool of size N for the WC robust LCMV problem.

    The classical literature (Vorobyov-Gershman-Luo 2003; Lobato et al. 2016)
    models the channel uncertainty as an ellipsoid around the presumed
    channel h_hat with radius eps.  We build a *direction-deterministic*
    grid of N steering vectors that span this ellipsoid:

      column 0: h_hat itself (primary constraint)
      columns 1..N-1: h_hat + eps * v_n, where v_n are eps-scaled deterministic
                       perturbations along the top principal directions of
                       a unit-trace random covariance derived from h_hat.

    Using the deterministic principal directions (rather than random draws)
    means the grid is reproducible and exercises the same DoF in every
    snapshot, which is necessary for the active-set logic to converge.

    Returns (M, N) complex matrix of presumed steering vectors.
    """
    M = h_hat_lk.size
    L = np.zeros((M, N), dtype=np.complex128)
    L[:, 0] = h_hat_lk
    if N == 1 or eps <= 0:
        return L
    # Build N-1 perturbation directions from the eigenvectors of (I - h_hat h_hat^H/||h||^2)
    # restricted to the subspace orthogonal to h_hat.  We pick the leading
    # M-1 such directions, but we only need N-1.
    h = h_hat_lk
    hh = h.conj() @ h
    if hh.real < 1e-12:
        return L
    # Householder reflector mapping h/||h|| to a multiple of e_1; its columns
    # 2..M are an orthonormal basis of the subspace orthogonal to h (eq. (6.11)).
    h_norm = math.sqrt(hh.real)
    x = h / h_norm
    alpha = -np.exp(1j * np.angle(x[0])) if abs(x[0]) > 0 else -1.0
    v = x.copy(); v[0] -= alpha
    vv = float(np.real(np.vdot(v, v)))
    Hh = np.eye(M, dtype=np.complex128) - (2.0 / vv) * np.outer(v, v.conj()) if vv > 1e-30 else np.eye(M, dtype=np.complex128)
    for n in range(1, N):
        L[:, n] = h + eps * h_norm * Hh[:, n]
    return L


def exact_wc(R, h, nu):
    """Exact single-ball worst-case robust MVDR:
       min w^H R w  s.t.  Re{w^H h} >= nu ||w|| + 1,  0 <= nu < ||h||.
    Stationarity gives w proportional to (R + mu I)^{-1} h with mu the unique root of
    mu ||(R + mu I)^{-1} h|| = nu (monotone in mu); the scale makes the constraint tight."""
    from scipy.optimize import brentq
    d, U = eigh(R); d = np.maximum(d, 1e-15); b = U.conj().T @ h; b2 = np.abs(b) ** 2
    hn = math.sqrt(float(b2.sum()))
    if nu < 0 or nu >= hn:
        return None
    if nu == 0:
        mu = 0.0
    else:
        f = lambda m: math.sqrt(float((m * m * b2 / (d + m) ** 2).sum())) - nu
        hi = max(float(d.max()), 1e-12)
        while f(hi) <= 0:
            hi *= 4.0
            if hi > 1e300:
                return None
        mu = brentq(f, 0.0, hi, xtol=1e-300, rtol=1e-14, maxiter=500)
    u = U @ (b / (d + mu))
    g = float(np.real(np.vdot(h, u))) - nu * norm(u)
    return None if g <= 0 else u / g


def cf_distributed_step(R_l: list, w_l: list, h_hat: np.ndarray,
                        eps_l: np.ndarray, nu_lk: np.ndarray,
                        rab_cfg: RABConfig, dist_cfg: DistConfig,
                        rho_state: np.ndarray, user_k: int,
                        sigma2: float = 1e-3):
    """
    One snapshot of D-CF-RAB for user k.

    Two-level structure:
      (a) Per-AP local RAB solve on R_l, producing a robust unit-gain
          weight w_l aimed at user k's per-AP estimated channel.
      (b) CPU consensus: solve a small L-dim Wiener problem in per-AP scalars
          c_l using per-AP projections w_l^H h_lk' for all users k' (signal
          and interferers).  The joint signal then coherently combines across
          APs, and the joint interference accounts for the cross-AP
          correlation structure that no per-AP solve can see.
      (c) Cross-AP binding-constraint check.

    Fronthaul cost: O(L*K) complex scalars per snapshot (projections
    w_l^H h_lk' for each user k'), plus O(L) for the returned c_l.  This is
    independent of M_l and linear in both L and K, far below the O(LM)
    raw-observation cost of fully-centralized schemes.
    """
    L = len(R_l)
    K = h_hat.shape[1]
    info = {"per_ap_iters": [], "Ax_size": 0}

    # ---- (a) Per-AP local step ----
    new_w = []
    A_locals = []
    for ell in range(L):
        l_pres = build_constraint_grid(h_hat[ell, user_k], rab_cfg.nu_c,
                                       N=nu_lk.shape[1])
        # nu carries the units of ||h_hat||: nu_i = eps_l * ||h_hat_{l,k}||
        nu_local = np.full(nu_lk.shape[1],
                           rab_cfg.nu_c * norm(h_hat[ell, user_k]))
        if rab_cfg.local_solver == 'exact':
            # exact single-ball worst-case solution: diagonally loaded MVDR with
            # the loading set by the bound (no working set to report)
            w_new = exact_wc(R_l[ell], h_hat[ell, user_k], nu_local[0])
            if w_new is None:
                w_new = h_hat[ell, user_k] / max(norm(h_hat[ell, user_k]) ** 2, 1e-30)
            lam, A, rho_new, ap_info = None, [], rho_state[ell], {"iters": 0}
        else:
            w_new, lam, A, rho_new, ap_info = rab_step(
                R_l[ell], w_l[ell], l_pres, nu_local, rab_cfg, rho_state[ell]
            )
        rho_state[ell] = rho_new
        new_w.append(w_new)
        A_locals.append(A)
        info["per_ap_iters"].append(ap_info["iters"])

    # ---- (b) CPU Wiener consensus over L per-AP scalars ----
    # Build the L-by-K matrix G[l,k'] = w_l^H h_lk' (projection of user k's
    # channel onto AP l's local beam).  This is what each AP reports to the
    # CPU each snapshot.
    G = np.zeros((L, K), dtype=np.complex128)
    for ell in range(L):
        for kk in range(K):
            G[ell, kk] = new_w[ell].conj() @ h_hat[ell, kk]
    # Per-AP noise variance after applying the local weight
    noise_per_ap = np.array([sigma2 * (new_w[ell].conj() @ new_w[ell]).real
                              for ell in range(L)])
    _eta = CPU_ETA.get('D-CF-RAB', 0.0)
    if _eta:
        noise_per_ap = noise_per_ap + np.array([
            _eta * eps_l[ell] ** 2 * float(np.sum(np.abs(h_hat[ell]) ** 2)) / h_hat.shape[2]
            * (new_w[ell].conj() @ new_w[ell]).real for ell in range(L)])

    # CPU Wiener: maximize  |c^H g_k|^2 / (sum_{k'!=k} |c^H g_k'|^2 + c^H diag(noise_per_ap) c)
    # Solution:  c = (sum_{k'!=k} g_k' g_k'^H + diag(noise_per_ap))^{-1} g_k
    g_des = G[:, user_k]
    Q = np.diag(noise_per_ap.astype(np.complex128))
    for kk in range(K):
        if kk == user_k:
            continue
        Q = Q + np.outer(G[:, kk], G[:, kk].conj())
    Q = Q + 1e-9 * np.eye(L)
    try:
        c = solve(Q, g_des)
    except LinAlgError:
        c = g_des / max(noise_per_ap.mean(), 1e-12)
    # Normalize so the joint signal is exactly 1
    s = np.vdot(c, g_des).real
    if abs(s) > 1e-12:
        c = c / s

    for ell in range(L):
        new_w[ell] = c[ell] * new_w[ell]

    # ---- (c) Cross-AP binding check ----
    joint_lhs = sum(np.real(new_w[ell].conj() @ h_hat[ell, user_k]) for ell in range(L))
    # Joint constraint:  sum_l Re{w_l^H h_l} >= sum_l nu_l ||w_l|| + 1.
    # The worst-case penalty is ADDED to the unit response, not subtracted;
    # with the sign reversed a larger uncertainty radius made the joint
    # constraint easier to satisfy and the check never fired.
    joint_rhs = 1.0 + sum(eps_l[ell] * norm(h_hat[ell, user_k]) * norm(new_w[ell])
                          for ell in range(L))
    fh_scalars = 2 * L * K + 2 * L + sum(len(A_locals[ell]) for ell in range(L))
    if joint_lhs < joint_rhs:
        info["Ax_size"] = 1
        # ---- Scale before rotating, at the JOINT level. -------------------
        # The joint constraint is homogeneous in the stacked weight except for
        # the trailing +1, so multiplying every per-AP weight by a common
        # gamma > 0 scales both the response and the worst-case penalty while
        # leaving the +1 fixed.  Feasibility by pure scaling is therefore
        # possible whenever  S := sum_l Re{w_l^H h_l}  exceeds
        # P := sum_l nu_l ||w_l||, and the rescale is free: the stacked weight
        # is scaled, and SINR is invariant to the scale of the combiner.
        S = sum(np.real(new_w[e2].conj() @ h_hat[e2, user_k]) for e2 in range(L))
        P = sum(eps_l[e2] * norm(h_hat[e2, user_k]) * norm(new_w[e2])
                for e2 in range(L))
        if S - P > 1e-12:
            gamma = 1.0 / (S - P)
            if gamma > 1.0:
                for e2 in range(L):
                    new_w[e2] = gamma * new_w[e2]
            info["joint_mode"] = "scaled"
            info["alpha"] = 0.0
            info["fronthaul_scalars"] = fh_scalars   # the CPU applies the scale itself
            return new_w, rho_state, info
        # Rescaling cannot restore (6.14): record the violation and keep the weights.
        # No direction-changing fallback is applied (it would need extra messages).
        info["joint_mode"] = "violated"
        info["alpha"] = 0.0
        info["fronthaul_scalars"] = fh_scalars
        return new_w, rho_state, info
        info["joint_mode"] = "rotated"
        # Additive correction along the presumed-channel direction at each AP,
        # w_l <- w_l + alpha * h_hat_l / ||h_hat_l||^2, with a single scalar
        # alpha >= 0 shared across APs (Algorithm 6.2, step 5).  alpha is found by
        # bisection rather than set to the raw deficit: the worst-case penalty
        # on the right-hand side also grows with ||w_l||, so a step sized by the
        # deficit alone overshoots and does not restore feasibility.
        def _residual(alpha):
            lhs = 0.0
            rhs = 1.0
            for e2 in range(L):
                hh = h_hat[e2, user_k]
                hn = max(norm(hh), 1e-12)
                wv = new_w[e2] + alpha * hh / (hn * hn)
                lhs += np.real(wv.conj() @ hh)
                rhs += eps_l[e2] * hn * norm(wv)
            return lhs - rhs

        lo, hi, ok = 0.0, 1.0, False
        for _ in range(60):                       # bracket
            if _residual(hi) >= 0.0:
                ok = True
                break
            lo, hi = hi, hi * 2.0
            if hi > 1e12:
                break
        if ok:
            for _ in range(80):                   # bisect
                mid = 0.5 * (lo + hi)
                if _residual(mid) >= 0.0:
                    hi = mid
                else:
                    lo = mid
                if hi - lo <= 1e-9 * max(1.0, hi):
                    break
            alpha = hi
            for ell in range(L):
                hh = h_hat[ell, user_k]
                hn = max(norm(hh), 1e-12)
                new_w[ell] = new_w[ell] + alpha * hh / (hn * hn)
            info["alpha"] = float(alpha)
        else:
            info["alpha"] = float("nan")          # infeasible bracket
        fh_scalars += L

    info["fronthaul_scalars"] = fh_scalars
    return new_w, rho_state, info


# =============================================================================
# 4.  Baselines
# =============================================================================

def beamformer_cf_rab(R_stack: np.ndarray, h_stack: np.ndarray,
                               nu_vec: np.ndarray, rab_cfg: RABConfig,
                               eps_total: float = 0.1):
    """B2: Centralized CF-RAB applied to the stacked LM-dimensional array.

    Constraint grid: primary column is h_stack; additional columns are
    h_stack perturbed by random vectors on the eps-radius sphere relative
    to ||h_stack||.  This is the classical Vorobyov-Gershman-Luo style grid,
    scaled for the LM-dimensional stacked channel.
    """
    M = h_stack.size
    l_pres = np.zeros((M, nu_vec.size), dtype=np.complex128)
    l_pres[:, 0] = h_stack
    l_pres = build_constraint_grid(h_stack, eps_total, N=nu_vec.size)
    w0 = np.zeros(M, dtype=np.complex128)
    w_new, _, _, _, _ = rab_step(R_stack, w0, l_pres, nu_vec, rab_cfg, 1.0)
    return w_new


# =============================================================================
#  CF-RAB-ANG: centralized CF-RAB with the angular constraint grid of Chapter 3,
#  [th, th-2deg, th+2deg, th-4deg] with nu = [1.9, 1.5, 1.5, 1.1], applied
#  block-wise across APs (not used in the chapter figures).  Those nu values
#  suit separated directions.  A grid of near-duplicate columns of h would
#  reduce the working-set correction to a rescaling of w and, because SINR is
#  scale-invariant, the solution to diagonally loaded MVDR; the main receiver
#  therefore uses the Householder grid of (6.11).
# =============================================================================

CFRAB_ANG_OFFSETS_DEG = (0.0, -2.0, +2.0, -4.0)


def _steer(theta: float, M: int) -> np.ndarray:
    return np.exp(1j * np.pi * np.arange(M) * np.sin(theta)) / np.sqrt(M)


def build_angular_grid_stacked(h_stack: np.ndarray, L: int, M: int,
                               offsets_deg=CFRAB_ANG_OFFSETS_DEG) -> np.ndarray:
    """Block-wise angular constraint grid for the stacked LM-dim channel.

    Column n is the concatenation over APs of a(theta_l + offset_n), each block
    rescaled to the per-AP channel norm so the relative AP weighting of the
    presumed channel is preserved.
    """
    G = np.zeros((L * M, len(offsets_deg)), dtype=np.complex128)
    for n, d in enumerate(offsets_deg):
        col = np.zeros(L * M, dtype=np.complex128)
        for ell in range(L):
            blk = h_stack[ell * M:(ell + 1) * M]
            th = estimate_aoa(blk)
            col[ell * M:(ell + 1) * M] = _steer(th + np.deg2rad(d), M) * norm(blk)
        G[:, n] = col
    return G


def beamformer_cf_rab_angular(R_stack: np.ndarray, h_stack: np.ndarray,
                              nu_vec: np.ndarray, rab_cfg: RABConfig,
                              L: int, M: int):
    """CF-RAB on the stacked array using the angular grid of Chapter 3."""
    l_pres = build_angular_grid_stacked(h_stack, L, M)
    nu = np.asarray(nu_vec, dtype=float)[:l_pres.shape[1]]
    w0 = np.zeros(h_stack.size, dtype=np.complex128)
    w_new, _, _, _, _ = rab_step(R_stack, w0, l_pres, nu, rab_cfg, 1.0)
    return w_new


# =============================================================================
#  MR / LP-MMSE / P-MMSE combining, as defined in E. Bjornson and
#  L. Sanguinetti, "Scalable Cell-Free Massive MIMO Systems," IEEE Trans.
#  Commun., 2020 (reference [6] of the chapter).
#
#  Two modelling choices:
#   (a) [6] normalises the noise to unity and carries the uplink power p; here
#       sigma^2 is explicit and p = 1.  The two parameterisations differ by a
#       common scale of the combiner, and SINR is invariant to that scale.
#   (b) [6] uses the estimation-error covariance C of its pilot MMSE estimator.
#       Here the CSI error is the bounded set of (6.2), so C = (eps ||h_hat||)^2
#       / M * I, the second moment of the generated boundary error (an isotropic
#       design choice, Section 6.7.1).
# =============================================================================

def _bj_C(h_hat_l, eps_l, M):
    tot = 0.0
    for kk in range(h_hat_l.shape[0]):
        tot += (eps_l * norm(h_hat_l[kk])) ** 2
    return (tot / M) * np.eye(M, dtype=np.complex128)


def beamformer_mr(h_view, user_k):
    """V_MR = Hhat(:,k), stacked over APs."""
    L = h_view.shape[0]
    return np.concatenate([h_view[l, user_k] for l in range(L)])


def beamformer_lp_mmse(h_view, eps_l, user_k, sigma2):
    """V_LP_MMSE = (V_MR V_MR^H + C + sigma2 I)^-1 V_MR, per AP."""
    L, K, M = h_view.shape
    out = []
    for l in range(L):
        V = h_view[l].T
        A = V @ V.conj().T + LP_C_SCALE * _bj_C(h_view[l], eps_l[l], M) + sigma2 * np.eye(M)
        out.append(solve(A, h_view[l, user_k]))
    return np.concatenate(out)


def cpu_wiener_combine(blocks, h_view, user_k, sigma2, eta=0.0, eps_l=None):
    """Apply the CPU rule of D-CF-RAB, eq. (6.13), to arbitrary local weights.
    blocks: list of L local weights (M,). Returns the stacked effective weight."""
    L, K, M = h_view.shape
    G = np.array([[np.vdot(blocks[l], h_view[l, i]) for i in range(K)] for l in range(L)])
    q = np.array([sigma2 * float(np.real(np.vdot(blocks[l], blocks[l]))) for l in range(L)])
    if eta:
        # isotropic projected-error proxy: eta * eps_l^2 * sum_k ||h_hat_lk||^2 / M per unit-norm block
        q = q + np.array([eta * eps_l[l] ** 2 * float(np.sum(np.abs(h_view[l]) ** 2)) / M
                          * float(np.real(np.vdot(blocks[l], blocks[l]))) for l in range(L)])
    Q = np.diag(q.astype(np.complex128))
    for i in range(K):
        if i != user_k:
            Q = Q + np.outer(G[:, i], G[:, i].conj())
    c = solve(Q + 1e-9 * np.eye(L), G[:, user_k])
    return np.concatenate([c[l] * blocks[l] for l in range(L)])


def beamformer_p_mmse(h_view, eps_l, user_k, sigma2):
    """v = (Hhat Hhat^H + C_blk + sigma2 I)^-1 Hhat(:,k) at the CPU."""
    L, K, M = h_view.shape
    Ha = np.zeros((L * M, K), dtype=np.complex128)
    Cb = np.zeros((L * M, L * M), dtype=np.complex128)
    for l in range(L):
        Ha[l * M:(l + 1) * M, :] = h_view[l].T
        Cb[l * M:(l + 1) * M, l * M:(l + 1) * M] = PMMSE_C_SCALE * _bj_C(h_view[l], eps_l[l], M)
    A = Ha @ Ha.conj().T + Cb + sigma2 * np.eye(L * M)
    return solve(A, Ha[:, user_k])


# =============================================================================
#  INCM-reconstruction baseline (RECON-RAB)
#  The interference-plus-noise covariance is rebuilt by integrating the Capon
#  spectrum over the field of view with a sector around the presumed direction
#  excluded, in the manner of Gu and Leshem (IEEE TSP, 2012), reference [20] of
#  the chapter.
#
#  This baseline needs an array manifold to integrate over, so it runs per AP on
#  the M-element half-wavelength ULA. It cannot be applied at the joint LM
#  dimension, where the stacked channel is not a steering vector of a uniform
#  array. It is therefore a per-AP competitor and is combined across APs with
#  the same equal-gain stacking used by the local baselines.
# =============================================================================

_FOV_CACHE = {}
_N_FOV = 181


def _fov(M):
    if M not in _FOV_CACHE:
        g = np.linspace(-np.pi / 2, np.pi / 2, _N_FOV)
        A = np.column_stack([np.exp(1j * np.pi * np.arange(M) * np.sin(t)) / np.sqrt(M)
                             for t in g])
        _FOV_CACHE[M] = (g, A)
    return _FOV_CACHE[M]


def _inc_reconstruct(R, th_c, half_deg):
    """Capon-integral INC reconstruction, excluding a sector around th_c."""
    M = R.shape[0]
    dl = 1e-3 * np.trace(R).real / M
    Rinv = inv(R + dl * np.eye(M))
    g, A = _fov(M)
    out = np.abs(g - th_c) > np.deg2rad(half_deg)
    Ao = A[:, out]
    p = 1.0 / np.real(np.einsum('ij,ik,kj->j', Ao.conj(), Rinv, Ao))
    return (Ao * p) @ Ao.conj().T + dl * np.eye(M)


def beamformer_recon_rab(R_l, h_view, user_k, half_deg=8.0):
    """Per-AP INCM reconstruction followed by MVDR on the reconstructed INC."""
    L = len(R_l)
    out = []
    for ell in range(L):
        h = h_view[ell, user_k]
        M = h.size
        th = estimate_aoa(h)
        try:
            R_rec = _inc_reconstruct((R_l[ell] + R_l[ell].conj().T) / 2, th, half_deg)
            v = solve(R_rec, h)
            w = v / max(abs(h.conj() @ v), 1e-12)
        except Exception:
            w = h / max(norm(h), 1e-12)
        if not np.all(np.isfinite(w)):
            w = h / max(norm(h), 1e-12)
        out.append(w)
    return np.concatenate(out)


def beamformer_local_mmse(R_l_IN: list, h_hat_lk: np.ndarray,
                          load_db: float = -10.0) -> list:
    """B3: Local diagonally-loaded MMSE per AP (no robustness, no CPU).

    Per-AP solution:  w_l = (R_IN,l + gamma I)^(-1) h_hat_l,
    normalized so that |w_l^H h_hat_l| = 1 (MVDR-style normalization).

    R_l_IN is the per-AP interference-plus-noise covariance (desired user
    excluded), matching the standard MVDR setup.  Diagonal loading gamma
    is a small fraction of tr(R)/M to regularize near-singular R, NOT the
    aggressive +10 dB loading used previously, which was masking gain.
    """
    out = []
    for ell, R in enumerate(R_l_IN):
        M = R.shape[0]
        # Mild diagonal loading: gamma = 10^(load_db/10) * tr(R)/M
        gamma = (10 ** (load_db / 10.0)) * np.trace(R).real / M
        Rload = R + gamma * np.eye(M)
        try:
            w = solve(Rload, h_hat_lk[ell])
        except LinAlgError:
            w = h_hat_lk[ell]
        gain = np.abs(h_hat_lk[ell].conj() @ w)
        out.append(w / max(gain, 1e-9))
    return out


def beamformer_centralized_sdp(R_stack: np.ndarray, h_stack: np.ndarray,
                               eps_total: float, time_limit: float = 5.0):
    """B1: Centralized SOCP gold standard, real-valued reformulation.

    Robust LCMV in complex variables can be rewritten with the real-valued
    block representation:   w_R = [Re(w); Im(w)], h_R similarly, and R_R is
    the 2M x 2M block:   [[Re(R), -Im(R)], [Im(R), Re(R)]].
    The robust constraint  Re{w^H h} >= eps ||w||_2 + 1  becomes a single
    second-order cone inequality on the real vector w_R.
    Solved only for total array sizes LM <= 320.
    """
    if not HAVE_CVXPY:
        return None
    M = h_stack.size
    if M > 320:
        return None
    R = R_stack
    R_R = np.block([[R.real, -R.imag], [R.imag, R.real]])
    R_R = (R_R + R_R.T) / 2 + 1e-8 * np.eye(2 * M)
    h_R = np.concatenate([h_stack.real, h_stack.imag])

    w_R = cp.Variable(2 * M)
    # Re{w^H h} = h_R^T w_R
    constraints = [h_R @ w_R >= eps_total * cp.norm(w_R, 2) + 1]
    obj = cp.Minimize(cp.quad_form(w_R, cp.psd_wrap(R_R)))
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False,
                   eps=1e-6, max_iters=10000)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        wR = w_R.value
        return wR[:M] + 1j * wR[M:]
    except Exception:
        return None


# =============================================================================
# 5.  Snapshot loop + SINR measurement
# =============================================================================

def sinr_from_weights(w_full: np.ndarray, h_true_k: np.ndarray,
                      H_true: np.ndarray, sigma2: float, k: int) -> float:
    """Output SINR for the joint receive combiner w_full on user k.
       h_true_k : stacked true channel for user k (LM,)
       H_true   : (K, LM) stacked channels for all users
    """
    sig = np.abs(w_full.conj() @ h_true_k) ** 2
    interf = 0.0
    for kk in range(H_true.shape[0]):
        if kk == k: continue
        interf += np.abs(w_full.conj() @ H_true[kk]) ** 2
    noise = sigma2 * (w_full.conj() @ w_full).real
    return float(sig / max(interf + noise, 1e-12))


def stack_w(w_l: list) -> np.ndarray:
    return np.concatenate(w_l)


UPDATE_TIMING = {}
_TKEY = [None]
def _eval_per_snapshot_t(key, *a, **kw):
    _TKEY[0] = key
    try:
        return _eval_per_snapshot(*a, **kw)
    finally:
        _TKEY[0] = None

def _eval_per_snapshot(make_w, n_snapshots, truth_at, K, k, sigma2,
                       stale_hat, stride=5):
    """Score a single-shot combiner the way a tracking combiner is scored.

    For each evaluation snapshot s the weight is rebuilt from the CSI actually
    available at s, then scored against the truth at s. Averaging one weight
    over a window instead scores it against instants it was never intended for,
    which rewards stale CSI and makes the delay axis non-monotone.
    """
    acc = []
    for s in range(n_snapshots // 4, n_snapshots, stride):
        _t0 = time.perf_counter()
        w = make_w(stale_hat(s))
        if _TKEY[0] is not None:
            _e = UPDATE_TIMING.setdefault(_TKEY[0], [0.0, 0])
            _e[0] += time.perf_counter() - _t0; _e[1] += 1
        if w is None:
            continue
        H_true_s = np.stack([truth_at(s)[:, kk, :].reshape(-1) for kk in range(K)])
        acc.append(sinr_from_weights(w, H_true_s[k], _ext(s, H_true_s), sigma2, k))
    return float(np.mean(acc)) if acc else float("nan")


JOINT_MODES = {}

def run_snapshot_trace(cfg: CFConfig, n_snapshots: int = 100,
                       SNR_dB: float = 0.0, INR_dB: float = 10.0,
                       fronthaul_delay: int = 0,
                       heterogeneous: bool = False,
                       eps_override: Optional[float] = None,
                       stale_centralized: bool = True,
                       methods=("D-CF-RAB", "CF-RAB", "local_mmse"),
                       seed: int = 0,
                        soi_in_cov: bool = True,
                        eval_at_last: bool = True,
                       gain_model: str = 'normalized',
                       cov_mode: str = 'fixed',
                       probe=None):
    """
    Run a full snapshot trace and return per-user SINR (dB) per method.

    Parameters of interest beyond the defaults:
      eps_override        : if set, overrides cfg.eps_uniform and re-injects
                            CSI error of that radius into h_hat. Used by E7.
      stale_centralized   : if True, the centralized RAB baseline also sees
                            the stale h_hat snapshot from tau steps ago,
                            making the E2 comparison apples-to-apples.
    """
    UPDATE_TIMING.clear()
    rng = np.random.default_rng(seed)
    _ERR_RNG[0] = np.random.default_rng([int(seed), 313])   # separate stream: directions and channels unchanged
    ap_xy, ue_xy = deploy_aps_users(cfg, rng)
    h_true, h_hat, eps_l, beta_lk = gen_channel_block(cfg, ap_xy, ue_xy, rng)

    L, K, M = cfg.L, cfg.K, cfg.M_l

    # Apply eps_override or heterogeneous tiers, then RE-INJECT CSI error
    # of the appropriate radius, so that a sweep of eps changes the CSI error
    # (gen_channel_block draws h_hat with the default cfg.eps_uniform).
    if eps_override is not None:
        eps_l = np.full(L, float(eps_override))
    elif heterogeneous:
        eps_l = per_ap_eps(cfg, heterogeneous=True, rng=rng)

    # Re-derive h_hat from h_true with the (possibly overridden) eps_l.
    h_hat = np.empty_like(h_true)
    for ell in range(L):
        for k in range(K):
            noise = (rng.normal(size=M) + 1j * rng.normal(size=M)) / math.sqrt(2)
            nrm = norm(noise)
            unit_err = noise / max(nrm, 1e-12)
            err_norm = _ball_radius(h_true[ell, k], unit_err, eps_l[ell])
            h_hat[ell, k] = h_true[ell, k] + err_norm * unit_err

    # ----- Link budget calibration -----
    # Standard cell-free benchmark calibration: each AP-user channel is
    # normalized to a per-element large-scale gain drawn around 1, so each
    # AP sees user k with a comparable but slightly varying received power.
    # This isolates the beamforming/combining algorithm performance from
    # the (separate) AP-clustering question.
    if gain_model == 'pathloss':
        # Full large-scale model: the 3GPP UMa-NLOS path loss of gen_channel_block
        # is kept, log-normal shadowing with 8 dB standard deviation is added,
        # and channels are expressed relative to the noise power for 23 dBm
        # user transmit power over 100 MHz with a 7 dB noise figure
        # (noise = -174 + 80 + 7 = -87 dBm). The noise variance is then 1.
        noise_rel = 10 ** ((-174.0 + 10 * math.log10(cfg.bw_MHz * 1e6) + 7.0 - 23.0) / 10.0)
        for ell in range(L):
            for kk in range(K):
                sc = 10 ** (rng.normal(0.0, 8.0) / 20.0) / math.sqrt(noise_rel)
                h_true[ell, kk] = h_true[ell, kk] * sc
                h_hat[ell, kk] = h_hat[ell, kk] * sc
        sigma2 = 1.0
    else:
        for ell in range(L):
            for kk in range(K):
                nrm = norm(h_true[ell, kk])
                if nrm > 1e-12:
                    # Per-AP shadow fade ~ uniform [-3, +3] dB
                    shadow_dB = rng.uniform(-3, 3)
                    target_nrm = math.sqrt(M) * 10 ** (shadow_dB / 20.0)
                    h_true[ell, kk] = h_true[ell, kk] * (target_nrm / nrm)
                    h_hat[ell, kk]  = h_hat[ell, kk]  * (target_nrm / nrm)
        sigma2 = 10 ** (-SNR_dB / 10.0)
    inr_lin = 10 ** (INR_dB / 10.0)   # used only when N_EXT_INTERF > 0
    _EXT_AT[0] = None
    _jam_stack = None
    if N_EXT_INTERF > 0:
        # External interferers: same channel model, normalization and aging as the users, with a
        # separate random stream so the users' channels are unchanged. They have no channel
        # estimates; they enter the covariance and the SINR only.
        from dataclasses import replace as _dc_replace
        rng_j = np.random.default_rng([int(seed), 717])
        J = int(N_EXT_INTERF)
        cfg_j = _dc_replace(cfg, K=J)
        _, jam_xy = deploy_aps_users(cfg_j, rng_j)
        g_t, _, _, _ = gen_channel_block(cfg_j, ap_xy, jam_xy, rng_j)
        for ell in range(L):
            for j in range(J):
                if gain_model == 'pathloss':
                    g_t[ell, j] *= 10 ** (rng_j.normal(0.0, 8.0) / 20.0) / math.sqrt(noise_rel)
                else:
                    g_t[ell, j] *= math.sqrt(M) * 10 ** (rng_j.uniform(-3, 3) / 20.0) / max(norm(g_t[ell, j]), 1e-12)
        g_t = g_t * math.sqrt(sigma2 * inr_lin)          # per-antenna interference-to-noise ratio INR_dB
        _seq = [g_t.copy()]
        for _s in range(1, n_snapshots):
            inn = (rng_j.normal(size=g_t.shape) + 1j * rng_j.normal(size=g_t.shape)) / math.sqrt(2)
            for ell in range(L):
                for j in range(J):
                    inn[ell, j] *= norm(g_t[ell, j]) * math.sqrt(1 - 0.995 ** 2) / math.sqrt(M)
            g_t = 0.995 * g_t + inn
            _seq.append(g_t.copy())
        _jam_stack = [np.stack([g[:, j, :].reshape(-1) for j in range(J)]) for g in _seq]
        _EXT_AT[0] = lambda s_, _st=_jam_stack: _st[s_]
    # Every user transmits with the same power p_k = 1. The covariance, the
    # SINR metric (6.5)-(6.6), the MMSE baselines and the Wiener consensus all use
    # this one power model.
    P_USER = 1.0

    # Per-AP total received-signal covariance: noise + ALL user contributions
    # (including desired). This is what the D-CF-RAB local stage sees, matching
    # the realistic adaptive sample-covariance model.
    # NOTE: R_l is per-AP and user-independent, so it can only carry the
    # contaminated model. When soi_in_cov is False the distributed kernel is
    # given the per-user INC instead, selected below.
    R_l = []
    for ell in range(L):
        R = sigma2 * np.eye(M, dtype=np.complex128)
        for kk in range(K):
            R = R + P_USER * np.outer(h_true[ell, kk], h_true[ell, kk].conj())
        R_l.append(R)

    # Per-user per-AP I+N covariance (excludes desired user k). Used by
    # local MMSE baseline so that B3 is a fair MVDR reference, not penalized
    # by self-nulling.
    R_l_IN_per_k = []
    for k in range(K):
        R_l_IN = []
        for ell in range(L):
            R = sigma2 * np.eye(M, dtype=np.complex128)
            for kk in range(K):
                if (not soi_in_cov) and kk == k:
                    continue
                R = R + P_USER * np.outer(h_true[ell, kk], h_true[ell, kk].conj())
            R_l_IN.append(R)
        R_l_IN_per_k.append(R_l_IN)

    H_stack_true = np.stack([h_true[:, k, :].reshape(-1) for k in range(K)])   # (K, LM)
    h_stack_hat = np.stack([h_hat[:, k, :].reshape(-1) for k in range(K)])
    # R_stack[k] for the centralized baselines B1/B2: full LM x LM
    # interference-plus-noise covariance for user k (excludes user k itself
    # from the covariance, matching the standard MVDR / robust-LCMV setup).
    # This is the optimistic gold-standard setup: it assumes the centralized
    # CPU can perfectly separate the desired user from the receive samples,
    # which a real distributed sim cannot do.
    # Covariance model, applied identically to every method.
    #   soi_in_cov = True  : noise + ALL users, i.e. the desired signal is
    #                        present, as in an adaptive sample covariance.
    #   soi_in_cov = False : noise + interferers only, i.e. the exact
    #                        interference-plus-noise covariance.
    # Every method receives the same covariance model, so the comparison is
    # like-for-like.
    R_stack_per_k = []
    for k in range(K):
        Rk = sigma2 * np.eye(L * M, dtype=np.complex128)
        for kk in range(K):
            if (not soi_in_cov) and kk == k:
                continue
            hk = H_stack_true[kk]
            Rk = Rk + P_USER * np.outer(hk, hk.conj())
        R_stack_per_k.append(Rk)

    if _jam_stack is not None:
        G0 = _jam_stack[0]                                   # (J, LM), includes interferer power
        Rj = G0.T @ G0.conj()
        for ell in range(L):
            blk = Rj[ell * M:(ell + 1) * M, ell * M:(ell + 1) * M]
            R_l[ell] = R_l[ell] + blk
            for kk in range(K):
                R_l_IN_per_k[kk][ell] = R_l_IN_per_k[kk][ell] + blk
        R_stack_per_k = [Rk + Rj for Rk in R_stack_per_k]
    if COV_MODEL == 'sample':
        if not soi_in_cov:
            raise ValueError('sample covariance contains every user; use soi_in_cov=True')
        rng_cov = np.random.default_rng([int(seed), 991])       # separate stream: truth unchanged
        ns = int(COV_SAMPLES)
        z = _symbols(rng_cov, (ns, K), USER_SYMBOLS)
        nz = (rng_cov.normal(size=(ns, L * M)) + 1j * rng_cov.normal(size=(ns, L * M))) / np.sqrt(2)
        X = np.sqrt(P_USER) * z @ H_stack_true + np.sqrt(sigma2) * nz   # rows are snapshots x^T
        if _jam_stack is not None:
            zj = _symbols(rng_cov, (ns, _jam_stack[0].shape[0]), EXT_SYMBOLS)
            X = X + zj @ _jam_stack[0]
        Rs = X.T @ X.conj() / ns
        Rs = (Rs + Rs.conj().T) / 2
        Rs = Rs + COV_LOADING * np.trace(Rs).real / (L * M) * np.eye(L * M)
        R_stack_per_k = [Rs for _ in range(K)]
        R_l = [Rs[l * M:(l + 1) * M, l * M:(l + 1) * M].copy() for l in range(L)]
        R_l_IN_per_k = [R_l for _ in range(K)]

    # Worst-case bound level.  Eq. (6.2) states the uncertainty relatively,
    # ||h - h_hat|| <= eps ||h_hat||, so the bound that enters the constraint
    # Re{w^H l_i} >= nu_i ||w|| + 1 must carry the same units as ||h_hat||:
    #     nu_{l,i} = eps_l * ||h_hat_{l,k}||.
    # A fixed vector would make the constraint level independent of the
    # uncertainty radius, so nu is built per AP and per user below.
    nu_lk = np.tile([1.0, 1.0, 1.0, 1.0], (L, 1))   # placeholder; set per user
    rab_cfg = RABConfig()
    if LOCAL_C_OVERRIDE is not None:
        rab_cfg.nu_c = float(LOCAL_C_OVERRIDE)
    dist_cfg = DistConfig(fronthaul_delay=fronthaul_delay)

    results = {m: np.zeros(K) for m in methods}
    if 'D-CF-RAB' in methods:
        results['D-CF-RAB_all'] = np.zeros(K)  # every snapshot (legacy schedule)
    timings = {m: 0.0 for m in methods}
    fh_bytes = 0
    Ax_hits = 0
    Ax_total = 0

    # --- Shared time-evolving channel sequence ---
    # We co-evolve a true channel sequence h_true_seq[s] and an estimated
    # channel sequence h_hat_seq[s] under AR(1) with correlation rho_t.
    # Both D-CF-RAB and the centralized methods see h_hat_seq[max(0, s-tau)] at snapshot
    # s; SINR is measured against the true channel at snapshot s and
    # averaged across all snapshots. This is what makes tau physically
    # meaningful: a stale h_hat is one whose underlying truth no longer
    # matches the current truth.
    rho_t = 0.995
    h_true_seq = [h_true.copy()]
    h_hat_seq = [h_hat.copy()]
    h_true_t = h_true.copy()
    h_hat_t = h_hat.copy()
    for s in range(1, n_snapshots):
        innovation_true = (rng.normal(size=h_true_t.shape) +
                           1j * rng.normal(size=h_true_t.shape)) / math.sqrt(2)
        for ell in range(L):
            for kk in range(K):
                target = norm(h_true_t[ell, kk]) * math.sqrt(1 - rho_t**2)
                innovation_true[ell, kk] *= target / math.sqrt(M)
        h_true_t = rho_t * h_true_t + innovation_true
        # h_hat at snapshot s = h_true at snapshot s + per-AP eps-radius CSI noise
        new_h_hat = np.empty_like(h_true_t)
        for ell in range(L):
            for kk in range(K):
                noise = (rng.normal(size=M) + 1j * rng.normal(size=M)) / math.sqrt(2)
                nrm = norm(noise)
                unit_err = noise / max(nrm, 1e-12)
                err_norm = _ball_radius(h_true_t[ell, kk], unit_err, eps_l[ell])
                new_h_hat[ell, kk] = h_true_t[ell, kk] + err_norm * unit_err
        h_hat_t = new_h_hat
        h_true_seq.append(h_true_t.copy())
        h_hat_seq.append(h_hat_t.copy())

    pre_hat = []
    if fronthaul_delay > 0:
        h_b = h_true.copy()
        for _j in range(fronthaul_delay):
            innov = (rng.normal(size=h_b.shape) + 1j * rng.normal(size=h_b.shape)) / math.sqrt(2)
            for ell in range(L):
                for kk in range(K):
                    innov[ell, kk] *= norm(h_b[ell, kk]) * math.sqrt(1 - rho_t ** 2) / math.sqrt(M)
            h_b = rho_t * h_b + innov
            est = np.empty_like(h_b)
            for ell in range(L):
                for kk in range(K):
                    nz = (rng.normal(size=M) + 1j * rng.normal(size=M)) / math.sqrt(2)
                    uu = nz / max(norm(nz), 1e-12)
                    est[ell, kk] = h_b[ell, kk] + _ball_radius(h_b[ell, kk], uu, eps_l[ell]) * uu
            pre_hat.append(est)

    def stale_hat(s):
        j = s - fronthaul_delay
        return h_hat_seq[j] if j >= 0 else pre_hat[-j - 1]

    def truth_at(s):
        return h_true_seq[s]

    if probe is not None:
        return probe(dict(truth_at=truth_at, stale_hat=stale_hat, R_l=R_l, sigma2=sigma2,
                          eps_l=eps_l, n_snapshots=n_snapshots, K=K, L=L, M=M))
    for k in range(K):
        # --- D-CF-RAB
        if "D-CF-RAB" in methods:
            t0 = time.perf_counter()
            w_l = [np.zeros(M, dtype=np.complex128) for _ in range(L)]
            rho_state = np.ones(L)
            sinr_acc = []; sinr_acc5 = []
            for s in range(n_snapshots):
                h_cpu_view = stale_hat(s)
                if cov_mode == 'tracking':
                    # population covariance of the CURRENT channels, same
                    # composition as R_l (desired user included when soi_in_cov)
                    _hs = truth_at(s)
                    _Rarg = []
                    for _l in range(L):
                        _R = sigma2 * np.eye(M, dtype=np.complex128)
                        for _kk in range(K):
                            if (not soi_in_cov) and _kk == k:
                                continue
                            _R = _R + P_USER * np.outer(_hs[_l, _kk], _hs[_l, _kk].conj())
                        _Rarg.append(_R)
                else:
                    _Rarg = (R_l if soi_in_cov else R_l_IN_per_k[k])
                _td = time.perf_counter()
                w_l, rho_state, info = cf_distributed_step(
                    _Rarg,
                    w_l, h_cpu_view, eps_l, nu_lk, rab_cfg, dist_cfg,
                    rho_state, k, sigma2=sigma2
                )
                _e = UPDATE_TIMING.setdefault('D-CF-RAB', [0.0, 0])
                _e[0] += time.perf_counter() - _td; _e[1] += 1
                fh_bytes += info["fronthaul_scalars"] * 8
                JOINT_MODES[info.get("joint_mode", "none")] = JOINT_MODES.get(info.get("joint_mode", "none"), 0) + 1
                JOINT_LOG.append((s, bool(s >= n_snapshots // 4 and (s - n_snapshots // 4) % 5 == 0), info.get("joint_mode", "none")))
                Ax_hits += info["Ax_size"]
                Ax_total += 1
                # Measure SINR at this snapshot vs CURRENT truth
                if s >= n_snapshots // 4:   # skip warm-up snapshots
                    H_true_s = np.stack([truth_at(s)[:, kk, :].reshape(-1)
                                         for kk in range(K)])
                    sinr_acc.append(sinr_from_weights(stack_w(w_l),
                                                     H_true_s[k], _ext(s, H_true_s),
                                                     sigma2, k))
                    if (s - n_snapshots // 4) % 5 == 0:
                        sinr_acc5.append(sinr_acc[-1])
                        _wf = stack_w(w_l); _hk = H_true_s[k]; _hh = np.concatenate([h_cpu_view[_l, k] for _l in range(L)])
                        RESPONSE_LOG.append((float(np.real(np.vdot(_wf, _hk))), float(np.real(np.vdot(_wf, _hh))), JOINT_LOG[-1][2] if JOINT_LOG else 'none'))
            timings["D-CF-RAB"] += time.perf_counter() - t0
            results["D-CF-RAB"][k] = float(np.mean(sinr_acc5))    # common schedule
            results["D-CF-RAB_all"][k] = float(np.mean(sinr_acc))

        # --- B2: CF-RAB. Resolve once on the latest stale view and
        # then *re-measure* per snapshot against the evolving truth. (Re-solving
        # per snapshot was 10x too slow for the full battery; in real systems
        # the centralized solve happens at the channel-coherence rate, not
        # every snapshot.)
        if "CF-RAB" in methods:
            t0 = time.perf_counter()
            if stale_centralized:
                h_view = h_hat_seq[max(0, n_snapshots - 1 - fronthaul_delay)]
            else:
                h_view = h_hat_seq[-1]
            h_stack_view = h_view[:, k, :].reshape(-1)
            eps_c = float(np.mean(eps_l))
            nu_joint = np.full(nu_lk.shape[1], eps_c * norm(h_stack_view))
            w_full = beamformer_cf_rab(R_stack_per_k[k], h_stack_view,
                                                 nu_joint, rab_cfg,
                                                 eps_total=eps_c)
            timings["CF-RAB"] += time.perf_counter() - t0
            def _mk_cf(hv):
                hs = hv[:, k, :].reshape(-1)
                nj = np.full(nu_lk.shape[1], float(np.mean(eps_l)) * norm(hs))
                return beamformer_cf_rab(R_stack_per_k[k], hs, nj, rab_cfg,
                                         eps_total=float(np.mean(eps_l)))
            results["CF-RAB"][k] = _eval_per_snapshot_t("CF-RAB", 
                _mk_cf, n_snapshots, truth_at, K, k, sigma2, stale_hat)

        # --- B2b: CF-RAB with the angular grid of Chapter 3 (same window/CSI as CF-RAB)
        if "CF-RAB-ang" in methods:
            t0 = time.perf_counter()
            if stale_centralized:
                h_view = h_hat_seq[max(0, n_snapshots - 1 - fronthaul_delay)]
            else:
                h_view = h_hat_seq[-1]
            h_stack_view = h_view[:, k, :].reshape(-1)
            nu_joint = np.full(nu_lk.shape[1],
                               float(np.mean(eps_l)) * norm(h_stack_view))
            w_full = beamformer_cf_rab_angular(R_stack_per_k[k], h_stack_view,
                                               nu_joint, rab_cfg, L, M)
            timings["CF-RAB-ang"] += time.perf_counter() - t0
            _rng = ([n_snapshots - 1] if eval_at_last
                    else list(range(n_snapshots // 4, n_snapshots)))
            sinr_acc = []
            for s in _rng:
                H_true_s = np.stack([truth_at(s)[:, kk, :].reshape(-1)
                                     for kk in range(K)])
                sinr_acc.append(sinr_from_weights(w_full, H_true_s[k],
                                                 _ext(s, H_true_s), sigma2, k))
            results["CF-RAB-ang"][k] = float(np.mean(sinr_acc))

        # --- B3: local MMSE (solve once on latest CSI, measure vs evolving truth)
        if 'LMVDR+W' in methods:
            # unconstrained loaded MVDR per AP on the same covariance D-CF-RAB uses,
            # followed by the same CPU combiner: isolates the worst-case constraint
            _Rsrc = R_l if soi_in_cov else R_l_IN_per_k[k]
            def _mkmv(hv, _R=_Rsrc):
                Lh, Kh, Mh = hv.shape
                blocks = []
                for l in range(Lh):
                    g = LMVDR_LOAD * np.trace(_R[l]).real / Mh
                    blocks.append(solve(_R[l] + g * np.eye(Mh), hv[l, k]))
                return cpu_wiener_combine(blocks, hv, k, sigma2, eta=CPU_ETA.get('LMVDR+W', 0.0), eps_l=eps_l)
            results['LMVDR+W'][k] = _eval_per_snapshot_t('LMVDR+W', 
                _mkmv, n_snapshots, truth_at, K, k, sigma2, stale_hat)
        for _wname in ('LP-MMSE+W', 'MR+W'):
            if _wname not in methods:
                continue
            def _mkw(hv, _n=_wname):
                Lh, Kh, Mh = hv.shape
                if _n == 'MR+W':
                    blocks = [hv[l, k] for l in range(Lh)]
                else:
                    v = beamformer_lp_mmse(hv, eps_l, k, sigma2)
                    blocks = [v[l * Mh:(l + 1) * Mh] for l in range(Lh)]
                return cpu_wiener_combine(blocks, hv, k, sigma2, eta=CPU_ETA.get(_n, 0.0), eps_l=eps_l)
            results[_wname][k] = _eval_per_snapshot_t(_wname, 
                _mkw, n_snapshots, truth_at, K, k, sigma2, stale_hat)
        for _bjname, _bjfn in (("MR", 0), ("LP-MMSE", 1), ("P-MMSE", 2)):
            if _bjname not in methods:
                continue
            t0 = time.perf_counter()
            if stale_centralized:
                h_view = h_hat_seq[max(0, n_snapshots - 1 - fronthaul_delay)]
            else:
                h_view = h_hat_seq[-1]
            if _bjfn == 0:
                w_full = beamformer_mr(h_view, k)
            elif _bjfn == 1:
                w_full = beamformer_lp_mmse(h_view, eps_l, k, sigma2)
            else:
                w_full = beamformer_p_mmse(h_view, eps_l, k, sigma2)
            timings[_bjname] += time.perf_counter() - t0
            def _mk(hv, _f=_bjfn):
                if _f == 0:
                    return beamformer_mr(hv, k)
                if _f == 1:
                    return beamformer_lp_mmse(hv, eps_l, k, sigma2)
                return beamformer_p_mmse(hv, eps_l, k, sigma2)
            results[_bjname][k] = _eval_per_snapshot_t(_bjname, 
                _mk, n_snapshots, truth_at, K, k, sigma2, stale_hat)

        if 'MR+ACPU' in methods:
            # MR at each AP, then a CPU combiner computed from the covariance of the local outputs
            _Rj = R_stack_per_k[k]
            def _mkacpu(hv, _R=_Rj):
                Lh, Kh, Mh = hv.shape
                B = np.zeros((Lh * Mh, Lh), dtype=np.complex128)
                for l in range(Lh):
                    B[l * Mh:(l + 1) * Mh, l] = hv[l, k]
                Ry = B.conj().T @ _R @ B
                gk = np.array([np.vdot(hv[l, k], hv[l, k]) for l in range(Lh)])
                c = solve(Ry + ACPU_LOADING * np.trace(Ry).real / Lh * np.eye(Lh), gk)
                return np.concatenate([c[l] * hv[l, k] for l in range(Lh)])
            results['MR+ACPU'][k] = _eval_per_snapshot_t('MR+ACPU', _mkacpu, n_snapshots, truth_at, K, k, sigma2, stale_hat)
        if 'RECON+W' in methods:
            _Rr = R_l if soi_in_cov else R_l_IN_per_k[k]
            def _mkrw(hv, _R=_Rr):
                Lh, Kh, Mh = hv.shape
                v = beamformer_recon_rab(_R, hv, k)
                return cpu_wiener_combine([v[l * Mh:(l + 1) * Mh] for l in range(Lh)], hv, k, sigma2,
                                          eta=CPU_ETA.get('RECON+W', 0.0), eps_l=eps_l)
            results['RECON+W'][k] = _eval_per_snapshot_t('RECON+W', _mkrw, n_snapshots, truth_at, K, k, sigma2, stale_hat)
        if "RECON-RAB" in methods:
            t0 = time.perf_counter()
            if stale_centralized:
                h_view = h_hat_seq[max(0, n_snapshots - 1 - fronthaul_delay)]
            else:
                h_view = h_hat_seq[-1]
            Rsel = R_l if soi_in_cov else R_l_IN_per_k[k]
            w_full = beamformer_recon_rab(Rsel, h_view, k)
            timings["RECON-RAB"] += time.perf_counter() - t0
            results["RECON-RAB"][k] = _eval_per_snapshot_t("RECON-RAB", 
                lambda hv: beamformer_recon_rab(Rsel, hv, k),
                n_snapshots, truth_at, K, k, sigma2, stale_hat)

        if "local_mmse" in methods:
            t0 = time.perf_counter()
            if stale_centralized:
                h_view = h_hat_seq[max(0, n_snapshots - 1 - fronthaul_delay)]
            else:
                h_view = h_hat_seq[-1]
            w_l = beamformer_local_mmse(R_l_IN_per_k[k], h_view[:, k, :])
            timings["local_mmse"] += time.perf_counter() - t0
            sinr_acc = []
            _rng = ([n_snapshots - 1] if eval_at_last
                    else list(range(n_snapshots // 4, n_snapshots)))
            for s in _rng:
                H_true_s = np.stack([truth_at(s)[:, kk, :].reshape(-1)
                                     for kk in range(K)])
                sinr_acc.append(sinr_from_weights(stack_w(w_l), H_true_s[k],
                                                 _ext(s, H_true_s), sigma2, k))
            results["local_mmse"][k] = float(np.mean(sinr_acc))

        # --- B4: local RAB (no consensus) -- solve once, measure per snapshot
        if "local_RAB_only" in methods:
            t0 = time.perf_counter()
            h_view = h_hat_seq[-1]
            w_l = []
            for ell in range(L):
                l_pres = build_constraint_grid(h_view[ell, k], rab_cfg.nu_c,
                                               N=nu_lk.shape[1])
                nu_local = np.full(nu_lk.shape[1],
                                   rab_cfg.nu_c * norm(h_view[ell, k]))
                w_new, *_ = rab_step((R_l if soi_in_cov else R_l_IN_per_k[k])[ell],
                                          np.zeros(M, dtype=np.complex128),
                                          l_pres, nu_local, rab_cfg, 1.0)
                w_l.append(w_new)
            timings["local_RAB_only"] += time.perf_counter() - t0
            sinr_acc = []
            _rng = ([n_snapshots - 1] if eval_at_last
                    else list(range(n_snapshots // 4, n_snapshots)))
            for s in _rng:
                H_true_s = np.stack([truth_at(s)[:, kk, :].reshape(-1)
                                     for kk in range(K)])
                sinr_acc.append(sinr_from_weights(stack_w(w_l), H_true_s[k],
                                                 _ext(s, H_true_s), sigma2, k))
            results["local_RAB_only"][k] = float(np.mean(sinr_acc))

        # --- B1: centralized SDP gold standard (only at small L*M)
        if "centralized_sdp" in methods and HAVE_CVXPY and L * M <= 320:
            t0 = time.perf_counter()
            # SDP is expensive: solve once on the final snapshot's CSI and
            # measure on the corresponding truth.
            if stale_centralized:
                h_view = h_hat_seq[max(0, n_snapshots - 1 - fronthaul_delay)]
            else:
                h_view = h_hat_seq[-1]
            h_stack_view = h_view[:, k, :].reshape(-1)
            # eq. (6.2) defines RELATIVE uncertainty ||dh|| <= eps*||h_hat||, so the
            # absolute SOCP bound must be scaled by the presumed-channel norm.
            w_full = beamformer_centralized_sdp(
                R_stack_per_k[k], h_stack_view,
                eps_total=float(np.mean(eps_l)) * norm(h_stack_view))
            timings["centralized_sdp"] += time.perf_counter() - t0
            def _mk_so(hv):
                hs = hv[:, k, :].reshape(-1)
                return beamformer_centralized_sdp(
                    R_stack_per_k[k], hs,
                    eps_total=float(np.mean(eps_l)) * norm(hs))
            results["centralized_sdp"][k] = _eval_per_snapshot_t("centralized_sdp", 
                _mk_so, n_snapshots, truth_at, K, k, sigma2, stale_hat)

    sinr_dB = {m: 10 * np.log10(np.clip(v, 1e-12, None)) for m, v in results.items()}
    return {
        "sinr_dB": sinr_dB,
        "timings_s": timings,
        "update_ms": {k_: 1000.0 * v_[0] / max(v_[1], 1) for k_, v_ in UPDATE_TIMING.items()},
        "update_calls": {k_: v_[1] for k_, v_ in UPDATE_TIMING.items()},
        "fh_bytes_total": fh_bytes,
        "Ax_hit_rate": (Ax_hits / max(Ax_total, 1)) if Ax_total else 0.0,
        "config": cfg,
    }


# =============================================================================
# 6.  Sweep drivers
# =============================================================================

def sweep_L(L_list=(4, 8, 16, 32), n_snapshots=60, n_trials=3, seed_base=100):
    """Sweep over number of APs L."""
    rows = []
    for L in L_list:
        for trial in range(n_trials):
            cfg = CFConfig(L=L, M_l=8, K=4)
            res = run_snapshot_trace(
                cfg, n_snapshots=n_snapshots,
                methods=("D-CF-RAB", "CF-RAB", "local_mmse", "local_RAB_only",
                         "centralized_sdp" if L * 8 <= 96 else None),
                seed=seed_base + trial,
            )
            rows.append((L, trial, res))
    return rows


def sweep_fronthaul_delay(tau_list=(0, 1, 4, 16), n_snapshots=60, seed_base=200):
    rows = []
    for tau in tau_list:
        cfg = CFConfig(L=16, M_l=8, K=4)
        res = run_snapshot_trace(
            cfg, n_snapshots=n_snapshots, fronthaul_delay=tau,
            methods=("D-CF-RAB", "CF-RAB", "local_mmse"),
            seed=seed_base + tau,
        )
        rows.append((tau, res))
    return rows


def sweep_heterogeneity(modes=("uniform", "tiered"), n_snapshots=60, seed_base=300):
    rows = []
    for mode in modes:
        cfg = CFConfig(L=16, M_l=8, K=4)
        res = run_snapshot_trace(
            cfg, n_snapshots=n_snapshots,
            heterogeneous=(mode == "tiered"),
            methods=("D-CF-RAB", "CF-RAB", "local_mmse"),
            seed=seed_base + (0 if mode == "uniform" else 1),
        )
        rows.append((mode, res))
    return rows


if __name__ == "__main__":
    print("Sanity check: small run on L=4, M_l=4, K=2")
    cfg = CFConfig(L=4, M_l=4, K=2)
    res = run_snapshot_trace(cfg, n_snapshots=30,
                             methods=("D-CF-RAB", "CF-RAB",
                                      "local_mmse", "local_RAB_only",
                                      "centralized_sdp"),
                             seed=0)
    for m, sinr in res["sinr_dB"].items():
        print(f"  {m:24s}: mean SINR = {np.nanmean(sinr):6.2f} dB   "
              f"5th-%ile = {np.nanpercentile(sinr, 5):6.2f} dB")
    print(f"  fronthaul_bytes = {res['fh_bytes_total']}")
    print(f"  Ax_hit_rate = {res['Ax_hit_rate']:.3f}")
    for m, t in res["timings_s"].items():
        print(f"  time[{m}] = {t*1000:.1f} ms")
