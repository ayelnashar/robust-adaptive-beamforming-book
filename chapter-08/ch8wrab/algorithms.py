"""
ch8wrab.algorithms -- numerical core of Chapter 8 (WRAB, the wideband robust adaptive beamformer).

Contents
  * ULA model at sub-terahertz frequencies: subcarrier frequencies (8.1), frequency-dependent steering vectors,
    carrier-fixed analog response (beam squint), 3-dB beamwidth.
  * Scenario: every setting of Table 8.1 with its default value.
  * generate_trial: one block-static trial (desired-direction jitter, optional multipath, interference-only or
    signal-present training snapshots), with separate random streams for geometry, noise and symbols.
  * WRAB (Algorithm 8.1): exponentially weighted covariance (8.10), scale-invariant gradient step (8.11)-(8.13),
    exact Euclidean projection onto the intersection of the worst-case constraints (8.14) through an active-set
    Newton solve of its dual, with Dykstra's method on the same sets as a checked fallback, and the covariance-anchored
    per-constraint margins (8.18)-(8.20).
  * Comparators of Section 8.4: analog-only, WC-single (carrier-referenced and frequency-aware), frequency-aware MVDR,
    robust Capon [24], worst-case SOCP [26] (spherical KKT root), simplified covariance reconstruction [30],
    fixed-margin multi-WC, exact multi-ball SOCP of (8.8) (CVXPY with CLARABEL) and the loaded multi-point LCMV [22].
  * Scoring: per-subcarrier output SINR (8.5), trial-level bootstrap intervals.
  * The frozen-norm one-step correction discussed after (8.14), used only for the feasibility comparison of
    Section 8.3.1, and the dynamic AR(1) array-rotation model of Section 8.2.3 (Figure 8.8(b)).

Every random number comes from seed_for(label, trial), so each experiment has its own stream and every result in
the chapter regenerates exactly.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import eigh, solve
from scipy.optimize import brentq, minimize

try:                                   # optional acceleration of the projection kernels
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:                    # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(fun):
            return fun
        return decorator

try:
    from threadpoolctl import threadpool_limits
    _BLAS_LIMIT = threadpool_limits(limits=1)   # single-thread BLAS, as for the timings quoted in the chapter
except ImportError:                    # pragma: no cover
    _BLAS_LIMIT = None

C0 = 299_792_458.0
BASE_SEED = 42

# method key -> legend name used in the chapter
METHOD_LABELS = {
    'analog': 'Analog-only',
    'wc': 'WC-single (carrier-referenced)',
    'mvdr': 'MVDR (frequency-aware)',
    'rcb': 'Robust Capon [24]',
    'socp': 'Worst-case SOCP [26]',
    'recon': 'Cov. reconstruction [30], simplified',
    'fixed': 'Fixed-margin multi-WC',
    'wrab': 'WRAB',
    'wc_faware': 'WC-single (frequency-aware)',
    'msocp': 'Exact multi-ball SOCP of (8.8)',
    'lcmv': 'Multi-point LCMV [22], loaded',
}
CORE_METHODS = ('analog', 'wc', 'wrab')
ALL_METHODS = ('analog', 'wc', 'mvdr', 'rcb', 'socp', 'recon', 'fixed', 'wrab')


def seed_for(label: str, trial: int = 0, base_seed: int | None = None) -> int:
    """Seed of one trial of one experiment; stable across sessions and independent of execution order."""
    if base_seed is None:
        base_seed = BASE_SEED
    digest = hashlib.sha256(f'{base_seed}|{label}|{trial}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little')


# ------------------------------------------------------------------------------------------------ array model
def subcarrier_frequencies(fc: float, B: float, Nsc: int) -> np.ndarray:
    """Centres of Nsc equal bins across the band B around fc, (8.1); the samples do not reach fc +/- B/2."""
    if not (np.isfinite(fc) and np.isfinite(B) and fc > 0 and 0 <= B < 2*fc):
        raise ValueError('Require fc>0 and 0<=B<2*fc.')
    if not isinstance(Nsc, (int, np.integer)) or Nsc < 1:
        raise ValueError('Nsc must be a positive integer.')
    return fc + (np.arange(Nsc) - (Nsc-1)/2)*(B/Nsc)


def steer_f(M, th, f, fc, d_over_lambda_c=0.5):
    """ULA steering vector a[m] = exp(+j 2 pi (f/fc) (d/lambda_c) m sin(th)), spacing fixed at the carrier."""
    if M < 1 or int(M) != M or f <= 0 or fc <= 0:
        raise ValueError('Invalid array size or frequency.')
    return np.exp(1j*2*np.pi*(f/fc)*d_over_lambda_c*np.arange(M)*np.sin(th))


def steering_matrix(M, theta, f, fc):
    """Steering vectors of the angles theta (radians) at frequency f, one per column."""
    return np.exp(1j*np.pi*(f/fc)*np.arange(M)[:, None]*np.sin(np.asarray(theta)[None, :]))


def analog_response(M, th_steer, th_look, f, fc, d_over_lambda_c=0.5):
    """Normalized array-factor magnitude of carrier-fixed phase weights steered to th_steer, seen at (th_look, f)."""
    a0 = steer_f(M, th_steer, fc, fc, d_over_lambda_c)
    at = steer_f(M, th_look, f, fc, d_over_lambda_c)
    return float(abs(np.vdot(a0, at))/M)


def beamwidth_rad(M, presumed_angle):
    """Approximate full 3-dB beamwidth 1.772/(M cos theta_p) at the presumed angle (8.15)."""
    if abs(presumed_angle) >= np.deg2rad(89):
        raise ValueError('The beamwidth approximation is not used at endfire.')
    return 1.772/(M*np.cos(presumed_angle))


# ------------------------------------------------------------------------------------------------ scenario
@dataclass(frozen=True)
class Scenario:
    """Settings of one experiment. The defaults are the principal settings of Table 8.1."""
    M: int = 64                          # number of elements (N in the chapter)
    fc: float = 300e9                    # carrier (Hz)
    B: float = 20e9                      # bandwidth (Hz)
    Nsc: int = 8                         # processed subcarriers
    K: int = 300                         # snapshots per run
    th_d_deg: float = 0.0                # nominal desired direction
    mismatch_deg: float = 2.0            # presumed minus nominal true direction
    interferers_deg: tuple = (45.0, -60.0)
    snr_db: float = 5.0                  # per element
    inr_db: float = 10.0                 # per element
    jitter_deg: float = 0.1              # RMS of the block-static desired-direction offset
    signal_free: bool = True             # interference-only training snapshots
    jitter_mode: str = 'desired_direction'   # or 'array_boresight' (the interferers move with the offset)
    n_paths: int = 1
    angular_spread_deg: float = 0.0
    eta: float = 0.97                    # forgetting factor (8.10)
    grid_fracs: tuple = (0.0, -0.6, 0.6, -1.2)   # constraint offsets in beamwidths (8.15)
    grid_offsets_deg: tuple | None = None        # fixed offsets in degrees (replace grid_fracs when given)
    initial_margins: tuple = (1.9, 1.5, 1.5, 1.1)
    margin_scale: float = 1.5            # anchor scale (8.18)
    margin_ceiling: float = 3.0
    subspace_fraction: float = 0.25      # retained eigenspace order ceil(0.25 M)
    warmup: int = 50                     # snapshots before the first margin update
    margin_period: int = 30              # margin update period T
    anchor_refresh: str = 'once'         # anchor computed once at the first margin update
    relaxation: float = 0.1              # step relaxation (8.13)
    single_radius_fraction: float = 0.6  # single-ball radius / sqrt(M)
    batch_period: int = 25               # rebuild period of the batch comparators
    covariance_loading: float = 1e-6     # relative diagonal load
    projection_tolerance: float = 1e-9
    projection_max_cycles: int = 500
    score_weights: str = 'last'          # score the final weight vector
    aggregation: str = 'mean_db'         # trial metric: mean over subcarriers of the SINR in dB
    squint_aware: bool = True            # constraint steering vectors at each subcarrier frequency
    self_anchor: bool = True             # margin adaptation (8.18)-(8.20) on/off

    def validate(self):
        subcarrier_frequencies(self.fc, self.B, self.Nsc)
        if self.M < 2 or self.K < 1 or self.M != int(self.M) or self.K != int(self.K):
            raise ValueError('M>=2 and K>=1 must be integers.')
        if not 0 < self.eta < 1 or self.relaxation <= 0 or self.relaxation > 1:
            raise ValueError('Require 0<eta<1 and 0<relaxation<=1.')
        if not 0 <= self.single_radius_fraction < 1:
            raise ValueError('Spherical radius must be strictly below sqrt(M).')
        if self.jitter_deg < 0 or self.n_paths < 1 or self.angular_spread_deg < 0:
            raise ValueError('Invalid channel parameters.')
        if self.jitter_mode not in ('desired_direction', 'array_boresight'):
            raise ValueError('Unknown jitter mode.')
        if self.anchor_refresh not in ('once', 'periodic'):
            raise ValueError('anchor_refresh must be once or periodic.')
        if self.margin_period < 1 or self.batch_period < 1 or not 0 < self.subspace_fraction < 1:
            raise ValueError('Invalid update schedule or retained subspace fraction.')
        if self.score_weights not in ('last', 'tail_mean'):
            raise ValueError('Unknown scoring weight rule.')
        if self.aggregation not in ('mean_db', 'linear'):
            raise ValueError('Unknown frequency aggregation.')
        if any(abs(t) >= 89.9 for t in (*self.interferers_deg, self.th_d_deg+self.mismatch_deg)):
            raise ValueError('Angles must be inside the visible ULA sector.')
        if not (0 <= self.margin_ceiling < np.sqrt(self.M)):
            raise ValueError('Margin ceiling must be smaller than each steering norm.')
        if not np.isfinite(self.projection_tolerance) or self.projection_tolerance <= 0:
            raise ValueError('Invalid projection tolerance.')
        return self


def design_grid(cfg: Scenario):
    """Presumed direction, constraint directions (8.15) and initial margins."""
    thp = np.deg2rad(cfg.th_d_deg + cfg.mismatch_deg)
    if cfg.grid_offsets_deg is None:
        offsets = np.asarray(cfg.grid_fracs, float)*beamwidth_rad(cfg.M, thp)
    else:
        offsets = np.deg2rad(np.asarray(cfg.grid_offsets_deg, float))
    if offsets.size == 0:
        raise ValueError('At least one constraint is required.')
    grid = thp + offsets
    if np.any(np.abs(grid) >= np.pi/2):
        raise ValueError('Constraint grid leaves the visible angular sector.')
    y = np.asarray(cfg.initial_margins, float).reshape(-1)
    if y.size == 1:
        y = np.repeat(y, grid.size)
    if y.size != grid.size or np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError('Supply one nonnegative initial margin per constraint, or a scalar tuple.')
    if np.any(y >= np.sqrt(cfg.M)):
        raise ValueError('An individual worst-case constraint is infeasible.')
    return thp, grid, y.copy()


def cn(rng, shape):
    """Circular complex Gaussian samples of unit variance."""
    return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)


def generate_trial(cfg: Scenario, seed: int):
    """One block-static channel and the training data shared by all methods.

    The desired-path parameters, the interference and noise, and the desired symbols use separate random streams.
    With signal_free=True no desired signal and no true desired direction reach a weight update; the jitter and the
    multipath enter the scored channel. All subcarriers share one jitter value and one normalized set of path gains.
    """
    cfg.validate()
    ss = np.random.SeedSequence(seed).spawn(3)
    geo, noise, symbols = (np.random.default_rng(s) for s in ss)
    jitter = float(geo.normal(0, np.deg2rad(cfg.jitter_deg)))
    thd = np.deg2rad(cfg.th_d_deg)+jitter
    ti = np.deg2rad(cfg.interferers_deg)
    if cfg.jitter_mode == 'array_boresight':
        ti = ti + jitter
    deviations = np.zeros(cfg.n_paths)
    gains = np.ones(cfg.n_paths, dtype=complex)
    if cfg.n_paths > 1:
        deviations[1:] = np.deg2rad(geo.uniform(-cfg.angular_spread_deg, cfg.angular_spread_deg, cfg.n_paths-1))
        magnitudes = np.r_[1.0, 0.6*geo.random(cfg.n_paths-1)]
        gains = magnitudes*np.exp(1j*geo.uniform(0, 2*np.pi, cfg.n_paths))
        gains /= np.linalg.norm(gains)            # unit total path power
    fl = subcarrier_frequencies(cfg.fc, cfg.B, cfg.Nsc)
    X, channels, ripn = [], [], []
    for f in fl:
        Ai = steering_matrix(cfg.M, ti, f, cfg.fc)
        H = steering_matrix(cfg.M, thd+deviations, f, cfg.fc) @ gains
        U = cn(noise, (len(ti), cfg.K))*10**(cfg.inr_db/20)
        xn = Ai@U + cn(noise, (cfg.M, cfg.K))
        if not cfg.signal_free:
            xn += H[:, None]*cn(symbols, (1, cfg.K))*10**(cfg.snr_db/20)
        Rn = np.eye(cfg.M, dtype=complex)+10**(cfg.inr_db/10)*(Ai@Ai.conj().T)
        X.append(np.ascontiguousarray(xn)); channels.append(H); ripn.append(Rn)
    return {'X': X, 'frequencies': fl, 'channel': channels, 'R_ipn': ripn,
            'jitter_rad': jitter, 'true_direction_rad': thd,
            'interferers_rad': ti, 'path_offsets_rad': deviations, 'path_gains': gains}


# ------------------------------------------------------------------------------------------------ projection (8.14)
@njit(cache=False, nogil=True)
def _single_projection(z, a, nu):
    """Exact Euclidean projection onto one worst-case set {w: Re(a^H w) >= nu ||w|| + 1}.

    In the real embedding this is prox_{lambda nu ||.||}(z + lambda a) with the scalar multiplier lambda >= 0 chosen
    by bisection so that the constraint is active.
    """
    t = np.real(np.vdot(a, z)); a2 = np.real(np.vdot(a, a))
    z2 = np.real(np.vdot(z, z)); an = np.sqrt(a2)
    if t-nu*np.sqrt(z2) >= 1.0:
        return z.copy()
    if nu == 0.0:
        return z+((1.0-t)/a2)*a
    hi = max(1e-12, (1.0-t+nu*np.sqrt(z2))/((an-nu)**2))
    for _ in range(100):
        vnorm = np.sqrt(max(z2+2*hi*t+hi*hi*a2, 0.0))
        shrink = max(0.0, 1-hi*nu/max(vnorm, 1e-300))
        response = shrink*(t+hi*a2-nu*vnorm)
        if response >= 1:
            break
        hi *= 2
    lo = 0.0
    for _ in range(65):
        mid = (lo+hi)/2
        vnorm = np.sqrt(max(z2+2*mid*t+mid*mid*a2, 0.0))
        shrink = max(0.0, 1-mid*nu/max(vnorm, 1e-300))
        response = shrink*(t+mid*a2-nu*vnorm)
        if response >= 1.0:
            hi = mid
        else:
            lo = mid
    v = z+hi*a
    return max(0.0, 1-hi*nu/max(np.linalg.norm(v), 1e-300))*v


@njit(cache=False, nogil=True)
def _dykstra(z, L, nu, tol, max_cycles):
    """Dykstra's method for the projection onto the intersection of the worst-case sets."""
    M, nc = L.shape
    x = z.copy()
    corr = np.zeros((nc, M), dtype=np.complex128)
    residual = 0.0
    change = np.inf
    for cycle in range(max_cycles):
        prev = x.copy()
        for i in range(nc):
            a = np.ascontiguousarray(L[:, i])
            v = x+corr[i]
            nxt = _single_projection(v, a, nu[i])
            corr[i] = v-nxt
            x = nxt
        residual = -np.inf
        xn = np.linalg.norm(x)
        for i in range(nc):
            val = 1+nu[i]*xn-np.real(np.vdot(np.ascontiguousarray(L[:, i]), x))
            residual = max(residual, val)
        change = np.linalg.norm(x-prev)
        if residual <= tol and change <= tol*(1+xn):
            return x, cycle+1, residual, True
    return x, max_cycles, residual, False


@njit(cache=False, nogil=True)
def _dual_eval(tau, c, G, z2, nu):
    h = c+G@tau
    s = np.sqrt(max(z2+2*np.dot(c, tau)+np.dot(tau, G@tau), 0.0))
    beta = np.dot(nu, tau)
    q = max(s-beta, 0.0)
    if s <= 1e-300 or q <= 0:
        return -np.sum(tau), -np.ones(len(tau)), np.eye(len(tau))*1e-8, s, q
    d = h/s-nu
    grad = q*d-1.0
    H = np.outer(d, d)+q*(G/s-np.outer(h, h)/(s*s*s))
    return 0.5*q*q-np.sum(tau), grad, H, s, q


@njit(cache=False, nogil=True)
def _dual_projection(z, L, nu, tol, max_iter=100):
    """Active-set Newton solve of the dual of the projection (8.14).

    With b = nu^T tau and v = z + L tau, the primal minimizer is w = max(1 - b/||v||, 0) v, and the dual minimizes
    0.5 max(||z + L tau|| - nu^T tau, 0)^2 - 1^T tau over tau >= 0, a problem in N_c real variables with closed-form
    gradient and Hessian. Bounds are enforced by re-solving on the free variables and a backtracking line search;
    the KKT residual is checked explicitly. A ridge of 1e-12 max(1, max|H|) keeps the Newton system regular.
    """
    nc = L.shape[1]
    G = np.ascontiguousarray(np.real(L.conj().T@L))
    c = np.ascontiguousarray(np.real(L.conj().T@z))
    z2 = np.real(np.vdot(z, z))
    tau = np.zeros(nc)
    for iteration in range(max_iter):
        obj, grad, H, s, q = _dual_eval(tau, c, G, z2, nu)
        kkt = 0.0
        for i in range(nc):
            err = abs(grad[i]) if tau[i] > 1e-13 else max(-grad[i], 0.0)
            kkt = max(kkt, err)
        if kkt <= tol and q > 0:
            v = z+L@tau.astype(np.complex128)
            return (q/max(s, 1e-300))*v, iteration, True
        free = (tau > 1e-13) | (grad < -tol)
        p = np.zeros(nc)
        for _ in range(nc+1):
            ix = np.nonzero(free)[0]
            if len(ix) == 0:
                break
            HH = np.zeros((len(ix), len(ix)))
            gg = np.zeros(len(ix))
            for i in range(len(ix)):
                gg[i] = grad[ix[i]]
                for j in range(len(ix)):
                    HH[i, j] = H[ix[i], ix[j]]
            ridge = 1e-12*max(1.0, np.max(np.abs(HH)))
            for i in range(len(ix)):
                HH[i, i] += ridge
            pp = np.linalg.solve(HH, -gg)
            p[:] = 0
            for i in range(len(ix)):
                p[ix[i]] = pp[i]
            bad = (tau <= 1e-13) & (p < 0) & free
            if not np.any(bad):
                break
            free[bad] = False
        slope = np.dot(grad, p)
        if slope >= -1e-25:
            for i in range(nc):
                p[i] = -grad[i] if tau[i] > 1e-13 else max(-grad[i], 0.0)
            slope = np.dot(grad, p)
        step = 1.0
        for i in range(nc):
            if p[i] < 0:
                step = min(step, -tau[i]/p[i])
        accepted = False
        for _ in range(50):
            candidate = np.maximum(tau+step*p, 0.0)
            ob2, _, _, _, _ = _dual_eval(candidate, c, G, z2, nu)
            if ob2 <= obj+1e-4*step*slope+1e-15*max(1.0, abs(obj)):
                tau = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted or step < 1e-16:
            break
    v = z+L@tau.astype(np.complex128)
    vn = np.linalg.norm(v)
    w = max(1-np.dot(nu, tau)/max(vn, 1e-300), 0.0)*v
    return w, max_iter, False


def wc_residual(w, L, nu):
    """Residual of the worst-case constraints of (8.8): positive entries are violations."""
    return 1+np.asarray(nu)*np.linalg.norm(w)-np.real(L.conj().T@w)


def project_wc(z, L, nu, tol=1e-9, max_cycles=500):
    """Euclidean projection (8.14) of z onto {w: Re(L_i^H w) >= nu_i ||w|| + 1 for every column i}.

    Returns the projected vector and the number of iterations. The dual Newton solve is tried first; if it fails its
    checks, Dykstra's method on the same sets is used, and an error is raised if that also fails.
    """
    z = np.asarray(z, complex); L = np.asarray(L, complex); nu = np.asarray(nu, float)
    if L.shape[0] != z.size or L.shape[1] != nu.size:
        raise ValueError('Incompatible projection dimensions.')
    if np.any(nu < 0) or np.any(nu >= np.linalg.norm(L, axis=0)):
        raise ValueError('Invalid or infeasible individual uncertainty radius.')
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(L)) and np.all(np.isfinite(nu))):
        raise FloatingPointError('Non-finite projection input.')
    if np.max(wc_residual(z, L, nu)) <= tol:
        return z.copy(), 0
    w, cycles, ok = _dual_projection(np.ascontiguousarray(z), np.ascontiguousarray(L), np.ascontiguousarray(nu), tol)
    residual = float(np.max(wc_residual(w, L, nu)))
    if not ok or residual > tol*1.01:
        w, cycles, residual, ok = _dykstra(np.ascontiguousarray(z), np.ascontiguousarray(L),
                                         np.ascontiguousarray(nu), tol, max_cycles)
    if not ok or not np.all(np.isfinite(w)):
        raise RuntimeError(f'WC projection did not converge: cycles={cycles}, residual={residual:.3g}. '
                           'Check joint feasibility of the constraints.')
    return w, int(cycles)


def projected_step(R, w, L, nu, cfg):
    """One WRAB update (8.11)-(8.14): scale-invariant gradient step on w^H R w, then the exact projection."""
    loading = cfg.covariance_loading*float(np.real(np.trace(R)))/len(w)
    g = R@w+loading*w
    gg = float(np.real(np.vdot(g, g)))
    curvature = float(np.real(np.vdot(g, R@g+loading*g)))
    if gg <= np.finfo(float).tiny:
        trial = w.copy()
    else:
        if curvature <= 0 or not np.isfinite(curvature):
            raise FloatingPointError('Nonpositive or nonfinite power-gradient curvature.')
        trial = w-(cfg.relaxation*gg/curvature)*g
    return project_wc(trial, L, nu, cfg.projection_tolerance, cfg.projection_max_cycles)


def margin_update(R, L, y, anchor, cfg):
    """Covariance-anchored margins (8.18)-(8.20).

    At the first call the anchor (8.18) uses the energy of each constraint steering vector outside the leading
    ceil(0.25 M) eigenvectors, scaled to mean margin_scale (mean-residual floor 1e-6), and the margins are reset to it.
    Every call then moves each margin toward its anchor according to the alignment of the dominant eigenvector with
    that constraint (8.19, alignment floor 0.05) and applies the common broadening factor (8.20), with the ceiling.
    """
    M = R.shape[0]
    need_anchor = anchor is None or cfg.anchor_refresh == 'periodic'
    rank = max(1, int(np.ceil(cfg.subspace_fraction*M))) if need_anchor else 1
    _, U = eigh((R+R.conj().T)/2, subset_by_index=(M-rank, M-1), check_finite=False)
    if need_anchor:
        beta = (np.sum(abs(L)**2, axis=0)-np.sum(abs(U.conj().T@L)**2, axis=0))/M
        beta = np.maximum(beta, 0)
        anchor = cfg.margin_scale*beta/max(float(beta.mean()), 1e-6)
        if cfg.anchor_refresh == 'once':
            y = anchor.copy()
    p = np.clip(abs(U[:, -1].conj()@L)**2/M, 0, 1)
    target_factor = np.clip(1/np.maximum(p, 0.05), 0.5, 2.0)
    y = 0.8*y+0.2*anchor*target_factor
    broaden = np.clip(0.15/max(float(p.max()), 0.05), 1.0, 2.5)
    y = np.clip(y*broaden, 0, cfg.margin_ceiling)
    return y, anchor, p


# ------------------------------------------------------------------------------------------------ comparators
def normalized_covariance(Rsum, mass, loading=1e-6):
    """Exponentially weighted covariance normalized by its total weight, with a relative diagonal load."""
    R = (Rsum+Rsum.conj().T)/(2*mass)
    floor = loading*float(np.real(np.trace(R)))/R.shape[0]
    return R+floor*np.eye(R.shape[0])


def mvdr_weights(R, a):
    v = solve(R, a, assume_a='pos', check_finite=False)
    den = np.vdot(a, v)
    if den.real <= 0 or abs(den.imag) > 1e-7*abs(den):
        raise FloatingPointError('Invalid MVDR normalization.')
    return v/den


def spherical_wc_from_eig(eig, a, radius):
    """Worst-case SOCP [26] with one spherical ball, solved through its scalar KKT root.

    gamma ||(R + gamma I)^-1 a|| = radius; this is the same convex problem as the single-ball SOCP.
    """
    lam, U = eig
    if radius < 0 or radius >= np.linalg.norm(a):
        raise ValueError('The spherical set includes the origin or has an invalid radius.')
    if np.min(lam) <= 0:
        raise ValueError('The covariance must be positive definite.')
    z = U.conj().T@a
    if radius == 0:
        gamma = 0.0
    else:
        def root(gamma):
            return np.linalg.norm((gamma/(lam+gamma))*z)-radius
        hi = max(float(lam.max()), 1e-12)
        for _ in range(100):
            if root(hi) >= 0:
                break
            hi *= 2
        else:
            raise RuntimeError('Could not bracket the spherical WC multiplier.')
        gamma = brentq(root, 0.0, hi, xtol=1e-13, rtol=1e-13)
    v = U@(z/(lam+gamma))
    den = float(np.real(np.vdot(a, v))-radius*np.linalg.norm(v))
    if den <= 0:
        raise RuntimeError('Nonpositive worst-case normalization.')
    w = v/den
    if abs(wc_residual(w, a[:, None], [radius])[0]) > 1e-6:
        raise RuntimeError('Spherical WC constraint failed after solution.')
    return w


def rcb_from_eig(eig, a, epsilon):
    """Robust Capon beamformer [24] with a spherical set; epsilon is the squared radius.

    For the same covariance and epsilon = radius**2, robust Capon and the single-ball worst-case beamformer give the
    same beam direction, which is why their curves coincide in Figure 8.11.
    """
    lam, U = eig
    if epsilon < 0 or epsilon >= np.real(np.vdot(a, a)):
        raise ValueError('Require 0<=epsilon<||a||^2.')
    z = U.conj().T@a
    if epsilon == 0:
        ah = a.copy()
    else:
        def equation(mu):
            return float(np.sum(abs(z/(1+mu*lam))**2)-epsilon)
        hi = 1.0/max(float(lam.min()), 1e-300)
        while equation(hi) > 0:
            hi *= 2
            if not np.isfinite(hi):
                raise RuntimeError('Could not bracket the RCB multiplier.')
        mu = brentq(equation, 0, hi, xtol=1e-13, rtol=1e-13)
        ah = a-U@(z/(1+mu*lam))
    v = U@((U.conj().T@ah)/lam)
    den = float(np.real(np.vdot(ah, v)))
    if den <= 0:
        raise RuntimeError('Nonpositive RCB normalization.')
    return v/den


def reconstruction_geometry(M, f, fc, thp, sector_halfwidth_rad):
    """Angular grid, desired-sector mask and trapezoidal weights for the covariance reconstruction."""
    step = min(np.deg2rad(0.25), beamwidth_rad(M, thp)/8)
    count = int(np.ceil(np.pi/step))+1
    theta = np.linspace(-np.pi/2, np.pi/2, count)
    inside = abs(theta-thp) <= sector_halfwidth_rad
    if not np.any(inside) or not np.any(~inside):
        raise ValueError('Reconstruction sector is empty or covers the whole field.')
    weights = np.full(count, theta[1]-theta[0]); weights[[0, -1]] *= 0.5
    return steering_matrix(M, theta, f, fc), inside, weights


def reconstruction_weights(eig, a, geometry, signal_present):
    """Simplified covariance reconstruction after [30].

    The interference-plus-noise covariance is reconstructed by integrating the Capon spectrum outside the declared
    sector. With interference-only training there is no desired signal to estimate, so the nominal steering vector is
    kept; with signal-present training the principal eigenvector of the in-sector reconstruction is used, with its
    phase aligned to the nominal vector. This is not the complete steering-vector optimization of [30].
    """
    lam, U = eig; A, inside, quad = geometry
    projections = U.conj().T@A
    capon = 1/np.maximum(np.sum(abs(projections)**2/lam[:, None], axis=0), 1e-300)
    outside_weights = capon[~inside]*quad[~inside]
    Ao = A[:, ~inside]
    Rin = (Ao*outside_weights)@Ao.conj().T
    Rin = (Rin+Rin.conj().T)/2
    Rin += 1e-6*float(np.trace(Rin).real)/len(a)*np.eye(len(a))
    if signal_present:
        Ai = A[:, inside]
        Rsig = (Ai*(capon[inside]*quad[inside]))@Ai.conj().T
        _, V = eigh((Rsig+Rsig.conj().T)/2, subset_by_index=(len(a)-1, len(a)-1))
        ah = V[:, -1]*np.linalg.norm(a)
        overlap = np.vdot(a, ah)
        if abs(overlap) > 1e-14:
            ah *= np.exp(-1j*np.angle(overlap))
    else:
        ah = a
    return mvdr_weights(Rin, ah)


def centred_steering(M, theta, f, fc):
    """Steering vectors with the phase reference at the array centre (used by the multi-point LCMV)."""
    A = steering_matrix(M, theta, f, fc)
    return A*np.exp(-1j*np.pi*(f/fc)*(M-1)/2*np.sin(np.atleast_1d(theta)))[None, :]


def lcmv_weights(Rb, Lc, load):
    """Multi-point LCMV [22]: unit response at every column of Lc, diagonal loading `load` (times the noise power)."""
    RiC = np.linalg.solve(Rb + load*np.eye(Rb.shape[0]), Lc)
    return RiC @ np.linalg.solve(Lc.conj().T @ RiC, np.ones(Lc.shape[1]))


def ewma_covariance(Xn, eta):
    """Exponentially weighted covariance (8.10) of a snapshot block, started from the identity, and its total weight."""
    Rs = np.eye(Xn.shape[0], dtype=complex); mass = 1.0
    for k in range(Xn.shape[1]):
        Rs *= eta; Rs += np.outer(Xn[:, k], Xn[:, k].conj()); mass = eta*mass + 1
    return Rs, mass


def multiball_socp(Rb, L, nu, solver='CLARABEL'):
    """Exact solution of (8.8) for a frozen covariance and frozen margins: min ||U w|| subject to
    Re(L_i^H w) >= 1 + nu_i ||w||, with R = U^H U, by an interior-point cone solver (CVXPY). Returns (w, seconds)."""
    import cvxpy as cp
    U = np.linalg.cholesky(Rb).conj().T
    w = cp.Variable(L.shape[0], complex=True)
    prob = cp.Problem(cp.Minimize(cp.norm(U @ w, 2)), [cp.real(L.conj().T @ w) >= 1 + nu*cp.norm(w, 2)])
    t0 = time.perf_counter(); prob.solve(solver=solver); dt = time.perf_counter()-t0
    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f'SOCP status {prob.status}')
    return w.value, dt


# ------------------------------------------------------------------------------------------------ training loop
def train_subband(Xn, f, cfg, methods=CORE_METHODS):
    """Run every requested method on the same training snapshots Xn of one subcarrier f.

    The true channel and the true covariance are not inputs. WRAB and the fixed-margin variant start from the
    projection of a(theta_p, f)/sqrt(M) onto the initial feasible set and update at every snapshot; the single-ball
    WC beamformers update the same way with their one ball; MVDR, robust Capon, the spherical SOCP and the
    reconstruction comparator are rebuilt every batch_period snapshots and at the last snapshot.
    """
    M, K = Xn.shape
    if (M, K) != (cfg.M, cfg.K):
        raise ValueError('Snapshot shape does not match the scenario.')
    if any(m not in METHOD_LABELS for m in methods):
        raise ValueError('Unknown comparator.')
    thp, grid, y0 = design_grid(cfg)
    L = steering_matrix(M, grid, f if cfg.squint_aware else cfg.fc, cfg.fc)
    af = steer_f(M, thp, f, cfg.fc)
    ac = steer_f(M, thp, cfg.fc, cfg.fc)
    radius = cfg.single_radius_fraction*np.sqrt(M)
    nu = y0.copy(); anchor = None
    histories = {m: np.zeros((K, M), complex) for m in methods}
    residuals = {m: np.zeros(K) for m in methods if m in ('wrab', 'fixed', 'wc', 'wc_faware')}
    cycles = {m: np.zeros(K, int) for m in residuals}
    yhist = np.zeros((K, len(y0)))
    updates = []
    w = {}
    for m in methods:
        if m in ('wrab', 'fixed'):
            w[m], _ = project_wc(af/np.sqrt(M), L, y0, cfg.projection_tolerance, cfg.projection_max_cycles)
        elif m in ('wc', 'wc_faware'):
            aa = ac if m == 'wc' else af
            w[m], _ = project_wc(aa/np.sqrt(M), aa[:, None], [radius], cfg.projection_tolerance,
                                 cfg.projection_max_cycles)
        else:
            w[m] = ac/M if m == 'analog' else af/M
    geometry = None
    if 'recon' in methods:
        sector = max(float(np.max(abs(grid-thp))), np.deg2rad(0.05))
        geometry = reconstruction_geometry(M, f, cfg.fc, thp, sector)
    Rsum = np.eye(M, dtype=complex); mass = 1.0
    started = time.perf_counter()
    for k in range(1, K+1):
        x = Xn[:, k-1]
        Rsum *= cfg.eta; Rsum += np.outer(x, x.conj())
        mass = cfg.eta*mass+1
        if 'wrab' in methods and cfg.self_anchor and k > cfg.warmup and k % cfg.margin_period == 0:
            nu, anchor, p = margin_update(Rsum, L, nu, anchor, cfg)
            updates.append({'snapshot': k, 'alignment': p.tolist(), 'margins': nu.tolist()})
        for m in residuals:
            if m == 'wrab':
                LM, yn = L, nu
            elif m == 'fixed':
                LM, yn = L, y0
            else:
                LM = (ac if m == 'wc' else af)[:, None]; yn = np.array([radius])
            w[m], nc = projected_step(Rsum, w[m], LM, yn, cfg)
            cycles[m][k-1] = nc
            residuals[m][k-1] = float(np.max(wc_residual(w[m], LM, yn)))
        if k % cfg.batch_period == 0 or k == K:
            batch = [m for m in methods if m in ('mvdr', 'rcb', 'socp', 'recon')]
            if batch:
                Rb = normalized_covariance(Rsum, mass, cfg.covariance_loading)
                eig = eigh(Rb, check_finite=False)
                if 'mvdr' in batch:
                    lam, U = eig
                    v = U@((U.conj().T@af)/lam)
                    w['mvdr'] = v/np.vdot(af, v)
                if 'rcb' in batch:
                    w['rcb'] = rcb_from_eig(eig, af, radius**2)
                if 'socp' in batch:
                    w['socp'] = spherical_wc_from_eig(eig, af, radius)
                if 'recon' in batch:
                    w['recon'] = reconstruction_weights(eig, af, geometry, not cfg.signal_free)
        for m in methods:
            if not np.all(np.isfinite(w[m])):
                raise FloatingPointError(f'{m}: non-finite weights at snapshot {k}.')
            histories[m][k-1] = w[m]
        yhist[k-1] = nu
    return {'histories': histories, 'residuals': residuals, 'projection_cycles': cycles,
            'margin_history': yhist, 'margin_updates': updates, 'L': L,
            'final_R': Rsum, 'covariance_mass': mass,
            'seconds': time.perf_counter()-started}


# ------------------------------------------------------------------------------------------------ scoring
def sinr_linear(w, channel, R_ipn, ps):
    """Output SINR (8.5) of weight w for the true channel, true interference-plus-noise covariance and signal power."""
    w = np.asarray(w, complex)
    num = float(ps*abs(np.vdot(w, channel))**2)
    den = float(np.real(np.vdot(w, R_ipn@w)))
    if not (np.isfinite(num) and np.isfinite(den) and den > 0):
        raise FloatingPointError('SINR has a non-finite numerator or nonpositive denominator.')
    return num/den


def selected_weight(history, cfg):
    return history[-1] if cfg.score_weights == 'last' else history[-min(50, len(history)):].mean(axis=0)


def evaluate_trial(cfg, seed, methods=CORE_METHODS, return_trace=False):
    """Train and score every method on every subcarrier of one trial."""
    trial = generate_trial(cfg, seed)
    values = {m: np.zeros(cfg.Nsc) for m in methods}
    residual = {m: -np.inf for m in methods if m in ('wrab', 'fixed', 'wc', 'wc_faware')}
    saturation = []; covered = []; times = []; final_margins = []
    saved_trace = None
    ps = 10**(cfg.snr_db/10)
    for n, f in enumerate(trial['frequencies']):
        trained = train_subband(trial['X'][n], f, cfg, methods)
        for m in methods:
            w = selected_weight(trained['histories'][m], cfg)
            values[m][n] = sinr_linear(w, trial['channel'][n], trial['R_ipn'][n], ps)
        for m in residual:
            residual[m] = max(residual[m], float(trained['residuals'][m].max()))
        saturation.append(float(np.mean(trained['margin_history'][-1] >= cfg.margin_ceiling-1e-10)))
        final_margins.append(trained['margin_history'][-1].tolist())
        at = steer_f(cfg.M, trial['true_direction_rad'], f, cfg.fc)   # scoring diagnostic only
        covered.append(bool(np.min(np.linalg.norm(trained['L']-at[:, None], axis=0)
                                   - trained['margin_history'][-1]) <= 1e-10))
        times.append(trained['seconds'])
        if return_trace and n == cfg.Nsc//2:
            saved_trace = trained
    return {'linear': values, 'max_residual': residual,
            'margin_saturation_fraction': float(np.mean(saturation)),
            'ball_coverage_fraction': float(np.mean(covered)),
            'final_margins': final_margins,
            'seconds': float(np.sum(times)), 'jitter_rad': trial['jitter_rad'],
            'trace': saved_trace}


def boot_ci(v, nb=2000, alpha=0.05, seed=0):
    """Mean and percentile-bootstrap 95% interval over trials (2000 resamples)."""
    v = np.asarray(v, float).reshape(-1)
    if v.size < 2 or not np.all(np.isfinite(v)):
        raise ValueError('Bootstrap needs at least two finite independent trial values.')
    r = np.random.default_rng(seed)
    means = v[r.integers(v.size, size=(nb, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return float(v.mean()), float(lo), float(hi)


def summarize(v, label):
    """Mean, bootstrap interval and sample standard deviation of per-trial values in dB."""
    v = np.asarray(v, float)
    mean, lo, hi = boot_ci(v, seed=seed_for(label))
    return dict(mean_db=mean, ci95_low_db=lo, ci95_high_db=hi, std_db=float(np.std(v, ddof=1)), samples=v.tolist())


def run_trials(cfg, NR, label, methods=CORE_METHODS, keep_trace=False):
    """NR trials of one configuration; the trial is the resampling unit (subcarriers are not extra trials)."""
    if NR < 2:
        raise ValueError('Use at least two trials for reported uncertainty.')
    raw = {m: np.zeros((NR, cfg.Nsc)) for m in methods}
    diagnostics = []
    trace = None
    for tr in range(NR):
        result = evaluate_trial(cfg, seed_for(label, tr), methods, return_trace=keep_trace and tr == 0)
        for m in methods:
            raw[m][tr] = result['linear'][m]
        diagnostics.append({k: v for k, v in result.items() if k not in ('linear', 'trace')})
        if result['trace'] is not None:
            trace = result['trace']
    summary, samples = {}, {}
    for m in methods:
        mean_db = 10*np.log10(np.maximum(raw[m], np.finfo(float).tiny)).mean(axis=1)
        linear_db = 10*np.log10(np.maximum(raw[m].mean(axis=1), np.finfo(float).tiny))
        v = mean_db if cfg.aggregation == 'mean_db' else linear_db
        mean, lo, hi = boot_ci(v, seed=seed_for(label+'|bootstrap|'+m))
        summary[m] = {'mean_db': mean, 'std_db': float(np.std(v, ddof=1)),
                      'ci95_low_db': lo, 'ci95_high_db': hi,
                      'mean_of_subcarrier_db': float(mean_db.mean()),
                      'db_of_mean_subcarrier_linear': float(linear_db.mean())}
        samples[m] = v
    return {'config': asdict(cfg), 'label': label, 'NR': NR, 'methods': list(methods),
            'summary': summary, 'samples': samples, 'raw_linear': raw,
            'diagnostics': diagnostics, 'trace': trace}


def reference_constrained_minimum(R, L, nu, initial=None, target=None):
    """Independent SciPy SLSQP solution of the same constrained problem (small arrays, for checks only).

    With `target` it minimizes 0.5 ||w - target||^2 (the projection); otherwise w^H R w for a frozen covariance.
    """
    M = L.shape[0]
    A = np.hstack([L.real.T, L.imag.T])
    nu = np.asarray(nu, float)

    def unpack(x):
        return x[:M]+1j*x[M:]
    if initial is None:
        initial, _ = project_wc(np.zeros(M, complex), L, nu, tol=1e-11, max_cycles=2000)
    x0 = np.r_[initial.real, initial.imag]

    def cons(x):
        return A@x-nu*np.linalg.norm(x)-1

    def jac(x):
        return A-nu[:, None]*x[None, :]/max(np.linalg.norm(x), 1e-300)
    if target is None:
        Q = np.block([[R.real, -R.imag], [R.imag, R.real]])
        scale = max(float(np.linalg.norm(Q, 2)), 1e-15)
        fun = lambda x: float(x@(Q@x)/scale)      # noqa: E731
        grad = lambda x: 2*(Q@x)/scale            # noqa: E731
    else:
        tz = np.r_[target.real, target.imag]
        fun = lambda x: float(0.5*np.sum((x-tz)**2))   # noqa: E731
        grad = lambda x: x-tz                           # noqa: E731
    out = minimize(fun, x0, jac=grad, constraints=[{'type': 'ineq', 'fun': cons, 'jac': jac}],
                   method='SLSQP', options={'ftol': 1e-12, 'maxiter': 2000})
    w = unpack(out.x)
    if not out.success or np.max(wc_residual(w, L, nu)) > 2e-7:
        raise RuntimeError(f'Reference optimization failed: {out.message}; residual '
                           f'{np.max(wc_residual(w, L, nu)):.3g}')
    return w, {'status': out.message, 'iterations': int(out.nit),
               'max_residual': float(np.max(wc_residual(w, L, nu)))}


# ------------------------------------------------------------------------------------------------ dynamic rotation
def generate_dynamic_trial(cfg, seed, rho):
    """Whole-array rotation phi_k following AR(1) with correlation rho per snapshot and RMS cfg.jitter_deg
    (Section 8.2.3). The rotation moves the desired source and the interferers relative to the array.
    Returns the training snapshots, the frequencies, and the per-snapshot desired and interferer steering vectors."""
    ss = np.random.SeedSequence(seed).spawn(3)
    geo, noise, _ = (np.random.default_rng(s) for s in ss)
    sig = np.deg2rad(cfg.jitter_deg); phi = np.zeros(cfg.K); phi[0] = geo.normal(0, sig)
    for k in range(1, cfg.K):
        phi[k] = rho*phi[k-1] + np.sqrt(1-rho**2)*geo.normal(0, sig)
    thd = np.deg2rad(cfg.th_d_deg); ti = np.deg2rad(np.asarray(cfg.interferers_deg))
    fl = subcarrier_frequencies(cfg.fc, cfg.B, cfg.Nsc); m = np.arange(cfg.M)[:, None]
    X, H, A = [], [], []
    for f in fl:
        U = cn(noise, (len(ti), cfg.K))*10**(cfg.inr_db/20)
        xn = cn(noise, (cfg.M, cfg.K)); Ak = []
        for q in range(len(ti)):
            a = np.exp(1j*np.pi*(f/cfg.fc)*m*np.sin(ti[q] + phi[None, :]))
            xn += a*U[q][None, :]; Ak.append(a)
        X.append(np.ascontiguousarray(xn)); H.append(np.exp(1j*np.pi*(f/cfg.fc)*m*np.sin(thd + phi[None, :])))
        A.append(np.stack(Ak, 1))
    return X, fl, H, A


# ------------------------------------------------------------------------------------------------ frozen-norm correction
def sequential_snapshots(M, K, fc, interferers_deg, inr_db, seed=42):
    """Interference-only snapshots of one carrier-frequency subband, drawn snapshot by snapshot from one generator
    (interferer amplitudes, then element noise). Used for the feasibility comparison of Section 8.3.1."""
    rng = np.random.default_rng(seed)
    intf = 10**(inr_db/20)
    rng.normal(0, 0.0, size=K)                 # zero pointing error for every snapshot (keeps the stream order)
    th = np.deg2rad(np.asarray(interferers_deg, float))
    X = np.zeros((M, K), dtype=complex)
    for k in range(K):
        for t in th:
            ui = intf*(rng.standard_normal()+1j*rng.standard_normal())/np.sqrt(2)
            X[:, k] += ui*steer_f(M, t, fc, fc)
        X[:, k] += (rng.standard_normal(M)+1j*rng.standard_normal(M))/np.sqrt(2)
    return X


def frozen_norm_trace(Xn, f, fc, th_p, grid, y0, eta=0.97, warmup=50, period=30):
    """Same gradient step and margin rule as WRAB, but with the cheaper one-step correction discussed after (8.14):
    the norm is frozen at ||w~||, the linear multiplier system of the violated constraints is solved with a small
    load and clipped componentwise, tau = max(solve(G + 1e-6 I, d), 0), and w = w~ + L_A tau.
    Returns the constraint residual of (8.8) after every update and the margins in use."""
    M, K = Xn.shape
    L = steering_matrix(M, np.asarray(grid), f, fc)
    Nc = L.shape[1]
    y = np.asarray(y0, float).copy(); anchor = y.copy()
    R = np.eye(M, dtype=complex)
    w = steer_f(M, th_p, f, fc); w = w/np.linalg.norm(w)
    anchored = False
    residuals, margins = np.zeros(K), np.zeros((K, Nc))
    ksig = max(1, int(0.25*M))
    for k in range(K):
        R = eta*R + np.outer(Xn[:, k], Xn[:, k].conj())
        if k > warmup:
            if not anchored:
                _, Us = eigh((R+R.conj().T)/2, subset_by_index=(M-ksig, M-1))
                anchor = np.maximum((np.sum(abs(L)**2, axis=0)-np.sum(abs(Us.conj().T@L)**2, axis=0))/M, 0.0)
                anchor = 1.5*anchor/max(anchor.mean(), 1e-6)
                y = anchor.copy(); anchored = True
            if k % period == 0:
                _, e = eigh((R+R.conj().T)/2, subset_by_index=(M-1, M-1))
                p = np.abs(e[:, 0].conj()@L)**2/M
                y = 0.8*y + 0.2*anchor*np.clip(1.0/np.maximum(p, 0.05), 0.5, 2.0)
                y = np.clip(y*np.clip(0.15/max(float(p.max()), 0.05), 1.0, 2.5), 0.0, 3.0)
        Rw = R@w
        mu = 0.1*np.real(np.vdot(Rw, Rw))/max(np.real(np.vdot(Rw, R@Rw)), 1e-20)
        wt = w - mu*Rw
        wtn = max(np.linalg.norm(wt), 1e-15)
        act = [i for i in range(Nc) if np.real(np.vdot(wt, L[:, i])) < y[i]*wtn + 1.0]
        if act:
            La = L[:, act]
            d = np.array([(y[i]*wtn + 1.0) - np.real(np.vdot(wt, L[:, i])) for i in act])
            G = np.real(La.conj().T@La)
            tau = np.maximum(np.linalg.solve(G + 1e-6*np.eye(len(act)), d), 0)
            w = wt + La@tau
        else:
            w = wt
        residuals[k] = float(np.max(wc_residual(w, L, y)))
        margins[k] = y
    return residuals, margins
