"""
ch5mprrab.algorithms -- MPR-RAB, the multi-primary robust adaptive beamformer of Chapter 5 (Sections 5.3-5.5).

At every update the kernel solves (5.8)

    min_w  w^H R_in w   s.t.   Re{w^H a~_k} >= nu_k ||w|| + beta_k,   k = 1..K,

numerically, with convergence and KKT residual diagnostics, with
  * R_in the interference-plus-noise covariance reconstructed from the Capon spectrum outside sectors of half the
    3-dB beamwidth around the tracked clusters; strong spectral peaks are located off the grid by a local search
    before the integration, so that the reconstruction does not depend on where a strong interferer falls relative
    to the grid (Section 5.4);
  * margins nu_k = kappa * sqrt(M / K) motivated by joint feasibility (Section 5.3.2); kappa < 1 alone is not a
    general feasibility certificate;
  * gain targets from the max-SINR beamformer of the estimated model (targets 'ms', Section 5.3.3): with
    G = A^H R_t^-1 A and P = diag(p_k), p_k the Capon powers of the tracked clusters, r is the principal eigenvector
    of G P, beta_k = |r_k| / max_j |r_j| and a~_k = a_k exp(j arg r_k).  R_t is the reconstruction with the excluded
    sectors widened by TARGET_SECT and filled with the noise level of the spectrum, which removes the skirts of strong
    cluster peaks that the narrower sectors of R_in leave behind; with R_t = R_in (target_sect=None) the solution at
    kappa = 0 is exactly the max-SINR beamformer of R_in (Proposition 5.2).  kappa > 0 adds a worst-case margin on
    every cluster.  Power-law targets beta_k = (p_k / p_max)^q with a~_k = a_k are available through a numeric q
    (q = 0: equal targets, beta_k = 1);
  * the active-set KKT method of Chapter 2 (AS-SNC) as the solver (Section 5.3.5; real embedding
    w_r = [Re w; Im w], Q = [[Re R, -Im R], [Im R, Re R]], l_k = [Re a~_k; Im a~_k]); the KKT point is
    w = (R_in + (mu/2) I)^-1 A~_A c with c >= 0, an LCMV beamformer whose loading mu/2 is fixed by the margins.

Schedule and tracking (Section 5.5).  The weight is recomputed every T_rec snapshots, at every snapshot during the
first T_rec, and whenever the cluster set changes.  After a warm-up the cluster directions are refined at every
update: a Capon search in a +/-WIN_DEG window on the interferer-subspace-deflated covariance (an eigencomponent above
ten times the noise level is flattened when less than DEFL of its energy lies in the span of the tracked steering
vectors), golden-section refinement between the scan points around the best one, and a step cap of MAX_STEP_DEG per
update.  The exponentially weighted covariance places a drifting cluster eta / (1 - eta) snapshots in the past; the
directions used in the constraints are advanced over that delay plus half an update interval with the least-squares
slope of the last PRED_WIN refinements (PRED_MIN needed; a capped acquisition step restarts the slope).  A cluster is
retired when its Capon output power over a short window (forgetting factor ETA_FAST) falls below DEATH_FRAC of the
strongest after a grace period; births follow an external hint.

The channel, the steering model and the SINR measures are in ch5mprrab.channel.  The planar array is supported
through the module-level geometry object GEOM (see ch5mprrab.planar).
"""
import numpy as np
import warnings
from numpy.linalg import solve, norm, pinv, LinAlgError
from scipy.linalg import eigh
from .channel import steer, ETA, EPS, M_DEFAULT  # noqa: F401  (M_DEFAULT is routed by ch5mprrab.planar)

# ----------------------------------------------------------------------------- design constants
KAPPA = 0.3            # margin: nu_k = KAPPA * sqrt(M / K)   (KAPPA < 1 suffices only for orthogonal directions; check the current Gram matrix)
TARGETS = 'ms'         # 'ms': max-SINR targets; a number q: beta_k = (p_k / p_max)^q
T_REC = 10             # update interval of the reconstruction, the targets and the weight (snapshots)
T_REF = None           # refinement interval after WARMUP; None: at every update (T_rec)
WARMUP = 20             # snapshots before the first refinement
EARLY_UPDATES = True   # recompute the weight at every snapshot during the first T_rec snapshots
WIN_DEG = 3.0          # refinement window half-width (deg)
N_SCAN = 31            # refinement scan points
MAX_STEP_DEG = 1.0     # refinement step cap per update (deg)
POLISH = True          # golden-section refinement between the scan points around the best one
PREDICT = True         # constant-velocity compensation of the delay of the tracked directions
PRED_WIN = 8           # refinements in the least-squares slope
PRED_MIN = 4           # refinements required before the slope is used
DEFL = 0.3             # interferer-subspace deflation threshold, 0 = off
DEFL_RULE = 'proj'     # 'proj': energy of the eigenvector in the span of the tracked steering vectors;
                       # 'max': largest normalized alignment with a single tracked steering vector
ETA_FAST = 0.9         # forgetting factor of the power estimate used for retirement (None: long-window Capon power)
DEATH_FRAC = 0.05      # lifecycle: retire below this fraction of the strongest cluster power
GRACE = 90             # lifecycle: grace period (snapshots)
LOAD_REF = 1e-4        # loading of the inverse used by refinement / Capon powers (x tr(R)/M)
LOAD_REC = 1e-6        # loading of the inverse used by the reconstruction (x tr(R)/M)
EXP_RAD = None         # None: sector half-width = half the 3-dB beamwidth of the array axis
PEAK_REFINE = True     # locate strong spectral peaks off the grid before the reconstruction
PEAK_RHO = 10.0        # a peak is strong if it exceeds PEAK_RHO times the median of the spectrum
SUB_MIN = 4000        # grids larger than this (the planar array) form the full spectral sum once per update and
                      # subtract the excluded points, which is the same sum with fewer operations
TARGET_SECT = 1.5      # the max-SINR targets use a covariance whose excluded sectors are TARGET_SECT times wider and
                       # filled with the noise level of the spectrum (None: the reconstruction itself)

# ----------------------------------------------------------------------------- active-set KKT solver (AS-SNC, Chapter 2)
def _solve_fixed_A(Q_eig, L_A, nu_A, beta_A, max_iter=100, tol=1e-11, mu_init=None, t_init=None):
    s, U = Q_eig
    Mr = U.shape[0]
    nA = L_A.shape[1]
    LU = U.T @ L_A
    mu = 1.0 if mu_init is None else mu_init
    t = 1.0 if t_init is None else t_init
    w = None
    for _ in range(max_iter):
        D = 1.0 / (2.0 * s + mu)
        M_lam = (LU.T * D) @ LU
        try:
            lam_new = solve(M_lam + 1e-14 * np.eye(nA), nu_A * t + beta_A)
        except LinAlgError:
            lam_new = pinv(M_lam) @ (nu_A * t + beta_A)
        if np.any(lam_new < -1e-10):
            return U @ (D * (LU @ lam_new)), lam_new, mu, t, False
        lam_new = np.maximum(lam_new, 0.0)
        w_new = U @ (D * (LU @ lam_new))
        t_new = norm(w_new)
        if t_new < 1e-14:
            return np.zeros(Mr), lam_new, mu, 0.0, True
        mu_new = (nu_A @ lam_new) / t_new
        err = abs(t_new - t) + abs(mu_new - mu)
        mu, t, w = mu_new, t_new, w_new
        if err < tol * max(1.0, t):
            return w, lam_new, mu, t, True
    return w, lam_new, mu, t, False


LAST_SOLVER_STATUS = {}
SOLVER_RESIDUAL_TOL = 1e-8


def _solver_status(Q_eig, L, nu, beta, w, active, lam_A, inner_ok, outer_capped, outer):
    """Independent scaled checks of all KKT conditions; no change to returned weights."""
    global LAST_SOLVER_STATUS
    if w is None or not np.all(np.isfinite(w)):
        LAST_SOLVER_STATUS = dict(certified=False, inner_converged=False, outer_capped=bool(outer_capped),
                                  outer_iterations=int(outer), reason='nonfinite iterate')
        return LAST_SOLVER_STATUS
    active = np.asarray(active, dtype=int)
    lm = np.zeros(L.shape[1])
    if len(active): lm[active] = lam_A
    t = norm(w)
    residual = L.T @ w - nu * t - beta
    scale = max(1., float(np.max(np.abs(L.T @ w))), float(np.max(np.abs(nu * t + beta))))
    primal = max(0., float(-np.min(residual))) / scale
    dual = max(0., float(-np.min(lm))) / max(1., float(norm(lm, np.inf)))
    q, U = Q_eig
    qw = U @ (q * (U.T @ w))
    load = float(nu @ lm) / max(t, 1e-300)
    st = 2 * qw + load * w - L @ lm
    stationarity = float(norm(st)) / max(1., float(norm(2 * qw)), float(norm(load * w)), float(norm(L @ lm)))
    complementarity = float(np.max(np.abs(lm * residual))) / max(1., abs(float(w @ qw)))
    gram = L.T @ L
    try:
        feasibility_value = float(nu @ np.linalg.solve(gram, nu))
        feasibility_sufficient = bool(np.linalg.eigvalsh(gram).min() > 0 and feasibility_value < 1.)
    except np.linalg.LinAlgError:
        feasibility_value = float('inf'); feasibility_sufficient = False
    finite = np.all(np.isfinite([primal, dual, stationarity, complementarity]))
    certified = bool(inner_ok and not outer_capped and finite and
                     max(primal, dual, stationarity, complementarity) <= SOLVER_RESIDUAL_TOL)
    LAST_SOLVER_STATUS = dict(certified=certified, inner_converged=bool(inner_ok),
        outer_capped=bool(outer_capped), outer_iterations=int(outer), primal=float(primal), dual=float(dual),
        stationarity=stationarity, complementarity=complementarity,
        feasibility_value=feasibility_value, feasibility_sufficient=feasibility_sufficient,
        residual_tolerance=SOLVER_RESIDUAL_TOL)
    return LAST_SOLVER_STATUS


def as_snc(Q_eig, L, nu, beta, max_outer=100, tol=1e-9, A0=None):
    """Numerical active-set solver (Section 5.3.5).  Returns (w, outer passes, active set, mu).
    LAST_SOLVER_STATUS carries convergence and KKT certification; capped output is approximate.
    """
    Mr, N = L.shape
    A = list(range(N)) if A0 is None else list(A0)
    mu_hint, t_hint = None, None
    w_A, mu_A = None, 0.0
    lam_A = np.zeros(len(A)); ok = False
    for it in range(max_outer):
        if len(A) == 0:
            if np.all(beta <= 0):
                w_A = np.zeros(Mr)
                _solver_status(Q_eig,L,nu,beta,w_A,[],[],True,False,it)
                return w_A, it, np.array([], dtype=int), 0.0
            A = [int(np.argmax(beta))]
            continue
        w_A, lam_A, mu_A, t_A, ok = _solve_fixed_A(Q_eig, L[:, A], nu[A], beta[A], mu_init=mu_hint, t_init=t_hint)
        if w_A is None or not np.all(np.isfinite(w_A)):
            A.pop()
            if not A:
                w_A = np.zeros(Mr)
                _solver_status(Q_eig,L,nu,beta,w_A,[],[],False,False,it)
                return w_A, it, np.array([], dtype=int), 0.0
            continue
        if np.any(lam_A < -tol):
            A.pop(int(np.argmin(lam_A)))
            continue
        residuals = L.T @ w_A - nu * norm(w_A) - beta
        viol = [(i, residuals[i]) for i in range(N) if i not in A and residuals[i] < -tol]
        if viol:
            A.append(min(viol, key=lambda x: x[1])[0])
            mu_hint, t_hint = mu_A, t_A
            continue
        _solver_status(Q_eig,L,nu,beta,w_A,A,lam_A,ok,False,it+1)
        return w_A, it + 1, np.array(A, dtype=int), mu_A
    # A may have changed after the last solve; the outer cap cannot be certified.
    global LAST_SOLVER_STATUS
    LAST_SOLVER_STATUS = dict(certified=False, inner_converged=bool(ok), outer_capped=True,
                              outer_iterations=int(max_outer), reason='outer iteration cap')
    return w_A, max_outer, np.array(A, dtype=int), mu_A


def real_form(R, A):
    """Complex quadratic form and constraint directions -> real SNC-SOCP data."""
    Q = np.block([[R.real, -R.imag], [R.imag, R.real]])
    L = np.vstack([A.real, A.imag])
    return Q, L


# ----------------------------------------------------------------------------- array geometry
def golden_max(f, lo, hi, iters=30):
    """Maximizer of a unimodal function on [lo, hi] by golden-section search."""
    gr = 0.6180339887498949
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


class ULAGeometry:
    """Scan grid, reconstruction grid and refinement moves for the uniform linear array.
    Directions are scalars (radians)."""
    name = 'ULA'

    def __init__(self, M):
        self.M = M
        self.bw = 0.886 * 2.0 / M                       # 3-dB beamwidth (rad), half-wavelength spacing
        self.scan = np.deg2rad(np.linspace(-90, 90, 361))
        self.As = np.column_stack([steer(M, t) for t in self.scan])
        self.dth = self.scan[1] - self.scan[0]

    def sector(self):
        return 0.5 * self.bw if EXP_RAD is None else EXP_RAD

    def mask(self, th_list, sec):
        m = np.ones(len(self.scan), dtype=bool)
        for t in th_list:
            m &= np.abs(self.scan - t) > sec
        return m

    def refine_candidates(self, th_old):
        return np.linspace(th_old - np.deg2rad(WIN_DEG), th_old + np.deg2rad(WIN_DEG), N_SCAN)

    def refine_peaks(self, spec, msk, Ri, theta_k, sec):
        """Strong local maxima of the Capon spectrum outside the excluded sectors, located off the grid by a
        golden-section search between the neighbouring grid points.  Returns the refined steering vectors, their
        Capon powers and the grid indices they replace."""
        M = self.M
        thr = PEAK_RHO * np.median(spec)
        p_of = lambda t: 1.0 / max(np.real(steer(M, t).conj() @ Ri @ steer(M, t)), 1e-12)
        cols, ws, drop = [], [], []
        for i in range(1, len(spec) - 1):
            if not msk[i] or spec[i] <= thr or spec[i] < spec[i - 1] or spec[i] < spec[i + 1]:
                continue
            th = golden_max(p_of, self.scan[i - 1], self.scan[i + 1])
            if any(abs(th - tk) <= sec for tk in theta_k):
                continue
            cols.append(steer(M, th))
            ws.append(p_of(th))
            drop.append(i)
        if not cols:
            return None, None, []
        return np.column_stack(cols), np.array(ws), drop

    def cap_step(self, th_old, th_new):
        step = th_new - th_old
        cap = np.deg2rad(MAX_STEP_DEG)
        return th_old + np.sign(step) * cap if abs(step) > cap else th_new

    def capped(self, th_old, th_new):
        return abs(th_new - th_old) > np.deg2rad(MAX_STEP_DEG)

    def polish(self, th, p_of):
        h = np.deg2rad(2.0 * WIN_DEG / (N_SCAN - 1))
        return golden_max(p_of, th - h, th + h, 20)

    def fit(self, meas):
        """Least-squares line through the (snapshot, direction) measurements: value at the last one and slope."""
        t = np.array([m[0] for m in meas], dtype=float)
        y = np.array([m[1] for m in meas])
        tc = t - t[-1]
        sl, ic = np.polyfit(tc, y, 1)
        return ic, sl

    def advance(self, th, v, tau):
        return th + v * tau


GEOM = None


def geometry(M):
    global GEOM
    if GEOM is None or GEOM.M != M:
        GEOM = ULAGeometry(M)
    return GEOM




# ----------------------------------------------------------------------------- the kernel
def maxsinr_responses(R, A, p):
    """Responses r = A^H w of the beamformer that maximizes sum_k p_k |w^H a_k|^2 / (w^H R w), i.e. of the largest
    generalized eigenvector of (A diag(p) A^H, R).  With G = A^H R^-1 A the responses satisfy G P r = lambda r; they
    are obtained from the Hermitian form P^1/2 G P^1/2.  Scale and common phase are arbitrary."""
    G = A.conj().T @ solve(R, A)
    G = 0.5 * (G + G.conj().T)
    sp = np.sqrt(p / p.max())
    _, U = eigh(sp[:, None] * G * sp[None, :])
    return U[:, -1] / sp


def variant_name(v):
    """'MPR k=0.3' for max-SINR targets, 'MPR k=0.3 q=0' for power-law targets."""
    if v[1] == 'msR':
        return 'MPRnoTS k=%g' % v[0]
    return 'MPR k=%g' % v[0] if v[1] == 'ms' else 'MPR k=%g q=%g' % (v[0], v[1])


def run_mpr_rab(X, a_initial, theta_init_rad, K_track=3, K_use=None, eta=ETA, use_tracking=True,
                   use_lifecycle=True, warmup=WARMUP, t_ref=T_REF, birth_hints=None,
                   exclude_zone_deg=3.0, variants=None, kappa=None, targets=None,
                   T_rec=T_REC, defl=DEFL, recon=True, verbose=False, eta_fast=-1, early=None, polish=None, predict=None,
                   target_sect=-1, peak=None):
    """MPR-RAB.  Returns (hist, log) for the default variant, or, when `variants` is a list of (kappa, targets)
    pairs (targets 'ms' or a number q), log['hist'][(kappa, targets)] holds the weight history of every variant,
    computed on the same tracked directions and the same reconstructed covariance (hist is then the first one).
    log['theta_hat'][k] are the directions of the constraints in force at snapshot k and log['p'][k] the Capon
    powers of the tracked clusters at the last update.  recon=False is the ablation that uses the loaded sample
    covariance as the objective; peak=False the ablation without the off-grid peak refinement.  Targets 'msR' are
    the max-SINR responses computed on the reconstruction itself (ablation of the reference covariance)."""
    M, T = X.shape
    g = geometry(M)
    if K_use is None:
        K_use = max(len(theta_init_rad), 2)
    K_use = min(K_use, K_track)
    birth_hints = list(birth_hints) if birth_hints else []
    if variants is None:
        variants = [(KAPPA if kappa is None else kappa, TARGETS if targets is None else targets)]
    eta_fast = ETA_FAST if eta_fast == -1 else eta_fast
    early = EARLY_UPDATES if early is None else early
    t_ref = T_rec if t_ref is None else t_ref
    polish = POLISH if polish is None else polish
    predict = PREDICT if predict is None else predict
    target_sect = TARGET_SECT if target_sect == -1 else target_sect
    peak = PEAK_REFINE if peak is None else peak
    big = g.As.shape[1] > SUB_MIN
    lag = eta / (1.0 - eta)                         # centroid delay of the exponentially weighted covariance (snapshots)
    theta_k = list(theta_init_rad)                  # tracked (measured) directions
    cluster_age = [0 for _ in theta_k]
    meas = [[] for _ in theta_k]                    # (snapshot, measured direction) per cluster
    th_use = list(theta_k)                          # directions of the constraints in force
    R = np.eye(M, dtype=complex)
    R_f = np.eye(M, dtype=complex)
    sec = g.sector()
    hists = {v: [] for v in variants}
    w = {v: a_initial / max(norm(a_initial), EPS) for v in variants}
    log = {'theta_hat': [], 'K_active': [], 'active': [], 'beta': [], 'mu': [], 'n_outer': [], 'p': []}
    log['solver'] = {v: [] for v in variants}
    need_update = True
    A_prev = {v: None for v in variants}
    last_act, last_beta, last_mu, last_n, last_p = [], [], 0.0, 0, []

    for k in range(T):
        x_k = X[:, k]
        R = eta * R + np.outer(x_k, x_k.conj())
        if eta_fast is not None:
            R_f = eta_fast * R_f + np.outer(x_k, x_k.conj())

        # ---- birth hints (external)
        for (t_birth, th_birth) in birth_hints:
            if t_birth == k:
                theta_k.append(float(th_birth))
                cluster_age.append(0)
                meas.append([])
                if len(theta_k) > K_use:
                    Rs_tmp = (R + R.conj().T) / 2
                    pws = [np.real(steer(M, th).conj() @ Rs_tmp @ steer(M, th)) / M for th in theta_k]
                    cand = [i for i in range(len(theta_k) - 1) if cluster_age[i] > GRACE] or list(range(len(theta_k) - 1))
                    weakest = cand[int(np.argmin([pws[i] for i in cand]))]
                    del theta_k[weakest]; del cluster_age[weakest]; del meas[weakest]
                need_update = True
        cluster_age = [a + 1 for a in cluster_age]

        # ---- lifecycle and refinement (every t_ref snapshots after the warm-up)
        if use_tracking and k > warmup and k % t_ref == 0 and theta_k:
            Rs = (R + R.conj().T) / 2
            Rinv_d = np.linalg.solve(Rs + LOAD_REF * np.trace(Rs).real / M * np.eye(M), np.eye(M))
            if use_lifecycle:
                if eta_fast is None:
                    local_p = [1.0 / max(np.real(steer(M, th).conj() @ Rinv_d @ steer(M, th)), 1e-10) for th in theta_k]
                else:
                    local_p = []
                    for th in theta_k:
                        a_t = steer(M, th); u = Rinv_d @ a_t; u = u / max(np.real(a_t.conj() @ u), 1e-10)
                        local_p.append(max(np.real(u.conj() @ R_f @ u), 1e-10))     # Capon output power, short window
                p_max = max(local_p)
                keep = [i for i, (p_loc, ag) in enumerate(zip(local_p, cluster_age)) if ag <= GRACE or p_loc > DEATH_FRAC * p_max]
                if keep and len(keep) < len(theta_k):
                    theta_k = [theta_k[i] for i in keep]
                    cluster_age = [cluster_age[i] for i in keep]
                    meas = [meas[i] for i in keep]
            # interferer-subspace deflation of the scan covariance
            Rd = Rs
            if defl > 0:
                ev, Uv = eigh(Rs)
                sig2 = np.mean(ev[:M // 2])
                Rd = Rs.copy()
                if DEFL_RULE == 'proj' and theta_k:
                    A_t = np.column_stack([steer(M, t) for t in theta_k])
                    P_A = A_t @ solve(A_t.conj().T @ A_t + 1e-9 * np.eye(len(theta_k)), A_t.conj().T)
                for i in range(M - 1, -1, -1):
                    if ev[i] <= 10 * sig2:
                        break
                    if DEFL_RULE == 'proj' and theta_k:
                        al = float(np.real(Uv[:, i].conj() @ P_A @ Uv[:, i]))
                    else:
                        al = max([np.abs(Uv[:, i].conj() @ steer(M, t)) ** 2 / M for t in theta_k] or [0.0])
                    if al < defl:
                        Rd = Rd - (ev[i] - sig2) * np.outer(Uv[:, i], Uv[:, i].conj())
            Rinv = np.linalg.solve(Rd + LOAD_REF * np.trace(Rd).real / M * np.eye(M), np.eye(M))
            p_of = lambda t: 1.0 / max(np.real(steer(M, t).conj() @ Rinv @ steer(M, t)), 1e-10)
            refined, new_meas = [], []
            for th_old, mh in zip(theta_k, meas):
                best_th, best_p = th_old, -np.inf
                for t in g.refine_candidates(th_old):
                    p = p_of(t)
                    if p > best_p:
                        best_p, best_th = p, t
                if polish:
                    best_th = g.polish(best_th, p_of)
                th_new = g.cap_step(th_old, best_th)
                if g.capped(th_old, best_th):          # an acquisition step, not drift: restart the slope
                    mh = []
                refined.append(th_new)
                new_meas.append((mh + [(k, th_new)])[-PRED_WIN:])
            theta_k, meas = refined, new_meas
            if len(theta_k) > K_use:                       # hard cap by received power
                pw = [np.real(steer(M, th).conj() @ Rs @ steer(M, th)) / M for th in theta_k]
                order = np.argsort(-np.array(pw))[:K_use]
                theta_k = [theta_k[i] for i in order]
                cluster_age = [cluster_age[i] for i in order]
                meas = [meas[i] for i in order]
            need_update = True

        if len(theta_k) > K_use:
            theta_k = theta_k[:K_use]
            cluster_age = cluster_age[:K_use]
            meas = meas[:K_use]
        K = len(theta_k)

        # ---- reconstruction, targets and exact solve
        if K and (need_update or k % T_rec == 0 or (early and k < T_rec)):
            Rs = (R + R.conj().T) / 2
            th_use = list(theta_k)
            if predict:
                for i, mh in enumerate(meas):
                    if len(mh) >= PRED_MIN:
                        z_fit, v_fit = g.fit(mh)
                        th_use[i] = g.advance(z_fit, v_fit, lag + (k - mh[-1][0]) + 0.5 * (T_rec - 1))
            A = np.column_stack([steer(M, t) for t in th_use])
            A_z = np.column_stack([steer(M, t) for t in theta_k])
            th_excl = list(th_use) + [t for t, tu in zip(theta_k, th_use) if t != tu]
            if recon:
                Ri_rec = np.linalg.inv(Rs + LOAD_REC * np.trace(Rs).real / M * np.eye(M))
                spec = 1.0 / np.maximum(np.real(np.sum(g.As.conj() * (Ri_rec @ g.As), axis=0)), 1e-12)
                msk = g.mask(th_excl, sec)
                pk_A, pk_w = None, None
                if peak:
                    pk_A, pk_w, drop = g.refine_peaks(spec, msk, Ri_rec, th_excl, sec)
                    if drop:
                        msk = msk.copy()
                        msk[drop] = False
                if big:
                    R_full = (g.As * spec) @ g.As.conj().T * g.dth
                    ex = ~msk
                    R_in = R_full - (g.As[:, ex] * spec[ex]) @ g.As[:, ex].conj().T * g.dth
                else:
                    R_in = (g.As[:, msk] * spec[msk]) @ g.As[:, msk].conj().T * g.dth
                if pk_A is not None:
                    R_in = R_in + (pk_A * pk_w) @ pk_A.conj().T * g.dth
                R_in = (R_in + R_in.conj().T) / 2
                R_in += LOAD_REC * np.trace(R_in).real / M * np.eye(M)
            else:
                R_in = Rs + LOAD_REC * np.trace(Rs).real / M * np.eye(M)
            Ri_p = np.linalg.inv(Rs + LOAD_REF * np.trace(Rs).real / M * np.eye(M))
            p = 1.0 / np.maximum(np.real(np.sum(A_z.conj() * (Ri_p @ A_z), axis=0)), 1e-12)
            R_n = R_in / (np.trace(R_in).real / M)
            Qr, Lr = real_form(R_n, A)
            s_eig, U_eig = eigh(0.5 * (Qr + Qr.T))
            Q_eig = (np.maximum(s_eig, 1e-14), U_eig)
            last_p = list(p)
            if any(v[1] == 'msR' for v in variants):
                r_r = maxsinr_responses(R_n, A, p)
                A_phr = A * np.exp(1j * np.angle(r_r))[None, :]
                L_msR = np.vstack([A_phr.real, A_phr.imag])
                beta_msR = np.abs(r_r) / np.abs(r_r).max()
            if any(v[1] == 'ms' for v in variants):
                R_t = R_n
                if recon and target_sect:
                    msk_t = g.mask(th_excl, target_sect * sec)
                    fl = np.median(spec)                                  # noise level of the spectrum
                    if big:
                        ex_t = ~msk_t
                        A_ex = g.As[:, ex_t]
                        R_t = R_full - (A_ex * spec[ex_t]) @ A_ex.conj().T * g.dth + fl * A_ex @ A_ex.conj().T * g.dth
                    else:
                        R_t = (g.As[:, msk_t] * spec[msk_t]) @ g.As[:, msk_t].conj().T * g.dth
                        R_t = R_t + fl * g.As[:, ~msk_t] @ g.As[:, ~msk_t].conj().T * g.dth
                    if peak:
                        pk_At, pk_wt, drop_t = g.refine_peaks(spec, msk_t, Ri_rec, th_excl, target_sect * sec)
                        if drop_t:
                            R_t = R_t - (g.As[:, drop_t] * spec[drop_t]) @ g.As[:, drop_t].conj().T * g.dth
                            R_t = R_t + (pk_At * pk_wt) @ pk_At.conj().T * g.dth
                    R_t = (R_t + R_t.conj().T) / 2
                    R_t += LOAD_REC * np.trace(R_t).real / M * np.eye(M)
                    R_t = R_t / (np.trace(R_t).real / M)
                r_ms = maxsinr_responses(R_t, A, p)
                A_ph = A * np.exp(1j * np.angle(r_ms))[None, :]
                L_ms = np.vstack([A_ph.real, A_ph.imag])
                beta_ms = np.abs(r_ms) / np.abs(r_ms).max()
            for v in variants:
                kap, tg = v
                nu = np.full(K, kap * np.sqrt(M / K))
                if tg == 'ms':
                    L_v, beta = L_ms, beta_ms
                elif tg == 'msR':
                    L_v, beta = L_msR, beta_msR
                else:
                    L_v, beta = Lr, (p / p.max()) ** tg
                A0 = A_prev[v] if (A_prev[v] is not None and len(A_prev[v]) <= K and (len(A_prev[v]) == 0 or A_prev[v].max() < K)) else None
                wr, n_out, act, mu = as_snc(Q_eig, L_v, nu, beta, A0=A0)
                log['solver'][v].append(dict(LAST_SOLVER_STATUS))
                if norm(wr) > 1e-12 and np.all(np.isfinite(wr)):
                    w[v] = wr[:M] + 1j * wr[M:]
                    A_prev[v] = act
                else:
                    w[v] = A.sum(axis=1) / max(norm(A.sum(axis=1)), EPS)
                    A_prev[v] = None
                if v == variants[0]:
                    last_act, last_beta, last_mu, last_n = list(act), list(beta), float(mu), int(n_out)
            need_update = False

        for v in variants:
            hists[v].append(w[v].copy())
        log['theta_hat'].append(list(th_use) if K else list(theta_k))
        if verbose:
            log.setdefault('theta_meas', []).append(list(theta_k))
        log['K_active'].append(K)
        log['active'].append(last_act)
        log['beta'].append(last_beta)
        log['mu'].append(last_mu)
        log['n_outer'].append(last_n)
        log['p'].append(last_p)

    for v, statuses in log['solver'].items():
        n_bad = sum(not st.get('certified', False) for st in statuses)
        if n_bad:
            warnings.warn('MPR-RAB variant %s: %d/%d updates are uncertified numerical approximations' % (v,n_bad,len(statuses)), RuntimeWarning)
    log['hist'] = hists
    return hists[variants[0]], log


def sinr_floor_db(w, theta_k_rad, power_k, R_ipn, ps):
    """Minimum-cluster expected SINR (not a worst-case uncertainty or instantaneous-fading bound): min over clusters with non-zero power of
    P_k |w^H a_k|^2 P_S / (w^H R_in w)."""
    den = np.real(w.conj() @ R_ipn @ w)
    if den < 1e-30:
        return -30.0
    vals = [10 * np.log10(max(p * ps * np.abs(w.conj() @ steer(len(w), th)) ** 2 / den, 1e-10))
            for th, p in zip(theta_k_rad, power_k) if p > 0]
    return min(vals) if vals else -30.0

