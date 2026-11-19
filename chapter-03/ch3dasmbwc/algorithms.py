"""
ch3dasmbwc.algorithms -- beamformers and metrics of Chapter 3.

Array model: uniform linear array, half-wavelength spacing, steering vectors with
the first element as phase reference (Eq. (3.2)).

Beamformers
  mvdr                 standard MVDR (Capon), Section 3.3.1
  dl_mvdr              diagonally loaded MVDR
  wc_mvdr              single worst-case beamformer [6], solved exactly (Section 3.3.3)
  adaptive_single_wc   adaptive single-WC beamformer of [12]
  inc_reconstruction   covariance-reconstruction comparator after Gu and Leshem [22]
  das_mbwc             DAS-MBWC, Algorithm 3.1 (dual-space active-set multiple-beam WC)
  mbwc_socp            exact solution of the multi-constraint problem (3.12) with SCS
  feasibility_margin   Slater margin t* of Eq. (3.A.4)
"""
import time
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq


# --------------------------------------------------------------------------- array and metrics
def steer(M, th):
    """ULA steering vector at angle `th` (radians), half-wavelength spacing, Eq. (3.2)."""
    return np.exp(1j * 2 * np.pi * 0.5 * np.arange(M) * np.sin(th))


def steer_err(M, th, gain=None, phase=None, pos=None):
    """Steering vector with element gain, phase and position perturbations.

    gain  : (M,) multiplicative gains, 1.0 = nominal
    phase : (M,) additive phase errors in radians
    pos   : (M,) element position deviations in wavelengths
    """
    idx = np.arange(M)
    p = 0.5 * idx if pos is None else 0.5 * idx + pos
    a = np.exp(1j * 2 * np.pi * p * np.sin(th))
    if phase is not None:
        a = a * np.exp(1j * phase)
    if gain is not None:
        a = a * gain
    return a


def sinr_db(w, a_true, R_ipn, ps):
    """Output SINR (dB) of a point source, Eq. (3.5): ps |w^H a|^2 / (w^H R_i+n w)."""
    num = ps * np.abs(w.conj() @ a_true) ** 2
    den = np.real(w.conj() @ R_ipn @ w)
    if den < 1e-30:
        return -20.0
    return 10 * np.log10(max(np.real(num) / np.real(den), 1e-10))


def desired_cov(M, th0, mp, ps):
    """Desired-signal covariance R_s of the multipath model (Section 3.2).

    The phases of the secondary rays are redrawn at every snapshot, so the rays add
    in power: R_s = ps * sum_n |g_n|^2 a(th0 + phi_n) a(th0 + phi_n)^H.
    """
    a0 = steer(M, th0)
    Rsd = ps * np.outer(a0, a0.conj())
    for dth, amp in (mp or []):
        ad = steer(M, th0 + dth)
        Rsd = Rsd + ps * amp ** 2 * np.outer(ad, ad.conj())
    return Rsd


def sinr_cov_db(w, Rsd, R_i):
    """Output SINR w^H R_s w / w^H R_i+n w (dB); equals sinr_db for a point source."""
    num = np.real(w.conj() @ Rsd @ w)
    den = np.real(w.conj() @ R_i @ w)
    if den < 1e-30:
        return -20.0
    return 10 * np.log10(max(num / den, 1e-10))


def sinr_opt_db(Rsd, R_i):
    """Clairvoyant optimum: largest generalized eigenvalue of (R_s, R_i+n), in dB."""
    return 10 * np.log10(np.max(np.real(np.linalg.eigvals(np.linalg.solve(R_i, Rsd)))))


def beampattern_db(w, M, angles):
    """Normalized beam pattern (dB, maximum 0 dB) of weight w over `angles` (radians)."""
    bp = np.array([20 * np.log10(max(np.abs(w.conj() @ steer(M, t)), 1e-15)) for t in angles])
    return bp - np.max(bp)


# --------------------------------------------------------------------------- batch baselines
def mvdr(R, a):
    """Standard MVDR (Capon) beamformer with a 1e-3 relative diagonal load for conditioning."""
    M = R.shape[0]
    dl = 1e-3 * np.trace(R).real / M
    Rinv_a = np.linalg.solve(R + dl * np.eye(M), a)
    return Rinv_a / (a.conj() @ Rinv_a)


def dl_mvdr(R, a, alpha):
    """Diagonally loaded MVDR with loading level alpha."""
    M = R.shape[0]
    Rinv_a = np.linalg.solve(R + alpha * np.eye(M), a)
    return Rinv_a / (a.conj() @ Rinv_a)


def wc_mvdr(R, a, zeta):
    """Single worst-case beamformer [6]:  min w^H R w  s.t.  Re{w^H a} >= zeta ||w|| + 1.

    Exact solution: w = v / (v^H R v) with v = (R + g I)^{-1} a, where the loading g > 0 is
    the unique root of g ||(R + g I)^{-1} a|| = zeta (requires zeta < ||a||), found from the
    eigendecomposition of R by a one-dimensional search. The constraint is active at the optimum.
    """
    Rh = (R + R.conj().T) / 2
    vals, vecs = eigh(Rh)
    ap = vecs.conj().T @ a
    if zeta >= np.linalg.norm(a):
        raise ValueError('zeta >= ||a||: worst-case constraint infeasible')
    f = lambda g: g * np.sqrt(np.sum(np.abs(ap) ** 2 / (vals + g) ** 2)) - zeta
    hi = max(1.0, float(np.max(vals)))
    while f(hi) < 0:
        hi *= 2.0
    g = brentq(f, 0.0, hi, xtol=1e-14 * hi, rtol=4 * np.finfo(float).eps, maxiter=200)
    v = vecs @ (ap / (vals + g))
    return v / np.real(v.conj() @ Rh @ v)


def inc_reconstruction(R, a_p, theta_p, sector_half_deg=6.0, n_scan=181):
    """Covariance-reconstruction comparator after Gu and Leshem [22].

    Reconstructs the interference-plus-noise covariance by integrating the Capon spectrum
    over the complement of a declared sector centred on the presumed direction, and takes
    the steering vector as the dominant eigenvector of the in-sector correlation. The
    declared sector (+-6 deg by default) covers the span of the constraint grid, so both
    methods are given the same uncertainty region. Cost O(M^3) per update.
    """
    Mv = R.shape[0]
    dl = 1e-3 * np.trace(R).real / Mv
    Rinv = np.linalg.inv(R + dl * np.eye(Mv))
    scan = np.linspace(-np.pi / 2, np.pi / 2, n_scan)
    in_sec = np.abs(scan - theta_p) <= np.deg2rad(sector_half_deg)

    R_inc = np.zeros((Mv, Mv), dtype=complex)            # Capon power outside the sector
    for t in scan[~in_sec]:
        at = steer(Mv, t)
        R_inc += (1.0 / np.real(at.conj() @ Rinv @ at)) * np.outer(at, at.conj())
    R_inc += dl * np.eye(Mv)

    C = np.zeros((Mv, Mv), dtype=complex)                # in-sector correlation
    for t in scan[in_sec]:
        at = steer(Mv, t)
        C += (1.0 / np.real(at.conj() @ Rinv @ at)) * np.outer(at, at.conj())
    _, evec = eigh((C + C.conj().T) / 2)
    a_hat = evec[:, -1]
    a_hat = a_hat * np.exp(-1j * np.angle(a_hat.conj() @ a_p))
    a_hat = a_hat / np.linalg.norm(a_hat) * np.sqrt(Mv)

    v = np.linalg.solve(R_inc, a_hat)
    return v / (a_hat.conj() @ v)


# --------------------------------------------------------------------------- adaptive beamformers
def adaptive_single_wc(X, a, zeta, eta=0.97, alpha=0.1):
    """Adaptive single-WC beamformer of [12]: one presumed steering vector, one radius.

    Gradient step with the scale-invariant step size (3.10), then a closed-form Lagrange
    multiplier from a quadratic equation whenever the WC constraint is violated (O(M) WC step).
    Returns the list of weight vectors, one per snapshot.
    """
    M, K = X.shape
    R = np.eye(M, dtype=complex)
    w = a / np.linalg.norm(a)
    hist = []
    for k in range(K):
        R = eta * R + np.outer(X[:, k], X[:, k].conj())
        Rw = R @ w
        rr = np.real(Rw.conj() @ Rw)
        rRr = np.real(Rw.conj() @ R @ Rw)
        mu = alpha * rr / max(rRr, 1e-20)
        wt = w - mu * Rw
        wtn = np.linalg.norm(wt)
        chi = np.real(wt.conj() @ a) - 1.0
        if chi < zeta * wtn:
            wn = max(np.linalg.norm(w), 1e-15)
            p = a - zeta * w / wn
            pHa = np.real(p.conj() @ a)
            pp = np.real(p.conj() @ p)
            pHwt = np.real(p.conj() @ wt)
            xv = np.real(wt.conj() @ a) - 1
            A = mu ** 2 * (pHa ** 2 - zeta ** 2 * pp)
            B = mu * (xv * pHa - zeta ** 2 * pHwt)
            C = xv ** 2 - zeta ** 2 * wtn ** 2
            disc = B ** 2 - A * C
            lam = 0.0
            if disc >= 0 and abs(A) > 1e-20:
                sd = np.sqrt(disc)
                cands = [x for x in ((-B + sd) / A, (-B - sd) / A) if np.isfinite(x) and x > 0]
                if cands:
                    lam = min(cands)
            w = wt + mu * lam * p
        else:
            w = wt
        if not np.all(np.isfinite(w)):
            w = a / np.linalg.norm(a)
        hist.append(w.copy())
    return hist


def das_mbwc(X, a, c_ang, nu0, eta=0.97, alpha=0.1, rescale=True, adapt=True, select='violated',
             smoothing=(0.8, 0.2), clip=(0.5, 2.0), period=30, warmup=50, ridge=1e-6, log=None):
    """DAS-MBWC, Algorithm 3.1.

    Per snapshot k:
      Step 1  covariance update R(k) = eta R(k-1) + x x^H (unnormalized, Eq. (3.4))
      Step 2  bound adaptation (3.20) at every `period`-th snapshot after `warmup` snapshots
              (first update at snapshot index 60 for the default values)
      Step 3  gradient step with the scale-invariant step size (3.10)
      Step 4  active set A = {i : g_i(w~) > 0}, rebuilt from scratch
      Step 6  deficit d, Gram solve (G + ridge I) tau = d with G = Re{L_A^H L_A}, negative tau set to 0
      Step 7  w = w~ + L_A tau
      Step 8  rescaling: if ||w|| > 1e12, w <- 1e6 w / ||w||

    Parameters
    ----------
    X        : (M, K) snapshots
    a        : (M,) presumed steering vector
    c_ang    : constraint directions (radians), length N
    nu0      : (N,) reference bounds
    smoothing: (weight on previous bound, weight on target) of (3.20)
    clip     : clipping range of the target in (3.20)
    select   : 'violated' (DAS-MBWC) or 'all' (all-constraint ablation: every constraint held
               at equality whenever any is violated, coefficients not clipped)
    log      : optional dict; records the active set per snapshot ('A'), the correction ratio
               ||L_A tau|| / ||w~|| ('dw', as (k, ratio)) and, if log['time'] is true, the time
               of each Gram correction ('tproj')

    Returns the list of K weight vectors.
    """
    M, K = X.shape
    Nc = len(c_ang)
    L = np.column_stack([steer(M, tc) for tc in c_ang])
    y0 = np.asarray(nu0, dtype=float)
    y = y0.copy()
    rho, rho_c = smoothing
    lo, hi = clip
    R = np.eye(M, dtype=complex)
    w = a / np.linalg.norm(a)
    hist = []
    for k in range(K):
        R = eta * R + np.outer(X[:, k], X[:, k].conj())

        if adapt and k > warmup and k % period == 0:
            _, evec = eigh((R + R.conj().T) / 2)
            e1 = evec[:, -1]                                  # dominant eigenvector
            for i in range(Nc):
                proj = np.abs(e1.conj() @ L[:, i]) ** 2 / M
                y[i] = rho * y[i] + rho_c * y0[i] * np.clip(1.0 / max(proj, 0.05), lo, hi)

        Rw = R @ w
        rr = np.real(Rw.conj() @ Rw)
        rRr = np.real(Rw.conj() @ R @ Rw)
        mu = alpha * rr / max(rRr, 1e-20)
        wt = w - mu * Rw

        wtn = max(np.linalg.norm(wt), 1e-15)
        act = [i for i in range(Nc) if np.real(wt.conj() @ L[:, i]) < y[i] * wtn + 1.0]
        if select == 'all' and act:
            act = list(range(Nc))
        if log is not None:
            log.setdefault('A', []).append(tuple(act))

        if not act:
            w = wt
        else:
            tp = time.perf_counter() if (log is not None and log.get('time')) else None
            Na = len(act)
            La = L[:, act]
            d = np.zeros(Na)
            for j, idx in enumerate(act):
                d[j] = (y[idx] * wtn + 1.0) - np.real(wt.conj() @ L[:, idx])
            G = np.real(La.conj().T @ La)
            try:
                tau = np.linalg.solve(G + ridge * np.eye(Na), d)
                if select != 'all':
                    tau = np.maximum(tau, 0)
            except np.linalg.LinAlgError:
                tau = np.zeros(Na)
            w = wt + La @ tau
            if tp is not None:
                log.setdefault('tproj', []).append(time.perf_counter() - tp)
            if log is not None:
                log.setdefault('dw', []).append(
                    (k, float(np.linalg.norm(La @ tau) / max(np.linalg.norm(wt), 1e-300))))

        if rescale:
            n = np.linalg.norm(w)
            if n > 1e12:
                w = w * (1e6 / n)
        if not np.all(np.isfinite(w)):
            w = a / np.linalg.norm(a)
        hist.append(w.copy())
    return hist


# --------------------------------------------------------------------------- convex references
def mbwc_socp(Rs, Lc, nu, solver='SCS', eps=1e-5):
    """Multi-constraint worst-case problem (3.12) solved exactly with CVXPY:

        min ||U w||  s.t.  Re{l_i^H w} >= nu_i ||w|| + 1,   R = U^H U.

    Returns (w, status, max constraint residual, solver time, wall time).
    """
    import cvxpy as cp
    Mv = Rs.shape[0]
    U = np.linalg.cholesky(Rs).conj().T
    w = cp.Variable(Mv, complex=True)
    cons = [cp.real(cp.conj(Lc[:, i]) @ w) >= nu[i] * cp.norm(w, 2) + 1 for i in range(Lc.shape[1])]
    prob = cp.Problem(cp.Minimize(cp.norm(U @ w, 2)), cons)
    t0 = time.perf_counter()
    if solver == 'SCS':
        prob.solve(solver='SCS', eps_abs=eps, eps_rel=eps)
    else:
        prob.solve(solver=solver)
    wall = time.perf_counter() - t0
    wv = w.value
    res = float(np.max(nu * np.linalg.norm(wv) + 1 - np.real(wv.conj() @ Lc))) if wv is not None else np.nan
    return wv, prob.status, res, prob.solver_stats.solve_time, wall


def feasibility_margin(Lc, nu):
    """Slater margin of Eq. (3.A.4): t* = max_{||w|| <= 1} min_i (Re{w^H l_i} - nu_i).

    t* > 0: a strictly feasible point exists (Theorem 3.3); t* <= 0: (3.12) has no feasible point.
    """
    import cvxpy as cp
    M = Lc.shape[0]
    wr, wi, t = cp.Variable(M), cp.Variable(M), cp.Variable()
    cons = [cp.norm(cp.hstack([wr, wi]), 2) <= 1]
    for i in range(Lc.shape[1]):
        cons.append(wr @ np.real(Lc[:, i]) + wi @ np.imag(Lc[:, i]) - nu[i] >= t)
    prob = cp.Problem(cp.Maximize(t), cons)
    try:
        prob.solve(solver='CLARABEL')
    except Exception:
        prob.solve()
    return float(t.value)
