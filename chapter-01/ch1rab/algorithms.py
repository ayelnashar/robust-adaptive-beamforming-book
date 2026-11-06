"""
ch1rab.algorithms -- robust adaptive beamforming algorithms of Chapter 1.

Ellipsoidal (variable-loading) and joint-constraint robust Capon beamformers (IEEE TAP 2006), and the
adaptive worst-case (WC) and multiple-beam worst-case (MBWC) beamformers (IET Signal Processing 2008),
together with the reference beamformers they are compared with.

Conventions: steering vector a_n(theta) = exp(+j*2*pi*d*n*sin(theta)), n = 0..M-1; the covariance is
updated as R(k) = eta*R(k-1) + x x^H with R(0) = I.
"""
import numpy as np
from scipy.optimize import brentq, root

# --------------------------------------------------------------------------- basics
def steer(theta, M, d=0.5):
    n = np.arange(M)
    return np.exp(1j * 2 * np.pi * d * n * np.sin(theta))


def sinr_single(w, H, noise_db, user=0):
    """Single-path SINR: H columns are amplitude-scaled steering vectors; column `user` is the SOI."""
    s2 = 10 ** (noise_db / 10)
    num = 0.0
    den = s2 * np.real(np.vdot(w, w))
    for j in range(H.shape[1]):
        p = np.abs(np.vdot(w, H[:, j])) ** 2
        if j == user:
            num = p
        else:
            den += p
    return num / den


def sinr_coherent(w, a_eff, p_s, interf, noise_db):
    """Coherent multipath SINR. interf: list of amplitude-scaled interferer steering vectors."""
    s2 = 10 ** (noise_db / 10)
    den = s2 * np.real(np.vdot(w, w)) + sum(np.abs(np.vdot(w, h)) ** 2 for h in interf)
    return p_s * np.abs(np.vdot(w, a_eff)) ** 2 / den


def beampattern(w, M, thetas, d=0.5):
    g = np.array([np.abs(np.vdot(w, steer(t, M, d))) ** 2 for t in thetas])
    return 10 * np.log10(g / g.max())


def capon(IR, a):
    v = IR @ a
    return v / np.real(np.vdot(a, v))


def soi_power(w, R):
    """Li-Stoica SOI power estimate for any beamformer: a_hat = R w scaled to norm sqrt(M)."""
    M = len(w)
    Rw = R @ w
    return np.real(np.vdot(Rw, Rw)) / (M * np.real(np.vdot(w, Rw)))


class RLS:
    """R(k) = eta R(k-1) + x x^H and its inverse by the matrix inversion lemma (R(0) = I)."""

    def __init__(self, M, eta=1.0, delta=1.0):
        self.eta = eta
        self.IR = delta * np.eye(M, dtype=complex)
        self.R = (1 / delta) * np.eye(M, dtype=complex)
        self.k = 0

    def update(self, x):
        e = self.eta
        self.R = e * self.R + np.outer(x, x.conj())
        IRx = self.IR @ x
        den = np.real(e + np.vdot(x, IRx))
        K = IRx / den
        self.IR = (self.IR - np.outer(K, x.conj() @ self.IR)) / e
        self.IR = 0.5 * (self.IR + self.IR.conj().T)
        self.k += 1

    @property
    def Rn(self):
        """covariance normalised by the effective number of snapshots"""
        if self.eta == 1.0:
            return self.R / (self.k + 1)
        return self.R * (1 - self.eta)


# --------------------------------------------------------------------------- TAP 2006: ellipsoidal
def _vl_lambda(pp1, pp2, mu, eps):
    """Variable-loading multiplier from ||pp2 - mu*lam*pp1||^2 = eps (smaller root, (1.50))."""
    cc = np.real(np.vdot(pp2, pp2)) - eps
    bb = 2 * mu * np.real(np.vdot(pp2, pp1))
    aa = mu ** 2 * np.real(np.vdot(pp1, pp1))
    if aa <= 1e-300:
        return None
    disc = bb ** 2 - 4 * aa * cc
    return (bb - np.sqrt(max(disc, 0.0))) / (2 * aa)


class VLRCB:
    """Adaptive robust Capon beamformer with variable loading.
    method='SD' (steepest descent) or 'CG' (conjugate gradient). Optional joint quadratic weight constraint tau."""

    def __init__(self, a0, eps, alpha, method='SD', tau=None):
        self.a0 = a0.copy()
        self.eps, self.alpha, self.method, self.tau = eps, alpha, method, tau
        self.Pr = a0.copy()
        self.Gr = a0.copy()          # IR(0) = I
        self.G = np.zeros_like(a0)
        self.C = self.Gr.copy()
        self.n_vl = 0

    def _correct(self, Pr1, mu):
        pp1 = self.Pr - self.a0
        pp2 = Pr1 - self.a0
        if np.real(np.vdot(pp2, pp2)) > self.eps:
            lam = _vl_lambda(pp1, pp2, mu, self.eps)
            if lam is None:                         # first snapshot: pp1 = 0, project on the sphere
                Pr1 = self.a0 + np.sqrt(self.eps) * pp2 / np.linalg.norm(pp2)
                self.G = np.zeros_like(pp1)
            else:
                Pr1 = Pr1 - mu * lam * pp1
                self.G = lam * pp1
            self.n_vl += 1
        else:
            self.G = np.zeros_like(pp1)
        return Pr1

    def update(self, IR):
        if self.method == 'SD':
            D = self.Gr + self.G
            mu = np.real(self.alpha * np.vdot(D, D) / (np.real(np.vdot(D, IR @ D)) + 1e-8))
            self.Gr = IR @ self.Pr
            Pr1 = self.Pr - mu * self.Gr
            self.Pr = self._correct(Pr1, mu)
        else:  # conjugate gradient (Polak-Ribiere)
            C, Gp = self.C, self.Gr
            if not np.all(np.isfinite(C)):          # restart CG if the direction blew up
                C = self.C = Gp.copy()
            mu = np.real(self.alpha * np.vdot(C, Gp) / (np.real(np.vdot(C, IR @ C)) + 1e-7))
            Gn = IR @ self.Pr + self.G
            Pr1 = self.Pr - mu * C
            self.Pr = self._correct(Pr1, mu)
            beta = np.vdot(Gn - Gp, Gn) / (np.real(np.vdot(Gp, Gp)) + 1e-7)
            self.C = Gn + beta * C
            self.Gr = Gn
        w = capon(IR, self.Pr)
        wj = None
        if self.tau is not None:
            wj = w.copy()
            if np.real(np.vdot(wj, wj)) > self.tau:
                Va = IR @ wj
                a = np.real(np.vdot(Va, Va))
                b = -2 * np.real(np.vdot(Va, wj))
                c = np.real(np.vdot(wj, wj)) - self.tau
                gam = (-b - np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
                wj = wj - gam * Va
        return w, wj


def newton1(z, gm, eps, a, itmax=200):
    """Li-Stoica RCB loading by Newton's method."""
    az2 = np.abs(z) ** 2
    lam_prev = 0.0
    lam = (np.linalg.norm(a) - np.sqrt(eps)) / (np.max(gm) * np.sqrt(eps))
    for _ in range(itmax):
        f = np.sum(az2 / (1 + lam * gm) ** 2) - eps
        fp = np.sum(az2 * gm / (1 + lam * gm) ** 3)
        lam_new = lam + 0.5 * f / fp
        if not (lam_new - lam > 1e-6):
            lam = lam_new
            break
        lam_prev, lam = lam, lam_new
    return lam


def rcb_eig(R, IR, a0, eps):
    """Batch eigendecomposition RCB (Li-Stoica)."""
    gm, U = np.linalg.eigh(R)
    lam = newton1(U.conj().T @ a0, gm, eps, a0)
    M = len(a0)
    Pd = a0 - np.linalg.solve(np.eye(M) + lam * R, a0)
    return capon(IR, Pd), Pd


def noojacom(r, w, PM, beta=0.001, gama=0.01):
    """Normalized orthogonal Oja (NOOja) subspace tracker."""
    y = w.conj().T @ r
    z = w @ y
    p1 = r - z
    B1 = PM * beta / (np.linalg.norm(r) ** 2 - np.linalg.norm(y) ** 2 + gama)
    ny2 = np.linalg.norm(y) ** 2
    fay = 1 / np.sqrt(1 + B1 ** 2 * np.linalg.norm(p1) ** 2 * ny2)
    T = (fay - 1) / ny2
    p2 = T * z / B1 + fay * p1
    u1 = p2 / np.linalg.norm(p2)
    v1 = w.conj().T @ u1
    return w - 2 * np.outer(u1, v1.conj())


class RCBSubspace:
    """Recursive eigendecomposition RCB with subspace tracking (Es/En by NOOjacom)."""

    def __init__(self, M, L, rng):
        q1, _ = np.linalg.qr(rng.random((M, M - L)) + 1j * rng.random((M, M - L)))
        q2, _ = np.linalg.qr(rng.random((M, L)) + 1j * rng.random((M, L)))
        self.En, self.Es = q1, q2

    def update(self, r, R, IR, a0, eps):
        self.Es = noojacom(r, self.Es, 1)
        self.En = noojacom(r, self.En, -1)
        self.Es = self.Es[:, ::-1]
        E = np.hstack([self.En, self.Es])
        EV = np.real(np.diag(E.conj().T @ R @ E))
        lam = newton1(E.conj().T @ a0, EV, eps, a0)
        M = len(a0)
        Ps = a0 - np.linalg.solve(np.eye(M) + lam * R, a0)
        return capon(IR, Ps), Ps


class WCSOCP:
    """Worst-case robust beamformer of Vorobyov-Gershman-Luo by SOCP (cvxpy, complex R):
       min ||U w||  s.t.  delta*||w|| <= Re(w^H a) - 1."""

    def __init__(self, M):
        import cvxpy as cp
        self.cp = cp
        self.U = cp.Parameter((M, M), complex=True)
        self.a = cp.Parameter(M, complex=True)
        self.delta = cp.Parameter(nonneg=True)
        self.w = cp.Variable(M, complex=True)
        cons = [self.delta * cp.norm(self.w, 2) <= cp.real(self.a.conj() @ self.w) - 1]
        self.prob = cp.Problem(cp.Minimize(cp.norm(self.U @ self.w, 2)), cons)
        self.last = None

    def solve(self, R, a, delta):
        Rn = R / np.real(np.trace(R)) * len(a)
        L = np.linalg.cholesky(0.5 * (Rn + Rn.conj().T))
        self.U.value = L.conj().T
        self.a.value = a
        self.delta.value = delta
        try:
            self.prob.solve(solver='CLARABEL')
            if self.w.value is not None:
                self.last = self.w.value.copy()
        except Exception:
            pass
        if self.last is None:
            self.last = a / len(a)
        return self.last


# --------------------------------------------------------------------------- IET 2008: worst case
def wc_eig(R, a, zeta):
    """WC beamformer w = (R + zeta/t I)^-1 a, t solving sum |z|^2/(zeta + t gamma)^2 = 1 (exact)."""
    gm, U = np.linalg.eigh(R)
    gm = np.maximum(gm, 1e-12)
    z2 = np.abs(U.conj().T @ a) ** 2
    f = lambda t: np.sum(z2 / (zeta + t * gm) ** 2) - 1.0
    if f(0) <= 0:                      # zeta >= ||a||: infeasible, fall back to heavy loading
        return a.copy(), np.nan
    hi = 1.0
    while f(hi) > 0:
        hi *= 4
    t = brentq(f, 0, hi, xtol=1e-14, rtol=1e-12)
    M = len(a)
    return np.linalg.solve(R + (zeta / t) * np.eye(M), a), t


class WCAdaptive:
    """Adaptive MVDR beamformer with a single WC constraint (Table 1.4)."""

    def __init__(self, a, zeta, alpha=0.1, modified=False, dzeta=0.1, per_snapshot=True):
        """modified=True enables the zeta-reduction rule of the modified Step 4.
        per_snapshot=True lowers zeta by dzeta at most once per snapshot and
        keeps the previous weight while no real root exists; False lowers it within the snapshot
        until a real root appears."""
        self.a, self.zeta, self.alpha = a.copy(), zeta, alpha
        self.modified, self.dzeta, self.per_snapshot = modified, dzeta, per_snapshot
        self.w = None
        self.log = []          # per snapshot: A, B, C, lambda, zeta, status

    def _solve(self, w, w1, mu, zeta):
        a = self.a
        pi = a - zeta * w / np.linalg.norm(w)
        chi = np.real(np.vdot(w1, a)) - 1
        pa = np.real(np.vdot(pi, a))
        A = mu ** 2 * (pa ** 2 - zeta ** 2 * np.real(np.vdot(pi, pi)))
        B = mu * (chi * pa - zeta ** 2 * np.real(np.vdot(w1, pi)))
        C = chi ** 2 - zeta ** 2 * np.real(np.vdot(w1, w1))
        disc = B * B - A * C
        lam, status = None, 'no real root'
        if disc >= 0 and abs(A) > 1e-300:
            cands = sorted(r for r in ((-B + np.sqrt(disc)) / A, (-B - np.sqrt(disc)) / A) if r >= 0)
            status = 'spurious roots only' if cands else 'negative roots only'
            for r in cands:
                wt = w1 + mu * r * pi
                if np.real(np.vdot(wt, a)) >= zeta * np.linalg.norm(wt) + 1 - 1e-8:
                    lam, status = r, 'active'
                    break
        return lam, status, pi, A, B, C, disc

    def update(self, R, IR):
        a = self.a
        if self.w is None:
            self.w = IR @ a
        w = self.w
        grd = R @ w
        mu = self.alpha * np.real(np.vdot(grd, grd)) / (np.real(np.vdot(grd, R @ grd)) + 1e-9)
        w1 = w - mu * grd
        zeta = self.zeta
        if np.real(np.vdot(w1, a)) >= zeta * np.linalg.norm(w1) + 1:
            self.w = w1
            self.log.append([0.0, 0.0, 0.0, 0.0, zeta, 'inactive'])
            return self.w
        lam, status, pi, A, B, C, disc = self._solve(w, w1, mu, zeta)
        rec = [A, B, C]                          # parameters at the zeta the snapshot started with
        if lam is None and self.modified:
            # modified Step 4 (explicitly relaxed experiment): lower zeta and recompute A, B, C
            while lam is None and zeta > self.dzeta + 1e-12:
                zeta -= self.dzeta
                lam, status, pi, A, B, C, disc = self._solve(w, w1, mu, zeta)
                if self.per_snapshot:
                    break
            self.zeta = zeta
        if lam is not None:
            self.w = w1 + mu * lam * pi
        elif zeta >= np.linalg.norm(a):
            # Cauchy-Schwarz: |w^H a| <= ||a|| ||w|| < zeta ||w|| + 1 for every w, so the WC problem has
            # no solution at this zeta. Keep the previous weight (Step 4 safeguard).
            status = 'infeasible (zeta >= ||a||)'
        elif disc < 0 and A < 0 and -B / A > 0:
            # feasible zeta but no real root along pi: take the vertex lambda = -B/A, the step that comes
            # closest to the boundary
            lam, status = -B / A, 'vertex (no real root)'
            self.w = w1 + mu * lam * pi
        # else: keep the previous weight (Step 4 safeguard)
        self.log.append(rec + [np.nan if lam is None else lam, zeta, status])
        return self.w


class MBWCAdaptive:
    """Robust LCMV beamformer with multiple-beam WC constraints (Table 1.5).
    Lam: M x N presumed steering matrix; ups: N margins; ivec: N right-hand sides."""

    def __init__(self, Lam, ups, ivec, a_init, alpha=0.3):
        self.Lam, self.ups, self.ivec = Lam, np.asarray(ups, float), np.asarray(ivec, float)
        self.alpha, self.a_init = alpha, a_init
        self.w = None
        self.fail = 0

    def update(self, R, IR):
        Lam, ups, iv = self.Lam, self.ups, self.ivec
        if self.w is None:
            self.w = IR @ self.a_init
        w = self.w
        grd = R @ w
        mu = self.alpha * np.real(np.vdot(grd, grd)) / (np.real(np.vdot(grd, R @ grd)) + 1e-9)
        w1 = w - mu * grd
        if np.any(np.real(w1.conj() @ Lam) < ups * np.linalg.norm(w1) + iv):
            Lbar = Lam - np.outer(w1 / np.linalg.norm(w1), ups)          # (1.108) modified matrix

            def F(t):
                wt = w1 + mu * Lbar @ t
                return np.real(Lam.conj().T @ wt) - ups * np.linalg.norm(wt) - iv
            t0 = np.real(np.diag(np.linalg.pinv(Lam.conj().T @ IR @ Lam)))
            sol = root(F, t0, method='hybr')          # Powell hybrid trust-region method
            if sol.success and np.all(np.isfinite(sol.x)):
                self.w = w1 + mu * Lbar @ sol.x
            else:
                self.w = w1
                self.fail += 1
        else:
            self.w = w1
        return self.w


def lcmv(IR, C, g):
    """w = R^-1 C (C^H R^-1 C)^-1 g"""
    RC = IR @ C
    return RC @ np.linalg.solve(C.conj().T @ RC, g)
