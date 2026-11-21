"""RDF-AS-ISAC: robust dual-function active-set receive beamformer with tracked-interference covariance
reconstruction (Section 7.5 and Algorithm 7.1 of Chapter 7).

Per snapshot k (O(M^2)), the forgetting covariance (7.13):
    R <- eta R + (1 - eta) x_k x_k^H,   eta = 0.96 at M = 32 and 1 - 1.28/M otherwise.

Every T_r = 10 snapshots (one refresh):
  1. Capon scan (7.14) of R on a 2-deg azimuth/elevation grid, evaluated exactly through the lag sums
     (7.15)-(7.16) of R^{-1}; local peaks above three times the noise level are refined on 0.25-deg and then
     0.05-deg grids and given the net power (7.17).
  2. Peak labelling with continuity: served users and targets keep the peak nearest their previous estimate,
     interference tracks keep their continuation peaks (gates 2.5 x 1.5 deg), unlabelled served directions claim
     the nearest free peak inside their a-priori window (+-9 x +-4 deg), and every other peak opens a new
     interference track. A served direction never claims a peak more than 6 dB above the nominal user SNR (power
     ceiling). After 2/(1 - eta) snapshots a user whose own peak lies outside its window may claim the nearest
     peak within twice the window whose power is within +-6 dB of its nominal SNR (extended claim), taking it over
     from an interference track if necessary.
  3. Each interference track is an alpha-beta filter (7.18a)-(7.18d) on azimuth and elevation (position gain
     0.5, rate parameter 0.3 on the smoothed increment) and is predicted ahead by h = eta/(1 - eta) + T_r/2 (7.19).
  4. Structured covariance (7.20) Q = I + sum_j (p_j / s2) a_j a_j^H over the predicted tracks matched at this
     refresh; peaks labelled as served signals never enter it.
  5. Exact solution of the worst-case problem (7.11)-(7.12) with every constraint kept, in the diagonal-loading
     form (7.23) w = (Q + g I)^{-1} L lam, g = nu^T lam / ||w||; lam from an N-dimensional nonnegative
     least-squares dual (7.23a), (g, ||w||) by a damped fixed point warm-started from the previous refresh.
     Each solve returns the duality-gap certificate (7.24), and the radial restoration (7.25) guarantees every
     constraint at the output.
Settings were chosen on development seeds (2001-2012 for the fixed scenes, 8001-8008 for random geometry).
No scene truth is used anywhere in this module.
"""
import numpy as np
from scipy.optimize import nnls

CONFIG = dict(eta=0.96, T=10, nu_mult=3.0, win_az=9.0, win_el=4.0, thr=3.0, beta_v=0.3,
              gate_az=2.5, gate_el=1.5, miss=3,
              ext=2.0, user_idx=None, user_snr_db=8.0, user_tol_db=6.0,      # user_idx None: all users
              ceiling=True,       # served directions never claim a peak above user_snr_db + user_tol_db
              alpha_p=0.5)        # track position smoothing (alpha-beta filter gain)


def upa_mat(Mx, My, az_deg, el_deg):
    az = np.deg2rad(np.atleast_1d(np.asarray(az_deg, float)))
    el = np.deg2rad(np.atleast_1d(np.asarray(el_deg, float)))
    ax = np.exp(1j*np.pi*np.outer(np.arange(Mx), np.sin(az)*np.cos(el)))
    ay = np.exp(1j*np.pi*np.outer(np.arange(My), np.sin(el)))
    return (ay[:, None, :]*ax[None, :, :]).reshape(Mx*My, -1)


class Scanner:
    """Coarse-to-fine Capon peak scan on a 2-deg angle grid; local peaks refined on a 0.25-deg grid and then on a
    0.05-deg grid (null depth at large M needs angle errors well below the beamwidth).
    The quadratic form a^H B a (B = R^{-1}) is evaluated through the Kronecker structure of the UPA steering
    vector a = a_y (x) a_x: with the lag sums c(p,q) of B, a^H B a = sum_{p,q} c(p,q) exp(j pi (p u + q v)),
    u = sin(az)cos(el), v = sin(el). This is exact; it costs (2Mx-1)(2My-1) per grid point instead of M^2."""
    def __init__(self, Mx=8, My=4, az=(-90, 90, 2.0), el=(-20, 20, 2.0), fine=2.0, fstep=0.25, fstep2=0.05):
        self.Mx, self.My = Mx, My
        self.fstep2 = fstep2
        if fstep2:
            d2 = np.arange(-fstep, fstep+1e-9, fstep2); D2A, D2E = np.meshgrid(d2, d2, indexing='ij')
            self.d2A, self.d2E = D2A.ravel(), D2E.ravel()
        self.az = np.arange(az[0], az[1]+1e-9, az[2]); self.el = np.arange(el[0], el[1]+1e-9, el[2])
        AZ, EL = np.meshgrid(self.az, self.el, indexing='ij'); self.shape = AZ.shape
        nx = np.tile(np.arange(Mx), My); ny = np.repeat(np.arange(My), Mx)
        self.pi = (nx[None, :] - nx[:, None]) + Mx - 1; self.qi = (ny[None, :] - ny[:, None]) + My - 1
        self.p = np.arange(-(Mx-1), Mx); self.q = np.arange(-(My-1), My)
        self.Eu, self.Ev = self._basis(AZ.ravel(), EL.ravel())
        d = np.arange(-fine, fine+1e-9, fstep); DA, DE = np.meshgrid(d, d, indexing='ij')
        self.dA, self.dE = DA.ravel(), DE.ravel()

    def _basis(self, az, el):
        az = np.deg2rad(az); el = np.deg2rad(el); u = np.sin(az)*np.cos(el); v = np.sin(el)
        return np.exp(1j*np.pi*np.outer(self.p, u)), np.exp(1j*np.pi*np.outer(self.q, v))

    def lag_sums(self, B):
        c = np.zeros((2*self.My-1, 2*self.Mx-1), complex); np.add.at(c, (self.qi, self.pi), B); return c

    @staticmethod
    def quad(c, Eu, Ev):
        return np.einsum('qg,qg->g', c @ Eu, Ev).real

    def peaks(self, R, thr=3.0, max_peaks=16):
        M = len(R); Rh = (R+R.conj().T)/2
        s2 = max(float(np.mean(np.linalg.eigvalsh(Rh)[:M//2])), 1e-12)
        Ri = np.linalg.inv(Rh + 1e-6*np.trace(Rh).real/M*np.eye(M)); c = self.lag_sums(Ri)
        P = (1/self.quad(c, self.Eu, self.Ev)).reshape(self.shape)
        pad = np.pad(P, 1, constant_values=-np.inf); na, ne = self.shape
        nb = np.max(np.stack([pad[1+i:1+i+na, 1+j:1+j+ne] for i in (-1, 0, 1) for j in (-1, 0, 1)
                              if (i, j) != (0, 0)]), 0)
        out = []
        for i, j in zip(*np.where((P > nb) & (P > thr*s2/M))):
            az = self.az[i]+self.dA; el = self.el[j]+self.dE
            q = 1/self.quad(c, *self._basis(az, el))
            b = int(np.argmax(q)); a0, e0, q0 = float(az[b]), float(el[b]), float(q[b])
            if self.fstep2:              # second refinement stage
                az2 = a0+self.d2A; el2 = e0+self.d2E; q2 = 1/self.quad(c, *self._basis(az2, el2))
                b = int(np.argmax(q2)); a0, e0, q0 = float(az2[b]), float(el2[b]), float(q2[b])
            out.append((a0, e0, q0-s2/M))
        out.sort(key=lambda t: -t[2]); merged = []
        for p in out:
            if all(abs(p[0]-q[0]) > 1.0 or abs(p[1]-q[1]) > 1.0 for q in merged): merged.append(p)
        return merged[:max_peaks], s2


class Tracker:
    """Desired/interference labelling with track continuity and smoothed angular rates."""
    def __init__(self, centers, win_az, win_el, gate_az=2.5, gate_el=1.5, miss=3,
                 ext=0.0, user_idx=(), user_snr_db=None, user_tol_db=6.0, kmin=0, ceiling=False,
                 alpha_p=1.0):
        self.alpha_p = alpha_p
        self.centers = list(centers); self.des = list(centers); self.trk = []
        self.wa, self.we, self.ga, self.ge, self.miss = win_az, win_el, gate_az, gate_el, miss
        self.ext, self.user_idx, self.usnr, self.utol, self.kmin = ext, set(user_idx), user_snr_db, user_tol_db, kmin
        self.claimed = [False]*len(self.centers)
        self.cap = (user_snr_db + user_tol_db) if (ceiling and user_snr_db is not None) else np.inf

    def _ok(self, p, s2):
        return 10*np.log10(max(p[2], 1e-12)/s2) <= self.cap

    def update(self, peaks, k, beta_v, s2=1.0):
        used_des = set(); trk_of = {}              # peak index -> interference track
        # 1) served directions re-claim the peak that continues their own estimate
        for j in range(len(self.centers)):
            if not self.claimed[j]: continue
            da, db = self.des[j]
            c = [(abs(p[0]-da)+abs(p[1]-db), i) for i, p in enumerate(peaks)
                 if i not in used_des and abs(p[0]-da) <= self.ga and abs(p[1]-db) <= self.ge and self._ok(p, s2)]
            if c: i = min(c)[1]; used_des.add(i); self.des[j] = tuple(peaks[i][:2])
        cont = set(used_des)
        # 2) interference tracks claim their continuation peaks
        for t in self.trk:                       # t = [az, el, p, misses, k_last, v_az, v_el]
            c = [(abs(p[0]-t[0])+abs(p[1]-t[1]), i) for i, p in enumerate(peaks)
                 if i not in used_des and i not in trk_of and abs(p[0]-t[0]) <= self.ga and abs(p[1]-t[1]) <= self.ge]
            if c:
                i = min(c)[1]; trk_of[i] = t; dk = k - t[4]
                pa, pe = t[0] + t[5]*dk, t[1] + t[6]*dk               # position predicted to this refresh
                na = pa + self.alpha_p*(peaks[i][0]-pa); ne = pe + self.alpha_p*(peaks[i][1]-pe)
                if dk > 0:
                    t[5] = (1-beta_v)*t[5] + beta_v*(na-t[0])/dk
                    t[6] = (1-beta_v)*t[6] + beta_v*(ne-t[1])/dk
                t[0], t[1], t[2], t[3], t[4] = na, ne, peaks[i][2], 0, k
            else:
                t[3] += 1
        # 3) unclaimed served directions take the nearest free peak inside their window
        for j, (a, b) in enumerate(self.centers):
            if self.claimed[j] and any(abs(peaks[i][0]-self.des[j][0]) <= 1e-9 and abs(peaks[i][1]-self.des[j][1]) <= 1e-9 for i in cont):
                continue
            da, db = self.des[j]
            c = [(((p[0]-da)/self.wa)**2+((p[1]-db)/self.we)**2, i) for i, p in enumerate(peaks)
                 if i not in used_des and i not in trk_of and abs(p[0]-a) <= self.wa and abs(p[1]-b) <= self.we
                 and self._ok(p, s2)]
            if not c and self.ext > 0 and j in self.user_idx and self.usnr is not None and k >= self.kmin:
                # 4) extended claim for a communication user whose own peak lies outside its window:
                # nearest peak within ext x window whose power matches the user's nominal received
                # power; it may take over an interference track (that track is then dropped)
                c = [(((p[0]-a)/self.wa)**2+((p[1]-b)/self.we)**2, i) for i, p in enumerate(peaks)
                     if i not in used_des and abs(p[0]-a) <= self.ext*self.wa and abs(p[1]-b) <= self.ext*self.we
                     and abs(10*np.log10(max(p[2], 1e-12)/s2) - self.usnr) <= self.utol]
                if c:
                    i = min(c)[1]
                    if i in trk_of: t = trk_of.pop(i); self.trk = [x for x in self.trk if x is not t]
            if c:
                i = min(c)[1]; used_des.add(i); self.des[j] = tuple(peaks[i][:2]); self.claimed[j] = True
        self.trk = [t for t in self.trk if t[3] <= self.miss]
        # 5) every other peak starts a new interference track
        for i, p in enumerate(peaks):
            if i not in used_des and i not in trk_of: self.trk.append([p[0], p[1], p[2], 0, k, 0.0, 0.0])

    def predicted(self, h):
        return [(t[0]+t[5]*h, t[1]+t[6]*h, t[2]) for t in self.trk if t[3] == 0]


def gains(w, L, nu):
    return (L.conj().T @ w).real - nu*np.linalg.norm(w)


def wc_loaded(R, L, nu, state=None, iters=300, tol=1e-9, damp=0.5, b=None):
    """MBWC worst-case problem  min w^H R w  s.t.  Re{L^H w} >= b + nu ||w||  in diagonal-loading form
    w = (R + g I)^{-1} L lam,  g = nu^T lam / ||w||  (exact optimum, N-dim active-set dual)."""
    if b is None: b = np.ones(L.shape[1])
    rho, U = np.linalg.eigh((R+R.conj().T)/2); Z = U.conj().T @ L
    g, t = (0.0, 0.0) if state is None else state
    for it in range(iters):
        d = 1/(rho+g); Q = (Z.conj().T @ (d[:, None]*Z)).real; Q = (Q+Q.T)/2
        A = np.linalg.cholesky(Q + 1e-12*np.trace(Q)/len(Q)*np.eye(len(Q))).T
        lam, _ = nnls(A, np.linalg.solve(A.T, b + nu*t), maxiter=1000)
        w = U @ (d*(Z @ lam)); tn = np.linalg.norm(w); gn = float(nu @ lam)/max(tn, 1e-30)
        done = abs(tn-t) <= tol*tn and abs(gn-g) <= tol*max(gn, 1e-12)
        t = (1-damp)*t + damp*tn; g = (1-damp)*g + damp*gn
        if done: break
    gg = gains(w, L, nu)
    if gg.min() <= 0: raise RuntimeError('worst-case solve did not reach a positive robust gain')
    wf = w*max(1.0, float(np.max(b/gg))*(1+1e-12))
    # Lagrangian lower bound: mu = 2 lam, s = nu^T mu, u = w/||w|| gives a dual-feasible point, so
    # D = b^T mu - (1/4) z^H R^{-1} z, z = L mu - s u, satisfies D <= optimum <= P = wf^H R wf.
    mu = 2*lam; s_ = float(nu @ mu); z = L @ mu - s_*w/np.linalg.norm(w); y = U.conj().T @ z
    D = float(b @ mu) - 0.25*float(np.sum(np.abs(y)**2/rho)); P = float(np.sum(rho*np.abs(U.conj().T @ wf)**2))
    return wf, (g, t), it+1, (P - D)/P


def gain_floors(alpha, NC, NS):
    """Communication priority: b_C = sqrt(max(2 alpha, 0.1)), b_S = sqrt(max(2(1-alpha), 0.1)); alpha=0.5 gives ones."""
    return np.r_[np.full(NC, np.sqrt(max(2*alpha, .1))), np.full(NS, np.sqrt(max(2*(1-alpha), .1)))]


def run_rdf_as_isac(X, L_C, L_S, nu0_C, nu0_S, centers, alpha=0.5, Mx=8, My=4, scanner=None,
                    return_diagnostics=False, **overrides):
    cfg = dict(CONFIG, **overrides)
    M, K = X.shape
    L = np.c_[L_C, L_S]; nu = cfg['nu_mult']*np.r_[nu0_C, nu0_S]
    b = gain_floors(alpha, L_C.shape[1], L_S.shape[1])
    if cfg['user_idx'] is None: cfg['user_idx'] = tuple(range(L_C.shape[1]//3))
    sc = scanner or Scanner(Mx, My)
    tr = Tracker(centers, cfg['win_az'], cfg['win_el'], cfg['gate_az'], cfg['gate_el'], cfg['miss'],
                 cfg['ext'], cfg['user_idx'], cfg['user_snr_db'], cfg['user_tol_db'], kmin=2/(1-cfg['eta']),
                 ceiling=cfg['ceiling'], alpha_p=cfg['alpha_p'])
    h = cfg['eta']/(1-cfg['eta']) + cfg['T']/2
    R = np.eye(M, dtype=complex)
    w, state, _, _ = wc_loaded(np.eye(M), L, nu, b=b)
    hist, diag = [], []; Qlast = np.eye(M, dtype=complex); iters = []; certs = []
    for k in range(K):
        x = X[:, k]; R = cfg['eta']*R + (1-cfg['eta'])*np.outer(x, x.conj())
        if k % cfg['T'] == cfg['T']-1:
            peaks, s2 = sc.peaks(R, thr=cfg['thr'])
            tr.update(peaks, k, cfg['beta_v'], s2)
            Rt = np.eye(M, dtype=complex)
            for a, e, p in tr.predicted(h):
                v = upa_mat(Mx, My, a, e)[:, 0]; Rt += (max(p, 0)/s2)*np.outer(v, v.conj())
            Qlast = Rt + 1e-6*np.eye(M)
            w, state, it, gap = wc_loaded(Qlast, L, nu, state=state, b=b); iters.append(it); certs.append(gap)
        hist.append(w.copy())
        if return_diagnostics:
            diag.append(dict(k=k, min_margin=float((gains(w, L, nu)-b).min()),
                             tracks=len(tr.predicted(h)) if k >= cfg['T']-1 else 0))
    if return_diagnostics:
        return hist, dict(per_snapshot=diag, Q_final=Qlast, solver_iterations=iters, certified_gaps=certs, b=b, nu=nu)
    return hist
