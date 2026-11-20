"""
ch5mprrab.baselines -- the comparators of Chapter 5 (Section 5.7.2).  Every direction-based method receives the
directions of the constraints of MPR-RAB at every snapshot and the Capon powers of the tracked clusters at its last
update (the shared tracking stage), and nothing else.

  lcmv_rule     LCMV-K with unit response on the K tracked directions: trace loading ('tr'), noise-floor loading
                ('nf', the LCMV-K of the tables) or Ledoit-Wolf shrinkage ('lw')
  parrcb_multi  Par-K-RCB: one robust Capon beamformer [Li, Stoica, Wang 2003] per cluster, summed
  lcmv_kd_dirs  LCMV-K-D: unit response and zero derivative on the K tracked directions
  incm_multi    INCM-LCMV, INCM-MVDR and INCM-GEV on the interference-plus-noise covariance reconstructed from the
                Capon spectrum [Gu and Leshem 2012], each with and without the off-grid peak refinement of MPR-RAB
"""
import numpy as np
from scipy.linalg import eigh as geigh
from . import channel as chn
from . import algorithms as alg


def lcmv_rule(X, dirs, rule, g, eta=chn.ETA, win=33):
    """LCMV-K with unity response at dirs[k]. rule 'tr': loading g*tr(R)/M; 'nf': g times the mean of the smallest M/2
    eigenvalues; 'lw': Ledoit-Wolf shrinkage of the windowed sample covariance (no parameter)."""
    Mx, T = X.shape
    R = np.eye(Mx, dtype=complex)
    out, key, C = [], None, None
    for k in range(T):
        x = X[:, k]
        R = eta * R + np.outer(x, x.conj())
        th = list(dirs[k])
        if key != tuple(th):
            C = np.column_stack([chn.steer(Mx, t) for t in th])
            key = tuple(th)
        f = np.ones(C.shape[1], dtype=complex)
        if rule == 'lw':
            Y = X[:, max(0, k - win + 1):k + 1]
            N = Y.shape[1]
            S = Y @ Y.conj().T / N
            mu = np.trace(S).real / Mx
            d2 = np.linalg.norm(S - mu * np.eye(Mx), 'fro') ** 2
            yn2 = np.sum(np.abs(Y) ** 2, axis=0)
            ySy = np.real(np.sum(Y.conj() * (S @ Y), axis=0))
            b2 = (np.sum(yn2 ** 2) - 2 * np.sum(ySy) + N * np.linalg.norm(S, 'fro') ** 2) / N ** 2
            rho = min(max(b2, 0.0), d2) / max(d2, 1e-30)
            Rl = rho * mu * np.eye(Mx) + (1 - rho) * S + 1e-9 * mu * np.eye(Mx)
        else:
            Rs = (R + R.conj().T) / 2
            dl = g * np.trace(Rs).real / Mx if rule == 'tr' else g * np.mean(np.linalg.eigvalsh(Rs)[:Mx // 2])
            Rl = Rs + dl * np.eye(Mx)
        try:
            RiC = np.linalg.solve(Rl, C)
            w = RiC @ np.linalg.solve(C.conj().T @ RiC + 1e-9 * np.eye(C.shape[1]), f)
        except np.linalg.LinAlgError:
            w = C @ f / max(np.linalg.norm(C @ f), chn.EPS)
        if not np.all(np.isfinite(w)):
            w = C @ f / max(np.linalg.norm(C @ f), chn.EPS)
        out.append(w)
    return out


def rcb_weight(vals, vecs, a, eps):
    """Robust Capon beamformer (Li, Stoica, Wang 2003), spherical uncertainty ||a - a_bar||^2 <= eps."""
    z = vecs.conj().T @ a
    z2 = np.abs(z) ** 2
    if eps >= z2.sum():
        return a / np.real(a.conj() @ a)
    g = lambda lam: float(np.sum(z2 / (1.0 + lam * vals) ** 2)) - eps
    lo, hi = 0.0, 1.0 / max(vals.min(), 1e-12)
    while g(hi) > 0:
        hi *= 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    a_hat = a - vecs @ (z / (1.0 + lam * vals))
    zh = vecs.conj().T @ a_hat
    w = vecs @ (zh / vals)
    return w / max(np.real(a_hat.conj() @ w), 1e-12)


def parrcb_multi(X, dirs, epss, beta=chn.BETA):
    """Par-K-RCB: one robust Capon beamformer per cluster, summed; several eps share one eigendecomposition."""
    Mx, T = X.shape
    R = chn.SIGMA2 * np.eye(Mx, dtype=complex)
    out = {e: [] for e in epss}
    for k in range(T):
        R = beta * R + (1 - beta) * np.outer(X[:, k], X[:, k].conj())
        Rs = (R + R.conj().T) / 2
        dl = 1e-3 * np.trace(Rs).real / Mx
        vals, vecs = np.linalg.eigh(Rs + dl * np.eye(Mx))
        vals = np.maximum(vals, 1e-12)
        aks = [chn.steer(Mx, th) for th in dirs[k]]
        for e in epss:
            w_sum = np.zeros(Mx, dtype=complex)
            for a_k in aks:
                w_sum = w_sum + rcb_weight(vals, vecs, a_k, e)
            if np.linalg.norm(w_sum) > chn.EPS:
                w_sum = w_sum / np.linalg.norm(w_sum)
            out[e].append(w_sum)
    return out


def lcmv_kd_dirs(X, dirs, gamma, eta=chn.ETA, h=1e-5):
    """LCMV-K-D with per-snapshot directions; azimuth derivative by central difference of the steering vector."""
    Mx, T = X.shape
    R = np.eye(Mx, dtype=complex)
    out, w, last, C, f = [], None, None, None, None
    for k in range(T):
        R = eta * R + np.outer(X[:, k], X[:, k].conj())
        th = tuple(dirs[k])
        if th != last:
            K = len(th)
            cols = [chn.steer(Mx, t) for t in th] + [(chn.steer(Mx, t + h) - chn.steer(Mx, t - h)) / (2 * h) for t in th]
            C = np.column_stack(cols)
            f = np.concatenate([np.ones(K), np.zeros(K)]).astype(complex)
            last = th
        Rs = (R + R.conj().T) / 2
        dl = gamma * np.trace(Rs).real / Mx
        try:
            RiC = np.linalg.solve(Rs + dl * np.eye(Mx), C)
            w = RiC @ np.linalg.solve(C.conj().T @ RiC + 1e-9 * np.eye(C.shape[1]), f)
        except np.linalg.LinAlgError:
            w = C[:, :C.shape[1] // 2].sum(axis=1)
        out.append(w)
    return out


def incm_multi(X, dirs, pows, keys, every=10, eta=chn.ETA):
    """Beamformers on the INCM reconstruction (Gu-Leshem type): the Capon spectrum of the loaded sample covariance is
    integrated over the grid of the array geometry outside +/-sect around every tracked direction.  keys are
    (mode, sect_deg) for the reconstruction on the grid and (mode, sect_deg, 'r') for the same reconstruction with
    the off-grid peak refinement of MPR-RAB (strong spectral peaks outside the excluded sectors located between the grid
    points, Section 5.4).  mode 'LCMV': K unit-gain constraints; 'MVDR': one constraint on the strongest tracked cluster
    by the tracker's Capon power; 'GEV': largest generalized eigenvector of (A diag(p) A^H, R_in), the max-SINR
    beamformer of the estimated model.  Updated every `every` snapshots and whenever the directions change; the
    covariance, its inverse and the spectrum are shared by all keys.  On large grids (the planar array) the full
    spectral sum is formed once per update and the excluded points are subtracted, the same sum in fewer operations."""
    geom = alg.geometry(X.shape[0])
    big = geom.As.shape[1] > alg.SUB_MIN
    Mx, T = X.shape
    R = np.eye(Mx, dtype=complex)
    out = {key: [] for key in keys}
    w = {key: None for key in keys}
    last = None
    for k in range(T):
        R = eta * R + np.outer(X[:, k], X[:, k].conj())
        th = tuple(dirs[k])
        if last is None or (k + 1) % every == 0 or th != last:
            Rs = (R + R.conj().T) / 2
            Ri = np.linalg.inv(Rs + 1e-6 * np.trace(Rs).real / Mx * np.eye(Mx))
            p = 1.0 / np.maximum(np.real(np.sum(geom.As.conj() * (Ri @ geom.As), axis=0)), 1e-12)
            A = np.column_stack([chn.steer(Mx, t) for t in th])
            pk = np.array(pows[k], dtype=float) if (pows is not None and len(pows[k]) == len(th)) else None
            if pk is None:
                Ri4 = np.linalg.inv(Rs + 1e-4 * np.trace(Rs).real / Mx * np.eye(Mx))
                pk = 1.0 / np.maximum(np.real(np.sum(A.conj() * (Ri4 @ A), axis=0)), 1e-12)
            Rin_cache, R_full = {}, None
            for key in keys:
                mode, sd, ref = key[0], key[1], len(key) > 2
                if (sd, ref) not in Rin_cache:
                    msk = geom.mask(th, np.deg2rad(sd))
                    pk_A = None
                    if ref:
                        pk_A, pk_w, drop = geom.refine_peaks(p, msk, Ri, list(th), np.deg2rad(sd))
                        if drop:
                            msk = msk.copy()
                            msk[drop] = False
                    if big:
                        if R_full is None:
                            R_full = (geom.As * p) @ geom.As.conj().T * geom.dth
                        ex = ~msk
                        Rin = R_full - (geom.As[:, ex] * p[ex]) @ geom.As[:, ex].conj().T * geom.dth
                    else:
                        Rin = (geom.As[:, msk] * p[msk]) @ geom.As[:, msk].conj().T * geom.dth
                    if pk_A is not None:
                        Rin = Rin + (pk_A * pk_w) @ pk_A.conj().T * geom.dth
                    Rin_cache[(sd, ref)] = (Rin + Rin.conj().T) / 2 + 1e-6 * np.trace(Rin).real / Mx * np.eye(Mx)
                Rin = Rin_cache[(sd, ref)]
                if mode == 'MVDR':
                    w[key] = np.linalg.solve(Rin, A[:, int(np.argmax(pk))])
                elif mode == 'GEV':
                    H = (A * pk) @ A.conj().T
                    w[key] = geigh((H + H.conj().T) / 2, Rin)[1][:, -1]
                else:
                    RiA = np.linalg.solve(Rin, A)
                    w[key] = RiA @ np.linalg.solve(A.conj().T @ RiA + 1e-9 * np.eye(A.shape[1]), np.ones(A.shape[1]))
            last = th
        for key in keys:
            out[key].append(w[key].copy())
    return out
