"""Conic and adaptive baselines of Section 7.7.2: exact SOCP and SDP solutions of the worst-case problem
(7.11)-(7.12) (B1, B2), VU-MBWC (B4) and the naive concatenated LCMV (B3)."""
import numpy as np
import scipy.linalg as sla
from scipy.linalg import eigh, solve
import cvxpy as cp


def run_dtras_virtual(X, L_all, nu0_all, eta=0.97, adapt_period=30, warmup=50, multiplier_ceiling=2.0):
    """VU-MBWC comparator (B4 in Section 7.7.2): the multi-beam worst-case recursion with the sensing columns
    treated as extra user constraints in one group, the first primary forced active, the frozen-norm Gram
    correction (7.8) with clipped multipliers, and the eigenstructure bound adaptation (7.9) with multiplier
    ceiling 2 at snapshot indices 60, 90, ..., 270. No priority weighting."""
    M, K = X.shape
    N = L_all.shape[1]
    nu = nu0_all.copy()
    nu0_base = nu0_all.copy()
    R = np.eye(M, dtype=complex)
    # Canonical non-directional initialization
    w = np.ones(M, dtype=complex) / np.sqrt(M)
    W_MAX = 2.0 * np.sqrt(M)
    hist = []
    for k in range(K):
        R = eta*R + np.outer(X[:, k], X[:, k].conj())
        if k > warmup and (k % adapt_period) == 0:
            Rs = (R + R.conj().T)/2
            ev, evec = eigh(Rs)
            e1 = evec[:, -1]
            for i in range(N):
                rho = np.abs(e1.conj() @ L_all[:, i])**2 / M
                nu[i] = 0.8*nu[i] + 0.2*nu0_base[i]*np.clip(1.0/max(rho, 0.05), 0.5, multiplier_ceiling)
        Rw = R @ w
        rr = np.real(Rw.conj() @ Rw)
        rRr = np.real(Rw.conj() @ R @ Rw)
        mu = 0.1*rr/max(rRr, 1e-20)
        w_hat = w - mu*Rw
        w_hat_n = max(np.linalg.norm(w_hat), 1e-15)
        # Primary = index 0, always active
        A = [0]
        for i in range(1, N):
            if np.real(w_hat.conj() @ L_all[:, i]) < nu[i]*w_hat_n + 1.0:
                A.append(i)
        if not A:
            w = w_hat
        else:
            L_A = L_all[:, A]
            d = np.array([(nu[i]*w_hat_n + 1.0) - np.real(w_hat.conj() @ L_all[:, i])
                          for i in A])
            G = np.real(L_A.conj().T @ L_A)
            try:
                tau = np.maximum(solve(G + 1e-6*np.eye(len(A)), d), 0)
            except np.linalg.LinAlgError:
                tau = np.zeros(len(A))
            w = w_hat + L_A @ tau
        if not np.all(np.isfinite(w)):
            w = L_all[:, 0]/np.linalg.norm(L_all[:, 0])
        nw = np.linalg.norm(w)
        if nw > W_MAX:
            w = w * (W_MAX / nw)
        hist.append(w.copy())
    return hist


def run_naive_lcmv(X, L_all, f=None, eta=0.97):
    """w = R^{-1} L (L^H R^{-1} L)^{-1} f, per snapshot."""
    M, K = X.shape
    N = L_all.shape[1]
    if f is None:
        f = np.ones(N)
    R = np.eye(M, dtype=complex)
    hist = []
    for k in range(K):
        R = eta*R + np.outer(X[:, k], X[:, k].conj())
        try:
            Rinv_L = solve(R + 1e-4*np.eye(M), L_all)
            w = Rinv_L @ solve(L_all.conj().T @ Rinv_L, f)
        except np.linalg.LinAlgError:
            w = L_all[:, 0]/np.linalg.norm(L_all[:, 0])
        hist.append(w.copy())
    return hist


def robust_margins(w, L, nu):
    return np.real(w.conj() @ L) - nu*np.linalg.norm(w) - 1.0


def _solve_robust(R_batch, L_C, L_S, nu_C, nu_S, use_sdp=False):
    """Exact convex fixed-bound WC problem; no surrogate or randomization.

    SDP represents ||w|| <= t by a Hermitian Schur-complement LMI.
    Both backends solve min w^H R w subject to Re(L^H w) >= 1+nu||w||.
    """
    M = R_batch.shape[0]
    L = np.column_stack([L_C,L_S]); nu = np.r_[nu_C,nu_S]
    R = (R_batch+R_batch.conj().T)/2 + 1e-4*np.eye(M)
    w = cp.Variable(M, complex=True)
    t = cp.Variable(nonneg=True)
    constraints = [cp.real(L.conj().T @ w) >= 1 + nu*t]
    if use_sdp:
        col = cp.reshape(w,(M,1),order='F')
        constraints += [cp.bmat([[cp.reshape(t,(1,1),order='F'), col.H],
                                [col,t*np.eye(M)]]) >> 0]
    else:
        constraints += [cp.norm(w,2) <= t]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(cp.abs(sla.cholesky(R,lower=False) @ w))),constraints)
    if not problem.is_dcp():
        raise AssertionError('The robust problem must be convex')
    if use_sdp:
        problem.solve(solver=cp.SCS, eps=1e-6, max_iters=50000, verbose=False)
    else:
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-8, tol_feas=1e-8)
    if problem.status not in ('optimal','optimal_inaccurate') or w.value is None:
        raise RuntimeError(f'Robust solve failed: {problem.status}')
    answer = np.asarray(w.value).ravel()
    # Reject materially infeasible solver output; do not silently substitute a beam.
    margin = robust_margins(answer,L,nu)
    if not np.isfinite(answer).all() or margin.min() < -1e-4:
        raise RuntimeError(f'Robust residual check failed: {margin.min()}')
    return answer


def run_sdp_isac(R_batch, L_C, L_S, nu_C, nu_S):
    return _solve_robust(R_batch,L_C,L_S,nu_C,nu_S,use_sdp=True)


def run_socp_isac(R_batch, L_C, L_S, nu_C, nu_S):
    return _solve_robust(R_batch,L_C,L_S,nu_C,nu_S,use_sdp=False)
