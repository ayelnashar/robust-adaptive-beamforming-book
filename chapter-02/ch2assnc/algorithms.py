"""
ch2assnc.algorithms -- the AS-SNC solver of Chapter 2 and the conic reference models.

Problem class (SNC-SOCP, real data):

    minimize    w' Q w
    subject to  L[:, i]' w  >=  nu[i] * ||w||  +  beta[i],    i = 1, ..., N

with Q symmetric positive definite and nu >= 0. Every constraint carries the same norm ||w||.

AS-SNC works in the eigenbasis of Q. For a working set it solves a small multiplier system together
with two scalar equations in (mu, t), where t = ||w|| and mu = nu' lambda / t (Section 2.3), applies a
reduced Newton correction when the scalar loop stalls, and accepts a point only after a scaled KKT
residual test on the input problem. If that test is not met, the shared-epigraph conic model is
solved instead (Clarabel, then SCS) and the result is labelled as a fallback (Algorithm 2.1).
"""
import warnings

import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh
from scipy.optimize import nnls

__all__ = ['as_snc', 'quality', 'conic_solve', 'reference_solution', 'CONIC_BACKENDS']


# ----------------------------------------------------------------------------------------------
# residual report
# ----------------------------------------------------------------------------------------------
def quality(Q, L, nu, beta, w, lam=None):
    """Scaled KKT residuals of a candidate w for the input problem.

    Returns a dict with the scaled feasibility violation ('primal'), the scaled stationarity
    residual, the scaled complementarity residual, their maximum ('residual') and the objective.
    When the multipliers lam are not given they are estimated by nonnegative least squares on the
    nearly active constraints.
    """
    if w is None or not np.all(np.isfinite(w)):
        return dict(primal=np.inf, stationarity=np.inf, complementarity=np.inf, residual=np.inf)
    t = norm(w)
    slack = L.T @ w - nu * t - beta
    scale = 1 + abs(beta) + abs(L.T @ w) + nu * t
    G = L - np.outer(w / max(t, 1e-300), nu)
    grad = 2 * Q @ w
    if lam is None:
        A = np.flatnonzero(slack / scale < 1e-5)
        lam = np.zeros(len(beta))
        if len(A):
            lam[A] = nnls(G[:, A], grad, maxiter=max(1000, 30 * len(A)))[0]
    primal = float(np.max(np.maximum(-slack, 0) / scale, initial=0))
    stationarity = float(norm(grad - G @ lam) / (1 + norm(grad) + norm(G @ lam)))
    comp = float(np.max(abs(lam * slack), initial=0) / (1 + abs(w @ Q @ w)))
    return dict(primal=primal, stationarity=stationarity, complementarity=comp,
                residual=max(primal, stationarity, comp), objective=float(w @ Q @ w))


# ----------------------------------------------------------------------------------------------
# AS-SNC
# ----------------------------------------------------------------------------------------------
def _linear(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=1e-13)[0]


def _residual(s, B, nu, b, v, lam):
    """Scaled residual of the equality-constrained working-set system in the eigenbasis."""
    t = norm(v)
    if t < 1e-30:
        return np.inf
    stat = 2 * s * v - B @ lam + (nu @ lam) * v / t
    slack = B.T @ v - nu * t - b
    return max(norm(stat) / (1 + norm(2 * s * v) + norm(B @ lam)),
               np.max(abs(slack) / (1 + abs(b) + abs(B.T @ v) + nu * t), initial=0))


def _newton(s, B, nu, b, v, lam, tol):
    """Reduced Newton KKT correction for a fixed working set.

    The Hessian block 2 diag(s) + mu (I - u u') is inverted as a diagonal-plus-rank-one matrix, so
    each step needs only an a x a solve, a = |working set|. Backtracking on the scaled residual.
    """
    for k in range(40):
        merit = _residual(s, B, nu, b, v, lam)
        if merit < tol:
            return v, lam, True, k
        t = norm(v)
        u = v / t
        mu = (nu @ lam) / t
        if mu < -1e-12:
            return v, lam, False, k
        mu = max(mu, 0.)
        d = 1 / (2 * s + mu)
        den = 1 - mu * np.dot(u * d, u)
        if den <= 1e-14:
            return v, lam, False, k

        def hinv(x):
            if x.ndim == 1:
                return d * x + mu / den * (d * u) * np.dot(d * u, x)
            return d[:, None] * x + mu / den * np.outer(d * u, (d * u) @ x)

        G = B - np.outer(u, nu)
        stat = 2 * s * v - G @ lam
        slack = B.T @ v - nu * t - b
        HG = hinv(G)
        hs = hinv(stat)
        dl = _linear(G.T @ HG, -slack + G.T @ hs)
        dv = HG @ dl - hs
        step = 1.
        for _ in range(25):
            vn = v + step * dv
            ln = lam + step * dl
            if _residual(s, B, nu, b, vn, ln) < merit * (1 - 1e-4 * step):
                break
            step *= .5
        else:
            return v, lam, False, k
        v, lam = vn, ln
    return v, lam, False, 40


def as_snc(Q, L, nu, beta, max_outer=100, tol=1e-9, return_trace=False, return_info=False):
    """AS-SNC: spectral active-set method for SNC-SOCPs with real data (Algorithm 2.1).

    Parameters
    ----------
    Q : (M, M) symmetric positive definite cost matrix.
    L : (M, N) constraint directions (column i is l_i).
    nu : (N,) nonnegative uncertainty radii.
    beta : (N,) offsets.
    max_outer : limit on working-set updates.
    tol : requested accuracy; the acceptance gate on the scaled KKT residual is max(tol, 1e-10).
    return_trace : also return the candidate history (one primal vector per recorded candidate).
    return_info : also return a dict with the multipliers, residuals, the number of Newton steps
        and the provenance ('AS-SNC' for the spectral path, 'fallback:CLARABEL' or 'fallback:SCS').

    Returns
    -------
    (w, outer_iterations, active_set[, trace][, info]).  Raises RuntimeError if neither the spectral
    path nor the conic fallback passes the residual test (for example an infeasible problem).
    """
    if any(np.iscomplexobj(x) for x in (Q, L, nu, beta)):
        raise ValueError('Real inputs required')
    Q, L, nu, beta = [np.asarray(x, dtype=float) for x in (Q, L, nu, beta)]
    if (L.ndim != 2 or Q.shape != (L.shape[0], L.shape[0]) or nu.shape != (L.shape[1],)
            or beta.shape != nu.shape):
        raise ValueError('Input shapes')
    if not all(np.all(np.isfinite(x)) for x in (Q, L, nu, beta)) or np.any(nu < 0) or tol <= 0:
        raise ValueError('Invalid inputs')
    if not np.allclose(Q, Q.T, rtol=1e-12, atol=1e-14):
        raise ValueError('Q must be symmetric')

    # One eigendecomposition of the cost; every later solve works in its eigenbasis (Section 2.3.3).
    s, U = eigh(Q, check_finite=False, driver='evd')
    scale = s[-1]
    if s[0] <= 0:
        raise ValueError('Q must be positive definite')
    s = s / scale                               # eigenvalues of Q / lambda_max(Q)
    B = U.T @ L                                 # rotated constraint directions, computed once
    M, N = L.shape
    A = list(range(N))                          # working set
    trace = []
    newton_steps = 0
    gate = max(tol, 1e-10)                      # acceptance threshold (multipliers, final check)
    inner = min(1e-8, max(tol * .05, 1e-13))    # tolerance of the scalar loop and Newton correction

    # Nonpositive offsets: the origin is feasible, hence optimal.
    if np.all(beta <= 0):
        result = (np.zeros(M), 0, np.array([], int))
        info = dict(method='AS-SNC', newton_steps=0, lam=np.zeros(N),
                    **quality(Q, L, nu, beta, result[0], np.zeros(N)))
        if return_trace:
            result += ([result[0].copy()],)
        return result + (info,) if return_info else result

    mu, t = 1., 1.
    visits = {}
    reason = 'iteration limit'
    for outer in range(max_outer):
        if not A:
            A = [int(np.argmax(beta))]
        key = tuple(A)
        visits[key] = visits.get(key, 0) + 1
        if visits[key] > 3:
            reason = 'repeated working set'
            break
        BA = B[:, A]
        na = nu[A]
        ba = beta[A]
        ok = False
        # Scalar fixed-point loop in (mu, t); lambda and v are exact for the current pair.
        for k in range(20):
            D = 1 / (2 * s + mu)
            S = (BA.T * D) @ BA
            lam = _linear(S, na * t + ba)
            v = D * (BA @ lam)
            if np.min(lam) < -gate:
                break
            tn = norm(v)
            if tn < 1e-30:
                break
            mun = (na @ lam) / tn
            err = max(abs(tn - t) / (1 + tn), abs(mun - mu) / (1 + abs(mun)))
            mu, t = max(mun, 0.), tn
            if err < inner:
                ok = True
                break
        if return_trace:
            trace.append(U @ v)
        if np.min(lam) < -gate:                 # release the most negative multiplier
            A.pop(int(np.argmin(lam)))
            continue
        if not ok or _residual(s, BA, na, ba, v, lam) > inner:
            v, lam, ok, steps = _newton(s, BA, na, ba, v, lam, inner)
            newton_steps += steps
            if return_trace and steps:
                trace.append(U @ v)
        if np.min(lam) < -gate:
            A.pop(int(np.argmin(lam)))
            continue
        # Add the most violated inactive constraint, if any.
        slack = B.T @ v - nu * norm(v) - beta
        inactive = [i for i in range(N) if i not in A]
        if inactive:
            j = min(inactive, key=lambda i: slack[i] / (1 + abs(beta[i])))
            if slack[j] < -gate * (1 + abs(beta[j])):
                A.append(j)
                t = norm(v)
                mu = max((na @ lam) / max(t, 1e-30), 0.)
                continue
        # Certify the candidate on the input problem before returning it.
        w = U @ v
        full = np.zeros(N)
        full[A] = np.maximum(lam, 0) * scale
        cert = quality(Q, L, nu, beta, w, full)
        if cert['residual'] <= gate:
            result = (w, outer + 1, np.array(A, int))
            info = dict(method='AS-SNC', newton_steps=newton_steps, lam=full, **cert)
            if return_trace:
                result += (trace,)
            return result + (info,) if return_info else result
        reason = 'final KKT check'
        break

    # Recovery route: the shared-epigraph conic model, labelled as a fallback.
    import cvxpy as cp
    x = cp.Variable(M)
    tt = cp.Variable()
    cs = [cp.norm(x) <= tt, L.T @ x - nu * tt >= beta]
    problem = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(Q / scale))), cs)
    for backend in ['CLARABEL', 'SCS']:
        try:
            if backend == 'CLARABEL':
                opts = dict(tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11, max_iter=500)
            else:
                opts = dict(eps=1e-10, max_iters=200000)
            problem.solve(solver=backend, **opts)
            if x.value is None:
                continue
            lam = np.maximum(np.asarray(cs[1].dual_value), 0) * scale
            cert = quality(Q, L, nu, beta, x.value, lam)
            if cert['residual'] > max(gate, 1e-8):
                continue
            w = np.asarray(x.value)
            slack = L.T @ w - nu * norm(w) - beta
            A = np.flatnonzero(abs(slack) < 1e-6 * (1 + abs(beta)))
            result = (w, max_outer, A)
            info = dict(method='fallback:' + backend, reason=reason, newton_steps=newton_steps,
                        lam=lam, **cert)
            if return_trace:
                result += (trace + [w.copy()],)
            return result + (info,) if return_info else result
        except cp.error.SolverError:
            continue
    raise RuntimeError('No checked solution: ' + reason)


# ----------------------------------------------------------------------------------------------
# conic models used as baselines and as the independent reference
# ----------------------------------------------------------------------------------------------
CONIC_BACKENDS = ['SCS', 'ECOS', 'CLARABEL', 'MOSEK', 'GUROBI']


def _options(backend, tol):
    return {'SCS': dict(eps=tol, max_iters=200000),
            'ECOS': dict(abstol=tol, reltol=tol, feastol=tol, max_iters=500),
            'CLARABEL': dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol, max_iter=500),
            'MOSEK': dict(mosek_params={'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                                        'MSK_IPAR_NUM_THREADS': 1}),
            'GUROBI': dict(FeasibilityTol=max(tol, 1e-9), OptimalityTol=max(tol, 1e-9),
                           BarQCPConvTol=max(tol, 1e-9), Threads=1)}[backend]


def conic_solve(Q, L, nu, beta, backend, tol=1e-9):
    """Solve the shared-epigraph conic model (Proposition 2.2) through CVXPY with one backend.

    The model is  min w'Qw  s.t.  ||w|| <= t,  l_i'w - nu_i t >= beta_i.  Returns (w, info), where
    info holds the solver status, the backend solve time in seconds and, when available, the
    multipliers of the N linear constraints.
    """
    import cvxpy as cp
    w = cp.Variable(len(Q))
    t = cp.Variable()
    cs = [cp.norm(w) <= t] + [L[:, i] @ w - nu[i] * t >= beta[i] for i in range(len(beta))]
    p = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Q))), cs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        p.solve(solver=backend, **_options(backend, tol))
    info = {'status': p.status, 'backend_seconds': p.solver_stats.solve_time}
    if w.value is not None:
        dual = [c.dual_value for c in cs[1:]]
        if all(x is not None for x in dual):
            info['lam'] = np.maximum(np.array(dual, dtype=float), 0)
    return w.value, info


def reference_solution(Q, L, nu, beta):
    """Independent reference: SCS at 1e-11, cross-checked against Clarabel at 1e-11.

    Returns (w_ref, summary). Raises RuntimeError if the two solvers disagree by more than 1e-6 in
    relative objective or the SCS point violates the constraints by more than 1e-8 (scaled). The
    strong-convexity Lagrangian lower bound is a numerical diagnostic, not a certified bound.
    """
    x = (Q, L, nu, beta)
    w, info = conic_solve(*x, 'SCS', 1e-11)
    wc, ic = conic_solve(*x, 'CLARABEL', 1e-11)
    if w is None or wc is None:
        raise RuntimeError('Independent reference failed')
    q = quality(*x, w, info.get('lam'))
    obj = w @ Q @ w
    den = max(abs(obj), 1e-12)
    disagreement = abs(wc @ Q @ wc - obj) / den
    if q['primal'] > 1e-8 or disagreement > 1e-6:
        raise RuntimeError(f'Reference disagreement {disagreement}, primal {q}')
    lam = info['lam']
    t = np.linalg.norm(w)
    g = nu * t + beta - L.T @ w
    grad = 2 * Q @ w - L @ lam + (nu @ lam) * w / max(t, 1e-300)
    lb = obj + lam @ g - np.dot(grad, grad) / (4 * np.linalg.eigvalsh(Q)[0])
    return w, dict(objective=float(obj), primal=q['primal'],
                   cross_solver_objective_difference=float(disagreement),
                   numerical_dual_lower_bound=float(lb), numerical_relative_gap=float((obj - lb) / den))
