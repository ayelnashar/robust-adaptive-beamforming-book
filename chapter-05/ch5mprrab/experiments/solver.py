"""Iteration counts of the active-set KKT solver at the selected margin with the max-SINR targets, and at kappa = 0.7
and 0.9: outer passes per update and inner fixed-point passes to the 1e-11 tolerance, two runs per scenario at the
operating point, linear array (argv[1] = 'ula') or planar array ('upa') (Section 5.3.5).  Writes DATA/solver_<array>.json:
    python -m ch5mprrab.experiments.solver <ula|upa>"""
import os, sys, json
import numpy as np
from . import _common  # noqa: F401
from .. import algorithms as alg
from .. import scenarios as MS

arr = sys.argv[1] if len(sys.argv) > 1 else 'ula'
MS.use_array(arr)
sel = MS.load_selection(arr)
k_sel = MS.kappa_of(sel['MPR-RAB'])
stats = {}
base_fixed = alg._solve_fixed_A
base_as = alg.as_snc
cur = {}


def fixed(Q_eig, L_A, nu_A, beta_A, max_iter=100, tol=1e-11, mu_init=None, t_init=None):
    s, U = Q_eig
    LU = U.T @ L_A
    nA = L_A.shape[1]
    mu = 1.0 if mu_init is None else mu_init
    t = 1.0 if t_init is None else t_init
    n = max_iter
    for it in range(max_iter):
        D = 1.0 / (2.0 * s + mu)
        M_lam = (LU.T * D) @ LU
        lam_new = np.linalg.solve(M_lam + 1e-14 * np.eye(nA), nu_A * t + beta_A)
        if np.any(lam_new < -1e-10):
            n = it + 1
            break
        lam_new = np.maximum(lam_new, 0.0)
        w_new = U @ (D * (LU @ lam_new))
        t_new = np.linalg.norm(w_new)
        mu_new = (nu_A @ lam_new) / max(t_new, 1e-300)
        err = abs(t_new - t) + abs(mu_new - mu)
        mu, t = mu_new, t_new
        if err < tol * max(1.0, t):
            n = it + 1
            break
    cur['inner'].append(n)
    return base_fixed(Q_eig, L_A, nu_A, beta_A, max_iter, tol, mu_init, t_init)


def as_snc(*a, **k):
    r = base_as(*a, **k)
    cur['outer'].append(int(r[1]))
    cur['status'].append(dict(alg.LAST_SOLVER_STATUS))
    return r


alg._solve_fixed_A = fixed
alg.as_snc = as_snc
for kap in sorted({k_sel, 0.7, 0.9}):
    cur = {'inner': [], 'outer': [], 'status': []}
    for sc in MS.cfg()['GEOS']:
        for rr in range(2):
            d = MS.make_data(sc, 5, 10, 0, 777000 + 100 * MS.cfg()['GEOS'].index(sc) + rr)
            c = d['c']
            alg.run_mpr_rab(d['X'], d['a0'], list(d['th0']), K_track=max(c['K_track'], len(d['th0'])), K_use=c['K_use'],
                              use_tracking=True, use_lifecycle=c['lc'], birth_hints=d['bh'], variants=[(kap, 'ms')])
    o, i = np.array(cur['outer']), np.array(cur['inner'])
    stats['%g' % kap] = dict(updates=int(len(o)), outer1=float(np.mean(o == 1)), outer_le2=float(np.mean(o <= 2)), outer_max=int(o.max()),
                             inner_p5=float(np.percentile(i, 5)), inner_p50=float(np.percentile(i, 50)), inner_p95=float(np.percentile(i, 95)),
                             inner_max=int(i.max()), inner_capped=float(np.mean(i >= 100)))
    stats['%g' % kap]['uncertified_updates'] = sum(not x['certified'] for x in cur['status'])
    stats['%g' % kap]['inner_not_converged'] = sum(not x['inner_converged'] for x in cur['status'])
    for metric in ('primal','dual','stationarity','complementarity'):
        stats['%g' % kap]['max_' + metric] = max(x.get(metric, float('inf')) for x in cur['status'])
    print(arr, 'kappa', kap, stats['%g' % kap], flush=True)
os.makedirs(MS.DATA, exist_ok=True)
json.dump({'kappa_sel': k_sel, 'stats': stats}, open(os.path.join(MS.DATA, 'solver_%s.json' % arr), 'w'), indent=1)
