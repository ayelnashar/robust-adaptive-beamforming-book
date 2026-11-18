"""
ch2assnc.scenarios -- problem generators, data and the experiments behind Figures 2.1-2.8.

Every experiment uses a fixed seed and returns plain records (lists and dicts) in the format of the
archived run in ``data/archived_run``, so the same plotting code draws either set of records.

Timing protocol (Section 2.7.1): one warm-up call, then one timed call. AS-SNC times are end to end
(eigendecomposition, iterations and the final residual check); conic baselines are timed through
CVXPY (model construction, canonicalization and backend solve). BLAS is limited to one thread.
Accuracy is measured outside the timings, against an independent tight SCS solution that is
cross-checked with Clarabel.
"""
import hashlib
import io
import time
import warnings
from importlib import resources

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .algorithms import as_snc, quality, conic_solve, reference_solution

METHODS = ['AS-SNC', 'SCS', 'ECOS', 'CLARABEL', 'MOSEK', 'GUROBI']
LABELS = {'CLARABEL': 'Clarabel', 'GUROBI': 'Gurobi'}       # display names used in the chapter


def label(method):
    return LABELS.get(method, method)


# ----------------------------------------------------------------------------------------------
# synthetic SNC-SOCP instances
# ----------------------------------------------------------------------------------------------
def random_instance(M, N, kappa, rng, nu_scale=0.2, sparse_A=True):
    """Random SNC-SOCP instance with controllable active-set sparsity.

    Q has eigenvalues logarithmically spaced in [1/kappa, 1] and a random orthonormal eigenbasis;
    the columns of L are random unit vectors. With sparse_A=True the first ceil(sqrt(N)) constraints
    get radii in [0.1, nu_scale] and offsets in [0.8, 1.2] (intended active), the others radii in
    [0.01, 0.05] and offsets in [0.05, 0.2] (decoys).
    """
    U = np.linalg.qr(rng.standard_normal((M, M)))[0]
    s = np.logspace(-np.log10(kappa), 0, M)
    Q = U @ np.diag(s) @ U.T
    Q = 0.5 * (Q + Q.T)

    if not sparse_A:
        L = rng.standard_normal((M, N))
        L /= np.maximum(np.linalg.norm(L, axis=0, keepdims=True), 1e-12)
        nu = rng.uniform(0.05, nu_scale, size=N)
        beta = rng.uniform(0.5, 1.5, size=N)
        return Q, L, nu, beta

    n_active = max(2, int(np.ceil(np.sqrt(N))))
    L = rng.standard_normal((M, N))
    L /= np.maximum(np.linalg.norm(L, axis=0, keepdims=True), 1e-12)
    nu = np.zeros(N)
    beta = np.zeros(N)
    nu[:n_active] = rng.uniform(0.1, nu_scale, size=n_active)
    beta[:n_active] = rng.uniform(0.8, 1.2, size=n_active)
    nu[n_active:] = rng.uniform(0.01, 0.05, size=N - n_active)
    beta[n_active:] = rng.uniform(0.05, 0.2, size=N - n_active)
    return Q, L, nu, beta


# ----------------------------------------------------------------------------------------------
# Fama-French test portfolios (Section 2.7.9)
# ----------------------------------------------------------------------------------------------
FF_INDUSTRIES_12 = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'Chems', 'BusEq',
                    'Telcm', 'Utils', 'Shops', 'Hlth', 'Money', 'Other']
FF_SIZE_VALUE_9 = ['S1V1', 'S1V3', 'S1V5', 'S3V1', 'S3V3', 'S3V5', 'S5V1', 'S5V3', 'S5V5']
FF_SIZE_MOM_9 = ['S1M1', 'S1M3', 'S1M5', 'S3M1', 'S3M3', 'S3M5', 'S5M1', 'S5M3', 'S5M5']
FF_SETS = {'FF-Ind12': FF_INDUSTRIES_12,
           'FF-All21': FF_INDUSTRIES_12 + FF_SIZE_VALUE_9,
           'FF-All30': FF_INDUSTRIES_12 + FF_SIZE_VALUE_9 + FF_SIZE_MOM_9}


def load_french_panel():
    """Monthly decimal returns, January 1949 to March 2017 (819 observations).

    Source: K. R. French Data Library, as distributed with the `linearmodels` package.
    """
    raw = resources.files('ch2assnc').joinpath('data/french_portfolios.csv.bz2').read_bytes()
    return pd.read_csv(io.BytesIO(raw), compression='bz2', parse_dates=['dates']).set_index('dates')


def load_ff(name, start='1949-01-01', end='2017-03-01'):
    """Return panel of one benchmark set: 'FF-Ind12', 'FF-All21' or 'FF-All30'."""
    if name not in FF_SETS:
        raise ValueError(f'Unknown set {name!r}; choose from {list(FF_SETS)}.')
    df = load_french_panel().loc[start:end, FF_SETS[name]]
    return df.dropna(how='any')


def build_portfolio_problem(returns_df, r_target=0.008, nu_return=0.005, nu_group=0.001,
                            n_groups=3, group_target=0.20):
    """Robust portfolio SNC-SOCP from a return panel.

    Risk:    Sigma = sample covariance, symmetrized and lifted by a small spectral floor.
    Return:  mu_hat' w - nu_return ||w|| >= r_target              (monthly, decimal returns)
    Groups:  (group average)' w - nu_group ||w|| >= group_target    (n_groups contiguous groups)

    A free-scale solver benchmark: no budget equality and no long-only restriction.
    Returns Sigma, L, nu, beta, mu_hat with L of shape (M, 1 + n_groups).
    """
    R = returns_df.values
    mu_hat = R.mean(axis=0)
    Sigma = np.cov(R.T)
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma += 1e-6 * np.trace(Sigma) / Sigma.shape[0] * np.eye(Sigma.shape[0])

    M = R.shape[1]
    group_size = M // n_groups
    Lg = np.zeros((M, n_groups))
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else M
        idx = np.arange(start, end)
        Lg[idx, g] = 1.0 / len(idx)

    L = np.column_stack([mu_hat, Lg])
    nu = np.concatenate([[nu_return], np.full(n_groups, nu_group)])
    beta = np.concatenate([[r_target], np.full(n_groups, group_target)])
    return Sigma, L, nu, beta, mu_hat


# ----------------------------------------------------------------------------------------------
# synthetic 50-asset allocation (Section 2.7.8)
# ----------------------------------------------------------------------------------------------
def portfolio_model(seed=2050, M=50):
    """Three-factor covariance, five sectors of ten assets, expected-return floor 0.15 with zero
    radius, sector-exposure floors 0.02. Returns Q, L, beta, mu."""
    rng = np.random.default_rng(seed)
    F = rng.normal(0, .04, (M, 3))
    Q = F @ F.T + np.diag(rng.uniform(.01, .03, M) ** 2)
    mu = rng.uniform(.08, .18, M)
    Lg = np.zeros((M, 5))
    for g in range(5):
        Lg[g * 10:(g + 1) * 10, g] = .1
    L = np.column_stack([mu, Lg])
    beta = np.r_[.15, np.full(5, .02)]
    return Q, L, beta, mu


# ----------------------------------------------------------------------------------------------
# benchmark harness
# ----------------------------------------------------------------------------------------------
def _brief(info):
    return {k: v for k, v in info.items() if k != 'lam'}


def timed(fn):
    """Warm-up call, then one timed call. Returns (milliseconds, output of the timed call)."""
    fn()
    t = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t) * 1000, out


class Benchmark:
    """Runs AS-SNC and the conic baselines on an instance and records time and accuracy.

    Backends that are not installed are recorded as unavailable. MOSEK needs a licence and Gurobi's
    free licence limits model size; their missing entries stay missing (nothing is extrapolated).
    """

    def __init__(self, methods=None):
        import cvxpy as cp
        self.methods = list(methods or METHODS)
        installed = set(cp.installed_solvers())
        self.disabled = {m: 'not installed' for m in self.methods
                         if m != 'AS-SNC' and m not in installed}
        self.records = []

    def _baseline(self, x, backend, tol):
        if backend in self.disabled:
            return None, {'status': 'unavailable', 'reason': self.disabled[backend]}
        try:
            return conic_solve(*x, backend, tol)
        except Exception as e:                                   # solver errors are recorded
            reason = str(e)
            if backend == 'MOSEK' and ('license' in reason.lower() or 'licence' in reason.lower()):
                self.disabled[backend] = 'No configured MOSEK licence'
            return None, {'status': 'error', 'reason': reason[:250]}

    def evaluate(self, case, x, tol=1e-9):
        Q, L, nu, b = x
        wr, ref = reference_solution(*x)
        digest = hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for a in x)).hexdigest()
        row = dict(case=case, M=len(Q), N=len(b), tol=tol, input_sha256=digest, reference=ref,
                   methods={})
        for method in self.methods:
            try:
                if method == 'AS-SNC':
                    ms, out = timed(lambda: as_snc(*x, tol=tol, return_info=True))
                    w, it, A, info = out
                    info = {**info, 'iterations': it, 'active': len(A)}
                else:
                    ms, (w, info) = timed(lambda: self._baseline(x, method, tol))
                rec = {k: v for k, v in info.items() if k != 'lam'}
                rec['ms'] = ms if w is not None else None
                if w is not None:
                    rec.update(quality(*x, w, info.get('lam')))
                    rec['relative_objective_difference'] = float(
                        abs(w @ Q @ w - ref['objective']) / max(abs(ref['objective']), 1e-12))
                    rec['relative_vector_difference'] = float(
                        np.linalg.norm(w - wr) / max(np.linalg.norm(wr), 1e-12))
                    rec['passes_common_target'] = bool(
                        rec['primal'] <= 1e-7 and rec['relative_objective_difference'] <= 1e-6)
                row['methods'][method] = rec
            except Exception as exc:
                row['methods'][method] = {'status': 'failed', 'reason': str(exc), 'ms': None}
        self.records.append(row)
        return row


# ----------------------------------------------------------------------------------------------
# timed experiments (Figures 2.1, 2.5, 2.6, 2.8; Tables 2.3, 2.4)
# ----------------------------------------------------------------------------------------------
def dimension_sweep(bench, dims=(25, 50, 100, 200, 400), per_dim=8, N=20, kappa=1000, seed=0):
    """Figure 2.1 and Table 2.3: one generator (seed 0) for all dimensions, eight instances each."""
    rows = []
    rng = np.random.default_rng(seed)
    with threadpool_limits(limits=1):
        for M in dims:
            for j in range(per_dim):
                rows.append(bench.evaluate(f'scaling_M{M}_{j}', random_instance(M, N, kappa, rng)))
    return rows


def conditioning_sweep(bench, kappas=(10, 100, 1000, 10000, 100000), per_kappa=10, M=100, N=20,
                       seed=100, seed_step=1000):
    """Figure 2.6: seed 100 + 1000 k for the k-th condition number, ten instances each."""
    rows = []
    with threadpool_limits(limits=1):
        for ki, k in enumerate(kappas):
            rng = np.random.default_rng(seed + ki * seed_step)
            for j in range(per_kappa):
                rows.append(bench.evaluate(f'conditioning_k{k}_{j}', random_instance(M, N, k, rng)))
    return rows


def tolerance_sweep(bench, tols=(1e-2, 1e-4, 1e-6, 1e-8, 1e-10), n=8, M=100, N=20, kappa=1000,
                    seed=10):
    """Figure 2.5: eight fixed instances (seed 10), every method at every requested tolerance."""
    rows = []
    rng = np.random.default_rng(seed)
    instances = [random_instance(M, N, kappa, rng) for _ in range(n)]
    with threadpool_limits(limits=1):
        for tol in tols:
            for j, x in enumerate(instances):
                rows.append(bench.evaluate(f'pareto_{tol}_{j}', x, tol))
    return rows


def famafrench_benchmark(bench, sets=('FF-Ind12', 'FF-All21', 'FF-All30'), repeats=10):
    """Figure 2.8 and Table 2.4: ten timed repetitions for each of the three data sets."""
    rows = []
    with threadpool_limits(limits=1):
        for name in sets:
            x = build_portfolio_problem(load_ff(name))[:4]
            for j in range(repeats):
                rows.append(bench.evaluate(f'{name}_{j}', x))
    return rows


# ----------------------------------------------------------------------------------------------
# solver behaviour (Figures 2.2-2.4) and checks
# ----------------------------------------------------------------------------------------------
def regression_checks():
    """Scalar instances with the closed form w* = beta / (1 - nu) (Section 2.2.6) and the
    two-dimensional instance with a repelling fixed point (Section 2.4.2)."""
    checks = []
    for radius in [0, .2, .9, .99, 1, 1.1]:
        try:
            w, it, A, info = as_snc(np.eye(1), np.eye(1), np.array([radius]), np.ones(1),
                                    return_info=True)
            assert radius < 1 and abs(w[0] - 1 / (1 - radius)) < 1e-6
            checks.append(dict(radius=radius, w=w.tolist(), **_brief(info)))
        except RuntimeError:
            assert radius >= 1
            checks.append(dict(radius=radius, status='infeasible, no point returned'))
    Q = np.array([[1., 2.], [2., 5.]])
    L = np.array([[3., 1.], [3.5, .5]])
    w, it, A, info = as_snc(Q, L, np.ones(2), np.array([2., 0.]), return_info=True)
    assert np.linalg.norm(w - np.array([1., 0.])) < 1e-6
    checks.append(dict(case='repelling fixed point (Section 2.4.2)', w=w.tolist(), **_brief(info)))
    return checks


def convergence_traces(n=5, M=100, N=20, kappa=1000, seed=20261020):
    """Figure 2.2: scaled KKT residual of every recorded candidate (including Newton corrections)."""
    rng = np.random.default_rng(seed)
    rows = []
    with threadpool_limits(limits=1):
        for _ in range(n):
            x = random_instance(M, N, kappa, rng)
            w, it, A, trace, info = as_snc(*x, return_trace=True, return_info=True)
            rows.append(dict(iterations=it, residuals=[quality(*x, v)['residual'] for v in trace],
                             **_brief(info)))
    return rows


def scale_invariance(n=20, M=100, N=20, kappa=1000, seed=2030, scales=None):
    """Figure 2.3: relative change of the AS-SNC solution when Q is multiplied by a positive scale."""
    scales = np.logspace(-4, 4, 9) if scales is None else np.asarray(scales, float)
    rng = np.random.default_rng(seed)
    with threadpool_limits(limits=1):
        xs = [random_instance(M, N, kappa, rng) for _ in range(n)]
        refs = [as_snc(*x)[0] for x in xs]
        errors = []
        for s in scales:
            es = []
            for (Q, L, nu, b), ref in zip(xs, refs):
                w, it, A, info = as_snc(s * Q, L, nu, b, return_info=True)
                es.append(float(np.linalg.norm(w - ref) / np.linalg.norm(ref)))
            errors.append(es)
    return dict(scales=scales.tolist(), errors=errors)


def active_set_counts(Ns=(20, 50), n=200, M=100, kappa=1000, seed=2040):
    """Figure 2.4: optimal active-set cardinality over n instances for each N (seed 2040 + N)."""
    counts = {}
    with threadpool_limits(limits=1):
        for N in Ns:
            rng = np.random.default_rng(seed + N)
            a = []
            for _ in range(n):
                w, it, A, info = as_snc(*random_instance(M, N, kappa, rng), return_info=True)
                assert info['method'] == 'AS-SNC'
                a.append(len(A))
            counts[str(N)] = a
    return counts


def near_boundary_diagnostic(radii=(.10, .11, .12, .13, .14, .15)):
    """Reduced Newton correction near the feasibility limit of the 50-asset model."""
    Q, L, b, _ = portfolio_model()
    rows = []
    with threadpool_limits(limits=1):
        for radius in radii:
            try:
                w, it, A, info = as_snc(Q, L, np.r_[0, np.full(5, radius)], b, return_info=True)
                rows.append(dict(radius=radius, risk=float(np.sqrt(w @ Q @ w)), **_brief(info)))
            except RuntimeError:
                rows.append(dict(radius=radius, status='no checked solution'))
    return rows


def supplement(quick=False):
    """Records behind Figures 2.2-2.4 plus the checks, in the archived format."""
    q = quick
    return dict(regression=regression_checks(),
                convergence=convergence_traces(n=2 if q else 5),
                scale=scale_invariance(n=4 if q else 20,
                                       scales=np.logspace(-4, 4, 5) if q else None),
                counts=active_set_counts(n=20 if q else 200),
                near_boundary_diagnostic=near_boundary_diagnostic())


def portfolio_sweep(kappas=None):
    """Figure 2.7: AS-SNC solutions of the 50-asset model as the common sector radius grows.

    The sector floors are feasible only for kappa < 1/sqrt(50). Each AS-SNC point is cross-checked
    against a tight Clarabel solve.
    """
    import cvxpy as cp
    Q, L, b, mu = portfolio_model()
    M = len(Q)
    kappas = np.round(np.linspace(0, .14, 29), 3) if kappas is None else kappas

    def clarabel_check(nu):
        x = cp.Variable(M)
        t = cp.Variable()
        cs = [cp.norm(x) <= t, L.T @ x - nu * t >= b]
        p = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(Q))), cs)
        p.solve(solver='CLARABEL', tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12,
                max_iter=500)
        return x.value

    rows = []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for k in kappas:
            nu = np.r_[0, np.full(5, k)]
            w, it, A, info = as_snc(Q, L, nu, b, return_info=True)
            wc = clarabel_check(nu)
            obj = float(w @ Q @ w)
            rows.append(dict(kappa=float(k), risk=float(np.sqrt(obj)), ret=float(mu @ w),
                             norm=float(np.linalg.norm(w)), active=int(len(A)),
                             method=info['method'], residual=info['residual'],
                             clarabel_risk=float(np.sqrt(wc @ Q @ wc)),
                             clarabel_residual=quality(Q, L, nu, b, wc)['residual'],
                             rel_obj_diff=abs(obj - float(wc @ Q @ wc)) / obj))
    return rows


def environment(bench=None):
    """Platform and library versions of a run."""
    import platform
    import cvxpy as cp
    env = dict(platform=platform.platform(), python=platform.python_version(),
               numpy=np.__version__, cvxpy=cp.__version__,
               disabled=dict(bench.disabled) if bench is not None else {}, blas_threads=1)
    try:
        env['cpu'] = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo')
                          if l.startswith('model name'))
    except (OSError, StopIteration):
        env['cpu'] = platform.processor() or platform.machine()
    try:
        import mosek
        env['mosek_version'] = list(mosek.Env.getversion())
    except Exception:
        pass
    return env


def run_all(quick=False, progress=print):
    """Run every experiment and return the records, keyed like the files in data/archived_run."""
    bench = Benchmark()
    rec = {}
    t0 = time.time()

    def step(name):
        progress(f'{name} done ({time.time() - t0:.0f} s)')

    rec['scaling'] = dimension_sweep(bench, dims=(25, 50, 100) if quick else (25, 50, 100, 200, 400),
                                     per_dim=2 if quick else 8)
    step('dimension sweep')
    rec['conditioning'] = conditioning_sweep(bench, per_kappa=2 if quick else 10)
    step('conditioning sweep')
    rec['pareto'] = tolerance_sweep(bench, n=2 if quick else 8)
    step('tolerance sweep')
    rec['famafrench'] = famafrench_benchmark(bench, repeats=2 if quick else 10)
    step('Fama-French benchmark')
    rec['supplement'] = supplement(quick)
    step('convergence, scale and active-set checks')
    rec['portfolio'] = portfolio_sweep(np.round(np.linspace(0, .14, 8), 3) if quick else None)
    step('portfolio sweep')
    rec['environment'] = environment(bench)
    return rec
