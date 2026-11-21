"""Experiment stages of Section 7.7. Each stage writes <datadir>/<stage>.json.

Stages and seeds (fixed, so every number can be regenerated exactly)
  main      Scenarios A-D, F, G (Table 7.2); reference seeds 1-15 and held-out seeds 5001-5015
            (Tables 7.3, 7.4, Figures 7.1, 7.4-7.6, 7.9 and the text of Sections 7.7.2, 7.7.3, 7.7.9, 7.7.12)
  array     M = 16, 32, 64, 128, benign and three strong interferers; seeds 6201-6210 (Table 7.5)
  load      U = T = 2, 3, 4 at M = 64, benign and three strong interferers; seeds 6301-6310 (Table 7.5)
  sweep     interferer separation, INR and joint mismatch at M = 32; seeds 6101-6108 (Figure 7.2)
  cross     target-to-user separation at M = 32; seeds 6401-6408 (Figure 7.3)
  sens      one-at-a-time parameter sensitivity, Scenarios D, F, G; seeds 6501-6508 (Table 7.6)
  alpha     priority sweep, Scenario D; seeds 7001-7010 (Figure 7.7)
  grid      Scenario E, 7 x 7 angular-error grid, K = 150; seeds 6001-6003 (Figure 7.8)
  tightproj projection reconstruction with an exact conic solve on held-out F and G scenes (Section 7.7.2)
  timing    single-thread runtime versus M; seeds 6901-6903 (Table 7.1)
The development seeds 2001-2012 and 8001-8008 that fixed the settings are not used here.

Method keys: rdf_as_isac, projection (projection reconstruction), socp / socp_radius (batch SOCP with nu_0 / 3 nu_0),
socp_online / socp_radius_online (re-solved every 10 observations), sdp, fixed (fixed beam), vu_mbwc, naive (LCMV),
true_cov (same problem with the true R_0), bound (single-function MVDR bound (7.6b)).
"""
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import cvxpy as cp
import scipy.linalg as sla

from . import baselines as m
from .scenes import M, K, C, S, L, n, nu, CENTERS, TRACK_KS, scene
from .rxmodel import Scene, evaluate_rx, RHO_O, N_FAST, steer
from .algorithms import run_rdf_as_isac, Scanner, wc_loaded
from . import projection as RI
from .projection import run_projection_reconstruction

warnings.filterwarnings('ignore')
TR_K = list(range(0, K, 5)) + [K-1]
GRID = np.linspace(-90, 90, 361)
FIXED_USERS, FIXED_TARGETS = [(20, 5), (-30, 8)], [(-5, 0), (40, -3)]
STAGES = ['main', 'array', 'load', 'sweep', 'cross', 'sens', 'alpha', 'grid', 'tightproj', 'timing']


def seeds(quick=False):
    """Seed sets of every stage. --quick keeps the structure with a few seeds per point."""
    if not quick:
        return dict(ref=range(1, 16), held=range(5001, 5016), array=range(6201, 6211), load=range(6301, 6311),
                    sweep=range(6101, 6109), cross=range(6401, 6409), sens=range(6501, 6509), alpha=range(7001, 7011),
                    grid=range(6001, 6004), timing=(6901, 6902, 6903), timing_sizes=[(4, 4), (8, 4), (8, 8), (16, 8), (16, 16)])
    return dict(ref=range(1, 3), held=range(5001, 5003), array=range(6201, 6203), load=range(6301, 6303),
                sweep=range(6101, 6102), cross=range(6401, 6402), sens=range(6501, 6502), alpha=range(7001, 7003),
                grid=range(6001, 6002), timing=(6901,), timing_sizes=[(4, 4), (8, 4), (8, 8)])


def nan4(k): return [float('nan')]*k


def socp(R, Cm, Sm, nv, mult):
    """Exact worst-case SOCP with bound vector mult*nv (nv has one entry per column of [Cm, Sm])."""
    assert len(nv) == Cm.shape[1] + Sm.shape[1]
    R = (R+R.conj().T)/2; R = R/(np.trace(R).real/len(R)); nc = Cm.shape[1]
    return m.run_socp_isac(R, Cm, Sm, mult*nv[:nc], mult*nv[nc:])


def safe(f, *a):
    try: return f(*a)
    except Exception: return None


def ev(w, tr, k):
    return evaluate_rx(w, tr) if w is not None else nan4(k)


def bound(tr):
    """Single-function MVDR bound (7.6b) for each user and target (dB): the largest SINR/SNR any receive filter
    can give that function alone with the true covariances."""
    R0 = tr['R_cci']
    Rc = R0 + RHO_O*sum(abs(b)**2*np.outer(a, a.conj()) for a, b in zip(tr['a_true_targets'], tr['beta']))
    Rs = R0 + RHO_O*sum(p*np.outer(h, h.conj()) for h, p in zip(tr['h_list'], tr['user_ps']))
    out = [10*np.log10(p*np.vdot(h, np.linalg.solve(Rc, h)).real) for h, p in zip(tr['h_list'], tr['user_ps'])]
    out += [10*np.log10(N_FAST*abs(b)**2*np.vdot(a, np.linalg.solve(Rs, a)).real) for a, b in zip(tr['a_true_targets'], tr['beta'])]
    return out


def online_socp(X, Cm, Sm, nv, mult, eta=0.97, T=10):
    """SOCP re-solved every T observations (k = 0, 10, ..., i.e. observations 1, 11, ..., 291) from a forgetting
    covariance with factor 0.97."""
    Mm = X.shape[0]; Re = np.eye(Mm, dtype=complex); W = []; w = None
    for k in range(X.shape[1]):
        x = X[:, k]; Re = eta*Re + (1-eta)*np.outer(x, x.conj())
        if k % T == 0:
            wn = safe(socp, Re, Cm, Sm, nv, mult)
            if wn is not None or w is None: w = wn
        W.append(w)
    return W


def matched_gap(Q, w, nu_, b, Lm):
    """Relative objective gap (7.28) of w against a tight conic solve of the same problem."""
    Q = (Q+Q.conj().T)/2; Mm = len(Q); x = cp.Variable(Mm, complex=True); t = cp.Variable(nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(sla.cholesky(Q, lower=False) @ x)),
                      [cp.real(Lm.conj().T @ x) >= b + nu_*t, cp.norm(x, 2) <= t])
    for tol in (1e-10, 1e-9, 1e-8):
        try:
            prob.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
            if x.value is not None: break
        except cp.error.SolverError:
            continue
    if x.value is None: return float('nan')
    ws = np.asarray(x.value).ravel(); o = lambda v: np.vdot(v, Q@v).real
    return float(o(w)/o(ws) - 1)


def pattern(w, Mx=8, My=4, el=2.0):
    A = np.column_stack([steer(Mx, My, th, el) for th in GRID]); u = w/np.linalg.norm(w)
    return (np.abs(u.conj() @ A)**2).tolist()


# ------------------------------------------------------------------------------------------------ main
def main_job(job):
    split, sc, seed = job; X, trs = scene(sc, seed); trk = sc in ('F', 'G')
    score = (lambda h: np.mean([ev(h[k], trs[k], 4) for k in TRACK_KS], 0).tolist()) if trk else \
            (lambda h: ev(h[-1], trs[-1], 4))
    H, tm = {}, {}
    t0 = time.perf_counter(); h, dg = run_rdf_as_isac(X, C, S, n, n, CENTERS, return_diagnostics=True)
    tm['rdf_as_isac'] = time.perf_counter()-t0
    H['rdf_as_isac'] = h
    t0 = time.perf_counter(); H['projection'] = run_projection_reconstruction(X, C, S, n, n); tm['projection'] = time.perf_counter()-t0
    H['vu_mbwc'] = m.run_dtras_virtual(X, L, nu)
    H['naive'] = m.run_naive_lcmv(X, L)
    wfix = socp(np.eye(M), C, S, nu, 3); H['fixed'] = [wfix]*K
    H['socp_online'] = online_socp(X, C, S, nu, 1); H['socp_radius_online'] = online_socp(X, C, S, nu, 3)
    R = (X@X.conj().T + np.eye(M))/(K+1)
    t0 = time.perf_counter(); ws = safe(socp, R, C, S, nu, 1); tm['socp'] = time.perf_counter()-t0
    H['socp'] = [ws]*K; H['socp_radius'] = [safe(socp, R, C, S, nu, 3)]*K
    H['sdp'] = [safe(m.run_sdp_isac, (R+R.conj().T)/2/(np.trace(R).real/M), C, S, n, n)]*K
    out = {a: score(hh) for a, hh in H.items()}
    if trk:
        ks = TRACK_KS[::5]
        out['true_cov'] = np.mean([ev(safe(socp, trs[k]['R_cci'], C, S, nu, 3), trs[k], 4) for k in ks], 0).tolist()
        out['bound'] = np.mean([bound(trs[k]) for k in ks], 0).tolist()
    else:
        out['true_cov'] = ev(safe(socp, trs[-1]['R_cci'], C, S, nu, 3), trs[-1], 4)
        out['bound'] = bound(trs[-1])
    tr_, pat = {}, {}
    if split == 'reference' and sc in ('C', 'D', 'F', 'G'):
        tr_ = {a: [ev(H[a][k], trs[k], 4) for k in TR_K] for a in
               ['rdf_as_isac', 'projection', 'socp_online', 'socp_radius_online', 'vu_mbwc', 'naive', 'fixed']}
        for a in ['socp', 'socp_radius']: tr_[a] = [out[a]]*len(TR_K) if not trk else [ev(H[a][k], trs[k], 4) for k in TR_K]
        tr_['true_cov'] = [ev(safe(socp, trs[k]['R_cci'], C, S, nu, 3), trs[k], 4) for k in TR_K[::5]]
    if sc in ('D', 'F'):
        kk = K-1
        for a in ['rdf_as_isac', 'projection', 'socp_radius' if sc == 'D' else 'socp_radius_online', 'fixed', 'vu_mbwc']:
            if H[a][kk] is not None: pat[a] = pattern(H[a][kk])
        wo = safe(socp, trs[kk]['R_cci'], C, S, nu, 3)
        if wo is not None: pat['true_cov'] = pattern(wo)
    gap = matched_gap(dg['Q_final'], h[-1], dg['nu'], dg['b'], L)
    return dict(split=split, scenario=sc, seed=seed, metrics=out, time=tm, traces=tr_, patterns=pat,
                min_margin=min(d['min_margin'] for d in dg['per_snapshot']), objective_gap=gap,
                max_cert=float(max(dg['certified_gaps'])), solver_iterations=dg['solver_iterations'])


# --------------------------------------------------------------------------------------- general scenes
def general(sc_obj, X, trs, Mx, My, with_projection=True):
    Mm = Mx*My; Cm, Sm = sc_obj.nominal_columns(); U, T = len(sc_obj.users), len(sc_obj.targets)
    n_ = np.array([.5, .35, .35]*(U+T))*np.sqrt(Mm/32); nC, nS = n_[:3*U], n_[3*U:]
    tr = trs[-1]; out, tm = {}, {}
    t0 = time.perf_counter()
    h = run_rdf_as_isac(X, Cm, Sm, nC, nS, sc_obj.users+sc_obj.targets, Mx=Mx, My=My, eta=1-1.28/Mm)
    tm['rdf_as_isac'] = time.perf_counter()-t0; out['rdf_as_isac'] = ev(h[-1], tr, U+T)
    if with_projection:
        t0 = time.perf_counter(); hp = safe(run_projection_reconstruction, X, Cm, Sm, nC, nS); tm['projection'] = time.perf_counter()-t0
        out['projection'] = ev(hp[-1] if hp else None, tr, U+T)
    R = (X@X.conj().T + np.eye(Mm))/(X.shape[1]+1)
    nn = np.r_[nC, nS]
    t0 = time.perf_counter(); out['socp'] = ev(safe(socp, R, Cm, Sm, nn, 1), tr, U+T); tm['socp'] = time.perf_counter()-t0
    out['socp_radius'] = ev(safe(socp, R, Cm, Sm, nn, 3), tr, U+T)
    out['fixed'] = ev(safe(socp, np.eye(Mm), Cm, Sm, nn, 3), tr, U+T)
    out['true_cov'] = ev(safe(socp, tr['R_cci'], Cm, Sm, nn, 3), tr, U+T)
    out['bound'] = bound(tr)
    return out, tm


def array_job(job):
    Mx, My, strong, seed = job
    sc = Scene(Mx, My, 2, 2, seed=seed, strong=strong); X, trs = sc.generate(K, 3, 3)
    out, tm = general(sc, X, trs, Mx, My)
    return dict(Mx=Mx, My=My, strong=strong, seed=seed, metrics=out, time=tm)


def load_job(job):
    U, strong, seed = job
    sc = Scene(8, 8, U, U, seed=seed, strong=strong); X, trs = sc.generate(K, 3, 3)
    out, tm = general(sc, X, trs, 8, 8)
    return dict(U=U, strong=strong, seed=seed, metrics=out, time=tm)


def sweep_job(job):
    kind, val, seed = job
    if kind == 'mismatch':
        X, trs = scene('D', seed, sigma=(val, val)); tr = trs[-1]
        R = (X@X.conj().T + np.eye(M))/(K+1)
        out = {'rdf_as_isac': ev(run_rdf_as_isac(X, C, S, n, n, CENTERS)[-1], tr, 4),
               'projection': ev(run_projection_reconstruction(X, C, S, n, n)[-1], tr, 4),
               'socp': ev(safe(socp, R, C, S, nu, 1), tr, 4), 'socp_radius': ev(safe(socp, R, C, S, nu, 3), tr, 4),
               'fixed': ev(safe(socp, np.eye(M), C, S, nu, 3), tr, 4),
               'true_cov': ev(safe(socp, tr['R_cci'], C, S, nu, 3), tr, 4), 'bound': bound(tr)}
        return dict(kind=kind, val=val, seed=seed, metrics=out)
    dsep, inr = (val, 25.0) if kind == 'separation' else (8.0, val)
    sc = Scene(8, 4, 2, 2, seed=seed, random_geometry=False, users=FIXED_USERS, targets=FIXED_TARGETS,
               strong_angles=[(FIXED_USERS[0][0]+dsep, FIXED_USERS[0][1])], inr_strong=inr)
    X, trs = sc.generate(K, 3, 3); out, _ = general(sc, X, trs, 8, 4)
    return dict(kind=kind, val=val, seed=seed, metrics=out)


def cross_job(job):
    d, seed = job
    tg = [(FIXED_USERS[0][0]+d, FIXED_USERS[0][1]), FIXED_TARGETS[1]]
    sc = Scene(8, 4, 2, 2, seed=seed, random_geometry=False, users=FIXED_USERS, targets=tg)
    X, trs = sc.generate(K, 2, 2); out, _ = general(sc, X, trs, 8, 4)
    return dict(delta=d, seed=seed, metrics=out)


SENS = {'default': {}, 'win_az=6': dict(win_az=6.0), 'win_az=12': dict(win_az=12.0),
        'eta=0.94': dict(eta=0.94), 'eta=0.98': dict(eta=0.98), 'T=5': dict(T=5), 'T=20': dict(T=20),
        'tol=3': dict(user_tol_db=3.0), 'tol=10': dict(user_tol_db=10.0),
        'beta_v=0.1': dict(beta_v=0.1), 'beta_v=0.5': dict(beta_v=0.5),
        'alpha_p=0.3': dict(alpha_p=0.3), 'alpha_p=1': dict(alpha_p=1.0),
        'nu_mult=2': dict(nu_mult=2.0), 'nu_mult=4': dict(nu_mult=4.0)}


def sens_job(job):
    sc, seed = job; X, trs = scene(sc, seed); trk = sc in ('F', 'G')
    score = (lambda h: np.mean([evaluate_rx(h[k], trs[k]) for k in TRACK_KS], 0).tolist()) if trk else \
            (lambda h: evaluate_rx(h[-1], trs[-1]))
    return dict(scenario=sc, seed=seed, metrics={k: score(run_rdf_as_isac(X, C, S, n, n, CENTERS, **v)) for k, v in SENS.items()})


def alpha_job(job):
    a, seed = job; X, trs = scene('D', seed)
    return dict(alpha=a, seed=seed, metrics=evaluate_rx(run_rdf_as_isac(X, C, S, n, n, CENTERS, alpha=a)[-1], trs[-1]))


def grid_job(job):
    c, s, seed = job; X, trs = scene('D', seed, KK=150, sigma=(c, s)); tr = trs[-1]
    R = (X@X.conj().T + np.eye(M))/151
    W = {'rdf_as_isac': run_rdf_as_isac(X, C, S, n, n, CENTERS)[-1], 'projection': run_projection_reconstruction(X, C, S, n, n)[-1],
         'socp': safe(socp, R, C, S, nu, 1), 'socp_radius': safe(socp, R, C, S, nu, 3),
         'vu_mbwc': m.run_dtras_virtual(X, L, nu)[-1]}
    return dict(c=c, s=s, seed=seed, metrics={a: ev(w, tr, 4) for a, w in W.items()})


def exact_projection(X):
    """Projection reconstruction with the same reconstructed covariance and 30-observation schedule, but each
    weight an exact radius-matched SOCP (diagnostic of Section 7.7.2)."""
    M_, K_ = X.shape; P, _ = RI.make_projector(L, .95); acc = np.zeros((M_, M_), complex); w = None; hist = []
    for k in range(K_):
        acc += np.outer(X[:, k], X[:, k].conj())
        if k == 0 or (k+1) % 30 == 0 or k == K_-1:
            R = (acc+np.eye(M_))/(k+2); RR, _ = RI.reconstruct(R, P)
            w = m.run_socp_isac(RR, C, S, 3*n, 3*n)
        hist.append(w)
    return hist


def tightproj_job(job):
    sc, seed = job; X, trs = scene(sc, seed); h = exact_projection(X)
    return dict(scenario=sc, seed=seed, metrics=np.mean([evaluate_rx(h[k], trs[k]) for k in TRACK_KS], 0).tolist())


def timing(sd):
    rows = []
    for Mx, My in sd['timing_sizes']:
        Mm = Mx*My; acc = {k: [] for k in ['rdf_pass', 'scan', 'solve', 'update', 'projection_pass', 'socp_solve']}
        for seed in sd['timing']:
            sc = Scene(Mx, My, 2, 2, seed=seed); X, _ = sc.generate(K, 3, 3); Cm, Sm = sc.nominal_columns()
            n_ = np.array([.5, .35, .35]*2)*np.sqrt(Mm/32)
            t0 = time.perf_counter(); run_rdf_as_isac(X, Cm, Sm, n_, n_, sc.users+sc.targets, Mx=Mx, My=My, eta=1-1.28/Mm)
            acc['rdf_pass'].append(time.perf_counter()-t0)
            R = (X@X.conj().T + np.eye(Mm))/(K+1); scn = Scanner(Mx, My)
            t0 = time.perf_counter()
            for _ in range(5): scn.peaks(R)
            acc['scan'].append((time.perf_counter()-t0)/5)
            Q = np.eye(Mm, dtype=complex) + 300*np.outer(steer(Mx, My, 33, 4), steer(Mx, My, 33, 4).conj())
            Lm = np.c_[Cm, Sm]; nuv = 3*np.r_[n_, n_]
            t0 = time.perf_counter()
            for _ in range(5): wc_loaded(Q, Lm, nuv)
            acc['solve'].append((time.perf_counter()-t0)/5)
            x = X[:, 0]; Rr = np.eye(Mm, dtype=complex); t0 = time.perf_counter()
            for k in range(200): Rr = .96*Rr + .04*np.outer(x, x.conj())
            acc['update'].append((time.perf_counter()-t0)/200)
            if Mm <= 128:
                t0 = time.perf_counter(); run_projection_reconstruction(X, Cm, Sm, n_, n_); acc['projection_pass'].append(time.perf_counter()-t0)
            t0 = time.perf_counter(); safe(socp, R, Cm, Sm, np.r_[n_, n_], 1); acc['socp_solve'].append(time.perf_counter()-t0)
        rows.append(dict(M=Mm, Mx=Mx, My=My, **{k: (float(np.mean(v)) if v else None) for k, v in acc.items()}))
        print('timing', rows[-1], flush=True)
    return rows


def _init_worker():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def jobs_for(stage, sd):
    if stage == 'main':
        return main_job, [(sp, sc, s) for sp, ss in [('reference', sd['ref']), ('heldout', sd['held'])] for sc in 'ABCDFG' for s in ss]
    if stage == 'array':
        return array_job, [(Mx, My, st, s) for Mx, My in [(4, 4), (8, 4), (8, 8), (16, 8)] for st in (0, 3) for s in sd['array']]
    if stage == 'load':
        return load_job, [(U, st, s) for U in (2, 3, 4) for st in (0, 3) for s in sd['load']]
    if stage == 'sweep':
        j = [('separation', d, s) for d in (3, 5, 7, 10, 14, 20) for s in sd['sweep']]
        j += [('inr', v, s) for v in (5, 10, 15, 20, 25, 30) for s in sd['sweep']]
        j += [('mismatch', v, s) for v in (0, 1, 2, 3, 4, 5) for s in sd['sweep']]
        return sweep_job, j
    if stage == 'cross':
        return cross_job, [(d, s) for d in (0, 2, 4, 6, 9, 12, 15) for s in sd['cross']]
    if stage == 'sens':
        return sens_job, [(sc, s) for sc in 'DFG' for s in sd['sens']]
    if stage == 'alpha':
        return alpha_job, [(a, s) for a in (0, .15, .3, .5, .7, .85, 1) for s in sd['alpha']]
    if stage == 'grid':
        return grid_job, [(c, s, seed) for c in range(7) for s in range(7) for seed in sd['grid']]
    if stage == 'tightproj':
        return tightproj_job, [(sc, s) for sc in 'FG' for s in sd['held']]
    raise KeyError(stage)


def run_stages(datadir, stages=None, jobs=2, quick=False):
    """Run the listed stages (default: all) and write <datadir>/<stage>.json. Timing runs last, alone."""
    datadir = Path(datadir); datadir.mkdir(parents=True, exist_ok=True)
    stages = list(stages or STAGES); sd = seeds(quick)
    par = [s for s in STAGES if s in stages and s != 'timing']
    if par:
        with ProcessPoolExecutor(jobs, initializer=_init_worker) as ex:
            for st in par:
                fn, jb = jobs_for(st, sd); t0 = time.time(); res = []
                for r in ex.map(fn, jb):
                    res.append(r)
                    if len(res) % 20 == 0: print(st, len(res), '/', len(jb), f'{time.time()-t0:.0f}s', flush=True)
                (datadir/f'{st}.json').write_text(json.dumps(res)); print(st, 'done', f'{time.time()-t0:.0f}s', flush=True)
    if 'timing' in stages:
        _init_worker()
        (datadir/'timing.json').write_text(json.dumps(timing(sd), indent=1)); print('timing done', flush=True)
