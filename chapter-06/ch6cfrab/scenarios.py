"""Chapter 6 experiments: settings, one stage per group of results, and the runner.

Every stage is a function that returns a JSON-serialisable dict; the runner executes each stage in its own process
(the simulator keeps its experiment switches in module globals) and writes <datadir>/<stage>.json and <stage>.log.
All random streams are seeded per run, so the stages can run in any order or in parallel and give the same numbers.

Stages and the results they produce
    fig1 ... fig10   figure data (Figures 6.1-6.10); fig6 measures wall-clock time and runs alone, after the others
    train            bound-level selection of the main study and the scalar-root solver check (Sections 6.7.1, 6.7.9)
    paired           20-realization paired comparisons and the local-stage comparison (Sections 6.7.2, 6.7.3, 6.7.9)
    audits           same-constraint conic audits, CF-RAB bound sweep, cross-AP rescaling outcomes (Section 6.7.9)
    lbound           load-dependent bound selection of D-CF-RAB, LP-MMSE and P-MMSE (Section 6.7.12)
    A1 ... A5        error laws (Table 6.4), bound selection per error law, equal-mean tiers (Section 6.7.6), joint-check
                     audit on scored snapshots and continuous-ball checks (Sections 6.7.9, 6.7.13)
    B1, B2           loading selection with exact and 256-snapshot covariances (Tables 6.2, 6.3); interferers without
                     channel estimates with non-Gaussian symbols (Table 6.5)
"""
import collections
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from numpy.linalg import cholesky, eigh, norm, solve
from scipy.stats import t as tdist

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)                         # single-thread BLAS, as for the timings in the chapter
except Exception:                                # pragma: no cover
    pass

from . import algorithms as C
from .algorithms import CFConfig, run_snapshot_trace

QUICK = os.environ.get('CH6_QUICK', '0') == '1'


def Q(full, quick):
    """Monte Carlo count: the chapter value, or a small value in --quick mode."""
    return quick if QUICK else full


# ------------------------------------------------------------------------------------------------ common settings
NSEED = 5                                        # realizations per figure point (95% Student-t intervals)
M9 = ('D-CF-RAB', 'MR+W', 'LP-MMSE+W', 'LMVDR+W', 'CF-RAB', 'centralized_sdp', 'P-MMSE', 'LP-MMSE', 'RECON-RAB')
M11 = M9 + ('MR',)


def paired(a, b):
    """Mean paired difference a - b with its 95% Student-t interval."""
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d)
    h = tdist.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def sweep(methods, cfg_kw, kw, base_seed):
    """(mean, 95% t half-width, worst-user SINR, per-realization means) per method, over NSEED realizations.
    The worst-user value is the SINR of the weakest user in each realization, averaged over realizations."""
    n = Q(NSEED, 2)
    acc = {m: [] for m in methods}; wst = {m: [] for m in methods}
    for t in range(n):
        r = run_snapshot_trace(CFConfig(**cfg_kw), methods=methods, seed=base_seed + 37 * t, soi_in_cov=True, **kw)
        for m in methods:
            v = np.asarray(r['sinr_dB'][m], float)
            acc[m].append(float(np.nanmean(v))); wst[m].append(float(np.nanmin(v)))
    tq = float(tdist.ppf(0.975, n - 1))
    return {m: (float(np.mean(acc[m])), float(tq * np.std(acc[m], ddof=1) / np.sqrt(n)),
                float(np.mean(wst[m])), [float(x) for x in acc[m]]) for m in methods}


def with_rab(**kw):
    """Return (original RABConfig, factory that overrides some of its fields)."""
    orig = C.RABConfig

    def mk(*a, **k):
        r = orig(*a, **k)
        for kk, vv in kw.items():
            setattr(r, kk, vv)
        return r
    return orig, mk


def capture_cf_rab(eps, seed, soi_in_cov=True):
    """Run one CF-RAB trace and return every (R, h, nu, cfg, eps, w) passed to Algorithm 6.1."""
    cc = []; _ac = C.beamformer_cf_rab

    def w(R, h, nu, cfg, e=0.1, *a, **k):
        x = _ac(R, h, nu, cfg, e); cc.append((R, h, nu, cfg, e, x)); return x
    C.beamformer_cf_rab = w
    try:
        C.run_snapshot_trace(CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10, eps_override=eps,
                             methods=('CF-RAB',), seed=seed, soi_in_cov=soi_in_cov)
    finally:
        C.beamformer_cf_rab = _ac
    return cc


def loaded_mvdr(R, h, cfg):
    m = h.size; gam = cfg.proj_reg * np.trace(R).real / m
    ri = solve(R + gam * np.eye(m), h)
    return ri / (h.conj() @ ri).real


# ------------------------------------------------------------------------------------------------ figure stages
EPS_FIG2 = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50]


def stage_fig1():
    """Figure 6.1: SINR against the number of APs, eps = 0.10, SNR = 10 dB."""
    xs = [4, 8, 16]
    return {str(L): sweep(M11, dict(L=L, M_l=8, K=4), dict(n_snapshots=60, SNR_dB=10, INR_dB=10), 100) for L in xs}


def stage_fig2():
    """Figure 6.2: mean and worst-user SINR against the CSI uncertainty radius, L = 16."""
    return {str(e): sweep(M11, dict(L=16, M_l=8, K=4), dict(n_snapshots=60, SNR_dB=10, INR_dB=10, eps_override=e), 700)
            for e in EPS_FIG2}


def stage_fig3():
    """Figure 6.3: SINR against the per-antenna SNR, L = 16, eps = 0.10."""
    return {str(s): sweep(M9, dict(L=16, M_l=8, K=4), dict(n_snapshots=60, SNR_dB=s), 600) for s in [-5, 0, 5, 10, 15, 20]}


def stage_fig4():
    """Figure 6.4: SINR against the CSI delay tau, L = 16."""
    return {str(t): sweep(M9, dict(L=16, M_l=8, K=4), dict(n_snapshots=60, fronthaul_delay=t, SNR_dB=10, INR_dB=10), 200)
            for t in [0, 1, 4, 16, 64]}


def stage_fig5():
    """Figure 6.5: uniform and tiered per-AP CSI uncertainty, L = 16."""
    return {('tiered' if md else 'uniform'): sweep(M9, dict(L=16, M_l=8, K=4),
                                                   dict(n_snapshots=60, SNR_dB=10, INR_dB=10, heterogeneous=md), 300)
            for md in [False, True]}


def stage_fig6():
    """Figure 6.6: measured time per weight update and listed CSI and control bytes per user update."""
    xs = [4, 8, 16]; ns = 30; n = Q(NSEED, 2)
    MT = ('D-CF-RAB', 'CF-RAB', 'centralized_sdp', 'P-MMSE', 'MR+W')
    rows = []
    for L in xs:
        cfg = CFConfig(L=L, M_l=8, K=4)
        acc = {m: [] for m in MT}; calls = {m: 0 for m in MT}; fhb = {}
        for t in range(n):
            r = run_snapshot_trace(cfg, n_snapshots=ns, SNR_dB=10, INR_dB=10, methods=MT, seed=400 + L + 37 * t,
                                   soi_in_cov=True)
            for m in MT:
                acc[m].append(r['update_ms'][m]); calls[m] += r['update_calls'][m]
            fhb['D-CF-RAB'] = r['fh_bytes_total'] / (ns * cfg.K)          # D-CF-RAB updates at every snapshot
        fhb['CF-RAB'] = (L * cfg.K * cfg.M_l * 16 + L * (cfg.M_l * (cfg.M_l + 1) // 2) * 16) / cfg.K
        fhb['P-MMSE'] = (L * cfg.K * cfg.M_l * 16) / cfg.K
        rows.append({'L': L, 't': {m: float(np.mean(v)) for m, v in acc.items()}, 'calls': calls, 'fh': dict(fhb)})
    return {'rows': rows}


def stage_fig7():
    """Figure 6.7: SINR against the trace length, L = 8, SNR = 0 dB."""
    return {str(n): sweep(M9, dict(L=8, M_l=8, K=4), dict(n_snapshots=n, SNR_dB=0), 500) for n in [5, 10, 20, 40, 80]}


def stage_fig8():
    """Figure 6.8 (left): objective excess of Algorithm 6.1 over the primary-ball conic benchmark."""
    EPS10 = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    rows = []
    for eps in EPS10:
        rat = []
        for seed in Q((700, 737, 774), (700,)):
            for R, h, nu, cfg, e, w_as in capture_cf_rab(eps, seed):
                epsT = float(nu[0])
                w_so = C.beamformer_centralized_sdp(R, h, eps_total=epsT)
                if w_so is None:
                    continue
                o = lambda w: float(np.real(np.vdot(w, R @ w)))
                if o(w_so) <= 0:
                    continue
                rat.append(o(w_as) / o(w_so))
        rows.append({'eps': eps, 'gap_pct': 100 * (float(np.mean(rat)) - 1)})
    return {'rows': rows}


def stage_fig9():
    """Figure 6.9: signal self-cancellation with the desired signal in the covariance and the channel in error."""
    cap = capture_cf_rab(0.10, 700, soi_in_cov=False)              # interference-plus-noise only
    R_IN, h_hat, _, cfg, _, _ = cap[0]; n = h_hat.size

    def trial(eps, snr, seed):
        rng = np.random.default_rng(seed); p = 10 ** (snr / 10.)
        d = rng.normal(size=n) + 1j * rng.normal(size=n); d = d / norm(d) * eps * norm(h_hat)
        ht = h_hat + d
        R = R_IN + p * np.outer(ht, ht.conj())
        nu = np.full(4, eps * norm(h_hat))                          # CF-RAB uses the declared radius, its best setting
        Lg = C.build_constraint_grid(h_hat, eps, N=4)
        gam = cfg.proj_reg * np.trace(R).real / n
        ri = solve(R + gam * np.eye(n), h_hat); wmv = ri / (h_hat.conj() @ ri).real
        was, *_ = C.rab_step(R, np.zeros(n, dtype=complex), Lg, nu, cfg, 1.0)
        wso = C.beamformer_centralized_sdp(R, h_hat, eps_total=eps * norm(h_hat))
        f = lambda w: 10 * np.log10(p * abs(np.vdot(w, ht)) ** 2 / float(np.real(np.vdot(w, R_IN @ w))))
        return f(wmv), f(was), (f(wso) if wso is not None else np.nan)

    rows = []
    for e in [0.05, 0.10, 0.20, 0.30]:
        for s in [0, 10, 20, 30]:
            v = np.array([trial(e, s, t) for t in range(Q(8, 2))]); m = np.nanmean(v, axis=0)
            rows.append({'eps': e, 'soi_dB': s, 'MVDR': float(m[0]), 'CF-RAB': float(m[1]), 'centralized_sdp': float(m[2])})
    return {'rows': rows}


def stage_fig10():
    """Figure 6.10: direction change of Algorithm 6.1 against the bound level, per AP and at the joint dimension."""
    EPS_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]   # c < 1
    cap = [(R, h, cfg) for R, h, _, cfg, _, _ in capture_cf_rab(0.10, 700)]
    if QUICK:
        cap = cap[:4]

    def cos_align(R, h, cfg):
        w = loaded_mvdr(R, h, cfg)
        return float(abs(np.vdot(w, h)) / (norm(w) * norm(h)))

    def departure(R, h, cfg, eps, N=4):
        m = h.size
        w, *_ = C.rab_step(R, np.zeros(m, dtype=complex), C.build_constraint_grid(h, eps, N=N),
                           np.full(N, eps * norm(h)), cfg, 1.0)
        wmv = loaded_mvdr(R, h, cfg)
        return float(1.0 - abs(np.vdot(w, wmv)) / (norm(w) * norm(wmv)))

    joint_star = np.array([cos_align(R, h, c) for R, h, c in cap])
    joint_dep = np.array([[departure(R, h, c, e) for R, h, c in cap] for e in EPS_GRID]).mean(axis=1)

    rng = np.random.default_rng(5); L, Mant, K = 16, 8, 4
    cfgc = CFConfig(L=L, M_l=Mant, K=K); rab = C.RABConfig()
    ap, ue = C.deploy_aps_users(cfgc, rng)
    h_true, h_hat2, eps_l, beta = C.gen_channel_block(cfgc, ap, ue, rng)
    for ell in range(L):
        for kk in range(K):
            nr = norm(h_true[ell, kk])
            if nr > 1e-12:
                tgt = np.sqrt(Mant) * 10 ** (rng.uniform(-3, 3) / 20.0)
                h_true[ell, kk] *= tgt / nr; h_hat2[ell, kk] *= tgt / nr
    sigma2 = 10 ** (-10 / 10.0)
    R_per = []
    for ell in range(L):
        Rl = sigma2 * np.eye(Mant, dtype=complex)
        for kk in range(K):                                          # desired user included, as in Figures 6.1-6.7
            Rl = Rl + 1.0 * np.outer(h_true[ell, kk], h_true[ell, kk].conj())
        R_per.append(Rl)
    per_star = np.array([cos_align(R_per[e], h_hat2[e, 0], rab) for e in range(L)])
    per_dep = np.array([[departure(R_per[e], h_hat2[e, 0], rab, x) for e in range(L)] for x in EPS_GRID]).mean(axis=1)
    return {'eps_grid': EPS_GRID, 'joint_star_mean': float(joint_star.mean()), 'per_star_mean': float(per_star.mean()),
            'joint_departure': joint_dep.tolist(), 'per_departure': per_dep.tolist()}


# ------------------------------------------------------------------------------------------------ paired comparisons
def exact_wc_grid(R_, h, nu):
    """Reference solution of the single-ball problem by a search over the loading (log grid plus golden section)."""
    d, U = eigh(R_); d = np.maximum(d, 1e-15); b = U.conj().T @ h; b2 = np.abs(b) ** 2

    def J(mu):
        r = b2 / (d + mu); gg = r.sum() - nu * np.sqrt((b2 / (d + mu) ** 2).sum())
        if gg <= 0:
            return np.inf, None
        a = 1.0 / gg
        return a * a * (d * b2 / (d + mu) ** 2).sum(), a
    lo, hi = np.log(1e-8 * d.max()), np.log(1e8 * d.max())
    grid = np.linspace(lo, hi, 161); vals = [J(np.exp(t))[0] for t in grid]
    i = int(np.argmin(vals)); a_, b_ = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    for _ in range(60):
        m1 = a_ + 0.382 * (b_ - a_); m2 = a_ + 0.618 * (b_ - a_)
        if J(np.exp(m1))[0] < J(np.exp(m2))[0]:
            b_ = m2
        else:
            a_ = m1
    mu = np.exp(0.5 * (a_ + b_)); val, alpha = J(mu)
    return None if alpha is None else alpha * (U @ (b / (d + mu)))


def stage_train():
    """Bound level of the main study (training realizations not used by any figure) and the scalar-root check."""
    out = {'train': {}}
    GRID = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
    for c in GRID:
        orig, mk = with_rab(nu_c=c); C.RABConfig = mk
        try:
            out['train'][str(c)] = {str(e): float(np.mean([np.nanmean(C.run_snapshot_trace(
                C.CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10, eps_override=e,
                methods=('D-CF-RAB',), seed=11000 + 97 * t, soi_in_cov=True)['sinr_dB']['D-CF-RAB'])
                for t in range(Q(3, 1))])) for e in (0.10, 0.50)}
        finally:
            C.RABConfig = orig
        print(f"  training c = {c}: {np.mean(list(out['train'][str(c)].values())):.3f} dB", flush=True)
    out['c_star'] = max(GRID, key=lambda c: np.mean(list(out['train'][str(c)].values())))
    rng = np.random.default_rng(5); rows = []
    for n in (2, 8, 32):
        for cond in (1, 100, 1e4):
            for c in (0.0, 0.1, 0.5, 0.98):
                Q_, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
                R_ = (Q_ * np.geomspace(1, cond, n)) @ Q_.conj().T; R_ = (R_ + R_.conj().T) / 2
                h = rng.normal(size=n) + 1j * rng.normal(size=n); nu = c * norm(h)
                ta = time.perf_counter(); w = C.exact_wc(R_, h, nu); tn = time.perf_counter() - ta
                ta = time.perf_counter(); wo = exact_wc_grid(R_, h, nu); to = time.perf_counter() - ta
                o = lambda v: float(np.real(np.vdot(v, R_ @ v)))
                rows.append(dict(n=n, cond=cond, c=c, rel_obj_vs_grid=abs(o(w) - o(wo)) / o(w),
                                 residual=abs(float(np.real(np.vdot(w, h))) - nu * norm(w) - 1),
                                 t_root_ms=1e3 * tn, t_grid_ms=1e3 * to))
    r8 = [r for r in rows if r['n'] == 8]
    out['solver_check'] = dict(cases=rows, max_rel_obj=max(r['rel_obj_vs_grid'] for r in rows),
                               max_residual=max(r['residual'] for r in rows),
                               speedup_n8=float(sum(r['t_grid_ms'] for r in r8) / sum(r['t_root_ms'] for r in r8)))
    return out


def stage_paired():
    """20-realization paired comparisons (normalized gains and path loss) and the exact versus Algorithm 6.1 local stage."""
    out = {'A_paired': {}, 'B_local_stage': {}}
    MET = ('D-CF-RAB', 'MR+W', 'LP-MMSE+W', 'LMVDR+W', 'LP-MMSE', 'P-MMSE')
    PTS = {'norm_L16_e010': (16, 0.10, 'normalized'), 'norm_L16_e050': (16, 0.50, 'normalized'),
           'norm_L8_e010': (8, 0.10, 'normalized'), 'norm_L4_e010': (4, 0.10, 'normalized'),
           'pl_L16_e010': (16, 0.10, 'pathloss'), 'pl_L16_e050': (16, 0.50, 'pathloss')}
    for key, (L, eps, gm) in PTS.items():
        runs = []
        for t in range(Q(20, 3)):
            sd = 5000 + 97 * t
            r = C.run_snapshot_trace(C.CFConfig(L=L, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10,
                                     eps_override=eps, methods=MET, seed=sd, soi_in_cov=True, gain_model=gm)
            runs.append({'seed': sd, **{m: float(np.nanmean(v)) for m, v in r['sinr_dB'].items()}})
        res = {'config': dict(L=L, eps=eps, gain_model=gm), 'runs': runs,
               'mean': {m: float(np.mean([x[m] for x in runs])) for m in MET}}
        for b in MET[1:]:
            res['D-' + b] = paired([x['D-CF-RAB'] for x in runs], [x[b] for x in runs])
        out['A_paired'][key] = res
        print(f"  {key}: D-CF-RAB {res['mean']['D-CF-RAB']:.2f} dB", flush=True)
    for eps in (0.10, 0.50):
        ex, asr = [], []
        for t in range(Q(20, 3)):
            sd = 12000 + 97 * t
            ex.append(float(np.nanmean(C.run_snapshot_trace(C.CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10,
                      INR_dB=10, eps_override=eps, methods=('D-CF-RAB',), seed=sd, soi_in_cov=True)['sinr_dB']['D-CF-RAB'])))
            orig, mk = with_rab(local_solver='active_set', nu_c=0.9); C.RABConfig = mk
            try:
                asr.append(float(np.nanmean(C.run_snapshot_trace(C.CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10,
                           INR_dB=10, eps_override=eps, methods=('D-CF-RAB',), seed=sd, soi_in_cov=True)['sinr_dB']['D-CF-RAB'])))
            finally:
                C.RABConfig = orig
        out['B_local_stage'][f'eps{eps}'] = {'exact': ex, 'active_set': asr, 'exact_minus_active_set': paired(ex, asr)}
    return out


def socp_same(R_, cols, nus):
    """Conic solution of (6.7) with the given constraint columns and bounds (Clarabel through CVXPY)."""
    import cvxpy as cp
    n = R_.shape[0]; Lc = cholesky(R_ + 1e-12 * np.eye(n)); w = cp.Variable(n, complex=True)
    cons = [cp.real(cols[:, i].conj() @ w) >= nus[i] * cp.norm(w, 2) + 1 for i in range(cols.shape[1])]
    p = cp.Problem(cp.Minimize(cp.sum_squares(Lc.conj().T @ w)), cons); p.solve(solver=cp.CLARABEL)
    return (w.value if p.status == 'optimal' else None), p.status


def stage_audits():
    """Same-constraint conic audits (joint and local), CF-RAB bound sweep and cross-AP rescaling outcomes."""
    out = {'C_joint': {}, 'D_cfrab_bound': {}, 'violations': {}}
    for eps in (0.01, 0.10, 0.50):
        cap = []; _ac = C.beamformer_cf_rab

        def spy(R_, h, nu, cfg, *a, **k):
            w = _ac(R_, h, nu, cfg, *a, **k)
            cap.append((R_, h, np.asarray(nu), k.get('eps_total', a[0] if a else 0.1), w)); return w
        C.beamformer_cf_rab = spy
        try:
            C.run_snapshot_trace(C.CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10, eps_override=eps,
                                 methods=('CF-RAB',), seed=700, soi_in_cov=True)
        finally:
            C.beamformer_cf_rab = _ac
        rows = []
        for (R_, h, nu, e, w) in cap[:Q(12, 3)]:
            cols = C.build_constraint_grid(h, e, N=len(nu)); o = lambda v: float(np.real(np.vdot(v, R_ @ v)))
            wm, sm = socp_same(R_, cols, nu); wp, sp = socp_same(R_, cols[:, :1], nu[:1])
            marg = np.real(w.conj() @ cols) - (nu * norm(w) + 1.0)
            rows.append({'status_same': sm, 'status_primary': sp, 'alg_obj': o(w), 'same_obj': o(wm) if wm is not None else None,
                         'primary_obj': o(wp) if wp is not None else None, 'alg_min_margin': float(marg.min())})
        ok = [r for r in rows if r['same_obj'] and r['primary_obj']]
        out['C_joint'][str(eps)] = {'instances': rows, 'n_ok': len(ok),
                                    'alg_vs_same_pct': [100 * (r['alg_obj'] / r['same_obj'] - 1) for r in ok],
                                    'same_vs_primary_pct': [100 * (r['same_obj'] / r['primary_obj'] - 1) for r in ok]}
    cap = []; _rs = C.rab_step

    def spy2(R_, w, l, nu, cfg, rho=1.0):
        o_ = _rs(R_, w, l, nu, cfg, rho)
        if l.shape[0] == 8 and len(cap) < Q(80, 10):
            cap.append((R_, l, np.asarray(nu), o_[0]))
        return o_
    orig, mk = with_rab(local_solver='active_set', nu_c=0.9); C.RABConfig = mk; C.rab_step = spy2
    try:
        C.run_snapshot_trace(C.CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10, eps_override=0.10,
                             methods=('D-CF-RAB',), seed=700, soi_in_cov=True)
    finally:
        C.RABConfig = orig; C.rab_step = _rs
    rows = []
    for R_, l, nu, w in cap:
        ws, st = socp_same(R_, l, nu); o = lambda v: float(np.real(np.vdot(v, R_ @ v)))
        if ws is None:
            continue
        rows.append({'excess_ratio': o(w) / o(ws), 'min_margin': float((np.real(w.conj() @ l) - (nu * norm(w) + 1)).min()),
                     'direction_cos': float(abs(np.vdot(w, ws)) / (norm(w) * norm(ws)))})
    out['C_local'] = {'n': len(rows), 'rows': rows}
    for c in ('declared', 0.5, 0.6, 0.7):
        vals = []
        for t in range(Q(5, 2)):
            _ac = C.beamformer_cf_rab
            if c != 'declared':
                def forced(R_, h, nu, cfg, *a, _c=c, **k):
                    k.pop('eps_total', None)
                    return _ac(R_, h, np.full(len(nu), _c * norm(h)), cfg, eps_total=_c)
                C.beamformer_cf_rab = forced
            try:
                r = C.run_snapshot_trace(C.CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10, eps_override=0.10,
                                         methods=('CF-RAB',), seed=700 + 37 * t, soi_in_cov=True)
            finally:
                C.beamformer_cf_rab = _ac
            vals.append(float(np.nanmean(r['sinr_dB']['CF-RAB'])))
        out['D_cfrab_bound'][str(c)] = {'values': vals, 'mean': float(np.mean(vals))}
    for eps in (0.10, 0.50):
        C.JOINT_MODES.clear()
        for t in range(Q(5, 2)):
            C.run_snapshot_trace(C.CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10, eps_override=eps,
                                 methods=('D-CF-RAB',), seed=700 + 37 * t, soi_in_cov=True)
        out['violations'][str(eps)] = dict(C.JOINT_MODES)
    return out


LB_SET = {'heavy_e010': (4, 16, 0.10, 'normalized'), 'heavy_e050': (4, 16, 0.50, 'normalized'),
          'small_e010': (4, 4, 0.10, 'normalized'), 'pl_e050': (16, 4, 0.50, 'pathloss'),
          'nom_e010': (16, 4, 0.10, 'normalized'), 'nom_e050': (16, 4, 0.50, 'normalized')}


def stage_lbound():
    """Load-dependent selection of the D-CF-RAB bound and of the LP-MMSE and P-MMSE error scales (Section 6.7.12)."""
    LB_C = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.98, 0.99, 0.995, 0.999)
    LB_LP = (0.001, 0.01, 0.03, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 10000.0)
    LB_P = (10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
    LB_TRAIN = [400000 + 37 * j for j in range(Q(8, 2))]; LB_TEST = [600000 + 37 * j for j in range(Q(20, 3))]

    def lb_run(L, K, eps, gm, seed, methods, c=None, pscale=1.0, lpscale=1.0):
        C.LOCAL_C_OVERRIDE = c; C.PMMSE_C_SCALE = pscale; C.LP_C_SCALE = lpscale
        try:
            r = C.run_snapshot_trace(C.CFConfig(L=L, M_l=8, K=K), n_snapshots=60, SNR_dB=10, INR_dB=10, eps_override=eps,
                                     methods=methods, seed=seed, soi_in_cov=True, gain_model=gm)
        finally:
            C.LOCAL_C_OVERRIDE = None; C.PMMSE_C_SCALE = 1.0; C.LP_C_SCALE = 1.0
        return ({m: float(np.nanmean(v)) for m, v in r['sinr_dB'].items()},
                {m: float(np.nanmin(v)) for m, v in r['sinr_dB'].items()})
    out = {}
    for key, (L, K, eps, gm) in LB_SET.items():
        tr = {}
        for c in LB_C:
            tr[f'D_c{c}'] = [lb_run(L, K, eps, gm, sd, ('D-CF-RAB',), c=c)[0]['D-CF-RAB'] for sd in LB_TRAIN]
        for s_ in LB_LP:
            tr[f'LP_s{s_}'] = [lb_run(L, K, eps, gm, sd, ('LP-MMSE+W',), lpscale=s_)[0]['LP-MMSE+W'] for sd in LB_TRAIN]
        for s_ in LB_P:
            tr[f'P_s{s_}'] = [lb_run(L, K, eps, gm, sd, ('P-MMSE',), pscale=s_)[0]['P-MMSE'] for sd in LB_TRAIN]
        sel = dict(c=max(LB_C, key=lambda c: np.mean(tr[f'D_c{c}'])),
                   lp_scale=max(LB_LP, key=lambda s_: np.mean(tr[f'LP_s{s_}'])),
                   p_scale=max(LB_P, key=lambda s_: np.mean(tr[f'P_s{s_}'])))
        rows = []
        for sd in LB_TEST:
            a, aw = lb_run(L, K, eps, gm, sd, ('D-CF-RAB',), c=sel['c'])
            b, bw = lb_run(L, K, eps, gm, sd, ('D-CF-RAB', 'MR+W', 'LP-MMSE+W', 'P-MMSE', 'LMVDR+W'))
            lp, _ = lb_run(L, K, eps, gm, sd, ('LP-MMSE+W',), lpscale=sel['lp_scale'])
            p, _ = lb_run(L, K, eps, gm, sd, ('P-MMSE',), pscale=sel['p_scale'])
            rows.append({'seed': sd, 'D_tuned': a['D-CF-RAB'], 'D_default': b['D-CF-RAB'], 'MR': b['MR+W'],
                         'LP_default': b['LP-MMSE+W'], 'P_default': b['P-MMSE'], 'LMVDR': b['LMVDR+W'],
                         'LP_tuned': lp['LP-MMSE+W'], 'P_tuned': p['P-MMSE'],
                         'D_tuned_worst': aw['D-CF-RAB'], 'MR_worst': bw['MR+W']})
        T = {'mean': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'seed'}, 'runs': rows}
        for a_, b_ in (('D_tuned', 'MR'), ('D_tuned', 'D_default'), ('D_tuned', 'LP_tuned'), ('D_tuned', 'LP_default'),
                       ('D_tuned', 'P_tuned'), ('D_tuned', 'P_default'), ('D_default', 'P_default'), ('D_default', 'P_tuned'),
                       ('D_tuned_worst', 'MR_worst')):
            T[f'{a_} - {b_}'] = paired([r[a_] for r in rows], [r[b_] for r in rows])
        out[key] = {'config': dict(L=L, K=K, eps=eps, gain_model=gm), 'selected': sel,
                    'train_means': {k: float(np.mean(v)) for k, v in tr.items()}, 'train_runs': tr, 'test': T}
        print(f"  {key}: c = {sel['c']}, D-CF-RAB {T['mean']['D_tuned']:.2f} dB", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ error laws, audits (A)
MODELS = ('boundary', 'interior', 'radius', 'gaussian')
MET_A = ('D-CF-RAB', 'MR+W', 'LP-MMSE+W', 'LMVDR+W', 'P-MMSE', 'LP-MMSE', 'CF-RAB')


def stage_A1():
    """Error laws with 20 paired realizations (Table 6.4)."""
    CFG = {'norm_L16_e010': (16, 0.10, 'normalized'), 'norm_L16_e050': (16, 0.50, 'normalized'),
           'pl_L16_e050': (16, 0.50, 'pathloss')}
    out = {}
    for key, (L, eps, gm) in CFG.items():
        for em in MODELS:
            tag = f'{key}|{em}'; C.ERR_MODEL = em; runs = []
            for t in range(Q(20, 3)):
                sd = 5000 + 97 * t; C.RESPONSE_LOG.clear(); C.JOINT_LOG.clear()
                r = C.run_snapshot_trace(C.CFConfig(L=L, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10, eps_override=eps,
                                         methods=MET_A, seed=sd, soi_in_cov=True, gain_model=gm)
                resp = [x[0] for x in C.RESPONSE_LOG]
                runs.append({'seed': sd, **{m: float(np.nanmean(r['sinr_dB'][m])) for m in MET_A},
                             **{m + '_worst': float(np.nanmin(r['sinr_dB'][m])) for m in MET_A},
                             'resp_min': float(min(resp)), 'resp_below1': int(sum(x < 1 - 1e-9 for x in resp)),
                             'resp_n': len(resp),
                             'joint_scored': dict(collections.Counter(m for _, sc, m in C.JOINT_LOG if sc))})
            C.ERR_MODEL = 'boundary'
            res = {'config': dict(L=L, eps=eps, gain_model=gm, err=em), 'runs': runs,
                   'mean': {m: float(np.mean([x[m] for x in runs])) for m in MET_A},
                   'worst': {m: float(np.mean([x[m + '_worst'] for x in runs])) for m in MET_A},
                   'resp_min': min(x['resp_min'] for x in runs), 'resp_below1': sum(x['resp_below1'] for x in runs),
                   'resp_n': sum(x['resp_n'] for x in runs)}
            for b in MET_A[1:]:
                res['D-' + b] = paired([x['D-CF-RAB'] for x in runs], [x[b] for x in runs])
            out[tag] = res
            print(f"  A1 {tag}: D-CF-RAB {res['mean']['D-CF-RAB']:.2f} dB", flush=True)
    return out


def stage_A2():
    """Bound selection of D-CF-RAB under each error law (eight training realizations)."""
    CG = (0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999)
    out = {}
    for eps in (0.10, 0.50):
        for em in MODELS:
            tag = f'eps{eps}|{em}'; C.ERR_MODEL = em; tr = {}
            for c in CG:
                C.LOCAL_C_OVERRIDE = c
                tr[str(c)] = [float(np.nanmean(C.run_snapshot_trace(C.CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10,
                              INR_dB=10, eps_override=eps, methods=('D-CF-RAB',), seed=400000 + 37 * j,
                              soi_in_cov=True)['sinr_dB']['D-CF-RAB'])) for j in range(Q(8, 2))]
            C.LOCAL_C_OVERRIDE = None; C.ERR_MODEL = 'boundary'
            means = {c: float(np.mean(v)) for c, v in tr.items()}
            out[tag] = {'train': tr, 'means': means, 'selected': max(means, key=means.get)}
    return out


def stage_A3():
    """Figure 6.5 configuration with equal-mean tiers (0.05 and 0.25, mean radius 0.10), same seeds as Figure 6.5."""
    C.TIER_MODE = 'equal_mean'; n = Q(5, 2); acc = {m: [] for m in M9}; wst = {m: [] for m in M9}
    for t in range(n):
        r = C.run_snapshot_trace(C.CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10, heterogeneous=True,
                                 methods=M9, seed=300 + 37 * t, soi_in_cov=True)
        for m in M9:
            v = np.asarray(r['sinr_dB'][m], float); acc[m].append(float(np.nanmean(v))); wst[m].append(float(np.nanmin(v)))
    C.TIER_MODE = 'random'
    tq = float(tdist.ppf(.975, n - 1))
    A3 = {m: [float(np.mean(acc[m])), float(tq * np.std(acc[m], ddof=1) / np.sqrt(n)), float(np.mean(wst[m])), acc[m]]
          for m in M9}
    return {'A3': A3, 'A3_paired': {b: paired(acc['D-CF-RAB'], acc[b]) for b in M9[1:]}}


def stage_A4():
    """Joint-check audit on all and on scored snapshots, and continuous-ball check of the passed D-CF-RAB updates."""
    out = {}; rng = np.random.default_rng(2026)
    for eps in (0.10, 0.50):
        C.JOINT_LOG.clear(); C.RESPONSE_LOG.clear(); cap = []
        _orig = C.cf_distributed_step

        def spy(R_l, w_l, h_hat, eps_l, nu_lk, rab_cfg, dist_cfg, rho_state, user_k, **kw):
            o = _orig(R_l, w_l, h_hat, eps_l, nu_lk, rab_cfg, dist_cfg, rho_state, user_k, **kw)
            cap.append(([x.copy() for x in o[0]], h_hat[:, user_k].copy(), np.asarray(eps_l).copy(), o[2].get('joint_mode')))
            return o
        C.cf_distributed_step = spy
        try:
            for t in range(Q(5, 1)):
                C.run_snapshot_trace(C.CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10, eps_override=eps,
                                     methods=('D-CF-RAB',), seed=700 + 37 * t, soi_in_cov=True)
        finally:
            C.cf_distributed_step = _orig
        allm = collections.Counter(m for _, _, m in C.JOINT_LOG)
        scm = collections.Counter(m for _, sc, m in C.JOINT_LOG if sc)
        wc, mc, n_ok = [], [], 0
        for wl, hk, el, mode in cap:
            if mode == 'violated':
                continue
            n_ok += 1
            base = sum(float(np.real(np.vdot(w_, h_))) for w_, h_ in zip(wl, hk))
            pen = sum(e_ * norm(h_) * norm(w_) for w_, h_, e_ in zip(wl, hk, el))
            wc.append(base - pen)
            if len(mc) < Q(300, 5):            # 2000 draws uniform inside each AP's ball, plus the analytic worst direction
                best = np.inf
                for _ in range(Q(2000, 200)):
                    tot = 0.0
                    for w_, h_, e_ in zip(wl, hk, el):
                        u = rng.normal(size=h_.size) + 1j * rng.normal(size=h_.size); u /= norm(u)
                        rr = e_ * norm(h_) * rng.random() ** (1 / (2 * h_.size))
                        tot += float(np.real(np.vdot(w_, h_ + rr * u)))
                    best = min(best, tot)
                worst_dir = sum(float(np.real(np.vdot(w_, h_ - e_ * norm(h_) * w_ / norm(w_)))) for w_, h_, e_ in zip(wl, hk, el))
                mc.append((best, worst_dir))
        out[str(eps)] = {'all_updates': dict(allm), 'scored_updates': dict(scm), 'n_checked': n_ok,
                         'analytic_worst_min': float(min(wc)), 'mc_min': float(min(b for b, _ in mc)),
                         'worst_dir_min': float(min(d for _, d in mc)), 'mc_n': len(mc)}
    return out


def stage_A5():
    """Continuous primary-ball check of CF-RAB (Algorithm 6.1 with the grid of (6.11))."""
    out = {}; rng = np.random.default_rng(7)
    for eps in (0.01, 0.10, 0.50):
        cap = []; _ac = C.beamformer_cf_rab

        def spy(R_, h, nu, cfg, *a, **k):
            w = _ac(R_, h, nu, cfg, *a, **k)
            cap.append((h.copy(), np.asarray(nu).copy(), k.get('eps_total', a[0] if a else None), w.copy())); return w
        C.beamformer_cf_rab = spy
        try:
            C.run_snapshot_trace(C.CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10, eps_override=eps,
                                 methods=('CF-RAB',), seed=700, soi_in_cov=True)
        finally:
            C.beamformer_cf_rab = _ac
        rows = []
        for h, nu, e_tot, w in cap[:Q(12, 3)]:
            nu0 = float(nu[0])                       # primary ball radius: continuous worst case over the whole ball
            analytic = float(np.real(np.vdot(w, h))) - nu0 * norm(w)
            best = np.inf
            for _ in range(Q(5000, 500)):
                u = rng.normal(size=h.size) + 1j * rng.normal(size=h.size); u /= norm(u)
                best = min(best, float(np.real(np.vdot(w, h + nu0 * rng.random() ** (1 / (2 * h.size)) * u))))
            vert = min(float(np.real(np.vdot(w, l))) - nu_i * norm(w)
                       for l, nu_i in zip(C.build_constraint_grid(h, e_tot, N=len(nu)).T, nu))
            rows.append({'analytic_primary': analytic, 'mc_min': best, 'grid_columns_min': vert})
        out[str(eps)] = {'n': len(rows), 'analytic_primary_min': min(r['analytic_primary'] for r in rows),
                         'mc_min': min(r['mc_min'] for r in rows), 'grid_min': min(r['grid_columns_min'] for r in rows)}
    return out


# ------------------------------------------------------------------------------------------------ loading and interferers (B)
KNOBS = dict(LOCAL_C_OVERRIDE=None, LMVDR_LOAD=1e-6, ACPU_LOADING=1e-3, LP_C_SCALE=1.0, PMMSE_C_SCALE=1.0, N_EXT_INTERF=0,
             COV_MODEL='population', USER_SYMBOLS='gaussian', EXT_SYMBOLS='gaussian')


def run_b(L, K, eps, gm, seed, methods, inr=10.0, **kn):
    """One trace with the given experiment switches; returns (mean, worst-user) SINR per method."""
    for k_, v_ in KNOBS.items():
        setattr(C, k_, v_)
    for k_, v_ in kn.items():
        setattr(C, k_, v_)
    try:
        r = C.run_snapshot_trace(C.CFConfig(L=L, M_l=8, K=K), n_snapshots=60, SNR_dB=10, INR_dB=inr, eps_override=eps,
                                 methods=methods, seed=seed, soi_in_cov=True, gain_model=gm)
    finally:
        for k_, v_ in KNOBS.items():
            setattr(C, k_, v_)
    return {m: (float(np.nanmean(v)), float(np.nanmin(v))) for m, v in r['sinr_dB'].items() if not m.endswith('_all')}


B_SET = {'heavy_e010': (4, 16, 0.10, 'normalized'), 'heavy_e050': (4, 16, 0.50, 'normalized'),
         'small_e010': (4, 4, 0.10, 'normalized'), 'nom_e010': (16, 4, 0.10, 'normalized'),
         'nom_e050': (16, 4, 0.50, 'normalized'), 'pl_e050': (16, 4, 0.50, 'pathloss')}


def stage_B1():
    """Table 6.2 and 6.3 configurations with training-selected bound, MVDR loading and error scales, exact and sample covariance."""
    CG = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.98, 0.99, 0.995, 0.999)
    LG = (1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)
    LPG = (0.001, 0.01, 0.03, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 10000.0)
    PG = (10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
    TRAIN = [400000 + 37 * j for j in range(Q(8, 2))]; TEST = [600000 + 37 * j for j in range(Q(20, 3))]
    out = {}
    for cov in ('population', 'sample'):
        for key, (L, K, eps, gm) in B_SET.items():
            tag = f'{key}|{cov}'; tr = {}
            for c in CG:
                tr[f'D_{c}'] = [run_b(L, K, eps, gm, sd, ('D-CF-RAB',), COV_MODEL=cov, LOCAL_C_OVERRIDE=c)['D-CF-RAB'][0] for sd in TRAIN]
            for g in LG:
                tr[f'MV_{g}'] = [run_b(L, K, eps, gm, sd, ('LMVDR+W',), COV_MODEL=cov, LMVDR_LOAD=g)['LMVDR+W'][0] for sd in TRAIN]
            for s_ in LPG:
                tr[f'LP_{s_}'] = [run_b(L, K, eps, gm, sd, ('LP-MMSE+W',), COV_MODEL=cov, LP_C_SCALE=s_)['LP-MMSE+W'][0] for sd in TRAIN]
            for s_ in PG:
                tr[f'P_{s_}'] = [run_b(L, K, eps, gm, sd, ('P-MMSE',), COV_MODEL=cov, PMMSE_C_SCALE=s_)['P-MMSE'][0] for sd in TRAIN]
            pick = lambda pre, grid: max(grid, key=lambda v: np.mean(tr[f'{pre}_{v}']))
            sel = dict(c=pick('D', CG), load=pick('MV', LG), lp=pick('LP', LPG), p=pick('P', PG))
            rows = []
            for sd in TEST:
                a = run_b(L, K, eps, gm, sd, ('D-CF-RAB',), COV_MODEL=cov, LOCAL_C_OVERRIDE=sel['c'])
                b = run_b(L, K, eps, gm, sd, ('D-CF-RAB', 'MR+W', 'LMVDR+W', 'LP-MMSE+W', 'P-MMSE'), COV_MODEL=cov)
                mv = run_b(L, K, eps, gm, sd, ('LMVDR+W',), COV_MODEL=cov, LMVDR_LOAD=sel['load'])
                lp = run_b(L, K, eps, gm, sd, ('LP-MMSE+W',), COV_MODEL=cov, LP_C_SCALE=sel['lp'])
                p = run_b(L, K, eps, gm, sd, ('P-MMSE',), COV_MODEL=cov, PMMSE_C_SCALE=sel['p'])
                rows.append({'seed': sd, 'D_tuned': a['D-CF-RAB'][0], 'D_tuned_worst': a['D-CF-RAB'][1],
                             'D_default': b['D-CF-RAB'][0], 'MR': b['MR+W'][0], 'MR_worst': b['MR+W'][1],
                             'MV_default': b['LMVDR+W'][0], 'MV_tuned': mv['LMVDR+W'][0], 'MV_tuned_worst': mv['LMVDR+W'][1],
                             'LP_tuned': lp['LP-MMSE+W'][0], 'P_tuned': p['P-MMSE'][0], 'P_default': b['P-MMSE'][0]})
            T = {'mean': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'seed'}, 'runs': rows}
            for a_, b_ in (('D_tuned', 'MR'), ('D_tuned', 'MV_tuned'), ('D_tuned', 'MV_default'), ('D_default', 'MV_default'),
                           ('D_tuned', 'D_default'), ('D_tuned', 'LP_tuned'), ('D_tuned', 'P_tuned'),
                           ('D_tuned_worst', 'MV_tuned_worst'), ('MV_tuned', 'MR')):
                T[f'{a_} - {b_}'] = paired([r[a_] for r in rows], [r[b_] for r in rows])
            out[tag] = {'config': dict(L=L, K=K, eps=eps, gain_model=gm, cov=cov), 'selected': sel,
                        'train_means': {k: float(np.mean(v)) for k, v in tr.items()}, 'test': T}
            print(f"  B1 {tag}: selected {sel}, D-CF-RAB {T['mean']['D_tuned']:.2f} dB", flush=True)
    return out


def stage_B2():
    """Interferers without channel estimates with Gaussian, QPSK, 16-QAM and impulsive symbols (Table 6.5)."""
    SC = {'J2_inr10': (2, 10.0), 'J2_inr20': (2, 20.0), 'J6_inr10': (6, 10.0), 'J6_inr20': (6, 20.0)}
    LAWS = {'gaussian': ('gaussian', 'gaussian'), 'qam_qpsk': ('16qam', 'qpsk'), 'qam_impulsive': ('16qam', 'impulsive')}
    GRID = {'D': ('LOCAL_C_OVERRIDE', (0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.98), 'D-CF-RAB'),
            'MV': ('LMVDR_LOAD', (1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0), 'LMVDR+W'),
            'ACPU': ('ACPU_LOADING', (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0), 'MR+ACPU'),
            'LP': ('LP_C_SCALE', (0.001, 1.0, 10000.0), 'LP-MMSE+W'),
            'P': ('PMMSE_C_SCALE', (1.0, 0.1, 0.01, 0.001), 'P-MMSE')}
    TR2 = [400000 + 37 * j for j in range(Q(6, 2))]; TE2 = [600000 + 37 * j for j in range(Q(12, 3))]
    out = {}
    for key, (J, inr) in SC.items():
        tr, sel = {}, {}
        for name, (knob, grid, meth) in GRID.items():
            for v in grid:
                tr[f'{name}_{v}'] = [run_b(16, 4, 0.10, 'normalized', sd, (meth,), inr=inr, N_EXT_INTERF=J, COV_MODEL='sample',
                                           **{knob: v})[meth][0] for sd in TR2]
            sel[name] = max(grid, key=lambda v: np.mean(tr[f'{name}_{v}']))
        out[key] = {'config': dict(J=J, INR_dB=inr, L=16, K=4, eps=0.10, cov='sample'), 'selected': sel,
                    'train_means': {k: float(np.mean(v)) for k, v in tr.items()}, 'test': {}}
        for law, (us, es) in LAWS.items():
            kw = dict(inr=inr, N_EXT_INTERF=J, COV_MODEL='sample', USER_SYMBOLS=us, EXT_SYMBOLS=es)
            rows = []
            for sd in TE2:
                row = {'seed': sd}
                base = run_b(16, 4, 0.10, 'normalized', sd, ('D-CF-RAB', 'MR+W', 'CF-RAB'), **kw)
                row['D_c098'] = base['D-CF-RAB'][0]; row['MR'], row['MR_worst'] = base['MR+W']
                row['CF'], row['CF_worst'] = base['CF-RAB']
                for name, (knob, grid, meth) in GRID.items():
                    row[name], row[name + '_worst'] = run_b(16, 4, 0.10, 'normalized', sd, (meth,), **{knob: sel[name]}, **kw)[meth]
                rows.append(row)
            T = {'mean': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'seed'}, 'runs': rows}
            for b_ in ('MR', 'LP', 'P', 'MV', 'ACPU', 'CF', 'D_c098'):
                T[f'D - {b_}'] = paired([r['D'] for r in rows], [r[b_] for r in rows])
            for b_ in ('MR', 'MV'):
                T[f'D_worst - {b_}_worst'] = paired([r['D_worst'] for r in rows], [r[b_ + '_worst'] for r in rows])
            out[key]['test'][law] = T
            print(f"  B2 {key} {law}: D-CF-RAB {T['mean']['D']:.2f} dB", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ runner
# longest stages first, so that two processes finish together; fig6 (timing) runs alone afterwards
GROUPS = ['B1', 'lbound', 'A1', 'A2', 'fig2', 'paired', 'B2', 'audits', 'train', 'fig1', 'fig3', 'fig4', 'fig7', 'fig8',
          'fig9', 'fig5', 'A4', 'fig10', 'A3', 'A5']
TIMING = 'fig6'
ALL_STAGES = GROUPS + [TIMING]

# stages needed by each figure / table / text block
NEEDS = {
    '6_1': ['fig1'], '6_2': ['fig2'], '6_3': ['fig3'], '6_4': ['fig4'], '6_5': ['fig5'], '6_6': ['fig6'], '6_7': ['fig7'],
    '6_8': ['fig8', 'fig2'], '6_9': ['fig9'], '6_10': ['fig10'],
    'tables': ['B1', 'B2', 'A1'],
    'text': ['train', 'paired', 'audits', 'lbound', 'A2', 'A3', 'A4', 'A5', 'fig6'],
}


def groups_for(keys):
    need = []
    for k in keys:
        for s in NEEDS[k]:
            if s not in need:
                need.append(s)
    return [s for s in ALL_STAGES if s in need]


def load(datadir, stage):
    p = os.path.join(datadir, stage + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def _tolist(o):
    if hasattr(o, 'tolist'):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def run_stage(stage, datadir):
    """Run one stage in this process and write <datadir>/<stage>.json."""
    os.makedirs(datadir, exist_ok=True)
    t0 = time.time()
    res = globals()['stage_' + stage]()
    res = {'stage': stage, 'quick': QUICK, 'seconds': round(time.time() - t0, 1), 'result': res}
    json.dump(res, open(os.path.join(datadir, stage + '.json'), 'w'), indent=1, default=_tolist)
    print(f'stage {stage} done in {time.time() - t0:.0f} s', flush=True)


def run_groups(datadir, groups=None, jobs=2, quick=False):
    """Run the stages, each in its own process, at most `jobs` at a time; fig6 (timing) is run alone at the end."""
    groups = list(GROUPS if groups is None else groups)
    os.makedirs(datadir, exist_ok=True)
    env = dict(os.environ, CH6_QUICK='1' if quick else '0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')

    def launch(s):
        log = open(os.path.join(datadir, s + '.log'), 'w')
        return subprocess.Popen([sys.executable, '-m', 'ch6cfrab.scenarios', s, datadir], stdout=log,
                                stderr=subprocess.STDOUT, env=env, cwd=root), log
    timed = [s for s in groups if s == TIMING]
    queue = [s for s in groups if s != TIMING]
    running = {}
    t0 = time.time()
    while queue or running:
        while queue and len(running) < max(1, jobs):
            s = queue.pop(0); running[s] = launch(s)
            print(f'[{time.time() - t0:6.0f} s] start {s}', flush=True)
        time.sleep(1.0)
        for s, (p, log) in list(running.items()):
            if p.poll() is not None:
                log.close(); del running[s]
                if p.returncode != 0:
                    raise RuntimeError(f'stage {s} failed; see {os.path.join(datadir, s + ".log")}')
                print(f'[{time.time() - t0:6.0f} s] done  {s}', flush=True)
    for s in timed:                                   # alone, so that no other process competes for the CPU
        p, log = launch(s); p.wait(); log.close()
        if p.returncode != 0:
            raise RuntimeError(f'stage {s} failed; see {os.path.join(datadir, s + ".log")}')
        print(f'[{time.time() - t0:6.0f} s] done  {s}', flush=True)


if __name__ == '__main__':
    run_stage(sys.argv[1], sys.argv[2])
