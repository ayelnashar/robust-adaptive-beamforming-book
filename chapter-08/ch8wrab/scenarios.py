"""
ch8wrab.scenarios -- the experiments behind Figures 8.1-8.11, Tables 8.2-8.3 and the numbers quoted in Chapter 8.

Each stage writes <datadir>/<stage>.json. The runner starts one process per stage, at most `jobs` at a time; the
timing stage runs alone at the end so that no other process competes for the CPU while it is measured.

    python -m ch8wrab.scenarios <stage> <datadir>        # one stage

Set CH8_QUICK=1 for a smoke test with two or three trials per point. Every trial draws its random numbers from
seed_for(stream, trial) with the stream names below, so a full run regenerates the chapter's values exactly.
"""
import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.special import roots_hermitenorm

from . import algorithms as C
from .algorithms import Scenario, run_trials, seed_for, summarize

QUICK = os.environ.get('CH8_QUICK', '0') == '1'


def Q(full, quick):
    return quick if QUICK else full


def slim(r):
    """JSON-ready record of run_trials: summaries, per-trial values, per-subcarrier linear SINR, diagnostics."""
    return {'label': r['label'], 'NR': r['NR'], 'methods': r['methods'], 'config': r['config'],
            'summary': r['summary'],
            'samples': {m: np.asarray(v).tolist() for m, v in r['samples'].items()},
            'raw_linear': {m: np.asarray(v).tolist() for m, v in r['raw_linear'].items()},
            'max_residual': max([v for d in r['diagnostics'] for v in d['max_residual'].values()] or [None]),
            'saturation': [d['margin_saturation_fraction'] for d in r['diagnostics']],
            'final_margins': [d['final_margins'] for d in r['diagnostics']],
            'seconds': [d['seconds'] for d in r['diagnostics']]}


def say(*a):
    print(*a, flush=True)


def point(tag, r):
    say(f'{tag}: ' + ', '.join(f'{m}={v["mean_db"]:.2f}' for m, v in r['summary'].items()))


# ------------------------------------------------------------------------------------------------ Figures 8.1, 8.2
def stage_analytic():
    """Figure 8.1 (carrier-fixed array-factor loss) and Figure 8.2 (expected power loss under Gaussian pointing error)."""
    fc, B, theta = 300e9, 20e9, np.deg2rad(30)
    offsets = np.linspace(-B/2, B/2, 201)
    f1 = {'frequency_offset_GHz': (offsets/1e9).tolist()}
    for M in (16, 64, 256):
        g = np.array([C.analog_response(M, theta, theta, fc+d, fc) for d in offsets])
        f1[f'N{M}'] = (20*np.log10(np.maximum(g, np.finfo(float).tiny))).tolist()
    sigma = np.linspace(0, .6, 13)
    nodes, weights = roots_hermitenorm(256); weights = weights/np.sqrt(2*np.pi)
    f2 = {'jitter_rms_deg': sigma.tolist()}
    for M in (64, 256):
        loss = []
        for sd in sigma:
            g = np.array([C.analog_response(M, 0, e, fc, fc) for e in nodes*np.deg2rad(sd)])
            loss.append(float(-10*np.log10(np.dot(weights, g**2))))       # expected POWER gain
        f2[f'N{M}'] = loss
    return {'fig1': f1, 'fig2': f2}


# ------------------------------------------------------------------------------------------------ Figures 8.3-8.11
def stage_fig3():
    M = 64
    cfg = Scenario(M=M, Nsc=11, grid_offsets_deg=(0, -1.5, 1.5, -3), single_radius_fraction=3.8/np.sqrt(M), jitter_deg=.1)
    r = run_trials(cfg, Q(15, 3), 'fig3'); point('Figure 8.3', r)
    return slim(r)


def stage_fig4():
    out = {}
    for mm in range(7):
        r = run_trials(Scenario(M=64, mismatch_deg=float(mm), jitter_deg=.15), Q(15, 3), 'fig4')
        point(f'mismatch {mm} deg', r); out[str(mm)] = slim(r)
    return out


def stage_fig5():
    out = {}
    methods = ('analog', 'wc', 'mvdr', 'wc_faware', 'fixed', 'wrab')
    for angle in (0, 15, 30, 45, 60):
        cfg = Scenario(M=64, Nsc=11, th_d_deg=angle, mismatch_deg=2, interferers_deg=(angle+25, angle-35), jitter_deg=.1)
        r = run_trials(cfg, Q(15, 3), 'fig5', methods); point(f'steering {angle} deg', r); out[str(angle)] = slim(r)
    return out


def stage_fig6():
    out = {}
    for sep in (3, 5, 8, 12, 20, 40):
        r = run_trials(Scenario(M=64, Nsc=8, interferers_deg=(sep, -55), jitter_deg=.1), Q(18, 3), 'fig6')
        point(f'separation {sep} deg', r); out[str(sep)] = slim(r)
    return out


def stage_fig7():
    """Figure 8.7(a): mismatch of 1.5 reference beamwidths and jitter of 0.3 beamwidths for N = 64 ... 512."""
    out = {}
    for M in (64, 128, 256, 512):
        bw = np.rad2deg(1.772/M)
        cfg = Scenario(M=M, Nsc=5, mismatch_deg=1.5*bw, jitter_deg=.3*bw)
        r = run_trials(cfg, Q(12, 3), 'fig7'); point(f'N={M}', r); out[str(M)] = slim(r)
    return out


def stage_fixedangle():
    """Figure 8.7(b): 1 degree of mismatch for every N, with the fan of Table 8.1 and with a fan of span +/-1.5 degrees
    and spacing 0.6 beamwidths (N_c = 2 ceil(1.5/(0.6 bw)) + 1 constraints)."""
    out = {}
    mismatch, budget, step_bw = 1.0, 1.5, 0.6
    for M in (64, 128, 256, 512):
        cfgA = Scenario(M=M, Nsc=5, mismatch_deg=mismatch, jitter_deg=0.1)
        rA = run_trials(cfgA, Q(12, 3), 'fig7f', ('analog', 'wc', 'wrab'))
        bw = np.rad2deg(1.772/(M*np.cos(np.deg2rad(mismatch))))
        n = int(np.ceil(budget/(step_bw*bw)))
        offs = tuple(float(x) for x in np.arange(-n, n+1)*step_bw*bw)
        cfgB = replace(cfgA, grid_offsets_deg=offs, initial_margins=(1.5,))
        rB = run_trials(cfgB, Q(12, 3), 'fig7f', ('wrab',))
        row = {m: rA['summary'][m] for m in ('analog', 'wc', 'wrab')}
        row['wrab_anglefan'] = rB['summary']['wrab']
        row.update(Nc_anglefan=len(offs), bw_deg=bw, max_residual=max(slim(rA)['max_residual'], slim(rB)['max_residual']),
                   ms_per_frequency_snapshot_anglefan=1e3*float(np.mean([d['seconds'] for d in rB['diagnostics']]))
                   / (cfgB.Nsc*cfgB.K))
        say(f'N={M}, N_c={len(offs)}:', {k: round(v['mean_db'], 2) for k, v in row.items() if isinstance(v, dict)})
        out[str(M)] = row
    return out


def stage_fig8():
    out = {}
    for j, sd in enumerate((0, .1, .2, .3, .4, .5, .6)):
        cfg = Scenario(M=128, Nsc=5, mismatch_deg=0, jitter_deg=sd, grid_offsets_deg=(0, -.4, .4, -.8))
        r = run_trials(cfg, Q(20, 3), 'fig8'); point(f'jitter {sd} deg', r); out[str(sd)] = slim(r)
    return out


def stage_dynamic(rho=0.99, window=50):
    """Figure 8.8(b): AR(1) whole-array rotation; linear SINR averaged over the last `window` snapshots, each with its
    own weight vector and geometry, then converted to dB."""
    methods = ('analog', 'wc', 'wrab'); out = {'rho': rho, 'window': window, 'points': {}}
    NR = Q(20, 3)
    for sd in (0, .1, .2, .3, .4, .5, .6):
        cfg = Scenario(M=128, Nsc=5, mismatch_deg=0, jitter_deg=sd, grid_offsets_deg=(0, -.4, .4, -.8))
        ps = 10**(cfg.snr_db/10); inr = 10**(cfg.inr_db/10); per = {m: [] for m in methods}
        for tr in range(NR):
            X, fl, H, A = C.generate_dynamic_trial(cfg, seed_for('fig8dyn', tr), rho)
            vals = {m: [] for m in methods}
            for n, f in enumerate(fl):
                t = C.train_subband(X[n], f, cfg, methods)
                for m in methods:
                    lin = []
                    for k in range(cfg.K-window, cfg.K):
                        Ai = A[n][:, :, k]; Rk = np.eye(cfg.M, dtype=complex) + inr*(Ai @ Ai.conj().T)
                        lin.append(C.sinr_linear(t['histories'][m][k], H[n][:, k], Rk, ps))
                    vals[m].append(10*np.log10(np.mean(lin)))
            for m in methods:
                per[m].append(float(np.mean(vals[m])))
        out['points'][str(sd)] = {m: summarize(per[m], f'fig8dyn|bootstrap|{m}') for m in methods}
        say(f'rotation {sd} deg:', {m: round(out['points'][str(sd)][m]['mean_db'], 2) for m in methods})
    return out


def stage_fig9():
    out = {}
    for aware in (True, False):
        cfg = Scenario(M=128, B=40e9, Nsc=11, grid_offsets_deg=(0, -1.5, 1.5, -3), jitter_deg=.1, squint_aware=aware)
        r = run_trials(cfg, Q(15, 3), 'fig9', ('wrab',)); point(f'squint-aware={aware}', r); out[str(int(aware))] = slim(r)
    return out


def stage_fig10():
    out = {}
    for adapt in (True, False):
        for i, init in enumerate((.1, 1.5, 3.0)):
            cfg = Scenario(M=128, Nsc=3, K=220, mismatch_deg=0, jitter_deg=.3, grid_offsets_deg=(0, -.4, .4, -.8),
                           initial_margins=(init,), self_anchor=adapt)
            r = run_trials(cfg, Q(20, 3), 'fig10', ('wrab',)); point(f'adapt={adapt}, initial margin {init}', r)
            out[f'{int(adapt)}_{init:g}'] = slim(r)
    return out


FIG11_MISMATCH = (0, 1, 2, 3, 4, 5, 6, 8)
FIG11_FRACS = tuple(1.2*np.array([0, -.5, .5, -1, 1]))     # symmetric fan 0, +/-0.6, +/-1.2 beamwidths


def fig11_config(mm):
    return Scenario(M=64, Nsc=5, K=600, grid_fracs=FIG11_FRACS, initial_margins=(.3,), interferers_deg=(-25, 40),
                    mismatch_deg=mm, jitter_deg=0, signal_free=True, single_radius_fraction=.85)


def stage_fig11():
    """The eight methods of Figure 8.11 that are trained online or rebuilt in batches."""
    out = {}
    for mm in FIG11_MISMATCH:
        r = run_trials(fig11_config(mm), Q(20, 2), 'fig11', C.ALL_METHODS); point(f'mismatch {mm} deg', r)
        out[str(mm)] = slim(r)
    return out


def stage_fig11ref(loads=(0.0, 1.0, 3.0, 10.0, 30.0, 100.0)):
    """Figure 8.11: exact multi-ball SOCP of (8.8) with WRAB's balls and final margins, and the multi-point LCMV at the
    fan directions (centre-referenced, loading swept; the loading with the best mean over 0-2 degrees is plotted)."""
    keys = ['wrab', 'msocp'] + [f'lcmv_{g:g}' for g in loads]
    out = {'loads': loads, 'points': {}}
    tw, ts, tl = [], [], []
    NR = Q(20, 2)
    for mm in FIG11_MISMATCH:
        cfg = fig11_config(mm)
        ps = 10**(cfg.snr_db/10); grid = C.design_grid(cfg)[1]
        per = {k: [] for k in keys}
        for tr in range(NR):
            trial = C.generate_trial(cfg, seed_for('fig11', tr))
            vals = {k: [] for k in keys}
            for n, f in enumerate(trial['frequencies']):
                t = C.train_subband(trial['X'][n], f, cfg, ('wrab',)); tw.append(1e3*t['seconds']/cfg.K)
                L = t['L']; nu = t['margin_history'][-1]
                Rb = C.normalized_covariance(t['final_R'], t['covariance_mass'], cfg.covariance_loading)
                ws = {'wrab': t['histories']['wrab'][-1]}
                ws['msocp'], dt = C.multiball_socp(Rb, L, nu); ts.append(1e3*dt)
                Lc = C.centred_steering(cfg.M, grid, f, cfg.fc)
                for g in loads:
                    t0 = time.perf_counter(); ws[f'lcmv_{g:g}'] = C.lcmv_weights(Rb, Lc, g)
                    tl.append(1e3*(time.perf_counter()-t0))
                for k in keys:
                    vals[k].append(10*np.log10(C.sinr_linear(ws[k], trial['channel'][n], trial['R_ipn'][n], ps)))
            for k in keys:
                per[k].append(float(np.mean(vals[k])))
        out['points'][str(mm)] = {k: summarize(per[k], f'fig11|bootstrap|{k.replace("msocp", "socp")}') for k in keys}
        say(f'mismatch {mm} deg:', {k: round(out['points'][str(mm)][k]['mean_db'], 2) for k in keys})
    score = {g: np.mean([out['points'][str(mm)][f'lcmv_{g:g}']['mean_db'] for mm in (0, 1, 2)]) for g in loads}
    out['lcmv_best_load'] = max(score, key=score.get)
    out['time_ms'] = {'wrab_per_snapshot': float(np.mean(tw)), 'socp_per_solve': float(np.mean(ts)),
                      'lcmv_per_solve': float(np.mean(tl))}
    say('LCMV loading selected:', out['lcmv_best_load'], 'times (ms):', out['time_ms'])
    return out


# ------------------------------------------------------------------------------------------------ Table 8.2 and margins
def stage_near(levels=(0.3, 1.0, 2.0, 3.0), loads=(0.0, 1.0, 3.0, 10.0, 30.0, 100.0)):
    """Table 8.2: one interferer 2-8 degrees from the source (Figure 8.6 setting and trials): adaptive margins, fixed
    margin levels, and the multi-point LCMV with its loading chosen per row. Also the final margins of every run."""
    out = {'levels': levels, 'loads': loads, 'points': {}}
    NR = Q(18, 3)
    for sep in (2, 3, 4, 5, 8):
        base = Scenario(M=64, Nsc=8, interferers_deg=(sep, -55), jitter_deg=.1)
        row = {}
        r = run_trials(base, NR, 'fig6', ('wrab',)); s = slim(r)
        row['adaptive'] = dict(r['summary']['wrab'], saturation=float(np.mean(s['saturation'])))
        fm = np.array(s['final_margins']).reshape(-1, len(C.design_grid(base)[2]))
        row['margins'] = {'grid_deg': np.rad2deg(C.design_grid(base)[1]).tolist(),
                          'smallest_median': float(np.median(fm.min(axis=1))),
                          'smallest_max': float(fm.min(axis=1).max()),
                          'below_ceiling_counts': np.bincount((fm < 3-1e-9).sum(axis=1), minlength=fm.shape[1]).tolist(),
                          'argmin_counts': np.bincount(fm.argmin(axis=1), minlength=fm.shape[1]).tolist(),
                          'runs': int(fm.shape[0])}
        row['max_residual'] = s['max_residual']
        for nu in levels:
            rf = run_trials(replace(base, initial_margins=(nu,)), NR, 'fig6', ('fixed',))
            row[f'fixed_{nu:g}'] = rf['summary']['fixed']
        ps = 10**(base.snr_db/10); grid = C.design_grid(base)[1]; per = {g: [] for g in loads}
        for tr in range(NR):
            trial = C.generate_trial(base, seed_for('fig6', tr)); vals = {g: [] for g in loads}
            for n, f in enumerate(trial['frequencies']):
                Rs, mass = C.ewma_covariance(trial['X'][n], base.eta)
                Rb = C.normalized_covariance(Rs, mass, base.covariance_loading)
                Lc = C.centred_steering(base.M, grid, f, base.fc)
                for g in loads:
                    w = C.lcmv_weights(Rb, Lc, g)
                    vals[g].append(10*np.log10(C.sinr_linear(w, trial['channel'][n], trial['R_ipn'][n], ps)))
            for g in loads:
                per[g].append(float(np.mean(vals[g])))
        allg = {g: summarize(per[g], f'fig6|bootstrap|lcmv{g:g}') for g in loads}
        best = max(allg, key=lambda g: allg[g]['mean_db'])
        row['lcmv'] = dict(allg[best], load=best)
        row['lcmv_all_loads'] = {f'{g:g}': allg[g]['mean_db'] for g in loads}
        out['points'][str(sep)] = row
        say(f'separation {sep} deg:', {k: round(v['mean_db'], 2) for k, v in row.items()
                                       if isinstance(v, dict) and 'mean_db' in v}, 'LCMV load', best)
    return out


def stage_controls():
    """Paired controls of Sections 8.5.3, 8.5.8 and 8.5.9: margin level at 64 elements (adaptive, fixed 0.3, fixed 3),
    signed mismatch with the asymmetric grid, and all methods with the desired signal present in training; plus the
    adaptive rule against fixed levels with the desired signal present."""
    NR = Q(8, 3); out = {'margins': {}, 'signed': {}, 'mixed': {}, 'mixed_levels': {}}
    for mm in (0, 2, 4):
        for name, initial, adapt in (('adaptive', .3, True), ('fixed_0.3', .3, False), ('fixed_3', 3., False)):
            cfg = Scenario(Nsc=3, K=300, mismatch_deg=mm, jitter_deg=0, grid_fracs=(0, -.6, .6, -1.2, 1.2),
                           initial_margins=(initial,), interferers_deg=(-25, 40), self_anchor=adapt,
                           single_radius_fraction=.85)
            r = run_trials(cfg, NR, 'audit_margins', ('wrab',)); point(f'margin control {name}, {mm} deg', r)
            out['margins'][f'{name}|{mm}'] = slim(r)
    for mm in (-2, 2):
        r = run_trials(Scenario(Nsc=3, K=300, mismatch_deg=mm, jitter_deg=.1), NR, 'audit_signed')
        point(f'signed {mm} deg', r); out['signed'][str(mm)] = slim(r)
    for mm in (0, 2, 4):
        base = Scenario(Nsc=3, K=300, mismatch_deg=mm, jitter_deg=0, signal_free=False,
                        grid_fracs=(0, -.6, .6, -1.2, 1.2), initial_margins=(.3,), interferers_deg=(-25, 40),
                        single_radius_fraction=.85)
        r = run_trials(base, NR, 'audit_mixed', C.ALL_METHODS); point(f'signal present {mm} deg', r)
        out['mixed'][str(mm)] = slim(r)
        lv = {}
        for nu in (0.3, 1.0, 2.0, 3.0):
            lv[f'fixed_{nu:g}'] = run_trials(replace(base, initial_margins=(nu,)), NR, 'audit_mixed',
                                             ('fixed',))['summary']['fixed']
        out['mixed_levels'][str(mm)] = lv
        say(f'signal present {mm} deg, fixed levels:', {k: round(v['mean_db'], 2) for k, v in lv.items()})
    return out


def stage_table3():
    """Table 8.3: scenarios S1-S4 at 64 elements with the asymmetric grid of Table 8.1."""
    out = {}
    for key, mm, jitter, paths, spread in (('S1', 2, .1, 1, 0), ('S2', 4, .2, 1, 0), ('S3', 6, .3, 1, 0),
                                           ('S4', 2, 0, 3, .2)):
        cfg = Scenario(mismatch_deg=mm, jitter_deg=jitter, n_paths=paths, angular_spread_deg=spread)
        r = run_trials(cfg, Q(20, 3), 'table2_' + key); point(key, r); out[key] = slim(r)
    return out


# ------------------------------------------------------------------------------------------------ Section 8.3 checks
def stage_feasibility():
    """Section 8.3.1: constraint residuals of the frozen-norm one-step correction and of the exact projection (8.14) on
    the same 64-element snapshots; Section 8.3.4: the online objective after 600 snapshots against the frozen final
    problem (Figure 8.11 setting, 2 degrees, three realizations)."""
    out = {}
    M, K, fc = 64, 300, 300e9
    thp = np.deg2rad(2); off = np.array([0, -.6, .6, -1.2])*1.772/M
    X = C.sequential_snapshots(M, K, fc, (45, -60), 10, seed=42)
    res, nus = C.frozen_norm_trace(X, fc, fc, thp, thp+off, [1.9, 1.5, 1.5, 1.1])
    t = C.train_subband(X, fc, Scenario(Nsc=1, grid_offsets_deg=tuple(np.rad2deg(off)), jitter_deg=0), ('wrab',))
    out['frozen_norm'] = {'violating_updates': int(np.count_nonzero(res > 1e-8)), 'updates': K,
                          'max_violation': float(res.max()), 'tail100_max_violation': float(res[-100:].max()),
                          'final_margins': nus[-1].tolist(),
                          'exact_projection_max_residual': float(t['residuals']['wrab'].max()),
                          'residual_trace': res.tolist(), 'exact_residual_trace': t['residuals']['wrab'].tolist()}
    say('frozen-norm correction:', {k: v for k, v in out['frozen_norm'].items() if 'trace' not in k})
    cfg = fig11_config(2); rows = []
    for tr in range(Q(3, 1)):
        trial = C.generate_trial(cfg, seed_for('fig11', tr)); n = cfg.Nsc//2
        t = C.train_subband(trial['X'][n], trial['frequencies'][n], cfg, ('wrab',))
        R = t['final_R']; nu = t['margin_history'][-1]; L = t['L']; w0 = t['histories']['wrab'][-1]
        obj = lambda v: float(np.real(np.vdot(v, R @ v)))      # noqa: E731
        w = w0.copy()
        for _ in range(Q(8000, 2000)):
            w, _ = C.projected_step(R, w, L, nu, cfg)
        wr, _ = C.reference_constrained_minimum(R/np.trace(R).real, L, nu, initial=w)
        rows.append({'trial': tr, 'online_objective': obj(w0), 'frozen_objective': obj(w), 'slsqp_objective': obj(wr),
                     'gap_percent': 100*(obj(w0)/obj(w)-1), 'frozen_vs_slsqp': obj(w)/obj(wr)-1,
                     'online_residual': float(C.wc_residual(w0, L, nu).max()),
                     'frozen_residual': float(C.wc_residual(w, L, nu).max()), 'margins': nu.tolist()})
        say('frozen problem:', {k: v for k, v in rows[-1].items() if k != 'margins'})
    out['frozen_gap'] = rows
    return out


def stage_timing():
    """Run alone. Time per subcarrier and snapshot in the Figure 8.7(a) loop (three methods), one WRAB update and one
    interior-point solve of (8.8) in the Figure 8.11 setting, and one WRAB update with the 27-constraint fan at 512
    elements. Timings depend on the machine."""
    out = {'scaling_ms_per_frequency_snapshot': {}}
    for M in (64, 128, 256, 512):
        bw = np.rad2deg(1.772/M)
        cfg = Scenario(M=M, Nsc=5, mismatch_deg=1.5*bw, jitter_deg=.3*bw)
        secs = [C.evaluate_trial(cfg, seed_for('fig7', tr))['seconds'] for tr in range(Q(2, 1))]
        out['scaling_ms_per_frequency_snapshot'][str(M)] = 1e3*float(np.mean(secs))/(cfg.Nsc*cfg.K)
    say('scaling:', out['scaling_ms_per_frequency_snapshot'])
    cfg = fig11_config(2); tw, ts = [], []
    for tr in range(Q(3, 1)):
        trial = C.generate_trial(cfg, seed_for('fig11', tr))
        for n, f in enumerate(trial['frequencies']):
            t = C.train_subband(trial['X'][n], f, cfg, ('wrab',)); tw.append(1e3*t['seconds']/cfg.K)
            Rb = C.normalized_covariance(t['final_R'], t['covariance_mass'], cfg.covariance_loading)
            _, dt = C.multiball_socp(Rb, t['L'], t['margin_history'][-1]); ts.append(1e3*dt)
    out['wrab_update_ms'] = float(np.mean(tw)); out['socp_solve_ms'] = float(np.mean(ts))
    M = 512; bw = np.rad2deg(1.772/(M*np.cos(np.deg2rad(1.0)))); n = int(np.ceil(1.5/(0.6*bw)))
    cfg = Scenario(M=M, Nsc=5, mismatch_deg=1.0, jitter_deg=0.1, initial_margins=(1.5,),
                   grid_offsets_deg=tuple(float(x) for x in np.arange(-n, n+1)*0.6*bw))
    trial = C.generate_trial(cfg, seed_for('fig7f', 0)); tf = []
    for k in range(Q(5, 2)):
        tf.append(1e3*C.train_subband(trial['X'][k], trial['frequencies'][k], cfg, ('wrab',))['seconds']/cfg.K)
    out['wrab_update_ms_27_constraints_N512'] = float(np.mean(tf))
    say('timing:', {k: v for k, v in out.items() if k != 'scaling_ms_per_frequency_snapshot'})
    return out


# ------------------------------------------------------------------------------------------------ runner
GROUPS = ['fig11', 'fig11ref', 'near', 'fixedangle', 'controls', 'fig5', 'dynamic', 'fig8', 'fig4', 'fig6', 'fig7',
          'table3', 'fig9', 'fig3', 'fig10', 'feasibility', 'analytic']
TIMING = 'timing'
ALL_STAGES = GROUPS + [TIMING]

# figure/table key -> stages it needs
NEEDS = {
    '8_1': ['analytic'], '8_2': ['analytic'], '8_3': ['fig3'], '8_4': ['fig4'], '8_5': ['fig5'], '8_6': ['fig6'],
    '8_7': ['fig7', 'fixedangle'], '8_8': ['fig8', 'dynamic'], '8_9': ['fig9'], '8_10': ['fig10'],
    '8_11': ['fig11', 'fig11ref'], 'tables': ['near', 'table3'],
    'text': ['controls', 'feasibility', 'timing'],
}


def groups_for(keys):
    out = []
    for k in keys:
        for s in NEEDS[k]:
            if s not in out:
                out.append(s)
    return [s for s in ALL_STAGES if s in out]


def _tolist(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, dict):
        return {str(k): _tolist(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_tolist(v) for v in o]
    return o


def run_stage(stage, datadir):
    t0 = time.time()
    result = globals()['stage_' + stage]()
    os.makedirs(datadir, exist_ok=True)
    with open(os.path.join(datadir, stage + '.json'), 'w') as f:
        json.dump({'stage': stage, 'quick': QUICK, 'seconds': time.time()-t0, 'result': _tolist(result)}, f)
    say(f'stage {stage} finished in {time.time()-t0:.0f} s')


def load(datadir, stage):
    with open(os.path.join(datadir, stage + '.json')) as f:
        return json.load(f)['result']


def run_groups(datadir, groups=None, jobs=2, quick=False):
    """Run the stages, each in its own process, at most `jobs` at a time; the timing stage runs alone at the end."""
    groups = list(GROUPS + [TIMING] if groups is None else groups)
    os.makedirs(datadir, exist_ok=True)
    env = dict(os.environ, CH8_QUICK='1' if quick else '0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')

    def launch(s):
        log = open(os.path.join(datadir, s + '.log'), 'w')
        return subprocess.Popen([sys.executable, '-m', 'ch8wrab.scenarios', s, datadir], stdout=log,
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
    for s in timed:
        p, log = launch(s); p.wait(); log.close()
        if p.returncode != 0:
            raise RuntimeError(f'stage {s} failed; see {os.path.join(datadir, s + ".log")}')
        print(f'[{time.time() - t0:6.0f} s] done  {s}', flush=True)


if __name__ == '__main__':
    run_stage(sys.argv[1], sys.argv[2])
