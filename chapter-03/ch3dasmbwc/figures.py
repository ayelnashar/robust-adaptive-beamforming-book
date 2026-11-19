"""
ch3dasmbwc.figures -- every simulation figure (Figures 3.1-3.14) and table (Tables 3.2-3.6) of
Chapter 3, plus the numbers quoted in the text of Sections 3.4-3.7 and Appendix 3.A.

Configured through environment variables read at import time:
  QUICK=1       smoke test (two Monte Carlo trials per point, short long runs)
  FIGDIR=path   output folder (default ./figures)
  RAB_CACHE=1   reuse the simulation data saved in FIGDIR/data (re-plot only, no re-simulation)
  RAB_INLINE=1  show each figure under its cell in Jupyter
Used by the notebook and by scripts/generate_figures.py.
"""
import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import scenarios as S
from .algorithms import steer, beampattern_db, desired_cov, sinr_opt_db

QUICK = os.environ.get('QUICK') == '1'
OUT = os.environ.get('FIGDIR', 'figures')
USE_CACHE = os.environ.get('RAB_CACHE') == '1'
INLINE = os.environ.get('RAB_INLINE') == '1'

# Monte Carlo counts of the chapter (QUICK: smoke test)
NR = dict(snap=25, sweep=25, sens=15, errors=20, size=15, stab=25, long=5, long_K=3000, abl=25, mc=200,
          t3=25, t5=25, signed=25, time_rep=5, t4_rep=10)
if QUICK:
    NR = dict(snap=2, sweep=2, sens=2, errors=2, size=2, stab=2, long=1, long_K=600, abl=2, mc=3,
              t3=2, t5=2, signed=2, time_rep=1, t4_rep=2)

RESULTS = {}
FRESH = set()            # data computed during this run (safe to read back from FIGDIR/data)
_MEM = {}

# --------------------------------------------------------------------------- visual identity (chapter figures)
COLOR = {'Standard MVDR': '#888888', 'DL-MVDR': '#1f77b4', 'WC-MVDR [6]': '#ff7f0e',
         'Adaptive single-WC [12]': '#2ca02c', 'DAS-MBWC': '#d62728', 'INC reconstruction [22]': '#9467bd',
         'Optimum': '#17becf'}
MARKER = dict(zip(S.METHODS, ['d', 'o', '^', '*', 'P', 's', 'v']))
LINE = dict(zip(S.METHODS, ['--', '-.', '-.', '-', '-', ':', '--']))

plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.linewidth': 0.9,
    'axes.spines.top': True, 'axes.spines.right': True,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
})
FIGSIZE = (7.2, 4.8)
LEG = dict(frameon=True, fancybox=False, framealpha=0.95, edgecolor='#444444', fontsize=8,
           borderpad=0.5, labelspacing=0.35, columnspacing=1.2, handlelength=2.6)

# --------------------------------------------------------------------------- simulation data
SNAP_SCEN = {  # figure label -> (Table 3.2 scenario, title, file name)
    '3_1': ('2° point source', 'Output SINR versus Snapshots (2° Mismatch, M=16)', 'Fig3_01_SINR_2deg_point_source'),
    '3_3': ('5° point source', 'Output SINR versus Snapshots (5° Mismatch, M=16)', 'Fig3_03_SINR_5deg_point_source'),
    '3_4': ('2° sm. spread', 'Output SINR versus Snapshots (2° Mismatch, Small Spread)', 'Fig3_04_SINR_2deg_small_spread'),
    '3_5': ('5° sm. spread', 'Output SINR versus Snapshots (5° Mismatch, Small Spread)', 'Fig3_05_SINR_5deg_small_spread'),
    '3_6': ('7° point source', 'Output SINR versus Snapshots (7° Mismatch, M=16)', 'Fig3_06_SINR_7deg_point_source'),
    '3_7': ('5° lg. spread', 'Output SINR versus Snapshots (5° Mismatch, Large Spread)', 'Fig3_07_SINR_5deg_large_spread'),
    '3_8': ('7° lg. spread', 'Output SINR versus Snapshots (7° Mismatch, Large Spread)', 'Fig3_08_SINR_7deg_large_spread'),
}


def _task(key):
    """(function, args, kwargs) of one simulation task."""
    if key.startswith('snap:'):
        return S.snapshot_scenario, (key[5:],), dict(NR=NR['snap'])
    table = {
        'beampattern': (S.beampattern_weights, (), {}),
        'sweep_point': (S.mismatch_sweep, (None, S.SEED_BASE + 2000), dict(NR=NR['sweep'])),
        'sweep_small': (S.mismatch_sweep, (S.MP_SMALL, S.SEED_BASE + 3000), dict(NR=NR['sweep'])),
        'array_errors': (S.array_errors, (), dict(NR=NR['errors'])),
        'array_size': (S.array_size, (), dict(NR=NR['size'])),
        'sensitivity': (S.sensitivity, (), dict(NR=NR['sens'])),
        'stability': (S.stability, (), dict(NR=NR['stab'])),
        'long_runs': (S.long_runs, (), dict(NR=NR['long'], K=NR['long_K'])),
        'feasibility': (S.feasibility_limits, (), {}),
        'subsets': (S.subset_limits, (), {}),
        'bound_ablation': (S.bound_ablation, (), dict(NR=NR['abl'])),
        'correction_ratio': (S.correction_ratio, (), dict(NR=NR['abl'])),
        'feasibility_check': (S.feasibility_check, (), dict(NR=NR['abl'])),
        'mc_check': (S.mc_check, (), dict(NR=NR['mc'])),
        'table3': (S.table3_socp, (), dict(NR=NR['t3'])),
        'table5': (S.table5_placement, (), dict(NR=NR['t5'])),
        'signed': (S.signed_mismatch, (), dict(NR=NR['signed'])),
        'time_recursion': (S.computation_time, (), dict(NREP=NR['time_rep'])),
        'table4': (S.table4_timing, (), dict(REP=NR['t4_rep'])),
    }
    return table[key]


SIM_TASKS = ['snap:' + s[0] for s in S.SCENARIOS] + [
    'mc_check', 'long_runs', 'sweep_point', 'sweep_small', 'table3', 'stability', 'array_size', 'signed',
    'array_errors', 'sensitivity', 'bound_ablation', 'correction_ratio', 'feasibility_check', 'table5',
    'beampattern', 'feasibility', 'subsets']
TIMING_TASKS = ['time_recursion', 'table4']        # run alone, so that nothing competes for the CPU


def compute(key):
    """Run one simulation task and save its result in FIGDIR/data/<key>.pkl."""
    f, args, kw = _task(key)
    res = f(*args, **kw)
    path = _path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(res, fh)
    _MEM[key] = res
    FRESH.add(key)
    return key


def _path(key):
    return os.path.join(OUT, 'data', key.replace(':', '_').replace('°', 'deg').replace(' ', '_').replace('.', '')
                        + '.pkl')


def data(key):
    if key in _MEM:
        return _MEM[key]
    path = _path(key)
    if (USE_CACHE or key in FRESH) and os.path.exists(path):
        with open(path, 'rb') as fh:
            _MEM[key] = pickle.load(fh)
        return _MEM[key]
    compute(key)
    return _MEM[key]


# --------------------------------------------------------------------------- plotting helpers
def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    if INLINE:
        import io
        from IPython.display import display, Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110)
        display(Image(data=buf.getvalue()))
    plt.close(fig)


def legend_below(ax, lo, hi, ncol=4, room=0.45):
    """Framed legend inside the axes, in a band added below the lowest curve."""
    span = hi - lo
    ylim = (lo - room * span, hi + 0.06 * span)
    ax.set_ylim(*ylim)
    ax.set_yticks([t for t in ax.get_yticks() if lo - 0.04 * span <= t <= ylim[1]])
    ax.set_ylim(*ylim)
    ax.legend(loc='lower center', ncol=ncol, **LEG)


def legend_above(ax, lo, hi, ncol=4, room=0.40, loc='upper center'):
    """Framed legend inside the axes, in a band added above the highest curve or bar."""
    span = hi - lo
    ylim = (lo, hi + room * span)
    ax.set_ylim(*ylim)
    ax.set_yticks([t for t in ax.get_yticks() if ylim[0] <= t <= hi + 0.04 * span])
    ax.set_ylim(*ylim)
    ax.legend(loc=loc, ncol=ncol, **LEG)


def _r(x, n=2):
    return float(np.round(x, n))


# --------------------------------------------------------------------------- Figures 3.1, 3.3-3.8
def snapshot_figure(fig_id):
    label, title, name = SNAP_SCEN[fig_id]
    Smean, TR = data('snap:' + label)
    K = Smean.shape[1]
    st = max(1, K // 14)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, m in enumerate(S.METHODS):
        ax.plot(np.arange(K)[::st], Smean[i, ::st], label=m, marker=MARKER[m], linestyle=LINE[m], color=COLOR[m],
                markersize=5.5, linewidth=1.3)
    ax.set_xlabel('Snapshots')
    ax.set_ylabel('SINR (dB)')
    ax.set_title(title)
    ax.set_xlim([0, K])
    legend_below(ax, float(Smean[:, ::st].min()), float(Smean[:, ::st].max()))
    save(fig, name)
    RESULTS['Fig' + fig_id.replace('_', '.')] = {
        'scenario': label, 'final_snapshot_sinr_dB': {m: _r(Smean[i, -1]) for i, m in enumerate(S.METHODS)}}


def fig_3_1():
    snapshot_figure('3_1')


def fig_3_3():
    snapshot_figure('3_3')


def fig_3_4():
    snapshot_figure('3_4')


def fig_3_5():
    snapshot_figure('3_5')


def fig_3_6():
    snapshot_figure('3_6')


def fig_3_7():
    snapshot_figure('3_7')


def fig_3_8():
    snapshot_figure('3_8')


# --------------------------------------------------------------------------- Figure 3.2
def fig_3_2():
    W = data('beampattern')
    tscan = np.linspace(-np.pi / 2, np.pi / 2, 600)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axvline(0, color='green', linestyle=':', alpha=0.6, linewidth=1)
    ax.axvline(5, color='red', linestyle=':', alpha=0.6, linewidth=1)
    nulls = {}
    for m, w in W.items():
        bp = beampattern_db(w, S.M, tscan)
        ax.plot(np.rad2deg(tscan), bp, label=m, linestyle=LINE[m], color=COLOR[m], linewidth=1.3)
        # null depth relative to the pattern maximum over the scan
        peak = np.max([20 * np.log10(max(abs(w.conj() @ steer(S.M, t)), 1e-15)) for t in tscan])
        nulls[m] = {f'{t}°': _r(20 * np.log10(max(abs(w.conj() @ steer(S.M, np.deg2rad(t))), 1e-15)) - peak, 1)
                    for t in (45, -60)}
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized Beam Pattern (dB)')
    ax.set_title('Steady-State Beam Patterns (5° Mismatch, M=16)')
    ax.set_xlim([-90, 90])
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_ylim([-60, 32])
    ax.set_yticks(np.arange(-60, 1, 10))
    ax.legend(loc='upper center', ncol=3, **LEG)
    save(fig, 'Fig3_02_beampattern_5deg')
    RESULTS['Fig3.2'] = {'null_depth_dB_at_interferers': nulls}


# --------------------------------------------------------------------------- Figures 3.9, 3.10
def sweep_figure(key, title, name, fig_label):
    sinr, abl = data(key)
    mm = np.arange(sinr.shape[1])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, m in enumerate(S.METHODS):
        ax.plot(mm, sinr[i], label=m, marker=MARKER[m], linestyle=LINE[m], color=COLOR[m], markersize=5.5,
                linewidth=1.3)
    ax.set_xlabel('Mismatch Angle (degrees)')
    ax.set_ylabel('SINR (dB)')
    ax.set_title(title)
    legend_below(ax, float(sinr.min()), float(sinr.max()))
    save(fig, name)
    RESULTS[fig_label] = {'mismatch_deg': mm.tolist(),
                          'sinr_dB': {m: [_r(v) for v in sinr[i]] for i, m in enumerate(S.METHODS)},
                          'all_constraint_ablation_dB': [_r(v) for v in abl]}


def fig_3_9():
    sweep_figure('sweep_point', 'Output SINR versus Mismatch (Point Source, M=16)', 'Fig3_09_SINR_vs_mismatch_point_source',
                 'Fig3.9')


def fig_3_10():
    sweep_figure('sweep_small', 'Output SINR versus Mismatch (Small Spread, M=16)', 'Fig3_10_SINR_vs_mismatch_small_spread',
                 'Fig3.10')


# --------------------------------------------------------------------------- Figure 3.11
def fig_3_11():
    res = data('array_errors')
    cases = list(res.keys())
    vals = np.array([res[c] for c in cases])          # columns: DAS-MBWC, INC, WC-MVDR, single-WC
    names = ['DAS-MBWC', 'INC reconstruction [22]', 'WC-MVDR [6]', 'Adaptive single-WC [12]']
    x = np.arange(len(cases)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for j, (nm, off) in enumerate(zip(names, (-1.5, -0.5, 0.5, 1.5))):
        ax.bar(x + off * w, vals[:, j], w, label=nm, color=COLOR[nm])
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=20, fontsize=9)
    ax.set_ylabel('Output SINR (dB)')
    ax.set_title('Robustness to Array Imperfections (5° Mismatch, M=16)')
    ax.grid(alpha=0.3, axis='y')
    ax.grid(False, axis='x')
    legend_above(ax, min(0.0, float(vals.min())), float(vals.max()), ncol=2, room=0.32)
    save(fig, 'Fig3_11_array_imperfections')
    RESULTS['Fig3.11'] = {c: {nm: _r(v) for nm, v in zip(names, res[c])} for c in cases}


# --------------------------------------------------------------------------- Figure 3.12
def fig_3_12():
    res = data('array_size')
    M_list = (16, 64, 128)
    names = ['DAS-MBWC', 'INC reconstruction [22]', 'WC-MVDR [6]', 'Adaptive single-WC [12]', 'Optimum']
    x = np.arange(len(M_list)); w = 0.17
    allv = np.array([v for P in res.values() for v in P]).ravel()
    lo, hi = float(allv.min()), float(allv.max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (gname, series) in zip(axes, res.items()):
        for j, (nm, off) in enumerate(zip(names, (-2, -1, 0, 1, 2))):
            ax.bar(x + off * w, series[j], w, label=nm, color=COLOR[nm])
        ax.set_xticks(x)
        ax.set_xticklabels([f'$M$ = {m}\n({114.0 / m:.2f}$\\degree$ beamwidth)' for m in M_list])
        ax.set_title(gname, fontsize=10)
        ax.axhline(0, color='k', lw=0.8)
        ax.grid(alpha=0.3, axis='y')
        ax.grid(False, axis='x')
    axes[0].set_ylabel('Output SINR (dB)')
    legend_above(axes[0], lo - 0.05 * (hi - lo), hi, ncol=3, room=0.36)
    fig.suptitle('Array Size Scaling at 7$\\degree$ Mismatch', fontsize=11)
    fig.tight_layout()
    save(fig, 'Fig3_12_array_size_scaling')
    RESULTS['Fig3.12'] = {g.replace('\n', ' '): {nm: dict(zip(map(str, M_list), [_r(v) for v in s]))
                                                 for nm, s in zip(names, series)} for g, series in res.items()}


# --------------------------------------------------------------------------- Figure 3.13
def fig_3_13():
    res = data('sensitivity')
    rhos, v_rho = zip(*res['rho'])
    clips, v_clip = zip(*res['clip'])
    pers, v_per = zip(*res['period'])
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
    ax[0].plot(rhos, v_rho, marker='o', color='#d62728')
    ax[0].axvline(0.8, color='k', ls=':', lw=1)
    ax[0].set_xlabel('Smoothing weight')
    ax[0].set_ylabel('Output SINR (dB)')
    xc = np.arange(len(clips))
    ax[1].plot(xc, v_clip, marker='s', color='#1f77b4')
    ax[1].axvline(list(clips).index((0.5, 2.0)), color='k', ls=':', lw=1)
    ax[1].set_xticks(xc)
    ax[1].set_xticklabels([f'[{a},{b}]' for a, b in clips], rotation=30, fontsize=7)
    ax[1].set_xlabel('Clipping range')
    ax[2].plot(pers, v_per, marker='^', color='#2ca02c')
    ax[2].axvline(30, color='k', ls=':', lw=1)
    ax[2].set_xlabel('Adaptation period (snapshots)')
    allv = list(v_rho) + list(v_clip) + list(v_per)
    for a in ax:
        a.set_ylim([min(allv) - 1, max(allv) + 1])
        a.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, 'Fig3_13_bound_adaptation_sensitivity')
    RESULTS['Fig3.13'] = {'smoothing_weight': {str(r): _r(v) for r, v in res['rho']},
                          'clipping_range': {f'[{a}, {b}]': _r(v) for (a, b), v in res['clip']},
                          'adaptation_period': {str(p): _r(v) for p, v in res['period']}}


# --------------------------------------------------------------------------- Figure 3.14
def fig_3_14():
    rows = data('time_recursion')
    Ms = np.array([r[0] for r in rows], dtype=float)
    tp = np.array([r[1] for r in rows]); te = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax[0].loglog(Ms, tp, marker='*', color='#d62728', markersize=9, label='DAS-MBWC')
    ax[0].loglog(Ms, te, marker='^', color='#ff7f0e', ls='--', label='WC-MVDR [6] (exact solver)')
    ax[0].loglog(Ms, tp[0] * (Ms / Ms[0]) ** 2, color='#888888', ls=':', lw=1, label=r'$O(M^2)$')
    ax[0].loglog(Ms, te[0] * (Ms / Ms[0]) ** 3, color='#444444', ls=':', lw=1, label=r'$O(M^3)$')
    ax[0].set_xlabel('Number of array elements $M$')
    ax[0].set_ylabel('Time per snapshot ($\\mu$s)')
    ax[0].set_xticks(Ms)
    ax[0].set_xticklabels([str(int(m)) for m in Ms])
    ax[0].minorticks_off()
    ax[0].grid(alpha=0.3, which='both')
    ax[0].legend(loc='upper left', **LEG)
    ratio = te / tp
    ax[1].semilogx(Ms, ratio, marker='o', color='#2ca02c')
    ax[1].set_xlabel('Number of array elements $M$')
    ax[1].set_ylabel('Speed-up over EigDec')
    ax[1].set_xticks(Ms)
    ax[1].set_xticklabels([str(int(m)) for m in Ms])
    ax[1].minorticks_off()
    ax[1].grid(alpha=0.3, which='both')
    fig.tight_layout()
    save(fig, 'Fig3_14_computation_time')
    RESULTS['Fig3.14'] = {'environment': S.environment(),
                          'rows': [{'M': int(m), 'das_mbwc_us_per_snapshot': _r(a, 1), 'wc_mvdr_us_per_solve': _r(b, 1),
                                    'speedup': _r(b / a, 1)} for m, a, b in zip(Ms, tp, te)]}


# --------------------------------------------------------------------------- tables
def table_3_2():
    rows = {}
    for lab in S.TABLE_ORDER:
        Smean, TR = data('snap:' + lab)
        rows[lab] = {'Std MVDR': _r(TR[:, 0].mean(), 1), 'DL-MVDR': _r(TR[:, 1].mean(), 1),
                     'WC [6]': _r(TR[:, 2].mean(), 1), '1-WC [12]': _r(TR[:, 3].mean(), 1),
                     'INC [22]': _r(TR[:, 5].mean(), 1), 'All-constr.': _r(TR[:, 6].mean(), 1),
                     'DAS-MBWC': _r(TR[:, 4].mean(), 1), 'Optimum': _r(Smean[6, -1], 1)}
    RESULTS['Table3.2'] = rows


def table_3_3():
    res = data('table3')
    RESULTS['Table3.3'] = {lab: {'DAS-MBWC': _r(res[lab]['das_mbwc'], 1), 'SOCP': _r(res[lab]['socp'], 1),
                                 'difference': _r(res[lab]['diff'], 2), 'difference_ci95': _r(res[lab]['diff_ci'], 2),
                                 'solver_status': res[lab]['status'], 'max_constraint_residual': res[lab]['max_resid']}
                           for lab in S.TABLE_ORDER}


def table_3_4():
    res = data('table4')
    RESULTS['Table3.4'] = {'environment': res['env'],
                           'rows': {str(m): {'SCS_ms': _r(r['scs_solve_ms'], 2), 'SCS_wall_ms': _r(r['scs_wall_ms'], 2),
                                             'DAS_MBWC_ms': _r(r['gram_ms'], 4), 'speedup': _r(r['speedup'], 0),
                                             'corrections_timed': r['n_proj']}
                                    for m, r in res['rows'].items()}}


def table_3_5():
    res = data('table5')
    RESULTS['Table3.5'] = {lab: {'5°': _r(v[5], 1), '7°': _r(v[7], 1)} for lab, v in res.items()}


def table_3_6():
    st = data('stability')
    lr = data('long_runs')
    RESULTS['Table3.6'] = {lab: {'SINR s.d. (dB)': _r(st[lab]['sinr_sd']), 'Set change rate': _r(st[lab]['change_rate'], 3),
                                 'Two-cycle fraction': _r(st[lab]['two_cycle'], 3), 'Mean |A|': _r(st[lab]['mean_A']),
                                 'Resets without': _r(lr[lab]['resets_without'], 1), 'Resets with': _r(lr[lab]['resets_with'], 1),
                                 'SINR last 500 (dB)': _r(lr[lab]['sinr_last500'], 1),
                                 'max ||w|| with rescaling': lr[lab]['max_norm']} for lab in S.TABLE_ORDER}


def text_checks():
    """Numbers quoted in the text that do not come from a figure or table."""
    out = {}
    a = lambda t: steer(S.M, np.deg2rad(t))
    out['steering_distances'] = {f'0-{t} deg': _r(np.linalg.norm(a(0) - a(t)), 2) for t in (1, 2, 3, 5, 7)}
    out['steering_distance_7deg_to_constraint_at_3deg'] = _r(np.linalg.norm(a(0) - a(3)), 2)
    out['clairvoyant_optimum_dB'] = {k: _r(sinr_opt_db(desired_cov(S.M, 0.0, mp, S.PS), S.R_IPN), 1)
                                     for k, mp in (('point source', None), ('small spread', S.MP_SMALL),
                                                   ('large spread', S.MP_LARGE))}
    mc = data('mc_check')
    names = ['Std MVDR', 'DL-MVDR', 'WC [6]', '1-WC [12]', 'INC [22]', 'DAS-MBWC', 'All-constr.']
    out['Table3.2_with_%d_trials' % NR['mc']] = {lab: {n: [_r(m, 2), _r(c, 2)] for n, m, c in
                                                      zip(names, mc[lab]['mean'], mc[lab]['ci95'])} for lab in mc}
    fe = data('feasibility')
    out['slater_margin'] = {f'{md} deg': {'t*(nu0)': _r(v['t'][1.0], 3), 't*(1.832 nu0)': _r(v['t'][1.832], 3),
                                          't*(2 nu0)': _r(v['t'][2.0], 3), 'limit s': _r(v['limit'], 3)}
                            for md, v in fe.items()}
    sub = data('subsets')
    out['subset_limits'] = {f'{md} deg': {str(s): _r(l, 3) for s, l in v} for md, v in sub.items()}
    out['bound_ramp_s_n'] = {n: _r(2 - 0.8 ** n, 3) for n in range(1, 10)}
    out['bound_ablation'] = {lab: {k: _r(v, 1) for k, v in d.items()} for lab, d in data('bound_ablation').items()}
    out['correction_ratio'] = {lab: {k: _r(v, 3) for k, v in d.items()} for lab, d in data('correction_ratio').items()}
    out['feasibility_check'] = {lab: {k: _r(v, 3) for k, v in d.items()} for lab, d in data('feasibility_check').items()}
    sg = data('signed')
    out['signed_mismatch'] = {'angles_deg': sg['angles'], 'asymmetric_grid_dB': [_r(v, 1) for v in sg['asymmetric']],
                              'symmetric_grid_dB': [_r(v, 1) for v in sg['symmetric']]}
    RESULTS['text'] = out


PART_FIGURES = [fig_3_1, fig_3_2, fig_3_3, fig_3_4, fig_3_5, fig_3_6, fig_3_7, fig_3_8, fig_3_9, fig_3_10,
                fig_3_11, fig_3_12, fig_3_13, fig_3_14]
PART_TABLES = [table_3_2, table_3_3, table_3_4, table_3_5, table_3_6, text_checks]


def write_summary(path=None):
    path = path or os.path.join(OUT, 'results_summary.json')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    meta = {'monte_carlo': NR, 'quick': QUICK}
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'settings': meta, **RESULTS}, fh, indent=1, ensure_ascii=False, default=str)
    return path
