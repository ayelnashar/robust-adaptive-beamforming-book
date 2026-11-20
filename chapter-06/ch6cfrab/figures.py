"""
ch6cfrab.figures -- Figures 6.1 to 6.10, Tables 6.2 to 6.6 and the results summary of Chapter 6, drawn from the data
that the stages of ch6cfrab.scenarios write into FIGDIR/data.

Configured through environment variables read at import time:
  FIGDIR=path    output folder (default ./figures); the simulation data are read from FIGDIR/data
  RAB_INLINE=1   also show each figure under its cell in Jupyter
"""
import csv
import json
import os

import numpy as np
from scipy.stats import t as tdist
import matplotlib
if os.environ.get('RAB_INLINE') != '1':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.environ.get('FIGDIR', 'figures')
DATA = os.path.join(OUT, 'data')
INLINE = os.environ.get('RAB_INLINE') == '1'
RESULTS = {}

plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True,
    'axes.linewidth': 0.9, 'axes.spines.top': True, 'axes.spines.right': True, 'axes.labelsize': 10.5,
    'lines.linewidth': 1.6, 'lines.markersize': 5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
})
LEG = dict(frameon=True, fancybox=False, framealpha=0.95, edgecolor='#444444', fontsize=7.4,
           borderpad=0.45, labelspacing=0.3, columnspacing=1.0, handlelength=2.6)

# method key of the simulator -> line style and the legend name used in the chapter
STYLE = {
    'D-CF-RAB': dict(linestyle='-', marker='o', color='C0', linewidth=2.2, markersize=5, zorder=8, label='D-CF-RAB'),
    'MR+W': dict(linestyle=(0, (3, 2)), marker='h', color='C8', linewidth=1.3, markersize=6, markerfacecolor='none',
                 markeredgewidth=1.2, zorder=10, label='MR + CPU combiner'),
    'LP-MMSE+W': dict(linestyle='-', marker='p', color='C9', linewidth=1.6, markersize=5, zorder=6,
                      label='LP-MMSE + CPU combiner'),
    'LMVDR+W': dict(linestyle='--', marker='^', color='C2', linewidth=1.5, markersize=5, zorder=5,
                    label='Local MVDR + CPU combiner'),
    'P-MMSE': dict(linestyle='-', marker='D', color='C4', linewidth=1.8, markersize=5, zorder=5, label='P-MMSE [6]'),
    'LP-MMSE': dict(linestyle='--', marker='P', color='C5', linewidth=1.5, markersize=5, zorder=3, label='LP-MMSE [6]'),
    'CF-RAB': dict(linestyle='-', marker='s', color='C3', linewidth=1.8, markersize=5, zorder=4, label='CF-RAB (centralized)'),
    'centralized_sdp': dict(linestyle=':', color='k', linewidth=1.5, zorder=9, label='Primary-ball SOCP'),
    'RECON-RAB': dict(linestyle=':', marker='X', color='C6', linewidth=1.5, markersize=5, zorder=3,
                      label='INCM reconstruction'),
    'MR': dict(linestyle=':', marker='*', color='C7', linewidth=1.4, markersize=7, zorder=1, label='MR [6]'),
}
ORDER = ('D-CF-RAB', 'MR+W', 'LP-MMSE+W', 'LMVDR+W', 'P-MMSE', 'LP-MMSE', 'CF-RAB', 'centralized_sdp', 'RECON-RAB', 'MR')
NAME = {k: v['label'] for k, v in STYLE.items()}


def J(stage):
    with open(os.path.join(DATA, stage + '.json')) as f:
        return json.load(f)['result']


def have(stage):
    return os.path.exists(os.path.join(DATA, stage + '.json'))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    if INLINE:
        plt.show()
    plt.close(fig)


def headroom(ax, frac, log=False):
    """Raise the top of the y-axis so that a legend at the top of the axes stays clear of the curves."""
    lo, hi = ax.get_ylim()
    if log:
        ax.set_ylim(lo, 10 ** (np.log10(hi) + frac * (np.log10(hi) - np.log10(lo))))
    else:
        ax.set_ylim(lo, hi + frac * (hi - lo))


def legend_inside(ax, loc='upper center', ncol=2, handles=None, **kw):
    h, l = handles if handles else ax.get_legend_handles_labels()
    return ax.legend(h, l, loc=loc, ncol=ncol, **{**LEG, **kw})


def draw(ax, xs, data, logx=False, idx=0):
    plot = ax.semilogx if logx else ax.plot
    for m in ORDER:
        if m in data[0]:
            plot(xs, [d[m][idx] for d in data], **STYLE[m])


def r3(x):
    return round(float(x), 3)


def per_point(d, keys):
    """{method name: [mean, 95% t half-width, worst-user]} per x value."""
    return {k: {NAME[m]: [r3(v[0]), r3(v[1]), r3(v[2])] for m, v in d[k].items()} for k in keys}


# ------------------------------------------------------------------------------------------------ figures
def fig_6_1():
    d = J('fig1'); xs = [4, 8, 16]; data = [d[str(x)] for x in xs]
    fig, ax = plt.subplots(figsize=(6.2, 4.6)); draw(ax, xs, data)
    ax.set_xlabel('Number of APs $L$'); ax.set_ylabel('Mean per-user SINR (dB)'); ax.set_xticks(xs)
    headroom(ax, 0.62); legend_inside(ax, ncol=2)
    save(fig, 'Fig6_01_SINR_vs_L')
    RESULTS['Figure 6.1'] = {'x': 'number of APs L', 'points': per_point(d, [str(x) for x in xs])}


EPS2 = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50]


def fig_6_2():
    d = J('fig2'); data = [d[str(e)] for e in EPS2]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.6))
    draw(a1, EPS2, data, logx=True); draw(a2, EPS2, data, logx=True, idx=2)
    for a, lab in ((a1, 'Mean per-user SINR (dB)'), (a2, 'Worst-user SINR (dB)')):
        a.set_xlabel(r'CSI uncertainty radius $\varepsilon$'); a.set_ylabel(lab); a.grid(True, which='both', alpha=0.3)
        headroom(a, 0.62)
    legend_inside(a1, ncol=2); legend_inside(a2, ncol=2)
    fig.tight_layout(); save(fig, 'Fig6_02_SINR_vs_eps')
    RESULTS['Figure 6.2'] = {'x': 'CSI uncertainty radius eps', 'points': per_point(d, [str(e) for e in EPS2])}


def fig_6_3():
    d = J('fig3'); xs = [-5, 0, 5, 10, 15, 20]; data = [d[str(x)] for x in xs]
    fig, ax = plt.subplots(figsize=(6.2, 4.6)); draw(ax, xs, data)
    ax.set_xlabel('Per-antenna SNR (dB)'); ax.set_ylabel('Mean per-user SINR (dB)')
    headroom(ax, 0.55); legend_inside(ax, ncol=2)
    save(fig, 'Fig6_03_SINR_vs_SNR')
    RESULTS['Figure 6.3'] = {'x': 'per-antenna SNR (dB)', 'points': per_point(d, [str(x) for x in xs])}


def fig_6_4():
    d = J('fig4'); xs = [0, 1, 4, 16, 64]; data = [d[str(x)] for x in xs]
    fig, ax = plt.subplots(figsize=(6.2, 4.6)); draw(ax, xs, data)
    ax.set_xscale('symlog', linthresh=1); ax.set_xticks(xs); ax.set_xticklabels([str(t) for t in xs])
    ax.set_xlabel(r'CSI delay $\tau$ (snapshots)'); ax.set_ylabel('Mean per-user SINR (dB)')
    ax.set_ylim(bottom=min(min(v[m][0] for m in v) for v in data) - 1.0)
    headroom(ax, 0.55); legend_inside(ax, loc='upper right', ncol=2)
    save(fig, 'Fig6_04_SINR_vs_delay')
    RESULTS['Figure 6.4'] = {'x': 'CSI delay tau (snapshots)', 'points': per_point(d, [str(x) for x in xs])}


def fig_6_5():
    d = J('fig5'); data = [d['uniform'], d['tiered']]
    present = [m for m in ORDER if m in data[0] and m != 'centralized_sdp']
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = np.arange(2); w = 0.8 / len(present)
    for i, m in enumerate(present):
        ax.bar(x + (i - len(present) / 2) * w, [dd[m][0] for dd in data], w, color=STYLE[m]['color'],
               edgecolor='k', linewidth=0.4, label=NAME[m], zorder=3)
    ax.plot(x - w / 2, [dd['centralized_sdp'][0] for dd in data], 'o', markersize=8, markerfacecolor='none',
            markeredgecolor='k', markeredgewidth=1.3, linestyle='none', label='Primary-ball SOCP', zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([r'Uniform $\varepsilon$', r'Tiered $\varepsilon$'])
    ax.set_ylabel('Mean per-user SINR (dB)'); ax.grid(False, axis='x')
    ax.set_ylim(0, max(max(dd[m][0] for m in dd) for dd in data) * 1.62)
    h, l = ax.get_legend_handles_labels()                     # method order of the chapter, SOCP after CF-RAB
    order = [NAME[m] for m in ORDER if NAME[m] in l]
    legend_inside(ax, ncol=2, handles=([h[l.index(n)] for n in order], order))
    save(fig, 'Fig6_05_heterogeneity')
    RESULTS['Figure 6.5'] = {'x': 'per-AP CSI uncertainty profile', 'points': per_point(d, ['uniform', 'tiered'])}


def fig_6_6():
    rows = J('fig6')['rows']; xs = [r['L'] for r in rows]
    MT = ('D-CF-RAB', 'CF-RAB', 'centralized_sdp', 'P-MMSE', 'MR+W')
    fig, (at, ab) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for m in MT:
        at.plot(xs, [r['t'][m] for r in rows], **STYLE[m])
    for m in ('D-CF-RAB', 'CF-RAB', 'P-MMSE'):
        ab.plot(xs, [r['fh'][m] for r in rows], **STYLE[m])
    at.set_yscale('log'); at.grid(True, which='both', alpha=0.3)
    at.set_xlabel('Number of APs $L$'); at.set_ylabel('Time per weight update (ms)'); at.set_xticks(xs)
    ab.set_xlabel('Number of APs $L$'); ab.set_ylabel('Listed bytes per user update'); ab.set_xticks(xs)
    headroom(at, 0.45, log=True); legend_inside(at, loc='upper left', ncol=2)
    ab.set_ylim(0, max(r['fh']['CF-RAB'] for r in rows) * 1.3); legend_inside(ab, loc='upper left', ncol=1)
    fig.tight_layout(); save(fig, 'Fig6_06_complexity')
    RESULTS['Figure 6.6'] = {'time per weight update (ms, this machine)': {str(r['L']): {NAME[m]: round(r['t'][m], 4) for m in MT}
                                                                         for r in rows},
                             'weight updates timed': {str(r['L']): {NAME[m]: r['calls'][m] for m in MT} for r in rows},
                             'listed bytes per user update': {str(r['L']): {NAME[m]: round(v, 1) for m, v in r['fh'].items()}
                                                              for r in rows}}


def fig_6_7():
    d = J('fig7'); xs = [5, 10, 20, 40, 80]; data = [d[str(x)] for x in xs]
    fig, ax = plt.subplots(figsize=(6.2, 4.6)); draw(ax, xs, data, logx=True)
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in xs]); ax.minorticks_off()
    ax.set_xlabel('Number of snapshots'); ax.set_ylabel('Mean per-user SINR (dB)')
    headroom(ax, 0.6); legend_inside(ax, ncol=2)
    save(fig, 'Fig6_07_convergence')
    RESULTS['Figure 6.7'] = {'x': 'trace length (snapshots)', 'points': per_point(d, [str(x) for x in xs])}


def fig_6_8():
    rows = J('fig8')['rows']; e2 = J('fig2')
    EPS10 = [r['eps'] for r in rows]
    sg = [abs(e2[str(e)]['CF-RAB'][0] - e2[str(e)]['centralized_sdp'][0]) for e in EPS2]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    a1.semilogx(EPS10, [r['gap_pct'] for r in rows], '-o', color='k', lw=2, ms=5,
                label='Algorithm 6.1 (CF-RAB) over primary-ball SOCP')
    a1.axhline(0, color='gray', lw=.8)
    a1.set_xlabel(r'CSI uncertainty radius $\varepsilon$'); a1.set_ylabel('Objective excess (%)')
    a1.grid(True, which='both', alpha=.3); headroom(a1, 0.25); legend_inside(a1, loc='upper left', ncol=1)
    a2.loglog(EPS2, np.maximum(sg, 1e-6), '-s', color='C3', lw=2, ms=5,
              label='|CF-RAB minus primary-ball SOCP|')
    a2.set_xlabel(r'CSI uncertainty radius $\varepsilon$'); a2.set_ylabel('|SINR difference| (dB)')
    a2.set_ylim(1e-6, 1.0); a2.grid(True, which='both', alpha=.3); legend_inside(a2, loc='upper left', ncol=1)
    fig.tight_layout(); save(fig, 'Fig6_08_optimality')
    RESULTS['Figure 6.8'] = {'objective excess (%)': {str(r['eps']): r3(r['gap_pct']) for r in rows},
                             '|CF-RAB minus primary-ball SOCP| (dB)': {str(e): round(float(s), 5) for e, s in zip(EPS2, sg)}}


def fig_6_9():
    rows = J('fig9')['rows']
    EPS8 = [0.05, 0.10, 0.20, 0.30]; SOI = [0, 10, 20, 30]
    ST8 = {'centralized_sdp': (':', 'none', 'k', 1.6, 'Primary-ball SOCP'),
           'CF-RAB': ('-', 's', 'C3', 2.2, 'CF-RAB (centralized)'),
           'MVDR': ('--', '^', 'C2', 1.8, 'Diagonally-loaded MVDR')}
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.3))
    for a, (xx, key, fix) in zip(axs, [(SOI, 'eps', 0.10), (EPS8, 'soi_dB', 20)]):
        for m, (ls, mk, col, lw, lab) in ST8.items():
            kw = dict(linestyle=ls, color=col, linewidth=lw, label=lab)
            if mk != 'none':
                kw.update(marker=mk, markersize=6)
            a.plot(xx, [r[m] for r in rows if abs(r[key] - fix) < 1e-9], **kw)
        a.set_ylabel('Mean per-user SINR (dB)'); headroom(a, 0.3)
        legend_inside(a, loc='upper left' if a is axs[0] else 'upper right', ncol=1)
    axs[0].set_xlabel('Desired-signal power in the covariance (dB)'); axs[0].set_xticks(SOI)
    axs[1].set_xlabel(r'CSI uncertainty radius $\varepsilon$')
    fig.tight_layout(); save(fig, 'Fig6_09_mismatch')
    RESULTS['Figure 6.9'] = {'rows': [{k: (r3(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]}


def fig_6_10():
    d = J('fig10'); g = d['eps_grid']
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.semilogx(g, d['per_departure'], '-o', color='C0', lw=2, ms=5, label=r'Per-AP problem ($M=8$)')
    ax.semilogx(g, d['joint_departure'], '-s', color='C3', lw=2, ms=5, label=r'Joint problem ($LM=128$)')
    ax.axvline(d['per_star_mean'], color='C0', ls=':', lw=1.6, label=r'Rescaling threshold, per-AP')
    ax.axvline(d['joint_star_mean'], color='C3', ls=':', lw=1.6, label=r'Rescaling threshold, joint')
    ax.set_xlabel(r'Design bound level $c$')
    ax.set_ylabel(r'$1-|\mathbf{w}^{\mathrm{H}}\mathbf{w}_{\mathrm{MV}}|\,/\,(\|\mathbf{w}\|_2\|\mathbf{w}_{\mathrm{MV}}\|_2)$')
    ax.grid(True, which='both', alpha=0.3); headroom(ax, 0.35); legend_inside(ax, loc='upper left', ncol=2)
    save(fig, 'Fig6_10_threshold')
    RESULTS['Figure 6.10'] = {'rescaling threshold, joint (LM = 128)': round(d['joint_star_mean'], 4),
                              'rescaling threshold, per-AP (M = 8)': round(d['per_star_mean'], 4),
                              'bound levels': g, 'departure, per-AP': [round(x, 5) + 0.0 for x in d['per_departure']],
                              'departure, joint': [round(x, 5) + 0.0 for x in d['joint_departure']]}


FIGURES = [fig_6_1, fig_6_2, fig_6_3, fig_6_4, fig_6_5, fig_6_6, fig_6_7, fig_6_8, fig_6_9, fig_6_10]


# ------------------------------------------------------------------------------------------------ tables
def write_table(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + '.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([header] + rows)
    with open(os.path.join(OUT, name + '.md'), 'w') as f:
        f.write('| ' + ' | '.join(header) + ' |\n|' + '---|' * len(header) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(x) for x in r) + ' |\n')
    RESULTS[name] = {'header': header, 'rows': rows}


n2 = lambda x: f'{x:.2f}'.replace('-', '−')
n3 = lambda x: f'{x:.3f}'.replace('-', '−')
iv = lambda v: f'{n2(v[0])} ({n2(v[1])}, {n2(v[2])})'
iv3 = lambda v: f'{n3(v[0])} ({n3(v[1])}, {n3(v[2])})'
LBK = ['heavy_e010', 'heavy_e050', 'small_e010', 'nom_e010', 'nom_e050', 'pl_e050']


def tables():
    """Tables 6.2 to 6.6 (CSV and Markdown) and the data of Figures 6.1-6.5 and 6.7 as one CSV table."""
    if have('B1'):
        B1 = J('B1'); T = []
        for k in LBK:
            v = B1[k + '|population']; c_ = v['config']; mm = v['test']['mean']
            T.append([c_['L'], c_['K'], f"{c_['eps']:.2f}", f"{v['selected']['c']:g}", n2(mm['D_tuned']), n2(mm['D_default']),
                      n2(mm['MR']), n2(mm['MV_tuned']), n2(mm['LP_tuned']), n2(mm['P_tuned']), n2(mm['D_tuned'] - mm['D_default'])])
        write_table('Table6_2_load_dependent_bound', ['L', 'K', 'eps', 'c', 'D-CF-RAB', 'D-CF-RAB, c = 0.98', 'MR + CPU',
                                                      'MVDR + CPU, selected', 'LP-MMSE + CPU', 'P-MMSE', 'Gain over c = 0.98'], T)
        T = []
        for k in LBK:
            v = B1[k + '|sample']; c_ = v['config']; t_ = v['test']; mm = t_['mean']
            T.append([c_['L'], c_['K'], f"{c_['eps']:.2f}", f"{v['selected']['c']:g}", n2(mm['D_tuned']), n2(mm['MV_tuned']),
                      n2(mm['MV_default']), iv3(t_['D_tuned - MV_tuned'])])
        write_table('Table6_3_sample_covariance', ['L', 'K', 'eps', 'c', 'D-CF-RAB', 'MVDR + CPU, selected', 'MVDR + CPU, 1e-6',
                                                   'D-CF-RAB minus MVDR + CPU, selected (95% interval)'], T)
    if have('A1'):
        A1 = J('A1'); T = []
        for e, nm in (('boundary', 'Boundary'), ('interior', 'Interior radial law'), ('radius', 'Radius-uniform'),
                      ('gaussian', 'Gaussian')):
            a = A1[f'norm_L16_e010|{e}']; b = A1[f'norm_L16_e050|{e}']; c = A1[f'pl_L16_e050|{e}']
            T.append([nm, n2(a['mean']['D-CF-RAB']), iv(a['D-P-MMSE']), n2(b['mean']['D-CF-RAB']), iv3(b['D-MR+W']),
                      iv(b['D-P-MMSE']), f"{b['resp_below1']} of {b['resp_n']}", iv(c['D-P-MMSE'])])
        write_table('Table6_4_error_laws', ['Error law', 'D-CF-RAB, eps = 0.10', 'D-CF-RAB minus P-MMSE, eps = 0.10',
                                            'D-CF-RAB, eps = 0.50', 'D-CF-RAB minus MR + CPU, eps = 0.50',
                                            'D-CF-RAB minus P-MMSE, eps = 0.50', 'Failed scored checks, eps = 0.50',
                                            'D-CF-RAB minus P-MMSE, path loss, eps = 0.50'], T)
    if have('B2'):
        B2 = J('B2'); T = []
        for s_, nm in (('J2_inr10', '2, 10 dB'), ('J2_inr20', '2, 20 dB'), ('J6_inr10', '6, 10 dB'), ('J6_inr20', '6, 20 dB')):
            mm = B2[s_]['test']['qam_impulsive']['mean']
            T.append([nm, f"{B2[s_]['selected']['D']:g}"] + [n2(mm[x]) for x in ('D', 'MV', 'ACPU', 'MR', 'LP', 'P', 'CF')])
        write_table('Table6_5_external_interferers', ['J, INR', 'Selected c', 'D-CF-RAB', 'MVDR + CPU', 'MR + adaptive CPU',
                                                      'MR + CPU', 'LP-MMSE + CPU', 'P-MMSE', 'CF-RAB'], T)
    T6 = [['16 APs, four users, c = 0.98', 'D-CF-RAB within 0.03 dB of MR + CPU combining'],
          ['Four APs, four users, nominal uncertainty', 'At c = 0.98, LP-MMSE + CPU combining is 4.2 dB higher. Load-dependent '
           'bound selection gains up to 2.7 dB in the four-AP configurations of Table 6.2 and up to 2.8 dB over MR + CPU combining.'],
          ['Nominal uncertainty, centralized processing', 'P-MMSE 2.5 dB above D-CF-RAB'],
          ['Largest tested radius (eps = 0.50)', 'D-CF-RAB 0.9 dB above P-MMSE for boundary errors, 0.7 dB below for '
           'radius-uniform errors; with path loss, above nominal but not tuned P-MMSE'],
          ['Covariance-based local processing', 'The bound acts as a loading rule: 3.2 to 4.0 dB above local MVDR with 1e-6 '
           'loading, within 0.14 dB of a training-selected loading (0.55 dB below it with path loss)'],
          ['Interferers without channel estimates', 'Covariance-based local processing 3.8 to 6.8 dB above the estimate-based '
           'combiners; tuned local MVDR 0.1 to 1.4 dB above D-CF-RAB'],
          ['Strong desired signal in the covariance with mismatch', 'Worst-case robust processing avoids self-cancellation '
           '(Fig. 6.9)']]
    write_table('Table6_6_observations', ['Operating regime', 'Observation in the tested range'], T6)
    S1 = []
    for st, fig, var in (('fig1', '6.1', 'L'), ('fig2', '6.2', 'eps'), ('fig3', '6.3', 'SNR_dB'), ('fig4', '6.4', 'tau'),
                         ('fig5', '6.5', 'profile'), ('fig7', '6.7', 'snapshots')):
        if not have(st):
            continue
        for x, dd in J(st).items():
            for m, v in dd.items():
                S1.append([f'Fig. {fig}', f'{var} = {x}', NAME.get(m, m), f'{v[0]:.3f}', f'{v[1]:.3f}', f'{v[2]:.3f}'])
    if S1:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, 'figure_data.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([['Figure', 'Point', 'Method', 'Mean SINR (dB)', '95% t half-width (dB)',
                                      'Worst-user SINR (dB)']] + S1)


# ------------------------------------------------------------------------------------------------ text results
def text_numbers():
    """Results quoted only in the text (Sections 6.7.1, 6.7.2, 6.7.6, 6.7.9, 6.7.11 to 6.7.15)."""
    T = {}
    if have('train'):
        tr = J('train')
        T['bound level selected on training realizations (Section 6.7.1)'] = tr['c_star']
        T['training SINR per bound level (dB)'] = {c: round(float(np.mean(list(v.values()))), 3) for c, v in tr['train'].items()}
        sc = tr['solver_check']
        T['scalar-root solver check (Section 6.3.6)'] = {'problems': len(sc['cases']), 'max relative objective difference':
                                                          sc['max_rel_obj'], 'max constraint residual': sc['max_residual'],
                                                          'speed-up over a loading-grid search, n = 8': round(sc['speedup_n8'], 1)}
    if have('paired'):
        p = J('paired')
        T['paired comparisons, 20 realizations (D-CF-RAB minus method, dB, 95% interval)'] = {
            k: {kk.replace('D-', '', 1): [r3(x) for x in vv] for kk, vv in v.items() if kk.startswith('D-')}
            for k, v in p['A_paired'].items()}
        T['exact minus Algorithm 6.1 local stage (dB, 95% interval)'] = {
            k: [r3(x) for x in v['exact_minus_active_set']] for k, v in p['B_local_stage'].items()}
    if have('audits'):
        a = J('audits')
        T['joint conic audit, median excess of Algorithm 6.1 (%)'] = {e: r3(np.median(v['alg_vs_same_pct'])) for e, v in a['C_joint'].items()}
        T['joint conic audit, median effect of the extra balls (%)'] = {e: r3(np.median(v['same_vs_primary_pct'])) for e, v in a['C_joint'].items()}
        rows = a['C_local']['rows']
        if rows:
            ratio = [r['excess_ratio'] for r in rows]
            T['local conic audit'] = {'solves': len(rows), 'objective ratio min/median/max': [r3(min(ratio)), r3(np.median(ratio)), r3(max(ratio))],
                                      'median direction difference (deg)': r3(np.median([np.degrees(np.arccos(min(1.0, r['direction_cos'])))
                                                                                       for r in rows]))}
        T['CF-RAB bound sweep, mean SINR (dB)'] = {c: r3(v['mean']) for c, v in a['D_cfrab_bound'].items()}
        T['cross-AP rescaling outcomes'] = a['violations']
    if have('lbound'):
        lb = J('lbound')
        T['load-dependent bound selection (Section 6.7.12)'] = {
            k: {'selected': v['selected'], 'D-CF-RAB tuned minus MR + CPU': [r3(x) for x in v['test']['D_tuned - MR']],
                'worst-user gain over MR + CPU': [r3(x) for x in v['test']['D_tuned_worst - MR_worst']],
                'D-CF-RAB (c = 0.98) minus P-MMSE nominal': [r3(x) for x in v['test']['D_default - P_default']],
                'D-CF-RAB (c = 0.98) minus P-MMSE tuned': [r3(x) for x in v['test']['D_default - P_tuned']]}
            for k, v in lb.items()}
    if have('A2'):
        T['bound selected per error law'] = {k: v['selected'] for k, v in J('A2').items()}
    if have('A3'):
        a3 = J('A3')
        T['equal-mean tiers (Section 6.7.6)'] = {'mean SINR (dB)': {NAME[m]: r3(v[0]) for m, v in a3['A3'].items()},
                                                 'D-CF-RAB minus method (dB, 95% interval)': {NAME[m]: [r3(x) for x in v]
                                                                                              for m, v in a3['A3_paired'].items()}}
    if have('A4'):
        T['joint-check audit (Section 6.7.9)'] = J('A4')
    if have('A5'):
        T['CF-RAB continuous primary-ball check (Section 6.7.13)'] = J('A5')
    RESULTS['Text'] = T


def write_summary():
    text_numbers()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'results_summary.json')
    with open(p, 'w') as f:
        json.dump(RESULTS, f, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return p


def make_all():
    for f in FIGURES:
        f()
    tables()
    return write_summary()
