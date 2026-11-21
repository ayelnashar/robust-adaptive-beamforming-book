"""
ch8wrab.figures -- Figures 8.1 to 8.11, Tables 8.1 to 8.3 and the results summary of Chapter 8, drawn from the data
that the stages of ch8wrab.scenarios write into FIGDIR/data.

Configured through environment variables read at import time:
  FIGDIR=path    output folder (default ./figures); the simulation data are read from FIGDIR/data
  RAB_INLINE=1   also show each figure under its cell in Jupyter
"""
import csv
import json
import os

import numpy as np
import matplotlib
if os.environ.get('RAB_INLINE') != '1':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .algorithms import METHOD_LABELS, seed_for, subcarrier_frequencies

OUT = os.environ.get('FIGDIR', 'figures')
DATA = os.path.join(OUT, 'data')
INLINE = os.environ.get('RAB_INLINE') == '1'
RESULTS = {}

plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True,
    'axes.linewidth': 0.9, 'axes.spines.top': True, 'axes.spines.right': True, 'axes.labelsize': 10.5,
    'lines.linewidth': 1.6, 'lines.markersize': 5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04, 'errorbar.capsize': 2.5,
})
LEG = dict(frameon=True, fancybox=False, framealpha=0.95, edgecolor='#444444', fontsize=7.6,
           borderpad=0.45, labelspacing=0.3, columnspacing=1.0, handlelength=2.6)

# method key -> colour, marker, fill; the same in every figure
STYLE = {
    'analog': ('tab:blue', 'o', 'full'), 'wc': ('tab:orange', 's', 'full'), 'mvdr': ('tab:olive', 'v', 'full'),
    'rcb': ('tab:red', 'D', 'full'), 'socp': ('k', 'D', 'none'), 'recon': ('tab:brown', 'X', 'full'),
    'fixed': ('tab:purple', 'P', 'full'), 'wrab': ('tab:green', '^', 'full'), 'wc_faware': ('tab:cyan', 'h', 'full'),
    'msocp': ('darkgreen', 'o', 'none'), 'lcmv': ('tab:pink', '*', 'full'),
    'wrab_bwfan': ('tab:green', '^', 'full'), 'wrab_anglefan': ('darkgreen', 'p', 'none'),
}
LABEL = dict(METHOD_LABELS, wrab_bwfan='WRAB, beamwidth-relative fan', wrab_anglefan='WRAB, fan sized in degrees')


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


def headroom(ax, frac):
    """Raise the top of the y-axis so that a legend at the top of the axes stays clear of the curves."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac*(hi - lo))


def footroom(ax, frac):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - frac*(hi - lo), hi)


def legend_inside(ax, loc='upper center', ncol=2, **kw):
    return ax.legend(loc=loc, ncol=ncol, **{**LEG, **kw})


def eb(ax, x, s, key, lw=None):
    """Mean with its 95% bootstrap interval; s is a list of summaries {mean_db, ci95_low_db, ci95_high_db}."""
    c, mk, fill = STYLE[key]
    mu = np.array([v['mean_db'] for v in s]); lo = np.array([v['ci95_low_db'] for v in s])
    hi = np.array([v['ci95_high_db'] for v in s])
    big = key in ('wrab', 'wrab_bwfan')
    ax.errorbar(x, mu, yerr=[np.maximum(mu - lo, 0), np.maximum(hi - mu, 0)], color=c, marker=mk, ls='-',
                fillstyle=fill, label=LABEL[key], lw=lw or (2.3 if big else 1.5), markeredgewidth=1.1,
                zorder=6 if big else 4)


def r3(x):
    return round(float(x), 3)


def pts(x, s):
    return {str(xx): {'mean_db': r3(v['mean_db']), 'ci95': [r3(v['ci95_low_db']), r3(v['ci95_high_db'])]}
            for xx, v in zip(x, s)}


def sweep_results(name, x, recs, methods):
    RESULTS[name] = {'x': list(x), 'points': {LABEL[m]: pts(x, [r['summary'][m] for r in recs]) for m in methods}}


def per_subcarrier(rec, method, seed):
    """Per-subcarrier mean SINR in dB with trial-bootstrap 95% intervals (2000 resamples)."""
    v = 10*np.log10(np.asarray(rec['raw_linear'][method]))
    rng = np.random.default_rng(seed); ix = rng.integers(v.shape[0], size=(2000, v.shape[0]))
    lo, hi = np.quantile(v[ix].mean(1), [.025, .975], axis=0)
    return v.mean(0), lo, hi


# ------------------------------------------------------------------------------------------------ figures
def fig_8_1():
    d = J('analytic')['fig1']
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for M, ls in ((16, '-'), (64, '--'), (256, '-.')):
        ax.plot(d['frequency_offset_GHz'], d[f'N{M}'], ls=ls, label=f'$N$ = {M}')
    ax.set(xlabel='Frequency offset (GHz)', ylabel='Normalized array gain (dB)', ylim=(-30, 1))
    legend_inside(ax, loc='lower center', ncol=1)
    save(fig, 'Fig8_01_squint_loss')
    f = np.asarray(d['frequency_offset_GHz'])
    RESULTS['Figure 8.1'] = {f'N={M}': {'band_edge_loss_db': r3(min(d[f'N{M}'][0], d[f'N{M}'][-1])),
                                        'deepest_db': r3(min(d[f'N{M}']))} for M in (16, 64, 256)}
    RESULTS['Figure 8.1']['offsets_GHz'] = [float(f[0]), float(f[-1])]


def fig_8_2():
    d = J('analytic')['fig2']
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for M, ls in ((64, '--'), (256, '-.')):
        ax.plot(d['jitter_rms_deg'], d[f'N{M}'], ls=ls, marker='o', label=f'$N$ = {M}')
    ax.set(xlabel='Pointing-error RMS (deg)', ylabel='Expected power-gain loss (dB)')
    legend_inside(ax, loc='upper left', ncol=1)
    save(fig, 'Fig8_02_pointing_loss')
    RESULTS['Figure 8.2'] = {f'N={M}': dict(zip([f'{s:g}' for s in d['jitter_rms_deg']], map(r3, d[f'N{M}'])))
                             for M in (64, 256)}


def fig_8_3():
    r = J('fig3'); c = r['config']
    fo = (subcarrier_frequencies(c['fc'], c['B'], c['Nsc']) - c['fc'])/1e9
    fig, ax = plt.subplots(figsize=(6.2, 4.2)); res = {}
    for m in r['methods']:
        mu, lo, hi = per_subcarrier(r, m, 3)
        eb(ax, fo, [dict(mean_db=a, ci95_low_db=b, ci95_high_db=h) for a, b, h in zip(mu, lo, hi)], m)
        res[LABEL[m]] = {'per_subcarrier_db': list(map(r3, mu)), 'band_mean_db': r3(r['summary'][m]['mean_db'])}
    ax.set(xlabel='Frequency offset from carrier (GHz)', ylabel='Output SINR (dB)')
    headroom(ax, 0.3); legend_inside(ax, loc='upper center', ncol=2)
    save(fig, 'Fig8_03_SINR_across_band')
    RESULTS['Figure 8.3'] = res


def fig_8_4():
    d = J('fig4'); x = list(range(7)); recs = [d[str(i)] for i in x]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in recs[0]['methods']:
        eb(ax, x, [r['summary'][m] for r in recs], m)
    ax.set(xlabel='Steering mismatch (deg)', ylabel='Output SINR (dB)')
    headroom(ax, 0.22); legend_inside(ax, loc='upper right', ncol=1)
    save(fig, 'Fig8_04_SINR_vs_mismatch')
    sweep_results('Figure 8.4', x, recs, recs[0]['methods'])


def fig_8_5():
    d = J('fig5'); x = [0, 15, 30, 45, 60]; recs = [d[str(a)] for a in x]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for m in recs[0]['methods']:
        eb(ax, x, [r['summary'][m] for r in recs], m)
    ax.set(xlabel='Nominal desired direction (deg)', ylabel='Output SINR (dB)', xticks=x)
    headroom(ax, 0.55); legend_inside(ax, ncol=2)
    save(fig, 'Fig8_05_SINR_vs_steering')
    sweep_results('Figure 8.5', x, recs, recs[0]['methods'])


def fig_8_6():
    d = J('fig6'); x = [3, 5, 8, 12, 20, 40]; recs = [d[str(s)] for s in x]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in recs[0]['methods']:
        eb(ax, x, [r['summary'][m] for r in recs], m)
    ax.set(xlabel='Interferer-source separation (deg)', ylabel='Output SINR (dB)')
    footroom(ax, 0.05); legend_inside(ax, loc='lower right', ncol=1)
    save(fig, 'Fig8_06_SINR_vs_separation')
    sweep_results('Figure 8.6', x, recs, recs[0]['methods'])


def fig7_gap(recs):
    """WRAB minus the better reference, with the paired bootstrap interval of each array size."""
    out = []
    for i, r in enumerate(recs):
        a, b, w = (np.asarray(r['samples'][m]) for m in ('analog', 'wc', 'wrab'))
        NR = len(w); rr = np.random.default_rng(seed_for('fig7_gap', i)); ix = rr.integers(NR, size=(2000, NR))
        smp = w[ix].mean(1) - np.maximum(a[ix].mean(1), b[ix].mean(1))
        lo, hi = np.quantile(smp, [.025, .975])
        out.append({'gap_db': r3(w.mean() - max(a.mean(), b.mean())), 'ci95': [r3(lo), r3(hi)]})
    return out


def fig_8_7():
    Ns = [64, 128, 256, 512]
    d = J('fig7'); recs = [d[str(M)] for M in Ns]; fa = J('fixedangle'); rows = [fa[str(M)] for M in Ns]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 7.4))
    for m in ('analog', 'wc'):
        eb(a1, Ns, [r['summary'][m] for r in recs], m)
        eb(a2, Ns, [r[m] for r in rows], m)
    eb(a1, Ns, [r['summary']['wrab'] for r in recs], 'wrab_bwfan')
    eb(a2, Ns, [r['wrab'] for r in rows], 'wrab_bwfan')
    eb(a2, Ns, [r['wrab_anglefan'] for r in rows], 'wrab_anglefan')
    for M, r in zip(Ns, rows):
        a2.annotate(f"$N_c$={r['Nc_anglefan']}", (M, r['wrab_anglefan']['mean_db']), textcoords='offset points',
                    xytext=(0, -15), ha='center', fontsize=8.5)
    for a, t in ((a1, '(a) Mismatch of 1.5 beamwidths'), (a2, r'(b) Mismatch of 1$\degree$ for every $N$')):
        a.set(xlabel='Number of elements $N$', ylabel='Output SINR (dB)', xscale='log', title=t)
        a.set_xticks(Ns); a.set_xticklabels([str(n) for n in Ns]); a.minorticks_off()
    headroom(a1, 0.35); legend_inside(a1, loc='upper left', ncol=1)
    headroom(a2, 0.45); legend_inside(a2, loc='upper center', ncol=2)
    fig.tight_layout(); save(fig, 'Fig8_07_array_scaling')
    gap = fig7_gap(recs)
    RESULTS['Figure 8.7(a)'] = {'x': Ns, 'points': {LABEL[m]: pts(Ns, [r['summary'][m] for r in recs])
                                                    for m in ('analog', 'wc', 'wrab')},
                                'WRAB minus better reference': dict(zip(map(str, Ns), gap))}
    RESULTS['Figure 8.7(b)'] = {'x': Ns, 'points': {LABEL[k]: pts(Ns, [r[m] for r in rows]) for k, m in
                                                    (('analog', 'analog'), ('wc', 'wc'), ('wrab_bwfan', 'wrab'),
                                                     ('wrab_anglefan', 'wrab_anglefan'))},
                                'N_c': {str(M): r['Nc_anglefan'] for M, r in zip(Ns, rows)}}


def fig_8_8():
    lev = [0, .1, .2, .3, .4, .5, .6]
    d = J('fig8'); recs = [d[str(s)] for s in lev]; dyn = J('dynamic'); drows = [dyn['points'][str(s)] for s in lev]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 7.2), sharey=True)
    for m in ('analog', 'wc', 'wrab'):
        eb(a1, lev, [r['summary'][m] for r in recs], m)
        eb(a2, lev, [r[m] for r in drows], m)
    a1.set(xlabel='Pointing-error RMS (deg)', ylabel='Output SINR (dB)', title='(a) Quasi-static source offset')
    a2.set(xlabel='Rotation RMS (deg)', ylabel='Output SINR (dB)',
           title=rf"(b) Dynamic array rotation, AR(1) $\rho$ = {dyn['rho']}")
    footroom(a1, 0.15); legend_inside(a1, loc='lower left', ncol=1); legend_inside(a2, loc='lower left', ncol=1)
    fig.tight_layout(); save(fig, 'Fig8_08_jitter')
    sweep_results('Figure 8.8(a)', lev, recs, ('analog', 'wc', 'wrab'))
    RESULTS['Figure 8.8(b)'] = {'x': lev, 'rho': dyn['rho'], 'window': dyn['window'],
                                'points': {LABEL[m]: pts(lev, [r[m] for r in drows]) for m in ('analog', 'wc', 'wrab')}}


def fig_8_9():
    d = J('fig9'); fig, ax = plt.subplots(figsize=(6.2, 4.2)); res = {}
    for key, mk, lab, col in (('1', '^', 'Frequency-consistent grid', 'tab:green'),
                              ('0', 'v', 'Carrier-referenced grid', 'tab:orange')):
        r = d[key]; c = r['config']
        fo = (subcarrier_frequencies(c['fc'], c['B'], c['Nsc']) - c['fc'])/1e9
        mu, lo, hi = per_subcarrier(r, 'wrab', 9)
        ax.errorbar(fo, mu, yerr=[np.maximum(mu-lo, 0), np.maximum(hi-mu, 0)], marker=mk, color=col, label=lab,
                    lw=2.0 if key == '1' else 1.5)
        res[lab] = {'offsets_GHz': list(map(r3, fo)), 'per_subcarrier_db': list(map(r3, mu))}
    ax.set(xlabel='Frequency offset (GHz)', ylabel='WRAB output SINR (dB)')
    footroom(ax, 0.1); legend_inside(ax, loc='lower center', ncol=2)
    save(fig, 'Fig8_09_grid_ablation')
    RESULTS['Figure 8.9'] = res


def fig_8_10():
    d = J('fig10'); inits = (0.1, 1.5, 3.0); x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.2, 4.2)); res = {}
    for shift, adapt, lab, col in ((-.18, 1, 'Adaptation on', 'tab:green'), (.18, 0, 'Margins frozen', 'tab:gray')):
        S = [d[f'{adapt}_{v:g}']['summary']['wrab'] for v in inits]
        mu = np.array([s['mean_db'] for s in S]); lo = np.array([s['ci95_low_db'] for s in S])
        hi = np.array([s['ci95_high_db'] for s in S])
        ax.bar(x + shift, mu, .36, yerr=[mu - lo, hi - mu], capsize=3, label=lab, color=col, edgecolor='k', lw=0.6)
        for xi, m, h in zip(x + shift, mu, hi):
            ax.text(xi, h + 0.08, f'{m:.2f}', ha='center', fontsize=8.5)
        res[lab] = {'means_db': dict(zip([f'{v:g}' for v in inits], map(r3, mu))), 'spread_db': r3(np.ptp(mu))}
    ax.set_xticks(x, [f'Initial margin {v:g}' for v in inits]); ax.set_ylim(20, 26.5)
    ax.set_ylabel('WRAB output SINR (dB)'); legend_inside(ax, loc='upper right', ncol=1)
    save(fig, 'Fig8_10_initialization')
    RESULTS['Figure 8.10'] = res


MM11 = [0, 1, 2, 3, 4, 5, 6, 8]


def fig_8_11():
    d = J('fig11'); recs = [d[str(m)] for m in MM11]; ref = J('fig11ref'); rows = [ref['points'][str(m)] for m in MM11]
    gk = f"lcmv_{ref['lcmv_best_load']:g}"
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for m in ('analog', 'wc', 'mvdr', 'rcb', 'socp', 'recon', 'fixed', 'wrab'):
        eb(ax, MM11, [r['summary'][m] for r in recs], m)
    eb(ax, MM11, [r['msocp'] for r in rows], 'msocp')
    eb(ax, MM11, [r[gk] for r in rows], 'lcmv')
    ax.set(xlabel='Steering mismatch (deg)', ylabel='Output SINR (dB)')
    headroom(ax, 0.40); legend_inside(ax, loc='upper center', ncol=2)
    save(fig, 'Fig8_11_comparison')
    RESULTS['Figure 8.11'] = {'x': MM11, 'points': {**{LABEL[m]: pts(MM11, [r['summary'][m] for r in recs])
                                                       for m in recs[0]['methods']},
                                                    LABEL['msocp']: pts(MM11, [r['msocp'] for r in rows]),
                                                    LABEL['lcmv']: pts(MM11, [r[gk] for r in rows])},
                              'lcmv_loading': ref['lcmv_best_load'],
                              'lcmv_all_loads_mean_db': {k.split('_')[1]: {str(m): r3(ref['points'][str(m)][k]['mean_db'])
                                                                         for m in MM11}
                                                         for k in rows[0] if k.startswith('lcmv_')}}


# ------------------------------------------------------------------------------------------------ tables
def write_table(name, header, rows, title):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + '.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with open(os.path.join(OUT, name + '.md'), 'w') as f:
        f.write(f'**{title}**\n\n| ' + ' | '.join(header) + ' |\n|' + '---|'*len(header) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(c) for c in r) + ' |\n')


def tables():
    write_table('Table8_1_settings', ['Parameter', 'Value'], [
        ['Carrier / bandwidth', '300 GHz / 20 GHz (40 GHz for the grid ablation)'],
        ['Array', 'ULA, half-wavelength spacing at the carrier'],
        ['Element counts', '64 (core sweeps); 128 (Figures 8.8-8.10); 64-512 (scaling)'],
        ['Processed subcarriers', '3-11, depending on the experiment'],
        ['Snapshots per run K', '300 (220 in Figure 8.10, 600 in Figure 8.11)'],
        ['Monte Carlo trials', '12-20 per point; 20 in Figures 8.8, 8.10 and 8.11'],
        ['SNR / INR per element', '5 dB / 10 dB'], ['Forgetting factor', '0.97'],
        ['Warm-up before margin adaptation', '50 snapshots'], ['Margin update period T', '30 snapshots'],
        ['Retained eigenspace order', 'ceil(0.25 M)'],
        ['Constraint grid', '(0, -0.6, +0.6, -1.2) beamwidths; fixed (0, +/-1.5, -3) deg in Figures 8.3 and 8.9 and '
                            '(0, +/-0.4, -0.8) deg in Figures 8.8 and 8.10; symmetric (0, +/-0.6, +/-1.2) beamwidths in '
                            'Figure 8.11'],
        ['Initial margins before anchoring', '(1.9, 1.5, 1.5, 1.1), then one-time anchors and (8.18)-(8.20) with '
                                             'ceiling 3; 0.3 in Figure 8.11'],
        ['Step relaxation', '0.1'],
        ['Single-ball radius (WC-single, robust Capon, SOCP)', '0.6 sqrt(M) (0.85 sqrt(M) in Figure 8.11; 3.8 in Figure 8.3)'],
        ['Multipath case', '3 block-static paths, +/-0.2 deg spread, unit total path power'],
        ['Random seed', '42, with an independent stream per experiment']], 'Table 8.1 - Principal simulation settings.')
    res = {}
    if have('near'):
        nr = J('near'); seps = ['2', '3', '4', '5', '8']; rows = []
        for s in seps:
            p = nr['points'][s]
            rows.append([s + ' deg', f"{p['adaptive']['mean_db']:.1f}"] +
                        [f"{p[f'fixed_{v:g}']['mean_db']:.1f}" for v in nr['levels']] + [f"{p['lcmv']['mean_db']:.1f}"])
        widest = max(v['ci95_high_db'] - v['ci95_low_db'] for s in seps for k, v in nr['points'][s].items()
                     if isinstance(v, dict) and 'ci95_high_db' in v and (k == 'adaptive' or k.startswith('fixed')
                                                                          or k == 'lcmv'))
        write_table('Table8_2_near_interferer', ['Sep.', 'Adaptive', 'Fixed 0.3', 'Fixed 1', 'Fixed 2', 'Fixed 3', 'LCMV'],
                    rows, 'Table 8.2 - Output SINR (dB) with one interferer near the source, N = 64, 2 deg mismatch. '
                          f'Means of 18 trials; widest 95% interval {widest:.1f} dB.')
        res['Table 8.2'] = {s: {'adaptive': r3(nr['points'][s]['adaptive']['mean_db']),
                                **{f'fixed {v:g}': r3(nr['points'][s][f'fixed_{v:g}']['mean_db']) for v in nr['levels']},
                                'LCMV': r3(nr['points'][s]['lcmv']['mean_db']),
                                'LCMV loading': nr['points'][s]['lcmv']['load'],
                                'margins': nr['points'][s]['margins']} for s in seps}
        res['Table 8.2']['widest_ci_db'] = r3(widest)
    if have('table3'):
        t3 = J('table3'); rows = []; names = {'S1': 'S1: 2 deg mismatch, 0.1 deg jitter',
                                              'S2': 'S2: 4 deg mismatch, 0.2 deg jitter',
                                              'S3': 'S3: 6 deg mismatch, 0.3 deg jitter',
                                              'S4': 'S4: 2 deg mismatch, 3-path spread'}
        for k in ('S1', 'S2', 'S3', 'S4'):
            sm = t3[k]['summary']
            rows.append([names[k]] + [f"{sm[m]['mean_db']:.1f} ± {sm[m]['std_db']:.1f}" for m in ('analog', 'wc', 'wrab')])
        write_table('Table8_3_scenarios', ['Scenario', 'Analog-only', 'WC-single', 'WRAB'], rows,
                    'Table 8.3 - Wideband output SINR (dB), mean ± standard deviation, N = 64.')
        res['Table 8.3'] = {k: {LABEL[m]: {'mean_db': r3(t3[k]['summary'][m]['mean_db']),
                                           'std_db': r3(t3[k]['summary'][m]['std_db'])} for m in ('analog', 'wc', 'wrab')}
                            for k in ('S1', 'S2', 'S3', 'S4')}
    RESULTS.update(res)


# ------------------------------------------------------------------------------------------------ text numbers
def text_numbers():
    T = {}
    if have('controls'):
        c = J('controls')
        T['margin control, 64 elements (Section 8.5.8)'] = {
            k: r3(v['summary']['wrab']['mean_db']) for k, v in c['margins'].items()}
        T['signed mismatch, asymmetric grid (Section 8.5.3)'] = {
            k: {LABEL[m]: r3(v['summary'][m]['mean_db']) for m in v['methods']} for k, v in c['signed'].items()}
        T['signal present in training (Section 8.5.9)'] = {
            k: {LABEL[m]: r3(v['summary'][m]['mean_db']) for m in v['methods']} for k, v in c['mixed'].items()}
        T['signal present, fixed margin levels'] = {k: {kk: r3(vv['mean_db']) for kk, vv in v.items()}
                                                    for k, v in c['mixed_levels'].items()}
    if have('feasibility'):
        fz = J('feasibility')
        T['frozen-norm correction against the exact projection (Section 8.3.1)'] = {
            k: v for k, v in fz['frozen_norm'].items() if 'trace' not in k}
        T['online objective against the frozen final problem (Section 8.3.4)'] = fz['frozen_gap']
    if have('fig11ref'):
        T['Figure 8.11 run times, ms (machine dependent)'] = J('fig11ref')['time_ms']
    if have('timing'):
        T['timing, ms (machine dependent; measured alone)'] = J('timing')
    if have('fixedangle'):
        T['fan sized in degrees, ms per subcarrier and snapshot'] = {
            M: r3(v['ms_per_frequency_snapshot_anglefan']) for M, v in J('fixedangle').items()}
    T['largest constraint residual over all runs'] = max_residual()
    RESULTS['Text'] = T


def max_residual(datadir=None):
    """Largest constraint residual of (8.8) over every adaptive update of every recorded run."""
    datadir = datadir or DATA
    vals = []
    for s in ('fig3', 'fig4', 'fig5', 'fig6', 'fig7', 'fig8', 'fig9', 'fig10', 'fig11', 'table3', 'controls', 'near',
              'fixedangle'):
        p = os.path.join(datadir, s + '.json')
        if not os.path.exists(p):
            continue
        stack = [json.load(open(p))['result']]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                if isinstance(o.get('max_residual'), (int, float)):
                    vals.append(o['max_residual'])
                stack.extend(v for k, v in o.items() if k != 'max_residual')
            elif isinstance(o, list):
                stack.extend(x for x in o if isinstance(x, (dict, list)))
    return max(vals) if vals else None


def write_summary():
    path = os.path.join(OUT, 'results_summary.json')
    with open(path, 'w') as f:
        json.dump(RESULTS, f, indent=1)
    return path


FIGS = ('8_1', '8_2', '8_3', '8_4', '8_5', '8_6', '8_7', '8_8', '8_9', '8_10', '8_11')


def make_all():
    for k in FIGS:
        globals()['fig_' + k]()
    tables(); text_numbers()
    return write_summary()
