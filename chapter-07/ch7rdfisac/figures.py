"""Figures 7.1-7.9 of Chapter 7, drawn from the stage data in <datadir>. One function per figure.

Every plot has a full frame and a framed legend inside the plot, placed clear of the curves (extra headroom is
added where needed). Each method keeps the same colour and marker in every figure."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

from .summary import pd_swerling

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.grid': True, 'grid.alpha': .22,
                     'axes.spines.top': True, 'axes.spines.right': True, 'axes.linewidth': 0.8,
                     'legend.frameon': True, 'legend.framealpha': 0.95, 'legend.edgecolor': '0.35',
                     'legend.fancybox': False, 'legend.fontsize': 6.5})
LAB = {'rdf_as_isac': 'RDF-AS-ISAC', 'projection': 'Projection reconstruction', 'socp': 'SOCP, base radius',
       'socp_radius': 'SOCP, matched radius', 'socp_online': 'SOCP, base radius, online',
       'socp_radius_online': 'SOCP, matched radius, online', 'fixed': 'Fixed beam', 'vu_mbwc': 'VU-MBWC',
       'true_cov': 'True covariance', 'naive': 'LCMV'}
COL = {'rdf_as_isac': '#087d65', 'projection': '#507aa6', 'socp': '#8b949e', 'socp_radius': '#d18a29', 'socp_online': '#8b949e',
       'socp_radius_online': '#c0392b', 'fixed': '#7f6a3c', 'vu_mbwc': '#843d95', 'true_cov': '#000000', 'naive': '#646b73'}
MK = {'rdf_as_isac': 'o', 'projection': 'D', 'socp': '^', 'socp_radius': 'v', 'socp_online': '^', 'socp_radius_online': 's',
      'fixed': 'X', 'vu_mbwc': 'P', 'true_cov': 'h', 'naive': '*'}
W1 = 3.45  # single-column width (in)
NAMES = {1: 'Fig7_01_paired_gains', 2: 'Fig7_02_separation_power_mismatch', 3: 'Fig7_03_target_crossing',
         4: 'Fig7_04_adaptation', 5: 'Fig7_05_beam_patterns', 6: 'Fig7_06_certificate', 7: 'Fig7_07_priority',
         8: 'Fig7_08_coverage', 9: 'Fig7_09_detection'}


class Data:
    def __init__(self, datadir):
        self.dir = Path(datadir)

    def __call__(self, name):
        return json.loads((self.dir/f'{name}.json').read_text())


def _save(fig, outdir, k):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(outdir/f'{NAMES[k]}.{ext}', dpi=300, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)
    return outdir/f'{NAMES[k]}.png'


def _ci(d):
    d = np.asarray(d, float); n = len(d); mu = d.mean(0)
    h = stats.t.ppf(.975, n-1)*d.std(0, ddof=1)/np.sqrt(n) if n > 1 else np.zeros_like(mu)
    return mu, h


def _line(ax, x, y, a, **kw):
    ax.plot(x, y, color=COL[a], marker=MK[a], lw=1.3, ms=4, label=LAB[a], **kw)


def _headroom(ax, frac):
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + frac*(hi-lo))


def _ptitle(ax, t): ax.set_title(t, loc='left', fontsize=8.5)


def fig_7_1(D, outdir):
    """Held-out paired gains of RDF-AS-ISAC (mean user SINR and mean target SNR), 95% intervals, symlog axis."""
    held = [r for r in D('main') if r['split'] == 'heldout']; base = ['projection', 'socp_radius', 'socp_radius_online']
    fig, axs = plt.subplots(2, 1, figsize=(W1, 4.3), sharex=True); fig.subplots_adjust(hspace=.28)
    for ax, sl, ttl in zip(axs, [slice(0, 2), slice(2, 4)], ['(a) Mean user SINR gain', '(b) Mean target SNR gain']):
        for j, b in enumerate(base):
            mu, h = [], []
            for sc in 'ABCDFG':
                d = [np.mean(r['metrics']['rdf_as_isac'][sl]) - np.mean(r['metrics'][b][sl]) for r in held if r['scenario'] == sc]
                m_, h_ = _ci(d); mu.append(m_); h.append(h_)
            ax.errorbar(np.arange(6) + (j-1)*.22, mu, yerr=h, fmt=MK[b], color=COL[b], ms=4.5, capsize=2.5, lw=1, label='vs ' + LAB[b])
        ax.axhline(0, color='k', lw=.7); ax.set_yscale('symlog', linthresh=1, linscale=1.2)
        tk = [-1, 0, 1, 3, 10, 30]; ax.set_yticks(tk); ax.set_yticklabels([str(t).replace('-', '−') for t in tk])
        ax.set_ylim(-2.6, 45); ax.set_ylabel('Paired gain (dB)'); _ptitle(ax, ttl)
    axs[0].set_ylim(-2.6, 4000)
    axs[0].legend(loc='upper left')
    axs[1].set_xticks(range(6)); axs[1].set_xticklabels(list('ABCDFG')); axs[1].set_xlabel('Scenario')
    return _save(fig, outdir, 1)


def fig_7_2(D, outdir):
    """User SINR versus interferer offset, interferer INR and joint angular error."""
    sw = D('sweep'); SM = ['rdf_as_isac', 'projection', 'socp_radius', 'fixed', 'true_cov']
    fig, axs = plt.subplots(3, 1, figsize=(W1, 6.6)); fig.subplots_adjust(hspace=.55)
    for ax, kind, xl, ttl, sl, yl in [(axs[0], 'separation', 'Interferer offset from user 1 (deg)', '(a) Separation, INR 25 dB', slice(0, 1), 'User 1 SINR (dB)'),
                                      (axs[1], 'inr', 'Interferer INR (dB)', '(b) Power, offset 8°', slice(0, 1), 'User 1 SINR (dB)'),
                                      (axs[2], 'mismatch', 'Angular error s.d. (deg)', '(c) Mismatch, Scenario D', slice(0, 2), 'Mean user SINR (dB)')]:
        rows = [r for r in sw if r['kind'] == kind]; xs = sorted({r['val'] for r in rows})
        for a in SM:
            _line(ax, xs, [np.mean([np.mean(r['metrics'][a][sl]) for r in rows if r['val'] == v]) for v in xs], a)
        ax.set_xlabel(xl); ax.set_ylabel(yl); _ptitle(ax, ttl)
    axs[2].set_ylim(-14, 13)
    axs[2].legend(loc='lower left', ncol=2)
    return _save(fig, outdir, 2)


def fig_7_3(D, outdir):
    """User 1 SINR and target 1 SNR as target 1 approaches user 1 in azimuth."""
    cr = D('cross'); ds = sorted({r['delta'] for r in cr}); SM = ['rdf_as_isac', 'projection', 'socp_radius', 'fixed', 'true_cov']
    fig, axs = plt.subplots(2, 1, figsize=(W1, 4.4), sharex=True); fig.subplots_adjust(hspace=.28)
    for ax, i, yl, ttl in [(axs[0], 0, 'User 1 SINR (dB)', '(a) Communication'), (axs[1], 2, 'Target 1 SNR (dB)', '(b) Sensing')]:
        for a in SM: _line(ax, ds, [np.mean([r['metrics'][a][i] for r in cr if r['delta'] == d]) for d in ds], a)
        ax.set_ylabel(yl); _ptitle(ax, ttl)
    _headroom(axs[0], 0.9); axs[0].legend(loc='upper center', ncol=2)
    axs[1].set_xlabel('Target 1 offset from user 1 (deg)')
    return _save(fig, outdir, 3)


def fig_7_4(D, outdir):
    """Causal user 1 SINR in Scenarios D, F and G, averaged over the reference scenes."""
    ref = [r for r in D('main') if r['split'] == 'reference']
    TR_K = np.array(list(range(0, 300, 5)) + [299]) + 1
    TM = ['rdf_as_isac', 'projection', 'socp_radius_online', 'socp_online', 'fixed']
    fig, axs = plt.subplots(3, 1, figsize=(W1, 6.4), sharex=True); fig.subplots_adjust(hspace=.3)
    for ax, sc in zip(axs, 'DFG'):
        g = [r for r in ref if r['scenario'] == sc]
        for a in TM:
            v = np.array([r['traces'][a] for r in g], float).mean(0)[:, 0]
            ax.plot(TR_K, v, color=COL[a], marker=MK[a], markevery=8, ms=3.2, lw=1.1, label=LAB[a])
        v = np.array([r['traces']['true_cov'] for r in g], float).mean(0)[:, 0]
        ax.plot(TR_K[::5], v, color='k', marker=MK['true_cov'], ms=3.2, lw=1.1, label=LAB['true_cov'])
        ax.set_ylabel('User 1 SINR (dB)'); _ptitle(ax, f'({"abc"["DFG".index(sc)]}) Scenario {sc}')
    axs[0].set_ylim(-1, 27); axs[0].legend(loc='upper center', ncol=2)
    axs[2].set_xlabel('Observation')
    return _save(fig, outdir, 4)


def fig_7_5(D, outdir):
    """Mean normalized array response at elevation 2 deg in Scenarios D and F."""
    ref = [r for r in D('main') if r['split'] == 'reference']; GRID = np.linspace(-90, 90, 361)
    fig, axs = plt.subplots(2, 1, figsize=(W1, 5.0), sharex=True); fig.subplots_adjust(hspace=.3)
    for ax, sc in zip(axs, 'DF'):
        g = [r for r in ref if r['scenario'] == sc]
        keys = [k for k in ['rdf_as_isac', 'projection', 'socp_radius', 'socp_radius_online', 'true_cov'] if k in g[0]['patterns']]
        for a in keys:
            P = np.mean([r['patterns'][a] for r in g], 0)
            ax.plot(GRID, 10*np.log10(P/P.max()), color=COL[a], marker=MK[a], markevery=(keys.index(a)*6, 30), ms=3.0, lw=.9, label=LAB[a])
        ax.plot([20, -30], [1.5, 1.5], ls='none', marker='v', ms=5.5, color='#087d65', label='User direction', clip_on=False)
        ax.plot([-5, 40], [1.5, 1.5], ls='none', marker='v', ms=5.5, color='#843d95', mfc='none', label='Target direction', clip_on=False)
        if sc == 'F':
            ax.plot([34, -17, 50], [1.5]*3, ls='none', marker='*', ms=6.5, color='#c0392b', label='Interferer (Scenario F)', clip_on=False)
        ax.set_ylim(-108, 3); ax.set_xlim(-90, 90); ax.set_ylabel('Normalized response (dB)')
        _ptitle(ax, f'({"ab"["DF".index(sc)]}) Scenario {sc}')
        ax.legend(loc='lower center', ncol=2)
    axs[1].set_xlabel('Azimuth (deg), elevation 2°')
    return _save(fig, outdir, 5)


def fig_7_6(D, outdir):
    """Largest certified duality gap (7.24) per scene and final-refresh gap (7.28) to a tight conic solve."""
    main = D('main')
    fig, axs = plt.subplots(2, 1, figsize=(W1, 4.0), sharex=True); fig.subplots_adjust(hspace=.3)
    for k, sc in enumerate('ABCDFG'):
        g = [r for r in main if r['scenario'] == sc]
        x = np.full(len(g), k) + np.random.default_rng(k).uniform(-.18, .18, len(g))
        axs[0].scatter(x, [max(r['max_cert'], 1e-16) for r in g], s=8, color='#087d65', label='Certificate (7.24)' if k == 0 else None)
        axs[1].scatter(x, [max(abs(r['objective_gap']), 1e-16) for r in g], s=8, color='#507aa6', label='Matched gap (7.28)' if k == 0 else None)
    for ax, yl, ttl in zip(axs, ['Largest certified gap', 'Final gap to conic solve'], ['(a) Duality-gap certificate', '(b) Matched-problem gap']):
        ax.set_yscale('log'); ax.set_ylabel(yl); _ptitle(ax, ttl)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi*30); ax.legend(loc='upper left')
    axs[1].set_xticks(range(6)); axs[1].set_xticklabels(list('ABCDFG')); axs[1].set_xlabel('Scenario')
    return _save(fig, outdir, 6)


def fig_7_7(D, outdir):
    """Mean user SINR and target SNR versus the priority alpha in Scenario D."""
    al = D('alpha'); As = sorted({r['alpha'] for r in al})
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    ax.plot(As, [np.mean([r['metrics'][:2] for r in al if r['alpha'] == a]) for a in As], 'o-', color='#087d65', label='Mean user SINR')
    ax.plot(As, [np.mean([r['metrics'][2:] for r in al if r['alpha'] == a]) for a in As], 's-', color='#843d95', label='Mean target SNR')
    ax.set_xlabel('Priority α'); ax.set_ylabel('dB'); ax.set_ylim(2, 19); ax.legend(loc='upper center', ncol=2)
    return _save(fig, outdir, 7)


def fig_7_8(D, outdir):
    """Scenario E service-threshold coverage on the 7 x 7 angular-error grid."""
    gr = D('grid'); GM = ['rdf_as_isac', 'projection', 'socp', 'socp_radius', 'vu_mbwc']
    TT8 = {'rdf_as_isac': 'RDF-AS-ISAC', 'projection': 'Projection reconstruction', 'socp': 'SOCP, base', 'socp_radius': 'SOCP, matched', 'vu_mbwc': 'VU-MBWC'}
    fig, axs = plt.subplots(3, 2, figsize=(W1, 5.0)); fig.subplots_adjust(hspace=.55, wspace=.15)
    for k, (ax, a) in enumerate(zip(axs.ravel(), GM)):
        Z = np.zeros((7, 7))
        for c in range(7):
            for s in range(7):
                v = np.array([r['metrics'][a] for r in gr if r['c'] == c and r['s'] == s], float).mean(0)
                Z[s, c] = float(v[0] >= -3 and v[1] >= -3 and v[2] >= 3 and v[3] >= 3)
        ax.imshow(Z, origin='lower', cmap='Greens', vmin=-.3, vmax=1.1); ax.grid(False)
        ax.set_title(f'({"abcde"[k]}) {TT8[a]}: {int(Z.sum())}/49', fontsize=7.5, loc='left')
        ax.set_xticks([0, 3, 6]); ax.set_yticks([0, 3, 6])
        if k % 2 == 0: ax.set_ylabel('Target σ (deg)', fontsize=8)
        else: ax.set_yticklabels([])
        if k >= 3: ax.set_xlabel('User σ (deg)', fontsize=8)
        else: ax.set_xticklabels([])
    axs[2, 1].axis('off'); cm = plt.get_cmap('Greens')
    axs[2, 1].legend(handles=[Patch(facecolor=cm(1/1.4 + .07), edgecolor='k', label='Pass'),
                              Patch(facecolor=cm(.3/1.4), edgecolor='k', label='Fail')], loc='center', fontsize=8)
    return _save(fig, outdir, 8)


def fig_7_9(D, outdir):
    """Mean worst-target detection probability versus false-alarm probability, Scenario D, Swerling 1 targets."""
    pfa = np.logspace(-8, -2, 13); DM = ['rdf_as_isac', 'projection', 'socp_radius', 'socp_radius_online', 'fixed', 'true_cov']
    rowsD = [r for r in D('main') if r['scenario'] == 'D']
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    for a in DM:
        G = 10**(np.array([r['metrics'][a][2:] for r in rowsD], float)/10)
        y = [np.mean([min(pd_swerling(g, -2*np.log(p), 1) for g in row) for row in G]) for p in pfa]
        ax.semilogx(pfa, y, color=COL[a], marker=MK[a], ms=3.2, lw=1.1, label=LAB[a])
    ax.set_ylim(0, 1.15); ax.set_ylabel('Worst-target $P_d$'); ax.set_xlabel('False-alarm probability')
    ax.legend(loc='upper left', ncol=2)
    return _save(fig, outdir, 9)


ALL = {k: globals()[f'fig_7_{k}'] for k in range(1, 10)}
NEEDS = {1: ['main'], 2: ['sweep'], 3: ['cross'], 4: ['main'], 5: ['main'], 6: ['main'], 7: ['alpha'], 8: ['grid'], 9: ['main']}


def draw(datadir, outdir, which=None):
    D = Data(datadir); done = []
    for k in (which or range(1, 10)):
        if all((Path(datadir)/f'{s}.json').exists() for s in NEEDS[k]):
            done.append(ALL[k](D, outdir))
    return done
