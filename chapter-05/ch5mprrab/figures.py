"""
ch5mprrab.figures -- Figures 5.1 to 5.10 of Chapter 5 and the results summary, drawn from the data that the
experiment stages write into FIGDIR/data.

Configured through environment variables read at import time:
  FIGDIR=path         output folder (default ./figures); the simulation data are read from FIGDIR/data
  CH5_LOCAL_TIMING=1  draw Figure 5.10 from the timing measured on this machine (data/timing.json) instead of the
                      timing recorded for the chapter (ch5mprrab/data/timing_recorded.json, Table 5.1)
  CH5_INLINE=1        also show each figure under its cell in Jupyter
"""
import json
import os
import numpy as np
import matplotlib
if os.environ.get('CH5_INLINE') != '1':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator

OUT = os.environ.get('FIGDIR', 'figures')
DATA = os.path.join(OUT, 'data')
INLINE = os.environ.get('CH5_INLINE') == '1'
LOCAL_TIMING = os.environ.get('CH5_LOCAL_TIMING') == '1'
PKG_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULTS = {}

plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'axes.linewidth': 0.9, 'axes.spines.top': True, 'axes.spines.right': True,
    'lines.linewidth': 1.3, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
})
LEG = dict(frameon=True, fancybox=False, framealpha=0.95, edgecolor='#444444', fontsize=7.8,
           borderpad=0.45, labelspacing=0.3, columnspacing=1.0, handlelength=2.4)

# method -> (colour, line width, marker): the visual identity of the chapter figures
STY = {'MPR-RAB': ('#2a78d6', 2.2, 'o'), 'MPR-RAB, equal targets': ('#6250d6', 1.4, 's'),
       'MPR-RAB, no delay compensation': ('#7fb3e8', 1.5, 'v'), 'MPR-RAB, refinement every 30': ('#9a9a9a', 1.5, 'X'),
       'INCM-LCMV': ('#eb6834', 1.3, '^'), 'INCM-MVDR': ('#8a6d00', 1.3, 'D'), 'INCM-GEV': ('#c2185b', 1.3, '*'),
       'Par-K-RCB': ('#1baf7a', 1.3, 'P'), 'LCMV-K': ('#e34948', 1.3, '<'), 'LCMV-K-D': ('#b05fa0', 1.2, '>'),
       'Oracle': ('#000000', 1.1, 'h')}
MSIZE = {'*': 8.0, 'P': 6.5, 'X': 6.5}
MAIN = ['MPR-RAB', 'INCM-LCMV', 'INCM-MVDR', 'INCM-GEV', 'Par-K-RCB', 'LCMV-K', 'Oracle']
CONDS = ['5_10_0', '5_10_2', '5_10_4', '10_20_0', '20_10_0', '5_35_0']
GEOS = list('ABCDEFGHI')
# columns of the result tables (Tables 5.4-5.7, 5.10, 5.11): table label -> family of the selection
COLS = [('LCMV-K', 'LCMV-K noise-floor-loaded'), ('Par-K-RCB', 'Par-K-RCB'), ('LCMV-K-D', 'LCMV-K-D'),
        ('INCM-LCMV', 'INCM-LCMV refined'), ('INCM-MVDR', 'INCM-MVDR refined'), ('INCM-GEV', 'INCM-GEV refined'),
        ('MPR-RAB, beta_k = 1', 'MPR-RAB equal targets'), ('MPR-RAB', 'MPR-RAB'), ('Oracle', 'Oracle')]


# --------------------------------------------------------------------------- helpers
def J(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def selection(arr):
    fn = os.path.join(DATA, 'selection_%s.json' % arr)
    if os.path.exists(fn):
        return json.load(open(fn))['best']
    return json.load(open(os.path.join(PKG_DATA, 'selection.json')))[arr]['best']


def names(arr):
    """Legend name -> result name of the selected setting (the reconstruction baselines with the peak refinement)."""
    sel = selection(arr)
    inc = lambda m: sel.get('INCM-%s refined' % m, sel['INCM-%s' % m])
    return {'MPR-RAB': sel['MPR-RAB'], 'MPR-RAB, equal targets': sel['MPR-RAB equal targets'],
            'MPR-RAB, no delay compensation': sel['MPR-RAB no delay compensation'],
            'INCM-LCMV': inc('LCMV'), 'INCM-MVDR': inc('MVDR'), 'INCM-GEV': inc('GEV'), 'Par-K-RCB': sel['Par-K-RCB'],
            'LCMV-K': sel['LCMV-K noise-floor-loaded'], 'Oracle': 'Oracle', 'LCMV-K-D': sel['LCMV-K-D']}


def kappa_of(n):
    return float(n.split('k=')[1].split()[0])


def load(arr, phase):
    R = {}
    for g in GEOS:
        fn = os.path.join(DATA, '%s_%s_%s.json' % (arr, phase, g))
        if os.path.exists(fn):
            R[g] = json.load(open(fn))
    return R


def pooled(R, n, suf='', first=None):
    """Mean over all cells (scenario x condition) of the mean over runs; first=N uses the first N runs of a cell."""
    return float(np.mean([np.mean(R[g][c][n + suf][:first]) for g in R for c in R[g]]))


def boot(dl, B=3000, seed=0):
    """Paired mean over cells weighted equally and its 95 % percentile bootstrap interval (resampling the runs within
    every cell)."""
    rng = np.random.RandomState(seed)
    m = np.mean([np.mean(d) for d in dl])
    bs = [np.mean([np.mean(d[rng.randint(0, len(d), len(d))]) for d in dl]) for _ in range(B)]
    return float(m), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def r2(v):
    return round(float(v), 2)


def line(ax, x, y, m, i=0, n_mark=10, label=None, **kw):
    """Curve of method m with its colour, width and marker; markers are staggered between methods."""
    col, lw, mk = STY[m]
    step = max(1, len(x) // n_mark)
    return ax.plot(x, y, color=col, ls='-', lw=lw, marker=mk, ms=MSIZE.get(mk, 5.0), mew=0.8, mec=col,
                   mfc='white' if m == 'Oracle' else col, markevery=((3 * i) % step, step),
                   label=m if label is None else label, **kw)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    if INLINE:
        plt.show()
    plt.close(fig)


def traces(arr, phase, sc):
    return np.load(os.path.join(DATA, '%s_%s_%s_traces.npz' % (arr, phase, sc)))


def trace_fig(sc, name, methods, xlim=None, ylim=None, shade=None, loc='lower right', ncol=2):
    NU = names('upa')
    tr = traces('upa', 'heldout', sc)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    if shade:
        ax.axvspan(shade[0], shade[1], color='#dddddd', alpha=0.6, lw=0, zorder=0)
    out = {}
    for i, m in enumerate(methods):
        y = np.asarray(tr[NU[m]])
        x = np.arange(1, len(y) + 1)
        out[m] = [r2(v) for v in y]
        if xlim:
            k = (x >= xlim[0]) & (x <= xlim[1])
            x, y = x[k], y[k]
        line(ax, x, y, m, i, zorder=3 if m == 'MPR-RAB' else 2)
    ax.set_xlabel('Snapshot'); ax.set_ylabel('Expected SINR (dB)')
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc=loc, ncol=ncol, **LEG)
    save(fig, name)
    return out


# --------------------------------------------------------------------------- Figures 5.1, 5.3 and 5.6
def fig_5_1():
    """Planar array, scenario A: expected SINR per snapshot (operating point, 30 runs)."""
    trace_fig('A', 'Fig5_01_planar_scenario_A', MAIN, ylim=(4, 25.5))
    RESULTS['Figure 5.1'] = {'note': 'mean traces in data/upa_heldout_A_traces.npz'}


def fig_5_3():
    """Planar array, scenario D: cluster birth at snapshot 150 and death at snapshot 280."""
    trace_fig('D', 'Fig5_03_planar_scenario_D', MAIN, ylim=(2, 25.5), loc='lower center')
    RESULTS['Figure 5.3'] = {'note': 'mean traces in data/upa_heldout_D_traces.npz'}


def fig_5_6():
    """Planar array, scenario I: expected SINR per snapshot; the strongest cluster is shadowed on [150, 250)."""
    trace_fig('I', 'Fig5_06_planar_scenario_I_shadowing',
              ['MPR-RAB', 'MPR-RAB, equal targets', 'INCM-LCMV', 'INCM-MVDR', 'INCM-GEV', 'Par-K-RCB', 'Oracle'],
              xlim=(100, 320), ylim=(0, 25), shade=(150, 250), loc='lower right', ncol=2)
    RESULTS['Figure 5.6'] = {'note': 'mean traces in data/upa_heldout_I_traces.npz'}


# --------------------------------------------------------------------------- Figure 5.2
def fig_5_2():
    """Planar array, scenario C: expected SINR (top) and mean angular error of the tracked directions (bottom)."""
    NU = names('upa')
    tr = traces('upa', 'heldout', 'C')
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    for i, m in enumerate(MAIN):
        y = tr[NU[m]]
        line(axs[0], np.arange(1, len(y) + 1), y, m, i, zorder=3 if m == 'MPR-RAB' else 2)
    e = tr['_trkerr']
    line(axs[1], np.arange(1, len(e) + 1), e, 'MPR-RAB', 0, label='MPR-RAB, mean direction error')
    axs[0].set_ylim(0, 25.5)
    axs[0].set_ylabel('Expected SINR (dB)'); axs[1].set_ylabel('Error (deg)'); axs[1].set_xlabel('Snapshot')
    axs[0].legend(loc='lower right', ncol=2, **LEG)
    axs[1].legend(loc='upper right', **LEG)
    fig.align_ylabels(axs)
    fig.tight_layout(h_pad=0.4)
    save(fig, 'Fig5_02_planar_scenario_C')
    RESULTS['Figure 5.2'] = {'mean direction error, last 300 snapshots (deg)': r2(np.nanmean(e[100:]))}


# --------------------------------------------------------------------------- Figure 5.4
RULES = (('MPR k=%g', '#2a78d6', 'o', 'MPR-RAB, max-SINR targets'), ('MPR k=%g q=0', '#6250d6', 's', 'MPR-RAB, equal targets'),
         ('MPR k=%g q=1', '#1baf7a', 'p', 'MPR-RAB, power targets'))
KAPPAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def fig_5_4():
    """Planar array: expected against minimum-cluster SINR over the condition study; the circle marks the selected
    MPR-RAB setting."""
    R = load('upa', 'heldout')
    NU = names('upa')
    sel = selection('upa')
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    axi = ax.inset_axes([0.50, 0.07, 0.47, 0.36])
    xa, ya, res = [], [], {}
    for fmt, col, mk, lab in RULES:
        xs = [pooled(R, fmt % k, '|floor') for k in KAPPAS]
        ys = [pooled(R, fmt % k) for k in KAPPAS]
        res[lab] = {'kappa': KAPPAS, 'expected': [r2(v) for v in ys], 'min-cluster': [r2(v) for v in xs]}
        xa += xs; ya += ys
        for a in (ax, axi):
            a.plot(xs, ys, '-', color=col, marker=mk, ms=4.2 if a is axi else 5.2, lw=1.3, label=lab if a is ax else None, zorder=3)
        if fmt == 'MPR k=%g':
            axi.annotate(r'$\kappa = 0$', (xs[0], ys[0]), textcoords='offset points', xytext=(3, -10), fontsize=7.5, ha='left', color=col)
            axi.annotate(r'$\kappa = 0.9$', (xs[-1], ys[-1]), textcoords='offset points', xytext=(-4, -3), fontsize=7.5, ha='right',
                         va='top', color=col)
    xsel, ysel = pooled(R, sel['MPR-RAB'], '|floor'), pooled(R, sel['MPR-RAB'])
    for a in (ax, axi):
        a.plot([xsel], [ysel], 'o', ms=11 if a is ax else 9, mfc='none', mec='#2a78d6', mew=1.4, zorder=5)
    for lab in ('LCMV-K', 'Par-K-RCB', 'LCMV-K-D', 'INCM-LCMV', 'INCM-MVDR', 'INCM-GEV', 'Oracle'):
        n = NU[lab]
        col, _, mk = STY[lab]
        x, y = pooled(R, n, '|floor'), pooled(R, n)
        res[lab] = {'expected': r2(y), 'min-cluster': r2(x)}
        ax.plot([x], [y], mk, color=col, ms=8 if mk != '*' else 11, mfc='white' if lab == 'Oracle' else col, mew=1.0, ls='none',
                label=lab, zorder=4)
        if lab.startswith('INCM'):
            axi.plot([x], [y], mk, color=col, ms=7 if mk != '*' else 9, ls='none', zorder=4)
            xa.append(x); ya.append(y)
    axi.set_xlim(min(xa) - 0.8, max(xa) + 0.8); axi.set_ylim(min(ya) - 0.12, max(ya) + 0.12)
    axi.tick_params(labelsize=7.5); axi.grid(alpha=0.25)
    ax.indicate_inset_zoom(axi, edgecolor='#999999', alpha=0.8)
    ax.set_xlabel('Minimum-cluster SINR (dB)'); ax.set_ylabel('Expected SINR (dB)')
    lo = min(min(ya), min(res[l]['expected'] for l in ('LCMV-K', 'Par-K-RCB', 'LCMV-K-D')))
    ax.set_ylim(lo - 14.0, max(ya + [res['Oracle']['expected']]) + 1.0)
    ax.legend(loc='lower left', ncol=1, **{**LEG, 'fontsize': 7.4})
    save(fig, 'Fig5_04_expected_vs_min_cluster')
    res['selected MPR-RAB'] = {'kappa': kappa_of(sel['MPR-RAB']), 'expected': r2(ysel), 'min-cluster': r2(xsel)}
    RESULTS['Figure 5.4'] = res


# --------------------------------------------------------------------------- Figure 5.5
def fig_5_5():
    """Planar array, scenario H: normalized gain of the final MPR-RAB weight."""
    P = np.load(os.path.join(DATA, 'pattern_H.npz'))
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    im = ax.imshow(P['G'], origin='lower', extent=[-90, 90, -60, 60], aspect='auto', cmap='Blues_r', vmin=-50, vmax=0)
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label('Normalized gain (dB)')
    ax.plot(P['clusters'][:, 0], P['clusters'][:, 1], 'o', ms=8, mfc='none', mec='#eb6834', mew=1.6, ls='none', label='clusters')
    ax.plot(P['interferers'][:, 0], P['interferers'][:, 1], 'x', ms=9, color='#e34948', mew=2.0, ls='none', label='interferers')
    ax.set_xlabel('Azimuth (deg)'); ax.set_ylabel('Elevation (deg)'); ax.grid(False)
    ax.legend(loc='lower left', **LEG)
    save(fig, 'Fig5_05_beam_pattern_scenario_H')
    RESULTS['Figure 5.5'] = {k: r2(v) for k, v in J('pattern_H.json')['H_gains_dB'].items()}


# --------------------------------------------------------------------------- Figure 5.7
def fig_5_7():
    """Linear array, scenario C: expected SINR (top) and an enlarged view of the top of the traces (bottom)."""
    NL = names('ula')
    tr = traces('ula', 'stream', 'C')
    tc = J('tracking_c.json')
    extra = {'MPR-RAB, refinement every 30': tc['sinr_trace']['every30']}
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
    order = ['MPR-RAB', 'INCM-LCMV', 'INCM-MVDR', 'INCM-GEV', 'Par-K-RCB', 'Oracle', 'MPR-RAB, no delay compensation',
             'MPR-RAB, refinement every 30']
    for m in ['MPR-RAB', 'INCM-LCMV', 'INCM-MVDR', 'INCM-GEV', 'Par-K-RCB', 'Oracle']:
        y = tr[NL[m]]
        line(axs[0], np.arange(1, len(y) + 1), y, m, order.index(m), zorder=3 if m == 'MPR-RAB' else 2)
    for m in ['MPR-RAB', 'MPR-RAB, no delay compensation', 'MPR-RAB, refinement every 30', 'INCM-MVDR', 'Oracle']:
        y = np.asarray(extra[m]) if m in extra else tr[NL[m]]
        line(axs[1], np.arange(1, len(y) + 1), y, m, order.index(m), zorder=3 if m == 'MPR-RAB' else 2)
    axs[0].set_ylim(0, 19.5); axs[1].set_ylim(15.7, 17.2)
    axs[0].set_ylabel('Expected SINR (dB)'); axs[1].set_ylabel('Zoom (dB)'); axs[1].set_xlabel('Snapshot')
    hl = {}
    for a in axs:
        for h_, l_ in zip(*a.get_legend_handles_labels()):
            hl.setdefault(l_, h_)
    axs[0].legend([hl[m] for m in order], order, loc='lower right', ncol=2, **{**LEG, 'fontsize': 7.4})
    fig.align_ylabels(axs)
    fig.tight_layout(h_pad=0.4)
    save(fig, 'Fig5_07_linear_scenario_C')
    RESULTS['Figure 5.7'] = {
        'mean |direction error| of the constraints, snapshots 100-400 (deg)': {k: round(v, 2) for k, v in tc['mean_abs_err_100_400'].items()},
        'mean expected SINR, snapshots 100-400 (dB)': {k: r2(v) for k, v in tc['sinr_mean_100_400'].items()},
        'largest step between snapshots after 60 (dB)': {k: r2(v) for k, v in tc['largest_step_after_60'].items()},
        'keys': {'pred': 'MPR-RAB', 'nopred': 'MPR-RAB, no delay compensation', 'every30': 'MPR-RAB, refinement every 30'}}


# --------------------------------------------------------------------------- Figure 5.8
def fig_5_8():
    """Planar array, training study: expected (left) and minimum-cluster (right) SINR against kappa; the vertical line
    marks the selected kappa."""
    Rt = load('upa', 'train')
    NU = names('upa')
    sel = selection('upa')
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.9), sharex=True)
    res = {}
    for fmt, col, mk, lab in RULES:
        lab = lab.replace('MPR-RAB, ', '')
        ye = [pooled(Rt, fmt % k) for k in KAPPAS]
        yf = [pooled(Rt, fmt % k, '|floor') for k in KAPPAS]
        res[lab] = {'kappa': KAPPAS, 'expected': [r2(v) for v in ye], 'min-cluster': [r2(v) for v in yf]}
        axs[0].plot(KAPPAS, ye, '-', marker=mk, color=col, ms=5, label=lab)
        axs[1].plot(KAPPAS, yf, '-', marker=mk, color=col, ms=5, label=lab)
    for i, f in enumerate(('INCM-LCMV', 'INCM-MVDR', 'INCM-GEV')):
        n = NU[f]
        col, _, mk = STY[f]
        xk = np.linspace(min(KAPPAS), max(KAPPAS), 19)
        res[f] = {'expected': r2(pooled(Rt, n)), 'min-cluster': r2(pooled(Rt, n, '|floor'))}
        for ax, suf in ((axs[0], ''), (axs[1], '|floor')):
            ax.plot(xk, np.full(len(xk), pooled(Rt, n, suf)), color=col, lw=1.2, marker=mk, ms=MSIZE.get(mk, 5.0),
                    mew=0.8, markevery=(2 + 2 * i, 6), label=f)
    k_sel = kappa_of(sel['MPR-RAB'])
    for ax in axs:
        ax.axvline(k_sel, color='#999999', lw=0.8, zorder=0)
        ax.set_xlabel(r'$\kappa$')
        ax.set_xticks([0, 0.3, 0.6, 0.9])
    lo = min(min(res[l]['expected']) for l in ('max-SINR targets', 'equal targets', 'power targets'))
    axs[0].set_ylim(lo - 2.6, max(max(res[l]['expected']) for l in res if isinstance(res[l]['expected'], list)) + 0.4)
    axs[0].set_ylabel('Expected SINR (dB)'); axs[1].set_ylabel('Minimum-cluster SINR (dB)')
    axs[0].legend(loc='lower left', ncol=2, **{**LEG, 'fontsize': 7.0, 'handlelength': 2.0, 'columnspacing': 0.8})
    fig.tight_layout(w_pad=1.0)
    save(fig, 'Fig5_08_kappa_training_study')
    res['selected kappa'] = k_sel
    RESULTS['Figure 5.8'] = res


# --------------------------------------------------------------------------- Figure 5.9
def fig_5_9():
    """Linear array: an interferer crossing a tracked cluster at snapshot 150, with and without deflation."""
    cr = J('sensitivity.json')['cross']
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    t = np.arange(1, 301)
    res = {}
    for i, (key, col, lab, mk) in enumerate((('tau=0.3', '#2a78d6', 'MPR-RAB, deflation on', 'o'),
                                             ('tau=0', '#e34948', 'MPR-RAB, deflation off', 's'))):
        kw = dict(color=col, ls='-', marker=mk, ms=5, mew=0.8, markevery=(10 * i, 30))
        s = np.mean([np.array(r['sinr']) for r in cr[key]], axis=0)
        e = np.mean([np.array(r['err']) for r in cr[key]], axis=0)
        axs[0].plot(t, s, label=lab, **kw)
        axs[1].plot(t, e, label=lab, **kw)
        res[lab] = {'min expected SINR (dB)': r2(s.min()), 'largest mean direction error (deg)': r2(e.max()),
                    'mean error, last 50 (deg)': r2(e[-50:].mean()), 'mean expected SINR, last 50 (dB)': r2(s[-50:].mean())}
    o = np.mean([np.array(r['sinr']) for r in cr['oracle']], axis=0)
    axs[0].plot(t, o, color='k', lw=1.1, marker='h', ms=5, mew=0.8, mfc='white', markevery=(20, 30), label='Oracle')
    res['Oracle'] = {'min expected SINR (dB)': r2(o.min()), 'mean expected SINR, last 50 (dB)': r2(o[-50:].mean())}
    for k in ('MPR-RAB, deflation on', 'MPR-RAB, deflation off'):
        res[k]['below the oracle, last 50 (dB)'] = r2(res['Oracle']['mean expected SINR, last 50 (dB)'] - res[k]['mean expected SINR, last 50 (dB)'])
    axs[0].axvline(150, color='#999999', lw=0.8); axs[1].axvline(150, color='#999999', lw=0.8)
    axs[0].set_ylim(min(-6, float(np.floor(min(r['min expected SINR (dB)'] for r in res.values())))) - 9, 21)
    axs[0].set_ylabel('Expected SINR (dB)'); axs[1].set_ylabel('Error (deg)'); axs[1].set_xlabel('Snapshot')
    axs[0].legend(loc='lower left', **LEG)
    fig.align_ylabels(axs)
    fig.tight_layout(h_pad=0.4)
    save(fig, 'Fig5_09_interferer_crossing')
    RESULTS['Figure 5.9'] = res


# --------------------------------------------------------------------------- Figure 5.10
TIMING_STY = {'MPR-RAB (T_rec = 10)': ('#2a78d6', 'o', 'MPR-RAB'), 'INCM-LCMV (every 10)': ('#eb6834', '^', 'INCM-LCMV'),
              'INCM-GEV (every 10)': ('#c2185b', '*', 'INCM-GEV'),
              'LCMV-K (every snapshot)': ('#e34948', '<', 'LCMV-K, every snapshot'), 'LCMV-K (every 10)': ('#e34948', 'd', 'LCMV-K, every 10'),
              'Par-K-RCB (every snapshot)': ('#1baf7a', 'P', 'Par-K-RCB, every snapshot'),
              'Par-K-RCB (every 10)': ('#1baf7a', 'X', 'Par-K-RCB, every 10'), 'LCMV-K-D (every snapshot)': ('#b05fa0', '>', 'LCMV-K-D')}


def fig_5_10():
    """Linear arrays: measured time per snapshot against array size, one core, median of 5 repetitions (the timing
    recorded for the chapter unless CH5_LOCAL_TIMING=1)."""
    tm = J('timing.json') if LOCAL_TIMING else json.load(open(os.path.join(PKG_DATA, 'timing_recorded.json')))
    sizes = tm['sizes']
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for n, (col, mk, lab) in TIMING_STY.items():
        ax.loglog(sizes, [tm['ms'][str(M)][n] for M in sizes], color=col, ls='-', marker=mk, ms=MSIZE.get(mk, 5.5), mew=0.8,
                  mec=col, mfc=col, label=lab)
    ax.set_xlabel('Array size $M$'); ax.set_ylabel('Time per snapshot (ms)')
    ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(top=ax.get_ylim()[1] * 300)
    ax.legend(loc='upper left', ncol=2, **{**LEG, 'fontsize': 7.4})
    save(fig, 'Fig5_10_time_per_snapshot')
    RESULTS['Figure 5.10'] = {'source': 'this machine' if LOCAL_TIMING else 'timing recorded for the chapter (Table 5.1)',
                              'sizes': sizes, 'ms per snapshot': {TIMING_STY[n][2]: [tm['ms'][str(M)][n] for M in sizes] for n in TIMING_STY}}
    if 'platform' in tm:
        RESULTS['Figure 5.10']['platform'] = {k: v for k, v in tm['platform'].items() if k != 'blas'}


# --------------------------------------------------------------------------- tables and text numbers
def _cols(arr):
    sel = selection(arr)
    return [(lab, sel.get(f, f)) for lab, f in COLS]


def _table_scen(R, cols, suf=''):
    return {g: {lab: r2(np.mean(R[g]['5_10_0'][n + suf])) for lab, n in cols} for g in R if '5_10_0' in R[g]}


def _paired(R, cols, mp, suf=''):
    out = {}
    for lab, n in cols:
        if n == mp:
            continue
        per = [np.array(R[g][c][mp + suf]) - np.array(R[g][c][n + suf]) for g in R for c in R[g]]
        m, lo, hi = boot(per)
        out[lab] = [round(m, 2), round(lo, 2), round(hi, 2)]
    return out


def _tables_planar():
    R = load('upa', 'heldout')
    if len(R) < 9:
        return
    sel = selection('upa')
    cols = _cols('upa')
    mp = sel['MPR-RAB']
    RESULTS['Table 5.4 (planar, operating point, expected SINR)'] = _table_scen(R, cols)
    t5 = {c.replace('_', '/'): {lab: r2(np.mean([np.mean(R[g][c][n]) for g in R])) for lab, n in cols} for c in CONDS}
    t5['all'] = {lab: r2(pooled(R, n)) for lab, n in cols}
    t5['min.'] = {lab: r2(pooled(R, n, '|floor')) for lab, n in cols}
    RESULTS['Table 5.5 (planar, expected SINR per condition)'] = t5
    t6 = {'expected, all cells': _paired(R, cols, mp), 'min-cluster, all cells': _paired(R, cols, mp, '|floor'),
          'expected, per condition': {lab: {c.replace('_', '/'): r2(np.mean([np.mean(np.array(R[g][c][mp]) - np.array(R[g][c][n]))
                                                                               for g in R])) for c in CONDS}
                                      for lab, n in cols if n != mp}}
    RESULTS['Table 5.6 (planar, paired difference MPR-RAB minus comparator)'] = t6
    RESULTS['Table 5.7 (planar, operating point, minimum-cluster SINR)'] = _table_scen(R, cols, '|floor')
    nab = 10
    rows = [('MPR-RAB', sel['MPR-RAB']), ('without margins', 'MPR k=0'),
            ('equal targets, beta_k = 1', sel['MPR-RAB equal targets']), ('power-proportional targets', sel['MPR-RAB power targets']),
            ('without delay compensation', sel['MPR-RAB no delay compensation']),
            ('targets computed on R_in instead of R_t', sel['MPR-RAB targets from the reconstruction']),
            ('without off-grid peak refinement', sel['MPR-RAB no peak refinement']),
            ('without reconstruction (sample covariance)', sel['MPR-RAB no reconstruction']),
            ('unit targets without margins', 'MPR k=0 q=0')]
    RESULTS['Table 5.8 (planar ablations, first 10 runs of every cell)'] = {
        '%s (%s)' % (lab, n): {'expected': r2(pooled(R, n, '', nab)), 'min-cluster': r2(pooled(R, n, '|floor', nab))} for lab, n in rows}
    r = R['I']['5_10_0']
    RESULTS['Table 5.9 (planar, scenario I)'] = {
        lab: {'last 50': r2(np.mean(r[n])), 'first 10 shadowed': r2(np.mean(r[n + '|dip'])), 'shadowed': r2(np.mean(r[n + '|blk'])),
              'first 10 after': r2(np.mean(r[n + '|rec'])), 'min-cluster': r2(np.mean(r[n + '|floor']))} for lab, n in cols}
    upd = sum(np.sum(R[g][c][mp + '|solver_updates']) for g in R for c in R[g])
    unc = sum(np.sum(R[g][c][mp + '|solver_uncertified']) for g in R for c in R[g])
    RESULTS['Solver certification, planar condition study (selected MPR-RAB)'] = {'updates': int(upd), 'uncertified': int(unc)}


def _tables_linear():
    R = load('ula', 'heldout')
    if len(R) < 9:
        return
    sel = selection('ula')
    cols = _cols('ula')
    mp = sel['MPR-RAB']
    RESULTS['Table 5.10 (linear, operating point, expected SINR)'] = _table_scen(R, cols)
    RESULTS['Table 5.11 (linear, paired difference MPR-RAB minus comparator)'] = {
        'expected, all cells': _paired(R, cols, mp), 'min-cluster, all cells': _paired(R, cols, mp, '|floor')}
    RESULTS['Linear condition study, pooled expected / minimum-cluster SINR'] = {
        lab: [r2(pooled(R, n)), r2(pooled(R, n, '|floor'))] for lab, n in cols}
    RESULTS['Linear condition study, scenario C pooled expected SINR'] = {
        lab: r2(np.mean([np.mean(R['C'][c][n]) for c in R['C']])) for lab, n in cols}
    upd = sum(np.sum(R[g][c][mp + '|solver_updates']) for g in R for c in R[g])
    unc = sum(np.sum(R[g][c][mp + '|solver_uncertified']) for g in R for c in R[g])
    RESULTS['Solver certification, linear condition study (selected MPR-RAB)'] = {'updates': int(upd), 'uncertified': int(unc)}


def _sensitivity():
    fn = os.path.join(DATA, 'sensitivity.json')
    if not os.path.exists(fn):
        return
    S = json.load(open(fn))
    pool = lambda d: [r2(np.mean([x[0] for sc in d for x in d[sc]])), r2(np.mean([x[1] for sc in d for x in d[sc]]))]
    RESULTS['Table 5.12 (linear, sensitivity: expected / minimum-cluster SINR)'] = {
        st: {s: pool(S[st][s]) for s in S[st]} for st in ('sector', 'eta', 'T_rec', 'window') if st in S}
    if 'deflation' in S:
        RESULTS['Deflation at 35 dB INR (C, E, H, I)'] = {
            'tau=%s' % t: {'pooled': pool(S['deflation'][t])[0],
                           **{sc: r2(np.mean([x[0] for x in S['deflation'][t][sc]])) for sc in S['deflation'][t]}}
            for t in S['deflation']}
    if 'notrack' in S:
        RESULTS['Tracking off, static scenarios A, B, F, G, H (expected / minimum-cluster SINR)'] = {
            k: pool(v) for k, v in S['notrack'].items()}
    if 'offgrid' in S:
        RESULTS['Interferers 0.25 deg off the reconstruction grid (expected / minimum-cluster SINR)'] = {
            k: {n: [r2(x) for x in v] for n, v in d.items()} for k, d in S['offgrid'].items()}


def _checks():
    out = {}
    for name, key in (('prop1.json', 'Proposition 5.1: oracle excess over the strongest whitened cluster (dB)'),
                      ('prop2.json', 'Proposition 5.2: zero-margin solution against the generalized eigenvector'),
                      ('feasibility.json', 'Joint feasibility nu^T G^-1 nu at kappa = 1'),
                      ('geometry.json', 'Planar Gram-matrix condition numbers')):
        fn = os.path.join(DATA, name)
        if not os.path.exists(fn):
            continue
        d = json.load(open(fn))
        if name == 'prop1.json':
            out[key] = {'max': round(d['max'], 3), 'per geometry': {k: round(v, 3) for k, v in d['excess_dB'].items()}}
        elif name == 'feasibility.json':
            out[key] = {'min': round(d['min'], 4), 'max': round(d['max'], 4), 'random phase vectors per geometry': d['n_phase']}
        elif name == 'geometry.json':
            out[key] = {'min': round(d['minimum'], 3), 'max': round(d['maximum'], 3)}
        else:
            out[key] = d
    for arr in ('ula', 'upa'):
        fn = os.path.join(DATA, 'solver_%s.json' % arr)
        if os.path.exists(fn):
            d = json.load(open(fn))
            out['Active-set solver, %s array' % {'ula': 'linear', 'upa': 'planar'}[arr]] = {
                'kappa %s' % k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in d['stats'].items()}
    for arr in ('upa', 'ula'):
        fn = os.path.join(DATA, 'selection_derived_%s.json' % arr)
        if os.path.exists(fn):
            d = json.load(open(fn))
            out['Selection derived from the %s training phase' % {'ula': 'linear', 'upa': 'planar'}[arr]] = {
                'matches the selection of the chapter': d['matches_chapter'], 'differences': d['differences']}
    RESULTS['Checks'] = out


def _dumps(obj):
    """JSON with every list of numbers on one line."""
    import re
    txt = json.dumps(obj, indent=1, ensure_ascii=False)
    return re.sub(r'\[\s*([-0-9.eE+,\s]+?)\s*\]', lambda m: '[' + ', '.join(x.strip() for x in m.group(1).split(',')) + ']', txt)


def write_summary():
    """Collects the tables and the quoted numbers of the chapter into OUT/results_summary.json."""
    for f in (_tables_planar, _tables_linear, _sensitivity, _checks):
        f()
    path = os.path.join(OUT, 'results_summary.json')
    with open(path, 'w') as fh:
        fh.write(_dumps(RESULTS) + '\n')
    return path


def show(key):
    """Print the numbers collected for a figure or table."""
    for k in RESULTS:
        if k == key or k.startswith(key + ' '):
            print(k)
            print(_dumps(RESULTS[k]))
