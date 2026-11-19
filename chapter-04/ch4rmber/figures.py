"""
ch4rmber.figures -- Figures 4.1 to 4.11 of Chapter 4 and the results summary, drawn from the data that the experiments
of ch4rmber.scenarios write into FIGDIR/data.

Configured through environment variables read at import time:
  FIGDIR=path        output folder (default ./figures); the simulation data are read from FIGDIR/data
  RAB_LOCAL_TIMING=1 draw Figure 4.1 from the runtime measured on this machine (res_main_time.json) instead of the
                     recorded run shipped with the package (ch4rmber/data/res_main_time_recorded.json)
  RAB_INLINE=1       also show each figure under its cell in Jupyter
"""
import json
import os
import numpy as np
from scipy.special import erfc
import matplotlib
if os.environ.get('RAB_INLINE') != '1':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.environ.get('FIGDIR', 'figures')
DATA = os.path.join(OUT, 'data')
INLINE = os.environ.get('RAB_INLINE') == '1'
LOCAL_TIMING = os.environ.get('RAB_LOCAL_TIMING') == '1'
PKG_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULTS = {}

plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True,
    'axes.linewidth': 0.9, 'axes.spines.top': True, 'axes.spines.right': True,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
})
LEG = dict(frameon=True, fancybox=False, framealpha=0.95, edgecolor='#444444', fontsize=7.6,
           borderpad=0.45, labelspacing=0.3, columnspacing=1.0, handlelength=2.4)

# method -> (legend name, colour, marker, line style, line width): the visual identity of the chapter figures
ST = {'gm': ('Guided R-MBER (blind)', '#d62728', 'P', '-', 2.2), 'g': ('Guide alone (blind)', '#e377c2', '^', '--', 1.2),
      'incm': ('INCM-REC', '#8c564b', 'X', ':', 1.5), 'sp': ('SPSS', '#7f7f7f', 'v', '--', 1.3),
      'wc': ('WC-MVDR/EigDec', '#ff7f0e', 'o', '-.', 1.3), 'bl': ('Blind R-MBER, full array', '#9467bd', 's', '-', 1.3),
      'tr': ('Trained R-MBER', '#1f77b4', 'd', '-.', 1.3), 'ta': ('TA-MMSE, every symbol known (trained)', '#2ca02c', '*', '-', 1.3),
      'ta30': ('TA-MMSE, 30 pilots (trained)', '#2ca02c', '*', (0, (1, 1.5)), 1.4),
      'tadd': ('TA-MMSE, 30 pilots + own decisions', '#17becf', 'h', '-.', 1.3), 'opt': ('Optimal MVDR', 'k', '', '--', 1.0)}
ORDER = [ST[k][0] for k in ST]
LO = -20


def J(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def line(ax, x, y, k, semilogy=False, **kw):
    n, c, m, l, lw = ST[k]
    f = ax.semilogy if semilogy else ax.plot
    if k == 'ta30':
        kw.setdefault('markerfacecolor', 'white')
    f(x, y, color=c, marker=m or None, linestyle=l, linewidth=lw, markersize=5.5, label=n, **kw)


def legend_inside(ax, axs=None, loc='upper center', ncol=2, **kw):
    """One framed legend inside `ax` with the entries of all axes in `axs`, in the chapter's method order."""
    h, l = [], []
    for a in (axs or [ax]):
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    idx = sorted(range(len(l)), key=lambda i: ORDER.index(l[i]) if l[i] in ORDER else len(ORDER) + i)
    return ax.legend([h[i] for i in idx], [l[i] for i in idx], loc=loc, ncol=ncol, **{**LEG, **kw})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    if INLINE:
        plt.show()
    plt.close(fig)


mean = lambda d, keys, i: np.array([np.array(d[k])[:, i].mean() for k in keys])
failed = lambda d, keys, i, o: [int((np.array(d[k])[:, i] < np.array(d[k])[:, o] - 3).sum()) for k in keys]
r1 = lambda v: [round(float(x), 2) for x in v]


def _sp():
    SP = J('res_main_spss.json')
    return SP, (lambda k: float(np.mean(np.maximum(SP[k], -100))))


# --------------------------------------------------------------------------- Figure 4.1
def fig_4_1():
    """Measured runtime versus array size (recorded run unless RAB_LOCAL_TIMING=1)."""
    t = J('res_main_time.json') if LOCAL_TIMING else json.load(open(os.path.join(PKG_DATA, 'res_main_time_recorded.json')))
    Ms = [16, 32, 64, 128]; T = np.array([t[str(m)] for m in Ms])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, k, hol in ((2, 'wc', 1), (6, 'incm', 1), (4, 'tr', 0), (5, 'bl', 0), (9, 'gm', 0)):
        n, c, m, l, lw = ST[k]
        ax.loglog(Ms, T[:, i], color=c, marker=m, linestyle=l, linewidth=lw, markersize=6.5,
                  label=n + (' (per solve)' if hol else ' (per snapshot)'), markerfacecolor='white' if hol else c)
    ax.loglog(Ms, T[:, 0], color='#7f7f7f', marker='v', linestyle='--', linewidth=1.1, markersize=6, markerfacecolor='white', label='MVDR (per solve)')
    ax.set_xticks(Ms); ax.set_xticklabels([str(m) for m in Ms]); ax.minorticks_off()
    ax.set_ylim(8, 2e6); ax.set_xlabel('Number of elements M'); ax.set_ylabel('Runtime (µs)')
    ax.legend(loc='upper left', ncol=2, **LEG)
    save(fig, 'Fig4_01_runtime')
    RESULTS['Figure 4.1'] = {'M': Ms, 'source': 'this machine' if LOCAL_TIMING else 'recorded run',
                             **{n: r1(T[:, i]) for i, n in enumerate(t['names'])}}


# --------------------------------------------------------------------------- Figure 4.2
def fig_4_2():
    """Output SINR versus steering-vector mismatch, point source, M = 8, 16, 32 and 64."""
    SP, spm = _sp(); res = {}
    fig, ax = plt.subplots(2, 2, figsize=(10, 8.4)); ax = ax.ravel()
    for a, Ml, lab in zip(ax, (8, 16, 32, 64), 'abcd'):
        d = J(f'res_main_sweep{Ml}.json'); keys = list(d); x = [float(k) for k in keys]
        for i, k in ((2, 'incm'), (3, 'wc'), (4, 'bl'), (5, 'tr'), (1, 'gm'), (7, 'opt')):
            line(a, x, mean(d, keys, i), k)
        line(a, x, [spm(f'sweep{Ml}|{k}') for k in keys], 'sp')
        bw = np.rad2deg(1.772 / Ml); a.axvline(bw / 2, color='#999999', lw=0.8, ls=':'); a.text(bw / 2, LO + 1.2, ' θ3dB/2', fontsize=8, color='#666666')
        a.set_ylim(LO, 50); a.set_xlabel('Steering-vector mismatch (degrees)'); a.set_ylabel('Output SINR (dB)')
        a.set_title(f'({lab}) M = {Ml}', fontsize=10)
        res[f'M = {Ml}'] = {'mismatch': x, **{ST[k][0]: r1(mean(d, keys, i)) for i, k in ((1, 'gm'), (2, 'incm'), (3, 'wc'), (4, 'bl'), (5, 'tr'), (7, 'opt'))},
                            'SPSS': r1([spm(f'sweep{Ml}|{k}') for k in keys]),
                            'failed, guided R-MBER': failed(d, keys, 1, 7), 'failed, INCM-REC': failed(d, keys, 2, 7),
                            'failed, WC-MVDR/EigDec': failed(d, keys, 3, 7)}
    for a in ax:
        legend_inside(a, ncol=2)
    fig.tight_layout(); save(fig, 'Fig4_02_SINR_vs_mismatch')
    RESULTS['Figure 4.2'] = res


# --------------------------------------------------------------------------- Figure 4.3
CASES = ['2 pt', '5 pt', '7 pt', '2 sm', '5 sm', '7 sm', '5 lg', '7 lg']


def fig_4_3():
    """Output SINR in the eight mismatch and propagation cases, M = 32."""
    SP, spm = _sp()
    t3 = J('res_trained_cases.json'); m = J('res_guided_mp.json'); c = np.array(J('res_fullarray_mp.json')['D']); keys = list(m)
    xt = ['2°\npoint', '5°\npoint', '7°\npoint', '2°\nsmall', '5°\nsmall', '7°\nsmall', '5°\nlarge', '7°\nlarge']
    V = [('g', mean(m, keys, 0), '++'), ('gm', mean(m, keys, 1), ''), ('incm', mean(m, keys, 2), 'xx'),
         ('sp', np.array([spm('case|' + k) for k in keys]), '--'), ('wc', np.array([t3[k][2] for k in keys]), '..'),
         ('bl', c[:, :, 1].mean(1), '\\\\'), ('tr', np.array([t3[k][4] for k in keys]), '//')]
    fig, ax = plt.subplots(figsize=(9, 5.4)); x = np.arange(8); w = 0.115
    for j, (k, v, h) in enumerate(V):
        n, col, _, _, _ = ST[k]
        ax.bar(x + (j - 3) * w, np.maximum(v, LO), w, bottom=0, label=n, color=col, hatch=h, edgecolor='white' if h else col, linewidth=0.3)
    ax.hlines(mean(m, keys, 4), x - 0.45, x + 0.45, colors='k', linestyles='--', linewidth=1.1, label='Optimal MVDR')
    ax.axhline(0, color='k', lw=0.6); ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=9); ax.set_ylim(LO, 42); ax.grid(axis='x', visible=False)
    ax.set_ylabel('Output SINR (dB)'); ax.set_xlabel('Mismatch and propagation case')
    legend_inside(ax, ncol=3)
    save(fig, 'Fig4_03_eight_cases')
    RESULTS['Figure 4.3'] = {'cases': keys, **{ST[k][0]: r1(v) for k, v, _ in V}, 'Optimal MVDR': r1(mean(m, keys, 4)),
                             'failed, guide alone': failed(m, keys, 0, 4), 'failed, guided R-MBER': failed(m, keys, 1, 4)}


# --------------------------------------------------------------------------- Figure 4.4
def fig_4_4():
    """Output SINR versus the number of snapshots processed, 7 degrees, point source and large spread."""
    tr = J('res_trained_traj.json'); cv = J('res_main_conv.json'); K = 150; ck = [10, 25, 50, 75, 100, 125, 150]; res = {}
    fig, ax = plt.subplots(2, 1, figsize=(7.6, 9.2))
    for a, tag, title in ((ax[0], 'pt', '(a) point source'), (ax[1], 'lg', '(b) large angular spread')):
        S = np.array(tr[tag]['S']); I = np.array(tr[tag]['I']); n = np.arange(1, K + 1); st = 10; g = cv[tag]
        line(a, n[st - 1::st], S[2, st - 1::st], 'wc'); line(a, ck, I, 'incm'); line(a, n[st - 1::st], np.array(g['blind'])[st - 1::st], 'bl')
        line(a, n[st - 1::st], S[4, st - 1::st], 'tr'); line(a, g['ks'], g['guided'], 'gm'); line(a, n[st - 1::st], S[7, st - 1::st], 'opt')
        a.set_ylim(-30, 45); a.set_xlim(0, K); a.set_xlabel('Snapshots processed'); a.set_ylabel('Output SINR (dB)')
        a.set_title(title + ', 7° mismatch, M = 32', fontsize=10)
        legend_inside(a, ncol=2)
        res[tag] = {'guided R-MBER, block lengths': g['ks'], 'guided R-MBER': r1(g['guided']), 'guide alone': r1(g['guide']),
                    'trained R-MBER at 10, 50, 100, 150': r1(S[4, [9, 49, 99, 149]]), 'WC-MVDR/EigDec max': round(float(S[2].max()), 2),
                    'WC-MVDR/EigDec final': round(float(S[2, -1]), 2), 'INCM-REC at 10..150': r1(I),
                    'blind R-MBER max': round(float(np.max(g['blind'])), 2), 'optimum': round(float(S[7, -1]), 2)}
    fig.tight_layout(); save(fig, 'Fig4_04_convergence')
    RESULTS['Figure 4.4'] = res


# --------------------------------------------------------------------------- Figures 4.5 and 4.6
def _ber(mode, name, num):
    SP, _ = _sp()
    b3 = J(f'res_berbase_{mode}.json'); p2 = J('res_berguided.json'); bb = J('res_main_berb.json'); res = {}
    xl = 'Steering-vector mismatch (degrees)' if mode == 'mm' else 'Element SNR (dB)'
    fig, ax = plt.subplots(3, 1, figsize=(7.6, 12.4))
    for a, (lab, l3), lab_ in zip(ax, (('point', 'point source'), ('small', 'small angular spread'), ('large', 'large angular spread')), 'abc'):
        R3 = np.array(b3['res'][l3]); x = b3['x']; P = np.array(p2[f'{mode}|{lab}']['Y']); xb = bb[f'{mode}|{lab}']['x']
        FL = 1e-11 if mode == 'mm' else 1e-16; fl = lambda v: np.maximum(v, FL)
        line(a, x, fl(R3[5]), 'incm', True); line(a, x, fl(SP[f'ber|{mode}|{lab}']), 'sp', True); line(a, x, fl(R3[2]), 'wc', True)
        line(a, xb, fl(bb[f'{mode}|{lab}']['Y']), 'bl', True); line(a, x, fl(R3[4]), 'tr', True)
        line(a, p2[f'{mode}|{lab}']['x'], fl(P[:, 1]), 'gm', True)
        if mode == 'mm':
            a.set_ylim(1e-11, 1); legend_inside(a, loc='lower center', ncol=3)
        else:
            a.set_ylim(3e-17, 1); a.axhline(FL, color='k', ls=':', lw=0.9); legend_inside(a, loc='lower left', ncol=1)
        a.set_xlabel(xl); a.set_ylabel('BER')
        a.set_title(f'({lab_}) {l3}, ' + ('element SNR −5 dB' if mode == 'mm' else '5° mismatch'), fontsize=10)
        res[l3] = {'x': x, 'guided R-MBER': [float(v) for v in P[:, 1]], 'guide alone': [float(v) for v in P[:, 0]],
                   'INCM-REC': [float(v) for v in R3[5]], 'WC-MVDR/EigDec': [float(v) for v in R3[2]], 'Trained R-MBER': [float(v) for v in R3[4]],
                   'Blind R-MBER, full array': [float(v) for v in bb[f'{mode}|{lab}']['Y']], 'SPSS': [float(v) for v in SP[f'ber|{mode}|{lab}']],
                   'TA-MMSE': [float(v) for v in R3[6]], 'Optimal MVDR': [float(v) for v in R3[7]]}
    fig.tight_layout(); save(fig, name)
    RESULTS[num] = res


def fig_4_5():
    """Model-based BER versus mismatch, element SNR -5 dB, M = 32."""
    _ber('mm', 'Fig4_05_BER_vs_mismatch', 'Figure 4.5')


def fig_4_6():
    """Model-based BER versus element SNR, 5 degrees of mismatch, M = 32."""
    _ber('snr', 'Fig4_06_BER_vs_SNR', 'Figure 4.6')


# --------------------------------------------------------------------------- Figure 4.7
def fig_4_7():
    """Output SINR versus the INR of each interferer, point source, M = 32, K = 150 and 1000."""
    SP, spm = _sp()
    si = J('res_guided_inr.json'); i1 = J('res_main_inr1000.json'); x6 = J('res_main_inrx.json'); t6 = J('res_trained_inr.json'); t6k = J('res_trained_inr1000.json')
    INR = [10, 15, 20, 25, 30]; res = {}
    fig, ax = plt.subplots(2, 2, figsize=(10, 8.4)); ax = ax.ravel()
    for a, (Kn, mm), lab in zip(ax, ((150, 5), (150, 7), (1000, 5), (1000, 7)), 'abcd'):
        src = si if Kn == 150 else i1; T6 = t6 if Kn == 150 else t6k; j = {5: 1, 7: 2}[mm]; ks = [f'{r}|{mm}' for r in INR]
        line(a, INR, mean(src, ks, 2), 'incm'); line(a, INR, [spm(f'inr|{Kn}|{r}|{mm}') for r in INR], 'sp')
        line(a, INR, [np.array(x6[f'{Kn}|{r}|{mm}'])[:, 0].mean() for r in INR], 'wc'); line(a, INR, [np.array(x6[f'{Kn}|{r}|{mm}'])[:, 1].mean() for r in INR], 'bl')
        line(a, INR, [T6[f'inr{r}'][0][j] for r in INR], 'tr'); line(a, INR, mean(src, ks, 1), 'gm'); line(a, INR, mean(src, ks, 4), 'opt')
        a.set_ylim(-30, 48); a.set_xticks(INR); a.set_xlabel('INR of each interferer (dB)'); a.set_ylabel('Output SINR (dB)')
        a.set_title(f'({lab}) K = {Kn}, {mm}° mismatch', fontsize=10); legend_inside(a, ncol=2)
        res[f'K = {Kn}, {mm} deg'] = {'INR': INR, 'guided R-MBER': r1(mean(src, ks, 1)), 'INCM-REC': r1(mean(src, ks, 2)),
                                     'Trained R-MBER': r1([T6[f'inr{r}'][0][j] for r in INR]), 'failed, guided R-MBER': failed(src, ks, 1, 4)}
    fig.tight_layout(); save(fig, 'Fig4_07_SINR_vs_INR')
    RESULTS['Figure 4.7'] = res


# --------------------------------------------------------------------------- Figure 4.8
def fig_4_8():
    """Output SINR versus element SNR (2 and 5 degrees) and versus the number of random interferers (3 and 7 degrees)."""
    SP, spm = _sp()
    cs = J('res_fullarray_snr.json'); ms = J('res_main_snr.json'); cn = J('res_fullarray_nint.json'); sn = J('res_guided_nint.json'); res = {}
    fig, ax = plt.subplots(2, 2, figsize=(10, 8.4)); ax = ax.ravel(); SN = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]
    for a, mm, lab in ((ax[0], 2, 'a'), (ax[1], 5, 'b')):
        D = np.array(cs['D']); idx = [cs['keys'].index(f'{mm}/{s:g}') for s in SN]; ks = [f'{mm}|{s:g}' for s in SN]
        line(a, SN, D[idx, :, 3].mean(1), 'incm'); line(a, SN, [spm(f'snr|{mm}|{s:g}') for s in SN], 'sp'); line(a, SN, D[idx, :, 1].mean(1), 'bl')
        line(a, SN, D[idx, :, 0].mean(1), 'tr'); line(a, SN, mean(ms, ks, 1), 'gm'); line(a, SN, mean(ms, ks, 2), 'opt')
        a.set_ylim(-30, 52); a.set_xlabel('Element SNR (dB)'); a.set_ylabel('Output SINR (dB)'); a.set_title(f'({lab}) point source, {mm}° mismatch', fontsize=10)
        res[f'SNR, {mm} deg'] = {'SNR': SN, 'guided R-MBER': r1(mean(ms, ks, 1)), 'optimum': r1(mean(ms, ks, 2)), 'INCM-REC': r1(D[idx, :, 3].mean(1)),
                                 'failed, guided R-MBER': failed(ms, ks, 1, 2)}
    NN = [2, 4, 6, 8, 10, 12, 16]
    for a, mm, lab in ((ax[2], 3, 'c'), (ax[3], 7, 'd')):
        D = np.array(cn['D']); idx = [cn['keys'].index(f'{n}/{mm}') for n in NN]; ks = [f'{n}|{mm}' for n in NN]
        line(a, NN, mean(sn, ks, 2), 'incm'); line(a, NN, [spm(f'nint|{n}|{mm}') for n in NN], 'sp'); line(a, NN, D[idx, :, 1].mean(1), 'bl')
        line(a, NN, D[idx, :, 0].mean(1), 'tr'); line(a, NN, mean(sn, ks, 1), 'gm'); line(a, NN, mean(sn, ks, 4), 'opt')
        a.set_ylim(-30, 48); a.set_xticks(NN); a.set_xlabel('Number of interferers'); a.set_ylabel('Output SINR (dB)')
        a.set_title(f'({lab}) random interferer directions, {mm}° mismatch', fontsize=10)
        res[f'interferers, {mm} deg'] = {'N': NN, 'guided R-MBER': r1(mean(sn, ks, 1)), 'INCM-REC': r1(mean(sn, ks, 2)),
                                         'failed, guided R-MBER': failed(sn, ks, 1, 4)}
    for a in ax:
        legend_inside(a, ncol=2)
    fig.tight_layout(); save(fig, 'Fig4_08_SINR_vs_SNR_and_interferers')
    RESULTS['Figure 4.8'] = res


# --------------------------------------------------------------------------- Figure 4.9
def fig_4_9():
    """Output SINR versus the direction of a third interferer, 3 degrees of mismatch, M = 32."""
    SP, spm = _sp()
    ne = J('res_guided_near.json'); ex = J('res_main_near.json')
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    keys = [k for k in ne if k.startswith('3|')]; x = [float(k.split('|')[1]) for k in keys]
    line(ax, x, mean(ne, keys, 2), 'incm'); line(ax, x, [spm('near|' + k) for k in keys], 'sp'); line(ax, x, [np.mean(ex[f'near|{k}']) for k in keys], 'bl')
    line(ax, x, mean(ne, keys, 5), 'tr'); line(ax, x, mean(ne, keys, 1), 'gm'); line(ax, x, mean(ne, keys, 4), 'opt')
    ax.set_xticks(x); ax.set_ylim(-30, 45); ax.set_xlabel('Direction of the third interferer (degrees)'); ax.set_ylabel('Output SINR (dB)')
    legend_inside(ax, ncol=3)
    save(fig, 'Fig4_09_near_interferer')
    k0 = [k for k in ne if k.startswith('0|')]
    RESULTS['Figure 4.9'] = {'direction': x, 'guided R-MBER': r1(mean(ne, keys, 1)), 'INCM-REC': r1(mean(ne, keys, 2)), 'SPSS': r1([spm('near|' + k) for k in keys]),
                             'Trained R-MBER': r1(mean(ne, keys, 5)), 'failed, guided R-MBER': failed(ne, keys, 1, 4),
                             'guided R-MBER, no mismatch': r1(mean(ne, k0, 1))}


# --------------------------------------------------------------------------- Figure 4.10
def fig_4_10():
    """Imperfect training reference, M = 32: contamination, interferer onset, wrong labels, source drift, late reference."""
    SP, spm = _sp(); res = {}
    fig, ax = plt.subplots(3, 2, figsize=(10, 12.4)); ax = ax.ravel()
    PAN = [('contam', 'Power of the contaminating user relative to the source (dB)', '(a) pilot contamination'),
           ('onset', 'INR of an interferer that appears after the pilots (dB)', '(b) interferer onset after the pilots'),
           ('labels', 'Wrong labels among the 30 pilots', '(c) wrong pilot labels'), ('drift', 'Source drift over the data part (degrees)', '(d) source drift after the pilots'),
           ('slip', 'Delay of the pilot reference (symbols)', '(e) late pilot reference')]
    for a, (stg, xl, tt) in zip(ax, PAN):
        d = J(f'res_imperfect_{stg}.json'); keys = list(d); xs = list(range(len(keys))) if keys[0] == 'None' else [float(k) for k in keys]
        for i, k in ((1, 'incm'), (4, 'ta'), (2, 'ta30'), (3, 'tadd'), (0, 'gm'), (6, 'opt')):
            line(a, xs, mean(d, keys, i), k)
        line(a, xs, [spm(f'ref|{stg}|{k}') for k in keys], 'sp')
        if keys[0] == 'None':
            a.set_xticks(xs); a.set_xticklabels(['none'] + [k.replace('.0', '') for k in keys[1:]])
        if stg == 'slip':
            a.set_xticks(xs)
        a.set_ylim(-10, 25); a.set_xlabel(xl); a.set_ylabel('Output SINR (dB)'); a.set_title(tt + ', 5° mismatch', fontsize=10)
        res[stg] = {'values': keys, 'guided R-MBER': r1(mean(d, keys, 0)), 'INCM-REC': r1(mean(d, keys, 1)), 'TA-MMSE, 30 pilots': r1(mean(d, keys, 2)),
                    'TA-MMSE, 30 pilots + own decisions': r1(mean(d, keys, 3)), 'TA-MMSE, every symbol known': r1(mean(d, keys, 4)),
                    'trained R-MBER, 30 pilots': r1(mean(d, keys, 5)), 'optimum': r1(mean(d, keys, 6)), 'SPSS': r1([spm(f'ref|{stg}|{k}') for k in keys]),
                    'failed, guided R-MBER': failed(d, keys, 0, 6), 'failed, TA-MMSE 30 pilots': failed(d, keys, 2, 6)}
    ax[5].axis('off'); legend_inside(ax[5], axs=list(ax[:5]), loc='center', ncol=1, fontsize=9)
    fig.tight_layout(); save(fig, 'Fig4_10_imperfect_reference')
    RESULTS['Figure 4.10'] = res


# --------------------------------------------------------------------------- Figure 4.11
def fig_4_11():
    """BER step under local scattering, M = 32, 5 degrees: output SINR and counted BER, guide alone vs guided R-MBER."""
    res = {}
    fig, ax = plt.subplots(2, 2, figsize=(10, 8.6)); ax = ax.ravel()
    for col, (stg, xl) in enumerate((('amp', 'Amplitude of each local ray (rays at +4° and −3°)'), ('off', 'Offset d of the local rays at ±d, amplitude 0.4 (degrees)'))):
        d = J(f'res_scat_{stg}.json'); keys = list(d); x = [float(k) for k in keys]
        S = lambda i: [np.array(d[k]['S'])[:, i].mean() for k in keys]; B = lambda i: [np.array(d[k]['B'])[:, i].mean() for k in keys]
        for i, k in ((2, 'incm'), (0, 'g'), (1, 'gm'), (3, 'opt')):
            line(ax[col], x, S(i), k); line(ax[col + 2], x, B(i), k, True)
        ax[col].set_ylim(15, 26); ax[col].set_xlabel(xl); ax[col].set_ylabel('Output SINR (dB)'); ax[col].set_title(f'({"ab"[col]}) output SINR, element SNR 5 dB', fontsize=10)
        ax[col + 2].set_ylim(3e-4, 4); ax[col + 2].set_xlabel(xl); ax[col + 2].set_ylabel('Counted BER'); ax[col + 2].set_title(f'({"cd"[col]}) counted BER, element SNR −10 dB', fontsize=10)
        if stg == 'off':
            ax[col].set_xticks([1, 2, 3, 4, 6, 8, 10, 15, 20]); ax[col + 2].set_xticks([1, 2, 3, 4, 6, 8, 10, 15, 20])
        o = lambda i: [int((np.array(d[k]['S'])[:, i] < np.array(d[k]['S'])[:, 3] - 3).sum()) for k in keys]
        res[stg] = {'x': x, 'SINR guide alone': r1(S(0)), 'SINR guided R-MBER': r1(S(1)), 'SINR INCM-REC': r1(S(2)), 'SINR optimum': r1(S(3)),
                    'gain of the BER step (dB)': r1(np.array(S(1)) - np.array(S(0))), 'counted BER guide alone': [float(v) for v in B(0)],
                    'counted BER guided R-MBER': [float(v) for v in B(1)], 'failed, guide alone': o(0), 'failed, guided R-MBER': o(1), 'failed, INCM-REC': o(2)}
    for a in ax:
        legend_inside(a, ncol=2)
    fig.tight_layout(); save(fig, 'Fig4_11_local_scattering')
    RESULTS['Figure 4.11'] = res


FIGURES = [fig_4_1, fig_4_2, fig_4_3, fig_4_4, fig_4_5, fig_4_6, fig_4_7, fig_4_8, fig_4_9, fig_4_10, fig_4_11]


# --------------------------------------------------------------------------- numbers quoted in the text
def text_numbers():
    """Collect the text-only results (Sections 4.3.1, 4.5, 4.7.1, 4.7.3, 4.7.5, 4.7.8, 4.7.9, 4.7.10)."""
    T = {}
    for name in ('floor_c', 'dykstra', 'distances', 'incm_sets', 'radius', 'ta_loading', 'wc_solver', 'spss_sinc'):
        p = os.path.join(DATA, f'res_text_{name}.json')
        if os.path.exists(p):
            T[name] = json.load(open(p))
    for key, f in (('bandwidth floor and held-out BER', 'res_floor.json'), ('BER step gain in the eight cases', 'res_scat_ci.json'),
                   ('WC-MVDR/EigDec coverage (deg)', 'res_scat_wc.json'), ('phase recovery', 'res_phase.json'),
                   ('trained R-MBER without half-spaces', 'res_nocon.json'), ('blind R-MBER without half-spaces', 'res_blindnc.json')):
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            T[key] = json.load(open(p))
    p = os.path.join(DATA, 'res_scat_dir.json')
    if os.path.exists(p):
        d = json.load(open(p))
        T['direction-only BER step'] = {k: {'guide': round(float(np.mean(np.array(v)[:, 0])), 2), 'guided': round(float(np.mean(np.array(v)[:, 1])), 2),
                                            'direction-only': round(float(np.mean(np.array(v)[:, 2])), 2),
                                            'norm ratio range': [round(float(np.min(np.array(v)[:, 4])), 2), round(float(np.max(np.array(v)[:, 4])), 2)]}
                                        for k, v in d.items()}
    p = os.path.join(DATA, 'res_main_ta30.json')
    if os.path.exists(p):
        d = json.load(open(p))
        T['TA-MMSE, 30 pilots (mean SINR, dB)'] = {k: round(float(np.mean(v)), 2) for k, v in d.items()}
    RESULTS['Text'] = T


def write_summary():
    text_numbers()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'results_summary.json')
    with open(p, 'w') as f:
        json.dump(RESULTS, f, indent=1, ensure_ascii=False)
    return p


def make_all():
    for f in FIGURES:
        f()
    return write_summary()
