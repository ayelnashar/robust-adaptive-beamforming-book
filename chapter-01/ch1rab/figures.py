"""
ch1rab.figures -- generates every simulation figure of Chapter 1 (Figs. 1.5-1.27) plus two diagnostics.

Configured through environment variables read at import time:
  QUICK=1       smoke test (few Monte Carlo runs)
  FIGDIR=path   output folder (default ./figures)
  RAB_CACHE=1   reuse the simulation data saved in FIGDIR/data (re-plot only, no re-simulation)
  RAB_INLINE=1  show each figure under its cell in Jupyter
Used by the notebook and by scripts/generate_figures.py.
"""
import os, json, time, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .scenarios import tap_mc, tap_run, iet_mc, iet_run, TAP_METHODS, WC_METHODS
from .algorithms import beampattern

QUICK = os.environ.get('QUICK') == '1'
RUNS_TAP = 3 if QUICK else 50        # TAP scenarios: 50 runs
RUNS_IET = 3 if QUICK else 100       # IET scenarios: 100 runs
RUNS_SWEEP = 2 if QUICK else 50      # runs per point of the SINR-vs-noise / mismatch sweeps
RUNS_N50 = 3 if QUICK else 300       # Fig. 1.19: N = 50 snapshots per run, so more runs per point
OUT = os.environ.get('FIGDIR', 'figures')
USE_CACHE = os.environ.get('RAB_CACHE') == '1'
INLINE = os.environ.get('RAB_INLINE') == '1'
M = 5
RESULTS = {}
KEEP_A = [m for m in TAP_METHODS if m != 'Adaptive joint-constraint RCB']   # figures without the joint variant
WCA = 'Robust MVDR-WC/Adaptive'

# --------------------------------------------------------------------------- visual identity
# One colour per method family, fixed across all figures (validated categorical palette, print steps);
# every series also has its own marker and dash pattern, so the figures read in grayscale.
INK, INK2, GRID = '#1f1f1d', '#52514e', '#e4e3de'
NEUTRAL, BENCH = '#8a8984', '#1f1f1d'
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN = \
    '#2a78d6', '#eb6834', '#199e70', '#c98500', '#d55181', '#008300'

STYLE = {
    # TAP 2006 family
    'Standard Capon':                dict(color=NEUTRAL, ls=(0, (5, 2)), marker='x', lw=1.3),
    'Batch RCB':                     dict(color=C_BLUE, ls='-', marker='s', lw=1.6),
    'Subspace-tracking RCB':         dict(color=C_AQUA, ls=(0, (6, 2, 1, 2)), marker='^', lw=1.5),
    'SOCP-based RCB':                dict(color=C_YELLOW, ls=(0, (3, 2)), marker='D', lw=1.6),
    'Adaptive spherical RCB':        dict(color=C_ORANGE, ls='-', marker='o', lw=2.8, zorder=2),
    'Adaptive joint-constraint RCB': dict(color=C_MAGENTA, ls=(0, (1, 1.5)), marker='v', lw=2.0),
    # IET 2008 family
    'Standard MVDR':                 dict(color=NEUTRAL, ls=(0, (5, 2)), marker='x', lw=1.3),
    'Robust MVDR-WC/SOCP':           dict(color=C_YELLOW, ls=(0, (3, 2)), marker='D', lw=1.6),
    'Robust MVDR-WC/EigDec':         dict(color=C_BLUE, ls='-', marker='s', lw=1.6),
    WCA:                             dict(color=C_ORANGE, ls='-', marker='o', lw=3.0, zorder=2),
    'Benchmark MVDR':                dict(color=BENCH, ls=(0, (7, 2, 1, 2)), marker='P', lw=1.3),
    'Benchmark LCMV':                dict(color=C_AQUA, ls=(0, (6, 2, 1, 2)), marker='^', lw=1.5),
    'Standard LCMV':                 dict(color=C_MAGENTA, ls=(0, (1, 1.5)), marker='v', lw=1.6),
    'Robust LCMV-MBWC':              dict(color=C_GREEN, ls='-', marker='h', lw=2.0),
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9.5,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'legend.frameon': False, 'legend.fontsize': 8.5, 'legend.handlelength': 3.2,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
})
FIGSIZE = (6.0, 3.9)


def db(x):
    return 10 * np.log10(np.maximum(np.abs(x), 1e-30))


def cached(key, func, *args, **kw):
    """Run func(*args, **kw) and keep the result in OUT/data/key.pkl; with RAB_CACHE=1 reuse it."""
    path = os.path.join(OUT, 'data', key + '.pkl')
    if USE_CACHE and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    res = func(*args, **kw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(res, f)
    return res


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, name + '.png')
    fig.savefig(png, dpi=300)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    if INLINE:                                   # inside Jupyter: show a light preview under the cell
        import io
        from IPython.display import display, Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110)
        display(Image(data=buf.getvalue()))
    plt.close(fig)


def legend_below(ax, n):
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3 if n > 4 else n,
              columnspacing=1.6, handletextpad=0.6)


def curve_plot(curves, ylabel, xlabel='Snapshot index', x=None, every=100, ref=None, ylim=None,
               legend=True):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for j, (m, y) in enumerate(curves.items()):
        st = dict(STYLE.get(m, dict(color=INK, ls='-', marker='o', lw=1.6)))
        lw = st.pop('lw')
        st.setdefault('zorder', 3)
        xx = np.arange(1, len(y) + 1) if x is None else x
        me = ((every // 2 + 13 * j) % every, every) if x is None else 1   # stagger markers of overlapping lines
        ax.plot(xx, y, lw=lw, markevery=me, ms=5.5, mfc='white', mew=1.3, label=m, **st)
    if ref is not None:
        ax.axhline(ref[0], color=INK2, lw=1.0, ls=(0, (1, 2)), label=ref[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x is None:
        ax.set_xlim(0, len(next(iter(curves.values()))))
    if ylim:
        ax.set_ylim(*ylim)
    if legend:
        legend_below(ax, len(curves) + (ref is not None))
    return fig, ax


def pattern_plot(W, name, marks):
    """marks: list of (angle_rad, label, kind) with kind 'soi' or 'jam'."""
    deg = np.arange(-90, 90.05, 0.1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for t, lab, kind in marks:
        c = C_GREEN if kind == 'soi' else '#b0afa8'
        ax.axvline(np.rad2deg(t), color=c, lw=1.0, ls=(0, (2, 2)), zorder=0)
        ax.text(np.rad2deg(t), 3.2, lab, ha='center', va='bottom', fontsize=8, color=INK2)
    for m, w in W.items():
        st = dict(STYLE.get(m, {}))
        lw = st.pop('lw', 1.6)
        st.pop('marker', None)
        st.setdefault('zorder', 3)
        ax.plot(deg, beampattern(w, M, np.deg2rad(deg)), lw=lw, label=m, **st)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized beampattern (dB)')
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_ylim(-60, 2)
    legend_below(ax, len(W))
    save(fig, name)


def socp_delta(eps_rcb):
    """SOCP norm bound equivalent to the RCB spherical set ||a - a0||^2 <= eps."""
    return np.sqrt(eps_rcb)


def smooth(y):
    """Display smoothing for the MSE curves: 1st-order Butterworth, Wn = 0.01, zero phase, even padding."""
    from scipy.signal import butter, filtfilt
    b, a = butter(1, 0.01)
    return filtfilt(b, a, y, padtype='even')


def msmooth(y, frac=0.25, hmax=50):
    """Display smoothing for SINR-vs-snapshot curves: centred moving average (linear scale) whose
    half-width grows as frac*k up to hmax snapshots (one-sided near the end), so the start-up transient
    is not smeared while the steady-state ripple is averaged out. Numbers quoted in the text are
    computed from the unsmoothed Monte Carlo means."""
    y = np.asarray(y, float)
    n = len(y)
    c = np.concatenate([[0.0], np.cumsum(y)])
    out = np.empty(n)
    for k in range(n):
        h = min(int(frac * k), hmax)
        lo, hi = k - h, min(k + h, n - 1)
        out[k] = (c[hi + 1] - c[lo]) / (hi - lo + 1)
    return out


def ss(y, n=100):
    """steady-state value: mean of the last n points (dB of the linear mean)"""
    return float(db(np.mean(y[-n:])))


# =========================================================================== PART A: TAP 2006
SC1 = dict(noise_db=-50, mis=0.03 * np.pi, eps=0.5, delta_socp=socp_delta(0.5), alpha=0.008, method='SD', tau=0.2)
SC2 = dict(noise_db=-20, mis=0.06 * np.pi, eps=1.0, delta_socp=socp_delta(1.0), alpha=0.01, method='CG', tau=0.2)
MARKS_A = [(0.0, 'SOI', 'soi'), (np.pi / 6, 'J2', 'jam'), (np.pi / 4, 'J1', 'jam')]


def fig_1_5_9():
    """Scenario 1: Figs 1.5-1.9 (steepest descent, 0.03pi mismatch, eps 0.5, noise -50 dB)"""
    acc = cached('sc1_mc', tap_mc, RUNS_TAP, base_seed=0, **SC1)
    fig, _ = curve_plot({m: db(msmooth(acc[m]['sinr'])) for m in TAP_METHODS}, 'Output SINR (dB)')
    save(fig, 'Fig1_05_SINR_scenario1')
    fig, _ = curve_plot({m: db(smooth(acc[m]['mse'])) for m in TAP_METHODS}, 'MSE (dB)')
    save(fig, 'Fig1_06_MSE_scenario1')
    fig, _ = curve_plot({m: db(acc[m]['soip']) for m in TAP_METHODS}, 'SOI power estimate (dB)',
                        ref=(0.0, 'True SOI power'))
    save(fig, 'Fig1_07_SOIP_scenario1')
    RESULTS['Fig1.5'] = {m: ss(acc[m]['sinr']) for m in TAP_METHODS}
    RESULTS['Fig1.6'] = {m: ss(acc[m]['mse']) for m in TAP_METHODS}
    RESULTS['Fig1.7'] = {m: ss(acc[m]['soip']) for m in TAP_METHODS}
    for seed, name in [(0, 'Fig1_08_beampattern_a'), (1, 'Fig1_09_beampattern_b')]:
        o = cached(f'sc1_run{seed}', tap_run, seed, **SC1)
        pattern_plot({m: o['_w'][m] for m in KEEP_A}, name, MARKS_A)
    # diagnostic: is the ellipsoidal constraint ever active?
    diag = {}
    for al in (0.008, 0.02, 0.05):
        o = cached(f'sc1_alpha{al}', tap_run, 0, **dict(SC1, alpha=al))
        diag[al] = (o['_vl_count'], float(db(o['Adaptive spherical RCB']['sinr'][-1])))
    RESULTS['diag_VL_activity_scenario1'] = {f'alpha={k}': {'VL activations': v[0], 'final SINR dB': round(v[1], 2)}
                                             for k, v in diag.items()}


def fig_1_10():
    """Scenario 2: Fig 1.10 (conjugate gradient, 0.06pi mismatch, eps 1, noise -20 dB)"""
    acc = cached('sc2_mc', tap_mc, RUNS_TAP, base_seed=100, **SC2)
    fig, _ = curve_plot({m: db(msmooth(acc[m]['sinr'])) for m in TAP_METHODS}, 'Output SINR (dB)')
    save(fig, 'Fig1_10_SINR_scenario2')
    RESULTS['Fig1.10'] = {m: ss(acc[m]['sinr']) for m in TAP_METHODS}
    o = cached('sc2_run100', tap_run, 100, **SC2)
    RESULTS['diag_VL_activity_scenario2'] = {'VL activations (one run)': o['_vl_count']}


def noise_sweep(noise, mis, al):
    res = {m: np.zeros(len(noise)) for m in TAP_METHODS}
    for i, nd in enumerate(noise):
        for r in range(RUNS_SWEEP):
            o = tap_run(500 + 37 * i + r, noise_db=nd, mis=mis, eps=1.0, delta_socp=socp_delta(1.0),
                        alpha=al, method='SD', tau=0.2, final_only=True)
            for m in TAP_METHODS:
                res[m][i] += o[m]['sinr'][-1] / RUNS_SWEEP
    return res


def fig_1_11_12():
    """Figs 1.11, 1.12: SINR vs noise power (final snapshot, N = 1000)"""
    noise = np.arange(30, -31, -4)
    for name, mis, al, key in [('Fig1_11_SINR_vs_noise_003pi', 0.03 * np.pi, 0.02, 'Fig1.11'),
                               ('Fig1_12_SINR_vs_noise_006pi', 0.06 * np.pi, 0.008, 'Fig1.12')]:
        res = cached(key.replace('.', '_'), noise_sweep, noise, mis, al)
        fig, ax = curve_plot({m: db(res[m]) for m in TAP_METHODS}, 'Output SINR (dB)',
                             xlabel='Noise power (dB)', x=noise)
        ax.invert_xaxis()
        save(fig, name)
        RESULTS[key] = {m: dict(zip([int(v) for v in noise[::3]], np.round(db(res[m])[::3], 2).tolist()))
                        for m in TAP_METHODS}


def fig_1_13_15():
    """Figs 1.13-1.15: moving jammers (noise -20 dB, 0.01pi mismatch)"""
    d2r = np.pi / 180
    moving = [('Fig1_13_moving_noncoherent', lambda k: (np.pi / 4 + 0.1 * d2r * k, np.pi / 6 + 0.01 * d2r * k)),
              ('Fig1_14_moving_coherent', lambda k: (np.pi / 4 + 0.1 * d2r * k, np.pi / 6 - 0.01 * d2r * k)),
              ('Fig1_15_moving_nonstationary', lambda k: (np.pi / 4 + 10 * d2r * np.sin(k / 15),
                                                          np.pi / 6 + 10 * d2r * np.sin(k / 15)))]
    for j, (name, traj) in enumerate(moving):
        acc = cached(name, tap_mc, RUNS_TAP, base_seed=200 + 100 * j, noise_db=-20, mis=0.01 * np.pi, eps=0.5,
                     delta_socp=socp_delta(0.5), alpha=0.01, method='SD', tau=None, traj=traj)
        fig, _ = curve_plot({m: db(msmooth(acc[m]['sinr'])) for m in KEEP_A}, 'Output SINR (dB)')
        save(fig, name)
        RESULTS['Fig1.' + name[5:7]] = {m: ss(acc[m]['sinr']) for m in KEEP_A}


def mismatch_sweep(angs):
    res = {m: np.zeros(len(angs)) for m in KEEP_A}
    for i, a in enumerate(angs):
        for r in range(RUNS_SWEEP):
            o = tap_run(900 + 41 * i + r, noise_db=-15, mis=np.deg2rad(a), eps=1.0, delta_socp=socp_delta(1.0),
                        alpha=0.008, method='SD', tau=None, final_only=True)
            for m in KEEP_A:
                res[m][i] += o[m]['sinr'][-1] / RUNS_SWEEP
    return res


def fig_1_16():
    """Fig 1.16: SINR vs mismatch angle (noise -15 dB, eps 1)"""
    angs = np.arange(0, 21, 1)
    res = cached('Fig1_16', mismatch_sweep, angs)
    fig, _ = curve_plot({m: db(res[m]) for m in KEEP_A}, 'Output SINR (dB)', xlabel='Mismatch angle (degrees)',
                        x=angs)
    save(fig, 'Fig1_16_SINR_vs_mismatch')
    RESULTS['Fig1.16'] = {m: dict(zip([int(v) for v in angs[::5]], np.round(db(res[m])[::5], 2).tolist()))
                          for m in KEEP_A}


# =========================================================================== PART B: IET 2008
BASE_B = dict(noise_db=0.0, mis_deg=5.0, zeta=1.8, alpha=0.1)
MARKS_B = [(0.0, 'SOI', 'soi'), (np.pi / 6, 'I1', 'jam'), (np.pi / 3, 'I2', 'jam')]


def wc_param_plot(log, name, show_zeta=False):
    L = np.array([[r[0], r[1], r[2], r[3], r[4]] for r in log], float)
    k = np.arange(1, len(L) + 1)
    act = np.array([r[5] != 'inactive' for r in log])
    ttl = [r'$A$', r'$B$', r'$C$', r'$\lambda$']
    ncol = 3 if show_zeta else 2
    fig, axs = plt.subplots(2, ncol, figsize=(7.4 if show_zeta else 6.0, 4.6))
    axs = axs.ravel()
    for q in range(4):
        ax = axs[q]
        y = np.where(act, L[:, q], np.nan)
        ax.plot(k, y, color=C_BLUE, lw=1.4)
        yv = y[19:][np.isfinite(y[19:])]
        if yv.size:                              # scale to snapshots 20..N; the start-up values (R ~ I) are off-scale
            lo, hi = yv.min(), yv.max()
            pad = 0.08 * (hi - lo if hi > lo else abs(hi) + 1e-30)
            ax.set_ylim(lo - pad, hi + pad)
        if not np.isfinite(y).any():
            ax.text(0.5, 0.5, 'no admissible root\n' + r'($\zeta \geq \|\mathbf{a}\|$: infeasible)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9, color=INK2)
            ax.set_yticks([])
        ax.set_xlim(1, len(k))
        ax.set_title(ttl[q], loc='left', fontsize=11, color=INK)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
    for ax in (axs[2:5] if show_zeta else axs[2:4]):
        ax.set_xlabel('Snapshot index')
    if show_zeta:
        kz = min(40, len(k))                     # zeta settles within a few snapshots: show the start
        axs[4].step(k[:kz], L[:kz, 4], where='post', color=C_ORANGE, lw=1.8)
        axs[4].axhline(np.sqrt(M), color=INK2, lw=0.9, ls=(0, (2, 2)))
        axs[4].text(kz, np.sqrt(M), r'$\|\mathbf{a}\|=\sqrt{5}$', ha='right', va='bottom', fontsize=8, color=INK2)
        axs[4].set_title(r'$\zeta$', loc='left', fontsize=11, color=INK)
        axs[4].set_xlim(1, kz)
        axs[4].set_xlabel('Snapshot index')
        axs[5].axis('off')
    fig.tight_layout()
    save(fig, name)


def fig_1_17():
    """Fig 1.17: SINR vs snapshot"""
    acc, o = cached('iet_mc', iet_mc, RUNS_IET, base_seed=1000, **BASE_B)
    fig, _ = curve_plot({m: db(msmooth(acc[m])) for m in WC_METHODS}, 'Output SINR (dB)')
    save(fig, 'Fig1_17_WC_SINR')
    RESULTS['Fig1.17'] = {m: ss(acc[m]) for m in WC_METHODS}
    RESULTS['Fig1.17_k10'] = {m: round(float(db(acc[m][9])), 2) for m in WC_METHODS}


def fig_1_18():
    """Fig 1.18: steady-state beampatterns (one run)"""
    o = cached('iet_run1000', iet_run, 1000, **BASE_B)
    pattern_plot(o['_w'], 'Fig1_18_WC_beampattern', MARKS_B)


def n50_sweep(noise):
    res = {m: np.zeros(len(noise)) for m in WC_METHODS}
    for i, nd in enumerate(noise):
        for r in range(RUNS_N50):
            o = iet_run(3000 + 101 * i + r, **dict(BASE_B, noise_db=nd, N=50, final_only=True))
            for m in WC_METHODS:
                res[m][i] += o['sinr'][m][-1] / RUNS_N50
    return res


def fig_1_19():
    """Fig 1.19: SINR vs noise power, N = 50 snapshots"""
    noise = np.arange(20, -31, -2)
    res = cached('Fig1_19', n50_sweep, noise)
    fig, ax = curve_plot({m: db(res[m]) for m in WC_METHODS}, 'Output SINR (dB)', xlabel='Noise power (dB)', x=noise)
    ax.invert_xaxis()
    save(fig, 'Fig1_19_WC_SINR_vs_noise_N50')
    RESULTS['Fig1.19'] = {m: dict(zip([int(v) for v in noise[::5]], np.round(db(res[m])[::5], 2).tolist()))
                          for m in WC_METHODS}


def fig_1_20_22_23():
    """Figs 1.20, 1.22, 1.23: WC parameters (single run)"""
    for z, name in [(1.8, 'Fig1_20_WC_params_zeta1p8'), (1.2, 'Fig1_22_WC_params_zeta1p2'),
                    (2.4, 'Fig1_23_WC_params_zeta2p4')]:
        o = cached(name, iet_run, 1000, **dict(BASE_B, zeta=z), methods=[WCA], log_params=True)
        wc_param_plot(o['log'], name)
        st = {}
        for r in o['log']:
            st[r[5]] = st.get(r[5], 0) + 1
        first = next((i + 1 for i, r in enumerate(o['log']) if r[5] != 'inactive'), None)
        RESULTS['Fig1.' + name[5:7]] = {'status counts': st, 'first WC snapshot': first,
                                        'final SINR dB': round(float(db(o['sinr'][WCA][-1])), 2)}


def fig_1_21():
    """Fig 1.21: effect of zeta (sequential colour ramp; the infeasible value in gray)"""
    zs = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    raw = {}
    for z in zs:
        acc, _ = cached(f'Fig1_21_zeta{z}', iet_mc, RUNS_IET, base_seed=4000, **dict(BASE_B, zeta=z), methods=[WCA])
        raw[z] = acc[WCA]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    feas = [z for z in zs if z < np.sqrt(M)]
    cmap = plt.get_cmap('Blues')
    mk = ['o', 's', '^', 'D', 'v', 'P']
    for j, z in enumerate(feas):
        y = db(msmooth(raw[z]))
        ax.plot(np.arange(1, len(y) + 1), y, color=cmap(0.35 + 0.6 * j / max(len(feas) - 1, 1)), lw=1.7,
                marker=mk[j % len(mk)], markevery=((50 + 13 * j) % 100, 100), ms=5, mfc='white', mew=1.2,
                label=fr'$\zeta$ = {z}')
    for z in zs:
        if z >= np.sqrt(M):
            y = db(msmooth(raw[z]))
            ax.plot(np.arange(1, len(y) + 1), y, color=NEUTRAL, lw=1.4, ls=(0, (5, 2)),
                    label=fr'$\zeta$ = {z} (infeasible)')
    ax.set_xlabel('Snapshot index'); ax.set_ylabel('Output SINR (dB)')
    ax.set_xlim(0, len(raw[zs[0]]))
    legend_below(ax, len(zs) + 1)
    save(fig, 'Fig1_21_WC_effect_of_zeta')
    RESULTS['Fig1.21'] = {f'ζ = {z}': ss(raw[z]) for z in zs}


def fig_1_24_25():
    """Figs 1.24, 1.25: modified algorithm starting at zeta = 3"""
    o = cached('Fig1_24', iet_run, 1000, **dict(BASE_B, zeta=3.0), methods=[WCA], log_params=True, modified=True)
    wc_param_plot(o['log'], 'Fig1_24_WC_params_modified_zeta3', show_zeta=True)
    zt = [r[4] for r in o['log']]
    acc, _ = cached('Fig1_25', iet_mc, RUNS_IET, base_seed=1000, **dict(BASE_B, zeta=3.0), methods=[WCA], modified=True)
    fig, ax = curve_plot({WCA: db(msmooth(acc[WCA]))}, 'Output SINR (dB)', legend=False)
    save(fig, 'Fig1_25_WC_SINR_modified')
    RESULTS['Fig1.24'] = {'zeta start': 3.0, 'zeta after first snapshot': zt[0], 'zeta final': round(zt[-1], 2),
                          'snapshots until zeta settles': int(np.argmax(np.array(zt) == zt[-1])) + 1}
    RESULTS['Fig1.25'] = {WCA + ' (modified)': ss(acc[WCA])}


def fig_1_26():
    """Fig 1.26: small angular spread"""
    m26 = ['Standard MVDR', 'Benchmark MVDR', 'Robust MVDR-WC/SOCP', 'Robust MVDR-WC/EigDec', WCA, 'Benchmark LCMV']
    acc, _ = cached('Fig1_26', iet_mc, RUNS_IET, base_seed=5000, **BASE_B, methods=m26,
                    multipath=dict(doas=[4.0, -3.0], amp=0.4, random_phase=False, bench='composite'))
    fig, _ = curve_plot({m: db(msmooth(acc[m])) for m in m26}, 'Output SINR (dB)')
    save(fig, 'Fig1_26_WC_small_angular_spread')
    RESULTS['Fig1.26'] = {m: ss(acc[m]) for m in m26}


def fig_1_27():
    """Fig 1.27: large angular spread with MBWC"""
    m27 = ['Benchmark MVDR', 'Standard LCMV', 'Robust MVDR-WC/SOCP', 'Robust MVDR-WC/EigDec', WCA,
           'Robust LCMV-MBWC', 'Benchmark LCMV']
    acc, o = cached('Fig1_27', iet_mc, RUNS_IET, base_seed=6000, **BASE_B, methods=m27,
                    multipath=dict(doas=[-30.0, -80.0], amp=0.4, random_phase=True),
                    mbwc=dict(ups=[1.6, 0.2, 0.2], ivec=[1.0, 0.5, 0.5], alpha=0.3))
    fig, _ = curve_plot({m: db(msmooth(acc[m])) for m in m27}, 'Output SINR (dB)')
    save(fig, 'Fig1_27_WC_large_angular_spread_MBWC')
    RESULTS['Fig1.27'] = {m: ss(acc[m]) for m in m27}


PART_A = [fig_1_5_9, fig_1_10, fig_1_11_12, fig_1_13_15, fig_1_16]
PART_B = [fig_1_17, fig_1_18, fig_1_19, fig_1_20_22_23, fig_1_21, fig_1_24_25, fig_1_26, fig_1_27]


def main():
    for f in PART_A + PART_B:
        t = time.time()
        f()
        print(f'{f.__name__}: {time.time() - t:.0f} s')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'results_summary.json'), 'w') as f:
        json.dump(RESULTS, f, indent=1, default=str)


if __name__ == '__main__':
    main()
