"""
ch1rab.figures -- generates every simulation figure of Chapter 1 (Figs. 1.5-1.27) plus two diagnostics.
Configured through environment variables read at import time: QUICK=1 (smoke test), FIGDIR (output
folder, default ./figures), RAB_INLINE=1 (show figures in Jupyter).
Used by the notebook and by scripts/generate_figures.py.
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .scenarios import tap_mc, tap_run, iet_mc, iet_run, TAP_METHODS, WC_METHODS
from .algorithms import beampattern

QUICK = os.environ.get('QUICK') == '1'
RUNS_TAP = 3 if QUICK else 50        # chapter: 50 runs for the ellipsoidal scenarios
RUNS_IET = 3 if QUICK else 100       # chapter: 100 runs for the WC scenarios
RUNS_SWEEP = 2 if QUICK else 50      # Monte Carlo runs per point of SINR-vs-noise / mismatch sweeps
RUNS_N50 = 3 if QUICK else 300       # Fig. 1.19: N = 50 snapshots per run, so more runs per point
OUT = os.environ.get('FIGDIR', 'figures')
INLINE = os.environ.get('RAB_INLINE') == '1'
M = 5
RESULTS = {}
KEEP_A = [m for m in TAP_METHODS if m != 'Proposed2']     # joint beamformer omitted where the chapter omits it

STYLE = {
    'Standard Capon': dict(color='0.45', ls='-', marker='s'),
    'Robust Capon (Batch)': dict(color='tab:green', ls='--', marker='*'),
    'Robust Capon (SS)': dict(color='tab:blue', ls='-.', marker='D'),
    'Robust (SOCP)': dict(color='tab:purple', ls=':', marker='o'),
    'Proposed1': dict(color='k', ls='-', marker='.'),
    'Proposed2': dict(color='tab:red', ls='-', marker='x'),
    'Standard MVDR': dict(color='0.45', ls='-', marker='d'),
    'Robust MVDR-WC/SOCP': dict(color='tab:purple', ls=':', marker='o'),
    'Robust MVDR-WC/EigDec': dict(color='tab:green', ls='--', marker='^'),
    'Robust MVDR-WC/Proposed': dict(color='k', ls='-', marker='*'),
    'Benchmark MVDR': dict(color='tab:blue', ls='-.', marker='s'),
    'Benchmark LCMV': dict(color='tab:orange', ls='--', marker='v'),
    'Standard LCMV': dict(color='tab:brown', ls='-', marker='h'),
    'Robust LCMV-MBWC': dict(color='tab:red', ls='-', marker='p'),
}
LABEL = {'Proposed1': 'Proposed (ellipsoidal)', 'Proposed2': 'Proposed (joint)'}


def db(x):
    return 10 * np.log10(np.maximum(np.abs(x), 1e-30))


def save(fig, name):
    fig.tight_layout()
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


def curve_plot(curves, ylabel, name, xlabel='Snapshot', x=None, every=100, ref=None, legend_loc='best',
               ylim=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m, y in curves.items():
        st = STYLE.get(m, {})
        xx = np.arange(1, len(y) + 1) if x is None else x
        me = every if x is None else 1
        ax.plot(xx, y, lw=1.3, markevery=me, ms=5, label=LABEL.get(m, m), **st)
    if ref is not None:
        ax.axhline(ref[0], color='0.6', lw=0.8, ls='--', label=ref[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, loc=legend_loc)
    return fig, ax


def pattern_plot(W, name, marks, title=None):
    th = np.deg2rad(np.arange(-90, 90.25, 0.25))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m, w in W.items():
        st = dict(STYLE.get(m, {}))
        st.pop('marker', None)
        ax.plot(th, beampattern(w, M, th), lw=1.3, label=LABEL.get(m, m), **st)
    for t in marks:
        ax.axvline(t, color='0.5', lw=0.8, ls=':')
    ax.set_xlabel('Angle (radian)')
    ax.set_ylabel('Beampattern (dB)')
    ax.set_ylim(-60, 2)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, loc='lower left')
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
    half-width grows as frac*k up to hmax snapshots (one-sided near the end), so the start-up transient is not smeared while
    the steady-state ripple (SOI / jammer waveform periods) is averaged out. Numbers quoted in the
    text are computed from the unsmoothed Monte Carlo means."""
    y = np.asarray(y, float)
    n = len(y)
    c = np.concatenate([[0.0], np.cumsum(y)])
    out = np.empty(n)
    for k in range(n):
        h = min(int(frac * k), hmax)
        lo, hi = k - h, min(k + h, n - 1)        # one-sided near the last snapshot, so the end is not left raw
        out[k] = (c[hi + 1] - c[lo]) / (hi - lo + 1)
    return out


def ss(y, n=100):
    """steady-state value: mean of the last n points (dB of the linear mean)"""
    return float(db(np.mean(y[-n:])))


# =========================================================================== PART A: TAP 2006
def fig_1_5_9():
    """Scenario 1: Figs 1.5-1.9 (SD, 0.03pi, eps 0.5, noise -50 dB)"""
    sc1 = dict(noise_db=-50, mis=0.03 * np.pi, eps=0.5, delta_socp=socp_delta(0.5), alpha=0.008,
               method='SD', tau=0.2)
    acc = tap_mc(RUNS_TAP, base_seed=0, **sc1)
    fig, _ = curve_plot({m: db(msmooth(acc[m]['sinr'])) for m in TAP_METHODS}, 'SINR (dB)', '')
    save(fig, 'Fig1_05_SINR_scenario1')
    fig, _ = curve_plot({m: db(smooth(acc[m]['mse'])) for m in TAP_METHODS}, 'MSE (dB)', '')
    save(fig, 'Fig1_06_MSE_scenario1')
    fig, _ = curve_plot({m: db(acc[m]['soip']) for m in TAP_METHODS}, 'SOI power estimate (dB)', '',
                        ref=(0.0, 'True SOI power'))
    save(fig, 'Fig1_07_SOIP_scenario1')
    RESULTS['Fig1.5'] = {m: ss(acc[m]['sinr']) for m in TAP_METHODS}
    RESULTS['Fig1.6'] = {m: ss(acc[m]['mse']) for m in TAP_METHODS}
    RESULTS['Fig1.7'] = {m: ss(acc[m]['soip']) for m in TAP_METHODS}
    keep = KEEP_A
    marks = [0.0, np.pi / 6, np.pi / 4]
    for j, (seed, name) in enumerate([(0, 'Fig1_08_beampattern_a'), (1, 'Fig1_09_beampattern_b')]):
        o = tap_run(seed, **sc1)
        pattern_plot({m: o['_w'][m] for m in keep}, name, marks)
    # diagnostic: is the ellipsoidal constraint ever active?
    diag = {}
    for al in (0.008, 0.02, 0.05):
        o = tap_run(0, **dict(sc1, alpha=al))
        diag[al] = (o['_vl_count'], float(db(o['Proposed1']['sinr'][-1])))
    RESULTS['diag_VL_activity_scenario1'] = {f'alpha={k}': {'VL activations': v[0], 'final SINR dB': round(v[1], 2)}
                                             for k, v in diag.items()}


def fig_1_10():
    """Scenario 2: Fig 1.10 (CG, 0.06pi, eps 1, noise -20 dB)"""
    sc2 = dict(noise_db=-20, mis=0.06 * np.pi, eps=1.0, delta_socp=socp_delta(1.0), alpha=0.01,
               method='CG', tau=0.2)
    acc = tap_mc(RUNS_TAP, base_seed=100, **sc2)
    fig, _ = curve_plot({m: db(msmooth(acc[m]['sinr'])) for m in TAP_METHODS}, 'SINR (dB)', '')
    save(fig, 'Fig1_10_SINR_scenario2')
    RESULTS['Fig1.10'] = {m: ss(acc[m]['sinr']) for m in TAP_METHODS}
    o = tap_run(100, **sc2)
    RESULTS['diag_VL_activity_scenario2'] = {'VL activations (one run)': o['_vl_count']}


def fig_1_11_12():
    """Figs 1.11, 1.12: SINR vs noise power (final snapshot, N = 1000)"""
    noise = np.arange(30, -31, -4)
    for name, mis, al, key in [('Fig1_11_SINR_vs_noise_003pi', 0.03 * np.pi, 0.02, 'Fig1.11'),
                               ('Fig1_12_SINR_vs_noise_006pi', 0.06 * np.pi, 0.008, 'Fig1.12')]:
        res = {m: np.zeros(len(noise)) for m in TAP_METHODS}
        for i, nd in enumerate(noise):
            for r in range(RUNS_SWEEP):
                o = tap_run(500 + 37 * i + r, noise_db=nd, mis=mis, eps=1.0, delta_socp=socp_delta(1.0),
                            alpha=al, method='SD', tau=0.2, final_only=True)
                for m in TAP_METHODS:
                    res[m][i] += o[m]['sinr'][-1] / RUNS_SWEEP
        fig, ax = curve_plot({m: db(res[m]) for m in TAP_METHODS}, 'Output SINR (dB)', '',
                             xlabel='Noise power (dB)', x=noise)
        ax.invert_xaxis()
        save(fig, name)
        RESULTS[key] = {m: dict(zip([int(v) for v in noise[::3]], np.round(db(res[m])[::3], 2).tolist()))
                        for m in TAP_METHODS}


def fig_1_13_15():
    """Figs 1.13-1.15: moving jammers (noise -20 dB, 0.01pi mismatch, joint excluded)"""
    d2r = np.pi / 180
    moving = [('Fig1_13_moving_noncoherent', lambda k: (np.pi / 4 + 0.1 * d2r * k, np.pi / 6 + 0.01 * d2r * k)),
              ('Fig1_14_moving_coherent', lambda k: (np.pi / 4 + 0.1 * d2r * k, np.pi / 6 - 0.01 * d2r * k)),
              ('Fig1_15_moving_nonstationary', lambda k: (np.pi / 4 + 10 * d2r * np.sin(k / 15),
                                                          np.pi / 6 + 10 * d2r * np.sin(k / 15)))]
    keep = KEEP_A
    for j, (name, traj) in enumerate(moving):
        acc = tap_mc(RUNS_TAP, base_seed=200 + 100 * j, noise_db=-20, mis=0.01 * np.pi, eps=0.5,
                     delta_socp=socp_delta(0.5), alpha=0.01, method='SD', tau=None, traj=traj)
        fig, _ = curve_plot({m: db(msmooth(acc[m]['sinr'])) for m in keep}, 'SINR (dB)', '')
        save(fig, name)
        RESULTS['Fig1.' + name[5:7]] = {m: ss(acc[m]['sinr']) for m in keep}


def fig_1_16():
    """Fig 1.16: SINR vs mismatch angle (noise -15 dB, eps 1)"""
    angs = np.arange(0, 21, 1)
    keep = KEEP_A
    res = {m: np.zeros(len(angs)) for m in keep}
    for i, a in enumerate(angs):
        for r in range(RUNS_SWEEP):
            o = tap_run(900 + 41 * i + r, noise_db=-15, mis=np.deg2rad(a), eps=1.0, delta_socp=socp_delta(1.0),
                        alpha=0.008, method='SD', tau=None, final_only=True)
            for m in keep:
                res[m][i] += o[m]['sinr'][-1] / RUNS_SWEEP
    fig, _ = curve_plot({m: db(res[m]) for m in keep}, 'Output SINR (dB)', '', xlabel='Mismatch angle (degrees)', x=angs)
    save(fig, 'Fig1_16_SINR_vs_mismatch')
    RESULTS['Fig1.16'] = {m: dict(zip([int(v) for v in angs[::5]], np.round(db(res[m])[::5], 2).tolist())) for m in keep}



# =========================================================================== PART B: IET 2008
BASE_B = dict(noise_db=0.0, mis_deg=5.0, zeta=1.8, alpha=0.1)


def wc_param_plot(log, name, show_zeta=False):
    L = np.array([[r[0], r[1], r[2], r[3], r[4]] for r in log], float)
    k = np.arange(1, len(L) + 1)
    act = np.array([r[5] != 'inactive' for r in log])
    ttl = ['A', 'B', 'C', r'$\lambda$']
    n = 5 if show_zeta else 4
    fig, axs = plt.subplots(2, 3 if show_zeta else 2, figsize=(9 if show_zeta else 6.4, 4.8))
    axs = axs.ravel()
    for q in range(4):
        y = np.where(act, L[:, q], np.nan)
        axs[q].plot(k, y, 'b-', lw=1, marker='*', markevery=100, ms=5)
        yv = y[2:][np.isfinite(y[2:])]
        if yv.size:                              # scale to snapshots 3..N; the first activation (R ~ I) is off-scale
            lo, hi = yv.min(), yv.max()
            pad = 0.08 * (hi - lo if hi > lo else abs(hi) + 1e-30)
            axs[q].set_ylim(lo - pad, hi + pad)
        if not np.isfinite(y).any():
            axs[q].text(0.5, 0.5, 'no admissible root\n' + r'($\zeta \geq \|\mathbf{a}\|$: infeasible)',
                        ha='center', va='center', transform=axs[q].transAxes, fontsize=9)
            axs[q].set_xlim(1, len(k))
            axs[q].set_yticks([])
        axs[q].set_title(ttl[q])
        axs[q].set_xlabel('Snapshot')
        axs[q].grid(True, alpha=0.35)
        axs[q].ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
    if show_zeta:
        axs[4].plot(k, L[:, 4], 'k-', lw=1.2)
        axs[4].set_title(r'$\zeta$')
        axs[4].set_xlabel('Snapshot')
        axs[4].grid(True, alpha=0.35)
        axs[5].axis('off')
    save(fig, name)


def fig_1_17():
    """Fig 1.17: SINR vs snapshot"""
    acc, o = iet_mc(RUNS_IET, base_seed=1000, **BASE_B)
    fig, _ = curve_plot({m: db(msmooth(acc[m])) for m in WC_METHODS}, 'SINR (dB)', '')
    save(fig, 'Fig1_17_WC_SINR')
    RESULTS['Fig1.17'] = {m: ss(acc[m]) for m in WC_METHODS}
    RESULTS['Fig1.17_k10'] = {m: round(float(db(acc[m][9])), 2) for m in WC_METHODS}


def fig_1_18():
    """Fig 1.18: steady-state beampatterns (one run)"""
    o = iet_run(1000, **BASE_B)
    pattern_plot(o['_w'], 'Fig1_18_WC_beampattern', [0.0, np.pi / 6, np.pi / 3])


def fig_1_19():
    """Fig 1.19: SINR vs noise power, N = 50 snapshots"""
    noise = np.arange(20, -31, -2)
    res = {m: np.zeros(len(noise)) for m in WC_METHODS}
    for i, nd in enumerate(noise):
        for r in range(RUNS_N50):
            o = iet_run(3000 + 101 * i + r, **dict(BASE_B, noise_db=nd, N=50, final_only=True))
            for m in WC_METHODS:
                res[m][i] += o['sinr'][m][-1] / RUNS_N50
    fig, ax = curve_plot({m: db(res[m]) for m in WC_METHODS}, 'SINR (dB)', '', xlabel='Noise power (dB)', x=noise)
    ax.invert_xaxis()
    save(fig, 'Fig1_19_WC_SINR_vs_noise_N50')
    RESULTS['Fig1.19'] = {m: dict(zip([int(v) for v in noise[::5]], np.round(db(res[m])[::5], 2).tolist()))
                          for m in WC_METHODS}


def fig_1_20_22_23():
    """Figs 1.20, 1.22, 1.23: WC parameters (single run)"""
    for z, name in [(1.8, 'Fig1_20_WC_params_zeta1p8'), (1.2, 'Fig1_22_WC_params_zeta1p2'),
                    (2.4, 'Fig1_23_WC_params_zeta2p4')]:
        o = iet_run(1000, **dict(BASE_B, zeta=z), methods=['Robust MVDR-WC/Proposed'], log_params=True)
        wc_param_plot(o['log'], name)
        st = {}
        for r in o['log']:
            st[r[5]] = st.get(r[5], 0) + 1
        first = next((i + 1 for i, r in enumerate(o['log']) if r[5] != 'inactive'), None)
        RESULTS['Fig1.' + name[5:7]] = {'status counts': st, 'first WC snapshot': first,
                                                'final SINR dB': round(float(db(o['sinr']['Robust MVDR-WC/Proposed'][-1])), 2)}


def fig_1_21():
    """Fig 1.21: effect of zeta"""
    zs = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    curves, raw = {}, {}
    for z in zs:
        acc, _ = iet_mc(RUNS_IET, base_seed=4000, **dict(BASE_B, zeta=z), methods=['Robust MVDR-WC/Proposed'])
        raw[f'ζ = {z}'] = acc['Robust MVDR-WC/Proposed']
        curves[f'ζ = {z}'] = db(msmooth(acc['Robust MVDR-WC/Proposed']))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    mk = ['d', '*', '^', 'o', 's', '+', 'p']
    for j, (lab, y) in enumerate(curves.items()):
        ax.plot(np.arange(1, len(y) + 1), y, lw=1.2, marker=mk[j], markevery=100, ms=5, label=lab)
    ax.set_xlabel('Snapshot'); ax.set_ylabel('SINR (dB)'); ax.grid(True, alpha=0.35); ax.legend(fontsize=8)
    save(fig, 'Fig1_21_WC_effect_of_zeta')
    RESULTS['Fig1.21'] = {lab: ss(y) for lab, y in raw.items()}


def fig_1_24_25():
    """Figs 1.24, 1.25: modified algorithm starting at zeta = 3"""
    o = iet_run(1000, **dict(BASE_B, zeta=3.0), methods=['Robust MVDR-WC/Proposed'], log_params=True, modified=True)
    wc_param_plot(o['log'], 'Fig1_24_WC_params_modified_zeta3', show_zeta=True)
    zt = [r[4] for r in o['log']]
    acc, _ = iet_mc(RUNS_IET, base_seed=1000, **dict(BASE_B, zeta=3.0), methods=['Robust MVDR-WC/Proposed'], modified=True)
    fig, _ = curve_plot({'Robust MVDR-WC/Proposed': db(msmooth(acc['Robust MVDR-WC/Proposed']))}, 'SINR (dB)', '')
    save(fig, 'Fig1_25_WC_SINR_modified')
    RESULTS['Fig1.24'] = {'zeta start': 3.0, 'zeta after first snapshot': zt[0], 'zeta final': round(zt[-1], 2),
                          'snapshots until zeta settles': int(np.argmax(np.array(zt) == zt[-1])) + 1}
    RESULTS['Fig1.25'] = {'Robust MVDR-WC/Proposed (modified)': ss(acc['Robust MVDR-WC/Proposed'])}


def fig_1_26():
    """Fig 1.26: small angular spread"""
    m26 = ['Standard MVDR', 'Benchmark MVDR', 'Robust MVDR-WC/SOCP', 'Robust MVDR-WC/EigDec',
           'Robust MVDR-WC/Proposed', 'Benchmark LCMV']
    acc, _ = iet_mc(RUNS_IET, base_seed=5000, **BASE_B, methods=m26,
                    multipath=dict(doas=[4.0, -3.0], amp=0.4, random_phase=False, bench='composite'))
    fig, _ = curve_plot({m: db(msmooth(acc[m])) for m in m26}, 'SINR (dB)', '')
    save(fig, 'Fig1_26_WC_small_angular_spread')
    RESULTS['Fig1.26'] = {m: ss(acc[m]) for m in m26}


def fig_1_27():
    """Fig 1.27: large angular spread with MBWC"""
    m27 = ['Benchmark MVDR', 'Standard LCMV', 'Robust MVDR-WC/SOCP', 'Robust MVDR-WC/EigDec',
           'Robust MVDR-WC/Proposed', 'Robust LCMV-MBWC', 'Benchmark LCMV']
    acc, o = iet_mc(RUNS_IET, base_seed=6000, **BASE_B, methods=m27,
                    multipath=dict(doas=[-30.0, -80.0], amp=0.4, random_phase=True),
                    mbwc=dict(ups=[1.6, 0.2, 0.2], ivec=[1.0, 0.5, 0.5], alpha=0.3))
    fig, _ = curve_plot({m: db(msmooth(acc[m])) for m in m27}, 'SINR (dB)', '')
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
