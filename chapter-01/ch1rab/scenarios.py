"""
ch1rab.scenarios -- Monte Carlo scenario runners for the Chapter 1 figures.
TAP 2006 part (ellipsoidal/joint constraints) and IET 2008 part (WC / MBWC constraints).
"""
import numpy as np
from .algorithms import (steer, sinr_single, sinr_coherent, capon, soi_power, RLS, VLRCB, rcb_eig,
                 RCBSubspace, WCSOCP, wc_eig, WCAdaptive, MBWCAdaptive, lcmv)

TAP_METHODS = ['Standard Capon', 'Robust Capon (Batch)', 'Robust Capon (SS)', 'Robust (SOCP)',
               'Proposed1', 'Proposed2']


# =========================================================================== TAP 2006
def tap_run(seed, *, M=5, N=1000, noise_db=-50.0, mis=0.03 * np.pi, eps=0.5, delta_socp=None,
            alpha=0.008, method='SD', tau=0.2, eta=1.0, socp_every=10, traj=None,
            final_only=False, jam=(np.pi / 4, np.pi / 6)):
    """One run of the ellipsoidal-constraint scenario (TAP 2006 scenarios).
    traj(k) -> (theta2, theta3) for moving jammers (k = 1..N); None = static.
    Returns dict of per-snapshot SINR / MSE / SOIP (or final SINR if final_only) and final weights."""
    rng = np.random.default_rng(seed)
    A1, A2, A3 = 1.0, 10 ** 0.5, 10 ** 0.5
    k = np.arange(1, N + 1)
    s1 = A1 * np.cos(2 * np.pi * 0.01 * k)
    s2 = A2 * np.sign(np.cos(2 * np.pi * 0.02 * k))
    s3 = A3 * np.sign(rng.standard_normal(N))
    na = 10 ** (noise_db / 20) / np.sqrt(2)
    V = na * (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N)))
    a_true = steer(0.0, M)
    a0 = steer(mis, M)                                  # presumed (mismatched) steering vector
    if delta_socp is None:
        delta_socp = np.sqrt(eps)                       # same uncertainty radius as the RCB sphere

    def jam_at(kk):
        return traj(kk) if traj is not None else jam

    rls = RLS(M, eta)
    vl = VLRCB(a0, eps, alpha, method, tau)
    ss = RCBSubspace(M, 3, rng)
    socp = WCSOCP(M)
    names = TAP_METHODS
    out = {m: {'sinr': np.zeros(N), 'mse': np.zeros(N), 'soip': np.zeros(N)} for m in names}
    wo = a0 / M
    for i in range(N):
        t2, t3 = jam_at(k[i])
        P = np.column_stack([a_true, steer(t2, M), steer(t3, M)])
        x = P @ np.array([s1[i], s2[i], s3[i]]) + V[:, i]
        rls.update(x)
        R, IR = rls.R, rls.IR
        wr, wj = vl.update(IR)
        if wj is None:
            wj = wr                                     # joint constraint not simulated
        ws, _ = ss.update(x, R, IR, a0, eps)
        last = (i == N - 1)
        if final_only and not last:
            continue
        wp = capon(IR, a0)
        wd, _ = rcb_eig(R, IR, a0, eps)
        if (i % socp_every == 0) or last:
            wo = socp.solve(R, a0, delta_socp)
        W = dict(zip(names, [wp, wd, ws, wo, wr, wj]))
        H = np.column_stack([A1 * a_true, A2 * P[:, 1], A3 * P[:, 2]])
        for m, w in W.items():
            out[m]['sinr'][i] = sinr_single(w, H, noise_db)
            if not final_only:
                out[m]['mse'][i] = (np.real(np.vdot(w, x)) - s1[i]) ** 2
                out[m]['soip'][i] = soi_power(w, rls.Rn)
    out['_w'] = W
    out['_vl_count'] = vl.n_vl
    return out


def tap_mc(runs, base_seed=0, **kw):
    """Average linear SINR / MSE / SOIP over Monte Carlo runs."""
    acc = None
    for r in range(runs):
        o = tap_run(base_seed + r, **kw)
        if acc is None:
            acc = {m: {q: np.zeros_like(v) for q, v in o[m].items()} for m in TAP_METHODS}
        for m in TAP_METHODS:
            for q in acc[m]:
                acc[m][q] += o[m][q] / runs
    acc['_w_last'] = o['_w']
    return acc


# =========================================================================== IET 2008
WC_METHODS = ['Standard MVDR', 'Robust MVDR-WC/SOCP', 'Robust MVDR-WC/EigDec',
              'Robust MVDR-WC/Proposed', 'Benchmark MVDR']


def iet_run(seed, *, M=5, N=1000, noise_db=0.0, mis_deg=5.0, zeta=1.8, zeta_socp=None,
            alpha=0.1, eta=1.0, socp_every=10, jam=(np.pi / 6, np.pi / 3),
            multipath=None, mbwc=None, methods=None, final_only=False, log_params=False,
            modified=False):
    """One run of the WC scenarios (IET 2008 scenarios).
    multipath: None, or dict(doas=[deg, deg], amp=0.4, random_phase=bool).
    mbwc: None, or dict(ups=[...], ivec=[...], alpha=...)."""
    rng = np.random.default_rng(seed)
    A1, A2, A3 = 10 ** (5 / 20), 10 ** 0.5, 10 ** 0.5
    k = np.arange(1, N + 1)
    s1 = A1 * np.cos(2 * np.pi * 0.03 * k)
    s2 = A2 * np.sign(np.cos(2 * np.pi * 0.02 * k))
    s3 = A3 * np.sign(rng.standard_normal(N))
    na = 10 ** (noise_db / 20) / np.sqrt(2)
    V = na * (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N)))
    a_main = steer(0.0, M)
    aj = [steer(jam[0], M), steer(jam[1], M)]
    a0 = steer(np.deg2rad(mis_deg), M)                  # presumed main-ray steering vector
    zeta_socp = zeta if zeta_socp is None else zeta_socp

    if multipath is None:
        a_eff = a_main
        rays_true = [a_main]
    else:
        ph = rng.uniform(-np.pi, np.pi, 3) if multipath.get('random_phase') else np.zeros(3)
        amp = multipath.get('amp', 0.4)
        rays_true = [np.exp(1j * ph[0]) * a_main] + \
                    [np.exp(1j * ph[f + 1]) * steer(np.deg2rad(d), M) for f, d in enumerate(multipath['doas'])]
        a_eff = rays_true[0] + amp * (rays_true[1] + rays_true[2])
        C_true = np.column_stack(rays_true)
        C_pres = np.column_stack([a0] + [steer(np.deg2rad(d), M) for d in multipath['doas']])
        g = np.array([1.0, amp, amp])
    interf = [A2 * aj[0], A3 * aj[1]]

    rls = RLS(M, eta)
    wca = WCAdaptive(a0, zeta, alpha, modified=modified)
    mb = None
    if mbwc is not None:
        mb = MBWCAdaptive(C_pres, mbwc['ups'], mbwc['ivec'], a0, mbwc.get('alpha', 0.3))
    socp = WCSOCP(M)
    names = methods or WC_METHODS
    sinr = {m: np.zeros(N) for m in names}
    wo = a0 / M
    for i in range(N):
        x = a_eff * s1[i] + aj[0] * s2[i] + aj[1] * s3[i] + V[:, i]
        rls.update(x)
        R, IR = rls.R, rls.IR
        wd = wca.update(R, IR)
        wm = mb.update(R, IR) if mb is not None else None
        last = (i == N - 1)
        if final_only and not last:
            continue
        W = {}
        if 'Standard MVDR' in names:
            W['Standard MVDR'] = capon(IR, a0)
        if 'Benchmark MVDR' in names:
            W['Benchmark MVDR'] = capon(IR, a_main)
        if 'Robust MVDR-WC/EigDec' in names:
            W['Robust MVDR-WC/EigDec'] = wc_eig(R, a0, zeta)[0]
        if 'Robust MVDR-WC/SOCP' in names:
            if (i % socp_every == 0) or last:
                wo = socp.solve(R, a0, zeta_socp)
            W['Robust MVDR-WC/SOCP'] = wo
        if 'Robust MVDR-WC/Proposed' in names:
            W['Robust MVDR-WC/Proposed'] = wd
        if 'Robust LCMV-MBWC' in names:
            W['Robust LCMV-MBWC'] = wm
        if 'Benchmark LCMV' in names:
            if multipath.get('bench') == 'composite':
                # closely spaced rays: the 3-constraint LCMV is ill-conditioned on a 5-element array,
                # so the benchmark uses the true composite (MRC-combined) steering vector
                W['Benchmark LCMV'] = capon(IR, a_eff)
            else:
                W['Benchmark LCMV'] = lcmv(IR, C_true, g)
        if 'Standard LCMV' in names:
            W['Standard LCMV'] = lcmv(IR, C_pres, g)
        for m, w in W.items():
            sinr[m][i] = sinr_coherent(w, a_eff, A1 ** 2, interf, noise_db)
    out = {'sinr': sinr, '_w': W}
    if log_params:
        out['log'] = wca.log
    if mb is not None:
        out['mbwc_fail'] = mb.fail
    return out


def iet_mc(runs, base_seed=1000, **kw):
    acc = None
    for r in range(runs):
        o = iet_run(base_seed + r, **kw)
        if acc is None:
            acc = {m: np.zeros_like(v) for m, v in o['sinr'].items()}
        for m in acc:
            acc[m] += o['sinr'][m] / runs
    return acc, o
