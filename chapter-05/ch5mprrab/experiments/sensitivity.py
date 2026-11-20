"""Sensitivity and stress studies of MPR-RAB on the linear array (Section 5.7.7, Table 5.12, Figure 5.9; independent
seeds):  python -m ch5mprrab.experiments.sensitivity [NR] [base]
Each study varies one design constant with the others at their default values (selected kappa, max-SINR targets).
Writes DATA/sensitivity.json: per study and setting, per scenario, the list over runs of [expected SINR, minimum-cluster
SINR] (mean of the last 50 snapshots).
Studies: sector (Delta_s as a multiple of the 3-dB beamwidth), eta, T_rec, window (refinement half-width), deflation
(INR 35), notrack (tracking off, initial pointing error D, margin kappa), cross (interferer sweeping through a
cluster), offgrid (interferers between the points of the reconstruction grid)."""
import os, sys, json, time
import numpy as np
from . import _common  # noqa: F401
from .. import channel as chn
from .. import algorithms as alg
from .. import scenarios as MS

SEL = MS.load_selection('ula')
KAP = MS.kappa_of(SEL['MPR-RAB'])
M = chn.M_DEFAULT
GEOS = list('ABCDEFGHI')
STUDIES = {'sector': [0.25, 0.5, 1.0, 1.5], 'eta': [0.95, 0.97, 0.99], 'T_rec': [1, 5, 10, 30], 'window': [2.0, 3.0, 5.0]}


def sect_of(fam):
    return float(SEL[fam].split('s=')[1])


def run(d, kappa=KAP, tracking=True, **kw):
    c = d['c']
    h, log = alg.run_mpr_rab(d['X'], d['a0'], list(d['th0']), K_track=max(c['K_track'], len(d['th0'])), K_use=c['K_use'],
                               use_tracking=tracking, use_lifecycle=c['lc'], birth_hints=d['bh'], kappa=kappa, targets='ms', **kw)
    return h, log


def met(h, d, sc):
    m = MS.metrics(h, d, sc)
    return [m[''], m['|floor']]


def main():
    NR = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    base = int(sys.argv[2]) if len(sys.argv) > 2 else 51000
    fn = os.path.join(MS.DATA, 'sensitivity.json')
    os.makedirs(MS.DATA, exist_ok=True)
    MS.validate_cache(fn, dict(study='sensitivity', runs=NR, base=base, selection=SEL))
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    t0 = time.time()
    for study, settings in STUDIES.items():
        if study in out:
            continue
        res = {}
        for sc in GEOS:
            for r in range(NR):
                d = MS.make_data(sc, 5, 10, 0, base + 10000 * GEOS.index(sc) + r)
                for s in settings:
                    saved = (alg.EXP_RAD, alg.WIN_DEG)
                    kw = {}
                    if study == 'sector':
                        alg.EXP_RAD = s * alg.geometry(M).bw
                    elif study == 'eta':
                        kw['eta'] = s
                    elif study == 'T_rec':
                        kw['T_rec'] = s
                    elif study == 'window':
                        alg.WIN_DEG = s
                    try:
                        h, _ = run(d, **kw)
                    finally:
                        alg.EXP_RAD, alg.WIN_DEG = saved
                    res.setdefault(str(s), {}).setdefault(sc, []).append(met(h, d, sc))
        out[study] = res
        json.dump(out, open(fn, 'w'))
        print('%s done %.0f s' % (study, time.time() - t0), flush=True)
    if 'deflation' not in out:
        res = {}
        for sc in 'CEHI':
            for r in range(NR):
                d = MS.make_data(sc, 5, 35, 0, base + 10000 * GEOS.index(sc) + 500 + r)
                for tau in (0.0, 0.3, 0.5):
                    h, _ = run(d, defl=tau)
                    res.setdefault(str(tau), {}).setdefault(sc, []).append(met(h, d, sc))
        out['deflation'] = res
        json.dump(out, open(fn, 'w'))
        print('deflation done %.0f s' % (time.time() - t0), flush=True)
    if 'notrack' not in out:
        res = {}
        for sc in 'ABFGH':
            for D in (0, 2, 4):
                for r in range(NR):
                    d = MS.make_data(sc, 5, 10, D, base + 10000 * GEOS.index(sc) + 100 * D + 900 + r)
                    for kap in (0.1, 0.3, 0.5, 0.7, 0.9):
                        h, _ = run(d, kappa=kap, tracking=False)
                        res.setdefault('D=%d k=%g' % (D, kap), {}).setdefault(sc, []).append(met(h, d, sc))
                    dirs = [list(d['th0'])] * d['T']                 # the wrong directions, held
                    keys = [('LCMV', sect_of('INCM-LCMV')), ('MVDR', sect_of('INCM-MVDR')), ('GEV', sect_of('INCM-GEV'))]
                    mo = MS.incm_multi(d['X'], dirs, None, keys)
                    for key in keys:
                        res.setdefault('D=%d INCM-%s' % (D, key[0]), {}).setdefault(sc, []).append(met(mo[key], d, sc))
        out['notrack'] = res
        json.dump(out, open(fn, 'w'))
        print('notrack done %.0f s' % (time.time() - t0), flush=True)
    # interferer sweeping through a cluster: clusters at -20 and +20 deg (powers 1, 1), interferer at -60 deg fixed,
    # second interferer moving from +40 deg to 0 deg over 300 snapshots (it crosses the +20 deg cluster at t = 150)
    if 'cross' not in out:
        res = {}
        for r in range(NR):
            chn.PS = chn.SIGMA2 * 10 ** 0.5
            chn.PIV = chn.SIGMA2 * 10 ** 1.0
            np.random.seed(base + 90000 + r)
            ch = chn.MultiClusterChannel(M, theta_k_deg=[-20, 20], power_k=[1.0, 1.0], theta_int_deg=[-60, 40], drift_rate_deg_per_snap=0.0)
            T = 300
            X = np.zeros((M, T), dtype=complex)
            thl, pwl, Rl = [], [], []
            for k in range(T):
                ch.theta_int[1] = np.deg2rad(40 - 40 * k / (T - 1))
                x, _, tn = ch.snapshot(k)
                X[:, k] = x
                thl.append(list(tn)); pwl.append(list(ch.power_k)); Rl.append(ch.true_R_ipn())
            th0 = [np.deg2rad(t) for t in (-20, 20)]
            a0 = sum(chn.steer(M, t) for t in th0); a0 = a0 / np.linalg.norm(a0)
            for tau in (0.0, 0.3):
                h, log = alg.run_mpr_rab(X, a0, list(th0), K_track=2, K_use=2, use_tracking=True, use_lifecycle=False,
                                           kappa=KAP, targets='ms', defl=tau)
                s = [chn.sinr_db_cluster(h[k], thl[k], pwl[k], Rl[k], chn.PS) for k in range(T)]
                err = [float(np.mean(MS.dir_errors(log['theta_hat'][k], thl[k]))) for k in range(T)]
                res.setdefault('tau=%g' % tau, []).append({'sinr': s, 'err': err})
            orc = [chn.sinr_db_cluster(chn.run_oracle_mc(Rl[k], thl[k], pwl[k], M), thl[k], pwl[k], Rl[k], chn.PS) for k in range(T)]
            res.setdefault('oracle', []).append({'sinr': orc})
        out['cross'] = res
        json.dump(out, open(fn, 'w'))
        print('cross done %.0f s' % (time.time() - t0), flush=True)
    # interferers off the reconstruction grid (0.25 deg from the 0.5 deg grid), INR 10 and 35, every scenario
    if 'offgrid' not in out:
        res = {}
        for sc in GEOS:
            for inr in (10, 35):
                for r in range(NR):
                    seed = base + 10000 * GEOS.index(sc) + 700 + 50 * (inr == 35) + r
                    d = MS.make_data(sc, 5, inr, 0, seed)
                    # same data with the interferers moved to +45.25 and -60.25 deg: regenerate with the new directions
                    chn.PS = chn.SIGMA2 * 10 ** 0.5
                    chn.PIV = chn.SIGMA2 * 10 ** (inr / 10)
                    c = MS.SCEN[sc]
                    np.random.seed(seed)
                    ch = c['fac']()
                    ch.theta_int = [np.deg2rad(45.25), np.deg2rad(-60.25)]
                    T = c['T']
                    X = np.zeros((M, T), dtype=complex); thl, pwl = [], []
                    p_full = list(ch.power_k)
                    for k in range(T):
                        if sc == 'I':
                            ch.power_k = list(p_full)
                            if MS.BLOCK[0] <= k < MS.BLOCK[1]:
                                ch.power_k[0] = 0.0
                        x, _, tn = ch.snapshot(k)
                        X[:, k] = x; thl.append(list(tn)); pwl.append(list(ch.power_k))
                    d.update(X=X, thl=thl, pwl=pwl, R_ipn=ch.true_R_ipn(), ch=ch)
                    h, log = run(d)
                    dirs, pows = log['theta_hat'], log['p']
                    saved = alg.PEAK_REFINE
                    alg.PEAK_REFINE = False
                    try:
                        hn, _ = run(d)
                    finally:
                        alg.PEAK_REFINE = saved
                    keys = [('LCMV', sect_of('INCM-LCMV')), ('MVDR', sect_of('INCM-MVDR')), ('GEV', sect_of('INCM-GEV'))]
                    mo = MS.incm_multi(X, dirs, pows, keys)
                    eps = float(SEL['Par-K-RCB'].split('=')[1]); gnf = float(SEL['LCMV-K noise-floor-loaded'].split('=')[1])
                    ent = {'MPR-RAB': h, 'MPR-RAB without peak refinement': hn,
                           'INCM-LCMV': mo[keys[0]], 'INCM-MVDR': mo[keys[1]], 'INCM-GEV': mo[keys[2]],
                           'Par-K-RCB': MS.parrcb_multi(X, dirs, [eps])[eps], 'LCMV-K': MS.lcmv_rule(X, dirs, 'nf', gnf),
                           'Oracle': [chn.run_oracle_mc(d['R_ipn'], thl[k], pwl[k], M) for k in range(T)]}
                    for n, hh in ent.items():
                        res.setdefault('INR=%d' % inr, {}).setdefault(n, []).append(met(hh, d, sc))
        out['offgrid'] = {k: {n: np.mean(v, axis=0).tolist() for n, v in dd.items()} for k, dd in res.items()}
        json.dump(out, open(fn, 'w'))
        print('offgrid done %.0f s' % (time.time() - t0), flush=True)
    print('SENS DONE', flush=True)


if __name__ == '__main__':
    main()
