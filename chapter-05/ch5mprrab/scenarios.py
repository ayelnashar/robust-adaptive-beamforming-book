"""
ch5mprrab.scenarios -- scenarios, protocol, metrics and selection of the numerical study of Chapter 5 (Section 5.7),
for the 16-element linear array ('ula') and the 8 x 8 planar array ('upa').

One process runs one array: use_array('upa') routes the steering model of ch5mprrab.channel and ch5mprrab.algorithms
to the planar array.

Phases (per array)
  train   : independent streams (seed = base + 10000*scenario + 100*condition + run), every MPR-RAB setting and every
            baseline parameter; one parameter per family is selected on the pooled expected SINR, and settings within
            TIE_DB of the best are treated as equivalent and resolved by the larger pooled minimum-cluster SINR
  heldout : the condition study: independent streams (base 2026000), every condition, NR runs per cell, paired
            bootstrap confidence intervals; its first condition is the operating point of the tables
  stream  : np.random.seed(42) once, scenarios in order, STREAM_NR runs per scenario, operating point only: the
            separate stream of the linear-array scenario C of Figure 5.7
Metrics per method: mean last-50 expected SINR (dB) and minimum-cluster SINR (worst of the K_use strongest true
clusters, key suffix '|floor'); scenario I adds the shadowing transient ('|dip', '|blk', '|rec').
Every direction-based method receives the directions of the constraints of MPR-RAB at every snapshot and the
Capon powers of the tracked clusters at its last update (the shared tracking stage), and nothing else.  No method
touches the global random stream, so every method in a cell sees identical data.

Results are written into DATA (environment variable CH5_DATA, default ./figures/data).
"""
import os, json, time, itertools
import numpy as np
from . import channel as chn
from . import algorithms as alg
from .baselines import lcmv_rule, rcb_weight, parrcb_multi, lcmv_kd_dirs, incm_multi  # noqa: F401

ARRAY = 'ula'
TIE_DB = 0.05
DATA = os.environ.get('CH5_DATA', os.path.join('figures', 'data'))
PKG_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# ----------------------------------------------------------------------------- ULA scenarios A..I
def _fac(theta, power, **kw):
    return lambda: chn.MultiClusterChannel(chn.M_DEFAULT, theta_k_deg=list(theta), power_k=list(power), **kw)


_thE, _pwE = chn.cdl_like_profile('A', K_clusters=5, seed=7)
_topE = list(np.argsort(-np.array(_pwE))[:3])
SCEN = {
    'A': dict(fac=_fac([0, 30], [1.0, 1.0]), th=[0, 30], T=300, K_track=2, K_use=2, trk=False, lc=False, bh=None),
    'B': dict(fac=_fac([-20, 5, 35], [1.0, 0.3, 0.8]), th=[-20, 5, 35], T=300, K_track=3, K_use=3, trk=False, lc=False, bh=None),
    'C': dict(fac=_fac([-10, 20], [1.0, 0.7], drift_rate_deg_per_snap=0.025), th=[-10, 20], T=400, K_track=2, K_use=2, trk=True, lc=False, bh=None),
    'D': dict(fac=lambda: chn.MultiClusterChannel(chn.M_DEFAULT, theta_k_deg=[0, 25], power_k=[1.0, 0.8],
                                                 lifecycle=[{'type': 'birth', 't': 150, 'theta': -30, 'power': 0.9}, {'type': 'death', 't': 280, 'idx': 0}]),
              th=[0, 25], T=400, K_track=4, K_use=3, trk=True, lc=True, bh=[(150, -30)]),
    'E': dict(fac=_fac(_thE, _pwE), th=[_thE[i] for i in _topE], T=400, K_track=3, K_use=3, trk=False, lc=False, bh=None),
    'F': dict(fac=_fac([-35, 0, 30], [1.0, 1.0, 1.0]), th=[-35, 0, 30], T=300, K_track=3, K_use=3, trk=False, lc=False, bh=None),
    'G': dict(fac=_fac([-40, -15, 5, 20, 35], [1.0] * 5), th=[-40, -15, 5, 20, 35], T=300, K_track=5, K_use=5, trk=False, lc=False, bh=None),
    'H': dict(fac=_fac([-30, -10, 15, 35], [1.0, 0.8, 1.0, 0.7]), th=[-30, -10, 15, 35], T=300, K_track=4, K_use=4, trk=False, lc=False, bh=None),
    'I': dict(fac=_fac([-30, -10, 15, 35], [1.0, 0.7, 0.8, 0.6]), th=[-30, -10, 15, 35], T=400, K_track=4, K_use=4, trk=False, lc=False, bh=None),
}
BLOCK = (150, 250)                       # scenario I: strongest cluster shadowed on [150, 250)

# ----------------------------------------------------------------------------- grids and protocol per array
KAPPAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CFG = {
    'ula': dict(GEOS=list('ABCDEFGHI'),
                CONDS=[(5, 10, 0), (5, 10, 2), (5, 10, 4), (10, 20, 0), (20, 10, 0), (5, 35, 0)],
                STREAM_NR={'A': 15, 'B': 15, 'C': 15, 'D': 15, 'E': 10, 'F': 15, 'G': 15, 'H': 15, 'I': 15},
                G_TR=[0.1, 0.3, 1, 3], G_NF=[30, 100, 300, 1000, 3000], EPSS=[0.25, 1, 4, 9],
                G_KD=[0.01, 0.03, 0.1, 0.3, 1], SECTS=[2, 3, 4.5, 6], NOPRED=True,
                NR_TRAIN=8, BASE_TRAIN=41000, NR_HELD=30, BASE_HELD=2026000, NR_ABL=30, HELD_TRACES=False),
    'upa': dict(GEOS=list('ABCDEFGHI'),
                CONDS=[(5, 10, 0), (5, 10, 2), (5, 10, 4), (10, 20, 0), (20, 10, 0), (5, 35, 0)],
                STREAM_NR={g: 15 for g in 'ABCDEFGHI'},
                G_TR=[0.1, 0.3, 1, 3, 10], G_NF=[100, 300, 1000, 3000, 10000, 30000], EPSS=[1, 4, 9, 16, 25, 36, 49],
                G_KD=[0.1, 0.3, 1, 3], SECTS=[6, 9, 12], NOPRED=True,
                NR_TRAIN=4, BASE_TRAIN=97000, NR_HELD=30, BASE_HELD=2026000, NR_ABL=10, HELD_TRACES=True),
}
VARIANTS = [(k, 'ms') for k in KAPPAS] + [(k, 0.0) for k in KAPPAS] + [(k, 1.0) for k in KAPPAS]
NOREC = [(k, 'ms') for k in (0.1, 0.3, 0.5, 0.7, 0.9)]
NOPRED = [(k, 'ms') for k in KAPPAS if k > 0]
NOTS = [(k, 'msR') for k in KAPPAS if k > 0]         # max-SINR targets computed on the reconstruction itself
NOPEAK = [(k, 'ms') for k in KAPPAS if k > 0]        # reconstruction without the off-grid peak refinement


def cfg():
    return CFG[ARRAY]


def use_array(name):
    """Select the array of this process ('ula' or 'upa'); 'upa' patches the steering model of the channel and
    of the kernel."""
    global ARRAY, U
    if name == 'upa' and ARRAY != 'upa':
        from . import planar as U
        U.patch_in()
    ARRAY = name


# ----------------------------------------------------------------------------- data and metrics
def make_data(sc, snr, inr, D, seed, reseed=True):
    chn.PS = chn.SIGMA2 * 10 ** (snr / 10)
    chn.PIV = chn.SIGMA2 * 10 ** (inr / 10)
    if ARRAY == 'upa':
        return _make_data_upa(sc, D, seed, reseed)
    c = SCEN[sc]
    if reseed:
        np.random.seed(seed)
    ch = c['fac']()
    T = c['T']
    X = np.zeros((chn.M_DEFAULT, T), dtype=complex)
    thl, pwl = [], []
    p_full = list(ch.power_k)
    for k in range(T):
        if sc == 'I':
            ch.power_k = list(p_full)
            if BLOCK[0] <= k < BLOCK[1]:
                ch.power_k[0] = 0.0
        x, _, tn = ch.snapshot(k)
        X[:, k] = x
        thl.append(list(tn))
        pwl.append(list(ch.power_k))
    R_ipn = ch.true_R_ipn()
    rm = np.random.RandomState(seed + 777)
    th0 = [np.deg2rad(t + d) for t, d in zip(c['th'], rm.uniform(-D, D, len(c['th'])))]
    a0 = sum(chn.steer(chn.M_DEFAULT, t) for t in th0)
    a0 = a0 / max(np.linalg.norm(a0), chn.EPS)
    bh = [(t, np.deg2rad(b)) for (t, b) in c['bh']] if c['bh'] else None
    return dict(X=X, thl=thl, pwl=pwl, R_ipn=R_ipn, th0=th0, a0=a0, bh=bh, T=T, ch=ch, c=c)


def _make_data_upa(sc, D, seed, reseed):
    s = U.UPA_SCEN[sc]
    M = U.M_UPA
    if reseed:
        np.random.seed(seed)
    ch, _ = U.make_channel(sc)
    T = s['T']
    X = np.zeros((M, T), dtype=complex)
    thl, pwl = [], []
    p_full = list(ch.power_k)
    for k in range(T):
        if sc == 'I':
            ch.power_k = list(p_full)
            if BLOCK[0] <= k < BLOCK[1]:
                ch.power_k[0] = 0.0
        U.apply_el_drift(ch, sc)
        x, _, tn = ch.snapshot(k)
        X[:, k] = x
        thl.append(list(tn))
        pwl.append(list(ch.power_k))
    R_ipn = ch.true_R_ipn()
    rm = np.random.RandomState(seed + 777)
    idx = s.get('th0_idx', list(range(len(s['az']))))
    th0 = [U.encode(np.deg2rad(s['az'][i] + da), np.deg2rad(s['el'][i] + de)) for i, da, de in
           zip(idx, rm.uniform(-D, D, len(idx)), rm.uniform(-D, D, len(idx)))]
    a0 = sum(chn.steer(M, t) for t in th0)
    a0 = a0 / max(np.linalg.norm(a0), chn.EPS)
    bh = [(t, np.deg2rad(U.enc_deg(a, e))) for (t, a, e) in s['bh']] if s.get('bh') else None
    c = dict(K_track=s['K_track'], K_use=s['K_use'], trk=s.get('trk', False), lc=s.get('lc', False), bh=bh, T=T)
    return dict(X=X, thl=thl, pwl=pwl, R_ipn=R_ipn, th0=th0, a0=a0, bh=bh, T=T, ch=ch, c=c)


def sinr_trace(h, d):
    return np.array([chn.sinr_db_cluster(h[k], d['thl'][k], d['pwl'][k], d['R_ipn'], chn.PS) for k in range(d['T'])])


def metrics(h, d, sc, s_all=None):
    thl, pwl, R_ipn, K_use, T = d['thl'], d['pwl'], d['R_ipn'], d['c']['K_use'], d['T']
    if s_all is None:
        s_all = sinr_trace(h, d)
    r = {'': float(np.mean(s_all[T - 50:]))}
    fl = []
    for k in range(T - 50, T):
        order = np.argsort(-np.array(pwl[k]))[:K_use]
        fl.append(alg.sinr_floor_db(h[k], [thl[k][i] for i in order], [pwl[k][i] for i in order], R_ipn, chn.PS))
    r['|floor'] = float(np.mean(fl))
    if sc == 'I':
        r['|dip'] = float(np.mean(s_all[BLOCK[0]:BLOCK[0] + 10]))
        r['|blk'] = float(np.mean(s_all[BLOCK[0]:BLOCK[1]]))
        r['|rec'] = float(np.mean(s_all[BLOCK[1]:BLOCK[1] + 10]))
    return r


def angle_deg(a, b):
    """Angular distance between two directions (deg): |difference| on the ULA, great-circle on the UPA."""
    if ARRAY == 'ula':
        return abs(np.rad2deg(a - b))
    az1, el1 = U.decode(a)
    az2, el2 = U.decode(b)
    u1 = np.array([np.cos(el1) * np.sin(az1), np.sin(el1), np.cos(el1) * np.cos(az1)])
    u2 = np.array([np.cos(el2) * np.sin(az2), np.sin(el2), np.cos(el2) * np.cos(az2)])
    return float(np.rad2deg(np.arccos(np.clip(u1 @ u2, -1.0, 1.0))))


def dir_errors(est, true):
    """Error (deg) of every true cluster under the one-to-one assignment of the estimates with the smallest total
    error; nan for a true cluster left without an estimate."""
    nt, ne = len(true), len(est)
    out = [float('nan')] * nt
    if nt == 0 or ne == 0:
        return out
    C = np.array([[angle_deg(e, t) for e in est] for t in true])          # nt x ne
    if ne >= nt:
        p = min(itertools.permutations(range(ne), nt), key=lambda p: sum(C[i, j] for i, j in enumerate(p)))
        for i, j in enumerate(p):
            out[i] = float(C[i, j])
    else:
        p = min(itertools.permutations(range(nt), ne), key=lambda p: sum(C[i, j] for j, i in enumerate(p)))
        for j, i in enumerate(p):
            out[i] = float(C[i, j])
    return out


# ----------------------------------------------------------------------------- one cell
def run_cell(sc, snr, inr, D, seed, reseed=True, sel=None, traces=False, ablate=True):
    """All methods on identical data.  sel=None evaluates every baseline parameter of the grid (training); with a
    selection only the selected baseline parameters are evaluated (MPR-RAB settings are always all evaluated, since
    they share one tracking run).  traces=True also returns the per-snapshot expected SINR of every method and the
    per-snapshot tracking error."""
    G = cfg()
    d = make_data(sc, snr, inr, D, seed, reseed)
    X, th0, a0, bh, c, T = d['X'], d['th0'], d['a0'], d['bh'], d['c'], d['T']
    r, trs = {}, {}

    def put(name, h):
        s_all = sinr_trace(h, d)
        if traces:
            trs[name] = s_all
        for suf, val in metrics(h, d, sc, s_all).items():
            r[name + suf] = val

    want = (lambda n: True) if sel is None else (lambda n: n in set(sel.values()))
    kw = dict(K_track=max(c['K_track'], len(th0)), K_use=c['K_use'], use_tracking=True, use_lifecycle=c['lc'], birth_hints=bh)
    var_main = VARIANTS + (NOTS if G['NOPRED'] else [])
    h, log = alg.run_mpr_rab(X, a0, list(th0), variants=var_main, **kw)
    dirs, pows = log['theta_hat'], log['p']
    for v in var_main:
        put(alg.variant_name(v), log['hist'][v])
        st = log['solver'][v]
        r[alg.variant_name(v) + '|solver_uncertified'] = int(sum(not x['certified'] for x in st))
        r[alg.variant_name(v) + '|solver_updates'] = len(st)
    ks = range(T) if traces else range(T - 50, T)
    errs = {k: dir_errors(dirs[k], d['thl'][k]) for k in ks}
    r['MPR|trkerr'] = [float(np.nanmean([errs[k][i] for k in range(T - 50, T)])) for i in range(len(d['thl'][-1]))]
    if traces:
        trs['_trkerr'] = np.array([np.nanmean(errs[k]) if len(errs[k]) else np.nan for k in range(T)])
    h2, log2 = alg.run_mpr_rab(X, a0, list(th0), variants=NOREC, recon=False, **kw)
    for v in NOREC:
        put('MPRnoRec k=%g' % v[0], log2['hist'][v])
    if G['NOPRED'] and ablate:
        h3, log3 = alg.run_mpr_rab(X, a0, list(th0), variants=NOPRED, predict=False, **kw)
        for v in NOPRED:
            put('MPRnoPred k=%g' % v[0], log3['hist'][v])
        h5, log5 = alg.run_mpr_rab(X, a0, list(th0), variants=NOPEAK, peak=False, **kw)
        for v in NOPEAK:
            put('MPRnoPeak k=%g' % v[0], log5['hist'][v])
    for g in G['G_TR']:
        if want('LCMVtr g=%g' % g):
            put('LCMVtr g=%g' % g, lcmv_rule(X, dirs, 'tr', g))
    for g in G['G_NF']:
        if want('LCMVnf g=%g' % g):
            put('LCMVnf g=%g' % g, lcmv_rule(X, dirs, 'nf', g))
    if want('LCMV-LW'):
        put('LCMV-LW', lcmv_rule(X, dirs, 'lw', 0))
    es = [e for e in G['EPSS'] if want('ParRCB e=%g' % e)]
    if es:
        pr = parrcb_multi(X, dirs, es)
        for e in es:
            put('ParRCB e=%g' % e, pr[e])
    for g in G['G_KD']:
        if want('LCMVKD g=%g' % g):
            put('LCMVKD g=%g' % g, lcmv_kd_dirs(X, dirs, g))
    keys = [(m, s_) for m in ('LCMV', 'MVDR', 'GEV') for s_ in G['SECTS'] if want('INCM-%s s=%g' % (m, s_))]
    keys += [(m, s_, 'r') for m in ('LCMV', 'MVDR', 'GEV') for s_ in G['SECTS'] if want('INCMr-%s s=%g' % (m, s_))]
    if keys:
        mo = incm_multi(X, dirs, pows, keys)
        for key in keys:
            put(('INCMr-%s s=%g' if len(key) > 2 else 'INCM-%s s=%g') % key[:2], mo[key])
    put('Oracle', [chn.run_oracle_mc(d['R_ipn'], d['thl'][k], d['pwl'][k], chn.M_DEFAULT) for k in range(T)])
    if traces:
        return r, trs
    return r


# ----------------------------------------------------------------------------- phases
def fname(phase, sc):
    return os.path.join(DATA, '%s_%s_%s.json' % (ARRAY, phase, sc))


def stream_states(base, geos):
    """RNG state at the start of every scenario of the seed-once stream, obtained by replaying the data generation
    only (no method touches the global RNG); lets the scenarios run in separate processes."""
    G = cfg()
    np.random.seed(base)
    states = {}
    for sc in G['GEOS']:
        states[sc] = np.random.get_state()
        if sc == geos[-1]:
            break
        for rr in range(G['STREAM_NR'][sc]):
            make_data(sc, 5, 10, 0, base, reseed=False)
    return states



def validate_cache(path, settings):
    """Reject cached results from a different protocol or source version."""
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parent
    sources = ('channel.py', 'algorithms.py', 'planar.py', 'baselines.py', 'scenarios.py', 'experiments/sensitivity.py')
    record = dict(settings=settings, source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                                  for name in sources})
    record = json.loads(json.dumps(record, sort_keys=True))
    metadata = str(path) + '.meta.json'
    if os.path.exists(path):
        if not os.path.exists(metadata) or json.load(open(metadata)) != record:
            raise RuntimeError('Cache metadata is missing or differs from this run: ' + str(path) +
                               '. Use a clean output folder or remove the cached file.')
    else:
        with open(metadata, 'w') as f:
            json.dump(record, f, indent=1, sort_keys=True)


def run_phase(phase, geos, NR=None, base=None, sel=None, verbose=True):
    """Runs a phase for the scenarios `geos` and caches one JSON file per scenario (resumable per condition).
    The stream phase, and the first condition of the planar condition study, also write the mean per-snapshot traces
    of every method (npz)."""
    G = cfg()
    conds = G['CONDS']
    if phase == 'stream':
        base = 42 if base is None else base
        conds = [(5, 10, 0)]
        states = stream_states(base, list(geos))
    elif phase == 'train':
        NR = G['NR_TRAIN'] if NR is None else NR
        base = G['BASE_TRAIN'] if base is None else base
    else:
        NR = G['NR_HELD'] if NR is None else NR
        base = G['BASE_HELD'] if base is None else base
    for sc in geos:
        fn = fname(phase, sc)
        os.makedirs(DATA, exist_ok=True)
        validate_cache(fn, dict(array=ARRAY, phase=phase, scenario=sc, conditions=conds,
                                runs=G['STREAM_NR'][sc] if phase == 'stream' else NR,
                                base=base, selection=sel, protocol=G))
        res = json.load(open(fn)) if os.path.exists(fn) else {}
        t0 = time.time()
        for ci, (snr, inr, D) in enumerate(conds):
            key = '%d_%d_%d' % (snr, inr, D)
            n_run = G['STREAM_NR'][sc] if phase == 'stream' else NR
            if key in res and len(res[key].get('Oracle', [])) >= n_run:
                continue
            cell, tracc = {}, {}
            if phase == 'stream':
                np.random.set_state(states[sc])
            want_tr = phase == 'stream' or (phase == 'heldout' and G['HELD_TRACES'] and ci == 0)
            for rr in range(n_run):
                seed = base + 10000 * G['GEOS'].index(sc) + 100 * ci + rr
                if phase == 'stream':
                    out, trs = run_cell(sc, snr, inr, D, seed, reseed=False, sel=sel, traces=True)
                else:
                    res_ = run_cell(sc, snr, inr, D, seed, reseed=True, sel=(sel if phase == 'heldout' else None),
                                    traces=want_tr, ablate=(phase != 'heldout' or rr < G['NR_ABL']))
                    out, trs = res_ if want_tr else (res_, {})
                for n, v in trs.items():
                    tracc.setdefault(n, []).append(v)
                for n, v in out.items():
                    cell.setdefault(n, []).append(v)
            res[key] = cell
            json.dump(res, open(fn, 'w'))
            if tracc:
                np.savez(fn.replace('.json', '_traces.npz'), **{n: np.nanmean(np.array(v), axis=0) for n, v in tracc.items()})
            if verbose:
                print('  %s %s %s %s %.0f s' % (ARRAY, phase, sc, key, time.time() - t0), flush=True)


def load_phase(phase, geos=None):
    R = {}
    for g in (geos or cfg()['GEOS']):
        fn = fname(phase, g)
        if os.path.exists(fn):
            R[g] = json.load(open(fn))
    return R


# ----------------------------------------------------------------------------- selection and reporting
def load_selection(arr=None):
    """Selected setting of every family for an array: DATA/selection_<arr>.json when a training phase has been rerun
    with --retrain, otherwise the selection of the chapter shipped in ch5mprrab/data/selection.json."""
    arr = arr or ARRAY
    fn = os.path.join(DATA, 'selection_%s.json' % arr)
    if os.path.exists(fn):
        return json.load(open(fn))['best']
    return json.load(open(os.path.join(PKG_DATA, 'selection.json')))[arr]['best']


def fam(n):
    """Family of a result name, or None for names that are not selectable methods."""
    if '|' in n:
        return None
    if n.startswith('MPRnoRec '):
        return 'MPR-RAB no reconstruction'
    if n.startswith('MPRnoPred '):
        return 'MPR-RAB no delay compensation'
    if n.startswith('MPRnoTS '):
        return 'MPR-RAB targets from the reconstruction'
    if n.startswith('MPRnoPeak '):
        return 'MPR-RAB no peak refinement'
    if n.startswith('MPR '):
        return {'': 'MPR-RAB', 'q=0': 'MPR-RAB equal targets', 'q=1': 'MPR-RAB power targets'}.get(' '.join(n.split()[2:]), None)
    for p, f in (('LCMVtr', 'LCMV-K trace-loaded'), ('LCMVnf', 'LCMV-K noise-floor-loaded'), ('ParRCB', 'Par-K-RCB'),
                 ('LCMVKD', 'LCMV-K-D'), ('INCM-LCMV', 'INCM-LCMV'), ('INCM-MVDR', 'INCM-MVDR'), ('INCM-GEV', 'INCM-GEV'),
                 ('INCMr-LCMV', 'INCM-LCMV refined'), ('INCMr-MVDR', 'INCM-MVDR refined'), ('INCMr-GEV', 'INCM-GEV refined')):
        if n.startswith(p):
            return f
    return n                                   # 'Oracle', 'LCMV-LW'


def kappa_of(n):
    return float(n.split('k=')[1].split()[0])


def pooled_table(R, suffix=''):
    """Mean over all scenarios, conditions and runs of every method present in every cell."""
    keys = set.intersection(*[set(R[g][c]) for g in R for c in R[g]])
    return {n: float(np.mean([np.mean(R[g][c][n + suffix]) for g in R for c in R[g]]))
            for n in keys if '|' not in n and (n + suffix) in keys}


def select(R, tie_db=TIE_DB):
    """One setting per family: the largest pooled expected SINR, with settings within tie_db of it treated as
    equivalent and resolved by the larger pooled minimum-cluster expected SINR.  MPR-RAB families: 0 < kappa < 1.
    Par-K-RCB: eps < M (an uncertainty set that contains the origin makes the robust Capon beamformer trivial)."""
    pe = pooled_table(R)
    pf = pooled_table(R, '|floor')
    M = chn.M_DEFAULT
    cands = {}
    for n in pe:
        f = fam(n)
        if f is None:
            continue
        if f.startswith('MPR-RAB') and not (0.0 < kappa_of(n) < 1.0):
            continue
        if f == 'Par-K-RCB' and float(n.split('=')[1]) >= M:
            continue
        cands.setdefault(f, []).append(n)
    best = {}
    for f, ns in cands.items():
        top = max(pe[n] for n in ns)
        ties = [n for n in ns if pe[n] >= top - tie_db]
        best[f] = max(ties, key=lambda n: (pf[n], pe[n]))
    return best, pe, pf


COLS = (('LCMV-K', 'LCMV-K noise-floor-loaded'), ('Par-K-RCB', 'Par-K-RCB'), ('LCMV-K-D', 'LCMV-K-D'),
        ('INCM-LCMV', 'INCM-LCMV'), ('INCM-MVDR', 'INCM-MVDR'), ('INCM-GEV', 'INCM-GEV'),
        ('MPR-RAB, beta_k = 1', 'MPR-RAB equal targets'), ('MPR-RAB', 'MPR-RAB'), ('Oracle', 'Oracle'))


def columns(sel):
    return [(lab, sel.get(f, f)) for lab, f in COLS]


def boot(dl, B=3000, seed=0):
    rng = np.random.RandomState(seed)
    m = np.mean([np.mean(d) for d in dl])
    bs = [np.mean([np.mean(d[rng.randint(0, len(d), len(d))]) for d in dl]) for _ in range(B)]
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def report(R, sel, title, suffix=''):
    cols = columns(sel)
    conds = sorted({c for g in R for c in R[g]}, key=lambda s: tuple(int(x) for x in s.split('_')))
    print(title)
    print('%-12s' % 'sc/cond' + ''.join('%11s' % c[0][:10] for c in cols))
    for g in R:
        for c in conds:
            if c in R[g]:
                print('%-12s' % ('%s %s' % (g, c.replace('_', '/'))) + ''.join('%11.2f' % np.mean(R[g][c][n + suffix]) for _, n in cols))
    print('%-12s' % 'pooled' + ''.join('%11.2f' % np.mean([np.mean(R[g][c][n + suffix]) for g in R for c in R[g]]) for _, n in cols))


def paired_ci(R, sel, suffix=''):
    cols = columns(sel)
    mp = sel['MPR-RAB']
    print('paired difference MPR-RAB minus comparator over all cells [95%% bootstrap CI]%s' % suffix)
    for lab, n in cols:
        if n == mp:
            continue
        per = [np.array(R[g][c][mp + suffix]) - np.array(R[g][c][n + suffix]) for g in R for c in R[g]]
        m, lo, hi = boot(per)
        print('   vs %-14s %+6.2f [%+.2f, %+.2f]' % (lab, m, lo, hi))


# ----------------------------------------------------------------------------- stages and runner
# (name, module and arguments, dependencies, estimated minutes on one core for the full counts); every stage runs in
# its own process because use_array('upa') patches module state.  {NR_U}, {NR_L}, {NR_T}, {NR_S} are replaced by the
# Monte Carlo counts (30, 30, 4 and 10 for the chapter; fewer with --quick).
_UPA_MIN = dict(A=20, B=29, C=26, D=46, E=39, F=29, G=49, H=39, I=52)
STAGES = (
    [('prop2', ['prop2'], [], 1), ('prop1', ['prop1'], [], 1), ('feasibility', ['feasibility'], [], 1),
     ('geometry', ['geometry'], [], 1)]
    + [('upa_heldout_%s' % g, ['study', 'upa', 'heldout', g, '{NR_U}'], [], _UPA_MIN[g]) for g in 'ABCDEFGHI']
    + [('upa_train_%s' % g, ['study', 'upa', 'train', g, '{NR_T}'], [], 12) for g in 'ABCDEFGHI']
    + [('upa_select', ['select', 'upa'], ['upa_train_%s' % g for g in 'ABCDEFGHI'], 1)]
    + [('ula_heldout_%s' % g, ['study', 'ula', 'heldout', g, '{NR_L}'], [], 13) for g in 'ABCDEFGHI']
    + [('ula_stream_C', ['study', 'ula', 'stream', 'C'], [], 2), ('tracking_c', ['tracking_c'], [], 3),
       ('sensitivity', ['sensitivity', '{NR_S}', '51000'], [], 20), ('solver_ula', ['solver', 'ula'], [], 2),
       ('solver_upa', ['solver', 'upa'], [], 6), ('pattern', ['pattern'], [], 1)]
)
TIMING = ('timing', ['timing', '{REPS}'], [], 4)
# stages each figure needs
NEEDS = {
    '5_1': ['upa_heldout_A'], '5_2': ['upa_heldout_C'], '5_3': ['upa_heldout_D'],
    '5_4': ['upa_heldout_%s' % g for g in 'ABCDEFGHI'], '5_5': ['pattern'], '5_6': ['upa_heldout_I'],
    '5_7': ['ula_stream_C', 'tracking_c'], '5_8': ['upa_train_%s' % g for g in 'ABCDEFGHI'], '5_9': ['sensitivity'],
    '5_10': [],
}
COUNTS = {'full': dict(NR_U=30, NR_L=30, NR_T=4, NR_TL=8, NR_S=10, REPS=5),
          'quick': dict(NR_U=2, NR_L=2, NR_T=1, NR_TL=1, NR_S=2, REPS=1)}


def stages_for(keys):
    """The stages needed by a subset of figures (keys such as '5_1')."""
    names = {s for k in keys for s in NEEDS[k]}
    return [s for s in STAGES if s[0] in names]


def run_stages(datadir, stages=STAGES, jobs=2, quick=False):
    """Runs the stages, at most `jobs` at a time, longest first among those whose dependencies are done.  Each stage
    writes its log to <datadir>/logs/<name>.log; a failed stage stops the run."""
    import subprocess
    import sys
    counts = COUNTS['quick' if quick else 'full']
    os.makedirs(os.path.join(datadir, 'logs'), exist_ok=True)
    env = dict(os.environ, CH5_DATA=os.path.abspath(datadir), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    names = {s[0] for s in stages}
    todo = sorted(stages, key=lambda s: -s[3])
    done, running = set(), {}
    t0 = time.time()
    while todo or running:
        for name, (p, f) in list(running.items()):
            if p.poll() is not None:
                f.close()
                if p.returncode != 0:
                    for q, g in running.values():
                        q.kill()
                    raise RuntimeError('stage %s failed; see %s' % (name, os.path.join(datadir, 'logs', name + '.log')))
                done.add(name)
                del running[name]
                print('  stage %-16s done  (%5.1f min elapsed)' % (name, (time.time() - t0) / 60), flush=True)
        while len(running) < jobs:
            ready = [s for s in todo if all(d in done or d not in names for d in s[2])]
            if not ready:
                break
            s = ready[0]
            todo.remove(s)
            args = [a.format(**counts) for a in s[1]]
            f = open(os.path.join(datadir, 'logs', s[0] + '.log'), 'w')
            cmd = [sys.executable, '-m', 'ch5mprrab.experiments.' + args[0]] + args[1:]
            running[s[0]] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=root), f)
            print('  stage %-16s start: %s' % (s[0], ' '.join(args)), flush=True)
        time.sleep(1)


def stage_done(datadir, stage, quick=False):
    """True when the output of a stage is present in datadir (study stages: every condition with all its runs)."""
    args = stage[1]
    counts = COUNTS['quick' if quick else 'full']
    if args[0] == 'study':
        fn = os.path.join(datadir, '%s_%s_%s.json' % (args[1], args[2], args[3]))
        if not os.path.exists(fn):
            return False
        res = json.load(open(fn))
        if args[2] == 'stream':
            return '5_10_0' in res and len(res['5_10_0'].get('Oracle', [])) >= CFG[args[1]]['STREAM_NR'][args[3]]
        nr = int(args[4].format(**counts))
        return all(len(res.get('%d_%d_%d' % c, {}).get('Oracle', [])) >= nr for c in CFG[args[1]]['CONDS'])
    out = {'select': 'selection_derived_%s.json' % (args[1] if len(args) > 1 else ''), 'sensitivity': 'sensitivity.json',
           'solver': 'solver_%s.json' % (args[1] if len(args) > 1 else ''), 'pattern': 'pattern_H.npz',
           'tracking_c': 'tracking_c.json', 'timing': 'timing.json'}.get(args[0], args[0] + '.json')
    fn = os.path.join(datadir, out)
    if not os.path.exists(fn):
        return False
    return args[0] != 'sensitivity' or 'offgrid' in json.load(open(fn))


def complete(datadir, stages=STAGES, quick=False):
    """True when every stage has written its output into datadir."""
    return all(stage_done(datadir, s, quick) for s in stages)
