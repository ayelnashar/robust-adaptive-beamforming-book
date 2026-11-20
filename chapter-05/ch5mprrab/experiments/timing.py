"""Implementation-component timings on linear arrays of every tested size (Section 5.7.8, Table 5.1, Figure 5.10):
    python -m ch5mprrab.experiments.timing [REPS]
MPR-RAB includes tracking and power estimation; baseline directions and powers are
precomputed outside their timers, and INCM reconstruction is unrefined. These are
not matched end-to-end receiver or UPA timings and cannot establish system speedup.
Wall-clock cost per snapshot against array size on one core: MPR-RAB (update every T_rec = 10, refinement at every
update), INCM-LCMV and INCM-GEV (every 10), LCMV-K and Par-K-RCB (every snapshot and every 10), LCMV-K-D (every
snapshot).  Three static clusters, two interferers, T snapshots, REPS repetitions; the median over repetitions is
reported.  Platform details are recorded in DATA/timing.json."""
import os, sys, json, time, platform
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from .. import channel as chn
from .. import algorithms as alg
from .. import scenarios as MS

T = 300
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SEL = MS.load_selection('ula')
KAP = MS.kappa_of(SEL['MPR-RAB'])
SIZES = [16, 64, 128, 256]


def lcmv_every(X, dirs, g, every):
    """LCMV-K with noise-floor loading, weight recomputed every `every` snapshots."""
    Mx, Tn = X.shape
    R = np.eye(Mx, dtype=complex)
    out, w = [], None
    for k in range(Tn):
        R = chn.ETA * R + np.outer(X[:, k], X[:, k].conj())
        if w is None or (k + 1) % every == 0:
            Rs = (R + R.conj().T) / 2
            ev = np.linalg.eigvalsh(Rs)
            dl = g * np.mean(ev[:Mx // 2])
            C = np.column_stack([chn.steer(Mx, t) for t in dirs[k]])
            RiC = np.linalg.solve(Rs + dl * np.eye(Mx), C)
            w = RiC @ np.linalg.solve(C.conj().T @ RiC, np.ones(C.shape[1]))
        out.append(w)
    return out


def rcb_every(X, dirs, eps, every):
    Mx, Tn = X.shape
    R = chn.SIGMA2 * np.eye(Mx, dtype=complex)
    out, w = [], None
    for k in range(Tn):
        R = chn.BETA * R + (1 - chn.BETA) * np.outer(X[:, k], X[:, k].conj())
        if w is None or (k + 1) % every == 0:
            Rs = (R + R.conj().T) / 2
            dl = 1e-3 * np.trace(Rs).real / Mx
            vals, vecs = np.linalg.eigh(Rs + dl * np.eye(Mx))
            vals = np.maximum(vals, 1e-12)
            w = sum(MS.rcb_weight(vals, vecs, chn.steer(Mx, th), eps) for th in dirs[k])
        out.append(w)
    return out


def main():
    res = {'sizes': SIZES, 'T': T, 'reps': REPS, 'kappa': KAP, 'ms': {},
           'scope': {'geometry': 'ULA at every size', 'mpr_tracking_included': True,
                     'baseline_tracking_included': False, 'incm_refined': False,
                     'matched_end_to_end': False,
                     'caution': 'Implementation-component timings; do not infer system-level speedup.'}}
    for M in SIZES:
        chn.M_DEFAULT = M
        alg.M_DEFAULT = M
        alg.GEOM = None
        np.random.seed(1)
        ch = chn.MultiClusterChannel(M, theta_k_deg=[-35, 0, 30], power_k=[1, 1, 1], theta_int_deg=[45, -60])
        X = np.zeros((M, T), dtype=complex)
        for k in range(T):
            X[:, k] = ch.snapshot(k)[0]
        th0 = [np.deg2rad(t) for t in (-35, 0, 30)]
        a0 = sum(chn.steer(M, t) for t in th0); a0 /= np.linalg.norm(a0)
        h, log = alg.run_mpr_rab(X, a0, th0, K_track=3, K_use=3, use_tracking=True, use_lifecycle=False, kappa=KAP, targets='ms')
        dirs, pows = log['theta_hat'], log['p']
        sd = float(SEL['INCM-LCMV'].split('s=')[1]); sg = float(SEL['INCM-GEV'].split('s=')[1])
        methods = {
            'MPR-RAB (T_rec = 10)': lambda: alg.run_mpr_rab(X, a0, th0, K_track=3, K_use=3, use_tracking=True, use_lifecycle=False,
                                                               kappa=KAP, targets='ms'),
            'INCM-LCMV (every 10)': lambda: MS.incm_multi(X, dirs, pows, [('LCMV', sd)]),
            'INCM-GEV (every 10)': lambda: MS.incm_multi(X, dirs, pows, [('GEV', sg)]),
            'LCMV-K (every snapshot)': lambda: lcmv_every(X, dirs, 300, 1),
            'LCMV-K (every 10)': lambda: lcmv_every(X, dirs, 300, 10),
            'Par-K-RCB (every snapshot)': lambda: rcb_every(X, dirs, 4, 1),
            'Par-K-RCB (every 10)': lambda: rcb_every(X, dirs, 4, 10),
            'LCMV-K-D (every snapshot)': lambda: MS.lcmv_kd_dirs(X, dirs, 0.1),
        }
        res['ms'][M] = {}
        for name, f in methods.items():
            f()                                    # warm-up
            ts = []
            for _ in range(REPS):
                t0 = time.perf_counter(); f(); ts.append((time.perf_counter() - t0) / T * 1e3)
            res['ms'][M][name] = float(np.median(ts))
            print('M=%3d %-28s %8.3f ms/snapshot' % (M, name, res['ms'][M][name]), flush=True)
        # share of the MPR-RAB time spent in the 2M x 2M eigendecomposition (one per update)
        Q = np.random.randn(2 * M, 2 * M); Q = Q @ Q.T
        t0 = time.perf_counter()
        for _ in range(20):
            np.linalg.eigh(Q)
        res.setdefault('eigh_ms', {})[M] = (time.perf_counter() - t0) / 20 * 1e3
    chn.M_DEFAULT = 16; alg.M_DEFAULT = 16; alg.GEOM = None
    cpu = ''
    try:
        cpu = [l for l in open('/proc/cpuinfo') if 'model name' in l][0].split(':', 1)[1].strip()
    except Exception:
        pass
    blas = ''
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        blas = buf.getvalue()
    except Exception:
        pass
    res['platform'] = dict(cpu=cpu, python=platform.python_version(), numpy=np.__version__, scipy=__import__('scipy').__version__,
                           os=platform.platform(), threads=os.environ.get('OMP_NUM_THREADS'), blas=blas[:2000])
    os.makedirs(MS.DATA, exist_ok=True)
    json.dump(res, open(os.path.join(MS.DATA, 'timing.json'), 'w'), indent=1)
    print('saved timing.json;', cpu)


if __name__ == '__main__':
    main()
