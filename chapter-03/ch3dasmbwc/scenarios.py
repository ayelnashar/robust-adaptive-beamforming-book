"""
ch3dasmbwc.scenarios -- simulation settings and experiment runners of Section 3.6.

Every Monte Carlo trial seeds NumPy's global generator before drawing its data, so each
experiment can be regenerated exactly, independently of the order in which experiments run.
The runners return numbers only; plotting lives in ch3dasmbwc.figures.
"""
import time
import platform
import numpy as np
from .algorithms import (steer, steer_err, sinr_db, sinr_cov_db, sinr_opt_db, desired_cov, mvdr, dl_mvdr,
                         wc_mvdr, adaptive_single_wc, inc_reconstruction, das_mbwc, mbwc_socp,
                         feasibility_margin)

# --------------------------------------------------------------------------- settings (Section 3.6)
M = 16                       # elements, half-wavelength ULA
SIGMA2 = 1.0                 # noise power per element (0 dB)
SNR = 5                      # dB, SOI at broadside
INR = 10                     # dB, each interferer
PS = SIGMA2 * 10 ** (SNR / 10)
ETA = 0.97                   # forgetting factor of the adaptive recursions, Eq. (3.4)
BETA = 0.99                  # EWMA weight of the batch covariance
ZETA = 3.8                   # worst-case radius, covers about 2 deg at M = 16
THETA_INTS = [np.deg2rad(45), np.deg2rad(-60)]
NU0 = np.array([1.9, 1.5, 1.5, 1.1])           # reference bounds
MP_SMALL = [(np.deg2rad(4), 0.4), (np.deg2rad(-3), 0.4)]     # rays at +4 and -3 deg, amplitude 0.4
MP_LARGE = [(np.deg2rad(-25), 0.5), (np.deg2rad(35), 0.3)]   # rays at -25 deg (0.5) and +35 deg (0.3)
RECON_SECTOR_DEG = 6.0       # declared sector of the reconstruction comparator, +-6 deg
SEED_BASE = 42000            # Table 3.2 / Figures 3.1-3.10

METHODS = ['Standard MVDR', 'DL-MVDR', 'WC-MVDR [6]', 'Adaptive single-WC [12]', 'DAS-MBWC',
           'INC reconstruction [22]', 'Optimum']


def ipn_cov(Mv=M, ints=THETA_INTS, steer_fn=None):
    """True interference-plus-noise covariance R_i+n."""
    steer_fn = steer_fn or (lambda t: steer(Mv, t))
    R = SIGMA2 * np.eye(Mv, dtype=complex)
    for ti in ints:
        ai = steer_fn(ti)
        R += SIGMA2 * 10 ** (INR / 10) * np.outer(ai, ai.conj())
    return R


R_IPN = ipn_cov()


def paper_grid(thp_deg):
    """Constraint grid [theta_0, theta_0 - 2, theta_0 + 2, theta_0 - 4] deg about the presumed direction."""
    return [np.deg2rad(thp_deg), np.deg2rad(thp_deg - 2), np.deg2rad(thp_deg + 2), np.deg2rad(thp_deg - 4)]


# Table 3.2 scenarios, in the order that fixes their seeds (SEED_BASE + 100 s + r)
SCENARIOS = [
    ('2° point source', 2, None), ('5° point source', 5, None), ('7° point source', 7, None),
    ('2° sm. spread', 2, MP_SMALL), ('5° sm. spread', 5, MP_SMALL), ('7° sm. spread', 7, MP_SMALL),
    ('5° lg. spread', 5, MP_LARGE), ('7° lg. spread', 7, MP_LARGE)]
# row order of Tables 3.2, 3.3 and 3.6
TABLE_ORDER = ['2° point source', '2° sm. spread', '5° point source', '5° sm. spread', '5° lg. spread',
               '7° point source', '7° sm. spread', '7° lg. spread']
SCEN = {lab: (md, mp) for lab, md, mp in SCENARIOS}


# --------------------------------------------------------------------------- data generation
def gen(Mv, K, th_s, mp, ints=THETA_INTS):
    """K snapshots: SOI at th_s (plus multipath rays with phases redrawn every snapshot),
    interferers at `ints` with INR, white noise. Uses NumPy's global generator."""
    piv = SIGMA2 * 10 ** (INR / 10)
    a_s = steer(Mv, th_s)
    X = np.zeros((Mv, K), dtype=complex)
    for k in range(K):
        s = np.sqrt(PS / 2) * (np.random.randn() + 1j * np.random.randn())
        sig = a_s * s
        if mp:
            for dth, amp in mp:
                g = amp * np.exp(1j * np.random.uniform(-np.pi, np.pi))
                sig += steer(Mv, th_s + dth) * s * g
        for ti in ints:
            u = np.sqrt(piv / 2) * (np.random.randn() + 1j * np.random.randn())
            sig += steer(Mv, ti) * u
        sig += np.sqrt(SIGMA2 / 2) * (np.random.randn(Mv) + 1j * np.random.randn(Mv))
        X[:, k] = sig
    return X


def gen_err(Mv, K, th_s, mp, gain=None, phase=None, pos=None):
    """As gen(), but the actual array response carries gain, phase and position errors."""
    piv = SIGMA2 * 10 ** (INR / 10)
    a_s = steer_err(Mv, th_s, gain, phase, pos)
    X = np.zeros((Mv, K), dtype=complex)
    for k in range(K):
        s = np.sqrt(PS / 2) * (np.random.randn() + 1j * np.random.randn())
        sig = a_s * s
        if mp:
            for dth, amp in mp:
                g = amp * np.exp(1j * np.random.uniform(-np.pi, np.pi))
                sig += steer_err(Mv, th_s + dth, gain, phase, pos) * s * g
        for ti in THETA_INTS:
            u = np.sqrt(piv / 2) * (np.random.randn() + 1j * np.random.randn())
            sig += steer_err(Mv, ti, gain, phase, pos) * u
        sig += np.sqrt(SIGMA2 / 2) * (np.random.randn(Mv) + 1j * np.random.randn(Mv))
        X[:, k] = sig
    return X


def gen_plain(Mv, K, ints=THETA_INTS):
    """Point source at broadside, interferers at `ints`, white noise (array-size and timing runs)."""
    piv = SIGMA2 * 10 ** (INR / 10)
    a_s = steer(Mv, 0.0)
    X = np.zeros((Mv, K), dtype=complex)
    for k in range(K):
        s = np.sqrt(PS / 2) * (np.random.randn() + 1j * np.random.randn())
        sig = a_s * s
        for ti in ints:
            sig += steer(Mv, ti) * (np.sqrt(piv / 2) * (np.random.randn() + 1j * np.random.randn()))
        sig += np.sqrt(SIGMA2 / 2) * (np.random.randn(Mv) + 1j * np.random.randn(Mv))
        X[:, k] = sig
    return X


def batch_cov(X):
    """EWMA covariance used by the batch beamformers (weight BETA, initialized at sigma^2 I)."""
    Re = np.eye(X.shape[0], dtype=complex) * SIGMA2
    for k in range(X.shape[1]):
        Re = BETA * Re + (1 - BETA) * np.outer(X[:, k], X[:, k].conj())
    return (Re + Re.conj().T) / 2


def _safe(f):
    try:
        return f()
    except Exception:
        return -20.0


# --------------------------------------------------------------------------- Figures 3.1, 3.3-3.8, Table 3.2
def snapshot_scenario(label, NR=25, K=300):
    """SINR versus snapshot for one Table 3.2 scenario.

    Returns S (7 x K Monte Carlo means, METHODS order) and TR (NR x 7 final-snapshot values per
    trial: methods 0-5 in METHODS order, column 6 = all-constraint ablation).
    The batch methods use the EWMA covariance up to each snapshot; the reconstruction comparator
    is evaluated every 20 snapshots (and at the last) and held in between, which saves run time
    and does not affect the final values.
    """
    si = [s[0] for s in SCENARIOS].index(label)
    md, mp = SCEN[label]
    thp = np.deg2rad(md)
    ap = steer(M, thp)
    ca = paper_grid(md)
    Rsd = desired_cov(M, 0.0, mp, PS)
    so = sinr_opt_db(Rsd, R_IPN)
    S = np.zeros((7, K))
    TR = np.zeros((NR, 7))
    for r in range(NR):
        np.random.seed(SEED_BASE + 100 * si + r)
        X = gen(M, K, 0.0, mp)
        w08 = adaptive_single_wc(X, ap, ZETA, ETA)
        wpr = das_mbwc(X, ap, ca, NU0, ETA)
        TR[r, 6] = sinr_cov_db(das_mbwc(X, ap, ca, NU0, ETA, select='all')[-1], Rsd, R_IPN)
        Re = np.eye(M, dtype=complex) * SIGMA2
        rc = -20.0
        cur = np.zeros((7, K))
        for k in range(K):
            Re = BETA * Re + (1 - BETA) * np.outer(X[:, k], X[:, k].conj())
            Rs = (Re + Re.conj().T) / 2
            cur[0, k] = _safe(lambda: sinr_cov_db(mvdr(Rs, ap), Rsd, R_IPN))
            cur[1, k] = _safe(lambda: sinr_cov_db(dl_mvdr(Rs, ap, 2 * SIGMA2), Rsd, R_IPN))
            cur[2, k] = _safe(lambda: sinr_cov_db(wc_mvdr(Rs, ap, ZETA), Rsd, R_IPN))
            cur[3, k] = sinr_cov_db(w08[k], Rsd, R_IPN)
            cur[4, k] = sinr_cov_db(wpr[k], Rsd, R_IPN)
            if k % 20 == 0 or k == K - 1:
                rc = _safe(lambda: sinr_cov_db(inc_reconstruction(Rs, ap, thp, RECON_SECTOR_DEG), Rsd, R_IPN))
            cur[5, k] = rc
            cur[6, k] = so
        S += cur
        TR[r, :6] = cur[:6, K - 1]
    return S / NR, TR


# --------------------------------------------------------------------------- Figure 3.2
def beampattern_weights(md=5, K=500, seed=SEED_BASE + 900):
    """Final weights of every method for the 5 deg point-source scenario (one realization)."""
    thp = np.deg2rad(md)
    ap = steer(M, thp)
    np.random.seed(seed)
    X = gen(M, K, 0.0, None)
    Rs = batch_cov(X)
    at = steer(M, 0.0)
    wo = np.linalg.solve(R_IPN, at)
    wo = wo / (at.conj() @ wo)
    return {
        'Standard MVDR': mvdr(Rs, ap),
        'DL-MVDR': dl_mvdr(Rs, ap, 2 * SIGMA2),
        'WC-MVDR [6]': wc_mvdr(Rs, ap, ZETA),
        'Adaptive single-WC [12]': adaptive_single_wc(X, ap, ZETA, ETA)[-1],
        'DAS-MBWC': das_mbwc(X, ap, paper_grid(md), NU0, ETA)[-1],
        'INC reconstruction [22]': inc_reconstruction(Rs, ap, thp, RECON_SECTOR_DEG),
        'Optimum': wo,
    }


# --------------------------------------------------------------------------- Figures 3.9, 3.10
def mismatch_sweep(mp, seed0, mm_range=tuple(range(9)), NR=25, K=300):
    """Final-snapshot SINR versus mismatch angle (true direction at broadside).
    Returns (7 x len(mm_range) means, all-constraint ablation means)."""
    sinr = np.zeros((7, len(mm_range)))
    abl = np.zeros(len(mm_range))
    Rsd = desired_cov(M, 0.0, mp, PS)
    so = sinr_opt_db(Rsd, R_IPN)
    for mi, md in enumerate(mm_range):
        thp = np.deg2rad(md)
        ap = steer(M, thp)
        ca = paper_grid(md)
        for r in range(NR):
            np.random.seed(seed0 + 100 * mi + r)
            X = gen(M, K, 0.0, mp)
            abl[mi] += sinr_cov_db(das_mbwc(X, ap, ca, NU0, ETA, select='all')[-1], Rsd, R_IPN) / NR
            Rs = batch_cov(X)
            sinr[0, mi] += _safe(lambda: sinr_cov_db(mvdr(Rs, ap), Rsd, R_IPN))
            sinr[1, mi] += _safe(lambda: sinr_cov_db(dl_mvdr(Rs, ap, 2 * SIGMA2), Rsd, R_IPN))
            sinr[2, mi] += _safe(lambda: sinr_cov_db(wc_mvdr(Rs, ap, ZETA), Rsd, R_IPN))
            sinr[3, mi] += sinr_cov_db(adaptive_single_wc(X, ap, ZETA, ETA)[-1], Rsd, R_IPN)
            sinr[4, mi] += sinr_cov_db(das_mbwc(X, ap, ca, NU0, ETA)[-1], Rsd, R_IPN)
            sinr[5, mi] += _safe(lambda: sinr_cov_db(inc_reconstruction(Rs, ap, thp, RECON_SECTOR_DEG), Rsd, R_IPN))
            sinr[6, mi] += so
        sinr[:, mi] /= NR
    return sinr, abl


# --------------------------------------------------------------------------- Figure 3.11
ERROR_CASES = [('Ideal', 0.00, 0.0, 0.000), ('Gain 5%', 0.05, 0.0, 0.000), ('Gain 10%', 0.10, 0.0, 0.000),
               ('Phase 5°', 0.00, 5.0, 0.000), ('Phase 10°', 0.00, 10.0, 0.000),
               ('Pos 0.02λ', 0.00, 0.0, 0.020), ('Pos 0.05λ', 0.00, 0.0, 0.050),
               ('Combined', 0.05, 5.0, 0.020)]


def array_errors(md=5, NR=20, K=300):
    """Gain, phase and position errors (standard deviations) on the actual array response; every
    beamformer uses the ideal steering model. Scored with the perturbed interference covariance.
    Returns {case: (DAS-MBWC, INC, WC-MVDR, adaptive single-WC)}."""
    thp = np.deg2rad(md)
    ap = steer(M, thp)
    ca = paper_grid(md)
    out = {}
    for ci, (name, g, ph, po) in enumerate(ERROR_CASES):
        p = e = a_ = rc = 0.0
        for r in range(NR):
            rng = np.random.RandomState(9000 + r)
            ge = 1.0 + g * rng.randn(M) if g > 0 else None
            pe = np.deg2rad(ph) * rng.randn(M) if ph > 0 else None
            pz = po * rng.randn(M) if po > 0 else None
            np.random.seed(46000 + 100 * ci + r)
            X = gen_err(M, K, 0.0, None, ge, pe, pz)
            at = steer_err(M, 0.0, ge, pe, pz)
            R_i = ipn_cov(steer_fn=lambda t: steer_err(M, t, ge, pe, pz))
            Rs = batch_cov(X)
            p += sinr_db(das_mbwc(X, ap, ca, NU0, ETA)[-1], at, R_i, PS)
            e += sinr_db(wc_mvdr(Rs, ap, ZETA), at, R_i, PS)
            a_ += sinr_db(adaptive_single_wc(X, ap, ZETA, ETA)[-1], at, R_i, PS)
            rc += sinr_db(inc_reconstruction(Rs, ap, thp, RECON_SECTOR_DEG), at, R_i, PS)
        out[name] = (p / NR, rc / NR, e / NR, a_ / NR)
    return out


# --------------------------------------------------------------------------- Figure 3.12
SIZE_GEOMS = [('Interference outside\nthe declared region', [45.0, -60.0]),
              ('One interferer inside\nthe declared region', [45.0, -60.0, 12.0])]


def array_size(md=7, M_list=(16, 64, 128), NR=15, K=300):
    """7 deg mismatch at 16, 64 and 128 elements. Six-point constraint fan from the true direction
    (0 deg, bound 1.9) to presumed + 1.75 deg (bounds 1.3), bounds and radius scaled by sqrt(M/16);
    the reconstruction sector is +-7 deg about the presumed direction.
    Returns {geometry: (DAS-MBWC, INC, WC-MVDR, adaptive single-WC, optimum) lists over M_list}."""
    fan = list(np.linspace(-md, md / 4, 6))
    res = {}
    for gi, (gname, ang) in enumerate(SIZE_GEOMS):
        P, RC, E, A, O = [], [], [], [], []
        ints = [np.deg2rad(t) for t in ang]
        for mi, Mv in enumerate(M_list):
            s_nu = np.sqrt(Mv / 16.0)
            R_i = ipn_cov(Mv, ints)
            at = steer(Mv, 0.0)
            ap = steer(Mv, np.deg2rad(md))
            ca = [np.deg2rad(md + o) for o in fan]
            nu = np.concatenate([[1.9], np.full(len(fan) - 1, 1.3)]) * s_nu
            p = rc = e = a_ = 0.0
            for r in range(NR):
                np.random.seed(47000 + 1000 * gi + 100 * mi + r)
                X = gen_plain(Mv, K, ints)
                p += sinr_db(das_mbwc(X, ap, ca, nu, ETA)[-1], at, R_i, PS)
                Rs = batch_cov(X)
                rc += sinr_db(inc_reconstruction(Rs, ap, np.deg2rad(md), md), at, R_i, PS)
                e += sinr_db(wc_mvdr(Rs, ap, ZETA * s_nu), at, R_i, PS)
                a_ += sinr_db(adaptive_single_wc(X, ap, ZETA * s_nu, ETA)[-1], at, R_i, PS)
            P.append(p / NR); RC.append(rc / NR); E.append(e / NR); A.append(a_ / NR)
            wo = np.linalg.solve(R_i, at)
            O.append(sinr_db(wo / (at.conj() @ wo), at, R_i, PS))
        res[gname] = (P, RC, E, A, O)
    return res


# --------------------------------------------------------------------------- Figure 3.13
SENS_RHO = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
SENS_CLIP = [(0.9, 1.1), (0.8, 1.25), (0.65, 1.5), (0.5, 2.0), (0.35, 3.0), (0.25, 4.0)]
SENS_PERIOD = [10, 20, 30, 40, 50, 70]


def sensitivity(md=5, NR=15, K=300, rhos=SENS_RHO, clips=SENS_CLIP, periods=SENS_PERIOD):
    """Final-snapshot SINR (point source, 5 deg) against the smoothing weight, clipping range and
    adaptation period of (3.20), each swept with the others at their nominal values."""
    at = steer(M, 0.0)
    ap = steer(M, np.deg2rad(md))
    ca = paper_grid(md)

    def one(rho, lo, hi, period):
        acc = 0.0
        for r in range(NR):
            np.random.seed(50000 + r)
            X = gen(M, K, 0.0, None)
            w = das_mbwc(X, ap, ca, NU0, ETA, rescale=False, smoothing=(rho, 1 - rho), clip=(lo, hi),
                         period=period)[-1]
            acc += sinr_db(w, at, R_IPN, PS)
        return acc / NR

    return dict(rho=[(r, one(r, 0.5, 2.0, 30)) for r in rhos],
                clip=[((lo, hi), one(0.8, lo, hi, 30)) for lo, hi in clips],
                period=[(p, one(0.8, 0.5, 2.0, p)) for p in periods])


# --------------------------------------------------------------------------- Figure 3.14 and Table 3.4 (timing)
def environment():
    import numpy, scipy
    env = dict(machine=platform.machine(), cores=__import__('os').cpu_count(), python=platform.python_version(),
               numpy=numpy.__version__, scipy=scipy.__version__)
    try:
        import cvxpy, scs
        env.update(cvxpy=cvxpy.__version__, scs=scs.__version__)
    except ImportError:
        pass
    try:
        with open('/proc/cpuinfo') as f:
            env['cpu'] = [l.split(':', 1)[1].strip() for l in f if l.startswith('model name')][0]
    except Exception:
        env['cpu'] = platform.processor() or platform.machine()
    return env


def computation_time(M_list=(16, 32, 64, 128, 256), K=100, NREP=5):
    """Per-snapshot time of the complete DAS-MBWC recursion (100-snapshot run, including the
    covariance update and the bound adaptation) against one exact WC-MVDR weight solve on a fixed
    covariance; medians over NREP repetitions (20 WC-MVDR solves per repetition), microseconds."""
    rows = []
    for Mv in M_list:
        np.random.seed(52000 + Mv)
        X = gen_plain(Mv, K)
        Rv = X @ X.conj().T / K
        apv = steer(Mv, np.deg2rad(5))
        cav = [np.deg2rad(5 + o) for o in (0, -2, 2, -4)]
        zeta_t = ZETA * np.sqrt(Mv / 16.0)
        for _ in range(10):
            wc_mvdr(Rv, apv, zeta_t)
        tps, tes = [], []
        for _ in range(NREP):
            t0 = time.perf_counter()
            das_mbwc(X, apv, cav, NU0, ETA)
            tps.append((time.perf_counter() - t0) / K * 1e6)
            t0 = time.perf_counter()
            for _ in range(20):
                wc_mvdr(Rv, apv, zeta_t)
            tes.append((time.perf_counter() - t0) / 20 * 1e6)
        rows.append((Mv, float(np.median(tps)), float(np.median(tes))))
    return rows


def table4_timing(M_list=(16, 32, 64, 128), REP=10, K=300):
    """Table 3.4: one SCS solve of (3.12) (median of REP) versus the DAS-MBWC Gram correction
    (median over a K-snapshot run), 5 deg grid, bounds scaled by sqrt(M/16)."""
    out = {'env': environment(), 'rows': {}}
    for Mv in M_list:
        np.random.seed(49000 + Mv)
        X = gen_plain(Mv, K)
        ca = paper_grid(5)
        Lc = np.column_stack([steer(Mv, t) for t in ca])
        nu = NU0 * np.sqrt(Mv / 16.0)
        Rs = batch_cov(X)
        mbwc_socp(Rs, Lc, nu)                                    # warm-up
        st, wl = [], []
        for _ in range(REP):
            _, _, _, s_t, w_t = mbwc_socp(Rs, Lc, nu)
            st.append(s_t); wl.append(w_t)
        log = {'time': True}
        das_mbwc(X, steer(Mv, np.deg2rad(5)), ca, nu, ETA, log=log)
        tp = np.array(log.get('tproj', [np.nan]))
        row = dict(scs_solve_ms=1e3 * float(np.median(st)), scs_wall_ms=1e3 * float(np.median(wl)),
                   gram_ms=1e3 * float(np.median(tp)), n_proj=int(len(tp)))
        row['speedup'] = row['scs_solve_ms'] / row['gram_ms']
        out['rows'][Mv] = row
    return out


# --------------------------------------------------------------------------- Table 3.3
def table3_socp(NR=25, K=300):
    """DAS-MBWC (adapted bounds) versus the SCS solution of (3.12) at the reference bounds on the
    batch covariance at the final snapshot of every Table 3.2 trial (same seeds)."""
    from scipy import stats
    out = {}
    for si, (lab, md, mp) in enumerate(SCENARIOS):
        thp = np.deg2rad(md); ap = steer(M, thp); ca = paper_grid(md)
        Lc = np.column_stack([steer(M, t) for t in ca]); Rsd = desired_cov(M, 0.0, mp, PS)
        P, S, ST, RS = [], [], [], []
        for r in range(NR):
            np.random.seed(SEED_BASE + 100 * si + r)
            X = gen(M, K, 0.0, mp)
            P.append(sinr_cov_db(das_mbwc(X, ap, ca, NU0, ETA)[-1], Rsd, R_IPN))
            wv, st, res, _, _ = mbwc_socp(batch_cov(X), Lc, NU0)
            ST.append(st); RS.append(res)
            S.append(sinr_cov_db(wv, Rsd, R_IPN) if wv is not None else np.nan)
        d = np.array(P) - np.array(S)
        h = float(stats.t.ppf(0.975, NR - 1) * d.std(ddof=1) / np.sqrt(NR)) if NR > 1 else float('nan')
        out[lab] = dict(das_mbwc=float(np.mean(P)), socp=float(np.nanmean(S)), diff=float(d.mean()), diff_ci=h,
                        status={s: ST.count(s) for s in set(ST)}, max_resid=float(np.nanmax(RS)))
    return out


# --------------------------------------------------------------------------- Table 3.5
PLACEMENTS = {'Point grid near presumed dir.': [],
              'Point grid + multipath dirs (−25°, +35°)': [-25.0, 35.0],
              'Point grid + constraint at +44°': [44.0],
              'Point grid + constraints on interferers (45°, −60°)': [45.0, -60.0]}


def table5_placement(NR=25, K=300):
    """Four constraint placements under large angular spread; every added constraint uses bound 1.1."""
    out = {}
    for ri, (lab, add) in enumerate(PLACEMENTS.items()):
        row = {}
        for ai, md in enumerate((5, 7)):
            thp = np.deg2rad(md); ap = steer(M, thp)
            ca = paper_grid(md) + [np.deg2rad(t) for t in add]
            nu = np.concatenate([NU0, np.full(len(add), 1.1)])
            Rsd = desired_cov(M, 0.0, MP_LARGE, PS)
            V = []
            for r in range(NR):
                np.random.seed(48000 + 1000 * ri + 100 * ai + r)
                X = gen(M, K, 0.0, MP_LARGE)
                V.append(sinr_cov_db(das_mbwc(X, ap, ca, nu, ETA)[-1], Rsd, R_IPN))
            row[md] = float(np.mean(V))
        out[lab] = row
    return out


# --------------------------------------------------------------------------- Table 3.6 and Section 3.6.7
def stability(NR=25, K=300, k0=150):
    """Table 3.6 columns 2-5 over snapshots k0..K-1: SINR standard deviation, set change rate,
    two-cycle fraction and mean active-set size (seeds 5000 + r)."""
    out = {}
    for lab in TABLE_ORDER:
        md, mp = SCEN[lab]
        Rsd = desired_cov(M, 0.0, mp, PS); thp = np.deg2rad(md); ap = steer(M, thp); ca = paper_grid(md)
        SD, CH, CY, AM = [], [], [], []
        for r in range(NR):
            np.random.seed(5000 + r)
            X = gen(M, K, 0.0, mp)
            log = {}
            h = das_mbwc(X, ap, ca, NU0, ETA, log=log)
            A = log['A'][k0:]
            SD.append(np.std([sinr_cov_db(h[k], Rsd, R_IPN) for k in range(k0, K)]))
            CH.append(np.mean([A[k] != A[k - 1] for k in range(1, len(A))]))
            CY.append(np.mean([A[k] == A[k - 2] and A[k] != A[k - 1] for k in range(2, len(A))]))
            AM.append(np.mean([len(x) for x in A]))
        out[lab] = dict(sinr_sd=float(np.mean(SD)), change_rate=float(np.mean(CH)),
                        two_cycle=float(np.mean(CY)), mean_A=float(np.mean(AM)))
    return out


def long_runs(NR=5, K=3000):
    """Table 3.6 columns 6-7: resets per run without and with the rescaling step, and the SINR over
    the last 500 snapshots with the step (seeds 30000 + r). A reset is an overflow of the weight
    norm, after which the recursion re-initializes w to a / ||a||."""
    out = {}
    for lab in TABLE_ORDER:
        md, mp = SCEN[lab]
        Rsd = desired_cov(M, 0.0, mp, PS); thp = np.deg2rad(md); ap = steer(M, thp); ca = paper_grid(md)
        RO, RR, MX, SL = [], [], [], []
        for r in range(NR):
            np.random.seed(30000 + r)
            X = gen(M, K, 0.0, mp)
            for resc, lst in ((False, RO), (True, RR)):
                with np.errstate(over='ignore', invalid='ignore'):
                    h = das_mbwc(X, ap, ca, NU0, ETA, rescale=resc)
                n = np.array([np.linalg.norm(w) for w in h])
                lst.append(int(np.sum((n[1:] < 1.01) & (n[:-1] > 10))))
                if resc:
                    MX.append(n.max())
                    SL.append(np.mean([sinr_cov_db(h[k], Rsd, R_IPN) for k in range(K - 500, K)]))
        out[lab] = dict(resets_without=float(np.mean(RO)), resets_with=float(np.mean(RR)),
                        max_norm=float(np.max(MX)), sinr_last500=float(np.mean(SL)))
    return out


def feasibility_limits(presumed=(2, 5, 7), scales=(1.0, 1.832, 2.0)):
    """Slater margin t* (3.A.4) of the grid at scaled bounds s * nu0 and the limit s where t* = 0
    (bisection, 40 steps), with the steering vectors of (3.2)."""
    out = {}
    for md in presumed:
        Lc = np.column_stack([steer(M, t) for t in paper_grid(md)])
        tstar = lambda s: feasibility_margin(Lc, NU0 * s)
        lo, hi = 1.0, 2.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if tstar(mid) > 0 else (lo, mid)
        out[md] = dict(t={s: tstar(s) for s in scales}, limit=lo)
    return out


def subset_limits(presumed=(2, 5, 7), sizes=(3, 2), s_max=4.0):
    """Feasibility limit s for every three- and two-constraint subset of the grid (Appendix 3.A.4).
    Limits at or above s_max are reported as s_max."""
    from itertools import combinations
    out = {}
    for md in presumed:
        grid = paper_grid(md)
        lims = []
        for size in sizes:
            for sub in combinations(range(4), size):
                Lc = np.column_stack([steer(M, grid[i]) for i in sub])
                nu = NU0[list(sub)]
                if feasibility_margin(Lc, nu * s_max) > 0:
                    lims.append((list(sub), s_max))
                    continue
                lo, hi = 1.0, s_max
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if feasibility_margin(Lc, nu * mid) > 0 else (lo, mid)
                lims.append((list(sub), lo))
        out[md] = lims
    return out


def bound_ablation(NR=25, K=300):
    """Final-snapshot SINR with the adaptation (3.20) and with the bounds held at nu0 (seeds 20000 + r)."""
    out = {}
    for lab in TABLE_ORDER:
        md, mp = SCEN[lab]
        Rsd = desired_cov(M, 0.0, mp, PS); thp = np.deg2rad(md); ap = steer(M, thp); ca = paper_grid(md)
        A, F = [], []
        for r in range(NR):
            np.random.seed(20000 + r)
            X = gen(M, K, 0.0, mp)
            A.append(sinr_cov_db(das_mbwc(X, ap, ca, NU0, ETA)[-1], Rsd, R_IPN))
            F.append(sinr_cov_db(das_mbwc(X, ap, ca, NU0, ETA, adapt=False)[-1], Rsd, R_IPN))
        out[lab] = dict(adaptive=float(np.mean(A)), fixed_nu0=float(np.mean(F)))
    return out


def correction_ratio(NR=25, K=300, k0=150):
    """||L_A tau|| / ||w~|| over snapshots k0..K-1 at which a correction is applied (seeds 20000 + r)."""
    out = {}
    for lab in TABLE_ORDER:
        md, mp = SCEN[lab]
        thp = np.deg2rad(md); ap = steer(M, thp); ca = paper_grid(md)
        V = []
        for r in range(NR):
            np.random.seed(20000 + r)
            X = gen(M, K, 0.0, mp)
            log = {}
            das_mbwc(X, ap, ca, NU0, ETA, log=log)
            V += [v for k, v in log.get('dw', []) if k >= k0]
        V = np.array(V)
        out[lab] = dict(median=float(np.median(V)), mean=float(V.mean()), p90=float(np.percentile(V, 90)))
    return out


def feasibility_check(NR=25, K=300):
    """Largest normalized constraint residual max_i g_i(w)/||w|| of the final weight at nu0
    (worst trial) and at 1.832 nu0 (best trial); negative means strictly feasible (seeds 20000 + r)."""
    out = {}
    for lab in TABLE_ORDER:
        md, mp = SCEN[lab]
        thp = np.deg2rad(md); ap = steer(M, thp); ca = paper_grid(md)
        Lc = np.column_stack([steer(M, t) for t in ca])
        G0, G1 = [], []
        for r in range(NR):
            np.random.seed(20000 + r)
            X = gen(M, K, 0.0, mp)
            w = das_mbwc(X, ap, ca, NU0, ETA)[-1]
            n = np.linalg.norm(w); re_ = np.real(w.conj() @ Lc)
            G0.append(np.max(1 + NU0 * n - re_) / n)
            G1.append(np.max(1 + 1.832 * NU0 * n - re_) / n)
        out[lab] = dict(worst_at_nu0=float(max(G0)), best_at_1832nu0=float(min(G1)))
    return out


def mc_check(NR=200, K=300):
    """Table 3.2 recomputed with NR trials (seeds 20000 + r): mean and 95 % half-width per method.
    Method order: Std MVDR, DL-MVDR, WC-MVDR, adaptive single-WC, INC, DAS-MBWC, all-constraint."""
    out = {}
    for lab in TABLE_ORDER:
        md, mp = SCEN[lab]
        Rsd = desired_cov(M, 0.0, mp, PS); thp = np.deg2rad(md); ap = steer(M, thp); ca = paper_grid(md)
        V = [[] for _ in range(7)]
        for r in range(NR):
            np.random.seed(20000 + r)
            X = gen(M, K, 0.0, mp)
            Rs = batch_cov(X)
            V[0].append(_safe(lambda: sinr_cov_db(mvdr(Rs, ap), Rsd, R_IPN)))
            V[1].append(_safe(lambda: sinr_cov_db(dl_mvdr(Rs, ap, 2 * SIGMA2), Rsd, R_IPN)))
            V[2].append(_safe(lambda: sinr_cov_db(wc_mvdr(Rs, ap, ZETA), Rsd, R_IPN)))
            V[3].append(sinr_cov_db(adaptive_single_wc(X, ap, ZETA, ETA)[-1], Rsd, R_IPN))
            V[4].append(_safe(lambda: sinr_cov_db(inc_reconstruction(Rs, ap, thp, RECON_SECTOR_DEG), Rsd, R_IPN)))
            V[5].append(sinr_cov_db(das_mbwc(X, ap, ca, NU0, ETA)[-1], Rsd, R_IPN))
            V[6].append(sinr_cov_db(das_mbwc(X, ap, ca, NU0, ETA, select='all')[-1], Rsd, R_IPN))
        mean = [float(np.mean(v)) for v in V]
        ci = [float(1.96 * np.std(v, ddof=1) / np.sqrt(NR)) if NR > 1 else float('nan') for v in V]
        out[lab] = dict(mean=mean, ci95=ci)
    return out


def signed_mismatch(NR=25, K=300, angles=tuple(range(-7, 8))):
    """Signed mismatch sweep (point source): asymmetric four-point grid versus the symmetric
    five-point grid [-4, -2, 0, +2, +4] deg with bounds [1.1, 1.5, 1.9, 1.5, 1.1]."""
    Rsd = desired_cov(M, 0.0, None, PS)
    res = {'angles': list(angles), 'asymmetric': [], 'symmetric': []}
    for md in angles:
        thp = np.deg2rad(md); ap = steer(M, thp)
        a_, s_ = [], []
        for r in range(NR):
            np.random.seed(61000 + 100 * (md + 10) + r)
            X = gen(M, K, 0.0, None)
            a_.append(sinr_cov_db(das_mbwc(X, ap, paper_grid(md), NU0, ETA)[-1], Rsd, R_IPN))
            s_.append(sinr_cov_db(das_mbwc(X, ap, [np.deg2rad(md + o) for o in (-4, -2, 0, 2, 4)],
                                           np.array([1.1, 1.5, 1.9, 1.5, 1.1]), ETA)[-1], Rsd, R_IPN))
        res['asymmetric'].append(float(np.mean(a_)))
        res['symmetric'].append(float(np.mean(s_)))
    return res
