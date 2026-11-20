"""Unit tests of the Chapter 6 package: solvers, error model, reproducibility and reference values."""
import numpy as np
import pytest
from numpy.linalg import norm

from ch6cfrab import algorithms as C
from ch6cfrab.algorithms import CFConfig, run_snapshot_trace
from ch6cfrab.scenarios import exact_wc_grid, paired

M9 = ('D-CF-RAB', 'MR+W', 'LP-MMSE+W', 'LMVDR+W', 'P-MMSE', 'LP-MMSE', 'CF-RAB', 'RECON-RAB', 'MR')

# mean SINR (dB) of each method for L = 4, 12 snapshots, SNR = INR = 10 dB, seed 123 (default switches)
REFERENCE = {'D-CF-RAB': 17.315852819109658, 'MR+W': 17.28963128846012, 'LP-MMSE+W': 19.839168202634337,
             'LMVDR+W': 8.941697744476745, 'P-MMSE': 21.529312127214254, 'LP-MMSE': 14.516809970839429,
             'CF-RAB': 8.366465133181805, 'RECON-RAB': 9.078981677229823, 'MR': 3.5695904709110557}


def _random_problem(rng, n, cond):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    R = (Q * np.geomspace(1, cond, n)) @ Q.conj().T
    R = (R + R.conj().T) / 2
    h = rng.normal(size=n) + 1j * rng.normal(size=n)
    return R, h


@pytest.mark.parametrize('n', [2, 8, 32])
@pytest.mark.parametrize('cond', [1.0, 1e2, 1e4])
@pytest.mark.parametrize('c', [0.0, 0.1, 0.5, 0.98])
def test_exact_single_ball_solution(n, cond, c):
    """Section 6.3.6: the scalar-root solution is feasible, tight and matches a loading-grid search."""
    rng = np.random.default_rng(int(1000 * c) + n + int(np.log10(cond)))
    R, h = _random_problem(rng, n, cond)
    nu = c * norm(h)
    w = C.exact_wc(R, h, nu)
    assert w is not None
    margin = float(np.real(np.vdot(w, h))) - nu * norm(w) - 1.0
    assert abs(margin) < 1e-9
    w_ref = exact_wc_grid(R, h, nu)
    obj = lambda v: float(np.real(np.vdot(v, R @ v)))
    assert obj(w) <= obj(w_ref) * (1 + 1e-7)


def test_error_model_stays_in_declared_ball():
    """Every generated CSI error satisfies the relative ball of eq. (6.2)."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        h = rng.normal(size=8) + 1j * rng.normal(size=8)
        u = rng.normal(size=8) + 1j * rng.normal(size=8); u /= norm(u)
        for e in (0.01, 0.10, 0.50):
            hp = h + C._ball_radius(h, u, e) * u
            assert norm(hp - h) <= e * norm(hp) * (1 + 1e-12)


def test_algorithm_6_1_feasible():
    """Algorithm 6.1 returns a weight that satisfies every worst-case constraint of its grid."""
    rng = np.random.default_rng(11)
    R, h = _random_problem(rng, 8, 1e3)
    for eps in (0.05, 0.10, 0.30):
        Lg = C.build_constraint_grid(h, eps, N=4)
        nu = np.full(4, eps * norm(h))
        w, *_ = C.rab_step(R, np.zeros(8, dtype=complex), Lg, nu, C.RABConfig(), 1.0)
        marg = np.real(w.conj() @ Lg) - (nu * norm(w) + 1.0)
        assert marg.min() > -1e-9


def test_traces_are_reproducible():
    kw = dict(n_snapshots=6, SNR_dB=10, INR_dB=10, methods=('D-CF-RAB', 'P-MMSE'), seed=7, soi_in_cov=True)
    a = run_snapshot_trace(CFConfig(L=4, M_l=8, K=4), **kw)
    b = run_snapshot_trace(CFConfig(L=4, M_l=8, K=4), **kw)
    for m in kw['methods']:
        np.testing.assert_array_equal(a['sinr_dB'][m], b['sinr_dB'][m])


def test_reference_values():
    """Fixed-seed SINR of every method, so that any change of the simulator is noticed."""
    r = run_snapshot_trace(CFConfig(L=4, M_l=8, K=4), n_snapshots=12, SNR_dB=10, INR_dB=10, methods=M9, seed=123,
                           soi_in_cov=True)
    for m, v in REFERENCE.items():
        assert float(np.nanmean(r['sinr_dB'][m])) == pytest.approx(v, abs=1e-8), m


def test_listed_fronthaul_bytes():
    """Sections 6.4.2 and 6.5.4: 1280, 4352 and 2048 bytes per user update at L = 16, M = 8, K = 4."""
    ns = 3
    r = run_snapshot_trace(CFConfig(L=16, M_l=8, K=4), n_snapshots=ns, SNR_dB=10, INR_dB=10, methods=('D-CF-RAB',), seed=1,
                           soi_in_cov=True)
    assert r['fh_bytes_total'] / (ns * 4) == 1280.0
    L, K, M = 16, 4, 8
    assert (L * K * M * 16 + L * (M * (M + 1) // 2) * 16) / K == 4352.0
    assert (L * K * M * 16) / K == 2048.0


def test_paired_interval():
    lo_hi = paired([1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.0, 4.5])
    assert lo_hi[1] < lo_hi[0] < lo_hi[2]
    assert lo_hi[0] == pytest.approx(0.375)


def test_legend_names_match_chapter():
    from ch6cfrab import figures as F
    names = {v['label'] for v in F.STYLE.values()}
    assert {'D-CF-RAB', 'MR + CPU combiner', 'LP-MMSE + CPU combiner', 'Local MVDR + CPU combiner', 'CF-RAB (centralized)',
            'Primary-ball SOCP', 'INCM reconstruction'} <= names
