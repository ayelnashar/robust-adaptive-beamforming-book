"""Sanity tests of the Chapter 5 code: solver and propositions, constraints, planar steering, oracle bound and seed
reproducibility.  Run with `pytest -q` from the chapter folder."""
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy.linalg import eigh as geigh

from ch5mprrab import algorithms as alg
from ch5mprrab import channel as chn
from ch5mprrab import scenarios as MS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter('ignore', RuntimeWarning)


def _random_problem(rng, M, K):
    B = rng.standard_normal((M, 2 * M)) + 1j * rng.standard_normal((M, 2 * M))
    R = B @ B.conj().T / (2 * M) + 0.05 * np.eye(M)
    R = R / (np.trace(R).real / M)
    th = np.sort(rng.uniform(-1.2, 1.2, K))
    A = np.column_stack([chn.steer(M, t) for t in th])
    p = rng.uniform(0.1, 1.0, K)
    return R, A, p


@pytest.mark.parametrize('M,K', [(16, 2), (16, 4), (64, 3), (64, 5)])
def test_proposition_5_2_zero_margin_is_max_sinr(M, K):
    """With the max-SINR targets and phases and zero margins the solution is the generalized eigenvector."""
    rng = np.random.default_rng(100 * M + K)
    R, A, p = _random_problem(rng, M, K)
    r = alg.maxsinr_responses(R, A, p)
    At = A * np.exp(1j * np.angle(r))[None, :]
    Q, L = alg.real_form(R, At)
    s, U = np.linalg.eigh(0.5 * (Q + Q.T))
    wr, _, _, _ = alg.as_snc((np.maximum(s, 1e-14), U), L, np.zeros(K), np.abs(r) / np.abs(r).max())
    w = wr[:M] + 1j * wr[M:]
    H = (A * p) @ A.conj().T
    ev, V = geigh((H + H.conj().T) / 2, R)
    cos = abs(w.conj() @ V[:, -1]) / np.linalg.norm(w) / np.linalg.norm(V[:, -1])
    assert cos > 1 - 1e-9
    sinr = np.real(w.conj() @ H @ w) / np.real(w.conj() @ R @ w)
    assert abs(sinr - ev[-1]) / ev[-1] < 1e-9


@pytest.mark.parametrize('kappa', [0.1, 0.3, 0.6])
def test_solver_meets_every_worst_case_constraint(kappa):
    """The active-set solution satisfies Re{w^H a~_k} >= nu_k ||w|| + beta_k and is certified by its KKT residuals."""
    rng = np.random.default_rng(int(10 * kappa))
    M, K = 16, 3
    R, A, p = _random_problem(rng, M, K)
    r = alg.maxsinr_responses(R, A, p)
    At = A * np.exp(1j * np.angle(r))[None, :]
    beta = np.abs(r) / np.abs(r).max()
    nu = np.full(K, kappa * np.sqrt(M / K))          # margins of Section 5.3.2
    Q, L = alg.real_form(R, At)
    s, U = np.linalg.eigh(0.5 * (Q + Q.T))
    wr, n_out, act, mu = alg.as_snc((np.maximum(s, 1e-14), U), L, nu, beta)
    assert alg.LAST_SOLVER_STATUS['certified']
    assert np.all(L.T @ wr - nu * np.linalg.norm(wr) - beta >= -1e-8)


def test_planar_steering_and_encoding():
    """Direction encoding round trip and unit-modulus planar steering vectors (squared norm 64)."""
    from ch5mprrab import planar as P
    for az, el in [(0.0, 10.0), (-30.0, 15.0), (45.0, 0.0), (-10.0, -25.0)]:
        th = P.encode(np.deg2rad(az), np.deg2rad(el))
        a_, e_ = P.az_el_deg(th)
        assert abs(a_ - az) < 1e-9 and abs(e_ - el) < 0.0125
        v = P.steer_upa(64, th)
        assert v.shape == (64,) and np.allclose(np.abs(v), 1.0)


def test_oracle_bounds_mpr_rab():
    """No beamformer exceeds the oracle expected SINR at any snapshot (Proposition 5.1 setting)."""
    d = MS.make_data('B', 5, 10, 0, 4242)
    c = d['c']
    h, _ = alg.run_mpr_rab(d['X'], d['a0'], list(d['th0']), K_track=c['K_track'], K_use=c['K_use'], use_tracking=True,
                           use_lifecycle=False, kappa=0.4, targets='ms')
    for k in range(0, d['T'], 25):
        wo = chn.run_oracle_mc(d['R_ipn'], d['thl'][k], d['pwl'][k], chn.M_DEFAULT)
        so = chn.sinr_db_cluster(wo, d['thl'][k], d['pwl'][k], d['R_ipn'], chn.PS)
        sm = chn.sinr_db_cluster(h[k], d['thl'][k], d['pwl'][k], d['R_ipn'], chn.PS)
        assert sm <= so + 1e-9


def test_methods_leave_the_random_stream_alone_and_seeds_reproduce():
    """make_data reproduces its data from the seed, and MPR-RAB does not touch the global random stream, so every
    method of a cell sees identical data."""
    d1 = MS.make_data('C', 5, 10, 2, 777)
    st = np.random.get_state()
    h, _ = alg.run_mpr_rab(d1['X'], d1['a0'], list(d1['th0']), K_track=2, K_use=2, use_tracking=True, use_lifecycle=False,
                           kappa=0.4, targets='ms')
    st2 = np.random.get_state()
    assert st[0] == st2[0] and np.array_equal(st[1], st2[1]) and st[2] == st2[2]
    d2 = MS.make_data('C', 5, 10, 2, 777)
    assert np.array_equal(d1['X'], d2['X'])
    m = MS.metrics(h, d1, 'C')
    # reference values of this run (NumPy 2.5, single-threaded BLAS)
    assert abs(m[''] - 16.836758495527683) < 1e-6
    assert abs(m['|floor'] - 6.3230974102625295) < 1e-6


def test_reference_run_scenario_A():
    d = MS.make_data('A', 5, 10, 0, 12345)
    h, _ = alg.run_mpr_rab(d['X'], d['a0'], list(d['th0']), K_track=2, K_use=2, use_tracking=True, use_lifecycle=False,
                           kappa=0.4, targets='ms')
    m = MS.metrics(h, d, 'A')
    assert abs(m[''] - 17.02801003323224) < 1e-6
    assert abs(m['|floor'] - 8.442276535327716) < 1e-6


def test_selection_rule_and_shipped_selection():
    """The shipped selection has the margins of the chapter: kappa = 0.3 on the planar and 0.4 on the linear array."""
    assert MS.kappa_of(MS.load_selection('upa')['MPR-RAB']) == 0.3
    assert MS.kappa_of(MS.load_selection('ula')['MPR-RAB']) == 0.4
    R = {'A': {'5_10_0': {'MPR k=0.3': [10.0], 'MPR k=0.3|floor': [5.0], 'MPR k=0.4': [10.03], 'MPR k=0.4|floor': [7.0],
                          'MPR k=0.5': [9.0], 'MPR k=0.5|floor': [9.0]}}}
    best, _, _ = MS.select(R)
    assert best['MPR-RAB'] == 'MPR k=0.4'          # within 0.05 dB of the best: larger minimum-cluster SINR decides


def test_planar_array_in_its_own_process():
    """use_array('upa') routes the steering model to the 8 x 8 array (one process per array)."""
    code = ('import warnings; warnings.simplefilter("ignore"); from ch5mprrab import scenarios as MS, channel as chn, algorithms as alg;'
            'MS.use_array("upa"); d = MS.make_data("A", 5, 10, 0, 7); assert d["X"].shape == (64, 300);'
            'assert chn.M_DEFAULT == 64 and alg.GEOM.name == "UPA"; print("ok")')
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS='1'))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith('ok')
