"""Sanity tests for the Chapter 7 package: exact lag-sum Capon scan, exact loaded worst-case solution with its
certificate, restoration, alpha-beta tracker form, metric scale invariance, power ceiling and fixed-seed values."""
import numpy as np
import pytest

from ch7rdfisac.algorithms import Scanner, Tracker, wc_loaded, gains, gain_floors, run_rdf_as_isac, upa_mat
from ch7rdfisac.projection import run_projection_reconstruction
from ch7rdfisac.rxmodel import evaluate_rx, steer
from ch7rdfisac.scenes import C, S, L, n, nu, CENTERS, TRACK_KS, scene
from ch7rdfisac import scenarios


def _rand_cov(M, seed=0, k=None):
    r = np.random.default_rng(seed); X = r.normal(size=(M, k or 3*M)) + 1j*r.normal(size=(M, k or 3*M))
    return X@X.conj().T/X.shape[1] + np.eye(M)


def test_lag_sums_equal_direct_capon():
    """(7.16): the lag-sum evaluation of a^H R^{-1} a equals the direct quadratic form."""
    for Mx, My in [(8, 4), (4, 4), (16, 8)]:
        M = Mx*My; B = np.linalg.inv(_rand_cov(M, Mx)); sc = Scanner(Mx, My); c = sc.lag_sums(B)
        az = np.array([-41.3, 0.0, 12.7, 63.0]); el = np.array([-7.0, 0.0, 4.5, 11.0])
        lag = sc.quad(c, *sc._basis(az, el))
        A = upa_mat(Mx, My, az, el); direct = np.einsum('mg,mn,ng->g', A.conj(), B, A).real
        assert np.allclose(lag, direct, rtol=1e-10, atol=1e-12)


def test_loaded_solution_feasible_certified_and_matches_socp():
    """(7.23)-(7.25): every constraint holds, the certificate is tiny and the objective equals a tight conic solve."""
    Q = np.eye(32, dtype=complex) + 300*np.outer(steer(8, 4, 33, 4), steer(8, 4, 33, 4).conj())
    b = gain_floors(0.5, 6, 6)
    w, _, it, cert = wc_loaded(Q, L, 3*nu, b=b)
    assert (gains(w, L, 3*nu) - b).min() >= 0
    assert 0 <= cert < 1e-6 and it <= 300
    assert abs(scenarios.matched_gap(Q, w, 3*nu, b, L)) < 1e-6


def test_gain_floors():
    """(7.22): alpha = 0.5 gives unit floors; the 0.1 floor protects both endpoints."""
    assert np.allclose(gain_floors(0.5, 6, 6), 1)
    b = gain_floors(1.0, 6, 6); assert np.allclose(b[:6], np.sqrt(2)) and np.allclose(b[6:], np.sqrt(0.1))


def test_alpha_beta_innovation_form():
    """(7.18d): the rate update equals rate + alpha_p beta_v (theta_p - predicted)/dk."""
    tr = Tracker([(20, 5)], 9, 4, alpha_p=0.5)
    tr.trk = [[40.0, 2.0, 100.0, 0, 9, 0.05, 0.0]]
    th_pred = 40.0 + 0.05*10; peak = (40.9, 2.0, 100.0)
    tr.update([peak], 19, beta_v=0.3)
    t = tr.trk[0]
    assert np.isclose(t[5], 0.05 + 0.5*0.3*(peak[0]-th_pred)/10)
    assert np.isclose(t[0], th_pred + 0.5*(peak[0]-th_pred))


def test_power_ceiling_blocks_strong_peak():
    """A served direction never claims a peak more than 6 dB above the nominal user SNR (8 dB)."""
    tr = Tracker([(20, 5)], 9, 4, user_snr_db=8.0, user_tol_db=6.0, ceiling=True)
    tr.update([(24.0, 5.0, 10**2.5)], 9, beta_v=0.3, s2=1.0)      # 25 dB peak inside the window
    assert not tr.claimed[0] and len(tr.trk) == 1
    tr = Tracker([(20, 5)], 9, 4, user_snr_db=8.0, user_tol_db=6.0, ceiling=True)
    tr.update([(24.0, 5.0, 10**0.8)], 9, beta_v=0.3, s2=1.0)      # 8 dB peak: claimed
    assert tr.claimed[0] and len(tr.trk) == 0


def test_metrics_scale_invariant():
    """(7.5)-(7.6) do not depend on the scale of w, so restoration leaves every metric unchanged."""
    X, trs = scene('D', 11, KK=40)
    w = np.random.default_rng(1).normal(size=32) + 1j*np.random.default_rng(2).normal(size=32)
    assert np.allclose(evaluate_rx(w, trs[-1]), evaluate_rx(-3.7j*w, trs[-1]))


def test_every_output_weight_feasible():
    """All output weights, including those held between refreshes, satisfy (7.12) (Section 7.6.2)."""
    X, _ = scene('G', 9901)
    h, d = run_rdf_as_isac(X, C, S, n, n, CENTERS, return_diagnostics=True)
    assert min(x['min_margin'] for x in d['per_snapshot']) >= -1e-9
    assert max(d['certified_gaps']) < 1e-6 and len(d['solver_iterations']) == 30


def test_projection_reconstruction_feasible():
    X, _ = scene('A', 3, KK=60)
    h = run_projection_reconstruction(X, C, S, n, n)
    assert (gains(h[-1], L, 3*nu) - 1).min() >= -1e-8


def test_seed_reproducibility():
    X1, _ = scene('F', 5003, KK=30); X2, _ = scene('F', 5003, KK=30)
    assert np.array_equal(X1, X2)


@pytest.mark.parametrize('sc, ref', [('D', [11.02, 9.49, 10.43, 9.51]), ('G', [8.80, 7.97, 4.64, 4.06])])
def test_fixed_seed_values(sc, ref):
    """Seed 9901 is used nowhere in the chapter; these values were recorded with the same code."""
    X, trs = scene(sc, 9901); h = run_rdf_as_isac(X, C, S, n, n, CENTERS)
    met = np.mean([evaluate_rx(h[k], trs[k]) for k in TRACK_KS], 0) if sc == 'G' else evaluate_rx(h[-1], trs[-1])
    assert np.allclose(np.round(met, 2), ref, atol=0.011)


def test_main_stage_reference_scene():
    """Reference scene A, seed 1: per-scene metrics used in Table 7.3."""
    r = scenarios.main_job(('reference', 'A', 1))
    assert np.allclose(r['metrics']['rdf_as_isac'], [10.815685043268697, 10.748417929223491, 10.394999144852184, 9.042946472962786], atol=1e-9)
    assert np.allclose(r['metrics']['socp_radius'], [10.481553175392921, 10.455174285238307, 10.11761896486643, 9.049242822681823], atol=1e-6)
    assert r['solver_iterations'][:5] == [62, 63, 63, 60, 57]
