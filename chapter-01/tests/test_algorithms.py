"""Sanity tests for ch1rab. Run with:  pytest -q"""
import numpy as np
import pytest
from ch1rab import (steer, rcb_eig, WCSOCP, wc_eig, WCAdaptive, VLRCB, RLS, capon, lcmv,
                    tap_run, iet_run, TAP_METHODS, WC_METHODS)

M = 5


def cosine(u, v):
    return abs(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))


@pytest.fixture
def cov():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, 40)) + 1j * rng.standard_normal((M, 40))
    R = A @ A.conj().T / 40 + 0.1 * np.eye(M)
    return R, np.linalg.inv(R)


def test_steering_vector_norm():
    a = steer(np.deg2rad(20), M)
    assert np.isclose(np.linalg.norm(a), np.sqrt(M))
    assert np.isclose(a[0], 1.0)


def test_rcb_matches_socp_with_same_radius(cov):
    """Li-Stoica equivalence: spherical RCB and the WC SOCP give the same direction."""
    R, IR = cov
    a0 = steer(np.deg2rad(5), M)
    eps = 0.5
    wd, _ = rcb_eig(R, IR, a0, eps)
    wo = WCSOCP(M).solve(R, a0, np.sqrt(eps))
    assert cosine(wd, wo) > 1 - 1e-6


def test_wc_eig_matches_socp(cov):
    R, _ = cov
    a0 = steer(np.deg2rad(5), M)
    we, _ = wc_eig(R, a0, 1.8)
    wo = WCSOCP(M).solve(R, a0, 1.8)
    assert cosine(we, wo) > 1 - 1e-6
    # the SOCP solution satisfies the WC constraint with equality
    assert np.isclose(np.real(np.vdot(wo, a0)) - 1.8 * np.linalg.norm(wo), 1.0, atol=1e-5)


def test_wc_adaptive_detects_infeasible_zeta(cov):
    R, IR = cov
    a0 = steer(np.deg2rad(5), M)
    wca = WCAdaptive(a0, zeta=2.4)          # > ||a0|| = sqrt(5)
    wca.update(R, IR)
    assert wca.log[-1][-1].startswith('infeasible')


def test_wc_adaptive_feasible_update_satisfies_constraint(cov):
    R, IR = cov
    a0 = steer(np.deg2rad(5), M)
    wca = WCAdaptive(a0, zeta=1.8)
    w = wca.update(R, IR)
    if wca.log[-1][-1] == 'active':
        assert np.real(np.vdot(w, a0)) >= 1.8 * np.linalg.norm(w) + 1 - 1e-6


def test_lcmv_meets_constraints(cov):
    _, IR = cov
    C = np.column_stack([steer(np.deg2rad(t), M) for t in (0, -30, -80)])
    g = np.array([1.0, 0.4, 0.4])
    w = lcmv(IR, C, g)
    assert np.allclose(C.conj().T @ w, g, atol=1e-8)


def test_capon_distortionless(cov):
    _, IR = cov
    a = steer(0.3, M)
    assert np.isclose(np.vdot(capon(IR, a), a), 1.0)


def test_rls_inverse_tracks_covariance():
    rng = np.random.default_rng(1)
    rls = RLS(M)
    for _ in range(50):
        rls.update(rng.standard_normal(M) + 1j * rng.standard_normal(M))
    assert np.allclose(rls.IR @ rls.R, np.eye(M), atol=1e-8)


def test_scenarios_smoke():
    o = tap_run(0, N=30)
    assert set(TAP_METHODS) <= set(o)
    assert all(np.all(np.isfinite(o[m]['sinr'])) for m in TAP_METHODS)
    o = iet_run(0, N=30)
    assert set(o['sinr']) == set(WC_METHODS)
    assert all(np.all(np.isfinite(v)) for v in o['sinr'].values())


def test_runs_are_reproducible():
    a = iet_run(7, N=20)['sinr']['Robust MVDR-WC/Proposed']
    b = iet_run(7, N=20)['sinr']['Robust MVDR-WC/Proposed']
    assert np.array_equal(a, b)
