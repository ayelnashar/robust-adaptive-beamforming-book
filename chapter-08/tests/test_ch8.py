"""Unit tests of the Chapter 8 package: projection (8.14), WRAB update, comparators, data model and reproducibility."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import eigh

from ch8wrab import algorithms as C
from ch8wrab.algorithms import Scenario

FC = 300e9


def test_steering_and_squint():
    a = C.steer_f(64, .2, FC, FC)
    assert np.allclose(abs(a), 1) and np.isclose(np.linalg.norm(a)**2, 64)
    assert np.isclose(C.analog_response(64, .2, .2, FC, FC), 1)
    f = 310e9; theta = np.deg2rad(30); peak = np.arcsin(FC/f*np.sin(theta))
    assert np.isclose(C.analog_response(128, theta, peak, f, FC), 1, atol=1e-12)   # carrier-fixed beam walks off


def test_subcarrier_centres():
    assert np.allclose((C.subcarrier_frequencies(FC, 20e9, 5) - FC)/1e9, [-8, -4, 0, 4, 8])


def test_noise_only_array_gain():
    a = C.steer_f(16, 0, FC, FC)
    assert np.isclose(C.sinr_linear(a/16, a, np.eye(16), 10**.5), 16*10**.5)


def test_projection_matches_slsqp():
    """(8.14): the dual Newton projection equals an independent SLSQP solution in eight complex cases with several
    active constraints and one zero-radius constraint."""
    rng = np.random.default_rng(121)
    for case in range(8):
        M = 8 + case % 3*4
        L = C.steering_matrix(M, np.deg2rad([-3, 0, 2]), FC, FC)
        nu = np.array([0 if case == 0 else .2, .5, .7])
        z = C.cn(rng, (M,))*1.3
        w, _ = C.project_wc(z, L, nu, tol=1e-11, max_cycles=3000)
        ref, _ = C.reference_constrained_minimum(np.eye(M), L, nu, target=z)
        assert np.linalg.norm(w - ref)/max(1, np.linalg.norm(ref)) < 2e-5
        assert np.max(C.wc_residual(w, L, nu)) < 2e-9


def test_dual_projection_equals_dykstra():
    rng = np.random.default_rng(5)
    L = C.steering_matrix(16, np.deg2rad([-3, 0, 3]), FC, FC); nu = np.array([.4, .5, .6]); z = C.cn(rng, (16,))
    wd, _, _, ok = C._dykstra(z, np.ascontiguousarray(L), nu, 1e-11, 3000)
    wn, _ = C.project_wc(z, L, nu, tol=1e-11)
    assert ok and np.linalg.norm(wd - wn) < 2e-7


def _fixed_problem():
    rng = np.random.default_rng(6)
    M = 12; T = C.cn(rng, (M, M)); R = T@T.conj().T/M + np.eye(M)
    L = C.steering_matrix(M, np.deg2rad([-4, 0, 4]), FC, FC); nu = np.full(3, .5)
    cfg = Scenario(M=M, margin_ceiling=1, initial_margins=(.5,), covariance_loading=0)
    return R, L, nu, cfg


def test_frozen_descent_reaches_constrained_optimum():
    """Section 8.3.4: with the covariance and margins frozen, the iteration reaches the constrained optimum."""
    R, L, nu, cfg = _fixed_problem()
    ref, _ = C.reference_constrained_minimum(R, L, nu)
    w, _ = C.project_wc(np.zeros(12, complex), L, nu)
    for _ in range(2500):
        w, _ = C.projected_step(R, w, L, nu, cfg)
    opt = float(np.real(np.vdot(ref, R@ref))); cost = float(np.real(np.vdot(w, R@w)))
    assert abs(cost - opt)/opt < 5e-5


def test_step_is_scale_invariant():
    R, L, nu, cfg = _fixed_problem()
    w, _ = C.project_wc(np.zeros(12, complex), L, nu)
    w1, _ = C.projected_step(R, w, L, nu, cfg); w2, _ = C.projected_step(1e7*R, w, L, nu, cfg)
    assert np.allclose(w1, w2, atol=1e-10, rtol=1e-9)


def test_spherical_wc_and_robust_capon():
    R, _, _, _ = _fixed_problem()
    a = C.steer_f(12, .1, FC, FC); radius = .7
    wk = C.spherical_wc_from_eig(eigh(R), a, radius)
    wr, _ = C.reference_constrained_minimum(R, a[:, None], [radius])
    assert abs(np.vdot(wk, R@wk).real/np.vdot(wr, R@wr).real - 1) < 2e-6
    wc = C.rcb_from_eig(eigh(R), a, radius**2)                      # same radius -> same beam direction
    assert abs(abs(np.vdot(wk, wc))/(np.linalg.norm(wk)*np.linalg.norm(wc)) - 1) < 1e-10


def test_invalid_and_infeasible_constraints_are_rejected():
    R, _, _, _ = _fixed_problem()
    a = C.steer_f(12, .1, FC, FC)
    for rad in (-1, np.linalg.norm(a), np.linalg.norm(a) + 1):
        with pytest.raises(ValueError):
            C.spherical_wc_from_eig(eigh(R), a, rad)
    with pytest.raises((ValueError, RuntimeError, np.linalg.LinAlgError)):
        C.project_wc(np.zeros(12, complex), np.column_stack([a, -a]), [0, 0], max_cycles=20)


def test_no_true_direction_reaches_the_update():
    """With interference-only training and a fixed presumed angle, the true direction does not change the weights."""
    cfg = Scenario(M=32, Nsc=1, K=120, margin_ceiling=2, initial_margins=(.5,), jitter_deg=0)
    changed = replace(cfg, th_d_deg=1, mismatch_deg=1)
    t1 = C.generate_trial(cfg, 77); t2 = C.generate_trial(changed, 77)
    assert np.array_equal(t1['X'][0], t2['X'][0])
    r1 = C.train_subband(t1['X'][0], FC, cfg, ('wrab',)); r2 = C.train_subband(t2['X'][0], FC, changed, ('wrab',))
    assert np.array_equal(r1['histories']['wrab'], r2['histories']['wrab'])


def test_streams_are_independent_and_reproducible():
    cfg = Scenario(M=32, Nsc=1, K=60)
    a = C.generate_trial(cfg, 77); C.generate_trial(cfg, C.seed_for('other', 3)); b = C.generate_trial(cfg, 77)
    assert np.array_equal(a['X'][0], b['X'][0])
    assert C.seed_for('fig3', 0) != C.seed_for('fig4', 0) != C.seed_for('fig4', 1)


def test_covariance_recursion():
    cfg = Scenario(M=16, Nsc=1, K=40)
    X = C.generate_trial(cfg, 9)['X'][0]
    Rs, mass = C.ewma_covariance(X, cfg.eta)
    wts = cfg.eta**np.arange(cfg.K-1, -1, -1)
    assert np.allclose(Rs, cfg.eta**cfg.K*np.eye(16) + (X*wts)@X.conj().T, rtol=1e-11, atol=1e-10)
    assert np.isclose(mass, (1 - cfg.eta**(cfg.K+1))/(1 - cfg.eta))


def test_every_update_is_feasible_for_all_methods():
    cfg = Scenario(Nsc=1, K=300, jitter_deg=.1)
    r = C.evaluate_trial(cfg, C.seed_for('online_validation'), C.ALL_METHODS)
    assert max(r['max_residual'].values()) < 2e-9
    assert all(np.isfinite(v).all() for v in r['linear'].values())


def test_asymmetric_grid():
    tp, grid, _ = C.design_grid(Scenario())
    assert np.allclose((grid - tp)/C.beamwidth_rad(64, tp), [0, -.6, .6, -1.2])


def test_frozen_norm_correction_is_not_feasible():
    """Section 8.3.1: the frozen-norm one-step correction violates the constraints in 296 of 300 updates, by up to
    0.67 (4.9e-3 over the last 100), while the exact projection stays within 1e-9 on the same snapshots."""
    M, K = 64, 300; thp = np.deg2rad(2); off = np.array([0, -.6, .6, -1.2])*1.772/M
    X = C.sequential_snapshots(M, K, FC, (45, -60), 10, seed=42)
    res, _ = C.frozen_norm_trace(X, FC, FC, thp, thp + off, [1.9, 1.5, 1.5, 1.1])
    assert np.count_nonzero(res > 1e-8) == 296
    assert round(res.max(), 2) == 0.67 and round(1e3*res[-100:].max(), 1) == 4.9
    t = C.train_subband(X, FC, Scenario(Nsc=1, grid_offsets_deg=tuple(np.rad2deg(off)), jitter_deg=0), ('wrab',))
    assert t['residuals']['wrab'].max() <= 1e-9


def test_multiball_socp_matches_wrab_fixed_point():
    """The exact solution of (8.8) by CVXPY equals the frozen-problem limit of the WRAB iteration."""
    pytest.importorskip('cvxpy')
    R, L, nu, cfg = _fixed_problem()
    ws, _ = C.multiball_socp(R, L, nu)
    w, _ = C.project_wc(np.zeros(12, complex), L, nu)
    for _ in range(2500):
        w, _ = C.projected_step(R, w, L, nu, cfg)
    obj = lambda v: float(np.real(np.vdot(v, R@v)))   # noqa: E731
    assert abs(obj(ws)/obj(w) - 1) < 1e-4
