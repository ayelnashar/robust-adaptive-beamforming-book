"""Sanity tests for ch2assnc. Run with:  pytest -q"""
import os

import numpy as np
import pytest

from ch2assnc import (as_snc, quality, conic_solve, random_instance, load_ff,
                      build_portfolio_problem, portfolio_model, regression_checks)
from ch2assnc.figures import load_records, summarize

ARCHIVE = os.path.join(os.path.dirname(__file__), '..', 'data', 'archived_run')


@pytest.mark.parametrize('nu', [0.0, 0.2, 0.5, 0.9, 0.99])
def test_scalar_closed_form(nu):
    """Section 2.2.6: min w^2 s.t. w >= nu|w| + beta has w* = beta / (1 - nu)."""
    beta = 1.0
    w, it, A, info = as_snc(np.eye(1), np.eye(1), np.array([nu]), np.array([beta]), return_info=True)
    assert w[0] == pytest.approx(beta / (1 - nu), rel=1e-9)
    assert info['method'] == 'AS-SNC'
    if nu > 0:                       # multiplier lambda* = 2 beta / (1 - nu)^2
        assert info['lam'][0] == pytest.approx(2 * beta / (1 - nu) ** 2, rel=1e-6)


@pytest.mark.parametrize('nu', [1.0, 1.1])
def test_scalar_infeasible_is_rejected(nu):
    with pytest.raises(RuntimeError):
        as_snc(np.eye(1), np.eye(1), np.array([nu]), np.ones(1))


def test_repelling_fixed_point_instance():
    """Section 2.4.2: the optimum is w* = (1, 0); a certified point must be returned."""
    Q = np.array([[1., 2.], [2., 5.]])
    L = np.array([[3., 1.], [3.5, .5]])
    w, it, A, info = as_snc(Q, L, np.ones(2), np.array([2., 0.]), return_info=True)
    assert np.allclose(w, [1., 0.], atol=1e-6)
    assert info['residual'] <= 1e-8


def test_nonpositive_offsets_return_origin():
    w, it, A, info = as_snc(np.eye(3), np.eye(3), np.full(3, .1), -np.ones(3), return_info=True)
    assert np.all(w == 0) and len(A) == 0 and info['primal'] == 0


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_agrees_with_clarabel(seed):
    rng = np.random.default_rng(seed)
    x = random_instance(40, 10, 100, rng)
    w, it, A, info = as_snc(*x, tol=1e-9, return_info=True)
    wc, _ = conic_solve(*x, 'CLARABEL', 1e-10)
    Q = x[0]
    assert info['residual'] <= 1e-9
    assert quality(*x, w, info['lam'])['primal'] <= 1e-9
    assert abs(w @ Q @ w - wc @ Q @ wc) / (wc @ Q @ wc) < 1e-6


def test_constraints_hold_on_returned_active_set():
    x = random_instance(60, 20, 1000, np.random.default_rng(5))
    Q, L, nu, beta = x
    w, it, A, info = as_snc(*x, return_info=True)
    slack = L.T @ w - nu * np.linalg.norm(w) - beta
    assert np.all(slack >= -1e-8 * (1 + abs(beta)))                 # feasible
    assert np.allclose(slack[A], 0, atol=1e-7)                      # active constraints are tight


def test_scale_invariance():
    x = random_instance(50, 10, 1000, np.random.default_rng(3))
    Q, L, nu, beta = x
    w0 = as_snc(*x)[0]
    for tau in (1e-3, 1e3):
        w = as_snc(tau * Q, L, nu, beta)[0]
        assert np.linalg.norm(w - w0) / np.linalg.norm(w0) < 1e-9


def test_generator_is_seed_reproducible():
    a = random_instance(30, 8, 100, np.random.default_rng(42))
    b = random_instance(30, 8, 100, np.random.default_rng(42))
    assert all(np.array_equal(p, q) for p, q in zip(a, b))


def test_regenerated_dimension_sweep_instances_match_archive():
    """The first instances of the dimension sweep (seed 0) give the archived solutions."""
    rows = load_records(ARCHIVE)['scaling']
    rng = np.random.default_rng(0)
    for row in rows[:4]:
        x = random_instance(25, 20, 1000, rng)
        w, it, A, info = as_snc(*x, tol=1e-9, return_info=True)
        rec = row['methods']['AS-SNC']
        assert len(A) == rec['active'] and it == rec['iterations']
        assert info['objective'] == pytest.approx(rec['objective'], rel=1e-12)


def test_fama_french_data():
    for name, M in [('FF-Ind12', 12), ('FF-All21', 21), ('FF-All30', 30)]:
        df = load_ff(name)
        assert df.shape == (819, M)
        Sigma, L, nu, beta, mu = build_portfolio_problem(df)
        assert L.shape == (M, 4) and np.all(np.linalg.eigvalsh(Sigma) > 0)
        w, it, A, info = as_snc(Sigma, L, nu, beta, return_info=True)
        assert info['method'] == 'AS-SNC' and info['residual'] <= 1e-9


def test_portfolio_feasibility_limit():
    Q, L, beta, mu = portfolio_model()
    as_snc(Q, L, np.r_[0, np.full(5, .14)], beta)                   # below 1/sqrt(50): solvable
    with pytest.raises(RuntimeError):
        as_snc(Q, L, np.r_[0, np.full(5, .15)], beta)               # above 1/sqrt(50): infeasible


def test_regression_checks_run():
    checks = regression_checks()
    assert len(checks) == 7 and checks[-1]['residual'] <= 1e-8


def test_input_validation():
    Q, L, nu, beta = random_instance(10, 3, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        as_snc(Q + 1j, L, nu, beta)
    with pytest.raises(ValueError):
        as_snc(Q + np.triu(np.ones_like(Q), 1), L, nu, beta)       # not symmetric
    with pytest.raises(ValueError):
        as_snc(Q, L, -nu, beta)                                     # negative radius
    with pytest.raises(ValueError):
        as_snc(-Q, L, nu, beta)                                     # not positive definite


def test_archived_summary():
    s = summarize(load_records(ARCHIVE))
    t = s['Table2.3_Fig2.1_dimension_sweep']
    assert set(t) == {'25', '50', '100', '200', '400'}
    assert all(row['AS-SNC']['passes_common_target'] == 8 for row in t.values())
    assert s['Fig2.4_active_set_cardinality']['N=20']['instances'] == 200
