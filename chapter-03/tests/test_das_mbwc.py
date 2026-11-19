"""Sanity tests for ch3dasmbwc. Run with:  pytest -q"""
import numpy as np
import pytest

from ch3dasmbwc import (steer, sinr_db, sinr_cov_db, sinr_opt_db, desired_cov, wc_mvdr, das_mbwc,
                        adaptive_single_wc, inc_reconstruction, mbwc_socp, feasibility_margin,
                        M, PS, ETA, ZETA, NU0, R_IPN, MP_SMALL, paper_grid, gen, batch_cov)


def cosine(u, v):
    return abs(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))


def data(md, mp=None, seed=7, K=300):
    np.random.seed(seed)
    return gen(M, K, 0.0, mp)


def test_steering_vector_and_distances():
    a = steer(M, np.deg2rad(20))
    assert np.isclose(np.linalg.norm(a), np.sqrt(M)) and np.isclose(a[0], 1.0)
    d = lambda t: np.linalg.norm(steer(M, 0.0) - steer(M, np.deg2rad(t)))
    # distances quoted in Section 3.2: non-monotonic in angle
    assert np.round(d(2), 2) == 3.59 and np.round(d(5), 2) == 6.13 and np.round(d(7), 2) == 5.73


def test_wc_mvdr_matches_conic_solver():
    """Exact single worst-case beamformer equals the SOCP of (3.8), with the constraint active."""
    X = data(5)
    R = batch_cov(X)
    a0 = steer(M, np.deg2rad(5))
    w = wc_mvdr(R, a0, ZETA)
    wo, status, res, _, _ = mbwc_socp(R, a0[:, None], np.array([ZETA]), solver='CLARABEL')
    assert status == 'optimal'
    assert cosine(w, wo) > 1 - 1e-5
    assert np.isclose(np.real(np.vdot(w, a0)) - ZETA * np.linalg.norm(w), 1.0, atol=1e-8)


def test_point_source_metrics_agree():
    a0 = steer(M, 0.0)
    Rs = desired_cov(M, 0.0, None, PS)
    w = np.linalg.solve(R_IPN, a0)
    assert np.isclose(sinr_cov_db(w, Rs, R_IPN), sinr_db(w, a0, R_IPN, PS))
    assert np.isclose(sinr_opt_db(Rs, R_IPN), sinr_db(w, a0, R_IPN, PS))


def test_gradient_step_is_scale_invariant_and_decreases_power():
    """Theorem 3.1: the step (3.10) is unchanged under R -> cR and lowers w^H R w."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((M, 40)) + 1j * rng.standard_normal((M, 40))
    R = A @ A.conj().T / 40 + 0.1 * np.eye(M)
    w = steer(M, 0.3)

    def step(R):
        r = R @ w
        mu = 0.1 * np.real(r.conj() @ r) / np.real(r.conj() @ R @ r)
        return w - mu * r

    w1, w2 = step(R), step(1e6 * R)
    assert np.allclose(w1, w2)
    assert np.real(w1.conj() @ R @ w1) < np.real(w.conj() @ R @ w)


def test_das_mbwc_is_reproducible():
    X = data(5)
    ca = paper_grid(5)
    ap = steer(M, np.deg2rad(5))
    w1 = das_mbwc(X, ap, ca, NU0, ETA)[-1]
    w2 = das_mbwc(data(5), ap, ca, NU0, ETA)[-1]
    assert np.array_equal(w1, w2)


def test_das_mbwc_final_weight_feasible_at_reference_bounds():
    """Section 3.6.6: the returned weight meets every constraint strictly at the reference bounds."""
    for md, mp in ((2, None), (7, None), (5, MP_SMALL)):
        X = data(md, mp)
        ca = paper_grid(md)
        w = das_mbwc(X, steer(M, np.deg2rad(md)), ca, NU0, ETA)[-1]
        L = np.column_stack([steer(M, t) for t in ca])
        g = 1 + NU0 * np.linalg.norm(w) - np.real(w.conj() @ L)
        assert np.max(g) < 0


def test_active_set_and_ablation():
    X = data(7)
    ca = paper_grid(7)
    ap = steer(M, np.deg2rad(7))
    log, log_all = {}, {}
    das_mbwc(X, ap, ca, NU0, ETA, log=log)
    das_mbwc(X, ap, ca, NU0, ETA, select='all', log=log_all)
    assert len(log['A']) == X.shape[1]
    assert any(0 < len(A) < len(ca) for A in log['A'])                 # selective enforcement
    assert all(len(A) in (0, len(ca)) for A in log_all['A'])          # ablation: all or none
    assert all(r >= 0 for _, r in log['dw'])


def test_rescaling_keeps_norm_bounded():
    X = data(7, K=1500)
    w = das_mbwc(X, steer(M, np.deg2rad(7)), paper_grid(7), NU0, ETA)
    assert max(np.linalg.norm(v) for v in w) <= 1e12


def test_gram_clipping_example():
    """Two-dimensional example of Section 3.4.3: the Gram solve has a negative coefficient."""
    l1, l2 = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])
    G = np.array([[l1 @ l1, l1 @ l2], [l2 @ l1, l2 @ l2]])
    tau = np.linalg.solve(G, np.array([1.0, 0.1]))
    assert np.allclose(tau, [19 / 15, -8 / 15])
    dw_clip = tau[0] * l1                       # clipped correction
    assert np.isclose(np.linalg.norm(dw_clip) / 1.0 - 1, 4 / 15)       # about 27 per cent longer


def test_slater_margin_and_feasibility_limit():
    """Appendix 3.A.4: t* = 1.08 at the reference bounds and t* = 0 near 1.72 nu0."""
    L = np.column_stack([steer(M, t) for t in paper_grid(5)])
    assert np.isclose(feasibility_margin(L, NU0), 1.08, atol=0.01)
    assert feasibility_margin(L, 1.70 * NU0) > 0 > feasibility_margin(L, 1.75 * NU0)


def test_mbwc_socp_meets_constraints():
    X = data(5)
    ca = paper_grid(5)
    L = np.column_stack([steer(M, t) for t in ca])
    w, status, res, _, _ = mbwc_socp(batch_cov(X), L, NU0)
    assert status in ('optimal', 'optimal_inaccurate') and res < 1e-3


def test_baselines_return_finite_weights():
    X = data(2)
    R = batch_cov(X)
    ap = steer(M, np.deg2rad(2))
    for w in (adaptive_single_wc(X, ap, ZETA)[-1], inc_reconstruction(R, ap, np.deg2rad(2))):
        assert np.all(np.isfinite(w))
