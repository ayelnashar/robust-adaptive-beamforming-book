"""Sanity tests for ch4rmber. Run with:  pytest -q"""
import numpy as np
from scipy.optimize import minimize

from ch4rmber import algorithms as A
from ch4rmber.algorithms import (steer, sinr_db, build_R_ipn, wc_grid, nu_pub, zeta, wc_mvdr_exact, proj_exact, run_rmber,
                                 run_blind, gen_with_symbols, gen_block, ber_of, qfunc, beamwidth_deg, pos_ula, run_sgrab_p,
                                 guide, ber_stage, guided, spss, ta_mmse, incm_rec, cov_ewma, M, K, ps, KAPPA, NU_FAC, R_ipn)


def block(seed=1000, Ml=M, Kl=K):
    np.random.seed(seed)
    return gen_with_symbols(Ml, Kl, 0.0, None)


def optimum(R=R_ipn, a=None):
    a = steer(M, 0.0).astype(complex) if a is None else a
    w = np.linalg.solve(R, a)
    return sinr_db(w, a, R, ps)


def test_steering_vector_and_beamwidth():
    a = steer(M, np.deg2rad(20))
    assert np.isclose(np.linalg.norm(a), np.sqrt(M)) and np.isclose(a[0], 1.0)
    # merging width of the guide: 1.772/(M - 1) rad for the half-wavelength ULA
    for Ml in (8, 16, 32, 64):
        assert np.isclose(beamwidth_deg(pos_ula(Ml)), np.rad2deg(1.772 / (Ml - 1)))


def test_wc_mvdr_exact_is_optimal_and_active():
    """The exact WC-MVDR/EigDec weight meets the worst-case constraint with equality and matches an independent solve."""
    X, S = block()
    Rs = cov_ewma(X); a = steer(M, np.deg2rad(5)); z = zeta(M)
    w = wc_mvdr_exact(Rs, a, z)
    assert np.isclose(np.real(np.vdot(a, w)) - z * np.linalg.norm(w), 1.0, atol=1e-9)
    # KKT stationarity of  min w^H R w  s.t.  Re{a^H w} - z ||w|| >= 1  (R with the solver's numerical loading):
    # R w = lam (a - z w / ||w||) for some lam > 0
    Rl = Rs + 1e-3 * np.trace(Rs).real / M * np.eye(M)
    v = a - z * w / np.linalg.norm(w)
    lam = np.real(np.vdot(v, Rl @ w)) / np.real(np.vdot(v, v))
    assert lam > 0 and np.linalg.norm(Rl @ w - lam * v) < 1e-8 * np.linalg.norm(Rl @ w)
    # and an independent local solve from a perturbed start does not find a lower feasible objective
    f = lambda u: float(np.real((u[:M] + 1j * u[M:]).conj() @ Rl @ (u[:M] + 1j * u[M:])))
    con = {'type': 'ineq', 'fun': lambda u: np.real((u[:M] + 1j * u[M:]).conj() @ a) - z * np.linalg.norm(u) - 1}
    r = minimize(f, np.concatenate([w.real, w.imag]) * 1.05, constraints=[con], method='SLSQP', options=dict(maxiter=2000, ftol=1e-15))
    w2 = r.x[:M] + 1j * r.x[M:]
    slack = np.real(np.vdot(a, w2)) - z * np.linalg.norm(w2) - 1
    assert slack < -1e-9 or np.real(w.conj() @ Rl @ w) <= np.real(w2.conj() @ Rl @ w2) * (1 + 1e-6)


def test_projection_feasible_and_idempotent():
    rng = np.random.default_rng(3)
    ca = wc_grid(5.0); L = np.column_stack([steer(M, t) for t in ca]); b = NU_FAC * nu_pub(M) * KAPPA + 1.0
    w0 = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) * 0.3
    w = proj_exact(w0, L, b, KAPPA)
    assert np.linalg.norm(w) <= KAPPA + 1e-9
    assert np.all(np.real(w.conj() @ L) >= b - 1e-8)
    assert np.allclose(proj_exact(w, L, b, KAPPA), w, atol=1e-10)


def test_trained_rmber_stays_feasible_and_beats_the_fixed_beam():
    X, S = block()
    a_p = steer(M, np.deg2rad(5)); ca = wc_grid(5.0); nu = NU_FAC * nu_pub(M)
    h = run_rmber(X, a_p, ca, nu, S)
    L = np.column_stack([steer(M, t) for t in ca]); b = nu * KAPPA + 1.0
    w = h[-1]; at = steer(M, 0.0).astype(complex)
    assert np.linalg.norm(w) <= KAPPA + 1e-8 and np.all(np.real(w.conj() @ L) >= b - 1e-6)
    assert sinr_db(w, at, R_ipn, ps) > sinr_db(a_p / M, at, R_ipn, ps) + 10


def test_guide_first_refresh_and_direction():
    """The guide first runs at snapshot 26 (k = 25) and finds the source at 7 degrees of mismatch."""
    X, S = block()
    thp = np.deg2rad(7.0)
    hist, log = run_sgrab_p(X, pos_ula(M), steer(M, thp), thp, A.Y0_G * np.sqrt(M / 16))
    assert log[0][0] == 25 and [l[0] for l in log] == list(range(25, K, 25))
    assert abs(log[-1][1]) <= 1.0
    w0, G, th = guide(X, 7.0)
    assert abs(th) <= 1.0


def test_scaling_matrix_spectrum_and_bound_4_13():
    """Psi = R_inc^-1 / lambda_max(R_inc^-1) has eigenvalues in [lambda_min/lambda_max, 1], and the bound (4.13) holds."""
    X, S = block()
    w0, G, th = guide(X, 7.0)
    ev = np.linalg.eigvalsh(G)
    assert np.isclose(ev[-1], 1.0) and ev[0] > 0
    Rinc = np.linalg.inv(G)                       # proportional to R_inc: lambda_min(Rinc) = 1 in this normalisation
    lmin = np.linalg.eigvalsh(Rinc)[0]
    rng = np.random.default_rng(0); mu = 0.1
    for _ in range(20):
        g = rng.standard_normal(M) + 1j * rng.standard_normal(M); w = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        p = -mu * G @ g
        xi = 2 * np.real(w.conj() @ Rinc @ p) + np.real(p.conj() @ Rinc @ p)
        assert abs(xi) <= mu * lmin * np.linalg.norm(g) * (2 * np.linalg.norm(w) + mu * np.linalg.norm(g)) + 1e-9


def test_guided_rmber_near_optimum_at_large_mismatch():
    X, S = block()
    at = steer(M, 0.0).astype(complex)
    wg, wb = guided(X, 7.0)
    assert sinr_db(wb, at, R_ipn, ps) > optimum() - 3
    assert np.linalg.norm(wb) <= KAPPA + 1e-9


def test_reproducible_with_fixed_seed():
    w1 = guided(block(1003)[0], 5.0)[1]
    w2 = guided(block(1003)[0], 5.0)[1]
    assert np.array_equal(w1, w2)
    rng1, rng2 = np.random.default_rng(1001), np.random.default_rng(1001)
    assert np.array_equal(gen_block(M, K, A.MP_SMALL, rng1)[0], gen_block(M, K, A.MP_SMALL, rng2)[0])


def test_comparators_and_ber_model():
    X, S = block()
    thp = np.deg2rad(2.0); a_p = steer(M, thp); at = steer(M, 0.0).astype(complex)
    for w in (spss(X, thp), incm_rec(X, a_p, thp, sector_halfwidth_deg=8.0)):
        assert np.isfinite(sinr_db(w, at, R_ipn, ps))
    w = ta_mmse(X, S)
    Rh = X @ X.conj().T / K; r = X @ np.conj(S) / K
    assert np.allclose((Rh + np.eye(M)) @ w, r)
    g = 10 ** (sinr_db(w, at, R_ipn, ps) / 10)
    assert np.isclose(ber_of(w, at), qfunc(np.sqrt(2 * g)), rtol=1e-9)


def test_blind_decisions_follow_the_trained_form_at_small_mismatch():
    X, S = block()
    a_p = steer(M, np.deg2rad(1.0)); ca = wc_grid(1.0); nu = NU_FAC * nu_pub(M); at = steer(M, 0.0).astype(complex)
    wt = run_rmber(X, a_p, ca, nu, S)[-1]; wbl = run_blind(X, a_p, ca, nu, phase=True)[-1]
    assert sinr_db(wbl, at, R_ipn, ps) > sinr_db(wt, at, R_ipn, ps) - 1.0
