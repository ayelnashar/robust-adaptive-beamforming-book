"""
ch4rmber.algorithms -- signal model and beamformers of Chapter 4.

  * signal model of Section 4.2: half-wavelength ULA, BPSK desired source, interferers, white noise, multipath
  * smoothed-BER cost and its gradient, Section 4.3: Parzen kernel, adaptive bandwidth with floor (4.7a), decision labels
    with blind phase recovery (Section 4.3.2)
  * guided R-MBER, Section 4.4: spectral guide (Capon spectrum, peak test, merging, sector, reconstruction (4.9),
    steering estimate, guide weight (4.10)) and the scaled BER step (4.11)
  * constraint-based R-MBER baselines, Section 4.5: trained and blind forms, worst-case grid (4.15), exact Dykstra
    projection onto the half-spaces and the norm ball
  * comparators of Section 4.7: WC-MVDR/EigDec (exact solution), INCM-REC, SPSS [14], TA-MMSE, optimum MVDR,
    standard and diagonally loaded MVDR and the unconstrained adaptive MBER beamformer
  * metrics: output SINR, model-based BPSK BER

Model constants (M, INR, theta_ints, ps, ...) are module globals. Some experiments change INR, theta_ints or ps for one
stage; every stage runs in its own process (ch4rmber.scenarios), so a change never leaks into another stage.
"""
import atexit
import numpy as np
from scipy.linalg import eigh
from scipy.special import erfc

# ---------------------------------------------------------------- settings of Section 4.7.1
M       = 32          # array elements (16, 64 and others are passed explicitly where used)
sigma2  = 1.0         # noise power per element
SNR     = 5           # desired-source SNR, dB
INR     = 10          # INR of each interferer, dB
ps      = sigma2 * 10 ** (SNR / 10)
eta     = 0.97        # forgetting factor (not used by run_mber, kept in its signature)
beta    = 0.99        # exponential weighting of the batch covariance (WC-MVDR/EigDec, MVDR, DL-MVDR)
K       = 150         # snapshots per block
theta_ints = [np.deg2rad(45.0), np.deg2rad(-60.0)]

KAPPA   = 3.0         # norm bound ||w|| <= kappa                       (Sections 4.4.2 and 4.5)
RHO_C   = 0.5         # bandwidth floor coefficient c of (4.7a)        (Section 4.3.1)
NU_FAC  = 0.5         # applied bounds nu = 0.5 nu_0                   (Section 4.5)



def nu_pub(M):
    """Nominal worst-case bounds nu_0 of Section 4.5, scaled with array size."""
    return np.array([1.9, 1.5, 1.5, 1.1]) * np.sqrt(M / 16.0)


def zeta(M):
    """Radius of WC-MVDR/EigDec, 3.8 sqrt(M/16) (Section 4.7.1)."""
    return 3.8 * np.sqrt(M / 16.0)


def steer(M, th):
    """ULA steering vector at angle `th` (radians), half-wavelength spacing, phase reference at the first element."""
    return np.exp(1j * 2 * np.pi * 0.5 * np.arange(M) * np.sin(th))


def sinr_db(w, a_true, R_ipn, ps):
    """Output SINR in dB of weight w for the true steering vector a_true:
    ps |w^H a_true|^2 / (w^H R_ipn w); a degenerate denominator returns -20 dB.
    """
    num = ps * np.abs(w.conj() @ a_true) ** 2
    den = np.real(w.conj() @ R_ipn @ w)
    if den < 1e-30:
        return -20.0
    return 10 * np.log10(max(np.real(num) / np.real(den), 1e-10))


def build_R_ipn(M):
    """Population interference-plus-noise covariance for the current INR and theta_ints."""
    R = sigma2 * np.eye(M, dtype=complex)
    for th in theta_ints:
        a = steer(M, th)
        R = R + (sigma2 * 10 ** (INR / 10)) * np.outer(a, a.conj())
    return R


R_ipn = build_R_ipn(M)


def wc_grid(thp_deg):
    """Constraint grid (4.15): presumed direction, -2, +2 and -4 degrees (input and output in degrees / radians)."""
    return [np.deg2rad(thp_deg), np.deg2rad(thp_deg - 2),
            np.deg2rad(thp_deg + 2), np.deg2rad(thp_deg - 4)]


# ---------------------------------------------------------------- comparators


def alg_mvdr(R, a):
    """Standard MVDR with a numerical loading of 1e-3 tr(R)/M."""
    M = R.shape[0]
    dl = 1e-3 * np.trace(R).real / M
    Rinv_a = np.linalg.solve(R + dl * np.eye(M), a)
    return Rinv_a / (a.conj() @ Rinv_a)


def alg_dl(R, a, alpha):
    """Diagonally loaded MVDR with loading alpha."""
    M = R.shape[0]
    Rinv_a = np.linalg.solve(R + alpha * np.eye(M), a)
    return Rinv_a / (a.conj() @ Rinv_a)


def wc_mvdr_exact(R, a, zeta):
    """WC-MVDR/EigDec [6]: exact solution of  min w^H R w  s.t.  Re{a^H w} - zeta ||w|| >= 1.

    Stationarity gives w proportional to v(d) = (R + d I)^{-1} a with d ||v(d)|| = zeta, found from the eigendecomposition
    of R (numerical loading 1e-3 tr(R)/M) by Brent's method; the scale is then set so that the constraint holds with
    equality. Requires zeta < ||a||.
    """
    from scipy.optimize import brentq
    M = R.shape[0]
    dl = 1e-3 * np.trace(R).real / M
    lam, U = eigh(R + dl * np.eye(M))
    b = U.conj().T @ a
    f = lambda d: d * np.linalg.norm(b / (lam + d)) - zeta
    hi = max(1.0, float(np.max(lam)))
    while f(hi) <= 0:
        hi *= 2.0
    d = brentq(f, 0.0, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    v = U @ (b / (lam + d))
    return v / (np.real(v.conj() @ a) - zeta * np.linalg.norm(v))


def wc_exact(R, a, z):
    """WC-MVDR/EigDec weight (alias of wc_mvdr_exact)."""
    return wc_mvdr_exact(R, a, z)


def run_mber(X, a, S_train, eta=0.97, step=0.1, ridge=0.01):
    """Adaptive MBER beamformer with a distortionless constraint at the presumed direction (reference only).

    One smoothed-BER gradient step per snapshot over a sliding window of 80 labelled snapshots, Silverman bandwidth, small
    ridge on ||w||^2, then projection onto w^H a = 1. Returns the weight after every snapshot.
    """
    M, K = X.shape
    aH_a = np.real(a.conj() @ a)
    w = a / aH_a
    hist = []
    
    L_buffer = 80  # sliding window size for gradient
    
    for k in range(K):
        # Use the most recent L_buffer samples
        k0 = max(0, k - L_buffer + 1)
        Xw = X[:, k0:k+1].T  # (L, M)
        sw = S_train[k0:k+1].real  # (L,)
        L = Xw.shape[0]
        
        # Margins on current w
        margins = np.real(Xw @ w.conj())  # (L,)
        
        # Silverman bandwidth on margins
        margin_std = np.std(margins - sw * np.mean(margins * sw))
        rho = max(1.06 * margin_std * max(L, 1)**(-1/5), 1e-3)
        
        # Gradient (Wirtinger: dJ/dw* = -(1/(L rho sqrt(2pi))) sum_n s_n exp(-m^2/2rho^2) x_n)
        weights_g = np.exp(-(margins ** 2) / (2 * rho ** 2))
        grad = -(1.0 / (L * rho * np.sqrt(2 * np.pi))) * (
            (weights_g * sw) @ Xw
        )
        if ridge > 0:
            grad = grad + 2.0 * ridge * w
        
        w_new = w - step * grad
        
        # Project onto distortionless: w^H a = 1
        gap = 1.0 - np.sum(np.conj(w_new) * a)
        w_new = w_new + np.conj(gap) * a / aH_a
        w = w_new
        
        if not np.all(np.isfinite(w)):
            w = a / aH_a
        hist.append(w.copy())
    return hist


def incm_rec(X, a_p, thp_rad, sector_halfwidth_deg=4.0, n_grid=200,
             scan_deg=(-90.0, 90.0), dl=1e-3):
    """INCM-REC: interference-plus-noise covariance reconstruction in the manner of [11] with a covariance-based
    steering-vector estimate. The sector |theta - thp| <= sector_halfwidth_deg is excluded from the reconstruction
    (Section 4.7 uses +-8 degrees and 200 scan directions over +-90 degrees).
    """
    M, K = X.shape
    R = (X @ X.conj().T) / K
    R = (R + R.conj().T) / 2
    Rinv = np.linalg.inv(R + dl * np.trace(R).real / M * np.eye(M))

    lo, hi = np.deg2rad(scan_deg[0]), np.deg2rad(scan_deg[1])
    hw = np.deg2rad(sector_halfwidth_deg)
    grid = np.linspace(lo, hi, n_grid)
    dth = grid[1] - grid[0]
    mask = np.abs(grid - thp_rad) > hw          # directions outside the excluded sector

    R_ipn = np.zeros((M, M), dtype=complex)
    for th in grid[mask]:
        a = steer(M, th)
        denom = np.real(a.conj() @ Rinv @ a)
        R_ipn += np.outer(a, a.conj()) / max(denom, 1e-12) * dth
    R_ipn = (R_ipn + R_ipn.conj().T) / 2

    # desired-signal covariance and SV estimate
    R_s = R - R_ipn
    a_hat = R_s @ a_p
    nrm = np.linalg.norm(a_hat)
    if not np.isfinite(nrm) or nrm < 1e-12:
        a_hat = a_p.copy()
    else:
        a_hat = a_hat / nrm * np.sqrt(M)

    Rl = R_ipn + dl * np.trace(R_ipn).real / M * np.eye(M)
    w = np.linalg.solve(Rl, a_hat)
    return w / (a_hat.conj() @ w)


def spss(X,th_p,sector_halfwidth_deg=6.0,taper='sinc',load=None):
    """SPSS beamformer of [14], Steps 1 to 4, with signal sector th_p +- 6 degrees.

    Watch points sin(alpha_k) = 2 x_k / M (integer x_k); sample matrix D over the watch points outside the sector; taper
    T = [sinc((m - n) Delta / pi)], Delta = asin(2/M), with sinc(u) = sin(u)/u as [14] defines it (taper='normalized'
    uses sin(pi u)/(pi u)); R_T = R_x o T, R^ = D R_T D, R~ = R^ o T, w = R~^-1 d / (d^H R~^-1 d). No loading, as in [14];
    load=c adds c tr(R~)/M (sensitivity check only). A failed solve falls back to least squares and is counted.
    """
    M,K=X.shape; Rx=X@X.conj().T/K; Delta=np.arcsin(2.0/M)
    mn=np.subtract.outer(np.arange(M),np.arange(M)).astype(float); u=mn*Delta/np.pi
    T=np.where(mn==0,1.0,np.sin(u)/np.where(u==0,1,u)) if taper=='sinc' else np.sinc(u)
    x=np.arange(int(np.ceil(-M/2)),int(np.ceil(M/2)))            # sin(alpha0) = 0
    s=2*x/M; s=s[(s>=-1)&(s<1)]; al=np.arcsin(s)
    hw=np.deg2rad(sector_halfwidth_deg); out=al[np.abs(al-th_p)>hw]
    Dm=sum(np.outer(steer(M,a),steer(M,a).conj()) for a in out)/M
    RT=Rx*T; Rh=Dm@RT@Dm; Rt=Rh*T; d=steer(M,th_p)
    if load: Rt=Rt+load*np.trace(Rt).real/M*np.eye(M)
    try: v=np.linalg.solve(Rt,d)
    except np.linalg.LinAlgError: FAIL[0]+=1; v=np.linalg.lstsq(Rt,d,rcond=None)[0]
    return v/(d.conj()@v)


FAIL = [0]   # SPSS solves that fell back to least squares


FALLBACKS = [0]


def fallback(name, Mloc):
    FALLBACKS[0] += 1
    print(f'WARNING: {name} failed, matched-filter fallback used (M = {Mloc})', flush=True)


atexit.register(lambda: FALLBACKS[0] and print(f'solver fallbacks in this run: {FALLBACKS[0]}'))


def ta_mmse(X,S,delta=None):
    """TA-MMSE: training-aided Wiener filter (R^ + delta I)^-1 r^ from the labelled snapshots X, S; delta = noise power."""
    Mloc,Kn=X.shape; delta=sigma2 if delta is None else delta
    R=X@X.conj().T/Kn; r=X@np.conj(S)/Kn
    return np.linalg.solve(R+delta*np.eye(Mloc),r)


def cov_ewma(X):
    """Exponentially weighted sample covariance (weight beta = 0.99, initialized at sigma2 I) of the batch beamformers."""
    Mloc=X.shape[0]; Rm=np.eye(Mloc,dtype=complex)*sigma2
    for k in range(X.shape[1]): Rm=beta*Rm+(1-beta)*np.outer(X[:,k],X[:,k].conj())
    return (Rm+Rm.conj().T)/2


NAMES8 = ['Standard MVDR', 'DL-MVDR', 'WC-MVDR/EigDec', 'MBER', 'Trained R-MBER', 'INCM-REC', 'TA-MMSE', 'Optimal MVDR']


def weights8(X,Sy,eff,a_p,thp,ca,p=None,R=None,Mloc=None):
    """Final weight of every comparator in NAMES8 order on one block (trained methods use the labels Sy)."""
    R=R_ipn if R is None else R; Mloc=M if Mloc is None else Mloc
    Rs=cov_ewma(X); ws=[]
    for nm,fn in (('MVDR',lambda: alg_mvdr(Rs,a_p)), ('DL-MVDR',lambda: alg_dl(Rs,a_p,2*sigma2)), ('WC-MVDR/EigDec',lambda: wc_exact(Rs,a_p,zeta(Mloc)))):
        try: ws.append(fn())
        except Exception: ws.append(a_p/Mloc); fallback(nm,Mloc)
    ws.append(run_mber(X,a_p,Sy,eta)[-1]); ws.append(run_rmber(X,a_p,ca,NU_FAC*nu_pub(Mloc),Sy)[-1])
    ws.append(incm_rec(X,a_p,thp,sector_halfwidth_deg=8.0)); ws.append(ta_mmse(X,Sy))
    wo=np.linalg.solve(R,eff); ws.append(wo/(eff.conj()@wo)); return ws


# ---------------------------------------------------------------- constraint-based R-MBER (Section 4.5)


def _proj_halfspace(w, c, b, c2):
    """Projection of w onto {z : Re(z^H c) >= b}; c2 = ||c||^2."""
    r = np.real(w.conj() @ c)
    if r >= b:
        return w
    return w + ((b - r) / c2) * c


def _proj_ball(w, kappa):
    """Projection onto the ball ||w|| <= kappa."""
    n = np.linalg.norm(w)
    return w if n <= kappa else (kappa / n) * w


def proj_exact(w0, L, b, kappa, iters=60, tol=1e-10):
    """Euclidean projection onto the intersection of the half-spaces Re(w^H c_i) >= b_i and the ball ||w|| <= kappa
    by Dykstra's method [42] with its correction vectors; at most `iters` sweeps, stop when the relative change < tol.
    L : (M, Nc) constraint vectors, b : (Nc,) thresholds.
    """
    Nc = L.shape[1]
    c2 = np.array([np.real(L[:, i].conj() @ L[:, i]) for i in range(Nc)])
    x = w0.copy()
    p = [np.zeros_like(w0) for _ in range(Nc + 1)]
    for _ in range(iters):
        x_prev = x.copy()
        for i in range(Nc):
            y = x + p[i]
            x_new = _proj_halfspace(y, L[:, i], b[i], c2[i])
            p[i] = y - x_new
            x = x_new
        y = x + p[Nc]
        x_new = _proj_ball(y, kappa)
        p[Nc] = y - x_new
        x = x_new
        if np.linalg.norm(x - x_prev) < tol * max(1.0, np.linalg.norm(x)):
            break
    return x


def run_rmber(X, a, c_ang, nu, S_train, step=0.1, Z=80, kappa=KAPPA, rho_c=RHO_C):
    """Trained constraint-based R-MBER (Section 4.5).

    One step per snapshot on the smoothed BER (4.5) over a window of Z labelled snapshots: Silverman bandwidth (4.7) with the
    floor rho_c * mean|margin| of (4.7a) and an absolute minimum of 1e-3, gradient (4.6) with its norm limited to 1/step
    (step norm at most 1), then the exact projection onto the half-spaces of the grid c_ang with bounds nu and the ball
    ||w|| <= kappa. Starts from a / ||a||^2. Returns the weight after every snapshot.
    """
    Mloc, Kloc = X.shape
    L = np.column_stack([steer(Mloc, t) for t in c_ang])
    b = nu * kappa + 1.0
    aH = np.real(a.conj() @ a)
    w = a / aH
    hist = []
    for k in range(Kloc):
        k0 = max(0, k - Z + 1)
        Xw = X[:, k0:k + 1]
        sw = S_train[k0:k + 1].real
        Ln = Xw.shape[1]
        m = np.real(Xw.T @ w.conj())                       # soft margins
        rho = max(1.06 * np.std(m - sw * np.mean(m * sw)) * Ln ** (-1 / 5), 1e-3)
        rho = max(rho, rho_c * np.mean(np.abs(m)))          # bandwidth floor
        g = np.exp(-(m ** 2) / (2 * rho ** 2))
        grad = -(1.0 / (Ln * rho * np.sqrt(2 * np.pi))) * ((g * sw) @ Xw.T)
        gn = np.linalg.norm(grad)
        if gn > 1.0 / step:                                 # step norm at most 1
            grad = grad * (1.0 / step) / gn
        w = proj_exact(w - step * grad, L, b, kappa)
        if not np.all(np.isfinite(w)):
            w = a / aH
        hist.append(w.copy())
    return hist


def run_blind(X, a, c_ang, nu, phase=True, w0=None, Lvec=None, step=0.1, Z=80, kappa=KAPPA, rho_c=RHO_C, ret_dec=False):
    """Blind constraint-based R-MBER: run_rmber with the training symbols replaced by decision labels.

    phase=True : decisions after blind BPSK phase recovery, phi = 0.5 arg(sum y^2) over the window (Section 4.3.2)
    phase=False: decisions sign(Re y) without phase recovery
    w0 replaces the start a/||a||^2, Lvec the grid vectors; ret_dec also returns the decision on the newest snapshot.
    """
    Mloc, Kloc = X.shape
    L = np.column_stack([steer(Mloc, t) for t in c_ang]) if Lvec is None else Lvec
    b = nu*kappa + 1.0
    aH = np.real(a.conj()@a); w = a/aH if w0 is None else w0.copy()
    hist=[]; dec=np.zeros(Kloc)
    for k in range(Kloc):
        k0=max(0,k-Z+1); Xw=X[:,k0:k+1]; Ln=Xw.shape[1]
        y = Xw.T@w.conj()                       # w^H x
        if phase:
            phi = 0.5*np.angle(np.sum(y**2)); rot=np.exp(-1j*phi)
            Xw = Xw*rot; y = y*rot
        m = np.real(y); sw = np.sign(m)+0.0
        dec[k]=sw[-1]
        rho=max(1.06*np.std(m-sw*np.mean(m*sw))*Ln**(-1/5),1e-3)
        rho=max(rho, rho_c*np.mean(np.abs(m)))
        g=np.exp(-(m**2)/(2*rho**2))
        grad=-(1.0/(Ln*rho*np.sqrt(2*np.pi)))*((g*sw)@Xw.T)
        gn=np.linalg.norm(grad)
        if gn>1.0/step: grad=grad*(1.0/step)/gn
        w=proj_exact(w-step*grad, L, b, kappa)
        if not np.all(np.isfinite(w)): w=a/aH
        hist.append(w.copy())
    return (hist,dec) if ret_dec else hist


def full(X,mm,S=None):
    """Final weight of the constraint-based R-MBER on the full array: trained if S is given, blind (phase recovery) if S is None."""
    Ml=X.shape[0]; a=steer(Ml,np.deg2rad(mm)); nu=NU_FAC*nu_pub(Ml)
    return run_rmber(X,a,wc_grid(mm),nu,S)[-1] if S is not None else run_blind(X,a,wc_grid(mm),nu,phase=True)[-1]


# ---------------------------------------------------------------- data generators (Section 4.7.1)


def gen_with_symbols(M, K, th_s, mp=None):
    """Point-source block: K snapshots of a BPSK source at th_s (power ps), the interferers theta_ints (INR each),
    white noise; optional multipath rays mp = [(offset, amplitude)] with phases redrawn at every snapshot.
    Draws from NumPy's global generator (seeded by the caller). Returns snapshots X and symbols S.
    """
    piv = sigma2 * 10**(INR / 10)
    a_s = steer(M, th_s)
    X = np.zeros((M, K), dtype=complex)
    S = np.zeros(K, dtype=complex)
    for k in range(K):
        # BPSK symbol
        bk = 1.0 if np.random.rand() > 0.5 else -1.0
        S[k] = bk
        s = np.sqrt(ps) * bk  # real-valued, no /sqrt(2)
        sig = a_s * s
        if mp:
            for dth, amp in mp:
                g = amp * np.exp(1j * np.random.uniform(-np.pi, np.pi))
                sig += steer(M, th_s + dth) * s * g
        for ti in theta_ints:
            u = np.sqrt(piv / 2) * (np.random.randn() + 1j * np.random.randn())
            sig += steer(M, ti) * u
        sig += np.sqrt(sigma2 / 2) * (np.random.randn(M) + 1j * np.random.randn(M))
        X[:, k] = sig
    return X, S


MP_SMALL = [(np.deg2rad(4), 0.4), (np.deg2rad(-3), 0.4)]    # small angular spread
MP_LARGE = [(np.deg2rad(-25), 0.5), (np.deg2rad(35), 0.3)]   # large angular spread


def gen_block(Mloc, Kloc, mp, rng):
    """Block-fading multipath (Section 4.7.1): ray phases drawn once per block from rng and held fixed.
    Returns snapshots, BPSK symbols and the realized effective steering vector of the desired signal.
    """
    a_s = steer(Mloc, 0.0)
    b = rng.choice([-1.0, 1.0], size=Kloc)
    s = np.sqrt(ps) * b
    X = np.outer(a_s, s)
    eff = a_s.astype(complex).copy()
    if mp:
        for dth, amp in mp:
            g = amp * np.exp(1j * rng.uniform(-np.pi, np.pi))
            X = X + np.outer(steer(Mloc, dth), s) * g
            eff = eff + g * steer(Mloc, dth)
    piv = sigma2 * 10 ** (INR / 10)
    for th in theta_ints:
        u = np.sqrt(piv / 2) * (rng.standard_normal(Kloc) + 1j * rng.standard_normal(Kloc))
        X = X + np.outer(steer(Mloc, th), u)
    X = X + np.sqrt(sigma2 / 2) * (rng.standard_normal((Mloc, Kloc))
                                   + 1j * rng.standard_normal((Mloc, Kloc)))
    return X, b, eff


def gen_block_p(mp, rng, p):
    """gen_block for M elements and K snapshots with the desired-source power p as a parameter (BER experiments)."""
    a_s = steer(M, 0.0); b = rng.choice([-1.0, 1.0], size=K); s = np.sqrt(p) * b; X = np.outer(a_s, s); eff = a_s.astype(complex).copy()
    if mp:
        for dth, amp in mp:
            g = amp * np.exp(1j * rng.uniform(-np.pi, np.pi)); X = X + np.outer(steer(M, dth), s) * g; eff = eff + g * steer(M, dth)
    piv = sigma2 * 10 ** (INR / 10)
    for th in theta_ints: X = X + np.outer(steer(M, th), np.sqrt(piv / 2) * (rng.standard_normal(K) + 1j * rng.standard_normal(K)))
    return X + np.sqrt(sigma2 / 2) * (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))), b, eff


def qfunc(x):
    """Gaussian tail function Q(x)."""
    return 0.5 * erfc(x / np.sqrt(2.0))


def ber_of(w, a_t, p=None, R=None):
    """Model-based BPSK BER Q(sqrt(2 SINR)) with ideal carrier-phase recovery; exact for Gaussian interference plus noise (Section 4.7.5)."""
    p = ps if p is None else p
    R = R_ipn if R is None else R
    den = np.sqrt(0.5 * np.real(w.conj() @ R @ w))
    return float(qfunc(np.sqrt(p) * np.abs(w.conj() @ a_t) / max(den, 1e-30)))


# ---------------------------------------------------------------- spectral guide (Section 4.4.1)
N_FOV = 361                         # spectrum grid over -90 to 90 degrees
eta_g = 0.97                        # exponential weighting alpha of the guide covariance
TAU_CFAR = 2.0                      # peak test: mean over one beamwidth above twice the local median
Y0_G = np.array([1.9, 1.5, 1.5, 1.1])
_CACHE = {}


def pos_ula(M):
    """Element positions of the half-wavelength ULA, in wavelengths."""
    return 0.5 * np.arange(M)


def steer_p(pos, th):
    """Steering vector for element positions pos (wavelengths)."""
    return np.exp(1j * 2 * np.pi * pos * np.sin(th))


def fov(pos):
    """Spectrum grid and its steering matrix for the geometry pos (cached)."""
    key = pos.tobytes()
    if key not in _CACHE:
        g = np.linspace(-np.pi / 2, np.pi / 2, N_FOV)
        _CACHE[key] = (g, np.column_stack([steer_p(pos, t) for t in g]))
    return _CACHE[key]


def _capon(pos, Rinv):
    """Capon spectrum (4.8) on the grid of fov(pos) for the inverse loaded covariance Rinv."""
    _, A = fov(pos)
    return 1.0 / np.real(np.einsum('ij,ik,kj->j', A.conj(), Rinv, A))


def cfar_peaks(g, P, step_deg, bw_deg, tau=2.0, n_max=10, sep_deg=None,
               guard_mult=1.0, ref_mult=3.0):
    """Peak test of the guide (Section 4.4.1): local maxima of the Capon spectrum whose mean power over one beamwidth
    exceeds tau times the median power in the two reference bands, each ref_mult beamwidths wide and separated from the test
    window by a guard of guard_mult beamwidths. At most n_max peaks, strongest score first, at least sep_deg apart.
    """
    N = len(P)
    is_max = np.zeros(N, bool)
    is_max[1:-1] = (P[1:-1] >= P[:-2]) & (P[1:-1] >= P[2:])
    is_max[0] = P[0] >= P[1]
    is_max[-1] = P[-1] >= P[-2]
    nt = max(1, int(round(bw_deg / step_deg)) // 2)
    ng = max(1, int(round(guard_mult * bw_deg / step_deg)))
    nr = max(3, int(round(ref_mult * bw_deg / step_deg)))
    c = np.concatenate([[0.0], np.cumsum(P)])
    s = np.full(N, -np.inf)
    for j in np.where(is_max)[0]:
        lo, hi = max(0, j - nt), min(N, j + nt + 1)
        test = (c[hi] - c[lo]) / max(hi - lo, 1)
        a = P[max(0, j - nt - ng - nr):max(0, j - nt - ng)]
        b = P[min(N, j + nt + ng + 1):min(N, j + nt + ng + nr + 1)]
        ref = np.concatenate([a, b])
        if ref.size >= 3:
            f = np.median(ref)
            if f > 0:
                s[j] = test / f
    sep = 0.5 * bw_deg if sep_deg is None else sep_deg
    guard = max(1, int(round(sep / step_deg)))
    sw = s.copy()
    out = []
    while len(out) < n_max:
        j = int(np.argmax(sw))
        if not np.isfinite(sw[j]) or sw[j] < tau:
            break
        out.append((g[j], float(P[j]), float(s[j])))
        sw[max(0, j - guard):j + guard + 1] = -np.inf
    return out


def beamwidth_deg(pos):
    """Half-power beamwidth from the aperture, 0.886 / D rad with D = max(pos) - min(pos): 1.772/(M - 1) rad for the ULA."""
    D = float(np.max(pos) - np.min(pos))
    return np.rad2deg(0.886 / max(D, 0.5))


def _cluster(peaks, span_deg):
    """Merge peaks closer than span_deg; each cluster is represented by its strongest peak and the sum of its peak powers."""
    if not peaks:
        return []
    pk = sorted(peaks, key=lambda z: z[0])
    cl, cur = [], [pk[0]]
    for t, p in pk[1:]:
        if abs(t - cur[-1][0]) <= np.deg2rad(span_deg):
            cur.append((t, p))
        else:
            cl.append(cur); cur = [(t, p)]
    cl.append(cur)
    return [(max(c, key=lambda z: z[1])[0], sum(p for _, p in c)) for c in cl]


def _select(cl, th_p, Delta, Dext, pos=None, P=None, sep_deg=None):
    """Source selection (Section 4.4.1). A cluster within Delta of the presumed direction th_p is taken (the weakest if
    several qualify); otherwise the weakest within the widened window Dext; otherwise the spectrum maximum in that window.
    Both windows allow one grid step at their edges. Also returns the other clusters that lie more than sep_deg from the
    selected source and carry at least 10% of its cluster power.
    """
    tol = np.deg2rad(180.0 / (N_FOV - 1))          # one scan-grid step
    dec, ext = np.deg2rad(Delta) + tol, np.deg2rad(Dext) + tol
    dsep = np.deg2rad(Delta if sep_deg is None else sep_deg)
    ins = [c for c in cl if abs(c[0] - th_p) <= dec]
    pw = None
    if len(ins) == 1:
        th, pw, mode = ins[0][0], ins[0][1], 'window'
    elif len(ins) > 1:
        c = min(ins, key=lambda z: z[1]); th, pw, mode = c[0], c[1], 'window-weakest'
    else:
        win = [c for c in cl if abs(c[0] - th_p) <= ext]
        if win:
            c = min(win, key=lambda z: z[1]); th, pw, mode = c[0], c[1], 'widened-weakest'
        elif pos is not None and P is not None:
            # no detected cluster in the widened window: take the spectrum maximum there
            g, _ = fov(pos)
            m = np.abs(g - th_p) <= ext
            th, pw, mode = g[m][np.argmax(P[m])], P[m].max(), 'widened-maximum'
        else:
            return th_p, [(c[0], c[1]) for c in cl], 'none'
    # clusters that the sector keeps clear of: more than sep_deg away and at least 10% of the source power
    others = [(c[0], c[1]) for c in cl
              if abs(c[0] - th) > dsep and c[1] >= 0.1 * pw]
    return th, others, mode


def _clip(th_c, half, others, half_min=2.0, margin=4.0):
    """Sector half-width narrowed toward a clearance `margin` from the other clusters, never below half_min (Section 4.4.1)."""
    h = half
    for t, _pw in others:
        d = abs(np.rad2deg(t - th_c))
        if d < h + margin:
            h = min(h, max(half_min, d - margin))
    return max(h, half_min)


def _inc(pos, Rs, th_c, half):
    """Reconstructed interference-plus-noise covariance (4.9) from the grid directions outside the sector, with loading."""
    M = Rs.shape[0]
    dl = 1e-3 * np.trace(Rs).real / M
    Rinv = np.linalg.inv(Rs + dl * np.eye(M))
    g, A = fov(pos)
    out = np.abs(g - th_c) > np.deg2rad(half)
    Ao = A[:, out]
    p = 1.0 / np.real(np.einsum('ij,ik,kj->j', Ao.conj(), Rinv, Ao))
    return (Ao * p) @ Ao.conj().T + dl * np.eye(M), Rinv


def _ahat(pos, Rinv, th_c, half, a_ref):
    """Steering-vector estimate: principal eigenvector of the spectrum-weighted sum over the sector, phase-aligned to a_ref, norm sqrt(M)."""
    M = len(pos)
    g, A = fov(pos)
    ins = np.abs(g - th_c) <= np.deg2rad(half)
    if ins.sum() < 2:
        return a_ref.copy()
    Ai = A[:, ins]
    p = 1.0 / np.real(np.einsum('ij,ik,kj->j', Ai.conj(), Rinv, Ai))
    C = (Ai * p) @ Ai.conj().T
    _, ev = eigh((C + C.conj().T) / 2)
    ah = ev[:, -1]
    ah = ah * np.exp(-1j * np.angle(ah.conj() @ a_ref))
    return ah / np.linalg.norm(ah) * np.sqrt(M)


def _mbwc(w, R, L, y):
    """One step of the multiple worst-case constraint recursion of Chapter 3 (scale-invariant gradient step, then a Gram
    correction on the violated constraints). The guide outputs this recursion, started at the presumed direction, until
    its first spectral refresh.
    """
    Rw = R @ w
    rr = np.real(Rw.conj() @ Rw)
    rRr = np.real(Rw.conj() @ R @ Rw)
    mu = 0.1 * rr / max(rRr, 1e-20)
    wt = w - mu * Rw
    wtn = max(np.linalg.norm(wt), 1e-15)
    act = [i for i in range(L.shape[1])
           if np.real(wt.conj() @ L[:, i]) < y[i] * wtn + 1.0]
    if not act:
        return wt
    La = L[:, act]
    d = np.array([(y[i] * wtn + 1.0) - np.real(wt.conj() @ L[:, i]) for i in act])
    G = np.real(La.conj().T @ La)
    try:
        tau = np.maximum(np.linalg.solve(G + 1e-6 * np.eye(len(act)), d), 0)
    except np.linalg.LinAlgError:
        tau = np.zeros(len(act))
    return wt + La @ tau


def grid_p(th_p_deg):
    """Constraint grid (4.15) in radians for a presumed direction in degrees."""
    return [np.deg2rad(th_p_deg), np.deg2rad(th_p_deg - 2),
            np.deg2rad(th_p_deg + 2), np.deg2rad(th_p_deg - 4)]


def run_sgrab_p(X, pos, a_p, th_p, y0, Delta=6.0, Dext=14.0, L_rec=25):
    """Spectral guide of guided R-MBER (Section 4.4.1, Table 4.3 Steps 1 to 4).

    Exponentially weighted covariance R(k) = alpha R(k-1) + x x^H with R(0) = I. At k = L_rec, 2 L_rec, ... (0-based,
    the first refresh at snapshot 26): loading 1e-3 tr(R)/M, Capon spectrum (4.8) on N_FOV directions, peak test, merging
    of peaks within one half-power beamwidth, source selection within Delta (widened to Dext), sector centred midway
    between the estimate and th_p with half-width Delta + |estimate - th_p| narrowed toward a 4 degree clearance from
    the other clusters (at least 2 degrees), reconstruction (4.9), steering estimate and guide weight (4.10). Before the
    first refresh the output is the multiple worst-case constraint recursion _mbwc started at the presumed direction.
    Returns the weight after every snapshot and a log (k, estimate in degrees, selection mode, sector half-width).
    """
    M, K = X.shape
    R = np.eye(M, dtype=complex)
    w_m = a_p / np.linalg.norm(a_p)
    w_r = a_p / np.linalg.norm(a_p)
    y = y0.copy()
    Lc = np.column_stack([steer_p(pos, t) for t in grid_p(np.rad2deg(th_p))])
    hist, log, use_rec = [], [], False
    g, _ = fov(pos)
    step = 180.0 / (N_FOV - 1)
    for k in range(K):
        R = eta_g * R + np.outer(X[:, k], X[:, k].conj())
        if k >= L_rec and k % L_rec == 0:
            Rs = (R + R.conj().T) / 2
            dl = 1e-3 * np.trace(Rs).real / M
            Rinv0 = np.linalg.inv(Rs + dl * np.eye(M))
            P = _capon(pos, Rinv0)                                                            # Capon spectrum (4.8)
            sp = 1.0 * beamwidth_deg(pos)                                                     # merging width
            det = cfar_peaks(g, P, step, beamwidth_deg(pos), tau=TAU_CFAR)
            cl = _cluster([(a, p) for a, p, _ in det], sp)
            th_hat, others, mode = _select(cl, th_p, Delta, Dext, pos, P, sp)
            th_c = 0.5 * (th_hat + th_p)
            half = _clip(th_c, Delta + np.rad2deg(abs(th_hat - th_p)), others)
            R_inc, Rinv = _inc(pos, Rs, th_c, half)
            ah = _ahat(pos, Rinv, th_hat, min(Delta, half), steer_p(pos, th_hat))
            v = np.linalg.solve(R_inc, ah)
            w_r = v / (ah.conj() @ v)
            use_rec = (mode != 'none')
            log.append((k, np.rad2deg(th_hat), mode, half))
        if not use_rec:
            w_m = _mbwc(w_m, (R + R.conj().T) / 2, Lc, y)
        w = w_r if use_rec else w_m
        if not np.all(np.isfinite(w)):
            w = a_p / np.linalg.norm(a_p)
        hist.append(w.copy())
    return hist, log


# ---------------------------------------------------------------- guided R-MBER (Sections 4.4.2 and 4.4.3)
Q = lambda x: 0.5 * erfc(x / np.sqrt(2))


def ber_stage(X,w0,G=None,step=0.1,Z=80,kappa=KAPPA,rho_c=RHO_C):
    """BER step of guided R-MBER (4.11): one pass over the block, one step per snapshot, from w0. Decision labels after
    blind phase recovery over a window of Z snapshots, bandwidth with floor, gradient multiplied by G (the scaling Psi)
    with its norm limited to 1/step, then projection onto the ball ||w|| <= kappa only.
    """
    w=w0.copy()
    for k in range(X.shape[1]):
        k0=max(0,k-Z+1); Xw=X[:,k0:k+1]; Ln=Xw.shape[1]; y=Xw.T@w.conj()
        rot=np.exp(-0.5j*np.angle(np.sum(y**2))); Xw=Xw*rot; y=y*rot
        m=np.real(y); sw=np.sign(m)+0.0
        rho=max(1.06*np.std(m-sw*np.mean(m*sw))*Ln**(-1/5),1e-3); rho=max(rho,rho_c*np.mean(np.abs(m)))
        g=np.exp(-(m**2)/(2*rho**2)); grad=-(1.0/(Ln*rho*np.sqrt(2*np.pi)))*((g*sw)@Xw.T)
        if G is not None: grad=G@grad
        gn=np.linalg.norm(grad)
        if gn>1.0/step: grad=grad*(1.0/step)/gn
        v=w-step*grad; n=np.linalg.norm(v); w=v if n<=kappa else v*(kappa/n)
    return w


def guide(X,mm):
    """Run the spectral guide on block X with presumed direction mm (degrees).

    Returns the guide weight scaled to unit response at the estimated direction, the scaling Psi = R_inc^-1 / lambda_max
    (R_inc rebuilt from the covariance at the end of the block with the last sector), and the estimate in degrees.
    """
    Ml=X.shape[0]; pos=pos_ula(Ml); thp=np.deg2rad(mm); h,log=run_sgrab_p(X,pos,steer(Ml,thp),thp,Y0_G*np.sqrt(Ml/16)); wg=h[-1]
    R=np.eye(Ml,dtype=complex)
    for k in range(X.shape[1]): R=eta_g*R+np.outer(X[:,k],X[:,k].conj())
    Rs=(R+R.conj().T)/2
    if log and log[-1][2]!='none':
        th_hat=np.deg2rad(log[-1][1]); R_inc,_=_inc(pos,Rs,0.5*(th_hat+thp),log[-1][3])
    else: th_hat=thp; R_inc=Rs
    Gi=np.linalg.inv(R_inc); Gi=(Gi+Gi.conj().T)/2; Gi=Gi/np.linalg.eigvalsh(Gi)[-1]
    ah=steer(Ml,th_hat); w0=wg/(wg.conj()@ah).conjugate() if abs(wg.conj()@ah)>1e-9 else ah/Ml
    return w0,Gi,np.rad2deg(th_hat)


def guided(X,mm):
    """Guided R-MBER on block X: returns (guide weight, guided R-MBER weight)."""
    w0,G,_=guide(X,mm); return w0,ber_stage(X,w0,G)


# ---------------------------------------------------------------- bandwidth floor (Section 4.3.1)


def trace_rmber(X, Sy, a_p, ca, nu, floor, step=0.1, Z=80):
    """Trained R-MBER at M elements recording the gradient norm, the output SINR (true direction 0 degrees) and the
    bandwidth after every snapshot; floor=False uses the Silverman bandwidth (4.7) alone.
    """
    L = np.column_stack([steer(M, t) for t in ca]); b = nu * KAPPA + 1.0
    aH = np.real(a_p.conj() @ a_p); w = a_p / aH
    gs, ss, rs = [], [], []
    a_t = steer(M, 0.0).astype(complex)
    for k in range(K):
        k0 = max(0, k - Z + 1); Xw = X[:, k0:k + 1]; sw = Sy[k0:k + 1].real
        Ln = Xw.shape[1]; m = np.real(Xw.T @ w.conj())
        rho = max(1.06 * np.std(m - sw * np.mean(m * sw)) * Ln ** (-1 / 5), 1e-3)
        if floor: rho = max(rho, RHO_C * np.mean(np.abs(m)))
        g = np.exp(-(m ** 2) / (2 * rho ** 2))
        grad = -(1.0 / (Ln * rho * np.sqrt(2 * np.pi))) * ((g * sw) @ Xw.T)
        gn = np.linalg.norm(grad)
        if gn > 1.0 / step: grad = grad * (1.0 / step) / gn
        w = proj_exact(w - step * grad, L, b, KAPPA)
        if not np.all(np.isfinite(w)): w = a_p / aH
        gs.append(gn); ss.append(sinr_db(w, a_t, R_ipn, ps)); rs.append(rho)
    return np.array(gs), np.array(ss), np.array(rs)


def feasible_reference(mm=5.0, iters=3000):
    """Best output SINR attainable inside the R-MBER feasible set with the true covariance, by projected ascent."""
    ca = wc_grid(mm); L = np.column_stack([steer(M, t) for t in ca]); nu = NU_FAC * nu_pub(M)
    b = nu * KAPPA + 1.0; at = steer(M, 0.0).astype(complex)
    w = np.linalg.solve(R_ipn, at); w = proj_exact(w / np.linalg.norm(w) * KAPPA, L, b, KAPPA, iters=2000, tol=0)
    best = sinr_db(w, at, R_ipn, ps)
    for _ in range(iters):
        num = np.real(w.conj() @ at); den = np.real(w.conj() @ R_ipn @ w)
        gdir = (at * np.conj(num) / max(abs(num), 1e-12)) / max(den, 1e-12) - (R_ipn @ w) * (abs(num) ** 2) / max(den, 1e-12) ** 2
        cand = proj_exact(w + 0.05 * gdir / max(np.linalg.norm(gdir), 1e-12) * KAPPA, L, b, KAPPA, iters=150, tol=0)
        s = sinr_db(cand, at, R_ipn, ps)
        if s > best: best, w = s, cand
    return best
