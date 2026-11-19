"""Numbers quoted in the text of Chapter 4 that no figure shows.

Stages: floor_c (floor coefficient c, Section 4.3.1), dykstra (Dykstra sweeps, Section 4.5), distances (1-degree
steering-vector distances and radius limits, Section 4.5), incm_sets (INCM-REC at 7 degrees in the two realization
sets, Section 4.7.1), radius (applied radii +-30%, Section 4.5), ta_loading (TA-MMSE loading, Section 4.7),
wc_solver (exact WC-MVDR/EigDec solver, Section 4.7.1), spss_sinc (SPSS taper, Section 4.7).
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
stage=sys.argv[1]
if stage=='floor_c':
    # floor coefficient c of (4.7a): mean SINR of trained R-MBER over 2, 5, 7 deg, 10 trials (Section 4.3.1)
    at = steer(M, 0.0).astype(complex)
    res = {}
    for c in [0.3, 0.5, 0.7]:
        v = []
        for mm in [2., 5., 7.]:
            thp = np.deg2rad(mm); a_p = steer(M, thp); ca = wc_grid(mm)
            for r in range(mc(10)):
                np.random.seed(1000 + r); X, S = gen_with_symbols(M, K, 0.0, None)
                v.append(sinr_db(run_rmber(X, a_p, ca, NU_FAC * nu_pub(M), S, rho_c=c)[-1], at, R_ipn, ps))
        res[c] = np.mean(v)
    OUT={str(k): float(v) for k, v in res.items()}
    print('floor coefficient', {k: round(v, 2) for k, v in res.items()}, '| loss at 0.3:', round(res[0.5] - res[0.3], 2), 'at 0.7:', round(res[0.5] - res[0.7], 2))
elif stage=='dykstra':
    # Dykstra sweeps inside trained R-MBER runs (5 runs, 5 deg, M = 32) (Section 4.5)
    sweeps, viol = [], []
    _proj = core.proj_exact
    def proj_logged(w0, L, b, kappa, iters=60, tol=1e-10):
        Nc = L.shape[1]; c2 = np.array([np.real(L[:, i].conj() @ L[:, i]) for i in range(Nc)])
        x = w0.copy(); p = [np.zeros_like(w0) for _ in range(Nc + 1)]
        for it in range(iters):
            xp = x.copy()
            for i in range(Nc):
                y = x + p[i]; rr = np.real(y.conj() @ L[:, i])
                xn = y if rr >= b[i] else y + ((b[i] - rr) / c2[i]) * L[:, i]; p[i] = y - xn; x = xn
            y = x + p[Nc]; n = np.linalg.norm(y); xn = y if n <= kappa else kappa / n * y; p[Nc] = y - xn; x = xn
            if np.linalg.norm(x - xp) < tol * max(1.0, np.linalg.norm(x)): break
        sweeps.append(it + 1)
        viol.append(max(0.0, max(b[i] - np.real(x.conj() @ L[:, i]) for i in range(Nc)), np.linalg.norm(x) - kappa))
        return x
    core.proj_exact = proj_logged
    a_p = steer(M, np.deg2rad(5.)); ca = wc_grid(5.)
    for r in range(mc(5)):
        np.random.seed(1000 + r); X, S = gen_with_symbols(M, K, 0.0, None); core.run_rmber(X, a_p, ca, NU_FAC * nu_pub(M), S)
    core.proj_exact = _proj
    sw = np.array(sweeps)
    OUT={'projections':len(sw),'median':float(np.median(sw)),'max':int(sw.max()),'cap_share':float(np.mean(sw >= 60)),'violation':float(max(viol))}
    print(f'Dykstra: {len(sw)} projections, median {np.median(sw):.0f} sweeps, max {sw.max()}, cap reached {np.mean(sw >= 60)*100:.1f}%, largest violation {max(viol):.1e}')
elif stage=='distances':
    # exact 1-deg broadside steering-vector distances, large-M approximation and the radius limits sqrt(M) - 1/kappa (Section 4.5)
    OUT={}
    for Mloc in [16, 32, 64]:
        d = np.linalg.norm(steer(Mloc, np.deg2rad(1.)) - steer(Mloc, 0.)); app = np.pi * np.deg2rad(1.) * Mloc ** 1.5 / np.sqrt(3)
        OUT[str(Mloc)]={'distance':float(d),'limit':float(np.sqrt(Mloc)-1/KAPPA)}
        print(f'M = {Mloc}: exact 1-deg distance {d:.2f}, large-M approximation {app:.2f} ({(app/d-1)*100:+.0f}%), radius limit {np.sqrt(Mloc)-1/KAPPA:.2f}')
elif stage=='incm_sets':
    # INCM-REC at 7 deg in the two independent realization sets (Figure 4.3 cases, Figure 4.2(c) sweep), 20 trials
    at = steer(M, 0.0).astype(complex)
    thp = np.deg2rad(7.); a_p = steer(M, thp); A = []; B_ = []
    for r in range(mc(20)):
        rng = np.random.default_rng(1000 + r); X, S, eff = gen_block(M, K, None, rng)
        A.append(sinr_db(incm_rec(X, a_p, thp, sector_halfwidth_deg=8.0), eff, R_ipn, ps))
        np.random.seed(1000 + r); X, S = gen_with_symbols(M, K, 0.0, None)
        B_.append(sinr_db(incm_rec(X, a_p, thp, sector_halfwidth_deg=8.0), at, R_ipn, ps))
    se = np.sqrt((np.var(A) + np.var(B_)) / 20)
    OUT={'cases':float(np.mean(A)),'sweep':float(np.mean(B_)),'se_units':float(abs(np.mean(A) - np.mean(B_)) / se)}
    print(f'INCM-REC at 7 deg: Figure 4.3 set {np.mean(A):.2f} dB (sd {np.std(A):.2f}); Figure 4.2(c) set {np.mean(B_):.2f} dB (sd {np.std(B_):.2f}); '
          f'difference {abs(np.mean(A) - np.mean(B_)) / se:.1f} standard errors')
elif stage=='radius':
    # applied radii +-30% (0.35, 0.50, 0.65 x nu0), five M = 32 cases, 12 trials (Section 4.5)
    worst = 0.0
    for lab, mm, mp in [('2 pt', 2., None), ('5 pt', 5., None), ('7 pt', 7., None), ('5 sm', 5., MP_SMALL), ('7 lg', 7., MP_LARGE)]:
        thp = np.deg2rad(mm); a_p = steer(M, thp); ca = wc_grid(mm); row = []
        for f in [0.35, 0.50, 0.65]:
            acc = 0.0
            for r in range(mc(12)):
                rng = np.random.default_rng(1000 + r); X, Sy, eff = gen_block(M, K, mp, rng)
                acc += sinr_db(run_rmber(X, a_p, ca, f * nu_pub(M), Sy)[-1], eff, R_ipn, ps)
            row.append(acc / 12)
        worst = max(worst, abs(row[0] - row[1]), abs(row[2] - row[1]))
        print(f'{lab:<6}' + ''.join(f'{v:8.2f}' for v in row))
    OUT={'largest_change_db':float(worst)}
    print('largest change from nominal:', round(worst, 2), 'dB')
elif stage=='ta_loading':
    # TA-MMSE loading between 0.1 and 10 times the noise power, 5 deg point-source cases of Figure 4.3 (Section 4.7)
    OUT={}
    for f in [0.1,0.3,1.0,3.0,10.0]:
        v=0.
        for r in range(mc(20)):
            rng=np.random.default_rng(1000+r); X,Sy,eff=gen_block(M,K,None,rng)
            v+=sinr_db(ta_mmse(X,Sy,delta=f*sigma2),eff,core.R_ipn,ps)/mc(20)
        OUT[str(f)]=float(v)
        print(f'loading {f:>4} x noise power: TA-MMSE SINR {v:.2f} dB')
elif stage=='wc_solver':
    # exact WC-MVDR/EigDec solver against an independent SLSQP solve, and the coverage of its ball (Section 4.7.1)
    from scipy.optimize import minimize
    OUT={}
    at=steer(M,0.0).astype(complex); z=zeta(M)
    print('WC-MVDR/EigDec, radius',round(z,3),', steering-vector norm',round(np.sqrt(M),3))
    for mm in [0.,1.,2.,5.,7.]:
        a_p=steer(M,np.deg2rad(mm)); dmin=np.sqrt(max(2*M-2*abs(a_p.conj()@at),0))
        np.random.seed(1000); X,S=gen_with_symbols(M,K,0.0,None); Rs=cov_ewma(X); w=wc_exact(Rs,a_p,z)
        f=lambda u: float(np.real((u[:M]+1j*u[M:]).conj()@Rs@(u[:M]+1j*u[M:])))
        con={'type':'ineq','fun':lambda u: np.real((u[:M]+1j*u[M:]).conj()@a_p)-z*np.linalg.norm(u)-1}
        r=minimize(f,np.concatenate([(a_p/M*40).real,(a_p/M*40).imag]),constraints=[con],method='SLSQP',options=dict(maxiter=2000,ftol=1e-14)); w2=r.x[:M]+1j*r.x[M:]
        print(f'   mismatch {mm:>3} deg: distance to true steering vector {dmin:5.2f} ({"inside" if dmin<=z else "outside"} the ball) | exact solver {sinr_db(w,at,core.R_ipn,ps):6.1f} dB | SLSQP {sinr_db(w2,at,core.R_ipn,ps):6.1f} dB | fixed beam {sinr_db(a_p/M,at,core.R_ipn,ps):6.1f} dB')
elif stage=='spss_sinc':
    # SPSS taper with sinc(u) = sin(u)/u (as [14] defines it) against the normalized sinc, M = 32, 0 and 2 deg (Section 4.7)
    OUT={}
    Ml=32; a=steer(Ml,0.0).astype(complex); R=core.build_R_ipn(Ml)
    for mm in (0.,2.):
        v0=[];v1=[]
        for r in range(mc(20)):
            np.random.seed(1000+r); X,S=core.gen_with_symbols(Ml,K,0.0,None)
            v0.append(sinr_db(spss(X,np.deg2rad(mm)),a,R,ps)); v1.append(sinr_db(spss(X,np.deg2rad(mm),taper='normalized'),a,R,ps))
        OUT[f'{mm:g}']=[float(np.mean(v0)),float(np.mean(v1))]
        print(f'SPSS, M = 32, {mm:g} deg: sinc(u) = sin(u)/u {np.mean(v0):.2f} dB, normalized sinc {np.mean(v1):.2f} dB, difference {np.mean(v1)-np.mean(v0):+.2f} dB')
json.dump(OUT,open(f'res_text_{stage}.json','w'))
