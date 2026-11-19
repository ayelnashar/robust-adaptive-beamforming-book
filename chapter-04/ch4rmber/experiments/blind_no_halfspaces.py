"""Blind constraint-based R-MBER (phase recovery) with and without the worst-case half-spaces, on the mismatch-sweep
realizations of Figure 4.2 for M = 8, 16 and 32 (Section 4.7.10). run_blind_nc is run_blind with the projection
onto the half-spaces and the ball replaced by the projection onto the ball alone.
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
def run_blind_nc(X, a, phase=True, step=0.1, Z=80, kappa=KAPPA, rho_c=RHO_C):
    Mloc, Kloc = X.shape
    aH = np.real(a.conj()@a); w = a/aH
    hist=[]
    for k in range(Kloc):
        k0=max(0,k-Z+1); Xw=X[:,k0:k+1]; Ln=Xw.shape[1]
        y = Xw.T@w.conj()
        if phase:
            phi = 0.5*np.angle(np.sum(y**2)); rot=np.exp(-1j*phi)
            Xw = Xw*rot; y = y*rot
        m = np.real(y); sw = np.sign(m)+0.0
        rho=max(1.06*np.std(m-sw*np.mean(m*sw))*Ln**(-1/5),1e-3)
        rho=max(rho, rho_c*np.mean(np.abs(m)))
        g=np.exp(-(m**2)/(2*rho**2))
        grad=-(1.0/(Ln*rho*np.sqrt(2*np.pi)))*((g*sw)@Xw.T)
        gn=np.linalg.norm(grad)
        if gn>1.0/step: grad=grad*(1.0/step)/gn
        v=w-step*grad; n=np.linalg.norm(v); w=v if n<=kappa else v*(kappa/n)      # ball only
        if not np.all(np.isfinite(w)): w=a/aH
        hist.append(w.copy())
    return hist
out={}
for Mloc,mms in [(8,[0.,2.,4.,6.,8.,10.,12.]),(16,[0.,1.,2.,3.,4.,5.]),(32,[0.,0.5,1.,1.5,2.,2.5,3.,5.,7.,9.,11.])]:
    R=build_R_ipn(Mloc); at=steer(Mloc,0.0).astype(complex); nu=NU_FAC*nu_pub(Mloc); A=np.zeros((4,len(mms))); F=np.zeros((2,len(mms)))
    for j,mm in enumerate(mms):
        thp=np.deg2rad(mm); a_p=steer(Mloc,thp); ca=wc_grid(mm)
        for r in range(mc(20)):
            np.random.seed(1000+r); X,S=gen_with_symbols(Mloc,K,0.0,None)
            wt=run_rmber(X,a_p,ca,nu,S)[-1]; st=sinr_db(wt,at,R,ps)
            ws=[run_blind(X,a_p,ca,nu,phase=True)[-1],run_blind_nc(X,a_p,phase=True)[-1],wt]
            v=[sinr_db(w,at,R,ps) for w in ws[:3]]
            A[:3,j]+=np.array(v)/mc(20); F[:,j]+=np.array([v[0]<st-3, v[1]<st-3])
    out[Mloc]={'mms':mms,'blind':A[0].tolist(),'blind_nc':A[1].tolist(),'trained':A[2].tolist(),'failed':F.tolist()}
    print(f'\nM = {Mloc}, point source, 20 realizations, SINR (dB) | failed runs (more than 3 dB below trained)')
    print(f'{"mismatch (deg)":<28}'+''.join(f'{m:>7.1f}' for m in mms))
    for n,row in [('blind R-MBER',A[0]),('blind, no half-spaces',A[1]),('trained R-MBER',A[2])]: print(f'{n:<28}'+''.join(f'{x:>7.1f}' for x in row))
    for n,row in [('failed, blind',F[0]),('failed, blind no half-sp.',F[1])]: print(f'{n:<28}'+''.join(f'{int(x):>7d}' for x in row),flush=True)
json.dump(out,open('res_blindnc.json','w'))
