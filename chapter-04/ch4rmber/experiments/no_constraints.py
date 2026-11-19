"""Trained constraint-based R-MBER with and without the worst-case half-spaces (Section 4.7.10). The variant without
them is run_rmber with the projection onto the ball alone; the norm bound, bandwidth floor, step and window are unchanged.
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
def run_rmber_nc(X, a, S_train, step=0.1, Z=80, kappa=KAPPA, rho_c=RHO_C):
    Mloc, Kloc = X.shape; aH=np.real(a.conj()@a); w=a/aH
    for k in range(Kloc):
        k0=max(0,k-Z+1); Xw=X[:,k0:k+1]; sw=S_train[k0:k+1].real; Ln=Xw.shape[1]
        m=np.real(Xw.T@w.conj())
        rho=max(1.06*np.std(m-sw*np.mean(m*sw))*Ln**(-1/5),1e-3); rho=max(rho,rho_c*np.mean(np.abs(m)))
        g=np.exp(-(m**2)/(2*rho**2)); grad=-(1.0/(Ln*rho*np.sqrt(2*np.pi)))*((g*sw)@Xw.T)
        gn=np.linalg.norm(grad)
        if gn>1.0/step: grad=grad*(1.0/step)/gn
        v=w-step*grad; n=np.linalg.norm(v); w=v if n<=kappa else v*(kappa/n)      # ball only
        if not np.all(np.isfinite(w)): w=a/aH
    return w
nu=NU_FAC*nu_pub(M); out={}
print('eight cases of Figure 4.3, 20 block-fading realizations: SINR (dB) of trained R-MBER, trained R-MBER without the half-spaces, TA-MMSE')
SCEN=[('2 pt',2.,None),('5 pt',5.,None),('7 pt',7.,None),('2 sm',2.,MP_SMALL),('5 sm',5.,MP_SMALL),('7 sm',7.,MP_SMALL),('5 lg',5.,MP_LARGE),('7 lg',7.,MP_LARGE)]
for lab,mm,mp in SCEN:
    thp=np.deg2rad(mm); a_p=steer(M,thp); ca=wc_grid(mm); s=np.zeros(3)
    for r in range(mc(20)):
        rng=np.random.default_rng(1000+r); X,Sy,eff=gen_block(M,K,mp,rng)
        s+=np.array([sinr_db(w,eff,core.R_ipn,ps) for w in (run_rmber(X,a_p,ca,nu,Sy)[-1],run_rmber_nc(X,a_p,Sy),ta_mmse(X,Sy))])/mc(20)
    out[lab]=s.tolist(); print(f'  {lab}: {s[0]:5.1f} {s[1]:5.1f} {s[2]:5.1f}',flush=True)
print('mismatch sweep on the realizations of Figure 4.2(c), point source, 20 realizations')
at=steer(M,0.0).astype(complex); mms=[0.,0.5,1.,2.,3.,5.,7.,9.,11.]; S9=np.zeros((3,len(mms)))
for j,mm in enumerate(mms):
    thp=np.deg2rad(mm); a_p=steer(M,thp); ca=wc_grid(mm)
    for r in range(mc(20)):
        np.random.seed(1000+r); X,Sy=gen_with_symbols(M,K,0.0,None)
        S9[:,j]+=np.array([sinr_db(w,at,core.R_ipn,ps) for w in (run_rmber(X,a_p,ca,nu,Sy)[-1],run_rmber_nc(X,a_p,Sy),ta_mmse(X,Sy))])/mc(20)
for n,row in zip(['trained R-MBER','no half-spaces','TA-MMSE'],S9): print(f'  {n:<15}',np.round(row,1).tolist())
out['sweep']=S9.tolist()
print('element SNR -5 dB, model-based BER versus mismatch, point source and large spread, 15 realizations (realizations of Figure 4.5)')
gen=gen_block_p
mm10=[0.,2.,5.,7.,9.,11.]
for lab,mp in [('point source',None),('large spread',MP_LARGE)]:
    B=np.zeros((3,len(mm10))); p=sigma2*10**(-0.5)
    for j,mm in enumerate(mm10):
        thp=np.deg2rad(mm); a_p=steer(M,thp); ca=wc_grid(mm)
        for r in range(mc(15)):
            X,b,eff=gen(mp,np.random.default_rng(6000+r),p)
            B[:,j]+=np.array([ber_of(w,eff,p=p) for w in (run_rmber(X,a_p,ca,nu,b)[-1],run_rmber_nc(X,a_p,b),ta_mmse(X,b))])/mc(15)
    print(' ',lab); [print(f'    {n:<15}',[f'{v:.1e}' for v in row]) for n,row in zip(['trained R-MBER','no half-spaces','TA-MMSE'],B)]
    out['low_'+lab]=B.tolist()
json.dump(out,open('res_nocon.json','w'))
