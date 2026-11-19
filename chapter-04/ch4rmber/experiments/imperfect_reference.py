"""Imperfect training reference (Section 4.7.8, Figure 4.10). Frame: KP = 30 known pilots, then 120 data symbols
(K = 150). Point source at 0 degrees, presumed direction 5 degrees for the blind methods, M = 32, SNR 5 dB,
interferers at 45 and -60 degrees (INR 10 dB), 20 realizations (seeds 31000 + r). SINR is the mean over the data part
with the data-part interference-plus-noise covariance (linear mean within a trial, then dB).
  TA-MMSE (30 pilots)        (R_p + sigma2 I)^-1 r_p from the pilots, then frozen
  TA-MMSE (pilots + DD)      one re-solve over the whole block with the pilots and its own decisions on the data
  TA-MMSE (all 150 known)    every symbol known, as in Sections 4.7.2 to 4.7.7
  trained R-MBER (30 pilots) constraint-based recursion on the pilots, then frozen
  guided R-MBER, INCM-REC    blind, whole block
  optimum                    per-snapshot MVDR for the true current steering vector
Stages: contam (a second user at 25 degrees sends the same pilots; power relative to the source), onset (an interferer
at 25 degrees switches on after the pilots; INR), labels (number of wrong pilot labels), drift (the source moves from
0 degrees to -D over the data), slip (pilot reference late by 0, 1, 2 symbols).
A run fails when it is more than 3 dB below the optimum.
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
KP=30; KD=120; KT=KP+KD; NR=mc(20); MM=5.0
NM=['guided R-MBER','INCM-REC','TA-MMSE (30 pilots)','TA-MMSE (pilots + DD)','TA-MMSE (all 150 known)','trained R-MBER (30 pilots)','optimum']
def realize(kind,val,r):
    rng=np.random.default_rng(31000+r); p=ps; th=np.zeros(KT)
    if kind=='drift': th[KP:]=np.linspace(0,-val,KD)
    A=np.column_stack([steer(M,np.deg2rad(t)) for t in th]); b=rng.choice([-1.,1.],KT)
    X=A*(np.sqrt(p)*b); piv=sigma2*10**(INR/10)
    for t in theta_ints: X=X+np.outer(steer(M,t),np.sqrt(piv/2)*(rng.standard_normal(KT)+1j*rng.standard_normal(KT)))
    X=X+np.sqrt(sigma2/2)*(rng.standard_normal((M,KT))+1j*rng.standard_normal((M,KT)))
    R=build_R_ipn(M).copy(); a3=steer(M,np.deg2rad(25.)); ph=np.exp(1j*rng.uniform(0,2*np.pi)); u3=rng.standard_normal(KD)+1j*rng.standard_normal(KD)
    flip=rng.permutation(KP); lead=rng.choice([-1.,1.],3)
    if kind=='contam' and val is not None:
        p3=p*10**(val/10); b3=np.r_[b[:KP],rng.choice([-1.,1.],KD)]*ph; X=X+np.outer(a3,np.sqrt(p3)*b3); R=R+p3*np.outer(a3,a3.conj())
    if kind=='onset' and val is not None:
        p3=sigma2*10**(val/10); X[:,KP:]+=np.outer(a3,np.sqrt(p3/2)*u3); R=R+p3*np.outer(a3,a3.conj())
    S=b.astype(complex); Sref=S.copy()
    if kind=='labels' and val: Sref[flip[:int(val)]]*=-1
    if kind=='slip' and val: Sref=np.r_[lead[:int(val)],b[:-int(val)]].astype(complex)
    return X,S,Sref,A[:,KP:],R,p
def seg(w,Ad,R,p): return 10*np.log10(np.mean(p*np.abs(Ad.conj().T@w)**2/np.real(w.conj()@R@w)))
def opt(Ad,R,p): Ri=np.linalg.inv(R); return 10*np.log10(np.mean(p*np.real(np.einsum('mk,mn,nk->k',Ad.conj(),Ri,Ad))))
def cell(kind,val):
    D=[]
    for r in range(NR):
        X,S,Sref,Ad,R,p=realize(kind,val,r); thp=np.deg2rad(MM); a_p=steer(M,thp)
        wb=guided(X,MM)[1]; wi=incm_rec(X,a_p,thp,sector_halfwidth_deg=8.0)
        wt=ta_mmse(X[:,:KP],Sref[:KP]); dec=np.sign(np.real(X[:,KP:].T@wt.conj())); wdd=ta_mmse(X,np.r_[Sref[:KP],dec].astype(complex))
        wfull=ta_mmse(X,S); wr=run_rmber(X[:,:KP],a_p,wc_grid(MM),NU_FAC*nu_pub(M),Sref[:KP])[-1]
        D.append([seg(w,Ad,R,p) for w in (wb,wi,wt,wdd,wfull,wr)]+[opt(Ad,R,p)])
    return np.array(D)
GRID={'contam':[None,-10.,-5.,0.,5.,10.],'onset':[None,0.,5.,10.,15.,20.,25.,30.],'labels':[0,1,2,3,5,8],'drift':[0.,1.,2.,3.,4.,6.],'slip':[0,1,2]}
if __name__=='__main__':
    stage=sys.argv[1]; out={}
    print(f'{stage}: mean SINR over the data part, dB [failed]: '+' | '.join(NM))
    for v in GRID[stage]:
        D=cell(stage,v); o=D[:,-1]; out[str(v)]=D.tolist()
        print(f'{str(v):>6} '+' '.join(f'{D[:,i].mean():7.1f}[{int((D[:,i]<o-3).sum()):>2}]' for i in range(6))+f'  opt {o.mean():5.1f}',flush=True)
    json.dump(out,open(f'res_imperfect_{stage}.json','w'))
