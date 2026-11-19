"""Decision-directed constraint-based R-MBER with and without blind phase recovery against the trained form
(Section 4.7.10): mismatch sweeps for M = 32, 16 and 64 on the realizations of Figure 4.2, and the eight cases of
Figure 4.3. Also reports the decision error rate over the first 20 snapshots and the runs ending more than 3 dB
below the trained form.
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
NR=mc(20); OUT={'cases':{}}
def sweep(Mloc, mms):
    R=core.build_R_ipn(Mloc); at=steer(Mloc,0.0).astype(complex); nu=NU_FAC*nu_pub(Mloc)
    A=np.zeros((3,len(mms))); E=np.zeros((2,len(mms))); F=np.zeros((2,len(mms)))
    for j,mm in enumerate(mms):
        thp=np.deg2rad(mm); a_p=steer(Mloc,thp); ca=wc_grid(mm)
        for r in range(NR):
            np.random.seed(1000+r); X,S=gen_with_symbols(Mloc,K,0.0,None)
            wt=run_rmber(X,a_p,ca,nu,S)[-1]
            h0,d0=run_blind(X,a_p,ca,nu,phase=False,ret_dec=True); h1,d1=run_blind(X,a_p,ca,nu,phase=True,ret_dec=True)
            st=sinr_db(wt,at,R,ps)
            for i,w in enumerate([wt,h0[-1],h1[-1]]): A[i,j]+=sinr_db(w,at,R,ps)/NR
            for i,(h,d) in enumerate([(h0,d0),(h1,d1)]):
                e=np.mean(d[:20]!=S.real[:20]); E[i,j]+=min(e,1-e)/NR           # decision errors, first 20 snapshots
                F[i,j]+=(sinr_db(h[-1],at,R,ps) < st-3.0)/NR                    # runs ending >3 dB below trained
    print(f'\nM={Mloc}, point source, {NR} trials; half-power beamwidth {np.rad2deg(0.886*2/Mloc):.1f} deg, first null {np.rad2deg(np.arcsin(2/Mloc)):.1f} deg')
    print('mismatch (deg)            '+''.join(f'{m:>7.1f}' for m in mms))
    for n,row in zip(['Trained R-MBER','Blind, no phase rec.','Blind, phase rec.'],A): print(f'{n:<26}'+''.join(f'{v:>7.1f}' for v in row))
    print('early dec. err, no phase  '+''.join(f'{v:>7.2f}' for v in E[0])); print('early dec. err, phase     '+''.join(f'{v:>7.2f}' for v in E[1]))
    print('failed runs, no phase     '+''.join(f'{v:>7.2f}' for v in F[0])); print('failed runs, phase        '+''.join(f'{v:>7.2f}' for v in F[1]))
    OUT[str(Mloc)]={'mms':mms,'A':A.tolist(),'E':E.tolist(),'F':F.tolist()}
sweep(32,[0.,0.5,1.,1.5,2.,2.5,3.,5.,7.,9.,11.])
sweep(16,[0.,1.,2.,3.,4.,5.,7.,9.,11.])
sweep(64,[0.,0.25,0.5,0.75,1.,1.25,1.5,2.,3.,5.])
# eight cases of Figure 4.3 (block fading, realized effective steering vector)
print('\neight cases of Figure 4.3, M=32, block fading, 20 trials')
SC=[('2 pt',2.,None),('5 pt',5.,None),('7 pt',7.,None),('2 sm',2.,MP_SMALL),('5 sm',5.,MP_SMALL),('7 sm',7.,MP_SMALL),('5 lg',5.,MP_LARGE),('7 lg',7.,MP_LARGE)]
nu=NU_FAC*nu_pub(M)
for lab,mm,mp in SC:
    thp=np.deg2rad(mm); a_p=steer(M,thp); ca=wc_grid(mm); v=np.zeros(3)
    for r in range(NR):
        rng=np.random.default_rng(1000+r); X,Sy,eff=gen_block(M,K,mp,rng)
        ws=[run_rmber(X,a_p,ca,nu,Sy)[-1],run_blind(X,a_p,ca,nu,phase=False)[-1],run_blind(X,a_p,ca,nu,phase=True)[-1]]
        v+=np.array([sinr_db(w,eff,core.R_ipn,ps) for w in ws])/NR
    print(f'{lab:<6} trained {v[0]:6.1f}   blind, no phase rec. {v[1]:6.1f}   blind, phase rec. {v[2]:6.1f}')
    OUT['cases'][lab]=v.tolist()
json.dump(OUT,open('res_phase.json','w'))
