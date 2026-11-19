"""Trained constraint-based R-MBER and the block comparators on the data of Figures 4.3, 4.4 and 4.7.

Stages:
  cases    final SINR of every comparator (ch4rmber.algorithms.NAMES8 order) in the eight cases of Figure 4.3
  traj     SINR after every snapshot, 7 degrees, point source and large spread (Figure 4.4)
  inr      trained R-MBER, INCM-REC, TA-MMSE versus INR, K = 150 (Figure 4.7(a), (b))
  inr1000  the same at K = 1000 (Figure 4.7(c), (d))
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
stage=sys.argv[1]; out={}
SCEN=[('2 pt',2.,None),('5 pt',5.,None),('7 pt',7.,None),('2 sm',2.,MP_SMALL),('5 sm',5.,MP_SMALL),('7 sm',7.,MP_SMALL),('5 lg',5.,MP_LARGE),('7 lg',7.,MP_LARGE)]
if stage=='cases':      # the eight mismatch and propagation cases (Figure 4.3)
    for lab,mm,mp in SCEN:
        thp=np.deg2rad(mm); a_p=steer(M,thp); ca=wc_grid(mm); acc=np.zeros(8)
        for r in range(mc(20)):
            rng=np.random.default_rng(1000+r); X,Sy,eff=gen_block(M,K,mp,rng)
            acc+=np.array([sinr_db(w,eff,core.R_ipn,ps) for w in weights8(X,Sy,eff,a_p,thp,ca)])/mc(20)
        out[lab]=acc.tolist(); print(lab,np.round(acc,2).tolist(),flush=True)
elif stage=='traj':     # SINR versus snapshots of the recursions and block methods, 7 degrees (Figure 4.4)
    for tag,mp in [('pt',None),('lg',MP_LARGE)]:
        mm=7.; ca=wc_grid(mm); thp=np.deg2rad(mm); a_p=steer(M,thp); ck=[9,24,49,74,99,124,149]; Sm=np.zeros((8,K)); Ia=np.zeros(len(ck)); NR=mc(12)
        for r in range(NR):
            rng=np.random.default_rng(1000+r); X,Sy,eff=gen_block(M,K,mp,rng)
            wo=np.linalg.solve(core.R_ipn,eff); so=sinr_db(wo,eff,core.R_ipn,ps)
            hm=run_mber(X,a_p,Sy,eta); hr=run_rmber(X,a_p,ca,NU_FAC*nu_pub(M),Sy); Rm=np.eye(M,dtype=complex)*sigma2
            for k in range(K):
                Rm=beta*Rm+(1-beta)*np.outer(X[:,k],X[:,k].conj()); Rs=(Rm+Rm.conj().T)/2
                for i,fn in enumerate((lambda: alg_mvdr(Rs,a_p),lambda: alg_dl(Rs,a_p,2*sigma2),lambda: wc_exact(Rs,a_p,zeta(M)))):
                    try: Sm[i,k]+=sinr_db(fn(),eff,core.R_ipn,ps)
                    except Exception: Sm[i,k]+=-30; fallback(['MVDR','DL-MVDR','WC-MVDR/EigDec'][i],M)
                Sm[3,k]+=sinr_db(hm[k],eff,core.R_ipn,ps); Sm[4,k]+=sinr_db(hr[k],eff,core.R_ipn,ps)
                Sm[6,k]+=sinr_db(ta_mmse(X[:,:k+1],Sy[:k+1]),eff,core.R_ipn,ps); Sm[7,k]+=so
            for c,kk in enumerate(ck): Ia[c]+=sinr_db(incm_rec(X[:,:kk+1],a_p,thp,sector_halfwidth_deg=8.0),eff,core.R_ipn,ps)
        Sm/=NR; Ia/=NR; out[tag]={'S':Sm.tolist(),'I':Ia.tolist()}
        print(tag,'final',np.round(Sm[:,-1],1).tolist(),'INCM',round(Ia[-1],1),'early',np.round(Sm[:,4],1).tolist(),flush=True)
elif stage=='inr':      # trained R-MBER, INCM-REC and TA-MMSE versus INR at K = 150 (Figure 4.7(a), (b))
    at=steer(M,0.0).astype(complex); nu=NU_FAC*nu_pub(M)
    def go(inr,ths,mms,Kn=K):
        core.INR=inr; core.theta_ints=[np.deg2rad(t) for t in ths]; R=core.build_R_ipn(M); o=np.zeros((3,len(mms)))
        for j,mm in enumerate(mms):
            thp=np.deg2rad(mm); a_p=steer(M,thp); ca=wc_grid(mm)
            for r in range(mc(20)):
                np.random.seed(1000+r); X,S=core.gen_with_symbols(M,Kn,0.0,None)
                o[:,j]+=np.array([sinr_db(w,at,R,ps) for w in (run_rmber(X,a_p,ca,nu,S)[-1],incm_rec(X,a_p,thp,sector_halfwidth_deg=8.0),ta_mmse(X,S))])/mc(20)
        return o
    for inr in [10,15,20,25,30]: o=go(inr,[45,-60],[0.,5.,7.]); out[f'inr{inr}']=o.tolist(); print(inr,np.round(o,1).tolist(),flush=True)
    o=go(30,[45,-60],[0.,5.],Kn=1000); out['K1000']=o.tolist(); print('K1000',np.round(o,1).tolist())
    o=go(20,[20,-25],[0.,2.,5.,7.]); out['close']=o.tolist(); print('close',np.round(o,1).tolist())
elif stage=='inr1000':   # trained R-MBER, INCM-REC and TA-MMSE versus INR at K = 1000 (Figure 4.7(c), (d))
    at=steer(M,0.0).astype(complex); nu=NU_FAC*nu_pub(M)
    for inr in [10,15,20,25,30]:
        core.INR=inr; core.theta_ints=[np.deg2rad(45.),np.deg2rad(-60.)]; R=core.build_R_ipn(M); o=np.zeros((3,3))
        for j,mm in enumerate([0.,5.,7.]):
            thp=np.deg2rad(mm); a_p=steer(M,thp); ca=wc_grid(mm)
            for r in range(mc(20)):
                np.random.seed(1000+r); X,S=core.gen_with_symbols(M,1000,0.0,None)
                o[:,j]+=np.array([sinr_db(w,at,R,ps) for w in (run_rmber(X,a_p,ca,nu,S)[-1],incm_rec(X,a_p,thp,sector_halfwidth_deg=8.0),ta_mmse(X,S))])/mc(20)
        out[f'inr{inr}']=o.tolist(); print(inr,np.round(o,1).tolist(),flush=True)
json.dump(out,open(f'res_trained_{stage}.json','w'))
