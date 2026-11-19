"""Constraint-based R-MBER on the full array, trained and blind, with TA-MMSE and INCM-REC on the same realizations.

Stages: mp (eight cases, Figure 4.3), nint (random interferer directions, Figure 4.8(c), (d)),
snr (element SNR, Figure 4.8(a), (b)). Columns: trained R-MBER, blind R-MBER, TA-MMSE, INCM-REC.
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
def cell(X,S,mm,at,R,p):
    """per-realization SINRs: trained R-MBER, blind R-MBER, TA-MMSE, INCM-REC"""
    thp=np.deg2rad(mm); a_p=steer(X.shape[0],thp)
    ws=[full(X,mm,S),full(X,mm,None),ta_mmse(X,S),incm_rec(X,a_p,thp,sector_halfwidth_deg=8.0)]
    return [sinr_db(w,at,R,p) for w in ws]
def report(title,keys,D,xlab):
    D=np.array(D)            # shape [len(keys), NR, 4]
    print('\n'+title); print(f'{xlab:<34}'+''.join(f'{str(k):>8}' for k in keys))
    names=['trained R-MBER','blind R-MBER','TA-MMSE (trained)','INCM-REC']
    for i,n in enumerate(names): print(f'{n:<34}'+''.join(f'{x:>8.1f}' for x in D[:,:,i].mean(1)))
    sys.stdout.flush()
if __name__=='__main__':
    stage=sys.argv[1]; out={}; NR=mc(20)
    if stage=='mp':
        SC=[('2 pt',2.,None),('5 pt',5.,None),('7 pt',7.,None),('2 sm',2.,MP_SMALL),('5 sm',5.,MP_SMALL),('7 sm',7.,MP_SMALL),('5 lg',5.,MP_LARGE),('7 lg',7.,MP_LARGE)]; D=[]
        for lab,mm,mp in SC:
            row=[]
            for r in range(NR):
                rng=np.random.default_rng(1000+r); X,Sy,eff=gen_block(M,K,mp,rng); row.append(cell(X,Sy.astype(complex),mm,eff,core.R_ipn,ps))
            D.append(row)
        report('eight cases of Figure 4.3 (block fading), M = 32: mean SINR (dB)',[s[0] for s in SC],D,'case'); out['D']=D
    elif stage=='nint':
        at=steer(M,0.0).astype(complex); D=[]; keys=[]
        for N in (2,4,6,8,10,12,16):
            for mm in (3.,7.):
                row=[]
                for r in range(NR):
                    g=np.random.default_rng(5000+r); ang=g.uniform(15,70,N)*g.choice([-1,1],N)
                    core.theta_ints=[np.deg2rad(t) for t in ang]; R=core.build_R_ipn(M)
                    np.random.seed(1000+r); X,S=core.gen_with_symbols(M,K,0.0,None); row.append(cell(X,S,mm,at,R,ps))
                D.append(row); keys.append(f'{N}/{mm:.0f}')
        core.theta_ints=[np.deg2rad(45.0),np.deg2rad(-60.0)]
        report('number of interferers (N / mismatch deg), random directions 15 to 70 deg either side, INR 10 dB, M = 32',keys,D,'N/mismatch'); out['D']=D; out['keys']=keys
    elif stage=='snr':
        at=steer(M,0.0).astype(complex); R=build_R_ipn(M); D=[]; keys=[]
        for mm in (2.,5.):
            for sn in (-10.,-7.5,-5.,-2.5,0.,2.5,5.,7.5,10.):
                core.ps=sigma2*10**(sn/10); row=[]
                for r in range(NR):
                    np.random.seed(1000+r); X,S=core.gen_with_symbols(M,K,0.0,None); row.append(cell(X,S,mm,at,R,core.ps))
                D.append(row); keys.append(f'{mm:.0f}/{sn:g}')
        report('element SNR (mismatch deg / SNR dB), point source, M = 32',keys,D,'mismatch/SNR'); out['D']=D; out['keys']=keys
    json.dump(out,open(f'res_fullarray_{stage}.json','w'))
