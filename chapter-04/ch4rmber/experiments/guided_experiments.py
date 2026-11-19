"""Guide alone, guided R-MBER, INCM-REC, TA-MMSE (every symbol known) and the optimum, M = 32.

Stages: mp (eight cases, Figure 4.3), near (third interferer near the source, Figure 4.9; adds trained R-MBER),
inr (INR 10 to 30 dB, K = 150, Figure 4.7(a), (b)), nint (2 to 16 interferers in random directions, Figure 4.8(c), (d)).
Columns of each realization: guide alone, guided R-MBER, INCM-REC, TA-MMSE, optimum[, trained R-MBER].
A run fails when its SINR is more than 3 dB below the optimum.
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
NR=mc(20)
NM=['guide alone','guided R-MBER','INCM-REC','TA-MMSE (trained)','optimum']
def row(X,S,mm,a,R,p,extra=False):
    Ml=X.shape[0]; thp=np.deg2rad(mm); wg,wb=guided(X,mm)
    ws=[wg,wb,incm_rec(X,steer(Ml,thp),thp,sector_halfwidth_deg=8.0),ta_mmse(X,S),np.linalg.solve(R,a)]
    if extra:
        ws+=[full(X,mm,S)]
    return [sinr_db(w,a,R,p) for w in ws]
def show(lab,D,names=NM):
    D=np.array(D); B=Q(np.sqrt(2*10**(D/10))); n=D.shape[1]
    print(f'{lab:<30}'+' '.join(f'{D[:,i].mean():6.1f}[{int((D[:,i]<D[:,4]-3).sum()):>2}]' for i in range(n) if i!=4)+f'  opt {D[:,4].mean():5.1f} | BER '+' '.join(f'{B[:,i].mean():.1e}' for i in range(4)),flush=True)
    return D.tolist()
if __name__=='__main__':
    stage=sys.argv[1]; out={}; at=steer(M,0.0).astype(complex)
    hdr='SINR dB [failed]: guide alone | guided R-MBER | INCM-REC | TA-MMSE (trained)'
    def std(Ml,mm,lab,cfo=0.0,extra=False):
        a=steer(Ml,0.0).astype(complex); R=core.build_R_ipn(Ml); D=[]
        for r in range(NR):
            np.random.seed(1000+r); X,S=core.gen_with_symbols(Ml,K,0.0,None)
            if cfo:
                sig=np.outer(a,np.sqrt(core.ps)*S); X=X-sig+sig*np.exp(1j*2*np.pi*cfo*np.arange(K)/K)
            D.append(row(X,S,mm,a,R,core.ps,extra))
        out[lab]=show(lab,D)
    if stage=='mp':        # the eight mismatch and propagation cases (Figure 4.3)
        print(hdr)
        for lab,mm,mp in [('2 pt',2.,None),('5 pt',5.,None),('7 pt',7.,None),('2 sm',2.,MP_SMALL),('5 sm',5.,MP_SMALL),('7 sm',7.,MP_SMALL),('5 lg',5.,MP_LARGE),('7 lg',7.,MP_LARGE)]:
            D=[]
            for r in range(NR):
                rng=np.random.default_rng(1000+r); X,Sy,eff=gen_block(M,K,mp,rng); D.append(row(X,Sy.astype(complex),mm,eff,core.R_ipn,ps))
            out[lab]=show(lab,D)
    elif stage=='near':    # third interferer near the source, INR 10 dB (Figure 4.9)
        print(hdr+' | trained R-MBER   (third interferer near the source, INR 10 dB)')
        for mm in (0.,3.):
            for th3 in (3.,4.,5.,6.,8.,10.,14.):
                core.theta_ints=[np.deg2rad(45.),np.deg2rad(-60.),np.deg2rad(th3)]; std(M,mm,f'{mm:g}|{th3:g}',extra=True)
    elif stage=='inr':     # interference level, K = 150 (Figure 4.7(a), (b))
        print(hdr)
        for inr in (10,15,20,25,30):
            for mm in (0.,5.,7.):
                core.INR=inr; std(M,mm,f'{inr}|{mm:g}')
    elif stage=='nint':    # 2 to 16 interferers in random directions (Figure 4.8(c), (d))
        print(hdr+'  (N interferers at random angles 15 to 70 deg either side)')
        for N in (2,4,6,8,10,12,16):
            for mm in (3.,7.):
                D=[]
                for r in range(NR):
                    g=np.random.default_rng(5000+r); ang=g.uniform(15,70,N)*g.choice([-1,1],N); core.theta_ints=[np.deg2rad(t) for t in ang]; R=core.build_R_ipn(M)
                    np.random.seed(1000+r); X,S=core.gen_with_symbols(M,K,0.0,None); D.append(row(X,S,mm,at,R,ps))
                out[f'{N}|{mm:g}']=show(f'N {N}, {mm:g} deg',D)
    json.dump(out,open(f'res_guided_{stage}.json','w'))
