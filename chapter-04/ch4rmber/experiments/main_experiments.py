"""Guided R-MBER and the comparators on the main data sets of Section 4.7.

Stages (python -m ch4rmber.experiments.main_experiments <stage>):
  sweep8, sweep16, sweep32, sweep64  mismatch sweep, same realizations for every method (Figure 4.2)
  time      per-snapshot runtime versus M, median of four runs, best of five repetitions (Figure 4.1)
  conv      SINR versus block length of the guided form, 7 degrees, point source and large spread (Figure 4.4)
  inr1000   guided form at K = 1000 for INR 10 to 30 dB (Figure 4.7(c), (d))
  snr       guided form versus element SNR, point source, 2 and 5 degrees (Figure 4.8(a), (b))
  near      blind R-MBER (full array) with a third interferer near the source (Figure 4.9)
  berb      blind R-MBER (full array) on the BER realizations (Figures 4.5, 4.6)
  inrx      WC-MVDR/EigDec and blind and trained R-MBER versus INR, K = 150 and 1000 (Figure 4.7)
  ta30      TA-MMSE from a preamble of 30 pilots on the data of Figures 4.2, 4.3, 4.7 and 4.8 (Section 4.7.8)
  spss      SPSS [14] on the realizations of Figures 4.2, 4.3 and 4.5 to 4.10
A run fails when its SINR is more than 3 dB below the optimum on the same block.
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
NR=mc(20); stage=sys.argv[1]; out={}
NAMES=['guide alone','guided R-MBER','INCM-REC','WC-MVDR/EigDec','blind R-MBER','trained R-MBER','TA-MMSE','optimum']
def cell(X,S,mm,a,R,p):
    Ml=X.shape[0]; thp=np.deg2rad(mm); a_p=steer(Ml,thp); wg,wb=guided(X,mm)
    try: wwc=wc_exact(cov_ewma(X),a_p,zeta(Ml))
    except Exception: wwc=a_p/Ml; fallback('WC-MVDR/EigDec',Ml)
    ws=[wg,wb,incm_rec(X,a_p,thp,sector_halfwidth_deg=8.0),wwc,full(X,mm,None),full(X,mm,S),ta_mmse(X,S),np.linalg.solve(R,a)]
    return [sinr_db(w,a,R,p) for w in ws]
def show(lab,D):
    D=np.array(D); o=D[:,-1]
    print(f'{lab:<8}'+' '.join(f'{D[:,i].mean():6.1f}[{int((D[:,i]<o-3).sum()):>2}]' for i in range(D.shape[1]-1))+f'  {o.mean():5.1f}',flush=True)
    return D.tolist()
if stage.startswith('sweep'):
    Ml=int(stage[5:]); mms={8:[0.,2.,4.,6.,8.,10.,12.],16:[0.,1.,2.,3.,4.,5.,7.,9.],32:[0.,0.5,1.,1.5,2.,2.5,3.,5.,7.,9.,11.],64:[0.,0.5,1.,1.5,2.,3.,5.,7.,9.]}[Ml]
    a=steer(Ml,0.0).astype(complex); R=core.build_R_ipn(Ml)
    print(f'M = {Ml}, mean SINR dB [failed]: '+' | '.join(NAMES[:-1])+' | optimum')
    for mm in mms:
        D=[]
        for r in range(NR):
            np.random.seed(1000+r); X,S=core.gen_with_symbols(Ml,K,0.0,None); D.append(cell(X,S,mm,a,R,ps))
        out[f'{mm:g}']=show(f'{mm:g}',D)
elif stage=='time':
    def best(f,inner,rep=5):
        o=[]
        for _ in range(rep):
            f(); t0=time.perf_counter()
            for _ in range(inner): f()
            o.append((time.perf_counter()-t0)/inner*1e6)
        return min(o)
    TN=['MVDR','DL-MVDR','WC-MVDR/EigDec','MBER','trained R-MBER','blind R-MBER','INCM-REC','TA-MMSE','guide alone','guided R-MBER']
    runs=[]
    for run_ in range(4):
        rows={}
        for Ml in (16,32,64,128):
            np.random.seed(1000); X,S=core.gen_with_symbols(Ml,K,0.0,None); thp=np.deg2rad(5.); a_p=steer(Ml,thp); ca=wc_grid(5.); nu=NU_FAC*nu_pub(Ml)
            Rs=cov_ewma(X)
            t=[best(lambda: alg_mvdr(Rs,a_p),50),best(lambda: alg_dl(Rs,a_p,2*sigma2),50),best(lambda: wc_exact(Rs,a_p,zeta(Ml)),20),
               best(lambda: run_mber(X,a_p,S,eta),1)/K,best(lambda: run_rmber(X,a_p,ca,nu,S),1)/K,best(lambda: run_blind(X,a_p,ca,nu,phase=True),1)/K,
               best(lambda: incm_rec(X,a_p,thp,sector_halfwidth_deg=8.0),3),best(lambda: ta_mmse(X,S),50),
               best(lambda: guide(X,5.),1)/K,best(lambda: guided(X,5.),1)/K]
            rows[Ml]=t
        runs.append(rows); print('run',run_+1,{m:[round(v,1) for v in rows[m]] for m in rows},flush=True)
    for Ml in (16,32,64,128):
        med=np.median(np.array([r[Ml] for r in runs]),axis=0).tolist(); out[str(Ml)]=med
        print(f'M = {Ml}: '+', '.join(f'{n} {v:.1f}' for n,v in zip(TN,med)),flush=True)
    out['names']=TN; out['runs']=[{str(m):r[m] for m in r} for r in runs]
    out['note']='block methods (WC, MVDR, DL, INCM-REC, TA-MMSE) timed per solve; recursions and the guided form per snapshot (block time / K)'
elif stage=='conv':
    ks=list(range(10,151,5))
    for tag,mp in (('pt',None),('lg',MP_LARGE)):
        T=np.zeros((2,len(ks))); NRc=mc(12)
        for r in range(NRc):
            rng=np.random.default_rng(1000+r); X,Sy,eff=gen_block(M,K,mp,rng)
            for j,k in enumerate(ks):
                wg,wb=guided(X[:,:k],7.); T[:,j]+=np.array([sinr_db(w,eff,core.R_ipn,ps) for w in (wg,wb)])/NRc
            hb=run_blind(X,steer(M,np.deg2rad(7.)),wc_grid(7.),NU_FAC*nu_pub(M),phase=True)
            if r==0: B=np.zeros(K)
            B+=np.array([sinr_db(hb[k],eff,core.R_ipn,ps) for k in range(K)])/NRc
        out[tag]={'ks':ks,'guide':T[0].tolist(),'guided':T[1].tolist(),'blind':B.tolist()}
        print(f'7 deg, {tag}: k = '+' '.join(f'{k:>5d}' for k in ks[::2]))
        print('  guide  '+' '.join(f'{x:5.1f}' for x in T[0][::2])); print('  guided '+' '.join(f'{x:5.1f}' for x in T[1][::2]),flush=True)
        print('  blind R-MBER at k = 25, 50, 100, 150: '+' '.join(f'{B[k-1]:5.1f}' for k in (25,50,100,150)))
elif stage=='inr1000':
    a=steer(M,0.0).astype(complex); print('K = 1000: INR/mismatch | guide | guided | INCM-REC | TA-MMSE | optimum')
    for inr in (10,15,20,25,30):
        for mm in (0.,5.,7.):
            core.INR=inr; R=core.build_R_ipn(M); D=[]
            for r in range(NR):
                np.random.seed(1000+r); X,S=core.gen_with_symbols(M,1000,0.0,None); thp=np.deg2rad(mm); wg,wb=guided(X,mm)
                D.append([sinr_db(w,a,R,ps) for w in (wg,wb,incm_rec(X,steer(M,thp),thp,sector_halfwidth_deg=8.0),ta_mmse(X,S),np.linalg.solve(R,a))])
            D=np.array(D); out[f'{inr}|{mm:g}']=D.tolist(); print(f'  {inr}/{mm:g}  '+' '.join(f'{D[:,i].mean():6.1f}[{int((D[:,i]<D[:,4]-3).sum()):>2}]' for i in range(4)),flush=True)
    core.INR=10
elif stage=='snr':
    a=steer(M,0.0).astype(complex); R=core.build_R_ipn(M); print('element SNR, point source: mismatch/SNR | guide | guided | optimum')
    for mm in (2.,5.):
        for sn in (-10.,-7.5,-5.,-2.5,0.,2.5,5.,7.5,10.):
            core.ps=sigma2*10**(sn/10); D=[]
            for r in range(NR):
                np.random.seed(1000+r); X,S=core.gen_with_symbols(M,K,0.0,None); wg,wb=guided(X,mm)
                D.append([sinr_db(w,a,R,core.ps) for w in (wg,wb,np.linalg.solve(R,a))])
            D=np.array(D); out[f'{mm:g}|{sn:g}']=D.tolist(); print(f'  {mm:g}/{sn:g}  '+' '.join(f'{D[:,i].mean():6.1f}[{int((D[:,i]<D[:,2]-3).sum()):>2}]' for i in range(2))+f'  {D[:,2].mean():5.1f}',flush=True)
    core.ps=ps
elif stage=='near':      # blind R-MBER (full array) with a third interferer near the source (Figure 4.9)
    a=steer(M,0.0).astype(complex)
    for mm in (0.,3.):
        for th3 in (3.,4.,5.,6.,8.,10.,14.):
            core.theta_ints=[np.deg2rad(45.),np.deg2rad(-60.),np.deg2rad(th3)]; R=core.build_R_ipn(M); v=[]
            for r in range(NR):
                np.random.seed(1000+r); X,S=core.gen_with_symbols(M,K,0.0,None); v.append(sinr_db(full(X,mm,None),a,R,ps))
            out[f'near|{mm:g}|{th3:g}']=v; print('near',mm,th3,round(np.mean(v),1),flush=True)
    core.theta_ints=[np.deg2rad(45.),np.deg2rad(-60.)]
elif stage=='berb':      # blind R-MBER (full array) on the BER realizations of Figures 4.5 and 4.6 (15 realizations)
    gen=gen_block_p
    for mode,xs,seed in (('mm',[0,1,2,3,5,7,9,11],6000),('snr',list(range(-16,9,2)),5000)):
        for lab,mp in (('point',None),('small',MP_SMALL),('large',MP_LARGE)):
            Y=[]
            for x in xs:
                mm,sn=(x,-5.) if mode=='mm' else (5.,x); p=sigma2*10**(sn/10.); o=0.
                for r in range(mc(15)):
                    X,b,eff=gen(mp,np.random.default_rng(seed+r),p); o+=ber_of(full(X,float(mm),None),eff,p=p)/mc(15)
                Y.append(o)
            out[f'{mode}|{lab}']={'x':xs,'Y':Y}; print(mode,lab,' '.join(f'{v:.0e}' for v in Y),flush=True)
elif stage=='inrx':      # WC-MVDR/EigDec and blind R-MBER on the interference-level data, K = 150 and 1000
    a=steer(M,0.0).astype(complex)
    for Kn in (150,1000):
        for inr in (10,15,20,25,30):
            for mm in (0.,5.,7.):
                core.INR=inr; R=core.build_R_ipn(M); D=[]
                for r in range(NR):
                    np.random.seed(1000+r); X,S=core.gen_with_symbols(M,Kn,0.0,None); a_p=steer(M,np.deg2rad(mm))
                    try: wwc=wc_exact(cov_ewma(X),a_p,zeta(M))
                    except Exception: wwc=a_p/M; fallback('WC-MVDR/EigDec',M)
                    D.append([sinr_db(w,a,R,ps) for w in (wwc,full(X,mm,None),full(X,mm,S))])
                D=np.array(D); out[f'{Kn}|{inr}|{mm:g}']=D.tolist(); print(Kn,inr,mm,np.round(D.mean(0),1).tolist(),flush=True)
    core.INR=10
elif stage=='ta30':      # TA-MMSE from the first KP = 30 snapshots only (pilot preamble, then frozen), on the data of Figures 4.2, 4.3, 4.7 and 4.8
    KP=30; diffs=[]
    def two(X,S,a,R,p): return [sinr_db(ta_mmse(X[:,:KP],S[:KP]),a,R,p),sinr_db(ta_mmse(X,S),a,R,p)]
    for Ml,mms in ((8,[0.,2.,4.,6.,8.,10.,12.]),(16,[0.,1.,2.,3.,4.,5.,7.,9.]),(32,[0.,0.5,1.,1.5,2.,2.5,3.,5.,7.,9.,11.]),(64,[0.,0.5,1.,1.5,2.,3.,5.,7.,9.])):
        a=steer(Ml,0.0).astype(complex); R=core.build_R_ipn(Ml); D=[]
        for r in range(NR):
            np.random.seed(1000+r); X,S=core.gen_with_symbols(Ml,K,0.0,None); D.append(two(X,S,a,R,ps))
        D=np.array(D); out[f'sweep{Ml}']=D[:,0].tolist(); g=json.load(open(f'res_main_sweep{Ml}.json')); diffs.append(abs(D[:,1]-np.array(g['0'])[:,6]).max())
        print(f'M = {Ml}: TA-MMSE 30 pilots {D[:,0].mean():.1f} dB [{int((D[:,0]<np.array(g["0"])[:,7]-3).sum())}], all {K}: {D[:,1].mean():.1f}',flush=True)
    for lab,mm,mp in [('2 pt',2.,None),('5 pt',5.,None),('7 pt',7.,None),('2 sm',2.,MP_SMALL),('5 sm',5.,MP_SMALL),('7 sm',7.,MP_SMALL),('5 lg',5.,MP_LARGE),('7 lg',7.,MP_LARGE)]:
        D=[]
        for r in range(NR):
            rng=np.random.default_rng(1000+r); X,Sy,eff=gen_block(M,K,mp,rng); D.append(two(X,Sy.astype(complex),eff,core.R_ipn,ps))
        D=np.array(D); out['case|'+lab]=D[:,0].tolist(); g=json.load(open('res_guided_mp.json')); diffs.append(abs(D[:,1]-np.array(g[lab])[:,3]).max())
        print(f'{lab}: TA-MMSE 30 pilots {D[:,0].mean():.1f} dB, all: {D[:,1].mean():.1f}',flush=True)
    a=steer(M,0.0).astype(complex)
    for Kn in (150,1000):
        for inr in (10,15,20,25,30):
            core.INR=inr; R=core.build_R_ipn(M); D=[]
            for r in range(NR):
                np.random.seed(1000+r); X,S=core.gen_with_symbols(M,Kn,0.0,None); D.append(two(X,S,a,R,ps))
            D=np.array(D); out[f'inr|{Kn}|{inr}']=D[:,0].tolist()
            g=json.load(open('res_guided_inr.json' if Kn==150 else 'res_main_inr1000.json')); diffs.append(abs(D[:,1]-np.array(g[f'{inr}|5'])[:,3]).max())
            print(f'K = {Kn}, INR {inr}: TA-MMSE 30 pilots {D[:,0].mean():.1f} dB, all: {D[:,1].mean():.1f}',flush=True)
    core.INR=10; R=core.build_R_ipn(M); cs=json.load(open('res_fullarray_snr.json'))
    for sn in (-10.,-7.5,-5.,-2.5,0.,2.5,5.,7.5,10.):
        core.ps=sigma2*10**(sn/10); D=[]
        for r in range(NR):
            np.random.seed(1000+r); X,S=core.gen_with_symbols(M,K,0.0,None); D.append(two(X,S,a,R,core.ps))
        D=np.array(D); out[f'snr|{sn:g}']=D[:,0].tolist(); diffs.append(abs(D[:,1]-np.array(cs['D'])[cs['keys'].index(f'5/{sn:g}'),:,2]).max())
        print(f'SNR {sn:g}: TA-MMSE 30 pilots {D[:,0].mean():.1f} dB, all: {D[:,1].mean():.1f}',flush=True)
    core.ps=ps; cn=json.load(open('res_fullarray_nint.json'))
    for N in (2,4,6,8,10,12,16):
        D=[]
        for r in range(NR):
            g_=np.random.default_rng(5000+r); ang=g_.uniform(15,70,N)*g_.choice([-1,1],N); core.theta_ints=[np.deg2rad(t) for t in ang]; R=core.build_R_ipn(M)
            np.random.seed(1000+r); X,S=core.gen_with_symbols(M,K,0.0,None); D.append(two(X,S,a,R,ps))
        D=np.array(D); out[f'nint|{N}']=D[:,0].tolist(); diffs.append(abs(D[:,1]-np.array(cn['D'])[cn['keys'].index(f'{N}/7'),:,2]).max())
        print(f'N = {N}: TA-MMSE 30 pilots {D[:,0].mean():.1f} dB, all: {D[:,1].mean():.1f}',flush=True)
    core.theta_ints=[np.deg2rad(45.),np.deg2rad(-60.)]
    print('largest difference of the all-symbol TA-MMSE from the stored results (same realizations):',max(diffs))
elif stage=='spss':      # SPSS [14] on the realizations of Figures 4.2, 4.3 and 4.5 to 4.10
    from ch4rmber.experiments import imperfect_reference as imperfect
    sp=lambda X,mm: spss(X,np.deg2rad(mm))
    for Ml,mms in ((8,[0.,2.,4.,6.,8.,10.,12.]),(16,[0.,1.,2.,3.,4.,5.,7.,9.]),(32,[0.,0.5,1.,1.5,2.,2.5,3.,5.,7.,9.,11.]),(64,[0.,0.5,1.,1.5,2.,3.,5.,7.,9.])):
        a=steer(Ml,0.0).astype(complex); R=core.build_R_ipn(Ml)
        for mm in mms:
            v=[]
            for r in range(NR):
                np.random.seed(1000+r); X,S=core.gen_with_symbols(Ml,K,0.0,None); v.append(sinr_db(sp(X,mm),a,R,ps))
            out[f'sweep{Ml}|{mm:g}']=v
        print(f'M = {Ml}:',' '.join(f'{np.mean(out[f"sweep{Ml}|{mm:g}"]):.1f}' for mm in mms),flush=True)
    for lab,mm,mp in [('2 pt',2.,None),('5 pt',5.,None),('7 pt',7.,None),('2 sm',2.,MP_SMALL),('5 sm',5.,MP_SMALL),('7 sm',7.,MP_SMALL),('5 lg',5.,MP_LARGE),('7 lg',7.,MP_LARGE)]:
        v=[]
        for r in range(NR):
            rng=np.random.default_rng(1000+r); X,Sy,eff=gen_block(M,K,mp,rng); v.append(sinr_db(sp(X,mm),eff,core.R_ipn,ps))
        out['case|'+lab]=v; print(lab,round(np.mean(v),1),flush=True)
    gen=gen_block_p
    for mode,xs,seed in (('mm',[0,1,2,3,5,7,9,11],6000),('snr',list(range(-16,9,2)),5000)):
        for lab,mp in (('point',None),('small',MP_SMALL),('large',MP_LARGE)):
            Y=[]
            for x in xs:
                mm,sn=(x,-5.) if mode=='mm' else (5.,x); p=sigma2*10**(sn/10.); o=0.
                for r in range(mc(15)):
                    X,b,eff=gen(mp,np.random.default_rng(seed+r),p); o+=ber_of(sp(X,float(mm)),eff,p=p)/mc(15)
                Y.append(o)
            out[f'ber|{mode}|{lab}']=Y; print('BER',mode,lab,' '.join(f'{v:.0e}' for v in Y),flush=True)
    a=steer(M,0.0).astype(complex)
    for Kn in (150,1000):
        for inr in (10,15,20,25,30):
            for mm in (0.,5.,7.):
                core.INR=inr; R=core.build_R_ipn(M); v=[]
                for r in range(NR):
                    np.random.seed(1000+r); X,S=core.gen_with_symbols(M,Kn,0.0,None); v.append(sinr_db(sp(X,mm),a,R,ps))
                out[f'inr|{Kn}|{inr}|{mm:g}']=v
        print('INR',Kn,[round(np.mean(out[f'inr|{Kn}|{i}|5']),1) for i in (10,20,30)],flush=True)
    core.INR=10; R=core.build_R_ipn(M)
    for mm in (2.,5.):
        for sn in (-10.,-7.5,-5.,-2.5,0.,2.5,5.,7.5,10.):
            core.ps=sigma2*10**(sn/10); v=[]
            for r in range(NR):
                np.random.seed(1000+r); X,S=core.gen_with_symbols(M,K,0.0,None); v.append(sinr_db(sp(X,mm),a,R,core.ps))
            out[f'snr|{mm:g}|{sn:g}']=v
    core.ps=ps; print('SNR 5 deg',[round(np.mean(out[f'snr|5|{s:g}']),1) for s in (-10.,0.,10.)],flush=True)
    for N in (2,4,6,8,10,12,16):
        for mm in (3.,7.):
            v=[]
            for r in range(NR):
                g_=np.random.default_rng(5000+r); ang=g_.uniform(15,70,N)*g_.choice([-1,1],N); core.theta_ints=[np.deg2rad(t) for t in ang]; R=core.build_R_ipn(M)
                np.random.seed(1000+r); X,S=core.gen_with_symbols(M,K,0.0,None); v.append(sinr_db(sp(X,mm),a,R,ps))
            out[f'nint|{N}|{mm:g}']=v
    core.theta_ints=[np.deg2rad(45.),np.deg2rad(-60.)]; print('nint 3 deg',[round(np.mean(out[f'nint|{n}|3']),1) for n in (2,8,16)],flush=True)
    for mm in (0.,3.):
        for th3 in (3.,4.,5.,6.,8.,10.,14.):
            core.theta_ints=[np.deg2rad(45.),np.deg2rad(-60.),np.deg2rad(th3)]; R=core.build_R_ipn(M); v=[]
            for r in range(NR):
                np.random.seed(1000+r); X,S=core.gen_with_symbols(M,K,0.0,None); v.append(sinr_db(sp(X,mm),a,R,ps))
            out[f'near|{mm:g}|{th3:g}']=v
    core.theta_ints=[np.deg2rad(45.),np.deg2rad(-60.)]
    print('near 3',[round(np.mean(out[f'near|3|{t:g}']),1) for t in (3.,6.,10.,14.)],flush=True)
    for stg,vals in imperfect.GRID.items():
        for val in vals:
            v=[]
            for r in range(imperfect.NR):
                X,S_,Sref,Ad,R,p=imperfect.realize(stg,val,r); v.append(imperfect.seg(sp(X,imperfect.MM),Ad,R,p))
            out[f'ref|{stg}|{val}']=v
        print('ref',stg,[round(np.mean(out[f'ref|{stg}|{v}']),1) for v in vals],flush=True)
    print('SPSS solve failures',FAIL[0])
json.dump(out,open(f'res_main_{stage}.json','w'))
