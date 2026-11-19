"""Model-based BER of the guide alone, guided R-MBER, INCM-REC and TA-MMSE on the BER realizations of Figures 4.5
(versus mismatch, element SNR -5 dB) and 4.6 (versus element SNR, 5 degrees). Stage: ber.
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
stage=sys.argv[1]; NR=mc(20); out={}
if stage=='ber':        # Figures 4.5 and 4.6
    gen=gen_block_p
    for mode,xs,seed in (('mm',[0,1,2,3,5,7,9,11],6000),('snr',list(range(-16,9,2)),5000)):
        for lab,mp in (('point',None),('small',MP_SMALL),('large',MP_LARGE)):
            Y=[]
            for x in xs:
                mm,sn=(x,-5.) if mode=='mm' else (5.,x); p=sigma2*10**(sn/10.); o=np.zeros(4)
                for r in range(mc(15)):
                    X,b,eff=gen(mp,np.random.default_rng(seed+r),p); thp=np.deg2rad(mm); wg,wb=guided(X,float(mm))
                    o+=np.array([ber_of(w,eff,p=p) for w in (wg,wb,incm_rec(X,steer(M,thp),thp,sector_halfwidth_deg=8.0),ta_mmse(X,b.astype(complex)))])/mc(15)
                Y.append(o.tolist())
            out[f'{mode}|{lab}']={'x':xs,'Y':Y}; Y=np.array(Y)
            print(f'{mode}, {lab}: x = {xs}'); [print(f'  {n:<14}'+' '.join(f'{v:.0e}' for v in Y[:,i]),flush=True) for i,n in enumerate(['guide','guided R-MBER','INCM-REC','TA-MMSE'])]
json.dump(out,open(f'res_berguided.json','w'))
