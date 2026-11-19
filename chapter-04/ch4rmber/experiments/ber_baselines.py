"""Model-based BER of the comparators (ch4rmber.algorithms.NAMES8 order), M = 32, K = 150, 15 block-fading
realizations per point, for the point source and the small and large angular spreads.
Stages: mm (versus mismatch at element SNR -5 dB, Figure 4.5), snr (versus element SNR at 5 degrees, Figure 4.6).
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
mode=sys.argv[1]; NR=15
SCN=[('point source',None),('small angular spread',MP_SMALL),('large angular spread',MP_LARGE)]
gen=gen_block_p
def point(mp,mm,sn,seed):
    p=sigma2*10**(sn/10.); thp=np.deg2rad(mm); a_p=steer(M,thp); ca=wc_grid(mm); o=np.zeros(8)
    for r in range(NR):
        X,b,eff=gen(mp,np.random.default_rng(seed+r),p)
        o+=np.array([ber_of(w,eff,p=p) for w in weights8(X,b,eff,a_p,thp,ca,p=p)])/NR
    return o
out={}
if mode=='mm':
    xs=[0.,1.,2.,3.,5.,7.,9.,11.]
    for lab,mp in SCN: out[lab]=np.column_stack([point(mp,mm,-5.,6000) for mm in xs]).tolist(); print(lab,flush=True)
else:
    xs=list(range(-16,9,2))
    for lab,mp in SCN: out[lab]=np.column_stack([point(mp,5.,float(s),5000) for s in xs]).tolist(); print(lab,flush=True)
json.dump({'x':xs,'res':out},open(f'res_berbase_{mode}.json','w'))
for lab in out:
    print('==',lab); [print(f'{n:<20}',[f'{v:.1e}' for v in row]) for n,row in zip(NAMES8,out[lab])]
