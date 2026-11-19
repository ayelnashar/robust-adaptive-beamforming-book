"""Bandwidth floor of (4.7a) at 5 degrees of mismatch (Section 4.3.1: final gradient norm and SINR with and without
the floor against the best SINR inside the feasible set) and the held-out check of the model-based BER at -4 dB element
SNR and 5 degrees (Section 4.7.5).
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
ref=feasible_reference(); print('feasible reference',round(ref,2))
# bandwidth floor: 10 realizations at 5 degrees, trained R-MBER with and without the floor (Section 4.3.1)
NF=mc(10)
ca=wc_grid(5.); a_p=steer(M,np.deg2rad(5.)); nu=NU_FAC*nu_pub(M); A=[np.zeros((NF,K)) for _ in range(6)]
for r in range(NF):
    np.random.seed(1000+r); X,Sy=gen_with_symbols(M,K,0.0,None)
    A[0][r],A[1][r],A[2][r]=trace_rmber(X,Sy,a_p,ca,nu,False); A[3][r],A[4][r],A[5][r]=trace_rmber(X,Sy,a_p,ca,nu,True)
g0,g1=np.median(A[0],0),np.median(A[3],0); s0,s1=np.median(A[1],0)[-1],np.median(A[4],0)[-1]
print('final grad no floor',g0[-1],'floor',g1[-1],'ratio',g1[-1]/g0[-1],' median ratio over last 50',np.median(g1[-50:]/g0[-50:]))
print('final SINR no floor',round(s0,2),'floor',round(s1,2),'gap to ref',round(ref-s0,2),round(ref-s1,2))
OUT={'reference':ref,'grad_final':[float(g0[-1]),float(g1[-1])],'sinr_final':[float(s0),float(s1)]}
# held-out BER check: final weights applied to 8 x 10 x 50000 held-out symbols per method (Section 4.7.5)
NHB=2000 if QUICK else 50000
snr=-4.0; p=sigma2*10**(snr/10); a_s=steer(M,0.0).astype(complex); thp=np.deg2rad(5.); a_p=steer(M,thp); ca=wc_grid(5.)
def gen_p(n,rng):
    b=rng.choice([-1.,1.],size=n); X=np.outer(a_s,np.sqrt(p)*b); piv=sigma2*10**(INR/10)
    for th in theta_ints: X=X+np.outer(steer(M,th),np.sqrt(piv/2)*(rng.standard_normal(n)+1j*rng.standard_normal(n)))
    return X+np.sqrt(sigma2/2)*(rng.standard_normal((M,n))+1j*rng.standard_normal((M,n))),b
rec={'Trained R-MBER':[],'WC-MVDR/EigDec':[],'TA-MMSE':[]}
for r in range(mc(8)):
    rng=np.random.default_rng(7000+r); X,S=gen_p(K,rng); Rs=cov_ewma(X)
    for name,w in [('Trained R-MBER',run_rmber(X,a_p,ca,NU_FAC*nu_pub(M),S)[-1]),('WC-MVDR/EigDec',wc_exact(Rs,a_p,zeta(M))),('TA-MMSE',ta_mmse(X,S))]:
        pm=ber_of(w,a_s,p=p); ph=np.exp(-1j*np.angle(w.conj()@a_s)); err=tot=0
        for _ in range(mc(10)):
            Xt,bt=gen_p(NHB,rng); err+=int(np.sum(np.sign(np.real(ph*(w.conj()@Xt)))!=bt)); tot+=NHB
        rec[name].append((pm,err/tot,tot))
for name,rows in rec.items():
    r_=np.array(rows); se=np.sqrt(r_[:,0]*(1-r_[:,0])/r_[:,2])                  # binomial s.e. from the model probability
    print(f'{name:<15} predicted {r_[:,0].mean():.3e} counted {r_[:,1].mean():.3e} errors {int((r_[:,1]*r_[:,2]).sum())} largest |z| {np.max(np.abs((r_[:,1]-r_[:,0])/np.maximum(se,1e-12))):.2f}')
    OUT[name]={'predicted':float(r_[:,0].mean()),'counted':float(r_[:,1].mean()),'errors':int((r_[:,1]*r_[:,2]).sum()),'symbols':int(r_[:,2].sum()),'max_abs_z':float(np.max(np.abs((r_[:,1]-r_[:,0])/np.maximum(se,1e-12))))}
json.dump(OUT,open('res_floor.json','w'))
