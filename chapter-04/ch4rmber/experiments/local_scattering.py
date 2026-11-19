"""BER step under local scattering (Section 4.7.9, Figure 4.11): guide alone against guided R-MBER, M = 32,
5 degrees of mismatch.
Stages
  amp  two local rays at +4 and -3 degrees (the small-spread geometry), ray amplitude 0 to 0.6
  off  two local rays of amplitude 0.4 at +d and -d degrees, d = 1 to 20
  ci   paired gain of the BER step over the guide alone in the eight cases of Figure 4.3 (from res_guided_mp.json),
       with 95% t intervals (Section 4.7.3)
  wc   pointing error covered by the WC-MVDR/EigDec radius 3.8 sqrt(M/16) for M = 8, 16, 32, 64, with the distance to
       the true steering vector taken after removing its common phase (Section 4.7.1)
  dir  the eight cases of Figure 4.3: guide, guided R-MBER and a direction-only BER step (gradient component along w
       removed, step 0.01 ||w||, norm held at that of the guide weight) (Section 4.7.9)
For amp and off the weights are computed from a K = 150 block at element SNR 5 dB (output SINR) and from a second block
at -10 dB (counted BER). The counted BER uses NH held-out BPSK symbols of the same channel; the common output phase is
estimated blindly as 0.5 arg sum y^2 over the held-out block and the sign ambiguity of BPSK is resolved with the known
sign of the realized response. Seeds 41000 + r (SINR) and 51000 + r (BER), with the held-out block drawn after the
training block from the same generator.
"""
import sys, json, time
import numpy as np
from ch4rmber import algorithms as core
from ch4rmber.algorithms import *
from ch4rmber.scenarios import mc, QUICK
NR=mc(20); NH=2000 if QUICK else 50000; MM=5.0
def gen(rays, p, rng, Kn):
    """training block and held-out block of one block-fading channel (ray phases fixed)."""
    a0=steer(M,0.0).astype(complex); eff=a0.copy()
    for d,amp in rays: eff=eff+amp*np.exp(1j*rng.uniform(-np.pi,np.pi))*steer(M,np.deg2rad(d))
    def block(n):
        b=rng.choice([-1.0,1.0],size=n); X=np.outer(eff,np.sqrt(p)*b); piv=sigma2*10**(INR/10)
        for th in theta_ints: X=X+np.outer(steer(M,th),np.sqrt(piv/2)*(rng.standard_normal(n)+1j*rng.standard_normal(n)))
        return X+np.sqrt(sigma2/2)*(rng.standard_normal((M,n))+1j*rng.standard_normal((M,n))),b
    X,b=block(Kn); Xh,bh=block(NH); return X,b,eff,Xh,bh
def counted(w,eff,Xh,bh):
    y=Xh.T@w.conj(); ph=0.5*np.angle(np.sum(y**2)); z=np.real(np.exp(-1j*ph)*y)
    if np.real(np.exp(-1j*ph)*(w.conj()@eff))<0: z=-z
    return float(np.mean(np.sign(z)!=bh))
def weights(X):
    thp=np.deg2rad(MM); w0,G,_=guide(X,MM); wb=ber_stage(X,w0,G)
    wi=incm_rec(X,steer(M,thp),thp,sector_halfwidth_deg=8.0); return w0,wb,wi
NAMES=['guide alone','guided R-MBER','INCM-REC','optimum']
def cell(rays):
    S=[];B=[]
    for r in range(NR):
        rng=np.random.default_rng(41000+r); X,b,eff,_,_=gen(rays,ps,rng,K)
        ws=list(weights(X))+[np.linalg.solve(R_ipn,eff)]; S.append([sinr_db(w,eff,R_ipn,ps) for w in ws])
        p10=sigma2*10**(-10/10); rng=np.random.default_rng(51000+r); X,b,eff,Xh,bh=gen(rays,p10,rng,K)
        ws=list(weights(X))+[np.linalg.solve(R_ipn,eff)]; B.append([counted(w,eff,Xh,bh) for w in ws])
    return S,B
def ber_dir(X,w0,G,alpha=0.01,Z=80,rho_c=RHO_C):
    w=w0.copy(); n0=np.linalg.norm(w0)
    for k in range(X.shape[1]):
        k0=max(0,k-Z+1); Xw=X[:,k0:k+1]; Ln=Xw.shape[1]; y=Xw.T@w.conj()
        rot=np.exp(-0.5j*np.angle(np.sum(y**2))); Xw=Xw*rot; y=y*rot
        m=np.real(y); sw=np.sign(m)+0.0
        rho=max(1.06*np.std(m-sw*np.mean(m*sw))*Ln**(-1/5),1e-3); rho=max(rho,rho_c*np.mean(np.abs(m)))
        g=np.exp(-(m**2)/(2*rho**2)); grad=-(1.0/(Ln*rho*np.sqrt(2*np.pi)))*((g*sw)@Xw.T)
        grad=G@grad; grad=grad-w*(w.conj()@grad)/(w.conj()@w); gn=np.linalg.norm(grad)
        if gn>0: w=w-alpha*np.linalg.norm(w)*grad/gn
        w=w*n0/np.linalg.norm(w)
    return w
if __name__=='__main__':
    stage=sys.argv[1]; out={}
    if stage in ('amp','off'):
        vals=[0.0,0.1,0.2,0.3,0.4,0.5,0.6] if stage=='amp' else [1.,2.,3.,4.,6.,8.,10.,15.,20.]
        print(f'{stage}: mean SINR dB [failed] and counted BER at -10 dB: '+' | '.join(NAMES))
        for v in vals:
            rays=[(4.,v),(-3.,v)] if stage=='amp' else [(v,0.4),(-v,0.4)]
            S,B=cell(rays); S=np.array(S); B=np.array(B); o=S[:,3]; out[f'{v:g}']={'S':S.tolist(),'B':B.tolist()}
            print(f'{v:>5g} '+' '.join(f'{S[:,i].mean():6.2f}[{int((S[:,i]<o-3).sum()):>2}]' for i in range(4))+' | '+' '.join(f'{B[:,i].mean():.2e}' for i in range(4))+f' | gain {np.mean(S[:,1]-S[:,0]):+.2f} dB',flush=True)
    elif stage=='dir':
        for lab,mm,mp in [('2 pt',2.,None),('5 pt',5.,None),('7 pt',7.,None),('2 sm',2.,MP_SMALL),('5 sm',5.,MP_SMALL),('7 sm',7.,MP_SMALL),('5 lg',5.,MP_LARGE),('7 lg',7.,MP_LARGE)]:
            D=[]
            for r in range(NR):
                rng=np.random.default_rng(1000+r); X,Sy,eff=gen_block(M,K,mp,rng); w0,G,_=guide(X,mm); wb=ber_stage(X,w0,G); wd=ber_dir(X,w0,G)
                D.append([sinr_db(w,eff,R_ipn,ps) for w in (w0,wb,wd,np.linalg.solve(R_ipn,eff))]+[np.linalg.norm(wb)/np.linalg.norm(w0)])
            D=np.array(D); out[lab]=D.tolist()
            print(f'{lab}: guide {D[:,0].mean():.2f} guided {D[:,1].mean():.2f} ({np.mean(D[:,1]-D[:,0]):+.2f}) direction-only {D[:,2].mean():.2f} ({np.mean(D[:,2]-D[:,0]):+.2f}) norm ratio {D[:,4].min():.1f}-{D[:,4].max():.1f}',flush=True)
    elif stage=='ci':
        from scipy import stats
        for k,v in json.load(open('res_guided_mp.json')).items():
            a=np.array(v); g=a[:,1]-a[:,0]; o=a[:,4]; h=stats.t.ppf(0.975,len(g)-1)*g.std(ddof=1)/np.sqrt(len(g))
            out[k]=[g.mean(),g.mean()-h,g.mean()+h,int((a[:,0]<o-3).sum()),int((a[:,1]<o-3).sum())]
            print(f'{k}: gain {g.mean():+.2f} dB, 95% CI [{g.mean()-h:+.2f}, {g.mean()+h:+.2f}], failed runs guide {out[k][3]} -> guided {out[k][4]}')
    elif stage=='wc':
        th=np.linspace(0,10,10001)
        for Ml in (8,16,32,64):
            z=3.8*np.sqrt(Ml/16); a0=steer(Ml,0.0); d=np.array([np.sqrt(max(2*Ml-2*abs(np.vdot(steer(Ml,np.deg2rad(t)),a0)),0.0)) for t in th])
            out[str(Ml)]=float(th[np.argmax(d>=z)]); print(f'M = {Ml}: radius {z:.2f} covers {out[str(Ml)]:.2f} deg')
    json.dump(out,open(f'res_scat_{stage}.json','w'))
