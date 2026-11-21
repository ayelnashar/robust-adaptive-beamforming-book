"""Projection reconstruction: the protected-subspace baseline of Section 7.7.2.

The part of the cumulative sample covariance inside the 95% energy subspace of the nominal steering matrix is
replaced by the noise level, and the radius-matched worst-case problem (bound vector 3 nu_0) is solved
iteratively: each weight is the best of at most 60 accelerated projected-gradient steps, each step projects onto
the constraint sets with at most 60 Dykstra cycles, and the weight is updated at the first observation and then
every 30 observations. It uses the supplied steering columns and the snapshots only; it never calls a conic
solver and never uses scene truth.
"""
import numpy as np
from scipy.optimize import brentq

CONFIG=dict(update_period=30,outer_iterations=60,projection_cycles=60,
            subspace_energy=.95,radius_multiplier=3.,projection_tol=1e-7)

def robust_margins(w,L,nu,rhs=None):
 if rhs is None:rhs=np.ones(len(nu))
 return (L.conj().T@w).real-nu*np.linalg.norm(w)-rhs

def project_one(v,a,nu,b):
 """Exact Euclidean projection onto Re(a^H w)-nu||w|| >= b."""
 if np.vdot(a,v).real-nu*np.linalg.norm(v)>=b:return v,False
 aa=np.vdot(a,a).real
 if nu<0 or nu>=np.sqrt(aa) or b<=0:raise ValueError('Invalid robust set')
 def at(lam):
  y=v+lam*a;r=np.linalg.norm(y)
  return max(0.,1-lam*nu/max(r,1e-30))*y
 def f(lam):
  x=at(lam);return np.vdot(a,x).real-nu*np.linalg.norm(x)-b
 hi=max(1.,(b-np.vdot(a,v).real+nu*np.linalg.norm(v))/(np.sqrt(aa)-nu)**2)
 for _ in range(100):
  if f(hi)>=0:break
  hi*=2
 else:raise RuntimeError('Projection root not bracketed')
 return at(brentq(f,0,hi,xtol=1e-12)),True

def projection(v,L,nu,rhs,cycles=60,tol=1e-7):
 """Dykstra with active single-set projections and exact radial restoration.
Finite cycles give an approximate nearest-point projection. Feasibility of the
returned weight is checked explicitly; no claim of exact finite-cycle FISTA.
"""
 x=v.copy();p=np.zeros_like(L);calls=0;restoration=1.
 for it in range(cycles):
  old=x.copy()
  for i in range(L.shape[1]):
   y=x+p[:,i];z,active=project_one(y,L[:,i],nu[i],rhs[i]);calls+=int(active)
   p[:,i]=y-z;x=z
  margins=robust_margins(x,L,nu,rhs)
  if np.linalg.norm(x-old)<tol and margins.min()>-tol:break
 gain=(L.conj().T@x).real-nu*np.linalg.norm(x)
 if gain.min()<=0:raise RuntimeError('Cannot restore a nonpositive robust direction')
 restoration=max(1.,float(np.max((rhs+1e-9)/gain)))
 x*=restoration
 if robust_margins(x,L,nu,rhs).min() < -1e-8:raise RuntimeError('Restoration failed')
 return x,dict(cycles=it+1,active_projections=calls,restoration_scale=restoration)

def make_projector(L,energy=.95):
 U,s,_=np.linalg.svd(L,full_matrices=False)
 rank=int(np.searchsorted(np.cumsum(s*s)/np.sum(s*s),energy)+1)
 U=U[:,:rank];return np.eye(L.shape[0])-U@U.conj().T,rank

def reconstruct(R,P):
 """Remove nominal desired-subspace covariance; retain noise in that subspace.
The complement covariance may retain off-subspace desired energy. This is an
estimator, not the true interference covariance and not oracle information.
"""
 R=(R+R.conj().T)/2;M=len(R)
 sigma=max(float(np.mean(np.linalg.eigvalsh(R)[:M//2])),1e-8)
 out=P@R@P+sigma*(np.eye(M)-P)
 return (out+out.conj().T)/2,sigma

def solve_active(R,L,nu,w=None,alpha=.5,nc=None,outer_iterations=60,projection_cycles=60):
 """Accelerated projected covariance descent, with monotone best-iterate retention.
alpha sets positive gain floors; at alpha=.5 every RHS equals one.
"""
 if not 0<=alpha<=1:raise ValueError('alpha must lie in [0,1]')
 if nc is None:nc=L.shape[1]//2
 rhs=np.r_[np.full(nc,np.sqrt(max(2*alpha,.1))),np.full(L.shape[1]-nc,np.sqrt(max(2*(1-alpha),.1)))]
 if w is None:w=L@np.ones(L.shape[1])/L.shape[0]
 w,d=projection(w,L,nu,rhs,cycles=projection_cycles)
 best=w.copy();bestobj=np.vdot(w,R@w).real;y=w.copy();t=1.;lip=np.linalg.eigvalsh(R)[-1]
 calls=d['active_projections'];maxrestore=d['restoration_scale'];completed=0
 for i in range(outer_iterations):
  nxt,d=projection(y-R@y/lip,L,nu,rhs,cycles=projection_cycles)
  calls+=d['active_projections'];maxrestore=max(maxrestore,d['restoration_scale'])
  obj=np.vdot(nxt,R@nxt).real
  if obj<bestobj:best=nxt.copy();bestobj=obj
  completed=i+1
  if np.linalg.norm(nxt-w)<1e-7:break
  tn=(1+np.sqrt(1+4*t*t))/2;y=nxt+(t-1)/tn*(nxt-w);w=nxt;t=tn
 return best,dict(iterations=completed,active_projections=calls,max_restoration=maxrestore,
                  min_margin=float(robust_margins(best,L,nu,rhs).min()),objective=float(bestobj))

def run_projection_reconstruction(X,L_C,L_S,nu0_C,nu0_S,alpha=.5,return_diagnostics=False):
 M,K=X.shape;L=np.c_[L_C,L_S];nu=CONFIG['radius_multiplier']*np.r_[nu0_C,nu0_S]
 P,rank=make_projector(L,CONFIG['subspace_energy']);acc=np.zeros((M,M),complex)
 w=None;hist=[];diag=[];last=None
 for k in range(K):
  acc+=np.outer(X[:,k],X[:,k].conj())
  update=(k==0 or (k+1)%CONFIG['update_period']==0 or k==K-1)
  if update:
   R=(acc+np.eye(M))/(k+2);RR,sigma=reconstruct(R,P)
   w,last=solve_active(RR+1e-4*np.eye(M),L,nu,w,alpha,L_C.shape[1],CONFIG['outer_iterations'],CONFIG['projection_cycles'])
  hist.append(w.copy());diag.append(dict(k=k,updated=update,rank=rank,min_margin=last['min_margin'],
    active_projections=last['active_projections'] if update else 0,iterations=last['iterations'] if update else 0,
    max_restoration=last['max_restoration'] if update else 1.,noise_estimate=sigma))
 return (hist,diag) if return_diagnostics else hist
