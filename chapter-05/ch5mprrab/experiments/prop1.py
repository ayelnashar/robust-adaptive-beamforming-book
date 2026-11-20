"""Proposition 5.1 check: excess of the oracle SINR over the strongest-cluster MVDR SINR (true covariance, operating
point) for the initial operating-point geometry of all nine scenarios on both arrays; not a bound for every dynamic
state.  Writes DATA/prop1.json:  python -m ch5mprrab.experiments.prop1"""
import os, json
import numpy as np
from .. import channel as chn
from .. import scenarios as MS
out = {}


def excess(R_in, A, P, PS):
    Ri = np.linalg.inv(R_in)
    Rs = (A * P) @ A.conj().T
    lam = np.max(np.real(np.linalg.eigvals(Ri @ Rs))) * PS
    single = max(PS * p * np.real(a.conj() @ Ri @ a) for a, p in zip(A.T, P))
    return float(10 * np.log10(lam / single))


chn.PS = chn.SIGMA2 * 10 ** 0.5; chn.PIV = chn.SIGMA2 * 10 ** 1.0
for sc in 'ABCDEFGHI':
    np.random.seed(1)
    ch = MS.SCEN[sc]['fac']()
    A = np.column_stack([chn.steer(16, t) for t in ch.theta_k])
    out['ULA ' + sc] = excess(ch.true_R_ipn(), A, np.array(ch.power_k, float), chn.PS)
MS.use_array('upa')
for sc in 'ABCDEFGHI':
    np.random.seed(1)
    ch, _ = MS.U.make_channel(sc)
    A = np.column_stack([chn.steer(64, t) for t in ch.theta_k])
    out['UPA ' + sc] = excess(ch.true_R_ipn(), A, np.array(ch.power_k, float), chn.PS)
for n, e in out.items():
    print('%s %.3f dB' % (n, e))
os.makedirs(MS.DATA, exist_ok=True)
json.dump({'excess_dB': out, 'max': max(out.values())}, open(os.path.join(MS.DATA, 'prop1.json'), 'w'), indent=1)
print('max %.3f' % max(out.values()))
