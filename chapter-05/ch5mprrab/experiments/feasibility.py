"""Joint-feasibility measure nu^T G^-1 nu at kappa = 1 (nu_k = sqrt(M/K)) for every linear and planar geometry of the
study (Section 5.3.2), with G = Re{A~^H A~} for the zero phases and for NPH random phase vectors per geometry.
Writes DATA/feasibility.json:  python -m ch5mprrab.experiments.feasibility"""
import os, json
import numpy as np
from .. import channel as chn
from .. import scenarios as MS
NPH = 10000
rng = np.random.RandomState(0)


def qvals(A):
    M, K = A.shape
    nu = np.full(K, np.sqrt(M / K))
    out = []
    for i in range(NPH + 1):
        ph = np.ones(K) if i == 0 else np.exp(1j * rng.uniform(-np.pi, np.pi, K))
        At = A * ph[None, :]
        G = np.real(At.conj().T @ At)
        out.append(float(nu @ np.linalg.solve(G, nu)))
    return out[0], min(out), max(out)


res = {}
for sc in 'ABCDEFGHI':
    c = MS.SCEN[sc]
    th = [np.deg2rad(t) for t in c['th']]
    if sc == 'D':
        th = [np.deg2rad(t) for t in (0, 25, -30)]          # the three tracked clusters after the birth
    res['ULA ' + sc] = qvals(np.column_stack([chn.steer(16, t) for t in th]))
MS.use_array('upa')
for sc in 'ABCDEFGHI':
    s = MS.U.UPA_SCEN[sc]
    th = [MS.U.encode(np.deg2rad(a), np.deg2rad(e)) for a, e in zip(s['az'], s['el'])]
    if sc == 'D':
        th.append(MS.U.encode(np.deg2rad(-30), np.deg2rad(5)))
    res['UPA ' + sc] = qvals(np.column_stack([chn.steer(64, t) for t in th]))
for k, v in res.items():
    print('%-6s zero phase %.4f   min %.4f   max %.4f' % (k, *v))
allv = [x for v in res.values() for x in v]
print('range over geometries and phases: %.4f .. %.4f' % (min(allv), max(allv)))
os.makedirs(MS.DATA, exist_ok=True)
json.dump({'per_geometry': res, 'min': min(allv), 'max': max(allv), 'n_phase': NPH}, open(os.path.join(MS.DATA, 'feasibility.json'), 'w'), indent=1)
