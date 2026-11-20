"""Nominal protected-cluster Gram-matrix conditioning and steering norms of the planar array (Section 5.5.3).
This check does not describe the time-varying tracking estimates.  Writes DATA/geometry.json:
    python -m ch5mprrab.experiments.geometry
"""
import os, json
import numpy as np
from .. import scenarios as MS
MS.use_array('upa')
U = MS.U
checks = {}
for sc, s in U.UPA_SCEN.items():
    indices = s.get('th0_idx', list(range(len(s['az']))))
    ae = [(s['az'][i], s['el'][i]) for i in indices]
    cases = {sc: ae}
    if sc == 'D':
        cases['D after birth'] = ae + [(-30, 5)]
        cases['D after death'] = ae[1:] + [(-30, 5)]
    for label, angles in cases.items():
        A = np.column_stack([U.steer_upa(64, U.encode(np.deg2rad(a), np.deg2rad(e))) for a, e in angles])
        G = A.conj().T @ A
        checks[label] = dict(gram_condition_number=float(np.linalg.cond(G)),
                             steering_norm_squared=np.sum(abs(A)**2, axis=0).tolist())
        assert np.allclose(np.diag(G), 64)
vals = [c['gram_condition_number'] for c in checks.values()]
result = dict(cases=checks, minimum=min(vals), maximum=max(vals))
os.makedirs(MS.DATA, exist_ok=True)
with open(os.path.join(MS.DATA, 'geometry.json'), 'w') as f:
    json.dump(result, f, indent=1)
print('Nominal protected-cluster Gram condition numbers: %.3f to %.3f' % (min(vals), max(vals)))
print('Every planar steering vector has squared norm 64.')
