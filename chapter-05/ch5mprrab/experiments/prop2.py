"""Proposition 5.2 check: with the max-SINR targets and phases and zero margins, the active-set solution of the
multi-primary problem equals the largest generalized eigenvector of (A diag(p) A^H, R), for random positive definite
R, random directions and powers, K = 2..5, M = 16 and 64.  Writes DATA/prop2.json:
    python -m ch5mprrab.experiments.prop2"""
import os, json
import numpy as np
from scipy.linalg import eigh as geigh
from .. import algorithms as alg
from .. import scenarios as MS
rng = np.random.default_rng(2026)
worst_cos, worst_rel = 1.0, 0.0
for trial in range(200):
    M = 16 if trial % 2 == 0 else 64
    K = 2 + trial % 4
    B = rng.standard_normal((M, 2 * M)) + 1j * rng.standard_normal((M, 2 * M))
    R = B @ B.conj().T / (2 * M) + 0.05 * np.eye(M)
    R = R / (np.trace(R).real / M)
    th = np.sort(rng.uniform(-1.2, 1.2, K))
    A = np.column_stack([np.exp(1j * np.pi * np.arange(M) * np.sin(t)) for t in th])
    p = rng.uniform(0.1, 1.0, K)
    r = alg.maxsinr_responses(R, A, p)
    At = A * np.exp(1j * np.angle(r))[None, :]
    Qr, Lr = alg.real_form(R, At)
    s, U = np.linalg.eigh(0.5 * (Qr + Qr.T))
    wr, n_out, act, mu = alg.as_snc((np.maximum(s, 1e-14), U), Lr, np.zeros(K), np.abs(r) / np.abs(r).max())
    w = wr[:M] + 1j * wr[M:]
    H = (A * p) @ A.conj().T
    ev, V = geigh((H + H.conj().T) / 2, R)
    wo = V[:, -1]
    cos = abs(w.conj() @ wo) / np.linalg.norm(w) / np.linalg.norm(wo)
    sinr = np.real(w.conj() @ H @ w) / np.real(w.conj() @ R @ w)
    worst_cos = min(worst_cos, cos)
    worst_rel = max(worst_rel, abs(sinr - ev[-1]) / ev[-1])
print('200 random cases: smallest |cos| between the zero-margin solution and the generalized eigenvector %.12f, '
      'largest relative SINR difference %.2e' % (worst_cos, worst_rel))
os.makedirs(MS.DATA, exist_ok=True)
json.dump({'cases': 200, 'smallest_abs_cos': float(worst_cos), 'largest_rel_sinr_diff': float(worst_rel)},
          open(os.path.join(MS.DATA, 'prop2.json'), 'w'), indent=1)
