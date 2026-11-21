"""Fixed scenes of Section 7.7.1 (Table 7.2). Scenarios A-D and E use the ISACEnv generator. Scenarios F and G
replace the two 8 dB interferers by three 25 dB interferers at (34, 4), (-17, 1) and (50, -2) deg, 6-12 deg in
azimuth from the nearest served direction; in G they drift linearly to (27, 4), (-11, 1) and (46, -2) deg over the
300 observations. Also defines the 12 steering columns (two azimuth neighbours at +-2 deg per served direction)."""
import numpy as np
from . import simulator as m
M, K = 32, 300
st = lambda a, e: m.steer_upa(8, 4, np.deg2rad(a), np.deg2rad(e))
C = np.column_stack([st(*x) for x in [(20, 5), (18, 5), (22, 5), (-30, 8), (-32, 8), (-28, 8)]])
S = np.column_stack([st(*x) for x in [(-5, 0), (-7, 0), (-3, 0), (40, -3), (38, -3), (42, -3)]])
L = np.c_[C, S]; n = np.array([.5, .35, .35]*2); nu = np.r_[n, n]
CENTERS = [(20, 5), (-30, 8), (-5, 0), (40, -3)]
SC = {'A': (.5, .5), 'B': (3.5, .5), 'C': (.5, 3.5), 'D': (3, 3), 'F': (3, 3), 'G': (3, 3)}
INTF0 = [(34, 4), (-17, 1), (50, -2)]; INTF1 = [(27, 4), (-11, 1), (46, -2)]; INR = 25.0
TRACK_KS = list(range(50, K, 5))          # tracking-average window for F and G

def interferer_angles(k, drift):
    f = k/(K-1) if drift else 0.0
    return [(a0+(a1-a0)*f, e0) for (a0, e0), (a1, _) in zip(INTF0, INTF1)]

def _Rcci(env, angs):
    R = env.sigma2_n*np.eye(M, dtype=complex)
    for a, e in angs: v = st(a, e); R += 10**(INR/10)*np.outer(v, v.conj())
    for th, ph in env.clutter_ang: v = m.steer_upa(8, 4, th, ph); R += env.clutter_power*np.outer(v, v.conj())
    return R

def _hard(seed, drift, KK=K):
    env = m.ISACEnv(8, 4, seed=seed); tr = env.draw_true(3, 3); r = env.rng
    cn = lambda: (r.normal()+1j*r.normal())/np.sqrt(2); X = np.zeros((M, KK), complex); angs = []
    for k in range(KK):
        A = interferer_angles(k, drift); angs.append(A); x = np.zeros(M, complex)
        for h, ps in zip(tr['h_list'], tr['user_ps']): x += np.sqrt(ps)*h*cn()
        for a, b in zip(tr['a_true_targets'], env.target_beta): x += b*a*cn()
        for a, e in A: x += np.sqrt(10**(INR/10))*st(a, e)*cn()
        for th, ph in env.clutter_ang: x += np.sqrt(env.clutter_power)*m.steer_upa(8, 4, th, ph)*cn()
        x += (r.normal(size=M)+1j*r.normal(size=M))/np.sqrt(2); X[:, k] = x
    base = dict(h_list=tr['h_list'], a_true_users=tr['a_true_users'], a_true_targets=tr['a_true_targets'],
                user_ps=tr['user_ps'], beta=env.target_beta)
    return X, [dict(base, R_cci=_Rcci(env, angs[k])) for k in range(KK)]

def scene(sc, seed, KK=K, sigma=None):
    """Returns snapshots X (M x K) and per-snapshot evaluation truth (used only by metrics)."""
    if sc in ('F', 'G'): return _hard(seed, sc == 'G', KK)
    c, s = SC[sc] if sigma is None else sigma
    X, tr = m.ISACEnv(seed=seed).generate_snapshots(KK, c, s); return X, [tr]*KK
