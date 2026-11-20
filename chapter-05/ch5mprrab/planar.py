"""
ch5mprrab.planar -- the 8 x 8 uniform planar array of the main study of Chapter 5 (Section 5.5.3): M = 64,
half-wavelength spacing on both axes, its nine scenarios with elevation (Table 5.2) and the two interferers.

Steering:  a(az, el) = a_y(el) (x) a_x(az, el),  a_x[m] = exp(j pi m sin(az) cos(el)),  a_y[n] = exp(j pi n sin(el)).

Direction encoding.  The simulator and the kernel carry a direction as one scalar (radians) and add
azimuth increments to it (drift, refinement moves).  A 2-D direction is encoded as
    th = az + S * k,   S = 10 rad,   k = round(el / DEL),   DEL = 0.025 deg,
so that azimuth arithmetic (|az| < pi/2 << S/2) stays inside the same elevation cell and elevation moves
are integer multiples of DEL (added as S * dk).  Degrees given to the channel are encoded as
    enc_deg = az_deg + rad2deg(S) * k.
"""
import numpy as np
from . import channel as chn
from . import algorithms as alg

NX, NY = 8, 8
M_UPA = NX * NY
S_ENC = 10.0
DEL = np.deg2rad(0.025)
S_DEG = np.rad2deg(S_ENC)


def decode(th):
    k = int(np.round(th / S_ENC))
    return th - k * S_ENC, k * DEL


def encode(az, el):
    return az + S_ENC * int(np.round(el / DEL))


def enc_deg(az_deg, el_deg):
    return az_deg + S_DEG * int(np.round(np.deg2rad(el_deg) / DEL))


def steer_upa(M, th):
    az, el = decode(th)
    ax = np.exp(1j * np.pi * np.arange(NX) * np.sin(az) * np.cos(el))
    ay = np.exp(1j * np.pi * np.arange(NY) * np.sin(el))
    return np.outer(ay, ax).ravel()        # = np.kron(ay, ax), bitwise, without the overhead of kron


def az_el_deg(th):
    az, el = decode(th)
    return np.rad2deg(az), np.rad2deg(el)


class UPAGeometry:
    """Reconstruction grid over the visible hemisphere and 2-D refinement moves for the kernel."""
    name = 'UPA'

    def __init__(self, M=M_UPA, daz=1.0, del_=2.0):
        self.M = M
        self.bw = 0.886 * 2.0 / NX                      # 3-dB beamwidth of one axis (rad)
        az = np.deg2rad(np.arange(-90, 90 + 1e-9, daz))
        el = np.deg2rad(np.arange(-90, 90 + 1e-9, del_))
        cols, dirs = [], []
        for e in el:
            ay = np.exp(1j * np.pi * np.arange(NY) * np.sin(e))
            for a in az:
                cols.append(np.kron(ay, np.exp(1j * np.pi * np.arange(NX) * np.sin(a) * np.cos(e))))
                dirs.append((a, e))
        self.As = np.column_stack(cols)
        self.grid = np.array(dirs)
        self.dth = np.deg2rad(daz) * np.deg2rad(del_)
        self.az_vals, self.el_vals = az, el
        self.n_az, self.n_el = len(az), len(el)

    def sector(self):
        return 0.5 * self.bw if alg.EXP_RAD is None else alg.EXP_RAD

    def mask(self, th_list, sec):
        m = np.ones(len(self.grid), dtype=bool)
        for t in th_list:
            az, el = decode(t)
            m &= (np.abs(self.grid[:, 0] - az) > sec) | (np.abs(self.grid[:, 1] - el) > sec)
        return m

    def refine_peaks(self, spec, msk, Ri, theta_k, sec):
        """Strong local maxima (8-neighbourhood) of the Capon spectrum outside the excluded boxes, located off the grid
        by alternating golden-section searches in azimuth and elevation between the neighbouring grid lines."""
        M = self.M
        S = spec.reshape(self.n_el, self.n_az)
        Mk = msk.reshape(self.n_el, self.n_az)
        thr = alg.PEAK_RHO * np.median(spec)
        dec = [decode(t) for t in theta_k]
        sa = lambda a, e: np.outer(np.exp(1j * np.pi * np.arange(NY) * np.sin(e)), np.exp(1j * np.pi * np.arange(NX) * np.sin(a) * np.cos(e))).ravel()
        def p_of(a, e):
            v = sa(a, e)
            return 1.0 / max(np.real(v.conj() @ (Ri @ v)), 1e-12)
        cols, ws, drop = [], [], []
        nb = np.full(S.shape, np.inf)
        core = S[1:-1, 1:-1]
        nbmax = np.max(np.stack([S[1 + dj:S.shape[0] - 1 + dj, 1 + di:S.shape[1] - 1 + di]
                                 for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)]), axis=0)
        nb[1:-1, 1:-1] = nbmax
        cand = np.argwhere(Mk & (S > thr) & (S >= nb))
        for j, i in cand:
            if True:
                a, e = self.az_vals[i], self.el_vals[j]
                for _ in range(2):
                    a = alg.golden_max(lambda x: p_of(x, e), self.az_vals[i - 1], self.az_vals[i + 1], 16)
                    e = alg.golden_max(lambda y: p_of(a, y), self.el_vals[j - 1], self.el_vals[j + 1], 16)
                if any(abs(a - ak) <= sec and abs(e - ek) <= sec for ak, ek in dec):
                    continue
                cols.append(sa(a, e))
                ws.append(p_of(a, e))
                drop.append(j * self.n_az + i)
        if not cols:
            return None, None, []
        return np.column_stack(cols), np.array(ws), drop

    def refine_candidates(self, th_old):
        az, el = decode(th_old)
        out = []
        for de in np.deg2rad(np.arange(-alg.WIN_DEG, alg.WIN_DEG + 1e-9, 1.0)):
            dk = int(np.round(de / DEL))
            for da in np.linspace(-np.deg2rad(alg.WIN_DEG), np.deg2rad(alg.WIN_DEG), 7):
                out.append(th_old + da + S_ENC * dk)
        return out

    def cap_step(self, th_old, th_new):
        az0, el0 = decode(th_old)
        az1, el1 = decode(th_new)
        cap = np.deg2rad(alg.MAX_STEP_DEG)
        daz = np.clip(az1 - az0, -cap, cap)
        dk = int(np.round(np.clip(el1 - el0, -cap, cap) / DEL))
        return th_old + daz + S_ENC * dk

    def capped(self, th_old, th_new):
        az0, el0 = decode(th_old)
        az1, el1 = decode(th_new)
        cap = np.deg2rad(alg.MAX_STEP_DEG)
        return abs(az1 - az0) > cap or abs(el1 - el0) > cap

    def polish(self, th, p_of):
        """Alternating golden-section searches in azimuth and elevation between the neighbouring scan points."""
        az, el = decode(th)
        h = np.deg2rad(1.0)
        a0, e0 = az, el
        for _ in range(2):
            az = alg.golden_max(lambda x: p_of(encode(x, el)), a0 - h, a0 + h, 16)
            el = alg.golden_max(lambda y: p_of(encode(az, y)), e0 - h, e0 + h, 16)
        return encode(az, el)

    def fit(self, meas):
        t = np.array([m[0] for m in meas], dtype=float)
        t = t - t[-1]
        ae = np.array([decode(m[1]) for m in meas])
        sa, ia = np.polyfit(t, ae[:, 0], 1)
        se, ie = np.polyfit(t, ae[:, 1], 1)
        return encode(ia, ie), np.array([sa, se])

    def advance(self, th, v, tau):
        az, el = decode(th)
        return encode(az + v[0] * tau, el + v[1] * tau)


_ORIG = {}


def patch_in():
    """Route every steering call of the channel and of the kernel to the planar array, set M = 64, install its geometry."""
    _ORIG['steer1'], _ORIG['M1'] = chn.steer, chn.M_DEFAULT
    _ORIG['steer3'], _ORIG['M3'], _ORIG['geom'] = alg.steer, alg.M_DEFAULT, alg.GEOM
    chn.steer = steer_upa
    chn.M_DEFAULT = M_UPA
    alg.steer = steer_upa
    alg.M_DEFAULT = M_UPA
    alg.GEOM = UPAGeometry()


def patch_out():
    chn.steer, chn.M_DEFAULT = _ORIG['steer1'], _ORIG['M1']
    alg.steer, alg.M_DEFAULT, alg.GEOM = _ORIG['steer3'], _ORIG['M3'], _ORIG['geom']


# ---------------------------------------------------------------- UPA scenarios (azimuth, elevation, power)
# The nine scenarios of the linear array with a distinct elevation for every cluster.  A, B, D, E and I keep every
# cluster outside the main lobes of the two interferers; G and H each have one cluster inside the main lobe of the
# interferer at (+45, 0), and C and H one cluster at the azimuth of the interferer at (-10, -25).
_AZ_E, _PW_E = chn.cdl_like_profile('A', K_clusters=5, seed=7)
_TOP_E = [int(i) for i in np.argsort(-np.array(_PW_E))[:3]]
UPA_SCEN = {
    'A': dict(az=[0, 30], el=[10, 20], pw=[1.0, 1.0], drift=0.0, el_drift=None, T=300, K_track=2, K_use=2),
    'B': dict(az=[-20, 5, 35], el=[15, -5, 20], pw=[1.0, 0.3, 0.8], drift=0.0, el_drift=None, T=300, K_track=3, K_use=3),
    'C': dict(az=[-10, 20], el=[10, -5], pw=[1.0, 0.7], drift=0.025, el_drift=[0.025, -0.025], T=400, K_track=2, K_use=2,
              trk=True),
    'D': dict(az=[0, 25], el=[10, -10], pw=[1.0, 0.8], drift=0.0, el_drift=None, T=400, K_track=4, K_use=3, trk=True,
              lc=True, life=[('birth', 150, -30, 5, 0.9), ('death', 280, 0)], bh=[(150, -30, 5)]),
    'E': dict(az=[round(a, 2) for a in _AZ_E], el=[-10, -20, -5, 15, 20], pw=list(_PW_E), drift=0.0, el_drift=None,
              T=400, K_track=3, K_use=3, th0_idx=_TOP_E),
    'F': dict(az=[-35, 0, 30], el=[10, -5, 15], pw=[1.0, 1.0, 1.0], drift=0.0, el_drift=None, T=300, K_track=3, K_use=3),
    'G': dict(az=[-40, -15, 5, 20, 35], el=[10, -5, 15, -10, 5], pw=[1.0] * 5, drift=0.0, el_drift=None, T=300, K_track=5, K_use=5),
    'H': dict(az=[-30, -10, 15, 35], el=[15, 10, -10, -5], pw=[1.0, 0.8, 1.0, 0.7], drift=0.0, el_drift=None, T=300, K_track=4, K_use=4),
    'I': dict(az=[-30, -10, 15, 35], el=[15, 10, -10, 20], pw=[1.0, 0.7, 0.8, 0.6], drift=0.0, el_drift=None, T=400, K_track=4, K_use=4),
}
INT_UPA = [(45.0, 0.0), (-10.0, -25.0)]        # second interferer shares the azimuth of a cluster in C and H


def make_channel(sc):
    s = UPA_SCEN[sc]
    th = [enc_deg(a, e) for a, e in zip(s['az'], s['el'])]
    life = []
    for ev in s.get('life', []):
        if ev[0] == 'birth':
            life.append({'type': 'birth', 't': ev[1], 'theta': enc_deg(ev[2], ev[3]), 'power': ev[4]})
        else:
            life.append({'type': 'death', 't': ev[1], 'idx': ev[2]})
    ch = chn.MultiClusterChannel(M_UPA, theta_k_deg=th, power_k=list(s['pw']),
                                theta_int_deg=[enc_deg(a, e) for a, e in INT_UPA], drift_rate_deg_per_snap=s['drift'],
                                lifecycle=life)
    return ch, th


def apply_el_drift(ch, sc):
    """Elevation drift for scenario C (one DEL step per snapshot per cluster, sign per cluster)."""
    s = UPA_SCEN[sc]
    if s.get('el_drift'):
        for i, r in enumerate(s['el_drift']):
            if i < len(ch.theta_k):
                ch.theta_k[i] += S_ENC * int(np.round(np.deg2rad(r) / DEL))
