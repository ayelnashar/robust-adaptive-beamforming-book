"""Receive-side dual-function model (Section 7.2): user SINR (7.5), target SNR (7.6) and the general scene
generator used for random geometries, array size, load, sweeps and the target crossing (Sections 7.7.4-7.7.6).

One receive spatial filter w at the base station serves U uplink users, each on its own orthogonal
time-frequency resource, and T monostatic radar echoes. Radar probing and the uplink allocations overlap in
a fraction RHO_O of the resources, in both directions.

  user u : SINR = p_u |w^H h_u|^2 / (w^H R_c w),   R_c = R_0 + RHO_O sum_t |beta_t|^2 a_t a_t^H
  target t: SNR = N_f |beta_t|^2 |w^H a_t|^2 / (w^H R_s w),   R_s = R_0 + RHO_O sum_u p_u h_u h_u^H
  R_0 = thermal noise + external co-channel interference + clutter.
Other users are not interference for user u because their uplinks occupy other resources. Both metrics are
invariant to the scale of w. The training snapshots contain every source, as the covariance accumulated over
all resources does.
"""
import numpy as np

RHO_O = 0.3
N_FAST = 64


def steer(Mx, My, az, el):
    az = np.deg2rad(az); el = np.deg2rad(el)
    ax = np.exp(1j*np.pi*np.arange(Mx)*np.sin(az)*np.cos(el)); ay = np.exp(1j*np.pi*np.arange(My)*np.sin(el))
    return np.kron(ay, ax)


def evaluate_rx(w, tr, n_fast=N_FAST, rho=RHO_O):
    """[SINR_u1..SINR_uU, SNR_t1..SNR_tT] in dB for the receive-side model."""
    R0 = tr['R_cci']
    Rc = R0 + rho*sum(abs(b)**2*np.outer(a, a.conj()) for a, b in zip(tr['a_true_targets'], tr['beta']))
    Rs = R0 + rho*sum(p*np.outer(h, h.conj()) for h, p in zip(tr['h_list'], tr['user_ps']))
    q = lambda R: max(float(np.vdot(w, R@w).real), 1e-30)
    out = [10*np.log10(max(p*abs(np.vdot(w, h))**2/q(Rc), 1e-12)) for h, p in zip(tr['h_list'], tr['user_ps'])]
    out += [10*np.log10(max(n_fast*abs(b)**2*abs(np.vdot(w, a))**2/q(Rs), 1e-12)) for a, b in zip(tr['a_true_targets'], tr['beta'])]
    return out


class Scene:
    """General scene: Mx x My UPA, U users, T targets, static clutter, weak far co-channel interferers, and
    optional strong interferers placed 8-14 deg in azimuth (and up to 3 deg in elevation) from served directions (static or drifting)."""

    def __init__(self, Mx=8, My=4, U=2, T=2, seed=0, random_geometry=True, users=None, targets=None,
                 strong=0, inr_strong=25.0, sep_range=(8.0, 14.0), drift=0.0, strong_angles=None):
        self.Mx, self.My, self.M = Mx, My, Mx*My
        self.rng = np.random.default_rng(seed)
        r = self.rng
        if random_geometry:     # azimuth U(-60, 60) deg with 12 deg minimum spacing, elevation U(-5, 10) deg
            pts = []
            while len(pts) < U+T:
                pts, tries = [], 0
                while len(pts) < U+T and tries < 500:
                    tries += 1; c = (float(r.uniform(-60, 60)), float(r.uniform(-5, 10)))
                    if all(abs(c[0]-p[0]) >= 12 for p in pts): pts.append(c)
            self.users, self.targets = pts[:U], pts[U:]
        else:
            self.users, self.targets = users, targets
        self.beta = [0.8, 0.7, 0.75, 0.65][:T] if T <= 4 else [0.75]*T
        self.user_snr_db = 8.0
        self.clutter = [(-55, -10), (-40, 8), (-15, 2), (5, -5), (25, 6), (65, -8)]; self.clutter_p = 0.04
        served = self.users + self.targets
        self.cci = [c for c in [(62, 2), (-65, 1)] if all(abs(c[0]-p[0]) >= 15 for p in served)]
        self.cci_p = 10**0.8
        self.strong0 = []
        if strong_angles is not None:
            self.strong0 = list(strong_angles)
        else:
            tries = 0
            while len(self.strong0) < strong and tries < 2000:
                tries += 1
                p = served[int(r.integers(len(served)))]
                c = (p[0] + float(r.choice([-1, 1]))*float(r.uniform(*sep_range)), p[1] + float(r.uniform(-3, 3)))
                if abs(c[0]) <= 70 and all(abs(c[0]-q[0]) >= sep_range[0] for q in served) and \
                        all(abs(c[0]-q[0]) >= 4 for q in self.strong0):
                    self.strong0.append(c)
        self.drift = drift          # each strong interferer moves this many degrees toward its nearest served direction
        self.strong_p = 10**(inr_strong/10)

    def strong_at(self, k, K):
        if not self.drift: return self.strong0
        out = []; f = k/max(K-1, 1); served = self.users + self.targets
        for a, e in self.strong0:
            near = min(served, key=lambda p: abs(p[0]-a)); s = np.sign(near[0]-a)
            out.append((a + s*self.drift*f, e))
        return out

    def nominal_columns(self, offsets=(0, -2, 2)):
        st = lambda a, e: steer(self.Mx, self.My, a, e)
        C = np.column_stack([st(a+o, e) for a, e in self.users for o in offsets])
        S = np.column_stack([st(a+o, e) for a, e in self.targets for o in offsets])
        return C, S

    def generate(self, K, sigma_c, sigma_s):
        r = self.rng; Mx, My, M = self.Mx, self.My, self.M; st = lambda a, e: steer(Mx, My, a, e)
        cn = lambda: (r.normal()+1j*r.normal())/np.sqrt(2)
        h_list, a_users, a_tg = [], [], []
        for a, e in self.users:
            at, et = a + r.normal(0, sigma_c), e + r.normal(0, sigma_c/2)
            aLoS = st(at, et); diff = np.zeros(M, complex)
            for _ in range(2): diff += cn()*st(at + r.normal(0, 1.0), et + r.normal(0, .5))
            h = np.sqrt(10/11)*aLoS + np.sqrt(1/11)*diff; h *= np.sqrt(M/np.vdot(h, h).real)
            h_list.append(h); a_users.append(aLoS)
        for a, e in self.targets:
            a_tg.append(st(a + r.normal(0, sigma_s), e + r.normal(0, sigma_s/2)))
        ps = [10**(self.user_snr_db/10)]*len(self.users)
        R_static = np.eye(M, dtype=complex)
        for a, e in self.cci: v = st(a, e); R_static += self.cci_p*np.outer(v, v.conj())
        for a, e in self.clutter: v = st(a, e); R_static += self.clutter_p*np.outer(v, v.conj())
        X = np.zeros((M, K), complex); trs = []
        Vc = [st(a, e) for a, e in self.cci]; Vl = [st(a, e) for a, e in self.clutter]
        for k in range(K):
            A = self.strong_at(k, K); Vs = [st(a, e) for a, e in A]
            x = np.zeros(M, complex)
            for h, p in zip(h_list, ps): x += np.sqrt(p)*h*cn()
            for a, b in zip(a_tg, self.beta): x += b*a*cn()
            for v in Vs: x += np.sqrt(self.strong_p)*v*cn()
            for v in Vc: x += np.sqrt(self.cci_p)*v*cn()
            for v in Vl: x += np.sqrt(self.clutter_p)*v*cn()
            x += (r.normal(size=M)+1j*r.normal(size=M))/np.sqrt(2); X[:, k] = x
            if self.drift or k == 0 or k == K-1:
                R0 = R_static + sum(self.strong_p*np.outer(v, v.conj()) for v in Vs)
            trs.append(dict(h_list=h_list, a_true_users=a_users, a_true_targets=a_tg, user_ps=ps, beta=self.beta, R_cci=R0))
        return X, trs
