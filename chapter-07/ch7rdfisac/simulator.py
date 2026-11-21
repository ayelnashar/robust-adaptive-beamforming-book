"""Scene generator for the fixed geometry of Scenarios A-E (Section 7.7.1): steering vectors (7.1) and
snapshots (7.3) with the clustered uplink channel (7.2), monostatic target echoes, clutter and interference."""
import numpy as np


def steer_upa(Mx, My, theta, phi):
    """UPA steering vector. theta=azimuth (rad), phi=elevation (rad).
    Half-wavelength spacing. Returns M = Mx*My complex vector."""
    nx = np.arange(Mx)
    ny = np.arange(My)
    ax = np.exp(1j*np.pi*nx*np.sin(theta)*np.cos(phi))
    ay = np.exp(1j*np.pi*ny*np.sin(phi))
    return np.kron(ay, ax)



class ISACEnv:
    """Fixed geometry of Scenarios A-E (Section 7.7.1): 8 x 4 UPA, users at (20, 5) and (-30, 8) deg, targets at
    (-5, 0) and (40, -3) deg, six static clutter components of power 0.04 and two 8 dB co-channel interferers."""
    def __init__(self, Mx=8, My=4, seed=None):
        self.Mx, self.My = Mx, My
        self.M = Mx*My
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        # Users: nominal AOI (theta, phi) in radians
        self.users_nom = [(np.deg2rad(20),  np.deg2rad(5)),
                          (np.deg2rad(-30), np.deg2rad(8))]
        self.user_snr_db = [8.0, 8.0]
        # Targets: nominal AOI and amplitude (path loss and reflection)
        self.targets_nom = [(np.deg2rad(-5),  np.deg2rad(0)),
                            (np.deg2rad(40),  np.deg2rad(-3))]
        self.target_beta = [0.8, 0.7]  # path-loss × RCS
        # Clutter: Q_c = 6 static scatterers in out-of-main-area
        self.Q_c = 6
        # Deterministic clutter placement so runs are reproducible
        clutter_az = np.deg2rad(np.array([-55, -40, -15, 5, 25, 65]))
        clutter_el = np.deg2rad(np.array([-10, 8, 2, -5, 6, -8]))
        self.clutter_ang = list(zip(clutter_az, clutter_el))
        self.clutter_power = 0.04   # relative to thermal
        # Co-channel interferers (cellular CCI), away from the targets and users
        self.interf_ang = [(np.deg2rad(62),  np.deg2rad(2)),
                           (np.deg2rad(-65), np.deg2rad(1))]
        self.interf_inr_db = [8.0, 8.0]
        # Thermal noise variance at each element
        self.sigma2_n = 1.0

    # --- per-realization draw of true channels/targets given uncertainty ---
    def draw_true(self, sigma_c_deg, sigma_s_deg, small_spread_deg=1.0):
        """Draw true user channels and target steerings.
        sigma_c_deg : CSI angular mismatch std (deg)  -> user AOI perturbation
        sigma_s_deg : target-location mismatch std (deg)
        Convention: user channel = true LoS steering vector + optional small-spread
        multipath, both unit-modulus-per-element (||h||^2 = M). Signal power is
        carried separately in `user_ps[u]` = 10^(SNR/10).
        """
        h_list = []            # effective channels normalized to squared norm M
        a_true_users = []      # TRUE LoS steering (dominant)
        user_ps = []
        a_nom_users = []
        small_sig = np.deg2rad(small_spread_deg)
        for (th_u, ph_u), snr_db in zip(self.users_nom, self.user_snr_db):
            dth = np.deg2rad(self.rng.normal(0, sigma_c_deg))
            dph = np.deg2rad(self.rng.normal(0, sigma_c_deg*0.5))
            th_true, ph_true = th_u + dth, ph_u + dph
            a_LoS = steer_upa(self.Mx, self.My, th_true, ph_true)
            # Rician composite: strong LoS + 2 diffuse paths within small spread
            K_rice = 10.0   # Rician K-factor
            diff = np.zeros_like(a_LoS)
            for _ in range(2):
                dth_r = self.rng.normal(0, small_sig)
                dph_r = self.rng.normal(0, small_sig*0.5)
                a_r = steer_upa(self.Mx, self.My, th_true+dth_r, ph_true+dph_r)
                g_r = (self.rng.normal()+1j*self.rng.normal())/np.sqrt(2)
                diff = diff + g_r * a_r
            h = np.sqrt(K_rice/(K_rice+1)) * a_LoS + np.sqrt(1/(K_rice+1)) * diff
            # Normalize so ||h||^2 = M (same scale as steering)
            h = h * np.sqrt(self.M / max(np.real(h.conj() @ h), 1e-12))
            h_list.append(h)
            a_true_users.append(a_LoS)
            user_ps.append(10**(snr_db/10.0))
            a_nom_users.append(steer_upa(self.Mx, self.My, th_u, ph_u))
        # Targets
        a_true_targets = []
        a_nom_targets = []
        for (th_t, ph_t) in self.targets_nom:
            dth = np.deg2rad(self.rng.normal(0, sigma_s_deg))
            dph = np.deg2rad(self.rng.normal(0, sigma_s_deg*0.5))
            a_true_targets.append(steer_upa(self.Mx, self.My, th_t+dth, ph_t+dph))
            a_nom_targets.append(steer_upa(self.Mx, self.My, th_t, ph_t))
        return dict(h_list=h_list, a_true_users=a_true_users,
                    a_true_targets=a_true_targets, a_nom_users=a_nom_users,
                    a_nom_targets=a_nom_targets, user_ps=user_ps)

    def interference_covariance(self):
        """Stationary covariance from CCI interferers and clutter + thermal noise."""
        R = self.sigma2_n * np.eye(self.M, dtype=complex)
        # CCI
        for (th, ph), inr_db in zip(self.interf_ang, self.interf_inr_db):
            a = steer_upa(self.Mx, self.My, th, ph)
            R += (10**(inr_db/10.0) * self.sigma2_n) * np.outer(a, a.conj())
        # Clutter
        for (th, ph) in self.clutter_ang:
            a = steer_upa(self.Mx, self.My, th, ph)
            R += self.clutter_power * self.sigma2_n * np.outer(a, a.conj())
        return R

    def generate_snapshots(self, K, sigma_c_deg, sigma_s_deg, small_spread_deg=1.0):
        """Generate K composite observation snapshots for the adaptive filter."""
        tr = self.draw_true(sigma_c_deg, sigma_s_deg, small_spread_deg)
        h_list = tr['h_list']
        a_true_targets = tr['a_true_targets']
        user_ps = tr['user_ps']
        U = len(h_list); T = len(a_true_targets)
        X = np.zeros((self.M, K), dtype=complex)
        interfs = [(steer_upa(self.Mx, self.My, th, ph), 10**(inr/10)*self.sigma2_n)
                   for (th, ph), inr in zip(self.interf_ang, self.interf_inr_db)]
        clutter = [(steer_upa(self.Mx, self.My, th, ph), self.clutter_power*self.sigma2_n)
                   for (th, ph) in self.clutter_ang]
        for k in range(K):
            xk = np.zeros(self.M, dtype=complex)
            # Users: s_u is unit-variance; signal power carried by user_ps
            for h, ps in zip(h_list, user_ps):
                s = (self.rng.normal()+1j*self.rng.normal())/np.sqrt(2)
                xk += np.sqrt(ps) * h * s
            # Target echoes
            for a_t, b_t in zip(a_true_targets, self.target_beta):
                s = (self.rng.normal()+1j*self.rng.normal())/np.sqrt(2)
                xk += b_t * a_t * s
            # CCI
            for a_i, p_i in interfs:
                s = (self.rng.normal()+1j*self.rng.normal())/np.sqrt(2)
                xk += np.sqrt(p_i) * a_i * s
            # Clutter
            for a_c, p_c in clutter:
                s = (self.rng.normal()+1j*self.rng.normal())/np.sqrt(2)
                xk += np.sqrt(p_c) * a_c * s
            # Thermal noise
            xk += (self.rng.normal(size=self.M)+1j*self.rng.normal(size=self.M))/np.sqrt(2) * np.sqrt(self.sigma2_n)
            X[:, k] = xk
        truth = dict(h_list=h_list, a_true_users=tr['a_true_users'],
                     a_true_targets=a_true_targets, a_nom_users=tr['a_nom_users'],
                     a_nom_targets=tr['a_nom_targets'], user_ps=user_ps,
                     beta=self.target_beta, R_cci=self.interference_covariance())
        return X, truth
