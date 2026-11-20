"""
ch5mprrab.channel -- signal model of Chapter 5: steering vectors, the multi-cluster channel, the expected SINR and the
oracle beamformer (Section 5.2).

The received snapshot is

    x(t) = (sum_k alpha_k(t) a(theta_k(t))) s(t) + sum_j u_j(t) a(theta_j^int) + n(t),

with independent complex Gaussian cluster gains alpha_k of power P_k, a complex Gaussian symbol s of power P_S,
interferers u_j of power P_I and white noise of power sigma^2 per element.  A direction is carried as one scalar in
radians; on the planar array it encodes azimuth and elevation (see ch5mprrab.planar, which routes `steer` and
`M_DEFAULT` of this module to the 8 x 8 array).

Module-level PS and PIV are the signal and interference powers of the current data set; they are set by
ch5mprrab.scenarios.make_data for every condition.
"""
import numpy as np
from scipy.linalg import eigh

# ----------------------------------------------------------------------------- global settings
M_DEFAULT = 16          # elements of the uniform linear array
SIGMA2 = 1.0            # noise power per element
SNR_DB = 5              # signal power per unit-power cluster (dB above the noise)
INR_DB = 10             # power of each interferer (dB above the noise)
PS = SIGMA2 * 10 ** (SNR_DB / 10)
PIV = SIGMA2 * 10 ** (INR_DB / 10)
ETA = 0.97              # forgetting factor of the exponentially weighted covariance
BETA = 0.99             # smoothing factor of the covariance used by Par-K-RCB
EPS = 1e-15

THETA_INTS_DEG = [45, -60]   # interferers of the linear array (deg)


def steer(M, th):
    """ULA steering vector, half-wavelength spacing; th in radians."""
    return np.exp(1j * 2 * np.pi * 0.5 * np.arange(M) * np.sin(th))


def sinr_db_cluster(w, theta_k_rad, power_k, R_ipn, ps):
    """
    Expected output SINR for a multi-cluster channel (5.4):

        E[|w^H a_eff|^2] = sum_k P_k |w^H a(theta_k)|^2,

    the output power averaged over the independent cluster phases, divided by w^H R_in w.
    """
    sig = 0.0
    M = len(w)
    for th, p in zip(theta_k_rad, power_k):
        a_k = steer(M, th)
        sig += p * np.abs(w.conj() @ a_k) ** 2
    sig *= ps
    den = np.real(w.conj() @ R_ipn @ w)
    if den < 1e-30:
        return -30.0
    return 10 * np.log10(max(sig / den, 1e-10))


# ----------------------------------------------------------------------------- multi-cluster channel
class MultiClusterChannel:
    """
    K-cluster desired signal with optional cluster dynamics.

    Parameters
    ----------
    M : int -- array elements
    theta_k_deg : array-like -- cluster directions in degrees (encoded directions on the planar array)
    power_k : array-like -- cluster powers, same length as theta_k_deg
    theta_int_deg : list -- interferer directions in degrees
    drift_rate_deg_per_snap : float -- angular drift per snapshot (scenario C)
    lifecycle : list of dicts -- birth and death events (scenario D), e.g.
        [{'type': 'birth', 't': 150, 'theta': -30, 'power': 0.9}, {'type': 'death', 't': 280, 'idx': 0}]
    """
    def __init__(self, M, theta_k_deg, power_k,
                 theta_int_deg=None, drift_rate_deg_per_snap=0.0,
                 lifecycle=None):
        self.M = M
        self.theta_k = list(np.deg2rad(theta_k_deg))
        self.power_k = list(power_k)
        self.theta_int = [np.deg2rad(t) for t in (theta_int_deg or THETA_INTS_DEG)]
        self.drift = np.deg2rad(drift_rate_deg_per_snap)
        self.lifecycle = lifecycle or []

        # cluster directions at every snapshot
        self.log_theta = []

    def _apply_lifecycle(self, t):
        for ev in self.lifecycle:
            if ev.get('_applied'): continue
            if ev['t'] == t:
                if ev['type'] == 'birth':
                    self.theta_k.append(np.deg2rad(ev['theta']))
                    self.power_k.append(ev['power'])
                elif ev['type'] == 'death':
                    idx = ev['idx']
                    if 0 <= idx < len(self.theta_k):
                        del self.theta_k[idx]
                        del self.power_k[idx]
                ev['_applied'] = True

    def snapshot(self, k_snap):
        """Generate one snapshot x (M,) and return (x, a_eff, theta_k_now)."""
        # lifecycle events fire at integer time steps
        self._apply_lifecycle(k_snap)

        # drift
        if self.drift != 0:
            self.theta_k = [th + self.drift for th in self.theta_k]

        # signal symbol (complex Gaussian, power PS)
        s = np.sqrt(PS / 2) * (np.random.randn() + 1j * np.random.randn())

        # per-cluster complex gains alpha_k with power power_k
        a_eff = np.zeros(self.M, dtype=complex)
        for th, p in zip(self.theta_k, self.power_k):
            alpha = np.sqrt(p / 2) * (np.random.randn() + 1j * np.random.randn())
            a_eff = a_eff + alpha * steer(self.M, th)

        sig = a_eff * s

        # interferers (uncorrelated complex Gaussian)
        for ti in self.theta_int:
            u = np.sqrt(PIV / 2) * (np.random.randn() + 1j * np.random.randn())
            sig = sig + steer(self.M, ti) * u

        # white noise
        n = np.sqrt(SIGMA2 / 2) * (np.random.randn(self.M) + 1j * np.random.randn(self.M))
        sig = sig + n

        self.log_theta.append(list(self.theta_k))
        return sig, a_eff, list(self.theta_k)

    def true_R_ipn(self):
        """Interference-plus-noise covariance (the desired signal excluded)."""
        R = SIGMA2 * np.eye(self.M, dtype=complex)
        for ti in self.theta_int:
            a = steer(self.M, ti)
            R += PIV * np.outer(a, a.conj())
        return R


def cdl_like_profile(profile='A', K_clusters=None, seed=0):
    """
    (theta_k_deg, power_k) of a synthetic clustered delay-line-like profile with a heavy-tailed power distribution
    (scenario E uses profile 'A' with five clusters, seed 7).
    """
    rng = np.random.RandomState(seed)
    if profile == 'A':  # NLOS-dominated, clusters spread
        K = K_clusters or 10
        theta = rng.uniform(-60, 60, K)
        powers = rng.dirichlet(np.ones(K) * 0.8)  # heavy-tailed
        powers = powers / powers.max()           # normalize peak to 1
    elif profile == 'B':
        K = K_clusters or 8
        theta = rng.uniform(-50, 50, K)
        powers = rng.dirichlet(np.ones(K) * 1.0)
        powers = powers / powers.max()
    elif profile == 'C':  # LOS + NLOS mixed: one strong + several weaker
        K = K_clusters or 6
        theta = np.concatenate([[0.0], rng.uniform(-45, 45, K - 1)])
        pw_nlos = rng.dirichlet(np.ones(K - 1) * 1.0) * 0.5
        powers = np.concatenate([[1.0], pw_nlos])
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return theta.tolist(), powers.tolist()


def run_oracle_mc(R_ipn_true, theta_k_rad, power_k, M):
    """
    Oracle beamformer for the multi-cluster channel (upper bound, Proposition 5.1):

        maximize (w^H H w) / (w^H R_in w),   H = sum_k P_k a(theta_k) a(theta_k)^H,

    the largest generalized eigenvector of (H, R_in) with the true covariance and directions.
    """
    H = np.zeros((M, M), dtype=complex)
    for th, p in zip(theta_k_rad, power_k):
        a = steer(M, th)
        H = H + p * np.outer(a, a.conj())
    H = (H + H.conj().T) / 2
    R = (R_ipn_true + R_ipn_true.conj().T) / 2
    # generalized eigenvalue problem H v = lambda R v, largest lambda
    vals, vecs = eigh(H, R + 1e-9 * np.eye(M))
    w = vecs[:, -1]
    return w
