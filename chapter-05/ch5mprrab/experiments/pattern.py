"""Beam pattern of the final MPR-RAB weight, planar array, scenario H, operating point, one run (seed 7), for
Figure 5.5:  python -m ch5mprrab.experiments.pattern
Writes DATA/pattern_H.npz (normalized gain over azimuth -90..90 deg and elevation -60..60 deg, cluster and interferer
directions) and DATA/pattern_H.json (normalized gains towards the clusters and the interferers)."""
import os
import json
import numpy as np
from . import _common  # noqa: F401
from .. import channel as chn
from .. import algorithms as alg
from .. import scenarios as MS


def main():
    MS.use_array('upa')
    U = MS.U
    sel = MS.load_selection('upa')
    d = MS.make_data('H', 5, 10, 0, 7)
    c = d['c']
    kap = MS.kappa_of(sel['MPR-RAB'])
    h, log = alg.run_mpr_rab(d['X'], d['a0'], list(d['th0']), K_track=c['K_track'], K_use=c['K_use'], use_tracking=True,
                            use_lifecycle=False, kappa=kap, targets='ms')
    w = h[-1]
    az = np.deg2rad(np.linspace(-90, 90, 361)); el = np.deg2rad(np.linspace(-60, 60, 241))
    G = np.zeros((len(el), len(az)))
    for i, e in enumerate(el):
        ay = np.exp(1j * np.pi * np.arange(U.NY) * np.sin(e))
        A = np.array([np.kron(ay, np.exp(1j * np.pi * np.arange(U.NX) * np.sin(a) * np.cos(e))) for a in az]).T
        G[i] = np.abs(w.conj() @ A) ** 2
    G = 10 * np.log10(G / G.max() + 1e-12)
    th_true, th_int = d['thl'][-1], d['ch'].theta_int
    clusters = np.array([U.az_el_deg(t_) for t_ in th_true])
    interferers = np.array([U.az_el_deg(t_) for t_ in th_int])
    os.makedirs(MS.DATA, exist_ok=True)
    np.savez(os.path.join(MS.DATA, 'pattern_H.npz'), G=G, az=np.rad2deg(az), el=np.rad2deg(el), clusters=clusters,
             interferers=interferers)
    nrm = lambda t_: float(10 * np.log10(np.abs(w.conj() @ chn.steer(64, t_)) ** 2 / (np.linalg.norm(w) ** 2 * 64) + 1e-15))
    gains = {'cluster (%.0f, %.0f)' % U.az_el_deg(t_): nrm(t_) for t_ in th_true}
    gains.update({'interferer (%.0f, %.0f)' % U.az_el_deg(t_): nrm(t_) for t_ in th_int})
    json.dump({'H_gains_dB': gains, 'kappa': kap}, open(os.path.join(MS.DATA, 'pattern_H.json'), 'w'), indent=1)
    print(json.dumps(gains, indent=1))


if __name__ == '__main__':
    main()
