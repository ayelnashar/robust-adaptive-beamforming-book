"""Scenario C (linear array) on the separate stream of Figure 5.7 (seed 42 once, scenarios in order, the 15 runs of C):
mean absolute direction error of the constraints and expected SINR per snapshot for MPR-RAB, for MPR-RAB without the
delay compensation, and for MPR-RAB with the directions refined every 30 snapshots on the scan grid without delay
compensation.  Writes DATA/tracking_c.json:  python -m ch5mprrab.experiments.tracking_c"""
import os, json
import numpy as np
from . import _common  # noqa: F401
from .. import algorithms as alg
from .. import scenarios as MS
sel = MS.load_selection('ula')
kap = MS.kappa_of(sel['MPR-RAB'])
CFG = {'pred': dict(), 'nopred': dict(predict=False), 'every30': dict(predict=False, polish=False, t_ref=30)}
states = MS.stream_states(42, list('ABC'))
np.random.set_state(states['C'])
err, sgn, sinr = {k: [] for k in CFG}, {k: [] for k in CFG}, {k: [] for k in CFG}
for r in range(MS.cfg()['STREAM_NR']['C']):
    d = MS.make_data('C', 5, 10, 0, 0, reseed=False)
    for key, kw in CFG.items():
        h, log = alg.run_mpr_rab(d['X'], d['a0'], list(d['th0']), K_track=2, K_use=2, use_tracking=True, use_lifecycle=False,
                                   kappa=kap, targets='ms', **kw)
        e = np.array([np.rad2deg(np.array(sorted(log['theta_hat'][k])) - np.array(sorted(d['thl'][k]))) for k in range(d['T'])])
        err[key].append(np.mean(np.abs(e), axis=1))
        sgn[key].append(np.mean(e, axis=1))
        sinr[key].append(MS.sinr_trace(h, d))
out = {'mean_abs_err_100_400': {k: float(np.mean(np.array(v)[:, 100:])) for k, v in err.items()},
       'mean_signed_err_100_400': {k: float(np.mean(np.array(v)[:, 100:])) for k, v in sgn.items()},
       'sinr_mean_100_400': {k: float(np.mean(np.mean(np.array(v), axis=0)[100:])) for k, v in sinr.items()},
       'largest_step_after_60': {k: float(np.max(np.abs(np.diff(np.mean(np.array(v), axis=0)[60:])))) for k, v in sinr.items()},
       'sinr_trace': {k: np.mean(np.array(v), axis=0).tolist() for k, v in sinr.items()},
       'kappa': kap}
os.makedirs(MS.DATA, exist_ok=True)
json.dump(out, open(os.path.join(MS.DATA, 'tracking_c.json'), 'w'))
print({k: v for k, v in out.items() if k != 'sinr_trace'})
