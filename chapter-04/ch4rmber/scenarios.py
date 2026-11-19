"""
ch4rmber.scenarios -- simulation settings of Section 4.7.1 and the ordered list of experiments.

Settings (defaults in ch4rmber.algorithms):
  * half-wavelength ULA, M = 32 unless stated (8, 16, 32, 64 in the mismatch sweep; 16 to 128 in the runtime test)
  * BPSK desired source at 0 degrees, SNR 5 dB; interferers at 45 and -60 degrees, INR 10 dB each; noise power 1
  * K = 150 snapshots per block (K = 1000 in Figure 4.7(c), (d))
  * multipath: small spread, rays at +4 and -3 degrees (amplitude 0.4 each); large spread, rays at -25 and +35 degrees
    (amplitudes 0.5 and 0.3); ray phases fixed over the block, SINR scored with the realized effective steering vector
  * guided R-MBER: L = 25, alpha = 0.97, delta = 1e-3 tr(R)/M, 361 directions, Delta = 6 degrees widened to 14,
    Z = 80, mu = 0.1, step norm at most 1, kappa = 3, c = 0.5
  * constraint-based R-MBER: grid (4.15), nu = 0.5 nu_0, kappa = 3, at most 60 Dykstra sweeps, tolerance 1e-10
  * WC-MVDR/EigDec: zeta = 3.8 sqrt(M/16), exponential weighting 0.99, loading 1e-3 tr(R)/M
  * INCM-REC: sector +-8 degrees, 200 directions; SPSS [14]: sector th_p +- 6 degrees
  * TA-MMSE: loading sigma2, every symbol of the block or a preamble of 30 pilots
  * a run fails when its SINR is more than 3 dB below the optimum on the same block
  * fixed seeds: point-source blocks np.random.seed(1000 + r); block-fading blocks default_rng(1000 + r);
    BER blocks default_rng(6000 + r) (mismatch) and (5000 + r) (element SNR); random interferer directions
    default_rng(5000 + r); imperfect reference default_rng(31000 + r); local scattering 41000 + r and 51000 + r

Every experiment is a module of ch4rmber.experiments run as `python -m ch4rmber.experiments.<module> <stage>` in the data
folder, one process per stage, two stages at a time; each stage prints its numbers and writes res_<name>.json.
GROUPS lists the stages in an order that respects the files one stage reads from another.
"""
import os
import subprocess
import sys
import time

QUICK = os.environ.get('RAB_QUICK') == '1'


def mc(n):
    """Monte Carlo count: n, or at most 2 in the quick smoke test."""
    return min(n, 2) if QUICK else n


# (module, stage) groups; stages inside a group are independent of each other
GROUPS = [
    [('main_experiments', 'sweep8'), ('main_experiments', 'sweep16')],
    [('main_experiments', 'sweep32'), ('main_experiments', 'sweep64')],
    [('trained_baselines', 'cases'), ('trained_baselines', 'traj')],
    [('trained_baselines', 'inr'), ('trained_baselines', 'inr1000')],
    [('ber_baselines', 'mm'), ('ber_baselines', 'snr')],
    [('main_experiments', 'conv'), ('main_experiments', 'inr1000')],
    [('main_experiments', 'snr'), ('main_experiments', 'near')],
    [('main_experiments', 'berb'), ('main_experiments', 'inrx')],
    [('ber_guided', 'ber'), ('guided_experiments', 'mp')],
    [('local_scattering', 'ci'), ('local_scattering', 'wc')],
    [('guided_experiments', 'near'), ('guided_experiments', 'inr')],
    [('guided_experiments', 'nint'), ('full_array_baselines', 'mp')],
    [('full_array_baselines', 'nint'), ('full_array_baselines', 'snr')],
    [('no_constraints', 'all'), ('blind_no_halfspaces', 'all')],
    [('phase_recovery', 'all'), ('floor_and_heldout', 'all')],
    [('text_checks', 'floor_c'), ('text_checks', 'dykstra')],
    [('text_checks', 'distances'), ('text_checks', 'incm_sets')],
    [('text_checks', 'radius'), ('text_checks', 'ta_loading')],
    [('text_checks', 'wc_solver'), ('text_checks', 'spss_sinc')],
    [('main_experiments', 'ta30'), ('imperfect_reference', 'contam')],
    [('imperfect_reference', 'onset'), ('imperfect_reference', 'labels')],
    [('imperfect_reference', 'drift'), ('imperfect_reference', 'slip')],
    [('main_experiments', 'spss'), ('local_scattering', 'dir')],
    [('local_scattering', 'amp'), ('local_scattering', 'off')],
]
TIMING = [('main_experiments', 'time')]     # runs alone, only when requested (Figure 4.1 uses the recorded run)

CHAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def run_groups(datadir, groups=GROUPS, jobs=2, quick=QUICK, log=print):
    """Run the stages of `groups` in `datadir`, `jobs` processes at a time; stop at the first failure."""
    os.makedirs(os.path.join(datadir, 'logs'), exist_ok=True)
    env = dict(os.environ, PYTHONPATH=CHAPTER_DIR + os.pathsep + os.environ.get('PYTHONPATH', ''),
               RAB_QUICK='1' if quick else '0', MPLBACKEND='Agg')
    t_all = time.time()
    for group in groups:
        t0 = time.time(); pending = list(group); running = []; done = {}
        while pending or running:
            while pending and len(running) < max(1, jobs):
                mod, stage = pending.pop(0)
                f = open(os.path.join(datadir, 'logs', f'{mod}_{stage}.txt'), 'w')
                p = subprocess.Popen([sys.executable, '-m', f'ch4rmber.experiments.{mod}', stage], cwd=datadir, env=env,
                                     stdout=f, stderr=subprocess.STDOUT, text=True)
                running.append(((mod, stage), p, f))
            for item in list(running):
                key, p, f = item
                if p.poll() is not None:
                    f.close(); running.remove(item); done[key] = p.returncode
            time.sleep(0.5)
        log(' + '.join(f'{m} {s}' for m, s in group) + f': {time.time() - t0:.0f} s')
        bad = [k for k, rc in done.items() if rc != 0]
        if bad:
            raise RuntimeError('failed: ' + ', '.join(f'{m} {s}' for m, s in bad) + f' (see {datadir}/logs)')
    log(f'all experiments finished in {(time.time() - t_all) / 60:.1f} min')


def log_text(datadir, mod, stage):
    """Printed output of one stage."""
    p = os.path.join(datadir, 'logs', f'{mod}_{stage}.txt')
    return open(p).read() if os.path.exists(p) else ''


# stages each figure needs (scripts/generate_figures.py --only)
NEEDS = {
    '4_1': [],
    '4_2': [('main_experiments', f'sweep{m}') for m in (8, 16, 32, 64)] + [('main_experiments', 'spss')],
    '4_3': [('trained_baselines', 'cases'), ('guided_experiments', 'mp'), ('full_array_baselines', 'mp'), ('main_experiments', 'spss')],
    '4_4': [('trained_baselines', 'traj'), ('main_experiments', 'conv')],
    '4_5': [('ber_baselines', 'mm'), ('ber_guided', 'ber'), ('main_experiments', 'berb'), ('main_experiments', 'spss')],
    '4_6': [('ber_baselines', 'snr'), ('ber_guided', 'ber'), ('main_experiments', 'berb'), ('main_experiments', 'spss')],
    '4_7': [('guided_experiments', 'inr'), ('main_experiments', 'inr1000'), ('main_experiments', 'inrx'), ('trained_baselines', 'inr'),
            ('trained_baselines', 'inr1000'), ('main_experiments', 'spss')],
    '4_8': [('full_array_baselines', 'snr'), ('main_experiments', 'snr'), ('full_array_baselines', 'nint'), ('guided_experiments', 'nint'),
            ('main_experiments', 'spss')],
    '4_9': [('guided_experiments', 'near'), ('main_experiments', 'near'), ('main_experiments', 'spss')],
    '4_10': [('imperfect_reference', s) for s in ('contam', 'onset', 'labels', 'drift', 'slip')] + [('main_experiments', 'spss')],
    '4_11': [('local_scattering', 'amp'), ('local_scattering', 'off')],
}


def groups_for(figs):
    """The stages of GROUPS that the figures `figs` (keys of NEEDS) need, in GROUPS order."""
    want = {t for f in figs for t in NEEDS[f]}
    return [g2 for g2 in ([t for t in g if t in want] for g in GROUPS) if g2]
