"""Check the regenerated results against the values quoted in Chapter 6.

    python -m ch6cfrab.verify figures              # data in figures/data
    python -m ch6cfrab.verify figures --docx ch6.docx   # also confirm that each quoted string appears in a .docx

Every check prints OK or !! with the regenerated value and the value quoted in the chapter. The tolerances are half a
unit of the last printed digit, so a check passes only when the regenerated number rounds to the quoted one. Values
from a --quick run use fewer realizations and are not expected to match.
"""
import json
import os
import re
import sys
import zipfile

import numpy as np
from numpy.linalg import norm
from scipy.stats import t as tdist

from . import algorithms as C
from .algorithms import CFConfig

FAILS = []
DOCX = None


def _docx_text(path):
    if not path:
        return None, None
    x = zipfile.ZipFile(path).read('word/document.xml').decode('utf8')
    t = re.sub(r'<(w|m):t(?: [^>]*)?>([^<]*)</\1:t>', r'\2', x)
    t = re.sub(r'<[^>]+>', '', t)
    return path, t.replace('\u2009', ' ').replace('\xa0', ' ')


def _load(datadir):
    def J(s):
        p = os.path.join(datadir, s + '.json')
        return json.load(open(p))['result'] if os.path.exists(p) else None
    R = {'E1': J('fig1'), 'E7': J('fig2'), 'E6': J('fig3'), 'E2': J('fig4'), 'E3': J('fig5'), 'E5': J('fig7'), 'E0': J('fig10')}
    for k, s in (('E4', 'fig6'), ('E10', 'fig8'), ('E8', 'fig9')):
        d = J(s); R[k] = d['rows'] if d else None
    EXP = {}
    for s in ('train', 'paired', 'audits'):
        EXP.update(J(s) or {})
    if J('lbound'):
        EXP['L_bound'] = J('lbound')
    A3 = J('A3') or {}
    RA = {'A1': J('A1'), 'A2': J('A2'), 'A3': A3.get('A3'), 'A3_paired': A3.get('A3_paired'), 'A4': J('A4'), 'A5': J('A5')}
    RB = {'B1': J('B1'), 'B2': J('B2')}
    return R, EXP, RA, RB


def check_figures(RESULTS, EXP):
    """Solver and error-model checks, figure numbers and the paired comparisons, audits and Section 6.7.12."""
    g = lambda t, m, k: RESULTS[t][str(k)][m][0]
    def flag(label, good, detail=''):
        if not good: FAILS.append(label)
        print(f'  [{"OK " if good else "!! "}] {label}' + (f'  ({detail})' if detail else ''))

    print('Exact local solver: feasibility and tightness of every per-AP solve')
    rec = []; _ex = C.exact_wc
    def _spy(R, h, nu):
        w = _ex(R, h, nu)
        if w is not None: rec.append(float(np.real(np.vdot(w, h)) - nu * norm(w) - 1.0))
        return w
    C.exact_wc = _spy
    C.run_snapshot_trace(CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10,
                         eps_override=0.10, methods=('D-CF-RAB',), seed=700, soi_in_cov=True)
    C.exact_wc = _ex
    rec = np.array(rec)
    flag(f'bound level c = {C.RABConfig().nu_c} is below 1', C.RABConfig().nu_c < 1)
    flag('every local solve is feasible and tight', len(rec) > 0 and rec.min() > -1e-9 and np.abs(rec).max() < 1e-6,
         f'{len(rec)} solves, margin range [{rec.min():.1e}, {rec.max():.1e}]')

    print('\nError model: coverage of the declared ball of eq. (6.2)')
    _rng = np.random.default_rng(3); _ins = 0; _tot = 0
    for _ in range(2000):
        _h = _rng.normal(size=8) + 1j * _rng.normal(size=8); _u = _rng.normal(size=8) + 1j * _rng.normal(size=8); _u /= norm(_u)
        for _e in (0.10, 0.50):
            _hp = _h + C._ball_radius(_h, _u, _e) * _u; _tot += 1; _ins += norm(_hp - _h) <= _e * norm(_hp) * (1 + 1e-12)
    flag('every generated error satisfies eq. (6.2)', _ins == _tot, f'{_ins}/{_tot}')
    r16 = [r for r in RESULTS['E4'] if r['L'] == 16][0]
    flag('per-update timing recorded for every timed method', all(r16['calls'][m] > 0 for m in r16['calls']), str(r16['calls']))

    print('\nInternal consistency')
    v = [g('E2', 'D-CF-RAB', x) for x in (0, 1, 4, 16, 64)]
    flag('delay response is monotone', all(v[i] >= v[i+1] for i in range(4)))
    flag('Fig. 6.4 at zero delay agrees with Fig. 6.3 at 10 dB',
         abs(g('E2', 'D-CF-RAB', 0) - g('E6', 'D-CF-RAB', 10)) < 1.0,
         f"{g('E2','D-CF-RAB',0):.2f} vs {g('E6','D-CF-RAB',10):.2f}")
    th = RESULTS['E0']
    flag('bound level lies in (eps*_per-AP, 1)', th['per_star_mean'] < C.RABConfig().nu_c < 1.0,
         f"eps*_per-AP = {th['per_star_mean']:.3f}")


    def ge(m, e, i=0):
        k = next(k for k in RESULTS['E7'] if abs(float(k) - e) < 1e-9); return RESULTS['E7'][k][m][i]
    EPSK = sorted(float(k) for k in RESULTS['E7'])
    def check_numbers(title, items):
        print('\n' + title)
        for label, fn, want, tol in items:
            try:
                got = float(fn())
            except Exception as e:
                flag(label, False, f'not available: {e}'); continue
            flag(label, abs(got - want) <= tol, f'{got:.5g} vs {want} in the chapter')
    check_numbers('Numbers quoted in the chapter: figures', [
        ('Fig. 6.1: D-CF-RAB at L = 4', lambda: g('E1','D-CF-RAB',4), 16.32, 0.006),
        ('Fig. 6.1: D-CF-RAB at L = 16', lambda: g('E1','D-CF-RAB',16), 25.74, 0.006),
        ('Fig. 6.1: largest |MR + CPU minus D-CF-RAB|', lambda: max(abs(g('E1','MR+W',L)-g('E1','D-CF-RAB',L)) for L in (4,8,16)), 0.022, 0.0006),
        ('Fig. 6.1: LP-MMSE + CPU minus D-CF-RAB at L = 4', lambda: g('E1','LP-MMSE+W',4)-g('E1','D-CF-RAB',4), 3.04, 0.006),
        ('Fig. 6.1: LP-MMSE + CPU minus D-CF-RAB at L = 8', lambda: g('E1','LP-MMSE+W',8)-g('E1','D-CF-RAB',8), 0.52, 0.006),
        ('Fig. 6.1: D-CF-RAB minus LP-MMSE + CPU at L = 16', lambda: g('E1','D-CF-RAB',16)-g('E1','LP-MMSE+W',16), 0.58, 0.006),
        ('Fig. 6.1: P-MMSE minus D-CF-RAB at L = 16', lambda: g('E1','P-MMSE',16)-g('E1','D-CF-RAB',16), 2.57, 0.006),
        ('Fig. 6.2: D-CF-RAB at eps = 0.10', lambda: ge('D-CF-RAB',.1), 26.38, 0.006),
        ('Fig. 6.2: largest |D-CF-RAB minus MR + CPU|', lambda: max(abs(ge('D-CF-RAB',e)-ge('MR+W',e)) for e in EPSK), 0.011, 0.0006),
        ('Fig. 6.2: D-CF-RAB minus LP-MMSE at eps = 0.005', lambda: ge('D-CF-RAB',.005)-ge('LP-MMSE',.005), 2.51, 0.006),
        ('Fig. 6.2: D-CF-RAB minus LP-MMSE at eps = 0.50', lambda: ge('D-CF-RAB',.5)-ge('LP-MMSE',.5), 6.28, 0.006),
        ('Fig. 6.2: smallest D-CF-RAB minus local MVDR + CPU', lambda: min(ge('D-CF-RAB',e)-ge('LMVDR+W',e) for e in EPSK), 3.79, 0.006),
        ('Fig. 6.2: largest D-CF-RAB minus local MVDR + CPU', lambda: max(ge('D-CF-RAB',e)-ge('LMVDR+W',e) for e in EPSK), 4.01, 0.006),
        ('Fig. 6.2: D-CF-RAB minus P-MMSE at eps = 0.50', lambda: ge('D-CF-RAB',.5)-ge('P-MMSE',.5), 1.09, 0.006),
        ('Fig. 6.2: P-MMSE above D-CF-RAB at eps = 0.3 (crossing between 0.3 and 0.5)', lambda: float(ge('P-MMSE',.3) > ge('D-CF-RAB',.3)), 1.0, 0.6),
        ('Fig. 6.3: smallest D-CF-RAB minus MR + CPU', lambda: min(g('E6','D-CF-RAB',k)-g('E6','MR+W',k) for k in RESULTS['E6']), -0.002, 0.0006),
        ('Fig. 6.3: largest D-CF-RAB minus MR + CPU', lambda: max(g('E6','D-CF-RAB',k)-g('E6','MR+W',k) for k in RESULTS['E6']), 0.008, 0.0006),
        ('Fig. 6.3: smallest P-MMSE minus D-CF-RAB', lambda: min(g('E6','P-MMSE',k)-g('E6','D-CF-RAB',k) for k in RESULTS['E6']), 2.11, 0.006),
        ('Fig. 6.3: largest P-MMSE minus D-CF-RAB', lambda: max(g('E6','P-MMSE',k)-g('E6','D-CF-RAB',k) for k in RESULTS['E6']), 2.31, 0.006),
        ('Fig. 6.4: D-CF-RAB at tau = 0', lambda: g('E2','D-CF-RAB',0), 27.21, 0.006),
        ('Fig. 6.4: D-CF-RAB at tau = 64', lambda: g('E2','D-CF-RAB',64), 14.07, 0.006),
        ('Fig. 6.4: MR + CPU at tau = 0', lambda: g('E2','MR+W',0), 27.21, 0.006),
        ('Fig. 6.4: MR + CPU at tau = 64', lambda: g('E2','MR+W',64), 14.11, 0.006),
        ('Fig. 6.4: P-MMSE at tau = 0', lambda: g('E2','P-MMSE',0), 29.2, 0.006),
        ('Fig. 6.4: P-MMSE at tau = 64', lambda: g('E2','P-MMSE',64), 13.18, 0.006),
        ('Fig. 6.5: D-CF-RAB, uniform profile', lambda: RESULTS['E3']['uniform']['D-CF-RAB'][0], 25.65, 0.006),
        ('Fig. 6.5: P-MMSE, uniform profile', lambda: RESULTS['E3']['uniform']['P-MMSE'][0], 28.35, 0.006),
        ('Fig. 6.5: D-CF-RAB, tiered profile', lambda: RESULTS['E3']['tiered']['D-CF-RAB'][0], 24.61, 0.006),
        ('Fig. 6.5: P-MMSE, tiered profile', lambda: RESULTS['E3']['tiered']['P-MMSE'][0], 28.42, 0.006),
        ('Fig. 6.5: paired D-CF-RAB minus MR+W, uniform, mean', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['uniform']['D-CF-RAB'][3])-np.array(RESULTS['E3']['uniform']['MR+W'][3]))[0], np.float64(0.007), 0.0006),
        ('Fig. 6.5: paired D-CF-RAB minus MR+W, uniform, lower bound', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['uniform']['D-CF-RAB'][3])-np.array(RESULTS['E3']['uniform']['MR+W'][3]))[1], np.float64(0.002), 0.0006),
        ('Fig. 6.5: paired D-CF-RAB minus MR+W, uniform, upper bound', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['uniform']['D-CF-RAB'][3])-np.array(RESULTS['E3']['uniform']['MR+W'][3]))[2], np.float64(0.013), 0.0006),
        ('Fig. 6.5: paired D-CF-RAB minus MR+W, tiered, mean', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['tiered']['D-CF-RAB'][3])-np.array(RESULTS['E3']['tiered']['MR+W'][3]))[0], np.float64(0.01), 0.0006),
        ('Fig. 6.5: paired D-CF-RAB minus MR+W, tiered, lower bound', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['tiered']['D-CF-RAB'][3])-np.array(RESULTS['E3']['tiered']['MR+W'][3]))[1], np.float64(-0.002), 0.0006),
        ('Fig. 6.5: paired D-CF-RAB minus MR+W, tiered, upper bound', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['tiered']['D-CF-RAB'][3])-np.array(RESULTS['E3']['tiered']['MR+W'][3]))[2], np.float64(0.022), 0.0006),
        ('Fig. 6.5: paired D-CF-RAB minus LP-MMSE+W, uniform, mean', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['uniform']['D-CF-RAB'][3])-np.array(RESULTS['E3']['uniform']['LP-MMSE+W'][3]))[0], np.float64(0.34), 0.006),
        ('Fig. 6.5: paired D-CF-RAB minus LP-MMSE+W, uniform, lower bound', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['uniform']['D-CF-RAB'][3])-np.array(RESULTS['E3']['uniform']['LP-MMSE+W'][3]))[1], np.float64(-0.3), 0.006),
        ('Fig. 6.5: paired D-CF-RAB minus LP-MMSE+W, uniform, upper bound', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['uniform']['D-CF-RAB'][3])-np.array(RESULTS['E3']['uniform']['LP-MMSE+W'][3]))[2], np.float64(0.99), 0.006),
        ('Fig. 6.5: paired D-CF-RAB minus LP-MMSE + CPU, tiered, mean', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['tiered']['D-CF-RAB'][3])-np.array(RESULTS['E3']['tiered']['LP-MMSE+W'][3]))[0], np.float64(-0.058), 0.0006),
        ('Fig. 6.5: paired D-CF-RAB minus LP-MMSE + CPU, tiered, lower bound', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['tiered']['D-CF-RAB'][3])-np.array(RESULTS['E3']['tiered']['LP-MMSE+W'][3]))[1], np.float64(-0.97), 0.006),
        ('Fig. 6.5: paired D-CF-RAB minus LP-MMSE + CPU, tiered, upper bound', lambda: (lambda d: (d.mean(), d.mean()-tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)), d.mean()+tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))))(np.array(RESULTS['E3']['tiered']['D-CF-RAB'][3])-np.array(RESULTS['E3']['tiered']['LP-MMSE+W'][3]))[2], np.float64(0.85), 0.006),
        ('Fig. 6.6: listed bytes, D-CF-RAB, L = 16', lambda: [r for r in RESULTS['E4'] if r['L']==16][0]['fh']['D-CF-RAB'], 1280.0, 0.5),
        ('Fig. 6.6: listed bytes, CF-RAB, L = 16', lambda: [r for r in RESULTS['E4'] if r['L']==16][0]['fh']['CF-RAB'], 4352.0, 0.5),
        ('Fig. 6.6: listed bytes, P-MMSE, L = 16', lambda: [r for r in RESULTS['E4'] if r['L']==16][0]['fh']['P-MMSE'], 2048.0, 0.5),
        ('Fig. 6.6: D-CF-RAB updates timed at L = 16', lambda: [r for r in RESULTS['E4'] if r['L']==16][0]['calls']['D-CF-RAB'], 600.0, 0.5),
        ('Fig. 6.6: CF-RAB updates timed at L = 16', lambda: [r for r in RESULTS['E4'] if r['L']==16][0]['calls']['CF-RAB'], 100.0, 0.5),
        ('Fig. 6.7: D-CF-RAB at 5 snapshots', lambda: g('E5','D-CF-RAB',5), 12.17, 0.006),
        ('Fig. 6.7: D-CF-RAB at 80 snapshots', lambda: g('E5','D-CF-RAB',80), 13.46, 0.006),
        ('Fig. 6.7: MR + CPU at 5 snapshots', lambda: g('E5','MR+W',5), 12.17, 0.006),
        ('Fig. 6.7: MR + CPU at 80 snapshots', lambda: g('E5','MR+W',80), 13.45, 0.006),
        ('Section 6.7.9: CF-RAB at eps = 0.10', lambda: ge('CF-RAB',.1), 16.39, 0.006),
        ('Section 6.7.9: D-CF-RAB minus CF-RAB at eps = 0.10', lambda: ge('D-CF-RAB',.1)-ge('CF-RAB',.1), 9.99, 0.006),
        ('Section 6.7.9: P-MMSE minus CF-RAB at eps = 0.10', lambda: ge('P-MMSE',.1)-ge('CF-RAB',.1), 12.29, 0.006),
        ('Section 6.7.9: INCM reconstruction at eps = 0.10', lambda: ge('RECON-RAB',.1), 14.48, 0.006),
        ('Section 6.7.9: largest |CF-RAB minus SOCP| over Figs. 6.1, 6.2, 6.3 and 6.5', lambda: max(abs(v['CF-RAB'][0]-v['centralized_sdp'][0]) for t_ in ('E1','E7','E6','E3') for v in RESULTS[t_].values()), 0.11, 0.006),
        ('Fig. 6.8: smallest objective excess (%)', lambda: min(r['gap_pct'] for r in RESULTS['E10']), 1.8, 0.06),
        ('Fig. 6.8: largest objective excess (%)', lambda: max(r['gap_pct'] for r in RESULTS['E10']), 6.6, 0.06),
        ('Fig. 6.9: loaded MVDR at 0 dB desired-signal power, eps = 0.10', lambda: [r for r in RESULTS['E8'] if abs(r['eps']-0.1)<1e-9 and r['soi_dB']==0][0]['MVDR'], -11.46, 0.006),
        ('Fig. 6.9: loaded MVDR at 30 dB desired-signal power, eps = 0.10', lambda: [r for r in RESULTS['E8'] if abs(r['eps']-0.1)<1e-9 and r['soi_dB']==30][0]['MVDR'], -41.35, 0.006),
        ('Fig. 6.9: CF-RAB at 0 dB desired-signal power, eps = 0.10', lambda: [r for r in RESULTS['E8'] if abs(r['eps']-0.1)<1e-9 and r['soi_dB']==0][0]['CF-RAB'], -6.56, 0.006),
        ('Fig. 6.9: CF-RAB at 30 dB desired-signal power, eps = 0.10', lambda: [r for r in RESULTS['E8'] if abs(r['eps']-0.1)<1e-9 and r['soi_dB']==30][0]['CF-RAB'], 31.42, 0.006),
        ('Fig. 6.9: primary-ball SOCP at 0 dB desired-signal power, eps = 0.10', lambda: [r for r in RESULTS['E8'] if abs(r['eps']-0.1)<1e-9 and r['soi_dB']==0][0]['centralized_sdp'], 19.38, 0.006),
        ('Fig. 6.9: primary-ball SOCP at 30 dB desired-signal power, eps = 0.10', lambda: [r for r in RESULTS['E8'] if abs(r['eps']-0.1)<1e-9 and r['soi_dB']==30][0]['centralized_sdp'], 25.84, 0.006),
        ('Fig. 6.10: c* at the joint dimension', lambda: RESULTS['E0']['joint_star_mean'], 0.535, 0.0006),
        ('Fig. 6.10: c* at the local dimension', lambda: RESULTS['E0']['per_star_mean'], 0.164, 0.0006)])
    if EXP.get('A_paired'):
        check_numbers('Numbers quoted in the chapter: paired comparisons, audits and parameter selection', [
        ('bound level selected on training realizations', lambda: EXP['c_star'], 0.98, 1e-09),
        ('training SINR at c = 0.9', lambda: float(np.mean(list(EXP['train']['0.9'].values()))), 21.75, 0.006),
        ('training SINR at c = 0.95', lambda: float(np.mean(list(EXP['train']['0.95'].values()))), 21.77, 0.006),
        ('training SINR at c = 0.98', lambda: float(np.mean(list(EXP['train']['0.98'].values()))), 21.77, 0.006),
        ('paired, L = 4: D-CF-RAB minus LP-MMSE + CPU, mean', lambda: EXP['A_paired']['norm_L4_e010']['D-LP-MMSE+W'][0], -4.17, 0.006),
        ('paired, L = 4: D-CF-RAB minus LP-MMSE + CPU, lower bound', lambda: EXP['A_paired']['norm_L4_e010']['D-LP-MMSE+W'][1], -5.06, 0.006),
        ('paired, L = 4: D-CF-RAB minus LP-MMSE + CPU, upper bound', lambda: EXP['A_paired']['norm_L4_e010']['D-LP-MMSE+W'][2], -3.28, 0.006),
        ('paired, L = 8: D-CF-RAB minus LP-MMSE + CPU, mean', lambda: EXP['A_paired']['norm_L8_e010']['D-LP-MMSE+W'][0], -0.4, 0.006),
        ('paired, L = 8: D-CF-RAB minus LP-MMSE + CPU, lower bound', lambda: EXP['A_paired']['norm_L8_e010']['D-LP-MMSE+W'][1], -0.79, 0.006),
        ('paired, L = 8: D-CF-RAB minus LP-MMSE + CPU, upper bound', lambda: EXP['A_paired']['norm_L8_e010']['D-LP-MMSE+W'][2], -0.0001, 6e-05),
        ('paired, L = 16: D-CF-RAB minus LP-MMSE + CPU, mean', lambda: EXP['A_paired']['norm_L16_e010']['D-LP-MMSE+W'][0], 0.61, 0.006),
        ('paired, L = 16: D-CF-RAB minus LP-MMSE + CPU, lower bound', lambda: EXP['A_paired']['norm_L16_e010']['D-LP-MMSE+W'][1], 0.32, 0.006),
        ('paired, L = 16: D-CF-RAB minus LP-MMSE + CPU, upper bound', lambda: EXP['A_paired']['norm_L16_e010']['D-LP-MMSE+W'][2], 0.89, 0.006),
        ('paired, eps = 0.10: D-CF-RAB minus P-MMSE, mean', lambda: EXP['A_paired']['norm_L16_e010']['D-P-MMSE'][0], -2.55, 0.006),
        ('paired, eps = 0.10: D-CF-RAB minus P-MMSE, lower bound', lambda: EXP['A_paired']['norm_L16_e010']['D-P-MMSE'][1], -2.77, 0.006),
        ('paired, eps = 0.10: D-CF-RAB minus P-MMSE, upper bound', lambda: EXP['A_paired']['norm_L16_e010']['D-P-MMSE'][2], -2.33, 0.006),
        ('paired, eps = 0.50: D-CF-RAB minus P-MMSE, mean', lambda: EXP['A_paired']['norm_L16_e050']['D-P-MMSE'][0], 0.87, 0.006),
        ('paired, eps = 0.50: D-CF-RAB minus P-MMSE, lower bound', lambda: EXP['A_paired']['norm_L16_e050']['D-P-MMSE'][1], 0.54, 0.006),
        ('paired, eps = 0.50: D-CF-RAB minus P-MMSE, upper bound', lambda: EXP['A_paired']['norm_L16_e050']['D-P-MMSE'][2], 1.2, 0.006),
        ('paired, path loss, eps = 0.50: D-CF-RAB minus P-MMSE, mean', lambda: EXP['A_paired']['pl_L16_e050']['D-P-MMSE'][0], 3.48, 0.006),
        ('paired, path loss, eps = 0.50: D-CF-RAB minus P-MMSE, lower bound', lambda: EXP['A_paired']['pl_L16_e050']['D-P-MMSE'][1], 2.12, 0.006),
        ('paired, path loss, eps = 0.50: D-CF-RAB minus P-MMSE, upper bound', lambda: EXP['A_paired']['pl_L16_e050']['D-P-MMSE'][2], 4.83, 0.006),
        ('paired, eps = 0.10: value of the worst-case constraint, mean', lambda: EXP['A_paired']['norm_L16_e010']['D-LMVDR+W'][0], 3.51, 0.006),
        ('paired, eps = 0.10: value of the worst-case constraint, lower bound', lambda: EXP['A_paired']['norm_L16_e010']['D-LMVDR+W'][1], 2.97, 0.006),
        ('paired, eps = 0.10: value of the worst-case constraint, upper bound', lambda: EXP['A_paired']['norm_L16_e010']['D-LMVDR+W'][2], 4.05, 0.006),
        ('paired, eps = 0.50: value of the worst-case constraint, mean', lambda: EXP['A_paired']['norm_L16_e050']['D-LMVDR+W'][0], 3.18, 0.006),
        ('paired, eps = 0.50: value of the worst-case constraint, lower bound', lambda: EXP['A_paired']['norm_L16_e050']['D-LMVDR+W'][1], 2.5, 0.006),
        ('paired, eps = 0.50: value of the worst-case constraint, upper bound', lambda: EXP['A_paired']['norm_L16_e050']['D-LMVDR+W'][2], 3.86, 0.006),
        ('paired: largest |D-CF-RAB minus MR + CPU|', lambda: max(abs(v['D-MR+W'][0]) for v in EXP['A_paired'].values()), 0.03, 0.006),
        ('exact minus Algorithm 6.1 local stage, eps0.1, mean', lambda: EXP['B_local_stage']['eps0.1']['exact_minus_active_set'][0], 0.13, 0.006),
        ('exact minus Algorithm 6.1 local stage, eps0.1, lower bound', lambda: EXP['B_local_stage']['eps0.1']['exact_minus_active_set'][1], 0.084, 0.0006),
        ('exact minus Algorithm 6.1 local stage, eps0.1, upper bound', lambda: EXP['B_local_stage']['eps0.1']['exact_minus_active_set'][2], 0.18, 0.006),
        ('exact minus Algorithm 6.1 local stage, eps0.5, mean', lambda: EXP['B_local_stage']['eps0.5']['exact_minus_active_set'][0], 0.27, 0.006),
        ('exact minus Algorithm 6.1 local stage, eps0.5, lower bound', lambda: EXP['B_local_stage']['eps0.5']['exact_minus_active_set'][1], 0.19, 0.006),
        ('exact minus Algorithm 6.1 local stage, eps0.5, upper bound', lambda: EXP['B_local_stage']['eps0.5']['exact_minus_active_set'][2], 0.35, 0.006),
        ('local audit: number of solves', lambda: float(EXP['C_local']['n']), 80.0, 0.5),
        ('local audit: smallest objective ratio', lambda: min(r['excess_ratio'] for r in EXP['C_local']['rows']), 2.2, 0.06),
        ('local audit: largest objective ratio', lambda: max(r['excess_ratio'] for r in EXP['C_local']['rows']), 17.8, 0.06),
        ('local audit: median objective ratio', lambda: float(np.median([r['excess_ratio'] for r in EXP['C_local']['rows']])), 4.2, 0.06),
        ('local audit: median direction difference (degrees)', lambda: float(np.median([np.degrees(np.arccos(min(1.0, r['direction_cos']))) for r in EXP['C_local']['rows']])), 16.5, 0.06),
        ('local audit: every weight feasible to 2e-14', lambda: float(min(r['min_margin'] for r in EXP['C_local']['rows']) > -2e-14), 1.0, 0.1),
        ('joint audit, eps = 0.01: median Algorithm 6.1 excess (%)', lambda: float(np.median(EXP['C_joint']['0.01']['alg_vs_same_pct'])), 0.0, 0.006),
        ('joint audit, eps = 0.1: median Algorithm 6.1 excess (%)', lambda: float(np.median(EXP['C_joint']['0.1']['alg_vs_same_pct'])), 0.02, 0.006),
        ('joint audit, eps = 0.5: median Algorithm 6.1 excess (%)', lambda: float(np.median(EXP['C_joint']['0.5']['alg_vs_same_pct'])), 11.6, 0.06),
        ('joint audit, eps = 0.5: median effect of the extra balls (%)', lambda: float(np.median(EXP['C_joint']['0.5']['same_vs_primary_pct'])), 0.11, 0.006),
        ('joint audit: 12 optimal solves per radius', lambda: float(all(v['n_ok'] == 12 for v in EXP['C_joint'].values())), 1.0, 0.1),
        ('joint audit: every weight feasible to 2e-15', lambda: float(min(r['alg_min_margin'] for v in EXP['C_joint'].values() for r in v['instances']) > -2e-15), 1.0, 0.1),
        ('CF-RAB bound sweep, declared', lambda: EXP['D_cfrab_bound']['declared']['mean'], 16.39, 0.006),
        ('CF-RAB bound sweep, 0.5', lambda: EXP['D_cfrab_bound']['0.5']['mean'], 14.07, 0.006),
        ('CF-RAB bound sweep, 0.6', lambda: EXP['D_cfrab_bound']['0.6']['mean'], 10.5, 0.006),
        ('CF-RAB bound sweep, 0.7', lambda: EXP['D_cfrab_bound']['0.7']['mean'], 4.86, 0.006),
        ('rescaling failures at eps = 0.10', lambda: float(EXP['violations']['0.1'].get('violated', 0)), 0.0, 0.5),
        ('rescaling failures at eps = 0.50', lambda: float(EXP['violations']['0.5'].get('violated', 0)), 12.0, 0.5),
        ('updates checked at eps = 0.50', lambda: float(sum(EXP['violations']['0.5'].values())), 1200.0, 0.5),
        ('solver check: 36 problems', lambda: float(len(EXP['solver_check']['cases'])), 36.0, 0.5),
        ('solver check: relative objective difference below 1e-9', lambda: float(EXP['solver_check']['max_rel_obj'] < 1e-9), 1.0, 0.1),
        ('solver check: constraint residual below 2e-14', lambda: float(EXP['solver_check']['max_residual'] < 2e-14), 1.0, 0.1)])
    if EXP.get('L_bound'):
        check_numbers('Numbers quoted in the chapter: Section 6.7.12 and Table 6.2', [
        ('Table 6.2, 4 APs, 16 users, eps = 0.10: selected bound', lambda: EXP['L_bound']['heavy_e010']['selected']['c'], 0.4, 1e-09),
        ('Table 6.2, 4 APs, 16 users, eps = 0.10: D-CF-RAB', lambda: EXP['L_bound']['heavy_e010']['test']['mean']['D_tuned'], 3.51, 0.006),
        ('Table 6.2, 4 APs, 16 users, eps = 0.10: MR + CPU', lambda: EXP['L_bound']['heavy_e010']['test']['mean']['MR'], 0.71, 0.006),
        ('Table 6.2, 4 APs, 16 users, eps = 0.10: tuned LP-MMSE + CPU', lambda: EXP['L_bound']['heavy_e010']['test']['mean']['LP_tuned'], 6.46, 0.006),
        ('Table 6.2, 4 APs, 16 users, eps = 0.10: tuned P-MMSE', lambda: EXP['L_bound']['heavy_e010']['test']['mean']['P_tuned'], 15.55, 0.006),
        ('Table 6.2, 4 APs, 16 users, eps = 0.10: gain over MR + CPU', lambda: EXP['L_bound']['heavy_e010']['test']['D_tuned - MR'][0], 2.81, 0.006),
        ('Table 6.2, 4 APs, 16 users, eps = 0.50: selected bound', lambda: EXP['L_bound']['heavy_e050']['selected']['c'], 0.6, 1e-09),
        ('Table 6.2, 4 APs, 16 users, eps = 0.50: D-CF-RAB', lambda: EXP['L_bound']['heavy_e050']['test']['mean']['D_tuned'], 0.71, 0.006),
        ('Table 6.2, 4 APs, 16 users, eps = 0.50: MR + CPU', lambda: EXP['L_bound']['heavy_e050']['test']['mean']['MR'], -0.7, 0.006),
        ('Table 6.2, 4 APs, 16 users, eps = 0.50: tuned LP-MMSE + CPU', lambda: EXP['L_bound']['heavy_e050']['test']['mean']['LP_tuned'], 1.01, 0.006),
        ('Table 6.2, 4 APs, 16 users, eps = 0.50: tuned P-MMSE', lambda: EXP['L_bound']['heavy_e050']['test']['mean']['P_tuned'], 3.38, 0.006),
        ('Table 6.2, 4 APs, 16 users, eps = 0.50: gain over MR + CPU', lambda: EXP['L_bound']['heavy_e050']['test']['D_tuned - MR'][0], 1.4, 0.006),
        ('Table 6.2, 4 APs, 4 users, eps = 0.10: selected bound', lambda: EXP['L_bound']['small_e010']['selected']['c'], 0.6, 1e-09),
        ('Table 6.2, 4 APs, 4 users, eps = 0.10: D-CF-RAB', lambda: EXP['L_bound']['small_e010']['test']['mean']['D_tuned'], 15.79, 0.006),
        ('Table 6.2, 4 APs, 4 users, eps = 0.10: MR + CPU', lambda: EXP['L_bound']['small_e010']['test']['mean']['MR'], 14.98, 0.006),
        ('Table 6.2, 4 APs, 4 users, eps = 0.10: tuned LP-MMSE + CPU', lambda: EXP['L_bound']['small_e010']['test']['mean']['LP_tuned'], 19.5, 0.006),
        ('Table 6.2, 4 APs, 4 users, eps = 0.10: tuned P-MMSE', lambda: EXP['L_bound']['small_e010']['test']['mean']['P_tuned'], 21.81, 0.006),
        ('Table 6.2, 4 APs, 4 users, eps = 0.10: gain over MR + CPU', lambda: EXP['L_bound']['small_e010']['test']['D_tuned - MR'][0], 0.82, 0.006),
        ('Table 6.2, 16 APs, eps = 0.10: selected bound', lambda: EXP['L_bound']['nom_e010']['selected']['c'], 0.8, 1e-09),
        ('Table 6.2, 16 APs, eps = 0.10: D-CF-RAB', lambda: EXP['L_bound']['nom_e010']['test']['mean']['D_tuned'], 26.7, 0.006),
        ('Table 6.2, 16 APs, eps = 0.10: MR + CPU', lambda: EXP['L_bound']['nom_e010']['test']['mean']['MR'], 26.66, 0.006),
        ('Table 6.2, 16 APs, eps = 0.10: tuned LP-MMSE + CPU', lambda: EXP['L_bound']['nom_e010']['test']['mean']['LP_tuned'], 26.65, 0.006),
        ('Table 6.2, 16 APs, eps = 0.10: tuned P-MMSE', lambda: EXP['L_bound']['nom_e010']['test']['mean']['P_tuned'], 28.9, 0.006),
        ('Table 6.2, 16 APs, eps = 0.10: gain over MR + CPU', lambda: EXP['L_bound']['nom_e010']['test']['D_tuned - MR'][0], 0.05, 0.006),
        ('Table 6.2, 16 APs, eps = 0.50: selected bound', lambda: EXP['L_bound']['nom_e050']['selected']['c'], 0.999, 1e-09),
        ('Table 6.2, 16 APs, eps = 0.50: D-CF-RAB', lambda: EXP['L_bound']['nom_e050']['test']['mean']['D_tuned'], 17.46, 0.006),
        ('Table 6.2, 16 APs, eps = 0.50: MR + CPU', lambda: EXP['L_bound']['nom_e050']['test']['mean']['MR'], 17.46, 0.006),
        ('Table 6.2, 16 APs, eps = 0.50: tuned LP-MMSE + CPU', lambda: EXP['L_bound']['nom_e050']['test']['mean']['LP_tuned'], 17.54, 0.006),
        ('Table 6.2, 16 APs, eps = 0.50: tuned P-MMSE', lambda: EXP['L_bound']['nom_e050']['test']['mean']['P_tuned'], 16.72, 0.006),
        ('Table 6.2, 16 APs, eps = 0.50: gain over MR + CPU', lambda: EXP['L_bound']['nom_e050']['test']['D_tuned - MR'][0], -0.0, 0.006),
        ('Table 6.2, 16 APs, path loss, eps = 0.50: selected bound', lambda: EXP['L_bound']['pl_e050']['selected']['c'], 0.98, 1e-09),
        ('Table 6.2, 16 APs, path loss, eps = 0.50: D-CF-RAB', lambda: EXP['L_bound']['pl_e050']['test']['mean']['D_tuned'], 20.41, 0.006),
        ('Table 6.2, 16 APs, path loss, eps = 0.50: MR + CPU', lambda: EXP['L_bound']['pl_e050']['test']['mean']['MR'], 20.48, 0.006),
        ('Table 6.2, 16 APs, path loss, eps = 0.50: tuned LP-MMSE + CPU', lambda: EXP['L_bound']['pl_e050']['test']['mean']['LP_tuned'], 20.92, 0.006),
        ('Table 6.2, 16 APs, path loss, eps = 0.50: tuned P-MMSE', lambda: EXP['L_bound']['pl_e050']['test']['mean']['P_tuned'], 20.29, 0.006),
        ('Table 6.2, 16 APs, path loss, eps = 0.50: gain over MR + CPU', lambda: EXP['L_bound']['pl_e050']['test']['D_tuned - MR'][0], -0.08, 0.006),
        ('Section 6.7.12, 4 APs, 16 users, eps = 0.10: gain over MR + CPU, lower bound', lambda: EXP['L_bound']['heavy_e010']['test']['D_tuned - MR'][1], 2.47, 0.006),
        ('Section 6.7.12, 4 APs, 16 users, eps = 0.10: gain over MR + CPU, upper bound', lambda: EXP['L_bound']['heavy_e010']['test']['D_tuned - MR'][2], 3.14, 0.006),
        ('Section 6.7.12, 4 APs, 16 users, eps = 0.10: worst-user gain over MR + CPU', lambda: EXP['L_bound']['heavy_e010']['test']['D_tuned_worst - MR_worst'][0], 3.81, 0.006),
        ('Section 6.7.12, 4 APs, 16 users, eps = 0.50: gain over MR + CPU, lower bound', lambda: EXP['L_bound']['heavy_e050']['test']['D_tuned - MR'][1], 1.22, 0.006),
        ('Section 6.7.12, 4 APs, 16 users, eps = 0.50: gain over MR + CPU, upper bound', lambda: EXP['L_bound']['heavy_e050']['test']['D_tuned - MR'][2], 1.58, 0.006),
        ('Section 6.7.12, 4 APs, 16 users, eps = 0.50: worst-user gain over MR + CPU', lambda: EXP['L_bound']['heavy_e050']['test']['D_tuned_worst - MR_worst'][0], 2.03, 0.006),
        ('Section 6.7.12, 4 APs, 4 users, eps = 0.10: gain over MR + CPU, lower bound', lambda: EXP['L_bound']['small_e010']['test']['D_tuned - MR'][1], 0.12, 0.006),
        ('Section 6.7.12, 4 APs, 4 users, eps = 0.10: gain over MR + CPU, upper bound', lambda: EXP['L_bound']['small_e010']['test']['D_tuned - MR'][2], 1.52, 0.006),
        ('Section 6.7.12, 4 APs, 4 users, eps = 0.10: worst-user gain over MR + CPU', lambda: EXP['L_bound']['small_e010']['test']['D_tuned_worst - MR_worst'][0], 1.57, 0.006),
        ('Chapter summary: largest gain over MR + CPU with four APs', lambda: max(EXP['L_bound'][k]['test']['D_tuned - MR'][0] for k in ('heavy_e010','heavy_e050','small_e010')), 2.8, 0.06),
        ('Chapter summary: largest worst-user gain with four APs', lambda: max(EXP['L_bound'][k]['test']['D_tuned_worst - MR_worst'][0] for k in ('heavy_e010','heavy_e050','small_e010')), 3.8, 0.06),
        ('Section 6.7.12: largest |gain over MR + CPU| with 16 APs', lambda: max(abs(EXP['L_bound'][k]['test']['D_tuned - MR'][0]) for k in ('nom_e010','nom_e050','pl_e050')), 0.08, 0.006),
        ('Section 6.7.12: smallest LP-MMSE + CPU lead with four APs', lambda: min(-EXP['L_bound'][k]['test']['D_tuned - LP_tuned'][0] for k in ('heavy_e010','heavy_e050','small_e010')), 0.31, 0.006),
        ('Section 6.7.12: largest LP-MMSE + CPU lead with four APs', lambda: max(-EXP['L_bound'][k]['test']['D_tuned - LP_tuned'][0] for k in ('heavy_e010','heavy_e050','small_e010')), 3.71, 0.006),
        ('Section 6.7.12: path loss, D-CF-RAB minus nominal P-MMSE, mean', lambda: EXP['L_bound']['pl_e050']['test']['D_default - P_default'][0], 3.26, 0.006),
        ('Section 6.7.12: path loss, D-CF-RAB minus nominal P-MMSE, lower bound', lambda: EXP['L_bound']['pl_e050']['test']['D_default - P_default'][1], 1.62, 0.006),
        ('Section 6.7.12: path loss, D-CF-RAB minus nominal P-MMSE, upper bound', lambda: EXP['L_bound']['pl_e050']['test']['D_default - P_default'][2], 4.9, 0.006),
        ('Section 6.7.12: path loss, D-CF-RAB minus tuned P-MMSE, mean', lambda: EXP['L_bound']['pl_e050']['test']['D_default - P_tuned'][0], 0.11, 0.006),
        ('Section 6.7.12: path loss, D-CF-RAB minus tuned P-MMSE, lower bound', lambda: EXP['L_bound']['pl_e050']['test']['D_default - P_tuned'][1], -0.46, 0.006),
        ('Section 6.7.12: path loss, D-CF-RAB minus tuned P-MMSE, upper bound', lambda: EXP['L_bound']['pl_e050']['test']['D_default - P_tuned'][2], 0.68, 0.006),
        ('Section 6.7.12: normalized gains, D-CF-RAB minus tuned P-MMSE, mean', lambda: EXP['L_bound']['nom_e050']['test']['D_default - P_tuned'][0], 0.74, 0.006),
        ('Section 6.7.12: normalized gains, D-CF-RAB minus tuned P-MMSE, lower bound', lambda: EXP['L_bound']['nom_e050']['test']['D_default - P_tuned'][1], 0.49, 0.006),
        ('Section 6.7.12: normalized gains, D-CF-RAB minus tuned P-MMSE, upper bound', lambda: EXP['L_bound']['nom_e050']['test']['D_default - P_tuned'][2], 0.99, 0.006),
        ('Section 6.7.12: path-loss difference from tuned P-MMSE not significant', lambda: float(EXP['L_bound']['pl_e050']['test']['D_default - P_tuned'][1] < 0 < EXP['L_bound']['pl_e050']['test']['D_default - P_tuned'][2]), 1.0, 0.1)])


def check_records(RA, RB, RES=None):
    """Sections 6.7.6, 6.7.9 and 6.7.12 to 6.7.15 and Tables 6.2 to 6.5."""
    paper, TXT = _docx_text(DOCX)
    if paper:
        print('chapter text checked for each quoted string:', paper)
    fails = []
    n = lambda x, d=2: f'{x:.{d}f}'.replace('-', '−')
    def chk(label, value, quoted, tol, text=None):
        good = abs(value - quoted) <= tol
        shown = True
        if TXT is not None and text is not None:
            shown = text.replace(' ', ' ') in TXT
        flag = 'OK ' if good and shown else '!! '
        if not (good and shown): fails.append(label)
        print(f'  [{flag}] {label}: {value:.4g} vs {quoted}' + ('' if shown else f'   (string "{text}" not found)'))
    A1, A3, A3p, A4, A5 = RA['A1'], RA['A3'], RA['A3_paired'], RA['A4'], RA['A5']
    B1, B2 = RB['B1'], RB['B2']
    ab = lambda k, f: A1[k][f]
    print('Section 6.7.13 and Table 6.4 (error laws)')
    EL = ('boundary', 'interior', 'radius', 'gaussian')
    for e in EL:
        a, b, c = A1[f'norm_L16_e010|{e}'], A1[f'norm_L16_e050|{e}'], A1[f'pl_L16_e050|{e}']
        for lab, v in ((f'{e} D eps0.10', a['mean']['D-CF-RAB']), (f'{e} D eps0.50', b['mean']['D-CF-RAB'])):
            chk('Table 6.4 ' + lab, v, round(v, 2), 0.005, n(v))
        for lab, v in ((f'{e} D-P eps0.10', a['D-P-MMSE']), (f'{e} D-MR+CPU eps0.50', b['D-MR+W']),
                       (f'{e} D-P eps0.50', b['D-P-MMSE']), (f'{e} D-P path loss', c['D-P-MMSE'])):
            d_ = 3 if 'MR+CPU' in lab else 2
            s = f'{n(v[0], d_)} ({n(v[1], d_)}, {n(v[2], d_)})'
            chk('Table 6.4 ' + lab, v[0], round(v[0], d_), 0.5 * 10 ** -d_, s)
        chk(f'Table 6.4 {e} failed scored checks', b['resp_below1'], b['resp_below1'], 0, f'{b["resp_below1"]} of {b["resp_n"]}')
        chk(f'{e}: no failure at eps 0.10 / path loss', a['resp_below1'] + c['resp_below1'], 0, 0)
    d10 = [A1[f'norm_L16_e010|{e}']['mean']['D-CF-RAB'] for e in EL]
    chk('D-CF-RAB range at eps 0.10 (low)', min(d10), 26.04, 0.005, '26.04 to 26.82 dB')
    chk('D-CF-RAB range at eps 0.10 (high)', max(d10), 26.82, 0.005)
    chk('|D - MR+CPU| at eps 0.10 below 0.01 dB', float(max(abs(A1[f'norm_L16_e010|{e}']['D-MR+W'][0]) for e in EL) < 0.01), 1, 0,
        'stays within 0.01 dB of MR + CPU combiner')
    chk('change of D-P and D-CF over error laws below 0.7 dB', float(max(max(A1[f'norm_L16_e010|{e}'][k][0] for e in EL) - min(A1[f'norm_L16_e010|{e}'][k][0] for e in EL)
            for k in ('D-P-MMSE', 'D-CF-RAB')) < 0.7), 1, 0, 'change by less than 0.7 dB')
    chk('D - P, gaussian, eps 0.50', ab('norm_L16_e050|gaussian', 'D-P-MMSE')[0], 0.66, 0.005, 'by 0.66 to 0.87 dB')
    r = ab('norm_L16_e050|radius', 'D-P-MMSE')
    chk('P - D, radius-uniform, eps 0.50', -r[0], 0.72, 0.005, f'{n(-r[0])} dB ({n(-r[2])} to {n(-r[1])} dB)')
    bR, rR = A1['norm_L16_e050|boundary']['mean'], A1['norm_L16_e050|radius']['mean']
    chk('P-MMSE gain, boundary to radius-uniform', rR['P-MMSE'] - bR['P-MMSE'], 6.37, 0.005, 'P-MMSE gains 6.37 dB')
    chk('D-CF-RAB gain, boundary to radius-uniform', rR['D-CF-RAB'] - bR['D-CF-RAB'], 4.78, 0.005, 'D-CF-RAB 4.78 dB')
    chk('D - P path loss (low)', min(A1[f'pl_L16_e050|{e}']['D-P-MMSE'][0] for e in EL), 3.41, 0.005, 'by 3.41 to 3.48 dB')
    chk('D - P path loss (high)', max(A1[f'pl_L16_e050|{e}']['D-P-MMSE'][0] for e in EL), 3.48, 0.005)
    chk('smallest realized response, eps 0.10 and path loss', min(A1[k]['resp_min'] for k in A1 if 'e010' in k or 'pl_' in k),
        1.08, 0.005, 'smallest realized response was 1.08')
    sel = {k: v['selected'] for k, v in RA['A2'].items()}
    chk('selected bound 0.8 at eps 0.10 for all laws', sum(sel[f'eps0.1|{e}'] == '0.8' for e in EL), 4, 0)
    chk('selected bound 0.999 at eps 0.50 for all laws', sum(sel[f'eps0.5|{e}'] == '0.999' for e in EL), 4, 0)
    chk('chapter summary: |D - MR+CPU| over all error laws', max(abs(v['D-MR+W'][0]) for v in A1.values()), 0.03, 0.005,
        'within 0.03 dB of matched-filter local processing')

    print('Section 6.7.15')
    nb = A1['norm_L16_e010|boundary']['mean']
    for lab, v, s in (('D-CF-RAB', nb['D-CF-RAB'], 'obtains 26.04 dB with 1280'), ('P-MMSE', nb['P-MMSE'], 'P-MMSE obtains 28.59 dB'),
                      ('CF-RAB', nb['CF-RAB'], 'CF-RAB 16.68 dB'), ('P - D', nb['P-MMSE'] - nb['D-CF-RAB'], 'stays 2.55 dB below')):
        chk('6.7.15 ' + lab, v, round(v, 2), 0.005, s)

    print('Section 6.7.6 (equal-mean tiers)')
    chk('D-CF-RAB', A3['D-CF-RAB'][0], 25.49, 0.005, 'D-CF-RAB obtains 25.49 dB')
    chk('MR + CPU', A3['MR+W'][0], 25.48, 0.005, 'MR + CPU combiner 25.48 dB')
    chk('P-MMSE', A3['P-MMSE'][0], 28.70, 0.005, 'P-MMSE 28.70 dB')
    v = A3p['LP-MMSE+W']; chk('D - LP-MMSE+CPU', v[0], 0.33, 0.005, f'{n(v[0])} dB ({n(v[1])} to {n(v[2])} dB)')

    print('Section 6.7.9 (scored-snapshot failures, continuous-ball checks)')
    chk('violations at eps 0.50 (all updates)', A4['0.5']['all_updates'].get('violated', 0), 12, 0)
    chk('violations on scored updates', sum(A4[e]['scored_updates'].get('violated', 0) for e in A4), 0, 0)
    chk('scored updates per radius', A4['0.5']['scored_updates']['scaled'], 180, 0)
    chk('closed-form worst case >= 1', min(A4[e]['analytic_worst_min'] for e in A4), 1.0, 1e-9)
    chk('sampled minimum, eps 0.10', A4['0.1']['mc_min'], 1.10, 0.005, 'response of 1.10 at')
    chk('sampled minimum, eps 0.50', A4['0.5']['mc_min'], 1.92, 0.005, '1.92 at')
    chk('CF-RAB primary-ball worst case >= 1', min(A5[e]['analytic_primary_min'] for e in A5), 1.0, 1e-9)
    chk('CF-RAB sampled minimum', min(A5[e]['mc_min'] for e in A5), 1.01, 0.005, 'minimum response of 1.01')
    chk('CF-RAB weights tested', sum(A5[e]['n'] for e in A5), 36, 0, 'all 36 tested weights')

    print('Section 6.7.12, Tables II and III')
    CF = ['heavy_e010', 'heavy_e050', 'small_e010', 'nom_e010', 'nom_e050', 'pl_e050']
    for c in CF:
        t = B1[f'{c}|population']['test']['mean']
        chk(f'Table 6.2 {c} MR + CPU', t['MR'], round(t['MR'], 2), 0.005, n(t['MR']))
        chk(f'Table 6.2 {c} MVDR + CPU', t['MV_tuned'], round(t['MV_tuned'], 2), 0.005, n(t['MV_tuned']))
        ts = B1[f'{c}|sample']['test']; v = ts['D_tuned - MV_tuned']
        chk(f'Table 6.3 {c} difference', v[0], round(v[0], 3), 0.0005, f'{n(v[0], 3)} ({n(v[1], 3)}, {n(v[2], 3)})')
    v = B1['heavy_e010|population']['test']['D_tuned - MR']
    chk('D - MR+CPU, 4 APs 16 users eps 0.10', v[0], 2.81, 0.005, f'{n(v[0])} dB ({n(v[1])} to {n(v[2])} dB)')
    chk('D - MR+CPU, 4 APs 16 users eps 0.50', B1['heavy_e050|population']['test']['D_tuned - MR'][0], 1.40, 0.005, '1.40 dB and 0.82 dB')
    chk('D - MR+CPU, 4 APs 4 users', B1['small_e010|population']['test']['D_tuned - MR'][0], 0.82, 0.005)
    chk('|D - MVDR tuned|, normalized gains', max(abs(B1[f'{c}|population']['test']['D_tuned - MV_tuned'][0]) for c in CF[:5]),
        0.14, 0.005, 'within 0.14 dB')
    v = B1['pl_e050|population']['test']['D_tuned - MV_tuned']
    chk('MVDR tuned - D, path loss', -v[0], 0.55, 0.005, f'{n(-v[0])} dB ({n(-v[2])} to {n(-v[1])} dB) above it under the path-loss')
    chk('largest population/sample difference', max(abs(B1[f'{c}|population']['test']['mean'][k] - B1[f'{c}|sample']['test']['mean'][k])
                                                     for c in CF for k in ('D_tuned', 'MV_tuned')), 0.021, 0.0005, 'at most 0.021 dB')
    chk('sample covariance, 16 APs eps 0.10: selected bound', B1['nom_e010|sample']['selected']['c'], 0.9, 0)
    chk('selected bounds otherwise unchanged', sum(B1[f'{c}|sample']['selected']['c'] != B1[f'{c}|population']['selected']['c']
                                                   for c in CF), 1, 0)
    chk('selected MVDR loadings equal', sum(B1[f'{c}|sample']['selected']['load'] != B1[f'{c}|population']['selected']['load']
                                            for c in CF), 0, 0)

    print('Section 6.7.14 and Table 6.5 (external interferers)')
    LAW = 'qam_impulsive'
    for s in B2:
        mm = B2[s]['test'][LAW]['mean']
        for x in ('D', 'MV', 'ACPU', 'MR', 'LP', 'P', 'CF'):
            chk(f'Table 6.5 {s} {x}', mm[x], round(mm[x], 2), 0.005, n(mm[x]))
    d = [B2[s]['test'][l]['mean']['D'] - B2[s]['test'][l]['mean'][x] for s in B2 for l in B2[s]['test'] for x in ('MR', 'LP', 'P')]
    chk('D minus estimate-based (low)', min(d), 3.75, 0.005, 'fall 3.75 to 6.77 dB below')
    chk('D minus estimate-based (high)', max(d), 6.77, 0.005)
    mv = [B2[s]['test'][l]['D - MV'][0] for s in B2 for l in B2[s]['test']]
    chk('MVDR tuned - D (low)', -max(mv), 0.11, 0.005, 'is 0.11 to 1.36 dB above D-CF-RAB')
    chk('MVDR tuned - D (high)', -min(mv), 1.36, 0.005)
    law = max(max(B2[s]['test'][l]['mean'][x] for l in B2[s]['test']) - min(B2[s]['test'][l]['mean'][x] for l in B2[s]['test'])
              for s in B2 for x in ('D', 'MV', 'ACPU', 'MR', 'LP', 'P'))
    chk('largest change with the symbol law', law, 0.11, 0.005, 'at most 0.11 dB')
    lcf = max(max(B2[s]['test'][l]['mean']['CF'] for l in B2[s]['test']) - min(B2[s]['test'][l]['mean']['CF'] for l in B2[s]['test'])
              for s in B2)
    chk('largest change with the symbol law, CF-RAB', lcf, 0.59, 0.005, 'CF-RAB changes by at most 0.59 dB')
    j6 = B2['J6_inr20']['test'][LAW]['mean']
    chk('J = 6, 20 dB: MR + adaptive CPU - D', j6['ACPU'] - j6['D'], 0.72, 0.005, 'are 0.72 and 0.96 dB above')
    chk('J = 6, 20 dB: CF-RAB - D', j6['CF'] - j6['D'], 0.96, 0.005)
    chk('D above ACPU and CF-RAB in the other scenarios', sum(B2[s]['test'][LAW]['mean']['D'] > max(B2[s]['test'][LAW]['mean']['ACPU'],
                                                                                         B2[s]['test'][LAW]['mean']['CF'])
                                                          for s in ('J2_inr10', 'J2_inr20', 'J6_inr10')), 3, 0)
    chk('selected bounds 0.2 to 0.3', sum(0.2 <= B2[s]['selected']['D'] <= 0.3 for s in B2), 4, 0)

    if RES is not None:
        print('MR + CPU against D-CF-RAB in the figures')
        mx = {t: max(abs(v['MR+W'][0] - v['D-CF-RAB'][0]) for v in RES[t].values()) for t in ('E1', 'E7', 'E6', 'E2', 'E3', 'E5')}
        chk('Fig. 6.1', mx['E1'], 0.022, 0.0005, 'within 0.022 dB')
        chk('Fig. 6.2', mx['E7'], 0.011, 0.0005, 'within 0.011 dB')
        chk('Figs. 6.1-6.5 and 6.7: largest difference below 0.05 dB', float(max(mx.values()) < 0.05), 1, 0, 'within 0.05 dB')
    print(f'\n{len(fails)} mismatch(es)' + (': ' + ', '.join(fails) if fails else ''))
    FAILS.extend(fails)
    return not fails

def main(argv=None):
    global DOCX
    import argparse
    p = argparse.ArgumentParser(description='Check the regenerated Chapter 6 results against the chapter.')
    p.add_argument('outdir', nargs='?', default='figures', help='output folder of generate_figures.py (data in <outdir>/data)')
    p.add_argument('--docx', default=None, help='optional .docx of the chapter: also confirm each quoted string')
    a = p.parse_args(argv)
    DOCX = a.docx
    R, EXP, RA, RB = _load(os.path.join(a.outdir, 'data'))
    missing = [k for k, v in R.items() if v is None]
    if missing:
        sys.exit('figure data missing for ' + ', '.join(missing) + '; run scripts/generate_figures.py first')
    check_figures(R, EXP)
    if all(v is not None for v in list(RA.values()) + list(RB.values())):
        print()
        check_records(RA, RB, R)
    else:
        print('\n(Tables 6.2-6.5 and Sections 6.7.6, 6.7.9, 6.7.12-6.7.15 not checked: their stages have not been run)')
    print(f'\n{len(FAILS)} check(s) differ from the chapter' + (':\n  ' + '\n  '.join(FAILS) if FAILS else ''))
    return len(FAILS)


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
