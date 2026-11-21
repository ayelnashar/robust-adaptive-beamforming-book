"""Check the regenerated results against the values quoted in Chapter 8.

    python -m ch8wrab.verify figures                      # data in figures/data
    python -m ch8wrab.verify figures --docx chapter8.docx  # also confirm that each quoted number appears in the .docx

Every check prints OK or !! with the regenerated value and the value quoted in the chapter. A number passes when the
regenerated value rounds to the quoted one (half a unit of the last printed digit), unless a wider tolerance is
stated with the check. Timings depend on the machine and are printed for information only. Values from a --quick run
use fewer trials and are not expected to match.
"""
import json
import os
import re
import sys
import zipfile

import numpy as np

from .algorithms import subcarrier_frequencies

FAILS = []
DOCX = None
COUNT = [0]


def _docx_text(path):
    x = zipfile.ZipFile(path).read('word/document.xml').decode('utf8')
    t = re.sub(r'<(w|m):t(?: [^>]*)?>([^<]*)</\1:t>', r'\2', x)
    t = re.sub(r'<[^>]+>', '', t)
    return t.replace(' ', ' ').replace('\xa0', ' ')


def _fmt(q, nd):
    s = f'{q:.{nd}f}'
    return s.replace('-', '−')


def num(label, value, quoted, nd=1, tol=None, in_docx=True):
    """Pass when |value - quoted| <= tol (default: half a unit of the last quoted digit)."""
    COUNT[0] += 1
    tol = 0.5*10**(-nd) + 1e-9 if tol is None else tol
    ok = value is not None and np.isfinite(value) and abs(value - quoted) <= tol
    extra = ''
    if DOCX is not None and in_docx:
        found = _fmt(quoted, nd) in DOCX
        extra = '' if found else '  [quoted string not found in the docx]'
        ok = ok and found
    if not ok:
        FAILS.append(label)
    v = 'n/a' if value is None else f'{value:.{max(nd, 1) + 2}f}'
    print(f'  [{"OK " if ok else "!! "}] {label}: regenerated {v}, chapter {_fmt(quoted, nd)}{extra}')


def flag(label, good, detail=''):
    COUNT[0] += 1
    if not good:
        FAILS.append(label)
    print(f'  [{"OK " if good else "!! "}] {label}' + (f'  ({detail})' if detail else ''))


def info(label, value, quoted):
    print(f'  [info] {label}: this run {value}, chapter {quoted}')


def main(argv):
    global DOCX
    if not argv:
        print(__doc__); return 1
    out = argv[0]; datadir = os.path.join(out, 'data')
    if '--docx' in argv:
        DOCX = _docx_text(argv[argv.index('--docx') + 1])

    def J(s):
        p = os.path.join(datadir, s + '.json')
        if not os.path.exists(p):
            return None
        d = json.load(open(p))
        if d.get('quick'):
            print(f'  note: {s} comes from a --quick run')
        return d['result']

    def mean(rec, m):
        return rec['summary'][m]['mean_db']

    def ci(s):
        return s['ci95_high_db'] - s['ci95_low_db']

    print('Section 8.4: subcarrier frequencies')
    f = subcarrier_frequencies(300e9, 20e9, 5)/1e9
    flag('five subcarriers at 292, 296, 300, 304 and 308 GHz', np.allclose(f, [292, 296, 300, 304, 308]))

    d = J('analytic')
    if d:
        print('Figures 8.1 and 8.2')
        a = d['fig1']
        flag('Figure 8.1: 16 elements nearly flat (less than 0.5 dB loss)', min(a['N16']) > -0.5, f"{min(a['N16']):.2f} dB")
        flag('Figure 8.1: 64 elements lose several decibels at the band edges', -10 < min(a['N64'][0], a['N64'][-1]) < -2,
             f"{min(a['N64'][0], a['N64'][-1]):.2f} dB")
        flag('Figure 8.1: 256 elements reach deep nulls (below -20 dB)', min(a['N256']) < -20, f"{min(a['N256']):.1f} dB")
        b = d['fig2']; i3 = b['jitter_rms_deg'].index(0.3)
        flag('Figure 8.2: 256 elements lose several dB at a few tenths of a degree; 64 elements stay tolerant',
             b['N256'][i3] > 2 and b['N64'][i3] < 0.5, f"0.3 deg: {b['N256'][i3]:.2f} and {b['N64'][i3]:.2f} dB")

    d = J('fig3')
    if d:
        print('Figure 8.3')
        num('WRAB band mean (near 15 dB)', mean(d, 'wrab'), 15, nd=0)
        num('analog-only band mean', mean(d, 'analog'), 1.3)
        num('WC-single band mean', mean(d, 'wc'), 1.8)

    d = J('fig4')
    if d:
        print('Figure 8.4')
        for mm, m, q in ((0, 'wrab', 16.8), (0, 'analog', 22.4), (0, 'wc', 22.9), (2, 'wrab', 18.9), (2, 'analog', -0.8),
                         (2, 'wc', 0.3), (3, 'wrab', 14.0), (4, 'wrab', 7.1), (6, 'wrab', 3.5)):
            num(f'{m} at {mm} deg', mean(d[str(mm)], m), q)
        flag('WRAB above both references from 2 deg onward',
             all(mean(d[str(mm)], 'wrab') > max(mean(d[str(mm)], 'analog'), mean(d[str(mm)], 'wc')) for mm in range(2, 7)))

    c = J('controls')
    if c:
        print('Section 8.5.3: signed mismatch with the asymmetric grid (three subcarriers)')
        num('WRAB at -2 deg', mean(c['signed']['-2'], 'wrab'), 14.1)

    d = J('fig5')
    if d:
        print('Figure 8.5')
        w = [mean(d[str(a)], 'wrab') for a in (0, 15, 30, 45)]
        num('WRAB lowest up to 45 deg', min(w), 18.3); num('WRAB highest up to 45 deg', max(w), 18.9)
        num('WRAB at 60 deg', mean(d['60'], 'wrab'), 16.7)
        g = [mean(d[str(a)], 'wrab') - mean(d[str(a)], 'fixed') for a in (0, 15, 30, 45)]
        num('fixed-margin variant below WRAB up to 45 deg, smallest gap', min(g), 3.6)
        num('fixed-margin variant below WRAB up to 45 deg, largest gap', max(g), 4.4)
        flag('fixed-margin variant within 0.3 dB of WRAB at 60 deg',
             abs(mean(d['60'], 'wrab') - mean(d['60'], 'fixed')) <= 0.3,
             f"{mean(d['60'], 'wrab') - mean(d['60'], 'fixed'):.2f} dB")
        num('WC-single (frequency-aware) at 60 deg', mean(d['60'], 'wc_faware'), 18.0)
        flag('WC-single (frequency-aware) exceeds WRAB at 60 deg', mean(d['60'], 'wc_faware') > mean(d['60'], 'wrab'))

    d = J('fig6')
    if d:
        print('Figure 8.6')
        num('WRAB at 3 deg', mean(d['3'], 'wrab'), 7.7)
        num('analog-only at 3 deg', mean(d['3'], 'analog'), -22.0); num('WC-single at 3 deg', mean(d['3'], 'wc'), -3.6)
        w = [mean(d[str(s)], 'wrab') for s in (5, 8, 12, 20, 40)]
        num('WRAB lowest from 5 deg', min(w), 18.8); num('WRAB highest from 5 deg', max(w), 18.9)

    d = J('fig7')
    if d:
        print('Figure 8.7(a)')
        from .figures import fig7_gap
        Ns = (64, 128, 256, 512); recs = [d[str(M)] for M in Ns]; gap = fig7_gap(recs)
        num('WRAB at 64 elements', mean(d['64'], 'wrab'), 16.4); num('WRAB at 512 elements', mean(d['512'], 'wrab'), 20.9)
        num('advantage over the better reference at 64 elements', gap[0]['gap_db'], 8.8)
        num('advantage over the better reference at 512 elements', gap[3]['gap_db'], 4.3)
        flag('paired bootstrap intervals of the advantage exclude zero at every size', all(g['ci95'][0] > 0 for g in gap))
        flag('all three methods gain SINR as the element count grows',
             all(np.all(np.diff([mean(d[str(M)], m) for M in Ns]) > 0) for m in ('analog', 'wc', 'wrab')))
        num('1.5 reference beamwidths at 64 elements (deg)', 1.5*np.rad2deg(1.772/64), 2.4)
        num('1.5 reference beamwidths at 512 elements (deg)', 1.5*np.rad2deg(1.772/512), 0.30, nd=2)

    d = J('fixedangle')
    if d:
        print('Figure 8.7(b)')
        v64 = [d['64'][m]['mean_db'] for m in ('analog', 'wc', 'wrab')]
        num('three methods at 64 elements, lowest', min(v64), 16.6); num('three methods at 64 elements, highest', max(v64), 17.2)
        num('WRAB (Table 8.1 fan) at 128 elements', d['128']['wrab']['mean_db'], 19.6)
        num('WRAB (Table 8.1 fan) at 512 elements', d['512']['wrab']['mean_db'], 7.1)
        num('analog-only at 512 elements', d['512']['analog']['mean_db'], 2.9)
        num('WC-single at 512 elements', d['512']['wc']['mean_db'], 2.8)
        flag('constraints of the fan sized in degrees: 5, 9, 15, 27',
             [d[str(M)]['Nc_anglefan'] for M in (64, 128, 256, 512)] == [5, 9, 15, 27])
        a = [d[str(M)]['wrab_anglefan']['mean_db'] for M in (64, 128, 256, 512)]
        num('fan sized in degrees, lowest', min(a), 16.3); num('fan sized in degrees, highest', max(a), 17.9)
        for M, q in ((64, 0.9), (128, 3.0)):
            num(f'cost of the wider fan at {M} elements', d[str(M)]['wrab']['mean_db'] - d[str(M)]['wrab_anglefan']['mean_db'], q)
        for M, q in ((256, 5.2), (512, 10.8)):
            num(f'gain of the wider fan at {M} elements', d[str(M)]['wrab_anglefan']['mean_db'] - d[str(M)]['wrab']['mean_db'], q)

    d = J('fig8')
    if d:
        print('Figure 8.8(a)')
        z = [mean(d['0'], m) for m in ('analog', 'wc', 'wrab')]; s6 = [mean(d['0.6'], m) for m in ('analog', 'wc', 'wrab')]
        num('SINR at zero jitter, lowest', min(z), 24.4); num('SINR at zero jitter, highest', max(z), 26.0)
        num('SINR at 0.6 deg, lowest', min(s6), 20.2); num('SINR at 0.6 deg, highest', max(s6), 20.8)
        g = [np.mean([mean(d[s], 'analog'), mean(d[s], 'wc')]) - mean(d[s], 'wrab') for s in ('0', '0.1', '0.2', '0.3')]
        num('WRAB below the references up to 0.3 deg (mean gap)', float(np.mean(g)), 1.4, tol=0.15)
        flag('WRAB slightly above the references from 0.5 deg',
             all(mean(d[s], 'wrab') > max(mean(d[s], 'analog'), mean(d[s], 'wc')) for s in ('0.5', '0.6')))
    d = J('dynamic')
    if d:
        print('Figure 8.8(b)')
        P = d['points']
        num('WRAB at zero rotation', P['0']['wrab']['mean_db'], 24.4)
        g0 = [P['0'][m]['mean_db'] - P['0']['wrab']['mean_db'] for m in ('analog', 'wc')]
        num('WRAB below the references at low jitter, smallest gap', min(g0), 1.3)
        num('WRAB below the references at low jitter, largest gap', max(g0), 1.5)
        flag('WRAB level with the references at 0.6 deg (within 0.2 dB)',
             max(abs(P['0.6'][m]['mean_db'] - P['0.6']['wrab']['mean_db']) for m in ('analog', 'wc')) <= 0.2)
        loss = {m: P['0'][m]['mean_db'] - P['0.6'][m]['mean_db'] for m in ('analog', 'wc', 'wrab')}
        num('WRAB loss from 0 to 0.6 deg', loss['wrab'], 3.1)
        num('reference loss from 0 to 0.6 deg, smallest', min(loss['analog'], loss['wc']), 4.4)
        num('reference loss from 0 to 0.6 deg, largest', max(loss['analog'], loss['wc']), 4.5)

    d = J('fig9')
    if d:
        print('Figure 8.9')
        a = np.mean(10*np.log10(np.asarray(d['1']['raw_linear']['wrab'])), axis=0)
        b = np.mean(10*np.log10(np.asarray(d['0']['raw_linear']['wrab'])), axis=0)
        num('lower edge, frequency-consistent grid', a[0], 14.4); num('lower edge, carrier-referenced grid', b[0], 12.7)
        num('upper edge, frequency-consistent grid', a[-1], 15.1); num('upper edge, carrier-referenced grid', b[-1], 15.7)
        num('upper edge, carrier-referenced advantage', b[-1] - a[-1], 0.7)
        flag('the two grids agree at the carrier (within 0.3 dB)', abs(a[len(a)//2] - b[len(b)//2]) <= 0.3,
             f'{a[len(a)//2]:.2f} and {b[len(b)//2]:.2f} dB')

    d = J('fig10')
    if d:
        print('Figure 8.10')
        on = [mean(d[f'1_{v:g}'], 'wrab') for v in (.1, 1.5, 3.0)]; off = [mean(d[f'0_{v:g}'], 'wrab') for v in (.1, 1.5, 3.0)]
        num('spread with adaptation', np.ptp(on), 0.05, nd=2); num('spread with frozen margins', np.ptp(off), 1.39, nd=2)
        num('frozen margins, smallest initial margin', off[0], 24.5)
        flag('frozen margins: the smallest margin gives the highest SINR', off[0] == max(off))

    if c:
        print('Section 8.5.8: margin-level control at 64 elements, 2 deg')
        g = lambda k: c['margins'][k]['summary']['wrab']['mean_db']    # noqa: E731
        num('adaptive margins', g('adaptive|2'), 18.31, nd=2); num('fixed margins of 3', g('fixed_3|2'), 18.31, nd=2)
        num('fixed margins of 0.3', g('fixed_0.3|2'), 11.58, nd=2)
        num('difference between adaptive and fixed 0.3', g('adaptive|2') - g('fixed_0.3|2'), 6.7)

    d = J('near')
    if d:
        print('Table 8.2')
        T = {'2': (6.2, -2.9, -2.9, -2.9, -2.9, -5.2), '3': (7.7, 0.2, -1.7, -2.9, -3.8, -5.2),
             '4': (17.8, 12.3, 15.3, 18.2, 19.1, 15.5), '5': (18.8, 11.4, 14.3, 17.3, 18.8, 18.5),
             '8': (18.9, 11.3, 14.3, 17.3, 18.9, 18.5)}
        cols = ['adaptive', 'fixed_0.3', 'fixed_1', 'fixed_2', 'fixed_3', 'lcmv']
        for s, row in T.items():
            for k, q in zip(cols, row):
                num(f'{s} deg, {k}', d['points'][s][k]['mean_db'], q)
        widest = max(ci(d['points'][s][k]) for s in T for k in cols)
        num('widest 95% interval in the table', widest, 0.9)
        num('smallest margin, median, 2 deg', d['points']['2']['margins']['smallest_median'], 0.01, nd=2)
        num('smallest margin, median, 3 deg', d['points']['3']['margins']['smallest_median'], 0.02, nd=2)
        for s in ('2', '3'):
            m = d['points'][s]['margins']
            flag(f'{s} deg: one constraint lowered and the others at the ceiling in most runs',
                 m['below_ceiling_counts'][1] > m['runs']/2, f"{m['below_ceiling_counts']} of {m['runs']} runs")
        flag('fixed levels at 2 and 3 deg at or below 0.2 dB',
             max(d['points'][s][f'fixed_{v:g}']['mean_db'] for s in ('2', '3') for v in d['levels']) <= 0.25)
        num('loss of the rule to a fixed margin of 3 at 4 deg',
            d['points']['4']['fixed_3']['mean_db'] - d['points']['4']['adaptive']['mean_db'], 1.3)
        flag('from 5 deg the rule and a fixed margin of 3 coincide (within 0.05 dB)',
             all(abs(d['points'][s]['adaptive']['mean_db'] - d['points'][s]['fixed_3']['mean_db']) <= 0.05 for s in ('5', '8')))

    if c:
        print('Section 8.5.9: desired signal present in training (8 trials, 3 subcarriers)')
        M2 = c['mixed']['2']
        num('WRAB at 2 deg', mean(M2, 'wrab'), 15.5); num('fixed-margin variant at 2 deg', mean(M2, 'fixed'), 8.8)
        conv = ('wc', 'mvdr', 'rcb', 'socp', 'recon')
        num('adaptive conventional beamformers at 2 deg, lowest', min(mean(M2, m) for m in conv), -23.0)
        num('adaptive conventional beamformers at 2 deg, highest', max(mean(M2, m) for m in conv), -10.4)
        num('WRAB at 0 deg', mean(c['mixed']['0'], 'wrab'), 14.3); num('MVDR at 0 deg', mean(c['mixed']['0'], 'mvdr'), -1.1)
        num('WRAB at 4 deg', mean(c['mixed']['4'], 'wrab'), -19.1)
        num('fixed margin of 3 at 2 deg', c['mixed_levels']['2']['fixed_3']['mean_db'], 15.5)

    d = J('fig11'); r = J('fig11ref')
    if d:
        print('Figure 8.11')
        MM = (0, 1, 2, 3, 4, 5, 6, 8); first8 = d['0']['methods']
        allci = {(mm, m): ci(d[str(mm)]['summary'][m]) for mm in MM for m in first8}
        if r:
            for mm in MM:
                for k in ('msocp', f"lcmv_{r['lcmv_best_load']:g}"):
                    allci[(mm, k)] = ci(r['points'][str(mm)][k])
        wk = max(allci, key=allci.get)
        num('widest 95% bootstrap interval', allci[wk], 3.6)
        flag('the widest interval is on the covariance-reconstruction curve', wk[1] == 'recon', str(wk))
        num('widest WRAB interval', max(allci[(mm, 'wrab')] for mm in MM), 0.14, nd=2)
        conv = ('analog', 'wc', 'mvdr', 'rcb', 'socp', 'recon')
        num('conventional methods at alignment, lowest', min(mean(d['0'], m) for m in conv), 19.5)
        num('conventional methods at alignment, highest', max(mean(d['0'], m) for m in conv), 22.9)
        num('WRAB at alignment', mean(d['0'], 'wrab'), 15.9)
        num('single-direction methods at 2 deg, lowest', min(mean(d['2'], m) for m in conv), 2.2)
        num('single-direction methods at 2 deg, highest', max(mean(d['2'], m) for m in conv), 6.8)
        num('WRAB at 2 deg', mean(d['2'], 'wrab'), 18.3)
        num('WRAB lead over the best single-direction method at 2 deg',
            mean(d['2'], 'wrab') - max(mean(d['2'], m) for m in conv), 11.5)
        num('WRAB at 3 deg', mean(d['3'], 'wrab'), 13.6); num('WRAB at 4 deg', mean(d['4'], 'wrab'), 7.6)
        flag('WRAB has the highest mean of the first eight methods from 2 to 8 deg',
             all(mean(d[str(mm)], 'wrab') == max(mean(d[str(mm)], m) for m in first8) for mm in (2, 3, 4, 5, 6, 8)))
        S = lambda mm, m: d[str(mm)]['summary'][m]                       # noqa: E731
        over = {mm: [m for m in first8 if m != 'wrab' and S(mm, m)['ci95_high_db'] >= S(mm, 'wrab')['ci95_low_db']]
                for mm in (2, 3, 4, 5, 6, 8)}
        flag('only at 8 deg does the WRAB interval overlap another (the frequency-aware MVDR)',
             over[8] == ['mvdr'] and all(not over[mm] for mm in (2, 3, 4, 5, 6)), str(over))
        num('fixed-margin variant at alignment', mean(d['0'], 'fixed'), 22.3)
        num('fixed-margin variant at 2 deg', mean(d['2'], 'fixed'), 12.1)
        flag('covariance reconstruction is the best conventional method at 1 and 2 deg',
             all(max(conv, key=lambda m: mean(d[str(mm)], m)) == 'recon' for mm in (1, 2)))
        num('covariance reconstruction at 3 deg', mean(d['3'], 'recon'), -0.9)
        num('covariance reconstruction at 4 deg', mean(d['4'], 'recon'), -25.1)
        flag('robust Capon and the worst-case SOCP coincide (within 0.01 dB)',
             all(abs(mean(d[str(mm)], 'rcb') - mean(d[str(mm)], 'socp')) <= 0.01 for mm in MM))
    if d and r:
        P = r['points']; gk = f"lcmv_{r['lcmv_best_load']:g}"
        flag('exact multi-ball SOCP within 0.2 dB of WRAB at every point',
             all(abs(P[str(mm)]['msocp']['mean_db'] - mean(d[str(mm)], 'wrab')) <= 0.2 for mm in MM),
             f"largest {max(abs(P[str(mm)]['msocp']['mean_db'] - mean(d[str(mm)], 'wrab')) for mm in MM):.3f} dB")
        flag('online WRAB of this stage equals the WRAB curve of Figure 8.11',
             all(abs(P[str(mm)]['wrab']['mean_db'] - mean(d[str(mm)], 'wrab')) < 1e-9 for mm in MM))
        num('LCMV loading selected over 0-2 deg (times the noise power)', r['lcmv_best_load'], 100, nd=0)
        for mm in (0, 1, 2):
            num(f'loaded LCMV at {mm} deg', P[str(mm)][gk]['mean_db'], 17.1)
        num('loaded LCMV at 3 deg', P['3'][gk]['mean_db'], 15.5)
        num('loaded LCMV at 4 deg', P['4'][gk]['mean_db'], 7.6)
        g5 = [mean(d[str(mm)], 'wrab') - P[str(mm)][gk]['mean_db'] for mm in (5, 6, 8)]
        num('WRAB above the loaded LCMV from 5 deg, smallest', min(g5), 2.4)
        num('WRAB above the loaded LCMV from 5 deg, largest', max(g5), 5.0)
        info('one interior-point solve / one WRAB update (ms)',
             f"{r['time_ms']['socp_per_solve']:.1f} / {r['time_ms']['wrab_per_snapshot']:.2f}", '27.6 / 0.16')

    d = J('table3')
    if d:
        print('Table 8.3')
        T = {'S1': ((0.1, 4.5), (0.6, 4.5), (18.9, 0.1)), 'S2': ((1.1, 4.8), (1.7, 5.0), (7.3, 0.3)),
             'S3': ((-0.2, 1.7), (0.4, 1.6), (3.5, 0.3)), 'S4': ((1.9, 2.7), (2.4, 2.8), (17.7, 3.1))}
        for s, row in T.items():
            for m, (mq, sq) in zip(('analog', 'wc', 'wrab'), row):
                num(f'{s} {m} mean', d[s]['summary'][m]['mean_db'], mq, in_docx=False)
                num(f'{s} {m} standard deviation', d[s]['summary'][m]['std_db'], sq, in_docx=False)

    d = J('feasibility')
    if d:
        print('Sections 8.3.1 and 8.3.4')
        fz = d['frozen_norm']
        flag('frozen-norm correction: violated constraints in 296 of 300 updates',
             (fz['violating_updates'], fz['updates']) == (296, 300), f"{fz['violating_updates']} of {fz['updates']}")
        num('frozen-norm correction: largest violation', fz['max_violation'], 0.67, nd=2)
        num('frozen-norm correction: largest violation over the last 100 snapshots (x 1e-3)',
            1e3*fz['tail100_max_violation'], 4.9, in_docx=False)
        flag('exact projection on the same snapshots: residual below 1e-9', fz['exact_projection_max_residual'] <= 1e-9,
             f"{fz['exact_projection_max_residual']:.2e}")
        gp = [row['gap_percent'] for row in d['frozen_gap']]
        num('online objective above the frozen optimum, smallest (percent)', min(gp), 14, nd=0)
        num('online objective above the frozen optimum, largest (percent)', max(gp), 20, nd=0)
        flag('continued frozen iterations reach the independent SLSQP optimum (relative difference below 5e-5)',
             all(abs(row['frozen_vs_slsqp']) < 5e-5 for row in d['frozen_gap']),
             ', '.join(f"{row['frozen_vs_slsqp']:.1e}" for row in d['frozen_gap']))
    from .figures import max_residual
    mr = max_residual(datadir)
    if mr is not None:
        flag('largest constraint residual over all reported runs is 1.0e-9 (at most 1.0e-9)', mr <= 1.0e-9 and mr > 0.95e-9,
             f'{mr:.3e}')

    t = J('timing')
    if t:
        print('Timings (machine dependent, for information)')
        s = t['scaling_ms_per_frequency_snapshot']
        info('ms per subcarrier and snapshot, 64/128/256/512 elements',
             ' / '.join(f'{s[M]:.2f}' for M in ('64', '128', '256', '512')), '0.51 / 0.68 / 1.66 / 6.6')
        info('one WRAB update, Figure 8.11 setting (ms)', f"{t['wrab_update_ms']:.2f}", '0.16')
        info('one interior-point solve of (8.8) (ms)', f"{t['socp_solve_ms']:.1f}", '27.6')
        info('ratio', f"{t['socp_solve_ms']/t['wrab_update_ms']:.0f}", 'about 170')
        info('one WRAB update with 27 constraints at 512 elements (ms)', f"{t['wrab_update_ms_27_constraints_N512']:.2f}",
             '3.8')

    print(f'\n{COUNT[0] - len(FAILS)} of {COUNT[0]} checks passed' + (': all values match the chapter.' if not FAILS else
                                                                    '; mismatches: ' + '; '.join(FAILS)))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
