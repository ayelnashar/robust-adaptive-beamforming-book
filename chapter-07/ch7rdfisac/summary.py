"""Summaries of the stage data: results_summary.json (every number quoted in Chapter 7) and Tables 7.1, 7.3-7.6
as CSV and Markdown. Values are means of per-scene dB metrics; intervals are 95% Student-t intervals."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats, integrate
from scipy.stats import ncx2

r2 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 2)
L4 = lambda a: [r2(v) for v in a]
SCEN = 'ABCDFG'


def ci(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]; n = len(d); mu = d.mean()
    h = stats.t.ppf(.975, n-1)*d.std(ddof=1)/np.sqrt(n) if n > 1 else float('nan')
    return dict(mean=r2(mu), lo=r2(mu-h), hi=r2(mu+h), n=n)


def pd_swerling(g, tau, k):
    """Square-law detection probability on one look: k = 0 Swerling 0 (7.30); k = 1, 2 gamma-distributed cross
    section with 2k degrees of freedom (Swerling 1 and 3), averaged by numerical integration (7.31)."""
    if k == 0: return ncx2.sf(tau, 2, 2*g)
    return integrate.quad(lambda x: ncx2.sf(tau, 2, 2*g*x)*stats.gamma.pdf(x, k, scale=1/k), 0, np.inf, limit=200)[0]


def gsum(rows, key):
    grp = {}
    for r in rows: grp.setdefault(key(r), []).append(r)
    res = {}
    for k, rs in grp.items():
        U = len(rs[0]['metrics']['bound']); u = U//2 if 'U' not in rs[0] else rs[0]['U']
        d = {}
        for a in rs[0]['metrics']:
            v = np.array([r['metrics'][a] for r in rs], float)
            d[a] = dict(user=r2(np.nanmean(v[:, :u])), target=r2(np.nanmean(v[:, u:])),
                        user_min=r2(np.nanmin(v[:, :u])), n_user_below0=int(np.sum(np.nanmin(v[:, :u], 1) < 0)),
                        per=L4(np.nanmean(v, 0)))
        res[str(k)] = d
    return res


def summarize(datadir):
    D = Path(datadir); out = {}
    ld = lambda n: json.loads((D/f'{n}.json').read_text()) if (D/f'{n}.json').exists() else None
    main = ld('main')
    if main:
        METH = sorted(main[0]['metrics'])
        means, paired, pav = {}, {}, {}
        for sp in ('reference', 'heldout'):
            means[sp], paired[sp], pav[sp] = {}, {}, {}
            for sc in SCEN:
                rows = [r for r in main if r['split'] == sp and r['scenario'] == sc]
                if not rows: continue
                A = {a: np.array([r['metrics'][a] for r in rows], float) for a in METH}
                means[sp][sc] = {a: L4(np.nanmean(v, 0)) for a, v in A.items()}
                paired[sp][sc] = {a: [ci(A['rdf_as_isac'][:, i]-A[a][:, i]) for i in range(4)] for a in METH if a != 'rdf_as_isac'}
                g = lambda a, sl: np.array([np.mean(r['metrics'][a][sl]) for r in rows], float)
                pav[sp][sc] = {a: dict(user=ci(g('rdf_as_isac', slice(0, 2)) - g(a, slice(0, 2))),
                                       target=ci(g('rdf_as_isac', slice(2, 4)) - g(a, slice(2, 4)))) for a in METH if a != 'rdf_as_isac'}
        out['main_means'] = means; out['main_paired'] = paired; out['main_paired_avg'] = pav
        out['sdp_socp_maxdiff'] = float(max(np.nanmax(abs(np.array(r['metrics']['sdp'], float) - np.array(r['metrics']['socp'], float))) for r in main))
        below = {}
        for a in METH:
            v = np.array([r['metrics'][a][:2] for r in main if r['scenario'] in 'ABCD'], float)
            below[a] = int(np.sum(np.nanmin(v, 1) < 0))
        out['users_below_0dB_AD'] = below; out['n_AD'] = sum(1 for r in main if r['scenario'] in 'ABCD')
        it = np.concatenate([r['solver_iterations'] for r in main])
        out['checks'] = dict(min_margin=float(min(r['min_margin'] for r in main)), max_cert=float(max(r['max_cert'] for r in main)),
                             max_gap=float(max(r['objective_gap'] for r in main)),
                             max_gap_AD=float(max(r['objective_gap'] for r in main if r['scenario'] in 'ABCD')),
                             median_iter=int(np.median(it)), frac_iter_cap=float(np.mean(it >= 300)), n_scenes=len(main),
                             n_weights=len(main)*300)
        rowsD = [r for r in main if r['scenario'] == 'D']; out['detection_by_pfa'] = {}
        for pfa in (1e-6, 1e-4):
            tau = -2*np.log(pfa); det = {}
            for a in ['rdf_as_isac', 'projection', 'socp', 'socp_radius', 'socp_radius_online', 'fixed', 'vu_mbwc', 'true_cov']:
                g = 10**(np.array([r['metrics'][a][2:] for r in rowsD], float)/10); res = {}
                for name, k in [('sw0', 0), ('sw1', 1), ('sw3', 2)]:
                    P = np.array([[pd_swerling(x, tau, k) for x in row] for row in g])
                    res[name] = dict(worst=r2(P.min(1).mean()), joint=r2(P.prod(1).mean()))
                det[a] = res
            out['detection_by_pfa'][f'{pfa:.0e}'] = det
            out['swerling1_closed_form_error'] = float(max(abs(pd_swerling(g, tau, 1) - pfa**(1/(1+g))) for g in (1, 10, 30, 100)))
        out['n_detection'] = len(rowsD)
    arr = ld('array')
    if arr: out['array'] = gsum(arr, lambda r: (r['Mx']*r['My'], r['strong']))
    load = ld('load')
    if load: out['load'] = gsum(load, lambda r: (r['U'], r['strong']))
    sw = ld('sweep')
    if sw: out['sweep'] = {kind: gsum([r for r in sw if r['kind'] == kind], lambda r: r['val']) for kind in ('separation', 'inr', 'mismatch')}
    cr = ld('cross')
    if cr: out['cross'] = gsum(cr, lambda r: r['delta'])
    se = ld('sens')
    if se:
        out['sens'] = {}
        for sc in 'DFG':
            rs = [r for r in se if r['scenario'] == sc]
            out['sens'][sc] = {k: dict(user=r2(np.mean([r['metrics'][k][:2] for r in rs])), target=r2(np.mean([r['metrics'][k][2:] for r in rs])),
                                       per=L4(np.mean([r['metrics'][k] for r in rs], 0))) for k in rs[0]['metrics']}
    al = ld('alpha')
    if al:
        out['alpha'] = {str(a): dict(user=r2(np.mean([r['metrics'][:2] for r in al if r['alpha'] == a])),
                                     target=r2(np.mean([r['metrics'][2:] for r in al if r['alpha'] == a])))
                        for a in sorted({r['alpha'] for r in al})}
    gr = ld('grid')
    if gr:
        cov, fails = {}, {}
        for a in gr[0]['metrics']:
            nn = 0; fl = []
            for c in range(7):
                for s in range(7):
                    v = np.array([r['metrics'][a] for r in gr if r['c'] == c and r['s'] == s], float).mean(0)
                    ok = bool(v[0] >= -3 and v[1] >= -3 and v[2] >= 3 and v[3] >= 3); nn += ok
                    if not ok: fl.append([c, s])
            cov[a] = int(nn); fails[a] = fl
        out['coverage'] = cov; out['coverage_fails'] = fails
    tp = ld('tightproj')
    if tp and main:
        out['exact_projection_gain'] = {}
        for sc in 'FG':
            rows = {r['seed']: r for r in main if r['split'] == 'heldout' and r['scenario'] == sc}
            ex = {r['seed']: r for r in tp if r['scenario'] == sc}
            ss = sorted(set(rows) & set(ex))
            it = [np.mean(rows[s]['metrics']['rdf_as_isac'][:2]) - np.mean(rows[s]['metrics']['projection'][:2]) for s in ss]
            ti = [np.mean(rows[s]['metrics']['rdf_as_isac'][:2]) - np.mean(ex[s]['metrics'][:2]) for s in ss]
            out['exact_projection_gain'][sc] = dict(iterative=r2(np.mean(it)), exact=r2(np.mean(ti)))
    tm = ld('timing')
    if tm: out['timing_vs_M'] = tm
    return out


TAB3 = [('rdf_as_isac', 'RDF-AS-ISAC'), ('projection', 'Projection reconstr.'), ('socp', 'SOCP ν0'), ('socp_radius', 'SOCP 3ν0'),
        ('socp_online', 'SOCP ν0, online'), ('socp_radius_online', 'SOCP 3ν0, online'), ('fixed', 'Fixed beam'),
        ('true_cov', 'True cov.'), ('bound', 'MVDR bound')]
TAB4 = [('projection', 'vs Projection reconstr.'), ('socp_radius', 'vs SOCP 3ν0, batch'),
        ('socp_radius_online', 'vs SOCP 3ν0, online'), ('true_cov', 'vs True covariance')]
TAB5 = [('rdf_as_isac', 'RDF-AS-ISAC'), ('projection', 'Projection reconstr.'), ('socp', 'SOCP ν0'), ('socp_radius', 'SOCP 3ν0'),
        ('fixed', 'Fixed beam'), ('true_cov', 'True cov.'), ('bound', 'MVDR bound')]
TAB6 = [('default', 'Default'), ('win_az=6', 'Window ±6°'), ('win_az=12', 'Window ±12°'), ('eta=0.94', 'η = 0.94'),
        ('eta=0.98', 'η = 0.98'), ('T=5', 'T_r = 5'), ('T=20', 'T_r = 20'), ('tol=3', 'Tolerance ±3 dB'),
        ('tol=10', 'Tolerance ±10 dB'), ('beta_v=0.1', 'Rate gain 0.1'), ('beta_v=0.5', 'Rate gain 0.5'),
        ('alpha_p=0.3', 'Position gain 0.3'), ('alpha_p=1', 'Position gain 1'), ('nu_mult=2', 'Radius 2ν0'),
        ('nu_mult=4', 'Radius 4ν0')]
fmt = lambda x: '' if x is None else f'{x:+.2f}'.replace('-', '−')
f1 = lambda x: '' if x is None else f'{x:.1f}'.replace('-', '−')


def tables(N):
    """Rows of Tables 7.1 and 7.3-7.6 as lists of lists (first row = header)."""
    T = {}
    if 'timing_vs_M' in N:
        T['Table7_1_runtime'] = [['M', 'Array', 'RDF-AS-ISAC, 300 obs. (ms)', 'Scan per refresh (ms)', 'Solve per refresh (ms)',
                                  'Update per obs. (µs)', 'Projection, 300 obs. (ms)', 'SOCP, one solve (ms)']] + \
            [[r['M'], f"{r['Mx']} × {r['My']}", round(r['rdf_pass']*1e3), f"{r['scan']*1e3:.2f}", f"{r['solve']*1e3:.2f}",
              round(r['update']*1e6), round(r['projection_pass']*1e3) if r['projection_pass'] else '—', f"{r['socp_solve']*1e3:.1f}"]
             for r in N['timing_vs_M']]
    if 'main_means' in N:
        ref = N['main_means']['reference']
        T['Table7_3_reference_means'] = [['Scen.', 'Metric'] + [t for _, t in TAB3]] + \
            [[sc if i == 0 else '', f'γ_{m} (dB)'] + [fmt(ref[sc][a][i]) for a, _ in TAB3]
             for sc in SCEN if sc in ref for i, m in enumerate(['u1', 'u2', 't1', 't2'])]
        pv = N['main_paired_avg']['heldout']
        cell = lambda c: '' if c['mean'] is None else f"{fmt(c['mean'])} [{c['lo']:.2f}, {c['hi']:.2f}]".replace('-', '−') if c['lo'] is not None else fmt(c['mean'])
        T['Table7_4_heldout_paired'] = [['Scen.', 'Metric'] + [t for _, t in TAB4]] + \
            [[sc if j == 0 else '', lab] + [cell(pv[sc][a][key]) for a, _ in TAB4]
             for sc in SCEN if sc in pv for j, (key, lab) in enumerate([('user', 'Users'), ('target', 'Targets')])]
    if 'array' in N and 'load' in N:
        rows = [['Case', 'Interference'] + [t for _, t in TAB5]]
        for grp, lab in [('array', 'M = {}'), ('load', 'U = T = {}')]:
            for key, v in sorted(N[grp].items(), key=lambda kv: tuple(int(x) for x in kv[0].strip('()').split(','))):
                a, st = (int(x) for x in key.strip('()').split(','))
                rows.append([lab.format(a), 'strong' if st else 'benign'] + [f"{f1(v[m]['user'])} / {f1(v[m]['target'])}" for m, _ in TAB5])
        T['Table7_5_random_geometry'] = rows
    if 'sens' in N:
        T['Table7_6_sensitivity'] = [['Setting', 'Scenario D', 'Scenario F', 'Scenario G']] + \
            [[lab] + [f"{N['sens'][sc][k]['user']:.2f} / {N['sens'][sc][k]['target']:.2f}" for sc in 'DFG'] for k, lab in TAB6]
    return T


def write(datadir, outdir):
    """Write results_summary.json and the table files into outdir; return the summary."""
    N = summarize(datadir); outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir/'results_summary.json').write_text(json.dumps(N, indent=1))
    for name, rows in tables(N).items():
        with open(outdir/f'{name}.csv', 'w', newline='', encoding='utf-8') as f: csv.writer(f).writerows(rows)
        md = ['| ' + ' | '.join(map(str, rows[0])) + ' |', '|' + '---|'*len(rows[0])] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows[1:]]
        (outdir/f'{name}.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return N
