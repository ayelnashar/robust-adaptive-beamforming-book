"""
ch2assnc.figures -- one function per figure of Chapter 2 (Figures 2.1-2.8) and the results summary.

Every function draws from a dict of records in the format of ``data/archived_run`` (keys 'scaling',
'conditioning', 'pareto', 'famafrench', 'supplement', 'portfolio', 'environment'), so the same code
draws the archived measurement run behind the chapter or a new run on another machine.
Figures are written as PNG (300 dpi) and PDF. Set ``INLINE = True`` (or CH2_INLINE=1) to also show
them under a Jupyter cell.
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .scenarios import METHODS, label  # noqa: E402

INLINE = os.environ.get('CH2_INLINE') == '1'
RECORD_FILES = ['scaling', 'conditioning', 'pareto', 'famafrench', 'supplement', 'portfolio',
                'environment']
# colour and marker per method; legend order follows the chapter
STYLES = {'AS-SNC': ('#111111', 'o'), 'SCS': ('#0072B2', 's'), 'ECOS': ('#D55E00', '^'),
          'CLARABEL': ('#009E73', 'D'), 'MOSEK': ('#8C564B', 'P'), 'GUROBI': ('#CC79A7', 'v')}
RC = {'font.family': 'serif', 'font.size': 10, 'axes.grid': True, 'grid.alpha': .22,
      'mathtext.fontset': 'dejavuserif'}
FF_NAMES = ['FF-Ind12', 'FF-All21', 'FF-All30']
FILE_NAMES = {'2_1': 'Fig2_01_scaling', '2_2': 'Fig2_02_convergence',
              '2_3': 'Fig2_03_scale_invariance', '2_4': 'Fig2_04_active_set_cardinality',
              '2_5': 'Fig2_05_accuracy_versus_time', '2_6': 'Fig2_06_conditioning',
              '2_7': 'Fig2_07_portfolio', '2_8': 'Fig2_08_fama_french'}


# ----------------------------------------------------------------------------------------------
# records and helpers
# ----------------------------------------------------------------------------------------------
def load_records(folder):
    """Load a run (archived or new) from a folder of JSON files."""
    rec = {}
    for name in RECORD_FILES:
        path = os.path.join(folder, name + '.json')
        if os.path.exists(path):
            with open(path) as fh:
                rec[name] = json.load(fh)
    return rec


def save_records(rec, folder):
    os.makedirs(folder, exist_ok=True)
    for name, value in rec.items():
        with open(os.path.join(folder, name + '.json'), 'w') as fh:
            json.dump(value, fh, indent=2, default=float)


def _kappa(row):
    return int(row['case'].split('_k')[1].split('_')[0])


def _ms(rows, method):
    return [r['methods'][method]['ms'] for r in rows
            if method in r['methods'] and r['methods'][method].get('ms') is not None]


def _median(values):
    return float(np.median(values)) if len(values) else None


def _save(fig, outdir, key):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, FILE_NAMES[key])
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(base + '.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    if INLINE:
        import io
        from IPython.display import display, Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
        display(Image(data=buf.getvalue()))
    plt.close(fig)
    return base + '.png'


def _finish(fig, axes, handles=None, labels=None):
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        for sp in ax.spines.values():
            sp.set_visible(True)
    if handles is None:
        handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .01), ncol=3,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .17, 1, 1))


def _plain_log_axis(axis, values=None):
    """Log axis labelled with plain numbers: at the given values, or at 1-2-5 steps."""
    from matplotlib.ticker import FixedLocator, LogLocator, FuncFormatter, NullFormatter
    axis.set_major_locator(FixedLocator(values) if values is not None
                           else LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_formatter(NullFormatter())


def _timing_plot(rows, groups, key, xlabel, xvalues=False):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method, (color, marker) in STYLES.items():
        ys = [_median(_ms([r for r in rows if key(r) == g], method)) for g in groups]
        ys = [np.nan if y is None else y for y in ys]
        if any(np.isfinite(ys)):
            ax.plot(groups, ys, '-', marker=marker, color=color, label=label(method), lw=1.7, ms=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if xvalues:
        _plain_log_axis(ax.xaxis, groups)
    _plain_log_axis(ax.yaxis)
    ax.set(xlabel=xlabel, ylabel='Median end-to-end time (ms)')
    _finish(fig, ax)
    return fig


# ----------------------------------------------------------------------------------------------
# one function per chapter figure
# ----------------------------------------------------------------------------------------------
def fig_2_1(rec, outdir='figures'):
    """Figure 2.1: median end-to-end time versus decision dimension (dimension sweep)."""
    rows = rec['scaling']
    with plt.rc_context(RC):
        fig = _timing_plot(rows, sorted({r['M'] for r in rows}), lambda r: r['M'],
                           'Decision dimension $M$', xvalues=True)
        return _save(fig, outdir, '2_1')


def fig_2_2(rec, outdir='figures'):
    """Figure 2.2: scaled KKT residual of each recorded candidate for five instances."""
    runs = rec['supplement']['convergence']
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for j, r in enumerate(runs):
            res = np.maximum(r['residuals'], 1e-15)
            ax.semilogy(np.arange(1, len(res) + 1), res, '-o', ms=4, label=f'Instance {j + 1}')
        ax.set(xlabel='Recorded candidate (including Newton correction)',
               ylabel='Scaled KKT residual')
        _finish(fig, ax)
        return _save(fig, outdir, '2_2')


def fig_2_3(rec, outdir='figures'):
    """Figure 2.3: relative change of the solution when the cost matrix is scaled."""
    sc = rec['supplement']['scale']
    e, s = np.asarray(sc['errors']), np.asarray(sc['scales'])
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.loglog(s, np.maximum(np.median(e, axis=1), 1e-16), '-o', label='Median')
        ax.loglog(s, np.maximum(np.max(e, axis=1), 1e-16), '-s', label='Maximum')
        ax.set(xlabel='Positive objective scale', ylabel='Relative solution variation')
        _finish(fig, ax)
        return _save(fig, outdir, '2_3')


def fig_2_4(rec, outdir='figures'):
    """Figure 2.4: distribution of the optimal active-set cardinality for N = 20 and N = 50."""
    counts = rec['supplement']['counts']
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for N, color in [('20', '#0072B2'), ('50', '#D55E00')]:
            if N not in counts:
                continue
            a = counts[N]
            ax.hist(a, bins=np.arange(min(a) - .5, max(a) + 1.5), alpha=.7, color=color,
                    label=f'N = {N}; mean = {np.mean(a):.2f}')
        ax.set(xlabel='Active-set cardinality', ylabel='Number of instances')
        _finish(fig, ax)
        return _save(fig, outdir, '2_4')


def fig_2_5(rec, outdir='figures'):
    """Figure 2.5: time versus accuracy when every solver's tolerance is swept."""
    rows = rec['pareto']
    tols = sorted({r['tol'] for r in rows}, reverse=True)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
        for method, (color, marker) in STYLES.items():
            tt, ee, pp = [], [], []
            for tol in tols:
                rr = [r['methods'][method] for r in rows
                      if r['tol'] == tol and method in r['methods']
                      and r['methods'][method].get('ms') is not None]
                if not rr:
                    continue
                tt.append(np.median([x['ms'] for x in rr]))
                ee.append(max(np.median([x['relative_objective_difference'] for x in rr]), 1e-15))
                pp.append(max(np.median([x['primal'] for x in rr]), 1e-15))
            if tt:
                for ax, xx in zip(axes, [ee, pp]):
                    ax.semilogx(xx, tt, '-', marker=marker, color=color, label=label(method),
                                lw=1.5, ms=5)
        axes[0].set(xlabel='Median relative objective difference',
                    ylabel='Median end-to-end time (ms)', title='Independent SCS reference')
        axes[1].set(xlabel='Median scaled feasibility violation', title='Constraint feasibility')
        _finish(fig, axes)
        return _save(fig, outdir, '2_5')


def fig_2_6(rec, outdir='figures'):
    """Figure 2.6: median end-to-end time versus the condition number of the cost matrix."""
    rows = rec['conditioning']
    with plt.rc_context(RC):
        fig = _timing_plot(rows, sorted({_kappa(r) for r in rows}), _kappa,
                           'Cost-matrix condition number')
        return _save(fig, outdir, '2_6')


def fig_2_7(rec, outdir='figures'):
    """Figure 2.7: risk and expected return of the 50-asset allocation as the sector radius grows."""
    rows = rec['portfolio']
    k = np.array([r['kappa'] for r in rows])
    blue = '#0072B2'
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.2, 3.7))
        ax2 = ax.twinx()
        h1, = ax.semilogy(k, [r['risk'] for r in rows], '-o', color='#111111', ms=4, lw=1.6,
                          label='Portfolio risk (left axis)')
        h2, = ax2.semilogy(k, [r['ret'] for r in rows], '-^', color=blue, ms=4, lw=1.4,
                           label='Expected return (right axis)')
        ax2.axhline(.15, color=blue, lw=.9, ls=':')               # expected-return floor
        ax.axvline(1 / np.sqrt(50), color='#888888', lw=.9)       # feasibility limit 1/sqrt(50)
        ax.set_xlim(-.003, .145)
        ax.set_xlabel(r'Sector uncertainty radius $\kappa$')
        ax.set_ylabel(r'Portfolio risk $\sqrt{\boldsymbol{w}^{\mathrm{T}}\boldsymbol{\Sigma}'
                      r'\boldsymbol{w}}$')
        ax2.set_ylabel(r'Expected return $\boldsymbol{\mu}^{\mathrm{T}}\boldsymbol{w}$', color=blue)
        ax2.tick_params(axis='y', which='both', colors=blue)
        ax2.spines['right'].set_color(blue)
        ax2.grid(False)
        ax.grid(True, which='major', alpha=.25)
        fig.legend([h1, h2], [h1.get_label(), h2.get_label()], loc='lower center',
                   bbox_to_anchor=(.5, 0), ncol=2, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, .08, 1, 1))
        return _save(fig, outdir, '2_7')


def fig_2_8(rec, outdir='figures'):
    """Figure 2.8: median end-to-end time on the three Fama-French problems."""
    rows = rec['famafrench']
    names = [n for n in FF_NAMES if any(r['case'].startswith(n + '_') for r in rows)]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7, 4.1))
        x = np.arange(len(names))
        width = .13
        for j, (method, (color, marker)) in enumerate(STYLES.items()):
            ys = [_median(_ms([r for r in rows if r['case'].startswith(n + '_')], method))
                  for n in names]
            if all(y is None for y in ys):
                continue
            ax.bar(x + (j - 2.5) * width, [np.nan if y is None else y for y in ys], width,
                   color=color, label=label(method))
        ax.set_xticks(x, names)
        ax.set(ylabel='Median end-to-end time (ms)')
        _finish(fig, ax)
        return _save(fig, outdir, '2_8')


FIGURES = [fig_2_1, fig_2_2, fig_2_3, fig_2_4, fig_2_5, fig_2_6, fig_2_7, fig_2_8]


# ----------------------------------------------------------------------------------------------
# numbers quoted in the chapter
# ----------------------------------------------------------------------------------------------
def _timing_table(rows, groups, key):
    table = {}
    for g in groups:
        rr = [r for r in rows if key(r) == g]
        row = {'instances': len(rr)}
        for m in METHODS:
            vals = [r['methods'][m] for r in rr if m in r['methods']
                    and r['methods'][m].get('ms') is not None]
            row[label(m)] = {'median_ms': _median([v['ms'] for v in vals]), 'returned': len(vals),
                             'passes_common_target': sum(v.get('passes_common_target', False)
                                                         for v in vals)}
        act = [r['methods']['AS-SNC'].get('active') for r in rr
               if r['methods'].get('AS-SNC', {}).get('active') is not None]
        row['mean_active_set'] = float(np.mean(act)) if act else None
        a, mo = row['AS-SNC']['median_ms'], row['MOSEK']['median_ms']
        row['MOSEK/AS-SNC'] = (mo / a) if (a and mo) else None
        table[str(g)] = row
    return table


def _solver_only(rows, key):
    """Solve time measured by each backend (CVXPY construction excluded); AS-SNC stays end to end."""
    groups = {}
    for r in rows:
        g = groups.setdefault(str(key(r)), {})
        for m, v in r['methods'].items():
            x = v.get('ms') if m == 'AS-SNC' else (None if v.get('backend_seconds') is None
                                                     else 1e3 * v['backend_seconds'])
            if x is not None:
                g.setdefault(label(m), []).append(x)
    return {k: {m: float(np.median(x)) for m, x in g.items()} for k, g in groups.items()}


def summarize(rec):
    """Every number the chapter quotes from the experiments, computed from a set of records."""
    out = {}
    if 'scaling' in rec:
        rows = rec['scaling']
        out['Table2.3_Fig2.1_dimension_sweep'] = _timing_table(
            rows, sorted({r['M'] for r in rows}), lambda r: r['M'])
    if 'supplement' in rec:
        sp = rec['supplement']
        out['Fig2.2_convergence'] = [
            dict(instance=j + 1, recorded_candidates=len(r['residuals']),
                 outer_iterations=r['iterations'], newton_steps=r['newton_steps'],
                 final_residual=r['residuals'][-1], method=r['method'])
            for j, r in enumerate(sp['convergence'])]
        e = np.asarray(sp['scale']['errors'])
        out['Fig2.3_scale_invariance'] = dict(
            scales=sp['scale']['scales'], median=np.median(e, axis=1).tolist(),
            maximum=np.max(e, axis=1).tolist(), overall_maximum=float(e.max()))
        out['Fig2.4_active_set_cardinality'] = {
            f'N={N}': dict(instances=len(a), mean=float(np.mean(a)), minimum=int(min(a)),
                           maximum=int(max(a))) for N, a in sp['counts'].items()}
        checks = []
        for c in sp['regression']:
            c = dict(c)
            if 'counterexample' in str(c.get('case', '')):
                c['case'] = 'repelling fixed point (Section 2.4.2)'
            if c.get('status') == 'correctly rejected':
                c['status'] = 'infeasible, no point returned'
            checks.append(c)
        out['checks'] = dict(scalar_and_two_dimensional=checks,
                             near_boundary=sp['near_boundary_diagnostic'])
    if 'pareto' in rec:
        rows = rec['pareto']
        tab = {}
        for tol in sorted({r['tol'] for r in rows}, reverse=True):
            tab[f'{tol:g}'] = {}
            for m in METHODS:
                rr = [r['methods'][m] for r in rows if r['tol'] == tol and m in r['methods']
                      and r['methods'][m].get('ms') is not None]
                if rr:
                    tab[f'{tol:g}'][label(m)] = dict(
                        median_ms=_median([x['ms'] for x in rr]),
                        median_relative_objective_difference=_median(
                            [x['relative_objective_difference'] for x in rr]),
                        median_feasibility_violation=_median([x['primal'] for x in rr]),
                        passes_common_target=sum(x.get('passes_common_target', False)
                                                 for x in rr), returned=len(rr))
        out['Fig2.5_tolerance_sweep'] = tab
    if 'conditioning' in rec:
        rows = rec['conditioning']
        out['Fig2.6_conditioning_sweep'] = _timing_table(rows, sorted({_kappa(r) for r in rows}),
                                                         _kappa)
    if 'portfolio' in rec:
        rows = rec['portfolio']
        out['Fig2.7_portfolio'] = dict(
            feasibility_limit=1 / np.sqrt(50),
            points=[{k: r[k] for k in ('kappa', 'risk', 'ret', 'norm', 'active', 'method')}
                    for r in rows],
            all_spectral_path=all(r['method'] == 'AS-SNC' for r in rows),
            max_relative_objective_difference_to_clarabel=max(r['rel_obj_diff'] for r in rows))
    if 'famafrench' in rec:
        rows = rec['famafrench']
        names = [n for n in FF_NAMES if any(r['case'].startswith(n + '_') for r in rows)]
        out['Table2.4_Fig2.8_fama_french'] = _timing_table(
            rows, names, lambda r: r['case'].rsplit('_', 1)[0])
    timed = [k for k in ('scaling', 'conditioning', 'pareto', 'famafrench') if k in rec]
    out['provenance_of_AS-SNC_results'] = {
        k: {m: sum(r['methods']['AS-SNC'].get('method') == m for r in rec[k])
            for m in sorted({r['methods']['AS-SNC'].get('method', 'failed') for r in rec[k]})}
        for k in timed}
    if 'scaling' in rec and 'famafrench' in rec:
        so = {f'M = {k}': v for k, v in _solver_only(rec['scaling'], lambda r: r['M']).items()}
        so.update(_solver_only(rec['famafrench'], lambda r: r['case'].rsplit('_', 1)[0]))
        out['supplementary_backend_solve_time_ms'] = so
    if 'environment' in rec:
        out['environment'] = rec['environment']
    return out
