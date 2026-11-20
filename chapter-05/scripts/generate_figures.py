"""Generate every Chapter 5 simulation figure (Figures 5.1-5.10) and results_summary.json.

    python scripts/generate_figures.py                     # full Monte Carlo counts, two processes (about 5 hours)
    python scripts/generate_figures.py --quick --outdir figures_quick   # smoke test (about 30 minutes)
    python scripts/generate_figures.py --only 5_1 5_7      # selected figures and the stages they need
    python scripts/generate_figures.py --replot            # redraw from <outdir>/data saved by an earlier run
    python scripts/generate_figures.py --timing            # also time this machine and draw Figure 5.10 from it
    python scripts/generate_figures.py --retrain           # also rerun the linear-array training phase and use the
                                                           # selections derived from both training phases

Each stage runs in its own process and writes its data and log into <outdir>/data.  Figure 5.10 is drawn from the
timing recorded for the chapter (Table 5.1) unless --timing is given; the timing stage then runs alone, after the other
simulations, so that no other process competes for the CPU while it is measured.  The planar training phase always
runs, since Figure 5.8 is drawn from it; by default the selection of the chapter (ch5mprrab/data/selection.json) is
used for the evaluation and the selection derived from the training phase is only compared with it.
"""
import argparse
import os
import sys

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--quick', action='store_true', help='2 runs per cell (1 for training) instead of the chapter counts')
p.add_argument('--replot', action='store_true', help='redraw from <outdir>/data saved by an earlier run (no simulation)')
p.add_argument('--outdir', default='figures', help='output folder (default: ./figures)')
p.add_argument('--only', nargs='*', default=None, help='subset of figures, e.g. 5_1 5_7 (default: all figures)')
p.add_argument('--jobs', type=int, default=2, help='parallel processes for the simulations (default 2)')
p.add_argument('--timing', action='store_true', help='measure the timing of Table 5.1 / Figure 5.10 on this machine')
p.add_argument('--retrain', action='store_true', help='rerun both training phases and use the derived selections')
args = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
outdir = os.path.abspath(args.outdir)
datadir = os.path.join(outdir, 'data')
os.environ['FIGDIR'] = outdir
os.environ['CH5_DATA'] = datadir
os.environ['CH5_LOCAL_TIMING'] = '1' if args.timing else os.environ.get('CH5_LOCAL_TIMING', '0')
from ch5mprrab import scenarios as S   # noqa: E402

if __name__ == '__main__':
    keys = args.only or list(S.NEEDS)
    bad = [k for k in keys if k not in S.NEEDS]
    if bad:
        sys.exit('unknown figure(s): ' + ' '.join(bad) + '; use keys such as 5_1 ... 5_10')
    if not args.replot:
        if args.retrain:
            tr = [('ula_train_%s' % g, ['study', 'ula', 'train', g, '{NR_TL}'], [], 5) for g in 'ABCDEFGHI']
            tr += [s for s in S.STAGES if s[0].startswith('upa_train')]
            S.run_stages(datadir, stages=tr, jobs=args.jobs, quick=args.quick)
            S.run_stages(datadir, stages=[('ula_select', ['select', 'ula', '--use'], [], 1),
                                          ('upa_select', ['select', 'upa', '--use'], [], 1)], jobs=args.jobs)
        stages = S.STAGES if not args.only else S.stages_for(keys)
        S.run_stages(datadir, stages=stages, jobs=args.jobs, quick=args.quick)
        if args.timing:
            S.run_stages(datadir, stages=[S.TIMING], jobs=1, quick=args.quick)
    from ch5mprrab import figures as F   # noqa: E402  (reads the environment at import)
    for k in keys:
        getattr(F, 'fig_' + k)()
        print('Figure', k.replace('_', '.'), 'written', flush=True)
    if not args.only:
        print('results summary:', F.write_summary())
