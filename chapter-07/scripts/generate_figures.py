"""Generate every Chapter 7 simulation figure (Figures 7.1-7.9), Tables 7.1 and 7.3-7.6 and results_summary.json.

    python scripts/generate_figures.py                     # full seed counts, two processes (about 30 minutes)
    python scripts/generate_figures.py --quick --outdir figures_quick   # smoke test with a few seeds
    python scripts/generate_figures.py --only 7_2 7_8      # selected figures and the stages they need
    python scripts/generate_figures.py --only tables text  # Tables 7.1, 7.3-7.6 and the numbers quoted in the text
    python scripts/generate_figures.py --replot            # redraw from <outdir>/data saved by an earlier run
    python scripts/generate_figures.py --verify            # then compare every number with the chapter

Stage data are written to <outdir>/data. The timing stage of Table 7.1 runs alone, after the other stages.
"""
import argparse
import os
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--quick', action='store_true', help='one or two seeds per point instead of the chapter counts')
p.add_argument('--replot', action='store_true', help='redraw from <outdir>/data saved by an earlier run (no simulation)')
p.add_argument('--outdir', default='figures', help='output folder (default: ./figures)')
p.add_argument('--only', nargs='*', default=None, help='subset, e.g. 7_2 7_8 tables text (default: everything)')
p.add_argument('--jobs', type=int, default=2, help='parallel processes for the simulations (default 2)')
p.add_argument('--verify', action='store_true', help='compare the regenerated numbers with the values quoted in the chapter')
args = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from ch7rdfisac import scenarios as S, summary, figures as F  # noqa: E402

NEEDS = {f'7_{k}': v for k, v in F.NEEDS.items()}
NEEDS['tables'] = ['main', 'array', 'load', 'sens', 'timing']
NEEDS['text'] = ['main', 'array', 'load', 'sweep', 'cross', 'alpha', 'grid', 'tightproj']

if __name__ == '__main__':
    outdir = os.path.abspath(args.outdir); datadir = os.path.join(outdir, 'data')
    keys = args.only or list(NEEDS)
    bad = [k for k in keys if k not in NEEDS]
    if bad:
        sys.exit('unknown key(s): ' + ' '.join(bad) + '; use 7_1 ... 7_9, tables or text')
    stages = [s for s in S.STAGES if any(s in NEEDS[k] for k in keys)]
    if not args.replot:
        S.run_stages(datadir, stages, jobs=args.jobs, quick=args.quick)
    figs = [int(k.split('_')[1]) for k in keys if k.startswith('7_')]
    for f in F.draw(datadir, outdir, figs):
        print('written', os.path.basename(f), flush=True)
    summary.write(datadir, outdir); print('results summary:', os.path.join(outdir, 'results_summary.json'))
    if args.verify:
        from ch7rdfisac import verify
        sys.exit(1 if verify.main([outdir]) else 0)
