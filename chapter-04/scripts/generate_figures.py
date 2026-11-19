"""Generate every Chapter 4 simulation figure (Figures 4.1-4.11) and results_summary.json.

    python scripts/generate_figures.py                     # full Monte Carlo counts, two processes
    python scripts/generate_figures.py --quick --outdir figures_quick   # smoke test (a few minutes)
    python scripts/generate_figures.py --only 4_2 4_9      # selected figures and the experiments they need
    python scripts/generate_figures.py --replot            # redraw from <outdir>/data saved by an earlier run
    python scripts/generate_figures.py --timing            # also time this machine and draw Figure 4.1 from it

Each experiment stage runs in its own process and writes its data and log into <outdir>/data. Figure 4.1 is drawn
from the runtime recorded for the chapter unless --timing is given; the timing stage then runs alone, after the
other simulations, so that no other process competes for the CPU while it is measured.
"""
import argparse
import os
import sys

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--quick', action='store_true', help='at most two Monte Carlo trials per point instead of the chapter counts')
p.add_argument('--replot', action='store_true', help='redraw from <outdir>/data saved by an earlier run (no simulation)')
p.add_argument('--outdir', default='figures', help='output folder (default: ./figures)')
p.add_argument('--only', nargs='*', default=None, help='subset of figures, e.g. 4_2 4_9 (default: all figures)')
p.add_argument('--jobs', type=int, default=2, help='parallel processes for the simulations (default 2)')
p.add_argument('--timing', action='store_true', help='measure the runtime of Figure 4.1 on this machine')
args = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
outdir = os.path.abspath(args.outdir)
os.environ['FIGDIR'] = outdir
os.environ['RAB_LOCAL_TIMING'] = '1' if args.timing else os.environ.get('RAB_LOCAL_TIMING', '0')
from ch4rmber import scenarios as S   # noqa: E402
from ch4rmber import figures as F     # noqa: E402  (reads the environment at import)

if __name__ == '__main__':
    datadir = os.path.join(outdir, 'data')
    keys = args.only or list(S.NEEDS)
    bad = [k for k in keys if k not in S.NEEDS]
    if bad:
        sys.exit('unknown figure(s): ' + ' '.join(bad) + '; use keys such as 4_1 ... 4_11')
    if not args.replot:
        groups = S.GROUPS if not args.only else S.groups_for(keys)
        S.run_groups(datadir, groups=groups, jobs=args.jobs, quick=args.quick)
        if args.timing:
            S.run_groups(datadir, groups=[S.TIMING], jobs=1, quick=args.quick)
    for k in keys:
        getattr(F, 'fig_' + k)()
        print('Figure', k.replace('_', '.'), 'written', flush=True)
    if not args.only:
        print('results summary:', F.write_summary())
