"""Generate every Chapter 8 simulation figure (Figures 8.1-8.11), Tables 8.1-8.3 and results_summary.json.

    python scripts/generate_figures.py                     # chapter trial counts, two processes
    python scripts/generate_figures.py --quick --outdir figures_quick   # smoke test with two or three trials per point
    python scripts/generate_figures.py --only 8_7 8_11     # selected figures and the experiments they need
    python scripts/generate_figures.py --only tables text  # Tables 8.1-8.3 and the results quoted in the text
    python scripts/generate_figures.py --replot            # redraw from <outdir>/data saved by an earlier run
    python scripts/generate_figures.py --verify            # then compare every number with the chapter

Each experiment stage runs in its own process and writes its data and log into <outdir>/data. The timing stage runs
alone, after the other stages, so that no other process competes for the CPU while it is measured.
"""
import argparse
import os
import sys

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--quick', action='store_true', help='two or three trials per point instead of the chapter counts')
p.add_argument('--replot', action='store_true', help='redraw from <outdir>/data saved by an earlier run (no simulation)')
p.add_argument('--outdir', default='figures', help='output folder (default: ./figures)')
p.add_argument('--only', nargs='*', default=None, help='subset, e.g. 8_7 8_11 tables text (default: everything)')
p.add_argument('--jobs', type=int, default=2, help='parallel processes for the simulations (default 2)')
p.add_argument('--verify', action='store_true', help='compare the regenerated numbers with the values quoted in the chapter')
args = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
outdir = os.path.abspath(args.outdir)
os.environ['FIGDIR'] = outdir
from ch8wrab import scenarios as S   # noqa: E402
from ch8wrab import figures as F     # noqa: E402  (reads the environment at import)

if __name__ == '__main__':
    datadir = os.path.join(outdir, 'data')
    keys = args.only or list(S.NEEDS)
    bad = [k for k in keys if k not in S.NEEDS]
    if bad:
        sys.exit('unknown key(s): ' + ' '.join(bad) + '; use 8_1 ... 8_11, tables or text')
    if not args.replot:
        S.run_groups(datadir, groups=S.groups_for(keys), jobs=args.jobs, quick=args.quick)
    for k in keys:
        if k == 'tables':
            F.tables(); print('Tables 8.1-8.3 written', flush=True)
        elif k == 'text':
            F.text_numbers()
        else:
            getattr(F, 'fig_' + k)(); print('Figure', k.replace('_', '.'), 'written', flush=True)
    if 'text' not in keys:
        F.text_numbers()
    print('results summary:', F.write_summary())
    if args.verify:
        from ch8wrab import verify
        sys.exit(1 if verify.main([outdir]) else 0)
