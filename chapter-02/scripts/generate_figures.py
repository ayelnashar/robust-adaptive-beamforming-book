"""Generate the Chapter 2 figures (Figures 2.1-2.8) and results_summary.json without Jupyter.

    python scripts/generate_figures.py                   # draw from the archived measurement run (seconds)
    python scripts/generate_figures.py --rerun           # re-run every experiment (about 1 min on two cores)
    python scripts/generate_figures.py --quick           # reduced re-run, smoke test (about 10 s)
    python scripts/generate_figures.py --only 2_1 2_8    # selected figures

The default draws the figures of the chapter from the records in data/archived_run. With --rerun
or --quick the experiments are measured again with fixed seeds; the new records are saved next to
the figures (records/ in the output folder). Timings depend on the machine; accuracy should not.
MOSEK needs a licence and Gurobi's free licence limits model size; unavailable solvers are skipped.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')      # one BLAS thread, as in the chapter
os.environ.setdefault('OMP_NUM_THREADS', '1')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
from ch2assnc import figures as F  # noqa: E402
from ch2assnc.scenarios import run_all  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--rerun', action='store_true', help='measure every experiment again (full counts)')
p.add_argument('--quick', action='store_true', help='measure again with reduced counts (smoke test)')
p.add_argument('--outdir', default=None,
               help='output folder (default: figures; figures_rerun with --rerun; figures_quick with --quick)')
p.add_argument('--records', default=os.path.join(ROOT, 'data', 'archived_run'),
               help='records to draw from when not re-running (default: data/archived_run)')
p.add_argument('--only', nargs='*', default=None, help='subset of figures, e.g. 2_1 2_5 (default: all)')
args = p.parse_args()

rerun = args.rerun or args.quick
outdir = args.outdir or ('figures_quick' if args.quick else 'figures_rerun' if args.rerun else 'figures')

if rerun:
    t0 = time.time()
    rec = run_all(quick=args.quick, progress=lambda s: print(s, flush=True))
    F.save_records(rec, os.path.join(outdir, 'records'))
    print(f'experiments finished in {time.time() - t0:.0f} s', flush=True)
else:
    rec = F.load_records(args.records)

funcs = F.FIGURES
if args.only:
    funcs = [f for f in funcs if any(f.__name__ == 'fig_' + k for k in args.only)]
    if not funcs:
        sys.exit('no figure function matches --only ' + ' '.join(args.only))
for f in funcs:
    path = f(rec, outdir)
    print(f'{f.__name__}: {path}', flush=True)

summary = F.summarize(rec)
summary['source'] = 'new run (reduced counts)' if args.quick else 'new run' if args.rerun else 'archived run'
with open(os.path.join(outdir, 'results_summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=1, default=float)
print('figures and results_summary.json written to', outdir)
