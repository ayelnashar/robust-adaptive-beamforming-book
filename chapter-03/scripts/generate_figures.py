"""Generate every Chapter 3 simulation figure (Figures 3.1-3.14) and table (Tables 3.2-3.6).

    python scripts/generate_figures.py                 # full Monte Carlo counts
    python scripts/generate_figures.py --quick         # smoke test (a few minutes)
    python scripts/generate_figures.py --only 3_1 3_9  # selected figures (and the data they need)
    python scripts/generate_figures.py --replot        # redraw from the data saved by an earlier run
    python scripts/generate_figures.py --jobs 2        # simulations in parallel processes

Timing experiments (Figure 3.14, Table 3.4) always run alone after the other simulations, so that
no other process competes for the CPU while they are measured.
"""
import argparse
import os
import sys
import time

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--quick', action='store_true', help='two Monte Carlo trials per point instead of the chapter counts')
p.add_argument('--replot', action='store_true', help='redraw from <outdir>/data saved by an earlier run (no re-simulation)')
p.add_argument('--outdir', default='figures', help='output folder (default: ./figures)')
p.add_argument('--only', nargs='*', default=None, help='subset of figures, e.g. 3_1 3_9 (default: all figures and tables)')
p.add_argument('--jobs', type=int, default=1, help='parallel processes for the simulations (default 1)')
args = p.parse_args()

os.environ['QUICK'] = '1' if args.quick else '0'
os.environ['FIGDIR'] = args.outdir
os.environ['RAB_CACHE'] = '1' if args.replot else '0'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ch3dasmbwc import figures as F   # noqa: E402  (reads the environment at import)


def _worker(key):
    t = time.time()
    F.compute(key)
    return key, time.time() - t


if __name__ == '__main__':
    funcs = F.PART_FIGURES + F.PART_TABLES
    if args.only:
        funcs = [f for f in F.PART_FIGURES if any(f.__name__ == 'fig_' + k for k in args.only)]
        if not funcs:
            sys.exit('no figure function matches --only ' + ' '.join(args.only))
    if not args.replot and not args.only:
        t0 = time.time()
        if args.jobs > 1:
            from multiprocessing import Pool
            with Pool(args.jobs) as pool:
                for key, dt in pool.imap_unordered(_worker, F.SIM_TASKS):
                    F.FRESH.add(key)
                    print(f'{key}: {dt:.0f} s', flush=True)
        else:
            for key in F.SIM_TASKS:
                print(f'{_worker(key)[0]}: done', flush=True)
        for key in F.TIMING_TASKS:
            print(f'{_worker(key)[0]}: done', flush=True)
        print(f'simulations finished in {time.time() - t0:.0f} s', flush=True)
    for f in funcs:
        t = time.time()
        f()
        print(f'{f.__name__}: {time.time() - t:.0f} s', flush=True)
    if not args.only:
        print('figures and results_summary.json written to', F.write_summary())
