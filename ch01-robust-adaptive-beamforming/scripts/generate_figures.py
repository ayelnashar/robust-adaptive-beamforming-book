"""Generate every Chapter 1 simulation figure (Figs. 1.5-1.27) without Jupyter.

    python scripts/generate_figures.py                 # full Monte Carlo counts (about 45 min on one core)
    python scripts/generate_figures.py --quick         # smoke test (about 2 min)
    python scripts/generate_figures.py --only 1_17 1_27
    python scripts/generate_figures.py --socp-radius original
"""
import argparse, os, sys, time, json

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--quick', action='store_true', help='3/3/2 Monte Carlo runs instead of 50/100/50')
p.add_argument('--socp-radius', choices=['matched', 'original'], default='matched',
               help="'matched': SOCP uses the same uncertainty radius as RCB/WC (default); "
                    "'original': the radius the MATLAB scripts effectively used")
p.add_argument('--outdir', default='figures', help='output folder (default: ./figures)')
p.add_argument('--only', nargs='*', default=None,
               help='subset of figure functions, e.g. 1_5_9 1_17 (default: all)')
args = p.parse_args()

os.environ['QUICK'] = '1' if args.quick else '0'
os.environ['SOCP_RADIUS'] = args.socp_radius
os.environ['FIGDIR'] = args.outdir
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ch1rab import figures as F   # noqa: E402  (reads the environment at import)

funcs = F.PART_A + F.PART_B
if args.only:
    funcs = [f for f in funcs if any(f.__name__ == 'fig_' + k for k in args.only)]
    if not funcs:
        sys.exit('no figure function matches --only ' + ' '.join(args.only))
for f in funcs:
    t = time.time()
    f()
    print(f'{f.__name__}: {time.time() - t:.0f} s', flush=True)
os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir, 'results_summary.json'), 'w') as fh:
    json.dump(F.RESULTS, fh, indent=1, default=str)
print('figures and results_summary.json written to', args.outdir)
