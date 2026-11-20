"""Selection on the training phase of one array:  python -m ch5mprrab.experiments.select <ula|upa> [--use]

One setting per family by the pooled expected SINR; settings within TIE_DB of the best are equivalent and the larger
pooled minimum-cluster SINR decides.  Writes DATA/selection_derived_<arr>.json and .txt and compares the result with
the selection used in the chapter; with --use the derived selection is also written to DATA/selection_<arr>.json,
which the evaluation phases then use instead of the shipped one."""
import json
import os
import sys
from . import _common  # noqa: F401
from .. import scenarios as MS


def main(argv):
    arr = argv[0]
    MS.use_array(arr)
    R = MS.load_phase('train')
    best, pe, pf = MS.select(R)
    lines = ['training phase, %s, %d scenarios, pooled over %d cells' % (arr, len(R), sum(len(R[g]) for g in R)),
             'family / candidates: pooled expected SINR, pooled minimum-cluster SINR (dB); * selected']
    for f in sorted(best):
        ns = sorted([n for n in pe if MS.fam(n) == f], key=lambda n: (MS.kappa_of(n) if 'k=' in n else 0, n))
        lines.append(f)
        for n in ns:
            lines.append('   %s %-22s %7.3f %7.2f' % ('*' if n == best[f] else ' ', n, pe[n], pf[n]))
    shipped = json.load(open(os.path.join(MS.PKG_DATA, 'selection.json')))[arr]['best']
    diff = {f: (best[f], shipped.get(f)) for f in best if best[f] != shipped.get(f)}
    lines.append('selection of the chapter reproduced: %s' % ('yes' if not diff else 'no, differs in %s' % diff))
    open(os.path.join(MS.DATA, 'selection_derived_%s.txt' % arr), 'w').write('\n'.join(lines) + '\n')
    rec = {'best': best, 'tie_db': MS.TIE_DB, 'matches_chapter': not diff, 'differences': diff}
    json.dump(rec, open(os.path.join(MS.DATA, 'selection_derived_%s.json' % arr), 'w'), indent=1)
    if '--use' in argv:
        json.dump({'best': best, 'tie_db': MS.TIE_DB}, open(os.path.join(MS.DATA, 'selection_%s.json' % arr), 'w'), indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1:])
