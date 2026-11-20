"""Prints the result tables of one array:  python -m ch5mprrab.experiments.report <ula|upa>

Operating point (first condition of the condition study, 30 runs): expected and minimum-cluster SINR per scenario;
condition study: expected and minimum-cluster SINR per condition and the paired differences MPR-RAB minus comparator
with 95 % bootstrap intervals."""
import json
import sys
import numpy as np
from . import _common  # noqa: F401
from .. import scenarios as MS


def main(argv):
    arr = argv[0] if argv else 'ula'
    MS.use_array(arr)
    sel = dict(MS.load_selection(arr))
    for family in ('INCM-LCMV', 'INCM-MVDR', 'INCM-GEV'):
        sel[family] = sel[family + ' refined']
    print('selection:', json.dumps(sel, indent=1))
    Rh = MS.load_phase('heldout')
    Rp = {sc: {'5_10_0': cells['5_10_0']} for sc, cells in Rh.items() if '5_10_0' in cells}
    if Rp:
        MS.report(Rp, sel, '%s, operating point: expected SINR (dB)' % arr)
        MS.report(Rp, sel, '%s, operating point: minimum-cluster SINR (dB)' % arr, '|floor')
        if 'I' in Rp:
            print('scenario I: last 50 / first 10 shadowed / shadowed / first 10 after the return')
            for lab, n in MS.columns(sel):
                r = Rp['I']['5_10_0']
                print('  %-20s %6.2f %6.2f %6.2f %6.2f' % (lab, np.mean(r[n]), np.mean(r[n + '|dip']), np.mean(r[n + '|blk']),
                                                        np.mean(r[n + '|rec'])))
    if Rh:
        MS.report(Rh, sel, '%s, condition study: expected SINR (dB)' % arr)
        MS.report(Rh, sel, '%s, condition study: minimum-cluster SINR (dB)' % arr, '|floor')
        MS.paired_ci(Rh, sel)
        MS.paired_ci(Rh, sel, '|floor')


if __name__ == '__main__':
    main(sys.argv[1:])
