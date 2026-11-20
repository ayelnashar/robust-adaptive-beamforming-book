"""One phase of the study for some scenarios of one array:

    python -m ch5mprrab.experiments.study <ula|upa> <train|heldout|stream> <scenarios> [NR] [base]

e.g. `upa heldout ABC 30`.  The evaluation phases (heldout, stream) use the selected setting of every family
(ch5mprrab.scenarios.load_selection).  One JSON file per scenario is written into DATA; a run that is interrupted
resumes at the next condition."""
import sys
import time
from . import _common  # noqa: F401
from .. import scenarios as MS


def main(argv):
    arr, phase, geos = argv[0], argv[1], argv[2]
    NR = int(argv[3]) if len(argv) > 3 and argv[3] != '-' else None
    base = int(argv[4]) if len(argv) > 4 else None
    MS.use_array(arr)
    sel = MS.load_selection(arr) if phase in ('stream', 'heldout') else None
    t0 = time.time()
    MS.run_phase(phase, list(geos), NR=NR, base=base, sel=sel)
    print('PHASE %s %s %s finished in %.0f s' % (arr, phase, geos, time.time() - t0), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
