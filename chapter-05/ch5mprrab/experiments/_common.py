"""Settings shared by the experiment stages: single-threaded BLAS, so that two stages can run side by side."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
