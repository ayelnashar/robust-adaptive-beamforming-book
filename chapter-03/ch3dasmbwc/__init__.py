"""Chapter 3 simulations: DAS-MBWC, the dual-space active-set multiple-beam worst-case beamformer.

Companion code for Chapter 3, "Efficient Solving Under Multiple Constraints", of
Robust Adaptive Beamforming and Its Applications by Ayman Elnashar (Wiley).
"""
from .algorithms import (steer, steer_err, sinr_db, sinr_cov_db, sinr_opt_db, desired_cov, beampattern_db,
                         mvdr, dl_mvdr, wc_mvdr, adaptive_single_wc, inc_reconstruction, das_mbwc,
                         mbwc_socp, feasibility_margin)
from .scenarios import (M, PS, ETA, BETA, ZETA, NU0, MP_SMALL, MP_LARGE, THETA_INTS, R_IPN, METHODS,
                        SCENARIOS, TABLE_ORDER, paper_grid, gen, batch_cov)

__version__ = '1.0.0'
