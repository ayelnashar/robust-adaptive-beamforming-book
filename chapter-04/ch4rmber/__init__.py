"""Chapter 4 simulations: guided R-MBER, a spectrally guided blind minimum-bit-error-rate beamformer, and its comparators.

Companion code for Chapter 4, "Robust Beamforming that Minimizes Bit-Error-Rate Directly", of
Robust Adaptive Beamforming and Its Applications by Ayman Elnashar (Wiley).
"""
from .algorithms import (steer, sinr_db, build_R_ipn, wc_grid, nu_pub, zeta, alg_mvdr, alg_dl, wc_mvdr_exact, run_mber,
                         incm_rec, spss, ta_mmse, proj_exact, run_rmber, run_blind, gen_with_symbols, gen_block,
                         gen_block_p, ber_of, run_sgrab_p, beamwidth_deg, guide, ber_stage, guided,
                         M, K, ps, sigma2, KAPPA, RHO_C, NU_FAC, MP_SMALL, MP_LARGE, R_ipn)

__version__ = '1.0.0'
