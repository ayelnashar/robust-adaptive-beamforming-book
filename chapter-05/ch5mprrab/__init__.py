"""
ch5mprrab -- simulations of Chapter 5, "Multi-Source Robust Beamforming with Subspace Tracking", of
Robust Adaptive Beamforming and Its Applications (A. Elnashar, Wiley).

MPR-RAB, the multi-primary robust adaptive beamformer: per-cluster worst-case constraints with margins from joint
feasibility, gain targets and constraint phases from the max-SINR beamformer of the estimated model, a reconstructed
interference-plus-noise covariance with off-grid peak refinement, the active-set KKT solver of Chapter 2, direction
tracking on a deflated covariance with delay compensation, and cluster lifecycle management.

Modules
  channel     signal model, multi-cluster channel, expected SINR, oracle
  algorithms  the MPR-RAB kernel (run_mpr_rab) and the active-set solver
  planar      the 8 x 8 planar array and its scenarios
  baselines   LCMV-K, Par-K-RCB, LCMV-K-D, INCM-LCMV, INCM-MVDR, INCM-GEV
  scenarios   scenarios, protocol, metrics, selection and the stage runner
  figures     Figures 5.1-5.10 and the results summary
"""
from .channel import steer, sinr_db_cluster, MultiClusterChannel, run_oracle_mc, M_DEFAULT  # noqa: F401
from .algorithms import run_mpr_rab, as_snc, real_form, maxsinr_responses, sinr_floor_db  # noqa: F401
from .scenarios import make_data, metrics, run_cell, use_array, load_selection, SCEN, CFG  # noqa: F401

__version__ = '1.0.0'
