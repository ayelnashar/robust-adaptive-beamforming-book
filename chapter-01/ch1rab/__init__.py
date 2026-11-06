"""ch1rab: Python implementation of the Chapter 1 simulations (robust adaptive beamforming,
Elnashar et al., IEEE TAP 2006, and Elnashar, IET Signal Processing 2008)."""
from .algorithms import (steer, sinr_single, sinr_coherent, beampattern, capon, soi_power, RLS, VLRCB,
                         newton1, rcb_eig, noojacom, RCBSubspace, WCSOCP, wc_eig, WCAdaptive, MBWCAdaptive,
                         lcmv)
from .scenarios import TAP_METHODS, WC_METHODS, tap_run, tap_mc, iet_run, iet_mc

__version__ = "1.0.0"
