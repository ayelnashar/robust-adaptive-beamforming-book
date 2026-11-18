"""ch2assnc: Python implementation of the Chapter 2 simulations (AS-SNC, a spectral active-set
method for second-order cone programs with shared norm coupling, and its benchmarks)."""
from .algorithms import as_snc, quality, conic_solve, reference_solution
from .scenarios import (METHODS, random_instance, load_ff, build_portfolio_problem, portfolio_model,
                        Benchmark, dimension_sweep, conditioning_sweep, tolerance_sweep,
                        famafrench_benchmark, regression_checks, convergence_traces,
                        scale_invariance, active_set_counts, near_boundary_diagnostic,
                        supplement, portfolio_sweep, run_all)

__version__ = "1.0.0"
