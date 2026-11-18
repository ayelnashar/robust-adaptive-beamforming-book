# Chapter 2: A General Constrained-Optimization Method for Robust Beamforming

Python code for Chapter 2 of *Robust Adaptive Beamforming and Its Applications* by Ayman Elnashar (Wiley, forthcoming). It reproduces every figure of the chapter (Figures 2.1–2.8) and the timing tables (Tables 2.3 and 2.4).

The chapter develops AS-SNC, a spectral active-set method for second-order cone programs with shared norm coupling (SNC-SOCPs):

```text
minimize    w' Q w
subject to  l_i' w  >=  nu_i ||w||  +  beta_i,      i = 1, ..., N
```

with `Q` symmetric positive definite and `nu >= 0`. The norm term is the worst-case penalty of a linear constraint whose direction is known only to within a ball of radius `nu_i`, and every constraint carries the same `||w||`. Robust beamforming with several worst-case constraints is one instance; robust portfolio selection, adversarially robust linear classification and robust control margins are others (Section 2.6).

AS-SNC computes one eigendecomposition of `Q`, reduces each working-set system to a small multiplier system and two scalar equations, applies a reduced Newton correction when the scalar iteration stalls, and returns a point only after a scaled KKT residual test on the input problem. If that test fails, it solves the shared-epigraph conic model with Clarabel (then SCS) and labels the result as a fallback (Algorithm 2.1).

The chapter builds on the article: A. Elnashar, "A Spectral Active-Set Method for Second-Order Cone Programs with Shared Norm Coupling," submitted.

## Quick start

```bash
cd chapter-02
pip install -r requirements.txt

# Figures 2.1-2.8 and results_summary.json from the archived measurement run (a few seconds)
python scripts/generate_figures.py

# measure every experiment again on this machine (about one minute on two cores)
python scripts/generate_figures.py --rerun

# reduced re-run, smoke test (about ten seconds)
python scripts/generate_figures.py --quick

# selected figures only
python scripts/generate_figures.py --only 2_1 2_8

# notebook (figures appear under each cell)
jupyter notebook notebooks/Chapter2_AS_SNC.ipynb

# tests
pytest -q
```

The package can also be installed with `pip install -e .` and used as the Python package `ch2assnc`:

```python
from ch2assnc import as_snc

w, outer_iterations, active_set, info = as_snc(Q, L, nu, beta, tol=1e-9, return_info=True)
print(info["method"], info["residual"])    # 'AS-SNC', or 'fallback:CLARABEL' / 'fallback:SCS'
```

`info` also holds the multipliers, the scaled feasibility, stationarity and complementarity residuals and the number of Newton steps. If neither the spectral path nor the fallback passes the residual test (for example, an infeasible problem), `as_snc` raises `RuntimeError`.

## Contents

```text
ch2assnc/
  algorithms.py     AS-SNC (Algorithm 2.1) with the reduced Newton correction and the final residual
                    check; scaled KKT residual report; shared-epigraph conic model (CVXPY) used by
                    the baselines and by the independent reference
  scenarios.py      instance generator, Fama-French data and portfolio problem, 50-asset allocation,
                    timing harness and one runner per experiment
  figures.py        one function per chapter figure, and the results summary
  data/             Fama-French monthly returns, January 1949 to March 2017
data/archived_run/  records of the measurement run behind the chapter figures
notebooks/          executed notebook, one section per figure
scripts/            command-line figure generation
figures/            PNG (300 dpi) and PDF of Figures 2.1-2.8, results_summary.json
tests/              sanity tests (closed forms, conic agreement, scale invariance, reproducibility)
```

## Figure map

| Chapter | Content | Function | Records |
|---|---|---|---|
| Figure 2.1, Table 2.3 | Median end-to-end time versus decision dimension M | `fig_2_1` | `scaling.json` |
| Figure 2.2 | Scaled KKT residual of every recorded candidate, five instances | `fig_2_2` | `supplement.json` |
| Figure 2.3 | Relative change of the solution when Q is scaled | `fig_2_3` | `supplement.json` |
| Figure 2.4 | Optimal active-set cardinality for N = 20 and N = 50 | `fig_2_4` | `supplement.json` |
| Figure 2.5 | Time versus accuracy when every solver's tolerance is swept | `fig_2_5` | `pareto.json` |
| Figure 2.6 | Median end-to-end time versus the condition number of Q | `fig_2_6` | `conditioning.json` |
| Figure 2.7 | Risk and expected return of the 50-asset allocation versus the sector radius κ | `fig_2_7` | `portfolio.json` |
| Figure 2.8, Table 2.4 | Median end-to-end time on the three Fama–French problems | `fig_2_8` | `famafrench.json` |

The scalar closed form of Section 2.2.6 and the repelling fixed point of Section 2.4.2 are checked by `regression_checks()` and by the tests.

## Experimental settings

* Synthetic instances (`random_instance`): Q has eigenvalues logarithmically spaced in [1/κ, 1] and a random orthonormal eigenbasis; constraint directions are random unit vectors. The first ⌈√N⌉ constraints get radii in [0.1, 0.2] and offsets in [0.8, 1.2], the others radii in [0.01, 0.05] and offsets in [0.05, 0.2].
* Dimension sweep: M = 25, 50, 100, 200, 400, N = 20, κ = 10³, eight instances per M, seed 0.
* Conditioning sweep: M = 100, N = 20, κ = 10, 10², ..., 10⁵, ten instances per κ, seed 100 + 1000k for the k-th κ.
* Tolerance sweep: M = 100, N = 20, κ = 10³, eight instances (seed 10); every solver is run at the requested tolerances 10⁻², 10⁻⁴, ..., 10⁻¹⁰.
* Fama–French problems: FF-Ind12 (M = 12), FF-All21 (M = 21), FF-All30 (M = 30); 819 monthly returns, January 1949 to March 2017; one robust return floor and three group floors (N = 4); ten timed repetitions each. A free-scale solver benchmark, with no budget equality and no long-only restriction.
* Convergence traces: five instances (seed 20261020). Scale test: 20 instances (seed 2030), scales 10⁻⁴ to 10⁴. Active-set counts: 200 instances per N (seed 2040 + N). All at M = 100, κ = 10³.
* 50-asset allocation: three-factor covariance (seed 2050), five sectors of ten assets, expected-return floor 0.15 with zero radius, sector floors 0.02 with a common radius κ from 0 to 0.14 (feasible only for κ < 1/√50). Every AS-SNC point is checked against a tight Clarabel solve.
* Timing: one warm-up call, then one timed call, with one BLAS thread. AS-SNC times are end to end (eigendecomposition, iterations and final check). Baseline times include CVXPY model construction, canonicalization and the backend solve; `results_summary.json` also lists the solve times measured by the backends on their own.
* Accuracy is measured outside the timings, against an independent reference: SCS at 10⁻¹¹, cross-checked with Clarabel at 10⁻¹¹. The common target is a scaled feasibility violation of at most 10⁻⁷ and a relative objective difference of at most 10⁻⁶.
* MOSEK needs a licence. Gurobi's free licence limits model size, so there are no Gurobi entries for M ≥ 100, including the conditioning and tolerance sweeps. Missing entries stay missing; nothing is extrapolated.

## Archived run and re-runs

The figures in `figures/` are drawn from `data/archived_run`, the measurement run behind the chapter: a Linux x86-64 cloud virtual machine with Python 3.12.14, NumPy 2.3.5, CVXPY 1.9.3 and MOSEK 11.2.6, using one BLAS thread. Timings on other machines will differ; the solutions should not.

A full re-run with `--rerun` on a two-core Intel Xeon (2.30 GHz) machine without MOSEK gave identical active sets and outer-iteration counts for all 160 timed instances, AS-SNC objectives within 1.1 × 10⁻¹³ (relative) of the archived ones, the same pass counts, and identical active-set histograms, convergence traces, portfolio sweep and checks. Every AS-SNC result came from the spectral path. The timing pattern was the same: 1.4–17 ms for AS-SNC against 12–267 ms for the baselines called through CVXPY in the dimension sweep.

## Key results (archived run)

| Result | Value |
|---|---|
| Table 2.3: median time at M = 25 / 100 / 400 (ms) | AS-SNC 1.81 / 4.31 / 17.29; SCS 16.48 / 22.57 / 89.43; MOSEK 17.23 / 24.08 / 117.41 |
| MOSEK-to-AS-SNC time ratio, dimension sweep | 5.59 (M = 100) to 9.52 (M = 25) |
| Mean optimal active-set size, dimension sweep | 12.75 (M = 25) to 17.50 (M = 400) |
| Figure 2.3: largest relative change, scales 10⁻⁴ to 10⁴ | 2.0 × 10⁻¹⁴ |
| Figure 2.4: mean active-set size | 15.21 (N = 20), 35.00 (N = 50) |
| Figure 2.5: median time over the tolerance sweep | AS-SNC 2.7–3.6 ms; baselines 17.5–28.9 ms |
| Figure 2.6: AS-SNC median time, κ = 10 to 10⁵ | 1.79 to 3.56 ms; MOSEK-to-AS-SNC ratio 11.66 to 6.15 |
| Table 2.4: median time, FF-Ind12 / FF-All21 / FF-All30 (ms) | AS-SNC 0.48 / 0.50 / 0.68; MOSEK 6.39 / 7.74 / 7.74 |
| Provenance of the 160 timed AS-SNC results | 160 spectral path, 0 fallback |
| Backend solve time alone at M = 25 / 400 (ms) | SCS 0.47 / 13.71; MOSEK 1.00 / 65.88; AS-SNC (end to end) 1.81 / 17.29 |

The last row matters when reading the end-to-end comparison: with CVXPY's modelling time excluded, the SCS backend solves faster than AS-SNC at every dimension, and MOSEK does at M ≤ 50 (`supplementary_backend_solve_time_ms` in `results_summary.json`).

## Data

The Fama–French monthly test portfolios come from the Kenneth R. French Data Library, as distributed with Kevin Sheppard's `linearmodels` package. They are stored in `ch2assnc/data/`, so no download is needed. All other instances are synthetic and fully determined by the seeds above.

## Requirements

Python 3.10 or later, NumPy, SciPy, Matplotlib, pandas, threadpoolctl, CVXPY, Clarabel, SCS and ECOS (`requirements.txt` pins the versions used to draw `figures/` and for the re-run above). MOSEK and Gurobi are optional. Jupyter is needed only for the notebook.

## How to cite

Please cite the book chapter and the article above (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
