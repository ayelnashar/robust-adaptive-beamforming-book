# Chapter 3: Efficient Solving Under Multiple Constraints

Python code for Chapter 3 of *Robust Adaptive Beamforming and Its Applications* by Ayman Elnashar (Wiley, forthcoming). It reproduces every simulation figure of the chapter (Figures 3.1–3.14), Tables 3.2–3.6 and the numbers quoted in the text and in Appendix 3.A.

The chapter develops **DAS-MBWC**, the dual-space active-set multiple-beam worst-case beamformer, a robust LCMV design with several worst-case (WC) constraints

```text
minimize    w^H R w
subject to  Re{w^H l_i}  >=  nu_i ||w||  +  1,      i = 1, ..., N
```

where `l_i` are the steering vectors of a grid of directions around the presumed one and `nu_i` their worst-case bounds. DAS-MBWC solves this once per snapshot (Algorithm 3.1): a gradient step with a scale-invariant step size, then a closed-form correction on only the constraints that the gradient step violates, obtained from a small Gram system `Re{L_A^H L_A} tau = d` that does not depend on the covariance scale. The bounds are adapted every 30 snapshots from the dominant eigenvector of the sample covariance, and a rescaling step keeps the weight norm finite once the adapted bounds pass the level at which all constraints can be met together (Appendix 3.A.4).

The chapter builds on the article: A. Elnashar, "Selective Constraint Enforcement in Robust LCMV Beamforming: A Dual-Space Active-Set Method," submitted.

## Quick start

```bash
cd chapter-03
pip install -r requirements.txt

# all figures, tables and results_summary.json, full Monte Carlo counts
# (about 3 minutes with two processes, about 5 minutes with one)
python scripts/generate_figures.py --jobs 2

# smoke test into a separate folder (under a minute)
python scripts/generate_figures.py --quick --outdir figures_quick

# selected figures only
python scripts/generate_figures.py --only 3_1 3_9

# redraw the figures from the simulation data saved by an earlier run
python scripts/generate_figures.py --replot

# notebook (figures appear under each cell)
jupyter notebook notebooks/Chapter3_DAS_MBWC.ipynb

# tests
pytest -q
```

The package can also be installed with `pip install -e .` and used as `ch3dasmbwc`:

```python
import numpy as np
from ch3dasmbwc import das_mbwc, steer, gen, paper_grid, NU0, M

np.random.seed(0)
X = gen(M, 300, 0.0, None)                        # 300 snapshots, SOI at broadside
a = steer(M, np.deg2rad(5))                        # presumed direction 5 deg
w = das_mbwc(X, a, paper_grid(5), NU0)[-1]         # weight after the last snapshot
```

## Contents

```text
ch3dasmbwc/
  algorithms.py   DAS-MBWC (Algorithm 3.1) with its all-constraint ablation; standard and diagonally
                  loaded MVDR; exact single worst-case beamformer WC-MVDR [6]; adaptive single-WC
                  beamformer [12]; covariance-reconstruction comparator after Gu and Leshem [22];
                  exact conic solution of (3.12) with CVXPY and SCS; Slater margin (3.A.4); SINR metrics
  scenarios.py    settings of Section 3.6, data generators and one runner per experiment, all with fixed seeds
  figures.py      one function per figure and per table, results collection
notebooks/        executed notebook, one section per figure and table
scripts/          command-line generation (--quick, --only, --replot, --jobs, --outdir)
figures/          PNG (300 dpi) and PDF of Figures 3.1-3.14, results_summary.json
tests/            sanity tests (exactness of WC-MVDR, scale invariance, feasibility, reproducibility)
```

## Figure and table map

| Chapter | Content | Function |
|---|---|---|
| Figure 3.1 | SINR vs snapshots, 2° mismatch, point source | `fig_3_1` |
| Figure 3.2 | Steady-state beam patterns, 5° mismatch | `fig_3_2` |
| Figure 3.3 | SINR vs snapshots, 5° mismatch, point source | `fig_3_3` |
| Figure 3.4 | SINR vs snapshots, 2° mismatch, small angular spread | `fig_3_4` |
| Figure 3.5 | SINR vs snapshots, 5° mismatch, small angular spread | `fig_3_5` |
| Figure 3.6 | SINR vs snapshots, 7° mismatch, point source | `fig_3_6` |
| Figure 3.7 | SINR vs snapshots, 5° mismatch, large angular spread | `fig_3_7` |
| Figure 3.8 | SINR vs snapshots, 7° mismatch, large angular spread | `fig_3_8` |
| Figure 3.9 | SINR vs mismatch angle, point source | `fig_3_9` |
| Figure 3.10 | SINR vs mismatch angle, small angular spread | `fig_3_10` |
| Figure 3.11 | Array gain, phase and position errors at 5° mismatch | `fig_3_11` |
| Figure 3.12 | Array size scaling at 7° mismatch (16, 64, 128 elements) | `fig_3_12` |
| Figure 3.13 | Sensitivity to the parameters of the bound adaptation (3.20) | `fig_3_13` |
| Figure 3.14 | Measured computation time against array size | `fig_3_14` |
| Table 3.2 | Steady-state SINR, eight scenarios, six methods and the all-constraint ablation | `table_3_2` |
| Table 3.3 | DAS-MBWC vs the exact SOCP of (3.12) at the reference bounds | `table_3_3` |
| Table 3.4 | SCS solve time vs DAS-MBWC correction time | `table_3_4` |
| Table 3.5 | Constraint placement under large angular spread | `table_3_5` |
| Table 3.6 | Stability of the recursion (active-set changes, long runs) | `table_3_6` |
| Text | 200-trial check of Table 3.2, Slater margin and feasibility limits, fixed-bound ablation, correction-to-weight ratio, constraint residuals, signed-mismatch sweep, steering-vector distances | `text_checks` |

## Simulation settings

* M = 16 half-wavelength uniform linear array. SOI at broadside with SNR 5 dB; two interferers at 45° and −60°, each 10 dB above the noise; noise 0 dB per element.
* Adaptive recursions (DAS-MBWC, adaptive single-WC): unnormalized covariance recursion with η = 0.97 and R(0) = I, step-size factor α = 0.1. Batch beamformers (MVDR, DL-MVDR with loading 2σ², WC-MVDR, reconstruction): EWMA covariance with weight 0.99, initialized at σ²I.
* ζ = 3.8. Constraint grid [θ₀, θ₀ − 2°, θ₀ + 2°, θ₀ − 4°] about the presumed direction θ₀, reference bounds ν₀ = [1.9, 1.5, 1.5, 1.1]. Bound adaptation (3.20) at every 30th snapshot after a 50-snapshot warm-up (first update at snapshot index 60), smoothing weight 0.8, clipping range [0.5, 2.0]. Gram solve with ridge 10⁻⁶. Rescaling when ‖w‖ > 10¹², back to 10⁶.
* Multipath: small spread with rays at +4° and −3° (amplitude 0.4), large spread with rays at −25° (0.5) and +35° (0.3); the phases of the secondary rays are redrawn at every snapshot, so the rays add in power, and these scenarios are scored with the desired-signal covariance against the generalized-eigenvalue optimum.
* Reconstruction comparator: declared sector ±6° about the presumed direction, covering the span of the constraint grid.
* Monte Carlo: 300 snapshots and 25 trials per scenario and per sweep point; 200 trials for the check of Table 3.2; 20 trials per case in Figure 3.11; 15 trials per case in Figures 3.12 and 3.13; five runs of 3000 snapshots for the long runs of Table 3.6. Every trial has a fixed seed, so every number can be regenerated exactly.
* Figure 3.12 uses a six-point constraint fan from the true direction (0°, bound 1.9) to 8.75° (bounds 1.3), with the bounds and ζ scaled by √(M/16).
* Figure 3.14 times the complete DAS-MBWC recursion (100-snapshot run, including the covariance update and the bound adaptation, median of five repetitions) against one exact WC-MVDR weight solve on a fixed covariance (median of five repetitions of 20 solves). Table 3.4 times one SCS solve of (3.12) (median of 10, solver time only) against the DAS-MBWC Gram correction alone (median over a 300-snapshot run).

## Key results

Final-snapshot output SINR in dB, 25 trials (Table 3.2):

| Scenario | Std MVDR | DL-MVDR | WC [6] | 1-WC [12] | INC [22] | All-constr. | DAS-MBWC |
|---|---|---|---|---|---|---|---|
| 2° point source | −11.0 | −2.2 | 13.6 | 13.5 | 16.0 | 11.6 | 11.6 |
| 2° small spread | 6.7 | 9.8 | 14.2 | 14.2 | 16.6 | 12.8 | 12.8 |
| 5° point source | −22.1 | −15.7 | 4.9 | 4.8 | 9.4 | 9.7 | 10.2 |
| 5° small spread | 3.0 | 5.8 | 10.0 | 9.9 | 10.2 | 11.1 | 11.5 |
| 5° large spread | −18.3 | −13.9 | 5.0 | 4.8 | 9.6 | 9.8 | 10.3 |
| 7° point source | −25.1 | −24.8 | −25.7 | −23.4 | −4.2 | 8.6 | 6.6 |
| 7° small spread | −1.1 | 2.2 | 5.9 | 5.8 | 6.7 | 9.8 | 8.9 |
| 7° large spread | −18.7 | −17.3 | −12.0 | −11.8 | −4.0 | 8.4 | 6.4 |

* Exact SOCP of (3.12) at the reference bounds (Table 3.3): DAS-MBWC is 0.2 to 0.9 dB below it at 2° and 5°; at 7° it holds 6.6 and 6.4 dB where the conic solution falls to −5.9 and −6.1 dB (point source, large spread). All 200 SCS solves return optimal.
* Slater margin t* = 1.08 at the reference bounds; the constraints can be met together only up to 1.72 ν₀, and every three-constraint subset only up to 1.72–1.85 ν₀ (Appendix 3.A.4). Holding the bounds at ν₀ gives about −16 dB (point source) and −14 dB (large spread) at 7°.
* Stability (Table 3.6): the active set is fixed after the transient at 2° and 7° and alternates at 5° without raising the SINR variation; without rescaling the weight norm overflows two to three times per 3000-snapshot run, with it never.
* Array imperfections (Figure 3.11): DAS-MBWC leads in all eight cases and loses 0.34 dB between the ideal array and the combined-error case (9.95 to 9.61 dB).
* Array size (Figure 3.12): 9.3, 14.2 and 18.3 dB at 16, 64 and 128 elements; with an interferer inside the declared region the reconstruction falls to −45.9 dB at 128 elements while DAS-MBWC reaches 10.4 dB.

Timings depend on the machine. On the machine used for `figures/` (one Intel Xeon core at 2.1 GHz, NumPy 2.5.3, SCS 3.3.2) the DAS-MBWC correction is 66, 109, 201 and 503 times faster than one SCS solve at 16, 32, 64 and 128 elements, and the complete recursion is 2.1 times faster per snapshot than one exact WC-MVDR solve at 16 elements and 17 times faster at 256. All values are in `figures/results_summary.json`.

Figures show each method with the same colour, marker and line style throughout, inside a full frame, with the legend in a framed box placed clear of the curves.

## Requirements

Python 3.10 or later, NumPy, SciPy, Matplotlib, CVXPY, Clarabel and SCS (`requirements.txt` pins the versions used to produce `figures/`). Jupyter is needed only for the notebook.

## How to cite

Please cite the book chapter and the papers in `CITATION.cff`.

## License

MIT, see `LICENSE`.
