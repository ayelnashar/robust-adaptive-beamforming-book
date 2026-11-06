# Chapter 1: Robust Adaptive Beamforming

Python code for Chapter 1 of *Robust Adaptive Beamforming Across Domains: Wireless, Radar, Sonar, and Biomedical Arrays, with Neural and Learning-Based Methods* by Ayman Elnashar (Wiley–IEEE Press, forthcoming). It reproduces every simulation figure of the chapter (Figs. 1.5–1.27).

The chapter builds on two papers:

* **TAP 2006.** A. Elnashar, S. Elnoubi, and H. El-Mikati, "Further study on robust adaptive beamforming with optimum diagonal loading," *IEEE Trans. Antennas Propag.*, vol. 54, no. 12, pp. 3647–3658, Dec. 2006. Adaptive variable-loading robust Capon beamformer with an ellipsoidal (spherical) steering-vector constraint, steepest-descent and conjugate-gradient implementations, and a joint quadratic constraint on the weight norm. Figs. 1.5–1.16.
* **IET 2008.** A. Elnashar, "On efficient implementation of robust adaptive beamforming based on worst-case performance optimization," *IET Signal Process.*, vol. 2, no. 4, pp. 381–393, Dec. 2008. Adaptive MVDR beamformer with a single worst-case (WC) constraint and closed-form Lagrange multiplier, the ζ-reduction rule, and a robust LCMV beamformer with multiple-beam WC (MBWC) constraints. Figs. 1.17–1.27.

## Quick start

```bash
cd chapter-01
pip install -r requirements.txt

# all figures, full Monte Carlo counts (about 45 minutes on one CPU core)
python scripts/generate_figures.py

# two-minute smoke test into a separate folder
python scripts/generate_figures.py --quick --outdir figures_quick

# selected figures only
python scripts/generate_figures.py --only 1_17 1_27

# notebook (figures appear under each cell)
jupyter notebook notebooks/Chapter1_RAB_TAP2006_IET2008.ipynb

# tests
pytest -q
```

The package can also be installed with `pip install -e .` and imported as `ch1rab`.

## Contents

```text
ch1rab/
  algorithms.py   variable-loading RCB (steepest descent / conjugate gradient), joint constraint,
                  batch and subspace-tracking RCB, WC SOCP (CVXPY + Clarabel), WC eigendecomposition,
                  adaptive WC MVDR with zeta-reduction, MBWC LCMV, RLS inverse-covariance update,
                  SINR and beampattern metrics
  scenarios.py    Monte Carlo scenario runners: tap_run / tap_mc (TAP 2006), iet_run / iet_mc (IET 2008)
  figures.py      plotting helpers and one function per chapter figure
notebooks/        executed notebook, one section per figure
scripts/          command-line figure generation
figures/          PNG (300 dpi) and PDF of Figs. 1.5-1.27, results_summary.json
tests/            sanity tests (equivalences, constraints, reproducibility)
```

## Figure map

| Chapter figure | Content | Function |
|---|---|---|
| 1.5, 1.6, 1.7 | SINR, MSE, SOI power vs snapshot, scenario 1 | `fig_1_5_9` |
| 1.8, 1.9 | Beampatterns, scenario 1 | `fig_1_5_9` |
| 1.10 | SINR vs snapshot, scenario 2 | `fig_1_10` |
| 1.11, 1.12 | SINR vs noise power, 0.03π and 0.06π mismatch | `fig_1_11_12` |
| 1.13–1.15 | Moving jammers (non-coherent, coherent, non-stationary) | `fig_1_13_15` |
| 1.16 | SINR vs mismatch angle | `fig_1_16` |
| 1.17, 1.18 | WC beamformers: SINR vs snapshot, beampatterns | `fig_1_17`, `fig_1_18` |
| 1.19 | WC beamformers: SINR vs noise, N = 50 | `fig_1_19` |
| 1.20, 1.22, 1.23 | WC parameters A, B, C, λ at ζ = 1.8, 1.2, 2.4 | `fig_1_20_22_23` |
| 1.21 | Effect of ζ | `fig_1_21` |
| 1.24, 1.25 | Modified algorithm (ζ-reduction), ζ₀ = 3 | `fig_1_24_25` |
| 1.26 | Small angular spread | `fig_1_26` |
| 1.27 | Large angular spread, MBWC | `fig_1_27` |

## Simulation settings

* M = 5 half-wavelength uniform linear array; sample covariance accumulated without forgetting (η = 1), R(0) = I.
* TAP scenarios: SOI at 0° (0 dB), jammers at 45° and 30° (INR 10 dB). Scenario 1: 0.03π mismatch, ε = 0.5, noise −50 dB, steepest descent with α = 0.008. Scenario 2: 0.06π mismatch, ε = 1, noise −20 dB, conjugate gradient with α = 0.01. Joint constraint τ = 0.2. 50 runs; 50 runs per point in the sweeps.
* IET scenarios: SOI at 0° (SNR 5 dB), presumed 5°, interferers at 30° and 60° (INR 10 dB), noise 0 dB, ζ = 1.8, α = 0.1. 100 runs; 300 runs per point in Fig. 1.19.
* All robust beamformers of a scenario share the same uncertainty set: the SOCP beamformer uses the norm bound √ε in the TAP scenarios and ζ in the IET scenarios. SOCP problems are solved with CVXPY and Clarabel and re-solved every 10 snapshots.
* Every run has a fixed seed, so the figures can be regenerated exactly.
* SINR-vs-snapshot curves are displayed after a centred moving average whose half-width grows as k/4 up to 50 snapshots; all numbers below and in `figures/results_summary.json` come from the unsmoothed Monte Carlo means.

## Key results (steady-state SINR in dB, mean of the last 100 snapshots)

| Figure | Scenario | Result |
|---|---|---|
| 1.5 | Scenario 1 (steepest descent) | Proposed (ellipsoidal) 20.4, proposed (joint) 20.0, batch RCB 20.0, SOCP 20.0, subspace RCB 17.4, Capon 1.1 |
| 1.10 | Scenario 2 (conjugate gradient) | Proposed 12.2, joint 12.2, batch RCB / SOCP 10.3, subspace RCB 6.5, Capon -13.6 |
| 1.13 | Moving jammers, non-coherent | Proposed 15.5, batch RCB 13.5, SOCP 13.4, subspace RCB 12.5, Capon 6.0 |
| 1.14 | Moving jammers, coherent | Proposed 8.1, batch RCB 3.9, SOCP 3.8, subspace RCB 3.1, Capon 5.4 |
| 1.15 | Moving jammers, non-stationary | Proposed 11.6, batch RCB 11.3, SOCP 11.3, subspace RCB 10.4, Capon 10.8 |
| 1.17 | Adaptive WC MVDR, DOA mismatch | WC/Proposed 10.9, WC/EigDec 10.9, WC/SOCP 10.9, benchmark MVDR 11.6, standard MVDR 2.3; at snapshot 10: WC/Proposed 7.7, WC/EigDec 6.5, WC/SOCP 3.3 |
| 1.25 | Modified algorithm (ζ-reduction from ζ = 3) | 11.0 |
| 1.26 | Small angular spread | WC beamformers 15.4, benchmark LCMV 16.1, benchmark MVDR 13.3, standard MVDR -1.0 |
| 1.27 | Large angular spread | MBWC LCMV 10.5, WC beamformers 10.1, benchmark LCMV 11.7, standard LCMV 7.1, benchmark MVDR -0.6 |

## Requirements

Python 3.10 or later, NumPy, SciPy, Matplotlib, CVXPY, Clarabel and pandas (`requirements.txt` pins the versions used to produce `figures/`). Jupyter is needed only for the notebook.

## How to cite

Please cite the book chapter and the two papers above (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
