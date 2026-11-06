# Chapter 1: Robust Adaptive Beamforming (Python simulations)

Companion code for Chapter 1 of *Robust Adaptive Beamforming Across Domains: Wireless, Radar, Sonar, and Biomedical Arrays, with Neural and Learning-Based Methods* by Ayman Elnashar (Wiley–IEEE Press, forthcoming).

The chapter consolidates two papers, and this repository reproduces every simulation figure of the chapter (Figs. 1.5–1.27):

* **TAP 2006.** A. Elnashar, S. Elnoubi, and H. El-Mikati, "Further study on robust adaptive beamforming with optimum diagonal loading," *IEEE Trans. Antennas Propag.*, vol. 54, no. 12, pp. 3647–3658, Dec. 2006. Adaptive variable-loading robust Capon beamformer with an ellipsoidal (spherical) steering-vector constraint, steepest-descent and conjugate-gradient implementations, and a joint quadratic constraint on the weight norm. Figs. 1.5–1.16.
* **IET 2008.** A. Elnashar, "On efficient implementation of robust adaptive beamforming based on worst-case performance optimization," *IET Signal Process.*, vol. 2, no. 4, pp. 381–393, Dec. 2008. Adaptive MVDR beamformer with a single worst-case (WC) constraint and closed-form Lagrange multiplier, the ζ-reduction rule, and a robust LCMV beamformer with multiple-beam WC (MBWC) constraints. Figs. 1.17–1.27.

The code is a Python port of the author's original MATLAB simulations, with the implementation errors listed below corrected.

## Quick start

```bash
git clone <this repository>
cd ch01-robust-adaptive-beamforming
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

# all figures, full Monte Carlo counts (about 45 minutes on one CPU core)
python scripts/generate_figures.py

# two-minute smoke test into a separate folder
python scripts/generate_figures.py --quick --outdir figures_quick

# selected figures only
python scripts/generate_figures.py --only 1_17 1_27

# notebook (figures appear under each cell)
jupyter notebook notebooks/Chapter1_RAB_TAP2006_IET2008.ipynb

# unit tests
pytest -q
```

The package can also be installed with `pip install -e .` and imported as `ch1rab`.

## Repository layout

```text
ch1rab/
  algorithms.py   beamformers: variable-loading RCB (SD/CG), joint constraint, batch and subspace-tracking RCB,
                  WC SOCP (CVXPY + Clarabel), WC eigendecomposition, adaptive WC MVDR with zeta-reduction,
                  MBWC LCMV, RLS inverse-covariance update, SINR and beampattern metrics
  scenarios.py    Monte Carlo scenario runners: tap_run / tap_mc (TAP 2006), iet_run / iet_mc (IET 2008)
  figures.py      plotting helpers and one function per chapter figure
notebooks/
  Chapter1_RAB_TAP2006_IET2008.ipynb   executed notebook, one section per figure
scripts/
  generate_figures.py                  command-line figure generation
figures/          PNG (300 dpi) and PDF of Figs. 1.5-1.27, results_summary.json
tests/            pytest sanity tests (equivalences, constraints, reproducibility)
```

## Figure map

| Chapter figure | Content | Original MATLAB script | Function |
|---|---|---|---|
| 1.5, 1.6, 1.7 | SINR, MSE, SOI power vs snapshot, scenario 1 | `RAB4.m`, `RAB_Joint.m` | `fig_1_5_9` |
| 1.8, 1.9 | Beampatterns, scenario 1 | `RAB4.m` | `fig_1_5_9` |
| 1.10 | SINR vs snapshot, scenario 2 (CG) | `RAB5.m`, `RAB_Joint1.m` | `fig_1_10` |
| 1.11, 1.12 | SINR vs noise power, 0.03π and 0.06π mismatch | `RAB6.m`, `RAB_Joint_SNR.m` | `fig_1_11_12` |
| 1.13–1.15 | Moving jammers (non-coherent, coherent, non-stationary) | `RAB_move.m`, `RAB7_Move*.m`, `RAB7_S_Move.m` | `fig_1_13_15` |
| 1.16 | SINR vs mismatch angle | `RAB_Mismatch.m` | `fig_1_16` |
| 1.17, 1.18 | WC beamformers: SINR vs snapshot, beampatterns | `RAB_WC3.m` | `fig_1_17`, `fig_1_18` |
| 1.19 | WC beamformers: SINR vs noise, N = 50 | `RAB_WC_SNR2.m` | `fig_1_19` |
| 1.20, 1.22, 1.23 | WC parameters A, B, C, λ at ζ = 1.8, 1.2, 2.4 | `RAB_WC3.m` | `fig_1_20_22_23` |
| 1.21 | Effect of ζ | `RAB_WC3.m` (ζ swept) | `fig_1_21` |
| 1.24, 1.25 | Modified algorithm (ζ-reduction), ζ₀ = 3 | `RAB_WC_epsc.m` | `fig_1_24_25` |
| 1.26 | Small angular spread | `RAB_WC_local2.m` | `fig_1_26` |
| 1.27 | Large angular spread, MBWC | `RAB_WC_LAS.m`, `RAB_WCg4ok.m`, `wcg.m` | `fig_1_27` |

## Simulation settings

* M = 5 half-wavelength uniform linear array; sample covariance accumulated without forgetting (η = 1), R(0) = I.
* TAP part: SOI at 0° (0 dB), jammers at 45° and 30° (INR 10 dB). Scenario 1: 0.03π mismatch, ε = 0.5, noise −50 dB, steepest descent with α = 0.008. Scenario 2: 0.06π mismatch, ε = 1, noise −20 dB, conjugate gradient with α = 0.01. Joint constraint τ = 0.2. 50 runs; 50 runs per point in the sweeps.
* IET part: SOI at 0° (SNR 5 dB), presumed 5°, interferers at 30° and 60° (INR 10 dB), noise 0 dB, ζ = 1.8, α = 0.1. 100 runs; 300 runs per point in Fig. 1.19.
* SOCP problems are solved with CVXPY and Clarabel and re-solved every 10 snapshots. By default the SOCP beamformer uses the same uncertainty radius as the other robust beamformers (`--socp-radius matched`); `--socp-radius original` reproduces the radius the MATLAB scripts effectively used.
* Every run has a fixed seed, so results are reproducible bit for bit on the same software stack.
* SINR-vs-snapshot curves are displayed after a centred moving average whose half-width grows as k/4 up to 50 snapshots; all numbers below and in `figures/results_summary.json` come from the unsmoothed Monte Carlo means.

## Key results (steady-state SINR in dB, mean of the last 100 snapshots)

| Figure | Scenario | Result |
|---|---|---|
| 1.5 | Scenario 1 (SD), steady-state SINR | Proposed (ellipsoidal) 20.4, joint 20.0, batch RCB 20.0, SOCP 20.0, subspace RCB 17.4, Capon 1.1 |
| 1.10 | Scenario 2 (CG) | Proposed 12.2, joint 12.2, batch RCB / SOCP 10.3, subspace RCB 6.5, Capon -13.6 |
| 1.13 | Moving jammers, non-coherent | Proposed 15.5, batch RCB 13.5, SOCP 13.4, subspace RCB 12.5, Capon 6.0 |
| 1.14 | Moving jammers, coherent | Proposed 8.1, batch RCB 3.9, SOCP 3.8, subspace RCB 3.1, Capon 5.4 |
| 1.15 | Moving jammers, non-stationary | Proposed 11.6, batch RCB 11.3, SOCP 11.3, subspace RCB 10.4, Capon 10.8 |
| 1.17 | Adaptive WC MVDR, DOA mismatch | WC/Proposed 10.9, WC/EigDec 10.9, WC/SOCP 10.9, benchmark MVDR 11.6, standard MVDR 2.3 |
| 1.25 | Modified algorithm (ζ-reduction from 3) | 11.0 |
| 1.26 | Small angular spread | WC (all three) 15.4, benchmark LCMV 16.1, benchmark MVDR 13.3, standard MVDR -1.0 |
| 1.27 | Large angular spread | MBWC LCMV 10.5, WC (all three) 10.1, benchmark LCMV 11.7, standard LCMV 7.1, benchmark MVDR -0.6 |

With the implementation errors corrected and a common uncertainty set, the SOCP, eigendecomposition and adaptive beamformers converge to the same weight vector; the adaptive algorithms reach it at lower cost and, for the WC beamformer, faster after start-up.

## Corrections relative to the original MATLAB code

```text
1. SOCP objective: robustbeam.m factorized chol(real(Rss)), i.e. it optimized w^H Re(R) w.
   The port uses the full complex covariance.
2. SOCP normalization: robustbeam.m was called with as = M*a/||a|| (||as|| = M, not sqrt(M)),
   which silently rescales the uncertainty radius by sqrt(M). The port uses the presumed
   steering vector itself; the radius is then directly comparable with the other methods.
3. Multipath SINR: scripts mixed SINRF (main ray only) and SINRFm (ray powers added
   incoherently). All rays carry the same symbol, so the port scores every beamformer with
   the coherent SINR  P_s |w^H a_eff|^2 / (w^H R_{i+n} w).
4. LCMV benchmarks used pinv(real(C^H R^-1 C)); the port uses the complex matrix.
5. WC eigendecomposition beamformer: newton2.m used a damped (x0.01) Newton step that can stop
   before the root; the port solves sum |z_m|^2/(zeta + t*gamma_m)^2 = 1 exactly (brentq).
6. WC adaptive root: MATLAB could return a complex lambda (sqrt of a negative discriminant).
   The port keeps the smallest real nonnegative root that satisfies the unsquared constraint
   (1.97). If zeta >= ||a|| the WC problem is infeasible and the previous weight is kept. For a
   feasible zeta with no real root it takes the vertex lambda = -B/A (the real part of the complex
   root MATLAB returned) when that is positive; otherwise it keeps the previous weight. If the
   zeta-reduction rule is enabled, zeta is lowered and A, B, C recomputed within the same snapshot.
7. VL quadratic: guards for a = 0 (first snapshot) and a negative discriminant (MATLAB took
   real(sqrt(.)) = 0, i.e. the vertex of the parabola; the port does the same, explicitly).
8. Beam patterns are |w^H a(theta)|^2 normalised to their maximum (MATLAB plotted an SINR-like
   quantity through SINRF).
9. SOI power: Li-Stoica estimate with the steering vector rescaled to norm sqrt(M) and the
   covariance normalised by the number of snapshots, so the curves are comparable with the
   true SOI power instead of growing with k.
10. Display smoothing: the scripts passed every curve through a zero-phase 1st-order Butterworth
   filter (Wn = 0.01) before plotting, including the sweep figures (1.11, 1.12, 1.16, 1.19), where
   it ran across the 16-26 sweep points and bent each curve towards a straight line (e.g. +3 dB
   output SINR at +30 dB noise in Fig. 1.11, where M = 5 allows at most -23 dB). The port plots
   unsmoothed Monte Carlo means; only the MSE curves (Fig. 1.6) keep the snapshot-axis smoothing.
```

## Requirements

Python 3.10 or later, NumPy, SciPy, Matplotlib, CVXPY, Clarabel and pandas (`requirements.txt` pins the versions used to produce `figures/`). Jupyter is needed only for the notebook.

## How to cite

If you use this code, please cite the book chapter and the original papers (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
