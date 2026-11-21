# Chapter 7: Dual-Function Beamforming for Integrated Sensing and Communication

Python code for Chapter 7 of *Robust Adaptive Beamforming and Its Applications* by Ayman Elnashar (Wiley, forthcoming). It reproduces every simulation figure of the chapter (Figures 7.1–7.9), Tables 7.1 and 7.3–7.6 and the numbers quoted in the text.

The chapter treats the receive side of a dual-function base station: one receive filter recovers uplink users, each on its own time-frequency resource, and collects monostatic radar echoes from the same array snapshots, while strong co-channel interference may arrive a few degrees from the served directions. Radar probing and uplink overlap in a fraction ρ = 0.3 of the resources. User SINR (7.5) and target SNR (7.6) are ratios of quadratic forms in the receive weight.

**RDF-AS-ISAC** (Algorithm 7.1), the robust dual-function active-set beamformer of the chapter, keeps every communication and sensing worst-case constraint and adapts only to identified interference:

1. A forgetting covariance (7.13) is scanned every 10 observations with the Capon spectrum (7.14), evaluated exactly through the lag sums (7.15)–(7.16) of the planar array on a 2° grid and refined to 0.25° and 0.05°.
2. Peaks are labelled as served users/targets or interference tracks. A served direction never claims a peak more than 6 dB above the nominal user SNR (power ceiling), and a power-gated extended claim recovers a user whose peak lies outside its window. Each track is an alpha-beta filter (7.18) predicted ahead by (7.19).
3. The structured covariance (7.20) holds the predicted tracks and the noise floor; peaks labelled as served signals never enter it.
4. The worst-case problem (7.11)–(7.12) is solved exactly in diagonal-loading form (7.23) through a nonnegative least-squares dual with one variable per constraint. Every solve returns the duality-gap certificate (7.24), and the radial restoration (7.25) certifies every constraint at the output.

The comparisons (Section 7.7.2) are base-radius and radius-matched SOCP, batch and re-solved online (CVXPY with Clarabel; SDP with SCS), projection reconstruction (protected-subspace covariance with an iterative solver), the fixed beam, VU-MBWC, the true-covariance solution of the same problem and the single-function MVDR bounds (7.6b).

The chapter builds on the article: A. Elnashar, "Robust Dual-Function Receive Beamforming With Tracked-Interference Covariance Reconstruction for 6G ISAC."

## Quick start

```bash
cd chapter-07
pip install -r requirements.txt

# all figures, tables and results_summary.json, chapter seed counts (about 30 minutes with two processes)
python scripts/generate_figures.py

# then compare every regenerated number with the value printed in the chapter
python -m ch7rdfisac.verify figures

# smoke test into a separate folder (about 6 minutes)
python scripts/generate_figures.py --quick --outdir figures_quick

# selected figures, the tables, or the text results only (runs only the stages they need)
python scripts/generate_figures.py --only 7_2 7_8
python scripts/generate_figures.py --only tables text

# redraw the figures from the stage data saved by an earlier run
python scripts/generate_figures.py --replot

# notebook (figures appear under each cell)
jupyter notebook notebooks/Chapter7_RDF_AS_ISAC.ipynb

# tests
pytest -q
```

The package can also be installed with `pip install -e .` and used as `ch7rdfisac`:

```python
import numpy as np
from ch7rdfisac.scenes import C, S, n, CENTERS, TRACK_KS, scene
from ch7rdfisac.algorithms import run_rdf_as_isac
from ch7rdfisac.rxmodel import evaluate_rx

X, truth = scene('G', 9901)                      # drifting strong interference, 300 snapshots
W, diag = run_rdf_as_isac(X, C, S, n, n, CENTERS, return_diagnostics=True)
print(np.round(np.mean([evaluate_rx(W[k], truth[k]) for k in TRACK_KS], 0), 2))   # SINR u1, u2, SNR t1, t2 (dB)
print(max(diag['certified_gaps']))                # largest certified relative duality gap
```

## Contents

```text
ch7rdfisac/
  algorithms.py   RDF-AS-ISAC (Algorithm 7.1): lag-sum Capon scanner, peak labelling with power ceiling, extended
                  user claim and alpha-beta tracks, exact loaded worst-case solution with certificate and restoration
  projection.py   projection reconstruction (protected subspace, accelerated projected gradient, Dykstra cycles)
  baselines.py    exact SOCP and SDP solutions of (7.11)-(7.12), VU-MBWC, naive concatenated LCMV
  simulator.py    steering vectors (7.1), clustered uplink channel (7.2), snapshots (7.3) for Scenarios A-E
  scenes.py       fixed Scenarios A-G (Table 7.2) and the 12 steering columns
  rxmodel.py      metrics (7.5)-(7.6) and the random-geometry scene generator (Sections 7.7.4-7.7.6)
  scenarios.py    experiment stages with their seeds (main, array, load, sweep, cross, sens, alpha, grid,
                  tightproj, timing) and the runner (two processes; timing runs alone at the end)
  summary.py      results_summary.json and Tables 7.1, 7.3-7.6 (CSV and Markdown)
  figures.py      one function per figure (fig_7_1 ... fig_7_9)
  verify.py       compares every regenerated number with the value printed in the chapter
notebooks/        executed notebook, one section per figure and table
scripts/          command-line generation (--quick, --only, --replot, --jobs, --outdir, --verify)
figures/          PNG (300 dpi) and PDF of Figures 7.1-7.9, Tables 7.1 and 7.3-7.6, results_summary.json
tests/            sanity tests (exact lag sums, loaded solution against a conic solve, certificate, restoration,
                  alpha-beta tracker, power ceiling, metric scale invariance, fixed-seed values)
```

## Figure map

| Chapter | Content | Function | Stage(s) |
|---|---|---|---|
| Figure 7.1 | Held-out paired gains in mean user SINR and mean target SNR | `fig_7_1` | `main` |
| Figure 7.2 | User SINR versus interferer offset, interferer INR and joint angular error | `fig_7_2` | `sweep` |
| Figure 7.3 | Target 1 crossing user 1 | `fig_7_3` | `cross` |
| Figure 7.4 | Causal user 1 SINR in Scenarios D, F and G | `fig_7_4` | `main` |
| Figure 7.5 | Mean array response at elevation 2° in Scenarios D and F | `fig_7_5` | `main` |
| Figure 7.6 | Certified duality gap (7.24) and matched-problem gap (7.28) | `fig_7_6` | `main` |
| Figure 7.7 | Mean user SINR and target SNR versus the priority α | `fig_7_7` | `alpha` |
| Figure 7.8 | Scenario E service-threshold coverage | `fig_7_8` | `grid` |
| Figure 7.9 | Worst-target detection probability, Swerling 1 | `fig_7_9` | `main` |
| Table 7.1 | Single-thread runtime versus array size | `summary.tables` | `timing` |
| Tables 7.3, 7.4 | Reference-scene means; held-out paired gains | `summary.tables` | `main` |
| Table 7.5 | Random geometry: array size and load | `summary.tables` | `array`, `load` |
| Table 7.6 | One-at-a-time parameter sensitivity | `summary.tables` | `sens` |
| Text | Exact conic solve on the projection covariance (7.7.2), users below 0 dB, SDP/SOCP agreement, constraint satisfaction and certificate (7.7.9), coverage (7.7.11), Swerling 0/1/3 detection (7.7.12) | `summary.summarize` | `main`, `tightproj`, `grid` |

## Simulation settings

* 8 × 4 half-wavelength UPA (arrays of 16 to 256 elements in Sections 7.5.5 and 7.7.4). Users at (20°, 5°) and (−30°, 8°), targets at (−5°, 0°) and (40°, −3°); user received SNR 8 dB, target amplitudes 0.8 and 0.7, matched-filter length N_f = 64; six static clutter components of power 0.04; two 8 dB interferers at (62°, 2°) and (−65°, 1°); overlap ρ = 0.3. Each served direction has two azimuth neighbours at ±2°, giving 12 constraints.
* Scenarios A–G (Table 7.2): user and target angular-error standard deviations from 0.5° to 3.5°; in F and G three 25 dB interferers 6–12° from the served directions, drifting by 4–7° in G. A–D are scored with the final weight after 300 observations, F and G with metrics averaged over k = 51, 56, …, 296.
* RDF-AS-ISAC: η = 0.96 (1 − 1.28/M for other arrays), T_r = 10, radius 3ν₀ (ν₀ = 0.5 primary, 0.35 neighbour, scaled by √(M/32)), windows ±9° × ±4°, gates 2.5° × 1.5°, position gain 0.5, rate parameter 0.3, power tolerance ±6 dB, α = 0.5. RDF-AS-ISAC refreshes at observations 10, 20, …, 300; the online SOCP baselines are re-solved at observations 1, 11, …, 291 from a forgetting covariance with factor 0.97; projection reconstruction updates at the first observation and every 30 observations.
* Seeds: reference 1–15, held-out 5001–5015, array size 6201–6210, load 6301–6310, sweeps 6101–6108, crossing 6401–6408, sensitivity 6501–6508, priority 7001–7010, Scenario E 6001–6003, timing 6901–6903. All settings were fixed on development seeds 2001–2012 and 8001–8008, which are not used here.

## Key results

Held-out paired gains of RDF-AS-ISAC in mean user SINR (dB, Table 7.4; 15 scenes per scenario, 95% Student-t intervals):

| Scenario | vs Projection reconstr. | vs SOCP 3ν₀, batch | vs SOCP 3ν₀, online | vs True covariance |
|---|---|---|---|---|
| A | +0.35 [0.27, 0.42] | +0.36 [0.16, 0.56] | +1.29 [0.84, 1.73] | −0.35 [−0.45, −0.26] |
| B | +0.50 [−0.01, 1.01] | +3.68 [1.55, 5.80] | +4.49 [2.63, 6.35] | −0.18 [−0.82, 0.45] |
| C | +0.43 [0.32, 0.54] | +0.41 [0.18, 0.64] | +1.41 [0.91, 1.92] | −0.37 [−0.50, −0.23] |
| D | +0.47 [0.06, 0.88] | +2.69 [1.20, 4.19] | +3.72 [2.36, 5.08] | −0.29 [−0.80, 0.23] |
| F | +25.68 [25.00, 26.36] | +6.19 [5.54, 6.85] | +6.79 [6.26, 7.31] | −0.50 [−0.63, −0.37] |
| G | +23.86 [23.12, 24.60] | +15.39 [14.95, 15.83] | +10.40 [9.76, 11.04] | −3.51 [−3.76, −3.26] |

* With three 25 dB interferers 6–12° from the served directions (Scenario F), projection reconstruction and the fixed beam fall below −16 dB in user SINR; RDF-AS-ISAC stays 0.50 dB below the true-covariance solution. An exact conic solve on the projection covariance changes the held-out gain from 25.68 to 25.92 dB in F and from 23.86 to 24.33 dB in G.
* Sensing trade-off: in Scenarios B and D the held-out mean target SNR is 1.56 and 1.07 dB below radius-matched SOCP, which suppresses uplink leakage into the radar resource at the cost of 3.68 and 2.69 dB in user SINR.
* Constraint satisfaction (Section 7.7.9): over 180 scenes and 54,000 output weights no constraint is violated (smallest margin 1.0e-12); largest certified duality gap 6.7e-08; final gap to a tight conic solve at most 7.6e-09; median 61 fixed-point steps per refresh.
* Scenario E coverage (of 49 cells): RDF-AS-ISAC 41, projection reconstruction 49, base-radius SOCP 23, radius-matched SOCP 34, VU-MBWC 14.

`python -m ch7rdfisac.verify figures` compares 492 values with the chapter (every cell of Tables 7.3–7.6 and the numbers quoted in Sections 7.7.2–7.7.12); all 492 agree. Every stage except the SDP column is bit-identical to the run that produced the chapter; the SDP backend (SCS, first-order) differs by at most 0.01 dB and is not tabulated (SDP and SOCP agree within 0.0052 dB).

Timings depend on the machine. On the machine that produced `figures/` (two cores, Python 3.13, NumPy 2.5, single-thread BLAS) the full 300-observation RDF-AS-ISAC pass takes 152 ms at M = 32 and 1510 ms at M = 256, against 15.3 and 381.8 ms for one SOCP solve (Table 7.1 in `figures/`).

Figures show each method with the same colour and marker throughout, inside a full frame, with the legend in a framed box placed clear of the curves.

## Requirements

Python 3.10 or later, NumPy, SciPy, Matplotlib, CVXPY with the Clarabel and SCS solvers, and threadpoolctl (`requirements.txt` pins the versions used to produce `figures/`). Jupyter is needed only for the notebook.

## How to cite

Please cite the book chapter and the papers in `CITATION.cff`.

## License

MIT, see `LICENSE`.
