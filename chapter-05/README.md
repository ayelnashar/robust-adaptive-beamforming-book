# Chapter 5: Multi-Source Robust Beamforming with Subspace Tracking

Python code for Chapter 5 of *Robust Adaptive Beamforming and Its Applications* by Ayman Elnashar (Wiley, forthcoming). It reproduces every simulation figure of the chapter (Figures 5.1–5.10), the result tables (Tables 5.4–5.12) and the numbers quoted in the text.

The chapter develops **MPR-RAB**, a multi-primary robust adaptive beamformer for sparse multipath channels in which the desired signal reaches the array through several clusters, as in the upper mid-band. At every update it solves the multi-primary worst-case problem (5.8)

```text
min_w  w^H R_in w   s.t.   Re{w^H a~_k} >= nu_k ||w|| + beta_k,   k = 1, ..., K
```

with four components:

1. **Reconstructed covariance** (Section 5.4). R_in is integrated from the Capon spectrum over the visible region outside sectors of half a beamwidth around the tracked clusters; strong spectral peaks are located between the grid points by a golden-section search before the integration, so that a strong interferer between grid lines is still nulled.
2. **Margins from joint feasibility** (Section 5.3.2). nu_k = κ √(M/K), with κ = 0.3 on the planar and κ = 0.4 on the linear array.
3. **Gain targets from the max-SINR beamformer** (Section 5.3.3). The targets beta_k and the constraint phases are the normalized responses of the max-SINR beamformer of the estimated model on a reference covariance R_t; at zero margin the solution is that beamformer (Proposition 5.2).
4. **Active-set KKT solver of Chapter 2** (Section 5.3.5) with certification of every update by its KKT residuals, direction tracking on an interferer-deflated covariance with compensation of the delay of the covariance memory, and a cluster lifecycle (Section 5.5).

The comparators, all fed with the same tracked directions and cluster powers, are LCMV-K (noise-floor loading), Par-K-RCB (one robust Capon beamformer per cluster), LCMV-K-D (derivative constraints), INCM-LCMV, INCM-MVDR and INCM-GEV on the reconstructed covariance, and the oracle of Proposition 5.1.

The chapter builds on the article: A. Elnashar, "Multi-Primary Robust Adaptive Beamforming With Per-Cluster Gain Guarantees for Sparse Multipath Channels."

## Quick start

```bash
cd chapter-05
pip install -r requirements.txt

# all figures and results_summary.json, full Monte Carlo counts (about 5 hours with two processes)
python scripts/generate_figures.py

# smoke test into a separate folder (about 30 minutes)
python scripts/generate_figures.py --quick --outdir figures_quick

# selected figures only (runs only the stages they need)
python scripts/generate_figures.py --only 5_7 5_9

# redraw the figures from the simulation data saved by an earlier run
python scripts/generate_figures.py --replot

# also time this machine and draw Figure 5.10 from that timing
python scripts/generate_figures.py --timing

# rerun both training phases and use the selections derived from them
python scripts/generate_figures.py --retrain

# notebook (figures appear under each cell)
jupyter notebook notebooks/Chapter5_MPR_RAB.ipynb

# tests
pytest -q
```

The package can also be installed with `pip install -e .` and used as `ch5mprrab`:

```python
import numpy as np
from ch5mprrab import scenarios as S, algorithms as A

d = S.make_data('C', 5, 10, 2, seed=777)           # linear array, drifting clusters, 2 deg initial pointing error
c = d['c']
h, log = A.run_mpr_rab(d['X'], d['a0'], list(d['th0']), K_track=c['K_track'], K_use=c['K_use'],
                       use_tracking=True, use_lifecycle=c['lc'], kappa=0.4, targets='ms')
m = S.metrics(h, d, 'C')
print(m[''], m['|floor'])                            # expected and minimum-cluster SINR (dB), last 50 snapshots
```

Call `S.use_array('upa')` first, in a fresh process, to work on the 8 × 8 planar array.

## Contents

```text
ch5mprrab/
  channel.py      signal model: steering vector, multi-cluster channel with drift, births and deaths, expected SINR (5.4),
                  oracle (Proposition 5.1)
  algorithms.py   MPR-RAB kernel (run_mpr_rab): reconstruction with off-grid peak refinement, max-SINR targets
                  (maxsinr_responses), margins, active-set KKT solver (as_snc) with residual certification, tracking on
                  the deflated covariance, delay compensation, lifecycle; minimum-cluster SINR (5.5)
  planar.py       8 x 8 planar array: steering model (5.20), direction encoding, reconstruction grid, the nine scenarios
  baselines.py    LCMV-K, Par-K-RCB, LCMV-K-D, INCM-LCMV, INCM-MVDR, INCM-GEV
  scenarios.py    scenarios A-I, conditions, protocol (training, condition study, separate stream), metrics, selection
                  rule, bootstrap, the list of stages and the runner (one process per stage)
  experiments/    one module per stage: study (phases), select, report, sensitivity, tracking_c, timing, pattern,
                  solver, prop1, prop2, feasibility, geometry; each writes its results into <outdir>/data
  figures.py      one function per figure (fig_5_1 ... fig_5_10), results collection (write_summary)
  data/           selection of the chapter (selection.json) and the timing of Table 5.1 (timing_recorded.json)
notebooks/        executed notebook, one section per figure
scripts/          command-line generation (--quick, --only, --replot, --timing, --retrain, --jobs, --outdir)
figures/          PNG (300 dpi) and PDF of Figures 5.1-5.10, results_summary.json
tests/            sanity tests (Proposition 5.2, constraints and certification, planar steering, oracle bound, seeds)
```

## Figure map

| Chapter | Content | Function | Stages |
|---|---|---|---|
| Figure 5.1 | Planar array, scenario A: expected SINR per snapshot | `fig_5_1` | `study upa heldout A` |
| Figure 5.2 | Planar array, scenario C: expected SINR and tracking error | `fig_5_2` | `study upa heldout C` |
| Figure 5.3 | Planar array, scenario D: cluster birth and death | `fig_5_3` | `study upa heldout D` |
| Figure 5.4 | Expected against minimum-cluster SINR, κ from 0 to 0.9, three target rules | `fig_5_4` | `study upa heldout A`–`I` |
| Figure 5.5 | Beam pattern of the final MPR-RAB weight, planar scenario H | `fig_5_5` | `pattern` |
| Figure 5.6 | Planar array, scenario I: shadowing of the strongest cluster | `fig_5_6` | `study upa heldout I` |
| Figure 5.7 | Linear array, scenario C, with and without delay compensation | `fig_5_7` | `study ula stream C`, `tracking_c` |
| Figure 5.8 | Sensitivity to κ on the planar training study | `fig_5_8` | `study upa train A`–`I` |
| Figure 5.9 | Interferer crossing a tracked cluster, linear array | `fig_5_9` | `sensitivity` |
| Figure 5.10 | Time per snapshot against array size | `fig_5_10` | recorded timing (Table 5.1) unless `--timing` |
| Tables 5.4–5.9 | Planar operating point, condition study, paired differences, minimum-cluster SINR, ablations, shadowing | `write_summary` | `study upa heldout A`–`I` |
| Tables 5.10, 5.11 | Linear operating point and paired differences | `write_summary` | `study ula heldout A`–`I` |
| Table 5.12, text | Sensitivity to the design constants; deflation, tracking off, off-grid interferers (Section 5.7.7) | `write_summary` | `sensitivity` |
| Text | Proposition 5.1 and 5.2 checks, joint feasibility, Gram conditioning, solver pass counts and certification | `write_summary` | `prop1`, `prop2`, `feasibility`, `geometry`, `solver ula`, `solver upa` |

## Simulation settings

* 8 × 8 planar array (M = 64, main study) and 16-element linear array, half-wavelength spacing. SNR 5 dB per unit-power cluster; two interferers with INR 10 dB each, at (+45°, 0°) and (−10°, −25°) in azimuth and elevation on the planar array and at +45° and −60° on the linear array; noise power 1; forgetting factor η = 0.97.
* Nine scenarios (Table 5.2): static clusters (A, B, E, F, G, H), drift of 0.025° per snapshot (C), birth at snapshot 150 and death at snapshot 280 (D), shadowing of the strongest cluster on snapshots 150–249 (I); scenario E is a synthetic five-cluster profile with the three strongest clusters protected. Six conditions (Table 5.3): SNR / INR / initial pointing error D of 5/10/0 (operating point), 5/10/2, 5/10/4, 10/20/0, 20/10/0 and 5/35/0.
* MPR-RAB (Table 5.13): update interval 10 snapshots, warm-up 20, grace period 90, reconstruction grid 0.5° (linear) and 1° × 2° (planar), sectors of half the 3 dB beamwidth, target sectors 1.5 times wider, off-grid threshold 10 times the median of the spectrum, refinement window ±3° with a 1° move cap, deflation threshold 0.3, delay compensation from the last 8 refinements (at least 4), retirement below 5 % of the strongest cluster, solver tolerance 10⁻⁸ on the scaled KKT residuals with at most 100 inner passes.
* Metrics: expected SINR (5.4) and minimum-cluster SINR (5.5) with the true interference-plus-noise covariance, averaged over the last 50 snapshots of every run.
* Protocol: every family's parameter (loading, uncertainty radius, derivative loading, exclusion sector, κ) is selected on a training stream (4 runs per cell on the planar array, 8 on the linear array) by the largest pooled expected SINR, settings within 0.05 dB being resolved by the larger minimum-cluster SINR; the results come from an independent stream with 30 runs per cell (seed 2026000 + 10000 × scenario + 100 × condition + run), and the paired differences carry 95 % percentile bootstrap intervals from 3000 resamples. The ablations of Table 5.8 use the first 10 runs of every cell. Figure 5.7 uses a separate stream (seed 42 once, scenarios in order, 15 runs). Every realization has a fixed seed and no method touches the global random stream, so every method of a cell sees identical data.

## Key results

RESULTS_PLACEHOLDER

## Requirements

Python 3.10 or later, NumPy, SciPy and Matplotlib (`requirements.txt` pins the versions used to produce `figures/`). Jupyter is needed only for the notebook. The stages set single-threaded BLAS so that two of them can run side by side.

## How to cite

Please cite the book chapter and the papers in `CITATION.cff`.

## License

MIT, see `LICENSE`.
