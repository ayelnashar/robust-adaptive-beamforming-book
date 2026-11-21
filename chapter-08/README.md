# Chapter 8: Wideband Robust Beamforming for Sub-Terahertz 6G Links

Python code for Chapter 8 of *Robust Adaptive Beamforming and Its Applications* by Ayman Elnashar (Wiley, forthcoming). It reproduces every simulation figure of the chapter (Figures 8.1–8.11), Tables 8.2 and 8.3 and the numbers quoted in the text.

The chapter develops WRAB, a wideband robust adaptive beamformer for fully digital sub-terahertz arrays, where large apertures and wide bands make carrier-referenced steering lose gain across the band (beam squint) and where a steering mismatch of a fraction of a degree already exceeds the beamwidth. WRAB treats every processed subcarrier separately and combines three mechanisms:

1. **Frequency-consistent constraints.** The steering vectors of each subcarrier's constraints are evaluated at that subcarrier's frequency, which removes the deterministic carrier-referenced steering error from the constraint set.
2. **A fan of worst-case constraints** around the presumed direction, Re{**w**ᴴ**a**(θᵢ, fₙ)} ≥ νᵢ‖**w**‖ + 1, spaced in beamwidths (8.15). Each snapshot takes a scale-invariant gradient step on the output power and then the exact Euclidean projection onto the intersection of all constraints (8.14), computed by an active-set Newton solve of its dual in Nc real variables, with Dykstra's method on the same sets as a checked fallback (Algorithm 8.1).
3. **Covariance-anchored margins.** The margins νᵢ are anchored once to the energy of each constraint outside the leading ⌈0.25M⌉ eigenvectors of the covariance (8.18) and then follow the alignment of the dominant eigenvector with each constraint, with a common broadening factor and a ceiling of 3, (8.19)–(8.20).

The comparators are analog-only steering, the single-ball worst-case beamformer with carrier-referenced and with frequency-aware steering (WC-single), frequency-aware MVDR, robust Capon ([24] in the chapter), the worst-case SOCP ([26], solved through its scalar KKT root), a simplified covariance-reconstruction beamformer after [30], the fixed-margin multi-constraint variant, the exact multi-ball SOCP of (8.8) (CVXPY with CLARABEL, WRAB's balls and final margins) and the loaded multi-point LCMV ([22]).

The chapter builds on the article: A. Elnashar, "Multi-Constraint Worst-Case Beamforming for Wideband Sub-Terahertz Arrays: Robustness to Steering Mismatch at Massive Array Sizes."

## Quick start

```bash
cd chapter-08
pip install -r requirements.txt

# all figures, tables and results_summary.json, chapter trial counts (about 12 minutes with two processes)
python scripts/generate_figures.py

# then compare every regenerated number with the value quoted in the chapter
python -m ch8wrab.verify figures

# smoke test into a separate folder (about 4 minutes)
python scripts/generate_figures.py --quick --outdir figures_quick

# selected figures, the tables, or the text results only (runs only the experiments they need)
python scripts/generate_figures.py --only 8_7 8_11
python scripts/generate_figures.py --only tables text

# redraw the figures from the simulation data saved by an earlier run
python scripts/generate_figures.py --replot

# notebook (figures appear under each cell)
jupyter notebook notebooks/Chapter8_WRAB.ipynb

# tests
pytest -q
```

The package can also be installed with `pip install -e .` and used as `ch8wrab`:

```python
import numpy as np
from ch8wrab import algorithms as C

cfg = C.Scenario(M=64, Nsc=3, K=300, mismatch_deg=2, jitter_deg=0.1)       # 300 GHz, 20 GHz band, Table 8.1 grid
r = C.evaluate_trial(cfg, C.seed_for('example', 0), C.CORE_METHODS)
print({C.METHOD_LABELS[m]: round(float(np.mean(10*np.log10(v))), 2) for m, v in r['linear'].items()})
```

## Contents

```text
ch8wrab/
  algorithms.py   ULA model at 300 GHz (subcarrier frequencies (8.1), frequency-dependent steering, carrier-fixed analog
                  response, 3-dB beamwidth); Scenario (settings of Table 8.1); trial generator (desired-direction jitter,
                  three-path angular spread, interference-only or signal-present training); WRAB (train_subband):
                  covariance (8.10), gradient step (8.11)-(8.13), exact projection (8.14) (project_wc), margins
                  (8.18)-(8.20) (margin_update); comparators of Section 8.4; exact multi-ball SOCP (multiball_socp);
                  multi-point LCMV; output SINR (8.5) and trial bootstrap; dynamic AR(1) array rotation (Figure 8.8(b));
                  frozen-norm one-step correction for the feasibility comparison of Section 8.3.1
  scenarios.py    one stage per group of results (analytic, fig3 ... fig11, fig11ref, fixedangle, dynamic, near,
                  controls, table3, feasibility, timing) and the runner (one process per stage; timing runs alone)
  figures.py      one function per figure (fig_8_1 ... fig_8_11), Tables 8.1-8.3, results collection
  verify.py       compares every regenerated number with the value quoted in the chapter
notebooks/        executed notebook, one section per figure
scripts/          command-line generation (--quick, --only, --replot, --jobs, --outdir, --verify)
figures/          PNG (300 dpi) and PDF of Figures 8.1-8.11, Tables 8.1-8.3 (CSV and Markdown), results_summary.json
tests/            sanity tests (projection against SLSQP and Dykstra, frozen-problem optimum, scale invariance, single-ball
                  equivalences, infeasibility rejection, no true-direction leakage, reproducibility, feasibility of every
                  update, the frozen-norm comparison of Section 8.3.1, exact SOCP)
```

## Figure map

| Chapter | Content | Function | Stage(s) |
|---|---|---|---|
| Figure 8.1 | Beam-squint array-factor loss across a 20 GHz band, N = 16, 64, 256 | `fig_8_1` | `analytic` |
| Figure 8.2 | Expected power-gain loss under Gaussian pointing error, N = 64, 256 | `fig_8_2` | `analytic` |
| Figure 8.3 | Per-subcarrier output SINR across the band, N = 64, 2° mismatch | `fig_8_3` | `fig3` |
| Figure 8.4 | Output SINR against steering mismatch, N = 64 | `fig_8_4` | `fig4` |
| Figure 8.5 | Output SINR against off-broadside steering angle | `fig_8_5` | `fig5` |
| Figure 8.6 | Output SINR against interferer–source separation | `fig_8_6` | `fig6` |
| Figure 8.7 | Array-size scaling: (a) 1.5 beamwidths of mismatch, (b) 1° for every N, two fans | `fig_8_7` | `fig7`, `fixedangle` |
| Figure 8.8 | Jitter at N = 128: (a) quasi-static source offset, (b) dynamic AR(1) array rotation | `fig_8_8` | `fig8`, `dynamic` |
| Figure 8.9 | Grid ablation: frequency-consistent against carrier-referenced constraints, 40 GHz | `fig_8_9` | `fig9` |
| Figure 8.10 | Margin initialization with adaptation on and with frozen margins | `fig_8_10` | `fig10` |
| Figure 8.11 | Ten-method comparison against steering mismatch | `fig_8_11` | `fig11`, `fig11ref` |
| Table 8.1 | Principal simulation settings | `tables` | |
| Table 8.2 | One interferer near the source: adaptive margins, fixed margins, multi-point LCMV | `tables` | `near` |
| Table 8.3 | Scenarios S1–S4, mean ± standard deviation | `tables` | `table3` |
| Text | Frozen-norm correction against the exact projection (8.3.1); online against frozen optimum (8.3.4); signed mismatch (8.5.3); margin-level control and margin state near interference (8.5.8); signal-present training (8.5.9); timings (8.3.4, 8.6) | `text_numbers` | `feasibility`, `controls`, `near`, `timing` |

## Simulation settings

* 300 GHz carrier, 20 GHz band (40 GHz in Figure 8.9), subcarriers at the centres of equal bins (8.1); ULA with half-wavelength spacing at the carrier; N = 64 elements in the core sweeps, 128 in Figures 8.8–8.10, 64–512 in Figure 8.7. SNR 5 dB and INR 10 dB per element; interferers at 45° and −60° unless a figure moves them.
* Forgetting factor 0.97, step relaxation 0.1, relative covariance load 10⁻⁶, projection tolerance 10⁻⁹. Constraint fan (0, −0.6, +0.6, −1.2) beamwidths around the presumed direction; fixed offsets (0, ±1.5, −3)° in Figures 8.3 and 8.9 and (0, ±0.4, −0.8)° in Figures 8.8 and 8.10; symmetric (0, ±0.6, ±1.2) beamwidths in Figure 8.11. Initial margins (1.9, 1.5, 1.5, 1.1) (0.3 in Figure 8.11), reset to the covariance anchor at the first margin update (warm-up 50 snapshots, period 30), retained eigenspace order ⌈0.25M⌉, ceiling 3. Single-ball radius 0.6√M (0.85√M in Figure 8.11, 3.8 in Figure 8.3).
* Training snapshots contain interference and noise only, except in the signal-present control of Section 8.5.9. The desired direction carries a block-static Gaussian offset (the jitter); Table 8.3 S4 adds three block-static paths with ±0.2° spread. Figure 8.8(b) uses an AR(1) rotation of the whole array with correlation 0.99 per snapshot.
* Each quasi-static run is scored with its final weight vector; Figure 8.8(b) averages the linear SINR of the last 50 snapshots, each with its own weight and geometry. A plotted point is the trial mean of the per-subcarrier SINR in dB; error bars are 95% bootstrap intervals over trials (2000 resamples). Trials per point: 15 (Figures 8.3–8.5, 8.9), 18 (Figure 8.6, Table 8.2), 12 (Figure 8.7), 20 (Figures 8.8, 8.10, 8.11, Table 8.3), 8 (controls).
* Seed 42 with an independent stream per experiment (`seed_for(stream, trial)` in `ch8wrab/algorithms.py`), so every number can be regenerated exactly.

## Key results

Output SINR in dB (trial mean of the per-subcarrier SINR), 64 elements, 300 GHz, 20 GHz band:

| Method | Fig. 8.4, 0° | Fig. 8.4, 2° | Fig. 8.11, 0° | Fig. 8.11, 2° | Fig. 8.11, 4° | Fig. 8.11, 8° |
|---|---|---|---|---|---|---|
| WRAB | 16.8 | 18.9 | 15.9 | 18.3 | 7.6 | 2.4 |
| Exact multi-ball SOCP of (8.8) | | | 15.8 | 18.2 | 7.4 | 2.3 |
| Multi-point LCMV [22], loaded | | | 17.1 | 17.1 | 7.6 | −1.9 |
| Fixed-margin multi-WC | | | 22.3 | 12.1 | 5.1 | 1.5 |
| Cov. reconstruction [30], simplified | | | 22.5 | 6.8 | −25.1 | −12.6 |
| WC-single (carrier-referenced) | 22.9 | 0.3 | 22.9 | 3.1 | 2.5 | 0.0 |
| Analog-only | 22.4 | −0.8 | 21.9 | 2.2 | 1.6 | −1.0 |

* At two degrees of mismatch in the ten-method comparison (Figure 8.11), WRAB is 11.5 dB above the best of the six single-direction methods, and the exact interior-point solution of (8.8) with the same balls stays within 0.2 dB of WRAB at every point. With distant interferers the heavily loaded multi-point LCMV matches or exceeds WRAB up to three degrees, except at two degrees; from five degrees on WRAB is 2.4 to 5.0 dB higher.
* Margins (Section 8.5.8, Table 8.2): away from interference the adaptive margins reach the ceiling and equal a fixed margin of 3 (18.31 dB at two degrees, against 11.58 dB for a fixed margin of 0.3). With one interferer two or three degrees from the source the rule lowers only the overlapping margin (median 0.01 to 0.02) and keeps 6.2 and 7.7 dB, while every fixed level stays at or below 0.2 dB and the multi-point LCMV gives −5.2 dB.
* Array size (Figure 8.7): at 1.5 beamwidths of mismatch WRAB rises from 16.4 dB at 64 elements to 20.9 dB at 512. At a fixed 1° error the beamwidth-relative fan falls to 7.1 dB at 512 elements, while a fan of span ±1.5° with 0.6-beamwidth spacing (5 to 27 constraints) holds 16.3 to 17.9 dB at every size.
* Signal present in training (Section 8.5.9): at two degrees WRAB keeps 15.5 dB while the adaptive conventional beamformers fall to −23.0 to −10.4 dB; at four degrees the source is outside the fan and WRAB also nulls it (−19.1 dB).
* Feasibility (Section 8.3.1): the largest constraint residual over all runs is 1.0 × 10⁻⁹. On the same 64-element snapshots the cheaper frozen-norm one-step correction leaves the constraints violated in 296 of 300 updates, by up to 0.67. After 600 online updates the objective is 14 to 20 percent above the optimum of the frozen final problem (Section 8.3.4).

Timings depend on the machine. On the machine that produced `figures/` (two cores, Python 3.13, NumPy 2.5, single-thread BLAS, Numba), the Figure 8.7(a) loop takes 0.20, 0.26, 0.80 and 3.49 ms per subcarrier and snapshot at 64, 128, 256 and 512 elements; one WRAB update in the Figure 8.11 setting takes 0.12 ms against 20.8 ms for one interior-point solve of (8.8) (ratio about 180); with 27 constraints at 512 elements one WRAB update takes 3.1 ms. All values are in `figures/results_summary.json`; `python -m ch8wrab.verify figures` compares 188 of them with the chapter.

Figures show each method with the same colour, marker and line style throughout, inside a full frame, with the legend in a framed box placed clear of the curves.

## Requirements

Python 3.10 or later, NumPy, SciPy, Matplotlib, CVXPY with the CLARABEL solver (exact multi-ball SOCP only) and threadpoolctl; Numba is optional and speeds up the projection kernels without changing the results (`requirements.txt` pins the versions used to produce `figures/`). Jupyter is needed only for the notebook.

## How to cite

Please cite the book chapter and the papers in `CITATION.cff`.

## License

MIT, see `LICENSE`.
