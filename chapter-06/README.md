# Chapter 6: Distributed (Cell-Free) Massive MIMO

Python code for Chapter 6 of *Robust Adaptive Beamforming and Its Applications* by Ayman Elnashar (Wiley, forthcoming). It reproduces every simulation figure of the chapter (Figures 6.1–6.10), Tables 6.2–6.6 and the numbers quoted in the text.

The chapter brings worst-case robust combining to cell-free massive MIMO. Every access point (AP) holds its own imperfect estimate of each user's channel, inside a bounded per-AP uncertainty ball whose radius is stated relative to the presumed channel (6.2). Two realizations share one multi-constraint robust LCMV solver (Algorithm 6.1: a nominal MVDR step followed by a working-set correction onto the binding worst-case constraints of the grid (6.11)):

1. **CF-RAB (centralized)** stacks all L APs and solves one joint robust problem of dimension LM at the central processing unit (CPU).
2. **D-CF-RAB** (Algorithm 6.2) solves one small exact single-ball problem per AP at the local bound level c (Section 6.3.6: the solution is a loaded MVDR weight whose loading is the root of a scalar equation), forwards the L local outputs, weights them with the L-dimensional CPU Wiener combiner (6.13) and rescales the combined weight when the cross-AP check of the worst-case combined response (6.14) requires it. Only L-dimensional quantities cross the fronthaul: 1280 listed bytes per user update at L = 16, against 2048 for P-MMSE and 4352 for CF-RAB.

The comparators are MR, LP-MMSE and P-MMSE combining as defined by Björnson and Sanguinetti (*IEEE Trans. Commun.*, 2020; [6] in the chapter), matched two-stage baselines that apply the CPU combiner (6.13) to a different local stage (MR + CPU combiner, LP-MMSE + CPU combiner, local MVDR + CPU combiner), INCM reconstruction ([20] in the chapter) and the centralized primary-ball SOCP benchmark (CVXPY with SCS).

The chapter builds on the article: A. Elnashar, "Robust Adaptive Beamforming for Cell-Free Massive MIMO under Heterogeneous Per-AP Uncertainty."

## Quick start

```bash
cd chapter-06
pip install -r requirements.txt

# all figures, tables and results_summary.json, full Monte Carlo counts (about 40 minutes with two processes)
python scripts/generate_figures.py

# then compare every regenerated number with the value quoted in the chapter
python -m ch6cfrab.verify figures

# smoke test into a separate folder (about 9 minutes)
python scripts/generate_figures.py --quick --outdir figures_quick

# selected figures, the tables, or the text results only (runs only the experiments they need)
python scripts/generate_figures.py --only 6_2 6_9
python scripts/generate_figures.py --only tables text

# redraw the figures from the simulation data saved by an earlier run
python scripts/generate_figures.py --replot

# notebook (figures appear under each cell)
jupyter notebook notebooks/Chapter6_CF_RAB.ipynb

# tests
pytest -q
```

The package can also be installed with `pip install -e .` and used as `ch6cfrab`:

```python
import numpy as np
from ch6cfrab.algorithms import CFConfig, run_snapshot_trace

r = run_snapshot_trace(CFConfig(L=16, M_l=8, K=4), n_snapshots=60, SNR_dB=10, INR_dB=10,
                       methods=('D-CF-RAB', 'CF-RAB', 'P-MMSE'), seed=700, soi_in_cov=True)
print({m: round(float(np.nanmean(v)), 2) for m, v in r['sinr_dB'].items()})   # mean per-user SINR in dB
```

## Contents

```text
ch6cfrab/
  algorithms.py   cell-free channel generator (Rician directions at 7.125 GHz, normalized per-AP gains or the path-loss
                  model of Section 6.7.1, AR(1) evolution, CSI delay, CSI errors on the boundary of the ball (6.2) or the
                  other error laws of Section 6.7.13); Algorithm 6.1 (rab_step) with the constraint grid (6.11); exact
                  single-ball solution (exact_wc); D-CF-RAB step of Algorithm 6.2 (cf_distributed_step); CF-RAB;
                  MR, LP-MMSE, P-MMSE, local MVDR and INCM-reconstruction combiners; primary-ball SOCP; snapshot loop
                  that scores every method on the same snapshots (run_snapshot_trace); timing and fronthaul inventory
  scenarios.py    one stage per group of results (fig1 ... fig10, train, paired, audits, lbound, A1 ... A5, B1, B2) and
                  the runner (one process per stage; the timing stage of Figure 6.6 runs alone at the end)
  figures.py      one function per figure (fig_6_1 ... fig_6_10), Tables 6.2-6.6, results collection
  verify.py       compares every regenerated number with the value quoted in the chapter
notebooks/        executed notebook, one section per figure
scripts/          command-line generation (--quick, --only, --replot, --jobs, --outdir, --verify)
figures/          PNG (300 dpi) and PDF of Figures 6.1-6.10, Tables 6.2-6.6 (CSV and Markdown), figure_data.csv,
                  results_summary.json
tests/            sanity tests (exact single-ball solution, ball (6.2), Algorithm 6.1 feasibility, listed fronthaul bytes,
                  reproducibility, fixed-seed reference values)
```

## Figure map

| Chapter | Content | Function | Stage(s) |
|---|---|---|---|
| Figure 6.1 | Mean per-user SINR against the number of APs L | `fig_6_1` | `fig1` |
| Figure 6.2 | Mean and worst-user SINR against the CSI uncertainty radius ε | `fig_6_2` | `fig2` |
| Figure 6.3 | Mean per-user SINR against the per-antenna SNR | `fig_6_3` | `fig3` |
| Figure 6.4 | Mean per-user SINR against the CSI delay τ | `fig_6_4` | `fig4` |
| Figure 6.5 | Uniform and tiered per-AP CSI uncertainty | `fig_6_5` | `fig5` |
| Figure 6.6 | Time per weight update and listed CSI and control bytes per user update | `fig_6_6` | `fig6` (timing, run alone) |
| Figure 6.7 | Mean per-user SINR against the trace length | `fig_6_7` | `fig7` |
| Figure 6.8 | Algorithm 6.1 against the primary-ball conic benchmark | `fig_6_8` | `fig8`, `fig2` |
| Figure 6.9 | Signal self-cancellation with the desired signal in the covariance | `fig_6_9` | `fig9` |
| Figure 6.10 | Direction change of Algorithm 6.1 against the bound level | `fig_6_10` | `fig10` |
| Tables 6.2, 6.3 | Load-dependent bound and loading selection, exact and 256-snapshot sample covariance | `tables` | `B1` |
| Table 6.4 | CSI-error laws | `tables` | `A1` |
| Table 6.5 | Interferers without channel estimates, non-Gaussian symbols | `tables` | `B2` |
| Table 6.6 | Observations in the simulated parameter ranges | `tables` | (summary of the above) |
| Text | Bound selection of the main study (6.7.1); scalar-root solver check (6.3.6); 20-realization paired comparisons (6.7.2, 6.7.3); equal-mean tiers (6.7.6); exact and Algorithm 6.1 local stages, conic audits, CF-RAB bound sweep, joint-check audit (6.7.9); rescaling outcomes (6.7.11); load-dependent selection (6.7.12); error laws and continuous-ball checks (6.7.13) | `text_numbers` | `train`, `paired`, `audits`, `lbound`, `A2`–`A5` |

The figures also show MR + CPU combiner, which the chapter describes in the text but does not plot because its curves overlap those of D-CF-RAB (within 0.05 dB in Figures 6.1–6.5 and 6.7).

## Simulation settings

* Service area 500 m × 500 m; L = 16 APs (4, 8 and 16 in Figures 6.1 and 6.6; 8 in Figure 6.7) with M = 8 antennas each, uniform linear arrays at half-wavelength spacing; K = 4 users with unit transmit power; Rician factor 7 dB at 7.125 GHz.
* Per-antenna SNR 10 dB, INR 10 dB, ε = 0.10 and zero CSI delay unless a figure varies them. Per-AP channel gains normalized to √M times a gain uniform within ±3 dB; Sections 6.7.12 and 6.7.13 also use the urban-macro path-loss model with 8 dB shadowing.
* Every CSI error lies on the boundary of the declared relative ball (6.2); true channels evolve as AR(1) with correlation 0.995 per snapshot. Covariance-based methods use the population covariance (6.4) at the first snapshot of each trace (the 256-snapshot sample covariance in Tables 6.3 and 6.5).
* D-CF-RAB updates its weights at every snapshot with the local bound level c = 0.98, selected on three training realizations not used in any figure; the other combiners are rebuilt every fifth snapshot after the first quarter of the trace. All methods are scored on the same nine rebuild snapshots of each 60-snapshot trace.
* Monte Carlo: five realizations per point in Figures 6.1–6.7 (pointwise 95% Student-t intervals); paired comparisons over 20 realizations; bound and loading selection on eight training realizations, evaluated on 20 held-out realizations (12 in Table 6.5). Every realization has a fixed seed (see `ch6cfrab/scenarios.py`), so every number can be regenerated exactly.

## Key results

Mean per-user SINR in dB, nominal point (L = 16, M = 8, K = 4, SNR = INR = 10 dB, ε = 0.10), five realizations:

| Method | Fig. 6.1, L = 4 | Fig. 6.1, L = 16 | Fig. 6.2, ε = 0.10 | Fig. 6.2, ε = 0.50 | Fig. 6.4, τ = 64 |
|---|---|---|---|---|---|
| D-CF-RAB | 16.32 | 25.74 | 26.38 | 17.40 | 14.07 |
| P-MMSE [6] | 21.91 | 28.30 | 28.68 | 16.31 | 13.18 |
| LP-MMSE + CPU combiner | 19.37 | 25.15 | | | |
| CF-RAB (centralized) | 10.12 | 15.53 | 16.39 | 13.39 | |

* D-CF-RAB stays within 0.03 dB of MR + CPU combiner in the nominal configuration, and 3.79 to 4.01 dB above local MVDR + CPU combiner with a 10⁻⁶ loading over the whole radius sweep of Figure 6.2: the bound acts as a loading rule. P-MMSE is 2.5 dB above D-CF-RAB at nominal uncertainty; the curves cross between ε = 0.3 and 0.5, and at ε = 0.50 D-CF-RAB is 1.09 dB above P-MMSE.
* Load-dependent bound selection (Table 6.2): with four APs the selected bound (0.4 or 0.6) raises D-CF-RAB by up to 2.74 dB over the fixed c = 0.98 and by up to 2.81 dB over MR + CPU combiner; with 16 APs the selected bound changes D-CF-RAB by at most 0.04 dB.
* Interferers without channel estimates (Table 6.5): covariance-based local processing is 3.75 to 6.77 dB above the estimate-based combiners (MR + CPU, LP-MMSE + CPU, P-MMSE); local MVDR with a training-selected loading is 0.11 to 1.36 dB above D-CF-RAB.
* Algorithm 6.1 against the primary-ball SOCP (Figure 6.8): objective excess 1.8% at ε = 0.005 to 6.6% at ε = 0.50; SINR difference at most 0.11 dB over Figures 6.1, 6.2, 6.3 and 6.5. With the desired signal in the covariance and a channel error (Figure 6.9, ε = 0.10), loaded MVDR falls from −11.46 to −41.35 dB as the desired-signal power rises from 0 to 30 dB, while CF-RAB rises from −6.56 to 31.42 dB.
* Rescaling thresholds (Figure 6.10): 0.164 for the per-AP problems (M = 8) and 0.535 for the joint problem (LM = 128).
* Listed CSI and control bytes per user update at L = 16: 1280 for D-CF-RAB, 2048 for P-MMSE, 4352 for CF-RAB.

Timings depend on the machine. On the machine that produced `figures/` (two cores, Python 3.13, NumPy 2.5, single-thread BLAS) a D-CF-RAB update at L = 16 (16 local solves and the CPU combination) takes 3.17 ms, against 0.72 ms for CF-RAB, 0.87 ms for P-MMSE and 32.4 ms for the primary-ball SOCP. All values are in `figures/results_summary.json`; `python -m ch6cfrab.verify figures` compares 320 of them with the chapter.

Figures show each method with the same colour, marker and line style throughout, inside a full frame, with the legend in a framed box placed clear of the curves.

## Requirements

Python 3.10 or later, NumPy, SciPy, Matplotlib, CVXPY with the SCS and Clarabel solvers, and threadpoolctl (`requirements.txt` pins the versions used to produce `figures/`). Jupyter is needed only for the notebook.

## How to cite

Please cite the book chapter and the papers in `CITATION.cff`.

## License

MIT, see `LICENSE`.
