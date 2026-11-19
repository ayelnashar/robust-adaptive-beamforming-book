# Chapter 4: Robust Beamforming that Minimizes Bit-Error-Rate Directly

Python code for Chapter 4 of *Robust Adaptive Beamforming and Its Applications* by Ayman Elnashar (Wiley, forthcoming). It reproduces every simulation figure of the chapter (Figures 4.1–4.11) and the numbers quoted in the text.

The chapter develops **guided R-MBER**, a blind minimum-bit-error-rate beamformer that needs no training symbols and no carrier-phase reference. It works in two stages:

1. **Spectral guide** (Section 4.4.1). Every L = 25 snapshots the guide forms the Capon spectrum of the exponentially weighted covariance on 361 directions, keeps the peaks that pass a local contrast test, merges peaks within one half-power beamwidth, selects the source within Δ = 6° of the presumed direction (widened to 14° if needed), reconstructs the interference-plus-noise covariance outside a sector around it (4.9), estimates the steering vector, and forms the guide weight (4.10).
2. **BER step** (Section 4.4.2). One pass of a smoothed-BER gradient step over the block on the beamformer's own decisions after blind phase recovery, with the step scaled by the inverse of the reconstructed covariance:

```text
p_k = -mu Psi g_k,     Psi = R_inc^-1 / lambda_max(R_inc^-1)                         (4.11)
```

The scaling makes the step the steepest-descent step of the smoothed BER in the metric of the reconstructed covariance and bounds the change of the reconstructed interference-plus-noise power by a term proportional to its smallest eigenvalue (4.13). The comparators are WC-MVDR/EigDec [6], INCM reconstruction (INCM-REC) after [11], spatial power spectrum sampling (SPSS) [14], the constraint-based R-MBER of Section 4.5 in its blind and trained forms, and a training-aided Wiener filter (TA-MMSE) with every symbol known or with a preamble of 30 pilots.

The chapter builds on the article: A. Elnashar, "Spectrally Guided Blind Minimum-Bit-Error-Rate Beamforming Robust to Steering-Vector and Training-Reference Mismatch."

## Quick start

```bash
cd chapter-04
pip install -r requirements.txt

# all figures and results_summary.json, full Monte Carlo counts (about 20 minutes with two processes)
python scripts/generate_figures.py

# smoke test into a separate folder (a few minutes)
python scripts/generate_figures.py --quick --outdir figures_quick

# selected figures only (runs only the experiments they need)
python scripts/generate_figures.py --only 4_2 4_9

# redraw the figures from the simulation data saved by an earlier run
python scripts/generate_figures.py --replot

# also time this machine and draw Figure 4.1 from that timing
python scripts/generate_figures.py --timing

# notebook (figures appear under each cell)
jupyter notebook notebooks/Chapter4_Guided_RMBER.ipynb

# tests
pytest -q
```

The package can also be installed with `pip install -e .` and used as `ch4rmber`:

```python
import numpy as np
from ch4rmber import guided, gen_with_symbols, steer, sinr_db, R_ipn, ps, M, K

np.random.seed(1000)
X, S = gen_with_symbols(M, K, 0.0, None)          # 150 snapshots, BPSK source at broadside
w_guide, w = guided(X, 7.0)                       # presumed direction 7 degrees, no training symbols
print(sinr_db(w, steer(M, 0.0), R_ipn, ps))        # output SINR in dB
```

## Contents

```text
ch4rmber/
  algorithms.py   signal model; smoothed-BER cost and gradient with the adaptive bandwidth and its floor; decision labels
                  with blind phase recovery; spectral guide (run_sgrab_p), scaled BER step (ber_stage), guide and guided;
                  constraint-based R-MBER, trained (run_rmber) and blind (run_blind), with the exact Dykstra projection;
                  WC-MVDR/EigDec (exact solution), INCM-REC, SPSS, TA-MMSE, MVDR, DL-MVDR and the adaptive MBER
                  beamformer; data generators; SINR and model-based BER
  scenarios.py    settings of Section 4.7.1, the ordered list of experiments and the runner (one process per stage)
  experiments/    one module per group of experiments; each stage prints its numbers and writes res_<name>.json
  figures.py      one function per figure (fig_4_1 ... fig_4_11), results collection
  data/           runtime recorded for Figure 4.1
notebooks/        executed notebook, one section per figure
scripts/          command-line generation (--quick, --only, --replot, --timing, --jobs, --outdir)
figures/          PNG (300 dpi) and PDF of Figures 4.1-4.11, results_summary.json
tests/            sanity tests (guide schedule and direction, exact WC-MVDR, projection, bound (4.13), reproducibility)
```

## Figure map

| Chapter | Content | Function | Experiments |
|---|---|---|---|
| Figure 4.1 | Measured runtime versus array size | `fig_4_1` | `main_experiments time` (recorded run unless `--timing`) |
| Figure 4.2 | Output SINR versus mismatch, M = 8, 16, 32, 64 | `fig_4_2` | `main_experiments sweep8 ... sweep64`, `spss` |
| Figure 4.3 | The eight mismatch and propagation cases, with the guide alone | `fig_4_3` | `trained_baselines cases`, `guided_experiments mp`, `full_array_baselines mp`, `main_experiments spss` |
| Figure 4.4 | Output SINR versus snapshots processed, 7° | `fig_4_4` | `trained_baselines traj`, `main_experiments conv` |
| Figure 4.5 | Model-based BER versus mismatch, element SNR −5 dB | `fig_4_5` | `ber_baselines mm`, `ber_guided ber`, `main_experiments berb`, `spss` |
| Figure 4.6 | Model-based BER versus element SNR, 5° | `fig_4_6` | `ber_baselines snr`, `ber_guided ber`, `main_experiments berb`, `spss` |
| Figure 4.7 | Interference level, K = 150 and 1000 | `fig_4_7` | `guided_experiments inr`, `main_experiments inr1000`, `inrx`, `trained_baselines inr`, `inr1000`, `spss` |
| Figure 4.8 | Element SNR and number of interferers | `fig_4_8` | `full_array_baselines snr`, `nint`, `main_experiments snr`, `guided_experiments nint`, `spss` |
| Figure 4.9 | A third interferer near the source | `fig_4_9` | `guided_experiments near`, `main_experiments near`, `spss` |
| Figure 4.10 | Imperfect training reference | `fig_4_10` | `imperfect_reference contam`, `onset`, `labels`, `drift`, `slip`, `main_experiments spss` |
| Figure 4.11 | BER step under local scattering | `fig_4_11` | `local_scattering amp`, `off` |
| Text | Bandwidth floor and floor coefficient (4.3.1); Dykstra sweeps, steering-vector distances, radius perturbation (4.5); WC-MVDR/EigDec coverage, the two realization sets (4.7.1); BER-step gains in the eight cases (4.7.3); held-out BER check (4.7.5); TA-MMSE from 30 pilots (4.7.8); direction-only step (4.7.9); constraint-based baselines with and without phase recovery and half-spaces (4.7.10); TA-MMSE loading and SPSS taper (4.7) | `text_numbers` | `floor_and_heldout`, `text_checks`, `local_scattering ci`, `wc`, `dir`, `main_experiments ta30`, `phase_recovery`, `no_constraints`, `blind_no_halfspaces` |

## Simulation settings

* Half-wavelength ULA, M = 32 unless stated (8, 16, 32 and 64 in the mismatch sweep; 16 to 128 in the runtime test). BPSK desired source at broadside with SNR 5 dB; two interferers at 45° and −60°, INR 10 dB each; noise power 1 per element. K = 150 snapshots per block (K = 1000 in Figure 4.7(c), (d)).
* Multipath: small angular spread with rays at +4° and −3° (amplitude 0.4 each), large angular spread with rays at −25° (0.5) and +35° (0.3). Ray phases are drawn once per block and held fixed; SINR is scored with the realized effective steering vector against the population interference-plus-noise covariance.
* Guided R-MBER: L = 25, α = 0.97, R̂(0) = I, loading δ = 10⁻³ tr(R̂)/M, 361 directions, peak test with one-beamwidth test window and guard and three-beamwidth reference bands, merging within 1.772/(M − 1) rad, Δ = 6° widened to 14°, sector 4° clear of other sources and at least 2° wide; the first spectral refresh is at snapshot 26 and earlier snapshots use the multiple worst-case constraint recursion of Chapter 3 started at the presumed direction. BER step: Z = 80, μ = 0.1, step norm at most 1, ball κ = 3, bandwidth floor c = 0.5.
* Constraint-based R-MBER: grid [θp, θp − 2°, θp + 2°, θp − 4°] (4.15), bounds ν = 0.5 ν₀ with ν₀ = [1.9, 1.5, 1.5, 1.1] √(M/16), κ = 3, at most 60 Dykstra sweeps with tolerance 10⁻¹⁰.
* WC-MVDR/EigDec: exact solution, radius ζ = 3.8 √(M/16), exponentially weighted covariance with weight 0.99. INCM-REC: sector ±8°, 200 directions over ±90°. SPSS [14]: sector θp ± 6°, no loading. TA-MMSE: loading σ².
* Monte Carlo: 20 realizations (Figures 4.2, 4.3, 4.7 to 4.11), 12 (Figure 4.4), 15 (Figures 4.5, 4.6). A run fails when its SINR is more than 3 dB below the optimum on the same block. Every realization has a fixed seed (see `ch4rmber/scenarios.py`), so every number can be regenerated exactly.

## Key results

Output SINR in dB, M = 32, mean of 20 realizations (Figure 4.3):

| Case | Guide alone | Guided R-MBER | INCM-REC | SPSS | WC-MVDR/EigDec | Blind R-MBER | Trained R-MBER | Optimum |
|---|---|---|---|---|---|---|---|---|
| 2° point source | 19.0 | 19.0 | 19.0 | 16.8 | 12.8 | 13.3 | 16.9 | 20.0 |
| 5° point source | 19.4 | 19.4 | 17.4 | 5.0 | −3.4 | 1.8 | 16.9 | 20.0 |
| 7° point source | 19.4 | 19.5 | 5.4 | −47.5 | −26.3 | −16.4 | 16.5 | 20.0 |
| 2° small spread | 19.3 | 20.0 | 19.9 | 17.1 | 11.6 | 13.1 | 17.1 | 21.2 |
| 5° small spread | 19.7 | 20.4 | 18.2 | 9.6 | 2.6 | 5.3 | 17.1 | 21.2 |
| 7° small spread | 19.8 | 20.5 | 13.0 | 3.1 | −17.0 | −9.6 | 16.5 | 21.2 |
| 5° large spread | 19.3 | 19.4 | 17.4 | 4.9 | −6.7 | 7.9 | 17.5 | 21.4 |
| 7° large spread | 19.4 | 19.5 | 4.1 | −35.0 | −23.8 | −7.5 | 17.3 | 21.4 |

* Mismatch sweep (Figure 4.2): guided R-MBER has no failed run up to 12°, 9°, 11° and 9° for M = 8, 16, 32 and 64 and stays within 1.8 dB of the optimum; INCM-REC fails in every run from 8°, 5°, 7° and 7°, WC-MVDR/EigDec from 0°, 2°, 1.5° and 1°.
* Interference level (Figure 4.7): guided R-MBER holds 19.4 to 19.8 dB at 5° and 7° up to 30 dB INR with K = 150 and K = 1000.
* Imperfect training reference (Figure 4.10): with one wrong pilot label TA-MMSE from 30 pilots falls from 19.1 to 13.2 dB, under pilot contamination at equal power to −0.1 dB; guided R-MBER, which uses no labels, holds 19.4 dB.
* Local scattering (Figure 4.11): with two rays of amplitude 0.6 at +4° and −3° the BER step adds 1.89 dB over the guide alone and lowers the counted BER at −10 dB from 9.4 × 10⁻³ to 2.1 × 10⁻³; for a point source it changes the SINR by 0.01 dB or less.
* Limits: guided R-MBER fails in every run when a third interferer lies 3° from the source (Figure 4.9) and when the source drifts by 6° within a block (Figure 4.10(d)).

Timings depend on the machine. Figure 4.1 is drawn from the run recorded for the chapter (Intel Xeon at 2.3 GHz, two cores, Python 3.13, NumPy 2.5, median of four runs, best of five repetitions): the guided form takes 134, 221, 506 and 1599 µs per snapshot at 16, 32, 64 and 128 elements, 0.6 to 1.2 times one WC-MVDR/EigDec solve. All values are in `figures/results_summary.json`.

Figures show each method with the same colour, marker and line style throughout, inside a full frame, with the legend in a framed box placed clear of the curves.

## Requirements

Python 3.10 or later, NumPy, SciPy and Matplotlib (`requirements.txt` pins the versions used to produce `figures/`). Jupyter is needed only for the notebook.

## How to cite

Please cite the book chapter and the papers in `CITATION.cff`.

## License

MIT, see `LICENSE`.
