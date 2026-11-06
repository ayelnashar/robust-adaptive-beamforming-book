# Robust Adaptive Beamforming Across Domains: Software

**Companion software for the book**
*Robust Adaptive Beamforming Across Domains: Wireless, Radar, Sonar, and Biomedical Arrays, with Neural and Learning-Based Methods*
Ayman Elnashar, Wiley–IEEE Press (forthcoming)

This repository contains the Python code that reproduces the simulation results of the book. Each chapter has its own folder with the algorithms, a notebook that regenerates every figure of the chapter, the generated figures, and tests.

## About the book

Adaptive beamforming can raise the output signal-to-interference-plus-noise ratio of an antenna array dramatically, but only when the quantities it relies on (the steering vector of the desired signal, the data covariance matrix, the directions of the interferers) are known accurately. Calibration errors, pointing errors, finite training data, multipath and array imperfections can make the optimal beamformer cancel the very signal it should protect. Robust adaptive beamforming designs beamformers that keep their performance under these errors.

The book develops robust adaptive array processing from first principles and carries it across wireless communications, radar, satellite and navigation systems, geophysics, acoustics, optics, radio astronomy and medical imaging, and then into neural and learning-based methods. Every method is backed by simulations, and this repository makes all of them reproducible.

## Organization

### Part I: Foundations and Methods

| Ch. | Title | Code |
|---|---|---|
| 1 | Robust Adaptive Beamforming | [chapter-01](chapter-01) |
| 2 | A General Method for Worst-Case Constrained Optimization | coming soon |
| 3 | An Efficient Solver for Multiple Worst-Case Constraints | coming soon |

### Part II: Wireless Communications

| Ch. | Title | Code |
|---|---|---|
| 4 | Robust Minimum-Bit-Error-Rate Beamforming | coming soon |
| 5 | Multi-Cluster Robust Beamforming with Subspace-Tracked Active Set | coming soon |
| 6 | Robust Adaptive Beamforming for Cell-Free Massive MIMO | coming soon |
| 8 | Wideband Sub-THz Beamforming | coming soon |
| 10 | LEO Satellite Robust Beamforming | coming soon |

### Part III: Sensing and Cross-Domain Applications

| Ch. | Title | Code |
|---|---|---|
| 7 | Robust Dual-Function Beamforming for 6G Integrated Sensing and Communication | coming soon |
| 9 | Movable-Antenna Integrated Sensing and Communication | coming soon |
| 11 | GNSS Anti-Jamming | coming soon |
| 12 | Automotive MIMO Radar | coming soon |
| 13 | Multi-Sensor Radar/IRST Fusion and Tracking | coming soon |
| 14 | Seismic and DAS Arrays under Velocity-Model Uncertainty | coming soon |
| 15 | Borehole Microseismic Monitoring | coming soon |
| 16 | Underwater Acoustic Arrays | coming soon |
| 17 | Microphone Arrays in Reverberant Environments | coming soon |
| 18 | Robust Beamforming for Optical Phased Arrays | coming soon |
| 19 | Radio Astronomy RFI Mitigation | coming soon |

### Part IV: Neural, Learning-Based, and Advanced Beamforming

| Ch. | Title | Code |
|---|---|---|
| 20 | When Does Machine Learning Help Robust Beamforming? | coming soon |
| 21 | Limits of Robust and Learned Beamforming in Medical Ultrasound | coming soon |
| 22 | Neural Receiver Detect-and-Rollback | coming soon |
| 23 | Safeguarded Robust Adaptive Beamforming | coming soon |

## Repository layout

```text
chapter-01/
  README.md        what the chapter code does, figure map, how to run it
  <package>/       algorithms and scenario code
  notebooks/       executed notebook, one section per figure
  scripts/         command-line figure generation
  figures/         PNG and PDF of the chapter figures, results summary
  tests/           sanity tests
chapter-02/
  ...
```

Every chapter folder is self-contained: open its README for installation and run instructions.

## Getting started

```bash
git clone https://github.com/<your-username>/<this-repository>.git
cd <this-repository>/chapter-01
pip install -r requirements.txt
jupyter notebook notebooks/
```

Requirements: Python 3.10 or later. Each chapter lists its own packages in `requirements.txt`.

## Conventions

* All results in the book are produced by the code in this repository; the original MATLAB simulations were ported to Python.
* Monte Carlo runs use fixed seeds, so every figure can be regenerated exactly.
* Figure numbers in file names follow the book (for example `Fig1_17_WC_SINR.png` is Figure 1.17).
* Where the port corrected an error in the original code, the correction is documented in the chapter README.

## How to cite

If you use this software, please cite the book and the chapter concerned. Each chapter folder contains a `CITATION.cff` file with the original journal papers.

## License

MIT, see [LICENSE](LICENSE).

## Contact

Ayman Elnashar, ayman.elnashar@ku.ac.ae
