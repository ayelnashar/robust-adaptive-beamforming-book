# Robust Adaptive Beamforming and Its Applications: Software

**Companion software for the book**
*Robust Adaptive Beamforming and Its Applications*
Ayman Elnashar, Wiley (forthcoming)

This repository contains the Python code that reproduces the simulation results of the book. Each chapter has its own folder with the algorithms, a notebook that regenerates every figure of the chapter, the generated figures, and tests.

## About the book

Adaptive beamforming can raise the output signal-to-interference-plus-noise ratio of an antenna array dramatically, but only when the quantities it relies on (the steering vector of the desired signal, the data covariance matrix, the directions of the interferers) are known accurately. Calibration errors, pointing errors, finite training data, multipath and array imperfections can make the optimal beamformer cancel the very signal it should protect. Robust adaptive beamforming designs beamformers that keep their performance under these errors.

The book develops robust adaptive array processing from first principles and carries it across wireless communications, radar, satellite and navigation systems, geophysics, acoustics, optics, radio astronomy and medical imaging, and then into neural and learning-based methods. Every method is backed by simulations, and this repository makes all of them reproducible.

## Organization

### Part I: Foundations and Core Methods

| Ch. | Title | Code |
|---|---|---|
| 1 | Robust Adaptive Beamforming: Foundations | [chapter-01](chapter-01) |
| 2 | A General Constrained-Optimization Method for Robust Beamforming | coming soon |
| 3 | Efficient Solving Under Multiple Constraints | coming soon |

### Part II: Wireless Communications

| Ch. | Title | Code |
|---|---|---|
| 4 | Robust Beamforming that Minimizes Bit-Error-Rate Directly | coming soon |
| 5 | Multi-Source Robust Beamforming with Subspace Tracking | coming soon |
| 6 | Distributed (Cell-Free) Massive MIMO | coming soon |
| 7 | Dual-Function Beamforming for Integrated Sensing and Communication | coming soon |
| 8 | Wideband Robust Beamforming for Sub-Terahertz 6G Links | coming soon |
| 9 | Geometry-Adaptive Dual-Function Beamforming for Movable-Antenna ISAC | coming soon |
| 10 | Satellite and Non-Terrestrial-Network Beamforming | coming soon |
| 11 | GNSS Anti-Jamming with a Controlled Reception Pattern Antenna | coming soon |

### Part III: Sensing and Cross-Domain Applications

| Ch. | Title | Code |
|---|---|---|
| 12 | Automotive Radar: Robust Beamforming and Target Tracking | coming soon |
| 13 | Robust Multi-Sensor Radar and IRST Fusion for Tracking Maneuvering Targets | coming soon |
| 14 | Dense Seismic Arrays and Distributed Acoustic Sensing | coming soon |
| 15 | Borehole Microseismic Monitoring under Velocity and Anisotropy Uncertainty | coming soon |
| 16 | Underwater Acoustic and Sonar Arrays | coming soon |
| 17 | Acoustic and Microphone Arrays in Reverberant Environments | coming soon |
| 18 | Robust Beamforming for Large-Scale Silicon-Photonic Optical Phased Arrays | coming soon |
| 19 | Radio Astronomy: RFI Mitigation under Megaconstellation Interference | coming soon |

### Part IV: Neural, Learning-Based, and Advanced Beamforming

| Ch. | Title | Code |
|---|---|---|
| 20 | When Does Machine Learning Help Robust Beamforming? | coming soon |
| 21 | Robust and Learned Beamforming for Medical Ultrasound: Limits under Sound-Speed Aberration | coming soon |
| 22 | Neural Receiver with Detect-and-Rollback | coming soon |
| 23 | Safeguarded Robust Adaptive Beamforming: Reconciling Covariance Reconstruction with Worst-Case Constraints | coming soon |

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
git clone https://github.com/ayelnashar/robust-adaptive-beamforming-book.git
cd robust-adaptive-beamforming-book/chapter-01
pip install -r requirements.txt
jupyter notebook notebooks/
```

Requirements: Python 3.10 or later. Each chapter lists its own packages in `requirements.txt`.

## Conventions

* All simulation results in the book are produced by the Python code in this repository.
* Monte Carlo runs use fixed seeds, so every figure can be regenerated exactly.
* Figure numbers in file names follow the book (for example `Fig1_17_WC_SINR.png` is Figure 1.17).

## How to cite

If you use this software, please cite the book and the chapter concerned. Each chapter folder contains a `CITATION.cff` file listing the journal papers the chapter builds on.

## License

MIT, see [LICENSE](LICENSE).

## Contact

Ayman Elnashar, ayman.elnashar@ku.ac.ae
