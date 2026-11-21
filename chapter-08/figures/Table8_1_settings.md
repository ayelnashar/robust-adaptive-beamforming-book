**Table 8.1 - Principal simulation settings.**

| Parameter | Value |
|---|---|
| Carrier / bandwidth | 300 GHz / 20 GHz (40 GHz for the grid ablation) |
| Array | ULA, half-wavelength spacing at the carrier |
| Element counts | 64 (core sweeps); 128 (Figures 8.8-8.10); 64-512 (scaling) |
| Processed subcarriers | 3-11, depending on the experiment |
| Snapshots per run K | 300 (220 in Figure 8.10, 600 in Figure 8.11) |
| Monte Carlo trials | 12-20 per point; 20 in Figures 8.8, 8.10 and 8.11 |
| SNR / INR per element | 5 dB / 10 dB |
| Forgetting factor | 0.97 |
| Warm-up before margin adaptation | 50 snapshots |
| Margin update period T | 30 snapshots |
| Retained eigenspace order | ceil(0.25 M) |
| Constraint grid | (0, -0.6, +0.6, -1.2) beamwidths; fixed (0, +/-1.5, -3) deg in Figures 8.3 and 8.9 and (0, +/-0.4, -0.8) deg in Figures 8.8 and 8.10; symmetric (0, +/-0.6, +/-1.2) beamwidths in Figure 8.11 |
| Initial margins before anchoring | (1.9, 1.5, 1.5, 1.1), then one-time anchors and (8.18)-(8.20) with ceiling 3; 0.3 in Figure 8.11 |
| Step relaxation | 0.1 |
| Single-ball radius (WC-single, robust Capon, SOCP) | 0.6 sqrt(M) (0.85 sqrt(M) in Figure 8.11; 3.8 in Figure 8.3) |
| Multipath case | 3 block-static paths, +/-0.2 deg spread, unit total path power |
| Random seed | 42, with an independent stream per experiment |
