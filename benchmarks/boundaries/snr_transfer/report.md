# P2-B — training and inference at different signal-to-noise ratios

<!-- GENERATED FILE. Produced by render_report.py from results/snr_transfer.json.
     Do not edit by hand: anything written here is dropped the next time it is
     regenerated. -->

Record generated 2026-09-26T15:01:10.156963+00:00 · record version 1 · 176.3 min wall clock · device `mps`

Registered design: `docs/preregistration/P2B-snr-transfer.md` (first registered `8779c76`, last revised before this run at `7603007`; code run from `7603007`, working tree clean). Resumed seeds: none. What the record means is stated in that document's Record section, not here.

**What this report's guard verifies, and what it does not.** Before rendering, `render_report.py` recomputes from the per-seed gains, with arithmetic written in that file, every aggregate, every sign count and its verdict against the registered thresholds, and the noise2clean diagonal set beside P2-A; it checks the design block against the registered literals; and it re-derives the `predictions` and `noise2clean_descriptive` trees with the measurement script's own functions and compares them field by field, including every p and Holm-adjusted p. That second part catches an edited or stale record, not an error inside those functions. **Not verified here at all:** the per-seed gains themselves, the self-check figures, the input SNRs, the training times, the environment and the provenance, which are printed as stored.

## Design, as recorded

Synthetic `C1s_adventitious` spectra, 256 points; exact Poisson (use_gaussian_approx=False) at every level. Architecture ResNet-FCNN (num_features=256, num_hidden_units=100, encoder_output_dim=64). 20 seeds; the seed is the replicate, and every ± below is the SD across seeds. 512 test frames per inference level per seed, the same arrays for both methods and every training level.

| λ (expected count at the maximum) | 4 | 9 | 20 | 45 | 100 |
|---|---|---|---|---|---|
| moving-average training frames | 12500 | 5556 | 2500 | 1111 | 500 |
| noise2clean training spectra | 2304 | 2304 | 2304 | 2304 | 2304 |
| input SNR, dB (as stored) | 2.4 | 5.9 | 9.4 | 12.9 | 16.3 |

**Confound, as registered:** a training level's S/N, its number of training frames and its number of optimiser updates are not separable in this design.

## M1 — SNR gain (dB), moving average (primary)

| train λ \ inference λ | 4 | 9 | 20 | 45 | 100 |
|---|---|---|---|---|---|
| **4** | +9.9 ± 0.5 | +9.7 ± 0.8 | +6.5 ± 1.3 | +2.3 ± 1.7 | -1.5 ± 1.9 |
| **9** | +10.1 ± 0.5 | +10.9 ± 0.5 | +9.7 ± 0.5 | +6.9 ± 1.0 | +3.5 ± 1.4 |
| **20** | +12.0 ± 1.3 | +12.9 ± 0.9 | +12.6 ± 0.5 | +10.7 ± 0.8 | +7.6 ± 1.4 |
| **45** | +16.5 ± 0.5 | +16.7 ± 0.5 | +16.2 ± 0.5 | +14.9 ± 0.5 | +13.0 ± 0.7 |
| **100** | +16.7 ± 0.4 | +17.1 ± 0.4 | +17.0 ± 0.4 | +16.3 ± 0.4 | +14.9 ± 0.4 |

## M1 — SNR gain (dB), noise2clean (baseline, descriptive)

| train λ \ inference λ | 4 | 9 | 20 | 45 | 100 |
|---|---|---|---|---|---|
| **4** | +17.8 ± 0.9 | +16.2 ± 1.3 | +13.5 ± 1.5 | +10.3 ± 1.7 | +6.9 ± 1.9 |
| **9** | +17.2 ± 0.7 | +16.4 ± 1.0 | +15.0 ± 1.4 | +12.8 ± 1.7 | +10.0 ± 2.0 |
| **20** | +16.7 ± 0.3 | +16.3 ± 0.6 | +15.4 ± 0.9 | +14.1 ± 1.3 | +12.3 ± 1.8 |
| **45** | +15.6 ± 0.2 | +15.4 ± 0.3 | +14.9 ± 0.4 | +14.3 ± 0.6 | +13.3 ± 1.0 |
| **100** | +14.4 ± 0.2 | +14.3 ± 0.3 | +14.0 ± 0.2 | +13.6 ± 0.3 | +13.1 ± 0.5 |

M1 at different inference levels is measured from different input SNRs and is not a like-for-like comparison across columns; M2 compares within a column.

## M2 — transfer penalty (dB), moving average (primary)

M1 of the cell minus M1 of the diagonal cell in the same column, within seed. Above the diagonal: training below inference; below it: training above.

| train λ \ inference λ | 4 | 9 | 20 | 45 | 100 |
|---|---|---|---|---|---|
| **4** | — | -1.2 ± 0.6 | -6.2 ± 1.4 | -12.6 ± 1.9 | -16.4 ± 1.9 |
| **9** | +0.1 ± 0.4 | — | -2.9 ± 0.6 | -8.0 ± 1.1 | -11.4 ± 1.5 |
| **20** | +2.1 ± 1.1 | +2.0 ± 0.9 | — | -4.2 ± 0.8 | -7.3 ± 1.4 |
| **45** | +6.5 ± 0.7 | +5.8 ± 0.7 | +3.6 ± 0.6 | — | -1.9 ± 0.7 |
| **100** | +6.8 ± 0.6 | +6.2 ± 0.6 | +4.4 ± 0.6 | +1.4 ± 0.6 | — |

## M2 — transfer penalty (dB), noise2clean (baseline, descriptive)

M1 of the cell minus M1 of the diagonal cell in the same column, within seed. Above the diagonal: training below inference; below it: training above.

| train λ \ inference λ | 4 | 9 | 20 | 45 | 100 |
|---|---|---|---|---|---|
| **4** | — | -0.2 ± 0.8 | -1.9 ± 1.2 | -4.0 ± 1.7 | -6.2 ± 1.7 |
| **9** | -0.6 ± 0.6 | — | -0.4 ± 0.8 | -1.5 ± 1.3 | -3.1 ± 1.8 |
| **20** | -1.2 ± 0.8 | -0.2 ± 0.6 | — | -0.2 ± 0.7 | -0.9 ± 1.5 |
| **45** | -2.3 ± 0.9 | -1.1 ± 0.8 | -0.5 ± 0.5 | — | +0.1 ± 0.7 |
| **100** | -3.4 ± 1.0 | -2.2 ± 0.9 | -1.4 ± 0.8 | -0.6 ± 0.4 | — |

## Registered predictions — moving average

| | verdict | family size | seeds required per cell |
|---|---|---|---|
| R1 positive control at λ = 20, 45, 100 | **PASS** | 3 | 19 |
| R2 training above inference costs little (M2 > −1 dB) | **PASS** | 10 | 17 |
| R3 training below inference costs (M2 < 0) | **PASS** | 10 | 17 |
| R4 the asymmetry, paired within seed | **PASS** | 10 | 17 |

Levels removed by R1's failure rule: none.

### R1 — diagonal M1 at λ = 20, 45, 100

| cell | M1, mean ± SD (dB) | seeds meeting the rule | required | p | Holm p | met |
|---|---|---|---|---|---|---|
| 20.0 | +12.6 ± 0.5 | 20/20 | 19 | 9.54e-07 | 2.86e-06 | yes |
| 45.0 | +14.9 ± 0.5 | 20/20 | 19 | 9.54e-07 | 2.86e-06 | yes |
| 100.0 | +14.9 ± 0.4 | 20/20 | 19 | 9.54e-07 | 2.86e-06 | yes |

Diagonal M1 at λ = 4 and 9, descriptive (no prediction names them): λ = 4: +9.9 ± 0.5 dB, positive in 20/20 seeds; λ = 9: +10.9 ± 0.5 dB, positive in 20/20 seeds.

### R2 — training above inference

The rule is M2 > −1 dB. Each cell's own M1 and the diagonal M1 beside it, as registered: where the diagonal is weak, a cell can meet the rule by losing little relative to little.

| cell | M2, mean ± SD (dB) | seeds meeting the rule | required | p | Holm p | met |
|---|---|---|---|---|---|---|
| 9.0 → 4.0 | +0.1 ± 0.4 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 20.0 → 4.0 | +2.1 ± 1.1 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 20.0 → 9.0 | +2.0 ± 0.9 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 45.0 → 4.0 | +6.5 ± 0.7 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 45.0 → 9.0 | +5.8 ± 0.7 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 45.0 → 20.0 | +3.6 ± 0.6 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 100.0 → 4.0 | +6.8 ± 0.6 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 100.0 → 9.0 | +6.2 ± 0.6 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 100.0 → 20.0 | +4.4 ± 0.6 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 100.0 → 45.0 | +1.4 ± 0.6 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |

| cell | cell M1 (dB) | diagonal M1 at the same inference λ (dB) |
|---|---|---|
| 9.0 → 4.0 | +10.1 | +9.9 |
| 20.0 → 4.0 | +12.0 | +9.9 |
| 20.0 → 9.0 | +12.9 | +10.9 |
| 45.0 → 4.0 | +16.5 | +9.9 |
| 45.0 → 9.0 | +16.7 | +10.9 |
| 45.0 → 20.0 | +16.2 | +12.6 |
| 100.0 → 4.0 | +16.7 | +9.9 |
| 100.0 → 9.0 | +17.1 | +10.9 |
| 100.0 → 20.0 | +17.0 | +12.6 |
| 100.0 → 45.0 | +16.3 | +14.9 |

### R3 — training below inference

| cell | M2, mean ± SD (dB) | seeds meeting the rule | required | p | Holm p | met |
|---|---|---|---|---|---|---|
| 4.0 → 9.0 | -1.2 ± 0.6 | 19/20 | 17 | 2e-05 | 2e-05 | yes |
| 4.0 → 20.0 | -6.2 ± 1.4 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 4.0 → 45.0 | -12.6 ± 1.9 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 4.0 → 100.0 | -16.4 ± 1.9 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 9.0 → 20.0 | -2.9 ± 0.6 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 9.0 → 45.0 | -8.0 ± 1.1 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 9.0 → 100.0 | -11.4 ± 1.5 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 20.0 → 45.0 | -4.2 ± 0.8 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 20.0 → 100.0 | -7.3 ± 1.4 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 45.0 → 100.0 | -1.9 ± 0.7 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |

### R4 — the asymmetry

Per pair a < b: −M2(a → b) − (−M2(b → a)), paired within seed; the rule is that it is positive.

| cell | difference, mean ± SD (dB) | seeds meeting the rule | required | p | Holm p | met |
|---|---|---|---|---|---|---|
| 4.0 vs 9.0 | +1.4 ± 0.9 | 18/20 | 17 | 0.000201 | 0.000201 | yes |
| 4.0 vs 20.0 | +8.3 ± 2.0 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 4.0 vs 45.0 | +19.1 ± 2.4 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 4.0 vs 100.0 | +23.1 ± 2.2 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 9.0 vs 20.0 | +4.9 ± 1.4 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 9.0 vs 45.0 | +13.8 ± 1.5 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 9.0 vs 100.0 | +17.6 ± 1.8 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 20.0 vs 45.0 | +7.8 ± 1.3 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 20.0 vs 100.0 | +11.6 ± 1.8 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 45.0 vs 100.0 | +3.3 ± 1.3 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |

## noise2clean — the same statistics, descriptive only

None of these is a prediction. They are computed for every cell and pair, with no exclusion and no stopping rule.

### R1-type cells — rule met in 3 of 3

| cell | M1, mean ± SD (dB) | seeds meeting the rule | required | p | Holm p | met |
|---|---|---|---|---|---|---|
| 20.0 | +15.4 ± 0.9 | 20/20 | 19 | 9.54e-07 | 2.86e-06 | yes |
| 45.0 | +14.3 ± 0.6 | 20/20 | 19 | 9.54e-07 | 2.86e-06 | yes |
| 100.0 | +13.1 ± 0.5 | 20/20 | 19 | 9.54e-07 | 2.86e-06 | yes |

### R2-type cells — rule met in 3 of 10

| cell | M2, mean ± SD (dB) | seeds meeting the rule | required | p | Holm p | met |
|---|---|---|---|---|---|---|
| 9.0 → 4.0 | -0.6 ± 0.6 | 15/20 | 17 | 0.0207 | 0.145 | no |
| 20.0 → 4.0 | -1.2 ± 0.8 | 7/20 | 17 | 0.942 | 1 | no |
| 20.0 → 9.0 | -0.2 ± 0.6 | 19/20 | 17 | 2e-05 | 0.0002 | yes |
| 45.0 → 4.0 | -2.3 ± 0.9 | 2/20 | 17 | 1 | 1 | no |
| 45.0 → 9.0 | -1.1 ± 0.8 | 7/20 | 17 | 0.942 | 1 | no |
| 45.0 → 20.0 | -0.5 ± 0.5 | 17/20 | 17 | 0.00129 | 0.0116 | yes |
| 100.0 → 4.0 | -3.4 ± 1.0 | 0/20 | 17 | 1 | 1 | no |
| 100.0 → 9.0 | -2.2 ± 0.9 | 2/20 | 17 | 1 | 1 | no |
| 100.0 → 20.0 | -1.4 ± 0.8 | 5/20 | 17 | 0.994 | 1 | no |
| 100.0 → 45.0 | -0.6 ± 0.4 | 17/20 | 17 | 0.00129 | 0.0116 | yes |

### R3-type cells — rule met in 5 of 10

| cell | M2, mean ± SD (dB) | seeds meeting the rule | required | p | Holm p | met |
|---|---|---|---|---|---|---|
| 4.0 → 9.0 | -0.2 ± 0.8 | 11/20 | 17 | 0.412 | 1 | no |
| 4.0 → 20.0 | -1.9 ± 1.2 | 19/20 | 17 | 2e-05 | 0.00016 | yes |
| 4.0 → 45.0 | -4.0 ± 1.7 | 19/20 | 17 | 2e-05 | 0.00016 | yes |
| 4.0 → 100.0 | -6.2 ± 1.7 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 9.0 → 20.0 | -0.4 ± 0.8 | 13/20 | 17 | 0.132 | 0.526 | no |
| 9.0 → 45.0 | -1.5 ± 1.3 | 18/20 | 17 | 0.000201 | 0.00121 | yes |
| 9.0 → 100.0 | -3.1 ± 1.8 | 20/20 | 17 | 9.54e-07 | 9.54e-06 | yes |
| 20.0 → 45.0 | -0.2 ± 0.7 | 11/20 | 17 | 0.412 | 1 | no |
| 20.0 → 100.0 | -0.9 ± 1.5 | 15/20 | 17 | 0.0207 | 0.103 | no |
| 45.0 → 100.0 | +0.1 ± 0.7 | 8/20 | 17 | 0.868 | 1 | no |

### R4-type pairs — rule met in 1 of 10

| cell | difference, mean ± SD (dB) | seeds meeting the rule | required | p | Holm p | met |
|---|---|---|---|---|---|---|
| 4.0 vs 9.0 | -0.4 ± 1.3 | 6/20 | 17 | 0.979 | 1 | no |
| 4.0 vs 20.0 | +0.7 ± 1.7 | 14/20 | 17 | 0.0577 | 0.461 | no |
| 4.0 vs 45.0 | +1.7 ± 2.2 | 16/20 | 17 | 0.00591 | 0.0532 | no |
| 4.0 vs 100.0 | +2.8 ± 2.3 | 19/20 | 17 | 2e-05 | 0.0002 | yes |
| 9.0 vs 20.0 | +0.2 ± 1.3 | 12/20 | 17 | 0.252 | 1 | no |
| 9.0 vs 45.0 | +0.4 ± 2.1 | 11/20 | 17 | 0.412 | 1 | no |
| 9.0 vs 100.0 | +0.9 ± 2.7 | 12/20 | 17 | 0.252 | 1 | no |
| 20.0 vs 45.0 | -0.3 ± 1.3 | 8/20 | 17 | 0.868 | 1 | no |
| 20.0 vs 100.0 | -0.5 ± 2.2 | 9/20 | 17 | 0.748 | 1 | no |
| 45.0 vs 100.0 | -0.7 ± 1.1 | 4/20 | 17 | 0.999 | 1 | no |

## noise2clean at λ = 100 beside P2-A's operating point — descriptive, no tolerance

noise2clean diagonal M1 at λ = 100: +13.1 dB. P2-A, arm A, level 1000, Δ = 0, as copied into this record: +11.4 dB. They differ in:

- noise: exact Poisson here, the Gaussian approximation in P2-A
- training: one level here, three in P2-A
- test set: 512 noisy frames of ONE clean spectrum per seed here, independently generated spectra per seed in P2-A, so a seed mean averages over different things and the spread across seeds means different things
- this is not a re-measurement of P2-A under the same conditions

## Self-checks, as stored (not verified by this report's guard)

The run refuses to write a record if any self-check fails; this record was written, so all passed. Their stored figures, over all seeds:

- 1 exact Poisson: worst distance from a whole count, frames and pools, 5.7e-06
- 2 noise level: λ from the variance over declared λ, 0.988 to 1.015 (tolerance a factor 1.3)
- 3 equal exposure: frames per level λ = 4.0: 12500, λ = 9.0: 5556, λ = 20.0: 2500, λ = 45.0: 1111, λ = 100.0: 500 (seed 0; every seed passed)
- 4 targets: every moving-average target row checked against its neighbours (22167 rows per seed)
- 5 no leakage: test frames found in training, 0; seed samples found in a pool, 0
- 6 model inputs: cells checked per seed, 50
- 7 architecture: parameter counts, 658177

## Training time per model (s), as stored, mean over seeds

| method | λ = 4 | λ = 9 | λ = 20 | λ = 45 | λ = 100 |
|---|---|---|---|---|---|
| moving_average | 147 | 66 | 29 | 13 | 6 |
| noise2clean | 55 | 54 | 53 | 51 | 51 |

## Environment, as stored

Python 3.12.11, torch 2.9.1, numpy 2.3.3, scipy 1.16.3, macOS-27.0-arm64-arm-64bit; lockfile `uv.lock` sha256 `66e2b40b18bd…`; in a virtual environment: True.
