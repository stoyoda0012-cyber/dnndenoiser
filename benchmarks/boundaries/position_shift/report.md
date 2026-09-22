# P2-A — the position-shift boundary, and what augmentation does to it

<!-- GENERATED FILE. Produced by render_report.py from
     results/position_shift_boundary.json. Do not edit by hand:
     anything written here is dropped the next time it is regenerated. -->

Record generated 2026-09-22T13:56:01.738654+00:00 · record version 1 · 162.7 min wall clock

Registered design: `docs/preregistration/P2A-position-shift-boundary.md` (registered `06fa8c0`, Revision 1 `ef25766`). Predictions were fixed before implementation.

## What was manipulated

A rigid energy shift delta (eV) applied to every peak at once, with the energy grid held fixed.

**Not in scope:** non-rigid shifts. A chemical shift moves components relative to one another; this manipulation is the charging/calibration case only.

| Arm | N | Training shift | Per-peak jitter |
|---|---|---|---|
| A narrow (N=2304) | 2304 | none | ±0.3 eV |
| B augmented ±1.5 eV (N=2304) | 2304 | U(±1.5) eV, rigid, per sample | ±0.3 eV |
| C narrow (N=461) | 461 | none | ±0.3 eV |
| D narrow (N=144) | 144 | none | ±0.3 eV |

arms A and B differ in TWO things, not one: augmentation, and training density on the position axis. At equal N, arm B has 1/5 of arm A's per-peak marginal density and 1/16 in the three-peak joint configuration space. Arms C and D are arm A at those two densities, so that R5 can separate the cost of augmentation from the cost of spreading a fixed N over a wider range.

## The boundary

|Δ|\* is the first crossing of zero mean SNR gain, per seed, median over 20 seeds. A censored seed never crossed inside the tested range and enters the order statistics at its bound; it is never dropped.

| Arm | level | direction | median \|Δ\|\* (eV) | censored | sustained (eV) | re-crossing seeds |
|---|---|---|---|---|---|---|
| A narrow (N=2304) | 100.0 | positive | not defined | — | — | — |
| A narrow (N=2304) | 100.0 | negative | not defined | — | — | — |
| B augmented ±1.5 eV (N=2304) | 100.0 | positive | not defined | — | — | — |
| B augmented ±1.5 eV (N=2304) | 100.0 | negative | not defined | — | — | — |
| C narrow (N=461) | 100.0 | positive | not defined | — | — | — |
| C narrow (N=461) | 100.0 | negative | not defined | — | — | — |
| D narrow (N=144) | 100.0 | positive | not defined | — | — | — |
| D narrow (N=144) | 100.0 | negative | not defined | — | — | — |
| A narrow (N=2304) | 1000.0 | positive | 0.47 | 0/20 | 0.47 | 0 |
| A narrow (N=2304) | 1000.0 | negative | 0.47 | 0/20 | 0.47 | 0 |
| B augmented ±1.5 eV (N=2304) | 1000.0 | positive | 1.83 | 0/20 | 1.83 | 0 |
| B augmented ±1.5 eV (N=2304) | 1000.0 | negative | 1.83 | 0/20 | 1.83 | 0 |
| C narrow (N=461) | 1000.0 | positive | 0.37 | 0/20 | 0.37 | 0 |
| C narrow (N=461) | 1000.0 | negative | 0.35 | 0/20 | 0.35 | 0 |
| D narrow (N=144) | 1000.0 | positive | 0.31 | 0/20 | 0.31 | 0 |
| D narrow (N=144) | 1000.0 | negative | 0.30 | 0/20 | 0.30 | 0 |
| A narrow (N=2304) | 10000.0 | positive | > 4.0 | 20/20 | -- | 0 |
| A narrow (N=2304) | 10000.0 | negative | > 4.0 | 20/20 | -- | 0 |
| B augmented ±1.5 eV (N=2304) | 10000.0 | positive | > 4.0 | 20/20 | -- | 0 |
| B augmented ±1.5 eV (N=2304) | 10000.0 | negative | > 4.0 | 20/20 | -- | 0 |
| C narrow (N=461) | 10000.0 | positive | > 4.0 | 20/20 | -- | 0 |
| C narrow (N=461) | 10000.0 | negative | > 4.0 | 20/20 | -- | 0 |
| D narrow (N=144) | 10000.0 | positive | > 4.0 | 20/20 | -- | 0 |
| D narrow (N=144) | 10000.0 | negative | > 4.0 | 20/20 | -- | 0 |

Where the mean gain at Δ = 0 is already negative there is no boundary to find, and the row says `not defined` rather than reporting Δ = 0.

## SNR gain against shift, level 1000.0 (primary)

Mean ± across-seed SD in dB. The replicate is the seed.

| Δ (eV) | A narrow (N=2304) | B augmented ±1.5 eV (N=2304) | C narrow (N=461) | D narrow (N=144) |
|---|---|---|---|---|
| -4.00 | -17.28 ± 0.09 | -16.43 ± 0.20 | -17.11 ± 0.18 | -17.26 ± 0.27 |
| -3.50 | -17.19 ± 0.09 | -15.79 ± 0.18 | -17.00 ± 0.18 | -17.13 ± 0.25 |
| -3.00 | -17.08 ± 0.10 | -14.71 ± 0.13 | -16.92 ± 0.17 | -16.98 ± 0.23 |
| -2.50 | -16.65 ± 0.11 | -12.22 ± 0.21 | -16.57 ± 0.20 | -16.59 ± 0.26 |
| -2.00 | -15.85 ± 0.12 | -4.58 ± 0.51 | -15.83 ± 0.22 | -15.88 ± 0.30 |
| -1.75 | -15.33 ± 0.13 | +2.05 ± 0.63 | -15.35 ± 0.22 | -15.39 ± 0.30 |
| -1.50 | -14.61 ± 0.14 | +8.16 ± 0.48 | -14.69 ± 0.20 | -14.75 ± 0.28 |
| -1.25 | -13.40 ± 0.16 | +9.87 ± 0.35 | -13.66 ± 0.16 | -13.77 ± 0.26 |
| -1.00 | -11.26 ± 0.22 | +10.07 ± 0.27 | -11.90 ± 0.15 | -12.17 ± 0.25 |
| -0.75 | -7.39 ± 0.31 | +10.11 ± 0.24 | -8.81 ± 0.22 | -9.40 ± 0.29 |
| -0.50 | -0.92 ± 0.42 | +10.13 ± 0.25 | -3.85 ± 0.29 | -4.81 ± 0.38 |
| -0.25 | +6.92 ± 0.42 | +10.15 ± 0.22 | +2.60 ± 0.29 | +1.25 ± 0.40 |
| +0.00 | +11.41 ± 0.24 | +10.10 ± 0.22 | +7.07 ± 0.27 | +5.64 ± 0.23 |
| +0.25 | +6.83 ± 0.38 | +10.07 ± 0.20 | +2.84 ± 0.34 | +1.34 ± 0.33 |
| +0.50 | -0.94 ± 0.35 | +9.98 ± 0.18 | -3.31 ± 0.30 | -4.41 ± 0.38 |
| +0.75 | -6.97 ± 0.24 | +9.91 ± 0.18 | -8.42 ± 0.25 | -9.14 ± 0.30 |
| +1.00 | -10.72 ± 0.17 | +9.90 ± 0.18 | -11.70 ± 0.19 | -12.07 ± 0.18 |
| +1.25 | -12.86 ± 0.14 | +9.64 ± 0.24 | -13.56 ± 0.16 | -13.75 ± 0.14 |
| +1.50 | -14.00 ± 0.13 | +7.83 ± 0.35 | -14.59 ± 0.17 | -14.73 ± 0.15 |
| +1.75 | -14.61 ± 0.15 | +2.03 ± 0.36 | -15.16 ± 0.19 | -15.31 ± 0.17 |
| +2.00 | -15.05 ± 0.17 | -3.95 ± 0.33 | -15.54 ± 0.21 | -15.71 ± 0.18 |
| +2.50 | -16.02 ± 0.14 | -11.35 ± 0.33 | -16.22 ± 0.17 | -16.33 ± 0.19 |
| +3.00 | -16.70 ± 0.08 | -13.79 ± 0.32 | -16.68 ± 0.15 | -16.72 ± 0.22 |
| +3.50 | -16.96 ± 0.07 | -14.93 ± 0.30 | -16.88 ± 0.15 | -16.92 ± 0.22 |
| +4.00 | -17.10 ± 0.09 | -15.86 ± 0.22 | -17.04 ± 0.18 | -17.10 ± 0.22 |

## Registered predictions

All evaluated at level 1000.0 only. Sign rules are one-sided because every prediction names its direction in advance; the uniform threshold is 15/20 (one-sided p = 0.0207), and the positive control's is 19/20.

| | Verdict | Statement |
|---|---|---|
| R1 | **PASS** | arm A gains at delta = 0 (positive control) |
| R2 | **PASS** | arm A's gain is negative at delta = +4.0 and -4.0 |
| R3 | **PASS** | arm A's |delta|* is <= 1.5 eV in at least one direction |
| R4 | **PASS** | at |delta| = 1.5, arm B gains more than arm A (paired within seed) |
| R5a | **PASS** | at delta = 0, arm A > arm C > arm D (training density alone costs) |
| R5b | **FAIL** | at delta = 0, arm B <= arm D (augmentation costs beyond the density penalty) |
| R6 | **PASS** | arm B's |delta|* exceeds arm A's where arm A has one |
| R7 | **PASS** | for arm A at |delta| >= 1.0, the BIAS-CORRECTED mean argmax displacement disp(delta) - disp(0) has the opposite sign to delta and grows with |delta| |

### Evidence

**R1** — arm A gains at delta = 0 (positive control)  
&nbsp;&nbsp;`.` mean +11.412 · 20/20 seeds · one-sided p = 0.0000 · d_z = +47.46  

**R2** — arm A's gain is negative at delta = +4.0 and -4.0  
&nbsp;&nbsp;`.per_direction[+4.00]` mean -17.097 · 20/20 seeds · one-sided p = 0.0000 · d_z = -188.20 · Holm p = 0.0000  
&nbsp;&nbsp;`.per_direction[-4.00]` mean -17.275 · 20/20 seeds · one-sided p = 0.0000 · d_z = -187.98 · Holm p = 0.0000  

**R3** — arm A's |delta|* is <= 1.5 eV in at least one direction  
&nbsp;&nbsp;`positive` median |Δ|* = 0.47 eV vs threshold 1.5 eV, 0 censored  
&nbsp;&nbsp;`negative` median |Δ|* = 0.47 eV vs threshold 1.5 eV, 0 censored  

**R4** — at |delta| = 1.5, arm B gains more than arm A (paired within seed)  
&nbsp;&nbsp;`.per_direction[+1.50]` mean +21.834 · 20/20 seeds · one-sided p = 0.0000 · d_z = +62.68 · Holm p = 0.0000  
&nbsp;&nbsp;`.per_direction[-1.50]` mean +22.765 · 20/20 seeds · one-sided p = 0.0000 · d_z = +45.73 · Holm p = 0.0000  

**R5a** — at delta = 0, arm A > arm C > arm D (training density alone costs)  
&nbsp;&nbsp;`.orderings[A_over_C]` mean +4.338 · 20/20 seeds · one-sided p = 0.0000 · d_z = +12.38  
&nbsp;&nbsp;`.orderings[C_over_D]` mean +1.437 · 20/20 seeds · one-sided p = 0.0000 · d_z = +5.93  

**R5b** — at delta = 0, arm B <= arm D (augmentation costs beyond the density penalty)  
&nbsp;&nbsp;`.` mean -4.466 · 0/20 seeds · one-sided p = 1.0000 · d_z = -13.68  

**R6** — arm B's |delta|* exceeds arm A's where arm A has one  
&nbsp;&nbsp;`positive` verdict **moved** — arm A 0.47 eV, arm B 1.83 eV, 20/20 seeds larger  
&nbsp;&nbsp;`negative` verdict **moved** — arm A 0.47 eV, arm B 1.83 eV, 20/20 seeds larger  

**R7** — for arm A at |delta| >= 1.0, the BIAS-CORRECTED mean argmax displacement disp(delta) - disp(0) has the opposite sign to delta and grows with |delta|  
&nbsp;&nbsp;`.sign_points[+1.00]` mean -0.668 · 20/20 seeds · one-sided p = 0.0000 · d_z = -19.61 · Holm p = 0.0000  
&nbsp;&nbsp;`.sign_points[-1.00]` mean +0.633 · 20/20 seeds · one-sided p = 0.0000 · d_z = +14.63 · Holm p = 0.0000  
&nbsp;&nbsp;`.sign_points[+4.00]` mean -4.013 · 20/20 seeds · one-sided p = 0.0000 · d_z = -70.22 · Holm p = 0.0000  
&nbsp;&nbsp;`.sign_points[-4.00]` mean +3.895 · 20/20 seeds · one-sided p = 0.0000 · d_z = +57.64 · Holm p = 0.0000  
&nbsp;&nbsp;`positive` |displacement| at |Δ| = 1, 1.5, 2, 3, 4: 0.668, 1.303, 1.936, 2.988, 4.013 eV; largest decrease 0.0000 eV against a tolerance of one grid step (0.069412 eV)  
&nbsp;&nbsp;`negative` |displacement| at |Δ| = 1, 1.5, 2, 3, 4: 0.633, 1.165, 1.713, 2.872, 3.895 eV; largest decrease 0.0000 eV against a tolerance of one grid step (0.069412 eV)  

## M3 — argmax displacement, and what it is not

a learning-free Gaussian smoother produces a SHIFT-INDEPENDENT positive offset of about +0.06/+0.21/+0.48 eV at sigma = 0.5/1.0/2.0 eV, present on clean spectra too -- three grid steps at sigma ~ 1 eV, which would help at delta < 0 and hurt at delta > 0

Comparator (ii), the learning-free Gaussian smoother, measured on this run's own test spectra at seed 0:

| Δ (eV) | sigma_0.5eV | sigma_1.0eV | sigma_2.0eV | noisy input |
|---|---|---|---|---|
| -4.00 | +0.056 | +0.211 | +0.472 | +0.005 |
| -3.50 | +0.057 | +0.209 | +0.474 | +0.008 |
| -3.00 | +0.053 | +0.210 | +0.474 | -0.001 |
| -2.50 | +0.057 | +0.210 | +0.476 | +0.007 |
| -2.00 | +0.057 | +0.211 | +0.476 | +0.004 |
| -1.75 | +0.057 | +0.209 | +0.475 | +0.007 |
| -1.50 | +0.056 | +0.209 | +0.476 | +0.003 |
| -1.25 | +0.056 | +0.209 | +0.476 | -0.004 |
| -1.00 | +0.054 | +0.208 | +0.475 | +0.003 |
| -0.75 | +0.055 | +0.208 | +0.475 | +0.005 |
| -0.50 | +0.056 | +0.209 | +0.477 | -0.004 |
| -0.25 | +0.053 | +0.209 | +0.475 | -0.001 |
| +0.00 | +0.055 | +0.211 | +0.477 | +0.008 |
| +0.25 | +0.056 | +0.211 | +0.476 | +0.004 |
| +0.50 | +0.056 | +0.210 | +0.476 | -0.004 |
| +0.75 | +0.058 | +0.212 | +0.477 | -0.005 |
| +1.00 | +0.058 | +0.211 | +0.476 | +0.010 |
| +1.25 | +0.060 | +0.213 | +0.477 | +0.001 |
| +1.50 | +0.059 | +0.211 | +0.476 | -0.001 |
| +1.75 | +0.057 | +0.208 | +0.476 | +0.012 |
| +2.00 | +0.055 | +0.210 | +0.477 | +0.005 |
| +2.50 | +0.056 | +0.208 | +0.477 | +0.009 |
| +3.00 | +0.053 | +0.207 | +0.475 | +0.007 |
| +3.50 | +0.056 | +0.208 | +0.475 | +0.004 |
| +4.00 | +0.057 | +0.211 | +0.478 | +0.005 |

A flat column is the point: none of truncation, background asymmetry, the three-peak envelope's own asymmetry or plain oversmoothing produces a displacement that depends on the shift, so a shift-dependent displacement in a trained arm is not attributable to them.

## Self-checks

Each of these has a stated failure condition and voids the record. Diagnostics are listed separately below because they cannot fail.

| Check | Result |
|---|---|
| 1 parameter count | 658177 (expected from the reference record) |
| 2, 3 grid invariance and test-sweep rigidity | 25 shifts, grid bit-identical, baselines are literals |
| 5 truncation | absolute retention at Δ=0 98.426 %; every shift within 1% of it |
| 6 input-SNR invariance | worst span 100.0 → 0.138 dB, 1000.0 → 0.160 dB, 10000.0 → 0.202 dB |
| 7 translation equivariance | worst residual 0.00456 of peak height, tolerance 0.01 |
| 9 argmax well-posedness and reference identity | worst fraction 1.0000, compared against 284.8 + delta, not 284.8 |
| 11 noise-model identity | field-by-field against the literals; the boolean is False at level 10000: the three levels do not share a noise code path, which is why the noise is not paired across shifts there |
| 4 training-pool rigidity | 20 seeds × 4 arms, 5 % sampled |
| 8, 8b pairing integrity and replay faithfulness | 12327 tuples per seed; 32 spectra rebuilt from replayed draws and required to be bit-identical |
| 10 augmentation | narrow arms all-zero; augmented within its range |
| 12 leakage | 0 identical spectra; near-duplicate ratio 1.002 |

**Diagnostic — run-to-run determinism.** Arm A trained twice on identical inputs at one seed: 0.000000 dB apart. Recorded, never asserted.

## Consistency anchor

Arm A at Δ = 0 is the reference benchmark's own primary condition for this architecture, re-drawn with this script's seeding. the draws differ, the seed count differs and the per-sample generation path differs; a flag is a prompt to explain the difference in this record.

| level | reference (dB) | here (dB) | difference | flag threshold | flagged |
|---|---|---|---|---|---|
| 100.0 | -5.205 | -5.381 | -0.176 | 2.479 | no |
| 1000.0 | +11.577 | +11.412 | -0.165 | 0.962 | no |
| 10000.0 | +20.624 | +20.634 | +0.010 | 0.727 | no |

## What this record does not support

- anything about MEASURED spectra: every spectrum here is synthetic, scored against a reference that exists only because it is synthetic, and the training noise and the test noise come from the same function, so the model's noise model is exactly correct by construction -- a condition measured data never satisfies
- a general position-shift threshold for XPS denoising: |delta|* is a property of THIS peak set, THIS jitter width, THIS architecture, THIS training-set size and THIS noise model, and one point was measured in each of those spaces
- anything about other training-set sizes, for any claim including R5: arms C and D bound the density penalty at one architecture and one recipe
- anything about other augmentation widths: one width (+/-1.5 eV) was tested, so R6 is a statement about that width and not about augmentation in general
- a full-spectrum translate: linear_background is level + slope*(x - x[0]), a ramp pinned to the WINDOW, so it does not travel with the peaks and a peak moving along it sits on a different background level. This manipulation is therefore 'peaks shift under a stationary background'. Self-check 7 measures the departure from a pure translate and bounds it; it does not remove it
- separation of degradation from window-edge effects beyond |delta| = 1.5 eV. Inside that range arm B IS an edge-proximity control, because it saw those edge distances in training; beyond it no arm did
- anything about non-rigid shifts: chemical shifts move components relative to one another and this manipulation cannot speak to them
- anything about optimisation budget: all arms get 50 epochs and one schedule, so a deficit arm B shows may be under-training rather than augmentation cost
- anything about peak areas, widths or fitted positions surviving denoising: M3 is a grid argmax and is not a peak fit
- a recommendation that augmentation is the right mitigation: R5 and R6 are designed to show its price and its edge, and measuring a mitigation is not endorsing it
- a mechanism for M3: opposite-signed displacement is CONSISTENT WITH a learned position prior, and the observable underdetermines it

**Device dependence.** R1, R2, R4, R5 and R7 are sign and ordering claims. R3 and R6 compare a recorded MAGNITUDE -- |delta|*, in eV -- against a fixed threshold, and are therefore subject to the same device caveat as any magnitude here. Floating-point reduction order differs between CPU, MPS and CUDA backends; no cross-device comparison is recorded unless one is run, and none is asserted.

**Descriptive-only rule.** Only the cells named in R1-R7, at the primary noise level, carry an inferential claim. Every other cell -- all other shifts, all other levels, all other arms, and both derived series -- is descriptive, is reported without a p-value, and no statement of the form 'gain dips at delta = x' or 'the curve is asymmetric at x' may be made about a cell not named in a prediction. A feature seen there is a candidate for a NEW preregistration, not a finding of this one.

**The evaluated quantity is agreement with a known synthetic reference. A high SNR gain does not establish that structure in the output is real; the network can oversmooth, suppress weak features, and produce plausible structure that was not in the input.**

