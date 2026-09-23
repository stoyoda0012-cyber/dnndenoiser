# P2-A — the position-shift boundary, and what augmentation does to it

<!-- GENERATED FILE. Produced by render_report.py from
     results/position_shift_boundary.json. Do not edit by hand:
     anything written here is dropped the next time it is regenerated. -->

Record generated 2026-09-23T15:00:32.890684+00:00 · record version 1 · 81.1 min wall clock

Registered design: `docs/preregistration/P2A-position-shift-boundary.md` (first registered `06fa8c0`, last revised before this run at `f4cb1ca`; code run from `f4cb1ca`, working tree clean). Predictions were fixed before implementation.

**What this report's guard verifies, and what it does not.** Before rendering, `render_report.py` recomputes every aggregate from the raw runs independently, and re-derives the `boundaries` and `predictions` trees from the raw runs with the measurement script's own functions — so an edited or stale record is refused, but an error *inside* those functions would be reproduced, not caught. **Not verified here at all:** the self-check figures, the environment, the consistency anchor and the M3 smoother comparator, which are not derivable from the raw runs and are printed as stored. Each section below that prints one of those says so.

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

eV is the unit that was measured. The two columns after it are conversions of the same number for this record's one setup, computed here from the record: bins on its 0.069427 eV grid, and multiples of the dominant peak's *nominal* FWHM, 1.2 eV. That FWHM is the generator's setting for the 284.8 eV peak — each synthetic spectrum varies it by up to ±10 %, and the three-peak envelope is wider — so the ratio is a conversion, not a measured width. **Whether any of these units carries over to a different grid or line width was not tested.**

| Arm | level | direction | median \|Δ\|\* (eV) | ≈ bins | ≈ × nominal FWHM | censored | sustained (eV) | re-crossing seeds |
|---|---|---|---|---|---|---|---|---|
| A narrow (N=2304) | 100.0 | positive | not defined | — | — | — | — | — |
| A narrow (N=2304) | 100.0 | negative | not defined | — | — | — | — | — |
| B augmented ±1.5 eV (N=2304) | 100.0 | positive | not defined | — | — | — | — | — |
| B augmented ±1.5 eV (N=2304) | 100.0 | negative | not defined | — | — | — | — | — |
| C narrow (N=461) | 100.0 | positive | not defined | — | — | — | — | — |
| C narrow (N=461) | 100.0 | negative | not defined | — | — | — | — | — |
| D narrow (N=144) | 100.0 | positive | not defined | — | — | — | — | — |
| D narrow (N=144) | 100.0 | negative | not defined | — | — | — | — | — |
| A narrow (N=2304) | 1000.0 | positive | 0.47 | 6.8 | 0.39 | 0/20 | 0.47 | 0 |
| A narrow (N=2304) | 1000.0 | negative | 0.47 | 6.8 | 0.40 | 0/20 | 0.47 | 0 |
| B augmented ±1.5 eV (N=2304) | 1000.0 | positive | 1.8 | 26 | 1.5 | 0/20 | 1.8 | 0 |
| B augmented ±1.5 eV (N=2304) | 1000.0 | negative | 1.8 | 26 | 1.5 | 0/20 | 1.8 | 0 |
| C narrow (N=461) | 1000.0 | positive | 0.37 | 5.3 | 0.31 | 0/20 | 0.37 | 0 |
| C narrow (N=461) | 1000.0 | negative | 0.35 | 5.0 | 0.29 | 0/20 | 0.35 | 0 |
| D narrow (N=144) | 1000.0 | positive | 0.31 | 4.5 | 0.26 | 0/20 | 0.31 | 0 |
| D narrow (N=144) | 1000.0 | negative | 0.30 | 4.3 | 0.25 | 0/20 | 0.30 | 0 |
| A narrow (N=2304) | 10000.0 | positive | > 4.0 | — | — | 20/20 | -- | 0 |
| A narrow (N=2304) | 10000.0 | negative | > 4.0 | — | — | 20/20 | -- | 0 |
| B augmented ±1.5 eV (N=2304) | 10000.0 | positive | > 4.0 | — | — | 20/20 | -- | 0 |
| B augmented ±1.5 eV (N=2304) | 10000.0 | negative | > 4.0 | — | — | 20/20 | -- | 0 |
| C narrow (N=461) | 10000.0 | positive | > 4.0 | — | — | 20/20 | -- | 0 |
| C narrow (N=461) | 10000.0 | negative | > 4.0 | — | — | 20/20 | -- | 0 |
| D narrow (N=144) | 10000.0 | positive | > 4.0 | — | — | 20/20 | -- | 0 |
| D narrow (N=144) | 10000.0 | negative | > 4.0 | — | — | 20/20 | -- | 0 |

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
&nbsp;&nbsp;`.` mean +11.412 · 20/20 seeds · one-sided p = 9.54e-07 · d_z = +47.46  

**R2** — arm A's gain is negative at delta = +4.0 and -4.0  
&nbsp;&nbsp;`.per_direction[+4.00]` mean -17.097 · 20/20 seeds · one-sided p = 9.54e-07 · d_z = -188.20 · Holm p = 1.34e-44  
&nbsp;&nbsp;`.per_direction[-4.00]` mean -17.275 · 20/20 seeds · one-sided p = 9.54e-07 · d_z = -187.98 · Holm p = 1.34e-44  

**R3** — arm A's |delta|* is <= 1.5 eV in at least one direction  
&nbsp;&nbsp;`positive` median |Δ|* = 0.47 eV vs threshold 1.5 eV, 0 censored  
&nbsp;&nbsp;`negative` median |Δ|* = 0.47 eV vs threshold 1.5 eV, 0 censored  

**R4** — at |delta| = 1.5, arm B gains more than arm A (paired within seed)  
&nbsp;&nbsp;`.per_direction[+1.50]` mean +21.834 · 20/20 seeds · one-sided p = 9.54e-07 · d_z = +62.68 · Holm p = 1.59e-35  
&nbsp;&nbsp;`.per_direction[-1.50]` mean +22.765 · 20/20 seeds · one-sided p = 9.54e-07 · d_z = +45.73 · Holm p = 3.16e-33  

**R5a** — at delta = 0, arm A > arm C > arm D (training density alone costs)  
&nbsp;&nbsp;`.orderings[A_over_C]` mean +4.338 · 20/20 seeds · one-sided p = 9.54e-07 · d_z = +12.38  
&nbsp;&nbsp;`.orderings[C_over_D]` mean +1.437 · 20/20 seeds · one-sided p = 9.54e-07 · d_z = +5.93  

**R5b** — at delta = 0, arm B <= arm D (augmentation costs beyond the density penalty)  
&nbsp;&nbsp;`.` mean -4.466 · 0/20 seeds · one-sided p = 1 · d_z = -13.68  

**R6** — arm B's |delta|* exceeds arm A's where arm A has one  
&nbsp;&nbsp;`positive` verdict **moved** — arm A 0.47 eV, arm B 1.83 eV, 20/20 seeds larger  
&nbsp;&nbsp;`negative` verdict **moved** — arm A 0.47 eV, arm B 1.83 eV, 20/20 seeds larger  

**R7** — for arm A at |delta| >= 1.0, the BIAS-CORRECTED mean argmax displacement disp(delta) - disp(0) has the opposite sign to delta and grows with |delta|  
&nbsp;&nbsp;`.sign_points[+1.00]` mean -0.668 · 20/20 seeds · one-sided p = 9.54e-07 · d_z = -19.61 · Holm p = 6.03e-26  
&nbsp;&nbsp;`.sign_points[-1.00]` mean +0.633 · 20/20 seeds · one-sided p = 9.54e-07 · d_z = +14.63 · Holm p = 7.69e-24  
&nbsp;&nbsp;`.sign_points[+4.00]` mean -4.013 · 20/20 seeds · one-sided p = 9.54e-07 · d_z = -70.22 · Holm p = 3.67e-36  
&nbsp;&nbsp;`.sign_points[-4.00]` mean +3.895 · 20/20 seeds · one-sided p = 9.54e-07 · d_z = +57.64 · Holm p = 1.17e-34  
&nbsp;&nbsp;`positive` |displacement| at |Δ| = 1, 1.5, 2, 3, 4: 0.668, 1.303, 1.936, 2.988, 4.013 eV; largest decrease 0.0000 eV against a tolerance of one grid step (0.069412 eV)  
&nbsp;&nbsp;`negative` |displacement| at |Δ| = 1, 1.5, 2, 3, 4: 0.633, 1.165, 1.713, 2.872, 3.895 eV; largest decrease 0.0000 eV against a tolerance of one grid step (0.069412 eV)  

## M3 — argmax displacement, and what it is not

a learning-free Gaussian smoother produces a SHIFT-INDEPENDENT positive offset of about +0.06/+0.21/+0.48 eV at sigma = 0.5/1.0/2.0 eV, present on clean spectra too -- three grid steps at sigma ~ 1 eV, which would help at delta < 0 and hurt at delta > 0

### Comparator (iii) — arm B on the identical test arrays

Bias-corrected displacement `disp(Δ) − disp(0)` in eV, level 1000.0, mean over seeds. Registered in advance: a *structural* window effect would give arms A and B the same profile; a learned position prior would leave arm B near zero inside its training range. These values come from `aggregates` and are covered by the guard.

| Δ (eV) | arm A | arm B |
|---|---|---|
| -4.00 | +3.8955 | +2.7148 |
| -3.50 | +3.4060 | +1.9541 |
| -3.00 | +2.8718 | +1.3584 |
| -2.50 | +2.2769 | +0.7768 |
| -2.00 | +1.7133 | +0.2759 |
| -1.75 | +1.4411 | +0.1048 |
| -1.50 | +1.1654 | +0.0207 |
| -1.25 | +0.8938 | +0.0046 |
| -1.00 | +0.6329 | +0.0012 |
| -0.75 | +0.3872 | +0.0011 |
| -0.50 | +0.1804 | +0.0007 |
| -0.25 | +0.0487 | +0.0000 |
| +0.00 | +0.0000 | +0.0000 |
| +0.25 | -0.0527 | +0.0012 |
| +0.50 | -0.1906 | -0.0010 |
| +0.75 | -0.4061 | -0.0003 |
| +1.00 | -0.6682 | -0.0010 |
| +1.25 | -0.9708 | -0.0023 |
| +1.50 | -1.3034 | -0.0227 |
| +1.75 | -1.6338 | -0.1109 |
| +2.00 | -1.9355 | -0.2787 |
| +2.50 | -2.4727 | -0.8514 |
| +3.00 | -2.9880 | -1.6578 |
| +3.50 | -3.5213 | -2.2426 |
| +4.00 | -4.0129 | -2.8412 |

### Comparator (ii) — a learning-free Gaussian smoother, every level

Mean displacement in eV on this run's own test spectra, **seed 0 only**. A diagnostic: **not verified by this report's guard**. Read the span across shifts, not the level: a flat column means the smoother's offset does not depend on the shift at that noise level.

**Level 100.0** — span across all shifts: sigma_0.5eV 0.0062, sigma_1.0eV 0.0039, sigma_2.0eV 0.0084, noisy_input 0.0065 eV

| Δ (eV) | sigma_0.5eV | sigma_1.0eV | sigma_2.0eV | noisy input |
|---|---|---|---|---|
| -4.00 | +0.057 | +0.211 | +0.470 | +0.000 |
| -3.50 | +0.057 | +0.213 | +0.474 | +0.002 |
| -3.00 | +0.061 | +0.213 | +0.477 | +0.001 |
| -2.50 | +0.058 | +0.213 | +0.476 | +0.001 |
| -2.00 | +0.058 | +0.210 | +0.476 | -0.002 |
| -1.75 | +0.060 | +0.214 | +0.478 | +0.002 |
| -1.50 | +0.056 | +0.211 | +0.473 | -0.002 |
| -1.25 | +0.058 | +0.212 | +0.476 | +0.000 |
| -1.00 | +0.057 | +0.213 | +0.478 | -0.001 |
| -0.75 | +0.057 | +0.210 | +0.475 | -0.001 |
| -0.50 | +0.060 | +0.213 | +0.478 | +0.001 |
| -0.25 | +0.057 | +0.210 | +0.473 | +0.002 |
| +0.00 | +0.058 | +0.212 | +0.476 | +0.002 |
| +0.25 | +0.057 | +0.212 | +0.477 | +0.001 |
| +0.50 | +0.058 | +0.210 | +0.475 | -0.000 |
| +0.75 | +0.062 | +0.213 | +0.478 | +0.002 |
| +1.00 | +0.057 | +0.210 | +0.474 | -0.004 |
| +1.25 | +0.059 | +0.212 | +0.475 | +0.000 |
| +1.50 | +0.058 | +0.212 | +0.477 | -0.000 |
| +1.75 | +0.058 | +0.211 | +0.475 | -0.002 |
| +2.00 | +0.062 | +0.214 | +0.479 | +0.002 |
| +2.50 | +0.058 | +0.211 | +0.475 | -0.004 |
| +3.00 | +0.058 | +0.212 | +0.475 | -0.000 |
| +3.50 | +0.056 | +0.210 | +0.472 | -0.004 |
| +4.00 | +0.057 | +0.213 | +0.476 | -0.002 |

**Level 1000.0** — span across all shifts: sigma_0.5eV 0.0066, sigma_1.0eV 0.0058, sigma_2.0eV 0.0053, noisy_input 0.0169 eV

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

**Level 10000.0** — span across all shifts: sigma_0.5eV 0.2410, sigma_1.0eV 0.4677, sigma_2.0eV 0.8106, noisy_input 0.3727 eV

| Δ (eV) | sigma_0.5eV | sigma_1.0eV | sigma_2.0eV | noisy input |
|---|---|---|---|---|
| -4.00 | +0.322 | +0.697 | +1.224 | -0.044 |
| -3.50 | +0.199 | +0.541 | +1.039 | -0.073 |
| -3.00 | +0.297 | +0.563 | +1.099 | -0.122 |
| -2.50 | +0.338 | +0.693 | +1.022 | -0.082 |
| -2.00 | +0.145 | +0.308 | +0.750 | -0.114 |
| -1.75 | +0.236 | +0.572 | +1.000 | -0.089 |
| -1.50 | +0.262 | +0.671 | +1.057 | -0.084 |
| -1.25 | +0.171 | +0.360 | +0.772 | -0.116 |
| -1.00 | +0.178 | +0.494 | +0.909 | -0.150 |
| -0.75 | +0.135 | +0.480 | +0.773 | -0.076 |
| -0.50 | +0.216 | +0.595 | +0.999 | -0.116 |
| -0.25 | +0.221 | +0.360 | +0.777 | -0.192 |
| +0.00 | +0.252 | +0.397 | +0.808 | -0.268 |
| +0.25 | +0.157 | +0.361 | +0.672 | -0.262 |
| +0.50 | +0.151 | +0.405 | +0.731 | -0.273 |
| +0.75 | +0.135 | +0.296 | +0.546 | -0.293 |
| +1.00 | +0.132 | +0.332 | +0.693 | -0.349 |
| +1.25 | +0.151 | +0.318 | +0.649 | -0.294 |
| +1.50 | +0.102 | +0.257 | +0.595 | -0.372 |
| +1.75 | +0.097 | +0.229 | +0.691 | -0.417 |
| +2.00 | +0.135 | +0.286 | +0.513 | -0.393 |
| +2.50 | +0.214 | +0.304 | +0.574 | -0.316 |
| +3.00 | +0.258 | +0.279 | +0.605 | -0.364 |
| +3.50 | +0.199 | +0.279 | +0.413 | -0.332 |
| +4.00 | +0.107 | +0.261 | +0.471 | -0.393 |

## Self-checks

Thirteen gates, each with a stated failure condition, each voiding the record. Diagnostics are listed separately below because they cannot fail. The figures in this section are stored, not recomputed: they are not derivable from the raw runs, and `render_report.py`'s guard does not cover them.

| Check | Result |
|---|---|
| 1 parameter count | 658177 (expected read from the reference record) |
| 2 grid invariance | bit-identical at all 25 shifts |
| 3 test-sweep peak-set construction | every centre equals its literal plus Δ. A run-time unit test of the peak-set constructor and of the registry staying unmutated — it does not inspect generated spectra; check 8b does that |
| 4 training-pool rigidity | 5% of every pool per seed — A narrow: 116, B augmented ±1.5 eV: 116, C narrow: 24, D narrow: 8 — rebuilt from the literal peaks, the recorded shift and the replayed draws, and required bit-identical to the pool |
| 5 truncation | absolute retention at Δ=0 98.426 %; every shift within 1% of it, each peak within 2% |
| 6 input-SNR invariance | worst span 100.0 → 0.138 dB, 1000.0 → 0.160 dB, 10000.0 → 0.202 dB |
| 7 translation equivariance | worst residual 0.00456 of peak height against a tolerance of 0.01, compared with `np.roll` at integer grid offsets |
| 8 pairing integrity | 2909 samples per seed, keyed on (level_index, sample_index) |
| 8b test-family rigidity | 300 test spectra per seed rebuilt from the shift-independent family seed and required bit-identical |
| 9 argmax well-posedness and reference identity | worst fraction 1.0000 against 284.8 + delta, not 284.8; identity asserted against the array the metric received |
| 10 augmentation | narrow arms all-zero; augmented SD 0.8777 against a uniform 0.8660, 10/10 deciles occupied |
| 11 noise-model identity | field by field against the literals; the boolean is False at level 10000: the three levels do not share a noise code path, which is why the noise is not paired across shifts there |
| 12 leakage | 0 identical spectra across 4 arms; near-duplicate ratio 1.002 |

**Diagnostic — run-to-run determinism.** Arm A trained twice on identical inputs at one seed: 0.000000 dB apart. Recorded, never asserted.

## Consistency anchor

Arm A at Δ = 0 is the reference benchmark's own primary condition for this architecture, re-drawn with this script's seeding. the draws differ, the seed count differs and the per-sample generation path differs; a flag is a prompt to explain the difference in this record. **Printed as stored; not verified by this report's guard.**

| level | reference (dB) | here (dB) | difference | flag threshold | flagged |
|---|---|---|---|---|---|
| 100.0 | -5.205 | -5.381 | -0.176 | 2.479 | no |
| 1000.0 | +11.577 | +11.412 | -0.165 | 0.962 | no |
| 10000.0 | +20.624 | +20.634 | +0.010 | 0.727 | no |

## What this record does not support

- anything about MEASURED spectra: every spectrum here is synthetic, scored against a reference that exists only because it is synthetic, and the training noise and the test noise come from the same function, so the model's noise model is exactly correct by construction -- a condition measured data never satisfies
- a general position-shift threshold for XPS denoising: |delta|* is a property of THIS peak set, THIS jitter width, THIS architecture, THIS training-set size and THIS noise model, and one point was measured in each of those spaces
- anything about other training-set sizes: each arm was trained at one pool size (2304, 461 or 144 spectra), with one architecture and one recipe
- an inferential claim at any noise level other than the primary one: every prediction was registered at level 1000.0, and the other two levels are descriptive
- a relation between |delta|* and any single property of the training distribution -- its position range, its size or its position density: one architecture and four training distributions were measured
- that augmentation is safe inside its training range in general: one augmentation width, one architecture and one noise level were measured
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

**Where the result-dependent cautions are.** This list was fixed before the results and states only what the design cannot support. Cautions that depend on the result -- among them how to read R5b's failure, what the other two noise levels showed, and what arms C and D do and do not show -- are in the Record section of docs/preregistration/P2A-position-shift-boundary.md, and in report.md, each with its source.

**The evaluated quantity is agreement with a known synthetic reference. A high SNR gain does not establish that structure in the output is real; the network can oversmooth, suppress weak features, and produce plausible structure that was not in the input.**

