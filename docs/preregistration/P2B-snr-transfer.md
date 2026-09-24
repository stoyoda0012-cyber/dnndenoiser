# Preregistration — P2-B: training and inference at different signal-to-noise ratios

**Status: DRAFT — not registered, not implemented, not run.** Items marked **OPEN**
are decisions the owner has not yet made. This document becomes a registration only
when the owner approves it and it is **published (pushed) before the first run**
(`AGENTS.md` §8.1). Nothing below may later be revised to match a result; a change
after registration is made visibly, with the reason and the date, in a Revision log.

## Conflict of interest

The owner of this repository, who approves this registration and decides what of its
record may be quoted, is the first author of the SIA paper discussed below
([10.1002/sia.70123](https://doi.org/10.1002/sia.70123)). A result that appears to
agree with that paper serves the author's interest; so does one framed to appear
independent of it. The safeguards are the ones this repository already uses: the
design, predictions and decision rules are fixed and published before any run; every
outcome is recorded whichever way it falls; the claims are independently reviewed
before any of them is cleared for quotation.

## What this record is, and is not, with respect to SIA

The SIA paper reports that cross-exposure deployment of a self-supervised denoiser
agreed, across the exposures it studied, with an **empirical directional rule** —
train signal-to-noise ratio (S/N) ≥ inference S/N. This document uses that phrase as
the paper does and does not treat the rule as a law.

**P2-B neither supports nor refutes the SIA paper's empirical directional rule, and
it is not a reproduction of that paper.** It asks a related question under different
conditions, and it says so before any result exists. The differences are:

| | SIA | P2-B |
|---|---|---|
| Data | measured AR-HAXPES frames of one multilayer sample | synthetic spectra from this package's generator |
| Where performance is judged | depth profiles, after an L1-regularised inversion, against a pseudo-ground truth | single-frame spectra, against the synthetic clean spectrum |
| Noise | the instrument's detector noise | exact Poisson noise, drawn by the same function for training and test |
| Signal | five core levels, resampled per element | one synthetic core-level envelope |
| Reference | pseudo-ground truth from the frame average | the noise-free spectrum, known exactly because it is synthetic |

A result here in either direction is recorded as what it is — a measurement under
these conditions — and is not read back onto the paper's data, its inversion stage or
its instrument.

## What is manipulated

Two things, crossed in full: the **per-frame flux at training** and the **per-frame
flux at inference**. Each model is trained at a single flux; every trained model is
evaluated at every flux. The diagonal cells (train = inference) and the off-diagonal
cells in both directions (train above inference, and train below) are all measured.

Flux is set by the generator's Poisson level: the expected count at the spectrum's
maximum, λ. Signal-to-noise in amplitude scales as √λ, so **a 25-fold range in flux
is a 5-fold range in amplitude S/N** (about 14 dB). This document states flux and S/N
separately wherever a ratio is quoted.

- **Levels (OPEN).** Proposed: λ = 4, 20, 100 — a 25-fold flux range, matching the
  range SIA's three exposures span. λ = 100 is the peak count of the reference
  benchmark's and P2-A's primary level, so the top level connects to those records.
  Optional, for seeing past the edges of that range: add λ = 0.8 and λ = 500 (a
  625-fold range over five levels, 25 cells instead of 9).

## The design

### Held fixed

- **Clean spectra.** `C1s_adventitious` (three peaks), 256 points, the generator's
  pseudo-Voigt profile and linear background, as in P2-A.
- **Noise.** Exact Poisson counts at every level — `use_gaussian_approx=False`, so no
  level uses the Gaussian approximation the reference benchmark and P2-A use. The
  training and test noise come from the same function, so the noise model is exactly
  right by construction; the record will say so.
- **Architecture.** ResNet-FCNN at `num_features=256, num_hidden_units=100,
  encoder_output_dim=64`.
- **Budget per level (OPEN).** Proposed: SIA's equal-total-exposure design, so the
  number of frames scales inversely with flux — 12 500 / 2 500 / 500 frames at
  λ = 4 / 20 / 100. The alternative, one fifth of each, cuts the run time roughly
  five-fold and departs from SIA's frame counts.

### Two training methods, on the same test data

1. **Self-supervised moving average (primary).** The method SIA used, as implemented
   in `dnndenoiser.training.selfsupervised` and verified against the archived
   reference in P1: `W = 1` — each frame's target is its temporally nearest other
   frame — with SIA's recipe, which is the library's (Adam, lr 1e-3, weight decay
   1e-9, `StepLR(step_size=25, gamma=0.5)`, Huber δ = 1, gradient-norm clip 4,
   50 epochs, batch 32). A "sample" is one clean spectrum; its frames are independent
   Poisson realisations of it at the training flux, and the model is trained on that
   sample's frames alone, as SIA trained per sample.
2. **noise2clean (baseline).** Trained against clean references on a pool of
   independently drawn spectra from the same generator at the same single training
   flux, with P2-A's recipe, so that its diagonal cell at λ = 100 can be set beside
   P2-A's operating point. The two methods therefore differ in information (clean
   targets versus none), in training data (a pool of spectra versus one sample's
   frames) and in recipe. **The comparison between them is descriptive**; it cannot
   attribute a difference to any one of those.

**Pairing.** For each seed, both methods and every training level are evaluated on
the **same test arrays**: fresh single frames of the seed's sample at each inference
flux, independent of every training frame. Differences between cells are computed
within seed.

### Replicates and split

- **The replicate is the seed.** A seed draws a new clean sample (its per-peak
  position, width and intensity), its training frames, its test frames, the
  noise2clean pool, and the model initialisation and batch order.
- **Seeds (OPEN).** Proposed: 20, as in P2-A. Rough cost at the proposed budget, from
  SIA's reported training time for one element on the same backend: about 17 minutes
  of moving-average training per seed across the three levels, so about 6 hours for
  20 seeds, plus noise2clean. Ten seeds would halve it; fewer than ten is not
  proposed.
- **Split.** Training frames, test frames and the noise2clean pool come from disjoint
  random streams; the unit that prevents leakage is the independently drawn frame (for
  the moving-average method) and the independently generated spectrum (for
  noise2clean). A self-check will verify that no test frame is byte-identical to a
  training frame.

### Regime

This record deliberately breaks the condition `AGENTS.md` §6 asks every claim to
state: **training and inference do not operate at the same signal-to-noise regime**
off the diagonal. That mismatch is the variable. Every cell states its training flux
and its inference flux.

## Metrics

- **M1 — SNR gain in dB (primary)**, exactly as in P2-A: output SNR minus input SNR,
  each `10·log10(mean(ref²) / mean((est − ref)²))` against the clean spectrum, per test
  frame, averaged over the test frames of a cell. The per-frame spread is not a
  replicate spread; dispersion is across seeds.
- **M2 — transfer penalty (derived).** For an off-diagonal cell, its M1 minus the M1
  of the diagonal cell at the same inference flux, computed within seed: what training
  at the other flux cost, or gained, relative to training at the flux being denoised.

## Registered predictions (PROPOSED — OPEN until the owner approves them)

Evaluated for the moving-average method; the same statistics are recorded for
noise2clean, descriptively. A sign rule is "the stated sign in at least 15 of 20
seeds" (one-sided *p* ≈ 0.021), Holm-adjusted within each prediction's family, as in
P2-A.

- **R1 — positive control.** Every diagonal cell has positive M1.
- **R2 — training above inference costs little.** For every cell with training flux
  above inference flux, M2 is greater than −1 dB. (**OPEN**: the −1 dB margin.)
- **R3 — training below inference costs.** For every cell with training flux below
  inference flux, M2 is negative.
- **R4 — the asymmetry.** For each pair of levels, the penalty of training below is
  larger in magnitude than the penalty of training above, paired within seed.

R2 and R3 together are this document's analogue, on synthetic spectra, of the SIA
paper's empirical directional rule. Whichever way they fall, both are recorded; a
failure is not repaired by redefining the cells, the margin or the levels after the
run.

## Self-checks that void the record

To be completed at implementation and reviewed before registration. At least:
- the realised count at each level matches its λ;
- every level's noise is exact Poisson (no Gaussian approximation anywhere);
- no test frame is byte-identical to a training frame, and the test arrays are
  identical across methods and training levels within a seed;
- the moving-average targets are the library's `moving_average_targets(W=1)`;
- the diagonal cell at λ = 100 for noise2clean lies within a stated tolerance of P2-A's
  Δ = 0 operating point, or the difference is explained by the noise model (exact
  Poisson here, the Gaussian approximation there) — a consistency anchor, not a
  prediction (**OPEN**: whether to register it).

## What this record will not support

- anything about measured spectra, depth profiles, the L1 inversion or any instrument;
- confirmation, refutation or reproduction of the SIA paper's empirical directional
  rule;
- a general S/N-transfer rule for XPS denoising: one peak set, one architecture, one
  target window, one budget design;
- attributing the difference between the two methods to information, data or recipe.

## Before registration

1. The owner decides every **OPEN** item.
2. The apparatus is implemented under `benchmarks/boundaries/snr_transfer/`, reusing
   what P2-A's apparatus does through a shared module; P2-A's own script is not
   edited, because its hash is in its record.
3. An independent audit of this document and the apparatus, with a fixed checklist.
4. The owner approves, and this document is **published before the first full run**.
