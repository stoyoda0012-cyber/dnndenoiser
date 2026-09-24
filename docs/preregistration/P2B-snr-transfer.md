# Preregistration — P2-B: training and inference at different signal-to-noise ratios

**Status: DRAFT — not registered, not implemented, not run.** Items marked
**PROPOSED** are decisions the owner has not yet made. This document becomes a
registration only when the owner approves it and it is **published (pushed) before the
first run**
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

- **Levels.** Five: λ = 4, 9, 20, 45, 100 — the 25-fold flux range SIA's three
  exposures span, in steps of about 2.24 in flux (about 1.5 in amplitude S/N), so the
  grid has 25 cells: 5 diagonal, 10 with training above inference, 10 with training
  below. λ = 100 is the peak count of the reference benchmark's and P2-A's primary
  level. Levels outside this range are not included: the equal-total-exposure design
  below cannot reach them without leaving SIA's frame counts.

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
- **Budget per level: equal total exposure, as in SIA.** The number of training
  frames is inversely proportional to flux, λ · N ≈ 50 000 at every level: 12 500,
  5 556, 2 500, 1 111 and 500 frames at λ = 4, 9, 20, 45, 100 (rounded to the nearest
  frame). The end points are SIA's frame counts.
- **What that confounds, stated now.** Because the frame count falls as flux rises,
  **a training level's S/N and its number of training frames cannot be separated in
  this design.** A difference between training levels is a difference in both; this
  record will not attribute it to either. That is SIA's design too, and the reason it
  is kept.

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
   flux — **PROPOSED:** 2304 spectra, P2-A's pool size, all at that one flux — with
   P2-A's recipe, so that its diagonal cell at λ = 100 can be set beside
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
- **Seeds: 20**, as in P2-A. Rough cost, scaling SIA's reported training time for
  one element on the same backend linearly with frames: about 25 minutes of
  moving-average training per seed over the five levels, so about 8 hours for 20
  seeds, plus noise2clean.
- **Saving and resuming.** Because the run is long, each seed's results are written
  to their own file as that seed finishes, with the commit, the working-tree state and
  the script's hash. A resumed run continues only from files written at the **same
  commit** from a clean tree, and refuses otherwise; the record states which seeds, if
  any, came from a resumed run.
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

## Registered predictions (PROPOSED — for the owner's decision)

Evaluated for the moving-average method; the same statistics are recorded for
noise2clean, descriptively. Every prediction is a **sign rule per cell, across the
20 seeds**, and a prediction holds only if every cell in its family meets it.

**How the threshold and Holm fit together.** The one-sided binomial *p* of *k* of 20
seeds in the stated direction, under a fair coin, is 0.0207 at 15, 0.0059 at 16,
0.0013 at 17, 0.00020 at 18 and 0.000020 at 19. Holm's first step for a family of
*m* tests at α = 0.05 is 0.05/*m*: 0.005 for 10 cells, 0.0167 for 3. So the per-cell
threshold is set at the smallest *k* whose *p* clears that first step — **17 of 20
for a family of 10, 16 of 20 for a family of 3** — and a family in which every cell
meets it passes Holm at every step. **Holm is therefore required, and is met through
the threshold**; the Holm-adjusted *p* of every cell is also reported. (The
alternative — a looser threshold with Holm reported only — is not proposed: it would
let a prediction "hold" that its own correction does not support.)

- **R1 — positive control, upper levels only.** At λ = 20, 45 and 100, the diagonal
  cell's M1 is positive in at least 19 of 20 seeds — P2-A's positive-control rule,
  stricter than the family-of-3 minimum of 16. The diagonal cells at λ = 4 and 9 are
  **descriptive**: whether the method helps at all there is part of what is being
  measured, not a precondition.
- **R2 — training above inference costs little.** In each of the 10 cells with
  training flux above inference flux, M2 is above the margin in at least 17 of 20
  seeds. **The margin, two options for the owner:**
  - *(a) fixed:* M2 > −1 dB.
  - *(b) relative:* M2 > −0.1 × the diagonal gain at the same inference flux, within
    seed — the off-diagonal model keeps at least 90 % of what training at the
    inference flux achieves. Defined only where that diagonal gain is positive; at
    λ = 4 and 9, where R1 does not require it to be, a seed with a non-positive
    diagonal gain counts against the prediction rather than being dropped.
- **R3 — training below inference costs.** In each of the 10 cells with training flux
  below inference flux, M2 is negative in at least 17 of 20 seeds.
- **R4 — the asymmetry.** For each of the 10 pairs of levels (a below b), the penalty
  of training at a and denoising at b is larger than that of training at b and
  denoising at a — −M2(a→b) > −M2(b→a), paired within seed — in at least 17 of 20
  seeds.

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
- **not a self-check, descriptive only:** the noise2clean diagonal cell at λ = 100 is
  reported beside P2-A's Δ = 0 operating point, with no tolerance and no condition. The
  two differ in noise model (exact Poisson here, the Gaussian approximation there) and
  in training data (one level here, three there), and the record says so.

## What this record will not support

- anything about measured spectra, depth profiles, the L1 inversion or any instrument;
- confirmation, refutation or reproduction of the SIA paper's empirical directional
  rule;
- a general S/N-transfer rule for XPS denoising: one peak set, one architecture, one
  target window, one budget design;
- separating a training level's S/N from its number of training frames;
- attributing the difference between the two methods to information, data or recipe.

## Before registration

1. The owner decides every **PROPOSED** item.
2. The apparatus is implemented under `benchmarks/boundaries/snr_transfer/`, reusing
   what P2-A's apparatus does through a shared module; P2-A's own script is not
   edited, because its hash is in its record.
3. An independent audit of this document and the apparatus, with a fixed checklist.
4. The owner approves, and this document is **published before the first full run**.
