# Preregistration — P2-B: training and inference at different signal-to-noise ratios

**Status: DRAFT — not registered, not implemented, not run.** The design and the
predictions were decided by the owner on 2026-09-25; the self-checks are completed at
implementation. This document becomes a registration only when the owner approves it,
after an independent audit, and it is **published (pushed) before the first run**
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
   flux — 2304 spectra, P2-A's pool size, all at that one flux — with P2-A's recipe, so that its diagonal cell at λ = 100 can be set beside
   P2-A's operating point. The two methods therefore differ in information (clean
   targets versus none), in training data (a pool of spectra versus one sample's
   frames) and in recipe. **The comparison between them is descriptive**; it cannot
   attribute a difference to any one of those.
   Because the noise2clean pool has the same 2304 spectra at every level, its row of
   the grid varies training S/N **without** varying the amount of training data. That
   makes it a descriptive handle on the moving-average method's confound between S/N
   and frame count — only a handle, since the two methods differ in the three ways
   just named.

**Pairing.** For each seed, both methods and every training level are evaluated on
the **same test arrays**: 512 fresh single frames of the seed's sample at each
inference flux, independent of every training frame. Differences between cells are
computed within seed.

**Normalisation, fixed at implementation.** The moving-average method normalises its
training stack by the stack's own global minimum and maximum, as the CLI's
moving-average path does. At inference, at any flux, the test frames are transformed
with the **training stack's** constants and the output is transformed back before it is
scored — so a model trained at one flux sees data from another flux through the
training flux's scale, as a model redeployed across exposures would. noise2clean uses
no normalisation beyond the generator's, as in P2-A. Every frame and every pool
spectrum is drawn by the generator's own noise function, `add_noise`.

### Replicates and split

- **The replicate is the seed.** A seed draws a new clean sample (its per-peak
  position, width and intensity), its training frames, its test frames, the
  noise2clean pool, and the model initialisation and batch order.
- **Seeds: 20**, as in P2-A. Measured at implementation on the development machine
  (MPS): about 10 minutes per seed for both methods over the five levels, so about
  3.5 hours for 20 seeds.
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

## Registered predictions

Evaluated for the moving-average method; the same statistics are recorded for
noise2clean, descriptively. Every prediction is a **sign rule per cell, across the
20 seeds**, and a prediction holds only if every cell in its family meets it.

**How the threshold and Holm fit together.** The one-sided binomial *p* of *k* of 20
seeds in the stated direction, under a fair coin, is 0.0207 at 15, 0.0059 at 16,
0.0013 at 17, 0.00020 at 18 and 0.000020 at 19. Holm's first step for a family of
*m* tests at α = 0.05 is 0.05/*m*. The per-cell threshold is the smallest *k* whose
*p* clears that first step, so a family in which every cell meets it passes Holm at
every step:

| Family size *m* | 1–2 | 3–8 | 9–10 |
|---|---|---|---|
| Per-cell threshold, of 20 seeds | 15 | 16 | 17 |

**Holm is required, and is met through the threshold**; the Holm-adjusted *p* of every
cell is also reported. A looser threshold with Holm reported only is not used: it
would let a prediction hold that its own correction does not support.

- **R1 — positive control, at λ = 20, 45 and 100.** The diagonal cell's M1 is positive
  in at least 19 of 20 seeds — P2-A's positive-control rule, stricter than the
  family-of-3 threshold of 16. **Why only these levels:** the SIA paper reports that
  training at its most photon-starved exposure, 4.8 s/frame, failed structurally — the
  denoiser's output lost its diversity, and all three cells trained there shared one
  wrong reconstruction whatever they were applied to. λ = 4 sits at that exposure's
  place in the 25-fold range (the lowest flux), and λ = 9 between it and SIA's 24 s;
  λ = 20, 45 and 100 cover SIA's 24 s and 120 s. The mapping is by relative flux only:
  the synthetic counts are not SIA's. So whether the method helps at λ = 4 and 9 is
  part of what is measured, and those two diagonal cells are **descriptive**.
- **R2 — training above inference costs little.** In each of the 10 cells with
  training flux above inference flux, M2 > −1 dB in at least 17 of 20 seeds. A fixed
  margin is a different fraction of the diagonal gain at each inference level; that
  is accepted, and the record says so. A relative margin was rejected because seven of
  the ten cells are at λ = 4 or 9, where the diagonal it would be relative to is
  descriptive and may not be positive. **Each R2 cell is reported with its own M1 and
  the diagonal cell's M1 beside it**: where the diagonal fails, an R2 cell can pass
  formally by losing little relative to nothing, and that is not to be read as
  transfer that worked.
- **R3 — training below inference costs.** In each of the 10 cells with training flux
  below inference flux, M2 is negative in at least 17 of 20 seeds.
- **R4 — the asymmetry.** For each of the 10 pairs of levels (a below b), the penalty
  of training at a and denoising at b is larger than that of training at b and
  denoising at a — −M2(a→b) > −M2(b→a), paired within seed — in at least 17 of 20
  seeds.

**If R1 fails at a level.** Decided now, before any result:
- R1's failure is reported first, per level.
- If R1 fails at one level L of λ = 20, 45, 100: every R2 and R3 cell whose inference
  level is L, and every R4 pair that includes L, becomes descriptive — its M2 is
  measured against a diagonal that did not work. R2, R3 and R4 are evaluated on the
  cells that remain, with the per-cell threshold read from the table above for the
  reduced family size. The record names the cells removed.
- If R1 fails at two or three of those levels, R2, R3 and R4 are not evaluated: the
  method did not work at enough of the grid for a transfer statement to mean anything.
  All their cells are reported descriptively.
- Cells at inference λ = 4 and 9 are not removed by this rule. Their diagonals were
  never required to work; R2's report of M1 beside M2 is what keeps them from being
  misread.

R2 and R3 together are this document's analogue, on synthetic spectra, of the SIA
paper's empirical directional rule. Whichever way they fall, both are recorded; a
failure is not repaired by redefining the cells, the margin or the levels after the
run.

## Self-checks that void the record

The run refuses to write a record if any fails. Each is shown to reject a named wrong
input in `tests/test_snr_transfer_gates.py`.

1. **Exact Poisson.** Every training and test frame, at every level, is a whole
   number of counts at that level's scale. Refuses Gaussian-approximated frames and
   frames drawn at another level.
2. **Realised flux.** The mean count at the spectrum's maximum over the training frames
   is λ within five standard errors. Refuses frames at another flux.
3. **Equal exposure.** λ · N equals the registered total at every level, to within one
   frame. Refuses equal frame counts.
4. **Targets.** At W = 1, every target row is bit-identical to one of its frame's
   adjacent frames in acquisition order — checked without the library's target
   function. Refuses W = 2 and a frame as its own target.
5. **No leakage.** No test frame is byte-identical to a training frame, and the seed's
   clean sample is not in any noise2clean pool. Refuses both.
6. **Same test arrays.** Every model of both methods, at every training level, was
   evaluated at each inference level on the array drawn for that level. Refuses a model
   evaluated on another level's array.
7. **Architecture.** The ResNet-FCNN's parameter count equals the reference benchmark's
   recorded count. Refuses another architecture.

The rule for a failed positive control, the threshold table and resuming only at the
same commit are tested in the same file.

**Not a self-check, descriptive only:** the noise2clean diagonal cell at λ = 100 is
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

1. ~~The owner decides the design and the predictions~~ — done 2026-09-25.
2. ~~The apparatus is implemented~~ — done: `benchmarks/boundaries/snr_transfer/`, reusing
   what P2-A's apparatus does through a shared module; P2-A's own script is not
   edited, because its hash is in its record.
3. An independent audit of this document and the apparatus, with a fixed checklist.
4. The owner approves, and this document is **published before the first full run**.
