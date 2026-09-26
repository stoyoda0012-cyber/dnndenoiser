# Preregistration — P2-B: training and inference at different signal-to-noise ratios

**Status: registered 2026-09-26, on the owner's approval, after two independent audits
(of `6459e5b` and `2046086`) and a check of the repair `d548985`; published before the
first full run (`AGENTS.md` §8.1). Implemented; no full run made yet.** The design and
the predictions were decided by the owner on 2026-09-25, before the apparatus was
written; what was tried while writing it is listed under "Before registration". Nothing
below may be revised to match a result; a change after registration is made visibly,
with the reason and the date, in the Revision log at the end.

## Conflict of interest

The owner of this repository, who approves this registration and decides what of its
record may be quoted, is the first author of the SIA paper discussed below
([10.1002/sia.70123](https://doi.org/10.1002/sia.70123)). A result that appears to
agree with that paper serves the author's interest; so does one framed to appear
independent of it. The safeguards are the ones this repository already uses: the
design, predictions and decision rules are fixed and published before the first full run; every
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
| Where performance is judged | mainly depth profiles, after an L1-regularised inversion, against a pseudo-ground truth; single-frame spectra are also compared (its Fig. 3) | single-frame spectra, against the synthetic clean spectrum |
| Noise | the instrument's detector noise, which the paper reports as purely Poisson with no detectable Gaussian read-noise floor | exact Poisson noise, drawn by the same function for training and test |
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
  this design** — nor its number of optimiser updates: at a fixed 50 epochs and batch
  32, the moving-average model takes about 19 550 updates at λ = 4 and 800 at λ = 100.
  A difference between training levels is a difference in all three; this record will
  not attribute it to any one. That is SIA's design too, and the reason it is kept.

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
- **Seeds: 20**, as in P2-A. Estimated, by extrapolating a 2-epoch timing run on the
  development machine (MPS): about 10 minutes per seed for both methods over the five levels, so about
  3.5 hours for 20 seeds.
- **Device.** The full run is made on the MPS backend of the development machine, as
  P2-A was; the script refuses a full run without an explicit device. Floating-point
  results are not expected to be bit-identical on another backend.
- **Saving and resuming.** Because the run is long, each seed's results are written
  to their own git-ignored file as that seed finishes, stamped with the commit, the
  working-tree state, the hashes of the script and the shared module, the device and
  library versions, and the settings. A resumed run reuses a file only if its stamp
  matches exactly, it holds the expected seed and all 50 cells; otherwise it refuses.
  So a record is always one environment, and every seed carries its own environment.
  The record states which seeds, if any, came from a resumed run. The output directory
  is created and proven writable before the first seed.
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
  training at its most photon-starved exposure, 4.8 s/frame, failed structurally — in
  its 4.8 s Self-DNN cell the pixel-level diversity of the output collapsed, and all
  three cells trained at 4.8 s showed the same kind of wrong layer arrangement whatever
  they were applied to (its §3.4 and Fig. 5). λ = 4 sits at that exposure's
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

1. **Exact Poisson.** (a) Every training and test frame, at every level, is a whole
   number of counts at that level's scale; (b) so is every noise2clean pool spectrum,
   each at its own maximum. Refuses Gaussian-approximated frames and pools, and frames
   drawn at a level that does not divide the declared one. This alone does not identify
   the level.
2. **Noise level.** λ estimated from the analytic Poisson variance — a bin of clean
   value c in a spectrum of maximum m has variance m·c/λ at the clean scale — lies
   within a factor 1.3 of the declared λ, for every training frame stack, test stack and
   noise2clean pool. The registered levels are about 2.2 apart, so any mix-up is refused;
   the test refuses all 20 ordered pairs of different registered levels. (A check of the
   mean count, used in the first draft, could not tell levels apart: the generator
   returns every level at the clean spectrum's amplitude.)
3. **Equal exposure.** λ · N equals the registered total at every level, to within one
   frame, with N the number of rows actually drawn. Refuses equal frame counts and a
   stack short of its planned count.
4. **Targets.** At W = 1, every target row is bit-identical to one of its frame's
   adjacent frames in acquisition order — checked without the library's target
   function. Refuses W = 2 and a frame as its own target.
5. **No leakage.** No test frame is byte-identical to a training frame, and the seed's
   clean sample is not in any noise2clean pool. Refuses both. It is an identity check,
   not a proof that the random streams are independent.
6. **Model inputs.** For all 50 cells, the array each network actually received —
   captured at the network's own call boundary by a forward pre-hook — is that
   inference level's test array under that model's normalisation, rebuilt apart from
   the evaluation code. Refuses an input shifted by five bins, a call that passes the
   network a shifted array while the prepared one stays right, another model's
   normalisation, a missing cell and an empty set.
7. **Architecture.** Every trained model's parameter count equals the reference
   benchmark's recorded count. Refuses another architecture; an architecture with the
   same count would pass.

The rule for a failed positive control, the threshold table, R1's Holm-adjusted *p*,
resuming, and preparing the output directory are tested in the same file.

**Statistics recorded beside the predictions.** R1 reports each cell's Holm-adjusted *p*
within its family of three. For noise2clean the same statistics as for the moving
average — the sign counts, *p* and Holm-adjusted *p* for R1 to R4, and the per-cell M1
and M2 — are recorded, **descriptively**: none of them is a prediction. They are
computed for every cell and pair, with no exclusion and no stopping rule, so R1's
failure cannot remove anything recorded for noise2clean.

**Every gain is a finite number.** A resumed seed file, and the whole record before it
is written, are refused if any gain is NaN, infinite or not a number: a NaN counts as
neither sign and would read as a non-supporting result it is not.

**Not a self-check, descriptive only:** the noise2clean diagonal cell at λ = 100 is
reported beside P2-A's Δ = 0 operating point, with no tolerance and no condition. The
two differ in noise model (exact Poisson here, the Gaussian approximation there), in
training data (one level here, three there) and in the test set (here 512 noisy frames
of one clean spectrum per seed, there independently generated spectra, so a seed mean
averages over different things and the spread across seeds means different things).
The record says so, and it is not a re-measurement of P2-A under the same conditions.

## What this record will not support

- anything about measured spectra, depth profiles, the L1 inversion or any instrument;
- confirmation, refutation or reproduction of the SIA paper's empirical directional
  rule;
- a general S/N-transfer rule for XPS denoising: one peak set, one architecture, one
  target window, one budget design;
- separating a training level's S/N from its number of training frames or optimiser
  updates;
- reading a lack of output diversity as SIA's structural failure. Within a seed there is
  one clean spectrum, so an output close to it and nearly the same for every frame
  scores well here; this record does not measure diversity;
- attributing the difference between the two methods to information, data or recipe.

## Before registration

1. ~~The owner decides the design and the predictions~~ — done 2026-09-25.
2. ~~The apparatus is implemented~~ — done: `benchmarks/boundaries/snr_transfer/`, reusing
   what P2-A's apparatus does through a shared module; P2-A's own script is not
   edited, because its hash is in its record.
3. An independent audit of this document and the apparatus, with a fixed checklist —
   the first was of commit `6459e5b`; its findings and their repair are in the commit
   that follows it (`2046086`). A second audit, of `2046086`, found one blocking defect
   — self-check 6 hashed the prepared input rather than what the network received —
   and three smaller ones; all four are repaired in the commit after it, each shown by
   a planted-error test. At the owner's decision there is no third audit: the second
   audit stated that the repair and its tests are sufficient to check.
4. ~~The owner approves, and this document is published before the first full run~~ —
   approved 2026-09-26; published with the commit that records the approval.

**What was run before registration, disclosed.** The predictions were fixed in
`dbea58f`, before the apparatus existed. While writing it: quick smoke runs (2 seeds,
2 epochs, frame counts divided by 50, 32 test frames, a 64-spectrum pool), whose
printed output — from 2-epoch models — was used only to see the code finish and the
positive-control rule branch; and a timing
run of 2 epochs of each method at λ = 4, which computed no gain. No full-size model was
trained and no registered cell was measured.

## Revision log

None since registration.
