# Preregistration — P2-B: training and inference at different signal-to-noise ratios

**Status: registered 2026-09-26, on the owner's approval, after two independent audits
(of `6459e5b` and `2046086`) and a check of the repair `d548985`; published before the
first full run (`AGENTS.md` §8.1).** ~~Implemented; no full run made yet.~~ *The first full
run was made on 2026-09-26 at `7603007`; its record and what it shows are under Record, added
in Revision 1 and revised in Revision 2 after an independent review. Nothing from it is cleared
for outward-facing quotation.* The design and
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

### Revision 1 — 2026-09-27, after the first full run: the write-up, not the design

The run at `7603007` completed and its record was committed unchanged in `d54acf7`. This
revision adds, after the result existed:

1. the italic sentences at the end of the Status line;
2. the Record section below;
3. `benchmarks/boundaries/snr_transfer/render_report.py`, which generates `report.md` and
   refuses a record it cannot reproduce from the per-seed gains, with the tamper tests in
   `tests/test_snr_transfer_record.py`;
4. `benchmarks/boundaries/snr_transfer/record_citations.py` and
   `tests/test_p2b_record_citations.py`, which pin every number the Record section quotes to
   a named field of the record, with planted-error tests.

No design choice, prediction, threshold, decision rule or self-check is changed, and the
measurement script is not edited: its sha256 is in the record.

### Revision 2 — 2026-09-27, after an independent review of Revision 1

An independent review of `32d7e55` (a different model, a fresh context, the fixed checklist
of `docs/VERIFICATION.md` §3) found no blocking defect and no quoted number wrong. It raised
seven items to fix and seven minor ones. Each is repaired as follows; nothing below changes
a prediction, a threshold, a decision rule, a self-check or the record.

1. **(S1)** The one cell where the output was worse than the input (train 4, inference 100)
   was stated only in `report.md`. It is now stated in the Record section, with its seed
   count and its worst seed.
2. **(S2)** "What this licenses" folded the unregistered M2 > 0 count into the licensed
   statement and mixed cell means with seed counts. It now states the four registered rules
   only, per seed, and labels the M2 > 0 count as descriptive and not licensed.
3. **(S3)** The moving average's training-stack normalisation — a recorded difference between
   the methods and between the two directions — was missing from the candidate readings and
   from the licensing conditions. It is added to both, with the stored constants.
4. **(S4)** Holm-adjusted *p* had no independent check. `render_report.py` now recomputes it
   with its own implementation, and `tests/test_snr_transfer_record.py` tests both that
   implementation and `boundary_common.holm` against a hand-computed family and shows that
   Bonferroni and the unadjusted *p* are told apart from it.
5. **(S5)** The guard accepted edits to the design block's strings and to the P2-A value, and
   its tamper tests did not pin a reason. The guard now checks the whole design block and
   the differences stated beside P2-A against literals, and the P2-A value against P2-A's
   committed record; the "not verified" list names what remains (generation time, wall clock
   among them). Every tamper test pins its reason, and a new test blinds the consistency
   check to show the independent path refuses on its own.
6. **(S6)** The citation test's description overstated what it derives from the gains. Every
   gain, penalty and count is now derived from the per-seed gains; the section says which
   figures are read as stored.
7. **(S7)** The Status line kept "no full run made yet" beside a sentence saying the run was
   made. The phrase is struck through, visibly.
8. **(Minor)** "grows with the distance" is replaced by the ranges it rested on; "never lost
   more than 1 dB" is stated over the 200 seed-cells; "float32 rounding" is removed as a cause
   the record does not store; quoted counts carry anchors; the absence of a `claim_scope`
   field is stated; the SIA paragraph no longer leans on the noise2clean juxtaposition. The
   guard's relative tolerance of one part in 10⁹ is left as it is: immaterial, as the review
   said.
9. **(Follow-up review of `51ab3b8`: minor)** The follow-up review confirmed items 1–8 and
   raised three minor points, repaired here: the guard now refuses a key added to the design
   block and checks the preregistration path and the confound statement, and the report's
   "not verified" list names the record version, the quick-mode flag, the resumed seeds and
   the device; when P2-A's record is absent the P2-A comparison is skipped and the report
   says the value was not checked, which a test shows; and the renderer states that the
   generator and noise2clean-recipe literals are the script's settings at `7603007`, since
   P2-A's record does not carry them to check against.

## Record

Run 2026-09-26 on the MPS backend of the development machine, in the environment pinned
by `uv.lock`, from a clean working tree at `7603007` — the commit that records the
registration. 20 seeds, both methods, all 25 cells; no seed was resumed;
176.3<!--r:run.minutes--> min, against the registered estimate of about
3.5<!--n:reg--> hours. The record is
`benchmarks/boundaries/snr_transfer/results/snr_transfer.json`, committed unchanged in
`d54acf7`; the rendered report is `report.md` beside it.

**Published before the run.** `7603007` was pushed and draft pull request
stoyoda0012-cyber/dnndenoiser#6 opened at 08:35 UTC on 2026-09-26, by GitHub's
timestamps. The run began about three and a half hours later: the record
was written at 15:01 UTC, after 2.9<!--r:run.hours--> hours of wall clock. P2-A could not
say this (its Record, "Limits of provenance"); this record can.

**All seven voiding self-checks passed.** Their stored figures, over every seed: the worst
distance of any frame or pool value from a whole count was 5.7e-6<!--r:chk1.worst-->
counts, against a tolerance of 0.001<!--n:impl-->; λ estimated from the
noise variance lay between 0.988<!--r:chk2.lo--> and 1.015<!--r:chk2.hi--> of the declared λ,
against a factor of 1.3<!--n:reg-->; every model had 658177<!--r:chk7.params--> parameters;
no test frame was found among the training frames and no seed's sample in a pool; in every
seed, all 50<!--r:chk6.cells--> cells' network inputs matched their rebuilt arrays. Each check is shown to reject a
named
wrong input in `tests/test_snr_transfer_gates.py`. These figures are stored by the run and
printed by `report.md` as stored; its guard does not verify them.

**How the numbers here are checked.** As in P2-A: every number below carries an anchor in
the source (`<!--r:…-->` or `<!--n:…-->`, invisible when rendered). An `r:` anchor names the
metric, condition, unit and derivation in
`benchmarks/boundaries/snr_transfer/record_citations.py`, and
`tests/test_p2b_record_citations.py` recomputes it from the record and requires the quoted
text to match at the precision quoted. Every gain, penalty, sign count and verdict count is
derived from the per-seed gains, not from the record's own `predictions` tree; self-check,
normalisation and timing figures are read from where the run stored them, which nothing
re-derives. A count quoted as a record value carries an anchor too; "20/20" in the table
below is anchored on its first number. `report.md` is generated by
`render_report.py`, which refuses a record it cannot reproduce from those gains;
`tests/test_snr_transfer_record.py` tampers with it field by field. The record owns every
number; if they disagree, this section is wrong.

**Units and the ±.** M1 and M2 are in dB. Every ± is the SD across the 20 seeds, which are
the replicates; the 512 test frames within a seed are not. One decimal is quoted: the
moving average's across-seed SDs of M1 and M2 run from 0.4<!--r:ma.sd.min--> to
1.9<!--r:ma.sd.max--> dB, so a second decimal would say more than the seeds support.

### Result — all four predictions held for the moving average

| | Verdict | The number that decided it |
|---|---|---|
| **R1** positive control, λ = 20, 45, 100 | **PASS** | diagonal M1 **+12.6<!--r:ma.m1.t20.i20--> ± 0.5<!--r:ma.m1.t20.i20.sd-->, +14.9<!--r:ma.m1.t45.i45--> ± 0.5<!--r:ma.m1.t45.i45.sd-->, +14.9<!--r:ma.m1.t100.i100--> ± 0.4<!--r:ma.m1.t100.i100.sd--> dB**; positive in 20<!--r:ma.R1.min.k-->/20 seeds each, 19 required |
| **R2** training above inference costs little | **PASS** | M2 > −1 dB in 20<!--r:ma.R2.min.k-->/20 seeds in every one of the ten cells; 17 required |
| **R3** training below inference costs | **PASS** | weakest cell 4 → 9: M2 **−1.2<!--r:ma.m2.t4.i9--> ± 0.6<!--r:ma.m2.t4.i9.sd--> dB**, negative in 19<!--r:ma.R3.k.4.9-->/20; every other cell 20<!--r:ma.R3.min.rest-->/20 |
| **R4** the asymmetry | **PASS** | weakest pair 4 vs 9: **+1.4<!--r:ma.r4.4.9--> ± 0.9<!--r:ma.r4.4.9.sd--> dB**, in 18<!--r:ma.R4.k.4.9-->/20; every other pair 20<!--r:ma.R4.min.rest-->/20 |

R1 held at all three levels, so the failure rule removed nothing and R2 to R4 were
evaluated on their full families of ten, at the registered threshold of 17 of 20. Every
Holm-adjusted *p* is in `report.md`. Unlike P2-A's, this record has no `claim_scope` field:
its scope is the registration's "What this record will not support" and this section.

The moving average's M2, train λ by inference λ (rows: training; columns: inference). Above
the diagonal is training below inference (R3); below it, training above inference (R2).

| train λ \ inference λ | 4 | 9 | 20 | 45 | 100 |
|---|---|---|---|---|---|
| **4** | — | −1.2<!--r:ma.m2.t4.i9--> | −6.2<!--r:ma.m2.t4.i20--> | −12.6<!--r:ma.m2.t4.i45--> | −16.4<!--r:ma.m2.t4.i100--> |
| **9** | +0.1<!--r:ma.m2.t9.i4--> | — | −2.9<!--r:ma.m2.t9.i20--> | −8.0<!--r:ma.m2.t9.i45--> | −11.4<!--r:ma.m2.t9.i100--> |
| **20** | +2.1<!--r:ma.m2.t20.i4--> | +2.0<!--r:ma.m2.t20.i9--> | — | −4.2<!--r:ma.m2.t20.i45--> | −7.3<!--r:ma.m2.t20.i100--> |
| **45** | +6.5<!--r:ma.m2.t45.i4--> | +5.8<!--r:ma.m2.t45.i9--> | +3.6<!--r:ma.m2.t45.i20--> | — | −1.9<!--r:ma.m2.t45.i100--> |
| **100** | +6.8<!--r:ma.m2.t100.i4--> | +6.2<!--r:ma.m2.t100.i9--> | +4.4<!--r:ma.m2.t100.i20--> | +1.4<!--r:ma.m2.t100.i45--> | — |

The R4 statistic ranges from +1.4<!--r:ma.r4.lo--> to +23.1<!--r:ma.r4.hi--> dB over the ten
pairs; over the four pairs of adjacent levels alone it ranges from +1.4<!--r:ma.r4.onestep.lo-->
to +7.8<!--r:ma.r4.onestep.hi--> dB. No prediction names how it varies across pairs, and it is
not argued from.

**One cell made the frames worse than they came in.** M2 is relative to the diagonal. In
absolute terms, the model trained at λ = 4 and applied at λ = 100 had M1
−1.5<!--r:ma.m1.t4.i100--> ± 1.9<!--r:ma.m1.t4.i100.sd--> dB, negative in
16<!--r:ma.m1neg.4.100-->/20 seeds and as low as −4.8<!--r:ma.m1.t4.i100.min--> dB: the output
was further from the clean spectrum than the input. It is the only one of the moving
average's 25 cells whose mean M1 is negative (1<!--r:ma.m1neg.cells--> cell); at 4 → 45 one
seed of 20 was negative (1<!--r:ma.m1neg.4.45-->). This is `AGENTS.md` §5's warning — output can
be worse than the input outside the training distribution — measured here in the direction
R3 names.

### R2 held in a form stronger than it was registered, and that is the result to read carefully

R2 registered that training above the inference flux **costs little**: M2 > −1 dB. What
was measured is not a small cost. In nine of the ten cells, M2 was **positive** — the model
trained at the higher flux denoised the lower-flux frames better than the model trained at
that lower flux — in at least 19<!--r:ma.m2pos.min.rest-->/20 seeds, by
+1.4<!--r:ma.m2.rest.lo--> to +6.8<!--r:ma.m2.rest.hi--> dB. The tenth, 9 → 4, is
+0.1<!--r:ma.m2.t9.i4--> ± 0.4<!--r:ma.m2.t9.i4.sd--> dB, positive in
13<!--r:ma.m2pos.9.4-->/20. **M2 > 0 is not a registered rule;** these counts are
descriptive. R2's verdict rests on M2 > −1 dB, which every cell met in every seed.

The registered report of each R2 cell's own M1 beside its diagonal (`report.md`) matters
less than it was expected to: the diagonals at λ = 4 and 9 did not fail (below), so no R2
cell passed by losing little relative to nothing.

**What "training at a higher flux" means in this design.** Under equal total exposure, a
higher training flux is also fewer training frames and fewer optimiser updates: the model
trained at λ = 100 saw 500 frames in about 800<!--n:reg--> updates, the one at λ = 4 saw
12 500 frames in about 19 550<!--n:reg--> updates. The registration stated that these three
cannot be separated here, and this record does not separate them. **The result is that
training at a higher flux, with fewer frames and fewer updates, transferred downward better
than training at the inference flux, with more of both — not that higher S/N, by itself, did.**

### noise2clean, the descriptive handle on that confound, does not show the pattern

noise2clean's row of the grid varies training flux with the pool fixed at 2304 spectra, so
there the training S/N moves without the amount of training data. Its M2, descriptive:

| train λ \ inference λ | 4 | 9 | 20 | 45 | 100 |
|---|---|---|---|---|---|
| **4** | — | −0.2<!--r:n2c.m2.t4.i9--> | −1.9<!--r:n2c.m2.t4.i20--> | −4.0<!--r:n2c.m2.t4.i45--> | −6.2<!--r:n2c.m2.t4.i100--> |
| **9** | −0.6<!--r:n2c.m2.t9.i4--> | — | −0.4<!--r:n2c.m2.t9.i20--> | −1.5<!--r:n2c.m2.t9.i45--> | −3.1<!--r:n2c.m2.t9.i100--> |
| **20** | −1.2<!--r:n2c.m2.t20.i4--> | −0.2<!--r:n2c.m2.t20.i9--> | — | −0.2<!--r:n2c.m2.t20.i45--> | −0.9<!--r:n2c.m2.t20.i100--> |
| **45** | −2.3<!--r:n2c.m2.t45.i4--> | −1.1<!--r:n2c.m2.t45.i9--> | −0.5<!--r:n2c.m2.t45.i20--> | — | +0.1<!--r:n2c.m2.t45.i100--> |
| **100** | −3.4<!--r:n2c.m2.t100.i4--> | −2.2<!--r:n2c.m2.t100.i9--> | −1.4<!--r:n2c.m2.t100.i20--> | −0.6<!--r:n2c.m2.t100.i45--> | — |

For noise2clean, training above inference cost something in every cell: mean M2 from
−3.4<!--r:n2c.above.lo--> to −0.2<!--r:n2c.above.hi--> dB, positive in
0<!--r:n2c.above.npos--> of the ten. Only 1<!--r:n2c.offdiag.npos--> of the twenty
off-diagonal means is positive. Measured with the moving average's rules, R2's rule is met
in 3<!--r:n2c.R2.met--> of ten cells, R3's in 5<!--r:n2c.R3.met--> of ten and R4's in
1<!--r:n2c.R4.met--> of ten pairs; the R4 statistic's means run from
−0.7<!--r:n2c.r4.lo--> to +2.8<!--r:n2c.r4.hi--> dB. None of this is a verdict.

**So the asymmetry appeared in the method whose training flux moved together with its frame
and update counts, and not in the method whose training flux moved alone.** That is a
juxtaposition, and it cannot be read as an attribution: the two methods also differ in
information (clean targets or none), in training data (a pool of other spectra, or frames
of the one spectrum under test) and in recipe, as the registration said. They also differ in
scaling: the moving average normalises by its training stack's minimum and maximum and applies
those constants at inference, and noise2clean does not. The stored constants differ by flux —
the λ = 4 stack's maximum averaged 3.78<!--r:norm.max.4--> and the λ = 100 stack's
1.35<!--r:norm.max.100--> on a clean maximum of one — so a model applied at another flux sees
inputs outside, or compressed within, the range it was trained on, differently in the two
directions. The juxtaposition is consistent with the moving average's asymmetry depending on
the frame and update counts; with it depending on that scaling; and with it depending on
something the self-supervised targets do and clean targets do not — or on more than one of
these. **This record does not decide between them, and no mechanism is claimed.**
Separating them needs, at least, the moving average trained at one frame count across flux
levels and with a normalisation that does not depend on the training stack — a new
preregistration; none is committed.

### The diagonals, including the two no prediction names

| λ | 4 | 9 | 20 | 45 | 100 |
|---|---|---|---|---|---|
| input SNR (dB) | 2.4<!--r:in.4--> | 5.9<!--r:in.9--> | 9.4<!--r:in.20--> | 12.9<!--r:in.45--> | 16.3<!--r:in.100--> |
| moving average, diagonal M1 (dB) | +9.9<!--r:ma.m1.t4.i4--> ± 0.5<!--r:ma.m1.t4.i4.sd--> | +10.9<!--r:ma.m1.t9.i9--> ± 0.5<!--r:ma.m1.t9.i9.sd--> | +12.6<!--r:ma.m1.t20.i20--> ± 0.5<!--r:ma.m1.t20.i20.sd--> | +14.9<!--r:ma.m1.t45.i45--> ± 0.5<!--r:ma.m1.t45.i45.sd--> | +14.9<!--r:ma.m1.t100.i100--> ± 0.4<!--r:ma.m1.t100.i100.sd--> |
| noise2clean, diagonal M1 (dB) | +17.8<!--r:n2c.m1.t4.i4--> ± 0.9<!--r:n2c.m1.t4.i4.sd--> | +16.4<!--r:n2c.m1.t9.i9--> ± 1.0<!--r:n2c.m1.t9.i9.sd--> | +15.4<!--r:n2c.m1.t20.i20--> ± 0.9<!--r:n2c.m1.t20.i20.sd--> | +14.3<!--r:n2c.m1.t45.i45--> ± 0.6<!--r:n2c.m1.t45.i45.sd--> | +13.1<!--r:n2c.m1.t100.i100--> ± 0.5<!--r:n2c.m1.t100.i100.sd--> |

At λ = 4 and 9 the moving average's diagonal was positive in every seed. These two cells are
**descriptive**: R1 did not include them, because the SIA paper reports that training at
its most photon-starved exposure failed structurally. **This record cannot speak to that
failure.** It measures a mean-squared error against the clean spectrum, not the diversity of
the output, and within a seed every test frame has the same clean spectrum — so an output
close to that spectrum and nearly the same for every frame scores well here. The flux levels
map onto SIA's exposures by relative flux only. A positive gain at λ = 4 is therefore not
evidence against what SIA reported.

**The two methods' diagonals are not a ranking of the methods.** The moving average is
trained on frames of the very spectrum it is then tested on; noise2clean on a pool of other
spectra from the same generator, with clean targets and another recipe. That noise2clean's diagonal is
+7.9<!--r:diag.n2c.minus.ma.4--> dB above the moving average's at λ = 4 and
−1.7<!--r:diag.n2c.minus.ma.100--> dB below it at λ = 100 is recorded — both ends, because
either alone reads as a ranking — and the design cannot attribute either difference.

### noise2clean at λ = 100 beside P2-A — descriptive, no tolerance

noise2clean's diagonal at λ = 100 is +13.1<!--r:n2c.m1.t100.i100--> dB; P2-A's arm A at
level 1000 and Δ = 0, as this run copied it into the record, is
+11.4<!--r:p2a.copied--> dB. They differ in noise model (exact Poisson here, the Gaussian
approximation there), in training data (one level here, three there) and in the test set
(one clean spectrum per seed here, independently generated spectra there, so the spread
across seeds means different things). **This is not a re-measurement of P2-A**, and the
difference between the two numbers is not interpreted.

### What this has to do with the SIA paper

R2 and R3 together were registered as this document's analogue, on synthetic spectra, of
the SIA paper's empirical directional rule, train S/N ≥ inference S/N. Both held. The
registration's statement stands unchanged: **P2-B neither supports nor refutes that rule and
is not a reproduction of the paper.** Even as an analogue, what held is a statement
about the moving average under equal exposure and its own normalisation, in which S/N is not
separated from the frame and update counts or from the scaling; the noise2clean juxtaposition
above is descriptive and does not settle which of them matters.

### What this licenses, and nothing stronger

That **this** ResNet-FCNN, trained by the library's self-supervised moving average at
W = 1 on the frames of one synthetic `C1s_adventitious` spectrum per seed, normalised by
its training stack's minimum and maximum, with exact Poisson noise and equal total exposure
across λ = 4 to 100, on the MPS backend, met the four registered rules:

- R1: its diagonal M1 was positive at λ = 20, 45 and 100 in every seed;
- R2: trained at a higher flux than the frames it denoised, its M2 was above
  −1<!--n:reg--> dB in all 10 cells and all 20 seeds — in none of the
  200<!--r:ma.R2.seedcells--> seed-cells did it lose more than 1<!--n:reg--> dB; the worst was
  −0.6<!--r:ma.R2.min.m2--> dB;
- R3: trained at a lower flux, its M2 was negative in at least 17<!--n:reg--> of 20 seeds in
  every cell (19<!--r:ma.R3.k.4.9--> in the weakest);
- R4: the loss going up exceeded the loss going down, paired within seed, in at least
  17<!--n:reg--> of 20 seeds in every pair (18<!--r:ma.R4.k.4.9--> in the weakest).

That M2 was *positive* in nine of the R2 cells is descriptive (above) and is not part of what
is licensed. None of this is separated from the frame and update counts, or from the
normalisation constants, that came with each flux. The registration's list "What this record
will not support" applies in full.

### Limits of provenance, stated rather than repaired

- **What was fixed before the result.** The predictions and decision rules were fixed in
  `dbea58f`, before the apparatus; the registration was approved and published at
  `7603007`, before the run. This section, the Status line, `render_report.py`,
  `report.md`, `record_citations.py` and their tests were written after the result existed
  (Revision 1). None of them changes a prediction, a threshold or a decision rule.
- **What a reader can check.** The record carries every seed's gains for every cell, so
  every aggregate, sign count, *p*, Holm-adjusted *p* and verdict is re-derivable from it,
  and `render_report.py` re-derives them. The per-seed gains themselves, the self-check
  figures and the environment can be checked only by re-running; a re-run on another
  backend is not expected to be bit-identical.
- **One run.** This is the first and only full run. Nothing here rests on agreement with
  another run.
- **Timestamps.** "Published before the run" rests on GitHub's timestamps for the push and
  the pull request, and on the record's own time; the record's time is written by the run
  on a machine its author controls.

Nothing from this record goes into the README, the package documentation or any release
note until the claims to be quoted, and the place quoting them, have been independently
reviewed and a person has decided what may be quoted (`docs/VERIFICATION.md` §4). No such
decision has been made.
