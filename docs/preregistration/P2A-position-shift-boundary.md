# Preregistration — P2-A: the position-shift boundary, and what augmentation does to it

**Status: registered 2026-09-22. Not implemented; not run; no shifted-test
metric has been computed.** Nothing below may be revised to match a result. When
a prediction or a rule turns out to be wrong it is changed *visibly*, with the
reason and the date, and the change is part of the record — see **Revision log**.

This is the first record of **P2** (`records/POST_IMPORT_PLAN.md`): failure
boundaries as reproducible records. P2 is a series; this document registers one
record of it, on one axis.

## Why this is registered before the code

`AGENTS.md` §5 states that distribution shift is the dominant failure mode of a
denoiser, and the reference benchmark's own README states that **nothing in it
probes distribution shift** — its training and test spectra are independent
draws from one distribution. So the package currently ships a measurement of
how well the denoiser does when everything is in-distribution, and no
measurement at all of where that stops being true.

A published number about *where a method fails* is a scientific claim under
`AGENTS.md` §6 and an independent-audit item under §8. A threshold chosen after
seeing the curve is not a threshold, so the design, the metrics, the decision
rules and the predictions are fixed here first.

**Nothing is taken from the private research ledger.** `POST_IMPORT_PLAN.md`
names "the position-shift cliff" as a candidate topic, and that naming is the
whole of the inheritance: no number, no threshold, no figure and no design
detail from `experiments/` or from any private note is used here or will be
used in the record. The design below is built from this repository's own
generator and its published reference benchmark. If the re-measurement disagrees
with anything privately believed, the re-measurement is what is published.

## What is manipulated, and what the claim will be about

A **rigid energy shift** Δ applied to every peak of the spectrum at once, with
the energy grid held fixed. In XPS this is the shape of a sample-charging
offset or a binding-energy calibration error: the whole envelope moves, the
structure within it does not. It is *not* the shape of a chemical shift, which
moves components relative to one another; that case is out of scope here.

Two models are trained and evaluated on the same test spectra:

| Training condition | Rigid shift in training | What it stands for |
|---|---|---|
| **narrow** | none (Δ = 0 for every training spectrum) | a model trained at one calibration, the ordinary case |
| **augmented** | Δ ~ U(−1.5, +1.5) eV, drawn per training spectrum | the obvious mitigation: train across the shifts you expect |

Both are tested over Δ ∈ ±3.0 eV, which reaches **beyond** the augmented
model's training range. That is deliberate. The registered question is not
"does augmentation fix it" but **"does augmentation remove the boundary or move
it"** — and a design whose test range stopped at the augmentation range could
not tell those apart.

Per-peak random jitter of ±0.3 eV is present in **both** conditions and is
**not** manipulated. It is the reference benchmark's own setting, and it matters
that it stays: the narrow model is not trained on a single frozen position, it
is trained on a ±0.3 eV neighbourhood. Whatever boundary is measured is a
boundary relative to *that* neighbourhood.

## The design

Everything below is fixed. Values that came from the reference benchmark are
marked, because a run here is meant to be readable next to that record.

### Held fixed

| | | From |
|---|---|---|
| Architecture | `ResNet-FCNN`, `num_features=256`, `num_hidden_units=100`, `encoder_output_dim=64`; 658,177 trainable parameters | reference benchmark |
| Optimiser | Adam, `weight_decay=1e-9` | reference benchmark |
| Loss | `HuberLoss(delta=1.0)` | reference benchmark |
| Schedule | `StepLR(step_size=10, gamma=0.1)`, gradient clip 4.0 | reference benchmark |
| Hyperparameters | `lr=0.001`, `epochs=50`, `batch_size=16` | the suggested values for this architecture |
| Peak set | `C1s_adventitious` — 284.8 (1.0), 286.3 (0.3), 288.5 (0.15) eV | reference benchmark |
| Energy grid | **pinned** to (277.8, 295.5) eV, 256 points, 0.069412 eV/point | the peak set's own auto-range, pinned so it cannot move with Δ |
| Generator | `eta=0.3`, pseudo-Voigt, linear background (level 0.05, slope 0.001), `intensity_variation=0.2`, `position_jitter=0.3`, `width_variation=0.1`, `normalize=True` | reference benchmark |
| Noise | Poisson at levels 100, 1000, 10000, with the reference benchmark's pinned `use_gaussian_approx` reconstruction | reference benchmark |
| Training pool | 768 spectra per noise level, all three levels mixed = 2304 | reference benchmark |
| Test set | 512 spectra per noise level **per Δ** | reference benchmark's per-level size |
| Seeds | **8** | see below |

Only **one architecture** is trained. This record is about distribution shift,
not about architectures, and fixing the architecture removes the confound the
reference benchmark deliberately keeps.

**Why eight seeds and not five.** The seed is the replicate unit and the only
thing the error bars mean. The reference benchmark used five because eighty
training runs at five seeds already cost half an hour. This design needs
sixteen training runs, so the same wall clock buys more replicates, and the
reference README's own rule — *cut sample count, not seeds* — says to spend it
there. Eight is registered; it is not to be reduced after seeing a result.

### Manipulated

**Test shift Δ (eV), 21 levels**, denser where the boundaries are expected:

```
0, ±0.25, ±0.5, ±0.75, ±1.0, ±1.25, ±1.5, ±1.75, ±2.0, ±2.5, ±3.0
```

Signed in both directions on purpose. The peak set is not symmetric within the
window — the dominant 284.8 eV peak has 7.0 eV of room to the left and the
weakest 288.5 eV peak has 7.0 eV to the right — so +Δ and −Δ push different
peaks toward different edges, and a boundary that turns out to be asymmetric is
a finding rather than an artefact to be averaged away.

**Δ is capped at 3.0 eV** so that truncation cannot masquerade as failure. See
the self-checks: at ±3.0 eV the clean spectrum retains more than 99.5 % of its
peak area inside the window, measured, not assumed.

### Pairing, and what the error bars are the dispersion of

- **Across Δ, within a seed and level: paired to the spectrum.** Every test
  condition at one (seed, noise level) is drawn with the *same* generator seed,
  so test spectrum *i* at Δ = 1.0 and test spectrum *i* at Δ = 0 share their
  per-peak jitter, intensity and width draws and differ only by the rigid
  shift. This makes the Δ-sweep a within-spectrum comparison and removes draw
  noise from the shape of the curve.
- **Across training conditions, within a seed: paired.** The narrow and
  augmented pools are generated from the same per-sample seed sequence, so they
  differ only in the shift drawn for each sample. Both models are evaluated on
  the identical test arrays.
- **Across seeds: independent.** One seed is one independently drawn training
  pool, one independent torch seed for initialisation and batch order, and one
  independently drawn test-spectrum family.

**Every reported error bar is the standard deviation across the eight seeds.**
The per-spectrum spread inside one test set is *not* a replicate spread — those
spectra share one trained model — and is stored under the key
`snr_gain_db_sd_over_spectra_not_a_replicate_sd`, following the reference
benchmark, so it cannot be mistaken for one.

### Split rule and leakage

Training and test draws use disjoint RNG stream bases, which is what creates
the split. A hash check over every test spectrum against every training
spectrum is what catches that separation silently failing, and a
nearest-neighbour statistic is recorded alongside it, because at Δ = 0 a test
spectrum is expected to be about as close to its nearest training spectrum as
training spectra are to each other — that is the correct outcome for an
independent draw from one distribution, and it is exactly why the Δ = 0 column
must be read as *in-distribution* performance.

### Regime

Training and inference operate at the **same signal-to-noise regime**: all
three noise levels are in the training pool and each is evaluated separately.
The noise level is not a manipulated variable, and no claim is made about
extrapolation to unseen noise levels. The manipulated axis is position only.

## Metrics

### M1 — SNR gain in dB (primary)

The reference benchmark's definition, unchanged:

```
signal power = mean(reference**2)
noise power  = mean((estimate - reference)**2)
SNR (dB)     = 10 * log10(signal power / noise power)
gain         = SNR(denoised) - SNR(noisy input)
```

The reference is the clean synthetic spectrum **at the same Δ** — that is, the
truth the model should have recovered, not the unshifted truth. Scoring against
the unshifted reference would measure something else entirely and is not done.

Two derived quantities are reported at every (condition, level, Δ):

- **absolute gain** — mean over spectra, then mean and SD over seeds;
- **degradation** — `gain(Δ) − gain(0)` computed *within a seed* before
  averaging, so it inherits the paired construction.

Degradation is well defined at every noise level. Absolute gain crosses zero
only where it starts positive, which is why the boundary statistic below is
conditioned.

### M2 — the boundary |Δ|\*, where it exists (primary, policy)

Per seed, per condition, per level, per direction: the smallest |Δ| at which
the seed's mean gain crosses zero, by linear interpolation between the two
adjacent grid points that bracket it. Reported as the **median** across seeds
with the inter-seed range, and with the number of **censored** seeds — seeds
whose gain never crosses within ±3.0 eV — stated next to it. A median is used,
not a mean, precisely because censoring is expected at the edges and a mean
over a censored sample would be a fabricated number.

**|Δ|\* is not computed where the mean gain at Δ = 0 is already negative.** The
reference benchmark records that this architecture *loses* about 5 dB at the
lowest noise level even fully in-distribution: the network's own reconstruction
error there exceeds the noise it removes. At such a level a "zero crossing" is
not a boundary, and the record will say `not defined at this level` rather than
report Δ = 0.

### M3 — argmax displacement in eV (secondary, confirmatory)

For each spectrum: `energy[argmax(denoised)] − energy[argmax(clean reference)]`,
signed, on the fixed 0.069412 eV grid. Reported as a mean over spectra then over
seeds, together with the same quantity computed on the **noisy input**, as a
comparator.

This exists because SNR gain alone under-reports the danger. The failure
`AGENTS.md` §5 actually warns about is not blur, it is a network placing
structure where it was *trained* to expect it. If that happens, the denoised
argmax sits back toward the training position and the signed displacement takes
the **opposite sign to Δ**. No other quantity in this record would show that.

Its limits, registered now rather than discovered later: it is a grid argmax,
not a fitted peak position; it is quantised to 0.069412 eV; it describes the
single dominant peak and says nothing about the two weak ones; and it supports
no claim about fitted binding energies. A self-check records the fraction of
clean references whose argmax lies within 0.5 eV of the nominal dominant peak
position, and the metric is reported as unreliable if that falls below 99 %.

## Self-checks that void the record

The script refuses to write a record if any of these fails. A failure here
means the measurement is broken, and a broken measurement is not reinterpreted
as a finding.

1. **Parameter count** — `ResNet-FCNN` at the pinned shape has exactly 658,177
   trainable parameters.
2. **Grid invariance** — the energy axis is bit-identical at every Δ.
3. **Rigidity** — at every Δ, each nominal peak centre equals its unshifted
   value plus Δ exactly, and all nominal inter-peak spacings are unchanged.
4. **Truncation** — the clean peak area retained inside the window at every Δ
   is within **1 %** of its value at Δ = 0. (Measured during design at ±3.0 eV:
   99.52 % at −3.0, 99.91 % at +3.0. The tolerance is set above the observed
   worst case and below anything that could explain a sign change in gain.)
5. **Input-SNR invariance** — the mean SNR of the *noisy input* against its own
   reference varies by no more than **0.2 dB** across all Δ, at every noise
   level. This is the check that matters most: if shifting changed the input
   SNR, a gain curve against Δ would be confounded at the source. (Measured
   during design across Δ ∈ {−3, 0, +3}: worst case 0.08 dB.)
6. **Augmentation actually happened** — the narrow pool's drawn shifts are all
   exactly 0.0; the augmented pool's span at least 95 % of (−1.5, +1.5) and
   have a mean within 0.1 eV of zero. Both pools' shift distributions are
   written into the record.
7. **Leakage** — no test spectrum is byte-identical to any training spectrum,
   in either condition; the nearest-neighbour statistic is recorded.
8. **Argmax well-posedness** — as in M3 above.
9. **Run-to-run determinism** — the narrow condition is trained twice on
   identical inputs at one seed and the scores compared; the result is
   *recorded*, not assumed.

## Registered predictions

Level **1000.0** (input SNR ≈ 16.3 dB) is the **primary** level, fixed here so
that the result is not selected from three afterwards. The other two levels are
reported in full and read descriptively.

Where a test is stated it is a two-sided one-sample *t*-test on the eight
per-seed values against zero, α = 0.05, **Holm-adjusted within the family named
in that prediction**, reported with Cohen's *d<sub>z</sub>* and the number of
seeds favouring — and, following the reference benchmark, sign consistency is
read alongside the *p*-value with neither decisive alone. Eight paired seeds is
a small sample and is not being presented as more.

| | Prediction | Decision rule |
|---|---|---|
| **R1** | *Positive control.* The narrow model gains at Δ = 0. | Mean gain > 0 **and** positive in 8/8 seeds. If this fails the apparatus is broken and nothing else in the record is read. |
| **R2** | *There is a cliff.* The narrow model's gain is negative at Δ = +3.0 and at Δ = −3.0. | Mean < 0, negative in ≥ 7/8 seeds, and *p* < 0.05 Holm-adjusted across the two directions. |
| **R3** | *The cliff is narrow.* The narrow model's \|Δ\|\* is ≤ **1.5 eV** in at least one direction. | Median across seeds ≤ 1.5, with ≤ 1 censored seed in that direction. |
| **R4** | *The shift is the cause, not the difficulty.* At \|Δ\| = 1.5 the augmented model gains more than the narrow model. | Paired within seed, augmented > narrow in ≥ 7/8 seeds, both directions. |
| **R5** | *Augmentation is not free.* At Δ = 0 the augmented model gains no more than the narrow model. | Augmented ≤ narrow in ≥ 6/8 seeds. |
| **R6** | *The boundary moves, it does not vanish.* The augmented model's \|Δ\|\* is larger than the narrow model's, and it still has one within ±3.0 eV in at least one direction. | Both parts required; the second part is the falsifiable half. |
| **R7** | *The network pulls the peak home.* For the narrow model at \|Δ\| ≥ 1.0, the mean signed argmax displacement has the **opposite sign to Δ**, and \|displacement\| grows with \|Δ\|. | Correct sign at all four of Δ ∈ {±1.0, ±3.0}; monotone in \|Δ\| over {1.0, 1.5, 2.0, 3.0} within each direction, allowing violations no larger than the across-seed SD. |

**R3 is the prediction most likely to be wrong, and it is the one that matters.**
A 1.5 eV calibration offset or charging shift is entirely ordinary in XPS. If
the boundary turns out to sit at 2.5 eV the registered prediction is *false* and
the record says so; if it sits at 0.5 eV the warning in the README has to be
much louder than anything currently written there.

**R6's second half is the one that could embarrass the mitigation.** If the
augmented model holds up all the way to ±3.0 eV with no crossing, R6 fails and
this record cannot claim that augmentation merely relocates the boundary — it
would then only have shown that the boundary is beyond the range tested, which
is a weaker and different statement, and is what would be written.

### A consistency anchor, not a prediction

The narrow condition is the reference benchmark's own primary condition for
this architecture, re-drawn with this script's seeding. Its Δ = 0 gains should
land near that record's published values — −5.205, +11.577, +20.624 dB at
levels 100, 1000, 10000. The script records the comparison and **flags** any
level differing by more than 3 dB. A flag is not a failure: the draws are
different, the seed count is different, and the per-sample generation path is
different. It is a prompt to explain the difference in the record before
publishing anything.

## What this record will not support

Written before the numbers exist, so that it cannot be trimmed to fit them.

- **Nothing about measured spectra.** Every spectrum here is synthetic, from
  this package's own generator, scored against a reference that exists only
  because the data is synthetic. The training noise and the test noise come
  from the same function, so the model's noise model is exactly correct by
  construction — a condition measured data never satisfies.
- **No general threshold for XPS denoising.** Whatever |Δ|\* comes out is a
  property of *this* peak set, *this* jitter width, *this* architecture, *this*
  training-set size and *this* noise model. One point was measured in each of
  those spaces.
- **Nothing about non-rigid shifts.** Chemical shifts move components relative
  to each other. This manipulation cannot speak to them.
- **Nothing about peak areas, widths, or fitted positions surviving denoising.**
  M3 is a grid argmax and is not a peak fit. Physically meaningful quantities
  have to be verified downstream (`AGENTS.md` §5).
- **No recommendation that augmentation is the right mitigation.** R5 and R6
  are designed to show its price and its edge. Measuring a mitigation is not
  endorsing it, and calibrating the instrument remains the better answer to a
  calibration error.
- **No claim about architectures.** One was trained.
- **Nothing bit-exact across devices.** Floating-point reduction order differs
  between CPU, MPS and CUDA backends. The predictions above are about signs,
  orderings and thresholds and are expected to survive a change of backend; the
  recorded magnitudes are properties of the machine named in the record's
  `environment`, and no cross-device comparison is recorded unless one is run.
- **A denoised spectrum is a model estimate, not a measurement.** A high gain
  says the output agrees with a known synthetic reference. It does not
  establish that structure in the output is real.

## If a prediction fails

It is recorded as failed, with the number that failed it, and the record is
published anyway. Negative and undecided results stay (`AGENTS.md` §6). A
failed prediction is not converted into a different prediction that the data
satisfies; if a follow-up design suggests itself, it is a *new* preregistration
with a new date.

A failed **self-check** is different: it voids the run. The record is not
written, the cause is fixed, and the run is repeated.

## Release and audit

- The design registered here is an independent-audit item under `AGENTS.md` §8
  on three counts: it adds a published scientific claim, it adds an evaluation
  metric (M3), and it states the conditions of a published measurement.
- **This document goes to independent audit before any of it is implemented**,
  as P1's did. P1's audit found that its criteria could be passed in full by a
  wrong implementation; the same question is the one to ask here.
- The completed record goes to a second independent audit before any number
  from it is quoted in the README or anywhere else outward-facing.
- Nothing from this record is written into the README, the package
  documentation, or any release note until both audits are done.
- Artefacts, following `benchmarks/reference/`: a script, a JSON record that is
  the only place numbers live, a renderer, a generated report that is never
  edited by hand, and a README stating what the record does not support. All
  under `benchmarks/boundaries/position_shift/`, which is inside the existing
  distribution boundary — `benchmarks/` is already denied by the `build` job,
  so no boundary moves and §3 is untouched.

## Revision log

Revisions are appended here with a date and the finding that caused them. The
registered text is preserved in git; nothing above is silently edited.

*(No revisions yet — registered 2026-09-22.)*

## Record

*(Empty until the run. The result section goes here, including any prediction
that failed.)*
