# Preregistration — P2-A: the position-shift boundary, and what augmentation does to it

**Status: registered 2026-09-22; revised ten times (see Revision log); run six
times — the first two discarded for defects in the apparatus, the third to fifth
superseded; the sixth, made under Revision 9, reproduced the fifth's measurement exactly.
The result is in the Record section, last. Seven of eight predictions held and R5b failed.
Not yet cleared for outward-facing quotation.** This line previously read "Not
implemented; not run" for a day after the record existed, because nothing checked it.
Nothing below may be revised to match a result. When a prediction or a rule turns out to be
wrong it is changed *visibly*, with the reason and the date — see **Revision log**.

> **Revision 1 — after two independent audits, before any implementation.** The
> audits found, between them, **six blocking defects**. Three are the P1 failure
> mode repeating: a *wrong* implementation would have satisfied all seven
> predictions and all nine self-checks — in particular, **nothing verified that
> the augmented training pool was shifted rigidly at all**, so a pool built with
> per-peak jitter (the manipulation this document puts explicitly out of scope)
> would have passed everything. Two are the opposite failure: a *correct*
> implementation would have been voided for reasons unrelated to the science.
> One is a confound that would have made a registered prediction pass with
> d<sub>z</sub> ≈ 10 for entirely the wrong reason. And **two numbers this
> document reported as "measured during design" were wrong**. All are repaired
> here; the registered text is preserved in git at commit `06fa8c0`, and every
> change is itemised in the revision log with the finding that caused it.

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

Four models are trained and evaluated on the same test spectra:

| Arm | Training shift | N | What it stands for |
|---|---|---|---|
| **A — narrow** | none (Δ = 0 for every training spectrum) | 2304 | a model trained at one calibration, the ordinary case |
| **B — augmented** | Δ ~ U(−1.5, +1.5) eV, rigid, drawn per training spectrum | 2304 | the obvious mitigation: train across the shifts you expect |
| **C — density control** | none | 461 | arm A at the *marginal* position-density of arm B |
| **D — density control** | none | 144 | arm A at the *joint* position-density of arm B |

Arms C and D exist because A and B differ in **two** things, not one. Arm A
draws each peak position from a 0.6 eV-wide window (per-peak jitter ±0.3 eV);
arm B draws from that convolved with ±1.5 eV, a 3.6 eV-wide trapezoid. At the
same N = 2304, arm B therefore has **one fifth** of arm A's training density in
the per-peak marginal, and **one sixteenth** in the three-peak joint
configuration space (Minkowski volume 3.456 vs 0.216 eV³). Without a control,
any deficit arm B shows at Δ = 0 is unattributable — and this repository has
already measured that the deficit would be large: its own training-set-size
record puts `ResNet-FCNN` at +2.056 dB for a 2.67× enlargement at level 1000,
against an across-seed SD of 0.321 dB, which extrapolates to **≈3.4 dB from
density alone**. Arms C and D cost 2.6 and 0.8 minutes of training.

Arms A–D are tested over Δ ∈ ±4.0 eV, which reaches **beyond** arm B's training
range. That is deliberate. The registered question is not "does augmentation fix
it" but **"does augmentation remove the boundary or move it"** — and a design
whose test range stopped at the augmentation range could not tell those apart.

Per-peak random jitter of ±0.3 eV is present in **all four** arms and is
**not** manipulated. It is the reference benchmark's own setting, and it matters
that it stays: arm A is not trained on a single frozen position, it is trained on
a ±0.3 eV neighbourhood. Whatever boundary is measured is a boundary relative to
*that* neighbourhood.

## The design

Everything below is fixed. Values that came from the reference benchmark are
marked, because a run here is meant to be readable next to that record.

### Held fixed

| | | From |
|---|---|---|
| Architecture | `ResNet-FCNN`, `num_features=256`, `num_hidden_units=100`, `encoder_output_dim=64` | reference benchmark |
| Parameter count | read at run time from `self_checks.parameter_counts.observed['ResNet-FCNN']` in the reference record, and asserted | reference benchmark |
| Optimiser | Adam, `weight_decay=1e-9` | reference benchmark |
| Loss | `HuberLoss(delta=1.0)` | reference benchmark |
| Schedule | `StepLR(step_size=10, gamma=0.1)`, gradient clip 4.0 | reference benchmark |
| Hyperparameters | `lr=0.001`, `epochs=50`, `batch_size=16` | the suggested values for this architecture |
| Peak set | `C1s_adventitious` — centres **284.8 / 286.3 / 288.5 eV**, FWHM 1.2 / 1.3 / 1.4, intensity 1.0 / 0.3 / 0.15. These literals are the baseline every rigidity check compares against; they are never re-read from `PEAK_SETS` at check time. | reference benchmark |
| Energy grid | **pinned** via `GeneratorConfig.energy_range` to (277.8, 295.5) eV, 256 points, 0.069412 eV/point | the peak set's own auto-range, pinned so it cannot move with Δ |
| Generator | `eta=0.3`, pseudo-Voigt, linear background (level 0.05, slope 0.001), `intensity_variation=0.2`, `position_jitter=0.3`, `width_variation=0.1`, `normalize=True` | reference benchmark |
| Noise | `NoiseConfig(noise_type='poisson', poisson_level=L, use_gaussian_approx=(10000.0/L)**2 > 20, gaussian_approx_min_rate=0.0)` at L ∈ {100, 1000, 10000} — see **Noise model**, below | reference benchmark |
| Training pool | 768 / 154 / 48 spectra per noise level, all three levels mixed = **2304 / 461 / 144** | reference benchmark, scaled for arms C and D |
| Test set | 512 spectra per noise level **per Δ** | see **Why 512**, below |
| Seeds | **20** | see below |

Only **one architecture** is trained. This record is about distribution shift,
not about architectures, and fixing the architecture removes the confound the
reference benchmark deliberately keeps.

**Why twenty seeds.** The seed is the replicate unit and the only thing the
error bars mean. At n = 8 — the count first registered — the exact two-sided
binomial tail for "≥ 7 of 8 seeds agree in sign" is 0.0703, so **no sign rule
that tolerates even one dissenting seed reaches α = 0.05**, and three of the
seven predictions were registered with rules at 0.070 and 0.289. The binding
constraint is not compute: at the reference record's measured 39.7 s per
training run for this architecture, all eighty runs across the four arms cost
**≈30 minutes**. At n = 20, "≥ 15 of 20" has one-sided p = 0.0207 and tolerates
five dissenters. Twenty is registered; it is not to be reduced after seeing a
result.

**Why 512 test spectra.** *Not* for precision on the mean gain — the replicate
is the seed, and enlarging a test set shrinks only the within-model sampling
term, which is why that spread is stored under the deliberately awkward key
`snr_gain_db_sd_over_spectra_not_a_replicate_sd`. 512 is retained because
**|Δ|\* is a first-crossing statistic**, a non-linear functional of each seed's
own gain curve, whose downward bias is governed by the noise in that curve and
is one-sided — so no amount of averaging across seeds removes it. Test-set size
is not the binding cost here; training is.

### Noise model, stated rather than inherited

The reference benchmark's noise configuration is a pinned reconstruction of this
package's **pre-2026-09-11** behaviour, and its source calls itself "the one call
site allowed to ask for the superseded model". This record adopts it — with the
literal config written into the table above, not named by reference — so that
the Δ = 0 column is comparable with that record. Two consequences are registered
rather than discovered:

- The boolean is **False** at level 10000 and **True** at levels 100 and 1000,
  so the three levels do not share a noise code path. An implementer who writes
  `use_gaussian_approx=True` throughout, or who takes the current per-bin
  default `gaussian_approx_min_rate=3.0`, changes the noise model — and the
  input-SNR check would not notice. Self-check 11 is what notices.
- Under this configuration, with `normalize=True` and `background_level=0.05`,
  the dimmest bin sits at ≈4.7 % of the peak and the worst-bin upward bias at
  level 1000 is ≈0.25 %. It is approximately Δ-independent, because the
  background floor is flat, which is why it does not confound the manipulated
  axis.

### Manipulated

**Test shift Δ (eV), 25 levels**, denser where the boundaries are expected:

```
0, ±0.25, ±0.5, ±0.75, ±1.0, ±1.25, ±1.5, ±1.75, ±2.0, ±2.5, ±3.0, ±3.5, ±4.0
```

Signed in both directions on purpose. The peak set is not symmetric within the
window — the dominant 284.8 eV peak has 7.0 eV of room to the left and the
weakest 288.5 eV peak has 7.0 eV to the right, but those are **5.83 and 5.00
FWHM** respectively — so +Δ and −Δ push different peaks toward different edges
by different amounts, and a boundary that turns out to be asymmetric is a
finding rather than an artefact to be averaged away. *Provided* it clears the
truncation asymmetry recorded in self-check 5, which is a condition, not a
formality: see there.

**Δ is capped at 4.0 eV** so that truncation cannot masquerade as failure. The
pseudo-Voigt tails (η = 0.3) put **≈1.6 % of the nominal peak area outside this
window at every Δ, Δ = 0 included**; what the cap controls is the *change*.
Measured at design time by analytic integration of the pseudo-Voigt profiles:

| Δ | area inside window | ratio to Δ = 0 |
|---|---|---|
| 0 | 98.426 % | 100.000 % |
| −3.0 / +3.0 | 97.948 % / 98.335 % | **99.514 % / 99.907 %** |
| −4.0 / +4.0 | 97.505 % / 98.168 % | **99.063 % / 99.737 %** |
| −4.5 / +4.5 | 97.151 % / 98.038 % | 98.704 % / 99.605 % |

The binding side is negative — the dominant peak — and it crosses the 1 %
tolerance at |Δ| ≈ 4.1 eV. Hence 4.0, not 3.0 and not 4.5.

### Pairing, and what the error bars are the dispersion of

- **Across Δ, within a seed and level: the clean spectra are paired to the
  spectrum.** Every test condition at one (seed, level) is drawn with the *same*
  generator seed, so test spectrum *i* at Δ = 1.0 and at Δ = 0 share their
  per-peak jitter, intensity and width draws; the clean spectra differ only by
  the rigid shift and a per-sample normalisation constant.
- **The noise realisation is a separate matter**, and its degree of pairing is a
  property of the noise branch rather than of this design. Measured at design
  time, corr(noise at Δ = 0, noise at Δ = 0.5): **+0.977 at level 100, +0.976 at
  level 1000, +0.177 at level 10000**. At the first two the pinned branch draws
  one standard-normal vector at an identical stream position and only the scale
  changes; at level 10000 the branch is `rng.poisson`, whose bit consumption is
  rate-dependent, so the stream diverges. **Degradation curves at level 10000
  therefore carry more draw noise than at the other two levels**, and this is
  stated in the record rather than left to be inferred. It is also the mechanism
  behind self-check 5's per-level tolerance.
- **Across arms, within a seed: paired.** All four pools are generated from the
  same per-sample seed sequence, so at equal index they differ only in the shift
  drawn (and, for C and D, in how many samples are kept). All four models are
  evaluated on the identical test arrays.
- **Across seeds: independent.** One seed is one independently drawn training
  pool, one independent torch seed for initialisation and batch order, and one
  independently drawn test-spectrum family.

**Every reported error bar is the standard deviation across the twenty seeds.**

### Split rule and leakage

Training and test draws use disjoint RNG stream bases, asserted arithmetically
as well as by hashing. A hash check over every test spectrum against every
training spectrum is what catches that separation silently failing, and a
nearest-neighbour statistic is recorded alongside it, because at Δ = 0 a test
spectrum is expected to be about as close to its nearest training spectrum as
training spectra are to each other — that is the correct outcome for an
independent draw from one distribution, and it is exactly why the Δ = 0 column
must be read as *in-distribution* performance.

Registered limitation: **the hash check can only detect a stream collision in
the Δ = 0 column**, because at Δ ≠ 0 a collided spectrum is shifted and no
longer byte-identical. The Δ = 0 test set is therefore always included in the
hash set, and the stream-base disjointness is asserted directly.

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

The reference is the clean synthetic spectrum **at the same Δ** — the truth the
model should have recovered, not the unshifted truth. Scoring against the
unshifted reference would measure something else entirely; self-check 9 is what
establishes that it was not done.

Reported at every (arm, level, Δ):

- **absolute gain** — mean over spectra, then mean and SD over seeds;
- **degradation** — `gain(Δ) − gain(0)` computed *within a seed* before
  averaging, so it inherits the paired construction.

### M2 — the boundary |Δ|\*, where it exists (primary, policy)

Per seed, per arm, per level, per direction, **three numbers are recorded
together**:

- the **first crossing** — the smallest |Δ| at which the seed's mean gain
  crosses zero, by linear interpolation in dB between the bracketing grid
  points. **Primary.**
- the **sustained crossing** — the smallest |Δ| beyond which the mean gain is
  negative at *every* larger grid point in that direction.
- the **number of sign changes** along the curve.

The first-crossing estimator is **biased low** wherever the curve wobbles near
zero, and that bias points in the direction that makes R3 easier to satisfy. The
sustained crossing is recorded precisely so that the size of that bias is
visible rather than argued about. Any seed with more than one sign change is
listed individually in the record; if more than a quarter of the seeds re-cross
at a given (arm, level, direction), |Δ|\* is reported as **unreliable** there
and R3's verdict at that cell is *undecided* rather than satisfied.

**Censoring.** A seed whose gain never crosses within ±4.0 eV is censored, and
**enters the order statistics at its censoring bound (> 4.0 eV); it is never
dropped.** Dropping it would select on the outcome and bias |Δ|\* down. The
summary is the median across seeds with the inter-seed range and the censored
count, reported as a **point value only when the censored count c ≤ 9** (for
n = 20); otherwise as "> 4.0 eV". Alongside it, the **Kaplan–Meier median**
treating non-crossing as right-censoring at 4.0 eV, with a Greenwood-based
interval, is reported as the censoring-aware estimator.

**|Δ|\* is not computed where the mean gain at Δ = 0 is already negative.** The
reference record has this architecture *losing* several dB at the lowest noise
level even fully in-distribution: the network's own reconstruction error there
exceeds the noise it removes. At such a level a "zero crossing" is not a
boundary, and the record will say `not defined at this level` rather than report
Δ = 0.

### M3 — argmax displacement in eV (secondary, confirmatory)

For each spectrum: `energy[argmax(denoised)] − energy[argmax(clean reference)]`,
signed, on the fixed 0.069412 eV grid.

**Primary form: `disp(Δ) − disp(0)`, computed within a seed before averaging.**
The raw `disp(Δ)` is reported alongside. The bias correction is not optional and
the reason is measured: a learning-free Gaussian smoother applied to these
spectra produces a **Δ-independent positive offset** of +0.06 / +0.21 / +0.48 eV
at σ = 0.5 / 1.0 / 2.0 eV — flat to ±0.004 eV across the whole sweep, and
present on the clean spectra too, so it is the asymmetry of the three-peak
envelope and not the noise. At σ ≈ 1 eV that offset is **three grid steps**. An
uncorrected M3 would therefore need a 0.21 eV displacement at Δ = +1.0 merely to
show the right *sign*, while at Δ = −1.0 the same offset would help — an
asymmetry between directions with nothing to do with the boundary. M1 already
has this device under the name *degradation*; M3 gets it too.

**Three comparators, all registered now:**

1. the **noisy input** — measured at design time to give ≈0.00 eV at every Δ;
2. a **learning-free Gaussian smoother** of matched effective width on the same
   noisy inputs — the flat offsets above. Any Δ-dependence in a trained model's
   M3 is therefore not attributable to envelope asymmetry, truncation,
   background or plain oversmoothing, because none of those produces one;
3. **arm B on the identical test arrays** — already trained, so free. If the
   displacement is a learned position prior, arm B's should be near zero inside
   ±1.5 and appear outside it; a structural window effect would give arms A and
   B the same profile.

Registered limits: a grid argmax, not a fitted peak position; quantised to
0.069412 eV; describes only the dominant peak; supports no claim about fitted
binding energies.

## Self-checks that void the record

The script refuses to write a record if any of these fails, **each of which has
a stated failure condition**. A failure here means the measurement is broken,
and a broken measurement is not reinterpreted as a finding.

1. **Parameter count** — matches the value read from the reference record.
2. **Grid invariance** — the energy axis is bit-identical at every Δ and in
   every arm.
3. **Rigidity of the test sweep** — at every Δ, each nominal peak centre equals
   its **literal** unshifted value (284.8 / 286.3 / 288.5) plus Δ exactly, and
   all nominal inter-peak spacings are unchanged.
4. **Rigidity of the training pools** — for a random 5 % of **every** pool,
   including arm A's and arms C's and D's, the realised per-peak centres
   recovered from the generator's own draws satisfy
   `mu_k = mu_k^literal + jitter_k + Δ_i` with **a single Δ_i common to all
   three peaks**, checked as `max_k(mu_k − mu_k^literal − jitter_k) −
   min_k(...) == 0` exactly. All four pools are asserted to have been produced
   by the same shifted-`PeakSet` construction function as the test sweep, with
   `position_jitter == 0.3` in every arm. Additionally, at the end of the run
   `PEAK_SETS['C1s_adventitious']` is compared against the literals; a
   difference voids the run.
5. **Truncation, total and per peak** — the clean peak area retained inside the
   window at every Δ is within **1 %** of its value at Δ = 0, and **each peak's**
   retained area is within **2 %** of its own Δ = 0 value. Both are written into
   the record per Δ. The per-peak figures matter because truncation is
   **≈5× asymmetric** between directions at ±3.0 (the dominant peak loses
   0.64 % at −3.0; the weakest loses 0.75 % at +3.0), pointing the same way as
   an asymmetric boundary would. **Any claimed directional asymmetry in the
   result must be shown to exceed the asymmetry recorded here.**
6. **Input-SNR invariance** — the mean SNR of the *noisy input* against its own
   reference varies, across the full 25-point Δ sweep, by no more than
   **0.2 dB at levels 100 and 1000** and **0.35 dB at level 10000**. Measured at
   design time over all Δ with six independent draws of 512 spectra: maxima
   0.099, 0.102 and **0.126** dB respectively, with per-level SD ≈ 0.022, 0.023
   and 0.033 dB. The level-10000 allowance is wider because the noise is not
   paired across Δ there (corr ≈ 0.18 against ≈ 0.98), so the statistic carries
   a Monte-Carlo component of its own; the tolerance sits above the measured
   maximum by ≈7 SD there and ≈4 SD at the other two levels.
7. **Translation equivariance** — at every Δ,
   `max|ref(Δ) − translate(ref(0), Δ)| / max(ref(0)) ≤ 0.01`, against the
   bandlimited translate, written per Δ into the record. This is the check that
   catches a background or normalisation term that fails to move with the peaks;
   checks 3 and 4 inspect *centres* and would pass on a spectrum that is not a
   translate at all.
8. **Pairing integrity** — for every sample index *i*, the recorded per-peak
   (intensity, jitter, width) tuples are bit-identical between arms, and
   identical across all 25 Δ within a test family. The number of tuples compared
   is written into the record. This is the check that fails if the arms'
   RNG streams desynchronise — which they will if one arm consumes a shift draw
   and another does not, silently breaking the pairing that R4, R5 and R6 rest
   on while every other check still passes.
9. **Argmax well-posedness and reference identity** — at every Δ, the fraction
   of clean references whose argmax lies within 0.5 eV of **284.8 + Δ** is
   ≥ 99 %. The array inspected is **the identical object passed to M1 as the
   reference**, asserted by identity, not a re-derived copy. This is what
   establishes that M1 scores against the Δ-matched truth.
10. **Augmentation actually happened** — arm A's, C's and D's drawn shifts are
    all exactly 0.0; arm B's lie in (−1.5, +1.5) with a mean within `5σ/√n` of
    zero, n being the pool size the check is applied to, which is stated in the
    record.
11. **Noise-model identity** — the realised `NoiseConfig` at each level is
    compared field by field against the literals in the *Held fixed* table; any
    difference voids the run.
12. **Leakage** — no test spectrum is byte-identical to any training spectrum,
    in any arm, and the train/test stream bases are asserted disjoint.

### Diagnostics recorded, never asserted

These have no failure condition and do not gate the record. They are listed
separately so that the count of actual gates is not overstated.

- Run-to-run determinism: arm A is trained twice on identical inputs at one
  seed and the scores compared.
- The nearest-neighbour statistic accompanying check 12.
- Every arm's realised shift distribution.
- The per-Δ values from checks 5, 6 and 7.

### A suspect-run rule

If **R2 and R7 both fail while R1 passes**, the run is treated as *suspect* and
the reference wiring is re-verified before any record is written. That pattern
is what a reference mix-up would produce — R1 is insensitive to it, because the
shifted and unshifted references coincide at Δ = 0 — and without this rule the
"publish failed predictions" policy below would publish a false negative result.

## Registered predictions

**Every prediction below is evaluated at level 1000.0 only** (input SNR ≈ 16.4 dB,
read at run time from the reference record). The other two levels are reported in
full and read descriptively; **no prediction is registered about them**, and R1 is
expected to *fail* at level 100.0 at this training-set size — the reference record
has this architecture several dB negative there — which is not a failure of the
apparatus.

Every prediction is **directional**: it names its direction in advance, so sign
rules are read one-sided and the exact one-sided binomial *p* is reported for
each. A single threshold is applied throughout rather than chosen per
prediction: **≥ 15 of 20 seeds, one-sided p = 0.0207**, except the positive
control at ≥ 19/20. Where a *t*-test is also stated it is a one-sample test on
the twenty per-seed values, Holm-adjusted within the family named in that row;
conjoining a sign rule with a *t*-test is strictly more stringent than either
alone, and R2's form is the template the others follow.

| | Prediction | Decision rule | Family |
|---|---|---|---|
| **R1** | *Positive control.* Arm A gains at Δ = 0. | Mean > 0 **and** positive in ≥ 19/20 seeds. If this fails the apparatus is broken and nothing else is read. | none |
| **R2** | *There is a cliff.* Arm A's gain is negative at Δ = +4.0 and at Δ = −4.0. | Mean < 0, negative in ≥ 15/20, and *p* < 0.05 Holm-adjusted across the two directions. | the two directions |
| **R3** | *The cliff is narrow.* Arm A's \|Δ\|\* is ≤ **1.5 eV** in at least one direction. | Median ≤ 1.5 with ≤ 1 censored seed in that direction, and M2 not flagged unreliable there. | none |
| **R4** | *The shift is the cause, not the difficulty.* At \|Δ\| = 1.5, arm B gains more than arm A. | Paired within seed, B > A in ≥ 15/20, both directions, *p* Holm-adjusted across the two. | the two directions |
| **R5a** | *Density alone costs.* At Δ = 0, arm A > arm C > arm D. | Each ordering paired within seed in ≥ 15/20. | the two orderings |
| **R5b** | *Augmentation costs beyond density.* At Δ = 0, arm B ≤ arm **D**. | Paired within seed, B ≤ D in ≥ 15/20, plus a paired mean difference with its CI. | none |
| **R6** | *The boundary moves rather than vanishing.* Arm B's \|Δ\|\* exceeds arm A's where arm A has one. | Paired within seed in ≥ 15/20; a seed censored in B but not A counts as "larger", a seed censored in both is a tie and is excluded with the count printed and the rule restated at the reduced n. **Second part is three-way** — see below. | none |
| **R7** | *Displacement opposite in sign to Δ.* For arm A at \|Δ\| ≥ 1.0, the **bias-corrected** mean displacement `disp(Δ) − disp(0)` has the opposite sign to Δ and grows with \|Δ\|. | Correct sign in ≥ 15/20 seeds at each of Δ ∈ {±1.0, ±4.0}, *p* Holm-adjusted across the four; and monotone over {1.0, 1.5, 2.0, 3.0, 4.0} within each direction, allowing violations no larger than **one grid step, 0.069412 eV**. | the four sign tests |

**R5b is compared against arm D, not arm C, on purpose.** Arm D is the *lower*
density bound, so it is the weaker competitor and "arm B ≤ arm D" is the
**stricter** of the two available tests. If R5b fails, the record states that no
augmentation cost beyond the density penalty was demonstrated — an undecided
verdict, not a finding that augmentation is free.

**R6's second part is three-way and pre-declared**, because "no crossing within
the tested range" is an *uninformative* outcome, not a falsification, and
labelling it as one would be a category error:

- **moved** — arm B has a crossing inside ±4.0 eV. The record may then say the
  boundary was relocated.
- **beyond range** — it does not. The record states that arm B's boundary lies
  beyond 4.0 eV and makes **no** claim that augmentation relocates rather than
  removes the boundary. This is an undecided verdict and is recorded as such.
- **not defined** — where the Δ = 0 gain is already negative.

**R3 is the prediction most likely to be wrong, and it is the one that matters.**
A 1.5 eV calibration offset or charging shift is entirely ordinary in XPS. If
the boundary turns out to sit at 2.5 eV the registered prediction is *false* and
the record says so; if it sits at 0.5 eV the warning in the README has to be
much louder than anything currently written there.

### Descriptive-only rule

The record will contain **4 arms × 3 levels × 25 Δ × 2 metrics ≈ 600 cells**,
plus the derived degradation and paired-difference series. The registered
predictions consume roughly twenty of them. Declaring level 1000.0 primary
collapses only the level axis.

Therefore: **only the cells named in R1–R7, at level 1000.0, carry an
inferential claim.** Every other cell — all other Δ, all other levels, all other
arms, and both derived series — is **descriptive, is reported without a
*p*-value, and no statement of the form "gain dips at Δ = x" or "the curve is
asymmetric at x" may be made about a cell not named above.** A feature seen
there is a candidate for a new preregistration, not a finding of this one.

**One exception, stated rather than taken.** *(Added on 2026-09-23, after the
results existed, and not logged at the time; recorded as Revision 8.)* A descriptive cell may be cited where
it **bounds the interpretation of a registered claim** — that is, where reading a
registered result without it would mislead. Such a citation asserts no direction,
no magnitude, no mechanism and no generalisation; it names the arm and level it
comes from; and it is accompanied either by a named follow-up registration or by
an explicit statement that none is committed. Anything beyond that is a finding
and needs its own registration. This exception is written here because the first
write-up invented it at the point of use, which is how a self-imposed rule stops
binding.

### A consistency anchor, not a prediction

Arm A is the reference benchmark's own primary condition for this architecture,
re-drawn with this script's seeding. The script **reads** that record's values
at run time from
`aggregates['suggested-hyperparameters']['ResNet-FCNN'][level]['snr_gain_db_mean']`
and its across-seed SD, writes both the reference value and the difference into
this record, and **flags** any level differing by more than **three of the
reference record's across-seed SDs**. No reference number is typed into this
document or into the script — §6 forbids restating a result in a second place,
and the reference README is explicit that a hand-typed number there has no
provenance.

A flag is not a failure: the draws differ, the seed count differs, and the
per-sample generation path differs. It is a prompt to explain the difference in
the record before publishing anything. The expectation is agreement within the
spread, since only the seed count and the stream base differ.

## What this record will not support

Written before the numbers exist, so that it cannot be trimmed to fit them.

- **Nothing about measured spectra.** Every spectrum here is synthetic, from
  this package's own generator, scored against a reference that exists only
  because the data is synthetic. The training noise and the test noise come
  from the same function, so the model's noise model is exactly correct by
  construction — a condition measured data never satisfies.
- **No general threshold for XPS denoising.** Whatever |Δ|\* comes out is a
  property of *this* peak set, *this* jitter width, *this* architecture, *this*
  training-set size and *this* noise model. One point was measured in each.
- **Nothing about other training-set sizes, for any claim including R5.** Arms
  C and D bound the density penalty at one architecture and one recipe. (R5b
  failed on this: arms C and D do not bound the cost of augmentation. See
  Revision 4, item 41, and the Record; this pointer was added in Revision 10.)
- **Nothing about other augmentation widths.** One width (±1.5 eV) is tested, so
  R6 is a statement about that width, not about augmentation in general.
- **The background does not move with the peaks.** `linear_background` returns
  `level + slope * (x - x[0])` — a ramp pinned to the **window**, not to the
  absolute energy axis, but either way it does not travel with the peaks, so a
  peak moving along it sits on a different background level. This manipulation
  is therefore *"peaks shift under a stationary background"*, not the
  full-spectrum translate a real charging shift produces. Self-check 7 bounds
  the departure at 1 % of peak height; measured, it reaches 0.46 % at the edge
  of the sweep. It is bounded, not removed. (The mechanism stated in Revision 1
  was wrong; see Revision 3.)
- **Window-edge effects are controlled only inside ±1.5 eV.** There, arm B *is*
  an edge-proximity control: it has seen those edge distances in training, so if
  R4 holds, edge proximity is excluded as the cause at 1.5 eV and therefore for
  R3's threshold. Beyond ±1.5 eV no arm has the corresponding edge proximity
  in-distribution, and degradation there is **not separable from window-edge
  effects**.
- **Nothing about non-rigid shifts.** Chemical shifts move components relative
  to each other. This manipulation cannot speak to them.
- **Nothing about optimisation budget.** All arms get 50 epochs and one
  schedule. Arm B has a harder problem, so a deficit it shows may be
  under-training rather than augmentation cost; this design does not separate
  them. The reference README flags the analogous confound for architectures.
- **Nothing about peak areas, widths, or fitted positions surviving denoising.**
  M3 is a grid argmax and is not a peak fit. Physically meaningful quantities
  have to be verified downstream (`AGENTS.md` §5).
- **No recommendation that augmentation is the right mitigation.** R5 and R6 are
  designed to show its price and its edge. Measuring a mitigation is not
  endorsing it.
- **No mechanism is claimed for M3.** Opposite-signed displacement is
  *consistent with* a learned position prior; the observable underdetermines it,
  and comparator (iii) is what separates it from a generic positional bias. §6
  forbids promoting an empirical regularity to a law.
- **Device dependence, stated precisely.** R1, R2, R4, R5 and R7 are sign and
  ordering claims. **R3 and R6 compare a recorded magnitude — |Δ|\*, in eV —
  against a fixed threshold, and are therefore subject to the same device
  caveat as any magnitude in this record.** Floating-point reduction order
  differs between CPU, MPS and CUDA backends; no cross-device comparison is
  recorded unless one is run, and none is asserted.
- **A denoised spectrum is a model estimate, not a measurement.** A high gain
  says the output agrees with a known synthetic reference. It does not
  establish that structure in the output is real.

## If a prediction fails

It is recorded as failed, with the number that failed it, and the record is
published anyway. Negative and undecided results stay (`AGENTS.md` §6). A failed
prediction is not converted into a different prediction that the data satisfies;
if a follow-up design suggests itself, it is a *new* preregistration with a new
date.

A failed **self-check** is different: it voids the run. The record is not
written, the cause is fixed, and the run is repeated. The **suspect-run rule**
above is a third case: a failure *pattern* that indicates broken wiring rather
than a negative result.

## Release and audit

- The design registered here is an independent-audit item under `AGENTS.md` §8
  on three counts: it adds a published scientific claim, it adds an evaluation
  metric (M3), and it states the conditions of a published measurement.
- **This document went to two independent audits before implementation**, and
  Revision 1 is their result. Both asked P1's question — can a wrong
  implementation pass? — and both answered yes by different routes.
- The completed record goes to a further independent audit before any number
  from it is quoted in the README or anywhere else outward-facing.
- Nothing from this record is written into the README, the package
  documentation, or any release note until that audit is done.
- Artefacts, following `benchmarks/reference/`: a script, a JSON record that is
  the only place numbers live, a renderer, a generated report that is never
  edited by hand, and a README stating what the record does not support. All
  under `benchmarks/boundaries/position_shift/`, which is inside the existing
  distribution boundary — `benchmarks/` is already denied by the `build` job,
  so no boundary moves and §3 is untouched.

## Revision log

### Revision 1 — 2026-09-22, after two independent audits, before implementation

Registered text preserved at commit `06fa8c0`. Audit A was adversarial
("can a wrong measurement pass, can a correct one fail?"); audit B was design
and statistics. Findings are itemised with the audit that raised them; where
both raised the same hole by different routes, both are named.

**Blocking — a wrong measurement would have passed**

1. **Nothing verified that arm B was shifted *rigidly*.** (A) An implementer
   using `GeneratorConfig(position_jitter=1.8)` — which
   `generate_single` applies **independently per peak** — would have trained arm
   B on non-rigid shifts, the manipulation this document puts explicitly out of
   scope, and **all nine self-checks and all seven predictions would still have
   passed**: check 10 inspected a logged scalar, and the rigidity check was
   scoped to the test sweep only. → new **self-check 4**, operating on realised
   training spectra in every arm.
2. **The rigidity baseline could be a tautology.** (A) `get_peak_set` returns
   the **shared module-level object**; `p.mu += delta` in a loop both
   accumulates and permanently corrupts `PEAK_SETS` for the process. A check
   that re-read its baseline through `get_peak_set` would compare a mutated
   baseline against a mutated result. → baselines are now the **literals**
   284.8 / 286.3 / 288.5, stated in the *Held fixed* table; fresh `PeakParams`
   required; end-of-run `PEAK_SETS` comparison added.
3. **The arms' RNG streams could desynchronise undetected.** (B) If one arm
   consumes a shift draw where another does not, every per-peak jitter,
   intensity and width differs and the pairing R4/R5/R6 rest on is silently
   broken — while nothing in the list looks at stream synchronisation. → new
   **self-check 8, pairing integrity**.
4. **The noise model was named, not specified.** (A) The reference's pin is two
   settings, the boolean is **False** at level 10000, and the package documents
   the `min_rate=0.0` floor as "not a setting to choose for new work". An
   implementer writing `use_gaussian_approx=True` throughout would change the
   noise model with no check noticing. → literal config in the table, a
   **Noise model** section stating the adoption as a decision, new
   **self-check 11**.

**Blocking — a correct measurement would have failed, or a number was wrong**

5. **Self-check 9 (was 8) was ambiguous, and one reading voids every run.** (A)
   "the nominal dominant peak position" was unqualified. Verified: the fraction
   within 0.5 eV of **284.8** is **0.0000** at Δ = ±3, of **284.8 + Δ** is
   **1.0000**. → disambiguated to `284.8 + Δ`, and an **identity assertion**
   against M1's reference array added, making it the check that establishes M1
   scores the Δ-matched truth.
6. **"retains more than 99.5 % of its peak area inside the window" was wrong.**
   (A) That is the **ratio to Δ = 0**; the **absolute** retention is
   **98.43 % at Δ = 0** and **97.95 % at −3.0**, because the η = 0.3 tails put
   ≈1.6 % outside the window at every Δ. Re-measured independently and
   confirmed. → the *Manipulated* section now states both, with a table.
7. **The input-SNR tolerance was set from the wrong support.** (A) The
   design-time figure was measured over **three** Δ values while the tolerance
   governs **twenty-five**, and the level-10000 branch is not paired across Δ.
   Re-measured over the full sweep with six independent draws: maxima
   **0.099 / 0.102 / 0.126 dB**, SD 0.022 / 0.023 / 0.033. → tolerance split
   **0.2 dB at levels 100 and 1000, 0.35 dB at level 10000**, with the mechanism
   and the measured noise-correlations (+0.977 / +0.976 / **+0.177**) recorded.
8. **R5 was confounded with training density and would have passed for the
   wrong reason.** (B) Arm B has **1/5** of arm A's per-peak marginal position
   density at equal N, and 1/16 in the joint space. The repository's own
   training-set-size record extrapolates to **≈3.4 dB of deficit from density
   alone** against a 0.321 dB seed SD — d<sub>z</sub> ≈ 10.6, so R5 would have
   passed at 20/20 carrying no information. → **arms C (N = 461) and D
   (N = 144)** added, 3.4 minutes of training; R5 split into **R5a** (density
   ordering) and **R5b** (augmentation cost beyond density, tested against the
   stricter arm D).
9. **n = 8 could not support the rules written on it.** (B) At n = 8 the
   two-sided tail for ≥7/8 is **0.0703** and for ≥6/8 is **0.2891**, so only R1's
   8/8 rule met α = 0.05. All eighty training runs cost **≈30 minutes**. →
   **n = 20**, a single uniform directional threshold of **≥15/20**
   (one-sided p = 0.0207), ≥19/20 for the positive control, with exact binomial
   *p* reported per prediction.

**Should-fix, folded into the same revision**

10. **M3 carried a large signed bias.** (A) Verified by applying a learning-free
    Gaussian smoother: a **Δ-independent** offset of +0.06 / +0.21 / +0.48 eV at
    σ = 0.5 / 1.0 / 2.0 eV, flat to ±0.004 eV and present on clean spectra. At
    σ ≈ 1 eV that is **three grid steps**, and it helps at Δ < 0 while hurting at
    Δ > 0. The registered comparator (the noisy input) reads ≈0.00 eV always and
    discriminates nothing. → M3's primary form is now **bias-corrected**,
    `disp(Δ) − disp(0)` within seed, with **three** comparators including the
    smoother and arm B. Audit A also **ruled out** all four confounding
    mechanisms the brief suspected — truncation, background, envelope asymmetry,
    oversmoothing — because each gives a flat offset, not an opposite-sign
    signature: R7's logic survives, its estimator did not.
11. **R7's tolerance was set by a quantity that would not exist until after the
    data.** (B) "violations no larger than the across-seed SD" is an automatic
    pass if the SD comes out large. → fixed at **one grid step, 0.069412 eV**,
    and the sign half is now per-seed rather than evaluated on the mean.
12. **R7's title asserted a mechanism the observable underdetermines.** (A)
    → renamed to state what is measured; mechanism moved to prose as
    *consistent with*, per §6's rule against promoting a regularity to a law.
13. **|Δ|\* was silent on re-crossing and biased toward R3.** (A, B) → first
    crossing (primary) plus **sustained crossing** and **sign-change count**;
    unreliable-and-undecided rule; censored seeds **enter at their bound, never
    dropped**; point-value-only-when-c ≤ 9 rule; **Kaplan–Meier** median added.
14. **R6's first half had no decision rule and its second half was not
    falsifiable.** (A, B) "Larger" was unquantified, and with arm B's envelope at
    ±1.8 eV the implied boundary sits at ≈3.0 eV — the old cap — making "no
    crossing" a coin flip decided by where the range stopped. → paired
    within-seed rule with an explicit tie convention; **three-way verdict**
    (moved / beyond range / not defined) with "beyond range" recorded as
    **undecided, not falsified**; sweep extended to **±4.0 eV**, which audit A
    showed is still inside the registered 1 % truncation tolerance (99.06 % at
    −4.0, crossing at |Δ| ≈ 4.1).
15. **No prediction named its noise level, and R1 was false across levels.**
    (A, B) → all predictions scoped to level 1000.0, with R1's expected failure
    at level 100.0 stated in advance.
16. **Multiplicity of the displayed grid was uncontrolled.** (B) ≈600 cells
    against ≈20 registered. → **descriptive-only rule** added, and each
    prediction now names its Holm family or "none".
17. **Reference numbers were typed by hand.** (A) §6: do not restate a result in
    a second place; the reference README: a hand-typed number there "has no
    provenance and should be treated as withdrawn". → the script **reads** them
    at run time and the values are gone from this document.
18. **The consistency anchor's 3 dB flag was ≈9 SD and would pass a broken
    pool.** (A) → flag at **three of the reference record's across-seed SDs**.
19. **The self-check list overstated how many gates existed.** (A) Determinism,
    the nearest-neighbour statistic and "written into the record" clauses cannot
    fail. → split into **twelve voiding checks** and a **diagnostics** list.
20. **What moves with Δ besides the peaks was uncontrolled.** (B) The background
    does not move with the peaks — by the mechanism corrected in Revision 3,
    not the one stated here; the normalisation
    constant drifts monotonically 0.9967 → 1.0023 across the sweep; edge
    proximity in FWHM is 5.00 at Δ = 0 and 2.86 at +3.0; per-peak truncation is
    **5× asymmetric**. The input-SNR check sees the aggregate and is blind to
    all of it. → new **self-check 7** (translation equivariance), per-peak
    truncation added to **self-check 5**, and three scope statements added,
    including that any claimed directional asymmetry must exceed the recorded
    truncation asymmetry.
21. **An unmeasured recommendation sat inside the not-supported list.** (B)
    "calibrating the instrument remains the better answer to a calibration
    error" is a comparative claim with no data. → **deleted**.
22. **The device caveat contradicted R3.** (B) "the predictions are about signs,
    orderings and thresholds" — but |Δ|\* against 1.5 eV **is** a magnitude.
    → rewritten to name which predictions are sign claims and which are
    magnitude claims.
23. **Missing scope items.** (B) training-set size for R5, augmentation width,
    background stationarity, edge effects beyond ±1.5, optimisation budget.
    → all five added.
24. **Self-check 10's tolerances were typo-catchers, one with a spurious-void
    risk.** (B) "mean within 0.1 eV of zero" is 3.2 SE at n = 768 and would void
    ≈3 % of correct runs if applied per level. → expressed as **5σ/√n** with the
    n stated.
25. **The leakage check is blind at Δ ≠ 0.** (A) A collided spectrum is shifted
    and no longer byte-identical. → the Δ = 0 test set is always in the hash
    set and stream-base disjointness is asserted arithmetically.
26. **512 test spectra were justified by citation, not argument.** (B) For M1
    they buy precision on a quantity that is not the error bar; for **M2** they
    control a one-sided bias that seeds cannot average away. → justification
    rewritten on M2 grounds.
27. **Self-check 5's tolerance was justified by assertion.** (B) "below anything
    that could explain a sign change in gain" had no number. → softened, and the
    per-peak figures that do the work are now stated.
28. **No rule existed for a failure *pattern* indicating broken wiring.** (A)
    A reference mix-up fails R2 and R7 while R1 passes, and the publish-failures
    policy would have published a false negative. → **suspect-run rule** added.
29. **A strength of the design was left implicit.** (B) Arm B is an
    edge-proximity control inside ±1.5 eV. → stated, and its limit — that it
    does not extend beyond ±1.5 — stated with it.

### Revision 2 — 2026-09-22, forced by implementation, before any result existed

Three registered items did not survive contact with the code. All three were found
by the smoke test, which runs two seeds at two epochs on tiny pools: **no
gain-versus-shift, boundary or displacement value had been computed when these
were written**, and the run that produced the record was started afterwards.

30. **Self-check 4's exactness form was not implementable.** It registered
    `max_k(mu_k − mu_k^literal − jitter_k) − min_k(...) == 0` *exactly*. But
    `(284.8 + delta) - 284.8` is not `delta` in binary floating point for an
    arbitrary delta, and the augmented arm's shifts are arbitrary: the check
    failed on a **correct** pool at the first smoke test, reporting
    `-1.0006910861176266 != -1.0006910861175995`. The implemented form is
    `centre_k == literal_k + delta`, evaluated as that expression — exact,
    testing the same property (one shift common to every peak), and still failing
    on a pool built with per-peak jitter, where each peak would carry its own
    offset. Inter-peak spacing preservation, which the registered text also
    required exactly, is likewise a floating-point residual and is now bounded at
    1e-9 eV and recorded rather than asserted equal.

31. **Self-check 8 was ambiguous about what "sample index" means, and vacuous as
    written.** Two faults, both surfaced by the smoke test:

    - *Ambiguity.* Arms C and D hold fewer samples per noise level than arms A
      and B, so equal **flat** indices point at different levels. The check
      failed on correct pools, reporting arm C's sample 154 against arm A's.
      Arms are now keyed on `(level_index, sample_index)`, which is what
      "the same per-sample seed sequence, and for C and D how many samples are
      kept" meant.
    - *Vacuity.* As registered, the check compared replayed per-peak draws
      between arms — but when the batch seeds match, that compares a replay
      against itself and passes whatever the generator did. A **new self-check
      8b, replay faithfulness**, rebuilds a sample of spectra from the replayed
      draws alone, with every random variation switched off, and requires them to
      be **bit-identical** to what the generator produced. Without it, check 8
      verified a model of the generator against itself. This is the same class of
      defect the audits found elsewhere in this document, found here by running
      the code.

32. **Self-check 6's tolerance was calibrated at one test-set size and applied at
    all of them.** The registered tolerances (0.2 / 0.2 / 0.35 dB) were measured
    at `n_test = 512`, where most of the statistic's spread is Monte-Carlo rather
    than signal — the deterministic signal-power variation is about 0.03 dB
    against a measured span near 0.10 dB. At the smoke test's `n_test = 64` the
    span reached 0.357 dB and voided a correct run. The allowance is now scaled
    by `sqrt(512 / n_test)` and **equals the registered value exactly at the
    registered size**, so the registered design point is unchanged and only
    off-design runs are affected.

    Re-measured at the registered size over the full **25-point** sweep with six
    independent draws, before the run: worst spans **0.104 / 0.107 / 0.107 dB**
    against tolerances 0.2 / 0.2 / 0.35 dB. Revision 1's figures were measured
    over 21 points, before the sweep was extended to ±4.0 eV.

Two further implementation notes, recorded but not revisions, because they change
nothing the document fixed: a negative-zero key collision (`-1.0 * 0.0` formats as
`-0.00` and keyed the origin twice) was an ordinary coding bug; and the
learning-free smoother comparator, computed on this run's own spectra, reproduced
the design-time offsets quoted in Revision 1 (+0.051 / +0.201 / +0.467 eV at
sigma = 0.5 / 1.0 / 2.0 eV against +0.06 / +0.21 / +0.48), which is a check on
that figure rather than a change to it.

### Revision 3 — 2026-09-22, from the first full run's own record

The first full run completed, all twelve self-checks reporting pass. Inspecting
the record before writing it up showed that **one of the twelve could not fail**.

33. **Self-check 7 was a tautology.** It compared the shifted spectrum against
    "the unshifted spectrum evaluated on a grid displaced by −Δ". Both sides
    expand to the same expression — the peak term to `f(e_i − Δ − mu_k)` and the
    background to a function of the grid index alone — so the residual was
    **exactly 0.000e+00 at all 25 shifts**, and would have been whatever the
    generator did. A gate that cannot fail was being counted among twelve that
    void the record, which makes the record's description of itself false. This
    is precisely the defect class the two audits were commissioned to find, and
    it was introduced *while implementing one of their findings*.

    Replaced with a comparison against `np.roll` of the unshifted spectrum at
    shifts that are **integer multiples of the energy step**, over the window
    interior so nothing wraps. That comparison is independent of the generator.
    The registered shift values are not grid-aligned, and the alternatives —
    interpolating, or re-evaluating the generator on a displaced grid — either
    carry more error than the effect or reproduce the tautology.

    Measured with the replacement: the residual is **real and grows with |Δ|**,
    reaching **0.46 % of peak height at +4.03 eV** and 0.40 % at −4.03 eV,
    inside the registered 1 % tolerance. So the tolerance was right and the
    design is unaffected; what was wrong was that nothing had checked it.

34. **Revision 1 stated the wrong mechanism for that residual, and this document
    and the record repeated it.** Revision 1 said, following audit B, that "the
    background is evaluated on the fixed absolute energy axis and does not move
    with the peaks". The source says otherwise:

    ```python
    def linear_background(x, level=0.1, slope=0.0):
        return level + slope * (x - x[0])
    ```

    It is a ramp pinned to the **window**, not to the absolute energy axis. The
    *conclusion* stands — the ramp does not travel with the peaks, so a peak
    moving along it sits on a different background level — and the magnitude
    follows from the slope directly: `slope × Δ = 0.001 × 4.0 = 0.4 %` of peak
    height, which is what the replacement check measures. Audit B's number was
    right; its explanation was not, and neither was this document's. Corrected
    in the claim-scope list, in `benchmarks/boundaries/position_shift/README.md`
    and in the script's `claim_scope`.

**The first full run's record is discarded and the measurement re-run**, from a
tree whose self-check 7 is the replacement. Its numbers were not wrong — the
seven-of-eight prediction outcome and every boundary value are unaffected by a
check that returned a constant — but a record that advertises twelve voiding
self-checks while one of them is inert is making a false statement about itself,
and that is the one thing this directory exists to prevent. Discarding it costs
31 minutes.

### Revision 4 — 2026-09-22, after two independent audits of the completed record

The completed record went to two independent audits, as this document required. Between
them they found **eight blocking defects**. Six are in the apparatus and are the same
failure as Revision 3, repeated: **a check that compares an expression against itself**.
Two are statements in the artefacts that the measurement itself refuted.

The first run's record is **discarded** and the measurement re-run from a repaired tree.

**The apparatus did not verify the study's independent variable**

35. **Self-check 4 was a tautology, on the check Revision 1 created to prevent exactly
    this.** It read the recorded shift, rebuilt a peak set from it, and compared that
    against the same expression. It never touched `pool["clean"]`. Audit A built the
    precise wrong pool Revision 1 names — arm B generated with `position_jitter=1.8`,
    the per-peak (non-rigid) manipulation this study puts out of scope — and it passed
    checks 4, 8, 8b and 10 while differing from the correct pool by 0.93 in spectrum
    units. I reproduced this before repairing it.

    Check 4 now **reconstructs each sampled training spectrum** from the literal peaks,
    the recorded rigid shift and the per-peak draws replayed from the generator's RNG at
    the pinned jitter, and requires it **bit-identical to the spectrum in the pool**.
    Verified to fail on three wrong pools — non-rigid jitter; a shift recorded but never
    applied; a shift applied at the wrong jitter width — and to pass the correct one. It
    subsumes the old check 8b, which compared a reconstruction against a regeneration
    rather than against the pool.

36. **Self-check 8's across-shift half never used the shift.** `delta` appeared only in
    the error message; the body recomputed the same expression 25 times and compared it
    with itself. It contributed 3600 of the 12327 tuples the record advertised. Replaced
    by **check 8b, test-family rigidity**, which reconstructs test spectra from the
    shift-independent family seed and requires bit-identity — so it fails if a family
    was drawn from a different seed, if the sweep is not rigid, or if the jitter moved.
    Check 8's remaining half — that the arms share their per-sample seeds — is real and
    kept, with its vacuous replay-against-itself comparison removed and its scope stated.

37. **Self-check 9's identity assertion was `clean is clean`.** The caller passed the
    same variable twice, so it could not fail, while the record advertised
    `reference_identity_asserted: true` and this document called it *"what establishes
    that M1 scores against the Δ-matched truth"*. `snr_db` now records the reference it
    was given, and check 9 asserts against that — and runs **after** the metric, not
    before.

38. **Self-check 10 passed on an arm B that was never augmented.** It tested min, max
    and mean, all of which an all-zero shift vector satisfies, under the title
    "Augmentation actually happened". It now also requires the realised SD to match the
    uniform SD within 5 SE and all ten deciles of the range to be occupied.

39. **Self-check 12 hashed arm A only**, while the registered text said "in any arm".
    It now hashes all four.

40. **The renderer's guard waved through the verdicts.** Audit A tamper-tested it field
    by field: **25 of 30 edits accepted**, including flipping a prediction's PASS to
    FAIL, rewriting a boundary median, and changing a Holm-adjusted *p* from 1.3e-44 to
    0.9 — while the docstring and the README both claimed everything was recomputed.
    This is the defect the reference benchmark's own record test exists for, repeated
    here without the test.

    `verify` now rebuilds the entire `boundaries` and `predictions` trees from `runs`
    and deep-diffs them. The audit's tamper set now **refuses 23 of 23**. And
    `tests/test_position_shift_boundary_record.py` — 25 cases — performs that tamper
    test in CI, so the claim is under test rather than asserted. The guard's two halves
    are now described separately: the aggregates are recomputed independently, the trees
    are a consistency check against the measurement module's own functions, and the
    self-check figures are **not** covered and the report says so.

**Statements the measurement refuted, or that were never true**

41. **`claim_scope` asserted what R5b disproved.** It said *"arms C and D bound the
    density penalty"* — and `render_report.py` printed it verbatim into `report.md`, so
    the artefact designated as the source of truth carried a statement its own data had
    refuted. Corrected, and three items the result newly requires were added: that the
    boundary is a single-noise-level quantity, that |Δ|\* is not determined by the
    training range (arms C and D share arm A's range exactly and have nearer
    boundaries), and that arm B's plateau is one width at one level in two quantities.

42. **The README promoted a four-point regularity to a law.** *"What generalises is the
    shape — a model is valid just past the position range it was trained on, and not
    beyond"* is contradicted twice by this record: at the highest noise level **all four
    arms are censored**, and arms C and D have **arm A's position range exactly** with
    boundaries 22 % and 35 % nearer. §6 forbids exactly this. Replaced.

43. **The Kaplan–Meier docstring claimed a coincidence the record disproves.** With
    administrative censoring beyond every event, KM reduces to the order-statistic
    estimator only **up to the even-n midpoint convention**: KM returns t_(10) and
    `np.median` the midpoint of t_(10) and t_(11). They differ in every defined cell, by
    about 5e-4 eV. Neither estimator is wrong; the claimed coincidence was. The
    convention is now recorded beside the number.

44. **R6's boolean would have rendered an undecided verdict as FAIL.** The registered
    text is explicit that "beyond range" is undecided and that calling it a falsification
    "would be a category error" — but `passed` is boolean and the renderer printed FAIL
    from it. R6 now carries a three-valued `verdict` and the renderer prints
    **UNDECIDED**. Both directions came out "moved", so this never bit.

45. **Self-check 3 was described as more than it is.** It compares
    `shifted_peak_set(delta)`'s centres against the literals plus Δ — a run-time unit
    test of that constructor and of the registry staying unmutated, not an inspection of
    any generated spectrum. Kept, with its scope stated; check 8b is what inspects the
    sweep's data.

46. **The gate count was wrong.** Thirteen, not twelve: 8b was added in Revision 2 and
    never folded into the count that this document, the README and the record all
    printed.

47. **R4's and R7's Holm resolution was undocumented.** R2's registered rule requires
    *p* < 0.05 Holm-adjusted; R4's and R7's say only "*p* Holm-adjusted across the
    two/four". The implementation resolved this as **reported, not required**, and
    computed `passed` from the sign test alone. That is a defensible reading but it was
    a decision made at implementation time that no revision recorded. It is recorded
    here, and it is **immaterial to this run**: R4's Holm *p* are 1.6e-35 and 3.2e-33,
    R7's 6.0e-26 to 3.7e-36, all far below 0.05, so requiring them changes no verdict.

**Why the run is discarded rather than the checks simply fixed**

The numbers were almost certainly right: none of the repairs touches data generation,
the models, or the metrics, and the previous two runs already reproduced each other
bit-for-bit. But *"all self-checks passed"* was the claim doing the work, and for arms
B, C and D it was false — nothing inspected their training data at all. A record whose
independent variable is unverified cannot be quoted, and re-running is how that stops
being an argument and becomes a checked fact. If a repaired check fails on the re-run,
the numbers were never quotable and this is how we find out.

### Revision 5 — 2026-09-23, after a review of the plan to repair Revision 4's write-up

A review of the repair plan found that the plan itself overclaimed in two places,
and a re-inspection before acting found that the Revision 4 write-up had not done
everything it said. Nothing here changes a registered prediction or the measurement,
and the record was not re-run.

48. **Revision 4's write-up claimed audit findings were reflected that were not.**
    Three renderer changes an audit recommended — reporting comparator (iii), labelling
    the smoother comparator with its noise level, and printing p-values that are not all
    `0.0000` — had not been made, and 48 of the 110 decimals quoted in the Record
    section were printed nowhere any guard reached. All three renderer changes are made.

49. **The gates had only ever been shown to pass.** `tests/test_position_shift_
    boundary_gates.py` now gives every self-check a correct input it must accept and a
    named wrong input it must reject, with the rejection pinned to its intended reason:
    run first with a bare `pytest.raises`, the pins were then shifted one test out of
    place by an editing error and 16 tests failed for the wrong reason, which is exactly
    what the pins exist to expose. Check 6 is inline in `run()` and cannot be tested this
    way; that is stated as a limit. No test found a defect affecting the measurement.

50. **"Put the numbers in `report.md` and the guard covers them" was proposed and
    rejected.** The renderer's own docstring already said the self-check figures, the
    anchor and the diagnostics are unverified, and that the trees are checked against
    the measurement module's own functions — so the guarantee would not have transferred.
    Nor would mere presence of a value have been enough: the same value can sit in a
    different arm, level or shift than the sentence says. Instead every number in the
    Record section now carries a per-occurrence source anchor, resolved against the
    record by `tests/test_p2a_record_citations.py`, which is itself tested against
    planted errors. Run for the first time, it found two further transcription errors
    in the Revision 4 text: an across-seed SD given as 0.012 that is 0.011, and a
    boundary said to be 35 % nearer that is 34 %.

51. **Two provenance limits are now stated in the Record section rather than left
    implicit**: that no registration commit was published before any run, so the
    ordering of registration and result is attested only by a history its author
    controls; and that the record's `design.preregistration.commits` omits Revision 4
    — the same hand-maintained-constant defect the previous write-up disclosed for
    Revision 3, repeated. Neither is repaired by editing history.

52. **This document's opening Status line said "Not implemented; not run"** for a day
    after the record existed. Corrected.


### Revision 6 — 2026-09-23, before publication: a developer-specific path

53. **The record carried the developer's home directory.** `consistency_anchor.
    read_from` was written with `str(path)`. `AGENTS.md` §10 forbids developer-specific
    absolute paths in tracked files, and `tests/test_paths.py` enforced that over the
    package's `.py` files only — a guard narrower than its rule, which is the defect
    class this document has spent five revisions on. It was found by a review of all
    the owner's projects, not by this repository.

54. **Removed from history, not only from the tree.** A push publishes every commit in
    the range, so a fix at the tip would have left the path in the two unpublished
    commits that carried it. Those commits were rewritten before anything was pushed:
    one field, in two records, replaced with the repository-relative path; nothing
    else changed. Commit hashes from the first record onward changed and every
    reference to them was remapped. The Record section states the edit where it
    matters — against the claim that the record is the script's unchanged output.

55. **The widened guard caught itself.** `tests/test_tracked_paths.py` scans every
    tracked file, reads its tokens from `test_paths.py` so the two cannot drift, and
    was shown to fail on the record before the fix. Committed, it then failed on its
    own two test lines, which used the token as a literal while its docstring said it
    carried none. Fixed before the rewrite was final, so no published commit carries
    either version.


### Revision 7 — 2026-09-23, the record regenerated in a pinned environment

56. **The environment was recorded but not enforced.** The runs were made in a conda
    base shared with other projects, with only lower bounds in `pyproject.toml`; any
    install for another project could have moved the next run onto different versions.
    `uv.lock`, `.python-version` and `[tool.uv]` constraints now pin Python 3.12.11,
    torch 2.9.1, numpy 2.3.3, scipy 1.16.3 and h5py 3.14.0 — the versions the record
    reports — for the development and benchmark environment only; `pip install` users
    still see lower bounds.

57. **Provenance is now recorded by the run.** The hand-typed list of registration
    commits is gone. The run reads the code commit, the working-tree state and this
    document's version from git, and refuses a full run from a dirty tree. The gate
    tests show it refusing a modified file and an untracked file.

58. **The record was regenerated, once, as the acceptance test.** Run in the pinned
    environment on macOS 27.0, it reproduced the third run in every per-run value,
    aggregate, boundary, prediction and diagnostic. So the record now published is the
    unedited output of its script, and the one-field edit of Revision 6 survives only
    in the superseded records in history, where it is disclosed.

59. **A false statement about history, corrected.** The Record section said the first
    two runs survive only in git history. The first was never committed.


### Revision 8 — 2026-09-23, after an independent review of the claims, before a fifth run

An independent review of the Record section, and of the record behind it, found that
the record's own self-check 4 did not run as registered, that a rule used in the Record
section had been written after the results existed without being logged, and that the
record's `claim_scope` — which the measurement script writes into every record — asserted
two things the design cannot support. The owner decided to correct the script and
regenerate the record once more rather than disclose and leave them.

60. **Self-check 4 sampled 12 spectra per arm per seed, not 5 % of every pool.** The
    registration says 5 %, and the first implementation did that. Revision 4 rewrote the
    check to inspect the pools themselves and, in doing so, took the `n_samples` form of
    the old check 8b it replaced — a flat 12, which is 0.5 % of arms A and B. No revision
    recorded the change; the claims review found it by comparing the design text with
    the code. The check is restored to the registered 5 %.

    **Post-hoc evidence about the pools, gathered before deciding.** Every training pool
    is deterministic in its seed, so the pools the fourth run trained on were
    regenerated and first anchored to statistics that run recorded over every sample —
    check 10's shift distributions and check 12's near-duplicate statistics, which
    matched exactly for all 20 seeds — and then check 4 was run on **all** of them:
    104,260 spectra rebuilt and compared bit-for-bit, all passing.
    `benchmarks/boundaries/position_shift/verify_pools_full.py --commit 8474c94`
    reproduces this. The anchor is direct for arms A and B and indirect for C and D,
    which share arm A's seed sequence and code path. That is strong evidence the pools
    were right; it is not the registered gate, which is why the run is repeated with the
    gate as registered.

61. **The descriptive-only rule's exception was added after the results existed.** It
    entered this document's design section on 2026-09-23, in the commit that published
    the third run's record, and no revision recorded it. The Record section's citation
    of a level-10000 cell relies on it. Recorded here, and marked where it stands.

62. **`claim_scope` asserted two things the design cannot.** It said training-set size
    "moves the boundary at fixed range", a causal claim from arms C and D whose
    boundaries no prediction names; and it referred to "Arm B's plateau", a shape word
    for cells no prediction names. Both are reworded in the script, so the regenerated
    record carries the correction as its own output rather than as an edit.

63. **The review's remaining findings are about the Record section's own text** — a
    mechanism stated for M3 against the record's own disclaimer, an unmeasured
    comparison with what a practitioner would tolerate, a generalisation made under an
    exception that forbids one, two false statements about the record, an internal
    contradiction about the number of runs, shape words, and quantities quoted below
    their standard error. They are corrected when this section is rewritten from the
    regenerated record, and listed as each is corrected.

64. **The regenerated record will not be assumed to equal the fourth.** It is compared
    with it field by field, and every difference is reported.

65. **The fifth run, compared with the fourth field by field.** Every per-run value,
    aggregate, boundary, prediction, diagnostic, the consistency anchor, the design
    record and the environment are identical. The differences are exactly the intended
    ones and nothing else: check 4's sample size in every arm and seed (116, 116, 24 and
    8 spectra — arm D's fell from 12, which had exceeded the registered 5 %), the added
    `registered_fraction` field, `train_seconds`, the two reworded `claim_scope` items,
    the provenance of the new commit, and the wall clock.

66. **The Record section is rewritten from the fifth record, and the claims review's
    findings are addressed as follows.** The mechanism stated for M3 is removed; −Δ is
    kept only as a reference value, defined as what the displacement would be if the
    denoised peak did not move. The comparison with what a practitioner would tolerate
    is removed, and the sentence now describes this model against its own training
    jitter. The level-10000 citation no longer states a general law about SNR gains and
    says that its exception was written post hoc. Two false statements are corrected:
    extending the sweep to ±4.0 eV is not what made R6 decidable, and R7's tolerance was
    evaluated at the registered float64 step, not the realised float32 one. The
    contradiction about the number of runs is gone. "At a cost", "plateau", "cliff of its
    own" and "steep" are gone. R7 and comparator (iii) are quoted to the precision their
    across-seed SD supports, in both directions, with the SD. Arm B's boundary is quoted
    to two significant figures with its spread. Arms C and D are cited in both
    directions, as descriptive, with no follow-up committed. A paragraph now says that
    only the third run was made before its own results were known, and what changed in
    the design section since. Smaller corrections: check 8b inspects test data; check
    10's figure names its seed and range; the sign convention of R5b's d<sub>z</sub> is
    stated; "close to 3.4 dB" became "2.7 SDs above it"; the `train_seconds` correction is
    attributed to the audits that found it; the noisiest level is named as such; the
    MPS backend is named beside the magnitudes it produced.


### Revision 9 — 2026-09-23, a structural change to the record, before a sixth and last run

A second independent review of the rewritten Record section found nothing blocking and
a set of corrections, two of which sat inside the record rather than in this document.
The cause was structural, and so is the repair.

67. **The record's `claim_scope` held interpretation, so every wording review meant
    re-measuring.** `claim_scope` is written into the record by the measurement script.
    Revisions 4 and 8 added to it statements that depend on the result — how R5b's
    failure should be read, what the other two noise levels showed, what arms C and D's
    boundaries imply, what arm B's gain looked like inside its training range. Each time
    a review asked for one of those to be worded differently, the only faithful ways to
    change it were to edit the record by hand or to run the measurement again.

    **Before:** `claim_scope` mixed the design's limits with four result-dependent
    items. **After:** it states only what the design cannot support, as could have been
    written before any result — one pool size per arm; predictions registered at one
    noise level; no relation between |Δ|\* and any single property of the training
    distribution; one augmentation width — and a new `result_dependent_interpretation`
    entry tells a reader of the JSON alone where the result-dependent cautions are: the
    Record section and `report.md`, each with its source. Nothing measured changes.

68. **A hand-typed assertion is removed from the record.**
    `design.preregistration.predictions_fixed_before_implementation: true` was typed
    into the script, and the run could not verify it — the class of field Revision 7
    removed elsewhere. What it asserted is stated in the Record section with the commits
    that establish it.

69. **After this run, prose corrections no longer require a measurement.** Everything
    in the record is either measured, derived from what was measured, or fixed before
    the results. Interpretation lives in this document and in `report.md`, whose unit
    conversions (bins and the ratio to the dominant peak's nominal FWHM, added at an
    external review's suggestion) are computed from the record at render time.

70. **The review's other findings concern this document's own text** and are corrected
    when the Record section is updated from the sixth record, which is compared with the
    fifth field by field.

71. **The sixth run, compared with the fifth field by field.** Every per-run value,
    aggregate, boundary, prediction, diagnostic, the consistency anchor and the
    environment are identical. The differences are exactly the intended ones and nothing
    else: the four `claim_scope` items, the added `result_dependent_interpretation`
    entry, the removed `predictions_fixed_before_implementation` field, the provenance of
    the new commit (code commit, script sha256, and the registration's last commit,
    commit count and sha256), `train_seconds` in every arm and seed, the wall clock and
    the generation time.

72. **The Record section is rewritten from the sixth record, and the second review's
    findings are addressed as follows.** Two false statements are corrected: M1 does
    penalise a misplaced peak, so the text no longer says it does not encode position,
    and says instead that a positive M1 gain does not establish where the output's peak
    sits; and the third run, not only the fourth, sampled 12 spectra in check 4. The
    claim that every gate is shown to fail now excepts check 6. The headline table
    names its level, its pairing and what ± means, and quotes gains at 0.1 dB. What the
    other two levels showed, formerly in `claim_scope`, is stated here. R5b's section no
    longer says which step of the design failed: the untested reading is removed, and
    the two readings the design cannot separate are stated side by side. Arms C and D's
    boundaries are quoted as values in eV, not as percentages of arm A's. R7 is compared
    with its across-seed SD, and the standard-error multiples and the percentage are
    gone. Comparator (ii) names its σ column and adds the widest one at level 10000. The
    sentence on charging and adventitious-carbon referencing, a statement about practice
    with no source here, is replaced by the shape of shift the axis stands for. The
    provenance paragraph says what was fixed at Revision 1 and before implementation,
    with the commits, and that the Revision 6 rewrite cannot be detected from the
    history. The wall-clock variation is no longer attributed to throttling, which was
    not measured. A Units paragraph states eV as the unit measured and quoted, with bins
    and the ratio to the dominant peak's nominal FWHM given as conversions for this one
    setup, not tested elsewhere; `report.md`'s boundary table carries the same columns.


### Revision 10 — 2026-09-24, after the sixth run, from a review of the history before publication

A review of every commit that publication would expose, made with a fixed checklist,
found no personal data and no boundary moved. It found statements that earlier entries
of this log, the README and the report state as fact and that nothing later corrects.
Nothing measured changes, and no prediction or decision rule changes.

73. **Errors in this log, corrected here rather than in place.** Revision 4 says "The
    first run's record is discarded"; it was the second run's, the first having been
    discarded under Revision 3. Revision 4, item 47, gives R7's Holm-adjusted *p* as
    "6.0e-26 to 3.7e-36"; the record's range is 3.7e-36 to 7.7e-24, the largest at
    Δ = −1.0 eV. Every value is still far below 0.05, so the item's conclusion stands.
    The design section's scope list still says that arms C and D bound the density
    penalty — the statement Revision 4, item 41, corrected in the record's `claim_scope`
    but not here. A pointer to R5b's result now sits beside it; nothing else in the
    design section changed.

74. **Other text brought into line with the Record section.** The README kept three
    statements the Record section had withdrawn: that the wall clock varied because of
    thermal throttling, which was not measured; that R5b "refutes the premise" of a
    matched handicap, followed by an untested reading of why; and that the
    training-set-size record shows what a change in *density* is worth, when it varies
    N. Each now says what the record supports. `report.md` no longer states on its own
    authority that the predictions were fixed before implementation; it points to the
    Record section, which names the commits. The sentence on arm A's gain table no
    longer argues in bold from the Δ = 0.5 eV cell, which no prediction names. The
    docstring of `tests/test_tracked_paths.py` said the path sat in three unpublished
    commits; it was two, as item 54 says.

75. **Errors in earlier commit messages, which stay in the published history.** The
    unpublished history is published as it stands, by the owner's decision taken before
    publication, so a commit message cannot be corrected in place. Those that state as fact what the records do
    not support are listed here, each with where the correct statement is.
    - `4c5ebca` says the wall clocks differed "because the machine throttled";
      throttling was not measured (item 72). It says R5b failed "because cutting N
      removes information about noise, intensity and width" and that arms C and D "are
      an N control, not a density control"; that is an untested reading, demoted in the
      next commit and removed in item 72. It says the denoised peak at a 4 eV shift sits
      "where the network was trained to expect one", a mechanism removed in item 66, and
      that extending the sweep to ±4.0 eV "is what made that decidable", which item 66
      corrects.
    - `05d3be7` gives the boundary's spread as "+/-0.012 eV"; the across-seed SDs are
      those under *Two significant figures, and why* in the Record section. It says the
      two directions split "9/20 each way"; the split the record gives is under *The two
      directions are not separated*, and the same sentence in the Record section was
      changed in Revision 5 without being listed there. It says arm B's displacement is
      "under a twentieth of a grid step inside its training range"; set against the
      grid step under *Units*, `report.md`'s comparator (iii) table shows that this holds
      in both directions only out to |Δ| = 1.0 eV, and that at ±1.5 eV, the edge of that
      range, the displacement is about a third of a step.
    - `4c5ebca` and `05d3be7` say "four volts" where four electronvolts is meant. Each
      commits a record carrying the one-field edit Revision 6 describes, without saying
      so; `b09b5d1` is the first commit that does.
    - `cc88d14` says a local ref "still holds the pre-rewrite P2-A record"; that was true
      when written, and the ref was deleted before publication.


## Record

Run 2026-09-23 in the environment pinned by `uv.lock`, from a clean working tree at the
commit the record names under `provenance` — the sixth full run, made under Revision 9.
20 seeds, 4 arms, 25 shifts, 3 noise levels, on the MPS backend;
81.1<!--r:run.minutes--> min. The record is
`benchmarks/boundaries/position_shift/results/position_shift_boundary.json`; the
rendered report is `report.md` beside it. Every per-run value agrees across the five runs
that were committed (see the last section).

**All thirteen voiding self-checks passed.** Four of them inspect the training data of
every arm, each by a different route: check 4 rebuilds 5 %<!--n:reg--> of every pool —
116, 116, 24 and 8 spectra per seed for arms A to D — from the literal peaks, the
recorded shift and the replayed draws, and requires them bit-identical to the pool;
check 8 compares every sample's seed across arms; check 10 tests every sample's shift
(arm B's realised SD was 0.8777<!--r:chk10.sd--> at seed 0 and between
0.855<!--r:chk10.sd.min--> and 0.886<!--r:chk10.sd.max--> across seeds, against a uniform
0.8660<!--r:chk10.sigma-->, with all ten deciles occupied); check 12 hashes every arm.
Check 8b does the same for the test data, rebuilding test spectra from each family's
seed. The suspect-run rule did not fire. The consistency anchor did not flag at any
level.

**This record's provenance is recorded by the run, not typed.** `provenance` names the
code commit, that the working tree was clean, the script's and the lockfile's sha256,
that the interpreter was a virtual environment, and this document's own version — its
first registration commit and its last commit before the run. Since Revision 9 the
record holds only what was measured, what is derived from it, and what was fixed before
the results; the result-dependent cautions are here and in `report.md`.

**Every gate but one is shown to fail, not only to pass.**
`tests/test_position_shift_boundary_gates.py` gives every self-check except check 6 a
pair — the correct input accepted, a named wrong input rejected, with the rejection
pinned to the reason the gate is for. **Check 6 is written inline in `run()` and has no
independent test**; it is exercised only by full runs. Rejecting one constructed failure
does not prove a gate catches every failure. None of these tests found a defect
affecting the measurement.

**How the numbers here are checked.** Every number below carries an anchor in the
source (`<!--r:…-->` or `<!--n:…-->`, invisible when rendered). An `r:` anchor names
the metric, condition, unit and derivation in
`benchmarks/boundaries/position_shift/record_citations.py`, and
`tests/test_p2a_record_citations.py` recomputes it from the record and requires the
quoted text to match at the precision quoted. An `n:` anchor marks a number that is
deliberately not from this record and says why. The record owns every `r:` number; if
they ever disagree, this section is wrong.

**Units.** Shifts and boundaries are measured and quoted in eV. Where a boundary is
also given in bins and as a multiple of the dominant peak's nominal FWHM, those are
conversions of the same number for this record's one setup — a grid of
0.069427<!--r:grid.step--> eV per bin, and the generator's nominal FWHM of
1.2<!--r:fwhm.nominal--> eV for the 284.8<!--n:reg--> eV peak, which each synthetic spectrum varies
by up to ±10 %<!--n:reg--> and which is narrower than the three-peak envelope. **Whether
any of these units carries over to another grid or line width was not tested.**

### Result — seven of eight predictions held; R5b failed

At level 1000 (the primary level); R4 and R5 are paired within seed; every ± is the
standard deviation across the twenty seeds.

| | Verdict | The number that decided it |
|---|---|---|
| **R1** positive control | **PASS** | arm A gains **+11.4<!--r:A.gain.0--> ± 0.2<!--r:A.gain.0.sd--> dB** at Δ = 0; 20/20 seeds |
| **R2** there is a cliff | **PASS** | **−17.1<!--r:R2.+4-->** and **−17.3<!--r:R2.-4--> dB** at Δ = ±4.0<!--n:reg-->; 20/20 each |
| **R3** the cliff is narrow | **PASS** | \|Δ\|\* ≈ **0.47<!--r:R3.pos--> eV**, against a registered threshold of 1.5<!--n:reg--> |
| **R4** the shift is the cause | **PASS** | arm B gains more than arm A by **+21.8<!--r:R4.+1.5--> / +22.8<!--r:R4.-1.5--> dB** at \|Δ\| = 1.5<!--n:reg-->; 20/20 |
| **R5a** density alone costs | **PASS** | A over C by **+4.3<!--r:R5a.AC--> dB**, C over D by **+1.4<!--r:R5a.CD--> dB**; 20/20 each |
| **R5b** augmentation costs beyond density | **FAIL** | arm B gains more than arm D by **+4.5<!--r:R5b.BD--> dB**; **0/20** seeds |
| **R6** the boundary moves | **PASS**, verdict **moved** | arm B's \|Δ\|\* ≈ **1.8<!--r:R6.pos--> eV**, inside the tested range; 20/20 |
| **R7** displacement opposite to Δ | **PASS** | **−4.01<!--r:R7.+4--> ± 0.06<!--r:R7.+4.sd--> / +3.90<!--r:R7.-4--> ± 0.07<!--r:R7.-4.sd--> eV** at Δ = ±4.0<!--n:reg-->; monotone, no violations |

**The other two levels, descriptive.** At level 100, R1 fails there as the design
anticipated: arm A's gain at Δ = 0 is −5.4<!--r:L100.A.gain.0--> ±
0.7<!--r:L100.A.gain.0.sd--> dB, so this model never helps at that level and no boundary
is defined. At level 10000, no arm's gain crosses zero anywhere inside ±4.0<!--n:reg-->
eV — the smallest mean gain over all four arms and all shifts is
+2.0<!--r:L10k.min.gain--> dB. So "there is a cliff" and the boundary below are
statements about level 1000; at one other level the model never helps, and at the other
there is no crossing inside the range tested.

### R3 was right in direction and badly wrong in magnitude

The registered text called R3 "the prediction most likely to be wrong, and the one
that matters", set the threshold at 1.5<!--n:reg--> eV, and said that if the boundary
sat near 0.5<!--n:reg--> eV the README's warning would have to be much louder than
anything written there. That is the branch the data took: **the boundary is about
0.47<!--r:R3.neg--> eV in both directions** — on this grid about
6.8<!--r:R3.bins--> bins, and about 0.39<!--r:R3.fwhm--> times the dominant peak's nominal
FWHM.

**Two significant figures, and why.** The across-seed SD of the boundary is
0.011<!--r:R3.sd--> eV in the positive direction and 0.013<!--r:R3.sd.neg--> eV in the
negative, and re-deriving each seed's crossing under other interpolants on the same
grid moves the positive-direction median by a further 0.002<!--r:R3.interp.span--> eV
— linear 0.4707<!--r:R3.linear-->, cubic spline 0.4698<!--r:R3.spline-->, PCHIP
0.4686<!--r:R3.pchip--> — a systematic span that seeds cannot reduce. A finer grid
between 0.25<!--n:reg--> and 0.75<!--n:reg--> eV would tighten it and is a separate
measurement.

**The two directions are not separated.** Paired within seed, negative minus positive
is +0.0007<!--r:R3.dir.mean--> ± 0.0175<!--r:R3.dir.sd--> eV; the negative-direction
crossing is the farther in 9<!--r:R3.dir.n.neg.farther--> seeds and the nearer in
11<!--r:R3.dir.n.neg.nearer-->. The across-seed range over both directions is
0.447<!--r:R3.range.lo-->–0.489<!--r:R3.range.hi--> eV. Self-check 5 records a
per-peak truncation asymmetry that any directional claim would have to clear first;
none is made.

The gain against shift, arm A at level 1000, dB:

| Δ (eV) | 0 | 0.25 | 0.50 | 0.75 | 1.00 | 2.00 | 4.00 | <!--n:reg-row-->
|---|---|---|---|---|---|---|---|
| gain (dB) | +11.4<!--r:A.gain.0--> | +6.8<!--r:A.gain.+0.25--> | −0.9<!--r:A.gain.+0.50--> | −7.0<!--r:A.gain.+0.75--> | −10.7<!--r:A.gain.+1.00--> | −15.0<!--r:A.gain.+2.00--> | −17.1<!--r:A.gain.+4.00--> |

**The boundary sits below half an electronvolt.** The cells other than Δ = 0 and
Δ = 4.0<!--n:reg--> are named in no prediction; they are shown and not argued from.
`report.md` gives every cell with its SD.

What the record establishes is about **this** model: trained on this package's synthetic
spectra with ±0.3<!--n:reg--> eV of per-peak jitter (±4.3<!--r:jitter.bins--> bins) and no
rigid shift, it stops helping at a rigid shift of about 0.47<!--r:R3.pos--> eV. A rigid
shift of the whole envelope is the shape of a sample-charging offset or a binding-energy
calibration error, which is why this axis was chosen; **no measured spectrum appears
anywhere in this record**, and nothing here compares the boundary with the shifts met in
practice.

### R5b failed in the opposite direction, and the design cannot say why

R5b predicted that arm B, trained across ±1.5<!--n:reg--> eV, would score **no better**
at Δ = 0 than arm D, a narrow arm cut to the same position density. Arm B scored
**+4.5<!--r:R5b.BD--> dB better**, in 20 of 20 seeds. The record stores the paired
difference as D − B, whose *d*<sub>z</sub> is −13.7<!--r:R5b.dz-->.

| at Δ = 0, level 1000 | gain (dB) | versus arm A |
|---|---|---|
| A narrow, N = 2304 | +11.4<!--r:A.gain.0--> | — |
| B augmented ±1.5<!--n:reg-->, N = 2304 | +10.1<!--r:B.gain.+0.00--> | −1.3<!--r:B.vsA--> |
| C narrow, N = 461 | +7.1<!--r:C.gain.0--> | −4.3<!--r:C.vsA--> |
| D narrow, N = 144 | +5.6<!--r:D.gain.0--> | −5.8<!--r:D.vsA--> |

*The A-over-C and C-over-D orderings are the registered R5a comparisons; **A minus B is
descriptive**, named in no prediction, and is shown rather than argued from.*

Arm C, cut to one fifth of arm A's N, lost 4.3<!--r:R5a.AC--> ± 0.4<!--r:R5a.AC.sd--> dB.
The design-time extrapolation from this repository's training-set-size record predicted
≈3.4<!--n:design--> dB; the measured loss is 2.7<!--r:R5a.AC.z--> of its own SDs above
that, so the extrapolation held in direction and order of magnitude and underestimated
the size.

**Either the density equivalence the design rested on does not hold, or augmentation has
an effect at Δ = 0 that the design did not anticipate.** No arm separates these, and
other readings — that the augmented pool is an easier optimisation problem, for example —
fit the same numbers. What follows under every reading is enough for the verdict:
**arms C and D cannot bound the cost of augmentation, in either direction.** Per the
registered fallback, *no augmentation cost beyond the density penalty was demonstrated —
an undecided verdict, not a finding that augmentation is free.* The cell that would
separate the readings is an augmented arm at reduced N, which is a new preregistration.

### R6 came out "moved", which was the falsifiable half

Arm B's boundary is about **1.8<!--r:R6.pos--> eV** — on this grid about
26<!--r:R6.bins--> bins, and about 1.5<!--r:R6.fwhm--> times the dominant peak's nominal
FWHM — inside the tested range, so the pre-declared verdict is *moved*, not *beyond
range*. Its across-seed SD is 0.014<!--r:R6.sd.pos--> eV in the positive direction and
0.022<!--r:R6.sd.neg--> eV in the negative. Training as arm B was trained — a rigid shift
drawn per spectrum from ±1.5<!--n:reg--> eV, at the same N — moved the boundary and, at
this width, did not remove it.

| Δ (eV) | 0 | 1.00 | 1.50 | 1.75 | 2.00 | 2.50 | 4.00 | <!--n:reg-row-->
|---|---|---|---|---|---|---|---|
| arm B gain (dB) | +10.1<!--r:B.gain.+0.00--> | +9.9<!--r:B.gain.+1.00--> | +7.8<!--r:B.gain.+1.50--> | +2.0<!--r:B.gain.+1.75--> | −4.0<!--r:B.gain.+2.00--> | −11.3<!--r:B.gain.+2.50--> | −15.9<!--r:B.gain.+4.00--> |

R4 names arm B at ±1.5<!--n:reg--> only, and R6 its crossing; the other cells are
descriptive. The sweep was extended from ±3.0<!--n:reg--> to ±4.0<!--n:reg--> eV in
Revision 1 against an expected boundary near 3.0<!--n:design--> eV; the measured
boundary lies well inside either cap, so the extension was not what made this verdict
decidable.

**No regularity is claimed across the arms.** R6 is not to be read as showing that the
training position range alone fixes |Δ|\*. Arms C and D share arm A's position range,
and their boundaries at level 1000 are reported for that reason — arm C
0.37<!--r:C.boundary.pos--> / 0.35<!--r:C.boundary.neg--> eV and arm D
0.31<!--r:D.boundary.pos--> / 0.30<!--r:D.boundary.neg--> eV, positive / negative,
descriptive. No prediction names them, and **no follow-up is committed**.

### R7 held, and it is the result a user should be most careful with

The bias-corrected displacement is **−4.01<!--r:R7.+4--> ± 0.06<!--r:R7.+4.sd--> eV at
Δ = +4.0<!--n:reg--> and +3.90<!--r:R7.-4--> ± 0.07<!--r:R7.-4.sd--> eV at
Δ = −4.0<!--n:reg-->**. Its reference value is −Δ, which is what the displacement would
be if the denoised peak did not move with the shift at all. In the +4.0<!--n:reg-->
direction the mean falls short of −Δ by 0.01<!--r:R7.+4.short--> eV, about
0.2<!--r:R7.+4.steps--> of a grid step and inside its across-seed SD of
0.06<!--r:R7.+4.sd-->; in the −4.0<!--n:reg--> direction by 0.10<!--r:R7.-4.short--> eV,
1.5<!--r:R7.-4.steps--> grid steps, more than its SD of 0.07<!--r:R7.-4.sd-->. **This
document declines to read that difference**: it is not a registered comparison, and
self-check 5 records a truncation asymmetry this record cannot separate from it. At
Δ = ±1.0<!--n:reg--> the displacement is already −0.67<!--r:R7.+1--> ±
0.03<!--r:R7.+1.sd--> and +0.63<!--r:R7.-1--> ± 0.04<!--r:R7.-1.sd--> eV — roughly
two-thirds of the shift in each direction. **No mechanism is claimed.** Opposite-signed
displacement is consistent with a learned position prior; the observable does not
decide it.

**Comparator (ii)**, the learning-free Gaussian smoother, spans
0.0058<!--r:M3.smoother.span.1000--> eV across all 25 shifts at level 1000 in its
σ = 1.0<!--n:reg--> eV column, so the offset that motivated the bias correction does not
depend on the shift there, and a shift-dependent displacement in a trained arm is not
attributable to truncation, background, envelope asymmetry or oversmoothing. **That
covers level 1000 only**: at level 10000 the same column spans
0.468<!--r:M3.smoother.span.10000--> eV and the widest, σ = 2.0<!--n:reg--> eV, spans
0.81<!--r:M3.smoother.span.10000.s2--> eV. It is a seed-0 diagnostic and is not covered by
`report.md`'s guard.

**Comparator (iii)**, arm B on the identical test arrays, was registered alongside the
smoother. Out to |Δ| = 1.25<!--n:reg--> in both directions arm B's bias-corrected
displacement stays within 0.005<!--r:B.disp.inner.maxabs--> eV of zero (SD at most
0.005<!--r:B.disp.inner.maxsd-->); it is −0.023<!--r:B.disp.+1.50--> and
+0.021<!--r:B.disp.-1.50--> at ±1.5<!--n:reg--> (SD 0.007<!--r:B.disp.sd.+1.5-->),
−0.28<!--r:B.disp.+2.00--> and +0.28<!--r:B.disp.-2.00--> at ±2.0<!--n:reg--> (SD at most
0.03<!--r:B.disp.sd.2.max-->), and −2.84<!--r:B.disp.+4.00--> ± 0.11<!--r:B.disp.sd.+4.0-->
and +2.71<!--r:B.disp.-4.00--> ± 0.28<!--r:B.disp.sd.-4.0--> at ±4.0<!--n:reg-->. The
registration stated **in advance** that a structural window effect would give arms A and
B the same profile. It does not. **That eliminates the alternative the registration
named, and no more.** (This comparator was omitted from the first write-up entirely; an
audit called that an incomplete record, correctly.)

### A descriptive cell cited under the exception to the descriptive-only rule

The exception was written after the results existed and was not logged at the time; see
Revision 8. The citation below keeps to its conditions and bounds how R1's and R4's
gains may be read.

Level 10000 is the noisiest level — its input SNR at Δ = 0 is
−3.6<!--r:L10k.in.0--> dB. In one cell there (arm A, Δ = +4.0<!--n:reg-->) M1 reads
+2.5<!--r:L10k.gain.+4--> dB while M3 reads −4.0<!--r:L10k.disp.+4--> eV; the input's own
SNR is −3.6<!--r:L10k.in.+4--> dB and the output's −1.1<!--r:L10k.out.+4--> dB. M1, a
mean-squared error against the Δ-matched reference, does penalise a misplaced peak — but
a smooth output with its peak in the wrong place can still beat an input whose noise
power exceeds its signal power. **A positive M1 gain does not establish that the
output's peak sits where the reference's does.** No follow-up measurement is committed
here.

### What this licenses, and nothing stronger

That **this** ResNet-FCNN, trained on **this** synthetic distribution at **this** jitter
width, size and noise model, on the MPS backend, loses all benefit at a rigid shift of
about 0.47<!--r:R3.pos--> eV at level 1000 — on this grid about 6.8<!--r:R3.bins--> bins,
about 0.39<!--r:R3.fwhm--> times the dominant peak's nominal FWHM; and that training with a
rigid shift drawn from ±1.5<!--n:reg--> eV moved that boundary to about
1.8<!--r:R6.pos--> eV. Arm B's gain at Δ = 0 differs from arm A's (descriptive,
−1.3<!--r:B.vsA--> dB), and this design cannot attribute that difference. The claim-scope
list in the record applies.

The consistency anchor licenses one thing beyond agreement: the Δ = 0 column here
**is** the reference benchmark's own primary condition for this architecture, agreeing
at every level inside the flag, so the boundary is measured from the operating point
that record describes. It licenses nothing about the other architectures there.

### Limits of provenance, stated rather than repaired

**The preregistration was never published before any run.** Every registration and
revision commit of this document was made locally and none was pushed before the
measurement. "Registered before the results existed" is attested only by this
repository's own history, whose timestamps its author controls. That cannot be
remedied now. Future registrations are to be published before their first run.

**What was fixed before the results, and what changed after.** R1–R7 and their decision
rules were fixed at Revision 1 (`ef25766`), before implementation began (`736e540`) and
before the first run. Revisions 2–10 changed self-checks, provenance, rendering, scope
wording and the record's structure, and recorded one decision-rule resolution made at
implementation time (item 47); none changed a prediction. Every run after the first was
made after earlier runs' numbers existed; the design section of this document differs
from Revision 4's only in its Status line, in the descriptive-only rule's exception,
added after the results and marked as such, and in one pointer to R5b's result added in
Revision 10. A reader who follows
`provenance.registration` lands on a version of this document that already contains
earlier runs' results; that is why.

**The earlier records' provenance defects are history.** The third run's record named
the wrong registration version, from a hand-maintained constant. It and the second run's
record carried a developer-specific absolute path in `consistency_anchor.read_from`,
removed before publication by rewriting the unpublished commits (Revision 6) — one field
in each, nothing else. That rewrite cannot be detected from the published history, whose
author and committer dates coincide, so "one field in each, nothing else" is attested
by the author. The third and fourth runs' self-check 4 sampled 12 spectra per arm per
seed instead of the registered 5 %<!--n:reg-->, about 0.5 %<!--n:derived-count--> of arms A
and B (Revision 8). None of these applies to this record, and all stay stated.

### Things worth recording about the runs themselves

**The measurement reproduced across six runs, and five of them can be checked from this
repository.** The second to sixth runs' records are committed, and every one of the
6000 per-run values, every aggregate and every boundary is identical across them, at
full precision. They differ in wall clock, in what the environment reports, and in the
fields later revisions added or removed. The fourth run moved from a shared conda base to
the pinned environment and from macOS 26.7<!--n:cross-record--> to 27.0<!--r:env.os-->; the
fifth ran check 4 at the registered 5 %<!--n:reg-->; the sixth wrote a `claim_scope` that
holds only what was fixed before the results. The wall clocks varied — 30.9<!--n:history-->,
162.7<!--n:cross-record-->, 34.7<!--n:cross-record-->, 65.2<!--n:cross-record-->,
129.5<!--n:cross-record--> and 81.1<!--r:run.minutes--> minutes.

What a reader **cannot** check is the first run. Its record was deleted before it was
ever committed, when the run was voided under Revision 3, so its agreement with the
others rests on a comparison made at the time.

**Self-check 6's margin is thinner than the pre-run estimate.** Revision 2 reported
worst spans of 0.104<!--n:design--> and 0.107<!--n:design--> dB over six draws; this
record's maxima over twenty seeds are 0.138<!--r:chk6.100.0-->,
0.160<!--r:chk6.1000.0--> and 0.202<!--r:chk6.10000.0--> dB, so level 1000 used
80 %<!--r:chk6.margin.pct--> of its 0.2<!--n:reg--> dB tolerance. Immaterial against the
gain swing, and better stated than left to a reader who diffs the two.

**The grid step.** R7's monotonicity tolerance was registered as one grid step,
0.069412<!--n:reg--> eV — span over 255 in float64 — and it was evaluated at that
registered value. The energy axis is float32, so the realised step is
0.069427<!--r:grid.step--> eV. R7 had no violations under either value.

**Kaplan–Meier.** The record stores the convention beside the number: KM returns
t<sub>(10)</sub> and `np.median` the midpoint of t<sub>(10)</sub> and t<sub>(11)</sub>,
which here differ by 5.3e-4<!--r:km.minus.median--> eV.

**A correction to an earlier write-up.** The second run's write-up gave that record's
arm A `train_seconds` range as 36.3<!--n:history--> to 364.1<!--n:history--> seconds;
its maximum was 482.9<!--n:history-->. The audits behind Revision 4 caught it.

Nothing from this record goes into the README, the package documentation or any
release note until a person has decided what may be published.
