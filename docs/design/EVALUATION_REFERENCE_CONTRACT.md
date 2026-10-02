# Design: what `evaluate` compares against, and what it may call the result

**Status: adopted 2026-10-01 (revision 4, after four independent audits); phases 1 and 2
implemented (see the CHANGELOG).** Step 1 of
the improvement plan that follows a reproducibility assessment of a published study. It
changes the meaning of an evaluation output, so under `AGENTS.md` §8 it was independently
audited before any code was written. The owner's adoption is recorded at the end.

## 1. The problem

`dnndenoiser evaluate` computes, per spectrum, an SNR in dB and an MSE of the noisy input and
of the denoised output against a dataset named `clean`, and reports the difference as a
gain. `compute_snr` is documented as truth-referenced. Nothing in a file says what `clean`
is:

- `generate` writes the noise-free spectra it drew from — the truth for that synthetic data;
- `SyntheticGenerator.save_hdf5` writes whatever arrays a caller passes as `clean`;
- for measured data there is no truth, and the reference most easily made is an estimate
  built from the data — for a frame stack, the mean of its frames;
- `infer` copies `clean` through with no record of its origin;
- if the evaluated file contains `clean`, `evaluate --clean other.h5` silently ignores the
  external reference.

Two constructed examples show why the name must follow the reference. Let the frames be
`X_i = s + e_i` and the reference the mean `X̄` of those same frames. A model that returns `X̄`
for every frame has zero discrepancy from the reference by construction, whatever `X̄`'s
distance from `s`; a model returning the same wrong spectrum `c` for every frame has
discrepancy `‖c − X̄‖²`, which can be large. So agreement with this reference neither
establishes nor excludes accuracy. And for a reference `R = s + η` and two estimates `X`, `Y`,

    E[‖X − R‖² − ‖Y − R‖²] = E[‖X − s‖² − ‖Y − s‖²] − 2 E⟨X − Y, η⟩ .

The difference of discrepancies equals the truth-referenced difference in expectation only
when the cross term vanishes. A sufficient condition is that, given the matched underlying
signal, `η` has zero mean and is independent of both `X` and `Y`. Independence alone is not
enough: with `X = s + 1`, `Y = s`, `R = s + 1 + Z` and `Z` independent and zero-mean, the
truth-referenced improvement is `+1` and the expected reference-based difference is `−1`.

The metadata cannot prove what a reference is. The aim is narrower: **a reference's origin
and conditions are declared and travel with the data; its relationship to the evaluated data
and to the model is declared separately, at evaluation; alignment is verified or explicitly
asserted; and the name of every reported quantity follows all three.**

## 2. What is declared

### 2.1 Reference origin — stored with the reference array

| `origin` | Meaning | Required fields |
|---|---|---|
| `synthetic_truth` | The noise-free signal synthetic noisy data were generated from | `generator` (name and version), `units` |
| `estimate` | An estimate of the underlying signal, however obtained | `construction`, `source`, `conditions`, `units`, `noise` |
| `undeclared` | Nothing is declared | — |

For `estimate`:

- `construction` — the operation: `frame_mean`, `leave_one_out_mean`, `smoothed`, `fitted`,
  `other` (with `other` requiring a description). The operation says nothing about which
  data it was applied to;
- `source` — which data it was computed from: an `acquisition_id` and the frame selection
  (`all`, a list or range of `frame_index` values, or `unrecorded`); no absolute paths or user
  names;
- `conditions` — the reference's own acquisition and calibration conditions (energy
  calibration, channel or angle, exposure normalisation, specimen state), each a value or
  `"unknown"`;
- `units` — see §5.3;
- `noise` — the reference's own statistics (e.g. frames averaged, exposure), or `"unknown"`.
  An invented estimate is worse than `"unknown"`.

### 2.2 Relationship — declared at evaluation, never stored with the reference

| Field | Values | Default |
|---|---|---|
| `overlap_with_evaluated` | `overlap`, `no_overlap_declared`, `unknown` | `unknown`, or `overlap` when established (below) |
| `used_in_model_development` | `yes`, `no_declared`, `unknown` | `unknown` |
| `signal_match` | a statement that the reference's `conditions` match the evaluated data's, or `"unknown"` | `"unknown"` |

**Overlap is established, not inferred from the construction.** It is `overlap` when the
reference's `source.acquisition_id` equals the evaluated data's `acquisition_id` (§3) and the
source frame selection is `all` or intersects the evaluated rows' `frame_index`. A declaration
contradicting an established overlap is rejected; anything else is accepted as declared. A mean
over a disjoint acquisition may be declared `no_overlap_declared`; that is a declaration, not a
proof of independence. Relationships are not copied by `infer`: the same reference can be
independent of one model and not of another. Until models carry provenance (step 2 of the
plan), every output carries `held_out_status: "unknown"`.

### 2.3 Scope of one declaration

One declaration applies to every row of the reference. The reader rejects metadata that
declares different origins for different rows, or conflicting declarations; it cannot detect
undeclared mixed provenance from the numbers, and does not claim to.

## 3. Storage and propagation

### 3.1 The reference declaration bundle

Two HDF5 attributes on the `clean` dataset, both required together:

- `reference_schema_version` — integer, 1 for this design;
- `reference_origin` — a JSON object as in §2.1 (`undeclared` is a valid, explicit value).

A third, optional attribute `reference_lineage` is a JSON array of transform records
`{"operation": …, "from_points": int, "to_points": int, "tool": "dnndenoiser <version>",
"step": "infer" | …}`, appended only when a transform is actually applied.

- **Neither bundle attribute present** → effective origin `undeclared` (legacy files).
- **Exactly one present, an unsupported version, invalid JSON, a missing required field, or a
  wrong type** → rejected as malformed, never read as `undeclared`.

### 3.2 Evaluated-data metadata

On `noisy` (or `frames`) and on `denoised`:

- `intensity_units` — one of `counts`, `counts_per_s`, `normalised_to_spectrum_max`,
  `normalised_to_global_max`, `generator_intensity`, or `other:<description>`;
- `acquisition_id` — optional, a string naming the acquisition the rows came from.

Existing datasets `energy`, `angles`, `times` and `frame_index` are the comparison and source
coordinates (§5).

### 3.3 Who writes and who preserves what

| Writer | Writes |
|---|---|
| `generate` (synthesises the data itself) | the bundle with `origin = synthetic_truth`, `generator`, and `units`; `intensity_units` on `noisy` equal to that `units` — `normalised_to_spectrum_max` for 1-D output, `normalised_to_global_max` for angle- or time-resolved output, `generator_intensity` under `--no-normalize` |
| `SyntheticGenerator.save_hdf5` (accepts caller-supplied arrays) | requires the declaration and the units as arguments and **rejects a call without them**, naming the argument |
| `infer` | copies the whole bundle and lineage unchanged; appends a lineage record when it resamples; copies `intensity_units` and `acquisition_id` from its input to the output `noisy`, and writes the same `intensity_units` on `denoised` (its inverse normalisation returns the input's units); copies `energy` (resampled if the input was), `angles`, `times` and `frame_index` |

A resampled `synthetic_truth` is the truth *for the resampled comparison*; it says nothing
about structure on the original grid.

## 4. Selecting the reference and resolving declarations

`evaluate` never writes to its input files. Declarations are resolved by **effective stored
origin**:

| Situation | Behaviour |
|---|---|
| `clean` in the evaluated file and `--clean` given | Rejected as ambiguous, unless `--reference {file,external}` selects one |
| Neither present | Rejected |
| Stored bundle absent **or** a valid explicit `undeclared`, `--reference-origin …` given with every required field | Accepted; the stored state and the effective declaration are both recorded, `declaration_source: "cli"` |
| Stored origin known (not `undeclared`), CLI declaration identical in origin and every field | Accepted; both recorded |
| Stored origin known, CLI declaration differs in origin or any field | Rejected, naming the selected reference and both declarations |
| Stored bundle malformed (§3.1) | Rejected |
| `--legacy-output` when the effective origin is not `undeclared`, or together with `--reference-origin` | Rejected (§6.5) |

## 5. Alignment before any metric

Three things are checked, and kept apart. **Detected metadata is always checked; an
assertion (§5.4) fills in only what is missing and never overrides a detected mismatch.**

**Applicability comes first.** An axis the data's layout does not have — angles or times for
plain 1-D spectra — is *not applicable*: it needs no metadata and no assertion, and is listed
under `alignment_not_applicable`. An axis the layout has (a non-energy dimension of the
arrays) is applicable; if its coordinates are missing, the check needs an assertion or the
evaluation is refused.

### 5.1 Comparison coordinates

- **Shape.** Equal shapes; or a single reference spectrum shared by every row, **only** with
  `--shared-reference` (no silent broadcasting).
- **Energy.** When both carry `energy`, equal within an absolute tolerance of 1e-6 of the
  energy unit.
- **Angles and times.** When both carry `angles` (or `times`), equal element by element
  within 1e-6, in order. A reversed axis is a mismatch.
- **Row correspondence** is established in one of three ways, recorded in the output:
  1. *same file* — the reference is the `clean` dataset of the evaluated file; positional
     correspondence is accepted under the file schema (not independent evidence that the
     arrays were written together);
  2. *identifiers* — identifiers that together cover **every** non-energy dimension of a
     row: `frame_index` within one acquisition namespace (§5.2) for a frame dimension,
     angle/time coordinates for those axes. A matching angle axis establishes angle
     correspondence only; it does not identify a sample or acquisition dimension, which then
     needs the assertion;
  3. *asserted* — `--assert-alignment rows`, positional correspondence declared by the user.
  Equal shapes alone establish structural compatibility, not observation identity; an
  external reference with no identifiers needs the assertion, a shared reference included
  (e.g. `evaluate -d denoised.h5 --clean ref.h5 --reference external --shared-reference
  --assert-alignment rows`).
- Rows may differ from one another (angles, times): matching is between corresponding rows,
  and flattening for aggregation (§6) happens only after it.

### 5.2 Source identifiers — used only within one acquisition namespace

`frame_index` is compared only when the reference's `source.acquisition_id` equals the
evaluated data's `acquisition_id`, and then only to check the declared correspondence (e.g.
row *i* of a leave-one-out reference was built for frame `frame_index[i]`). A reference from
another acquisition may carry unrelated indices; equal integers from different acquisitions
establish nothing.

### 5.3 Units

`intensity_units` on the evaluated arrays and `units` in the reference declaration must be
equal strings. A mismatch is rejected. A reference with effective origin `undeclared` has no
declared units, so the units check is missing (§5.4).

### 5.4 Missing metadata

When a check cannot be made because the metadata is absent, evaluation is rejected unless the
user asserts it: `--assert-alignment energy,units,rows,angles,times`. The output lists,
separately, `alignment_verified`, `alignment_asserted` and `alignment_not_applicable`. An asserted check is never
reported as verified. An assertion for a check whose metadata is present and contradictory is
rejected, naming the mismatch.

## 6. What `evaluate` reports

All arithmetic in float64 except in legacy output (§6.5). Empty arrays are rejected. Inputs
are checked finite before any metric; every computed value is checked finite before output.
JSON is strict: no NaN or Infinity is ever written; an undefined value is `null` with a
`status` string giving the reason. The aggregation unit is the flattened spectrum; the output
says so and states that flattened rows are not independent specimens.

### 6.1 Primary quantities, for every origin

`mse_in_mean`, `mse_out_mean` and `mse_difference = mse_in_mean − mse_out_mean` (positive:
the output is closer to the reference than the input), each against the selected reference.
This is a descriptive difference in discrepancy. It estimates the truth-referenced difference
only under the condition in §1; the output states that condition for every non-truth origin.

### 6.2 Relative summaries, in percent, named by their weighting

Positive means a lower output MSE than input MSE.

- `relative_mse_change_aggregate_pct = 100 × (1 − Σ mse_out / Σ mse_in)`. Where every
  `mse_in_i > 0` it is the input-error-weighted mean of the per-spectrum changes. It stays
  defined when an individual `mse_in_i = 0`, and is then not such a mean.
- `mean_relative_mse_change_per_spectrum_pct = 100 × mean(1 − mse_out_i / mse_in_i)` over
  spectra with `mse_in_i > 0`; the number of spectra excluded for `mse_in_i = 0` is reported.

They can disagree in sign: `mse_in = (0.01, 1)`, `mse_out = (0.02, 0.5)` gives −25 % and
≈ +48.5 %. A zero denominator for the whole aggregate gives `null` with a status.

### 6.3 dB quantities

**Base per-spectrum score.** `score = 10 log10(P_ref / max(P_res, 1e-10))`, with
`P_ref = mean(reference²)` and `P_res = mean((estimate − reference)²)` over the spectrum, in the
reference's units. It is computed for the input (`estimate = noisy`) and the output
(`estimate = denoised`).

- **Eligible spectra**: those with `P_ref > 0`. A spectrum with `P_ref = 0` has no score; it
  is counted (`zero_reference_power_count`) and excluded from every dB statistic and from
  the floor counts.
- **Floor counts**, over eligible spectra: `floor_active_input_count` and
  `floor_active_output_count`, a spectrum counting when its `P_res ≤ 1e-10` in the reference's
  units squared.
- **Derived statistics**, over eligible spectra: the input mean and output mean of the score;
  the per-spectrum change `score_out − score_in`, its mean, and its population standard
  deviation (divisor *n*, as the existing `snr_gain_std` uses).
- **No eligible spectra**: every dB statistic is `null` with a status.

Changing the floor is a separate audited metric change.

### 6.4 Naming by effective origin and relationship

| Case | Name of the primary quantities | dB quantities | Heading and key words never used |
|---|---|---|---|
| `synthetic_truth` | error against the synthetic truth | `snr_input_mean`, `snr_output_mean`, `snr_gain_mean`, `snr_gain_std` (existing keys; §6.3 defines them) | — |
| `estimate`, `overlap_with_evaluated` ≠ `overlap` | discrepancy from a reference estimate | `agreement_db_input_mean`, `agreement_db_output_mean`, `agreement_db_change_mean`, `agreement_db_change_std` (§6.3) | snr, quality, accuracy, improvement, gain |
| `estimate` with `overlap` established or declared | discrepancy from a reference built from the evaluated data | none | snr, quality, accuracy, improvement, gain |
| `undeclared` | discrepancy from an undeclared reference | none | snr, quality, accuracy, improvement, gain |

In normal (non-legacy) output, every non-truth case carries: *This measures agreement with a reference estimate. The
reference's noise, bias, and dependence on the evaluated data or the model can affect it; it
is not an established error against the underlying signal.* When overlap is established or
declared and `construction` is `frame_mean`, it adds: *A model returning this mean for every
frame has zero discrepancy from it by construction. That agreement does not establish
accuracy.* A change between input and output is named "change in discrepancy from the
reference".

Every output carries an `evaluation_context` object: `evaluate_output_version` (`"2"`), the
evaluated dataset's and the reference's content digests (§6.6), the stored and effective
declarations and their source, the relationship fields, `held_out_status`,
`alignment_verified`, `alignment_asserted`, the aggregation unit, and the model's identity if
available (else `"unknown"`).

### 6.5 Legacy output

`--legacy-output` exists to reproduce historical numbers exactly, and only when the effective
origin is `undeclared` and no `--reference-origin` is given; otherwise it is rejected (§4). It
is an explicit, stated exception to §6's float64 rule and to §6.4's naming table: it runs the
pre-change arithmetic (the input's own dtype, the pre-change formulas of `cmd_evaluate` at
commit `cb5e000`) and writes the pre-change keys — including `snr_*` and `mse_reduction_mean`
— with their pre-change values.

**Exactness is bounded by the platform.** The historical arithmetic runs in the input's
own dtype — float32 for files written by `generate`, and for the committed fixture — and
float32 summation differs in the last bits between NumPy versions and platforms. The values
are exact on the platform and NumPy version that produced them; elsewhere the portable
golden test allows `rel = 1e-6` and `abs = 1e-9`, and a second test compares the output
exactly with an independent transcription of the `cb5e000` arithmetic in the same runtime,
so a change of arithmetic (e.g. to float64) is still caught. **It is also bounded by strict JSON.** If any historical value is not finite (reachable
from finite inputs: a spectrum with zero input MSE and positive output MSE makes the
historical `mse_reduction_mean` −∞), legacy output is refused, naming the non-finite metric.
It never substitutes a value.

It adds, in the reserved object `evaluation_context`, everything §6.4 lists, with
`evaluate_output_version: "1-legacy"`, `legacy_baseline: "cmd_evaluate at cb5e000"`, and a
`limitations` string: *reference origin not declared; values are the pre-change arithmetic and
are truth-referenced only if the reference is the truth.* Nothing else is added to the
top-level keys. Alignment (§5) still applies; a legacy file without units metadata needs the
corresponding assertion.

### 6.6 Content digests

Digest format `dnd-digest-1`: SHA-256 over a sequence of named components, each framed as
`name` (UTF-8) `0x00`, an 8-byte little-endian length, then the payload. An absent component
is framed with the name and length 0 and listed in the output as absent.

- An array component's payload is an ASCII header `<dtype-str>|<shape as comma list>` `0x00`,
  then the array bytes in little-endian, C order.
- A JSON component's payload is the object serialised with sorted keys, separators `,` and
  `:`, non-ASCII kept, UTF-8.

Components, in this order:

- **reference digest**: `reference` (array), `energy`, `angles`, `times`, `frame_index` (as
  carried with the reference), `declaration_stored` (the bundle as stored, or absent). The
  effective CLI declaration is recorded separately, not hashed into this digest;
- **evaluated digest**: `noisy`, `denoised`, `energy`, `angles`, `times`, `frame_index`.

Test vectors — small arrays with their expected digests — are frozen with the implementation.
A digest identifies content; it does not establish provenance.

## 7. What this does not do

- It does not detect a falsely declared reference or false independence. It rejects a
  declaration contradicted by recorded identifiers (§2.2); it cannot detect undisclosed
  provenance or undeclared mixed origins.
- It does not decide held-out status (needs step 2), add a contraction measure (step 5), or
  support per-group declarations.
- Declared origin, relationship and alignment do not support a performance claim on their
  own; the data provenance, split unit, noise model, seeds and comparison conditions of
  `AGENTS.md` §6 still apply. The command does not certify a publication-ready comparison.
- It changes no committed measurement record. The benchmark and boundary records compute
  their own metrics with their own historical definitions and are not reinterpreted.

## 8. Tests that must exist before this ships

Each check is shown to reject a named wrong input, pinned to its reason
(`docs/VERIFICATION.md` §1), and each rejection test has every other field valid. Expected
values and metadata are literals or independent computations in the test, never the code
under test; legacy expectations come from a golden file produced by `cmd_evaluate` at commit
`cb5e000` and committed with the tests. Each group has positive counterparts, so an
implementation that rejects every difficult case fails.

1. **Declaration at write.** `generate` writes the complete bundle with `synthetic_truth`,
   `generator` and `units`, and the matching `intensity_units` on `noisy`, for 1-D,
   angle-resolved and `--no-normalize` output; `save_hdf5` without the declaration or without
   units is rejected naming the argument; an invalid `origin`, one bundle attribute without
   the other, and an unsupported version are each rejected as malformed; a file with neither
   attribute reads as `undeclared`, and a file with an explicit `undeclared` bundle reads as
   `undeclared` too.
2. **Propagation.** For every origin and both `infer` branches, the complete bundle (version
   included), `intensity_units` (on `noisy` and `denoised`), `acquisition_id`, `angles`,
   `times` and `frame_index` are preserved unchanged, and `energy` is unchanged without
   resampling and equals the independently computed resampled grid with it; the file stays
   readable; a lineage
   record is appended only when resampling; planted copiers that drop the version, the units
   or the acquisition id, alter a field, upgrade the origin, or copy a relationship each fail.
3. **End to end.** `generate → infer → evaluate` with no assertions yields the truth output
   (1-D, with angles and times listed as not applicable, and angle-resolved). A measured-style stack with a `frame_mean` reference from the
   same acquisition keeps its established overlap after `infer`.
4. **Truth arithmetic.** Per-spectrum scores, changes and each aggregate equal independent
   computations, including `snr_gain_std` with divisor *n*; floor counts equal exact input and
   output counts on a fixture with some spectra under the floor, and are zero on one with
   none; a zero-reference-power spectrum is excluded from scores and floor counts and
   counted; no eligible spectra gives `null` with a status; the JSON holds no NaN or Infinity.
5. **Non-truth outputs (normal mode).** For each non-truth case, no key or heading contains a
   forbidden word (planted outputs containing each are rejected); `agreement_db_*` values
   equal an independent computation (a planted constant fails).
6. **Legacy.** A legacy synthetic file yields no `snr_*` key by default. With
   `--reference-origin synthetic_truth --generator … --units …` and
   `--assert-alignment units` it yields the same SNR and MSE values as the golden file within
   float32-to-float64 tolerance; that declaration without `--generator` is rejected. With
   `--legacy-output --assert-alignment units` it yields the golden file's top-level JSON
   exactly plus `evaluation_context`. A fixture with a zero-input-MSE spectrum and positive
   output MSE makes legacy output refuse, naming `mse_reduction_mean`. `--legacy-output` with a
   known effective origin, or with `--reference-origin`, is rejected. Absent and explicit
   `undeclared` bundles are each tested.
7. **Resolution table.** Every row of §4, including an explicit `undeclared` bundle completed
   from the command line.
8. **Estimates and overlap.** Missing, empty or whitespace-only required fields are rejected;
   `other` without a description is rejected; `no_overlap_declared` contradicting an
   established overlap is rejected; positive counterpart: a mean over a disjoint acquisition
   declared `no_overlap_declared` is accepted and reported as discrepancy from a reference
   estimate, without the same-frames warning.
9. **Aggregation.** The opposite-sign example of §6.2 by hand; a spectrum with `mse_in = 0` is
   excluded from the per-spectrum mean and counted while the aggregate stays defined; all
   spectra excluded gives `null` with a status; a whole-aggregate zero denominator gives
   `null` with a status.
10. **Alignment.** Each rejected with every other field valid and the reason pinned: swapped
    rows detectable through `frame_index` within one acquisition; mismatched energy grids;
    reversed angle and time axes; mismatched units; broadcast without `--shared-reference`;
    non-finite arrays; an external reference with no identifiers and no `rows` assertion;
    an assertion given for a check whose metadata is present and contradictory; an
    angle-resolved external reference missing its `angles` with no assertion; an external
    reference of shape (samples, angles, energy) with matching angles but its sample blocks
    swapped and no `rows` assertion (matching angles alone must not verify sample order). Positive
    counterparts: a correctly aligned separate acquisition with different `frame_index`
    values; a legitimate shared reference; valid multi-angle data; an asserted check listed
    under `alignment_asserted`, not `alignment_verified`.
11. **Digests.** The frozen test vectors reproduce; changing any one component (a byte of the
    array, the dtype, the shape, an axis, the stored declaration) changes the digest; the
    effective CLI declaration does not.

## 9. Migration

CHANGELOG and QUICK_START state the breaking change and both routes (declaration, legacy
output). `generate` output from this version on is declared, so a fresh synthetic workflow is
unchanged.

## 10. Questions settled by the audits

- The overlap default for a frame mean is `unknown`, set to `overlap` only when established
  (§2.2) — second audit.
- A reference is identified by a canonical content digest that binds array, coordinates and
  declaration (§6.6) — second audit; its serialisation profile fixed as `dnd-digest-1` —
  third audit.
- Legacy output refuses rather than substitutes when a historical value is not finite (§6.5)
  — third audit.

## Revision log

- **Revision 1 (2026-09-30), after the first independent audit.** Origin separated from the
  relationship to the evaluated data and the model; a generic `estimate` origin with
  required fields; `undeclared` no longer yields SNR keys by default (breaking; owner's
  decision), with declaration and `--legacy-output` as routes; reference selection,
  declaration resolution and alignment specified; the stack-mean warning corrected;
  aggregation primary on mean MSEs; numerical edge cases and the SNR floor specified; tests
  expanded; the private study's numbers removed in favour of constructed examples.
- **Revision 2 (2026-10-01), after the second independent audit.** Legacy output restricted
  to undeclared references, made an explicit exception that runs the pre-change arithmetic
  and carries the full `evaluation_context` (the author's choice between the two options the
  audit offered, confirmed by the owner on 2026-10-01); alignment split into comparison coordinates,
  source identifiers and units, with missing metadata rejected unless asserted and asserted
  checks reported apart from verified ones; overlap now established from source membership
  rather than inferred from the construction (`stack_mean` became the operation
  `frame_mean`), `signal_match` moved to the evaluation-time relationship and the reference's
  own `conditions` kept with it; the unbiasedness statement replaced by the exact cross-term
  condition and a counterexample; relative summaries in percent and named by weighting; one
  dB formula with zero-reference-power handling and separate floor counts; strict JSON; the
  attribute bundle and lineage format defined, with malformed metadata rejected; tests
  pinned to reasons with positive counterparts.

- **Revision 3 (2026-10-01), after the third independent audit.** Legacy output bounded by
  strict JSON: it refuses, naming the metric, when a historical value is not finite, and its
  baseline is `cmd_evaluate` at `cb5e000` via a committed golden file; the full metadata path
  specified — `synthetic_truth` requires `units`, `generate` writes units matching its
  normalisation, `infer` preserves units, acquisition id and all coordinates — with an
  end-to-end test; angle and time coordinates compared, row correspondence established by
  same file, identifiers or assertion, and assertions never override a detected mismatch;
  declarations resolved by effective origin, so an explicit `undeclared` bundle can be
  completed like an absent one, and legacy eligibility defined by effective origin; the base
  dB score separated from its derived statistics, with eligible spectra, floor counts and the
  population standard deviation defined; forbidden-word tests scoped to normal output; the
  digest format enumerated as `dnd-digest-1`. The owner approved the five choices the audit
  left open (2026-10-01); that approved the choices, not yet the design as a whole.
- **Revision 4 (2026-10-01), after the fourth independent audit — the last design audit.**
  The audit found no blocking finding and nothing affecting the meaning of an evaluation
  output. Added: axis applicability before completeness (`alignment_not_applicable`);
  identifier-based row correspondence must cover every non-energy dimension, so matching
  angles alone cannot verify sample order; same-file correspondence described as accepted
  under the file schema; an example for an external shared reference; the resampling
  exception in the propagation test. By the owner's decision of 2026-10-01 the design audits
  stop here: what remains is decided in implementation and fixed by tests, and the frozen
  implementation and its tests get an independent review before release.
- **Implementation note (2026-10-02), after the independent review of the phase-1
  implementation (`315c78f`).** Two clarifications the review made necessary, neither a
  change of meaning: legacy exactness is bounded by the platform and NumPy version (§6.5;
  the CI on another NumPy differed in the last float32 bits); `estimate.conditions` has
  exactly four keys — `energy_calibration`, `channel_or_angle`, `exposure_normalisation`,
  `specimen_state` — each a non-empty string or a finite number, or `"unknown"`, with
  unknown fields in an origin refused. A follow-up review of the fixes (`506bd6a`) added: a
  resampled integer energy axis becomes floating point; frame identifiers are compared as
  exact integers of any width; legacy output is also compared exactly with an independent
  transcription of the `cb5e000` arithmetic in the same runtime.
- **Implementation note (2026-10-02), phase 2.** Decided in implementation and fixed by
  tests, as revision 4 provides: (a) the coordinate axes of a layout are the `ndim − 2`
  axes between the row axis and the energy axis; with one such axis, the coordinate
  dataset the evaluated file carries (`angles` or `times`) names it, and a file carrying
  neither has an unnamed axis that the user names and asserts with `--assert-alignment
  angles` or `--assert-alignment times`; with two, the layout is `(rows, times, angles,
  energy)` as `generate` writes it; more are refused; (b) metadata that contradicts the
  arrays it describes — a coordinate of the wrong length, a non-integer `frame_index`, a
  one-axis layout carrying both `angles` and `times` — is refused outright; (c) an
  assertion is accepted only for a check that could not be made: one for a verified
  check, or for an axis the layout does not have, is refused, so a verified check is
  never reported as asserted nor the reverse; (d) a shared reference
  (`--shared-reference`: one spectrum, shape `(energy,)` or with leading dimensions of
  one) carries no row identifiers and no angle or time axis, so each of those checks
  needs its assertion; in the same file its rows are matched under the file schema; (e)
  in the same file a present coordinate is verified and an absent one is still absent;
  (f) `intensity_units` must be present and equal on `noisy` and `denoised`: present on
  one only counts as absent, different values are a mismatch; (g) the
  `evaluation_context` fields are `alignment_verified`, `alignment_asserted`,
  `alignment_not_applicable`, `row_correspondence` and `shared_reference`, and the
  digests are under `digests` with `format` and, per side, `sha256` and `absent`; (h) in
  `dnd-digest-1` the dtype string is NumPy's dtype name (`float32`, `int64`, …), and
  `declaration_stored` is the JSON object `{"reference_origin": …,
  "reference_schema_version": …}` as read from the attributes; the lineage is not part of
  the digest, as the array bytes already reflect any transform. After the independent
  review of `f1b335a`: (i) every carried coordinate dataset, on either side and whether or
  not its axis applies, must be a finite one-dimensional numeric array, and a one-axis
  layout refuses `angles` and `times` together on the reference as on the evaluated file;
  a dataset for an axis the layout does not have is otherwise ignored for alignment and
  still hashed as carried; a carried `frame_index` obeys the integer and length rules in
  every layout, a shared reference's included (it is still not a row identifier there);
  (j) shapes are compared before the reference's coordinate lengths, so a reference of
  another `ndim` is refused as a shape mismatch; (k) a repeated `--assert-alignment` flag
  accumulates.
- **Implementation phases.** Phase 1: the reference declaration and its propagation
  (`generate`, `save_hdf5`, `infer`), naming by origin (no SNR for an undeclared reference),
  legacy output, and no silent ignoring of `--clean`. Phase 2: the rest of the alignment
  contract (§5) and the content digests (§6.6).

## Confirmation

**2026-10-01 — the owner adopted this design, revision 4, as a whole**, after four independent
audits (the last of which found no blocking finding), and approved implementation in the two
phases listed in the revision log, starting with phase 1. The status line changes to
"implemented" only when an implementation is merged.
