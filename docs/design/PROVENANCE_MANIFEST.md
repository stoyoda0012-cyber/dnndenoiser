# Design: what a trained model records about how it was made

**Status: adopted 2026-10-02 (revision 3, after three independent audits); phases A and B
implemented (see the CHANGELOG).** Step 2 of the improvement plan that follows a reproducibility assessment of a
published study. Its recording part (phase A) changes no evaluation output; its
relationship part (phase B) lets `evaluate` *establish* whether the evaluated data or the
reference took part in training, and renames the outputs when they did, so under
`AGENTS.md` §8 the design is independently audited before any code is written. It builds
on `docs/design/EVALUATION_REFERENCE_CONTRACT.md` (step 1, adopted 2026-10-01, implemented
in phases 1 and 2), whose vocabulary and `dnd-digest-1` format it uses.

## 1. The problem

A checkpoint written by `dnndenoiser train` holds the weights, the shape needed to
rebuild the network, the training method, a few hyperparameters, and (for
`moving-average`) the normalisation and energy axis. It does not say:

- which data it was trained on — not the file, not its content, not the acquisition or
  the rows;
- what the targets were made from (a synthetic truth, a second synthetic realisation, a
  window mean of the training frames);
- the seeds, the hyperparameters the run actually used, the software versions, the code
  commit, or the device;
- whether the weights are still the ones that were trained.

So after the fact, which model was trained on which data has to be reconstructed by hand
from file names and notes, and it can be reconstructed wrongly. Step 1 left two fields
open for this reason: every `evaluate` output carries `model: "unknown"` and
`held_out_status: "unknown"`, and `used_in_model_development` can only be declared. A model
evaluated on the very rows it was trained on reports the same names as one evaluated on an
independent acquisition.

The aim is narrow: **a model records, at training time and by the run itself, what it was
trained on and how; that record travels with the model's output; and `evaluate`
establishes from identifiers and digests — never infers from names — when evaluated rows
or the reference took part in training, and then names its outputs accordingly.** The
record identifies content; it does not prove origin, and it does not make training
reproducible (§7).

## 2. The manifest

`train` writes a manifest into the checkpoint under the key `provenance`. It is computed
by the run, never typed by the user (`docs/VERIFICATION.md` §1).

**Types and serialisation.** The manifest holds only JSON types: objects with string keys,
arrays, strings, finite numbers (Python `int` and `float`, never NumPy scalars), booleans
and `null`. Its canonical serialisation is step 1's JSON payload profile (§6.6: sorted keys,
separators `,` and `:`, non-ASCII kept, UTF-8) with non-finite numbers forbidden. A value
the run produced that is not finite (a diverged `final_loss`) is stored as `null`, and the
manifest's `statuses` object maps the field's dotted path to the reason.

| Field | Content |
|---|---|
| `schema` | `"dnd-provenance-1"` |
| `created_utc` | ISO 8601, seconds, `Z` |
| `software` | `dnndenoiser`, `python`, `numpy`, `torch`, `h5py` versions; `torch_cuda` (`torch.version.cuda` or `null`); `platform` (`platform.system()`, `platform.machine()`); `device` (the resolved backend) |
| `code` | §2.1 |
| `command` | §2.2 |
| `training_data` | §2.3 |
| `targets` | §2.4 |
| `preprocessing` | `resampling` (`from_points`, `to_points`) or `null`; `normalisation` as applied (`kind`, `min`, `max`) or `null` |
| `result` | `epochs` (completed), `final_loss` (the mean training loss of the last epoch, every method; `train_selfsupervised` is changed to return it). `train` refuses `--epochs` below 1, which today crashes after training with no loss to save |
| `statuses` | as above; `{}` when every value is defined |

**Privacy.** The values the run produces itself — arguments, effective settings,
environment, platform, code — contain no path, host name or user name: input and output
paths are excluded from `arguments`, `platform` holds only the system and machine type, and
`code` holds a commit or `"unknown"`. **Data-derived identifiers are copied verbatim**: the
training file's `acquisition_id`, its reference declaration (including free-text
`conditions` and descriptions) and its frame structure. Sharing a checkpoint, or any output
of it, shares them. Keeping names of people, places or specimens out of `acquisition_id`
and declarations is the user's responsibility, and so is keeping an `acquisition_id` unique
per acquisition (a reused one makes §5 establish relationships that are false); QUICK_START
says both.

### 2.1 Code

`commit` is recorded only when all of the following hold; otherwise `commit` and
`tree_clean` are both `"unknown"`:

- the imported package directory is inside a git work tree;
- that work tree's top level contains `src/dnndenoiser/`, which resolves to the imported
  package directory;
- `src/dnndenoiser/__init__.py` is tracked there.

This excludes the common case of a package installed into an ignored virtual environment
inside some other repository, whose commit is neither the code's nor public. A private
repository that vendors this package at a tracked top-level `src/dnndenoiser/` still passes;
QUICK_START says so. `tree_clean`
is `true` when no tracked file is modified or staged and no untracked file exists under
`src/dnndenoiser/`, else `false`. No path is recorded.

### 2.2 Command

- `method`;
- `arguments`: the values, as parsed (defaults included), of the `train` options on an
  **allow list** kept in the code; a test classifies every option of the `train` parser as
  recorded or excluded, so an option added later (a path, say) fails the test until it is
  classified. `--data` and `--output` are excluded; nothing else from the argument namespace
  (`func`, `command`, internal attributes) is recorded;
- `flags_passed`: the option names given on the command line, from the argument list the
  parser received, filtered to known option names as `_flags_passed` does (no values);
- `effective`: the settings the training code actually used, recorded from the code path
  that uses them, not re-derived from `arguments` — `architecture`, `num_features`,
  `num_hidden_units`, `encoder_output_dim`, `optimiser` (name and parameters), `scheduler`
  (name and parameters, or `null`), `loss`, `grad_clip`, `epochs`, `batch_size`, and for
  moving-average `window` after clamping. For moving-average the architecture, optimiser,
  scheduler, loss and clip are the fixed values of `train_selfsupervised`, whatever the
  parser defaults say;
- `seeds`: `torch` (the value passed to `torch.manual_seed`, or `null` when none was) and,
  for noise2noise, `targets` (the seeds its target generators used, defaults included).

### 2.3 Training data

All rows of the training file are used (the CLI splits off nothing), so the training set is
the file's content.

- `digest`: `dnd-digest-1` over the components `noisy`, `frames`, `clean`, `energy`,
  `angles`, `times`, `frame_index` **as stored** in the file — whether or not the method
  reads them — absent ones framed as absent and listed. It identifies these datasets, not
  the whole file (`seeds` and every attribute are outside it).
- `array_digests`: for `noisy` (or `frames`) and `clean` when present, a **matching
  digest**: `dnd-digest-1` over one component named `array` whose payload is the array
  cast to float32 in its stored shape. It is name- and dtype-independent on purpose:
  `infer` renames `frames` to `noisy` and writes float32, and step 1 frames a reference as
  `reference`; a digest that included the name or the stored dtype would never match the
  training data's own `infer` output.
- `layout`: the stored shape and dtype of each component; `rows_used: "all"`.
- `intensity_units` and `acquisition_id` of the input array, or `null`; `signal_identity` of
  `clean` (§5.4), or `null`.
- `frame_index_runs`: the **set** of training `frame_index` values as sorted inclusive
  `[first, last]` runs of consecutive integers (compact for a contiguous stack, exact for
  any set), or `null` without a `frame_index`. `train` validates a `frame_index` it records
  by step 1's rules — one-dimensional, integer, one value per first-axis row — and refuses
  otherwise, naming the dataset. It also refuses non-finite training arrays (today a NaN
  frame passes the frame-stack reader and reaches the normalisation constants).
- `reference_declaration`: the effective origin of the stored declaration of `clean`
  (step 1, §2.1) when the file has one, else `null`.

### 2.4 Targets

What the loss compared the output with: `{"kind": "clean"}` for noise2clean,
`{"kind": "synthesised_realisation", "noise_level": …}` for noise2noise,
`{"kind": "leave_one_out_window_mean", "window": …}` for moving-average (the window after
clamping).

### 2.5 Model digest

Two digests, each stored as `{"format": "dnd-digest-1", "sha256": <hex>}`.

**`model_body_digest`**, over what `infer` applies:

- each entry of `checkpoint["model_state_dict"]` **as stored** (loaded on the CPU), in key
  order by Unicode code point, as an array component named by its key, its payload the
  §6.6 array framing of `t.detach().cpu().contiguous().numpy()`. Persistent buffers are
  entries of the state dict and are included (BatchNorm running statistics and their
  integer counters, the Transformer's positional table). An entry that is not a tensor, or
  whose dtype NumPy cannot represent, makes `train` refuse to write a manifest and `infer`
  refuse to verify one, naming the key;
- then a JSON component named `config`: what `infer` uses to rebuild and apply the model —
  the values `checkpoint_model_config` resolves, under its canonical names
  (`num_features`, `num_hidden_units`, `encoder_output_dim`, `architecture`), plus
  `training_method` and `normalisation` (or `null`);

**`model_digest`**, stored beside `provenance` in the checkpoint, over two components: a
JSON component `body` (the `model_body_digest` object) and a JSON component `provenance`
(the manifest's canonical serialisation).

So an edit to the weights, a buffer, the architecture keys, the normalisation constants or
the manifest without a matching `model_digest` is detected — in `infer` from the
checkpoint, and in `evaluate` from what `infer` wrote (§3), without the weights.

## 3. Propagation and refusals

| Writer | Writes |
|---|---|
| `train` | `provenance` and `model_digest` in the checkpoint |
| `infer` | verifies the checkpoint (below); writes a scalar UTF-8 string dataset `model_provenance` holding exactly the manifest's canonical serialisation (a dataset, so its size is not limited), and the attributes `model_digest` and `model_body_digest` (each object as canonical JSON)
on `denoised`; writes on the output `noisy` the attribute `input_array_digest`, the matching digest (§2.3) of its input array **as read, before resampling**, so that a resampled output can still be matched with its training data; carries `signal_identity` on `clean` with the reference declaration |

**Absent, partial, malformed.** Step 1's rule (§3.1) applies to both records, in `infer`
and in `evaluate`:

- `provenance` and `model_digest` both absent (a checkpoint written before this version):
  `infer` writes neither; `evaluate` reports `"unknown"`. Not an error.
- exactly one present, or either malformed — an unknown `schema`, a missing or unknown
  field, a wrong type, a non-JSON value (possible through `--trust-checkpoint`'s full
  unpickler), a stored `model_provenance` that is not its own canonical serialisation —
  **refused**, naming the field. Never read as absent.
- `model_digest` present and different from the digest of the checkpoint as loaded:
  `infer` **refuses**: the manifest no longer describes what would be applied. There is no
  override; a checkpoint moved between devices is unaffected (it is loaded on the CPU and
  float32 bytes do not change).
- in an `infer` output, `evaluate` recomputes `model_digest` from the stored
  `model_body_digest` and the stored `model_provenance` bytes and **refuses** a mismatch,
  naming it: the copied manifest was edited, or attributes of another model were attached.
  The three records (`model_provenance`, `model_digest`, `model_body_digest`) are all present
  or all absent; anything else is refused. `evaluate` cannot check that the body digest
  describes the weights that produced `denoised` (it does not have them); §7 says so.

`infer` does not copy a `model_provenance` from its input: the output records the model
that was just applied. The reference declaration and lineage (step 1, §3.3) are carried as
before.

## 4. What `evaluate` reports — phase A

`evaluation_context.model` becomes, when the evaluated file carries a manifest:
`{"model_digest", "manifest_digest", "method", "training_data": {"digest",
"acquisition_id", "intensity_units"}, "software": {"dnndenoiser"}, "code": {"commit",
"tree_clean"}}`, where `manifest_digest` is the SHA-256 of the stored `model_provenance`
bytes. Otherwise it stays `"unknown"`. Phase A changes nothing else.

## 5. Established relationships — phase B

Established from recorded identifiers and digests, in the sense of step 1 §2.2: a
declaration that contradicts an established relationship is rejected; anything not
established keeps its declared or default value. Keys, all in `evaluation_context`:
`held_out_status`, `held_out_basis`, `rows_in_training`, `same_acquisition_rows_unidentified`,
`used_in_model_development` with `used_in_model_development_basis` (in the existing
`relationship` object), `reference_rows_in_training`, `shares_source_with_training_data`.
Bases take `established`, `declared` or `default`; `held_out_basis` is `established` or
`default`, never `declared` (there is no flag that declares it). When nothing is
established: `held_out_status` `"unknown"`, `rows_in_training` `null`,
`same_acquisition_rows_unidentified` `false`, `reference_rows_in_training` `null`,
`shares_source_with_training_data` `"unknown"`. `held_out_status` and `rows_in_training`
are also top-level keys of every normal output, equal to the context's.

### 5.1 `held_out_status` of the evaluated data

The evaluated side offers the matching digest of its `noisy` (§2.3), its `acquisition_id`
and its `frame_index`. The first rule that applies decides:

| # | Value | Established when |
|---|---|---|
| 1 | `not_held_out` | the evaluated `noisy` matching digest, or its recorded `input_array_digest` (§3), equals the training `noisy`/`frames` matching digest — every row; `rows_in_training = n` |
| 2 | `not_held_out` | both `acquisition_id`s are present and equal, both sides carry `frame_index`, and at least one evaluated `frame_index` lies in `frame_index_runs`; `rows_in_training` = the number of evaluated rows whose `frame_index` does |
| 3 | `disjoint_by_identifiers` | both `acquisition_id`s are present and differ; or they are equal, both sides carry `frame_index`, and no evaluated value lies in `frame_index_runs` |
| 4 | `unknown` | otherwise. When the `acquisition_id`s are equal but `frame_index` is missing on either side, the output also reports `same_acquisition_rows_unidentified: true` — reported, not established: nothing separates the rows, and a user may have split one acquisition into two files |

Rows are first-axis rows (frames or samples), not the flattened spectra `n_spectra` counts.
`not_held_out` means **at least one** evaluated row took part in training; `rows_in_training`
says how many. It is a count only under rules 1 and 2, and `null` under rules 3 and 4: rule 3
does not establish that no row was trained on. `disjoint_by_identifiers` is
not "held out": data copied under another identifier, a resampled or recast copy, or the
same specimen measured twice, is not detected; the name says what was checked.

### 5.2 The reference

Two separate fields; step 1's `used_in_model_development` keeps its literal meaning.

- **`used_in_model_development`** is established `yes`, for a model whose targets were built
  from `clean` (noise2clean trains against it; noise2noise synthesises its targets from
  it), when either
  - the reference array's matching digest equals the training `clean` matching digest
    (equal after the float32 cast); `reference_rows_in_training = n`; or
  - the reference carries a `signal_identity` (§5.4) equal to the training `clean`'s, and
    the reference's `frame_index` (the evaluated file's, for a same-file reference)
    intersects `frame_index_runs`; `reference_rows_in_training` = the count.

  This covers the ordinary synthetic transfer experiment: train on one `generate` file,
  evaluate on another with the same seed and another noise level. The evaluated noisy rows
  are new draws (§5.1 may say `disjoint_by_identifiers`, truly), but the truth rows are the
  training targets, and the output says so. `--used-in-model-development no_declared` contradicting it is rejected. Use for
  *selection* (validation, model choice) is never established — a training manifest does
  not record it — and stays declared.
- **`shares_source_with_training_data`** (new, established only: `yes` or `unknown`) is
  `yes` when the reference is an `estimate` whose `source.acquisition_id` equals the
  training `acquisition_id` and whose source frames intersect the training rows: `all`
  intersects when the training data have any rows of that acquisition; a list or a
  `{"range": …}` is compared with `frame_index_runs` (a range at its ends, as step 1 does);
  `unrecorded`, or `frame_index_runs: null` with anything but `all`, does not establish it.
  A moving-average model's targets were window means of its training frames, so a frame
  mean over those frames shares their data without being the training target.

### 5.3 Names and caveats

Step 1's rule — the name follows the relationship — extends to the training data.

- **Under `not_held_out`**, exactly these keys are renamed with the prefix
  `training_fit_`: `snr_input_mean`, `snr_output_mean`, `snr_gain_mean`, `snr_gain_std`
  (truth) and `agreement_db_input_mean`, `agreement_db_output_mean`,
  `agreement_db_change_mean`, `agreement_db_change_std` (an estimate not overlapping the
  evaluated data). Entries of `status` keyed by them follow the rename. The MSE keys and the
  counts (`zero_reference_power_count`, `floor_active_*_count`,
  `per_spectrum_excluded_zero_input_mse`) keep their names. The printed heading becomes
  *Fit to the training data — <step 1 heading>*; the JSON has no heading. In the cases with
  no dB keys (an estimate overlapping the evaluated data, an undeclared reference) only the
  top-level keys and the caveat change. The new names contain none of step 1's forbidden
  words. The caveat: *The evaluated data include rows (first axis) of the model's training
  data (k of n); these values describe the fit to those rows, not performance on new
  data.*
- **Every** normal output gains the top-level keys `held_out_status` and
  `rows_in_training`, so that a script reading the metrics alone sees them.
- `used_in_model_development` established `yes` adds, **in every case, truth included**:
  *The reference took part in training this model.* `shares_source_with_training_data:
  yes` adds: *The reference was computed from data the model was trained on.*
- **Legacy output** (step 1, §6.5) is refused when `not_held_out` is established: it exists
  to reproduce historical numbers under historical names, and those names would be false.
  Files written before this version carry no manifest and are unaffected.

### 5.4 Synthetic data

`generate` writes `frame_index = arange(n_samples)` and two identities, each `"generate:"`
or `"generate-signal:"` plus the first 16 hex digits of the SHA-256 of the canonical JSON
of a configuration object:

- the **signal identity**, the attribute `signal_identity` on `clean`: the peak set's
  parameters (its peaks and energy range as resolved, not its preset name), the seed, and
  every field that changes the stored `clean` in the chosen mode;
- the **noise identity**, the attribute `acquisition_id` on `noisy`: the signal identity's
  fields plus every noise field that changes the stored `noisy` in that mode.

Sample *i* depends only on the seed, the configuration and *i*, so a smaller `-n` with the
same identity is a prefix of a larger one, and with `frame_index` this is rule 2's
intersection; `clean` does not depend on the noise fields, so the same seed with another
noise level gives the same signal identity and §5.2 recognises the training targets.

**Which fields.** A field enters an identity in a mode exactly when changing it changes the
corresponding array in that mode, **and its value is not the field's no-effect value**
(a zero shift rate, a zero variation, `use_gaussian_approx` off). The second clause keeps
identities stable when a later version adds a field at its no-effect value; **a field added
later must have a no-effect value that reproduces the earlier arrays, or be hashed under a
new identity prefix** so that old and new identities are never compared as equal or
disjoint. Values are hashed **as resolved** (a `gaussian_approx_min_rate` of `None` as the
value the code uses), and configurations that produce identical arrays are canonicalised
to one form: noise type `none`, a zero `poisson_level` and a zero `gaussian_std` all as
"no noise"; background `shirley` with level 0, `linear` with level and slope 0, and `none`
all as "no background". The code's table states each field's no-effect value, or "none"
for fields that have none (`eta`, `normalize`, `n_energy_points`, …). The
classification lives in the code as one table per mode and is checked empirically by tests
(§8, group 8). The cases the second audit found must be classified as follows:

| Field | In the identity when |
|---|---|
| `n_samples`, output path, `--manifest`, the dnndenoiser and NumPy versions | never |
| noise type, `poisson_level`, `gaussian_std`, `use_gaussian_approx`, `gaussian_approx_min_rate` | noise identity only, each when its noise type reads it |
| `background_level` | the background type is not `none` |
| `background_slope` | the type is `linear` |
| angle fields (`angle_range` bounds, model parameters, `angle_shift_rate`) | `n_angles > 1`, each bound or parameter when the chosen angle model or a non-zero shift reads it (neither bound under model `none` with zero shift; both bounds under every other model, `linear` included) |
| time fields (`time_range` bounds, `time_decay_constant`, oscillation parameters, `time_shift_rate`) | `n_times > 1`, likewise per time model (both bounds under every model but `none`, `linear_decay` included) |
| every other field (`n_energy_points`, the energy range as resolved, `use_pseudo_voigt`, `eta`, background type, the three variations, `normalize`) | always, unless at its no-effect value |

A test freezes one identifier for a fixed configuration, so a change that alters existing
identifiers fails it.

**Versions.** The truth declaration's `generator` names the dnndenoiser version. When two
identities are equal but the training and evaluated (or reference) generator versions
differ, the output adds: *Generated by different versions; equal identities assume equal
draws.* If the draws did change between versions, an established relationship can then be
false; the note says so.

Different seeds give different identities and are reported `disjoint_by_identifiers`
without a distributional qualification: sharing the generator's distribution is what an
in-distribution test is, and the split rule and seed count belong to the claim
(`AGENTS.md` §6), not to this field. Per-sample seeds are drawn from `[0, 2^31)`, so two
synthetic sets of `N_t` and `N_e` samples share about `N_t·N_e/2^31` identical draws by
chance; `disjoint_by_identifiers` does not exclude them.

## 6. Storage limits and compatibility

- The manifest is small except `frame_index_runs`, at most one run per frame; it is stored
  in the checkpoint and in a dataset, not an attribute, in the `infer` output.
- Checkpoints written before this version load as before; `infer` writes no manifest for
  them and `evaluate` reports `"unknown"`. No existing checkpoint key changes.
- `load_checkpoint`'s restricted mode reads the manifest (plain types only; checked with
  nested objects, `null`, booleans, lists and large integers).
- New datasets and attributes: in `generate` output, `frame_index`, `acquisition_id` on
  `noisy` and `signal_identity` on `clean`; in `infer` output, the dataset
  `model_provenance`, the attributes `model_digest` and `model_body_digest` on `denoised`,
  `input_array_digest` on `noisy`, and `signal_identity` carried on `clean`.
- `generate` output gains `frame_index` and `acquisition_id`. Step 1's alignment checks the
  new `frame_index` (an integer per row) and otherwise behaves as before: a same-file
  evaluation is matched by the file, and an external synthetic-truth reference still needs
  `--assert-alignment rows`, since a truth declaration names no acquisition. Step 1's
  refusal message for that case is corrected to say so (today it says the identifiers come
  "from different acquisitions").

## 7. What this does not do

- It does not prove where a model came from. A manifest can be edited together with its
  `model_digest`; the digest detects changes made *without* updating it. There is no
  signature. In an `infer` output, `evaluate` verifies that the manifest and the body
  digest belong together, not that the body digest describes the weights that produced
  `denoised`.
- It does not detect stale row metadata left by a user's own tools: a `frame_index`
  renumbered after taking a subset, or an `input_array_digest` or `signal_identity` kept on
  an array that was replaced. QUICK_START says that such tools must drop or rewrite them.
- It does not detect training data that reached the model under another identity: a copy
  without its `acquisition_id`, a resampled, recast, reshaped, reordered or row-subset copy
  without `frame_index`, frames of the same specimen under another acquisition.
- It does not record anything outside `dnndenoiser train` or anything done to the file
  before `train` read it.
- **It does not make training reproducible.** It records what was run; non-deterministic
  kernels, dropout and device differences remain (`--seed`'s help already says so).
- It changes no metric definition and no alignment rule of step 1, and no committed
  measurement record.

## 8. Tests that must exist before this ships

Each check is shown to reject a named wrong input, pinned to its reason, with every other
field valid and a positive counterpart; expected values are computed in the tests, never by
the code under test.

1. **Manifest at write.** For each method: every field present with the right type and no
   other; `digest` and `array_digests` equal an independent computation over the file;
   `frame_index_runs` equals an independent run-length encoding of the set (contiguous,
   gapped, unsorted, duplicated, single frame); `seeds.torch: null` without `--seed`;
   noise2noise records its target seeds, defaults included; moving-average's `effective`
   holds the fixed values of `train_selfsupervised` and the clamped window, not the parser
   defaults (a planted manifest copying the defaults is rejected); `arguments` reflects a
   passed flag and a default and excludes the paths; no run-produced value contains the
   input or output path, the host name or the user name (planted values caught);
   `code.commit` is `"unknown"` for a package inside an ignored directory of another git
   repository, and recorded for this checkout; a diverged run stores `final_loss: null` with
   its status; the canonical serialisation is strict JSON; every `train` option is
   classified for `arguments`; `--epochs 0`, a float, two-dimensional or wrong-length
   training `frame_index`, and a NaN frame are each refused, naming the reason.
2. **Model digest.** Equals an independent computation, frozen as a test vector for a small
   model; changes when one weight, one buffer, an architecture key or a normalisation
   constant or the manifest changes; a checkpoint carrying both spellings of a shape key
   hashes the resolved values; `infer` refuses each altered checkpoint naming the digest
   (including a manifest edited alone), and `evaluate` refuses an `infer` output whose
   `model_provenance` was edited alone, or whose three records do not all match, accepts an
   unaltered one, accepts a pre-manifest checkpoint and writes no manifest; a non-tensor
   state-dict entry is refused naming the key.
3. **Absent, partial, malformed.** Each case of §3 refused naming the field, in `infer`
   and in `evaluate`, with every other field valid; both absent accepted as `"unknown"`.
4. **Propagation.** `infer` writes `model_provenance` (the checkpoint's manifest,
   canonical) and `model_digest`; does not copy an input's `model_provenance`; the
   restricted loader reads a checkpoint with a manifest.
5. **Phase A output.** `evaluation_context.model` equals the fields of §4 computed
   independently, `manifest_digest` against a frozen vector; `"unknown"` without a manifest.
6. **Phase B, held-out.** Each rule of §5.1 with every other field valid: the training file
   itself after `infer` (rule 1); a float64 frame stack and its `infer` output (rule 1, by
   the matching digest despite the rename and the cast); the same acquisition with
   intersecting frames, counted (rule 2); one intersecting row of many (rule 2, `k` of
   `n`); disjoint frames and a different acquisition (rule 3); digests equal but
   identifiers different (rule 1 wins); one acquisition split into two files without
   `frame_index` (`unknown` with `same_acquisition_rows_unidentified`, **not**
   `not_held_out`); a resampled copy without identifiers (`unknown`). Renamed keys and
   heading exactly under `not_held_out`, the full key set asserted, `status` entries
   included; `rows_in_training` `null` under rules 3 and 4; a 300-point frame stack
   trained with moving-average and evaluated on its own resampled `infer` output (rule 1
   through `input_array_digest`); an angle-resolved `generate → train → infer → evaluate`
   with a first-axis subset (count against first-axis rows, not `n_spectra`); the top-level
   keys in every output, equal to the context's; legacy output refused under
   `not_held_out`.
7. **Phase B, the reference.** The training `clean` as an external reference, for
   noise2clean and noise2noise (`used_in_model_development` established `yes`, caveat in
   the truth case); the same seed with another `--poisson-level` and another `-n`
   (`held_out_status` `disjoint_by_identifiers`, `used_in_model_development` established
   `yes` by signal identity with `k` of `n`); a frame mean over the training acquisition (`shares_source…: yes`, `used…`
   not established); `unrecorded`, a range intersecting and not, `frame_index_runs: null`;
   `no_declared` contradicting an established `yes` rejected; positive counterparts.
8. **Synthetic identity.** `-n 100` and `-n 1000` with the same seed share both identities
   and rule 2 counts all 100 rows; another noise level changes the noise identity and not
   the signal identity, and `clean` is unchanged; the upper angle and time bounds under
   `linear` and `linear_decay` change both identity and arrays with lower bounds 0 and 10;
   the canonicalised equivalents (no noise, no background, `min_rate` resolved) give one
   identity and identical arrays; for each mode, changing a field where it
   is included changes the identity **and** the array, and changing it where it is excluded
   changes neither (the classification checked against the generator, not against
   itself); a field at its no-effect value is omitted; every field of `GeneratorConfig`
   and `NoiseConfig` is classified; one identifier frozen for a fixed configuration; the
   versions note when the generators differ.
9. **Regressions of the `generate` change.** Same-file evaluation, an external truth
   reference with `--assert-alignment rows` (with the corrected message without it), and a
   shared reference behave as before on new `generate` output.

## 9. Migration

Additive for checkpoints and data. `evaluate`'s normal output gains `held_out_status` and
`rows_in_training`; under an established `not_held_out` its dB keys are renamed (breaking
for a script that reads `snr_*` from an evaluation of training data, which is the case the
rename exists for). CHANGELOG and QUICK_START describe the manifest, the refusals, the new
fields, the rename, and that data-derived identifiers travel with a shared model.

## 10. Implementation phases

Phase A: §2–§4 (manifest, model digest, propagation and refusals, the model identity in
`evaluate`). Phase B: §5 (established relationships, names and caveats, synthetic
identity). Phase B changes the meaning of evaluation outputs; both are implemented only
after the design as a whole is adopted, and each frozen implementation gets an independent
review before merge.

## Revision log

- **Revision 0 (2026-10-02).** First draft.
- **Revision 1 (2026-10-02), after the first independent audit** (three blocking, eight
  should fix, four minor). §5.1 no longer establishes `not_held_out` when `frame_index` is
  missing (it reports `same_acquisition_rows_unidentified` instead), gives its rules a
  precedence, and counts `rows_in_training`; the synthetic identity hashes only fields that
  change the arrays, excludes `n_samples`, and `generate` writes `frame_index`; `code.commit`
  is recorded only for this package's own checkout; matching digests are name- and
  dtype-independent; the model digest is specified (stored state dict, buffers, key order,
  refusals) and covers the keys `infer` uses; the manifest's types, canonical form and
  non-finite values are defined; absent, partial and malformed records follow step 1's
  refusal rule; `command` separates parsed arguments from the settings actually used and
  records every seed; the privacy statement is limited to run-produced values. Owner's
  decisions (2026-10-02): outputs are renamed under `not_held_out` (`training_fit_`) and
  gain top-level `held_out_status`; a reference computed from training data gets the new
  field `shares_source_with_training_data` rather than a wider `used_in_model_development`;
  synthetic identity by the array-changing fields plus `frame_index`; data-derived
  identifiers are copied verbatim. No override of the `infer` refusal (the audit found no
  legitimate case needing one).
- **Revision 2 (2026-10-02), after the second independent audit** (one blocking, seven
  should fix, nine minor; fourteen of the first audit's fifteen findings closed). `generate`
  writes a signal identity on `clean` beside the noise identity on `noisy`, and §5.2
  establishes `used_in_model_development` from it, so a model evaluated against its own
  training targets under another noise level is recognised; `rows_in_training` is `null`
  unless rows were found; the identity's fields are classified per mode, at their
  no-effect value omitted, with the missing noise and shift fields; `infer` records its
  input's matching digest before resampling; `model_digest` covers the manifest and the
  resolved model configuration; the renamed keys are listed exactly; training-side
  `frame_index`, non-finite frames and `--epochs 0` are refused; `arguments` uses an allow
  list; new keys are named; a versions note when generators differ; step 1's message for an
  external truth reference is corrected. Owner's decisions (2026-10-02): a separate signal
  identity rather than dropping the noise fields; identities kept stable by omitting fields
  at their no-effect value.
- **Revision 3 (2026-10-02), after the third independent audit** (verdict: adopt with
  named changes; N1–N17 closed but for one table entry). The upper angle and time bounds
  are in the identity under `linear` and `linear_decay` too; the identity hashes resolved
  values, the peak set's parameters rather than its name, and one canonical form for
  configurations giving identical arrays; a field added later must reproduce the earlier
  arrays at its no-effect value or use a new prefix; `model_digest` is split into a body
  digest and the manifest so that `evaluate` verifies an `infer` output's manifest (owner's
  decision, 2026-10-02); unestablished values are stated; the compatibility list and the
  limits on stale row metadata are added.
- **Implementation note (2026-10-02), phase B.** Decided in implementation and fixed by
  tests: the identity classification is `CLASSIFICATION` in `src/dnndenoiser/data/identity.py`,
  checked against the generator field by field and mode by mode; with a lower bound of 0
  the linear angle and time models read only the ratio to the upper bound, so the arrays
  agree to float32 rounding while the identity, as adopted, still includes the bound (it
  can call identical draws different, never the reverse); noise components are
  canonicalised to the active ones (a zero level or standard deviation is no component,
  so `mixed` with one zero equals the other type); a moving-average model never
  establishes `used_in_model_development` (its targets were window means); an external
  reference that cannot be matched by identifiers is refused naming the true reason
  (a truth declaration names no acquisition, or the evaluated data carry none).
  After the independent review of the phase-B implementation: rule 1 compares the
  array the method trained on (`frames` for moving-average, `noisy` otherwise), since a
  file may hold both; the versions note is given only when identities were compared
  (rule 2 or the signal identity), not on a content match; `input_array_digest` is
  validated only where it is used. §5.2's "new draws" holds for Poisson noise; with the
  same seed, Gaussian noise repeats its deviates rescaled, so `disjoint_by_identifiers`
  stays literally true but the noise is not independent (QUICK_START says so).

## Confirmation

**2026-10-02 — the owner adopted this design, revision 3, as a whole**, after three
independent audits (the last returning "adopt with named changes", all applied in this
revision), and approved implementation in the two phases of §10, starting with phase A.
The status line changes to "implemented" only when an implementation is merged.
