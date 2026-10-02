# Design: frame stacks with angle channels, and what the acquisition order rests on

**Status: adopted 2026-10-02 (revision 2, after two independent audits); implemented (see
the CHANGELOG).** Step 3 of the improvement plan that follows a reproducibility assessment of a
published study. It changes what the self-supervised `moving-average` method trains on for
a new input form — which frames are a frame's neighbours, and over what the normalisation
is taken — so under `AGENTS.md` §8 it is independently audited before any code is written.
It uses the vocabulary of `docs/design/EVALUATION_REFERENCE_CONTRACT.md` (step 1) and
`docs/design/PROVENANCE_MANIFEST.md` (step 2).

## 1. The problem

An angle-resolved measurement repeated in time gives an array of frames × angle channels ×
energy. One way to train the `moving-average` method on it is one model for all channels,
with each frame's target the mean of its temporal neighbours **in the same channel**, and
one normalisation over all channels. Today:

- **The frame stack has no channel axis.** `frames` is `(n_frames, energy)`. The only way
  to train one model across channels is the documented workaround: one stack, channel *k*
  at indices *k* × *stride* + *t* with *t* = 0, 1, …. It keeps every target inside its
  channel only while stride ≥ (the largest channel's frame count) + `--window` and the
  window is smaller than every channel's frame count; past either bound, a frame's nearest
  others reach into the next channel, silently. (QUICK_START gave a weaker bound until it
  was corrected in a separate change, merged before this design was adopted.)
- **Nothing says what the channels are.** An angle axis may be the analyser's angle or the
  emission angle, in either order; a reader of the file cannot tell.
- **Nothing says what the acquisition order rests on.** `frame_index` is defined as
  acquisition order, and the method's targets depend on it, but a file cannot say whether
  the order was recorded by the instrument or inferred by whoever converted the data. A
  wrong inferred order changes every target without any visible error.

**What this design does about each — and what it does not.** It gives the stack a channel
axis on which the method builds every target inside its own channel, whatever the window.
It lets a file *declare* the angle kind, units and order basis, and records the
declarations. It **checks none of them**: a permuted, offset or reversed channel axis
changes no target and no normalisation constant, so nothing at training can detect it;
at `evaluate`, the angle *values* are compared with a reference that carries `angles`
(step 1), but `angle_kind` and `angle_units` are not; the order basis is never checked.
Determining the order, converting vendor files and denoising jointly across angles stay
out of scope (`AGENTS.md` §1, §5).

## 2. Schema

Additive to the current frame stack (`src/dnndenoiser/data/frame_stack.py`). **Energy is the
last axis everywhere** (`n_energy = frames.shape[-1]`), for 2-D and 3-D stacks alike.

| Dataset | Shape | Notes |
|---|---|---|
| `frames` | `(n_frames, energy)` as today, **or** `(n_frames, n_angles, energy)` | one frame is one acquisition of every channel |
| `energy` | `(energy,)` | its length equals `frames.shape[-1]` |
| `frame_index` | `(n_frames,)` | unchanged: integer, unique, acquisition order, one per frame, shared by its channels. New optional attribute `order_basis`: `"recorded"` (from the instrument's record), `"inferred"` (derived by the user, e.g. from an array layout) or `"unknown"`; absent reads as `"unknown"`. It is a declaration and is never checked |
| `angles` | `(n_angles,)` | **required** with a 3-D `frames`; with a 2-D `frames` it is accepted as today and not read as a channel coordinate. Numeric of kind integer or float (not boolean), finite, **no repeated values**, one per channel, in the order of the channel axis (any order; never sorted by the reader). Attributes, both required: `angle_kind` ∈ {`"emission"`, `"analyser"`, `"other:<description>"`} (the description non-empty, at most 200 characters, copied verbatim into a manifest — step 2's privacy note) and `angle_units` = `"deg"` |
| `times` | — | **forbidden with a 3-D `frames`**: per-frame times would make an `infer` output carry both `angles` and `times` on a one-axis layout, which step 1 refuses. With a 2-D `frames` it is accepted as today |

The coordinate is named `angles`, as in `generate` output and step 1's alignment, so an
`infer` output of a channel stack is an ordinary `(rows, angles, energy)` file there.
Non-angle channel axes are not covered by this revision.

**2-D stacks are unchanged** (owner's decision, 2026-10-02): what the reader accepts
today for a 2-D `frames` it keeps accepting, `angles` and `times` included. For a 3-D
`frames` the reader refuses each of: no `angles`; an `angles` length different from the
channel axis; non-finite, boolean or non-numeric values; repeated values; a missing or
unknown `angle_kind`, an empty or over-long `other:` description, a missing or other
`angle_units`; a `times` dataset. For any stack it refuses an `order_basis` outside the
three values (the attribute is new, so no existing file carries one) and an `energy`
length different from `frames.shape[-1]` (for 2-D stacks this is today's check).

**Writer.** `write_frame_stack` takes `angles`, `angle_kind`, `angle_units` and
`order_basis`; `order_basis` has no default value — when it is not given, the attribute is
not written and reads as `"unknown"`. (Its existing default for `frame_index`, `arange`,
already asserts an order; the docstring says so.)

## 3. Training with channels

For a 3-D stack, `moving-average`:

1. **Targets inside the channel.** For each frame *t* there is one neighbour set — the `W`
   frames whose `frame_index` is nearest to frame *t*'s, excluding *t*, chosen exactly as
   `moving_average_targets` chooses them (including its `argsort` tie-breaking) — and it is
   the same in every channel, because every channel shares `frame_index`. The target of
   frame *t* in channel *c* is the mean of that set's frames **in channel *c***. This is
   `moving_average_targets(frames[:, c, :], frame_index, W)` for each *c*, and equals
   `moving_average_targets(frames.reshape(n_frames, n_angles * E), frame_index, W)`
   reshaped back, bit for bit for the float32 frames `train` passes (their sums are exact
   in float64; for float64 inputs the two can differ in the last bit through pairwise
   summation). A frame of another channel is never in a neighbourhood, whatever `W`.
2. **One normalisation over all channels.** The min-max constants are taken over every
   value of every frame and channel after resampling, as for a 2-D stack, and are recorded
   in the checkpoint as today. **Consequence:** the loss is weighted by absolute intensity,
   so a channel much dimmer than the brightest contributes little to training; the targets
   stay unbiased (they are on the scale of their inputs); and the trained distribution is
   defined by the global range, so a channel stack whose range differs at `infer` puts
   every channel out of distribution (`AGENTS.md` §5).
3. **Window.** `W` is clamped to `n_frames − 1`, as today (every channel has `n_frames`
   frames).
4. **Rows.** The network is still spectrum-wise. Training rows are the `(frame, channel)`
   pairs in C order of `(n_frames, n_angles)` — **frame-major is part of the method's
   definition for 3-D stacks**: under a seed, the batch order and therefore the weights
   depend on it. A workaround stack (channel-major) does not reproduce these weights.
5. **The 2-D case is unchanged**, bit for bit, and a 3-D stack with one channel trains to
   the same weights as the equivalent 2-D stack — for every energy length: resampling is
   applied to the frame-major rows, `frames.reshape(n_frames * n_angles, E)`, exactly as a
   2-D stack's rows are resampled (resampling the 3-D array directly differs from the 2-D
   call in the last bits when the axis has one channel).
6. **Order basis.** `train` prints a warning when a moving-average stack's basis — 2-D or
   3-D — is `"inferred"` or `"unknown"` (targets are only as right as the order). Other
   methods do not use the order and do not warn.

The P1 preregistration's reproduction covers the 2-D path only; the 3-D path is not part
of what P1 reproduces.

**From the workaround to the channel axis.** Targets agree wherever no neighbour choice
is a tie. On tie rows (evenly spaced indices, odd `W`) `argsort` may pick differently on
the workaround's longer index row, so the two can differ there; and summation order can
differ in the last bit.

## 4. Inference and evaluation

`infer`, when its input is a frame stack:

1. checks the channel axis of a 3-D stack — the 3-D rules of §2 for `angles`, its
   attributes and `times` — and nothing that only training needs: a stack with one
   frame, without `frame_index`, or with non-integer or repeated indices is accepted by
   `infer` as today (owner's decision, 2026-10-02). A 2-D stack is read as today;
2. writes `noisy` and `denoised` as `(n_frames, n_angles, energy)` (it already does);
3. carries `angles` **with** `angle_kind` and `angle_units`, and `frame_index` **with**
   `order_basis` (today it copies the datasets without their attributes).

`evaluate` then treats the output as a `(rows, angles, energy)` file: step 1 compares
`angles` element by element with a reference that carries them (a reversed or offset axis
is a mismatch); `angle_kind` and `angle_units` are **not** compared. Step 2's held-out
rules identify **frames** (first-axis rows), not channels: a model trained on some
channels of an acquisition and evaluated on all of them counts every frame as a training
row (conservative); and a model trained on a workaround stack whose indices are not
acquisition order can give a false `disjoint_by_identifiers` if that stack carried the
acquisition's `acquisition_id` (QUICK_START tells users not to). The natural reference for
a channel stack — the per-channel mean, shape `(n_angles, energy)` — is not a single
spectrum and cannot be used with `--shared-reference`; repeat it over the frames.

## 5. Provenance

The manifest schema becomes `dnd-provenance-2`, written by **every** `train` run from this
version on, 2-D included (so new checkpoints differ from earlier versions' and older
dnndenoiser versions refuse them; existing checkpoints are unchanged). It adds two fields
to `training_data`, for every method:

- `angles`: `null` exactly when the training file has no `angles` dataset (equivalently,
  `layout` has no `angles`); otherwise exactly `{"kind": …, "units": …}`, where `kind` is
  `"emission"`, `"analyser"`, `"other:<description>"` or `null` (not declared, as in a
  `generate` file) and `units` is `"deg"` or `null`. The number of channels is already
  `layout.angles.shape[0]`;
- `frame_index_basis`: one of `"recorded"`, `"inferred"`, `"unknown"` — the `order_basis`
  of the training file's `frame_index`, or `"unknown"` when it has none or has no
  `frame_index`.

**Out-of-vocabulary declarations are refused at `train`, for every method** (owner's
decision, 2026-10-02): an `angle_kind`, `angle_units` or `order_basis` attribute outside
the vocabulary is refused naming the attribute; a file without them trains as today.
Attributes may be stored as `str` or `bytes`; both are read.

**Version rules.** A version-1 manifest is validated with exactly the version-1 fields, a
version-2 manifest with exactly the version-2 fields; anything else — a version-1 manifest
with the new fields, a version-2 manifest without them, any other schema value — is
refused. **A stored manifest and its bytes are never changed**: `model_digest` and the
canonical-serialisation check stay exact, and the defaults (`angles: null`,
`frame_index_basis: "unknown"`) are applied only where a field is used. Versions of
dnndenoiser before this one refuse version-2 records ("unknown provenance schema"), the
expected forward incompatibility.

`evaluate` reports, under `evaluation_context.model.training_data`, the keys
`declared_angles` and `declared_frame_index_basis` — named so that they read as
declarations, not checked. For a version-2 manifest they are its two fields; for a
version-1 manifest both are `"not recorded"` (not `null`, which would say the file had no
angles). `evaluate_output_version` stays `"2"`: the change is additive keys in
`evaluation_context`.

## 6. Compatibility

- 2-D stacks are read and trained as before; **new** checkpoints carry a version-2
  manifest (older versions refuse them); existing version-1 checkpoints still verify in
  `infer`, and their outputs carry the version-1 text byte for byte.
- `train` refuses an out-of-vocabulary `angle_kind`, `angle_units` or `order_basis`
  attribute (new attributes, so no existing file is affected unless it happens to carry
  one).
- `generate` is unchanged (it writes the `noisy` schema).
- A workaround stack keeps training as before; QUICK_START marks the workaround
  deprecated, states its corrected bound, and says that its indices are not acquisition
  order and must not carry the acquisition's `acquisition_id`.
- A 3-D frame stack without `angles`, with malformed `angles`, or with `times` used to pass
  `infer` and is now refused — a breaking change for such files, listed in the CHANGELOG.

## 7. What this does not do

- It does not determine, check or detect a wrong acquisition order; `order_basis` is a
  declaration.
- It does not detect a permuted, offset or reversed channel axis at training, and does not
  compare `angle_kind` or `angle_units` at `evaluate` (comparing them would amend step 1
  §5.1, a separate audited change).
- It does not convert vendor files or decide which angle a vendor axis is.
- It does not denoise jointly across angles: the model stays spectrum-wise.
- It does not cover non-angle channel axes, or stacks whose channels have different
  numbers of frames.

## 8. Tests that must exist before this ships

Each check is shown to reject a named wrong input, pinned to its reason, with every other
field valid and a positive counterpart; expected values are computed in the tests.

1. **Reader and writer.** Each refusal of §2 with every other field valid (including a
   boolean `angles`, a repeated value, a `times` dataset, an energy length equal to
   `n_angles`); valid 3-D and 2-D stacks accepted; `angles` returned in file order (a
   descending or interleaved axis stays as written); `order_basis` absent reads as
   `"unknown"`; the writer writes no `order_basis` unless given.
2. **Targets inside the channel.** (a) On tie-free index sets (distinct pairwise
   distances), each target equals an independent brute-force mean over the same channel,
   for every `W` from 1 to `n_frames − 1`; (b) on any index set and float32 frames, channel
   *c*'s target equals `moving_average_targets(frames[:, c, :], frame_index, W)` bit for bit
   (the definition, not an independent check); (c) changing one channel's frames leaves the others' targets
   bit-identical; (d) a planted implementation that pools channels fails.
3. **Equivalence with the workaround**, on tie-free index sets, with a stride larger than
   2 × (the largest index span within a channel) + 1 so that channels cannot interleave
   whatever the gaps, and `W` < frames: the same targets row for row after reordering, to 1e-12, and the same
   normalisation constants exactly; and a tie case where they legitimately differ, stated.
4. **Normalisation** equals the independent min and max over all channels.
5. **Unchanged paths.** A 2-D stack trains to bit-identical weights at this version and the
   previous one (same seed, CPU); a `(n, 1, E)` stack trains to the same weights as the
   `(n, E)` stack, at E = 256 and at an E that needs resampling.
6. **Rows.** End to end, training row *r*'s input and target belong to the same
   `(frame, channel)`, in frame-major order.
7. **Resampling** for a 3-D stack: E = 256 records `resampling: null`; E = 300 records
   `from_points: 300`.
8. **Inference.** `infer` refuses a 3-D stack without `angles`, with malformed `angles`,
   with `times`; keeps accepting a 2-D stack with `times`/`angles`, a one-frame stack and a
   stack without `frame_index`; carries `angles` with its attributes and `frame_index` with
   `order_basis`, also through resampling, with attributes stored as `str` or `bytes`;
   `evaluate` verifies `angles` against a matching reference and refuses a reversed one.
   The window is clamped to `n_frames − 1` for a 3-D stack at the CLI.
9. **Provenance.** A version-2 manifest records `angles` and `frame_index_basis` for each
   method (a `generate` 3-D file gives `kind`/`units` `null`); the value rules of §5,
   each with a refused counterexample (an `angles` declaration without an `angles` layout,
   a basis outside the vocabulary, units other than `"deg"`); out-of-vocabulary attributes
   refused at `train` for a supervised and a moving-average file; exact key sets per version;
   a version-1 manifest with version-2 fields and a version-2 manifest without them are
   refused; a version-1 checkpoint still verifies in `infer` and its output's
   `model_provenance` is byte-identical to the version-1 text; `evaluate` reports
   `declared_angles` and `declared_frame_index_basis` for version 2 and `"not recorded"`
   for version 1 (a version-1 model trained on a 3-D `generate` file included); the warning
   appears exactly for moving-average, 2-D and 3-D, with `"inferred"` or `"unknown"`.
10. **Held-out across channels.** A model trained on one channel and evaluated on the full
    stack counts every frame (documented); the workaround case of §4 is pinned as the
    documented false `disjoint_by_identifiers` when the workaround stack carries the
    acquisition's `acquisition_id`.

## 9. Migration

Additive except the 3-D `infer` refusals and the out-of-vocabulary refusal of §6. CHANGELOG and QUICK_START describe the channel
axis, the declarations and what is (not) checked, the deprecation of the workaround, and
the consequence of one global normalisation; the README limitation ("spectrum-wise
processing") says it holds for channel stacks too.

## Revision log

- **Revision 0 (2026-10-02).** First draft.
- **Revision 1 (2026-10-02), after the first independent audit** (two blocking, eight
  should fix, five minor). The workaround's bound is stated correctly (stride ≥ frames +
  window), and QUICK_START's wrong bound was corrected in a separate change; tests 2 and 3
  are restated so a correct implementation can pass them (tie-free index sets for the
  independent checks, the definition checked separately); the version-1/version-2 manifest
  rules keep stored records byte-exact; the new manifest fields are defined for every
  method and the warning is moving-average only; `infer`'s three changes are listed and
  `times` is forbidden in a stack; §1 and §7 say which declaration is checked where; the
  held-out rules are stated to identify frames; energy is the last axis everywhere; the
  consequence of one global normalisation is stated; the writer has no default
  `order_basis`; boolean and repeated angles are refused; frame-major rows are part of the
  definition and P1 covers the 2-D path only; a claim about the assessed study is
  softened. Owner's decisions (2026-10-02): repeated angle values refused, any order
  allowed; `order_basis` and the angle kind recorded and reported, never checked, the
  provenance schema bumped to version 2; the workaround documented as deprecated, not
  removed; QUICK_START's wrong bound corrected now, separately.
- **Revision 2 (2026-10-02), after the second independent audit** (verdict: adopt with
  named changes; twelve of fifteen earlier findings closed, three partly). Test 3's stride
  accounts for gapped tie-free indices; a version-1 manifest is reported as "not
  recorded", never as "no angles"; 2-D stacks keep accepting `angles` and `times`, and
  `infer` checks only the channel axis, not training-only conditions; the manifest fields'
  value rules are stated, out-of-vocabulary attributes are refused at `train`; a one-channel
  stack is resampled as frame-major rows so it matches the 2-D weights at every energy
  length; every new checkpoint carries version 2; the reported keys are named; the
  float32 condition of the bit-exact reshaping is stated; the tests cover the gaps the
  audit named. Owner's decisions (2026-10-02): 2-D stacks keep `times`/`angles`; `infer`
  adds only the channel-axis checks; out-of-vocabulary attributes refused naming them.

## Confirmation

**2026-10-02 — the owner adopted this design, revision 2, as a whole**, after two
independent audits (the second returning "adopt with named changes", all applied in this
revision). The status line changes to "implemented" only when an implementation is merged.
