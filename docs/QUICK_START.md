# Quick Start Guide

## Install

```bash
pip install -e .          # from the repository root; Python 3.10+
dnndenoiser --help
```

Every command below assumes the installed `dnndenoiser` console command. From a
source checkout without installing, substitute `PYTHONPATH=src python3 -m dnndenoiser.cli`.

## The four-step workflow

```bash
# 1. Generate synthetic training data
dnndenoiser generate -o train.h5 -n 8192 --peak-set C1s_single

# 2. Train a model
dnndenoiser train -d train.h5 -o model.pt --arch bi-LSTM --epochs 30

# 3. Denoise a held-out set: a different seed, so no test spectrum was trained on
dnndenoiser generate -o test.h5 -n 512 --peak-set C1s_single --seed 7
dnndenoiser infer -d test.h5 -m model.pt -o denoised.h5

# 4. Evaluate against the clean reference
dnndenoiser evaluate -d denoised.h5
```

## `generate` — synthetic XPS spectra

```bash
# Peak-set presets: C1s_single, C1s_adventitious, C1s_polymer,
#                   O1s_oxide, Si2p_oxide, N1s_amine, test_doublet
dnndenoiser generate -o data.h5 -n 1000 --peak-set C1s_adventitious

# Noise models: poisson (default), gaussian, mixed, none
# `poisson` draws counts from the Poisson distribution in every bin. A
# Gaussian approximation is available in the library (`add_poisson_noise(..., use_gaussian_approx=True)`)
# and is applied only to bins whose expected count is at least 3, because below that
# its floor at zero biases the bin upward. It is not the default and the CLI does not reach it.
dnndenoiser generate -o data.h5 -n 500 --noise-type poisson --poisson-level 1000

# Sample-to-sample variation (important for generalization)
dnndenoiser generate -o data.h5 -n 8192 \
  --position-jitter 0.5 --width-var 0.2 --intensity-var 0.3

# Angle-resolved 3D data (energy x angle)
dnndenoiser generate -o arpes.h5 -n 100 --n-angles 16 --angle-model cosine

# Time-resolved 3D data (energy x time)
dnndenoiser generate -o time.h5 -n 100 --n-times 32 --time-model exponential_decay

# 4D data (energy x angle x time)
dnndenoiser generate -o 4d.h5 -n 50 --n-angles 16 --n-times 32
```

Backgrounds: `--background {none,linear,shirley}`. Peak shapes are
pseudo-Voigt by default (`--true-voigt` evaluates the Voigt profile exactly via
the Faddeeva function, slower; its width parameters are converted from FWHM and
mixing fraction by an approximation).

## `train` — model training

```bash
# Architectures: FCNN, ResNet-FCNN, 1D-CNN, ResNet-1DCNN, GRU, LSTM, bi-LSTM, Transformer
dnndenoiser train -d data.h5 -o model.pt --arch ResNet-FCNN --epochs 50 --lr 0.001

# Training methods (see README for exact semantics):
#   noise2clean (default) — supervised, needs 'clean' in the HDF5
#   noise2noise           — synthesizes a second independent noisy realization from 'clean'
#   moving-average        — self-supervised from a frame stack; needs no 'clean'
# noise2noise needs the Poisson level the data was generated with, so that the
# synthesized realization sits in the same noise regime as the input:
dnndenoiser train -d data.h5 -o model.pt --method noise2noise --noise-level 1000

# Device selection (auto-detected by default)
dnndenoiser train -d data.h5 -o model.pt --device mps
```

Rule-of-thumb learning rates: RNN/CNN/FCNN families 0.01; ResNet and
Transformer families 0.001. Use at least 8k training samples; 32k is better.

Multidimensional (angle/time) arrays are flattened and trained spectrum-wise.

## `infer` / `evaluate` / `diagnose`

```bash
dnndenoiser infer -d noisy.h5 -m model.pt -o denoised.h5 --batch-size 256
dnndenoiser evaluate -d denoised.h5 -o metrics.json          # reference inside the file
dnndenoiser evaluate -d denoised.h5 --clean ref.h5 --reference external -o metrics.json
```

**What `evaluate` reports depends on what the reference is**, and the reference says so
itself: `generate` declares its `clean` arrays as the *synthetic truth*, `infer` carries
that declaration through, and `evaluate` reads it. The design, and why, is
[docs/design/EVALUATION_REFERENCE_CONTRACT.md](design/EVALUATION_REFERENCE_CONTRACT.md).

| Reference | What is reported |
|---|---|
| synthetic truth | mean MSEs, SNR in dB and SNR gain — an error against the truth |
| an estimate whose overlap with the evaluated data is not established or declared (including `unknown`) | mean MSEs and an *agreement* in dB — not an SNR |
| an estimate whose overlap with the evaluated data is established or declared (e.g. the mean of the same frames) | mean MSEs and relative changes; no dB quantity |
| undeclared | mean MSEs and relative changes; no dB quantity |

A reference that is not the synthetic truth is an estimate of the signal: agreement with
it is not an error against the signal, and a model that returns the mean of the frames
for every frame agrees with that mean perfectly. The JSON output carries an
`evaluation_context` with the reference's declaration, where it came from, its declared
relationship to the data and the model, and the checks made.

**Files written before this version** have no declaration, so `evaluate` reports no SNR
for them. For a synthetic file, declare it:

```bash
dnndenoiser evaluate -d denoised.h5 --reference-origin synthetic_truth \
    --generator "dnndenoiser 0.1.x generate" --units normalised_to_spectrum_max \
    --assert-alignment units -o m.json
```

or reproduce the old output with `--legacy-output --assert-alignment units` (undeclared
references only; it runs the old arithmetic in the input's own dtype, so it is exact on
the same platform and NumPy version and can differ in the last bits elsewhere; it refuses
if an old value would be infinite). Both need `--assert-alignment units` because such a
file carries no `intensity_units` on its arrays, so the units check cannot be made (below);
with `--legacy-output` the reference's units are not declared either. The JSON keys changed;
the CHANGELOG lists them. A reference
estimated from measured data is declared in a JSON file passed with
`--reference-declaration`; its required fields are in the design document. `--overlap`,
`--used-in-model-development` and `--signal-match` state its relationship to the data being
evaluated. `evaluate` never writes to its input files, never ignores `--clean` silently, and
never broadcasts a reference of another shape.

**Alignment is verified before any metric.** `evaluate` compares the energy grid, the
angle and time axes the layout has, the units (`intensity_units` on the evaluated arrays
against the reference's declared units) and the row correspondence, from the metadata both
sides carry. A mismatch is refused, and no option overrides it. A check whose metadata is
absent is refused too, unless you assert it:

```bash
# an external reference without frame identifiers: its rows are the data's rows, you say
# (--reference external is needed when the evaluated file carries its own `clean`, as an
# `infer` output from a `generate` file does)
dnndenoiser evaluate -d denoised.h5 --clean ref.h5 --reference external \
    --assert-alignment rows -o m.json
# one reference spectrum for every row
dnndenoiser evaluate -d denoised.h5 --clean ref.h5 --reference external \
    --shared-reference --assert-alignment rows
# a file written before this version: no units on its arrays, none declared
dnndenoiser evaluate -d old.h5 --legacy-output --assert-alignment units
```

The output lists what was verified, what was asserted and what does not apply (angles and
times for plain spectra), and how the rows were matched: the same file, identifiers
(`frame_index` within one acquisition, plus matching axes), or your assertion. An assertion
for a check that was made, or for an axis the layout does not have, is refused, so a
verified check is never confused with an asserted one. The output also carries a content
digest (`dnd-digest-1`) of the evaluated arrays and of the reference, each with its
coordinates and stored declaration; a digest identifies what was compared, not where it
came from.

### `diagnose` — how the output depends on the input

```bash
dnndenoiser diagnose -d frames.h5 -m model.pt -o report.json
dnndenoiser diagnose -d frames.h5 -m model.pt -o report.json --probe 285:0.6:3 --probe 287:0.6:-2
```

On a **frame stack** only (rows of a `noisy` file are different spectra by default;
convert repeated acquisitions with `write_frame_stack`). The model is applied exactly as
`infer` applies it (resampling to the network length, the checkpoint's normalisation and
its inverse, dropout off), on `--device cpu` by default. Per channel it reports:

- the **contraction ratio** `C`: the output's frame-to-frame variation over the input's.
  `C = 1` for a model returning its input, `C = 0` for one returning a fixed spectrum. Its
  denominator holds the noise *and* any real change across frames, and an oversmoothed or
  wrong spectrum returned for every frame also gives 0. **It is not a noise reduction and
  must not be read as one in dB**: that would be a reference-free SNR;
- the **injection response** `R` and the **area ratio** `A`, for Gaussian probes added to
  every frame: amplitude `k·σ`, `σ` the frames' noise scale from differences of frames with
  adjacent `frame_index` values. `R` is the fraction of the probe that reaches the output,
  projected on the probe; attenuation, broadening and shift all reduce it, and `R = 1` is
  what returning the input gives, so larger is not better. It is not confined to [0, 1]: an
  output that moves against the probe gives a negative value, one that amplifies it a value
  above 1, and neither is clipped. `A = 1` does not show that the
  areas of real features are preserved. By default five probes (10 to 90 % of the way along
  the energy axis, FWHM 3 % of its span, `k = 3`), each reported separately; `--probe
  E0:FWHM:k` replaces them (a negative `k` is a dip; for a negative `E0` use the
  `=` form, `--probe=-5:0.6:3`).

Medians and `[q25, q75]` are over frames; the spread is not an uncertainty. The report
also carries the held-out status of the frames from the model's manifest, the model's
identity and a digest of the frames as stored. None of these is an accuracy, a noise
reduction or an SNR, and no threshold says a model has collapsed. **Observed, not
established:** on a model's own stationary training stack, a moving-average model on toy
data (one seed) responded very little to a probe away from the peaks and only partly on them; this
may be the ordinary outcome of the method, so `diagnose` is not a collapse detector.
Design and two independent audits:
[docs/design/OUTPUT_CONTRACTION.md](design/OUTPUT_CONTRACTION.md).

## HDF5 schema

| Dataset | Shape | Notes |
|---------|-------|-------|
| `noisy` | (n, …, energy) | input spectra; extra axes (angle/time) allowed. Attributes `intensity_units` and, optionally, `acquisition_id` |
| `clean` | (n, …, energy) | reference; required for noise2clean/noise2noise and evaluate. Attributes `reference_schema_version` and `reference_origin` declare what it is |
| `energy` | (energy,) | energy axis; `evaluate` compares it between the data and an external reference |
| `denoised` | (n, …, energy) | written by `infer`, with the same attributes as `noisy` |
| `angles` / `times` | (angles,) / (times,) | written by `generate` for 3D/4D data; they name the coordinate axes (a 3D layout has one, a 4D layout is `(n, times, angles, energy)`) |
| `frame_index` | (n,) | optional; integer acquisition order, carried by `infer`; `evaluate` matches rows by it within one acquisition |

### Frame stacks — `--method moving-average`

A different layout, for training from repeated acquisitions with **no clean
reference**. Each frame's target is the mean of its `--window` temporally
nearest *other* frames (leave-one-out).

| Dataset | Shape | Notes |
|---------|-------|-------|
| `frames` | (n_frames, energy) or (n_frames, n_angles, energy) | the acquired frames, in any row order; energy is the last axis |
| `energy` | (energy,) | energy axis |
| `frame_index` | (n_frames,) | acquisition order; **integer and unique**; one per frame, shared by its channels. Optional attribute `order_basis`: `recorded`, `inferred` or `unknown` |
| `angles` | (n_angles,) | required with a 3-D `frames`: one finite, non-repeated value per channel, in the order of the axis; attributes `angle_kind` (`emission`, `analyser` or `other:<description>`) and `angle_units` (`deg`) |

`frame_index` is what "temporally nearest" is measured on, so a file that
omits it cannot be told from one whose frames were shuffled. **Duplicate
indices are rejected**: a frame is excluded from its own neighbourhood by
position, not by index value, so two frames sharing an index become distance-0
neighbours of each other and each leaks straight into the other's target.

```bash
dnndenoiser train -d stack.h5 -o model.pt --method moving-average \
    --window 5 --epochs 50
```

The optimiser, schedule, loss and architecture are fixed — they are part of the
method being reproduced — so `--arch`, `--lr`, `--lr-drop-period`,
`--lr-drop-factor`, `--scheduler`, `--warmup-epochs`, `--weight-decay`,
`--grad-clip`, `--hidden-units`, `--encoder-dim` and `--noise-level` are
**refused rather than ignored**, in whichever form they are written (`--lr 0.05`,
`--lr=0.05`, or an abbreviation argparse would accept). `--epochs`,
`--batch-size`, `--seed`, `--window`, `--device` and `--threads` still apply.

Stacks that are not 256 points are resampled — by `train`, and by `infer` in the
same way, so the stack a model was trained on can be passed to it as it is:

```bash
dnndenoiser infer -d stack.h5 -m model.pt -o denoised.h5
```

`infer` reads a frame stack's `frames` when the file has no `noisy`. When it
resamples, every array it writes — `noisy`, `clean`, `denoised` and `energy` —
is on the resampled grid.

**Angle channels.** To train one model on the frames of several angle channels, give
`frames` a channel axis, `(n_frames, n_angles, energy)`, and name the channels in
`angles`. Each frame's target is then built from its `--window` nearest other frames **in
its own channel**, whatever the window; the min–max normalisation is taken over every
channel together (so a channel much dimmer than the brightest contributes little to the
loss, and the trained range is the whole stack's). Training rows are the (frame, channel)
pairs, frame-major. `infer` checks the channel axis and carries `angles` and its
attributes; `evaluate` compares `angles` with a reference that carries them.

```python
from dnndenoiser.data.frame_stack import write_frame_stack
write_frame_stack("stack.h5", frames, energy, frame_index,           # frames: (n, A, E)
                  angles=angles, angle_kind="emission", angle_units="deg",
                  order_basis="inferred")
```

**What is declared and what is checked.** `angle_kind`, `angle_units` and `order_basis`
are recorded — in the stack, in `infer` output and in the model's manifest
(`evaluation_context.model.training_data.declared_angles`,
`declared_frame_index_basis`) — and **never checked**. A permuted, offset or reversed
channel axis changes no target, so training cannot detect it; only the angle *values* are
compared, by `evaluate`, against a reference that carries them. `order_basis` says
whether the order came from the instrument (`recorded`) or was derived by you
(`inferred`); `train --method moving-average` warns when it is not `recorded`, because
every target is only as right as the order (other methods do not use the order). Leave it out rather than write `recorded` by default.

**Several channels in one 2-D stack (deprecated).** Before the channel axis, the recipe
was one 2-D stack with channel *k* at the indices
*k* × *stride* + *t*, with *t* = 0, 1, … the acquisition order within the channel and

    stride ≥ (the largest channel's number of frames) + --window

and `--window` smaller than the number of frames in every channel. Each frame's
`--window` nearest others are then in its own channel, and the min–max normalisation is
taken over all channels together. **A smaller stride fails silently:** the first frame of
a channel is only *stride* − (*n* − 1) away from the last frame of the previous one, so
with a stride of just *n* + 1 it takes that frame as a neighbour from a window of 3 on,
and at 2 when a tie falls that way (an earlier version of this page gave that stride as
safe). **So does a window as large
as a channel:** `train` limits the window by the frames in the whole stack, not in one
channel, so the nearest others of a frame run out of its own channel and reach into the
next. These indices are not acquisition order: do not give such a stack the
acquisition's `acquisition_id`, or `evaluate` can compare them with the acquisition's own
`frame_index` and report its frames as disjoint from the training data. Numbering the channels one after another
instead would make the last frame of one channel a neighbour of the first frame
of the next.

### Loading a checkpoint safely

`infer` does not unpickle a checkpoint by default. `torch.load`'s full
unpickler runs code from the file, so a checkpoint someone sent you is a
script someone sent you. Files written by v0.1.2 and later need nothing
special; a checkpoint from v0.1.0 or v0.1.1 stored its energy axis as a NumPy
array and needs `--trust-checkpoint`, which reaches the unpickler. Use it only
on files you produced or otherwise trust.

**Evaluating this is not straightforward.** A measured stack has no clean
reference, and the obvious substitute — a mean over the same frames — is *not*
independent of targets built from subsets of those frames. An SNR computed
that way is not a held-out result and must not be reported as one. A mean
reference also rewards an output that barely changes from frame to frame: a
model that returns nearly the same spectrum for every frame sits close to the
mean and scores well by SNR against it, however little it tells about any one
frame. How the output varies across frames cannot tell such a model from a good
denoiser either, since both vary little; `dnndenoiser diagnose` (its section is under
`infer` / `evaluate` / `diagnose`) adds a known
change to the input and reports how much of it reaches the output (the injection
response). See
[the preregistration](preregistration/P1-selfsupervised-moving-average.md) for
what the port does and does not establish.

### What a checkpoint records about its training

`train` writes a provenance manifest into the checkpoint, and `infer` carries it into
its output, where `evaluate` reports it under `evaluation_context.model`. It records the
training data's content digest and identifiers, the targets, the options and the settings
actually used, every seed, the software versions, the device, and the commit when the
package runs from its own git checkout. A `model_digest` binds it to the weights and the
normalisation: `infer` refuses a checkpoint edited without updating it, and `evaluate`
refuses an output whose carried records do not belong together. The design is
[docs/design/PROVENANCE_MANIFEST.md](design/PROVENANCE_MANIFEST.md).

- **Identifiers travel with the model.** The training file's `acquisition_id`, its
  reference declaration (including free-text conditions) and an `other:<description>`
  angle kind are copied verbatim into the manifest, so sharing a checkpoint or any of its outputs shares them. Keep names of
  people, places and specimens out of them, and give each acquisition its own
  `acquisition_id`: one reused for two acquisitions makes them look like the same data.
- **Tools that edit files must keep the metadata true.** A tool that takes a subset of
  rows and renumbers `frame_index`, or replaces `noisy` or `clean` but keeps
  `input_array_digest` or `signal_identity`, leaves metadata that describes other data.
  Drop or rewrite it.
- **The commit** is recorded only for this package's own checkout, never for another
  repository the package happens to sit in; a repository that vendors the package at a
  tracked top-level `src/dnndenoiser/` is recorded as such a checkout.
- A manifest records what was run. It does not make training reproducible.

**Was the evaluated data training data?** `evaluate` establishes it from the manifest and
reports it as `held_out_status` (also a top-level key), with `rows_in_training`:

| `held_out_status` | Means |
|---|---|
| `not_held_out` | at least one evaluated row (first axis) was a training row: the same array, or the same `acquisition_id` with intersecting `frame_index`. The dB keys become `training_fit_*` and the heading *Fit to the training data*; `--legacy-output` is refused |
| `disjoint_by_identifiers` | the identifiers show other rows. Not proof of independence: a copy under another identifier is not detected |
| `unknown` | nothing could be established (no manifest, no identifiers, or one acquisition without `frame_index`, reported as `same_acquisition_rows_unidentified`) |

`generate` writes `frame_index`, an `acquisition_id` on `noisy` and a `signal_identity` on
`clean`, so this works for synthetic data without anything to declare: a test file made
with the same seed and settings and a smaller `-n` is the training file's first rows
(`not_held_out`); one with the same seed and another noise level holds the same signals
(`disjoint_by_identifiers`, and the reference is reported as the model's training target).
Its noise is **not independent** of the training noise: the same seed draws it from the
same random stream, so Gaussian noise repeats rescaled, mixed noise stays highly
correlated when only `--gaussian-std` changes, and Poisson noise at a nearby level can be
strongly correlated. `disjoint_by_identifiers` says only that the arrays differ. Use
another `--seed` for a held-out synthetic test set.

### What `--seed` reproduces

Two tiers ([docs/design/REPRODUCIBILITY.md](design/REPRODUCIBILITY.md)):

- **Promised, bit for bit** — the same `model_body_digest` from two training processes —
  on the same machine, in the same installed environment (Python and the same builds of
  torch, NumPy and h5py), with the same code (one commit with a clean tree, or one installed
  release), `--device cpu`, an integer `--seed`, the same thread count, and the same input
  and arguments. `--device auto` picks MPS or CUDA where available, so pass `--device cpu`;
  pass `--threads N` to fix the thread count (on the development machine the Transformer's
  and bi-LSTM's weights changed with it).
- **Not promised** — anything else: another machine or environment, MPS or CUDA, another
  thread count, other code, or no seed. Such runs may agree or differ; nothing is claimed.

The manifest records the conditions it can (`software.torch_threads`, the device, the seed,
the code, the input digest, the arguments) and a label, `reproducibility.tier`:
`tier-1-eligible` for a cpu run with a seed and a tree that is not dirty, `tier-2`
otherwise. It cannot record the machine or the builds behind a version string, so two
manifests can show that two runs were *not* comparable, never that they were. `evaluate`
reports `torch_threads` and the tier under `evaluation_context.model`. Checkpoints store
their weights on the CPU whatever device trained them.

## Python API (minimal example)

```python
import torch
from dnndenoiser import DenoisingNetwork

model = DenoisingNetwork(num_features=256, num_hidden_units=100,
                         layer_type='ResNet-FCNN', encoder_output_dim=64)
ckpt = torch.load('model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

with torch.no_grad():
    out = model(noisy_batch)          # (batch, 256) float32 tensor
    denoised = out[0] if isinstance(out, tuple) else out
```

## Troubleshooting

- `ModuleNotFoundError: dnndenoiser` — run `pip install -e .` from the
  repository root, or set `PYTHONPATH=src`.
- `command not found: dnndenoiser` — the console script is installed into the
  active environment; activate the venv you installed into.
- Error: method requires clean data — noise2clean/noise2noise need a `clean`
  dataset in the training HDF5 (the `generate` command always writes one).
- GPU check: `python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"`
