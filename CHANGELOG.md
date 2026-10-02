# Changelog

All notable user-visible changes to `dnndenoiser` will be documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A trained model records how it was made.** `train` writes a provenance manifest
  into the checkpoint: a content digest of the training data and the identifiers it
  carries (`acquisition_id`, `frame_index`), what the targets were made from, every
  recorded option as parsed, the settings the run actually used (for `moving-average`,
  its fixed optimiser and schedule rather than the parser defaults), every seed used
  (including noise2noise's target seeds when `--seed` is not given), the software
  versions, the device, and the commit of this package's own checkout (or `"unknown"`).
  A `model_digest` binds it to the weights, the normalisation and the configuration.
  `infer` refuses a checkpoint whose digest no longer matches, carries the manifest into
  its output, and records a digest of its input as read; `evaluate` checks that the
  carried records belong together and reports the model's identity under
  `evaluation_context.model` (it was always `"unknown"`). A record that is present is
  valid or refused, never read as absent; checkpoints written before this version have
  none and work as before. The manifest records what was run; it does not make training
  reproducible. Design and three independent audits:
  `docs/design/PROVENANCE_MANIFEST.md`.
- **`evaluate` says when the evaluated data were training data, and names the result
  accordingly.** From the model's manifest it establishes `held_out_status`: `not_held_out`
  when the evaluated noisy array is the training input (by content, also after `infer`
  resampled it) or when its rows share the training acquisition and frames, with
  `rows_in_training` counting them; `disjoint_by_identifiers` when the identifiers show
  other rows (not proof of independence); `unknown` otherwise. It also establishes
  `used_in_model_development: yes` when the reference was the model's training target
  (by content, or by a synthetic signal identity with intersecting rows), and reports
  `shares_source_with_training_data` for an estimate computed from the training frames.
  A declaration contradicting an established relationship is refused. `held_out_status`
  and `rows_in_training` are top-level keys of every normal output.
- **`generate` records which draws a file holds**: `frame_index` (`0 … n-1`), an
  `acquisition_id` on `noisy` and a `signal_identity` on `clean`, each a hash of the
  settings that change those arrays. The same seed and settings with a smaller `-n` are
  recognised as a subset; the same seed with another noise level is recognised as the
  same signals (for Gaussian noise the noise draws repeat too, rescaled; another `--seed`
  gives held-out synthetic data).
- **[`docs/WHEN_TO_TRUST.md`](docs/WHEN_TO_TRUST.md)** — what has been measured about
  where a trained model stops being trustworthy, one question at a time. The first
  answer is how far a shift in energy can be trusted, from the P2-A record: the
  conditions, what to do, and what was not established. Every number on the page is
  anchored to the record or to its registered design; a test recomputes the record's
  and matches the design's, within limits the preregistration lists. The prose around
  the numbers is checked by review, not by the test.

### Changed

- **Breaking: an evaluation of a model on its own training rows is renamed.** When
  `held_out_status` is `not_held_out`, the dB keys carry the prefix `training_fit_`
  (`training_fit_snr_gain_mean`, …, `training_fit_agreement_db_change_std`), the printed
  heading says *Fit to the training data*, a caveat states how many rows (first axis)
  were training rows, and `--legacy-output` is refused. A script that reads `snr_*` from an
  evaluation of training data no longer finds it, which is the point. An external
  synthetic-truth reference that cannot be matched by identifiers is now refused with
  the true reason (a truth declaration names no acquisition).
- **Breaking: `evaluate` reports an SNR only against a declared synthetic truth.**
  A reference now declares what it is (`generate` declares its `clean` arrays as the
  synthetic truth; `infer` carries the declaration through), and every reported
  quantity is named by it: an SNR against the synthetic truth, an *agreement* in dB
  against a declared estimate from other data, and mean MSEs and relative changes —
  no dB quantity — against an estimate built from the evaluated data or an undeclared
  reference. Files written before this
  version are undeclared and get no SNR by default; declare them with
  `--reference-origin synthetic_truth --generator … --units …`, or reproduce the old
  output with `--legacy-output`. It runs the old arithmetic in the input's own dtype
  (float32 for files written by `generate`), so its values are exact on the same platform
  and NumPy version and can differ in the last bits elsewhere, where float32 summation
  differs between NumPy versions. The JSON output gains `evaluation_context`, refuses NaN and Infinity, and
  renames and adds keys:

  | Before | Now | Note |
  |---|---|---|
  | `mse_input_mean` | `mse_in_mean` | float64 |
  | `mse_output_mean` | `mse_out_mean` | float64 |
  | `mse_reduction_mean` | `mean_relative_mse_change_per_spectrum_pct` | excludes spectra with zero input MSE, counted in `per_spectrum_excluded_zero_input_mse` |
  | — | `mse_difference`, `relative_mse_change_aggregate_pct` | new |
  | `n_samples` | `n_spectra` | |
  | `snr_*` | `snr_*` against a synthetic truth only; `agreement_db_*` against an estimate whose overlap with the evaluated data is not `overlap` (that includes `unknown`); none otherwise | |

  `--legacy-output` writes the old keys. `evaluate` also no longer ignores `--clean` when the input has its
  own reference, and refuses a reference of another shape instead of broadcasting it.
  `SyntheticGenerator.save_hdf5` now requires `reference_declaration` and
  `intensity_units`. Design, four independent audits and the owner's adoption:
  `docs/design/EVALUATION_REFERENCE_CONTRACT.md`.
- **`evaluate` verifies alignment before any metric, or refuses.** The energy grid, the
  angle and time axes the layout has, the units (`intensity_units` on the evaluated
  arrays against the reference's declared units) and the row correspondence between the
  evaluated arrays and the reference are checked from the metadata both carry. A
  detected mismatch — an energy grid off by more than 1e-6, a reversed angle or time
  axis, different units, frame identifiers that disagree within one acquisition — is
  refused, and no option overrides it. A check whose metadata is absent is refused too
  unless asserted with `--assert-alignment energy,units,rows,angles,times`; an assertion
  for a check that was made, or for an axis the layout does not have, is refused. The
  output lists `alignment_verified`, `alignment_asserted` and `alignment_not_applicable`
  separately, with `row_correspondence` (`same_file`, `identifiers` or `asserted`). A
  single reference spectrum shared by every row needs `--shared-reference`. **An
  undeclared reference has no declared units, so the two migration routes now need
  `--assert-alignment units`**: `--legacy-output --assert-alignment units`, and
  `--reference-origin synthetic_truth --generator … --units … --assert-alignment units`
  when the file's arrays carry no `intensity_units` (files written before this version).
  The output also carries content digests of the evaluated arrays and of the reference
  (`dnd-digest-1`: SHA-256 over the arrays, their coordinates and the stored
  declaration) under `evaluation_context.digests`; a digest identifies what was
  compared, it does not establish where it came from. This completes the design's
  phase 2.

### Fixed

- **`train --epochs 0` crashed after "training"**, with no loss to save. It is now
  refused. A training file with non-finite values in any stored array or coordinate, a
  non-numeric coordinate, or a malformed `frame_index` is refused too, naming the dataset,
  before anything is trained.
- **A content digest of a 0-d array hashed it as shape (1,)**, against the format's
  definition. No array `evaluate` digests is 0-d, so no evaluation output changes; a
  model's BatchNorm counters are.
- **`infer` could not take the frame stack a moving-average model was trained on.**
  It read only a `noisy` dataset, where a stack holds `frames`, and it did not
  resample, where `train` resamples a stack that is not 256 points — so a measured
  1024-channel stack failed with a `KeyError`, or with a PyTorch shape error that did
  not say why. `infer` now reads `frames` when there is no `noisy`, and resamples to
  the checkpoint's length with the function `train` uses; every array it writes is
  then on the resampled grid. Found by running the self-supervised route on measured
  frames from the public repository alone.
- **`train --seed` was ignored by every method but `moving-average`.** Its help
  promised that the seed is passed to torch before the model is built; the
  noise2clean and noise2noise paths built the model and shuffled batches from
  torch's unseeded stream, and noise2noise synthesized its targets from generators
  fixed at a default seed. `--seed` now seeds all three on every method. It still
  does not make training reproducible on every device. Found by an external audit.
- **`infer` dropped a frame stack's `frame_index`**, so the denoised frames could not
  be matched back to the acquisition. It now writes it to the output.
- **`evaluate` printed the MSE of measured-scale data as `0.000000`.** It now prints
  it in scientific notation. The saved JSON was already exact.

- **The documentation did not say that GRU, LSTM and bi-LSTM read the spectrum as
  one time step.** Their names suggest recurrence along the energy axis; each takes
  the whole spectrum as a single step (sequence length 1), so the recurrence never
  runs. The README and the reference benchmark's README now say so, and a test pins
  it. Nothing about the models or the measured numbers changed.
- **Text files were opened in the locale's encoding, so a Japanese Windows
  (cp932) machine failed three tests and could mis-write output.** Every
  `open()`, `read_text()` and `write_text()` in the package and the tests now
  names UTF-8: `evaluate -o`'s metrics JSON, `SyntheticGenerator.save_manifest`
  and the tests reading `pyproject.toml`, `CITATION.cff`, `CHANGELOG.md` and the
  P1 fixture. `PYTHONUTF8=1` was the only workaround before. CI now runs on
  Windows on every push, and with `PYTHONWARNDEFAULTENCODING=1` on every OS, so
  a call without an encoding fails there.

### Documentation

- **QUICK_START says how to train one model on several channels of one stack** —
  the emission-angle channels of an angle-resolved measurement, for example — by
  offsetting each channel's `frame_index`, and why numbering them one after another
  would mix neighbours across channels. A test shows both. The recipe holds only while
  `--window` is smaller than every channel's number of frames; QUICK_START now says
  so, and a test shows it failing beyond that.
- **README Limitations: synthetic noise is independent between energy channels.**
  Measured detectors can spread one count over neighbouring channels, so a model
  trained on synthetic spectra can meet correlated noise it was not trained on.
- **QUICK_START warns that a mean reference rewards an output that barely changes
  from frame to frame**, so an SNR against it can rank a model that returns nearly
  the same spectrum for every frame above one that does not.

## [0.1.2] - 2026-09-22

### Added

- **[`docs/FROM_THE_PAPERS.md`](docs/FROM_THE_PAPERS.md)** — a route from the
  published JVST A and SIA papers to a reader's own measurements: read the files
  with `toyomacro`, put the frames in the frame-stack layout, train the
  self-supervised method on them, and take the depth-profile metrics from the
  papers' Zenodo deposit rather than from here. It is as explicit about what is
  *not* available — the inversion solver, the trained weights and the ~60 GB of
  raw frames were not deposited — as about what is, so the papers' figures
  cannot be reproduced end to end from public artifacts and the page says so.

### Fixed

- **Packaging metadata: the license and the project URLs were both absent.** An
  installed copy declared no licence terms and offered no route back to the
  source, the changelog or the archive. `license = "MIT"`, `license-files` and
  five `[project.urls]` entries now reach the wheel's `METADATA`; the archive
  URL is the concept DOI, so it does not go stale at the next release.

- **CI actions are pinned to commit SHAs, and the workflow is read-only.** Every
  action was on a movable tag, so `@v4` meant "whatever that repository decides
  v4 is when CI next runs"; `openjournals-draft-action` was on `@master`, with
  no release to pin to at all. The workflow also declared no `permissions`,
  taking whatever the repository default grants. It is `contents: read` now, and
  a job that ever needs more has to say so where it can be seen.

- **The Transformer refuses an unusable spectrum length at construction.** It
  reads the spectrum as patches of 8 points, so `num_features` must be
  divisible by 8. That used to surface on the first forward pass as
  `RuntimeError: shape '[2, 31, 8]' is invalid for input of size 500`, naming
  neither the constraint nor the architecture — after the model was built and
  training set up. It is now a `ValueError` at construction that names the
  patch size, the nearest usable lengths, and the fact that every other
  architecture accepts any length. Documented in the README's architecture
  table.

- **The evaluation metrics are importable, and are what the tests test.**
  `compute_snr` and `compute_mse` were nested inside `cmd_evaluate`, so
  `tests/test_evaluation.py` carried its own copies and tested those. The
  copies were not the same function: the test's `compute_snr` divided by
  `noise_power + 1e-10` where the shipped one floors with
  `np.maximum(noise_power, 1e-10)`, and the two disagree by up to 0.41 dB near
  perfect reconstruction — exactly where a denoiser is doing best. A third
  copy, `compute_psnr`, was tested and is not part of the package at all. The
  metrics are now `dnndenoiser.cli.compute_snr` / `compute_mse`, the tests
  import them, and the floor convention is tested as the convention it is.

- **`infer` applies the normalisation a checkpoint records, and inverts it.**
  The moving-average path stores the constants it trained under; `infer` read
  neither. A model trained on data scaled to [0, 1] was fed raw counts, and its
  output was written beside a `noisy` dataset in counts, in a space the file did
  not name. Both directions are now applied, and the command states that the
  constants are the training stack's — applying them to another dataset is an
  assumption, not a conversion.
- **`infer` reads the model's shape out of the checkpoint instead of guessing.**
  The two training paths wrote it under different key names (`n_features` from
  noise2clean, `num_features` from the moving-average path in v0.1.1) and only
  the first set was read, so the second fell through to defaults. Both
  spellings are read now, and a checkpoint that records neither is an error
  naming the file rather than a shape mismatch inside `load_state_dict`.

### Security

- **`infer` no longer unpickles a checkpoint unless asked to.** `torch.load`'s
  full unpickler executes code from the file, and the CLI invites the risky
  case: `infer -m someone-elses-model.pt` would have run whatever that file said
  to run. v0.1.0 and v0.1.1 both shipped with it unconditional. Loading is now
  restricted by default; the unpickler is reached only with
  `--trust-checkpoint`, whose help says what accepting it means.

  Checkpoints written from this version on need no flag. Those from v0.1.0 and
  v0.1.1 stored the energy axis as a NumPy array, which the restricted loader
  refuses — the error names the flag rather than leaving it to be found. The
  energy axis is now stored as a tensor.

## [0.1.1] - 2026-09-21

### Added

- **A self-supervised training method that needs no clean reference.**
  `dnndenoiser` implements the leave-one-out moving-average self-supervised
  training target and the ResNet-FCNN training loop of the archived
  `arhaxpes_denoise` reference implementation (Zenodo
  [10.5281/zenodo.22092109](https://doi.org/10.5281/zenodo.22092109) v1.0.0).
  On that deposit's own synthetic fixture — 200 Poisson frames of a single
  Gaussian core level on a flat background, 16 held-out frames, CPU — the
  port's targets are exactly equal to the reference's for `W ∈ {1, 2, 5, 10}`
  on contiguous acquisition order, for `W = 5` on a permuted one, and on the
  window-clamp, error and duplicate-index cases. At `W = 5` its trained outputs
  are bit-identical to the reference's at seed 0 (relative L∞ 0.0, against a
  registered bound of 1e-4), and the two implementations' clean-referenced
  output SNR differ by 0.000 dB on each of five further seeds (registered bound
  0.5 dB). ResNet-FCNN `state_dict`s interchange between the two packages
  without renaming at `num_features=256, num_hidden_units=100,
  encoder_output_dim=64`.

  **The two agree to the bit because they share a code lineage** — the deposit
  vendored its network from this project — not because an unrelated
  reimplementation would; the registered tolerances exist for that case. The
  evidence does not pin the gradient-clip threshold: removing clipping entirely
  reproduces the reference exactly on this fixture. **No measured data was
  used, and nothing here measures how well either implementation denoises.**

  Acceptance criteria were registered before the implementation existed and
  audited twice — once before it was written, once after — in
  [`docs/preregistration/P1-selfsupervised-moving-average.md`](docs/preregistration/P1-selfsupervised-moving-average.md),
  which also records what the criteria do *not* establish.

- An HDF5 frame-stack schema (`frames`, `energy`, `frame_index`) for repeated
  acquisitions, with a reader that rejects duplicate acquisition indices.
- `train --method moving-average --window W`. Training only; `infer` does not
  read the frame-stack schema yet.
- `train --seed`.

### Fixed

- The sdist shipped `tests/` without `tests/fixtures/`, so test collection
  aborted and no test ran from the released archive.

## [0.1.0] - 2026-09-21

First public release, published from a single import of the audited tree. The
Changed and Fixed entries below record work done before that import, against
the unreleased code; they are kept because they change what the software
produces, and one of them changes it for a given seed.

### Added

- First public release: the `dnndenoiser` package and its `dnndenoiser` command
  (`generate` → `train` → `infer` → `evaluate`), the test suite, the
  documentation, and `benchmarks/reference/` with its machine-readable record.
- Changelog and contribution templates for future public development.

### Changed

- `train --method noise2noise` now requires `--noise-level`: the Poisson level
  the data was generated with, in the units of `generate --poisson-level`. It
  sets the noise regime of the synthesized second realization, which has to
  match the input, and the data file does not record it. Levels that cannot be
  honoured are refused rather than silently training against a clean or
  unusable target.

### Fixed

- `train --method noise2noise` failed with `AttributeError` before training
  started: it read a Poisson level off a configuration class that does not carry
  one. The advertised option had never worked.
- `train --method noise2noise` now synthesizes the second realization at the
  same noise magnitude as the data. The generator and `Noise2Noise` parameterise
  Poisson noise differently — counts scale as the square of the ratio in one and
  linearly in the other — so a level carried across unchanged produced a target
  `sqrt(10000 / level)` times noisier than the input: tenfold at the generator's
  default level of 100. The conversion is applied by the command; callers using
  `Noise2Noise` directly still state the level in its own units.
- `train` records the stated noise level in the checkpoint, and warns when the
  clean spectra are not each normalized to their own peak, which is what the
  conversion assumes.
- The synthetic generator draws true Poisson counts by default. It used to
  default to a Gaussian approximation, chosen once from the **peak** expected
  count and then applied to every bin. The approximation can go negative and
  negatives were floored at zero, so a bin came back biased upward by an amount
  that depends only on its own expected count — and a bright peak licensed the
  approximation for the dark bins where it is worst. At `--poisson-level 1000` a
  bin at 1% of the peak was 8% high and one at 0.1% of the peak was 82% high.
  The approximation is still available as `use_gaussian_approx=True`, now
  applied per bin and only where the expected count is at least 3, which holds
  its relative bias under 1%. **`generate` produces different data than before
  for the same seed and level.** This is the same defect the Noise2Noise entry
  below records, in the other place it occurred; `benchmarks/reference/` names
  the superseded rule explicitly so its published measurement stays exactly
  reproducible.
- Noise2Noise realizations no longer floor the Poisson rate at 0.01 counts.
  Bins with fewer expected counts than that came back biased upward, so the
  training target did not return the clean signal on average — the property
  Noise2Noise rests on. The rate is floored at zero, matching the synthetic
  generator and the other training methods.

[Unreleased]: https://github.com/stoyoda0012-cyber/dnndenoiser/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/stoyoda0012-cyber/dnndenoiser/releases/tag/v0.1.2
[0.1.1]: https://github.com/stoyoda0012-cyber/dnndenoiser/releases/tag/v0.1.1
[0.1.0]: https://github.com/stoyoda0012-cyber/dnndenoiser/releases/tag/v0.1.0
