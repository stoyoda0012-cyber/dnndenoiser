# Reference denoising benchmark

A deterministic, re-runnable measurement of per-architecture denoising performance on
this package's own synthetic spectra. It exists so that any per-architecture number
shown to a user — in the README or anywhere else — has a record behind it that
satisfies `AGENTS.md` §6.

Nothing in this directory is distributed in the sdist or wheel (`AGENTS.md` §3).

## Files

| File | What it is |
|---|---|
| `reference_benchmark.py` | The measurement. Run it to produce a record. |
| `results/reference_benchmark.json` | **The primary record.** The only place numbers live. |
| `results/reference_benchmark_ntrain6144.json` | A second record at 2.67× the training data, used as a robustness check on training-set size (see below). |
| `render_report.py` | Regenerates the Markdown report *from* a record. |
| `report.md` | Generated output of `render_report.py --write`. Do not edit by hand. |

The one rule that keeps this honest: **numbers are never typed by hand** — and
nor is anything else in `report.md`. Every line of it comes out of
`render_report.py`; a paragraph added to the file by hand is dropped the next
time the report is regenerated, which is how six lines of the self-check
commentary nearly went missing on 2026-09-11. The table and
the GUI strings are rendered from the JSON, so re-running the benchmark and re-running
the renderer keeps every displayed figure tied to the run that produced it. If you find
a per-architecture number anywhere in this repository that did not come out of
`render_report.py`, it has no provenance and should be treated as withdrawn.

`render_report.py` recomputes, from the per-run raw numbers, everything it is about
to render — the aggregate means, the across-seed standard deviations, the input SNR
that labels each column, the parameter counts, and every field of the paired
comparison including the t statistic, the raw and Holm-adjusted p values, Cohen's dz
and the seeds-favouring count — and refuses to render if any of them disagree. It
reports all disagreements together rather than stopping at the first.

That claim is itself under test: `tests/test_reference_benchmark_record.py` tampers
with a copy of the record one field at a time and requires the guard to reject each
one. An earlier version of the guard checked only the mean and the seed count while
this paragraph claimed it checked everything, so tampered standard deviations and a
fabricated paired table rendered without complaint.

## Re-running it

```bash
pip install -e ".[dev]"

# Full run, as recorded (device auto-selects cuda > mps > cpu).
python benchmarks/reference/reference_benchmark.py \
    --seeds 5 --n-train-per-level 768 --n-test-per-level 512 \
    --matched-budget --matched-seeds 5

# The training-set-size robustness check (primary condition only, 2.67x the data).
python benchmarks/reference/reference_benchmark.py \
    --seeds 5 --n-train-per-level 2048 --n-test-per-level 512 \
    --skip-determinism-check \
    --output-name reference_benchmark_ntrain6144.json

# Regenerate the report and the GUI strings from the records. `--compare-record`
# is not optional: without it `--write` renders a report with the robustness
# section missing, and silently overwrites the one that had it.
python benchmarks/reference/render_report.py --write \
    --compare-record benchmarks/reference/results/reference_benchmark_ntrain6144.json

# Fast smoke test of the whole pipeline (seconds; the numbers are not meaningful).
# Give it an output directory: the default is results/, and a quick run written
# there overwrites the published record.
python benchmarks/reference/reference_benchmark.py --quick --output-dir "$(mktemp -d)"

# Lint, matching the project's configured rules.
ruff check benchmarks/reference/
```

Useful flags: `--device {auto,cpu,cuda,mps}`, `--output-dir`, `--output-name`,
`--epochs-cap`, `--skip-determinism-check`.

Note that CI's ruff step runs over `src/` and `tests/` only, so this directory is not
linted automatically; run the command above yourself after editing it.

### Expected wall clock

On the Apple-silicon machine that produced the committed records (`mps` backend), the
first command above — 80 training runs, both conditions — took around half an hour,
roughly two thirds of it in the primary condition. The larger training-set robustness
check (40 runs at 2.67× the data) takes comparable time again. `--quick` finishes in
well under a minute. Each record's own `total_wall_clock_seconds` and each run's
`train_seconds` are authoritative for the machine that produced it; a CPU-only machine
will be several times slower, and none of this is a performance claim about anything.

Cost scales with `--seeds` and `--n-train-per-level`. If you have to cut, **cut sample
count, not seeds** — the dispersion across seeds is what makes the comparison mean
anything, and a tighter absolute magnitude with n = 1 would be worth less than a looser
one with n = 5.

### Self-checks that run every time

The script refuses to produce a record if any of these fail:

1. **Parameter counts** — every architecture's trainable-parameter count is re-derived
   and compared against the expected value, so a silent architecture change cannot
   quietly move the numbers.
2. **Suggested hyperparameters** — `SUGGESTED_HYPERPARAMS` is the primary
   condition's recipe, and every value it uses is a literal in
   `reference_benchmark.py`, so a run here is self-contained. The values
   originated in a graphical training tool that is not part of this repository;
   where that tool is present the script parses its source with `ast` and aborts
   if the two disagree, and where it is absent the cross-check records
   `gui-source-not-found` and the run proceeds. The record of the published
   measurement embeds the values it was compared against, under
   `self_checks.gui_hyperparameter_crosscheck`, so the comparison that was made
   remains readable here.
3. **Leakage** — every test spectrum is hashed and checked against every training
   spectrum. The split is created by using disjoint RNG streams, not by this check;
   the check is what catches that separation silently failing. The same step also
   records a *near-duplicate* statistic (see below), which the identity check alone
   would miss.
4. **Run-to-run determinism** — the baseline architecture is trained twice on identical
   inputs and the scores compared, recording whether this device reproduces itself.

## What the numbers support

Read `results/reference_benchmark.json` → `claim_scope`, and `report.md` for the
rendered form. In short:

**The measurement supports:** *architecture X, at the hyperparameters this software
suggests for it, trained on the described synthetic training pool and evaluated on
independently drawn synthetic test spectra, scored the reported SNR gain.*

**It does not support:**

- that architecture X is **intrinsically** better than architecture Y. The primary
  condition deliberately gives each architecture its own suggested epochs, batch size
  and learning rate, because that is what the software did for the user when the
  measurement was taken: a graphical training tool, which is not part of this
  repository, filled those settings in per architecture. The CLI does not — it
  applies one default to every architecture — so the primary condition describes the
  measured recipe, not what `dnndenoiser train` does unprompted. Budget is
  therefore confounded with architecture by design. The secondary `matched-budget`
  condition exists precisely so that this confound can be inspected rather than
  forgotten — and where the two conditions disagree, the disagreement *is* the finding.
- that the ranking holds on **measured** spectra. Everything here is synthetic, drawn
  from this package's own generator, and evaluated against a reference that only exists
  because the data is synthetic. Distribution shift is the dominant failure mode of a
  denoiser (`AGENTS.md` §5); nothing in this benchmark probes it.
  [`benchmarks/boundaries/position_shift/`](../boundaries/position_shift/) does, on
  one axis — a rigid energy shift — for `ResNet-FCNN` only, starting from *this*
  benchmark's own primary condition for that architecture, which it reproduces
  within its consistency anchor. Read its README for what it supports; no number
  from it is repeated here. Note in particular
  that the training noise and the test noise come from the *same* function, so the
  model's noise model is exactly correct by construction — a condition measured data
  never satisfies.
- that the ranking holds at other peak sets, other position jitter, other backgrounds,
  or noise levels outside those listed. Exactly one point in each of those spaces was
  measured. Training-set size is the one axis that *was* varied — see the robustness
  check in `report.md` — and even there only two points exist.
- that the *magnitudes* transfer. Every architecture gained several dB when the
  training set was enlarged 2.67×, so the absolute figures describe a training-set
  size, not a ceiling. Quote the ordering more confidently than the numbers.
- that a difference smaller than the reported across-seed spread is real. Where two
  architectures overlap within their spread, the honest statement is that this
  benchmark did not separate them.
- anything about whether peak areas, positions or widths survive denoising. That was
  not measured. Denoising is preprocessing, and physically meaningful quantities have
  to be verified downstream (`AGENTS.md` §5).

**A denoised spectrum is a model estimate, not a measurement.** A large SNR gain says
the output agrees with a known synthetic reference; it does not establish that
structure in the output is real. The network can oversmooth, suppress weak features,
and produce plausible structure that was not in the input.

### Results in the record that are easy to misread

- **Negative gain at low noise is a real result, not a bug.** At the lowest noise level
  the input is already close to the reference, and the network's own reconstruction
  error is larger than the noise it removes, so denoising makes the spectrum *worse*.
  This is the `AGENTS.md` §5 warning showing up as a number. Do not present the
  benchmark's best figure without the level at which the sign flips.
- **"No leakage" is not the same as "a hard test set".** The identity check passes —
  no test spectrum is byte-identical to a training spectrum — but the record also
  reports the near-duplicate statistic, and it comes out near 1: a test spectrum is
  about as close to its nearest training spectrum as training spectra are to each
  other. That is the correct and expected outcome for an independent draw from one
  distribution, and it is exactly why the result must be read as *in-distribution*
  performance. The generator varies only per-peak intensity, position and width around
  one fixed peak set, so the test set samples the same small parameter family densely.
  Nothing here measures generalization to spectral structure the model has not seen.
- **The ordering is more durable than the numbers.** Across the two training-set sizes
  the mean gains moved by up to several dB while the ordering at the two lower noise
  levels barely changed. At the highest noise level, by contrast, every architecture
  lands within about one dB of every other, so the ordering *there* carries almost no
  information and should not be quoted as a ranking at all.
- **The learning-rate schedule is not budget-controlled in the primary condition.**
  `StepLR(step_size=10, gamma=0.1)` is fixed while the suggested epoch counts differ
  (30 vs 50), so an architecture given 50 epochs spends its extra epochs at a learning
  rate already cut by a factor of 100 or more. That is what the software does, and the
  primary condition faithfully reproduces it; it is not a controlled comparison of
  training length.
- **The matched-budget condition is not neutral either.** It fixes the learning rate at
  0.001, which is the value the software already suggests for `ResNet-FCNN`,
  `ResNet-1DCNN` and `Transformer`, and is ten times lower than the value it suggests
  for the other five. So the secondary condition removes one confound (differing epoch
  and batch budgets) by introducing another (a learning rate that suits some
  architectures better than others). It is a useful second look, not a neutral
  arbiter, and a fully controlled comparison would require sweeping the learning rate
  per architecture — which this benchmark does not do.
- **One model covers all three noise levels.** Each run trains a single model on a pool
  containing all noise levels in equal proportion, then scores it at each level
  separately. That is a realistic setup, but it means the low-noise figures partly
  reflect a compromise the model made in order to also handle the high-noise data. A
  model trained at one noise level only would very likely score differently there.
  Nothing in these records measures that.

## Independence assumption behind the error bars

The reported spread is the standard deviation across **seeds**, where one seed is one
independent training run on an independently drawn training set. That is the unit that
supports comparing architectures.

Per-spectrum SNR values inside a single test set are *not* independent replicates: they
share one trained model, so averaging more test spectra shrinks the sampling error of
that model's score without saying anything about how much the score would move if the
model were retrained. The record stores the per-spectrum spread under the deliberately
awkward key `snr_gain_db_sd_over_spectra_not_a_replicate_sd` so it cannot be mistaken
for an error bar.

The paired comparison against the baseline architecture uses the seed as the pairing
unit: within a seed, all architectures see the same training arrays and the same test
spectra. Paired *t*-tests on n = 5 are weak, so the record reports the mean paired
difference, its standard deviation, Cohen's *d<sub>z</sub>*, the raw *p*, a
Holm-adjusted *p* across the comparisons made at each noise level, and how many of the
seeds favoured the architecture. Read the sign consistency alongside the *p*-value, and
treat neither as decisive on its own.

## Reproducibility

Bit-exact reproduction **across devices is not guaranteed**, and the record says so.
Floating-point reduction order differs between the CPU, MPS and CUDA backends, and
GPU-backend kernels are not required to be run-to-run deterministic.

What is reproducible:

- **the data** — generated by `numpy.random.Generator`, which is platform-independent,
  so the same seeds give the same spectra everywhere;
- **the parameter counts** — asserted on every run;
- **the design** — every condition required by `AGENTS.md` §6 is written into the
  record, so the measurement can be repeated and disagreed with;
- **the procedure** — this file plus `reference_benchmark.py`.

Run-to-run determinism on a single device is *measured* rather than assumed: see
`self_checks.repeat_run_determinism` in the record. Cross-device agreement can be
checked by running `--quick` twice with `--device cpu` and `--device mps` and diffing
the two records. No such comparison is recorded here, so this file states no number
for it; `design.reproducibility.bit_exact_across_devices` in the record is `false`,
and that is the whole of the claim. A previous version of this paragraph quoted a
result that no record backed, and re-running the stated procedure did not reproduce
it — which is the failure mode this directory exists to prevent.

## If you change something

- Knobs exposed on the command line (`--seeds`, `--n-train-per-level`,
  `--n-test-per-level`, `--device`) are written into each record, so two records that
  differ only in those are comparable *as long as you say which is which* — that is
  exactly how the training-set-size check above is done.
- Changing a fixed condition in the source (peak set, generator config, noise levels,
  optimizer recipe, metric) makes the new record incomparable with the old one. Bump
  `RECORD_VERSION` and say what changed.
- Changing the noise model, the generator, or the meaning of the metric is an
  independent-audit item (`AGENTS.md` §8).
- Do not delete a superseded record to make the new one look tidier. Negative and
  superseded results stay (`AGENTS.md` §6).
