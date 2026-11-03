# Using dnndenoiser with an AI agent — a starting point

> This document is the developers' recommended starting point for using
> dnndenoiser together with an AI agent. It is not a required procedure.
> Copy it and edit it for your research question, samples, instrument
> and laboratory practice. Where you depart from it, record what you
> changed and why in your analysis notes. Nothing in it widens what the
> software supports or makes a result scientifically valid.
>
> **Written for: dnndenoiser v0.2.0.** Read the copy that came with the
> version you have installed (`python -m dnndenoiser.guides`), not the
> repository's main branch, which may describe a newer version.

If you keep an edited copy, keep this header and the version line, and
keep what you added about *your usual* instrument or settings apart from
what you checked for *this* data set. An agent should not treat the
first as a fact about the second.

## 1. What dnndenoiser is for, and what it is not

Good at: generating synthetic XPS spectra with a declared noise model,
training a denoising network on them or — self-supervised — on repeated
acquisitions of your own spectrum (a frame stack), applying it, and
saying what an evaluation can and cannot show.

Not for: peak fitting, quantification, depth profiling, or reading
vendor file formats (use `toyomacro` for files). **No pretrained model is
shipped**: you train one before you can denoise anything. A checkpoint
from someone else is a file from someone else: it is loaded with a
restricted loader, and one that needs the full unpickler — which executes
code from the file — is refused unless you pass `--trust-checkpoint`. Do
not pass it to get past the refusal on a file whose origin you cannot
establish.

## 2. Before an analysis, check

- **Where the model's training data came from**, and whether the
  spectra you want to denoise are inside that distribution: peak
  structure, peak positions, noise level, background, normalisation,
  instrument response. Outside it, the output can be worse than the
  input. A model trained on synthetic spectra meets measured spectra as
  a distribution shift, even when the peaks look alike.
- **The noise along the energy axis.** The synthetic generator draws
  noise independently channel by channel. A detector or the analyser
  software's rebinning can correlate it across neighbouring channels;
  that is a shift the synthetic training never saw.
- **The order of your frames.** For self-supervised training,
  `frame_index` is the acquisition order and defines which frames are
  neighbours. If the files are not stored in that order, the index must
  say so.
- **The intensity scale.** Spectra are normalised by default; a model
  trained on one scale and applied to another is applied outside its
  distribution.

## 3. When something is not known

- No clean reference: there is **no reference-free SNR** for a measured
  spectrum. Do not compute one, and do not substitute the mean of the
  same frames — it is not independent of what was trained on. Say that
  the accuracy on this data is not evaluated.
- Unknown training distribution (a model whose origin you cannot
  establish): do not use it for results. Train on data you can describe.
- Unknown frame order: say so; do not let the file order stand in for it.

## 4. How to work

Train on data that represent what you will denoise — ideally your own
frames, with `train --method moving-average`. Settle the training
conditions once, then apply the same model unchanged to the batch.
Keep the noisy input beside the output: every downstream number should
be checkable against it.

## 5. Reading a result

**Denoised output is a model estimate, not a measurement.** The network
can oversmooth, suppress weak features, or produce plausible structure
that is not in the data. Denoising is preprocessing: check peak areas,
positions and widths downstream against the noisy input, not on the
output alone.

What each command's numbers mean:

1. `evaluate` reports an **SNR only against a reference declared as the
   synthetic truth**. Against an estimate it reports agreement or
   relative changes, which are not errors against the signal. The one
   exception is `--legacy-output`, which prints `snr_*` keys for an
   undeclared reference: they are an error against the signal only if
   that reference really is the truth — do not report them as an SNR
   otherwise. A gain measured on synthetic data is a statement about that
   synthetic data, not about your measurements.
2. `evaluate` and `diagnose` report `held_out_status`; report it
   verbatim with the result. `not_held_out` means the data were training
   data, and a result on them is not a held-out result. `unknown` is not
   held out either. Only `disjoint_by_identifiers` supports a held-out
   claim, and only as far as the identifiers the files carry are right.
3. `diagnose` describes how the output follows a change added to the
   input. None of its numbers is an accuracy, a noise reduction or an
   SNR, and it does not detect a collapsed model.
4. Several models trained with `train --seeds` and applied together by
   `infer` show **how much training runs differ** on one input. That
   spread is not an uncertainty of the spectrum.

So, to "how much did denoising improve my spectrum?", an answer on
measured data reads like: "on measured data there is no reference, so
the improvement is not evaluated. On synthetic spectra generated
separately from the training set but with the same generator settings
(`held_out_status` as reported, the noise level stated), the gain against
the synthetic truth was X, given as the spread over N spectra and M
training seeds, not as one run. Fitting the denoised spectrum instead of
the noisy one moved the peak areas and positions by Y — a difference
between two estimates, not an error of either."

## 6. Keep the record with the result

`train` writes a provenance manifest into the checkpoint (training-data
digest, options, seeds, software versions, the package's commit or
`"unknown"`), bound to the weights by a `model_digest`; `infer` carries
it into its output and refuses a checkpoint whose `model_digest` no
longer matches; `evaluate` reports the model's identity. A checkpoint
written before the manifest existed has none, and is accepted with
"Provenance: none" — say so when you use one.
Keep the checkpoint, the noisy input and the evaluation JSON together.
The manifest records what was run; it does not make training
reproducible.

## 7. The command line

```bash
dnndenoiser generate ...   # synthetic spectra with a declared truth
dnndenoiser train ...      # noise2clean, noise2noise, or moving-average on a frame stack
dnndenoiser infer ...      # apply one model, or several with repeated -m
dnndenoiser evaluate ...   # metrics whose meaning depends on the declared reference
dnndenoiser diagnose ...   # how the output depends on the input (frame stack only)
```

`dnndenoiser <command> --help` lists the options of the installed
version. There is no MCP server; an agent drives the CLI or the Python
API.

## 8. Where the details are

For the version this guide was written for:

- Quick start and the HDF5 layouts: <https://github.com/stoyoda0012-cyber/dnndenoiser/blob/v0.2.0/docs/QUICK_START.md>
- Applying the method to your own measurements: <https://github.com/stoyoda0012-cyber/dnndenoiser/blob/v0.2.0/docs/FROM_THE_PAPERS.md>
- What an evaluation reports, and why: <https://github.com/stoyoda0012-cyber/dnndenoiser/blob/v0.2.0/docs/design/EVALUATION_REFERENCE_CONTRACT.md>
- Limitations: <https://github.com/stoyoda0012-cyber/dnndenoiser/blob/v0.2.0/README.md#limitations>

The repository's `AGENTS.md` and `CONTRIBUTING.md` apply when you change
the package itself or its public documents — not to an analysis that
only uses it.
