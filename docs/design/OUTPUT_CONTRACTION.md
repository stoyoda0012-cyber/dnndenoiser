# Design: does the output follow its input? A diagnostic, not a verdict

**Status: adopted 2026-10-03 (revision 2, after two independent audits); implemented (see
the CHANGELOG).** Step 5 of the improvement plan that follows a reproducibility assessment of a
published study. It adds a `diagnose` command that describes how a model's output depends on
its input, and tests with evaluation-level negative controls. It adds no accuracy judgement,
no noise-reduction figure and no threshold. Under `AGENTS.md` §8 it is audited before code is
written, because it adds outputs whose meaning must be fixed and could be misread.

## 1. The problem

A self-supervised model can return nearly the same spectrum for every frame. Against the
natural reference for measured frames — their mean — such a model has zero discrepancy by
construction (step 1, `docs/design/EVALUATION_REFERENCE_CONTRACT.md` §6.4, already says so
in a caveat). That score is **uninformative**, not necessarily wrong: on a stationary stack
the frame mean is itself a good estimate of the signal, so returning it may score well
against the true signal too. What the score cannot say is whether the model denoises its
input or returns a stored spectrum whatever the input.

Frame-to-frame variation cannot say it either. For a stationary signal, an ideal denoiser
also removes the frame-to-frame variation, so small output variation is what good denoising
and a stored spectrum both look like. What separates them is whether the output still
follows a change that is really in the input.

## 2. The surface

`dnndenoiser diagnose -d frames.h5 -m model.pt -o report.json [--probe E0:FWHM:k …]`, on a
**frame stack** (2-D or 3-D, `docs/design/FRAME_STACK_CHANNELS.md`) only (owner's decision,
2026-10-03: rows of a `noisy` file are, by default, different spectra; repeated acquisitions
are converted with `write_frame_stack`).

The computation is a library function `diagnose(f, frames, energy, frame_index, …)` taking
frames already on the network grid and a callable `f`. `f` takes and returns arrays shaped
`(n_frames, [n_channels,] L)` on the network grid, in input units; it is assumed to act row
by row. `diagnose` calls it once on the frames and once per probe, and computes each
channel's mean `x̄` once, before the calls. The CLI does the resampling and builds `f`
from the checkpoint exactly as `infer` does — resampling to the
network length, the checkpoint's normalisation and its inverse, `model.eval()` (dropout off),
batches on the device given (`--device`, default `cpu`; on another device the two passes of
§3.2 may differ by device nondeterminism, and the report records the device). Tests pass
other callables (§5).

**The grid.** The quantities — the frames `x_i`, σ, the probe `g`, the projection — are on
the network grid: the CLI resamples the frames to the network length if they are not already
(reported as `resampled: {from_points, to_points}` or `null`); the probe is sampled at the
resampled energy points and added in input units after resampling and before normalisation.
The digest and the held-out check (§3.3) use the frames **as stored**, before resampling.
Normalisation is affine, so it cancels in both ratios below; it matters only for σ and the
probe amplitude, which are in input units.

## 3. What is reported

All quantities per channel (one for a 2-D stack). Nothing is reported in dB or as a
percentage reduction.

### 3.1 Contraction ratio

    C = Σ_i ‖y_i − ȳ‖² / Σ_i ‖x_i − x̄‖²

over the frames of one channel, `x̄`, `ȳ` the means over frames. `C = 1` for a model returning
its input, `C = 0` for one returning a fixed spectrum.

**C is not an accuracy, a noise reduction or an SNR change.** Its denominator contains the
input's noise **and any change of the signal across frames** (drift, a reaction); its
numerator counts removed signal change as removed variation; and it is blind to bias — an
oversmoothed or wrong spectrum returned for every frame gives `C = 0`. Reading `−10 log₁₀ C`
as "noise reduction in dB" would present a reference-free SNR, which `AGENTS.md` §5 forbids;
no output carries it.

Overall: the ratio of the summed numerators to the summed denominators over channels,
labelled as weighted by each channel's frame-to-frame variation. A channel with zero
denominator (identical frames after resampling) is refused.

### 3.2 Injection response

To see whether the output follows a real change, a known change is added and the change in
output measured:

- **σ**, per channel, the noise scale used to size the probe: the frames are put in
  `frame_index` order, differences are taken **only between frames whose indices are
  adjacent** (index difference 1), and σ is the median over grid points of
  `std(Δx)/√2` over those differences (population standard deviation, ddof 0). Differences of
  adjacent frames cancel slow drift; a gap between runs is not differenced. Fewer than two
  adjacent pairs, or σ ≤ 1e-12 · max|x| (no measurable noise, e.g. very low counts in few
  frames or noiseless data), is refused. The report records σ and the order basis (step 3).
- **The probe** `g`: a Gaussian of unit height at `E₀` with full width `FWHM`, sampled on the
  network grid; amplitude `a = k·σ`, `k` signed (a negative `k` is a dip). Defaults (owner's
  decision, 2026-10-03): `FWHM` = 3 % of |span| (span = last − first energy of the network
  grid), `k = 3`, and **five positions** at 10, 30, 50, 70 and 90 % of the way from the
  first to the last energy point (so on a descending axis 10 % is near the highest energy),
  each reported separately and never averaged. `--probe E0:FWHM:k` (repeatable; `=` form,
  `--probe=-5:0.6:3`, for a negative E0) **replaces** the defaults. A position closer than
  2·FWHM to either end, a FWHM below twice the grid spacing, `FWHM ≤ 0` or `k = 0` is
  refused.
- **The response**, per frame, a secant at amplitude `a` (not a derivative; nonlinearity makes
  it depend on `k`):

      R = ⟨f(x + a g) − f(x), g⟩ / (a ⟨g, g⟩)

  reported per probe and per channel (each channel at its own amplitude `k·σ_c`) as the median
  and `[q25, q75]` over frames (NumPy's linear quantiles), and for a 3-D stack also pooled
  over all (frame, channel) rows, labelled as such. The spread is over frames; it is not an
  uncertainty of R.
- **The area ratio** `A = Σ(f(x + a g) − f(x)) / (a Σ g)` beside it, the same way.

**Reading R.** R is the projection of the output change on the probe: attenuation,
broadening and shift all reduce it (an area-preserving smoother gives `R < 1` with `A = 1`).
`R = 1` is what returning the input unchanged gives, so larger is not better; values outside
[0, 1] are reported as computed, never clipped. **A** sums the output change over the whole
spectrum: changes far from the probe count, changes of opposite sign cancel, and a shift
gives 1, so `A = 1` does not show that the areas of real features are preserved. A small
`R` says the output does not follow a change of that position, width and size — collapse, or smoothing that removes such a
feature. It says nothing about how far the output is from the signal.

### 3.3 Whether the frames are the training data

The report includes `held_out_status`, `held_out_basis` and `rows_in_training` from the
model's manifest, by step 2's rules: `held_out(manifest, noisy=<frames as stored>,
input_array_digest=None, acquisition_id=<the frames' acquisition_id attribute, read by the
CLI>, frame_index=<the stack's frame_index>)`. Without a manifest, the same defaults as
`evaluate` (`unknown`). When the status is `not_held_out`, the interpretation adds: *The frames include the model's training data; these
values describe the trained model on its training frames.* **Observed, not established:** on a
model's own stationary training stack, a moving-average model on toy data gave `R` near 0 away
from the peaks and well below 1 on them (one seed); this may be the ordinary outcome of the
method, so `diagnose` is not a collapse detector (owner's decision, 2026-10-03, to say so in
QUICK_START, without numbers).

## 4. Output

`report.json` (strict JSON): per channel and overall `contraction_ratio`, `n_frames`, and per
channel `sigma`; `probes`, a list of `{E0, fwhm, k, channels: [{amplitude, response_median,
response_iqr, area_median, area_iqr}], pooled: {…} or null}`; `resampled`, `device`,
`order_basis`; the held-out fields of §3.3; the model's identity (`evaluation_context.model`'s
fields, step 2 §4, built from the manifest and the verified records); the input's
`provenance.matching_digest` of the frames **as stored**; and a fixed `interpretation`:

*These describe how the output depends on the input. They are not measures of accuracy, of
noise reduction or of SNR: the contraction ratio counts removed signal change as well as
removed noise and is blind to bias; the injection response is reduced by attenuation,
broadening and shift alike, and 1 is what returning the input gives; an area ratio of 1 does
not show that the areas of real features are preserved.*

No key or printed label contains a word of step 1's forbidden list (snr, quality, accuracy,
improvement, gain, reduction); a printed label is the text before a value, and the
interpretation and the §3.3 sentence are printed as `Note:` lines, as `evaluate`'s caveats
are, which the rule does not cover.

## 5. Evaluation-level negative controls (owner's decision, step 4)

Callables built in the tests, not shipped, on the network grid in float64:

- **identity** `f(x) = x`: `C = 1`, `R = A = 1`;
- **constant** `f(x) = c` (the frames' mean, computed once): `C = 0`, `R = A = 0`;
- **smoother** `f(x) = S x`, `S` an explicit band matrix with a stated boundary rule: `C`
  small, `R = gᵀSg / gᵀg` and `A = 1ᵀSg / 1ᵀg` exactly, independent of `x` and `a`;
- **shrink** `f(x) = x̄ + λ(x − x̄)` with `λ = √C_S`, matched to the smoother's contraction:
  the same `C` as the smoother, `R = λ`;
- **mirror** `f(x) = 2x̄ − x` (x̄ fixed per channel before the calls): `C = 1` and `R = A = −1`
  to rounding (C is blind to sign; R is not clipped).

Claims pinned by tests:

- against the frame-mean reference (step 1's `estimate_same_data` case), the constant has zero
  discrepancy, below both the smoother and the identity — the reference gives the constant
  the best score, and the score cannot say why;
- `C` does not separate two models that treat the input differently: the smoother and the
  matched shrink have equal `C` (to 1e-12);
- `R` does: `gᵀSg/gᵀg` for the smoother against `λ` for the shrink, and 0 for the constant.

The `evaluate` control writes the reference array itself as the constant's `denoised`, so the
zero is exact; the caveats of that case appear for every model in it, since they depend on
the case, not on the output.

## 6. Documentation

QUICK_START's frame-stack evaluation warning currently ends by telling the reader to look at
how the output varies across frames before reading a frame-mean score. That contradicts §1
(frame-to-frame variation cannot separate denoising from a stored spectrum). In the change
that adds `diagnose`, the sentence is replaced by a pointer to `diagnose` and to the injection
response, and `evaluate` prints one `Note:` line pointing to `diagnose` exactly when it
prints the same-frames caveat (a frame mean over the evaluated frames), worded around a
frame stack and its model and within step 1's forbidden-word rule. The JSON caveat of step 1
is unchanged (owner's decision, 2026-10-03: changing it is a contract revision).

## 7. What this does not do

- It does not judge accuracy, set thresholds, or call a model collapsed.
- It does not change `evaluate`'s metrics, names or JSON.
- It does not cover the `noisy` schema.
- The probe is one shape; it says nothing about features of other shapes, positions not
  probed, or a real signal change of another form.

## 8. Tests

1. **Controls** (§5) in float64 on the callable, on stated fixtures (values of order 1, so
   rounding is near float64 epsilon; tolerance 1e-12): identity, constant, smoother (the
   expected `R` and `A` from the matrix product written in the test), matched shrink, mirror;
   and a stub nonlinear in a known way (`f(z) = z²` elementwise), whose secant `R` depends on
   the amplitude as computed in the test, so a probe added at the wrong scale fails.
2. **The `evaluate` claims** of §5, on a frame stack with the frame-mean reference.
3. **End to end** through the CLI: the wrapper's network replaced through a named seam by an
   identity stub, so the normalisation and its inverse are covered, with a float32 tolerance
   stated as a multiple of float32 epsilon × span / amplitude; a trained model, repeatability
   (two runs, equal reports, on CPU); the defaults give five probes and no averaged field.
4. **The grid:** a 511-point stack made by interleaving midpoints, so `resample(·, 256)`
   returns the original points — the independent truth for `C` and σ; the probe sampled on
   the resampled axis.
5. **σ:** rows stored out of `frame_index` order; a gap with a step between two runs (not
   differenced); slow drift (σ near the noise, where the per-point standard deviation over
   frames would not be); each against an independent computation; σ ≤ the floor and fewer than
   two adjacent pairs refused.
6. **Channels:** per-channel values independent (a constant in one channel, identity in
   another); the overall `C` as the ratio of sums.
7. **Refusals**, each pinned with a positive counterpart: no frames, one frame, non-finite
   input or output, identical frames (zero denominator), a non-finite R or A, a probe within
   2·FWHM of an end, a FWHM below twice the spacing, `FWHM ≤ 0`, `k = 0`, a malformed
   `--probe`; a negative E0 accepted in the `=` form.
8. **Held-out:** rule 1 on the model's own training stack of a length other than 256
   (digest of the stored frames), with the added sentence; rule 2 for a subset of the training
   frames (`not_held_out`, counted) and for a held-out subset of the same acquisition
   (`disjoint_by_identifiers`); `unknown` without a manifest.
9. **Words:** no forbidden word in any key or printed label, using step 1's `forbidden_in`
   (planted words caught); the interpretation in JSON and as `Note:` lines; `evaluate`'s
   pointer line present exactly with the same-frames caveat and covered by its word test.

## Revision log

- **Revision 0 (2026-10-03).** First draft.
- **Revision 1 (2026-10-03), after the first independent audit** (two blocking, seven should
  fix, five minor). C is stated not to be a noise reduction or an SNR change, never shown in
  dB, and the forbidden words cover `diagnose`; the grid, the channel aggregation, the input
  digest (`matching_digest`) and a callable interface for the controls are specified, with
  `model.eval()`; the controls add a linear smoother and a mirror, and the claim is restated
  (the frame-mean score is uninformative, `C` does not separate, `R` does); held-out status is
  reported; σ from consecutive differences; R's reading covers shift and broadening, is not
  clipped, and the area ratio is reported beside it; probes near the ends are refused; exact
  values only in float64 on the callable; QUICK_START's contradictory sentence is replaced.
  Owner's decisions (2026-10-03): frame stacks only; five default probe positions, never
  averaged; the pointer from `evaluate` in QUICK_START and print only; the observation about
  training stacks documented, without numbers.
- **Revision 2 (2026-10-03), after the second independent audit** (verdict: adopt with
  named changes N1–N12). σ, amplitude and responses are per channel with a labelled pooled
  summary and `[q25, q75]`; `f`'s array shape and call pattern are fixed and the CLI owns the
  resampling; the digest and the held-out check use the frames as stored, with the call to
  `held_out` written out; a matched-contraction shrink makes "C does not separate" testable;
  σ differences only adjacent indices and is refused when negligible; A's reading is stated;
  the word rule excludes `Note:` lines; the probe syntax, the axis direction and the minimum
  FWHM are fixed; tests add a nonlinear stub, an interleaved 511-point grid and the defaults;
  tolerances are tied to fixtures; the `evaluate` pointer prints only with the same-frames
  caveat; the spread is said not to be an uncertainty.

## Confirmation

**2026-10-03 — the owner adopted this design, revision 2, as a whole**, after two
independent audits (the second returning "adopt with named changes", all applied in this
revision). The status line says "implemented" only in the change that merges the
implementation.
