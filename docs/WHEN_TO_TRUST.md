# When to trust the denoiser

A trained denoiser is only valid inside the distribution it was trained on. Outside
it, the output can be worse than the input — and still look plausible. This page
collects, one question at a time, what has been **measured** about where that happens.
Each answer gives the number, the conditions it was measured under, and a link to the
record that owns it. Anything not listed here has not been measured.

<!-- Every number on this page carries an r: anchor (a value taken from the record)
or an n: anchor (one that deliberately is not), and tests/test_p2a_record_citations.py
recomputes each r: value from the record. Only the sentences listed in the P2-A
preregistration's Revision 11 are cleared for quotation; the test refuses any other
record key on this page. -->

## My spectra sit at a different energy than the training data. How far can I trust the model?

**Which workflows this is about.** It applies when a model is trained on synthetic
spectra — the `noise2clean` and `noise2noise` methods, which learn from clean synthetic
references — or when a model trained on one measurement is reused on another whose
energy scale differs, for example through sample charging or a calibration offset.
The self-supervised moving-average method (`train --method moving-average`) trains on
the frames of the measurement it denoises. **This measurement did not include it**:
it says nothing about shifts between the frames of one measurement, such as charging
that drifts during acquisition, and nothing about reusing a self-supervised model on
another measurement.

**What was measured.** A rigid shift of the whole spectrum, applied at inference to a
model trained without one:

- A model trained with ±0.3<!--n:reg--> eV of per-peak position jitter and no rigid
  shift **lost all of its SNR-gain benefit at a shift of about 0.5<!--r:R3.pos--> eV**
  (0.47<!--r:R3.pos--> eV toward higher energy, 0.47<!--r:R3.neg--> eV toward lower).
  Beyond that it made the spectrum worse than the noisy input.
- Training with a rigid shift drawn per spectrum from ±1.5<!--n:reg--> eV **moved that
  point to about 1.8<!--r:R6.pos--> eV** (1.8<!--r:R6.neg--> eV the other way). **It did
  not remove it**: beyond that, this model failed too.
- **The denoised peak does not follow the shift.** For the model trained without a
  rigid shift, at a shift of ±1.0<!--n:reg--> eV the denoised peak moved the other way,
  by −0.67<!--r:R7.+1--> / +0.63<!--r:R7.-1--> eV — roughly two-thirds of the shift.
  No mechanism is claimed.

**Under these conditions only.** Synthetic C 1s spectra from this package's generator
(three peaks, 256 points on a 0.069<!--r:grid.step--> eV grid); a ResNet-FCNN at the
reference recipe, trained on 2304 spectra, an equal mix of three Poisson noise levels;
the numbers above are for the middle level, where the noisy input's SNR at zero shift
is about 16<!--r:L1k.in.0--> dB; 20 independently seeded training runs; the MPS
backend; the shift applied to the peaks under a background that does not move. On that
grid, 0.47<!--r:R3.pos--> eV is about 6.8<!--r:R3.bins--> bins, or about
0.39<!--r:R3.fwhm--> times the main peak's nominal width (1.2<!--r:fwhm.nominal--> eV
FWHM). **Whether the eV figure, the bin figure or the width ratio carries over to your
grid and line widths was not tested** — none of them should be assumed to.

**What to do.**

1. Put the energy scale where the model expects it before denoising — correct for
   charging and calibration first.
2. If you train on synthetic spectra, train across the shifts you expect, and test
   beyond them: the boundary moves outward, it does not disappear.
3. Check peak positions against the noisy input, and verify any position, area or
   width downstream. A denoised spectrum is a model estimate, not a measurement.

**Not established here.** Anything about measured spectra; other peak shapes,
architectures, training-set sizes or noise levels (at the least noisy of the three
levels this model did not help even without a shift; at the noisiest, no failure point
appeared within ±4.0<!--n:reg--> eV); shifts that move peaks relative to each other,
such as chemical shifts; and any general threshold for XPS.

**Record:** [`benchmarks/boundaries/position_shift/`](../benchmarks/boundaries/position_shift/)
— the report, the preregistration it was measured against, and how to re-run it.

## Not yet measured

- A model trained at one signal-to-noise ratio and applied at another.
- Which architecture holds up best when the peaks move.
- Anything on measured spectra with a known reference.

These are planned as further records of the same kind. Until one exists, this page
says nothing about them.
