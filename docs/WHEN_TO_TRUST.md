# When to trust the denoiser

A trained denoiser is only valid inside the distribution it was trained on. Outside
it, the output can be worse than the input. This page gives what has been measured
about where that happens, with the conditions and the record each answer comes from.
It quotes a limited set of statements cleared for quotation here; the record holds
more.

Each answer below names the model, the conditions and the record it comes from.

**Which model was measured, for the energy-shift question.** A ResNet-FCNN trained with the recipe of this
repository's reference benchmark, by the measurement's own training loop. The CLI's
`train` command, at its defaults (the FCNN architecture and different optimiser
settings), was not measured, and these numbers should not be assumed to hold for it.
Nor were the CLI's `generate` defaults, which differ from the data below: no position
jitter, one noise level, and exact Poisson noise.

<!-- Numbers on this page carry r: anchors (recomputed from the record) or n:reg anchors
(registered design values, matched against the record's design); see
tests/test_p2a_record_citations.py, and item 78 of the P2-A preregistration's Revision 11
for what the test does not check. -->

## My spectra sit at a different energy than the training data. How far can I trust the model?

**What was measured.** That model, trained on synthetic spectra against their clean
references, then given spectra whose peaks were all shifted in energy by the same
amount — the shape of a charging offset or a calibration error.

- **Trained without shifts, it lost all of its benefit at about
  0.5<!--r:R3.pos--> eV.** The boundary was 0.47<!--r:R3.pos--> eV toward higher binding
  energy and 0.47<!--r:R3.neg--> eV toward lower; over the 20<!--r:n.seeds--> training runs
  and both directions it fell between 0.45<!--r:R3.range.lo--> and
  0.49<!--r:R3.range.hi--> eV. Beyond it the
  SNR gain was negative: by this metric the output was further from the clean spectrum
  than the noisy input was.
- **Trained with shifts drawn from ±1.5<!--n:reg--> eV, the boundary moved to about
  1.8<!--r:R6.pos--> eV** (1.8<!--r:R6.neg--> eV the other way; between
  1.79<!--r:R6.range.lo--> and 1.87<!--r:R6.range.hi--> eV over runs and directions). It
  did not remove it: beyond that, this model failed too. At zero shift its improvement
  was consistently smaller than the first model's; this design cannot attribute why.
- **At ±1.0<!--n:reg--> eV, the denoised peak followed only about a third of the
  shift.** For the model trained without shifts, at a shift of +1.0<!--n:reg--> eV the
  maximum of the denoised spectrum sat −0.67<!--r:R7.+1--> ± 0.03<!--r:R7.+1.sd--> eV from
  that of the clean spectrum at the same shift, and at −1.0<!--n:reg--> eV
  +0.63<!--r:R7.-1--> ± 0.04<!--r:R7.-1.sd--> eV (mean ± SD across runs, after
  subtracting the same quantity at zero shift): short of where it should have been by
  roughly two-thirds of the shift. The shortfall grew with the shift; at
  ±4.0<!--n:reg--> eV the peak hardly followed at all. The maximum is a grid point of
  the whole spectrum, not a fitted peak position. No
  mechanism is claimed. A positive M1 gain does not establish that the output's peak
  sits where the reference's does. (M1 is the SNR gain.)

**Under these conditions only.**

- *Data.* Synthetic C 1s spectra from this package's generator: three peaks,
  256<!--r:n.points--> points on a 0.069<!--r:grid.step--> eV grid, with ±0.3<!--n:reg--> eV
  of independent position jitter per peak in training and test spectra alike. The
  peaks shift under a linear background that stays in place, so this is not a
  translation of the whole spectrum.
- *Training.* 2304<!--r:n.train.A--> spectra per model: an equal mix of three Poisson
  noise levels.
- *Noise model.* Training and test noise come from the same function, so the model's
  noise model is exactly right by construction — which measured data never is. Two of
  the three levels, including the one quoted here, use a Gaussian approximation to the
  Poisson noise.
- *Evaluation.* The numbers above are for the middle level, where the noisy input's SNR
  at zero shift is about 16<!--r:L1k.in.0--> dB. Every level was in training, so
  training and inference share one signal-to-noise regime.
- *Metric.* SNR gain in dB against the clean spectrum at the same shift, averaged over
  512<!--r:n.test--> test spectra per shift. The boundary is the median over runs of
  each run's first crossing of zero gain, linearly interpolated between the shifts
  tested; this estimator is biased low wherever the curve wobbles near zero. The
  displacement is likewise a mean over each run's test spectra, so it does not say how
  individual spectra behaved.
- *Replicates and split.* The replicate is the training run, each with its own seed:
  every ± above is across runs and every range across runs and both directions, and
  the spectra within one test set are not treated as independent replicates. The two
  models were tested on identical spectra. Training and test spectra are generated
  independently from disjoint random streams; the independently generated spectrum is
  the unit that keeps them apart.
- *Units.* On this grid 0.47<!--r:R3.pos--> eV is about 6.8<!--r:R3.bins--> bins, or about
  0.39<!--r:R3.fwhm--> times the dominant peak's nominal width of
  1.2<!--r:fwhm.nominal--> eV FWHM — which each spectrum varies, and which is narrower than
  the three-peak envelope. Whether the eV figure, the bin figure or the width ratio
  carries over to your grid and line widths was not tested.
- *Backend and provenance.* Measured on the MPS backend; the numbers were not compared
  across devices. The preregistration was not published before the measurement ran,
  so that it came first is attested only by this repository's own history.

**What to do.**

1. Put the energy scale where the model expects it before denoising — correct for
   charging and calibration first.
2. Verify any position, area or width downstream. A denoised spectrum is a model
   estimate, not a measurement.

**Not established here.**

- Anything about measured spectra.
- The `noise2noise` and self-supervised moving-average methods, including shifts
  between the frames of one measurement, and reusing any trained model on another
  measurement.
- Noise levels other than the one quoted.
- Other peak shapes, grids, architectures, training-set sizes, jitter widths, noise
  models, optimisation budgets, or shift ranges in training.
- Separating the loss from effects of the window's edge beyond ±1.5<!--n:reg--> eV,
  where the boundary of the model trained with shifts and the displacement at
  ±4.0<!--n:reg--> eV lie.
- Training with shifts as a remedy.
- Shifts that move peaks relative to each other, such as chemical shifts.
- Any general threshold for XPS.

**Record:** [`benchmarks/boundaries/position_shift/`](../benchmarks/boundaries/position_shift/)
— the report, the preregistration it was measured against, and how to re-run it.

<!-- record:P2-B -->
<!-- DRAFT, not cleared: proposed for owner review. Numbers carry r: anchors (recomputed
from the P2-B record) or n:reg anchors (registered design values); see
tests/test_p2b_record_citations.py. -->

## I trained the self-supervised model at one count rate. Can I use it at another?

**What was measured.** The self-supervised moving-average method with
W = 1<!--n:reg-->, trained on the frames of one synthetic spectrum and then applied to
fresh frames of the same spectrum at another count rate — another per-frame exposure. The
count rate is the expected count at the spectrum's maximum, λ; five levels from
λ = 4<!--n:reg--> to λ = 100<!--n:reg--> were measured, a 25<!--n:reg-->-fold range, with
every level applied to every other in both directions. "Lost" and "cost" below compare a
model with one trained at the count rate it was applied to, not with the noisy input.

- **At the three count rates where this was registered, it helped at the rate it was
  trained on.** At λ = 20<!--n:reg-->, 45<!--n:reg--> and 100<!--n:reg--> its SNR gain was
  positive in 20<!--r:ma.R1.min.k--> of 20<!--r:n.seeds--> runs each.
- **Applied to noisier frames than it was trained on, it never lost more than
  1<!--n:reg--> dB** against a model trained at that count rate, in any of the
  200<!--r:ma.R2.seedcells--> combinations of run and pair of levels. The single worst
  combination lost 0.6<!--r:ma.R2.worst.loss--> dB.
- **Applied to cleaner frames than it was trained on, it did worse than a model trained
  at that count rate**, in every pair of levels, in at least
  19<!--r:ma.R3.min.k--> of 20<!--r:n.seeds--> runs. On average it still improved on the
  noisy input in all but one pair (below).
- **Going to cleaner frames cost more than going to noisier ones.** For every pair of
  levels, the loss against the matched model when going up exceeded the loss going
  down, paired within run, in at least 18<!--r:ma.R4.min.k--> of
  20<!--r:n.seeds--> runs.
- **At the extreme, on average, it made the frames worse than the input.** Trained at
  λ = 4<!--n:reg--> and applied at λ = 100<!--n:reg-->, its SNR gain was
  −1.5<!--r:ma.m1.t4.i100--> ± 1.9<!--r:ma.m1.t4.i100.sd--> dB, negative in
  16<!--r:ma.m1neg.4.100--> of 20<!--r:n.seeds--> runs: there the output was further from
  the clean spectrum than the noisy input was. That was the only one of the
  25<!--r:ma.cells--> combinations of training and applied rate whose mean gain was
  negative (1<!--r:ma.m1neg.cells-->); in every other, the output still improved on the
  input on average.

**Why the direction cannot be attributed to the signal-to-noise ratio.** Every level had
the same total exposure, as when one measurement time is split into more or fewer
frames. So a model trained at a higher count rate also saw fewer frames —
500<!--n:reg--> at λ = 100<!--n:reg--> against 12500<!--n:reg--> at λ = 4<!--n:reg--> — and
took fewer optimisation steps, about 800<!--n:reg--> against 19550<!--n:reg-->. And the
method scales its input by the minimum and maximum of its training frames, which differ
with the count rate, so a model applied at another rate sees inputs outside, or squeezed
within, the range it was trained on. None of these is separated from the others, or from
the signal-to-noise ratio. No mechanism is claimed.

**Which model was measured.** A ResNet-FCNN trained by the library's moving-average
recipe with W = 1<!--n:reg--> and 50<!--n:reg--> epochs. The CLI's
`train --method moving-average` at its defaults, whose window and number of epochs
differ, was not measured, and these results should not be assumed to hold for it.

**Under these conditions only.**

- *Data.* Synthetic C 1s spectra from this package's generator: three peaks,
  256<!--r:n.points--> points. Each run draws one clean spectrum; its training and test
  frames are independent noisy copies of it.
- *Training.* Batch 32<!--n:reg-->, trained per run on that run's frames alone, with the
  same total exposure at every count rate.
- *Noise model.* Exact Poisson counts at every level, drawn by the same function for
  training and test, so the noise model is exactly right by construction — which
  measured data never is.
- *Evaluation.* SNR gain in dB against the clean spectrum, averaged over
  512<!--r:n.test--> fresh test frames per level. The noisy input's SNR ran from about
  2.4<!--r:in.4--> dB at λ = 4<!--n:reg--> to 16.3<!--r:in.100--> dB at
  λ = 100<!--n:reg-->. This metric does not measure whether the output varies from frame to
  frame: every frame of a run shares one clean spectrum, so an output that stays close to
  it scores well whatever it does with the frame.
- *Replicates and split.* The replicate is the run, each with its own spectrum, frames
  and initialisation; every ± is across runs, and the frames within a run are not
  treated as independent replicates. Training and test frames come from disjoint random
  streams.
- *Backend and provenance.* Measured once, on the MPS backend; not compared across
  devices or repeated. Every run's gain for every combination is in the record, so each
  number here can be recomputed from it or, like the input SNR, read from it; the gains themselves can be checked only by
  running the measurement again. The preregistration was published before the
  measurement ran, which rests on GitHub's timestamps and on a record time written by the
  run itself.

**What to do.**

1. Train on frames at the count rate you will denoise, when you can: that is where this
   measurement registered, as a test, that the method helps, and it held.
2. If you apply a model to frames with more counts than it was trained on, check the
   result: in this measurement that direction did worse than a model trained at that
   count rate in nearly every run, and at the extreme the output was worse than the
   input.
3. Verify any position, area or width downstream. A denoised spectrum is a model
   estimate, not a measurement.

**Not established here.**

- Anything about measured spectra, depth profiles or an instrument.
- Applying a model to frames of a different spectrum, or to another measurement: here
  the frames were always of the spectrum it was trained on.
- Other ways of spending the exposure — for example the same number of frames at every
  count rate.
- Other noise models, such as a read-noise floor.
- Why one direction cost more: signal-to-noise, frame count, optimisation steps and
  input scaling were not separated.
- Whether the output keeps frame-to-frame variation, or how the method fails at the
  lowest count rates beyond this metric; that was not measured.
- Other training methods, window widths, architectures, spectrum shapes, or count-rate
  ranges wider than the one tested.
- Any general rule for XPS.

**Record:** [`benchmarks/boundaries/snr_transfer/`](../benchmarks/boundaries/snr_transfer/)
— the report, and the preregistration it was measured against, with its Record section.
<!-- /record:P2-B -->
