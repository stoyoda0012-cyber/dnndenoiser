# When to trust the denoiser

A trained denoiser is only valid inside the distribution it was trained on. Outside
it, the output can be worse than the input. This page gives what has been measured
about where that happens, with the conditions and the record each answer comes from.
It quotes a limited set of statements cleared for quotation here; the record holds
more.

**Which model was measured.** A ResNet-FCNN trained with the recipe of this
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
