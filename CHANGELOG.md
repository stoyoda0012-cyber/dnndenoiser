# Changelog

All notable user-visible changes to `dnndenoiser` will be documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once public releases begin.

## [Unreleased]

### Added

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

No public release has been issued from this repository yet.
