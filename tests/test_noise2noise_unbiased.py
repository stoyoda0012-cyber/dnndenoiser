"""Noise2Noise pairs have to return the clean signal on average.

Noise2Noise learns the clean signal from noisy targets because the target is an
unbiased draw around it. `_add_poisson_noise` used to floor the Poisson rate at
0.01 counts, so every bin whose expected counts fell below that came back biased
upward — in both realizations, since they share the floor. The rate is now
floored at zero, the same treatment `_default_poisson_noise` and the synthetic
generator already used.

The acceptance criteria are the constants below — levels, spectra, realization
count, seed, and thresholds — rather than values read back off the result.
Standardized deviations
``z = (mean - clean) / SE`` with ``SE = sqrt(clean / (scale * N))`` are
approximately standard normal, so a threshold of 5 is a wide margin. The normal
approximation understates the tail at the low-count end, where Poisson
discreteness dominates; measured across seeds the family-wise false-alarm rate
for this file is near 2e-3 rather than the 1.5e-4 the normal tail would suggest.
The seed is fixed, so each run is deterministic; the margin against the
threshold is a few standard errors rather than many, and changing the
realization count or the spectra means measuring it again. Bins whose expected
counts are too small for
the approximation are excluded from the per-bin check and still covered by the
aggregate one, which is where a one-signed bias shows up most strongly.

Run with:
    pytest tests/test_noise2noise_unbiased.py -v
"""
import numpy as np
import pytest

from dnndenoiser.data.synthetic_generator import add_poisson_noise
from dnndenoiser.training.methods import Noise2Noise

# The criteria, as constants rather than values read off the result.
N_REALIZATIONS = 4000
SEED = 12345
Z_THRESHOLD = 5.0
MIN_EXPECTED_COUNTS = 10.0 / N_REALIZATIONS  # per-bin normality guard
# generate --poisson-level {100, 1000, 10000} in the pair parameterisation.
PAIR_LEVELS = [1.0, 100.0, 10000.0]
N_BINS = 256


def spectra():
    """The four shapes the criteria name, keyed by name."""
    energy = np.linspace(0, 1, N_BINS)
    peak = np.exp(-((energy - 0.5) ** 2) / (2 * 0.08 ** 2))
    peak = peak / peak.max()

    single = np.zeros(N_BINS, dtype=np.float32)
    single[N_BINS // 2] = 1.0

    # The tail is set to exact zero: the criteria name a shape reaching zero, and a
    # merely small tail would never exercise the zero-bin criterion.
    decaying = np.where(peak < 1e-6, 0.0, peak)

    return {
        'peak_on_background': (0.85 * peak + 0.15).astype(np.float32),
        'peak_decaying_to_zero': decaying.astype(np.float32),
        'all_zero': np.zeros(N_BINS, dtype=np.float32),
        'single_nonzero_bin': single,
    }


SPECTRA = spectra()


def realizations(clean, pair_level):
    """N input/target pairs from one seeded generator."""
    method = Noise2Noise(seed=SEED)
    pairs = [method.generate_pair(clean, pair_level) for _ in range(N_REALIZATIONS)]
    return (np.stack([p.input for p in pairs]),
            np.stack([p.target for p in pairs]))


def assert_unbiased(draws, clean, pair_level, what):
    """Apply the criteria to one set of realizations."""
    scale = 10000.0 / pair_level
    expected_counts = np.maximum(clean, 0) * scale
    mean = draws.mean(axis=0)

    zero = expected_counts == 0
    if zero.any():
        assert np.array_equal(draws[:, zero], np.zeros_like(draws[:, zero])), (
            f'{what}: bins with no expected counts must come back exactly zero'
        )

    total_counts = expected_counts.sum()
    if total_counts > 0:
        standard_error = np.sqrt(total_counts) / scale / np.sqrt(N_REALIZATIONS)
        z_total = (mean.sum() - clean.sum()) / standard_error
        assert abs(z_total) < Z_THRESHOLD, f'{what}: aggregate z = {z_total:+.2f}'

    testable = expected_counts >= MIN_EXPECTED_COUNTS
    if testable.any():
        standard_error = np.sqrt(expected_counts[testable] / N_REALIZATIONS) / scale
        z = (mean[testable] - clean[testable]) / standard_error
        worst = np.abs(z).max()
        assert worst < Z_THRESHOLD, f'{what}: worst per-bin z = {worst:.2f}'


@pytest.mark.parametrize('shape', sorted(SPECTRA))
@pytest.mark.parametrize('pair_level', PAIR_LEVELS)
class TestConditionalMean:
    """E[realization | clean] == clean, for both draws of the pair."""

    def test_input_realization_is_unbiased(self, shape, pair_level):
        inputs, _ = realizations(SPECTRA[shape], pair_level)
        assert_unbiased(inputs, SPECTRA[shape], pair_level, f'{shape} input')

    def test_target_realization_is_unbiased(self, shape, pair_level):
        _, targets = realizations(SPECTRA[shape], pair_level)
        assert_unbiased(targets, SPECTRA[shape], pair_level, f'{shape} target')


@pytest.mark.parametrize('pair_level', PAIR_LEVELS)
def test_generator_poisson_draw_is_unbiased_under_the_same_criteria(pair_level):
    """The draw this is being aligned with, held to the same bar.

    Scope, stated because the name would otherwise overreach: this exercises the
    generator's exact Poisson branch. Its default is a Gaussian approximation,
    taken whenever the scaled counts exceed 20 — which covers the levels
    ``generate`` is actually used at — and that branch clips each realization at
    zero, so it does not meet this bar in dark bins. That is a property of the
    generator, not of this change, and is out of its scope.

    The generator's level is stated in its own units, so the level that matches
    ``pair_level`` is the one that produces the same expected counts.
    """
    clean = SPECTRA['peak_decaying_to_zero']
    generator_level = np.sqrt(pair_level * 10000.0)
    rng = np.random.default_rng(SEED)
    draws = np.stack([
        add_poisson_noise(clean, generator_level, rng=rng, use_gaussian_approx=False)
        for _ in range(N_REALIZATIONS)
    ])

    assert_unbiased(draws, clean, pair_level, f'generator at level {generator_level:g}')


class TestExternalNoiseFunction:
    """The `_use_internal_noise=False` path keeps its previous meaning."""

    def test_an_external_function_is_used_for_both_draws(self):
        calls = []

        def noise_fn(clean, noise_level):
            calls.append((clean.copy(), noise_level))
            return clean + len(calls)

        pair = Noise2Noise(noise_fn=noise_fn, seed=SEED).generate_pair(
            SPECTRA['peak_on_background'], 100.0
        )

        assert len(calls) == 2, 'both realizations come from the supplied function'
        assert {level for _, level in calls} == {100.0}
        assert np.array_equal(pair.input, SPECTRA['peak_on_background'] + 1)
        assert np.array_equal(pair.target, SPECTRA['peak_on_background'] + 2)

    def test_the_internal_poisson_path_is_not_taken(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError('internal Poisson noise must not run here')

        monkeypatch.setattr(Noise2Noise, '_add_poisson_noise', fail)

        Noise2Noise(noise_fn=lambda clean, level: clean, seed=SEED).generate_pair(
            SPECTRA['peak_on_background'], 100.0
        )
