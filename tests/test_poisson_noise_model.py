"""The synthetic generator's noise model is Poisson, and the approximation is bounded.

Until 2026-09-11, `add_poisson_noise` defaulted to a Gaussian approximation
selected once from the *peak* expected count and applied to every bin. The
approximation can go negative, negatives are floored at zero, and the resulting
upward bias depends only on a bin's own expected count -- so a bright peak
licensed the approximation for dark bins where it is worst.

The admissibility floor is derived, not chosen: the clipped Gaussian's relative
bias is ``phi(sqrt(l))/sqrt(l) - Phi(-sqrt(l))``, and a 1% tolerance fixes ``l``.
See ``GAUSSIAN_APPROX_MIN_RATE``.
"""
import numpy as np
import pytest

from dnndenoiser.data.synthetic_generator import (
    GAUSSIAN_APPROX_MIN_RATE,
    NoiseConfig,
    add_noise,
    add_poisson_noise,
)

# Levels whose peak expected count is an exact power of two. With a spectrum
# normalised to a maximum of exactly 1.0, recovering counts from the returned
# float32 is a pure exponent shift, so integrality separates a Poisson draw from
# a Gaussian one exactly rather than within a tolerance.
EXACT = {78.125: 16384.0, 312.5: 1024.0, 1250.0: 64.0, 2500.0: 16.0}


@pytest.fixture
def unit_peak():
    e = np.linspace(-10, 10, 256)
    p = np.exp(-e ** 2 / 2)
    return p / p.max()


def counts(returned, rate_at_peak):
    return returned.astype(np.float64) * rate_at_peak


def is_integral(x):
    return np.all(x == np.rint(x))


@pytest.mark.parametrize("level,peak_rate", sorted(EXACT.items()))
def test_default_draws_poisson_counts_in_every_bin(unit_peak, level, peak_rate):
    """The default is Poisson, for the dark bins too -- not only where it is cheap."""
    got = counts(add_poisson_noise(unit_peak, level, rng=np.random.default_rng(0)),
                 peak_rate)
    assert is_integral(got), "the default path returned a non-integer count"
    assert np.all(got >= 0)


@pytest.mark.parametrize("level,peak_rate", sorted(EXACT.items()))
def test_opt_in_approximation_never_reaches_a_low_rate_bin(unit_peak, level, peak_rate):
    """Admissibility is a per-bin property and is enforced per bin."""
    rates = peak_rate * unit_peak
    low = rates < GAUSSIAN_APPROX_MIN_RATE
    high = ~low
    assert low.any() and high.any(), "this level does not exercise both branches"

    got = counts(add_poisson_noise(unit_peak, level, use_gaussian_approx=True,
                                   rng=np.random.default_rng(0)), peak_rate)
    assert is_integral(got[low]), "a bin below the floor was approximated"
    # A clipped draw is exactly 0.0, which is integral and is still a Gaussian
    # output, so it cannot discriminate; every other high-rate bin must be
    # continuous.
    above = got[high]
    assert not is_integral(above[above != 0.0])


def test_the_floor_is_the_one_percent_bias_point():
    """The threshold is derived, not chosen: 1% relative bias fixes it."""
    from scipy.stats import norm
    s = np.sqrt(GAUSSIAN_APPROX_MIN_RATE)
    rel_bias = norm.pdf(s) / s - norm.cdf(-s)
    assert rel_bias <= 0.01, f"the floor admits {rel_bias:.3%} bias, over the 1% tolerance"

    # Pinning only "<= 1%" leaves a whole window open -- 3.5 and 3.9 satisfy it
    # too -- so the constant could drift upward without a test noticing. Pin it
    # to the root of the tolerance equation, which is what actually determines
    # it, rather than to a bound it happens to clear.
    from scipy.optimize import brentq
    exact = brentq(lambda x: norm.pdf(np.sqrt(x)) / np.sqrt(x)
                   - norm.cdf(-np.sqrt(x)) - 0.01, 1e-6, 1e4)
    assert exact < GAUSSIAN_APPROX_MIN_RATE <= np.ceil(exact), (
        f"the floor should be ceil({exact:.4f}) = {np.ceil(exact)}, "
        f"not {GAUSSIAN_APPROX_MIN_RATE}")


def test_noise_config_carries_both_knobs(unit_peak):
    """A caller can name the model instead of depending on what the default is."""
    cfg = NoiseConfig(noise_type="poisson", poisson_level=312.5)
    assert cfg.use_gaussian_approx is False and cfg.gaussian_approx_min_rate is None
    plain = add_noise(unit_peak, cfg, rng=np.random.default_rng(3))
    assert is_integral(counts(plain, 1024.0))


def test_the_superseded_rule_stays_reachable_and_exact(unit_peak):
    """`benchmarks/reference/` reproduces a measurement taken under the old model.

    Pinning the flag alone would not do it: the flag's meaning changed. The old
    rule was "approximate every bin iff the PEAK rate exceeds 20", and a floor of
    zero is what reconstructs it.
    """
    def superseded_rule(data, level, rng):
        dm = np.maximum(data, 0).max()
        scale = (10000.0 / level) ** 2
        scaled = np.clip(scale * (np.maximum(data, 0) / dm), 0, 1e12)
        if scale > 20:
            drawn = np.maximum(
                scaled + rng.standard_normal(scaled.shape) * np.sqrt(scaled), 0)
        else:
            drawn = rng.poisson(scaled).astype(np.float64)
        return (drawn / scale * dm).astype(np.float32)

    for level in (100.0, 1000.0, 10000.0):
        cfg = NoiseConfig(noise_type="poisson", poisson_level=level,
                          use_gaussian_approx=(10000.0 / level) ** 2 > 20,
                          gaussian_approx_min_rate=0.0)
        mine = add_noise(unit_peak, cfg, rng=np.random.default_rng(11))
        theirs = superseded_rule(unit_peak, level, np.random.default_rng(11))
        assert np.array_equal(mine, theirs), f"level {level} is no longer reproducible"
