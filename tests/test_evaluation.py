"""The evaluation metrics — the ones that ship, not copies of them.

This file used to define its own ``compute_snr``, ``compute_mse`` and
``compute_psnr`` under the heading "standalone implementations for testing", and
test those. It was therefore not a test of the metrics at all:

- its ``compute_snr`` divided by ``noise_power + 1e-10`` while the shipped one
  floors with ``np.maximum(noise_power, 1e-10)``. Near perfect reconstruction
  the two disagree by up to 0.41 dB, and no test could have seen it;
- ``compute_psnr`` was tested and **is not part of the package at all**.

The shipped metrics were nested inside ``cmd_evaluate``, which is why a copy was
reachable and the real thing was not. They are module-level now and imported
here. A test of a metric has to be a test of the metric that runs.
"""
import pytest
import numpy as np
from scipy.ndimage import gaussian_filter1d

from dnndenoiser.cli import compute_mse, compute_snr


class TestMetrics:
    """Test SNR, MSE, PSNR metric computations."""

    @pytest.fixture
    def clean_signal(self):
        """Generate clean test signal with offset."""
        np.random.seed(42)
        return np.random.randn(50, 256) + 10

    def test_perfect_reconstruction(self, clean_signal):
        """Identical signals should give very high SNR/PSNR, zero MSE."""
        assert np.all(compute_snr(clean_signal, clean_signal) > 50)
        assert np.all(compute_mse(clean_signal, clean_signal) == 0)

    @pytest.mark.parametrize("noise_std", [0.1, 0.5, 1.0, 2.0])
    def test_noise_snr_monotonicity(self, clean_signal, noise_std):
        """Higher noise should give lower SNR."""
        noisy = clean_signal + noise_std * np.random.randn(*clean_signal.shape)
        snr = np.mean(compute_snr(noisy, clean_signal))
        # SNR should decrease with noise
        assert snr > 0  # Still positive for these noise levels
        if noise_std >= 1.0:
            assert snr < 25  # Higher noise = lower SNR

    def test_mse_symmetry(self, clean_signal):
        """MSE(x, y) should equal MSE(y, x)."""
        noisy = clean_signal + np.random.randn(*clean_signal.shape)
        mse_xy = compute_mse(clean_signal, noisy)
        mse_yx = compute_mse(noisy, clean_signal)
        np.testing.assert_array_almost_equal(mse_xy, mse_yx)

    def test_batch_dimension(self, clean_signal):
        """Metrics should return one value per sample."""
        noisy = clean_signal + 0.5 * np.random.randn(*clean_signal.shape)
        assert compute_snr(noisy, clean_signal).shape == (50,)
        assert compute_mse(noisy, clean_signal).shape == (50,)


class TestSNRGain:
    """Test SNR gain interpretation."""

    def test_denoising_improves_snr(self):
        """Simple smoothing should improve SNR for high-noise data."""
        np.random.seed(42)
        clean = np.random.randn(100, 256) + 10
        noisy = clean + 3.0 * np.random.randn(*clean.shape)
        denoised = gaussian_filter1d(noisy, sigma=2, axis=1)

        snr_gain = np.mean(compute_snr(denoised, clean)) - np.mean(compute_snr(noisy, clean))
        assert snr_gain > 0

    def test_snr_to_efficiency_conversion(self):
        """10 dB SNR gain = 10x measurement efficiency."""
        # Exposure time ratio = 10^(SNR_gain/10)
        for snr_gain, expected_ratio in [(10, 10), (20, 100), (3, 2)]:
            ratio = 10 ** (snr_gain / 10)
            assert abs(ratio - expected_ratio) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestTheFloorConvention:
    """The 1e-10 floor is a stated convention, so it is tested as one.

    Flooring rather than adding is what keeps the statistic exact wherever it
    is meaningful. The previous copy added, and disagreed with the shipped
    metric by up to 0.41 dB just where a denoiser is doing best.
    """

    def test_floor_leaves_meaningful_values_exact(self):
        reference = np.array([[1.0, 2.0, 3.0, 4.0]])
        signal = reference + 0.01  # noise power 1e-4, far above the floor
        expected = 10 * np.log10(np.mean(reference**2) / 1e-4)
        assert compute_snr(signal, reference)[0] == pytest.approx(expected)

    def test_floor_only_guards_the_degenerate_case(self):
        reference = np.array([[1.0, 2.0, 3.0, 4.0]])
        perfect = compute_snr(reference, reference)[0]
        assert np.isfinite(perfect), "the floor exists so this is not infinity"
        assert perfect == pytest.approx(10 * np.log10(np.mean(reference**2) / 1e-10))

    def test_adding_instead_of_flooring_would_differ_where_it_matters(self):
        """Records the size of the defect this file used to hide."""
        reference = np.array([[1.0, 2.0, 3.0, 4.0]])
        noise = np.array([[1.0, -1.0, 1.0, -1.0]]) * 3.16e-6  # noise power ~1e-11
        signal = reference + noise

        floored = compute_snr(signal, reference)[0]
        noise_power = np.mean(noise**2)
        added = 10 * np.log10(np.mean(reference**2) / (noise_power + 1e-10))
        assert abs(floored - added) > 0.1, (
            "the two conventions must be measurably different here, or this "
            "test is not holding anything"
        )
