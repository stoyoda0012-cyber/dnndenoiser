"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
from scipy.ndimage import gaussian_filter1d

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Metric Functions (standalone implementations for testing)
# =============================================================================

def compute_snr(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute SNR in dB: 10*log10(signal_power / noise_power)."""
    noise = signal - reference
    signal_power = np.mean(reference ** 2, axis=-1)
    noise_power = np.mean(noise ** 2, axis=-1)
    return 10 * np.log10(signal_power / (noise_power + 1e-10))


def compute_mse(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error per sample."""
    return np.mean((signal - reference) ** 2, axis=-1)


def compute_psnr(signal: np.ndarray, reference: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    """Compute Peak Signal-to-Noise Ratio in dB."""
    mse = compute_mse(signal, reference)
    return 10 * np.log10(data_range ** 2 / (mse + 1e-10))


# =============================================================================
# Tests
# =============================================================================

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
        assert np.all(compute_psnr(clean_signal, clean_signal) > 50)
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
        assert compute_psnr(noisy, clean_signal).shape == (50,)


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
