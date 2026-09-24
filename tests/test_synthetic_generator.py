"""Unit tests for synthetic data generator."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dnndenoiser.data.synthetic_generator import (
    gaussian, lorentzian, voigt_profile, pseudo_voigt,
    SyntheticGenerator, GeneratorConfig, NoiseConfig,
    list_peak_sets
)


# =============================================================================
# Peak Shapes
# =============================================================================

class TestPeakShapes:
    """Test peak shape functions: normalization and limiting behavior."""

    @pytest.mark.parametrize("func,kwargs,x_range", [
        (gaussian, {'mu': 0, 'sigma': 1}, (-10, 10, 1000)),
        (lorentzian, {'mu': 0, 'gamma': 1}, (-100, 100, 10000)),
    ])
    def test_normalization(self, func, kwargs, x_range):
        """Peak shapes should integrate to 1."""
        x = np.linspace(*x_range)
        y = func(x, **kwargs)
        # np.trapezoid is NumPy >= 2.0; pyproject declares numpy >= 1.24.
        trapezoid = getattr(np, 'trapezoid', None) or np.trapz
        integral = trapezoid(y, x)
        assert abs(integral - 1.0) < 0.01

    def test_voigt_limiting_cases(self):
        """Voigt should reduce to Gaussian/Lorentzian at limits."""
        x = np.linspace(-10, 10, 1000)

        # gamma → 0: should be Gaussian
        v_gauss = voigt_profile(x, mu=0, sigma=1, gamma=1e-15)
        g = gaussian(x, mu=0, sigma=1)
        np.testing.assert_allclose(v_gauss, g, rtol=1e-3)

        # sigma → 0: should be Lorentzian
        v_lor = voigt_profile(x, mu=0, sigma=1e-15, gamma=1)
        lor = lorentzian(x, mu=0, gamma=1)
        np.testing.assert_allclose(v_lor, lor, rtol=1e-3)

    def test_pseudo_voigt_eta(self):
        """Pseudo-Voigt eta controls Gaussian/Lorentzian mix."""
        x = np.linspace(-10, 10, 1000)
        pv_g = pseudo_voigt(x, mu=0, fwhm=2.0, eta=0.0)  # Pure Gaussian
        pv_l = pseudo_voigt(x, mu=0, fwhm=2.0, eta=1.0)  # Pure Lorentzian

        # Same peak position, but Lorentzian has wider tails
        assert np.argmax(pv_g) == np.argmax(pv_l)
        tail_idx = np.abs(x) > 4
        assert np.sum(pv_l[tail_idx]) > np.sum(pv_g[tail_idx])


# =============================================================================
# Generator
# =============================================================================

class TestSyntheticGenerator:
    """Test synthetic spectrum generation."""

    @pytest.fixture
    def generator(self):
        """Create a standard generator."""
        gen_config = GeneratorConfig(n_energy_points=128)
        noise_config = NoiseConfig(noise_type='poisson', poisson_level=1000)
        return SyntheticGenerator('C1s_single', noise_config, gen_config)

    def test_output_shapes(self, generator):
        """Generated data should have correct shapes."""
        clean, noisy, energy, meta = generator.generate_batch(10, seed=42)
        assert clean.shape == (10, 128)
        assert noisy.shape == (10, 128)
        assert energy.shape == (128,)
        assert len(meta) == 10

    def test_reproducibility(self, generator):
        """Same seed should produce identical results."""
        c1, n1, _, _ = generator.generate_batch(5, seed=42)
        c2, n2, _, _ = generator.generate_batch(5, seed=42)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(n1, n2)

    def test_noise_effect(self, generator):
        """Noisy data should differ from clean."""
        clean, noisy, _, _ = generator.generate_batch(50, seed=42)
        mse = np.mean((noisy - clean) ** 2)
        assert mse > 1e-6

    @pytest.mark.parametrize("peak_set", list_peak_sets())
    def test_all_peak_sets(self, peak_set):
        """All peak set presets should generate valid data."""
        gen = SyntheticGenerator(
            peak_set,
            NoiseConfig(poisson_level=100),
            GeneratorConfig(n_energy_points=64)
        )
        clean, noisy, _, _ = gen.generate_batch(2, seed=42)
        assert not np.any(np.isnan(clean))
        assert not np.any(np.isnan(noisy))


# =============================================================================
# HDF5 Export
# =============================================================================

class TestHDF5Export:
    """Test HDF5 file operations."""

    def test_save_and_load(self):
        """Should save to HDF5 and contain expected datasets."""
        gen = SyntheticGenerator(
            'C1s_single',
            NoiseConfig(poisson_level=1000),
            GeneratorConfig(n_energy_points=64)
        )
        clean, noisy, energy, meta = gen.generate_batch(20, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.h5'
            SyntheticGenerator.save_hdf5(path, clean, noisy, energy, meta)

            import h5py
            with h5py.File(path, 'r') as f:
                assert set(['clean', 'noisy', 'energy']).issubset(f.keys())
                assert f['clean'].shape == (20, 64)

    @pytest.mark.parametrize("fmt,suffix", [('jsonl', '.jsonl'), ('csv', '.csv')])
    def test_save_manifest(self, fmt, suffix, tmp_path):
        """Should write one record per sample, with the encoding named.

        Under CI's ``PYTHONWARNDEFAULTENCODING=1`` and the ``EncodingWarning``
        filter in ``pyproject.toml``, an ``open`` without ``encoding`` fails here
        rather than only on a non-UTF-8 locale such as cp932.
        """
        gen = SyntheticGenerator(
            'C1s_single',
            NoiseConfig(poisson_level=1000),
            GeneratorConfig(n_energy_points=64)
        )
        _, _, _, meta = gen.generate_batch(3, seed=42)

        path = SyntheticGenerator.save_manifest(tmp_path / 'manifest', meta, format=fmt)

        assert path.suffix == suffix
        lines = path.read_text(encoding='utf-8').splitlines()
        assert len(lines) == 3 + (fmt == 'csv')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
