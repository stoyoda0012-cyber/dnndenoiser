"""Training paths reachable from the command line.

`--method noise2noise` used to raise ``AttributeError`` before training ever
started: the command read ``poisson_level`` off the wrong ``NoiseConfig`` — the
one in ``dnndenoiser.training.noise`` carries ``level`` instead. The end-to-end
smoke only exercises the default method, so nothing caught it.

Fixing the exception is not enough on its own. The level has to arrive at the
noise model as the regime the data actually sits in, and the two Poisson
parameterisations in this package differ: the generator draws counts with
``lambda = (10000/level)**2`` while ``Noise2Noise`` uses ``10000/level``. A fix
that merely stops the crash can still pair every spectrum with a target an order
of magnitude off, and exit zero. These tests pin the arriving number and the
resulting noise magnitude, not just the exit code.

Run with:
    pytest tests/test_cli_train.py -v
"""
import sys

import h5py
import numpy as np
import pytest
import torch

from dnndenoiser.cli import main, poisson_level_to_pair_level
from dnndenoiser.data.synthetic_generator import add_poisson_noise
from dnndenoiser.training.methods import POISSON_NOISE_FLOOR, Noise2Noise

# Small enough to keep the CLI round trip near a second on CPU.
N_SAMPLES = 16
N_ENERGY = 64
GENERATED_LEVEL = 500.0


def run_cli(monkeypatch, *argv):
    """Invoke the console entry point as the shell would."""
    monkeypatch.setattr(sys, 'argv', ['dnndenoiser', *argv])
    main()


@pytest.fixture
def dataset(monkeypatch, tmp_path):
    """A tiny generated dataset, produced through the CLI itself."""
    path = tmp_path / 'data.h5'
    run_cli(
        monkeypatch, 'generate',
        '-o', str(path),
        '-n', str(N_SAMPLES),
        '--n-energy', str(N_ENERGY),
        '--peak-set', 'C1s_single',
        '--poisson-level', str(GENERATED_LEVEL),
        '--seed', '0',
    )
    return path


def train_argv(dataset, model_path, *extra):
    return (
        'train',
        '-d', str(dataset),
        '-o', str(model_path),
        '--arch', 'FCNN',
        '--epochs', '1',
        '--batch-size', '8',
        '--device', 'cpu',
        *extra,
    )


@pytest.fixture
def levels_seen(monkeypatch):
    """Record the levels the command hands to the noise model."""
    seen = []
    original = Noise2Noise.generate_pair

    def spy(self, clean, noise_level):
        seen.append(noise_level)
        return original(self, clean, noise_level)

    monkeypatch.setattr(Noise2Noise, 'generate_pair', spy)
    return seen


class TestCliTraining:
    """dnndenoiser train"""

    @pytest.mark.parametrize('method', ['noise2clean', 'noise2noise'])
    def test_method_trains_and_writes_a_checkpoint(self, method, monkeypatch, dataset,
                                                   tmp_path):
        model_path = tmp_path / f'{method}.pt'
        extra = ('--noise-level', str(GENERATED_LEVEL)) if method == 'noise2noise' else ()

        run_cli(monkeypatch, *train_argv(dataset, model_path, '--method', method, *extra))

        assert model_path.is_file()
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        assert checkpoint['architecture'] == 'FCNN'
        assert checkpoint['n_features'] == N_ENERGY

    def test_noise2noise_requires_a_level(self, monkeypatch, dataset, tmp_path):
        """No default: the data file does not record the regime it was made in."""
        model_path = tmp_path / 'missing.pt'

        with pytest.raises(SystemExit) as exit_info:
            run_cli(monkeypatch, *train_argv(dataset, model_path, '--method', 'noise2noise'))

        assert exit_info.value.code == 1
        assert not model_path.exists()

    def test_the_level_reaches_the_noise_model_converted(self, monkeypatch, dataset,
                                                         tmp_path, levels_seen):
        """The number handed to the noise model, not the one printed.

        Passing the generate-style level straight through would leave 500 here,
        making the target sqrt(10000/500) = 4.5 times noisier than the input.
        """
        model_path = tmp_path / 'converted.pt'

        run_cli(monkeypatch, *train_argv(
            dataset, model_path, '--method', 'noise2noise',
            '--noise-level', str(GENERATED_LEVEL),
        ))

        assert levels_seen, 'the noise model was never asked for a pair'
        assert set(levels_seen) == {poisson_level_to_pair_level(GENERATED_LEVEL)}

    @pytest.mark.parametrize('level', ['0', '-1', 'nan', 'inf', '0.3'])
    def test_noise2noise_rejects_levels_it_cannot_honour(self, level, monkeypatch,
                                                         dataset, tmp_path):
        """Refusing beats training against a clean or NaN target."""
        model_path = tmp_path / 'rejected.pt'

        with pytest.raises(SystemExit) as exit_info:
            run_cli(monkeypatch, *train_argv(
                dataset, model_path, '--method', 'noise2noise', '--noise-level', level,
            ))

        assert exit_info.value.code == 1
        assert not model_path.exists()

    def test_spectra_not_normalized_to_their_own_peak_are_flagged(self, monkeypatch,
                                                                   tmp_path, capsys):
        """Angle-resolved generation normalizes by one global maximum.

        The array then peaks at 1 while individual spectra do not, so a check on
        the array maximum passes exactly when the conversion is off. This one is
        per spectrum.
        """
        data = tmp_path / 'angles.h5'
        run_cli(
            monkeypatch, 'generate', '-o', str(data), '-n', '4',
            '--n-energy', str(N_ENERGY), '--peak-set', 'C1s_single',
            '--n-angles', '12', '--angle-max', '80',
            '--poisson-level', str(GENERATED_LEVEL), '--seed', '0',
        )

        run_cli(monkeypatch, *train_argv(
            tmp_path / 'angles.h5', tmp_path / 'angles.pt',
            '--method', 'noise2noise', '--noise-level', str(GENERATED_LEVEL),
        ))

        assert 'individual clean spectra peak between' in capsys.readouterr().err

    def test_spectra_normalized_to_their_own_peak_are_not_flagged(self, monkeypatch,
                                                                  dataset, tmp_path,
                                                                  capsys):
        run_cli(monkeypatch, *train_argv(
            dataset, tmp_path / 'quiet.pt',
            '--method', 'noise2noise', '--noise-level', str(GENERATED_LEVEL),
        ))

        assert 'individual clean spectra peak between' not in capsys.readouterr().err

    def test_the_checkpoint_records_the_level(self, monkeypatch, dataset, tmp_path):
        """The argument for stating the level is that files do not record it."""
        model_path = tmp_path / 'recorded.pt'

        run_cli(monkeypatch, *train_argv(
            dataset, model_path, '--method', 'noise2noise',
            '--noise-level', str(GENERATED_LEVEL),
        ))

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        assert checkpoint['noise_level'] == GENERATED_LEVEL

    def test_the_checkpoint_records_no_level_for_noise2clean(self, monkeypatch, dataset,
                                                             tmp_path):
        """The option is documented as ignored there, so nothing is recorded."""
        model_path = tmp_path / 'no-level.pt'

        run_cli(monkeypatch, *train_argv(
            dataset, model_path, '--method', 'noise2clean', '--noise-level', '500',
        ))

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        assert checkpoint['noise_level'] is None

    def test_non_finite_clean_spectra_are_refused(self, monkeypatch, dataset,
                                                  tmp_path):
        """A NaN rate does not raise where it is introduced, only later.

        Poisson sampling turns it into an opaque error about the rate being too
        large, after the run has already spent its setup.
        """
        corrupted = tmp_path / 'corrupted.h5'
        with h5py.File(dataset, 'r') as source, h5py.File(corrupted, 'w') as target:
            for name in source:
                target.create_dataset(name, data=source[name][:])
            target['clean'][0, 0] = np.nan

        model_path = tmp_path / 'corrupted.pt'
        with pytest.raises(SystemExit) as exit_info:
            run_cli(monkeypatch, *train_argv(
                corrupted, model_path, '--method', 'noise2noise',
                '--noise-level', str(GENERATED_LEVEL),
            ))

        assert exit_info.value.code == 1
        assert not model_path.exists()

    def test_noise2clean_ignores_the_level(self, monkeypatch, dataset, tmp_path):
        """A level that noise2noise would reject must not affect noise2clean."""
        model_path = tmp_path / 'unaffected.pt'

        run_cli(monkeypatch, *train_argv(
            dataset, model_path, '--method', 'noise2clean', '--noise-level', '0',
        ))

        assert model_path.is_file()


class TestLevelConversion:
    """The two Poisson parameterisations have to line up."""

    @pytest.fixture
    def clean(self):
        """A peak on a background, max-normalized as the generator emits."""
        energy = np.linspace(0, 1, 256)
        peak = np.exp(-((energy - 0.5) ** 2) / (2 * 0.08 ** 2))
        spectrum = 0.85 * peak / peak.max() + 0.15
        return (spectrum / spectrum.max()).astype(np.float32)

    @staticmethod
    def residual_scale(spectra, clean):
        return float(np.std(np.stack(spectra) - clean))

    @pytest.mark.parametrize('level', [100.0, 500.0, 1000.0])
    def test_synthesized_realization_matches_the_generator(self, clean, level):
        """Same stated level, same noise magnitude — the claim the CLI makes."""
        rng = np.random.default_rng(0)
        generated = [
            add_poisson_noise(clean, level, rng=rng, use_gaussian_approx=False)
            for _ in range(40)
        ]
        pair_level = poisson_level_to_pair_level(level)
        synthesized = [
            Noise2Noise(seed=seed).generate_pair(clean, pair_level).target
            for seed in range(40)
        ]

        ratio = self.residual_scale(synthesized, clean) / self.residual_scale(generated, clean)
        assert 0.85 < ratio < 1.18, f'noise magnitudes differ by {ratio:.2f}x'

    @pytest.mark.parametrize('level', [100.0, 500.0, 1000.0])
    def test_passing_the_level_through_unconverted_would_not_match(self, clean, level):
        """Guards the conversion itself: the naive wiring is measurably wrong."""
        rng = np.random.default_rng(0)
        generated = [
            add_poisson_noise(clean, level, rng=rng, use_gaussian_approx=False)
            for _ in range(40)
        ]
        unconverted = [
            Noise2Noise(seed=seed).generate_pair(clean, level).target for seed in range(40)
        ]

        ratio = self.residual_scale(unconverted, clean) / self.residual_scale(generated, clean)
        assert ratio > 2.0, f'expected the unconverted level to be far noisier, got {ratio:.2f}x'


class TestSynthesizedRealization:
    """Properties noise2noise training rests on."""

    @pytest.fixture
    def clean(self):
        """A peak on a background, as a measured spectrum has.

        The background matters: a peak decaying to zero concentrates almost all
        the variance in a handful of bins, so pooling residuals across bins buys
        far less precision than the bin count suggests.
        """
        energy = np.linspace(0, 1, N_ENERGY)
        peak = np.exp(-((energy - 0.5) ** 2) / (2 * 0.1 ** 2))
        return (0.85 * peak / peak.max() + 0.15).astype(np.float32)

    def test_the_two_realizations_are_independent(self, clean):
        """Noise2Noise rests on the two noise draws being uncorrelated.

        Correlated per bin across realizations, then averaged, rather than
        pooling every bin into one sample: bin variances span orders of
        magnitude across a peak, and pooling lets the few peak bins set the
        effective sample size. Per bin the draws are identically scaled, so 128
        realizations give each of the 64 estimates the same weight. Across 40
        seeds this statistic stays under 0.024, half the threshold.
        """
        method = Noise2Noise(seed=0)
        pairs = [method.generate_pair(clean, 100.0) for _ in range(128)]

        first = np.stack([p.input - clean for p in pairs])
        second = np.stack([p.target - clean for p in pairs])

        per_bin = [
            np.corrcoef(first[:, bin_], second[:, bin_])[0, 1]
            for bin_ in range(clean.size)
        ]
        correlation = float(np.mean(per_bin))
        assert abs(correlation) < 0.05, f'residuals correlate: {correlation:+.3f}'

    @pytest.mark.parametrize('level', [10.0, 100.0, 1000.0])
    def test_a_higher_level_means_more_noise(self, clean, level):
        quieter = Noise2Noise(seed=0).generate_pair(clean, level).target
        noisier = Noise2Noise(seed=0).generate_pair(clean, level * 10).target

        assert np.std(noisier - clean) > np.std(quieter - clean)

    def test_a_level_at_the_floor_returns_the_clean_spectrum(self, clean):
        """Why the CLI refuses such a level instead of passing it through."""
        pair = Noise2Noise(seed=0).generate_pair(clean, POISSON_NOISE_FLOOR)

        assert np.array_equal(pair.target, clean)
        assert np.array_equal(pair.input, clean)
