"""Training Methods for Spectral Denoising

Implements three training paradigms:
- Noise2Clean: Learn to map noisy → clean (requires clean reference)
- Noise2Noise: Learn to map noisy → noisy (independent noise realizations)
- Noise2Self: Self-supervised using masking (single noisy measurement)

Each method defines how to generate input/target pairs from clean spectra.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

# Poisson levels at or below this add no noise at all: the "noisy" realization
# would be the clean spectrum. Callers that mean to add noise should treat a
# level this small as a configuration error rather than a quiet no-op.
# This bounds the level, not the Poisson rate; the rate is floored at zero.
POISSON_NOISE_FLOOR = 1e-5


class TrainingMethodType(Enum):
    """Available training methods."""
    NOISE2CLEAN = "noise2clean"
    NOISE2NOISE = "noise2noise"
    NOISE2SELF = "noise2self"


@dataclass
class TrainingPair:
    """Container for input/target training pairs.

    Attributes:
        input: Noisy input spectrum/spectra
        target: Target for training (clean, noisy, or masked)
        clean: Original clean spectrum (for evaluation, optional)
        mask: Binary mask for Noise2Self (optional)
    """
    input: np.ndarray
    target: np.ndarray
    clean: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


class TrainingMethod(ABC):
    """Abstract base class for training methods."""

    def __init__(self, noise_fn: Optional[Callable] = None):
        """Initialize training method.

        Args:
            noise_fn: Function to add noise. Signature: noise_fn(clean, noise_level) -> noisy
                      If None, uses default Poisson noise.
        """
        self.noise_fn = noise_fn or self._default_poisson_noise

    @staticmethod
    def _default_poisson_noise(clean: np.ndarray, noise_level: float) -> np.ndarray:
        """Default Poisson noise model for XPS data.

        Args:
            clean: Clean spectrum
            noise_level: Noise scaling factor (lower = more noise)

        Returns:
            Noisy spectrum
        """
        # Avoid division by zero
        if noise_level <= 0:
            noise_level = 1e-6

        # Scale, apply Poisson, scale back
        scaled = clean / noise_level
        # Ensure non-negative for Poisson
        scaled = np.maximum(scaled, 0)
        noisy = np.random.poisson(scaled).astype(np.float32)
        return noisy * noise_level

    @property
    @abstractmethod
    def method_type(self) -> TrainingMethodType:
        """Return the training method type."""
        pass

    @property
    @abstractmethod
    def requires_clean_target(self) -> bool:
        """Whether this method requires clean spectra for training."""
        pass

    @property
    def shuffle_each_epoch(self) -> bool:
        """Whether to shuffle data each epoch. Override if needed."""
        return True

    @abstractmethod
    def generate_pair(
        self,
        clean: np.ndarray,
        noise_level: float
    ) -> TrainingPair:
        """Generate input/target pair from clean spectrum.

        Args:
            clean: Clean spectrum or batch of spectra
            noise_level: Noise level parameter

        Returns:
            TrainingPair with input and target
        """
        pass

    def generate_batch(
        self,
        clean_batch: np.ndarray,
        noise_levels: np.ndarray
    ) -> TrainingPair:
        """Generate batch of input/target pairs.

        Args:
            clean_batch: (N, ...) batch of clean spectra
            noise_levels: (N,) noise levels for each sample

        Returns:
            TrainingPair with batched input and target
        """
        inputs = []
        targets = []
        masks = []

        for i, (clean, noise_level) in enumerate(zip(clean_batch, noise_levels)):
            pair = self.generate_pair(clean, noise_level)
            inputs.append(pair.input)
            targets.append(pair.target)
            if pair.mask is not None:
                masks.append(pair.mask)

        return TrainingPair(
            input=np.stack(inputs),
            target=np.stack(targets),
            clean=clean_batch,
            mask=np.stack(masks) if masks else None
        )


class Noise2Clean(TrainingMethod):
    """Noise2Clean: Supervised denoising.

    - Input: Noisy spectrum
    - Target: Clean spectrum
    - Requires: Clean reference data

    This is the standard supervised approach where we learn
    a direct mapping from noisy to clean.
    """

    @property
    def method_type(self) -> TrainingMethodType:
        return TrainingMethodType.NOISE2CLEAN

    @property
    def requires_clean_target(self) -> bool:
        return True

    def generate_pair(
        self,
        clean: np.ndarray,
        noise_level: float
    ) -> TrainingPair:
        """Generate noisy input with clean target."""
        noisy = self.noise_fn(clean, noise_level)
        return TrainingPair(
            input=noisy,
            target=clean.copy(),
            clean=clean.copy()
        )


class Noise2Noise(TrainingMethod):
    """Noise2Noise: Learning from noisy pairs.

    - Input: Noisy spectrum (realization 1)
    - Target: Noisy spectrum (realization 2)
    - Requires: Ability to generate multiple noise realizations

    Key insight: If noise is zero-mean and independent between
    input and target, the network learns to predict the clean signal.

    IMPORTANT: Uses two independent RNGs to ensure noise independence.
    This is critical for correct noise2noise training.

    Reference: Lehtinen et al., "Noise2Noise: Learning Image Restoration
    without Clean Data", ICML 2018.
    """

    def __init__(self, noise_fn: Optional[Callable] = None, seed: int = 42):
        """Initialize Noise2Noise with independent RNGs.

        Args:
            noise_fn: Custom noise function (if None, uses internal Poisson)
            seed: Base seed for RNGs (two RNGs use seed and seed+1000)
        """
        super().__init__(noise_fn)
        # Two independent RNGs for generating independent noise
        self._rng1 = np.random.default_rng(seed)
        self._rng2 = np.random.default_rng(seed + 1000)
        self._use_internal_noise = noise_fn is None

    @property
    def method_type(self) -> TrainingMethodType:
        return TrainingMethodType.NOISE2NOISE

    @property
    def requires_clean_target(self) -> bool:
        # Still need clean to generate two independent noisy versions
        # But doesn't need clean "target" per se
        return True

    def _add_poisson_noise(self, clean: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
        """Add Poisson noise using specific RNG for independence.

        The counts are floored at zero and no higher. Noise2Noise learns the
        clean signal because each realization returns it on average, and a
        positive floor would break exactly that: bins whose expected counts fall
        below it come back biased upward, identically in both realizations.
        ``_default_poisson_noise`` above and
        ``data.synthetic_generator.add_poisson_noise`` already floor at zero, so
        this is their treatment rather than a second noise model. A generator
        that floors at a positive value instead carries exactly the bias this
        avoids; none is reachable from here.

        ``noise_level`` is this method's own parameter, not the one
        ``generate --poisson-level`` takes: counts scale as ``10000 / level``
        here and as ``(10000 / level) ** 2`` there. ``cli`` converts between
        them; a caller who does not is choosing a different noise regime.

        Args:
            clean: Clean spectrum. Assumed non-negative and finite. Negative
                values, negative infinity included, are clipped to zero, as
                Poisson rates cannot be negative; NaN and positive infinity
                raise from the sampler.
            noise_level: Noise level (higher = more noise)
            rng: Random number generator to use

        Returns:
            Noisy spectrum, with ``E[noisy] == clean`` for non-negative input.
        """
        if noise_level <= POISSON_NOISE_FLOOR:
            return clean.copy()

        scale_factor = 10000.0 / noise_level
        scaled = np.maximum(clean * scale_factor, 0.0)
        noisy = rng.poisson(scaled).astype(np.float32)
        return noisy / scale_factor

    def generate_pair(
        self,
        clean: np.ndarray,
        noise_level: float
    ) -> TrainingPair:
        """Generate two independent noisy realizations.

        Uses two independent RNGs to guarantee noise independence.
        """
        if self._use_internal_noise:
            # Use internal Poisson noise with independent RNGs
            noisy_input = self._add_poisson_noise(clean, noise_level, self._rng1)
            noisy_target = self._add_poisson_noise(clean, noise_level, self._rng2)
        else:
            # Use provided noise function (note: may not guarantee independence!)
            noisy_input = self.noise_fn(clean, noise_level)
            noisy_target = self.noise_fn(clean, noise_level)

        return TrainingPair(
            input=noisy_input,
            target=noisy_target,
            clean=clean.copy()
        )


class Noise2Self(TrainingMethod):
    """Noise2Self: Self-supervised denoising with masking.

    - Input: Noisy spectrum with some values masked
    - Target: Original noisy values at masked positions
    - Requires: Only the noisy measurement itself

    The network learns to predict masked values from unmasked neighbors.
    Only works if the underlying signal is locally correlated (smooth).

    Reference: Batson & Royer, "Noise2Self: Blind Denoising by
    Self-Supervision", ICML 2019.
    """

    def __init__(
        self,
        noise_fn: Optional[Callable] = None,
        mask_ratio: float = 0.2,
        mask_type: str = "random"
    ):
        """Initialize Noise2Self method.

        Args:
            noise_fn: Noise function (used for data augmentation if needed)
            mask_ratio: Fraction of values to mask (0.1-0.3 typical)
            mask_type: "random" for random pixels, "structured" for patterns
        """
        super().__init__(noise_fn)
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

    @property
    def method_type(self) -> TrainingMethodType:
        return TrainingMethodType.NOISE2SELF

    @property
    def requires_clean_target(self) -> bool:
        return False  # Only needs noisy data!

    @property
    def shuffle_each_epoch(self) -> bool:
        # Noise2Self benefits from consistent masking patterns
        # across epochs for stability
        return False

    def _generate_mask(self, shape: tuple) -> np.ndarray:
        """Generate binary mask.

        Args:
            shape: Shape of the spectrum

        Returns:
            Binary mask (1 = keep, 0 = mask out)
        """
        if self.mask_type == "random":
            mask = np.random.random(shape) > self.mask_ratio
        elif self.mask_type == "structured":
            # Structured masking: mask every Nth point
            n = int(1 / self.mask_ratio)
            mask = np.ones(shape, dtype=bool)
            mask[::n] = False
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

        return mask.astype(np.float32)

    def generate_pair(
        self,
        clean: np.ndarray,
        noise_level: float
    ) -> TrainingPair:
        """Generate masked input with unmasked noisy target.

        For Noise2Self, we work directly with the noisy measurement.
        The 'clean' input here is actually the measured (noisy) spectrum.
        """
        # Generate noisy version (or use clean as-is if it's already noisy)
        noisy = self.noise_fn(clean, noise_level)

        # Generate mask
        mask = self._generate_mask(noisy.shape)

        # Masked input: set masked positions to local mean or zero
        masked_input = noisy * mask

        # For 1D spectra, interpolate masked values
        if noisy.ndim == 1:
            masked_input = self._interpolate_masked(noisy, mask)

        return TrainingPair(
            input=masked_input,
            target=noisy.copy(),  # Target is the full noisy spectrum
            clean=clean.copy() if clean is not noisy else None,
            mask=mask
        )

    def _interpolate_masked(
        self,
        spectrum: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Interpolate masked values from neighbors.

        This helps the network by not having discontinuities at masked points.
        """
        result = spectrum.copy()
        masked_indices = np.where(mask == 0)[0]

        for idx in masked_indices:
            # Simple linear interpolation from neighbors
            left = idx - 1
            right = idx + 1

            while left >= 0 and mask[left] == 0:
                left -= 1
            while right < len(spectrum) and mask[right] == 0:
                right -= 1

            if left >= 0 and right < len(spectrum):
                # Interpolate
                result[idx] = (spectrum[left] + spectrum[right]) / 2
            elif left >= 0:
                result[idx] = spectrum[left]
            elif right < len(spectrum):
                result[idx] = spectrum[right]
            # else: keep original (shouldn't happen with reasonable mask_ratio)

        return result


class Noise2SelfFromNoisy(Noise2Self):
    """Noise2Self variant that works directly with pre-existing noisy data.

    Use this when you only have noisy measurements (no clean reference).
    """

    def generate_pair(
        self,
        noisy: np.ndarray,
        noise_level: float = 0.0  # Ignored
    ) -> TrainingPair:
        """Generate masked input from noisy spectrum.

        Args:
            noisy: Already noisy spectrum
            noise_level: Ignored (noise already present)
        """
        mask = self._generate_mask(noisy.shape)

        if noisy.ndim == 1:
            masked_input = self._interpolate_masked(noisy, mask)
        else:
            masked_input = noisy * mask

        return TrainingPair(
            input=masked_input,
            target=noisy.copy(),
            clean=None,
            mask=mask
        )


def create_training_method(
    method_type: TrainingMethodType,
    noise_fn: Optional[Callable] = None,
    **kwargs
) -> TrainingMethod:
    """Factory function to create training method.

    Args:
        method_type: Type of training method
        noise_fn: Optional custom noise function
        **kwargs: Additional arguments for specific methods
            - mask_ratio: For Noise2Self (default: 0.2)
            - mask_type: For Noise2Self (default: 'random')
            - seed: For Noise2Noise (default: 42)

    Returns:
        TrainingMethod instance
    """
    if method_type == TrainingMethodType.NOISE2CLEAN:
        return Noise2Clean(noise_fn=noise_fn)
    elif method_type == TrainingMethodType.NOISE2NOISE:
        seed = kwargs.get('seed', 42)
        return Noise2Noise(noise_fn=noise_fn, seed=seed)
    elif method_type == TrainingMethodType.NOISE2SELF:
        mask_ratio = kwargs.get('mask_ratio', 0.2)
        mask_type = kwargs.get('mask_type', 'random')
        return Noise2Self(
            noise_fn=noise_fn,
            mask_ratio=mask_ratio,
            mask_type=mask_type
        )
    else:
        raise ValueError(f"Unknown training method: {method_type}")


# Convenience aliases
N2C = Noise2Clean
N2N = Noise2Noise
N2S = Noise2Self
