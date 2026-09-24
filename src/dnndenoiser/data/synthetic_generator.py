"""
Synthetic XPS Spectrum Generator for Denoising Training Data.

Generates synthetic XPS spectra with:
- Voigt/Pseudo-Voigt peaks with configurable K (Lorentzian fraction)
- Peak sets defining mu (center) and FWHM
- Poisson/Gaussian/Mixed noise models
- Simple background (linear, constant, Shirley-like)

Output:
- HDF5 files with /clean and /noisy datasets in (N, E) row-chunked format
- Manifest files (JSONL/CSV) recording noise conditions, seeds, and peak_set_id

Based on voigtfit reference implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Literal, Union
from pathlib import Path
import json
import csv
import h5py
from scipy.special import wofz  # Faddeeva function for true Voigt


# =============================================================================
# Peak Shape Functions
# =============================================================================

def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Normalized Gaussian profile.

    Args:
        x: Energy axis
        mu: Center position
        sigma: Standard deviation (NOT FWHM)

    Returns:
        Normalized Gaussian profile
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def lorentzian(x: np.ndarray, mu: float, gamma: float) -> np.ndarray:
    """Normalized Lorentzian (Cauchy) profile.

    Args:
        x: Energy axis
        mu: Center position
        gamma: Half-width at half-maximum (HWHM)

    Returns:
        Normalized Lorentzian profile
    """
    return gamma / (np.pi * ((x - mu) ** 2 + gamma ** 2))


def voigt_profile(x: np.ndarray, mu: float, sigma: float, gamma: float) -> np.ndarray:
    """
    True Voigt profile using Faddeeva function.

    The Voigt profile is the convolution of Gaussian and Lorentzian:
        V(x) = Re[w(z)] / (sigma * sqrt(2*pi))
    where w(z) is the Faddeeva function and z = (x - mu + i*gamma) / (sigma * sqrt(2))

    Args:
        x: Energy axis
        mu: Center position
        sigma: Gaussian standard deviation
        gamma: Lorentzian half-width (HWHM)

    Returns:
        Normalized Voigt profile
    """
    if sigma < 1e-10:
        # Pure Lorentzian
        return lorentzian(x, mu, gamma)
    if gamma < 1e-10:
        # Pure Gaussian
        return gaussian(x, mu, sigma)

    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def pseudo_voigt(x: np.ndarray, mu: float, fwhm: float, eta: float) -> np.ndarray:
    """
    Pseudo-Voigt profile: linear combination of Gaussian and Lorentzian.

    PV(x) = eta * L(x) + (1 - eta) * G(x)

    This is faster than true Voigt and often sufficient for XPS.

    Args:
        x: Energy axis
        mu: Center position
        fwhm: Full width at half maximum (shared by both components)
        eta: Lorentzian mixing fraction (0 = pure Gaussian, 1 = pure Lorentzian)

    Returns:
        Pseudo-Voigt profile (normalized to peak height = 1)
    """
    # Convert FWHM to component widths
    # Gaussian: FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.3548 * sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    # Lorentzian: FWHM = 2 * gamma
    gamma = fwhm / 2

    # Normalized profiles (peak height = 1 for comparison)
    g = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    lor = 1 / (1 + ((x - mu) / gamma) ** 2)

    return eta * lor + (1 - eta) * g


def voigt_from_fwhm_and_eta(
    x: np.ndarray,
    mu: float,
    fwhm: float,
    eta: float,
    use_pseudo: bool = True
) -> np.ndarray:
    """
    Generate Voigt-like profile from FWHM and Lorentzian fraction (eta/K).

    Args:
        x: Energy axis
        mu: Center position
        fwhm: Total FWHM
        eta: Lorentzian fraction (K parameter, 0-1)
            0 = pure Gaussian
            1 = pure Lorentzian
        use_pseudo: If True, use pseudo-Voigt (faster). If False, use true Voigt.

    Returns:
        Peak profile (normalized to peak height = 1)
    """
    if use_pseudo:
        return pseudo_voigt(x, mu, fwhm, eta)
    else:
        # Convert FWHM and eta to sigma and gamma for true Voigt
        # Approximate relationship (Thompson et al. 1987):
        # fwhm_G = fwhm * (1 - eta)^0.5 approximately
        # fwhm_L = fwhm * eta approximately
        # More accurate: use iterative fitting or lookup tables

        # Simple approximation:
        fwhm_g = fwhm * np.sqrt(1 - eta) if eta < 1 else 0
        fwhm_l = fwhm * eta

        sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2))) if fwhm_g > 0 else 1e-10
        gamma = fwhm_l / 2 if fwhm_l > 0 else 1e-10

        profile = voigt_profile(x, mu, sigma, gamma)
        # Normalize to peak height = 1
        if profile.max() > 0:
            profile = profile / profile.max()
        return profile


# =============================================================================
# Peak Set Configuration
# =============================================================================

@dataclass
class PeakParams:
    """Parameters for a single peak."""
    mu: float           # Center position (eV)
    fwhm: float         # Full width at half maximum (eV)
    intensity: float = 1.0  # Relative intensity

    def __post_init__(self):
        if self.fwhm <= 0:
            raise ValueError(f"FWHM must be positive, got {self.fwhm}")


@dataclass
class PeakSet:
    """A named collection of peaks that define a spectrum template."""
    id: str                     # Unique identifier (e.g., "C1s_single", "O1s_oxide")
    name: str                   # Display name
    peaks: List[PeakParams]     # List of peaks
    energy_range: Tuple[float, float] = None  # Optional energy range override
    description: str = ""       # Optional description

    def __post_init__(self):
        if not self.peaks:
            raise ValueError("PeakSet must have at least one peak")

        # Auto-calculate energy range if not specified
        if self.energy_range is None:
            centers = [p.mu for p in self.peaks]
            margin = max(p.fwhm for p in self.peaks) * 5
            self.energy_range = (min(centers) - margin, max(centers) + margin)


# Predefined peak sets for common XPS lines
PEAK_SETS: Dict[str, PeakSet] = {
    "C1s_single": PeakSet(
        id="C1s_single",
        name="C 1s (Single Peak)",
        peaks=[PeakParams(mu=284.8, fwhm=1.2)],
        description="Single C-C/C-H peak"
    ),
    "C1s_adventitious": PeakSet(
        id="C1s_adventitious",
        name="C 1s (Adventitious Carbon)",
        peaks=[
            PeakParams(mu=284.8, fwhm=1.2, intensity=1.0),   # C-C/C-H
            PeakParams(mu=286.3, fwhm=1.3, intensity=0.3),   # C-O
            PeakParams(mu=288.5, fwhm=1.4, intensity=0.15),  # C=O
        ],
        description="Typical adventitious carbon with C-O, C=O"
    ),
    "C1s_polymer": PeakSet(
        id="C1s_polymer",
        name="C 1s (Polymer)",
        peaks=[
            PeakParams(mu=285.0, fwhm=1.0, intensity=1.0),   # C-C
            PeakParams(mu=286.5, fwhm=1.1, intensity=0.5),   # C-O
            PeakParams(mu=289.0, fwhm=1.2, intensity=0.2),   # O-C=O
        ],
        description="Polymer with ester groups"
    ),
    "O1s_oxide": PeakSet(
        id="O1s_oxide",
        name="O 1s (Metal Oxide)",
        peaks=[
            PeakParams(mu=530.0, fwhm=1.3, intensity=1.0),   # Lattice O
            PeakParams(mu=531.5, fwhm=1.5, intensity=0.4),   # OH/vacancy
        ],
        description="Metal oxide with hydroxyl/vacancy"
    ),
    "Si2p_oxide": PeakSet(
        id="Si2p_oxide",
        name="Si 2p (SiO2)",
        peaks=[
            PeakParams(mu=99.3, fwhm=0.8, intensity=0.3),    # Si(0)
            PeakParams(mu=103.3, fwhm=1.5, intensity=1.0),   # SiO2
        ],
        description="Silicon with native oxide"
    ),
    "N1s_amine": PeakSet(
        id="N1s_amine",
        name="N 1s (Amine)",
        peaks=[PeakParams(mu=399.5, fwhm=1.4)],
        description="Primary amine"
    ),
    "test_doublet": PeakSet(
        id="test_doublet",
        name="Test Doublet",
        peaks=[
            PeakParams(mu=285.0, fwhm=1.0, intensity=1.0),
            PeakParams(mu=287.0, fwhm=1.2, intensity=0.6),
        ],
        description="Test doublet for validation"
    ),
}


def get_peak_set(peak_set_id: str) -> PeakSet:
    """Get a predefined peak set by ID."""
    if peak_set_id not in PEAK_SETS:
        raise ValueError(f"Unknown peak set: {peak_set_id}. Available: {list(PEAK_SETS.keys())}")
    return PEAK_SETS[peak_set_id]


def list_peak_sets() -> List[str]:
    """List available peak set IDs."""
    return list(PEAK_SETS.keys())


# =============================================================================
# Background Models
# =============================================================================

def linear_background(
    x: np.ndarray,
    level: float = 0.1,
    slope: float = 0.0
) -> np.ndarray:
    """Simple linear background.

    Args:
        x: Energy axis
        level: Base level (at x[0])
        slope: Slope (per eV)

    Returns:
        Background array
    """
    return level + slope * (x - x[0])


def shirley_like_background(
    spectrum: np.ndarray,
    x: np.ndarray,
    level_low: float = 0.05,
    level_high: float = 0.1
) -> np.ndarray:
    """
    Generate a Shirley-like background (simplified, non-iterative).

    This creates a step-like background that increases on the high-binding-energy
    side of peaks, mimicking inelastic scattering.

    Args:
        spectrum: The peak spectrum (without background)
        x: Energy axis
        level_low: Background level at low binding energy
        level_high: Background level at high binding energy

    Returns:
        Background array
    """
    # Cumulative integral from high to low energy
    cumsum = np.cumsum(spectrum[::-1])[::-1]
    if cumsum.max() > 0:
        cumsum = cumsum / cumsum.max()

    # Scale between low and high levels
    background = level_low + (level_high - level_low) * cumsum
    return background


# =============================================================================
# Noise Models
# =============================================================================

@dataclass
class NoiseConfig:
    """Configuration for noise generation."""
    noise_type: Literal['poisson', 'gaussian', 'mixed', 'none'] = 'poisson'

    # Poisson noise: level controls SNR via SNR = 10000/level
    # level=1 -> SNR~10000, level=100 -> SNR~100, level=10000 -> SNR~1
    poisson_level: float = 100.0

    # Gaussian noise: std as fraction of peak intensity
    gaussian_std: float = 0.01

    # Opt in to the Gaussian approximation to Poisson, in the bins where it is
    # admissible (see `add_poisson_noise` and `GAUSSIAN_APPROX_MIN_RATE`).
    # Off by default: Poisson is the model, this is an auxiliary. It exists so
    # that a measurement taken through the approximation can be reproduced by
    # naming it, rather than by depending on what the default happens to be.
    use_gaussian_approx: bool = False

    # Override the admissibility floor; None means GAUSSIAN_APPROX_MIN_RATE.
    gaussian_approx_min_rate: float | None = None

    def __str__(self) -> str:
        if self.noise_type == 'none':
            return 'none'
        elif self.noise_type == 'poisson':
            return f'poisson_{self.poisson_level:.0e}'
        elif self.noise_type == 'gaussian':
            return f'gaussian_{self.gaussian_std:.0e}'
        else:
            return f'mixed_p{self.poisson_level:.0e}_g{self.gaussian_std:.0e}'


#: Smallest per-bin expected count at which the Gaussian approximation to the
#: Poisson distribution is admissible here. The approximation can return
#: negative values, which are floored at zero, so a bin comes back biased
#: upward by a factor that depends only on its own expected count ``l``:
#:
#:     rel_bias(l) = phi(sqrt(l)) / sqrt(l) - Phi(-sqrt(l))
#:
#: That is a closed form, not a fit. It fixes the admissible range once a
#: tolerance is chosen: 1% relative bias gives l = 2.9725, which is rounded up
#: to 3. At l = 3 the Poisson relative standard deviation is 1/sqrt(3) = 58%,
#: so the bias is under 2% of the noise it sits in.
GAUSSIAN_APPROX_MIN_RATE = 3.0


def add_poisson_noise(
    data: np.ndarray,
    level: float,
    use_gaussian_approx: bool = False,
    rng: np.random.Generator = None,
    gaussian_approx_min_rate: float | None = None,
) -> np.ndarray:
    """
    Add Poisson (shot) noise.

    Counts are drawn from the Poisson distribution. The spectrum is normalised
    by its own maximum and scaled so that the **peak** bin has expected count
    ``(10000 / level) ** 2``; every other bin's expected count is that times its
    fraction of the peak, so a dark background sits near zero however bright the
    peak is.

    The noise level controls SNR:
        level=1     -> SNR ~ 10000 (negligible noise)
        level=10    -> SNR ~ 1000  (minimal noise)
        level=100   -> SNR ~ 100   (slight noise)
        level=1000  -> SNR ~ 10    (visible noise)
        level=1e4   -> SNR ~ 1     (very noisy)

    Args:
        data: Input intensity data
        level: Noise level (higher = more noise)
        use_gaussian_approx: Opt in to the Gaussian approximation **in the bins
            where it is admissible** -- those whose expected count is at least
            ``GAUSSIAN_APPROX_MIN_RATE``. Bins below that are drawn from the
            Poisson distribution regardless, so the approximation cannot reach
            the low-count bins where it biases upward. It is faster on bright
            spectra and it is not the default, because Poisson is the model and
            this is an auxiliary.
        rng: Random number generator (for reproducibility)
        gaussian_approx_min_rate: Override the admissibility floor. Defaults to
            ``GAUSSIAN_APPROX_MIN_RATE``. Zero approximates every bin, which is
            how this package behaved before 2026-09-11 and is kept reachable so
            that a measurement taken then can still be reproduced exactly; it
            is not a setting to choose for new work.

    Returns:
        Noisy data

    Note:
        Before 2026-09-11 the approximation was the default and was selected
        once, from the **peak** expected count, for the whole spectrum. A bright
        peak therefore licensed it for bins where it was invalid: at
        ``level=1000`` a bin at 1% of the peak has an expected count of 1 and
        came back 8% high, and one at 0.1% of the peak came back 82% high. The
        admissibility test is per bin now. This is the same defect that
        ``Noise2Noise`` fixed on its own sampler by flooring the rate at zero
        rather than at 0.01.
    """
    if level <= 0:
        return data.copy()

    if rng is None:
        rng = np.random.default_rng()

    data_pos = np.maximum(data, 0)
    data_max = data_pos.max()
    if data_max == 0:
        return data.copy()

    # Normalize and scale
    normalized = data_pos / data_max
    snr_target = 10000.0 / level
    lambda_scale = snr_target * snr_target
    scaled = lambda_scale * normalized
    scaled = np.clip(scaled, 0, 1e12)

    if use_gaussian_approx:
        # Per bin, not once for the spectrum: admissibility is a property of
        # each bin's own expected count. The floor at zero stays -- a negative
        # count is not a count -- and the threshold is what bounds the bias it
        # introduces.
        min_rate = (GAUSSIAN_APPROX_MIN_RATE if gaussian_approx_min_rate is None
                    else float(gaussian_approx_min_rate))
        approximable = scaled >= min_rate
        noisy = np.empty(scaled.shape, dtype=np.float64)
        if approximable.any():
            rates = scaled[approximable]
            draws = rates + rng.standard_normal(rates.shape) * np.sqrt(rates)
            noisy[approximable] = np.maximum(draws, 0.0)
        if not approximable.all():
            noisy[~approximable] = rng.poisson(scaled[~approximable])
    else:
        noisy = rng.poisson(scaled).astype(np.float64)

    # Scale back
    return (noisy / lambda_scale * data_max).astype(np.float32)


def add_gaussian_noise(
    data: np.ndarray,
    std_fraction: float,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Add Gaussian noise.

    Args:
        data: Input intensity data
        std_fraction: Noise std as fraction of peak intensity
        rng: Random number generator

    Returns:
        Noisy data
    """
    if std_fraction <= 0:
        return data.copy()

    if rng is None:
        rng = np.random.default_rng()

    std = std_fraction * np.max(data)
    noisy = data + rng.normal(0, std, data.shape).astype(np.float32)
    return np.maximum(noisy, 0)


def add_noise(
    data: np.ndarray,
    config: NoiseConfig,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Add noise according to config.

    Args:
        data: Input data
        config: Noise configuration
        rng: Random number generator

    Returns:
        Noisy data
    """
    if config.noise_type == 'none':
        return data.copy()
    elif config.noise_type == 'poisson':
        return add_poisson_noise(
            data, config.poisson_level, rng=rng,
            use_gaussian_approx=config.use_gaussian_approx,
            gaussian_approx_min_rate=config.gaussian_approx_min_rate)
    elif config.noise_type == 'gaussian':
        return add_gaussian_noise(data, config.gaussian_std, rng=rng)
    elif config.noise_type == 'mixed':
        result = add_poisson_noise(
            data, config.poisson_level, rng=rng,
            use_gaussian_approx=config.use_gaussian_approx,
            gaussian_approx_min_rate=config.gaussian_approx_min_rate)
        result = add_gaussian_noise(result, config.gaussian_std, rng=rng)
        return result
    else:
        raise ValueError(f"Unknown noise type: {config.noise_type}")


# =============================================================================
# Spectrum Generation
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for synthetic spectrum generation."""
    # Peak shape
    eta: float = 0.3  # Lorentzian fraction (K parameter), fixed for all peaks
    use_pseudo_voigt: bool = True  # Use pseudo-Voigt (faster) vs true Voigt

    # Energy axis
    n_energy_points: int = 256
    energy_range: Optional[Tuple[float, float]] = None  # Override from peak set

    # Background
    background_type: Literal['none', 'linear', 'shirley'] = 'linear'
    background_level: float = 0.05  # Base background level (fraction of peak)
    background_slope: float = 0.001  # For linear background

    # Intensity variation
    intensity_variation: float = 0.2  # Random variation in peak intensities
    position_jitter: float = 0.0  # Random jitter in peak positions (eV)
    width_variation: float = 0.1  # Random variation in peak widths

    # Normalization
    normalize: bool = True  # Normalize clean spectra to [0, 1]

    # === Angle-resolved (ARPES-like) settings ===
    # When n_angles > 1, generates 3D data: (N_samples, N_angles, N_energy)
    n_angles: int = 1  # Number of angle points (1 = standard 2D mode)
    angle_range: Tuple[float, float] = (0.0, 60.0)  # Angle range in degrees
    angle_intensity_model: Literal['cosine', 'exponential', 'linear', 'none'] = 'cosine'
    # Controls how peak intensity varies with angle:
    #   'cosine': I(θ) = I₀ * cos(θ)^n  (typical ARPES behavior)
    #   'exponential': I(θ) = I₀ * exp(-θ/θ₀)  (surface sensitivity)
    #   'linear': I(θ) = I₀ * (1 - θ/θ_max)  (simple attenuation)
    #   'none': No angle dependence (same spectrum at all angles)
    angle_cosine_power: float = 1.0  # Exponent n for cosine model
    angle_exp_decay: float = 30.0  # θ₀ decay constant for exponential model (degrees)
    angle_shift_rate: float = 0.0  # Peak position shift per degree (eV/deg)

    # === Time-resolved settings ===
    # When n_times > 1, generates time-series data
    # Combined with angles: 4D data (N_samples, N_times, N_angles, N_energy)
    # Without angles: 3D data (N_samples, N_times, N_energy)
    n_times: int = 1  # Number of time points (1 = no time dimension)
    time_range: Tuple[float, float] = (0.0, 100.0)  # Time range (arbitrary units)
    time_intensity_model: Literal['exponential_decay', 'linear_decay', 'oscillation', 'none'] = 'exponential_decay'
    # Controls how peak intensity varies with time:
    #   'exponential_decay': I(t) = I₀ * exp(-t/τ) (chemical reaction, degradation)
    #   'linear_decay': I(t) = I₀ * (1 - t/t_max) (simple linear decay)
    #   'oscillation': I(t) = I₀ * (1 + A*sin(2π*f*t)) (periodic variation)
    #   'none': No time dependence
    time_decay_constant: float = 50.0  # τ for exponential decay
    time_oscillation_amplitude: float = 0.2  # A for oscillation model
    time_oscillation_frequency: float = 0.05  # f for oscillation model (cycles per unit time)
    time_shift_rate: float = 0.0  # Peak position shift per time unit (eV/time)


def compute_angle_intensity_factor(
    angle: float,
    model: str,
    cosine_power: float = 1.0,
    exp_decay: float = 30.0,
    angle_max: float = 60.0,
) -> float:
    """
    Compute intensity modulation factor based on emission angle.

    In ARPES and angle-resolved XPS, peak intensity typically decreases
    at higher emission angles due to:
    - Reduced photoelectron escape probability
    - Surface sensitivity effects
    - Matrix element variations

    Args:
        angle: Emission angle in degrees
        model: Intensity model ('cosine', 'exponential', 'linear', 'none')
        cosine_power: Exponent for cosine model
        exp_decay: Decay constant for exponential model (degrees)
        angle_max: Maximum angle for linear model normalization

    Returns:
        Intensity factor in [0, 1]
    """
    if model == 'none':
        return 1.0

    # Convert to radians for cosine
    theta_rad = np.radians(angle)

    if model == 'cosine':
        # I(θ) = cos(θ)^n - typical angular distribution
        factor = np.cos(theta_rad) ** cosine_power
    elif model == 'exponential':
        # I(θ) = exp(-θ/θ₀) - exponential decay with angle
        factor = np.exp(-angle / exp_decay)
    elif model == 'linear':
        # I(θ) = 1 - θ/θ_max - simple linear decrease
        factor = max(0.0, 1.0 - angle / angle_max)
    else:
        factor = 1.0

    return float(np.clip(factor, 0.0, 1.0))


def compute_time_intensity_factor(
    time: float,
    model: str,
    decay_constant: float = 50.0,
    time_max: float = 100.0,
    oscillation_amplitude: float = 0.2,
    oscillation_frequency: float = 0.05,
) -> float:
    """
    Compute intensity modulation factor based on time.

    Simulates time-dependent phenomena in XPS:
    - Chemical reactions (exponential decay)
    - Surface degradation (linear decay)
    - Periodic processes (oscillation)

    Args:
        time: Time value (arbitrary units)
        model: Intensity model ('exponential_decay', 'linear_decay', 'oscillation', 'none')
        decay_constant: τ for exponential decay
        time_max: Maximum time for linear decay normalization
        oscillation_amplitude: A for oscillation model (0 to 1)
        oscillation_frequency: f for oscillation model (cycles per unit time)

    Returns:
        Intensity factor (typically in [0, 1], but oscillation can exceed 1)
    """
    if model == 'none':
        return 1.0

    if model == 'exponential_decay':
        # I(t) = exp(-t/τ) - exponential decay
        factor = np.exp(-time / decay_constant) if decay_constant > 0 else 1.0
    elif model == 'linear_decay':
        # I(t) = 1 - t/t_max - linear decrease
        factor = max(0.0, 1.0 - time / time_max) if time_max > 0 else 1.0
    elif model == 'oscillation':
        # I(t) = 1 + A*sin(2π*f*t) - periodic variation
        factor = 1.0 + oscillation_amplitude * np.sin(2 * np.pi * oscillation_frequency * time)
    else:
        factor = 1.0

    return float(np.clip(factor, 0.0, 2.0))  # Allow up to 2x for oscillation peaks


def compute_difficulty_score(
    noise_type: str,
    poisson_level: float,
    gaussian_std: float,
    n_peaks: int = 1,
) -> float:
    """
    Compute a difficulty score for a sample based on noise and complexity.

    Score is in range [0, 1] where:
    - 0.0 = easiest (no noise, single peak)
    - 1.0 = hardest (extreme noise, many overlapping peaks)

    Components:
    - Poisson noise: log10(level) mapped from [-2, 6] to [0, 0.5]
    - Gaussian noise: std mapped from [0, 0.1] to [0, 0.3]
    - Peak complexity: n_peaks mapped from [1, 5] to [0, 0.2]

    Args:
        noise_type: 'none', 'poisson', 'gaussian', 'mixed'
        poisson_level: Poisson noise level (SNR = 10000/level)
        gaussian_std: Gaussian noise std (fraction of signal)
        n_peaks: Number of peaks

    Returns:
        Difficulty score in [0, 1]
    """
    score = 0.0

    # Poisson component (0 to 0.5)
    # level=1 (SNR=10000) -> 0.0, level=1e6 (SNR=0.01) -> 0.5
    if noise_type in ('poisson', 'mixed') and poisson_level > 0:
        log_level = np.log10(max(poisson_level, 1))
        # Map log10(level) from [0, 6] to [0, 0.5]
        poisson_score = np.clip(log_level / 12.0, 0, 0.5)
        score += poisson_score

    # Gaussian component (0 to 0.3)
    # std=0 -> 0.0, std=0.1 -> 0.3
    if noise_type in ('gaussian', 'mixed') and gaussian_std > 0:
        gaussian_score = np.clip(gaussian_std * 3.0, 0, 0.3)
        score += gaussian_score

    # Peak complexity component (0 to 0.2)
    # 1 peak -> 0.0, 5+ peaks -> 0.2
    if n_peaks > 1:
        peak_score = np.clip((n_peaks - 1) / 20.0, 0, 0.2)
        score += peak_score

    return float(np.clip(score, 0, 1))


@dataclass
class SampleMetadata:
    """Metadata for a generated sample."""
    sample_idx: int
    peak_set_id: str
    noise_type: str
    poisson_level: float
    gaussian_std: float
    seed: int
    energy_range: Tuple[float, float]
    n_peaks: int
    eta: float  # Lorentzian fraction
    difficulty: float = 0.0  # Difficulty score [0, 1]

    def __post_init__(self):
        """Compute difficulty score if not provided."""
        if self.difficulty == 0.0 and self.noise_type != 'none':
            self.difficulty = compute_difficulty_score(
                self.noise_type,
                self.poisson_level,
                self.gaussian_std,
                self.n_peaks,
            )

    def to_dict(self) -> Dict:
        return {
            'sample_idx': int(self.sample_idx),
            'peak_set_id': self.peak_set_id,
            'noise_type': self.noise_type,
            'poisson_level': float(self.poisson_level),
            'gaussian_std': float(self.gaussian_std),
            'seed': int(self.seed),
            'energy_min': float(self.energy_range[0]),
            'energy_max': float(self.energy_range[1]),
            'n_peaks': int(self.n_peaks),
            'eta': float(self.eta),
            'difficulty': float(self.difficulty),
        }


class SyntheticGenerator:
    """
    Generate synthetic XPS spectra for denoising training.

    Example usage:
        gen = SyntheticGenerator(
            peak_set="C1s_adventitious",
            noise_config=NoiseConfig(noise_type='poisson', poisson_level=100),
            config=GeneratorConfig(eta=0.3)
        )
        clean, noisy, energy, metadata = gen.generate_batch(100, seed=42)
        gen.save_hdf5("output.h5", clean, noisy, energy, metadata)
    """

    def __init__(
        self,
        peak_set: Union[str, PeakSet],
        noise_config: NoiseConfig = None,
        config: GeneratorConfig = None,
    ):
        """
        Initialize generator.

        Args:
            peak_set: Peak set ID string or PeakSet object
            noise_config: Noise configuration
            config: Generator configuration
        """
        if isinstance(peak_set, str):
            self.peak_set = get_peak_set(peak_set)
        else:
            self.peak_set = peak_set

        self.noise_config = noise_config or NoiseConfig()
        self.config = config or GeneratorConfig()

        # Setup energy axis
        if self.config.energy_range is not None:
            self.energy_range = self.config.energy_range
        else:
            self.energy_range = self.peak_set.energy_range

        self.energy = np.linspace(
            self.energy_range[0],
            self.energy_range[1],
            self.config.n_energy_points,
            dtype=np.float32
        )

        # Setup angle axis for 3D/4D mode
        if self.config.n_angles > 1:
            self.angles = np.linspace(
                self.config.angle_range[0],
                self.config.angle_range[1],
                self.config.n_angles,
                dtype=np.float32
            )
        else:
            self.angles = None

        # Setup time axis for 3D/4D mode
        if self.config.n_times > 1:
            self.times = np.linspace(
                self.config.time_range[0],
                self.config.time_range[1],
                self.config.n_times,
                dtype=np.float32
            )
        else:
            self.times = None

    def generate_single(
        self,
        rng: np.random.Generator = None,
        angle: float = 0.0,
        time: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single clean spectrum with optional variations.

        Args:
            rng: Random number generator for variations
            angle: Emission angle in degrees (for angle-resolved mode)
            time: Time value (for time-resolved mode)

        Returns:
            Tuple of (spectrum, background)
        """
        if rng is None:
            rng = np.random.default_rng()

        spectrum = np.zeros_like(self.energy)

        # Compute angle-dependent intensity factor
        angle_factor = compute_angle_intensity_factor(
            angle,
            self.config.angle_intensity_model,
            cosine_power=self.config.angle_cosine_power,
            exp_decay=self.config.angle_exp_decay,
            angle_max=self.config.angle_range[1],
        )

        # Compute time-dependent intensity factor
        time_factor = compute_time_intensity_factor(
            time,
            self.config.time_intensity_model,
            decay_constant=self.config.time_decay_constant,
            time_max=self.config.time_range[1],
            oscillation_amplitude=self.config.time_oscillation_amplitude,
            oscillation_frequency=self.config.time_oscillation_frequency,
        )

        # Combined modulation factor
        modulation_factor = angle_factor * time_factor

        for peak in self.peak_set.peaks:
            # Apply random variations
            intensity = peak.intensity
            mu = peak.mu
            fwhm = peak.fwhm

            if self.config.intensity_variation > 0:
                intensity *= (1 + rng.uniform(
                    -self.config.intensity_variation,
                    self.config.intensity_variation
                ))

            if self.config.position_jitter > 0:
                mu += rng.uniform(
                    -self.config.position_jitter,
                    self.config.position_jitter
                )

            if self.config.width_variation > 0:
                fwhm *= (1 + rng.uniform(
                    -self.config.width_variation,
                    self.config.width_variation
                ))

            # Apply angle-dependent position shift
            if self.config.angle_shift_rate != 0.0:
                mu += angle * self.config.angle_shift_rate

            # Apply time-dependent position shift
            if self.config.time_shift_rate != 0.0:
                mu += time * self.config.time_shift_rate

            # Apply combined (angle + time) intensity modulation
            intensity *= modulation_factor

            # Generate peak
            peak_profile = voigt_from_fwhm_and_eta(
                self.energy, mu, fwhm, self.config.eta,
                use_pseudo=self.config.use_pseudo_voigt
            )
            spectrum += intensity * peak_profile

        # Generate background (also scaled by modulation factor for realism)
        if self.config.background_type == 'none':
            background = np.zeros_like(self.energy)
        elif self.config.background_type == 'linear':
            background = linear_background(
                self.energy,
                level=self.config.background_level * spectrum.max() * angle_factor,
                slope=self.config.background_slope
            )
        elif self.config.background_type == 'shirley':
            background = shirley_like_background(
                spectrum, self.energy,
                level_low=self.config.background_level * 0.5 * spectrum.max(),
                level_high=self.config.background_level * spectrum.max()
            )
        else:
            raise ValueError(f"Unknown background type: {self.config.background_type}")

        return spectrum + background, background

    def generate_batch(
        self,
        n_samples: int,
        seed: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[SampleMetadata]]:
        """
        Generate a batch of spectra with noise.

        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            Tuple of (clean, noisy, energy, metadata_list)
            For 2D mode (n_angles=1, n_times=1):
                - clean: (n_samples, n_energy) clean spectra
                - noisy: (n_samples, n_energy) noisy spectra
            For 3D angle mode (n_angles>1, n_times=1):
                - clean: (n_samples, n_angles, n_energy) clean spectra
                - noisy: (n_samples, n_angles, n_energy) noisy spectra
            For 3D time mode (n_angles=1, n_times>1):
                - clean: (n_samples, n_times, n_energy) clean spectra
                - noisy: (n_samples, n_times, n_energy) noisy spectra
            For 4D mode (n_angles>1, n_times>1):
                - clean: (n_samples, n_times, n_angles, n_energy) clean spectra
                - noisy: (n_samples, n_times, n_angles, n_energy) noisy spectra
            - energy: (n_energy,) energy axis
            - metadata_list: List of SampleMetadata for each sample
        """
        base_rng = np.random.default_rng(seed)

        # Determine dimensionality
        has_angles = self.config.n_angles > 1 and self.angles is not None
        has_times = self.config.n_times > 1 and self.times is not None

        # Allocate arrays based on dimensions
        if has_times and has_angles:
            # 4D: (N_samples, N_times, N_angles, N_energy)
            clean = np.zeros(
                (n_samples, len(self.times), len(self.angles), len(self.energy)),
                dtype=np.float32
            )
        elif has_times:
            # 3D time-only: (N_samples, N_times, N_energy)
            clean = np.zeros(
                (n_samples, len(self.times), len(self.energy)),
                dtype=np.float32
            )
        elif has_angles:
            # 3D angle-only: (N_samples, N_angles, N_energy)
            clean = np.zeros(
                (n_samples, len(self.angles), len(self.energy)),
                dtype=np.float32
            )
        else:
            # 2D: (N_samples, N_energy)
            clean = np.zeros((n_samples, len(self.energy)), dtype=np.float32)

        noisy = np.zeros_like(clean)
        metadata_list = []

        for i in range(n_samples):
            # Per-sample seed for reproducibility
            sample_seed = base_rng.integers(0, 2**31)
            sample_rng = np.random.default_rng(sample_seed)

            if has_times and has_angles:
                # 4D generation: Time × Angle
                sample_clean = np.zeros(
                    (len(self.times), len(self.angles), len(self.energy)),
                    dtype=np.float32
                )
                global_max = 0.0

                for t_idx, time_val in enumerate(self.times):
                    for a_idx, angle_val in enumerate(self.angles):
                        # Unique RNG for each (time, angle) combination
                        slice_rng = np.random.default_rng(
                            sample_seed + t_idx * 1000 + a_idx
                        )
                        spectrum, _ = self.generate_single(
                            slice_rng, angle=angle_val, time=time_val
                        )
                        sample_clean[t_idx, a_idx] = spectrum
                        global_max = max(global_max, spectrum.max())

                # Normalize across all time×angle slices
                if self.config.normalize and global_max > 0:
                    sample_clean = sample_clean / global_max

                # Add noise to each slice
                for t_idx in range(len(self.times)):
                    for a_idx in range(len(self.angles)):
                        noise_rng = np.random.default_rng(
                            sample_seed + 100000 + t_idx * 1000 + a_idx
                        )
                        noisy[i, t_idx, a_idx] = add_noise(
                            sample_clean[t_idx, a_idx], self.noise_config, noise_rng
                        )

                clean[i] = sample_clean

            elif has_times:
                # 3D time-only generation
                sample_clean = np.zeros(
                    (len(self.times), len(self.energy)),
                    dtype=np.float32
                )
                global_max = 0.0

                for t_idx, time_val in enumerate(self.times):
                    time_rng = np.random.default_rng(sample_seed + t_idx)
                    spectrum, _ = self.generate_single(
                        time_rng, angle=0.0, time=time_val
                    )
                    sample_clean[t_idx] = spectrum
                    global_max = max(global_max, spectrum.max())

                # Normalize across all time slices
                if self.config.normalize and global_max > 0:
                    sample_clean = sample_clean / global_max

                # Add noise to each time slice
                for t_idx in range(len(self.times)):
                    noise_rng = np.random.default_rng(sample_seed + 1000 + t_idx)
                    noisy[i, t_idx] = add_noise(
                        sample_clean[t_idx], self.noise_config, noise_rng
                    )

                clean[i] = sample_clean

            elif has_angles:
                # 3D angle-only generation (existing behavior)
                sample_clean = np.zeros(
                    (len(self.angles), len(self.energy)),
                    dtype=np.float32
                )
                global_max = 0.0

                for j, angle in enumerate(self.angles):
                    angle_rng = np.random.default_rng(sample_seed + j)
                    spectrum, _ = self.generate_single(angle_rng, angle=angle, time=0.0)
                    sample_clean[j] = spectrum
                    global_max = max(global_max, spectrum.max())

                if self.config.normalize and global_max > 0:
                    sample_clean = sample_clean / global_max

                for j in range(len(self.angles)):
                    noise_rng = np.random.default_rng(sample_seed + 1000 + j)
                    noisy[i, j] = add_noise(
                        sample_clean[j], self.noise_config, noise_rng
                    )

                clean[i] = sample_clean

            else:
                # Standard 2D generation
                spectrum, _ = self.generate_single(sample_rng, angle=0.0, time=0.0)

                if self.config.normalize and spectrum.max() > 0:
                    spectrum = spectrum / spectrum.max()

                clean[i] = spectrum
                noisy[i] = add_noise(spectrum, self.noise_config, sample_rng)

            # Record metadata
            metadata = SampleMetadata(
                sample_idx=i,
                peak_set_id=self.peak_set.id,
                noise_type=self.noise_config.noise_type,
                poisson_level=self.noise_config.poisson_level,
                gaussian_std=self.noise_config.gaussian_std,
                seed=sample_seed,
                energy_range=self.energy_range,
                n_peaks=len(self.peak_set.peaks),
                eta=self.config.eta,
            )
            metadata_list.append(metadata)

        return clean, noisy, self.energy.copy(), metadata_list

    def generate_batch_3d(
        self,
        n_samples: int,
        seed: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[SampleMetadata]]:
        """
        Generate a batch of 3D angle-resolved spectra.

        Convenience method that also returns the angle axis.

        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            Tuple of (clean, noisy, energy, angles, metadata_list)
            - clean: (n_samples, n_angles, n_energy) clean spectra
            - noisy: (n_samples, n_angles, n_energy) noisy spectra
            - energy: (n_energy,) energy axis
            - angles: (n_angles,) angle axis in degrees
            - metadata_list: List of SampleMetadata for each sample
        """
        if self.angles is None:
            raise ValueError(
                "3D angle mode requires n_angles > 1 in GeneratorConfig"
            )

        clean, noisy, energy, metadata = self.generate_batch(n_samples, seed)

        return clean, noisy, energy, self.angles.copy(), metadata

    def generate_batch_4d(
        self,
        n_samples: int,
        seed: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[SampleMetadata]]:
        """
        Generate a batch of 4D time×angle-resolved spectra.

        Convenience method that returns all axes.

        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            Tuple of (clean, noisy, energy, times, angles, metadata_list)
            - clean: (n_samples, n_times, n_angles, n_energy) clean spectra
            - noisy: (n_samples, n_times, n_angles, n_energy) noisy spectra
            - energy: (n_energy,) energy axis
            - times: (n_times,) time axis
            - angles: (n_angles,) angle axis in degrees
            - metadata_list: List of SampleMetadata for each sample
        """
        if self.times is None or self.angles is None:
            raise ValueError(
                "4D mode requires n_times > 1 AND n_angles > 1 in GeneratorConfig"
            )

        clean, noisy, energy, metadata = self.generate_batch(n_samples, seed)

        return clean, noisy, energy, self.times.copy(), self.angles.copy(), metadata

    @staticmethod
    def save_hdf5(
        filepath: Union[str, Path],
        clean: np.ndarray,
        noisy: np.ndarray,
        energy: np.ndarray,
        metadata_list: List[SampleMetadata],
        angles: np.ndarray = None,
        times: np.ndarray = None,
        chunk_size: int = 128,
        compression: str = 'gzip',
    ) -> Path:
        """
        Save generated data to HDF5 file.

        Structure for 2D data:
            /clean      - (N, E) clean spectra
            /noisy      - (N, E) noisy spectra
            /energy     - (E,) energy axis
            /metadata   - JSON string of generation parameters

        Structure for 3D angle data:
            /clean      - (N, A, E) clean spectra
            /noisy      - (N, A, E) noisy spectra
            /energy     - (E,) energy axis
            /angles     - (A,) angle axis in degrees
            /metadata   - JSON string of generation parameters

        Structure for 3D time data:
            /clean      - (N, T, E) clean spectra
            /noisy      - (N, T, E) noisy spectra
            /energy     - (E,) energy axis
            /times      - (T,) time axis
            /metadata   - JSON string of generation parameters

        Structure for 4D data:
            /clean      - (N, T, A, E) clean spectra
            /noisy      - (N, T, A, E) noisy spectra
            /energy     - (E,) energy axis
            /times      - (T,) time axis
            /angles     - (A,) angle axis in degrees
            /metadata   - JSON string of generation parameters

        Args:
            filepath: Output file path
            clean: Clean spectra array (2D, 3D, or 4D)
            noisy: Noisy spectra array (2D, 3D, or 4D)
            energy: Energy axis
            metadata_list: List of sample metadata
            angles: Angle axis (optional, for 3D/4D data)
            times: Time axis (optional, for 3D/4D data)
            chunk_size: Chunk size for row-based access
            compression: Compression algorithm ('gzip', 'lzf', or None)

        Returns:
            Path to saved file
        """
        filepath = Path(filepath)

        # Determine dimensionality and extract shape
        ndim = clean.ndim
        n_samples = clean.shape[0]
        n_energy = clean.shape[-1]
        n_times = 1
        n_angles = 1

        if ndim == 4:
            # 4D: (N, T, A, E)
            n_times = clean.shape[1]
            n_angles = clean.shape[2]
            chunks = (min(chunk_size, n_samples), n_times, n_angles, n_energy)
        elif ndim == 3:
            # 3D: could be (N, T, E) or (N, A, E)
            if times is not None and angles is None:
                # Time-only 3D
                n_times = clean.shape[1]
            elif angles is not None and times is None:
                # Angle-only 3D
                n_angles = clean.shape[1]
            else:
                # Ambiguous - assume angles for backward compatibility
                n_angles = clean.shape[1]
            chunks = (min(chunk_size, n_samples), clean.shape[1], n_energy)
        else:
            # 2D: (N, E)
            chunks = (min(chunk_size, n_samples), n_energy)

        with h5py.File(filepath, 'w') as f:
            # Create datasets with row-chunked storage for efficient access
            f.create_dataset(
                'clean', data=clean, dtype='float32',
                chunks=chunks, compression=compression
            )
            f.create_dataset(
                'noisy', data=noisy, dtype='float32',
                chunks=chunks, compression=compression
            )
            f.create_dataset('energy', data=energy, dtype='float32')

            # Store time axis for 3D/4D data
            if times is not None:
                f.create_dataset('times', data=times, dtype='float32')

            # Store angles axis for 3D/4D data
            if angles is not None:
                f.create_dataset('angles', data=angles, dtype='float32')

            # Store metadata as JSON
            meta_dict = {
                'n_samples': n_samples,
                'n_energy': n_energy,
                'n_times': n_times,
                'n_angles': n_angles,
                'ndim': ndim,
                'energy_min': float(energy.min()),
                'energy_max': float(energy.max()),
                'peak_set_id': metadata_list[0].peak_set_id if metadata_list else '',
                'noise_type': metadata_list[0].noise_type if metadata_list else '',
                'eta': metadata_list[0].eta if metadata_list else 0.3,
            }

            if times is not None:
                meta_dict['time_min'] = float(times.min())
                meta_dict['time_max'] = float(times.max())

            if angles is not None:
                meta_dict['angle_min'] = float(angles.min())
                meta_dict['angle_max'] = float(angles.max())

            f.attrs['metadata'] = json.dumps(meta_dict)

            # Store per-sample seeds for reproducibility
            seeds = np.array([m.seed for m in metadata_list], dtype=np.int64)
            f.create_dataset('seeds', data=seeds)

        return filepath

    @staticmethod
    def save_manifest(
        filepath: Union[str, Path],
        metadata_list: List[SampleMetadata],
        format: Literal['jsonl', 'csv'] = 'jsonl'
    ) -> Path:
        """
        Save manifest file recording sample metadata.

        Args:
            filepath: Output file path (extension will be adjusted)
            metadata_list: List of sample metadata
            format: Output format ('jsonl' or 'csv')

        Returns:
            Path to saved manifest
        """
        filepath = Path(filepath)

        if format == 'jsonl':
            filepath = filepath.with_suffix('.jsonl')
            with open(filepath, 'w', encoding='utf-8') as f:
                for meta in metadata_list:
                    f.write(json.dumps(meta.to_dict()) + '\n')

        elif format == 'csv':
            filepath = filepath.with_suffix('.csv')
            if metadata_list:
                fieldnames = list(metadata_list[0].to_dict().keys())
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for meta in metadata_list:
                        writer.writerow(meta.to_dict())

        return filepath


def generate_dataset(
    output_dir: Union[str, Path],
    peak_set_id: str = "C1s_adventitious",
    n_samples: int = 1000,
    noise_configs: List[NoiseConfig] = None,
    eta: float = 0.3,
    seed: int = 42,
    manifest_format: Literal['jsonl', 'csv'] = 'jsonl',
) -> Dict[str, Path]:
    """
    Generate a complete dataset with multiple noise levels.

    Args:
        output_dir: Output directory
        peak_set_id: Peak set to use
        n_samples: Number of samples per noise level
        noise_configs: List of noise configurations (default: several Poisson levels)
        eta: Lorentzian fraction (K parameter)
        seed: Base random seed
        manifest_format: Manifest file format

    Returns:
        Dict mapping noise description to output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if noise_configs is None:
        noise_configs = [
            NoiseConfig(noise_type='none'),
            NoiseConfig(noise_type='poisson', poisson_level=10),
            NoiseConfig(noise_type='poisson', poisson_level=100),
            NoiseConfig(noise_type='poisson', poisson_level=1000),
            NoiseConfig(noise_type='mixed', poisson_level=100, gaussian_std=0.02),
        ]

    results = {}
    config = GeneratorConfig(eta=eta)

    for i, noise_config in enumerate(noise_configs):
        noise_str = str(noise_config)
        h5_path = output_dir / f"{peak_set_id}_{noise_str}.h5"
        manifest_path = output_dir / f"{peak_set_id}_{noise_str}_manifest"

        gen = SyntheticGenerator(peak_set_id, noise_config, config)
        clean, noisy, energy, metadata = gen.generate_batch(n_samples, seed=seed + i)

        SyntheticGenerator.save_hdf5(h5_path, clean, noisy, energy, metadata)
        SyntheticGenerator.save_manifest(manifest_path, metadata, manifest_format)

        results[noise_str] = {
            'h5': h5_path,
            'manifest': manifest_path.with_suffix(f'.{manifest_format}')
        }

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic XPS spectra')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--peak-set', '-p', type=str, default='C1s_adventitious',
                        choices=list_peak_sets(), help='Peak set to use')
    parser.add_argument('--n-samples', '-n', type=int, default=1000, help='Number of samples')
    parser.add_argument('--eta', type=float, default=0.3, help='Lorentzian fraction (0-1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--noise-type', type=str, default='poisson',
                        choices=['none', 'poisson', 'gaussian', 'mixed'])
    parser.add_argument('--poisson-level', type=float, default=100.0)
    parser.add_argument('--gaussian-std', type=float, default=0.01)

    args = parser.parse_args()

    noise_config = NoiseConfig(
        noise_type=args.noise_type,
        poisson_level=args.poisson_level,
        gaussian_std=args.gaussian_std,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GeneratorConfig(eta=args.eta)
    gen = SyntheticGenerator(args.peak_set, noise_config, config)
    clean, noisy, energy, metadata = gen.generate_batch(args.n_samples, seed=args.seed)

    h5_path = output_dir / f"{args.peak_set}_{noise_config}.h5"
    manifest_path = output_dir / f"{args.peak_set}_{noise_config}_manifest"

    SyntheticGenerator.save_hdf5(h5_path, clean, noisy, energy, metadata)
    SyntheticGenerator.save_manifest(manifest_path, metadata)

    print(f"Generated {args.n_samples} samples")
    print(f"  HDF5: {h5_path}")
    print(f"  Manifest: {manifest_path.with_suffix('.jsonl')}")
