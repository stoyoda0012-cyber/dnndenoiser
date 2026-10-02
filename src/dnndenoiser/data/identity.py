"""Identities of synthetic data: which draws a ``generate`` file holds.

Implements §5.4 of ``docs/design/PROVENANCE_MANIFEST.md`` (adopted 2026-10-02, phase B).
``generate`` writes two identities, each a prefix plus the first 16 hex digits of the
SHA-256 of the canonical JSON of a configuration object:

- the **signal identity** (``"generate-signal:"``, the attribute ``signal_identity`` on
  ``clean``): the peak set's parameters, the seed, and every field that changes the stored
  ``clean`` in the chosen mode;
- the **noise identity** (``"generate:"``, the attribute ``acquisition_id`` on ``noisy``):
  the signal identity's fields plus every noise field that changes the stored ``noisy``.

Sample *i* depends only on the seed, the configuration and *i*, so a smaller ``-n`` with
the same identity is a prefix of a larger one; ``n_samples`` is never part of it.

**Which fields.** A field enters an identity in a mode exactly when changing it changes the
corresponding array in that mode, and its value is not the field's no-effect value. Values
are hashed as resolved, and configurations that give identical arrays are canonicalised to
one form (no noise, no background). The classification is :data:`CLASSIFICATION`; tests
check it against the generator, field by field and mode by mode, and freeze one identifier.
A field added later must reproduce the earlier arrays at its no-effect value, or the
prefixes change, so that old and new identities are never compared.
"""
from __future__ import annotations

import hashlib

from dnndenoiser.provenance import canonical

SIGNAL_PREFIX = "generate-signal:"
NOISE_PREFIX = "generate:"

# Every field of GeneratorConfig and NoiseConfig, with the rule that puts it into an
# identity and its no-effect value ("none" when it has none). Tests require this to
# cover every field and check each rule against the generator.
CLASSIFICATION = {
    # GeneratorConfig
    "eta": ("always", "none"),
    "use_pseudo_voigt": ("always", "none"),
    "n_energy_points": ("always", "none"),
    "energy_range": ("always, as resolved from the peak set when None", "none"),
    "background_type": ("always, canonicalised with its level and slope", "none"),
    "background_level": ("background type is not none", "0 (with zero slope: no background)"),
    "background_slope": ("background type is linear", "0"),
    "intensity_variation": ("always", 0.0),
    "position_jitter": ("always", 0.0),
    "width_variation": ("always", 0.0),
    "normalize": ("always", "none"),
    "n_angles": ("n_angles > 1", "1"),
    "angle_range": ("n_angles > 1 and (model is not none or a non-zero shift)", "none"),
    "angle_intensity_model": ("n_angles > 1", "none"),
    "angle_cosine_power": ("n_angles > 1 and model cosine", "none"),
    "angle_exp_decay": ("n_angles > 1 and model exponential", "none"),
    "angle_shift_rate": ("n_angles > 1", 0.0),
    "n_times": ("n_times > 1", "1"),
    "time_range": ("n_times > 1 and (model is not none or a non-zero shift)", "none"),
    "time_intensity_model": ("n_times > 1", "none"),
    "time_decay_constant": ("n_times > 1 and model exponential_decay", "none"),
    "time_oscillation_amplitude": ("n_times > 1 and model oscillation", "none"),
    "time_oscillation_frequency": ("n_times > 1 and model oscillation", "none"),
    "time_shift_rate": ("n_times > 1", 0.0),
    # NoiseConfig (noise identity only)
    "noise_type": ("noise identity, canonicalised to the active components", "none"),
    "poisson_level": ("noise identity, type poisson or mixed", "0 (no Poisson noise)"),
    "gaussian_std": ("noise identity, type gaussian or mixed", "0 (no Gaussian noise)"),
    "use_gaussian_approx": ("noise identity, Poisson active", False),
    "gaussian_approx_min_rate": ("noise identity, Poisson active and approximation on; "
                                 "None resolved", "none"),
}


def _f(v) -> float:
    return float(v)


def _background(config) -> dict | None:
    kind = config.background_type
    level, slope = _f(config.background_level), _f(config.background_slope)
    if kind == "none" or (kind == "shirley" and level == 0) or (
            kind == "linear" and level == 0 and slope == 0):
        return None
    if kind == "shirley":
        return {"type": "shirley", "level": level}
    return {"type": "linear", "level": level, "slope": slope}


def _angles(config) -> dict | None:
    if config.n_angles <= 1:
        return None
    out = {"n": int(config.n_angles), "model": config.angle_intensity_model}
    shift = _f(config.angle_shift_rate)
    if config.angle_intensity_model != "none" or shift != 0:
        out["range"] = [_f(config.angle_range[0]), _f(config.angle_range[1])]
    if config.angle_intensity_model == "cosine":
        out["cosine_power"] = _f(config.angle_cosine_power)
    elif config.angle_intensity_model == "exponential":
        out["exp_decay"] = _f(config.angle_exp_decay)
    if shift != 0:
        out["shift_rate"] = shift
    return out


def _times(config) -> dict | None:
    if config.n_times <= 1:
        return None
    model = config.time_intensity_model
    out = {"n": int(config.n_times), "model": model}
    shift = _f(config.time_shift_rate)
    if model != "none" or shift != 0:
        out["range"] = [_f(config.time_range[0]), _f(config.time_range[1])]
    if model == "exponential_decay":
        out["decay_constant"] = _f(config.time_decay_constant)
    elif model == "oscillation":
        out["oscillation_amplitude"] = _f(config.time_oscillation_amplitude)
        out["oscillation_frequency"] = _f(config.time_oscillation_frequency)
    if shift != 0:
        out["shift_rate"] = shift
    return out


def _noise(noise) -> dict:
    from dnndenoiser.data.synthetic_generator import GAUSSIAN_APPROX_MIN_RATE
    out = {}
    kind = noise.noise_type
    if kind in ("poisson", "mixed") and _f(noise.poisson_level) > 0:
        poisson = {"level": _f(noise.poisson_level)}
        if noise.use_gaussian_approx:
            rate = (GAUSSIAN_APPROX_MIN_RATE if noise.gaussian_approx_min_rate is None
                    else noise.gaussian_approx_min_rate)
            poisson["gaussian_approx_min_rate"] = _f(rate)
        out["poisson"] = poisson
    if kind in ("gaussian", "mixed") and _f(noise.gaussian_std) > 0:
        out["gaussian"] = {"std": _f(noise.gaussian_std)}
    return out


def signal_configuration(generator, seed: int) -> dict:
    """The configuration object the signal identity hashes."""
    config = generator.config
    out = {
        "seed": int(seed),
        "peaks": [[_f(p.mu), _f(p.fwhm), _f(p.intensity)] for p in generator.peak_set.peaks],
        "energy_range": [_f(generator.energy_range[0]), _f(generator.energy_range[1])],
        "n_energy_points": int(config.n_energy_points),
        "pseudo_voigt": bool(config.use_pseudo_voigt),
        "eta": _f(config.eta),
        "background": _background(config),
        "normalize": bool(config.normalize),
    }
    for name in ("intensity_variation", "position_jitter", "width_variation"):
        if _f(getattr(config, name)) != 0:
            out[name] = _f(getattr(config, name))
    for name, value in (("angles", _angles(config)), ("times", _times(config))):
        if value is not None:
            out[name] = value
    return out


def noise_configuration(generator, seed: int) -> dict:
    """The configuration object the noise identity hashes."""
    return {**signal_configuration(generator, seed), "noise": _noise(generator.noise_config)}


def _identity(prefix: str, obj: dict) -> str:
    return prefix + hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:16]


def identities(generator, seed: int) -> tuple[str, str]:
    """``(acquisition_id, signal_identity)`` for the arrays ``generator.generate_batch(n,
    seed)`` returns, whatever ``n``."""
    return (_identity(NOISE_PREFIX, noise_configuration(generator, seed)),
            _identity(SIGNAL_PREFIX, signal_configuration(generator, seed)))
