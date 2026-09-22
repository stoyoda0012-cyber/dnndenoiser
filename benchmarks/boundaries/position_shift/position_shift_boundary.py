"""P2-A: the position-shift boundary, and what augmentation does to it.

Registered design: `docs/preregistration/P2A-position-shift-boundary.md`
(registered 2026-09-22 at commit 06fa8c0, amended the same day as Revision 1
after two independent audits, at commit ef25766). Nothing in this script may
depart from that document; where it must, the document is revised first and the
revision log says why.

What is measured: a rigid energy shift of the whole spectrum -- the shape of a
sample-charging offset or a binding-energy calibration error -- swept to
+/-4.0 eV, against four models that differ in what position distribution they
were trained on. The registered question is not "does augmentation fix it" but
"does augmentation REMOVE the boundary or MOVE it", which is why the test range
reaches beyond the augmented arm's training range.

  arms        A narrow N=2304 | B augmented U(-1.5,1.5) N=2304
              C narrow N=461  | D narrow N=144   (position-density controls)
  seeds       20; a seed varies the data draw, the model init and the batch order
  shifts      25 values, 0 to +/-4.0 eV, denser near the expected boundaries
  metrics     M1 SNR gain in dB; M2 the boundary |delta|*; M3 argmax displacement
  primary     noise level 1000.0 ONLY -- the other two are descriptive

This script refuses to write a record if any of twelve self-checks fails. A
broken measurement is not reinterpreted as a finding.

Usage:

    python benchmarks/boundaries/position_shift/position_shift_boundary.py
    python benchmarks/boundaries/position_shift/position_shift_boundary.py --quick \
        --output-dir "$(mktemp -d)"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from math import comb
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy import stats
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset

from dnndenoiser.data.synthetic_generator import (
    GeneratorConfig,
    NoiseConfig,
    PeakParams,
    PeakSet,
    SyntheticGenerator,
    get_peak_set,
)
from dnndenoiser.models.network import DenoisingNetwork

RECORD_VERSION = "1"
PREREGISTRATION = "docs/preregistration/P2A-position-shift-boundary.md"
PREREGISTRATION_COMMITS = {"registered": "06fa8c0", "revision_1": "ef25766"}

# --------------------------------------------------------------------------------------
# Pinned design constants. Every one of these is fixed in the preregistration.
# --------------------------------------------------------------------------------------

# LITERALS. Self-checks 3 and 4 compare against these and never re-read PEAK_SETS,
# because `get_peak_set` returns the shared module-level object: an implementation
# that mutated it in place would otherwise be compared against its own mutation.
PEAK_SET_ID = "C1s_adventitious"
NOMINAL_CENTRES = (284.8, 286.3, 288.5)
NOMINAL_FWHM = (1.2, 1.3, 1.4)
NOMINAL_INTENSITY = (1.0, 0.3, 0.15)
DOMINANT_CENTRE = 284.8
ENERGY_RANGE = (277.8, 295.5)
N_ENERGY_POINTS = 256

MODEL_CONFIG = {"num_features": 256, "num_hidden_units": 100, "encoder_output_dim": 64}
ARCH = "ResNet-FCNN"
HYPERPARAMS = {"lr": 0.001, "epochs": 50, "batch_size": 16}
OPTIMIZER = "Adam"
WEIGHT_DECAY = 1e-9
LOSS = "HuberLoss(delta=1.0)"
LR_SCHEDULER = "StepLR(step_size=10, gamma=0.1)"
LR_DROP_PERIOD = 10
LR_DROP_FACTOR = 0.1
GRADIENT_CLIP_NORM = 4.0

GENERATOR_CONFIG_KWARGS = {
    "n_energy_points": N_ENERGY_POINTS,
    "eta": 0.3,
    "use_pseudo_voigt": True,
    "background_type": "linear",
    "background_level": 0.05,
    "background_slope": 0.001,
    "intensity_variation": 0.2,
    "position_jitter": 0.3,
    "width_variation": 0.1,
    "normalize": True,
    "energy_range": ENERGY_RANGE,
}
POSITION_JITTER = 0.3

NOISE_LEVELS = (100.0, 1000.0, 10000.0)
PRIMARY_LEVEL = 1000.0

# Arms. `train_per_level` sums to the registered pool size; `aug_halfwidth` is the
# half-width of the per-sample RIGID shift, 0.0 meaning no augmentation.
ARMS = {
    "A_narrow_2304": {"train_per_level": (768, 768, 768), "aug_halfwidth": 0.0},
    "B_augmented_2304": {"train_per_level": (768, 768, 768), "aug_halfwidth": 1.5},
    "C_narrow_461": {"train_per_level": (154, 154, 153), "aug_halfwidth": 0.0},
    "D_narrow_144": {"train_per_level": (48, 48, 48), "aug_halfwidth": 0.0},
}
ARM_ORDER = ("A_narrow_2304", "B_augmented_2304", "C_narrow_461", "D_narrow_144")
AUG_HALFWIDTH = 1.5

_POSITIVE_SHIFTS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0)
DELTAS = tuple(sorted({0.0} | {s * d for d in _POSITIVE_SHIFTS for s in (1.0, -1.0)}))
DELTA_MAX = 4.0

N_TEST_PER_LEVEL = 512
N_SEEDS = 20

# RNG stream bases. Train and test never share a base, and the offsets cannot collide
# for the seed / level / sample ranges used here; self-check 12 asserts it arithmetically
# rather than trusting this comment.
TRAIN_STREAM_BASE = 10_000_000
TEST_STREAM_BASE = 90_000_000
AUG_SHIFT_STREAM_BASE = 50_000_000
TORCH_SEED_BASE = 20_000

# Registered decision thresholds. `SIGN_K_OF_N` is the single uniform directional
# rule; `SIGN_K_CONTROL` is the positive control's.
SIGN_RULE = {"k": 15, "n": 20}
SIGN_RULE_CONTROL = {"k": 19, "n": 20}
ALPHA = 0.05

# Self-check tolerances, each set in the preregistration against a measured worst case.
TOL_TRUNCATION_TOTAL = 0.01
TOL_TRUNCATION_PER_PEAK = 0.02
TOL_INPUT_SNR_DB = {100.0: 0.2, 1000.0: 0.2, 10000.0: 0.35}
TOL_TRANSLATION_EQUIVARIANCE = 0.01
MIN_ARGMAX_WELL_POSED = 0.99
# Inter-peak spacing is preserved by construction; this bounds the floating-point
# residual of comparing shifted centres, not a physical tolerance.
SPACING_TOLERANCE_EV = 1e-9
CONSISTENCY_ANCHOR_SD_MULTIPLE = 3.0

QUICK_OVERRIDES = {"n_seeds": 2, "n_test_per_level": 64, "epochs_cap": 2}


class SelfCheckFailure(RuntimeError):
    """Raised when a self-check fails. The record is not written."""


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def snr_db(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-spectrum SNR in dB against a clean reference.

        signal power = mean(reference**2)          over the energy axis
        noise power  = mean((estimate - reference)**2)
        SNR (dB)     = 10 * log10(signal power / noise power)

    The reference is the clean synthetic spectrum AT THE SAME SHIFT -- the truth the
    model should have recovered, not the unshifted truth. Scoring against the unshifted
    reference would measure something else entirely; self-check 9 is what establishes
    that it was not done, by asserting identity between the array it inspects and the
    array this function divides by.

    Definition unchanged from `benchmarks/reference/reference_benchmark.py`, so the
    delta = 0 column is readable next to that record.
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    signal_power = np.mean(reference**2, axis=-1)
    noise_power = np.mean((estimate - reference) ** 2, axis=-1)
    if not np.all(noise_power > 0.0):
        raise ValueError(
            "zero noise power: an estimate is bit-identical to the reference, "
            "which makes SNR undefined rather than infinite"
        )
    if not np.all(signal_power > 0.0):
        raise ValueError("zero signal power: a reference spectrum is all zeros")
    return 10.0 * np.log10(signal_power / noise_power)


def argmax_energy(spectra: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Energy of the maximum bin. A GRID argmax, not a fitted peak position."""
    return energy[np.argmax(np.asarray(spectra), axis=-1)]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def shifted_peak_set(delta: float) -> PeakSet:
    """A peak set rigidly shifted by `delta`, built from FRESH PeakParams.

    Never mutates `PEAK_SETS`. `get_peak_set` returns the shared module-level object and
    its `PeakParams` are shared with it, so `p.mu += delta` in a loop would both
    accumulate across iterations and corrupt the registry for the whole process.
    `energy_range` is passed explicitly so `PeakSet.__post_init__` cannot auto-range the
    grid off the shifted centres.
    """
    return PeakSet(
        id=f"{PEAK_SET_ID}_shift{delta:+.3f}",
        name=f"{PEAK_SET_ID} shifted {delta:+.3f} eV",
        peaks=[
            PeakParams(mu=mu + delta, fwhm=fwhm, intensity=intensity)
            for mu, fwhm, intensity in zip(NOMINAL_CENTRES, NOMINAL_FWHM, NOMINAL_INTENSITY)
        ],
        energy_range=ENERGY_RANGE,
    )


def noise_config(level: float) -> NoiseConfig:
    """The reference benchmark's pinned reconstruction of the pre-2026-09-11 model.

    Adopted, not inherited: the preregistration's `Noise model` section states this as a
    decision, with its bias quantified. Note the boolean is FALSE at level 10000, so the
    three levels do not share a code path -- which is also why the noise is not paired
    across shifts at that level. Self-check 11 compares the realised config against these
    literals field by field.
    """
    return NoiseConfig(
        noise_type="poisson",
        poisson_level=level,
        use_gaussian_approx=(10000.0 / level) ** 2 > 20,
        gaussian_approx_min_rate=0.0,
    )


def make_generator(delta: float, level: float, **config_overrides) -> SyntheticGenerator:
    kwargs = dict(GENERATOR_CONFIG_KWARGS)
    kwargs.update(config_overrides)
    return SyntheticGenerator(
        peak_set=shifted_peak_set(delta),
        noise_config=noise_config(level),
        config=GeneratorConfig(**kwargs),
    )


def replay_sample_draws(sample_seed: int) -> list:
    """Reproduce `generate_single`'s per-peak draws for one sample.

    `generate_single` draws, per peak and in this order, the intensity multiplier, the
    position jitter and the width multiplier, each from the sample RNG, and none of them
    depends on `mu`. Replaying them is what lets self-check 8 compare two arms' realised
    per-peak draws without instrumenting the generator -- and it is what fails if the
    arms' RNG streams desynchronise, which is the failure no other check would see.
    """
    rng = np.random.default_rng(sample_seed)
    draws = []
    for _ in NOMINAL_CENTRES:
        intensity = rng.uniform(
            -GENERATOR_CONFIG_KWARGS["intensity_variation"],
            GENERATOR_CONFIG_KWARGS["intensity_variation"],
        )
        jitter = rng.uniform(-POSITION_JITTER, POSITION_JITTER)
        width = rng.uniform(
            -GENERATOR_CONFIG_KWARGS["width_variation"],
            GENERATOR_CONFIG_KWARGS["width_variation"],
        )
        draws.append((float(intensity), float(jitter), float(width)))
    return draws


def sample_seeds_of_batch(batch_seed: int, n_samples: int) -> list:
    """The per-sample seeds `generate_batch` derives from one batch seed."""
    base_rng = np.random.default_rng(batch_seed)
    return [int(base_rng.integers(0, 2**31)) for _ in range(n_samples)]


def train_sample_seed(seed_index: int, level_index: int, sample_index: int) -> int:
    return TRAIN_STREAM_BASE + seed_index * 1_000_000 + level_index * 100_000 + sample_index


def test_batch_seed(seed_index: int, level_index: int) -> int:
    """One batch seed per (seed, level), REUSED AT EVERY SHIFT.

    This is what pairs the test spectra to the spectrum across the sweep: the same seed
    with a peak set differing only in `mu` gives identical per-peak jitter, intensity and
    width draws, so the clean spectra differ only by the rigid shift and a per-sample
    normalisation constant.
    """
    return TEST_STREAM_BASE + seed_index * 1000 + level_index


def build_training_pool(seed_index: int, arm: str, n_per_level, rng_shift_base: int):
    """One arm's training pool, generated one sample at a time.

    Per-sample generation is what allows a per-sample RIGID shift. The shift is drawn
    from a SEPARATE stream, never from the sample RNG, so that every arm's per-sample
    generator seed -- and therefore its jitter, intensity and width draws -- is identical
    at equal index. That is the pairing self-check 8 verifies.
    """
    halfwidth = ARMS[arm]["aug_halfwidth"]
    shift_rng = np.random.default_rng(rng_shift_base + seed_index)
    clean_parts, noisy_parts, shifts, seeds_used, keys, levels = [], [], [], [], [], []
    energy = None
    for level_index, level in enumerate(NOISE_LEVELS):
        for sample_index in range(n_per_level[level_index]):
            delta = float(shift_rng.uniform(-halfwidth, halfwidth)) if halfwidth > 0 else 0.0
            batch_seed = train_sample_seed(seed_index, level_index, sample_index)
            generator = make_generator(delta, level)
            clean, noisy, energy, _meta = generator.generate_batch(1, seed=batch_seed)
            clean_parts.append(clean[0])
            noisy_parts.append(noisy[0])
            shifts.append(delta)
            seeds_used.append(batch_seed)
            keys.append((level_index, sample_index))
            levels.append(level)
    return {
        "clean": np.asarray(clean_parts, dtype=np.float32),
        "noisy": np.asarray(noisy_parts, dtype=np.float32),
        "shifts": np.asarray(shifts, dtype=np.float64),
        "batch_seeds": seeds_used,
        # Arms C and D are SUBSETS of arm A's sequence, so the pools align on
        # (level_index, sample_index) and not on flat position -- the per-level block
        # sizes differ. Self-check 8 keys on this.
        "keys": keys,
        "levels": levels,
        "energy": energy,
    }


def draw_test(seed_index: int, level_index: int, level: float, delta: float, n_samples: int):
    generator = make_generator(delta, level)
    clean, noisy, energy, _meta = generator.generate_batch(
        n_samples, seed=test_batch_seed(seed_index, level_index)
    )
    return clean.astype(np.float32), noisy.astype(np.float32), energy


# --------------------------------------------------------------------------------------
# Self-checks. Each has a stated failure condition; failing one voids the record.
# --------------------------------------------------------------------------------------


def _row_hashes(array: np.ndarray) -> set:
    return {hashlib.sha1(row.tobytes()).hexdigest() for row in np.ascontiguousarray(array)}


def _nearest_neighbour_rms(query: np.ndarray, pool: np.ndarray, exclude_self: bool):
    distances = cdist(query.astype(np.float64), pool.astype(np.float64), metric="euclidean")
    if exclude_self:
        np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1) / np.sqrt(query.shape[1])


def check_parameter_count(reference_record: dict) -> dict:
    """Self-check 1. The expected value is READ from the reference record, not typed."""
    expected = reference_record["self_checks"]["parameter_counts"]["observed"][ARCH]
    observed = sum(p.numel() for p in build_model().parameters() if p.requires_grad)
    if observed != expected:
        raise SelfCheckFailure(
            f"parameter count: {ARCH} has {observed}, reference record says {expected}"
        )
    return {"architecture": ARCH, "expected_from_reference_record": int(expected),
            "observed": int(observed), "passed": True}


def check_grid_and_rigidity() -> dict:
    """Self-checks 2 and 3: grid invariance, and rigidity of the test sweep.

    Rigidity is checked against the LITERAL nominal centres. A check that re-read its
    baseline through `get_peak_set` would compare a mutated registry against a result
    produced from the same mutation and pass on both.

    The comparison is `centre_k == literal_k + delta`, evaluated as that expression, and
    NOT `centre_k - literal_k == delta`. The preregistration registered the second form
    and it is not implementable: `(284.8 + delta) - 284.8` is not `delta` in binary
    floating point for an arbitrary delta, so the registered check failed on a correct
    augmented pool at the first smoke test. The first form is exact, tests the same
    property -- one shift common to every peak -- and still fails on a pool built with
    per-peak jitter, where each peak would carry its own offset. Recorded as Revision 2.
    """
    baseline_energy = make_generator(0.0, NOISE_LEVELS[0]).energy
    nominal_spacings = np.diff(NOMINAL_CENTRES)
    per_delta = {}
    for delta in DELTAS:
        generator = make_generator(delta, NOISE_LEVELS[0])
        if not np.array_equal(generator.energy, baseline_energy):
            raise SelfCheckFailure(f"grid moved at delta={delta}: energy axis is not identical")
        centres = np.array([p.mu for p in generator.peak_set.peaks], dtype=np.float64)
        expected = np.array([c + delta for c in NOMINAL_CENTRES], dtype=np.float64)
        residual = centres - expected
        if not np.all(residual == 0.0):
            raise SelfCheckFailure(
                f"non-rigid test sweep at delta={delta}: centres {centres.tolist()} "
                f"differ from literal+delta {expected.tolist()}"
            )
        spacing_error = float(np.max(np.abs(np.diff(centres) - nominal_spacings)))
        if spacing_error > SPACING_TOLERANCE_EV:
            raise SelfCheckFailure(
                f"inter-peak spacings changed at delta={delta} by {spacing_error:g} eV")
        per_delta[f"{delta:+.2f}"] = {
            "centres_eV": centres.tolist(),
            "residual_from_literal_plus_delta": residual.tolist(),
            "max_spacing_error_eV": spacing_error,
        }
    return {
        "grid_bit_identical_at_every_delta": True,
        "energy_step_eV": float(baseline_energy[1] - baseline_energy[0]),
        "baselines_are_literals": list(NOMINAL_CENTRES),
        "n_deltas_checked": len(DELTAS),
        "passed": True,
        "per_delta_offsets": per_delta,
    }


def check_pool_rigidity(pools: dict, rng: np.random.Generator) -> dict:
    """Self-check 4. Rigidity of every TRAINING pool, on a random 5% of each.

    The registered text's point: a pool built with `position_jitter = 1.8` would be
    trained on NON-rigid shifts -- the manipulation this study puts out of scope -- and
    every other check would pass. This is the one that fails.
    """
    report = {}
    for arm, pool in pools.items():
        n = len(pool["shifts"])
        sample = rng.choice(n, size=max(1, int(np.ceil(0.05 * n))), replace=False)
        for index in sample:
            delta = pool["shifts"][index]
            generator = make_generator(delta, NOISE_LEVELS[0])
            centres = np.array([p.mu for p in generator.peak_set.peaks], dtype=np.float64)
            expected = np.array([c + delta for c in NOMINAL_CENTRES], dtype=np.float64)
            if not np.all(centres - expected == 0.0):
                raise SelfCheckFailure(
                    f"{arm} sample {index}: centres {centres.tolist()} differ from "
                    f"literal+delta {expected.tolist()}, so the training shift is either "
                    "not rigid or not the shift that was recorded"
                )
        if GENERATOR_CONFIG_KWARGS["position_jitter"] != POSITION_JITTER:
            raise SelfCheckFailure("position_jitter is not 0.3 in the generator config")
        report[arm] = {"n_samples": int(n), "n_checked": int(len(sample)),
                       "position_jitter": POSITION_JITTER, "passed": True}
    live = get_peak_set(PEAK_SET_ID)
    live_centres = [p.mu for p in live.peaks]
    if live_centres != list(NOMINAL_CENTRES):
        raise SelfCheckFailure(
            f"PEAK_SETS['{PEAK_SET_ID}'] was mutated during the run: {live_centres}"
        )
    report["registry_unmutated"] = {"centres": live_centres, "passed": True}
    return report


def check_truncation() -> dict:
    """Self-check 5. Retained peak area, total and per peak, against the delta=0 value.

    The tails of an eta=0.3 pseudo-Voigt put about 1.6% of the nominal area outside this
    window at EVERY shift, zero included. What the cap on |delta| controls is the CHANGE,
    so both the absolute retention and the ratio are recorded: an earlier draft of the
    preregistration stated the ratio as if it were the absolute figure, which is the
    error this records against.

    The per-peak figures matter on their own: truncation is asymmetric between
    directions, in the same direction an asymmetric boundary would be, so any claimed
    directional asymmetry in the result has to clear the asymmetry recorded here.
    """
    from dnndenoiser.data.synthetic_generator import voigt_from_fwhm_and_eta

    wide = np.linspace(ENERGY_RANGE[0] - 60.0, ENERGY_RANGE[1] + 60.0, 200_001)
    window = np.linspace(ENERGY_RANGE[0], ENERGY_RANGE[1], 200_001)
    eta = GENERATOR_CONFIG_KWARGS["eta"]

    def areas(delta):
        total, inside = [], []
        for mu, fwhm, intensity in zip(NOMINAL_CENTRES, NOMINAL_FWHM, NOMINAL_INTENSITY):
            total.append(intensity * np.trapezoid(
                voigt_from_fwhm_and_eta(wide, mu + delta, fwhm, eta, use_pseudo=True), wide))
            inside.append(intensity * np.trapezoid(
                voigt_from_fwhm_and_eta(window, mu + delta, fwhm, eta, use_pseudo=True), window))
        return np.array(total), np.array(inside)

    total0, inside0 = areas(0.0)
    baseline_total_ratio = inside0.sum() / total0.sum()
    baseline_per_peak = inside0 / total0

    per_delta = {}
    for delta in DELTAS:
        total, inside = areas(delta)
        ratio_total = float((inside.sum() / total.sum()) / baseline_total_ratio)
        ratio_peaks = (inside / total) / baseline_per_peak
        if abs(ratio_total - 1.0) > TOL_TRUNCATION_TOTAL:
            raise SelfCheckFailure(
                f"truncation at delta={delta}: total retained area is {ratio_total:.5f} "
                f"of its delta=0 value, outside the {TOL_TRUNCATION_TOTAL:.0%} tolerance"
            )
        if np.any(np.abs(ratio_peaks - 1.0) > TOL_TRUNCATION_PER_PEAK):
            raise SelfCheckFailure(
                f"truncation at delta={delta}: per-peak ratios {ratio_peaks.tolist()}"
            )
        per_delta[f"{delta:+.2f}"] = {
            "absolute_retained_fraction": float(inside.sum() / total.sum()),
            "ratio_to_delta_zero": ratio_total,
            "per_peak_ratio_to_delta_zero": [float(x) for x in ratio_peaks],
        }
    return {
        "what_it_measures": (
            "fraction of the nominal pseudo-Voigt peak area falling inside the pinned "
            "energy window, absolute and relative to the delta=0 value"
        ),
        "absolute_retained_fraction_at_delta_zero": float(baseline_total_ratio),
        "tolerance_total_relative": TOL_TRUNCATION_TOTAL,
        "tolerance_per_peak_relative": TOL_TRUNCATION_PER_PEAK,
        "passed": True,
        "per_delta": per_delta,
    }


def check_translation_equivariance() -> dict:
    """Self-check 7. Is the shifted spectrum actually a TRANSLATE of the unshifted one?

    Self-checks 3 and 4 inspect peak CENTRES and would pass on a spectrum that is not a
    translate at all. The background is the reason it is not exactly one:
    `linear_background` is evaluated on the fixed absolute energy axis, so it does not
    move with the peaks. The exact translate is obtained by evaluating the unshifted
    spectrum on a grid displaced by -delta, which moves the background with it; the
    residual between the two is precisely the part of this manipulation that is not a
    rigid translation of the whole spectrum.
    """
    deterministic = {"intensity_variation": 0.0, "position_jitter": 0.0,
                     "width_variation": 0.0, "normalize": True}
    reference = make_generator(0.0, NOISE_LEVELS[0], **deterministic)
    baseline, _ = reference.generate_single(np.random.default_rng(0))
    scale = float(np.max(baseline))
    per_delta = {}
    worst = 0.0
    for delta in DELTAS:
        shifted = make_generator(delta, NOISE_LEVELS[0], **deterministic)
        actual, _ = shifted.generate_single(np.random.default_rng(0))
        translate_generator = make_generator(
            0.0, NOISE_LEVELS[0],
            energy_range=(ENERGY_RANGE[0] - delta, ENERGY_RANGE[1] - delta), **deterministic)
        translate, _ = translate_generator.generate_single(np.random.default_rng(0))
        residual = float(np.max(np.abs(actual - translate)) / scale)
        worst = max(worst, residual)
        if residual > TOL_TRANSLATION_EQUIVARIANCE:
            raise SelfCheckFailure(
                f"translation equivariance at delta={delta}: residual {residual:.5f} "
                f"exceeds {TOL_TRANSLATION_EQUIVARIANCE}"
            )
        per_delta[f"{delta:+.2f}"] = residual
    return {
        "what_it_measures": (
            "max|clean(delta) - translate(clean(0), delta)| / max(clean(0)), where the "
            "translate is the unshifted spectrum evaluated on a grid displaced by -delta. "
            "Non-zero because the linear background is evaluated on the fixed absolute "
            "energy axis and therefore does NOT move with the peaks."
        ),
        "tolerance": TOL_TRANSLATION_EQUIVARIANCE,
        "worst_residual": worst,
        "passed": True,
        "per_delta_residual": per_delta,
    }


def _realised_peak_set(delta: float, draws: list) -> PeakSet:
    """The peak set a sample actually got, reconstructed from the replayed draws."""
    return PeakSet(
        id="realised", name="realised", energy_range=ENERGY_RANGE,
        peaks=[
            PeakParams(mu=mu + delta + jitter, fwhm=fwhm * (1 + width),
                       intensity=intensity * (1 + amplitude))
            for (mu, fwhm, intensity), (amplitude, jitter, width)
            in zip(zip(NOMINAL_CENTRES, NOMINAL_FWHM, NOMINAL_INTENSITY), draws)
        ],
    )


def check_replay_faithfulness(pools: dict, rng: np.random.Generator, n_samples: int = 8) -> dict:
    """Is the replay in `replay_sample_draws` actually what the generator did?

    Self-check 8 compares replayed per-peak draws between arms. That comparison is only
    worth anything if the replay models the generator faithfully, so this rebuilds a
    handful of spectra from the replayed draws alone -- with every random variation
    switched off -- and requires them to be bit-identical to what the generator produced.
    Without this step, check 8 would be comparing one model of the generator against
    itself and would pass whatever the generator did.
    """
    deterministic = {"intensity_variation": 0.0, "position_jitter": 0.0, "width_variation": 0.0}
    checked = 0
    for arm in ARM_ORDER:
        pool = pools[arm]
        for index in rng.choice(len(pool["batch_seeds"]), size=min(n_samples, len(pool["batch_seeds"])),
                                replace=False):
            batch_seed = pool["batch_seeds"][index]
            delta = float(pool["shifts"][index])
            level = pool["levels"][index]
            sample_seed = sample_seeds_of_batch(batch_seed, 1)[0]
            draws = replay_sample_draws(sample_seed)

            actual, _ = make_generator(delta, level).generate_single(
                np.random.default_rng(sample_seed))
            kwargs = dict(GENERATOR_CONFIG_KWARGS)
            kwargs.update(deterministic)
            rebuilt, _ = SyntheticGenerator(
                peak_set=_realised_peak_set(delta, draws),
                noise_config=noise_config(level),
                config=GeneratorConfig(**kwargs),
            ).generate_single(np.random.default_rng(0))
            if not np.array_equal(actual, rebuilt):
                raise SelfCheckFailure(
                    f"{arm} sample {index}: the replayed draws do not reproduce the "
                    f"generator's spectrum (max abs diff "
                    f"{float(np.max(np.abs(actual - rebuilt))):g})"
                )
            checked += 1
    return {"n_spectra_rebuilt_from_replayed_draws": checked, "bit_identical": True,
            "passed": True}


def check_pairing_integrity(pools: dict, seed_index: int, n_test: int) -> dict:
    """Self-check 8. Do the arms, and the shifts, actually share their per-peak draws?

    If one arm consumed a shift draw from the sample RNG where another did not, every
    per-peak jitter, intensity and width would differ and the pairing that R4, R5 and R6
    rest on would be silently broken -- while every other self-check still passed.

    The arms are keyed on (level_index, sample_index), not on flat position: arms C and D
    hold fewer samples per level, so equal flat indices point at different levels. The
    first smoke test failed here, on a correct pair of pools, for exactly that reason.
    """
    reference_arm = ARM_ORDER[0]
    reference = dict(zip(pools[reference_arm]["keys"], pools[reference_arm]["batch_seeds"]))
    compared = 0
    for arm in ARM_ORDER[1:]:
        pool = pools[arm]
        for key, batch_seed in zip(pool["keys"], pool["batch_seeds"]):
            if key not in reference:
                raise SelfCheckFailure(f"{arm} has sample {key}, which {reference_arm} lacks")
            if batch_seed != reference[key]:
                raise SelfCheckFailure(
                    f"{arm} sample {key} uses batch seed {batch_seed}, {reference_arm} "
                    f"uses {reference[key]}: streams desynchronised"
                )
            sample_seed = sample_seeds_of_batch(batch_seed, 1)[0]
            reference_seed = sample_seeds_of_batch(reference[key], 1)[0]
            if replay_sample_draws(sample_seed) != replay_sample_draws(reference_seed):
                raise SelfCheckFailure(f"{arm} sample {key}: per-peak draws differ")
            compared += 1

    # The same question across the shift sweep: one batch seed per (seed, level), reused
    # at every shift, so the per-peak draws are identical and only the rigid shift differs.
    baseline_draws = {}
    for level_index in range(len(NOISE_LEVELS)):
        seeds = sample_seeds_of_batch(test_batch_seed(seed_index, level_index), n_test)
        baseline_draws[level_index] = [replay_sample_draws(s) for s in seeds[: min(16, n_test)]]
    for delta in DELTAS:
        for level_index in range(len(NOISE_LEVELS)):
            seeds = sample_seeds_of_batch(test_batch_seed(seed_index, level_index), n_test)
            again = [replay_sample_draws(s) for s in seeds[: len(baseline_draws[level_index])]]
            if again != baseline_draws[level_index]:
                raise SelfCheckFailure(
                    f"test per-peak draws differ at delta={delta}, level index {level_index}"
                )
            compared += len(again)
    return {"n_tuples_compared": int(compared * len(NOMINAL_CENTRES)),
            "arms_keyed_on": "(level_index, sample_index)", "passed": True}


def check_argmax_well_posed_and_reference_identity(
    clean: np.ndarray, scored_reference: np.ndarray, energy: np.ndarray, delta: float
) -> float:
    """Self-check 9. Argmax well-posedness AND the identity of M1's reference.

    The registered wording was ambiguous: read as "within 0.5 eV of 284.8" it scores
    0.0000 at delta = +/-3 and would void every run; read as "284.8 + delta" it scores
    1.0000. It is the second, and the identity assertion is what makes this check also
    the guarantee that M1 divides by the DELTA-MATCHED truth rather than the unshifted one.
    """
    if scored_reference is not clean:
        raise SelfCheckFailure(
            "the array inspected for argmax well-posedness is not the same object "
            "passed to M1 as the reference"
        )
    fraction = float(np.mean(np.abs(argmax_energy(clean, energy) - (DOMINANT_CENTRE + delta)) <= 0.5))
    if fraction < MIN_ARGMAX_WELL_POSED:
        raise SelfCheckFailure(
            f"argmax ill-posed at delta={delta}: only {fraction:.4f} of clean references "
            f"peak within 0.5 eV of {DOMINANT_CENTRE + delta:.2f} eV"
        )
    return fraction


def check_augmentation(pools: dict) -> dict:
    """Self-check 10. The narrow arms drew no shift; the augmented arm drew the range.

    The tolerance on the augmented arm's mean is 5*sigma/sqrt(n) with n stated, not a
    fixed 0.1 eV: a fixed bound would be 3.2 SE at one pool size and would void a few
    percent of correct runs.
    """
    report = {}
    for arm in ARM_ORDER:
        shifts = pools[arm]["shifts"]
        halfwidth = ARMS[arm]["aug_halfwidth"]
        n = len(shifts)
        if halfwidth == 0.0:
            if not np.all(shifts == 0.0):
                raise SelfCheckFailure(f"{arm} is a narrow arm but drew non-zero shifts")
            report[arm] = {"n": int(n), "all_zero": True, "passed": True}
            continue
        sigma = 2 * halfwidth / np.sqrt(12.0)
        tolerance = 5.0 * sigma / np.sqrt(n)
        if shifts.min() < -halfwidth or shifts.max() > halfwidth:
            raise SelfCheckFailure(f"{arm} drew a shift outside +/-{halfwidth}")
        if abs(float(np.mean(shifts))) > tolerance:
            raise SelfCheckFailure(
                f"{arm} shift mean {np.mean(shifts):.4f} exceeds 5*sigma/sqrt(n) = {tolerance:.4f}"
            )
        report[arm] = {
            "n": int(n), "halfwidth": halfwidth,
            "min": float(shifts.min()), "max": float(shifts.max()),
            "mean": float(np.mean(shifts)), "tolerance_5sigma_over_sqrt_n": float(tolerance),
            "passed": True,
        }
    return report


def input_snr_tolerance(level: float, n_test: int) -> float:
    """Self-check 6's tolerance, which is calibrated at the REGISTERED test size.

    Most of this statistic's spread across shifts is Monte-Carlo, not signal: the
    deterministic signal-power variation is about 0.03 dB while the measured span at
    n = 512 is around 0.10 dB. So a run at a smaller test set has a proportionally
    noisier statistic, and the registered tolerance would void it for a reason that has
    nothing to do with the manipulation -- which is what the first smoke test did at
    n = 64. The allowance is therefore scaled by sqrt(512 / n_test) and equals the
    registered value exactly at the registered size. Recorded as Revision 2.
    """
    return TOL_INPUT_SNR_DB[level] * float(np.sqrt(N_TEST_PER_LEVEL / max(1, n_test)))


def check_noise_model() -> dict:
    """Self-check 11. The realised NoiseConfig, field by field, against the literals."""
    expected = {
        100.0: {"noise_type": "poisson", "poisson_level": 100.0,
                "use_gaussian_approx": True, "gaussian_approx_min_rate": 0.0},
        1000.0: {"noise_type": "poisson", "poisson_level": 1000.0,
                 "use_gaussian_approx": True, "gaussian_approx_min_rate": 0.0},
        10000.0: {"noise_type": "poisson", "poisson_level": 10000.0,
                  "use_gaussian_approx": False, "gaussian_approx_min_rate": 0.0},
    }
    realised = {}
    for level in NOISE_LEVELS:
        config = noise_config(level)
        fields = {
            "noise_type": config.noise_type,
            "poisson_level": float(config.poisson_level),
            "use_gaussian_approx": bool(config.use_gaussian_approx),
            "gaussian_approx_min_rate": float(config.gaussian_approx_min_rate),
        }
        if fields != expected[level]:
            raise SelfCheckFailure(
                f"noise config at level {level}: realised {fields}, expected {expected[level]}"
            )
        realised[str(level)] = fields
    return {"realised": realised, "passed": True,
            "note": "the boolean is False at level 10000: the three levels do not "
                    "share a noise code path, which is why the noise is not paired "
                    "across shifts there"}


def check_leakage(pools: dict, delta_zero_test: dict) -> dict:
    """Self-check 12. Stream disjointness, and byte-identity at delta = 0.

    Registered limitation, stated rather than discovered: this can only detect a stream
    collision in the DELTA = 0 column, because at any other shift a collided spectrum is
    shifted and no longer byte-identical. The delta = 0 test set is therefore always in
    the hash set, and the stream-base disjointness is asserted arithmetically as well.
    """
    train_seeds = set()
    for pool in pools.values():
        train_seeds.update(pool["batch_seeds"])
    test_seeds = {test_batch_seed(seed, level)
                  for seed in range(N_SEEDS) for level in range(len(NOISE_LEVELS))}
    if train_seeds & test_seeds:
        raise SelfCheckFailure("train and test RNG stream bases collide")

    report = {"stream_bases_disjoint": True, "collisions_clean": 0, "collisions_noisy": 0}
    train_clean = pools[ARM_ORDER[0]]["clean"]
    clean_hashes = _row_hashes(train_clean)
    noisy_hashes = _row_hashes(pools[ARM_ORDER[0]]["noisy"])
    neighbour = []
    for payload in delta_zero_test.values():
        report["collisions_clean"] += len(_row_hashes(payload["clean"]) & clean_hashes)
        report["collisions_noisy"] += len(_row_hashes(payload["noisy"]) & noisy_hashes)
        neighbour.append(_nearest_neighbour_rms(payload["clean"], train_clean, exclude_self=False))
    if report["collisions_clean"] or report["collisions_noisy"]:
        raise SelfCheckFailure("a test spectrum is byte-identical to a training spectrum")
    test_to_train = np.concatenate(neighbour)
    train_to_train = _nearest_neighbour_rms(train_clean, train_clean, exclude_self=True)
    report["passed"] = True
    report["only_detects_collisions_at_delta_zero"] = (
        "at any other shift a collided spectrum is shifted and no longer byte-identical"
    )
    report["near_duplicate_analysis"] = {
        "what_it_measures": (
            "RMS distance from each clean delta=0 test spectrum to its nearest clean "
            "training spectrum, against the same statistic within the training set. A "
            "ratio near 1 is the correct outcome for an independent draw from one "
            "distribution, and is why the delta=0 column is in-distribution performance."
        ),
        "test_to_train_nn_rms_median": float(np.median(test_to_train)),
        "train_to_train_nn_rms_median": float(np.median(train_to_train)),
        "ratio_median_test_over_train": float(
            np.median(test_to_train) / np.median(train_to_train)),
    }
    return report


# --------------------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------------------


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def build_model() -> DenoisingNetwork:
    return DenoisingNetwork(
        num_features=MODEL_CONFIG["num_features"],
        num_hidden_units=MODEL_CONFIG["num_hidden_units"],
        layer_type=ARCH,
        encoder_output_dim=MODEL_CONFIG["encoder_output_dim"],
    )


def train_one(pool: dict, device: str, torch_seed: int, epochs_cap=None):
    """One training run. The recipe is the reference benchmark's, unchanged."""
    torch.manual_seed(torch_seed)
    model = build_model().to(device)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(torch_seed + 1)
    loader = DataLoader(
        TensorDataset(torch.tensor(pool["noisy"]), torch.tensor(pool["clean"])),
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=True,
        generator=loader_generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=HYPERPARAMS["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DROP_PERIOD, gamma=LR_DROP_FACTOR)
    criterion = torch.nn.HuberLoss(delta=1.0)
    epochs = HYPERPARAMS["epochs"] if epochs_cap is None else min(HYPERPARAMS["epochs"], epochs_cap)

    model.train()
    start = time.perf_counter()
    final_loss = float("nan")
    for _epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            prediction, _peak = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        final_loss = epoch_loss / max(1, n_batches)
    _sync(device)
    return model, {"epochs_run": epochs, "final_train_loss": final_loss,
                   "train_seconds": time.perf_counter() - start,
                   "n_train_spectra": int(len(pool["clean"]))}


@torch.no_grad()
def denoise(model, noisy: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    outputs = []
    tensor = torch.tensor(noisy, dtype=torch.float32)
    for start in range(0, len(tensor), batch_size):
        prediction, _peak = model(tensor[start : start + batch_size].to(device))
        outputs.append(prediction.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def gaussian_smoother_comparator(noisy: np.ndarray, clean: np.ndarray,
                                 energy: np.ndarray, sigmas_ev) -> dict:
    """M3 comparator (ii): a learning-free smoother.

    Registered because the four mechanisms an audit suspected of confounding M3 --
    asymmetric truncation, background asymmetry, the three-peak envelope's own asymmetry,
    and plain oversmoothing -- all produce a SHIFT-INDEPENDENT offset rather than an
    opposite-sign-to-delta signature. Recording the smoother's displacement at every
    shift is what keeps that statement checkable in this record rather than inherited
    from the audit.
    """
    from scipy.ndimage import gaussian_filter1d

    step = float(energy[1] - energy[0])
    out = {}
    reference_argmax = argmax_energy(clean, energy)
    for sigma in sigmas_ev:
        smoothed = gaussian_filter1d(noisy.astype(np.float64), sigma / step, axis=-1, mode="nearest")
        out[f"sigma_{sigma:.1f}eV"] = float(
            np.mean(argmax_energy(smoothed, energy) - reference_argmax))
    return out


# --------------------------------------------------------------------------------------
# M2 -- the boundary statistic
# --------------------------------------------------------------------------------------


def boundary_statistics(abs_deltas, gains) -> dict:
    """First crossing, sustained crossing and sign-change count for one seed's curve.

    `abs_deltas` is ascending and starts at 0.0; `gains` is the seed's mean SNR gain at
    each. The FIRST crossing is primary and is biased LOW wherever the curve wobbles near
    zero -- a bias that points in the direction that makes R3 easier to satisfy, which is
    why the SUSTAINED crossing is recorded beside it rather than argued about.
    """
    gains = np.asarray(gains, dtype=np.float64)
    abs_deltas = np.asarray(abs_deltas, dtype=np.float64)
    sign_changes = int(np.sum(np.diff(np.signbit(gains)) != 0))

    if gains[0] < 0.0:
        return {"defined": False, "reason": "gain at delta=0 is already negative",
                "first_crossing": None, "sustained_crossing": None,
                "sign_changes": sign_changes, "censored": False}

    def interpolate(i):
        g0, g1 = gains[i], gains[i + 1]
        if g0 == g1:
            return float(abs_deltas[i + 1])
        return float(abs_deltas[i] + (abs_deltas[i + 1] - abs_deltas[i]) * g0 / (g0 - g1))

    first = None
    for i in range(len(gains) - 1):
        if gains[i] >= 0.0 > gains[i + 1]:
            first = interpolate(i)
            break

    sustained = None
    if gains[-1] < 0.0:
        j = len(gains) - 1
        while j > 0 and gains[j - 1] < 0.0:
            j -= 1
        sustained = interpolate(j - 1) if j > 0 else float(abs_deltas[0])

    return {"defined": True, "first_crossing": first, "sustained_crossing": sustained,
            "sign_changes": sign_changes, "censored": first is None}


def kaplan_meier_median(crossings, censored_count: int, bound: float):
    """KM median of |delta|*, treating non-crossing as right-censoring at `bound`.

    Registered as the censoring-aware estimator. It is reported with the observation
    that, because every censoring time here is administrative and falls at `bound` --
    beyond every observed event -- the Kaplan-Meier estimate coincides exactly with the
    order-statistic median. The registered estimator therefore adds no information in
    this design. That is recorded rather than quietly dropped.
    """
    events = sorted(float(x) for x in crossings)
    n = len(events) + censored_count
    if n == 0:
        return None
    survival = 1.0
    at_risk = n
    for index, time_point in enumerate(events):
        survival *= 1.0 - 1.0 / at_risk
        at_risk -= 1
        if survival <= 0.5:
            return time_point
    return None


def summarise_boundary(per_seed: list, n_seeds: int) -> dict:
    """The registered summary: median with the censored count, plus the KM median.

    Censored seeds ENTER THE ORDER STATISTICS AT THEIR BOUND and are never dropped;
    dropping them would select on the outcome and bias the boundary down. The median is
    a point value only while the censored count leaves the middle order statistics
    uncensored.
    """
    defined = [s for s in per_seed if s["defined"]]
    if not defined:
        return {"defined": False, "reason": "gain at delta=0 is already negative",
                "n_seeds_defined": 0}
    crossings = [s["first_crossing"] for s in defined if not s["censored"]]
    censored = sum(1 for s in defined if s["censored"])
    n = len(defined)
    ordered = sorted(crossings) + [float("inf")] * censored
    max_censored_for_point = (n + 1) // 2 - 1
    if censored <= max_censored_for_point:
        median = float(np.median([x for x in ordered]))
    else:
        median = None
    re_crossing = sum(1 for s in defined if s["sign_changes"] > 1)
    return {
        "defined": True,
        "n_seeds_defined": n,
        "n_censored": censored,
        "censoring_bound_eV": DELTA_MAX,
        "median_first_crossing_eV": median,
        "median_reported_as": (f"> {DELTA_MAX}" if median is None else "point value"),
        "point_value_allowed_while_censored_at_most": max_censored_for_point,
        "min_first_crossing_eV": float(min(crossings)) if crossings else None,
        "max_first_crossing_eV": float(max(crossings)) if crossings else None,
        "kaplan_meier_median_eV": kaplan_meier_median(crossings, censored, DELTA_MAX),
        "median_sustained_crossing_eV": (
            float(np.median([s["sustained_crossing"] for s in defined
                             if s["sustained_crossing"] is not None]))
            if any(s["sustained_crossing"] is not None for s in defined) else None),
        "n_seeds_re_crossing": re_crossing,
        "unreliable": re_crossing > n / 4.0,
        "per_seed_first_crossing_eV": [s["first_crossing"] for s in defined],
        "per_seed_sign_changes": [s["sign_changes"] for s in defined],
    }


# --------------------------------------------------------------------------------------
# Inference rules
# --------------------------------------------------------------------------------------


def binomial_one_sided(k: int, n: int) -> float:
    """P(X >= k | n, p = 0.5). Every sign rule here is DIRECTIONAL, so one-sided."""
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2**n)


def sign_test(values, positive: bool, k: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    favouring = int(np.sum(values > 0) if positive else np.sum(values < 0))
    return {
        "n_seeds": n, "n_favouring": favouring, "k_required": k,
        "one_sided_binomial_p": binomial_one_sided(favouring, n) if n else None,
        "passed": favouring >= k,
    }


def paired_t(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or np.all(values == values[0]):
        return {"mean": float(np.mean(values)), "sd": 0.0, "t": None, "p": None, "cohens_dz": None}
    result = stats.ttest_1samp(values, 0.0)
    sd = float(np.std(values, ddof=1))
    return {
        "mean": float(np.mean(values)), "sd": sd,
        "ci95_halfwidth": float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))),
        "t": float(result.statistic), "p": float(result.pvalue),
        "cohens_dz": float(np.mean(values) / sd) if sd > 0 else None,
    }


def holm(pvalues: list) -> list:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = float(min(1.0, running))
    return adjusted


# --------------------------------------------------------------------------------------
# Registered predictions
# --------------------------------------------------------------------------------------

POSITIVE_ABS_DELTAS = (0.0,) + _POSITIVE_SHIFTS


def dkey(delta: float) -> str:
    """Canonical JSON key for a shift.

    `+ 0.0` normalises negative zero: the negative-direction sweep is built as
    `-1.0 * d`, which gives -0.0 at the origin and would key the same cell twice.
    """
    return f"{delta + 0.0:+.2f}"


def index_runs(runs: list) -> dict:
    """{metric: {arm: {level: {delta_key: [per-seed value, ordered by seed]}}}}."""
    indexed = {}
    for entry in runs:
        for metric in ("snr_gain_db_mean", "argmax_displacement_ev_mean", "input_snr_db_mean"):
            indexed.setdefault(metric, {}).setdefault(entry["arm"], {}).setdefault(
                str(entry["level"]), {}).setdefault(dkey(entry["delta"]), []).append(
                    (entry["seed_index"], entry[metric]))
    out = {}
    for metric, arms in indexed.items():
        out[metric] = {}
        for arm, levels in arms.items():
            out[metric][arm] = {}
            for level, deltas in levels.items():
                out[metric][arm][level] = {
                    k: np.array([v for _s, v in sorted(pairs)], dtype=np.float64)
                    for k, pairs in deltas.items()}
    return out


def boundary_table(gains: dict, level: str) -> dict:
    """M2 for every arm and both directions at one level."""
    table = {}
    for arm in ARM_ORDER:
        table[arm] = {}
        for direction, sign in (("positive", 1.0), ("negative", -1.0)):
            n_seeds = len(gains[arm][level][dkey(0.0)])
            per_seed = []
            for seed_index in range(n_seeds):
                curve = [gains[arm][level][dkey(sign * d)][seed_index] for d in POSITIVE_ABS_DELTAS]
                per_seed.append(boundary_statistics(POSITIVE_ABS_DELTAS, curve))
            table[arm][direction] = summarise_boundary(per_seed, n_seeds)
            table[arm][direction]["_per_seed_raw"] = per_seed
    return table


def evaluate_predictions(indexed: dict, boundaries: dict) -> dict:
    """R1-R7, at the primary level only, exactly as registered."""
    level = str(PRIMARY_LEVEL)
    gain = indexed["snr_gain_db_mean"]
    disp = indexed["argmax_displacement_ev_mean"]
    k = SIGN_RULE["k"]
    results = {}

    # R1 -- positive control, arm A at delta = 0.
    values = gain["A_narrow_2304"][level][dkey(0.0)]
    sign = sign_test(values, positive=True, k=SIGN_RULE_CONTROL["k"])
    results["R1"] = {
        "statement": "arm A gains at delta = 0 (positive control)",
        "rule": "mean > 0 AND positive in >= %d of %d seeds" % (SIGN_RULE_CONTROL["k"], SIGN_RULE_CONTROL["n"]),
        "family": "none", "level": PRIMARY_LEVEL,
        "stats": paired_t(values), "sign": sign,
        "passed": bool(np.mean(values) > 0 and sign["passed"]),
        "gates_the_record": True,
    }

    # R2 -- the cliff, arm A at +/-4.0.
    per_direction, pvalues = {}, []
    for delta in (DELTA_MAX, -DELTA_MAX):
        values = gain["A_narrow_2304"][level][dkey(delta)]
        t = paired_t(values)
        per_direction[dkey(delta)] = {"stats": t, "sign": sign_test(values, positive=False, k=k)}
        pvalues.append(t["p"] if t["p"] is not None else 1.0)
    for adjusted, key in zip(holm(pvalues), per_direction):
        per_direction[key]["holm_adjusted_p"] = adjusted
        per_direction[key]["passed"] = bool(
            per_direction[key]["stats"]["mean"] < 0
            and per_direction[key]["sign"]["passed"] and adjusted < ALPHA)
    results["R2"] = {
        "statement": f"arm A's gain is negative at delta = +{DELTA_MAX} and -{DELTA_MAX}",
        "rule": f"in EACH direction: mean < 0 AND negative in >= {k} of "
                f"{SIGN_RULE['n']} seeds AND Holm-adjusted t-test p < {ALPHA}",
        "family": "the two directions", "level": PRIMARY_LEVEL,
        "per_direction": per_direction,
        "passed": all(v["passed"] for v in per_direction.values()),
    }

    # R3 -- the cliff is narrow: |delta|* <= 1.5 in at least one direction.
    per_direction = {}
    for direction in ("positive", "negative"):
        summary = boundaries[level]["A_narrow_2304"][direction]
        median = summary.get("median_first_crossing_eV")
        per_direction[direction] = {
            "median_first_crossing_eV": median,
            "n_censored": summary.get("n_censored"),
            "unreliable": summary.get("unreliable"),
            "passed": bool(summary.get("defined") and median is not None and median <= 1.5
                           and summary.get("n_censored", 99) <= 1
                           and not summary.get("unreliable")),
        }
    results["R3"] = {
        "statement": "arm A's |delta|* is <= 1.5 eV in at least one direction",
        "rule": "in AT LEAST ONE direction: median first crossing <= 1.5 eV AND "
                "at most 1 censored seed AND M2 not flagged unreliable",
        "family": "none", "level": PRIMARY_LEVEL, "threshold_eV": 1.5,
        "per_direction": per_direction,
        "passed": any(v["passed"] for v in per_direction.values()),
        "note": ("the registered text names this the prediction most likely to be wrong "
                 "and the one that matters: a 1.5 eV charging or calibration offset is "
                 "ordinary in XPS"),
    }

    # R4 -- at |delta| = 1.5, arm B gains more than arm A.
    per_direction, pvalues = {}, []
    for delta in (1.5, -1.5):
        diff = gain["B_augmented_2304"][level][dkey(delta)] - gain["A_narrow_2304"][level][dkey(delta)]
        t = paired_t(diff)
        per_direction[dkey(delta)] = {"stats": t, "sign": sign_test(diff, positive=True, k=k)}
        pvalues.append(t["p"] if t["p"] is not None else 1.0)
    for adjusted, key in zip(holm(pvalues), per_direction):
        per_direction[key]["holm_adjusted_p"] = adjusted
        per_direction[key]["passed"] = bool(per_direction[key]["sign"]["passed"])
    results["R4"] = {
        "statement": "at |delta| = 1.5, arm B gains more than arm A (paired within seed)",
        "rule": f"in BOTH directions: B - A positive in >= {k} of {SIGN_RULE['n']} "
                f"seeds. The Holm-adjusted t-test p is REPORTED, not required",
        "family": "the two directions", "level": PRIMARY_LEVEL,
        "per_direction": per_direction,
        "passed": all(v["passed"] for v in per_direction.values()),
    }

    # R5a -- density alone costs: A > C > D at delta = 0.
    orderings = {}
    for label, high, low in (("A_over_C", "A_narrow_2304", "C_narrow_461"),
                             ("C_over_D", "C_narrow_461", "D_narrow_144")):
        diff = gain[high][level][dkey(0.0)] - gain[low][level][dkey(0.0)]
        orderings[label] = {"stats": paired_t(diff), "sign": sign_test(diff, positive=True, k=k)}
        orderings[label]["passed"] = orderings[label]["sign"]["passed"]
    results["R5a"] = {
        "statement": "at delta = 0, arm A > arm C > arm D (training density alone costs)",
        "rule": f"BOTH orderings positive in >= {k} of {SIGN_RULE['n']} seeds",
        "family": "the two orderings", "level": PRIMARY_LEVEL, "orderings": orderings,
        "passed": all(v["passed"] for v in orderings.values()),
    }

    # R5b -- augmentation costs beyond density: B <= D at delta = 0.
    diff = gain["D_narrow_144"][level][dkey(0.0)] - gain["B_augmented_2304"][level][dkey(0.0)]
    sign = sign_test(diff, positive=True, k=k)
    results["R5b"] = {
        "statement": "at delta = 0, arm B <= arm D (augmentation costs beyond the density penalty)",
        "rule": f"D - B positive in >= {k} of {SIGN_RULE['n']} seeds; the paired mean "
                f"difference and its CI are reported alongside",
        "family": "none", "level": PRIMARY_LEVEL,
        "compared_against": "arm D, the LOWER density bound and therefore the stricter test",
        "stats": paired_t(diff), "sign": sign, "passed": sign["passed"],
        "if_failed": ("no augmentation cost beyond the density penalty was demonstrated -- "
                      "an undecided verdict, not a finding that augmentation is free"),
    }

    # R6 -- the boundary moves rather than vanishing.
    per_direction = {}
    for direction in ("positive", "negative"):
        a = boundaries[level]["A_narrow_2304"][direction]
        b = boundaries[level]["B_augmented_2304"][direction]
        if not (a.get("defined") and b.get("defined")):
            per_direction[direction] = {"defined": False, "verdict": "not defined",
                                        "reason": a.get("reason") or b.get("reason")}
            continue
        larger, ties, n_compared = 0, 0, 0
        for a_seed, b_seed in zip(a["_per_seed_raw"], b["_per_seed_raw"]):
            if a_seed["censored"] and b_seed["censored"]:
                ties += 1
                continue
            n_compared += 1
            if b_seed["censored"]:
                larger += 1
            elif not a_seed["censored"] and b_seed["first_crossing"] > a_seed["first_crossing"]:
                larger += 1
        k_reduced = int(np.ceil(k / SIGN_RULE["n"] * n_compared)) if n_compared else 0
        second_half = ("moved" if b["n_censored"] < b["n_seeds_defined"] else "beyond range")
        per_direction[direction] = {
            "defined": True,
            "n_compared": n_compared, "n_ties_excluded": ties,
            "n_b_larger": larger, "k_required_at_reduced_n": k_reduced,
            "one_sided_binomial_p": binomial_one_sided(larger, n_compared) if n_compared else None,
            "first_part_passed": bool(n_compared and larger >= k_reduced),
            "verdict": second_half,
            "arm_A_median_eV": a.get("median_first_crossing_eV"),
            "arm_B_median_eV": b.get("median_first_crossing_eV"),
            "arm_B_n_censored": b.get("n_censored"),
        }
    results["R6"] = {
        "statement": "arm B's |delta|* exceeds arm A's where arm A has one",
        "rule": "first part: B's crossing larger than A's, paired within seed, in "
                "at least k of the non-tied seeds (k scaled to the reduced n). "
                "Second part: the three-way verdict below. R6 PASSES only when a "
                "direction satisfies the first part AND its verdict is 'moved'",
        "family": "none", "level": PRIMARY_LEVEL, "per_direction": per_direction,
        "three_way_verdict": {
            "moved": "arm B has a crossing inside the tested range; the boundary was relocated",
            "beyond range": ("arm B has no crossing inside the tested range. The record states "
                             "that its boundary lies beyond it and makes NO claim that "
                             "augmentation relocates rather than removes the boundary. This is "
                             "an UNDECIDED verdict, not a falsification of R6."),
            "not defined": "the gain at delta = 0 is already negative at this level",
        },
        "passed": any(v.get("first_part_passed") and v.get("verdict") == "moved"
                      for v in per_direction.values()),
    }

    # R7 -- bias-corrected argmax displacement opposite in sign to delta.
    baseline = disp["A_narrow_2304"][level][dkey(0.0)]
    sign_points, pvalues = {}, []
    for delta in (1.0, -1.0, DELTA_MAX, -DELTA_MAX):
        corrected = disp["A_narrow_2304"][level][dkey(delta)] - baseline
        expect_positive = delta < 0
        t = paired_t(corrected)
        sign_points[dkey(delta)] = {
            "stats": t, "sign": sign_test(corrected, positive=expect_positive, k=k),
            "expected_sign": "positive" if expect_positive else "negative"}
        pvalues.append(t["p"] if t["p"] is not None else 1.0)
    for adjusted, key in zip(holm(pvalues), sign_points):
        sign_points[key]["holm_adjusted_p"] = adjusted
        sign_points[key]["passed"] = sign_points[key]["sign"]["passed"]

    grid_step = (ENERGY_RANGE[1] - ENERGY_RANGE[0]) / (N_ENERGY_POINTS - 1)
    monotone = {}
    for direction, s in (("positive", 1.0), ("negative", -1.0)):
        magnitudes = [float(np.mean(np.abs(disp["A_narrow_2304"][level][dkey(s * d)] - baseline)))
                      for d in (1.0, 1.5, 2.0, 3.0, 4.0)]
        violations = [magnitudes[i] - magnitudes[i + 1] for i in range(len(magnitudes) - 1)]
        worst = max(violations) if violations else 0.0
        monotone[direction] = {
            "abs_displacement_eV_at_1_1p5_2_3_4": magnitudes,
            "worst_violation_eV": worst, "tolerance_one_grid_step_eV": grid_step,
            "passed": worst <= grid_step}
    results["R7"] = {
        "statement": ("for arm A at |delta| >= 1.0, the BIAS-CORRECTED mean argmax "
                      "displacement disp(delta) - disp(0) has the opposite sign to delta "
                      "and grows with |delta|"),
        "family": "the four sign tests", "level": PRIMARY_LEVEL,
        "rule": (f"correct sign in >= {k} of {SIGN_RULE['n']} seeds at EACH of "
                 "delta = +/-1.0 and +/-4.0, AND |displacement| non-decreasing over "
                 "|delta| = 1.0, 1.5, 2.0, 3.0, 4.0 within each direction, allowing "
                 "decreases no larger than one grid step. The Holm-adjusted t-test p "
                 "is REPORTED, not required"),
        "sign_points": sign_points, "monotonicity": monotone,
        "passed": (all(v["passed"] for v in sign_points.values())
                   and all(v["passed"] for v in monotone.values())),
        "no_mechanism_claimed": ("opposite-signed displacement is CONSISTENT WITH a learned "
                                 "position prior; the observable underdetermines it, and the "
                                 "arm-B comparator is what separates it from a generic "
                                 "positional bias"),
    }
    return results


def suspect_run_rule(predictions: dict) -> dict:
    """R2 and R7 both failing while R1 passes is what a reference mix-up looks like.

    R1 is insensitive to it because the shifted and unshifted references coincide at
    delta = 0. Without this rule the publish-failed-predictions policy would publish a
    false negative result.
    """
    suspect = (predictions["R1"]["passed"]
               and not predictions["R2"]["passed"]
               and not predictions["R7"]["passed"])
    return {"suspect": bool(suspect),
            "what_it_would_indicate": "the reference wiring, not a negative result",
            "action_if_suspect": "re-verify M1's reference before writing a record"}


# --------------------------------------------------------------------------------------
# Record scaffolding
# --------------------------------------------------------------------------------------

CLAIM_SCOPE = {
    "supports": (
        "Under a rigid energy shift of the whole spectrum -- the shape of a charging "
        "offset or a binding-energy calibration error -- a ResNet-FCNN trained on this "
        "package's synthetic spectra at the recorded position distribution scored the "
        "reported SNR gain, and its boundary |delta|* fell where the record says it fell."
    ),
    "does_not_support": [
        "anything about MEASURED spectra: every spectrum here is synthetic, scored "
        "against a reference that exists only because it is synthetic, and the training "
        "noise and the test noise come from the same function, so the model's noise "
        "model is exactly correct by construction -- a condition measured data never "
        "satisfies",
        "a general position-shift threshold for XPS denoising: |delta|* is a property of "
        "THIS peak set, THIS jitter width, THIS architecture, THIS training-set size and "
        "THIS noise model, and one point was measured in each of those spaces",
        "anything about other training-set sizes, for any claim including R5: arms C and "
        "D bound the density penalty at one architecture and one recipe",
        "anything about other augmentation widths: one width (+/-1.5 eV) was tested, so "
        "R6 is a statement about that width and not about augmentation in general",
        "a full-spectrum translate: the linear background is evaluated on the fixed "
        "absolute energy axis and does NOT move with the peaks, so this manipulation is "
        "'peaks shift under a stationary background'. Self-check 7 bounds the resulting "
        "departure from a pure translate; it does not remove it",
        "separation of degradation from window-edge effects beyond |delta| = 1.5 eV. "
        "Inside that range arm B IS an edge-proximity control, because it saw those edge "
        "distances in training; beyond it no arm did",
        "anything about non-rigid shifts: chemical shifts move components relative to "
        "one another and this manipulation cannot speak to them",
        "anything about optimisation budget: all arms get 50 epochs and one schedule, so "
        "a deficit arm B shows may be under-training rather than augmentation cost",
        "anything about peak areas, widths or fitted positions surviving denoising: M3 "
        "is a grid argmax and is not a peak fit",
        "a recommendation that augmentation is the right mitigation: R5 and R6 are "
        "designed to show its price and its edge, and measuring a mitigation is not "
        "endorsing it",
        "a mechanism for M3: opposite-signed displacement is CONSISTENT WITH a learned "
        "position prior, and the observable underdetermines it",
    ],
    "device_dependence": (
        "R1, R2, R4, R5 and R7 are sign and ordering claims. R3 and R6 compare a recorded "
        "MAGNITUDE -- |delta|*, in eV -- against a fixed threshold, and are therefore "
        "subject to the same device caveat as any magnitude here. Floating-point "
        "reduction order differs between CPU, MPS and CUDA backends; no cross-device "
        "comparison is recorded unless one is run, and none is asserted."
    ),
    "descriptive_only_rule": (
        "Only the cells named in R1-R7, at the primary noise level, carry an inferential "
        "claim. Every other cell -- all other shifts, all other levels, all other arms, "
        "and both derived series -- is descriptive, is reported without a p-value, and no "
        "statement of the form 'gain dips at delta = x' or 'the curve is asymmetric at x' "
        "may be made about a cell not named in a prediction. A feature seen there is a "
        "candidate for a NEW preregistration, not a finding of this one."
    ),
    "denoised_output_is_a_model_estimate": (
        "The evaluated quantity is agreement with a known synthetic reference. A high SNR "
        "gain does not establish that structure in the output is real; the network can "
        "oversmooth, suppress weak features, and produce plausible structure that was not "
        "in the input."
    ),
}


def environment_record(device: str) -> dict:
    record = {
        "device_requested_resolved_to": device,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if device == "cuda" and torch.cuda.is_available():
        record["cuda_device_name"] = torch.cuda.get_device_name(0)
    return record


def design_record(n_seeds: int, n_test_per_level: int, epochs_cap) -> dict:
    return {
        "preregistration": {
            "document": PREREGISTRATION,
            "commits": PREREGISTRATION_COMMITS,
            "predictions_fixed_before_implementation": True,
        },
        "manipulated": {
            "variable": "rigid energy shift delta (eV) applied to every peak at once",
            "grid_held_fixed": True,
            "delta_values_eV": list(DELTAS),
            "delta_max_eV": DELTA_MAX,
            "why_capped_here": (
                "so truncation cannot masquerade as failure. The eta=0.3 tails put about "
                "1.6% of the nominal peak area outside this window at EVERY delta, zero "
                "included; the cap controls the CHANGE, which crosses the 1% tolerance at "
                "|delta| ~ 4.1 eV on the negative side -- the dominant peak."
            ),
            "not_in_scope": (
                "non-rigid shifts. A chemical shift moves components relative to one "
                "another; this manipulation is the charging/calibration case only."
            ),
        },
        "arms": {
            arm: {
                "train_per_level": list(spec["train_per_level"]),
                "n_train": int(sum(spec["train_per_level"])),
                "augmentation_halfwidth_eV": spec["aug_halfwidth"],
                "per_peak_jitter_eV": POSITION_JITTER,
            }
            for arm, spec in ARMS.items()
        },
        "why_density_controls_exist": (
            "arms A and B differ in TWO things, not one: augmentation, and training "
            "density on the position axis. At equal N, arm B has 1/5 of arm A's per-peak "
            "marginal density and 1/16 in the three-peak joint configuration space. Arms "
            "C and D are arm A at those two densities, so that R5 can separate the cost "
            "of augmentation from the cost of spreading a fixed N over a wider range."
        ),
        "data": {
            "provenance": ("generated in-process by dnndenoiser.data.synthetic_generator."
                           "SyntheticGenerator; no measured data, no external files"),
            "peak_set_id": PEAK_SET_ID,
            "nominal_centres_eV": list(NOMINAL_CENTRES),
            "nominal_fwhm_eV": list(NOMINAL_FWHM),
            "nominal_intensity": list(NOMINAL_INTENSITY),
            "energy_range_eV": list(ENERGY_RANGE),
            "n_energy_points": N_ENERGY_POINTS,
            "generator_config": dict(GENERATOR_CONFIG_KWARGS),
            "n_test_per_level_per_delta": n_test_per_level,
            "why_this_test_size": (
                "NOT for precision on the mean gain -- the replicate is the seed. 512 is "
                "retained because |delta|* is a first-crossing statistic whose downward "
                "bias is governed by the noise in each seed's own gain curve and is "
                "one-sided, so no amount of averaging across seeds removes it."
            ),
        },
        "noise_model": {
            "levels": list(NOISE_LEVELS),
            "config_per_level": {
                str(level): {
                    "noise_type": "poisson", "poisson_level": level,
                    "use_gaussian_approx": (10000.0 / level) ** 2 > 20,
                    "gaussian_approx_min_rate": 0.0,
                } for level in NOISE_LEVELS},
            "adopted_not_inherited": (
                "this is the reference benchmark's pinned reconstruction of the package's "
                "PRE-2026-09-11 behaviour, adopted so the delta=0 column is comparable "
                "with that record. The boolean is FALSE at level 10000, so the three "
                "levels do not share a code path. Under this configuration the worst-bin "
                "upward bias at level 1000 is about 0.25% and is approximately "
                "delta-independent, which is why it does not confound the manipulated axis."
            ),
        },
        "split_rule": {
            "rule": "train and test draw from disjoint RNG stream bases",
            "leakage_preventing_unit": "the independently generated spectrum",
            "limitation": ("the byte-identity check can only detect a stream collision in "
                           "the delta = 0 column; at any other shift a collided spectrum "
                           "is shifted and no longer byte-identical"),
        },
        "regime": (
            "training and inference operate at the same signal-to-noise regime: all three "
            "noise levels are in every training pool and each is evaluated separately. "
            "The noise level is not manipulated; position is."
        ),
        "pairing": {
            "across_delta_within_seed_and_level": (
                "one generator batch seed per (seed, level), REUSED at every shift, so a "
                "test spectrum at one shift and the same index at another share their "
                "per-peak jitter, intensity and width draws; the CLEAN spectra differ only "
                "by the rigid shift and a per-sample normalisation constant"),
            "noise_realisation_across_delta": (
                "a property of the noise branch, not of this design. Measured at design "
                "time, corr(noise at delta=0, noise at delta=0.5) is +0.977 at level 100, "
                "+0.976 at level 1000 and +0.177 at level 10000, because the level-10000 "
                "branch is rng.poisson, whose bit consumption is rate-dependent. "
                "Degradation curves at level 10000 therefore carry more draw noise."),
            "across_arms_within_seed": (
                "all arms are generated from the same per-sample seed sequence and "
                "evaluated on identical test arrays; the shift is drawn from a SEPARATE "
                "stream so it cannot desynchronise the sample RNG"),
            "across_seeds": "independent: data draw, model init and batch order all vary",
            "replicate_unit": "the seed",
        },
        "model": {
            "architecture": ARCH, "model_config": dict(MODEL_CONFIG),
            "hyperparameters": dict(HYPERPARAMS), "optimizer": OPTIMIZER,
            "weight_decay": WEIGHT_DECAY, "loss": LOSS, "lr_scheduler": LR_SCHEDULER,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "epochs_cap_applied": epochs_cap,
        },
        "metrics": {
            "M1_snr_gain_db": {
                "definition": "10*log10(mean(ref^2)/mean((est-ref)^2)), output minus input",
                "reference": "the clean synthetic spectrum AT THE SAME SHIFT",
                "derived": "degradation = gain(delta) - gain(0), computed within a seed",
            },
            "M2_boundary": {
                "primary": "first crossing of zero mean gain, linear interpolation in dB",
                "also_recorded": ["sustained crossing", "number of sign changes"],
                "bias": ("the first-crossing estimator is biased LOW wherever the curve "
                         "wobbles near zero, and that bias makes R3 easier to satisfy; the "
                         "sustained crossing is recorded so its size is visible"),
                "censoring": ("a seed that never crosses enters the order statistics at the "
                              "bound and is NEVER dropped; dropping would select on the "
                              "outcome and bias the boundary down"),
                "not_computed_where": "the mean gain at delta = 0 is already negative",
            },
            "M3_argmax_displacement_ev": {
                "primary_form": "disp(delta) - disp(0), computed within a seed",
                "why_bias_corrected": (
                    "a learning-free Gaussian smoother produces a SHIFT-INDEPENDENT "
                    "positive offset of about +0.06/+0.21/+0.48 eV at sigma = 0.5/1.0/2.0 "
                    "eV, present on clean spectra too -- three grid steps at sigma ~ 1 eV, "
                    "which would help at delta < 0 and hurt at delta > 0"),
                "comparators": ["the noisy input", "a learning-free Gaussian smoother",
                                "arm B on identical test arrays"],
                "limits": ("a grid argmax quantised to the energy step, not a fitted peak "
                           "position; describes the dominant peak only; supports no claim "
                           "about fitted binding energies"),
            },
        },
        "inference": {
            "n_seeds": n_seeds,
            "primary_level": PRIMARY_LEVEL,
            "all_predictions_evaluated_at_primary_level_only": True,
            "sign_rule": dict(SIGN_RULE),
            "sign_rule_positive_control": dict(SIGN_RULE_CONTROL),
            "sign_rules_are_one_sided_because_every_prediction_is_directional": True,
            "one_sided_p_of_sign_rule": binomial_one_sided(SIGN_RULE["k"], SIGN_RULE["n"]),
            "alpha": ALPHA,
            "multiplicity": "Holm within the family each prediction names",
            "error_bars": ("standard deviation across seeds. The per-spectrum spread inside "
                           "one test set is NOT a replicate spread and is stored under "
                           "snr_gain_db_sd_over_spectra_not_a_replicate_sd"),
        },
        "reproducibility": {
            "bit_exact_across_devices": False,
            "data_is_platform_independent": True,
            "note": "numpy.random.Generator draws identically everywhere; torch reductions do not",
        },
    }


def read_reference_record() -> tuple:
    """The reference benchmark's record, read at run time.

    AGENTS.md section 6 forbids restating a numeric result in a second place, and the
    reference README is explicit that a hand-typed number there has no provenance. So no
    reference value is a literal in this script or in the preregistration: the parameter
    count, the consistency-anchor gains, their across-seed SDs and the input-SNR label
    are all read from the file that owns them.
    """
    path = (Path(__file__).resolve().parents[2] / "reference" / "results"
            / "reference_benchmark.json")
    if not path.exists():
        raise SelfCheckFailure(f"reference record not found at {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def consistency_anchor(reference_record: dict, path: Path, aggregates: dict) -> dict:
    """Not a prediction. Arm A at delta = 0 should land near the record that owns it.

    A flag is not a failure: the draws differ, the seed count differs and the per-sample
    generation path differs. It is a prompt to explain the difference before publishing.
    """
    reference = reference_record["aggregates"]["suggested-hyperparameters"][ARCH]
    per_level = {}
    flagged = False
    for level in NOISE_LEVELS:
        key = str(level)
        if key not in reference:
            continue
        reference_mean = float(reference[key]["snr_gain_db_mean"])
        reference_sd = reference[key].get("snr_gain_db_sd_across_seeds")
        here = aggregates["A_narrow_2304"][key][dkey(0.0)]["snr_gain_db_mean"]
        difference = here - reference_mean
        threshold = (CONSISTENCY_ANCHOR_SD_MULTIPLE * float(reference_sd)
                     if reference_sd else None)
        level_flagged = bool(threshold is not None and abs(difference) > threshold)
        flagged = flagged or level_flagged
        per_level[key] = {
            "reference_snr_gain_db_mean": reference_mean,
            "reference_snr_gain_db_sd_across_seeds": reference_sd,
            "here_arm_A_delta_zero": here,
            "difference_db": difference,
            "flag_threshold_db": threshold,
            "flagged": level_flagged,
        }
    return {
        "read_from": str(path),
        "key_path": "aggregates['suggested-hyperparameters']['" + ARCH + "'][level]",
        "no_reference_number_is_typed_into_this_script": True,
        "flag_rule": f"{CONSISTENCY_ANCHOR_SD_MULTIPLE} x the reference record's across-seed SD",
        "a_flag_is_not_a_failure": (
            "the draws differ, the seed count differs and the per-sample generation path "
            "differs; a flag is a prompt to explain the difference in this record"),
        "any_flagged": flagged,
        "per_level": per_level,
    }


def run(args) -> dict:
    started = time.perf_counter()
    device = resolve_device(args.device)
    n_seeds = args.seeds
    n_test = args.n_test_per_level
    reference_record, reference_path = read_reference_record()

    print(f"device={device}  seeds={n_seeds}  deltas={len(DELTAS)}  "
          f"arms={len(ARM_ORDER)}  n_test/level/delta={n_test}")

    # Seed-independent self-checks, cheapest first.
    self_checks = {
        "1_parameter_count": check_parameter_count(reference_record),
        "2_and_3_grid_and_test_sweep_rigidity": check_grid_and_rigidity(),
        "11_noise_model_identity": check_noise_model(),
        "5_truncation": check_truncation(),
        "7_translation_equivariance": check_translation_equivariance(),
    }
    print("seed-independent self-checks passed "
          f"(worst translation residual {self_checks['7_translation_equivariance']['worst_residual']:.5f})")

    runs = []
    per_seed_checks = []
    input_snr_spans = {str(level): [] for level in NOISE_LEVELS}
    argmax_well_posed = {}
    smoother_by_delta = {}
    determinism = None

    for seed_index in range(n_seeds):
        seed_started = time.perf_counter()
        pools = {arm: build_training_pool(seed_index, arm, ARMS[arm]["train_per_level"],
                                          AUG_SHIFT_STREAM_BASE)
                 for arm in ARM_ORDER}
        checks = {
            "4_training_pool_rigidity": check_pool_rigidity(
                pools, np.random.default_rng(777 + seed_index)),
            "8_pairing_integrity": check_pairing_integrity(pools, seed_index, n_test),
            "8b_replay_faithfulness": check_replay_faithfulness(
                pools, np.random.default_rng(31337 + seed_index)),
            "10_augmentation": check_augmentation(pools),
        }

        models = {}
        for arm in ARM_ORDER:
            model, info = train_one(pools[arm], device, TORCH_SEED_BASE + seed_index,
                                    epochs_cap=args.epochs_cap)
            models[arm] = model
            info["arm"] = arm
            checks.setdefault("training", {})[arm] = info

        if seed_index == 0 and not args.skip_determinism_check:
            repeat, _ = train_one(pools["A_narrow_2304"], device, TORCH_SEED_BASE,
                                  epochs_cap=args.epochs_cap)

        delta_zero_test = {}
        for level_index, level in enumerate(NOISE_LEVELS):
            input_snr_here = []
            for delta in DELTAS:
                clean, noisy, energy = draw_test(seed_index, level_index, level, delta, n_test)
                fraction = check_argmax_well_posed_and_reference_identity(
                    clean, clean, energy, delta)
                argmax_well_posed[dkey(delta)] = min(
                    argmax_well_posed.get(dkey(delta), 1.0), fraction)

                snr_in = snr_db(noisy, clean)
                input_snr_here.append(float(np.mean(snr_in)))
                clean_argmax = argmax_energy(clean, energy)

                if seed_index == 0:
                    smoother_by_delta.setdefault(str(level), {})[dkey(delta)] = (
                        gaussian_smoother_comparator(noisy, clean, energy, (0.5, 1.0, 2.0)))
                    smoother_by_delta[str(level)][dkey(delta)]["noisy_input"] = float(
                        np.mean(argmax_energy(noisy, energy) - clean_argmax))
                if delta == 0.0:
                    delta_zero_test[level] = {"clean": clean, "noisy": noisy}

                for arm in ARM_ORDER:
                    estimate = denoise(models[arm], noisy, device)
                    gain = snr_db(estimate, clean) - snr_in
                    runs.append({
                        "seed_index": seed_index, "arm": arm, "level": level, "delta": delta,
                        "n_test_spectra": int(len(gain)),
                        "input_snr_db_mean": float(np.mean(snr_in)),
                        "output_snr_db_mean": float(np.mean(snr_db(estimate, clean))),
                        "snr_gain_db_mean": float(np.mean(gain)),
                        "snr_gain_db_sd_over_spectra_not_a_replicate_sd": float(
                            np.std(gain, ddof=1)),
                        "argmax_displacement_ev_mean": float(
                            np.mean(argmax_energy(estimate, energy) - clean_argmax)),
                    })
                    if seed_index == 0 and delta == 0.0 and level == PRIMARY_LEVEL \
                            and arm == "A_narrow_2304" and not args.skip_determinism_check:
                        repeat_gain = snr_db(denoise(repeat, noisy, device), clean) - snr_in
                        determinism = {
                            "what_it_measures": ("arm A trained twice on identical inputs at "
                                                 "one seed; recorded, never asserted"),
                            "first_snr_gain_db_mean": float(np.mean(gain)),
                            "repeat_snr_gain_db_mean": float(np.mean(repeat_gain)),
                            "absolute_difference_db": float(
                                abs(np.mean(gain) - np.mean(repeat_gain))),
                        }

            span = max(input_snr_here) - min(input_snr_here)
            tolerance = input_snr_tolerance(level, n_test)
            if span > tolerance:
                raise SelfCheckFailure(
                    f"input-SNR invariance at level {level}, seed {seed_index}: span "
                    f"{span:.4f} dB over the {len(DELTAS)}-point sweep exceeds "
                    f"{tolerance:.4f} dB (registered {TOL_INPUT_SNR_DB[level]} dB at "
                    f"n_test = {N_TEST_PER_LEVEL})"
                )
            input_snr_spans[str(level)].append(float(span))

        checks["12_leakage"] = check_leakage(pools, delta_zero_test)
        checks["9_argmax_well_posedness_worst_fraction"] = float(min(argmax_well_posed.values()))
        per_seed_checks.append(checks)
        print(f"  seed {seed_index + 1}/{n_seeds} done in "
              f"{(time.perf_counter() - seed_started) / 60:.1f} min")

    self_checks["6_input_snr_invariance"] = {
        "tolerance_db_per_level_as_registered": {str(k): v for k, v in TOL_INPUT_SNR_DB.items()},
        "tolerance_db_per_level_applied": {
            str(level): input_snr_tolerance(level, n_test) for level in NOISE_LEVELS},
        "tolerance_calibrated_at_n_test": N_TEST_PER_LEVEL,
        "n_test_used": n_test,
        "measured_span_db_per_level": {
            level: {"max": float(max(spans)), "mean": float(np.mean(spans)),
                    "per_seed": spans}
            for level, spans in input_snr_spans.items() if spans},
        "why_level_10000_is_wider": (
            "its noise is not paired across shifts (corr ~ 0.18 against ~ 0.98), so the "
            "statistic carries a Monte-Carlo component of its own"),
        "passed": True,
    }
    self_checks["9_argmax_well_posedness_and_reference_identity"] = {
        "compared_against": "284.8 + delta, not 284.8",
        "reference_identity_asserted": True,
        "minimum_fraction_required": MIN_ARGMAX_WELL_POSED,
        "worst_fraction_observed": float(min(argmax_well_posed.values())),
        "per_delta_worst": argmax_well_posed,
        "passed": True,
    }
    self_checks["4_8_10_12_per_seed"] = per_seed_checks

    # Aggregation.
    indexed = index_runs(runs)
    aggregates = {}
    for arm in ARM_ORDER:
        aggregates[arm] = {}
        for level in NOISE_LEVELS:
            key = str(level)
            aggregates[arm][key] = {}
            baseline = indexed["snr_gain_db_mean"][arm][key][dkey(0.0)]
            for delta in DELTAS:
                values = indexed["snr_gain_db_mean"][arm][key][dkey(delta)]
                degradation = values - baseline
                displacement = indexed["argmax_displacement_ev_mean"][arm][key][dkey(delta)]
                disp_baseline = indexed["argmax_displacement_ev_mean"][arm][key][dkey(0.0)]
                aggregates[arm][key][dkey(delta)] = {
                    "n_seeds": int(len(values)),
                    "snr_gain_db_mean": float(np.mean(values)),
                    "snr_gain_db_sd_across_seeds": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else None),
                    "per_seed_snr_gain_db": [float(x) for x in values],
                    "degradation_db_mean": float(np.mean(degradation)),
                    "degradation_db_sd_across_seeds": (
                        float(np.std(degradation, ddof=1)) if len(values) > 1 else None),
                    "argmax_displacement_ev_mean": float(np.mean(displacement)),
                    "argmax_displacement_bias_corrected_ev_mean": float(
                        np.mean(displacement - disp_baseline)),
                    "input_snr_db_mean": float(
                        np.mean(indexed["input_snr_db_mean"][arm][key][dkey(delta)])),
                }

    boundaries = {str(level): boundary_table(indexed["snr_gain_db_mean"], str(level))
                  for level in NOISE_LEVELS}
    predictions = evaluate_predictions(indexed, boundaries)
    suspect = suspect_run_rule(predictions)

    for level_table in boundaries.values():
        for arm_table in level_table.values():
            for direction_summary in arm_table.values():
                direction_summary.pop("_per_seed_raw", None)

    return {
        "record_version": RECORD_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).name),
        "claim_scope": CLAIM_SCOPE,
        "environment": environment_record(device),
        "design": design_record(n_seeds, n_test, args.epochs_cap),
        "self_checks": self_checks,
        "diagnostics": {
            "note": ("recorded, never asserted: nothing here can void the record, and it is "
                     "listed separately so the count of actual gates is not overstated"),
            "repeat_run_determinism": determinism,
            "m3_comparators_seed_0": smoother_by_delta,
        },
        "runs": runs,
        "aggregates": aggregates,
        "boundaries": boundaries,
        "predictions": predictions,
        "suspect_run_rule": suspect,
        "consistency_anchor": consistency_anchor(reference_record, reference_path, aggregates),
        "total_wall_clock_seconds": time.perf_counter() - started,
        "quick_mode": bool(args.quick),
    }


def main(argv=None) -> int:
    default_output = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--output-name", default="position_shift_boundary.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--n-test-per-level", type=int, default=N_TEST_PER_LEVEL)
    parser.add_argument("--epochs-cap", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke test; the numbers are not meaningful")
    parser.add_argument("--skip-determinism-check", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds = QUICK_OVERRIDES["n_seeds"]
        args.n_test_per_level = QUICK_OVERRIDES["n_test_per_level"]
        args.epochs_cap = QUICK_OVERRIDES["epochs_cap"]

    record = run(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / args.output_name
    destination.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\nwrote {destination}")
    print(f"total wall clock: {record['total_wall_clock_seconds'] / 60:.1f} min")
    verdicts = {name: value["passed"] for name, value in record["predictions"].items()}
    print("predictions: " + "  ".join(f"{n}={'PASS' if v else 'FAIL'}" for n, v in verdicts.items()))
    if record["suspect_run_rule"]["suspect"]:
        print("SUSPECT RUN: R2 and R7 both failed while R1 passed -- re-verify the reference wiring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
