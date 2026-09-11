"""Reference denoising benchmark: per-architecture SNR gain on synthetic XPS spectra.

This script exists so that any per-architecture number shown to a user has a record
behind it. It measures one thing and states exactly what that thing is:

    "Architecture X, at the hyperparameters this software suggests for it, trained
     and tested on synthetic spectra drawn from the generator described in the
     record, reached an SNR gain of ... dB."

It does NOT measure, and the record must not be read as measuring, that one
architecture is intrinsically better than another. See `README.md` next to this file
and the `claim_scope` block of the emitted JSON.

Design summary (the full, machine-readable version is written into the JSON record):

  data          synthetic spectra from `dnndenoiser.data.synthetic_generator`
  split rule    train and test are independent draws from disjoint RNG streams;
                the leakage-preventing unit is the independently generated spectrum
  noise model   `NoiseConfig(noise_type='poisson', poisson_level=L)` at several L
  seeds         5 seeds; a seed varies both the data draw and the model init
  metric        SNR gain in dB, defined explicitly in `snr_db()` below
  conditions    each architecture at its own suggested hyperparameters (primary),
                optionally also at one identical budget for all (secondary)

Usage:

    python benchmarks/reference/reference_benchmark.py --output-dir <dir>
    python benchmarks/reference/reference_benchmark.py --quick        # smoke test
    python benchmarks/reference/reference_benchmark.py --matched-budget --matched-seeds 3
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
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
    SyntheticGenerator,
    get_peak_set,
)
from dnndenoiser.models.network import DenoisingNetwork

RECORD_VERSION = "1"

# --------------------------------------------------------------------------------------
# Fixed experimental conditions. Changing any of these invalidates comparison with an
# older record; bump RECORD_VERSION and say so in the README when you do.
# --------------------------------------------------------------------------------------

# Baseline first: it is the reference of the paired comparison. The rest follow the
# order the GUI dropdown lists them in.
BASELINE_ARCH = "FCNN"
ARCHITECTURES = (
    "FCNN",
    "ResNet-FCNN",
    "1D-CNN",
    "ResNet-1DCNN",
    "GRU",
    "LSTM",
    "bi-LSTM",
    "Transformer",
)

# Model shape. These are the GUI's default spin-box values, and the CLI's defaults for
# --hidden-units / --encoder-dim.
MODEL_CONFIG = {"num_features": 256, "num_hidden_units": 100, "encoder_output_dim": 64}

# Trainable-parameter counts at MODEL_CONFIG. Not a measurement — a property of the
# model that the script re-derives and asserts, so a silent architecture change fails
# loudly instead of quietly moving the numbers.
EXPECTED_PARAM_COUNTS = {
    "FCNN": 37345,
    "ResNet-1DCNN": 91682,
    "1D-CNN": 151785,
    "GRU": 193957,
    "LSTM": 249957,
    "bi-LSTM": 579657,
    "ResNet-FCNN": 658177,
    "Transformer": 861706,
}

# Primary comparison condition: each architecture at the hyperparameters a graphical
# training tool suggests for it. That tool is not part of this repository; the values
# originated in its `panels/training_panel.py::OPTIMAL_HYPERPARAMS` and are literals
# here, so this script is self-contained. Where the tool is present alongside this
# checkout, `crosscheck_gui_hyperparams()` parses its source and fails if the two ever
# disagree; where it is absent, the cross-check records that and the run proceeds.
SUGGESTED_HYPERPARAMS = {
    "bi-LSTM": {"lr": 0.01, "epochs": 30, "batch_size": 32},
    "LSTM": {"lr": 0.01, "epochs": 30, "batch_size": 32},
    "GRU": {"lr": 0.01, "epochs": 30, "batch_size": 32},
    "Transformer": {"lr": 0.001, "epochs": 50, "batch_size": 32},
    "ResNet-FCNN": {"lr": 0.001, "epochs": 50, "batch_size": 16},
    "ResNet-1DCNN": {"lr": 0.001, "epochs": 50, "batch_size": 16},
    "FCNN": {"lr": 0.01, "epochs": 30, "batch_size": 32},
    "1D-CNN": {"lr": 0.01, "epochs": 30, "batch_size": 32},
}

# Secondary condition: one identical budget for every architecture. lr=0.001 is the more
# conservative of the two learning rates the GUI suggests.
MATCHED_BUDGET_HYPERPARAMS = {"lr": 0.001, "epochs": 30, "batch_size": 32}

# Optimizer / loss / schedule: the recipe the suggested hyperparameters above are the
# companions of, from the same tool (`workers/training_worker.py`). Literals here, so a
# run needs nothing outside this repository.
OPTIMIZER = "Adam"
WEIGHT_DECAY = 1e-9
LOSS = "HuberLoss(delta=1.0)"
LR_SCHEDULER = "StepLR(step_size=10, gamma=0.1)"
LR_DROP_PERIOD = 10
LR_DROP_FACTOR = 0.1
GRADIENT_CLIP_NORM = 4.0

# Data.
PEAK_SET = "C1s_adventitious"
GENERATOR_CONFIG_KWARGS = {
    "n_energy_points": 256,
    "eta": 0.3,
    "use_pseudo_voigt": True,
    "background_type": "linear",
    "background_level": 0.05,
    "background_slope": 0.001,
    "intensity_variation": 0.2,
    "position_jitter": 0.3,
    "width_variation": 0.1,
    "normalize": True,
}
NOISE_LEVELS = (100.0, 1000.0, 10000.0)
N_TRAIN_PER_LEVEL = 768
N_TEST_PER_LEVEL = 512
N_SEEDS = 5

# RNG stream bases. Train and test draws never share a base, and the offsets cannot
# collide for the seed/level ranges used here.
TRAIN_STREAM_BASE = 10_000_000
TEST_STREAM_BASE = 90_000_000
TORCH_SEED_BASE = 20_000

QUICK_OVERRIDES = {
    "n_train_per_level": 128,
    "n_test_per_level": 64,
    "n_seeds": 2,
    "epochs_cap": 3,
}


# --------------------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------------------


def snr_db(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-spectrum SNR in dB against a clean reference.

        signal power = mean(reference**2)          over the energy axis
        noise power  = mean((estimate - reference)**2)
        SNR (dB)     = 10 * log10(signal power / noise power)

    Both the input spectrum and the denoised estimate are scored against the same
    reference, and SNR gain is (output SNR - input SNR).

    The reference is the clean synthetic spectrum. This metric is only available
    because the data is synthetic; there is no reference-free equivalent for measured
    spectra, and none is implied here.

    The whole spectrum, background included, counts as signal. A definition that scored
    only the peak region would give different numbers.
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


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def make_generator(noise_level: float) -> SyntheticGenerator:
    return SyntheticGenerator(
        peak_set=PEAK_SET,
        # PINNED, and not to be "tidied" to the defaults. The published record
        # was measured when the Gaussian approximation to Poisson was this
        # package's default AND was selected once from the peak expected count
        # for the whole spectrum. On 2026-09-11 Poisson became the default and
        # admissibility of the approximation became a per-bin test.
        #
        # Naming the flag alone would NOT reproduce the record, because the
        # meaning of the flag changed. The earlier rule was exactly "every bin
        # approximated, if and only if the PEAK expected count exceeds 20", and
        # that is what is reconstructed here: a floor of zero approximates every
        # bin, and the condition selects the branch the old code would have
        # taken. Verified bit-identical to the earlier implementation at every
        # level this benchmark uses, in one, two and three dimensions -- over
        # THIS benchmark's inputs, which are finite, non-negative and drawn at
        # NOISE_LEVELS. It is not a drop-in replacement in general: the old
        # function returned a copy for `level <= 0` and for an all-zero
        # spectrum, and returned all-NaN rather than raising when the input
        # carried NaN or infinity. None of those is reachable from here.
        #
        # This is the one call site allowed to ask for the superseded model, and
        # only so that a published measurement stays reproducible rather than
        # being silently re-measured. See `design.noise_model.implementation_note`.
        noise_config=NoiseConfig(
            noise_type="poisson", poisson_level=noise_level,
            use_gaussian_approx=(10000.0 / noise_level) ** 2 > 20,
            gaussian_approx_min_rate=0.0),
        config=GeneratorConfig(**GENERATOR_CONFIG_KWARGS),
    )


def draw(noise_level: float, n_samples: int, stream_seed: int):
    """One independent draw of (clean, noisy) at a fixed noise level."""
    generator = make_generator(noise_level)
    clean, noisy, energy, _metadata = generator.generate_batch(n_samples, seed=stream_seed)
    return clean.astype(np.float32), noisy.astype(np.float32), energy


def train_stream_seed(seed_index: int, level_index: int) -> int:
    return TRAIN_STREAM_BASE + seed_index * 1000 + level_index


def test_stream_seed(seed_index: int, level_index: int) -> int:
    return TEST_STREAM_BASE + seed_index * 1000 + level_index


def build_seed_datasets(seed_index: int, n_train_per_level: int, n_test_per_level: int):
    """Build the train pool and the per-level test sets for one seed.

    The training pool mixes all noise levels; the test sets stay separate so results can
    be reported per level. Every architecture in this seed sees exactly these arrays, so
    architectures are compared paired on identical data.
    """
    train_clean, train_noisy = [], []
    test_sets = {}
    energy = None
    for level_index, level in enumerate(NOISE_LEVELS):
        clean, noisy, energy = draw(level, n_train_per_level, train_stream_seed(seed_index, level_index))
        train_clean.append(clean)
        train_noisy.append(noisy)

        t_clean, t_noisy, _ = draw(level, n_test_per_level, test_stream_seed(seed_index, level_index))
        test_sets[level] = {"clean": t_clean, "noisy": t_noisy}

    return {
        "train_clean": np.concatenate(train_clean, axis=0),
        "train_noisy": np.concatenate(train_noisy, axis=0),
        "test": test_sets,
        "energy": energy,
    }


def _row_hashes(array: np.ndarray) -> set:
    return {hashlib.sha1(row.tobytes()).hexdigest() for row in np.ascontiguousarray(array)}


def _nearest_neighbour_rms(query: np.ndarray, pool: np.ndarray, exclude_self: bool) -> np.ndarray:
    """Per-row RMS distance to the closest row of `pool`."""
    distances = cdist(query.astype(np.float64), pool.astype(np.float64), metric="euclidean")
    if exclude_self:
        np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1) / np.sqrt(query.shape[1])


def leakage_check(datasets: dict) -> dict:
    """Two separate questions about the split, kept separate.

    1. Is any test spectrum byte-identical to a training spectrum? That would mean the
       split silently failed (a reused seed, an accidentally shared generator). The
       split rule — independent draws from disjoint RNG streams — is what creates the
       separation; this is what catches it not happening.

    2. How close are test spectra to the nearest training spectrum, compared with how
       close training spectra are to each other? The generator has few free parameters
       (per-peak intensity, position and width), so a fresh draw naturally lands near
       previous draws. That is not leakage — it is the definition of an in-distribution
       test set — but the ratio has to be reported rather than assumed, because it is
       the quantity that says how much of the measured performance is generalization
       and how much is a dense sampling of a small parameter space.
    """
    train_clean = datasets["train_clean"]
    train_clean_hashes = _row_hashes(train_clean)
    train_noisy_hashes = _row_hashes(datasets["train_noisy"])
    collisions_clean = 0
    collisions_noisy = 0
    n_test = 0
    test_neighbour_distances = []
    for payload in datasets["test"].values():
        n_test += len(payload["clean"])
        collisions_clean += len(_row_hashes(payload["clean"]) & train_clean_hashes)
        collisions_noisy += len(_row_hashes(payload["noisy"]) & train_noisy_hashes)
        test_neighbour_distances.append(
            _nearest_neighbour_rms(payload["clean"], train_clean, exclude_self=False)
        )
    test_to_train = np.concatenate(test_neighbour_distances)
    train_to_train = _nearest_neighbour_rms(train_clean, train_clean, exclude_self=True)
    return {
        "n_train_spectra": int(len(train_clean)),
        "n_test_spectra": int(n_test),
        "identical_clean_spectra": int(collisions_clean),
        "identical_noisy_spectra": int(collisions_noisy),
        "passed": collisions_clean == 0 and collisions_noisy == 0,
        "near_duplicate_analysis": {
            "what_it_measures": (
                "RMS distance from each clean test spectrum to its nearest clean "
                "training spectrum, against the same statistic computed within the "
                "training set. A ratio near 1 means the test set is a fresh draw from "
                "the same distribution, not a disguised copy of the training set; it "
                "also means this benchmark measures in-distribution performance only."
            ),
            "test_to_train_nn_rms_median": float(np.median(test_to_train)),
            "test_to_train_nn_rms_min": float(np.min(test_to_train)),
            "train_to_train_nn_rms_median": float(np.median(train_to_train)),
            "ratio_median_test_over_train": float(
                np.median(test_to_train) / np.median(train_to_train)
            ),
        },
    }


# --------------------------------------------------------------------------------------
# Training / evaluation
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


def build_model(arch: str) -> DenoisingNetwork:
    return DenoisingNetwork(
        num_features=MODEL_CONFIG["num_features"],
        num_hidden_units=MODEL_CONFIG["num_hidden_units"],
        layer_type=arch,
        encoder_output_dim=MODEL_CONFIG["encoder_output_dim"],
    )


def train_one(arch, hyperparams, x_train, y_train, device, torch_seed, epochs_cap=None):
    """Train one model. The same torch seed drives init and batch order."""
    torch.manual_seed(torch_seed)
    model = build_model(arch).to(device)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(torch_seed + 1)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        generator=loader_generator,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparams["lr"], weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DROP_PERIOD, gamma=LR_DROP_FACTOR
    )
    criterion = torch.nn.HuberLoss(delta=1.0)

    epochs = hyperparams["epochs"] if epochs_cap is None else min(hyperparams["epochs"], epochs_cap)

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
    seconds = time.perf_counter() - start
    return model, {"epochs_run": epochs, "final_train_loss": final_loss, "train_seconds": seconds}


@torch.no_grad()
def denoise(model, noisy: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    outputs = []
    tensor = torch.tensor(noisy, dtype=torch.float32)
    for start in range(0, len(tensor), batch_size):
        chunk = tensor[start : start + batch_size].to(device)
        prediction, _peak = model(chunk)
        outputs.append(prediction.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate(model, datasets: dict, device: str) -> dict:
    """Evaluate one trained model on every per-level test set."""
    per_level = {}
    for level, payload in datasets["test"].items():
        estimate = denoise(model, payload["noisy"], device)
        snr_in = snr_db(payload["noisy"], payload["clean"])
        snr_out = snr_db(estimate, payload["clean"])
        gain = snr_out - snr_in
        per_level[str(level)] = {
            "n_test_spectra": int(len(gain)),
            "input_snr_db_mean": float(np.mean(snr_in)),
            "output_snr_db_mean": float(np.mean(snr_out)),
            "snr_gain_db_mean": float(np.mean(gain)),
            # Spread over spectra inside one test set. Reported for shape information
            # only: these spectra share one trained model, so this is NOT the dispersion
            # that a comparison between architectures may be based on.
            "snr_gain_db_sd_over_spectra_not_a_replicate_sd": float(np.std(gain, ddof=1)),
        }
    return per_level


# --------------------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------------------


def check_param_counts() -> dict:
    observed = {}
    for arch in ARCHITECTURES:
        model = build_model(arch)
        observed[arch] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    mismatches = {
        arch: {"expected": EXPECTED_PARAM_COUNTS[arch], "observed": observed[arch]}
        for arch in ARCHITECTURES
        if observed[arch] != EXPECTED_PARAM_COUNTS[arch]
    }
    return {"observed": observed, "mismatches": mismatches, "passed": not mismatches}


def _find_gui_hyperparams_file() -> Path | None:
    candidate = Path(__file__).resolve().parents[2] / "gui" / "panels" / "training_panel.py"
    return candidate if candidate.is_file() else None


def crosscheck_gui_hyperparams() -> dict:
    """Compare SUGGESTED_HYPERPARAMS against the GUI source without importing PyQt6."""
    path = _find_gui_hyperparams_file()
    if path is None:
        return {"status": "gui-source-not-found", "passed": None}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "OPTIMAL_HYPERPARAMS":
                    found = ast.literal_eval(node.value)
    if found is None:
        return {"status": "OPTIMAL_HYPERPARAMS-not-found", "passed": False}
    agrees = all(
        found.get(arch) == SUGGESTED_HYPERPARAMS.get(arch) for arch in ARCHITECTURES
    )
    return {
        "status": "compared",
        "gui_source": str(path.relative_to(Path(__file__).resolve().parents[2])),
        "gui_values": found,
        "passed": bool(agrees),
    }


def repeat_run_determinism(datasets, device, epochs_cap) -> dict:
    """Train the baseline architecture twice with identical inputs and compare.

    This measures run-to-run determinism on *this* device. It says nothing about
    reproducing the numbers on a different device or torch build; see
    `design.reproducibility`.
    """
    x_train = torch.tensor(datasets["train_noisy"], dtype=torch.float32)
    y_train = torch.tensor(datasets["train_clean"], dtype=torch.float32)
    scores = []
    for _attempt in range(2):
        model, _info = train_one(
            BASELINE_ARCH,
            SUGGESTED_HYPERPARAMS[BASELINE_ARCH],
            x_train,
            y_train,
            device,
            TORCH_SEED_BASE,
            epochs_cap=epochs_cap,
        )
        per_level = evaluate(model, datasets, device)
        scores.append([per_level[key]["snr_gain_db_mean"] for key in sorted(per_level)])
        del model
    differences = [abs(a - b) for a, b in zip(scores[0], scores[1])]
    return {
        "architecture": BASELINE_ARCH,
        "seed_index": 0,
        "first_run_snr_gain_db": scores[0],
        "second_run_snr_gain_db": scores[1],
        "max_abs_difference_db": max(differences),
        "bit_identical": all(difference == 0.0 for difference in differences),
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


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def aggregate(runs: list, condition: str) -> dict:
    """Mean +/- SD of the per-seed run means. The replicate is the training run."""
    out = {}
    for arch in ARCHITECTURES:
        out[arch] = {}
        for level in NOISE_LEVELS:
            values = [
                r["per_level"][str(level)]["snr_gain_db_mean"]
                for r in runs
                if r["architecture"] == arch and r["condition"] == condition
            ]
            inputs = [
                r["per_level"][str(level)]["input_snr_db_mean"]
                for r in runs
                if r["architecture"] == arch and r["condition"] == condition
            ]
            if not values:
                continue
            out[arch][str(level)] = {
                "n_seeds": len(values),
                "per_seed_snr_gain_db": values,
                "snr_gain_db_mean": float(np.mean(values)),
                "snr_gain_db_sd_across_seeds": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                ),
                "input_snr_db_mean": float(np.mean(inputs)),
            }
    return out


def _holm(pvalues: list) -> list:
    """Holm-Bonferroni step-down adjustment."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = (m - rank) * pvalues[index]
        running = max(running, value)
        adjusted[index] = float(min(1.0, running))
    return adjusted


def paired_comparison(runs: list, condition: str) -> dict:
    """Paired difference of each architecture against the baseline, within seed.

    Architectures in the same seed were trained on the same data and scored on the same
    test spectra, so the seed pairs them. n is the number of seeds, not the number of
    spectra.
    """
    result = {"baseline": BASELINE_ARCH, "by_noise_level": {}}
    for level in NOISE_LEVELS:

        def value_of(arch, seed_index, level=level):
            for r in runs:
                if (
                    r["architecture"] == arch
                    and r["condition"] == condition
                    and r["seed_index"] == seed_index
                ):
                    return r["per_level"][str(level)]["snr_gain_db_mean"]
            return None

        seed_indices = sorted(
            {r["seed_index"] for r in runs if r["condition"] == condition}
        )
        baseline_values = [value_of(BASELINE_ARCH, s) for s in seed_indices]
        if any(v is None for v in baseline_values):
            continue

        entries, pvalues, arch_order = {}, [], []
        for arch in ARCHITECTURES:
            if arch == BASELINE_ARCH:
                continue
            values = [value_of(arch, s) for s in seed_indices]
            if any(v is None for v in values):
                continue
            differences = np.array(values, dtype=float) - np.array(baseline_values, dtype=float)
            mean_difference = float(np.mean(differences))
            sd_difference = float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan")
            if len(differences) > 1 and sd_difference > 0:
                t_statistic, p_value = stats.ttest_rel(values, baseline_values)
                t_statistic, p_value = float(t_statistic), float(p_value)
                cohens_dz = mean_difference / sd_difference
            else:
                t_statistic, p_value, cohens_dz = float("nan"), float("nan"), float("nan")
            entries[arch] = {
                "n_seeds": len(differences),
                "per_seed_difference_db": [float(d) for d in differences],
                "mean_difference_db": mean_difference,
                "sd_difference_db": sd_difference,
                "paired_t": t_statistic,
                "p_value_raw": p_value,
                "cohens_dz": cohens_dz,
                "seeds_favouring_arch": int(np.sum(differences > 0)),
            }
            arch_order.append(arch)
            pvalues.append(p_value if np.isfinite(p_value) else 1.0)

        for arch, adjusted in zip(arch_order, _holm(pvalues)):
            entries[arch]["p_value_holm_adjusted"] = adjusted

        result["by_noise_level"][str(level)] = {
            "seed_indices": seed_indices,
            "baseline_per_seed_snr_gain_db": baseline_values,
            "comparisons": entries,
        }
    return result


# --------------------------------------------------------------------------------------
# Record assembly
# --------------------------------------------------------------------------------------


CLAIM_SCOPE = {
    "supports": (
        "Architecture X, at the hyperparameters this software suggests for it, trained "
        "on the synthetic training pool described under `design.data` and evaluated on "
        "independently drawn synthetic test spectra, scored the reported SNR gain."
    ),
    "does_not_support": [
        "that architecture X is intrinsically better than architecture Y",
        "that this ranking holds on measured spectra, on other peak sets, at other "
        "training-set sizes, or outside the noise levels listed here",
        "that a difference smaller than the reported across-seed spread is real",
        "any statement about physically meaningful quantities (peak areas, positions, "
        "widths) surviving denoising: those were not measured",
    ],
    "regime_note": (
        "Training and inference operate at the same signal-to-noise regime: every "
        "evaluated noise level is also present in the training pool. No claim is made "
        "about extrapolation to unseen noise levels."
    ),
    "denoised_output_is_a_model_estimate": (
        "The evaluated quantity is agreement with a known synthetic reference. A high "
        "SNR gain does not establish that structure in the output is real; the network "
        "can oversmooth, suppress weak features, and produce plausible structure that "
        "was not in the input."
    ),
}


def design_record(n_train_per_level: int, n_test_per_level: int, n_seeds: int, epochs_cap) -> dict:
    peak_set = get_peak_set(PEAK_SET)
    generator = make_generator(NOISE_LEVELS[0])
    return {
        "data": {
            "provenance": (
                "generated in-process by dnndenoiser.data.synthetic_generator."
                "SyntheticGenerator; no measured data, no external files"
            ),
            "peak_set_id": PEAK_SET,
            "peak_set_peaks_mu_fwhm_intensity": [
                [p.mu, p.fwhm, p.intensity] for p in peak_set.peaks
            ],
            "energy_range_ev": [float(generator.energy[0]), float(generator.energy[-1])],
            "energy_step_ev": float(generator.energy[1] - generator.energy[0]),
            "generator_config": dict(GENERATOR_CONFIG_KWARGS),
            "n_train_spectra_per_noise_level": n_train_per_level,
            "n_train_spectra_total_pooled": n_train_per_level * len(NOISE_LEVELS),
            "n_test_spectra_per_noise_level": n_test_per_level,
        },
        "split_rule": {
            "rule": (
                "train and test are independent draws from disjoint RNG streams, not a "
                "shuffle of one draw"
            ),
            "leakage_preventing_unit": "the independently generated spectrum",
            "train_stream_seed_formula": f"{TRAIN_STREAM_BASE} + 1000*seed_index + level_index",
            "test_stream_seed_formula": f"{TEST_STREAM_BASE} + 1000*seed_index + level_index",
            "verified": (
                "row-hash comparison for identity, plus a nearest-neighbour distance "
                "comparison for near-duplication; see self_checks.leakage"
            ),
            "shared_test_set": (
                "within one seed every architecture is trained on the same arrays and "
                "scored on the same test spectra, which is what makes the comparison paired"
            ),
        },
        "noise_model": {
            "config": "NoiseConfig(noise_type='poisson', poisson_level=L)",
            "levels": list(NOISE_LEVELS),
            "implementation_note": (
                "dnndenoiser.data.synthetic_generator.add_poisson_noise normalizes "
                "the spectrum by its maximum and scales it to lambda = (10000/L)**2 "
                "at the peak. This benchmark does not use the package default; it "
                "names its noise rule explicitly: the Gaussian approximation to the "
                "Poisson distribution (heteroscedastic, sd proportional to "
                "sqrt(intensity)), selected once from the peak lambda and applied "
                "to every bin when that lambda exceeds 20, and true Poisson counts "
                "otherwise -- so the approximation is used at L=100 and L=1000, and "
                "at L=10000 (lambda 1) Poisson counts are drawn. That rule was the "
                "package default until 2026-09-11; the default is now true Poisson "
                "with a per-bin admissibility test, and the benchmark keeps the "
                "earlier rule so that its published measurement stays exactly "
                "reproducible. This is the package's own noise model, not an "
                "independent one."
            ),
            "training_pool": (
                "one model per architecture per seed, trained on all noise levels pooled "
                "in equal proportion; evaluated on each level separately"
            ),
        },
        "seeds": {
            "n_seeds": n_seeds,
            "what_a_seed_varies": (
                "both the data draw (train and test RNG stream seeds) and the model init "
                "and batch order (torch.manual_seed)"
            ),
            "torch_seed_formula": f"{TORCH_SEED_BASE} + seed_index",
        },
        "metric": {
            "name": "SNR gain (dB)",
            "definition": (
                "signal power = mean(reference**2); noise power = mean((estimate - "
                "reference)**2); SNR_dB = 10*log10(signal power / noise power); "
                "gain = output SNR - input SNR"
            ),
            "reference": "the clean synthetic spectrum",
            "computed_over": "the full 256-point energy axis, background included",
            "aggregation": (
                "mean over the test spectra of one run gives that run's score; the "
                "reported mean and SD are over runs (seeds)"
            ),
            "independence_assumption": (
                "dispersion is reported across independent training runs (seeds), which is "
                "the unit that supports comparing architectures. Per-spectrum SNR values "
                "within one test set share a single trained model and are not independent "
                "replicates; they must not be used to shrink the error bar."
            ),
        },
        "training_recipe": {
            "optimizer": OPTIMIZER,
            "weight_decay": WEIGHT_DECAY,
            "loss": LOSS,
            "lr_scheduler": LR_SCHEDULER,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "target": "noise2clean (the clean synthetic spectrum)",
            "model_config": dict(MODEL_CONFIG),
            "epochs_cap_applied": epochs_cap,
            "schedule_caveat": (
                "the LR schedule is fixed at StepLR(step_size=10, gamma=0.1) while the "
                "suggested epoch counts differ (30 vs 50), so architectures assigned 50 "
                "epochs spend their extra epochs at a learning rate already reduced by "
                "1e-2 or more. This is what the software does; it is not a controlled "
                "budget."
            ),
        },
        "conditions": {
            "primary": {
                "name": "suggested-hyperparameters",
                "hyperparameters": SUGGESTED_HYPERPARAMS,
                "why": (
                    "this is what the software fills in for the user, so the result "
                    "describes the software's behaviour rather than the architectures "
                    "in the abstract"
                ),
                "supports": (
                    "architecture X at its suggested settings scored ... "
                    "and NOT architecture X is intrinsically better"
                ),
            },
            "secondary": {
                "name": "matched-budget",
                "hyperparameters": MATCHED_BUDGET_HYPERPARAMS,
                "why": (
                    "identical epochs, batch size and learning rate for every "
                    "architecture, so the primary result can be read against one "
                    "condition where the budget is not confounded with the architecture"
                ),
            },
        },
        "reproducibility": {
            "bit_exact_across_devices": False,
            "detail": (
                "Floating-point reduction order differs between CPU, MPS and CUDA "
                "backends, and MPS/CUDA kernels are not required to be run-to-run "
                "deterministic. Re-running this script on a different device, or on a "
                "different torch build, will not reproduce these numbers bit-for-bit and "
                "may move them by a few tenths of a dB. What is reproducible: the "
                "generated data (numpy Generator, platform independent), the parameter "
                "counts, the experimental design, and — the thing the record is for — the "
                "procedure by which the numbers can be regenerated and checked."
            ),
            "measured_here": (
                "self_checks.repeat_run_determinism trains the baseline architecture "
                "twice with identical inputs on this device and reports whether the "
                "scores came out bit-identical. That is a same-device statement only."
            ),
        },
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def run(args) -> dict:
    device = resolve_device(args.device)

    n_train = args.n_train_per_level
    n_test = args.n_test_per_level
    n_seeds = args.seeds
    epochs_cap = args.epochs_cap
    if args.quick:
        n_train = QUICK_OVERRIDES["n_train_per_level"]
        n_test = QUICK_OVERRIDES["n_test_per_level"]
        n_seeds = QUICK_OVERRIDES["n_seeds"]
        epochs_cap = QUICK_OVERRIDES["epochs_cap"]

    print(f"device: {device}")
    print(f"seeds: {n_seeds}  train/level: {n_train}  test/level: {n_test}")

    param_check = check_param_counts()
    if not param_check["passed"]:
        raise SystemExit(f"parameter-count self-check failed: {param_check['mismatches']}")
    gui_check = crosscheck_gui_hyperparams()
    if gui_check.get("passed") is False:
        raise SystemExit(f"GUI hyperparameter cross-check failed: {gui_check}")
    print(f"self-check: parameter counts ok; GUI hyperparameters {gui_check['status']}")

    conditions = [("suggested-hyperparameters", n_seeds)]
    if args.matched_budget:
        conditions.append(("matched-budget", args.matched_seeds or n_seeds))

    runs = []
    leakage = None
    determinism = None
    wall_start = time.perf_counter()

    for condition, condition_seeds in conditions:
        for seed_index in range(condition_seeds):
            datasets = build_seed_datasets(seed_index, n_train, n_test)
            if leakage is None:
                leakage = leakage_check(datasets)
                if not leakage["passed"]:
                    raise SystemExit(f"leakage check failed: {leakage}")
                print(f"self-check: leakage {leakage}")
                if not args.skip_determinism_check:
                    determinism = repeat_run_determinism(datasets, device, epochs_cap)
                    print(
                        f"self-check: repeat-run determinism on {device}: "
                        f"bit_identical={determinism['bit_identical']} "
                        f"max|diff|={determinism['max_abs_difference_db']:.3g} dB"
                    )

            x_train = torch.tensor(datasets["train_noisy"], dtype=torch.float32)
            y_train = torch.tensor(datasets["train_clean"], dtype=torch.float32)

            for arch in ARCHITECTURES:
                hyperparams = (
                    SUGGESTED_HYPERPARAMS[arch]
                    if condition == "suggested-hyperparameters"
                    else MATCHED_BUDGET_HYPERPARAMS
                )
                model, train_info = train_one(
                    arch,
                    hyperparams,
                    x_train,
                    y_train,
                    device,
                    TORCH_SEED_BASE + seed_index,
                    epochs_cap=epochs_cap,
                )
                per_level = evaluate(model, datasets, device)
                runs.append(
                    {
                        "condition": condition,
                        "architecture": arch,
                        "seed_index": seed_index,
                        "hyperparameters": dict(hyperparams),
                        "n_trainable_parameters": param_check["observed"][arch],
                        "per_level": per_level,
                        **train_info,
                    }
                )
                summary = "  ".join(
                    f"L{int(level)}:{per_level[str(level)]['snr_gain_db_mean']:+6.2f}"
                    for level in NOISE_LEVELS
                )
                print(
                    f"[{condition}] seed {seed_index} {arch:13s} "
                    f"{train_info['train_seconds']:6.1f}s  {summary}"
                )
                del model

    total_seconds = time.perf_counter() - wall_start

    record = {
        "record_version": RECORD_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "benchmarks/reference/reference_benchmark.py",
        "claim_scope": CLAIM_SCOPE,
        "environment": environment_record(device),
        "design": design_record(n_train, n_test, n_seeds, epochs_cap),
        "self_checks": {
            "parameter_counts": param_check,
            "gui_hyperparameter_crosscheck": gui_check,
            "leakage": leakage,
            "repeat_run_determinism": determinism,
        },
        "runs": runs,
        "aggregates": {
            condition: aggregate(runs, condition) for condition, _ in conditions
        },
        "paired_vs_baseline": {
            condition: paired_comparison(runs, condition) for condition, _ in conditions
        },
        "total_wall_clock_seconds": total_seconds,
        "quick_mode": bool(args.quick),
    }
    return record


def main(argv=None) -> int:
    default_output = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="directory for the JSON record (default: results/ next to this script)",
    )
    parser.add_argument(
        "--output-name",
        default="reference_benchmark.json",
        help="file name of the JSON record inside --output-dir",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--n-train-per-level", type=int, default=N_TRAIN_PER_LEVEL)
    parser.add_argument("--n-test-per-level", type=int, default=N_TEST_PER_LEVEL)
    parser.add_argument(
        "--epochs-cap", type=int, default=None, help="cap every architecture's epoch count"
    )
    parser.add_argument(
        "--matched-budget",
        action="store_true",
        help="also run the secondary condition with identical epochs/batch/LR for all",
    )
    parser.add_argument(
        "--matched-seeds",
        type=int,
        default=None,
        help="seeds for the secondary condition (default: same as --seeds)",
    )
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    parser.add_argument(
        "--skip-determinism-check",
        action="store_true",
        help="skip the duplicate baseline training run used to measure run-to-run determinism",
    )
    args = parser.parse_args(argv)

    record = run(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / args.output_name
    destination.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\nwrote {destination}")
    print(f"total wall clock: {record['total_wall_clock_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
