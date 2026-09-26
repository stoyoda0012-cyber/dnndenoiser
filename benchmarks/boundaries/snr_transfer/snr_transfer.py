"""P2-B: training and inference at different signal-to-noise ratios.

Registered design: `docs/preregistration/P2B-snr-transfer.md`. Nothing in this script
may depart from that document; where it must, the document is revised first and its
revision log says why.

What is measured: a 5 x 5 grid of training flux x inference flux, on synthetic
spectra with exact Poisson noise, for two training methods evaluated on the same test
frames:

  levels      lambda = 4, 9, 20, 45, 100 (expected count at the spectrum maximum)
  budget      equal total exposure: N_frames = round(50000 / lambda) per level
  primary     self-supervised moving average, W = 1, the library's recipe
  baseline    noise2clean on 2304 independently drawn spectra per level, P2-A's recipe
  seeds       20; a seed draws the sample, every frame, the pool and the model init
  metrics     M1 SNR gain in dB; M2 = M1(off-diagonal) - M1(diagonal at same inference)

Each seed's results are written to their own file as the seed finishes. A resumed run
reuses a seed file only if it was written at the same commit, from a clean tree, by the
same script and shared module; otherwise it refuses.

This script refuses to write a record if any self-check fails.

Usage:

    python benchmarks/boundaries/snr_transfer/snr_transfer.py
    python benchmarks/boundaries/snr_transfer/snr_transfer.py --quick \
        --output-dir "$(mktemp -d)"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import boundary_common as common  # noqa: E402
from boundary_common import SelfCheckFailure  # noqa: E402

from dnndenoiser.data.synthetic_generator import (  # noqa: E402
    GeneratorConfig,
    NoiseConfig,
    SyntheticGenerator,
    add_noise,
    get_peak_set,
)
from dnndenoiser.models.network import DenoisingNetwork  # noqa: E402
from dnndenoiser.training.selfsupervised import (  # noqa: E402
    moving_average_targets,
    train_selfsupervised,
)

RECORD_VERSION = "1"
PREREGISTRATION = "docs/preregistration/P2B-snr-transfer.md"
REPO_ROOT = common.REPO_ROOT
HERE = Path(__file__).resolve().parent
SCRIPTS = [Path(__file__).resolve(), Path(common.__file__).resolve()]

# --------------------------------------------------------------------------------------
# Registered design
# --------------------------------------------------------------------------------------

PEAK_SET_ID = "C1s_adventitious"
ENERGY_RANGE = (277.8, 295.5)
N_ENERGY_POINTS = 256
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

LAMBDAS = (4.0, 9.0, 20.0, 45.0, 100.0)
TOTAL_EXPOSURE = 50_000.0              # lambda * N_frames, equal at every level
R1_LEVELS = (20.0, 45.0, 100.0)        # positive control; lambda = 4, 9 descriptive
N_TEST_PER_LEVEL = 512
N_SEEDS = 20
W = 1

MODEL_CONFIG = {"num_features": 256, "num_hidden_units": 100, "encoder_output_dim": 64}
ARCH = "ResNet-FCNN"
MA_RECIPE = {  # the library's train_selfsupervised; recorded, not passed -- it takes none
    "optimizer": "Adam", "lr": 1e-3, "weight_decay": 1e-9,
    "lr_scheduler": "StepLR(step_size=25, gamma=0.5)", "loss": "HuberLoss(delta=1.0)",
    "gradient_clip_norm": 4.0, "epochs": 50, "batch_size": 32,
    "normalisation": "element-global min-max over the training stack, applied at "
                     "inference with the training constants and inverted on the output",
}
N2C_POOL = 2304
N2C_RECIPE = {  # P2-A's train_one, unchanged
    "optimizer": "Adam", "lr": 1e-3, "weight_decay": 1e-9,
    "lr_scheduler": "StepLR(step_size=10, gamma=0.1)", "loss": "HuberLoss(delta=1.0)",
    "gradient_clip_norm": 4.0, "epochs": 50, "batch_size": 16,
}

# Disjoint RNG streams. Each base is far enough from the next that no seed, level or
# sample index can carry one stream into another.
SAMPLE_STREAM_BASE = 1_000_000
TRAIN_FRAME_STREAM_BASE = 2_000_000
TEST_FRAME_STREAM_BASE = 3_000_000
N2C_POOL_STREAM_BASE = 4_000_000
TORCH_SEED_BASE = 5_000_000

R1_K = 19                 # P2-A's positive-control rule
R2_MARGIN_DB = -1.0
ALPHA = 0.05

QUICK_OVERRIDES = {"n_seeds": 2, "frame_divisor": 50, "n_test_per_level": 32,
                   "n2c_pool": 64, "epochs": 2}

P2A_RECORD = REPO_ROOT / "benchmarks/boundaries/position_shift/results/position_shift_boundary.json"


def poisson_level(lam: float) -> float:
    """The generator's `poisson_level` whose expected count at the maximum is `lam`.

    `add_poisson_noise` scales the spectrum so its maximum has expected count
    (10000 / level)**2; solving for the level gives 10000 / sqrt(lam).
    """
    return 10000.0 / float(np.sqrt(lam))


def n_frames(lam: float, divisor: int = 1) -> int:
    return max(2, int(round(TOTAL_EXPOSURE / lam / divisor)))


def noise_config(lam: float) -> NoiseConfig:
    """Exact Poisson at every level: no Gaussian approximation anywhere."""
    return NoiseConfig(noise_type="poisson", poisson_level=poisson_level(lam),
                       use_gaussian_approx=False)


def make_generator(lam: float) -> SyntheticGenerator:
    return SyntheticGenerator(
        peak_set=get_peak_set(PEAK_SET_ID),
        noise_config=noise_config(lam),
        config=GeneratorConfig(**GENERATOR_CONFIG_KWARGS),
    )


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def draw_sample(seed_index: int) -> np.ndarray:
    """One clean sample: a draw of the peak set's per-peak position, width and intensity.

    The noisy half of the generator's output is discarded; only the clean spectrum is a
    sample. Which level's generator draws it does not matter -- the clean spectrum does
    not depend on the noise config -- and the lowest-flux generator is used by rule.
    """
    clean, _noisy, _energy, _meta = make_generator(LAMBDAS[0]).generate_batch(
        1, seed=SAMPLE_STREAM_BASE + seed_index)
    return clean[0].astype(np.float64)


def draw_frames(clean: np.ndarray, lam: float, n: int, stream: int) -> np.ndarray:
    """n independent Poisson realisations of one clean spectrum at flux `lam`.

    Through `add_noise`, the function the generator itself uses, so training frames, test
    frames and the noise2clean pool are all drawn by one function.
    """
    rng = np.random.default_rng(stream)
    config = noise_config(lam)
    return np.stack([add_noise(clean, config, rng) for _ in range(n)]).astype(np.float32)


def train_frame_stream(seed_index: int, level_index: int) -> int:
    return TRAIN_FRAME_STREAM_BASE + seed_index * 100 + level_index


def test_frame_stream(seed_index: int, level_index: int) -> int:
    return TEST_FRAME_STREAM_BASE + seed_index * 100 + level_index


def n2c_pool(seed_index: int, level_index: int, n: int) -> dict:
    clean, noisy, _energy, _meta = make_generator(LAMBDAS[level_index]).generate_batch(
        n, seed=N2C_POOL_STREAM_BASE + seed_index * 100 + level_index)
    return {"clean": clean.astype(np.float32), "noisy": noisy.astype(np.float32)}


def _row_hashes(array: np.ndarray) -> set:
    return {hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in array}


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# --------------------------------------------------------------------------------------
# Self-checks. Each raises SelfCheckFailure; each is shown to reject a named wrong input
# in tests/test_snr_transfer_gates.py.
# --------------------------------------------------------------------------------------


def check_exact_poisson(frames: np.ndarray, clean: np.ndarray, lam: float) -> dict:
    """Self-check 1a: every value is a whole number of counts at this level's scale.

    Exact Poisson draws are integers before `add_poisson_noise` scales them back by
    data_max / lambda. A Gaussian approximation, or a scale that is not an integer
    multiple of the true one, leaves non-integers. Tolerance: float32 rounding of counts
    up to ~10**3. This alone does NOT identify the level: frames drawn at a lambda that
    divides the declared one also land on whole counts. Self-check 2 does that.
    """
    counts = frames.astype(np.float64) * lam / float(np.max(np.maximum(clean, 0.0)))
    worst = float(np.max(np.abs(counts - np.round(counts))))
    if worst > 1e-3:
        raise SelfCheckFailure(
            f"frames at lambda = {lam} are not whole counts (worst residual {worst:.3g}); "
            "the noise is not exact Poisson at this level")
    return {"lambda": lam, "worst_integer_residual": worst}


def check_exact_poisson_pool(pool: dict, lam: float) -> dict:
    """Self-check 1b: the same for the noise2clean pool, each row at its own scale."""
    clean = pool["clean"].astype(np.float64)
    scale = lam / np.max(np.maximum(clean, 0.0), axis=1, keepdims=True)
    counts = pool["noisy"].astype(np.float64) * scale
    worst = float(np.max(np.abs(counts - np.round(counts))))
    if worst > 1e-3:
        raise SelfCheckFailure(
            f"the noise2clean pool at lambda = {lam} is not whole counts (worst residual "
            f"{worst:.3g}); its noise is not exact Poisson at this level")
    return {"lambda": lam, "worst_integer_residual": worst}


LEVEL_TOLERANCE = 1.3   # registered levels are about 2.2x apart


def _lambda_from_variance(noisy: np.ndarray, clean: np.ndarray) -> float:
    """lambda estimated from the analytic Poisson variance, not from the generator.

    A bin with clean value c, in a spectrum of maximum m, has expected count lambda*c/m,
    returned as count*m/lambda, so its variance is m*c/lambda. Pooled over every bin and
    every row: lambda = sum(m*c) / sum((noisy - clean)**2).
    """
    clean = np.maximum(clean.astype(np.float64), 0.0)
    m = np.max(clean, axis=-1, keepdims=True)
    residual = noisy.astype(np.float64) - clean
    return float(np.sum(np.broadcast_to(m * clean, residual.shape)) / np.sum(residual**2))


def check_noise_level(noisy: np.ndarray, clean: np.ndarray, lam: float) -> dict:
    """Self-check 2: the level is the declared one, from the noise variance.

    The mean cannot tell levels apart -- the generator returns every level at the clean
    spectrum's amplitude -- but the variance falls as 1/lambda. The estimate must lie
    within a factor LEVEL_TOLERANCE of the declared lambda; every registered level is
    about 2.2x from its neighbour, so any mix-up is refused. `clean` is one spectrum
    (broadcast over frames) or one per row (the noise2clean pool).
    """
    estimate = _lambda_from_variance(noisy, np.broadcast_to(clean, noisy.shape))
    ratio = estimate / lam
    if not (1.0 / LEVEL_TOLERANCE <= ratio <= LEVEL_TOLERANCE):
        raise SelfCheckFailure(
            f"noise variance gives lambda = {estimate:.2f}, declared {lam}; the frames "
            "were not drawn at the declared level")
    return {"lambda": lam, "lambda_from_variance": estimate}


def check_equal_exposure(train_frames: dict, divisor: int) -> dict:
    """Self-check 3: lambda * N is the registered total at every level, to one frame --
    N being the number of rows actually drawn, not the number planned."""
    counts = {lam: int(len(frames)) for lam, frames in train_frames.items()}
    for lam, n in counts.items():
        if abs(lam * n - TOTAL_EXPOSURE / divisor) > lam:
            raise SelfCheckFailure(
                f"lambda = {lam} has {n} frames; lambda * N = {lam * n}, not "
                f"{TOTAL_EXPOSURE / divisor} to within one frame")
    return {"frames_per_level": {str(k): v for k, v in counts.items()}}


def check_targets_are_neighbours(normalised: np.ndarray, targets: np.ndarray) -> dict:
    """Self-check 4: at W = 1 every target row is one of its frame's adjacent frames.

    Independent of `moving_average_targets`: it asks only that row i of the targets be
    bit-identical to row i-1 or row i+1 of the normalised stack, which is what "the
    temporally nearest other frame" means for frames in acquisition order.
    """
    n = len(normalised)
    stack = normalised.astype(np.float64)
    for i in range(n):
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n]
        if not any(np.array_equal(targets[i], stack[j]) for j in neighbours):
            raise SelfCheckFailure(
                f"target row {i} is not frame {neighbours}; W = 1 targets must be the "
                "temporally nearest other frame")
    return {"rows_checked": n}


def check_no_leakage(train_frames: dict, test_frames: dict, pool_clean: dict,
                     sample: np.ndarray) -> dict:
    """Self-check 5: no test frame is a training frame; no pool spectrum is the sample."""
    train_rows = set().union(*(_row_hashes(f) for f in train_frames.values()))
    collisions = sum(len(_row_hashes(f) & train_rows) for f in test_frames.values())
    if collisions:
        raise SelfCheckFailure(f"{collisions} test frame(s) are byte-identical to training frames")
    sample_hash = _array_hash(sample.astype(np.float32))
    in_pool = [str(lam) for lam, c in pool_clean.items() if sample_hash in _row_hashes(c)]
    if in_pool:
        raise SelfCheckFailure(f"the sample's clean spectrum is in the noise2clean pool at {in_pool}")
    return {"test_frames_in_training": 0, "sample_in_pool": []}


def expected_model_input(test: np.ndarray, norm) -> np.ndarray:
    """What a model should receive for a test array: the array, or for the moving
    average the array under the training stack's min-max, as float32 -- computed here,
    apart from `evaluate`, so that self-check 6 has something to compare against."""
    if norm is None:
        return np.asarray(test, dtype=np.float32)
    return ((test.astype(np.float64) - norm["min"]) / (norm["max"] - norm["min"])).astype(np.float32)


def check_same_test_arrays(evaluated: dict, test_frames: dict, norms: dict) -> dict:
    """Self-check 6: every model saw, at every inference level, exactly the input it should.

    `evaluated` holds, per (method, train level, inference level), the hash of the array
    actually passed to the network. It is compared with the hash of the input rebuilt
    from the drawn test array and that model's normalisation. All 50 cells must be there.
    """
    required = {(m, t, i) for m in ("moving_average", "noise2clean")
                for t in LAMBDAS for i in LAMBDAS}
    missing = required - set(evaluated)
    if missing:
        raise SelfCheckFailure(f"{len(missing)} cell(s) were never evaluated: {sorted(missing)[:3]}")
    for (method, train, infer), digest in evaluated.items():
        norm = norms[train] if method == "moving_average" else None
        if digest != _array_hash(expected_model_input(test_frames[infer], norm)):
            raise SelfCheckFailure(
                f"{method} trained at {train} was evaluated at {infer} on an input that is "
                "not that level's test array under the model's normalisation")
    return {"cells_checked": len(evaluated)}


def check_parameter_count(model: DenoisingNetwork, expected: int) -> dict:
    """Self-check 7: the architecture is the one the reference benchmark counts.

    Run on every trained model. A different architecture with the same count would pass;
    the count is the reference benchmark's recorded value, not this script's."""
    count = sum(p.numel() for p in model.parameters())
    if count != expected:
        raise SelfCheckFailure(f"{ARCH} has {count} parameters, expected {expected}")
    return {"parameters": count}


def reference_parameter_count() -> int:
    record = json.loads((REPO_ROOT / "benchmarks/reference/results/reference_benchmark.json")
                        .read_text(encoding="utf-8"))
    return int(record["self_checks"]["parameter_counts"]["observed"][ARCH])


# --------------------------------------------------------------------------------------
# Training and inference
# --------------------------------------------------------------------------------------


def build_model() -> DenoisingNetwork:
    return DenoisingNetwork(layer_type=ARCH, **MODEL_CONFIG)


def train_moving_average(frames: np.ndarray, device: str, torch_seed: int, epochs: int):
    g_min, g_max = float(frames.min()), float(frames.max())
    if g_max <= g_min:
        raise SelfCheckFailure("a training stack is constant; min-max normalisation is undefined")
    normalised = (frames.astype(np.float64) - g_min) / (g_max - g_min)
    frame_indices = np.arange(len(frames))
    targets = moving_average_targets(normalised, frame_indices, W)
    target_check = check_targets_are_neighbours(normalised, targets)
    start = time.perf_counter()
    model = train_selfsupervised(normalised.astype(np.float32), targets.astype(np.float32),
                                 epochs=epochs, batch_size=MA_RECIPE["batch_size"],
                                 device=device, seed=torch_seed)
    common.sync(device)
    return model, {"min": g_min, "max": g_max}, {
        "train_seconds": time.perf_counter() - start, "n_frames": int(len(frames)),
        "targets": target_check}


def train_noise2clean(pool: dict, device: str, torch_seed: int, epochs: int):
    """P2-A's `train_one`, unchanged in recipe."""
    torch.manual_seed(torch_seed)
    model = build_model().to(device)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(torch_seed + 1)
    loader = DataLoader(TensorDataset(torch.tensor(pool["noisy"]), torch.tensor(pool["clean"])),
                        batch_size=N2C_RECIPE["batch_size"], shuffle=True,
                        generator=loader_generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=N2C_RECIPE["lr"],
                                 weight_decay=N2C_RECIPE["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.HuberLoss(delta=1.0)
    model.train()
    start = time.perf_counter()
    for _epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            prediction, _peak = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), N2C_RECIPE["gradient_clip_norm"])
            optimizer.step()
        scheduler.step()
    common.sync(device)
    model.eval()
    return model, {"train_seconds": time.perf_counter() - start, "n_train": int(len(pool["clean"]))}


@torch.no_grad()
def denoise(model, inputs: np.ndarray, device: str, batch_size: int = 512) -> np.ndarray:
    model.eval()
    tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs = [model(tensor[s:s + batch_size].to(device))[0].cpu().numpy()
               for s in range(0, len(tensor), batch_size)]
    return np.concatenate(outputs, axis=0).astype(np.float64)


def evaluate(model, test: np.ndarray, clean: np.ndarray, device: str, norm=None) -> tuple:
    """Mean SNR gain over the test frames, and the hash of the array the network received."""
    inputs = test.astype(np.float64)
    if norm is not None:
        inputs = (inputs - norm["min"]) / (norm["max"] - norm["min"])
    inputs = inputs.astype(np.float32)
    output = denoise(model, inputs, device)
    if norm is not None:
        output = output * (norm["max"] - norm["min"]) + norm["min"]
    reference = np.broadcast_to(clean, test.shape)
    gain = common.snr_db(output, reference) - common.snr_db(test, reference)
    return float(np.mean(gain)), _array_hash(inputs)


# --------------------------------------------------------------------------------------
# One seed
# --------------------------------------------------------------------------------------


def run_seed(seed_index: int, device: str, settings: dict, expected_params: int) -> dict:
    sample = draw_sample(seed_index)
    frame_counts = {lam: n_frames(lam, settings["frame_divisor"]) for lam in LAMBDAS}

    train_frames = {lam: draw_frames(sample, lam, frame_counts[lam], train_frame_stream(seed_index, i))
                    for i, lam in enumerate(LAMBDAS)}
    test_frames = {lam: draw_frames(sample, lam, settings["n_test_per_level"],
                                    test_frame_stream(seed_index, i))
                   for i, lam in enumerate(LAMBDAS)}
    pools = {lam: n2c_pool(seed_index, i, settings["n2c_pool"]) for i, lam in enumerate(LAMBDAS)}

    checks = {
        "1a_exact_poisson_frames": [check_exact_poisson(f, sample, lam)
                                    for frames in (train_frames, test_frames)
                                    for lam, f in frames.items()],
        "1b_exact_poisson_pool": [check_exact_poisson_pool(pools[lam], lam) for lam in LAMBDAS],
        "2_noise_level": (
            [check_noise_level(f, sample, lam) for frames in (train_frames, test_frames)
             for lam, f in frames.items()]
            + [check_noise_level(pools[lam]["noisy"], pools[lam]["clean"], lam) for lam in LAMBDAS]),
        "3_equal_exposure": check_equal_exposure(train_frames, settings["frame_divisor"]),
        "5_no_leakage": check_no_leakage(
            train_frames, test_frames, {lam: p["clean"] for lam, p in pools.items()}, sample),
        "7_parameter_count": [],
    }
    input_snr = {str(lam): float(np.mean(common.snr_db(t, np.broadcast_to(sample, t.shape))))
                 for lam, t in test_frames.items()}

    gains = {"moving_average": {}, "noise2clean": {}}
    training = {"moving_average": {}, "noise2clean": {}}
    evaluated, norms = {}, {}
    for i, train_lam in enumerate(LAMBDAS):
        model, norm, info = train_moving_average(
            train_frames[train_lam], device, TORCH_SEED_BASE + seed_index * 100 + i,
            settings["epochs"])
        checks["7_parameter_count"].append(check_parameter_count(model, expected_params))
        norms[train_lam] = norm
        training["moving_average"][str(train_lam)] = {**info, "normalisation": norm}
        for infer_lam in LAMBDAS:
            gain, digest = evaluate(model, test_frames[infer_lam], sample, device, norm)
            gains["moving_average"][cell(train_lam, infer_lam)] = gain
            evaluated[("moving_average", train_lam, infer_lam)] = digest

        model, info = train_noise2clean(pools[train_lam], device,
                                        TORCH_SEED_BASE + seed_index * 100 + 50 + i,
                                        settings["epochs"])
        checks["7_parameter_count"].append(check_parameter_count(model, expected_params))
        training["noise2clean"][str(train_lam)] = info
        for infer_lam in LAMBDAS:
            gain, digest = evaluate(model, test_frames[infer_lam], sample, device)
            gains["noise2clean"][cell(train_lam, infer_lam)] = gain
            evaluated[("noise2clean", train_lam, infer_lam)] = digest

    checks["6_same_test_arrays"] = check_same_test_arrays(evaluated, test_frames, norms)
    return {"seed_index": seed_index, "gains_db": gains, "input_snr_db": input_snr,
            "training": training, "self_checks": checks,
            "environment": common.environment_record(device),
            "sample_sha256": _array_hash(sample)}


# --------------------------------------------------------------------------------------
# Per-seed files and resuming
# --------------------------------------------------------------------------------------


def seed_stamp(device: str, settings: dict) -> dict:
    """What a seed file must match for a resumed run to reuse it: the commit, a clean
    tree, the scripts, the device and library versions, and the settings. A run that
    resumes is therefore one environment, and the record says which it was."""
    return {"code_commit": common.git("rev-parse", "HEAD"),
            "working_tree_clean": common.working_tree_status() == "",
            "scripts": {Path(s).name: common.sha256(s) for s in SCRIPTS},
            "environment": common.environment_record(device),
            "settings": settings}


def load_resumable(path: Path, stamp: dict, seed_index: int) -> dict | None:
    """A saved seed, if it matches the stamp, is the expected seed and is complete."""
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("stamp") != stamp or not stamp["working_tree_clean"]:
        raise SelfCheckFailure(
            f"{path.name} was written at a different commit, tree state, script, device, "
            "library version or setting; a run resumes only from files written at the "
            "same commit from a clean tree in the same environment. Move the seed files "
            "away to start again.")
    result = saved.get("result", {})
    if result.get("seed_index") != seed_index:
        raise SelfCheckFailure(f"{path.name} holds seed {result.get('seed_index')}, not {seed_index}")
    cells = {cell(t, i) for t in LAMBDAS for i in LAMBDAS}
    for method in ("moving_average", "noise2clean"):
        if set(result.get("gains_db", {}).get(method, {})) != cells:
            raise SelfCheckFailure(f"{path.name} does not hold all 25 cells for {method}")
    return saved


def prepare_output(out_dir: Path) -> Path:
    """Create the output directory and prove it writable BEFORE any seed runs, so a
    run of hours cannot end on a failed save."""
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = out_dir / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return out_dir / "snr_transfer.json"


# --------------------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------------------


def cell(train, infer) -> str:
    return f"{float(train)}->{float(infer)}"


def m1(seeds: list, method: str, train, infer) -> np.ndarray:
    return np.array([s["gains_db"][method][cell(train, infer)] for s in seeds])


def m2(seeds: list, method: str, train, infer) -> np.ndarray:
    return m1(seeds, method, train, infer) - m1(seeds, method, infer, infer)


def _family(tests: dict, n: int) -> dict:
    """Evaluate a family of per-cell sign rules at the Holm-consistent threshold."""
    if not tests:
        return {"evaluated": False, "cells": {}}
    k = common.threshold_for_family(len(tests), n, ALPHA)
    cells = {}
    for name, (values, positive) in tests.items():
        cells[name] = common.sign_test(values, positive, k)
    adjusted = common.holm([c["one_sided_binomial_p"] for c in cells.values()])
    for c, p in zip(cells.values(), adjusted):
        c["holm_adjusted_p"] = p
    return {"evaluated": True, "family_size": len(tests), "k_required": k,
            "passed": all(c["passed"] for c in cells.values()), "cells": cells}


def evaluate_predictions(seeds: list, method: str = "moving_average") -> dict:
    """The registered predictions for the moving average; for noise2clean, the same
    statistics, recorded descriptively."""
    n = len(seeds)
    r1_cells = {str(lam): common.sign_test(m1(seeds, method, lam, lam), True, R1_K)
                for lam in R1_LEVELS}
    for c, p in zip(r1_cells.values(),
                    common.holm([c["one_sided_binomial_p"] for c in r1_cells.values()])):
        c["holm_adjusted_p"] = p
    failed = [float(lam) for lam, c in r1_cells.items() if not c["passed"]]
    r1 = {"k_required": R1_K, "cells": r1_cells, "failed_levels": failed,
          "passed": not failed,
          "descriptive_diagonals": {str(lam): m1(seeds, method, lam, lam).tolist()
                                    for lam in LAMBDAS if lam not in R1_LEVELS}}

    if len(failed) >= 2:
        return {"R1": r1, "R2": {"evaluated": False, "why": "R1 failed at two or more levels"},
                "R3": {"evaluated": False, "why": "R1 failed at two or more levels"},
                "R4": {"evaluated": False, "why": "R1 failed at two or more levels"}}

    removed = set(failed)
    above = [(t, i) for t in LAMBDAS for i in LAMBDAS if t > i]
    below = [(t, i) for t in LAMBDAS for i in LAMBDAS if t < i]
    r2_tests = {cell(t, i): (m2(seeds, method, t, i) - R2_MARGIN_DB, True)
                for t, i in above if i not in removed}
    r3_tests = {cell(t, i): (m2(seeds, method, t, i), False) for t, i in below if i not in removed}
    r4_tests = {f"{a}|{b}": (-m2(seeds, method, a, b) + m2(seeds, method, b, a), True)
                for a, b in below if a not in removed and b not in removed}
    r2 = _family(r2_tests, n)
    r2["margin_db"] = R2_MARGIN_DB
    r2["m1_beside_each_cell"] = {
        cell(t, i): {"cell_m1_mean": float(np.mean(m1(seeds, method, t, i))),
                     "diagonal_m1_mean": float(np.mean(m1(seeds, method, i, i)))}
        for t, i in above}
    reduced = sorted(removed)
    r2["cells_made_descriptive_by_R1"] = [cell(t, i) for t, i in above if i in removed]
    r3 = _family(r3_tests, n)
    r3["cells_made_descriptive_by_R1"] = [cell(t, i) for t, i in below if i in removed]
    r4 = _family(r4_tests, n)
    r4["pairs_made_descriptive_by_R1"] = [f"{a}|{b}" for a, b in below
                                          if a in removed or b in removed]
    return {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "levels_removed_by_R1": reduced}


def aggregates(seeds: list) -> dict:
    out = {}
    for method in ("moving_average", "noise2clean"):
        out[method] = {}
        for t in LAMBDAS:
            for i in LAMBDAS:
                v1 = m1(seeds, method, t, i)
                entry = {"m1_mean": float(np.mean(v1)), "m1_sd": float(np.std(v1, ddof=1))
                         if len(v1) > 1 else 0.0, "m1_per_seed": v1.tolist()}
                if t != i:
                    v2 = m2(seeds, method, t, i)
                    entry.update({"m2_mean": float(np.mean(v2)),
                                  "m2_sd": float(np.std(v2, ddof=1)) if len(v2) > 1 else 0.0})
                out[method][cell(t, i)] = entry
    return out


def p2a_beside_noise2clean(seeds: list) -> dict:
    """Descriptive only: no tolerance, no condition (registered)."""
    if not P2A_RECORD.exists():
        return {"available": False}
    record = json.loads(P2A_RECORD.read_text(encoding="utf-8"))
    p2a = record["aggregates"]["A_narrow_2304"]["1000.0"]["+0.00"]["snr_gain_db_mean"]
    here = float(np.mean(m1(seeds, "noise2clean", 100.0, 100.0)))
    return {"available": True, "p2a_arm_A_level_1000_delta_0_gain_db": p2a,
            "noise2clean_lambda_100_diagonal_gain_db": here,
            "differences_stated": [
                "noise: exact Poisson here, the Gaussian approximation in P2-A",
                "training: one level here, three in P2-A",
                "test set: 512 noisy frames of ONE clean spectrum per seed here, "
                "independently generated spectra per seed in P2-A, so a seed mean averages "
                "over different things and the spread across seeds means different things",
                "this is not a re-measurement of P2-A under the same conditions"]}


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------


def run(args) -> dict:
    quick = args.quick
    settings = {"n_seeds": N_SEEDS, "frame_divisor": 1, "n_test_per_level": N_TEST_PER_LEVEL,
                "n2c_pool": N2C_POOL, "epochs": MA_RECIPE["epochs"]}
    if quick:
        settings.update(QUICK_OVERRIDES)
        if args.output_dir is None:
            raise SystemExit("--quick requires --output-dir outside results/")
    if not quick and args.device == "auto":
        raise SystemExit("a full run needs an explicit --device (registered: mps)")
    device = common.resolve_device(args.device)
    provenance = common.provenance_record(quick=quick, preregistration=PREREGISTRATION,
                                          scripts=SCRIPTS)
    out_dir = Path(args.output_dir) if args.output_dir else HERE / "results"
    record_path = prepare_output(out_dir)
    # Per-seed files live in a git-ignored directory, so that writing them does not make
    # the tree dirty and a resumed run can still verify it is clean.
    seed_dir = out_dir / "seeds" if quick else HERE / ".partial"
    seed_dir.mkdir(parents=True, exist_ok=True)
    stamp = seed_stamp(device, settings)
    expected_params = reference_parameter_count()

    seeds, resumed = [], []
    start = time.perf_counter()
    for seed_index in range(settings["n_seeds"]):
        path = seed_dir / f"seed_{seed_index:02d}.json"
        saved = None if quick else load_resumable(path, stamp, seed_index)
        if saved is not None:
            seeds.append(saved["result"])
            resumed.append(seed_index)
            continue
        result = run_seed(seed_index, device, settings, expected_params)
        path.write_text(json.dumps({"stamp": stamp, "result": result}, indent=1),
                        encoding="utf-8")
        seeds.append(result)
        print(f"seed {seed_index} done ({time.perf_counter() - start:.0f} s)", flush=True)

    record = {
        "record_version": RECORD_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "quick_mode": quick,
        "provenance": provenance,
        "environment": common.environment_record(device),
        "design": {
            "preregistration": PREREGISTRATION, "lambdas": list(LAMBDAS),
            "poisson_levels": [poisson_level(lam) for lam in LAMBDAS],
            "total_exposure": TOTAL_EXPOSURE, "frames_per_level":
                {str(lam): n_frames(lam, settings["frame_divisor"]) for lam in LAMBDAS},
            "W": W, "architecture": ARCH, "model_config": MODEL_CONFIG,
            "moving_average_recipe": MA_RECIPE, "noise2clean_recipe": N2C_RECIPE,
            "noise2clean_pool": settings["n2c_pool"], "n_test_per_level": settings["n_test_per_level"],
            "n_seeds": settings["n_seeds"], "epochs": settings["epochs"],
            "generator_config": GENERATOR_CONFIG_KWARGS, "peak_set_id": PEAK_SET_ID,
            "noise": "exact Poisson (use_gaussian_approx=False) at every level",
            "confound": "a training level's S/N, its number of training frames and its "
                        "number of optimiser updates are not separable in this design",
        },
        "resumed_seeds": resumed,
        "seeds": seeds,
        "aggregates": aggregates(seeds),
        "predictions": evaluate_predictions(seeds),
        "noise2clean_descriptive": evaluate_predictions(seeds, "noise2clean"),
        "p2a_beside_noise2clean": p2a_beside_noise2clean(seeds),
        "total_wall_clock_seconds": time.perf_counter() - start,
    }
    return record, record_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke run: not a record")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto",
                        help="a full run requires it explicitly; the registration uses mps")
    args = parser.parse_args(argv)
    try:
        record, path = run(args)
    except SelfCheckFailure as failure:
        print(f"SELF-CHECK FAILED -- no record written:\n{failure}", file=sys.stderr)
        return 2
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"record written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
