"""Every P2-B self-check accepts the correct input and rejects a named wrong one.

A check that has only ever passed has not been shown to check anything (AGENTS.md
§8.1). Each test here pairs the input the check is meant to accept with one it exists
to refuse, and requires the refusal to name the reason. The threshold table and the
rule for a failed positive control are tested the same way, against the values the
preregistration registers.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarks" / "boundaries" / "snr_transfer" / "snr_transfer.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.is_file(),
    reason="benchmarks/ is a source-checkout tree, not part of the distribution",
)


@pytest.fixture(scope="module")
def p2b():
    sys.path.insert(0, str(SCRIPT.parents[1]))
    spec = importlib.util.spec_from_file_location("_p2b_snr_transfer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sample(p2b):
    return p2b.draw_sample(0)


def test_exact_poisson_accepts_the_library_draw_and_refuses_a_gaussian_one(p2b, sample):
    frames = p2b.draw_frames(sample, 20.0, 64, 1)
    p2b.check_exact_poisson(frames, sample, 20.0)
    rng = np.random.default_rng(0)
    rate = 20.0 * np.maximum(sample, 0) / sample.max()
    gaussian = (rate + rng.standard_normal((64, rate.size)) * np.sqrt(rate)) / 20.0 * sample.max()
    with pytest.raises(p2b.SelfCheckFailure, match="not whole counts"):
        p2b.check_exact_poisson(gaussian.astype(np.float32), sample, 20.0)


def test_exact_poisson_refuses_frames_at_a_level_that_does_not_divide_the_declared_one(p2b, sample):
    frames = p2b.draw_frames(sample, 45.0, 64, 1)
    with pytest.raises(p2b.SelfCheckFailure, match="not whole counts"):
        p2b.check_exact_poisson(frames, sample, 20.0)


def test_exact_poisson_for_the_pool_refuses_a_gaussian_pool(p2b):
    pool = p2b.n2c_pool(0, 2, 32)
    p2b.check_exact_poisson_pool(pool, 20.0)
    rng = np.random.default_rng(0)
    noisy = pool["clean"] + rng.standard_normal(pool["clean"].shape) * 0.05
    with pytest.raises(p2b.SelfCheckFailure, match="pool at lambda = 20.0 is not whole counts"):
        p2b.check_exact_poisson_pool({"clean": pool["clean"], "noisy": noisy}, 20.0)


LEVEL_MIXUPS = [(real, declared) for real in (4.0, 9.0, 20.0, 45.0, 100.0)
                for declared in (4.0, 9.0, 20.0, 45.0, 100.0) if real != declared]


def test_noise_level_accepts_every_registered_level(p2b, sample):
    for i, lam in enumerate(p2b.LAMBDAS):
        p2b.check_noise_level(p2b.draw_frames(sample, lam, 200, 10 + i), sample, lam)


@pytest.mark.parametrize("real,declared", LEVEL_MIXUPS)
def test_noise_level_refuses_every_mix_up_of_registered_levels(p2b, sample, real, declared):
    """The audit's case: frames drawn at 4 declared as 20 passed the old mean-based check,
    and the integer check, because the generator returns every level at one amplitude."""
    frames = p2b.draw_frames(sample, real, 200, 7)
    with pytest.raises(p2b.SelfCheckFailure, match="not drawn at the declared level"):
        p2b.check_noise_level(frames, sample, declared)


def test_noise_level_refuses_a_pool_drawn_at_another_level(p2b):
    pool = p2b.n2c_pool(0, 0, 64)          # drawn at lambda = 4
    p2b.check_noise_level(pool["noisy"], pool["clean"], 4.0)
    with pytest.raises(p2b.SelfCheckFailure, match="not drawn at the declared level"):
        p2b.check_noise_level(pool["noisy"], pool["clean"], 20.0)


def test_equal_exposure_accepts_the_registered_counts_and_refuses_equal_frames(p2b, sample):
    registered = {lam: p2b.n_frames(lam) for lam in p2b.LAMBDAS}
    assert registered == {4.0: 12500, 9.0: 5556, 20.0: 2500, 45.0: 1111, 100.0: 500}
    drawn = {lam: np.zeros((n, 1)) for lam, n in registered.items()}
    p2b.check_equal_exposure(drawn, 1)
    with pytest.raises(p2b.SelfCheckFailure, match="lambda \\* N"):
        p2b.check_equal_exposure({lam: np.zeros((2500, 1)) for lam in p2b.LAMBDAS}, 1)


def test_equal_exposure_counts_the_rows_drawn_not_the_rows_planned(p2b):
    """The audit's case: the old check saw only the planned counts."""
    drawn = {lam: np.zeros((p2b.n_frames(lam), 1)) for lam in p2b.LAMBDAS}
    drawn[4.0] = drawn[4.0][:-5]
    with pytest.raises(p2b.SelfCheckFailure, match="lambda = 4.0 has 12495 frames"):
        p2b.check_equal_exposure(drawn, 1)


def test_targets_check_accepts_w1_and_refuses_w2(p2b, sample):
    frames = p2b.draw_frames(sample, 20.0, 12, 3).astype(np.float64)
    indices = np.arange(len(frames))
    p2b.check_targets_are_neighbours(frames, p2b.moving_average_targets(frames, indices, 1))
    with pytest.raises(p2b.SelfCheckFailure, match="temporally nearest other frame"):
        p2b.check_targets_are_neighbours(frames, p2b.moving_average_targets(frames, indices, 2))


def test_targets_check_refuses_a_frame_as_its_own_target(p2b, sample):
    frames = p2b.draw_frames(sample, 20.0, 12, 3).astype(np.float64)
    with pytest.raises(p2b.SelfCheckFailure, match="temporally nearest other frame"):
        p2b.check_targets_are_neighbours(frames, frames.copy())


def test_leakage_check_refuses_a_training_frame_in_the_test_set(p2b, sample):
    train = {20.0: p2b.draw_frames(sample, 20.0, 16, 4)}
    test = {20.0: p2b.draw_frames(sample, 20.0, 16, 5)}
    pool = {20.0: p2b.n2c_pool(0, 2, 8)["clean"]}
    p2b.check_no_leakage(train, test, pool, sample)
    leaked = {20.0: np.concatenate([test[20.0][:-1], train[20.0][:1]])}
    with pytest.raises(p2b.SelfCheckFailure, match="byte-identical to training frames"):
        p2b.check_no_leakage(train, leaked, pool, sample)


def test_leakage_check_refuses_the_sample_in_the_pool(p2b, sample):
    train = {20.0: p2b.draw_frames(sample, 20.0, 16, 4)}
    test = {20.0: p2b.draw_frames(sample, 20.0, 16, 5)}
    pool = {20.0: np.concatenate([p2b.n2c_pool(0, 2, 8)["clean"], sample[None].astype(np.float32)])}
    with pytest.raises(p2b.SelfCheckFailure, match="clean spectrum is in the noise2clean pool"):
        p2b.check_no_leakage(train, test, pool, sample)


def _evaluated_cells(p2b, sample, shift_one=None):
    """Run `evaluate` for every cell with a stand-in model; optionally corrupt one input."""
    test = {lam: p2b.draw_frames(sample, lam, 8, 20 + i) for i, lam in enumerate(p2b.LAMBDAS)}
    norms = {lam: {"min": -0.1, "max": 1.3 + lam / 1000} for lam in p2b.LAMBDAS}
    model = p2b.build_model()
    evaluated = {}
    for method in ("moving_average", "noise2clean"):
        for t in p2b.LAMBDAS:
            for i in p2b.LAMBDAS:
                array = test[i]
                if shift_one == (method, t, i):
                    array = np.roll(array, 5, axis=1)
                norm = norms[t] if method == "moving_average" else None
                _gain, digest = p2b.evaluate(model, array, sample, "cpu", norm)
                evaluated[(method, t, i)] = digest
    return evaluated, test, norms


def test_same_test_arrays_accepts_what_evaluate_actually_fed(p2b, sample):
    evaluated, test, norms = _evaluated_cells(p2b, sample)
    p2b.check_same_test_arrays(evaluated, test, norms)


def test_same_test_arrays_refuses_an_input_shifted_by_five_bins(p2b, sample):
    """The audit's case: the old check hashed the array handed in, not what the network saw."""
    evaluated, test, norms = _evaluated_cells(p2b, sample, ("moving_average", 20.0, 45.0))
    with pytest.raises(p2b.SelfCheckFailure, match="not that level's test array"):
        p2b.check_same_test_arrays(evaluated, test, norms)


def test_same_test_arrays_refuses_another_levels_normalisation(p2b, sample):
    evaluated, test, norms = _evaluated_cells(p2b, sample)
    swapped = {**norms, 9.0: norms[45.0]}
    with pytest.raises(p2b.SelfCheckFailure, match="not that level's test array"):
        p2b.check_same_test_arrays(evaluated, test, swapped)


def test_same_test_arrays_refuses_a_missing_cell_and_an_empty_set(p2b, sample):
    evaluated, test, norms = _evaluated_cells(p2b, sample)
    del evaluated[("noise2clean", 4.0, 100.0)]
    with pytest.raises(p2b.SelfCheckFailure, match="never evaluated"):
        p2b.check_same_test_arrays(evaluated, test, norms)
    with pytest.raises(p2b.SelfCheckFailure, match="never evaluated"):
        p2b.check_same_test_arrays({}, test, norms)


def test_parameter_count_refuses_another_architecture(p2b):
    expected = p2b.reference_parameter_count()
    p2b.check_parameter_count(p2b.build_model(), expected)
    other = p2b.DenoisingNetwork(layer_type="FCNN", **p2b.MODEL_CONFIG)
    with pytest.raises(p2b.SelfCheckFailure, match="parameters, expected"):
        p2b.check_parameter_count(other, expected)


def test_the_threshold_table_is_the_registered_one(p2b):
    """The preregistration's table: 15 for 1-2 tests, 16 for 3-8, 17 for 9-10."""
    got = {m: p2b.common.threshold_for_family(m, 20) for m in range(1, 11)}
    assert got == {1: 15, 2: 15, 3: 16, 4: 16, 5: 16, 6: 16, 7: 16, 8: 16, 9: 17, 10: 17}


def _fake_seeds(p2b, diag_positive: dict, n=20, above=0.0, below=-2.0):
    """Seeds whose moving-average gains are set per cell, for the prediction logic."""
    seeds = []
    for s in range(n):
        gains = {}
        for t in p2b.LAMBDAS:
            for i in p2b.LAMBDAS:
                if t == i:
                    g = 5.0 if diag_positive.get(t, True) or s >= 10 else -1.0
                elif t > i:
                    g = 5.0 + above
                else:
                    g = 5.0 + below - (i - t) / 100.0
                gains[p2b.cell(t, i)] = g
        seeds.append({"gains_db": {"moving_average": gains, "noise2clean": gains}})
    return seeds


def test_when_r1_holds_every_family_is_evaluated_in_full(p2b):
    p = p2b.evaluate_predictions(_fake_seeds(p2b, {}))
    assert p["R1"]["passed"]
    assert (p["R2"]["family_size"], p["R3"]["family_size"], p["R4"]["family_size"]) == (10, 10, 10)
    assert p["R2"]["k_required"] == 17


def test_when_r1_fails_at_one_level_its_cells_become_descriptive(p2b):
    p = p2b.evaluate_predictions(_fake_seeds(p2b, {45.0: False}))
    assert p["R1"]["failed_levels"] == [45.0]
    assert p["R2"]["cells_made_descriptive_by_R1"] == ["100.0->45.0"]
    assert p["R2"]["family_size"] == 9 and p["R2"]["k_required"] == 17
    assert set(p["R3"]["cells_made_descriptive_by_R1"]) == {"4.0->45.0", "9.0->45.0", "20.0->45.0"}
    assert p["R3"]["family_size"] == 7 and p["R3"]["k_required"] == 16
    assert p["R4"]["family_size"] == 6


def test_when_r1_fails_at_two_levels_r2_to_r4_are_not_evaluated(p2b):
    p = p2b.evaluate_predictions(_fake_seeds(p2b, {45.0: False, 100.0: False}))
    assert all(p[r]["evaluated"] is False for r in ("R2", "R3", "R4"))


def _stamp(**over):
    base = {"code_commit": "a" * 40, "working_tree_clean": True, "scripts": {"x.py": "1"},
            "environment": {"device_requested_resolved_to": "mps", "torch": "2.9.1"},
            "settings": {"n_seeds": 20}}
    return {**base, **over}


def _complete_result(p2b, seed_index):
    cells = {p2b.cell(t, i): 0.0 for t in p2b.LAMBDAS for i in p2b.LAMBDAS}
    return {"seed_index": seed_index, "gains_db": {"moving_average": cells, "noise2clean": cells}}


def _write(path, stamp, result):
    path.write_text(json.dumps({"stamp": stamp, "result": result}), encoding="utf-8")


def test_a_resumed_run_accepts_a_matching_complete_seed(p2b, tmp_path):
    path = tmp_path / "seed_03.json"
    _write(path, _stamp(), _complete_result(p2b, 3))
    assert p2b.load_resumable(path, _stamp(), 3)["result"]["seed_index"] == 3


@pytest.mark.parametrize("field,value", [
    ("code_commit", "b" * 40),
    ("environment", {"device_requested_resolved_to": "cpu", "torch": "2.9.1"}),
    ("environment", {"device_requested_resolved_to": "mps", "torch": "2.11.0"}),
    ("settings", {"n_seeds": 2}),
    ("scripts", {"x.py": "2"}),
])
def test_a_resumed_run_refuses_another_commit_environment_or_setting(p2b, tmp_path, field, value):
    """The audit's case: a seed from MPS reused under CPU would have been recorded as CPU."""
    path = tmp_path / "seed_00.json"
    _write(path, _stamp(**{field: value}), _complete_result(p2b, 0))
    with pytest.raises(p2b.SelfCheckFailure, match="same environment"):
        p2b.load_resumable(path, _stamp(), 0)


def test_a_resumed_run_refuses_the_wrong_seed_and_an_incomplete_one(p2b, tmp_path):
    path = tmp_path / "seed_05.json"
    _write(path, _stamp(), _complete_result(p2b, 4))
    with pytest.raises(p2b.SelfCheckFailure, match="holds seed 4, not 5"):
        p2b.load_resumable(path, _stamp(), 5)
    partial = _complete_result(p2b, 5)
    partial["gains_db"]["noise2clean"] = {}
    _write(path, _stamp(), partial)
    with pytest.raises(p2b.SelfCheckFailure, match="all 25 cells for noise2clean"):
        p2b.load_resumable(path, _stamp(), 5)


def test_the_output_directory_is_made_and_proven_writable_before_any_seed(p2b, tmp_path):
    """The audit's case: a full run created only .partial, so the final save would fail."""
    target = tmp_path / "not" / "yet" / "there"
    assert p2b.prepare_output(target) == target / "snr_transfer.json"
    assert target.is_dir() and not any(target.iterdir())


def test_the_output_directory_refuses_an_unwritable_place(p2b, tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        p2b.prepare_output(blocker / "sub")


def test_r1_reports_holm_adjusted_p(p2b):
    p = p2b.evaluate_predictions(_fake_seeds(p2b, {}))
    assert all("holm_adjusted_p" in c for c in p["R1"]["cells"].values())


def test_same_test_arrays_refuses_a_call_that_passes_the_network_something_else(p2b, sample,
                                                                                monkeypatch):
    """The second audit's case: the array handed to `denoise` altered at the call, while the
    local `inputs` stays right. Only a hash taken at the network's call boundary sees it."""
    original = p2b.denoise
    monkeypatch.setattr(p2b, "denoise",
                        lambda model, inputs, device: original(model, np.roll(inputs, 5, axis=1), device))
    evaluated, test, norms = _evaluated_cells(p2b, sample)
    with pytest.raises(p2b.SelfCheckFailure, match="not that level's test array"):
        p2b.check_same_test_arrays(evaluated, test, norms)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "1.0", None, True])
def test_a_resumed_seed_with_a_non_finite_or_non_numeric_gain_is_refused(p2b, tmp_path, bad):
    result = _complete_result(p2b, 0)
    result["gains_db"]["moving_average"] = {**result["gains_db"]["moving_average"],
                                            p2b.cell(4.0, 9.0): bad}
    path = tmp_path / "seed_00.json"
    _write(path, _stamp(), result)
    with pytest.raises(p2b.SelfCheckFailure, match="not a finite number"):
        p2b.load_resumable(path, _stamp(), 0)


def test_noise2clean_statistics_keep_every_cell_when_r1_fails(p2b):
    """The second audit's case: reusing the prediction logic dropped noise2clean's cells."""
    seeds = _fake_seeds(p2b, {45.0: False, 100.0: False})
    d = p2b.descriptive_statistics(seeds, "noise2clean")
    assert d["descriptive_only"] is True
    assert (len(d["R1_cells"]), len(d["R2_cells"]), len(d["R3_cells"]), len(d["R4_pairs"])) == (3, 10, 10, 10)
