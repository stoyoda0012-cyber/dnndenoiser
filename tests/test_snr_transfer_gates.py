"""Every P2-B self-check accepts the correct input and rejects a named wrong one.

A check that has only ever passed has not been shown to check anything (AGENTS.md
§8.1). Each test here pairs the input the check is meant to accept with one it exists
to refuse, and requires the refusal to name the reason. The threshold table and the
rule for a failed positive control are tested the same way, against the values the
preregistration registers.
"""
from __future__ import annotations

import importlib.util
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


def test_exact_poisson_refuses_frames_drawn_at_another_level(p2b, sample):
    frames = p2b.draw_frames(sample, 45.0, 64, 1)
    with pytest.raises(p2b.SelfCheckFailure, match="not whole counts"):
        p2b.check_exact_poisson(frames, sample, 20.0)


def test_realised_flux_accepts_the_level_and_refuses_another(p2b, sample):
    frames = p2b.draw_frames(sample, 9.0, 2000, 2)
    p2b.check_realised_flux(frames, sample, 9.0)
    with pytest.raises(p2b.SelfCheckFailure, match="mean count at the maximum"):
        p2b.check_realised_flux(frames * (20.0 / 9.0), sample, 9.0)


def test_equal_exposure_accepts_the_registered_counts_and_refuses_equal_frames(p2b):
    registered = {lam: p2b.n_frames(lam) for lam in p2b.LAMBDAS}
    assert registered == {4.0: 12500, 9.0: 5556, 20.0: 2500, 45.0: 1111, 100.0: 500}
    p2b.check_equal_exposure(registered, 1)
    with pytest.raises(p2b.SelfCheckFailure, match="lambda \\* N"):
        p2b.check_equal_exposure({lam: 2500 for lam in p2b.LAMBDAS}, 1)


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


def test_same_test_arrays_refuses_a_model_evaluated_on_another_level(p2b):
    hashes = {4.0: "a", 9.0: "b"}
    p2b.check_same_test_arrays({("moving_average", 4.0, 9.0): "b"}, hashes)
    with pytest.raises(p2b.SelfCheckFailure, match="different test array"):
        p2b.check_same_test_arrays({("noise2clean", 4.0, 9.0): "a"}, hashes)


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


def test_a_resumed_run_refuses_a_seed_file_from_another_commit(p2b, tmp_path):
    stamp = {"code_commit": "a" * 40, "working_tree_clean": True, "scripts": {}}
    path = tmp_path / "seed_00.json"
    path.write_text('{"stamp": ' + __import__("json").dumps({**stamp, "code_commit": "b" * 40})
                    + ', "result": {}}', encoding="utf-8")
    with pytest.raises(p2b.SelfCheckFailure, match="same commit from a clean tree"):
        p2b.load_resumable(path, stamp)
    path.write_text('{"stamp": ' + __import__("json").dumps(stamp) + ', "result": {"x": 1}}',
                    encoding="utf-8")
    assert p2b.load_resumable(path, stamp) == {"stamp": stamp, "result": {"x": 1}}
