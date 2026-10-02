"""The provenance manifest, phase B (docs/design/PROVENANCE_MANIFEST.md §5 and §8,
groups 6-9): established relationships, names and caveats, synthetic identities.

Every rejection is pinned to its reason with every other field valid, and has a positive
counterpart. The synthetic-identity classification is checked against the generator itself
(do the arrays change?), never against the classification; one identifier is frozen from a
literal canonical JSON string hashed here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import h5py
import numpy as np
import pytest

from dnndenoiser import evaluation as ev
from dnndenoiser import provenance as prov
from dnndenoiser import reference as ref
from dnndenoiser.data.identity import CLASSIFICATION, identities
from dnndenoiser.data.synthetic_generator import (GeneratorConfig, NoiseConfig,
                                                  SyntheticGenerator)
from tests.test_evaluate_reference import (TRUTH, arrays, estimate, evaluate, forbidden_in,
                                           refuse, run, write)
from tests.test_provenance_manifest import stack_file, supervised_file, train

DB_KEYS = ("snr_input_mean", "snr_output_mean", "snr_gain_mean", "snr_gain_std")


def infer(monkeypatch, data, model, out):
    run(monkeypatch, "infer", "-d", str(data), "-m", str(model), "-o", str(out), "--device", "cpu")
    return out


def generate(monkeypatch, path, *extra, n=8, seed=42):
    run(monkeypatch, "generate", "-o", str(path), "-n", str(n), "--n-energy", "32",
        "--peak-set", "C1s_single", "--seed", str(seed), *extra)
    return path


def ctx(m):
    return m["evaluation_context"]


@pytest.fixture(scope="module")
def sup(tmp_path_factory):
    """A noise2clean model trained on a file with acquisition acq-T and frames 0..7."""
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("sup")
    try:
        data = supervised_file(d / "train.h5", acquisition_id="acq-T", frame_index=np.arange(8))
        train(mp, data, d / "m.pt")
    finally:
        mp.undo()
    return data, d / "m.pt"


def evaluated_file(path, noisy, clean, *, declaration=TRUTH, acquisition_id=None,
                   frame_index=None, model_output=None):
    """An evaluate input carrying the model records of ``model_output`` (an infer output)."""
    write(path, noisy, noisy, clean, declaration=declaration, acquisition_id=acquisition_id,
          frame_index=frame_index)
    if model_output is not None:
        with h5py.File(model_output) as src, h5py.File(path, "a") as dst:
            dst.create_dataset(prov.OUTPUT_MANIFEST, data=src[prov.OUTPUT_MANIFEST][()],
                               dtype=h5py.string_dtype("utf-8"))
            for key in (prov.OUTPUT_DIGEST, prov.OUTPUT_BODY_DIGEST):
                dst["denoised"].attrs[key] = src["denoised"].attrs[key]
    return path


@pytest.fixture
def records(monkeypatch, tmp_path, sup):
    data, model = sup
    return infer(monkeypatch, data, model, tmp_path / "records.h5")


# ---------------------------------------------------------------------------------- 6


def test_the_training_file_itself_is_not_held_out_and_renamed(monkeypatch, capsys, tmp_path, sup,
                                                             records):
    capsys.readouterr()
    m = evaluate(monkeypatch, tmp_path, records)
    printed = capsys.readouterr().out
    c = ctx(m)
    assert (c["held_out_status"], c["held_out_basis"], c["rows_in_training"]) == (
        "not_held_out", "established", 8)
    assert (m["held_out_status"], m["rows_in_training"]) == ("not_held_out", 8)
    expected = {"n_spectra", "mse_in_mean", "mse_out_mean", "mse_difference",
                "relative_mse_change_aggregate_pct", "per_spectrum_excluded_zero_input_mse",
                "mean_relative_mse_change_per_spectrum_pct", "zero_reference_power_count",
                "floor_active_input_count", "floor_active_output_count",
                *("training_fit_" + k for k in DB_KEYS), "held_out_status", "rows_in_training",
                "evaluation_context"}
    assert set(m) == expected
    assert "=== Fit to the training data — Error against the synthetic truth ===" in printed
    assert ev.training_fit_caveat(8, 8) in c["caveats"]


def test_status_entries_follow_the_rename():
    m = {"snr_gain_mean": None, "mse_in_mean": 1.0,
         "status": {"snr_gain_mean": "overflow", "mse_in_mean": "x"}}
    assert ev.rename_for_training_fit(m) == {
        "training_fit_snr_gain_mean": None, "mse_in_mean": 1.0,
        "status": {"training_fit_snr_gain_mean": "overflow", "mse_in_mean": "x"}}
    assert set(ev.TRAINING_FIT_KEYS) == {*DB_KEYS, "agreement_db_input_mean",
                                         "agreement_db_output_mean", "agreement_db_change_mean",
                                         "agreement_db_change_std"}


def test_a_float64_frame_stack_matches_through_the_rename_and_the_cast(monkeypatch, tmp_path):
    data = stack_file(tmp_path / "s.h5", e=256, dtype=np.float64)
    train(monkeypatch, data, tmp_path / "m.pt", method="moving-average")
    out = infer(monkeypatch, data, tmp_path / "m.pt", tmp_path / "o.h5")
    with h5py.File(out, "a") as f:            # an estimate reference, so evaluate can run
        mean = np.repeat(f["noisy"][:].mean(0, keepdims=True), f["noisy"].shape[0], 0)
        f.create_dataset("clean", data=mean.astype(np.float32))
        ref.write_declaration(f["clean"], estimate(acquisition="other"))
        for name in ("noisy", "denoised"):
            f[name].attrs["intensity_units"] = "counts"
    c = ctx(evaluate(monkeypatch, tmp_path, out))
    assert (c["held_out_status"], c["rows_in_training"]) == ("not_held_out", 12)


def test_a_resampled_output_matches_through_its_input_digest(monkeypatch, tmp_path):
    data = stack_file(tmp_path / "s.h5", e=64)
    train(monkeypatch, data, tmp_path / "m.pt", method="moving-average")
    out = infer(monkeypatch, data, tmp_path / "m.pt", tmp_path / "o.h5")
    with h5py.File(out, "a") as f:
        assert f["noisy"].shape[-1] == 256
        f.create_dataset("clean", data=f["noisy"][:])
        ref.write_declaration(f["clean"], estimate(acquisition="other"))
        for name in ("noisy", "denoised"):
            f[name].attrs["intensity_units"] = "counts"
    c = ctx(evaluate(monkeypatch, tmp_path, out))
    assert (c["held_out_status"], c["rows_in_training"]) == ("not_held_out", 12)
    with h5py.File(out, "a") as f:            # the same output without the recorded digest
        del f["noisy"].attrs[prov.INPUT_ARRAY_DIGEST]
    assert ctx(evaluate(monkeypatch, tmp_path, out))["held_out_status"] == "unknown"


@pytest.mark.parametrize("acq, frames, status, rows, unidentified", [
    ("acq-T", np.arange(5, 13), "not_held_out", 3, False),        # rule 2, 3 of 8
    ("acq-T", np.arange(7, 15), "not_held_out", 1, False),        # one row of many
    ("acq-T", np.arange(20, 28), "disjoint_by_identifiers", None, False),
    ("acq-X", np.arange(0, 8), "disjoint_by_identifiers", None, False),
    ("acq-T", None, "unknown", None, True),                       # split without frame_index
    (None, np.arange(0, 8), "unknown", None, False),
], ids=["intersecting", "one-row", "disjoint-frames", "other-acquisition",
        "same-acquisition-unidentified", "no-acquisition"])
def test_the_rules_on_identifiers(monkeypatch, tmp_path, records, acq, frames, status, rows,
                                  unidentified):
    n, _d, c = arrays(n=8, e=32, seed=11)       # other arrays: no digest match
    path = evaluated_file(tmp_path / "e.h5", n, c, acquisition_id=acq, frame_index=frames,
                          model_output=records)
    m = evaluate(monkeypatch, tmp_path, path)
    k = ctx(m)
    assert (k["held_out_status"], k["rows_in_training"], k["same_acquisition_rows_unidentified"]) == (
        status, rows, unidentified)
    assert k["held_out_basis"] == ("default" if status == "unknown" else "established")
    renamed = status == "not_held_out"
    assert ("training_fit_snr_gain_mean" in m) is renamed and ("snr_gain_mean" in m) is not renamed


def test_equal_digests_win_over_different_identifiers(monkeypatch, tmp_path, sup, records):
    with h5py.File(records, "a") as f:
        f["noisy"].attrs["acquisition_id"] = "acq-X"
        del f["noisy"].attrs[prov.INPUT_ARRAY_DIGEST]
    assert ctx(evaluate(monkeypatch, tmp_path, records))["held_out_status"] == "not_held_out"


def test_legacy_output_is_refused_on_training_data(monkeypatch, capsys, tmp_path):
    n, _d, c = arrays(n=8, e=32)
    data = write(tmp_path / "u.h5", n, None, c, units=None)          # undeclared reference
    train(monkeypatch, data, tmp_path / "m.pt")
    out = infer(monkeypatch, data, tmp_path / "m.pt", tmp_path / "o.h5")
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(out), "--legacy-output",
                 "--assert-alignment", "units")
    assert "--legacy-output reproduces historical names" in err
    n2, _d2, c2 = arrays(n=8, e=32, seed=5)                       # other rows, no identifiers
    other = infer(monkeypatch, write(tmp_path / "v.h5", n2, None, c2, units=None),
                  tmp_path / "m.pt", tmp_path / "o2.h5")
    m = evaluate(monkeypatch, tmp_path, other, "--legacy-output", "--assert-alignment", "units")
    assert "snr_gain_mean" in m and "held_out_status" not in m            # unknown: allowed


def test_an_estimate_reference_on_training_rows_renames_the_agreement_keys(monkeypatch, tmp_path,
                                                                          records):
    with h5py.File(records, "a") as f:
        ref_arr = f["clean"][:]
        del f["clean"]
        f.create_dataset("clean", data=ref_arr)
        ref.write_declaration(f["clean"], estimate(acquisition="other"))
    m = evaluate(monkeypatch, tmp_path, records)
    assert {"training_fit_" + k for k in ("agreement_db_input_mean", "agreement_db_output_mean",
                                          "agreement_db_change_mean", "agreement_db_change_std")
            } <= set(m)
    keys = [k for k in m if k not in ("evaluation_context", "status")]
    assert forbidden_in(keys, []) == []


# ---------------------------------------------------------------------------------- 7


@pytest.mark.parametrize("method", ["noise2clean", "noise2noise"])
def test_the_training_clean_as_reference_was_used_in_development(monkeypatch, capsys, tmp_path,
                                                                 sup, method):
    data, model = sup
    if method == "noise2noise":
        model = tmp_path / "n2n.pt"
        train(monkeypatch, data, model, method="noise2noise")
    n, _d, _c = arrays(n=8, e=32, seed=21)                      # new noisy rows
    test = write(tmp_path / "t.h5", n, None, None)
    out = infer(monkeypatch, test, model, tmp_path / "o.h5")
    m = evaluate(monkeypatch, tmp_path, out, "--clean", str(data), "--assert-alignment", "rows")
    c = ctx(m)
    assert c["relationship"]["used_in_model_development"] == "yes"
    assert c["relationship"]["used_in_model_development_basis"] == "established"
    assert c["reference_rows_in_training"] == 8
    assert ev.REFERENCE_TRAINED_CAVEAT in c["caveats"]                 # truth case included
    assert "snr_gain_mean" in m                                        # new noisy rows
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(out), "--clean", str(data),
                 "--assert-alignment", "rows", "--used-in-model-development", "no_declared")
    assert "contradicts the model's provenance" in err


def test_no_declared_is_accepted_when_nothing_is_established(monkeypatch, tmp_path, sup):
    data, model = sup
    n, _d, c = arrays(n=8, e=32, seed=21)
    test = write(tmp_path / "t.h5", n, None, c, declaration=TRUTH)
    out = infer(monkeypatch, test, model, tmp_path / "o.h5")
    m = evaluate(monkeypatch, tmp_path, out, "--used-in-model-development", "no_declared")
    assert ctx(m)["relationship"]["used_in_model_development"] == "no_declared"
    assert ctx(m)["relationship"]["used_in_model_development_basis"] == "declared"


def test_same_seed_another_noise_level_recognises_the_training_targets(monkeypatch, tmp_path):
    train_file = generate(monkeypatch, tmp_path / "train.h5", "--poisson-level", "100", n=16)
    test_file = generate(monkeypatch, tmp_path / "test.h5", "--poisson-level", "300", n=8)
    train(monkeypatch, train_file, tmp_path / "m.pt")
    out = infer(monkeypatch, test_file, tmp_path / "m.pt", tmp_path / "o.h5")
    m = evaluate(monkeypatch, tmp_path, out)
    c = ctx(m)
    assert c["held_out_status"] == "disjoint_by_identifiers" and c["rows_in_training"] is None
    assert c["relationship"]["used_in_model_development"] == "yes"
    assert c["reference_rows_in_training"] == 8
    assert "snr_gain_mean" in m


def test_a_frame_mean_over_the_training_acquisition_shares_its_source(monkeypatch, tmp_path):
    data = stack_file(tmp_path / "s.h5", e=256)
    with h5py.File(data, "a") as f:
        f["frames"].attrs["acquisition_id"] = "acq-S"
        f["frames"].attrs["intensity_units"] = "counts"
    train(monkeypatch, data, tmp_path / "m.pt", method="moving-average")
    out = infer(monkeypatch, data, tmp_path / "m.pt", tmp_path / "o.h5")
    with h5py.File(out, "a") as f:
        mean = np.repeat(f["noisy"][:].mean(0, keepdims=True), f["noisy"].shape[0], 0)
        f.create_dataset("clean", data=mean.astype(np.float32))
        ref.write_declaration(f["clean"], estimate(acquisition="acq-S"))
    c = ctx(evaluate(monkeypatch, tmp_path, out))
    assert c["shares_source_with_training_data"] == "yes"
    assert ev.REFERENCE_SHARES_CAVEAT in c["caveats"]
    assert c["relationship"]["used_in_model_development"] == "unknown"   # not the target


MANIFEST_TD = {"acquisition_id": "acq-S", "frame_index_runs": [[0, 4], [10, 12]]}


@pytest.mark.parametrize("frames, runs, expected", [
    ("all", [[0, 4]], "yes"),
    ("unrecorded", [[0, 4]], "unknown"),
    ([3, 30], [[0, 4]], "yes"),
    ([5, 9], [[0, 4], [10, 12]], "unknown"),
    ({"range": [6, 11]}, [[0, 4], [10, 12]], "yes"),
    ({"range": [5, 9]}, [[0, 4], [10, 12]], "unknown"),
    ({"range": [5, 13]}, [[7, 8]], "yes"),                       # spans a run, ends outside
    ([3], None, "unknown"),
    ("all", None, "yes"),
], ids=["all", "unrecorded", "list-hit", "list-miss", "range-hit", "range-miss", "range-spans",
        "runs-null-list", "runs-null-all"])
def test_shares_source_rules(frames, runs, expected):
    manifest = {"training_data": {"acquisition_id": "acq-S", "frame_index_runs": runs}}
    assert prov.shares_source(manifest, estimate(acquisition="acq-S", frames=frames)) == expected
    assert prov.shares_source(manifest, estimate(acquisition="acq-Q", frames=frames)) == "unknown"


# ---------------------------------------------------------------------------------- 8


def test_a_prefix_with_the_same_seed_is_counted_as_training_rows(monkeypatch, tmp_path):
    big = generate(monkeypatch, tmp_path / "big.h5", n=30)
    small = generate(monkeypatch, tmp_path / "small.h5", n=10)
    with h5py.File(big) as b, h5py.File(small) as s:
        assert b["noisy"].attrs["acquisition_id"] == s["noisy"].attrs["acquisition_id"]
        assert b["clean"].attrs["signal_identity"] == s["clean"].attrs["signal_identity"]
        np.testing.assert_array_equal(b["noisy"][:10], s["noisy"][:])
        np.testing.assert_array_equal(s["frame_index"][:], np.arange(10))
    train(monkeypatch, big, tmp_path / "m.pt")
    out = infer(monkeypatch, small, tmp_path / "m.pt", tmp_path / "o.h5")
    c = ctx(evaluate(monkeypatch, tmp_path, out))
    assert (c["held_out_status"], c["rows_in_training"]) == ("not_held_out", 10)


def test_an_angle_resolved_subset_counts_first_axis_rows(monkeypatch, tmp_path):
    big = generate(monkeypatch, tmp_path / "big.h5", "--n-angles", "3", n=12)
    small = generate(monkeypatch, tmp_path / "small.h5", "--n-angles", "3", n=4)
    train(monkeypatch, big, tmp_path / "m.pt")
    out = infer(monkeypatch, small, tmp_path / "m.pt", tmp_path / "o.h5")
    m = evaluate(monkeypatch, tmp_path, out)
    assert m["n_spectra"] == 12 and m["rows_in_training"] == 4      # rows, not spectra
    assert ev.training_fit_caveat(4, 4) in ctx(m)["caveats"]


def gen_arrays(noise=None, n=3, **config):
    g = SyntheticGenerator("C1s_single", noise or NoiseConfig(),
                           GeneratorConfig(**{"n_energy_points": 32, **config}))
    clean, noisy, _e, _m = g.generate_batch(n, seed=42)
    return clean, noisy, identities(g, 42)


def same(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


ANG = {"n_angles": 3}
TIM = {"n_times": 3}
# (base config, changed config, included?) -- included means the clean array and both
# identities change; excluded means none of them does.
SIGNAL_CASES = [
    ({}, {"eta": 0.5}, True),
    ({}, {"use_pseudo_voigt": False}, True),
    ({}, {"n_energy_points": 40}, True),
    ({}, {"energy_range": (279.0, 291.0)}, True),
    ({}, {"background_type": "shirley"}, True),
    ({}, {"background_level": 0.1}, True),
    ({}, {"background_slope": 0.01}, True),
    ({"background_type": "shirley"}, {"background_type": "shirley", "background_slope": 0.01}, False),
    ({"background_type": "none"}, {"background_type": "none", "background_level": 0.3}, False),
    ({}, {"intensity_variation": 0.3}, True),
    ({}, {"position_jitter": 0.5}, True),
    ({}, {"width_variation": 0.2}, True),
    ({}, {"normalize": False}, True),
    ({}, {"angle_range": (5.0, 50.0), "angle_cosine_power": 2.0, "angle_shift_rate": 0.0}, False),
    ({}, {"time_range": (1.0, 50.0), "time_decay_constant": 9.0}, False),
    ({**ANG, "angle_intensity_model": "none"}, {**ANG, "angle_intensity_model": "none",
                                                "angle_range": (10.0, 70.0)}, False),
    ({**ANG, "angle_intensity_model": "none"}, {**ANG, "angle_intensity_model": "none",
                                                "angle_cosine_power": 3.0}, False),
    ({**ANG, "angle_intensity_model": "none"}, {**ANG, "angle_intensity_model": "none",
                                                "angle_shift_rate": 0.05}, True),
    ({**ANG}, {**ANG, "angle_range": (10.0, 60.0)}, True),
    ({**ANG}, {**ANG, "angle_cosine_power": 2.0}, True),
    ({**ANG}, {**ANG, "angle_exp_decay": 10.0}, False),
    ({**ANG, "angle_intensity_model": "exponential"},
     {**ANG, "angle_intensity_model": "exponential", "angle_exp_decay": 10.0}, True),
    ({**ANG, "angle_intensity_model": "linear", "angle_range": (10.0, 60.0)},
     {**ANG, "angle_intensity_model": "linear", "angle_range": (10.0, 80.0)}, True),
    ({**TIM, "time_intensity_model": "none"}, {**TIM, "time_intensity_model": "none",
                                               "time_range": (5.0, 50.0)}, False),
    ({**TIM}, {**TIM, "time_decay_constant": 20.0}, True),
    ({**TIM}, {**TIM, "time_oscillation_amplitude": 0.5}, False),
    ({**TIM, "time_intensity_model": "oscillation"},
     {**TIM, "time_intensity_model": "oscillation", "time_oscillation_amplitude": 0.5}, True),
    ({**TIM, "time_intensity_model": "oscillation"},
     {**TIM, "time_intensity_model": "oscillation", "time_oscillation_frequency": 0.2}, True),
    ({**TIM, "time_intensity_model": "oscillation"},
     {**TIM, "time_intensity_model": "oscillation", "time_decay_constant": 9.0}, False),
    ({**TIM, "time_intensity_model": "linear_decay", "time_range": (10.0, 100.0)},
     {**TIM, "time_intensity_model": "linear_decay", "time_range": (10.0, 140.0)}, True),
    ({**TIM, "time_intensity_model": "none"}, {**TIM, "time_intensity_model": "none",
                                               "time_shift_rate": 0.01}, True),
]


@pytest.mark.parametrize("base, changed, included", SIGNAL_CASES,
                         ids=[f"{sorted(c.items())}"[:60] for _b, c, _i in SIGNAL_CASES])
def test_a_signal_field_is_in_the_identity_exactly_when_it_changes_the_arrays(base, changed,
                                                                              included):
    c0, n0, (a0, s0) = gen_arrays(**base)
    c1, n1, (a1, s1) = gen_arrays(**changed)
    assert (not same(c0, c1)) is included, "the generator disagrees with the expectation"
    assert (s0 != s1) is included and (a0 != a1) is included


NOISE_CASES = [
    (NoiseConfig(poisson_level=100.0), NoiseConfig(poisson_level=300.0), True),
    (NoiseConfig(), NoiseConfig(gaussian_std=0.5), False),
    (NoiseConfig(noise_type="gaussian"), NoiseConfig(noise_type="gaussian", poisson_level=9.0), False),
    (NoiseConfig(noise_type="gaussian"), NoiseConfig(noise_type="gaussian", gaussian_std=0.05), True),
    (NoiseConfig(), NoiseConfig(use_gaussian_approx=True), True),
    (NoiseConfig(), NoiseConfig(gaussian_approx_min_rate=0.0), False),
    (NoiseConfig(use_gaussian_approx=True), NoiseConfig(use_gaussian_approx=True,
                                                        gaussian_approx_min_rate=3.0), False),
    # At this level many bins sit below the default floor of 3, so the floor matters.
    (NoiseConfig(poisson_level=5000.0, use_gaussian_approx=True),
     NoiseConfig(poisson_level=5000.0, use_gaussian_approx=True, gaussian_approx_min_rate=0.0),
     True),
    (NoiseConfig(noise_type="none"), NoiseConfig(poisson_level=0.0), False),
    (NoiseConfig(noise_type="none"), NoiseConfig(noise_type="gaussian", gaussian_std=0.0), False),
    (NoiseConfig(noise_type="gaussian", gaussian_std=0.02),
     NoiseConfig(noise_type="mixed", poisson_level=0.0, gaussian_std=0.02), False),
]


@pytest.mark.parametrize("base, changed, included", NOISE_CASES,
                         ids=[str(i) for i in range(len(NOISE_CASES))])
def test_a_noise_field_is_in_the_noise_identity_only(base, changed, included):
    c0, n0, (a0, s0) = gen_arrays(noise=base)
    c1, n1, (a1, s1) = gen_arrays(noise=changed)
    assert same(c0, c1) and s0 == s1                      # clean never depends on noise
    assert (not same(n0, n1)) is included, "the generator disagrees with the expectation"
    assert (a0 != a1) is included


@pytest.mark.parametrize("model, key", [("angle", "angle_range"), ("time", "time_range")])
def test_a_linear_model_upper_bound_is_in_the_identity_even_from_zero(model, key):
    """With a lower bound of 0 the linear models read only the ratio to the upper bound,
    so the arrays agree to float32 rounding; the adopted design (third audit) keeps the
    bound in the identity, which can only call identical draws different, never the
    reverse."""
    extra = ({"n_angles": 3, "angle_intensity_model": "linear"} if model == "angle"
             else {"n_times": 3, "time_intensity_model": "linear_decay"})
    hi = 80.0 if model == "angle" else 140.0
    c0, _n0, ids0 = gen_arrays(**extra, **{key: (0.0, 60.0 if model == "angle" else 100.0)})
    c1, _n1, ids1 = gen_arrays(**extra, **{key: (0.0, hi)})
    np.testing.assert_allclose(c0, c1, rtol=1e-6, atol=1e-7)
    assert ids0[0] != ids1[0] and ids0[1] != ids1[1]


@pytest.mark.parametrize("variant", [{"background_type": "none"},
                                     {"background_type": "shirley", "background_level": 0.0},
                                     {"background_type": "linear", "background_level": 0.0,
                                      "background_slope": 0.0}])
def test_configurations_with_identical_arrays_share_one_identity(variant):
    c0, n0, ids0 = gen_arrays(background_type="none")
    c1, n1, ids1 = gen_arrays(**variant)
    assert same(c0, c1) and same(n0, n1) and ids0 == ids1


def test_a_field_at_its_no_effect_value_is_omitted():
    _c, _n, base = gen_arrays()
    _c, _n, explicit = gen_arrays(position_jitter=0.0, angle_shift_rate=0.0, time_shift_rate=0.0)
    assert base == explicit


def test_every_configuration_field_is_classified():
    fields = {f.name for f in dataclasses.fields(GeneratorConfig)} | {
        f.name for f in dataclasses.fields(NoiseConfig)}
    assert set(CLASSIFICATION) == fields


FROZEN_JSON = ('{"background":{"level":0.05,"slope":0.001,"type":"linear"},"energy_range":'
               '[278.8,290.8],"eta":0.3,"intensity_variation":0.2,"n_energy_points":32,"noise":'
               '{"poisson":{"level":100.0}},"normalize":true,"peaks":[[284.8,1.2,1.0]],'
               '"pseudo_voigt":true,"seed":42,"width_variation":0.1}')


def test_one_identifier_is_frozen():
    expected = "generate:" + hashlib.sha256(FROZEN_JSON.encode("utf-8")).hexdigest()[:16]
    assert expected == "generate:6b2e5a27bb586a1a"
    _c, _n, (acq, signal) = gen_arrays(noise=NoiseConfig(poisson_level=100.0))
    assert acq == expected and signal == "generate-signal:8c637641d25e8bf4"


def test_the_versions_note_when_generators_differ(monkeypatch, tmp_path):
    data = generate(monkeypatch, tmp_path / "d.h5", n=8)
    train(monkeypatch, data, tmp_path / "m.pt")
    out = infer(monkeypatch, data, tmp_path / "m.pt", tmp_path / "o.h5")
    assert ev.VERSIONS_CAVEAT not in ctx(evaluate(monkeypatch, tmp_path, out))["caveats"]
    with h5py.File(out, "a") as f:
        decl = json.loads(f["clean"].attrs["reference_origin"])
        decl["generator"] = "dnndenoiser 9.9.9 SyntheticGenerator"
        ref.write_declaration(f["clean"], decl)
    assert ev.VERSIONS_CAVEAT in ctx(evaluate(monkeypatch, tmp_path, out))["caveats"]


# ---------------------------------------------------------------------------------- 9


def test_new_generate_output_still_evaluates_as_before(monkeypatch, capsys, tmp_path):
    data = generate(monkeypatch, tmp_path / "d.h5", n=8)
    other = generate(monkeypatch, tmp_path / "e.h5", n=8, seed=3)
    with h5py.File(data) as f:
        assert f["frame_index"].dtype == np.int64
        noisy, energy = f["noisy"][:], f["energy"][:]
        acq, index = f["noisy"].attrs["acquisition_id"], f["frame_index"][:]
    path = write(tmp_path / "x.h5", noisy, noisy, None, units="normalised_to_spectrum_max",
                 energy=energy, acquisition_id=acq, frame_index=index)
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(path), "--clean", str(other))
    assert "the reference names no source acquisition (a synthetic-truth declaration has none)" in err
    m = evaluate(monkeypatch, tmp_path, path, "--clean", str(other), "--assert-alignment", "rows")
    assert ctx(m)["row_correspondence"] == "asserted" and "snr_gain_mean" in m
    single = tmp_path / "single.h5"
    with h5py.File(other) as g, h5py.File(single, "w") as s:
        s.create_dataset("clean", data=g["clean"][:1])
        for k, v in g["clean"].attrs.items():
            s["clean"].attrs[k] = v
        s.create_dataset("energy", data=g["energy"][:])
    m = evaluate(monkeypatch, tmp_path, path, "--clean", str(single), "--shared-reference",
                 "--assert-alignment", "rows")
    assert ctx(m)["shared_reference"] is True


def small_manifest(kind="clean", signal="generate-signal:00", runs=([0, 7],), clean_digest=None):
    return {"targets": {"kind": kind},
            "training_data": {"array_digests": {} if clean_digest is None else
                              {"clean": clean_digest},
                              "signal_identity": signal, "frame_index_runs": [list(r) for r in runs]}}


@pytest.mark.parametrize("frames, expected", [(np.arange(20, 28), (False, None)),
                                              (np.arange(5, 13), (True, 3)),
                                              (None, (False, None))],
                         ids=["disjoint-frames", "intersecting", "no-frames"])
def test_a_signal_identity_establishes_use_only_with_intersecting_frames(frames, expected):
    reference = np.ones((8, 4), dtype=np.float32)
    got = prov.reference_in_training(small_manifest(), reference=reference,
                                     signal_identity="generate-signal:00", frame_index=frames)
    assert got == expected
    other = prov.reference_in_training(small_manifest(), reference=reference,
                                       signal_identity="generate-signal:11", frame_index=frames)
    assert other == (False, None)


def test_a_moving_average_model_never_establishes_use_of_its_files_clean(monkeypatch, tmp_path):
    """Its targets were window means of the frames, not the file's clean."""
    data = stack_file(tmp_path / "s.h5", e=256)
    with h5py.File(data, "a") as f:
        f.create_dataset("clean", data=f["frames"][:])
        ref.write_declaration(f["clean"], estimate(acquisition="other"))
    train(monkeypatch, data, tmp_path / "m.pt", method="moving-average")
    out = infer(monkeypatch, data, tmp_path / "m.pt", tmp_path / "o.h5")
    with h5py.File(out, "a") as f:
        for name in ("noisy", "denoised"):
            f[name].attrs["intensity_units"] = "counts"
    c = ctx(evaluate(monkeypatch, tmp_path, out))
    assert c["relationship"]["used_in_model_development"] == "unknown"
    assert c["reference_rows_in_training"] is None
    # The same reference against a clean-target manifest would establish it.
    digest = prov.matching_digest(np.asarray(h5py.File(data)["clean"][:]))
    assert prov.reference_in_training(small_manifest(clean_digest=digest),
                                      reference=h5py.File(data)["clean"][:],
                                      signal_identity=None, frame_index=None)[0] is True
    assert prov.reference_in_training(small_manifest(kind="leave_one_out_window_mean",
                                                     clean_digest=digest),
                                      reference=h5py.File(data)["clean"][:],
                                      signal_identity=None, frame_index=None) == (False, None)
