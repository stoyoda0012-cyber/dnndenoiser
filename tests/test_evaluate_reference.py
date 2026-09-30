"""The evaluation-reference contract, phase 1 (docs/design/EVALUATION_REFERENCE_CONTRACT.md §8).

Every rejection is pinned to its reason, with every other field valid; every rejection has a
positive counterpart, so an implementation that refuses every hard case fails too. Expected
values are computed here from the arrays, never by the code under test, and the legacy
expectations come from a golden file produced by ``cmd_evaluate`` at ``cb5e000``.

Phase 2 (the rest of the alignment contract, and content digests) is not tested here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from dnndenoiser import evaluation as ev
from dnndenoiser import reference as ref
from dnndenoiser.cli import main
from dnndenoiser.data.synthetic_generator import GeneratorConfig, SyntheticGenerator

DATA = Path(__file__).resolve().parent / "data"
FORBIDDEN = ("snr", "quality", "accuracy", "improvement", "gain", "reduction")


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["dnndenoiser", *argv])
    return main()


def refuse(monkeypatch, capsys, *argv) -> str:
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, *argv)
    assert exc.value.code == 1
    return capsys.readouterr().err


def load_strict(path) -> dict:
    def no_constant(name):
        raise AssertionError(f"non-strict JSON constant {name} in {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=no_constant)


TRUTH = {"origin": "synthetic_truth", "generator": "test fixture", "units": "counts"}


def estimate(acquisition="acq-A", frames="all", construction="frame_mean", **extra):
    decl = {"origin": "estimate", "construction": construction,
            "source": {"acquisition_id": acquisition, "frames": frames},
            "conditions": {"energy_calibration": "unknown"}, "units": "counts",
            "noise": "unknown"}
    decl.update(extra)
    return decl


def write(path, noisy, denoised, clean, declaration=None, raw_attrs=None, units="counts",
          acquisition_id=None, frame_index=None, energy=None):
    """An evaluate input. ``declaration`` writes a valid bundle; ``raw_attrs`` writes
    attributes as given (for malformed cases)."""
    noisy = np.asarray(noisy, dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("noisy", data=noisy)
        if units is not None:
            f["noisy"].attrs["intensity_units"] = units
        if acquisition_id is not None:
            f["noisy"].attrs["acquisition_id"] = acquisition_id
        if denoised is not None:
            f.create_dataset("denoised", data=np.asarray(denoised, dtype=np.float32))
        if clean is not None:
            f.create_dataset("clean", data=np.asarray(clean, dtype=np.float32))
            if declaration is not None:
                ref.write_declaration(f["clean"], declaration)
            for key, value in (raw_attrs or {}).items():
                f["clean"].attrs[key] = value
        f.create_dataset("energy", data=np.linspace(0, 1, noisy.shape[-1]) if energy is None
                         else energy)
        if frame_index is not None:
            f.create_dataset("frame_index", data=np.asarray(frame_index))
    return path


def arrays(n=6, e=16, seed=0):
    rng = np.random.default_rng(seed)
    clean = rng.uniform(1.0, 2.0, (n, e))
    noisy = clean + rng.normal(0, 0.2, (n, e))
    denoised = clean + rng.normal(0, 0.05, (n, e))
    return noisy.astype(np.float32), denoised.astype(np.float32), clean.astype(np.float32)


def evaluate(monkeypatch, tmp_path, path, *extra):
    out = tmp_path / "m.json"
    run(monkeypatch, "evaluate", "-d", str(path), "-o", str(out), *extra)
    return load_strict(out)


# ---------------------------------------------------------------------------------- 1


@pytest.mark.parametrize("extra, ndim, units", [
    ((), 2, "normalised_to_spectrum_max"),
    (("--n-angles", "3"), 3, "normalised_to_global_max"),
    (("--no-normalize",), 2, "generator_intensity"),
])
def test_generate_declares_its_clean_arrays_as_synthetic_truth(monkeypatch, tmp_path, extra, ndim, units):
    path = tmp_path / "g.h5"
    run(monkeypatch, "generate", "-o", str(path), "-n", "4", "--n-energy", "32",
        "--peak-set", "C1s_single", *extra)
    with h5py.File(path) as f:
        assert f["clean"].ndim == ndim
        stored, effective = ref.read_declaration(f["clean"])
        assert f["clean"].attrs["reference_schema_version"] == 1
        assert stored["origin"] == "synthetic_truth" and stored["units"] == units
        assert stored["generator"].startswith("dnndenoiser ")
        assert f["noisy"].attrs["intensity_units"] == units


@pytest.mark.parametrize("missing", ["reference_declaration", "intensity_units"])
def test_save_hdf5_refuses_to_write_without_a_declaration(tmp_path, missing):
    gen = SyntheticGenerator("C1s_single", config=GeneratorConfig(n_energy_points=16))
    clean, noisy, energy, meta = gen.generate_batch(2, seed=0)
    kwargs = gen.truth_arguments(clean.ndim)
    kwargs.pop(missing)
    with pytest.raises(TypeError, match=missing):
        SyntheticGenerator.save_hdf5(tmp_path / "x.h5", clean, noisy, energy, meta, **kwargs)


def test_save_hdf5_refuses_units_that_contradict_the_declaration(tmp_path):
    gen = SyntheticGenerator("C1s_single", config=GeneratorConfig(n_energy_points=16))
    clean, noisy, energy, meta = gen.generate_batch(2, seed=0)
    with pytest.raises(ValueError, match="declared in 'counts'"):
        SyntheticGenerator.save_hdf5(tmp_path / "x.h5", clean, noisy, energy, meta,
                                     reference_declaration=TRUTH,
                                     intensity_units="normalised_to_spectrum_max")


@pytest.mark.parametrize("attrs, reason", [
    ({"reference_schema_version": 1, "reference_origin": json.dumps({"origin": "truth"})},
     "'origin' must be one of"),
    ({"reference_origin": json.dumps(TRUTH)}, "present without its pair"),
    ({"reference_schema_version": 1}, "present without its pair"),
    ({"reference_schema_version": 2, "reference_origin": json.dumps(TRUTH)},
     "unsupported reference_schema_version 2"),
    ({"reference_schema_version": 1, "reference_origin": "{not json"}, "not valid JSON"),
    ({"reference_schema_version": 1, "reference_origin": json.dumps({"origin": "synthetic_truth"})},
     "synthetic_truth needs"),
])
def test_a_malformed_declaration_is_refused_not_read_as_undeclared(monkeypatch, capsys, tmp_path, attrs, reason):
    n, d, c = arrays()
    path = write(tmp_path / "x.h5", n, d, c, raw_attrs=attrs)
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(path))
    assert "malformed reference declaration" in err and reason in err


def test_absent_and_explicit_undeclared_both_read_as_undeclared(monkeypatch, tmp_path):
    n, d, c = arrays()
    for name, decl in (("absent.h5", None), ("explicit.h5", {"origin": "undeclared"})):
        out = evaluate(monkeypatch, tmp_path, write(tmp_path / name, n, d, c, declaration=decl))
        ctx = out["evaluation_context"]["reference"]
        assert ctx["effective_declaration"] == {"origin": "undeclared"}
        assert ctx["stored_declaration"] == (None if decl is None else decl)


# ---------------------------------------------------------------------------------- 2


@pytest.fixture(scope="module")
def model_64(tmp_path_factory):
    """A 64-point noise2clean model, trained once for the propagation tests."""
    tmp = tmp_path_factory.mktemp("m64")
    data = tmp / "d.h5"
    mp = pytest.MonkeyPatch()
    try:
        run(mp, "generate", "-o", str(data), "-n", "16", "--n-energy", "64", "--peak-set", "C1s_single")
        model = tmp / "m.pt"
        run(mp, "train", "-d", str(data), "-o", str(model), "--arch", "FCNN", "--epochs", "1",
            "--batch-size", "8", "--device", "cpu", "--seed", "0")
    finally:
        mp.undo()
    return model


def check_propagated(src: Path, out: Path, resampled_to=None):
    """The whole bundle, units, acquisition id and coordinates survive infer."""
    with h5py.File(src) as s, h5py.File(out) as o:
        for key in ("reference_schema_version", "reference_origin"):
            if key in s["clean"].attrs:
                assert key in o["clean"].attrs, f"infer dropped {key}"
                assert o["clean"].attrs[key] == s["clean"].attrs[key], f"infer altered {key}"
            else:
                assert key not in o["clean"].attrs, f"infer invented {key}"
        allowed = {"reference_schema_version", "reference_origin", "reference_lineage"}
        assert set(o["clean"].attrs) <= allowed, "infer stored something beyond the bundle"
        lineage = json.loads(o["clean"].attrs.get("reference_lineage", "[]"))
        if resampled_to is None:
            assert lineage == []
            # infer has always stored energy as float32; unchanged at that precision
            np.testing.assert_array_equal(o["energy"][:], s["energy"][:].astype(np.float32))
        else:
            assert len(lineage) == 1
            assert {k: lineage[0][k] for k in ("operation", "from_points", "to_points", "step")} == \
                {"operation": "linear resample", "from_points": s["energy"].shape[0],
                 "to_points": resampled_to, "step": "infer"}
            e = s["energy"][:]
            np.testing.assert_allclose(o["energy"][:], np.linspace(e[0], e[-1], resampled_to), rtol=1e-6)
        for name in ("noisy", "denoised"):
            for key in ("intensity_units", "acquisition_id"):
                assert key in o[name].attrs, f"{key} lost on {name}"
                assert o[name].attrs[key] == s["noisy"].attrs[key], f"{key} altered on {name}"
        np.testing.assert_array_equal(o["frame_index"][:], s["frame_index"][:])


@pytest.mark.parametrize("decl", [TRUTH, estimate(), {"origin": "undeclared"}, None],
                         ids=["truth", "estimate", "explicit-undeclared", "absent"])
@pytest.mark.parametrize("points", [64, 80], ids=["no-resample", "resample"])
def test_infer_carries_the_declaration_and_its_context(monkeypatch, tmp_path, model_64, decl, points):
    n, _d, c = arrays(n=5, e=points)
    src = write(tmp_path / "in.h5", n, None, c, declaration=decl, acquisition_id="acq-A",
                frame_index=np.arange(5) * 10)
    out = tmp_path / "out.h5"
    run(monkeypatch, "infer", "-d", str(src), "-m", str(model_64), "-o", str(out), "--device", "cpu")
    check_propagated(src, out, None if points == 64 else 64)


@pytest.mark.parametrize("plant", ["drop-version", "alter-field", "upgrade-origin",
                                   "drop-units", "drop-acquisition", "store-relationship"])
def test_the_propagation_check_rejects_a_wrong_copier(monkeypatch, tmp_path, model_64, plant):
    n, _d, c = arrays(n=5, e=64)
    src = write(tmp_path / "in.h5", n, None, c, declaration=estimate(), acquisition_id="acq-A",
                frame_index=np.arange(5))
    out = tmp_path / "out.h5"
    run(monkeypatch, "infer", "-d", str(src), "-m", str(model_64), "-o", str(out), "--device", "cpu")
    check_propagated(src, out)
    with h5py.File(out, "a") as o:
        if plant == "drop-version":
            del o["clean"].attrs["reference_schema_version"]
        elif plant == "alter-field":
            o["clean"].attrs["reference_origin"] = ref.canonical(estimate(acquisition="acq-B"))
        elif plant == "upgrade-origin":
            o["clean"].attrs["reference_origin"] = ref.canonical(TRUTH)
        elif plant == "drop-units":
            del o["denoised"].attrs["intensity_units"]
        elif plant == "drop-acquisition":
            del o["noisy"].attrs["acquisition_id"]
        else:
            o["clean"].attrs["overlap_with_evaluated"] = "no_overlap_declared"
    with pytest.raises(AssertionError):
        check_propagated(src, out)


def test_infer_refuses_a_malformed_declaration(monkeypatch, capsys, tmp_path, model_64):
    n, _d, c = arrays(n=3, e=64)
    src = write(tmp_path / "in.h5", n, None, c, raw_attrs={"reference_schema_version": 1})
    err = refuse(monkeypatch, capsys, "infer", "-d", str(src), "-m", str(model_64),
                 "-o", str(tmp_path / "o.h5"), "--device", "cpu")
    assert "malformed reference declaration" in err


# ---------------------------------------------------------------------------------- 3


@pytest.mark.parametrize("extra", [(), ("--n-angles", "3")], ids=["1-D", "angle-resolved"])
def test_generate_infer_evaluate_yields_the_truth_output_with_no_extra_flags(monkeypatch, tmp_path, extra):
    data, model, out = tmp_path / "d.h5", tmp_path / "m.pt", tmp_path / "o.h5"
    run(monkeypatch, "generate", "-o", str(data), "-n", "8", "--n-energy", "32",
        "--peak-set", "C1s_single", *extra)
    run(monkeypatch, "train", "-d", str(data), "-o", str(model), "--arch", "FCNN", "--epochs", "1",
        "--batch-size", "8", "--device", "cpu")
    run(monkeypatch, "infer", "-d", str(data), "-m", str(model), "-o", str(out), "--device", "cpu")
    metrics = evaluate(monkeypatch, tmp_path, out)
    assert {"snr_input_mean", "snr_gain_mean"} <= set(metrics)
    assert metrics["evaluation_context"]["reference"]["declaration_source"] == "file"


def test_a_frame_mean_from_the_same_acquisition_keeps_its_overlap_through_infer(monkeypatch, tmp_path, model_64):
    rng = np.random.default_rng(3)
    frames = (1.5 + rng.normal(0, 0.2, (6, 64))).astype(np.float32)
    mean = np.repeat(frames.mean(0, keepdims=True), 6, 0)
    src = write(tmp_path / "in.h5", frames, None, mean, declaration=estimate(acquisition="acq-A"),
                acquisition_id="acq-A", frame_index=np.arange(6))
    out = tmp_path / "out.h5"
    run(monkeypatch, "infer", "-d", str(src), "-m", str(model_64), "-o", str(out), "--device", "cpu")
    metrics = evaluate(monkeypatch, tmp_path, out)
    rel = metrics["evaluation_context"]["relationship"]
    assert rel["overlap_with_evaluated"] == "overlap" and rel["overlap_basis"] == "established"
    assert not any("snr" in k or "agreement" in k for k in metrics)
    assert ev.SAME_FRAMES_CAVEAT in metrics["evaluation_context"]["caveats"]


# ---------------------------------------------------------------------------------- 4


def expected(noisy, denoised, clean):
    x, y, r = (np.asarray(a, dtype=np.float64) for a in (noisy, denoised, clean))
    mi, mo = ((x - r) ** 2).mean(-1), ((y - r) ** 2).mean(-1)
    pr = (r ** 2).mean(-1)
    ok = pr > 0
    si = 10 * np.log10(pr[ok] / np.maximum(mi[ok], 1e-10))
    so = 10 * np.log10(pr[ok] / np.maximum(mo[ok], 1e-10))
    return mi, mo, pr, ok, si, so


def test_truth_arithmetic_matches_an_independent_computation(monkeypatch, tmp_path):
    n, d, c = arrays(n=8, e=16, seed=5)
    d[0] = c[0]                      # output exactly on the truth: output floor active
    n[1] = c[1]                      # input exactly on the truth: input floor active
    c[2] = 0.0                       # zero reference power: excluded and counted
    n[2], d[2] = 0.1, 0.2
    m = evaluate(monkeypatch, tmp_path, write(tmp_path / "t.h5", n, d, c, declaration=TRUTH))
    mi, mo, pr, ok, si, so = expected(n, d, c)
    assert m["mse_in_mean"] == pytest.approx(mi.mean(), rel=1e-12)
    assert m["mse_out_mean"] == pytest.approx(mo.mean(), rel=1e-12)
    assert m["mse_difference"] == pytest.approx(mi.mean() - mo.mean(), rel=1e-12)
    assert m["snr_input_mean"] == pytest.approx(si.mean(), rel=1e-12)
    assert m["snr_output_mean"] == pytest.approx(so.mean(), rel=1e-12)
    assert m["snr_gain_mean"] == pytest.approx((so - si).mean(), rel=1e-12)
    assert m["snr_gain_std"] == pytest.approx(np.std(so - si, ddof=0), rel=1e-12)
    assert m["zero_reference_power_count"] == 1
    assert m["floor_active_input_count"] == 1 and m["floor_active_output_count"] == 1


def test_floor_counts_are_zero_when_nothing_is_under_the_floor(monkeypatch, tmp_path):
    n, d, c = arrays(seed=6)
    m = evaluate(monkeypatch, tmp_path, write(tmp_path / "t.h5", n, d, c, declaration=TRUTH))
    assert m["floor_active_input_count"] == 0 and m["floor_active_output_count"] == 0
    assert m["zero_reference_power_count"] == 0


def test_no_eligible_spectrum_gives_null_with_a_status(monkeypatch, tmp_path):
    n, d, _c = arrays()
    m = evaluate(monkeypatch, tmp_path, write(tmp_path / "t.h5", n, d, np.zeros_like(n), declaration=TRUTH))
    for key in ("snr_input_mean", "snr_output_mean", "snr_gain_mean", "snr_gain_std"):
        assert m[key] is None and "no spectrum has non-zero reference power" in m["status"][key]


# ---------------------------------------------------------------------------------- 5


def forbidden_in(keys, headings):
    return [w for w in FORBIDDEN for k in list(keys) + list(headings) if w in k.lower()]


@pytest.mark.parametrize("case", ["estimate", "estimate_same_data", "undeclared"])
def test_non_truth_output_uses_no_quality_words(monkeypatch, capsys, tmp_path, case):
    n, d, c = arrays(seed=7)
    decl = {"estimate": estimate(acquisition="other"), "estimate_same_data": estimate(),
            "undeclared": None}[case]
    path = write(tmp_path / "x.h5", n, d, c, declaration=decl, acquisition_id="acq-A")
    capsys.readouterr()
    m = evaluate(monkeypatch, tmp_path, path)
    printed = capsys.readouterr().out.splitlines()
    start = printed.index(next(line for line in printed if line.startswith("=== ") and "Evaluation" not in line))
    headings = [line.split("  ")[0] for line in printed[start:] if line and not line.startswith("Note:")
                and not line.startswith("Metrics saved") and line != "Done."]
    keys = [k for k in m if k not in ("evaluation_context", "status")]
    assert forbidden_in(keys, headings) == []


@pytest.mark.parametrize("planted", ["snr_gain_mean", "accuracy_pct", "quality", "mse_reduction_mean",
                                     "improvement_db"])
def test_the_naming_check_rejects_planted_words(planted):
    assert forbidden_in(["mse_in_mean", planted], []) != []


def test_agreement_db_values_match_an_independent_computation(monkeypatch, tmp_path):
    n, d, c = arrays(seed=8)
    m = evaluate(monkeypatch, tmp_path, write(tmp_path / "e.h5", n, d, c,
                                              declaration=estimate(acquisition="other"),
                                              acquisition_id="acq-A"))
    _mi, _mo, _pr, _ok, si, so = expected(n, d, c)
    assert m["agreement_db_input_mean"] == pytest.approx(si.mean(), rel=1e-12)
    assert m["agreement_db_output_mean"] == pytest.approx(so.mean(), rel=1e-12)
    assert m["agreement_db_change_mean"] == pytest.approx((so - si).mean(), rel=1e-12)
    assert not any(k.startswith("snr") for k in m)


# ---------------------------------------------------------------------------------- 6


@pytest.fixture
def legacy_file(tmp_path):
    g = json.loads((DATA / "legacy_evaluate_fixture.json").read_text(encoding="utf-8"))
    return write(tmp_path / "legacy.h5", np.asarray(g["noisy"]), np.asarray(g["denoised"]),
                 np.asarray(g["clean"]), units=None, energy=np.asarray(g["energy"], dtype=np.float32))


GOLDEN = json.loads((DATA / "legacy_evaluate_golden_cb5e000.json").read_text(encoding="utf-8"))


def test_an_undeclared_legacy_file_yields_no_snr_by_default(monkeypatch, tmp_path, legacy_file):
    m = evaluate(monkeypatch, tmp_path, legacy_file)
    assert not any(k.startswith("snr") for k in m)


def test_declaring_the_legacy_file_as_truth_restores_the_values(monkeypatch, tmp_path, legacy_file):
    m = evaluate(monkeypatch, tmp_path, legacy_file, "--reference-origin", "synthetic_truth",
                 "--generator", "unknown legacy generator", "--units", "normalised_to_spectrum_max")
    for new, old in (("snr_input_mean", "snr_input_mean"), ("snr_output_mean", "snr_output_mean"),
                     ("snr_gain_mean", "snr_gain_mean"), ("mse_in_mean", "mse_input_mean"),
                     ("mse_out_mean", "mse_output_mean")):
        assert m[new] == pytest.approx(GOLDEN[old], rel=1e-5), new
    assert m["evaluation_context"]["reference"]["declaration_source"] == "cli"


def test_a_truth_declaration_without_its_generator_is_refused(monkeypatch, capsys, tmp_path, legacy_file):
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(legacy_file), "--reference-origin",
                 "synthetic_truth", "--units", "normalised_to_spectrum_max")
    assert "synthetic_truth needs ['generator']" in err


def test_legacy_output_reproduces_the_golden_file_exactly(monkeypatch, tmp_path, legacy_file):
    m = evaluate(monkeypatch, tmp_path, legacy_file, "--legacy-output")
    ctx = m.pop("evaluation_context")
    assert m == GOLDEN
    assert ctx["evaluate_output_version"] == "1-legacy"
    assert ctx["legacy_baseline"] == "cmd_evaluate at cb5e000" and "not declared" in ctx["limitations"]


def test_legacy_output_refuses_a_non_finite_historical_value(monkeypatch, capsys, tmp_path):
    n, d, c = arrays(seed=9)
    n[0] = c[0]                      # zero input MSE, positive output MSE: reduction is -inf
    path = write(tmp_path / "x.h5", n, d, c, units=None)
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(path), "--legacy-output")
    assert "legacy output refused" in err and "'mse_reduction_mean'" in err


def test_legacy_output_is_refused_for_a_declared_reference(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "t.h5", n, d, c, declaration=TRUTH)
    assert "this reference is declared 'synthetic_truth'" in refuse(
        monkeypatch, capsys, "evaluate", "-d", str(path), "--legacy-output")


def test_legacy_output_is_refused_with_a_cli_declaration(monkeypatch, capsys, tmp_path, legacy_file):
    assert "cannot be combined with a declaration" in refuse(
        monkeypatch, capsys, "evaluate", "-d", str(legacy_file), "--legacy-output",
        "--reference-origin", "synthetic_truth", "--generator", "g", "--units", "counts")


def test_legacy_output_accepts_an_explicit_undeclared_bundle(monkeypatch, tmp_path):
    g = json.loads((DATA / "legacy_evaluate_fixture.json").read_text(encoding="utf-8"))
    path = write(tmp_path / "u.h5", np.asarray(g["noisy"]), np.asarray(g["denoised"]),
                 np.asarray(g["clean"]), declaration={"origin": "undeclared"}, units=None,
                 energy=np.asarray(g["energy"], dtype=np.float32))
    m = evaluate(monkeypatch, tmp_path, path, "--legacy-output")
    m.pop("evaluation_context")
    assert m == GOLDEN


# ---------------------------------------------------------------------------------- 7


def test_two_references_without_a_choice_are_refused(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "a.h5", n, d, c, declaration=TRUTH)
    other = write(tmp_path / "b.h5", n, None, c + 1, declaration=TRUTH)
    assert "choose one with --reference" in refuse(
        monkeypatch, capsys, "evaluate", "-d", str(path), "--clean", str(other))
    m = evaluate(monkeypatch, tmp_path, path, "--clean", str(other), "--reference", "external")
    _mi, mo, *_ = expected(n, d, c + 1)
    assert m["mse_out_mean"] == pytest.approx(mo.mean(), rel=1e-12)
    assert m["evaluation_context"]["reference"]["selected"] == "external"


def test_no_reference_is_refused(monkeypatch, capsys, tmp_path):
    n, d, _c = arrays()
    path = write(tmp_path / "a.h5", n, d, None)
    assert "a reference is required" in refuse(monkeypatch, capsys, "evaluate", "-d", str(path))


def test_a_known_origin_accepts_an_identical_declaration_and_refuses_a_different_one(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "t.h5", n, d, c, declaration=TRUTH)
    same = evaluate(monkeypatch, tmp_path, path, "--reference-origin", "synthetic_truth",
                    "--generator", TRUTH["generator"], "--units", TRUTH["units"])
    assert same["evaluation_context"]["reference"]["declaration_source"] == "file"
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(path), "--reference-origin",
                 "synthetic_truth", "--generator", "another", "--units", TRUTH["units"])
    assert "differs from the one stored" in err and "another" in err


def test_an_explicit_undeclared_bundle_can_be_completed_from_the_command_line(monkeypatch, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "u.h5", n, d, c, declaration={"origin": "undeclared"})
    m = evaluate(monkeypatch, tmp_path, path, "--reference-origin", "synthetic_truth",
                 "--generator", "g", "--units", "counts")
    r = m["evaluation_context"]["reference"]
    assert r["declaration_source"] == "cli" and r["stored_declaration"] == {"origin": "undeclared"}
    assert "snr_gain_mean" in m


# ---------------------------------------------------------------------------------- 8


@pytest.mark.parametrize("mutate, reason", [
    (lambda e: e.pop("noise"), "estimate needs ['noise']"),
    (lambda e: e.__setitem__("units", "  "), "'units' must be a non-empty string"),
    (lambda e: e["source"].__setitem__("acquisition_id", ""), "'source.acquisition_id' must be"),
    (lambda e: e.__setitem__("construction", "other"), "'description' must be a non-empty string"),
    (lambda e: e.__setitem__("conditions", {}), "'conditions' must be a non-empty object"),
    (lambda e: e["source"].__setitem__("frames", []), "'source.frames' must be"),
])
def test_an_incomplete_estimate_is_refused(monkeypatch, capsys, tmp_path, mutate, reason):
    decl = estimate()
    mutate(decl)
    decl_path = tmp_path / "decl.json"
    decl_path.write_text(json.dumps(decl), encoding="utf-8")
    n, d, c = arrays()
    path = write(tmp_path / "u.h5", n, d, c)
    assert reason in refuse(monkeypatch, capsys, "evaluate", "-d", str(path),
                            "--reference-declaration", str(decl_path))


def test_no_overlap_contradicting_an_established_overlap_is_refused(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "e.h5", n, d, c, declaration=estimate(acquisition="acq-A"),
                 acquisition_id="acq-A", frame_index=np.arange(len(n)))
    assert "contradicts an established overlap" in refuse(
        monkeypatch, capsys, "evaluate", "-d", str(path), "--overlap", "no_overlap_declared")


def test_a_disjoint_acquisition_declared_without_overlap_is_an_estimate(monkeypatch, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "e.h5", n, d, c, declaration=estimate(acquisition="acq-B"),
                 acquisition_id="acq-A")
    m = evaluate(monkeypatch, tmp_path, path, "--overlap", "no_overlap_declared")
    ctx = m["evaluation_context"]
    assert ctx["relationship"]["overlap_with_evaluated"] == "no_overlap_declared"
    assert "agreement_db_change_mean" in m
    assert ev.SAME_FRAMES_CAVEAT not in ctx["caveats"] and ev.ESTIMATE_CAVEAT in ctx["caveats"]


# ---------------------------------------------------------------------------------- 9


def test_the_two_relative_summaries_can_disagree_in_sign():
    # mse_in = (0.01, 1), mse_out = (0.02, 0.5): by hand -25 % and 1 - 0.52/1.01 = 48.51 %
    r = np.zeros((2, 1))
    x = np.array([[0.1], [1.0]])
    y = np.array([[np.sqrt(0.02)], [np.sqrt(0.5)]])
    m = ev.evaluate_arrays(x, y, r, "undeclared")
    assert m["mean_relative_mse_change_per_spectrum_pct"] == pytest.approx(-25.0)
    assert m["relative_mse_change_aggregate_pct"] == pytest.approx(100 * (1 - 0.52 / 1.01))


def test_a_zero_input_mse_spectrum_is_excluded_from_the_per_spectrum_mean_only():
    r = np.ones((3, 4))
    x = r.copy()
    x[1:] += 1.0
    y = r + 0.5
    m = ev.evaluate_arrays(x, y, r, "undeclared")
    assert m["per_spectrum_excluded_zero_input_mse"] == 1
    assert m["mean_relative_mse_change_per_spectrum_pct"] == pytest.approx(75.0)
    assert m["relative_mse_change_aggregate_pct"] == pytest.approx(100 * (1 - 0.75 / 2))


def test_all_spectra_with_zero_input_mse_give_null_with_a_status():
    r = np.ones((2, 4))
    m = ev.evaluate_arrays(r, r + 1, r, "undeclared")
    assert m["mean_relative_mse_change_per_spectrum_pct"] is None
    assert m["relative_mse_change_aggregate_pct"] is None
    assert "zero" in m["status"]["relative_mse_change_aggregate_pct"]


# ------------------------------------------------------------- phase-1 alignment


def test_a_reference_of_another_shape_is_never_broadcast(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "b.h5", n, d, c[:1], declaration=TRUTH)
    assert "never broadcast" in refuse(monkeypatch, capsys, "evaluate", "-d", str(path))


def test_non_finite_inputs_are_refused(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    d[2, 3] = np.nan
    path = write(tmp_path / "nan.h5", n, d, c, declaration=TRUTH)
    assert "'denoised' contains non-finite values" in refuse(monkeypatch, capsys, "evaluate", "-d", str(path))
