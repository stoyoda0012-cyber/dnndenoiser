"""The evaluation-reference contract, phase 2: alignment (design §5, test group 10) and
content digests (§6.6, group 11).

Every rejection is pinned to its reason with every other field valid, and has a positive
counterpart. Expected digests come from an independent transcription of ``dnd-digest-1``
written here, and from literals frozen with the implementation.
"""
from __future__ import annotations

import hashlib
import json
import struct

import h5py
import numpy as np
import pytest

from dnndenoiser import alignment as al
from dnndenoiser import digest as dg
from tests.test_evaluate_reference import (DATA, TRUTH, arrays, estimate, evaluate, refuse,
                                           run, write)

ROWS = ("--assert-alignment", "rows")


def context(m):
    return m["evaluation_context"]


def arrays3(n=4, k=3, e=16, seed=1):
    rng = np.random.default_rng(seed)
    clean = rng.uniform(1.0, 2.0, (n, k, e))
    noisy = clean + rng.normal(0, 0.2, clean.shape)
    denoised = clean + rng.normal(0, 0.05, clean.shape)
    return noisy.astype(np.float32), denoised.astype(np.float32), clean.astype(np.float32)


def pair(tmp_path, n, d, c, data=None, reference=None, declaration=TRUTH):
    """An evaluated file without a reference, and an external reference file."""
    data = write(tmp_path / "data.h5", n, d, None, **(data or {}))
    other = write(tmp_path / "ref.h5", n, None, c, declaration=declaration, **(reference or {}))
    return data, other


def external(monkeypatch, tmp_path, data, other, *extra):
    return evaluate(monkeypatch, tmp_path, data, "--clean", str(other), *extra)


def refuse_external(monkeypatch, capsys, data, other, *extra):
    return refuse(monkeypatch, capsys, "evaluate", "-d", str(data), "--clean", str(other), *extra)


# --------------------------------------------------------------------------------- 10


def test_a_same_file_truth_verifies_every_applicable_check(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "t.h5", n, d, c, declaration=TRUTH)
    capsys.readouterr()
    m = evaluate(monkeypatch, tmp_path, path)
    ctx = context(m)
    assert ctx["alignment_verified"] == ["shape", "energy", "units", "rows"]
    assert ctx["alignment_asserted"] == []
    assert ctx["alignment_not_applicable"] == ["angles", "times"]
    assert ctx["row_correspondence"] == "same_file" and ctx["shared_reference"] is False
    assert ("Alignment verified: shape, energy, units, rows (same file); asserted: none; "
            "not applicable: angles, times") in capsys.readouterr().out


@pytest.mark.parametrize("extra", [(), ROWS], ids=["no-assertion", "rows-asserted"])
def test_swapped_rows_are_detected_through_frame_index_within_one_acquisition(monkeypatch, capsys,
                                                                             tmp_path, extra):
    """Identifiers in one acquisition namespace are compared; an assertion never overrides
    the mismatch they detect."""
    n, d, c = arrays()
    data, other = pair(tmp_path, n, d, c, declaration=estimate(acquisition="acq-A"),
                       data={"acquisition_id": "acq-A", "frame_index": np.arange(6)},
                       reference={"frame_index": np.array([0, 2, 1, 3, 4, 5])})
    err = refuse_external(monkeypatch, capsys, data, other, *extra)
    assert "rows do not correspond" in err and "differs at 2 of 6 rows" in err
    assert "first at row 1: evaluated 1, reference 2" in err


def test_matching_identifiers_in_one_acquisition_verify_the_rows(monkeypatch, tmp_path):
    n, d, c = arrays()
    data, other = pair(tmp_path, n, d, c, declaration=estimate(acquisition="acq-A"),
                       data={"acquisition_id": "acq-A", "frame_index": np.arange(6) + 100},
                       reference={"frame_index": np.arange(6) + 100})
    ctx = context(external(monkeypatch, tmp_path, data, other))
    assert ctx["row_correspondence"] == "identifiers"
    assert ctx["alignment_verified"] == ["shape", "energy", "units", "rows"]
    assert ctx["alignment_asserted"] == []


@pytest.mark.parametrize("extra", [ROWS, ("--assert-alignment", "rows,energy")],
                         ids=["no-assertion", "energy-asserted"])
def test_mismatched_energy_grids_are_refused(monkeypatch, capsys, tmp_path, extra):
    n, d, c = arrays()
    grid = np.linspace(0, 1, n.shape[-1])
    data, other = pair(tmp_path, n, d, c, reference={"energy": grid + 1e-3})
    err = refuse_external(monkeypatch, capsys, data, other, *extra)
    assert "'energy' differs between the evaluated data and the reference" in err
    assert "largest difference 0.001" in err


def test_an_energy_grid_within_tolerance_is_verified(monkeypatch, tmp_path):
    n, d, c = arrays()
    grid = np.linspace(0, 1, n.shape[-1])
    data, other = pair(tmp_path, n, d, c, reference={"energy": grid + 1e-9})
    assert "energy" in context(external(monkeypatch, tmp_path, data, other, *ROWS))["alignment_verified"]


@pytest.mark.parametrize("axis", ["angles", "times"])
def test_a_reversed_coordinate_axis_is_refused(monkeypatch, capsys, tmp_path, axis):
    n, d, c = arrays3()
    coords = np.array([10.0, 20.0, 30.0])
    data, other = pair(tmp_path, n, d, c, data={axis: coords}, reference={axis: coords[::-1]})
    err = refuse_external(monkeypatch, capsys, data, other, *ROWS)
    assert f"'{axis}' differs" in err and "axis is reversed" in err


@pytest.mark.parametrize("axis", ["angles", "times"])
def test_a_matching_coordinate_axis_is_verified(monkeypatch, tmp_path, axis):
    n, d, c = arrays3()
    coords = np.array([10.0, 20.0, 30.0])
    data, other = pair(tmp_path, n, d, c, data={axis: coords}, reference={axis: coords})
    ctx = context(external(monkeypatch, tmp_path, data, other, *ROWS))
    assert axis in ctx["alignment_verified"]
    assert ctx["alignment_not_applicable"] == ["times" if axis == "angles" else "angles"]


@pytest.mark.parametrize("extra", [(), ("--assert-alignment", "units")])
def test_mismatched_units_are_refused(monkeypatch, capsys, tmp_path, extra):
    n, d, c = arrays()
    path = write(tmp_path / "u.h5", n, d, c, declaration=TRUTH, units="counts_per_s")
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(path), *extra)
    assert "units differ: the evaluated arrays are in 'counts_per_s', the reference is declared in 'counts'" in err


def test_units_that_differ_between_the_evaluated_arrays_are_refused(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "u.h5", n, d, c, declaration=TRUTH)
    with h5py.File(path, "a") as f:
        f["denoised"].attrs["intensity_units"] = "counts_per_s"
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(path))
    assert "units differ between the evaluated arrays" in err


def test_an_unknown_unit_string_is_refused(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "u.h5", n, d, c, declaration=TRUTH, units="apples")
    assert "'noisy.intensity_units' must be one of" in refuse(monkeypatch, capsys, "evaluate", "-d", str(path))


def test_partial_units_are_absent_not_a_mismatch(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "u.h5", n, d, c, declaration=TRUTH)
    with h5py.File(path, "a") as f:
        del f["denoised"].attrs["intensity_units"]
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(path))
    assert "units ('denoised' carries no intensity_units)" in err and "--assert-alignment units" in err
    ctx = context(evaluate(monkeypatch, tmp_path, path, "--assert-alignment", "units"))
    assert ctx["alignment_asserted"] == ["units"] and "units" not in ctx["alignment_verified"]


def test_a_single_spectrum_is_never_broadcast_without_the_flag(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    data, other = pair(tmp_path, n, d, c[:1])
    err = refuse_external(monkeypatch, capsys, data, other, *ROWS)
    assert "never broadcast" in err and "--shared-reference" in err


def test_a_shared_reference_needs_the_rows_assertion(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    data, other = pair(tmp_path, n, d, c[:1])
    err = refuse_external(monkeypatch, capsys, data, other, "--shared-reference")
    assert "rows (a shared reference carries no row identifiers)" in err


@pytest.mark.parametrize("shape", ["(1, e)", "(e,)"])
def test_a_legitimate_shared_reference_is_compared_with_every_row(monkeypatch, tmp_path, shape):
    n, d, c = arrays()
    single = c[:1] if shape == "(1, e)" else c[0]
    data = write(tmp_path / "data.h5", n, d, None)
    other = tmp_path / "ref.h5"
    with h5py.File(other, "w") as f:
        f.create_dataset("clean", data=single)
        from dnndenoiser import reference as ref
        ref.write_declaration(f["clean"], TRUTH)
        f.create_dataset("energy", data=np.linspace(0, 1, n.shape[-1]))
    m = external(monkeypatch, tmp_path, data, other, "--shared-reference", *ROWS)
    ctx = context(m)
    assert ctx["shared_reference"] is True and ctx["row_correspondence"] == "asserted"
    assert ctx["alignment_asserted"] == ["rows"]
    r = np.broadcast_to(c[0].astype(np.float64), n.shape)
    assert m["mse_in_mean"] == pytest.approx(np.mean((n.astype(np.float64) - r) ** 2), rel=1e-12)
    assert m["mse_out_mean"] == pytest.approx(np.mean((d.astype(np.float64) - r) ** 2), rel=1e-12)


def test_the_shared_flag_refuses_a_reference_that_is_not_one_spectrum(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    data, other = pair(tmp_path, n, d, c)
    err = refuse_external(monkeypatch, capsys, data, other, "--shared-reference", *ROWS)
    assert "--shared-reference needs a single spectrum of 16 points, got a reference of shape (6, 16)" in err


def test_an_external_reference_without_identifiers_needs_the_rows_assertion(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    data, other = pair(tmp_path, n, d, c)
    err = refuse_external(monkeypatch, capsys, data, other)
    assert "rows (the evaluated file carries no 'frame_index')" in err
    assert err.rstrip().endswith("--assert-alignment rows")
    ctx = context(external(monkeypatch, tmp_path, data, other, *ROWS))
    assert ctx["alignment_asserted"] == ["rows"] and ctx["row_correspondence"] == "asserted"
    assert ctx["alignment_verified"] == ["shape", "energy", "units"]


def test_identifiers_from_different_acquisitions_establish_nothing(monkeypatch, capsys, tmp_path):
    """Equal frame indices outside one namespace neither verify nor contradict; a correctly
    aligned separate acquisition is accepted with the assertion."""
    n, d, c = arrays()
    data, other = pair(tmp_path, n, d, c, declaration=estimate(acquisition="acq-B"),
                       data={"acquisition_id": "acq-A", "frame_index": np.arange(6) + 10},
                       reference={"frame_index": np.arange(6)})
    err = refuse_external(monkeypatch, capsys, data, other, "--overlap", "no_overlap_declared")
    assert "equal frame indices from different acquisitions establish nothing" in err
    m = external(monkeypatch, tmp_path, data, other, "--overlap", "no_overlap_declared", *ROWS)
    assert context(m)["row_correspondence"] == "asserted" and "agreement_db_change_mean" in m


@pytest.mark.parametrize("name, reason", [
    ("units", "was made from the metadata and verified"),
    ("energy", "was made from the metadata and verified"),
    ("rows", "was made from the metadata and verified"),
    ("angles", "nothing to assert: 'angles' does not apply to a layout of shape (6, 16)"),
    ("rows,bogus", "names checks among"),
    ("", "names checks among"),
])
def test_an_assertion_is_accepted_only_for_a_check_that_could_not_be_made(monkeypatch, capsys,
                                                                          tmp_path, name, reason):
    n, d, c = arrays()
    path = write(tmp_path / "t.h5", n, d, c, declaration=TRUTH)
    assert reason in refuse(monkeypatch, capsys, "evaluate", "-d", str(path), "--assert-alignment", name)


def test_an_angle_resolved_external_reference_missing_its_angles_needs_an_assertion(monkeypatch, capsys,
                                                                                   tmp_path):
    n, d, c = arrays3()
    data, other = pair(tmp_path, n, d, c, data={"angles": np.array([1.0, 2.0, 3.0])})
    err = refuse_external(monkeypatch, capsys, data, other)
    assert "angles (the reference carries no 'angles')" in err
    assert err.rstrip().endswith("--assert-alignment rows,angles")
    ctx = context(external(monkeypatch, tmp_path, data, other, "--assert-alignment", "angles,rows"))
    assert ctx["alignment_asserted"] == ["rows", "angles"]
    assert ctx["alignment_verified"] == ["shape", "energy", "units"]


def test_matching_angles_alone_do_not_verify_sample_order(monkeypatch, capsys, tmp_path):
    n, d, c = arrays3()
    angles = np.array([1.0, 2.0, 3.0])
    data, other = pair(tmp_path, n, d, c[::-1], data={"angles": angles}, reference={"angles": angles})
    err = refuse_external(monkeypatch, capsys, data, other)
    assert "rows (the evaluated file carries no 'frame_index')" in err


def test_identifiers_must_cover_every_non_energy_dimension(monkeypatch, capsys, tmp_path):
    """Frame identifiers verify the row axis; with the angle axis only asserted, the rows
    are not verified by identifiers and need their own assertion."""
    n, d, c = arrays3()
    angles = np.array([1.0, 2.0, 3.0])
    decl = estimate(acquisition="acq-A")
    data, other = pair(tmp_path, n, d, c, declaration=decl,
                       data={"acquisition_id": "acq-A", "frame_index": np.arange(4), "angles": angles},
                       reference={"frame_index": np.arange(4), "angles": angles})
    ctx = context(external(monkeypatch, tmp_path, data, other))
    assert ctx["row_correspondence"] == "identifiers" and ctx["alignment_verified"] == [
        "shape", "energy", "units", "rows", "angles"]
    data2, other2 = pair(tmp_path, n, d, c, declaration=decl,
                         data={"acquisition_id": "acq-A", "frame_index": np.arange(4), "angles": angles},
                         reference={"frame_index": np.arange(4)})
    err = refuse_external(monkeypatch, capsys, data2, other2, "--assert-alignment", "angles")
    assert "rows (the angles axis is not verified, so identifiers do not cover every non-energy dimension)" in err


def test_an_unnamed_coordinate_axis_must_be_named(monkeypatch, capsys, tmp_path):
    n, d, c = arrays3()
    path = write(tmp_path / "k.h5", n, d, c, declaration=TRUTH)
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(path))
    assert "the coordinate axis of this (rows, axis, energy) layout is unnamed" in err
    assert "the coordinate axis" in refuse(monkeypatch, capsys, "evaluate", "-d", str(path),
                                           "--assert-alignment", "angles,times")
    ctx = context(evaluate(monkeypatch, tmp_path, path, "--assert-alignment", "times"))
    assert ctx["alignment_asserted"] == ["times"] and ctx["alignment_not_applicable"] == ["angles"]
    assert ctx["alignment_verified"] == ["shape", "energy", "units", "rows"]


def test_a_three_dimensional_layout_with_both_axes_is_refused(monkeypatch, capsys, tmp_path):
    n, d, c = arrays3()
    path = write(tmp_path / "k.h5", n, d, c, declaration=TRUTH, angles=np.arange(3.0),
                 times=np.arange(3.0))
    assert "carries both 'angles' and 'times'" in refuse(monkeypatch, capsys, "evaluate", "-d", str(path))


@pytest.mark.parametrize("field, value, reason", [
    ("angles", np.arange(5.0), "'angles' of the evaluated file has 5 values but its angles axis has 3"),
    ("energy", np.arange(10.0), "'energy' of the evaluated file has 10 values but its spectra have 16 points"),
    ("frame_index", np.arange(3), "'frame_index' of the evaluated file has 3 values but its row axis has 4"),
    ("frame_index", np.arange(4.0), "'frame_index' of the evaluated file must hold integers"),
    ("angles", np.ones((3, 1)), "'angles' of the evaluated file must be a one-dimensional numeric array"),
])
def test_metadata_that_contradicts_the_arrays_is_refused(monkeypatch, capsys, tmp_path, field, value, reason):
    n, d, c = arrays3()
    kwargs = {"angles": np.arange(3.0), field: value}
    path = write(tmp_path / "k.h5", n, d, c, declaration=TRUTH, **kwargs)
    assert reason in refuse(monkeypatch, capsys, "evaluate", "-d", str(path))


def test_four_dimensional_data_has_times_then_angles(monkeypatch, capsys, tmp_path):
    rng = np.random.default_rng(4)
    c = rng.uniform(1, 2, (2, 3, 4, 8)).astype(np.float32)
    n, d = c + 0.1, c + 0.05
    times, angles = np.arange(3.0), np.arange(4.0) * 10
    path = write(tmp_path / "f.h5", n, d, c, declaration=TRUTH, times=times, angles=angles)
    ctx = context(evaluate(monkeypatch, tmp_path, path))
    assert ctx["alignment_verified"] == ["shape", "energy", "units", "rows", "angles", "times"]
    assert ctx["alignment_not_applicable"] == []
    data, other = pair(tmp_path, n, d, c, data={"times": times}, reference={"times": times, "angles": angles})
    err = refuse_external(monkeypatch, capsys, data, other, *ROWS)
    assert "angles (the evaluated file carries no 'angles')" in err
    bad = write(tmp_path / "g.h5", n, d, c, declaration=TRUTH, times=angles, angles=times)
    assert "'times' of the evaluated file has 4 values but its times axis has 3" in refuse(
        monkeypatch, capsys, "evaluate", "-d", str(bad))


def test_more_than_two_coordinate_axes_are_refused(monkeypatch, capsys, tmp_path):
    c = np.ones((2, 2, 2, 2, 8), dtype=np.float32)
    path = write(tmp_path / "x.h5", c + 0.1, c, c, declaration=TRUTH)
    assert "more than two coordinate axes" in refuse(monkeypatch, capsys, "evaluate", "-d", str(path))


def test_a_non_finite_coordinate_is_refused(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    grid = np.linspace(0, 1, n.shape[-1])
    grid[3] = np.nan
    data, other = pair(tmp_path, n, d, c, reference={"energy": grid})
    assert "'energy' of the reference contains non-finite values" in refuse_external(
        monkeypatch, capsys, data, other, *ROWS)


def test_a_missing_energy_axis_needs_an_assertion(monkeypatch, capsys, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "e.h5", n, d, c, declaration=TRUTH)
    with h5py.File(path, "a") as f:
        del f["energy"]
    assert "energy (the file carries no 'energy')" in refuse(monkeypatch, capsys, "evaluate", "-d", str(path))
    ctx = context(evaluate(monkeypatch, tmp_path, path, "--assert-alignment", "energy"))
    assert ctx["alignment_asserted"] == ["energy"] and ctx["alignment_verified"] == ["shape", "units", "rows"]


def test_legacy_output_needs_the_units_assertion_and_reports_alignment(monkeypatch, capsys, tmp_path):
    g = json.loads((DATA / "legacy_evaluate_fixture.json").read_text(encoding="utf-8"))
    path = write(tmp_path / "legacy.h5", np.asarray(g["noisy"]), np.asarray(g["denoised"]),
                 np.asarray(g["clean"]), units=None, energy=np.asarray(g["energy"], dtype=np.float32))
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(path), "--legacy-output")
    assert "units (the evaluated arrays carry no intensity_units; the reference is undeclared" in err
    ctx = context(evaluate(monkeypatch, tmp_path, path, "--legacy-output", "--assert-alignment", "units"))
    assert ctx["alignment_asserted"] == ["units"]
    assert ctx["alignment_verified"] == ["shape", "energy", "rows"]
    assert ctx["digests"]["format"] == "dnd-digest-1"


def test_generate_infer_evaluate_verifies_angles_without_assertions(monkeypatch, tmp_path):
    data, model, out = tmp_path / "d.h5", tmp_path / "m.pt", tmp_path / "o.h5"
    run(monkeypatch, "generate", "-o", str(data), "-n", "4", "--n-energy", "32",
        "--peak-set", "C1s_single", "--n-angles", "3")
    run(monkeypatch, "train", "-d", str(data), "-o", str(model), "--arch", "FCNN", "--epochs", "1",
        "--batch-size", "8", "--device", "cpu")
    run(monkeypatch, "infer", "-d", str(data), "-m", str(model), "-o", str(out), "--device", "cpu")
    ctx = context(evaluate(monkeypatch, tmp_path, out))
    assert ctx["alignment_verified"] == ["shape", "energy", "units", "rows", "angles"]
    assert ctx["alignment_not_applicable"] == ["times"] and ctx["alignment_asserted"] == []


# ------------------------------------------------- review of f1b335a, fixed findings


@pytest.mark.parametrize("field, bad", [("energy", "nan"), ("angles", "inf")])
def test_a_same_file_non_finite_coordinate_is_refused(monkeypatch, capsys, tmp_path, field, bad):
    """Finding 1: finiteness was checked only when comparing two files."""
    n, d, c = arrays3()
    angles = np.array([1.0, 2.0, 3.0])
    energy = np.linspace(0, 1, 16)
    if field == "energy":
        energy[3] = np.nan
    else:
        angles[1] = np.inf
    path = write(tmp_path / "f.h5", n, d, c, declaration=TRUTH, angles=angles, energy=energy)
    assert f"'{field}' of the evaluated file contains non-finite values" in refuse(
        monkeypatch, capsys, "evaluate", "-d", str(path))


@pytest.mark.parametrize("case", ["truth-no-ids", "estimate-evaluated-without-id"])
def test_identifiers_need_a_namespace_on_both_sides(monkeypatch, capsys, tmp_path, case):
    """Finding 2: equal frame indices without an acquisition namespace establish nothing,
    even when neither side has one (None is not a namespace)."""
    n, d, c = arrays()
    if case == "truth-no-ids":
        data, other = pair(tmp_path, n, d, c[::-1], data={"frame_index": np.arange(6)},
                           reference={"frame_index": np.arange(6)})
    else:
        data, other = pair(tmp_path, n, d, c[::-1], declaration=estimate(acquisition="acq-A"),
                           data={"frame_index": np.arange(6)}, reference={"frame_index": np.arange(6)})
    err = refuse_external(monkeypatch, capsys, data, other)
    assert "equal frame indices from different acquisitions establish nothing" in err


def test_noisy_and_denoised_must_have_the_same_shape(monkeypatch, capsys, tmp_path):
    """Finding 3: the check existed, no test pinned it."""
    n, d, c = arrays()
    path = write(tmp_path / "s.h5", n, d[:1], c, declaration=TRUTH)
    assert "shapes differ: noisy (6, 16), denoised (1, 16)" in refuse(
        monkeypatch, capsys, "evaluate", "-d", str(path))


def test_the_reference_digest_uses_the_coordinates_carried_with_the_reference(monkeypatch, tmp_path):
    """Finding 4: an external reference with its own coordinates (same values as the
    evaluated file's, other dtypes) hashes its own, not the evaluated file's."""
    n, d, c = arrays3(n=2, k=3, e=8)
    angles = np.array([5.0, 15.0, 25.0])
    decl = estimate(acquisition="acq-A")
    data = write(tmp_path / "data.h5", n, d, None, acquisition_id="acq-A",
                 frame_index=np.array([7, 9], dtype=np.int64), angles=angles,
                 energy=np.linspace(0, 1, 8))
    other = write(tmp_path / "ref.h5", n, None, c, declaration=decl,
                  frame_index=np.array([7, 9], dtype=np.int32), angles=angles.astype(np.float32),
                  energy=np.linspace(0, 1, 8).astype(np.float32))
    m = external(monkeypatch, tmp_path, data, other)
    assert context(m)["row_correspondence"] == "identifiers"
    with h5py.File(other) as g, h5py.File(data) as f:
        stored = {"reference_origin": json.loads(g["clean"].attrs["reference_origin"]),
                  "reference_schema_version": 1}
        own = _digest(reference_parts(g["clean"][:], g["energy"][:], g["angles"][:], None,
                                      g["frame_index"][:], stored))
        borrowed = _digest(reference_parts(g["clean"][:], f["energy"][:], f["angles"][:], None,
                                           f["frame_index"][:], stored))
    assert own != borrowed
    assert context(m)["digests"]["reference"] == {"sha256": own, "absent": ["times"]}


@pytest.mark.parametrize("offset, accepted", [(2e-6, False), (5e-7, True)])
def test_the_energy_tolerance_is_one_millionth(monkeypatch, capsys, tmp_path, offset, accepted):
    """Finding 5: the tolerance's value was not pinned."""
    n, d, c = arrays()
    grid = np.linspace(0, 1, n.shape[-1])
    data, other = pair(tmp_path, n, d, c, reference={"energy": grid + offset})
    if accepted:
        assert "energy" in context(external(monkeypatch, tmp_path, data, other, *ROWS))["alignment_verified"]
    else:
        assert "'energy' differs" in refuse_external(monkeypatch, capsys, data, other, *ROWS)


@pytest.mark.parametrize("case", ["3-D data, 1-D reference with angles", "4-D data, 2-D reference with times"])
def test_a_reference_of_another_ndim_carrying_coordinates_is_a_shape_mismatch(monkeypatch, capsys,
                                                                             tmp_path, case):
    """Finding 6: the reference's axis lengths were indexed by the evaluated layout before
    the shapes were compared (an IndexError, or a refusal naming the wrong axis)."""
    if case.startswith("3-D"):
        n, d, c = arrays3()
        data = write(tmp_path / "data.h5", n, d, None, angles=np.array([1.0, 2.0, 3.0]))
        other = tmp_path / "ref.h5"
        with h5py.File(other, "w") as g:
            g.create_dataset("clean", data=c[0, 0])
            from dnndenoiser import reference as ref
            ref.write_declaration(g["clean"], TRUTH)
            g.create_dataset("energy", data=np.linspace(0, 1, 16))
            g.create_dataset("angles", data=np.array([1.0, 2.0, 3.0]))
        err = refuse_external(monkeypatch, capsys, data, other, *ROWS)
        assert "shapes differ: evaluated (4, 3, 16), reference (16,)" in err
    else:
        rng = np.random.default_rng(5)
        c = rng.uniform(1, 2, (2, 3, 4, 8)).astype(np.float32)
        data = write(tmp_path / "data.h5", c + 0.1, c + 0.05, None, times=np.arange(3.0),
                     angles=np.arange(4.0))
        other = write(tmp_path / "ref.h5", c[0, 0], None, c[0, 0], declaration=TRUTH,
                      times=np.arange(3.0))
        err = refuse_external(monkeypatch, capsys, data, other)
        assert "shapes differ: evaluated (2, 3, 4, 8), reference (4, 8)" in err
    assert "--shared-reference" in err


@pytest.mark.parametrize("field, value, reason", [
    ("energy", np.linspace(0, 1, 10), "'energy' of the reference has 10 values but its spectra have 16 points"),
    ("frame_index", np.arange(3), "'frame_index' of the reference has 3 values but its row axis has 6"),
])
def test_reference_side_coordinate_lengths_are_checked(monkeypatch, capsys, tmp_path, field, value, reason):
    """Finding 7: the check existed, no test pinned it."""
    n, d, c = arrays()
    data, other = pair(tmp_path, n, d, c, declaration=estimate(acquisition="acq-A"),
                       data={"acquisition_id": "acq-A", "frame_index": np.arange(6)},
                       reference={"frame_index": np.arange(6), field: value})
    assert reason in refuse_external(monkeypatch, capsys, data, other)


@pytest.mark.parametrize("case", ["string angles on 2-D data", "string frame_index on a shared reference"])
def test_a_non_numeric_coordinate_is_refused_even_where_its_axis_does_not_apply(monkeypatch, capsys,
                                                                             tmp_path, case):
    """Finding 8: an unvalidated dataset reached the digest and crashed it."""
    n, d, c = arrays()
    if case.startswith("string angles"):
        path = write(tmp_path / "a.h5", n, d, c, declaration=TRUTH)
        with h5py.File(path, "a") as f:
            f.create_dataset("angles", data=np.array(["a", "b"], dtype=h5py.string_dtype()))
        err = refuse(monkeypatch, capsys, "evaluate", "-d", str(path))
        assert "'angles' of the evaluated file must be a one-dimensional numeric array" in err
    else:
        data, other = pair(tmp_path, n, d, c[:1])
        with h5py.File(other, "a") as g:
            g.create_dataset("frame_index", data=np.array(["x"], dtype=h5py.string_dtype()))
        err = refuse_external(monkeypatch, capsys, data, other, "--shared-reference", *ROWS)
        assert "'frame_index' of the reference must be a one-dimensional numeric array" in err


def test_a_reference_carrying_both_axes_in_a_one_axis_layout_is_refused(monkeypatch, capsys, tmp_path):
    """Finding 9: the rule applied to the evaluated file only."""
    n, d, c = arrays3()
    coords = np.array([1.0, 2.0, 3.0])
    data, other = pair(tmp_path, n, d, c, data={"angles": coords}, reference={"angles": coords, "times": coords})
    assert "the reference carries both 'angles' and 'times'" in refuse_external(
        monkeypatch, capsys, data, other, *ROWS)


def test_a_repeated_assert_alignment_flag_accumulates(monkeypatch, tmp_path):
    """Finding 11: argparse kept only the last value."""
    n, d, c = arrays()
    data, other = pair(tmp_path, n, d, c)
    with h5py.File(data, "a") as f:
        del f["noisy"].attrs["intensity_units"], f["denoised"].attrs["intensity_units"]
    ctx = context(external(monkeypatch, tmp_path, data, other, "--assert-alignment", "rows",
                           "--assert-alignment", "units"))
    assert ctx["alignment_asserted"] == ["units", "rows"]
    assert al.parse_assertions(["rows", "units,energy"]) == {"rows", "units", "energy"}


# ------------------------------------- follow-up review of 6b6d55f, fixed findings


@pytest.mark.parametrize("index, reason", [
    (np.array([0.5]), "'frame_index' of the reference must hold integers"),
    (np.arange(4), "'frame_index' of the reference has 4 values but its row axis has 1"),
])
def test_a_shared_reference_frame_index_obeys_the_integer_and_length_rules(monkeypatch, capsys,
                                                                           tmp_path, index, reason):
    """Follow-up finding 1: a shared reference's frame_index passed only the type check."""
    n, d, c = arrays()
    data, other = pair(tmp_path, n, d, c[:1], reference={"frame_index": index})
    assert reason in refuse_external(monkeypatch, capsys, data, other, "--shared-reference", *ROWS)


def test_a_shape_mismatch_is_named_before_the_reference_both_axes_rule(monkeypatch, capsys, tmp_path):
    """Follow-up finding 2: a (16,) reference carrying angles and times against 3-D data is
    a shape mismatch, not a one-axis layout carrying both."""
    n, d, c = arrays3()
    coords = np.array([1.0, 2.0, 3.0])
    data = write(tmp_path / "data.h5", n, d, None, angles=coords)
    other = tmp_path / "ref.h5"
    with h5py.File(other, "w") as g:
        g.create_dataset("clean", data=c[0, 0])
        from dnndenoiser import reference as ref
        ref.write_declaration(g["clean"], TRUTH)
        g.create_dataset("energy", data=np.linspace(0, 1, 16))
        g.create_dataset("angles", data=coords)
        g.create_dataset("times", data=coords)
    err = refuse_external(monkeypatch, capsys, data, other, *ROWS)
    assert "shapes differ: evaluated (4, 3, 16), reference (16,)" in err
    assert "carries both" not in err


# ---------------------------------------------------- the check as a function


def test_parse_assertions():
    assert al.parse_assertions(None) == set()
    assert al.parse_assertions(" rows, units ") == {"rows", "units"}
    with pytest.raises(al.AlignmentError, match="names checks among"):
        al.parse_assertions("rows,")


def test_evaluated_units_decodes_bytes_and_names_the_absent_array():
    assert al.evaluated_units(b"counts", "counts") == ("counts", None)
    assert al.evaluated_units(None, None) == (None, "the evaluated arrays carry no intensity_units")
    assert al.evaluated_units("counts", None) == (None, "'denoised' carries no intensity_units")
    with pytest.raises(al.AlignmentError, match="units differ between the evaluated arrays"):
        al.evaluated_units("counts", "counts_per_s")


def test_the_result_lists_are_disjoint_and_ordered():
    side = al.Side(shape=(2, 3, 8), energy=np.arange(8.0), times=np.arange(3.0), units="counts",
                   frame_index=np.arange(2), acquisition_id="a")
    result = al.check(side, side, same_file=True, shared=False, asserted=set())
    assert result == {"verified": ["shape", "energy", "units", "rows", "times"], "asserted": [],
                      "not_applicable": ["angles"], "row_correspondence": "same_file"}
    assert al.report_line(result) == ("Alignment verified: shape, energy, units, rows (same file), "
                                      "times; asserted: none; not applicable: angles")


# --------------------------------------------------------------------------------- 11
# An independent transcription of dnd-digest-1 (design section 6.6), not the module under test.


def _frame(name, payload):
    body = b"" if payload is None else payload
    return name.encode("utf-8") + b"\x00" + struct.pack("<Q", len(body)) + body


def _array(a):
    a = np.asarray(a)
    if a.dtype.byteorder in ("=", ">"):
        a = a.astype(a.dtype.newbyteorder("<"))
    header = (a.dtype.name + "|" + ",".join(str(n) for n in a.shape)).encode("ascii")
    return header + b"\x00" + a.tobytes("C")


def _json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(parts):
    sha = hashlib.sha256()
    for name, payload in parts:
        sha.update(_frame(name, payload))
    return sha.hexdigest()


REF = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.5]], dtype=np.float32)
ENERGY = np.array([280.0, 281.0, 282.0])
INDEX = np.array([3, 4], dtype=np.int64)
BUNDLE = {"reference_origin": {"origin": "synthetic_truth", "generator": "gén 1", "units": "counts"},
          "reference_schema_version": 1}
NOISY = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 7.0]], dtype=np.float32)
DENOISED = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.6]], dtype=np.float32)

# Frozen with the implementation (2026-10-02): the transcription above and the module must
# both reproduce these.
FROZEN_REFERENCE = "70ba92971f62c349f623298810e9284b0842741019212c4d50cad9e053a03e0a"
FROZEN_EVALUATED = "da049835fa644bcee7f1216397bff8aa1912e3d11dcd14a24d74385779959b9b"
FROZEN_MINIMAL = "078e894e0bf00f2709932b110124bba93909ab5111076e1917d07db760d364e9"


def reference_parts(reference=REF, energy=ENERGY, angles=None, times=None, frame_index=INDEX,
                    bundle=BUNDLE):
    opt = lambda v: None if v is None else _array(v)                        # noqa: E731
    return [("reference", _array(reference)), ("energy", opt(energy)), ("angles", opt(angles)),
            ("times", opt(times)), ("frame_index", opt(frame_index)),
            ("declaration_stored", None if bundle is None else _json(bundle))]


def test_the_frozen_vectors_reproduce():
    assert _digest(reference_parts()) == FROZEN_REFERENCE
    assert dg.reference_digest(REF, energy=ENERGY, frame_index=INDEX,
                               declaration_stored=BUNDLE) == {"sha256": FROZEN_REFERENCE,
                                                              "absent": ["angles", "times"]}
    evaluated = [("noisy", _array(NOISY)), ("denoised", _array(DENOISED)), ("energy", None),
                 ("angles", None), ("times", None), ("frame_index", None)]
    assert _digest(evaluated) == FROZEN_EVALUATED
    assert dg.evaluated_digest(NOISY, DENOISED) == {
        "sha256": FROZEN_EVALUATED, "absent": ["energy", "angles", "times", "frame_index"]}
    one = np.array([7], dtype=np.uint8)
    assert _digest(reference_parts(one, None, frame_index=None, bundle=None)) == FROZEN_MINIMAL
    assert dg.reference_digest(one)["sha256"] == FROZEN_MINIMAL


def test_an_absent_component_is_framed_with_its_name_and_length_zero():
    assert dg.frame("angles", None) == b"angles\x00" + bytes(8)
    assert dg.frame("x", b"ab") == b"x\x00" + b"\x02" + bytes(7) + b"ab"
    assert dg.array_payload(np.array([1, 256], dtype=np.uint16)) == (
        b"uint16|2\x00" + b"\x01\x00" + b"\x00\x01")


@pytest.mark.parametrize("change", ["a byte of the array", "the dtype", "the shape", "an axis",
                                    "the frame index", "the stored declaration", "an axis made absent"])
def test_changing_any_one_component_changes_the_digest(change):
    base = dg.reference_digest(REF, energy=ENERGY, frame_index=INDEX, declaration_stored=BUNDLE)
    kwargs = dict(energy=ENERGY, frame_index=INDEX, declaration_stored=BUNDLE)
    reference = REF
    if change == "a byte of the array":
        reference = REF.copy()
        reference[1, 2] = np.float32(6.5000005)
    elif change == "the dtype":
        reference = REF.astype(np.float64)          # the same values
    elif change == "the shape":
        reference = REF.reshape(3, 2)               # the same bytes
    elif change == "an axis":
        kwargs["energy"] = ENERGY + 1e-9
    elif change == "the frame index":
        kwargs["frame_index"] = INDEX.astype(np.uint64)    # the same integers, another dtype
    elif change == "the stored declaration":
        kwargs["declaration_stored"] = {**BUNDLE, "reference_origin": {**BUNDLE["reference_origin"], "units": "counts_per_s"}}
    elif change == "an axis made absent":
        kwargs["energy"] = None
    changed = dg.reference_digest(reference, **kwargs)
    assert changed["sha256"] != base["sha256"]
    assert ("energy" in changed["absent"]) == (change == "an axis made absent")


def test_a_big_endian_array_has_the_little_endian_digest():
    little = dg.array_payload(np.array([1.5, -2.0], dtype="<f4"))
    assert dg.array_payload(np.array([1.5, -2.0], dtype=">f4")) == little
    assert little.startswith(b"float32|2\x00")


def test_an_array_that_cannot_be_digested_is_refused():
    with pytest.raises(TypeError, match="has no digest"):
        dg.array_payload(np.array(["a"], dtype=object))


def test_evaluate_digests_match_the_transcription_of_the_file(monkeypatch, tmp_path):
    n, d, c = arrays3(n=2, k=3, e=8)
    angles = np.array([5.0, 15.0, 25.0])
    index = np.array([7, 9], dtype=np.int32)
    path = write(tmp_path / "t.h5", n, d, c, declaration=TRUTH, angles=angles, frame_index=index,
                 acquisition_id="acq-A")
    m = evaluate(monkeypatch, tmp_path, path)
    with h5py.File(path) as f:
        stored = {"reference_origin": json.loads(f["clean"].attrs["reference_origin"]),
                  "reference_schema_version": int(f["clean"].attrs["reference_schema_version"])}
        energy = f["energy"][:]
        expect_ref = _digest(reference_parts(f["clean"][:], energy, angles, None, f["frame_index"][:], stored))
        expect_eval = _digest([("noisy", _array(f["noisy"][:])), ("denoised", _array(f["denoised"][:])),
                               ("energy", _array(energy)), ("angles", _array(angles)), ("times", None),
                               ("frame_index", _array(f["frame_index"][:]))])
    digests = context(m)["digests"]
    assert digests == {"format": "dnd-digest-1",
                       "evaluated": {"sha256": expect_eval, "absent": ["times"]},
                       "reference": {"sha256": expect_ref, "absent": ["times"]}}


def test_the_effective_cli_declaration_is_not_part_of_the_reference_digest(monkeypatch, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "u.h5", n, d, c)
    plain = context(evaluate(monkeypatch, tmp_path, path, "--assert-alignment", "units"))["digests"]
    declared = context(evaluate(monkeypatch, tmp_path, path, "--reference-origin", "synthetic_truth",
                                "--generator", "g", "--units", "counts"))["digests"]
    assert plain["reference"] == declared["reference"]
    assert "declaration_stored" in plain["reference"]["absent"]
    stored = write(tmp_path / "s.h5", n, d, c, declaration=TRUTH)
    with_bundle = context(evaluate(monkeypatch, tmp_path, stored))["digests"]
    assert with_bundle["reference"]["sha256"] != plain["reference"]["sha256"]
    assert with_bundle["reference"]["absent"] == ["angles", "times", "frame_index"]
    assert with_bundle["evaluated"] == plain["evaluated"]
    # An explicit undeclared bundle completed from the command line: the stored bundle is
    # hashed, the effective declaration is not.
    explicit = write(tmp_path / "x.h5", n, d, c, declaration={"origin": "undeclared"})
    as_stored = context(evaluate(monkeypatch, tmp_path, explicit, "--assert-alignment", "units"))["digests"]
    completed = context(evaluate(monkeypatch, tmp_path, explicit, "--reference-origin", "synthetic_truth",
                                 "--generator", "g", "--units", "counts"))["digests"]
    assert as_stored["reference"] == completed["reference"]
    assert as_stored["reference"]["absent"] == ["angles", "times", "frame_index"]
    assert as_stored["reference"]["sha256"] not in (plain["reference"]["sha256"], with_bundle["reference"]["sha256"])


def test_a_shared_reference_is_digested_as_stored(monkeypatch, tmp_path):
    n, d, c = arrays()
    data, other = pair(tmp_path, n, d, c[:1])
    digests = context(external(monkeypatch, tmp_path, data, other, "--shared-reference", *ROWS))["digests"]
    with h5py.File(other) as g:
        stored = {"reference_origin": json.loads(g["clean"].attrs["reference_origin"]),
                  "reference_schema_version": 1}
        expect = _digest(reference_parts(g["clean"][:], g["energy"][:], frame_index=None, bundle=stored))
    assert digests["reference"]["sha256"] == expect
