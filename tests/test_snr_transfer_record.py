"""The P2-B renderer must refuse a record it cannot reproduce from the per-seed gains.

`benchmarks/boundaries/snr_transfer/render_report.py` turns the record into the grids
and verdict tables a reader acts on. P2-A's first guard accepted 25 of 30 hand edits,
including a flipped verdict and a rewritten Holm p, because it copied what it printed
from the record unverified (`tests/test_position_shift_boundary_record.py`). This file
tampers with a copy of the P2-B record one field at a time, for every kind of field the
guard claims to cover, and requires the guard to refuse it.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_DIR = REPO_ROOT / "benchmarks" / "boundaries" / "snr_transfer"
RENDERER = BOUNDARY_DIR / "render_report.py"
RECORD = BOUNDARY_DIR / "results" / "snr_transfer.json"
REPORT = BOUNDARY_DIR / "report.md"

pytestmark = pytest.mark.skipif(
    not RENDERER.is_file() or not RECORD.is_file(),
    reason="benchmarks/boundaries is a source-checkout tree, not part of the distribution",
)


@pytest.fixture(scope="module")
def renderer():
    spec = importlib.util.spec_from_file_location("_p2b_render_report", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _pred(rec, family):
    return rec["predictions"][family]


def _n2c(rec, family):
    return rec["noise2clean_descriptive"][family]


def _agg(rec, method="moving_average", key="100.0->4.0"):
    return rec["aggregates"][method][key]


# (label, mutation, the reason the guard must give). The reason is pinned so that a case
# cannot pass because some other check happened to fire (docs/VERIFICATION.md section 1).
# Where both the independent recomputation and the consistency check would fire, the
# reason pinned is the independent one; `test_the_independent_path_alone_refuses` below
# shows that path works with the consistency check blinded.
TAMPERS = [
    ("R1 verdict flipped", lambda r: _pred(r, "R1").__setitem__("passed", False),
     r"predictions\.R1\.passed: recomputed True"),
    ("R2 verdict flipped", lambda r: _pred(r, "R2").__setitem__("passed", False),
     r"predictions\.R2\.passed: recomputed True"),
    ("R3 verdict flipped", lambda r: _pred(r, "R3").__setitem__("passed", False),
     r"predictions\.R3\.passed: recomputed True"),
    ("R4 verdict flipped", lambda r: _pred(r, "R4").__setitem__("passed", False),
     r"predictions\.R4\.passed: recomputed True"),
    ("R1 cell verdict flipped",
     lambda r: _pred(r, "R1")["cells"]["20.0"].__setitem__("passed", False),
     r"moving_average\.R1\[20\.0\]\.passed"),
    ("R3 cell sign count", lambda r: _pred(r, "R3")["cells"]["4.0->9.0"].__setitem__("n_favouring", 20),
     r"moving_average\.R3\[4\.0->9\.0\]\.n_favouring"),
    ("R4 cell binomial p", lambda r: _pred(r, "R4")["cells"]["4.0|9.0"].__setitem__(
        "one_sided_binomial_p", 0.9), r"moving_average\.R4\[4\.0\|9\.0\]\.one_sided_binomial_p"),
    ("R2 Holm-adjusted p", lambda r: _pred(r, "R2")["cells"]["9.0->4.0"].__setitem__(
        "holm_adjusted_p", 0.9), r"moving_average\.R2\[9\.0->4\.0\]\.holm_adjusted_p"),
    ("R2 threshold loosened", lambda r: _pred(r, "R2").__setitem__("k_required", 15),
     r"predictions\.R2\.k_required"),
    ("R2 cell threshold loosened",
     lambda r: _pred(r, "R2")["cells"]["9.0->4.0"].__setitem__("k_required", 15),
     r"moving_average\.R2\[9\.0->4\.0\]\.k_required"),
    ("R2 margin rewritten", lambda r: _pred(r, "R2").__setitem__("margin_db", -3.0),
     r"predictions\.R2\.margin_db"),
    ("R2 M1 beside a cell", lambda r: _pred(r, "R2")["m1_beside_each_cell"]["100.0->4.0"].__setitem__(
        "diagonal_m1_mean", 16.0), r"m1_beside_each_cell\.100\.0->4\.0\.diagonal_m1_mean"),
    ("a cell made descriptive by R1", lambda r: _pred(r, "R3").__setitem__(
        "cells_made_descriptive_by_R1", ["4.0->9.0"]), r"R3\.cells_made_descriptive_by_R1"),
    ("a level removed by R1", lambda r: _pred(r, "R1").__setitem__("failed_levels", [20.0]),
     r"R1\.failed_levels"),
    ("an R2 cell deleted", lambda r: _pred(r, "R2")["cells"].pop("9.0->4.0"),
     r"moving_average\.R2\[9\.0->4\.0\]\.n_favouring: recomputed 20, record says None"),
    ("a descriptive diagonal", lambda r: _pred(r, "R1")["descriptive_diagonals"]["4.0"].__setitem__(0, 99.0),
     r"descriptive_diagonals\.4\.0\[0\]"),
    ("noise2clean R2 count", lambda r: _n2c(r, "R2_cells")["100.0->4.0"].__setitem__("n_favouring", 20),
     r"noise2clean\.R2\[100\.0->4\.0\]\.n_favouring"),
    ("noise2clean R4 verdict", lambda r: _n2c(r, "R4_pairs")["45.0|100.0"].__setitem__("passed", True),
     r"noise2clean\.R4\[45\.0\|100\.0\]\.passed"),
    ("noise2clean Holm p", lambda r: _n2c(r, "R3_cells")["4.0->9.0"].__setitem__("holm_adjusted_p", 0.01),
     r"noise2clean\.R3\[4\.0->9\.0\]\.holm_adjusted_p"),
    ("aggregate M1 mean", lambda r: _agg(r).__setitem__("m1_mean", 9.9),
     r"aggregates\[moving_average\]\[100\.0->4\.0\]\.m1_mean"),
    ("aggregate M1 SD", lambda r: _agg(r).__setitem__("m1_sd", 0.1),
     r"aggregates\[moving_average\]\[100\.0->4\.0\]\.m1_sd"),
    ("aggregate M2 mean", lambda r: _agg(r).__setitem__("m2_mean", -1.0),
     r"aggregates\[moving_average\]\[100\.0->4\.0\]\.m2_mean"),
    ("aggregate M2 SD", lambda r: _agg(r, "noise2clean").__setitem__("m2_sd", 0.1),
     r"aggregates\[noise2clean\]\[100\.0->4\.0\]\.m2_sd"),
    ("aggregate per-seed list", lambda r: _agg(r)["m1_per_seed"].__setitem__(3, 0.0),
     r"aggregates\[moving_average\]\[100\.0->4\.0\]\.m1_per_seed"),
    ("an aggregate cell deleted", lambda r: r["aggregates"]["noise2clean"].pop("4.0->4.0"),
     r"aggregates\[noise2clean\]: cells differ"),
    ("noise2clean diagonal beside P2-A", lambda r: r["p2a_beside_noise2clean"].__setitem__(
        "noise2clean_lambda_100_diagonal_gain_db", 11.4), r"noise2clean_lambda_100_diagonal_gain_db"),
    ("P2-A value copied wrong", lambda r: r["p2a_beside_noise2clean"].__setitem__(
        "p2a_arm_A_level_1000_delta_0_gain_db", 5.0), r"P2-A's record"),
    ("differences beside P2-A emptied", lambda r: r["p2a_beside_noise2clean"].__setitem__(
        "differences_stated", []), r"differences_stated: not the registered list"),
    ("a raw per-seed gain", lambda r: r["seeds"][7]["gains_db"]["moving_average"].__setitem__(
        "4.0->100.0", 5.0), r"aggregates\[moving_average\]\[4\.0->100\.0\]\.m1_mean"),
    ("a seed dropped", lambda r: r["seeds"].pop(), r"seeds: indices"),
    ("a seed duplicated", lambda r: r["seeds"].__setitem__(1, copy.deepcopy(r["seeds"][0])),
     r"seeds: indices"),
    ("design levels", lambda r: r["design"].__setitem__("lambdas", [4.0, 9.0, 20.0, 45.0, 90.0]),
     r"design\.lambdas"),
    ("design frame counts", lambda r: r["design"]["frames_per_level"].__setitem__("100.0", 5000),
     r"design\.frames_per_level"),
    ("design seed count", lambda r: r["design"].__setitem__("n_seeds", 10), r"design\.n_seeds"),
    ("design pool size", lambda r: r["design"].__setitem__("noise2clean_pool", 461),
     r"design\.noise2clean_pool"),
    ("design architecture", lambda r: r["design"].__setitem__("architecture", "GRU"),
     r"design\.architecture \(registered\)"),
    ("design noise model", lambda r: r["design"].__setitem__("noise", "Gaussian approximation"),
     r"design\.noise \(registered\)"),
    ("design hidden units", lambda r: r["design"]["model_config"].__setitem__("num_hidden_units", 200),
     r"design\.model_config \(registered\)\.num_hidden_units"),
    ("design W", lambda r: r["design"].__setitem__("W", 2), r"design\.W \(registered\)"),
    ("design epochs", lambda r: r["design"].__setitem__("epochs", 5), r"design\.epochs \(registered\)"),
    ("design peak set", lambda r: r["design"].__setitem__("peak_set_id", "O1s"),
     r"design\.peak_set_id \(registered\)"),
    ("design normalisation", lambda r: r["design"]["moving_average_recipe"].__setitem__(
        "normalisation", "per-frame"), r"design\.moving_average_recipe \(registered\)\.normalisation"),
    ("design key added", lambda r: r["design"].__setitem__("augmentation", "none"),
     r"design: keys added \['augmentation'\]"),
    ("design confound rewritten", lambda r: r["design"].__setitem__("confound", "none"),
     r"design\.confound \(registered\)"),
    ("design generator", lambda r: r["design"]["generator_config"].__setitem__("position_jitter", 0.0),
     r"design\.generator_config \(registered\)\.position_jitter"),
]


@pytest.mark.parametrize("label,mutate,reason", TAMPERS, ids=[t[0] for t in TAMPERS])
def test_guard_refuses_tampered_record(renderer, record, label, mutate, reason):
    tampered = copy.deepcopy(record)
    mutate(tampered)
    with pytest.raises(renderer.RecordDisagreement, match=reason):
        renderer.verify(tampered)


# The tampers whose pinned reason comes from the independent recomputation. With the
# consistency check blinded -- the measurement functions made to echo whatever the record
# stores -- each must still be refused, so the independent path is shown to work alone.
INDEPENDENT = [t for t in TAMPERS if t[0] in {
    "R1 verdict flipped", "R3 cell sign count", "R4 cell binomial p", "R2 Holm-adjusted p",
    "R2 cell threshold loosened", "noise2clean R2 count", "noise2clean R4 verdict",
    "noise2clean Holm p", "an R2 cell deleted"}]


@pytest.mark.parametrize("label,mutate,reason", INDEPENDENT, ids=[t[0] for t in INDEPENDENT])
def test_the_independent_path_alone_refuses(renderer, record, monkeypatch, label, mutate, reason):
    tampered = copy.deepcopy(record)
    mutate(tampered)
    monkeypatch.setattr(renderer.measurement, "evaluate_predictions",
                        lambda seeds: copy.deepcopy(tampered["predictions"]))
    monkeypatch.setattr(renderer.measurement, "descriptive_statistics",
                        lambda seeds, method: copy.deepcopy(tampered["noise2clean_descriptive"]))
    with pytest.raises(renderer.RecordDisagreement, match=reason):
        renderer.verify(tampered)


def test_the_blinded_consistency_check_is_really_blind(renderer, record, monkeypatch):
    """The converse of the test above: with the measurement functions echoing the record, an
    edit only the consistency check covers passes -- so the blinding is real."""
    tampered = copy.deepcopy(record)
    _pred(tampered, "R3")["cells_made_descriptive_by_R1"] = ["4.0->9.0"]
    monkeypatch.setattr(renderer.measurement, "evaluate_predictions",
                        lambda seeds: copy.deepcopy(tampered["predictions"]))
    monkeypatch.setattr(renderer.measurement, "descriptive_statistics",
                        lambda seeds, method: copy.deepcopy(tampered["noise2clean_descriptive"]))
    renderer.verify(tampered)


# Holm's adjustment against values computed by hand, for both implementations: the
# renderer's and the one in boundary_common that wrote the record. Family: noise2clean's
# R2-type cells, p (by cell) 0.0207, 0.942, 2.0e-5, 1.0, 0.942, 0.00129, 1.0, 1.0, 0.994,
# 0.00129. Sorted: 2.0e-5 x 10, 0.00129 x 9, 0.00129 x 8 -> 0.0116 (monotone), 0.0207 x 7,
# then (m - i) * p >= 1 for the rest, capped at 1.
HOLM_P = [0.020694732666015625, 0.9423408508300781, 2.002716064453125e-05, 0.9999799728393555,
          0.9423408508300781, 0.0012884140014648438, 1.0, 0.9999799728393555, 0.9940910339355469,
          0.0012884140014648438]
HOLM_BY_HAND = [7 * 0.020694732666015625, 1.0, 10 * 2.002716064453125e-05, 1.0, 1.0,
                9 * 0.0012884140014648438, 1.0, 1.0, 1.0, 9 * 0.0012884140014648438]


def _boundary_common():
    spec = importlib.util.spec_from_file_location(
        "_boundary_common", REPO_ROOT / "benchmarks" / "boundaries" / "boundary_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("which", ["renderer", "boundary_common"])
def test_holm_matches_a_hand_computed_family(renderer, which):
    holm = renderer.holm_adjusted if which == "renderer" else _boundary_common().holm
    assert holm(HOLM_P) == pytest.approx(HOLM_BY_HAND, rel=1e-12)


@pytest.mark.parametrize("which", ["renderer", "boundary_common"])
def test_holm_refuses_bonferroni_and_the_unadjusted_p(renderer, which):
    """A wrong adjustment is told apart from the right one: Bonferroni (every p times m) and
    no adjustment both differ from the hand-computed values."""
    holm = renderer.holm_adjusted if which == "renderer" else _boundary_common().holm
    bonferroni = [min(1.0, 10 * p) for p in HOLM_P]
    assert bonferroni != pytest.approx(HOLM_BY_HAND, rel=1e-6)
    assert HOLM_P != pytest.approx(HOLM_BY_HAND, rel=1e-6)
    assert holm(HOLM_P) != pytest.approx(bonferroni, rel=1e-6)


def test_guard_accepts_the_shipped_record(renderer, record):
    """The converse. A guard that refuses everything would pass every test above."""
    renderer.verify(copy.deepcopy(record))


def test_sign_count_tampered_self_consistently_is_still_refused(renderer, record):
    """A count, its p, its Holm p and its verdict changed together are internally
    consistent; the count has to be re-derived from the per-seed gains to catch it."""
    tampered = copy.deepcopy(record)
    c = _pred(tampered, "R3")["cells"]["4.0->9.0"]
    c["n_favouring"] = 3
    c["one_sided_binomial_p"] = renderer.binomial_one_sided(3, 20)
    c["holm_adjusted_p"] = 1.0
    c["passed"] = False
    _pred(tampered, "R3")["passed"] = False
    with pytest.raises(renderer.RecordDisagreement, match=r"R3\[4\.0->9\.0\]\.n_favouring"):
        renderer.verify(tampered)


def test_the_committed_report_is_what_the_renderer_produces(renderer, record):
    """report.md is generated; a hand edit, or a record changed without re-rendering,
    leaves it different from what the renderer writes now."""
    assert REPORT.is_file(), "report.md is missing; run render_report.py --write"
    expected = renderer.render(record, renderer.verify(copy.deepcopy(record)))
    assert REPORT.read_text(encoding="utf-8") == expected


def test_a_hand_edited_report_is_detected(renderer, record):
    expected = renderer.render(record, renderer.verify(copy.deepcopy(record)))
    edited = expected.replace("| **PASS** | 10 | 17 |", "| **FAIL** | 10 | 17 |", 1)
    assert edited != expected
    assert edited != REPORT.read_text(encoding="utf-8")


def test_without_p2a_record_the_p2a_gain_is_unchecked_and_the_report_says_so(renderer, record,
                                                                              monkeypatch, tmp_path):
    """The guard's P2-A comparison is skipped when P2-A's record is absent: a wrong copied
    value then passes, and the report must say the value was not checked."""
    shipped = renderer.render(record, renderer.verify(copy.deepcopy(record)))
    assert "**Not checked:**" not in shipped
    monkeypatch.setattr(renderer, "P2A_RECORD", tmp_path / "absent.json")
    tampered = copy.deepcopy(record)
    tampered["p2a_beside_noise2clean"]["p2a_arm_A_level_1000_delta_0_gain_db"] = 5.0
    text = renderer.render(tampered, renderer.verify(tampered))
    assert "**Not checked:** P2-A's record was not present" in text
