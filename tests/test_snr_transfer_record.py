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


TAMPERS = [
    ("R1 verdict flipped", lambda r: _pred(r, "R1").__setitem__("passed", False)),
    ("R2 verdict flipped", lambda r: _pred(r, "R2").__setitem__("passed", False)),
    ("R3 verdict flipped", lambda r: _pred(r, "R3").__setitem__("passed", False)),
    ("R4 verdict flipped", lambda r: _pred(r, "R4").__setitem__("passed", False)),
    ("R1 cell verdict flipped",
     lambda r: _pred(r, "R1")["cells"]["20.0"].__setitem__("passed", False)),
    ("R3 cell sign count", lambda r: _pred(r, "R3")["cells"]["4.0->9.0"].__setitem__("n_favouring", 20)),
    ("R4 cell binomial p", lambda r: _pred(r, "R4")["cells"]["4.0|9.0"].__setitem__(
        "one_sided_binomial_p", 0.9)),
    ("R2 Holm-adjusted p", lambda r: _pred(r, "R2")["cells"]["9.0->4.0"].__setitem__(
        "holm_adjusted_p", 0.9)),
    ("R2 threshold loosened", lambda r: _pred(r, "R2").__setitem__("k_required", 15)),
    ("R2 cell threshold loosened",
     lambda r: _pred(r, "R2")["cells"]["9.0->4.0"].__setitem__("k_required", 15)),
    ("R2 margin rewritten", lambda r: _pred(r, "R2").__setitem__("margin_db", -3.0)),
    ("R2 M1 beside a cell", lambda r: _pred(r, "R2")["m1_beside_each_cell"]["100.0->4.0"].__setitem__(
        "diagonal_m1_mean", 16.0)),
    ("a cell made descriptive by R1", lambda r: _pred(r, "R3").__setitem__(
        "cells_made_descriptive_by_R1", ["4.0->9.0"])),
    ("a level removed by R1", lambda r: _pred(r, "R1").__setitem__("failed_levels", [20.0])),
    ("an R2 cell deleted", lambda r: _pred(r, "R2")["cells"].pop("9.0->4.0")),
    ("a descriptive diagonal", lambda r: _pred(r, "R1")["descriptive_diagonals"]["4.0"].__setitem__(0, 99.0)),
    ("noise2clean R2 count", lambda r: _n2c(r, "R2_cells")["100.0->4.0"].__setitem__("n_favouring", 20)),
    ("noise2clean R4 verdict", lambda r: _n2c(r, "R4_pairs")["45.0|100.0"].__setitem__("passed", True)),
    ("noise2clean Holm p", lambda r: _n2c(r, "R3_cells")["4.0->9.0"].__setitem__("holm_adjusted_p", 0.01)),
    ("aggregate M1 mean", lambda r: _agg(r).__setitem__("m1_mean", 9.9)),
    ("aggregate M1 SD", lambda r: _agg(r).__setitem__("m1_sd", 0.1)),
    ("aggregate M2 mean", lambda r: _agg(r).__setitem__("m2_mean", -1.0)),
    ("aggregate M2 SD", lambda r: _agg(r, "noise2clean").__setitem__("m2_sd", 0.1)),
    ("aggregate per-seed list", lambda r: _agg(r)["m1_per_seed"].__setitem__(3, 0.0)),
    ("an aggregate cell deleted", lambda r: r["aggregates"]["noise2clean"].pop("4.0->4.0")),
    ("noise2clean diagonal beside P2-A", lambda r: r["p2a_beside_noise2clean"].__setitem__(
        "noise2clean_lambda_100_diagonal_gain_db", 11.4)),
    ("a raw per-seed gain", lambda r: r["seeds"][7]["gains_db"]["moving_average"].__setitem__(
        "4.0->100.0", 5.0)),
    ("a seed dropped", lambda r: r["seeds"].pop()),
    ("a seed duplicated", lambda r: r["seeds"].__setitem__(1, copy.deepcopy(r["seeds"][0]))),
    ("design levels", lambda r: r["design"].__setitem__("lambdas", [4.0, 9.0, 20.0, 45.0, 90.0])),
    ("design frame counts", lambda r: r["design"]["frames_per_level"].__setitem__("100.0", 5000)),
    ("design seed count", lambda r: r["design"].__setitem__("n_seeds", 10)),
    ("design pool size", lambda r: r["design"].__setitem__("noise2clean_pool", 461)),
]


@pytest.mark.parametrize("label,mutate", TAMPERS, ids=[label for label, _ in TAMPERS])
def test_guard_refuses_tampered_record(renderer, record, label, mutate):
    tampered = copy.deepcopy(record)
    mutate(tampered)
    with pytest.raises(renderer.RecordDisagreement):
        renderer.verify(tampered)


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
