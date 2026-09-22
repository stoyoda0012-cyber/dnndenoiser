"""The P2-A renderer must refuse a record it cannot reproduce from the raw runs.

`benchmarks/boundaries/position_shift/render_report.py` turns a record into the
boundary table, the verdict table and the evidence a reader acts on, so a record
that disagrees with its own `runs` array must never render.

This file exists because an independent audit tamper-tested the first version of
that guard field by field and it accepted 25 of 30 edits -- including flipping a
prediction's PASS to FAIL, rewriting a boundary median, and changing a
Holm-adjusted p from 1.3e-44 to 0.9. The verdict table, the most consequential
thing the report prints, was copied from the record unverified while the
renderer's docstring and the benchmark README both claimed everything was
recomputed. That is the same defect the reference benchmark's own record test was
written for, repeated here, and this is the test that would have caught it.

Each case tampers with a copy of the shipped record, one field at a time, and
requires the guard to reject it. They are the reason the claim in
`benchmarks/boundaries/position_shift/README.md` can be made at all.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_DIR = REPO_ROOT / "benchmarks" / "boundaries" / "position_shift"
RENDERER = BOUNDARY_DIR / "render_report.py"
RECORD = BOUNDARY_DIR / "results" / "position_shift_boundary.json"

pytestmark = pytest.mark.skipif(
    not RENDERER.is_file() or not RECORD.is_file(),
    reason="benchmarks/boundaries is a source-checkout tree, not part of the distribution",
)


@pytest.fixture(scope="module")
def renderer():
    spec = importlib.util.spec_from_file_location("_p2a_render_report", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


PRIMARY = "1000.0"
ARM = "A_narrow_2304"


def _prediction(rec, name):
    return rec["predictions"][name]


def _boundary(rec, arm=ARM, direction="positive"):
    return rec["boundaries"][PRIMARY][arm][direction]


def _aggregate(rec, delta="+0.00", arm=ARM):
    return rec["aggregates"][arm][PRIMARY][delta]


# Each entry is (label, mutation). The labels are the audit's own tamper set plus the
# fields the repaired guard newly covers.
TAMPERS = [
    ("prediction verdict flipped to PASS",
     lambda r: _prediction(r, "R5b").__setitem__("passed", True)),
    ("prediction verdict flipped to FAIL",
     lambda r: _prediction(r, "R3").__setitem__("passed", False)),
    ("R6 three-way verdict rewritten",
     lambda r: _prediction(r, "R6")["per_direction"]["positive"].__setitem__(
         "verdict", "beyond range")),
    ("R6 paired seed count",
     lambda r: _prediction(r, "R6")["per_direction"]["positive"].__setitem__("n_b_larger", 1)),
    ("R3 boundary median restated",
     lambda r: _prediction(r, "R3")["per_direction"]["positive"].__setitem__(
         "median_first_crossing_eV", 9.9)),
    ("Holm-adjusted p rewritten",
     lambda r: _prediction(r, "R2")["per_direction"]["+4.00"].__setitem__(
         "holm_adjusted_p", 0.9)),
    ("paired mean rewritten",
     lambda r: _prediction(r, "R1")["stats"].__setitem__("mean", -99.0)),
    ("Cohen's dz rewritten",
     lambda r: _prediction(r, "R1")["stats"].__setitem__("cohens_dz", 0.1)),
    ("R7 monotonicity series scrambled",
     lambda r: _prediction(r, "R7")["monotonicity"]["positive"].__setitem__(
         "abs_displacement_eV_at_1_1p5_2_3_4", [9.0, 1.0, 9.0, 1.0, 9.0])),
    ("R7 worst violation rewritten",
     lambda r: _prediction(r, "R7")["monotonicity"]["positive"].__setitem__(
         "worst_violation_eV", 5.0)),
    ("boundary first crossing",
     lambda r: _boundary(r).__setitem__("median_first_crossing_eV", 9.9)),
    ("boundary sustained crossing",
     lambda r: _boundary(r).__setitem__("median_sustained_crossing_eV", 9.9)),
    ("boundary Kaplan-Meier median",
     lambda r: _boundary(r).__setitem__("kaplan_meier_median_eV", 9.9)),
    ("boundary censored count",
     lambda r: _boundary(r).__setitem__("n_censored", 7)),
    ("boundary re-crossing count",
     lambda r: _boundary(r).__setitem__("n_seeds_re_crossing", 7)),
    ("boundary defined-seed count",
     lambda r: _boundary(r).__setitem__("n_seeds_defined", 3)),
    ("boundary unreliable flag raised",
     lambda r: _boundary(r).__setitem__("unreliable", True)),
    ("suspect-run rule flipped",
     lambda r: r["suspect_run_rule"].__setitem__("suspect", True)),
    ("aggregate gain mean",
     lambda r: _aggregate(r).__setitem__("snr_gain_db_mean", 99.0)),
    ("aggregate across-seed SD",
     lambda r: _aggregate(r).__setitem__("snr_gain_db_sd_across_seeds", 99.0)),
    ("aggregate degradation",
     lambda r: _aggregate(r, "+4.00").__setitem__("degradation_db_mean", 0.0)),
    ("aggregate bias-corrected displacement",
     lambda r: _aggregate(r, "+4.00").__setitem__(
         "argmax_displacement_bias_corrected_ev_mean", 0.0)),
    ("a single raw run",
     lambda r: r["runs"][500].__setitem__("snr_gain_db_mean", 99.0)),
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
    """The subtle case the first guard missed.

    It recomputed the binomial p FROM the stored count, so an edit that changed the
    count and its p together was internally consistent and passed. The count has to be
    re-derived from the per-seed series instead.
    """
    tampered = copy.deepcopy(record)
    sign = _prediction(tampered, "R1")["sign"]
    sign["n_favouring"] = 3
    sign["one_sided_binomial_p"] = renderer.binomial_one_sided(3, sign["n_seeds"])
    sign["passed"] = False
    with pytest.raises(renderer.RecordDisagreement):
        renderer.verify(tampered)
