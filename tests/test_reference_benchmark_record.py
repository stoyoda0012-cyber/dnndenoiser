"""The reference benchmark's renderer must refuse a record it cannot reproduce.

`render_report.py` turns a record into the table, the paired comparison and the
strings a user eventually reads, so a record that disagrees with its own raw runs
must never render. An earlier version of `verify_record` checked only the
aggregate mean and the seed count while its docstring and the benchmark README
both claimed it recomputed everything; tampered standard deviations and an
entirely fabricated paired table rendered without complaint.

These tests tamper with a copy of the shipped record, one field at a time, and
require the guard to reject it. They are the reason the claim in the README can
be made at all.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER = REPO_ROOT / "benchmarks" / "reference" / "render_report.py"
RECORD = REPO_ROOT / "benchmarks" / "reference" / "results" / "reference_benchmark.json"

pytestmark = pytest.mark.skipif(
    not RENDERER.is_file() or not RECORD.is_file(),
    reason="benchmarks/reference is a source-checkout tree, not part of the distribution",
)


@pytest.fixture(scope="module")
def verify_record():
    spec = importlib.util.spec_from_file_location("_reference_render_report", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_record


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _first_aggregate(rec: dict) -> dict:
    condition = next(iter(rec["aggregates"]))
    arch = next(iter(rec["aggregates"][condition]))
    level = next(iter(rec["aggregates"][condition][arch]))
    return rec["aggregates"][condition][arch][level]


def _first_comparison(rec: dict) -> dict:
    condition = next(iter(rec["paired_vs_baseline"]))
    level = next(iter(rec["paired_vs_baseline"][condition]["by_noise_level"]))
    comparisons = rec["paired_vs_baseline"][condition]["by_noise_level"][level]["comparisons"]
    return comparisons[next(iter(comparisons))]


def test_the_shipped_record_verifies(verify_record, record):
    verify_record(copy.deepcopy(record))


@pytest.mark.parametrize(
    "field",
    [
        "snr_gain_db_mean",
        "snr_gain_db_sd_across_seeds",
        "input_snr_db_mean",
    ],
)
def test_tampering_an_aggregate_is_rejected(verify_record, record, field):
    tampered = copy.deepcopy(record)
    _first_aggregate(tampered)[field] = 0.01
    with pytest.raises(SystemExit):
        verify_record(tampered)


def test_tampering_the_per_seed_series_is_rejected(verify_record, record):
    tampered = copy.deepcopy(record)
    entry = _first_aggregate(tampered)
    entry["per_seed_snr_gain_db"] = [0.0] * len(entry["per_seed_snr_gain_db"])
    with pytest.raises(SystemExit):
        verify_record(tampered)


@pytest.mark.parametrize(
    "field, value",
    [
        ("mean_difference_db", 50.0),
        ("sd_difference_db", 0.0001),
        ("paired_t", 9999.0),
        ("p_value_raw", 0.0),
        ("p_value_holm_adjusted", 0.0),
        ("cohens_dz", 999.0),
        ("seeds_favouring_arch", 0),
    ],
)
def test_tampering_the_paired_table_is_rejected(verify_record, record, field, value):
    tampered = copy.deepcopy(record)
    _first_comparison(tampered)[field] = value
    with pytest.raises(SystemExit):
        verify_record(tampered)


def test_tampering_a_parameter_count_is_rejected(verify_record, record):
    tampered = copy.deepcopy(record)
    tampered["runs"][0]["n_trainable_parameters"] = 1
    with pytest.raises(SystemExit):
        verify_record(tampered)


def test_deleting_a_run_is_rejected(verify_record, record):
    tampered = copy.deepcopy(record)
    del tampered["runs"][0]
    with pytest.raises(SystemExit):
        verify_record(tampered)


def test_every_problem_is_reported_together(verify_record, record):
    """Accumulate-and-fail: one run must show every disagreement, not just the first."""
    tampered = copy.deepcopy(record)
    _first_aggregate(tampered)["snr_gain_db_mean"] = 0.01
    _first_comparison(tampered)["mean_difference_db"] = 50.0
    with pytest.raises(SystemExit) as excinfo:
        verify_record(tampered)
    message = str(excinfo.value)
    assert "snr_gain_db_mean" in message or "mean:" in message
    assert "mean_difference_db" in message
