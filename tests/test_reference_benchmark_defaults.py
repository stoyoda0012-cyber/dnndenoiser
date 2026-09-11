"""The reference benchmark's defaults must match the record it ships with.

The constants under test sit beneath a comment saying that changing them
invalidates comparison with an older record. They were once out of step with the
record, so the documented one-line invocation silently ran a different
experiment from the one the published numbers came from. These checks are cheap
and catch that class of drift.

The module is parsed rather than imported: it pulls in torch and the whole
package, which is far more than this needs, and `benchmarks/` is not part of the
built distribution, so the tests skip when it is absent.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarks" / "reference" / "reference_benchmark.py"
RECORD = REPO_ROOT / "benchmarks" / "reference" / "results" / "reference_benchmark.json"

pytestmark = pytest.mark.skipif(
    not SCRIPT.is_file() or not RECORD.is_file(),
    reason="benchmarks/reference is a source-checkout tree, not part of the distribution",
)


def _module_constants() -> dict[str, object]:
    """Return the script's module-level literal assignments, without importing it."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue
    return out


@pytest.fixture(scope="module")
def constants() -> dict[str, object]:
    return _module_constants()


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_train_size_default_matches_the_record(constants, record):
    per_level = record["design"]["data"]["n_train_spectra_per_noise_level"]
    assert per_level == 768
    assert constants["N_TRAIN_PER_LEVEL"] == per_level


def test_test_size_default_matches_the_record(constants, record):
    assert constants["N_TEST_PER_LEVEL"] == record["design"]["data"]["n_test_spectra_per_noise_level"]


def test_pooled_total_is_the_product_of_the_two_defaults(constants, record):
    n_levels = len(constants["NOISE_LEVELS"])
    expected = constants["N_TRAIN_PER_LEVEL"] * n_levels
    assert expected == 2304
    assert record["design"]["data"]["n_train_spectra_total_pooled"] == expected


def test_noise_levels_and_seed_count_match_the_record(constants, record):
    assert record["design"]["seeds"]["n_seeds"] == constants["N_SEEDS"]
    recorded_levels = {float(k) for k in record["aggregates"]["suggested-hyperparameters"]["FCNN"]}
    assert recorded_levels == set(constants["NOISE_LEVELS"])
