"""Every number quoted in the P2-B Record section is pinned to a stated source.

The rules are P2-A's, and so is the parser: this file loads it from
`tests/test_p2a_record_citations.py` rather than keeping a second copy, so the two
records are held to one definition of "an anchored number". See
`benchmarks/boundaries/snr_transfer/record_citations.py` for what the anchors mean.

Rules enforced over the Record section of `docs/preregistration/P2B-snr-transfer.md`:
- every decimal or percentage is followed, before the next number, by an anchor
  `<!--r:KEY-->` (from the record) or `<!--n:KEY-->` (deliberately not from it); an
  integer is checked when an anchor follows it directly;
- each `r:` value, recomputed from the record, equals the quoted text at the precision
  quoted;
- every `n:` key has a stated reason, and every `r:` key in the registry is cited.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = REPO_ROOT / "benchmarks" / "boundaries" / "snr_transfer"
REGISTRY = BOUNDARY / "record_citations.py"
RECORD = BOUNDARY / "results" / "snr_transfer.json"
DOCUMENT = REPO_ROOT / "docs" / "preregistration" / "P2B-snr-transfer.md"
PARSER = Path(__file__).resolve().parent / "test_p2a_record_citations.py"

pytestmark = pytest.mark.skipif(
    not (REGISTRY.is_file() and RECORD.is_file() and DOCUMENT.is_file()),
    reason="benchmarks/ and docs/ are source-checkout trees, not part of the distribution",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parser():
    return _load("_p2a_citation_parser", PARSER)


@pytest.fixture(scope="module")
def registry():
    return _load("_p2b_citations", REGISTRY)


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def section():
    text = DOCUMENT.read_text(encoding="utf-8")
    return text[text.index("\n## Record\n"):]


def check_anchored(parser, text):
    bare = [(ln, tok) for ln, tok, _pct, kind, _key in parser._occurrences(text) if kind is None]
    assert not bare, "numbers quoted without a source anchor (line, token): " + repr(bare)


def check_matches(parser, registry, record, text):
    problems = []
    for lineno, token, _pct, kind, key in parser._occurrences(text):
        if kind != "r":
            continue
        if key not in registry.CITATIONS:
            problems.append(f"line {lineno}: unknown citation key {key!r}")
            continue
        description, derive = registry.CITATIONS[key]
        actual = float(derive(record))
        quoted, half_unit = parser._quoted_value(token)
        if abs(actual - quoted) > half_unit + 1e-12:
            problems.append(f"line {lineno}: quoted {token} for {key!r} ({description}); "
                            f"the record gives {actual!r}")
    assert not problems, "\n".join(problems)


def check_reasons(parser, registry, text):
    unknown = sorted({key for _ln, _tok, _pct, kind, key in parser._occurrences(text)
                      if kind == "n" and key not in registry.NON_RECORD})
    assert not unknown, f"non-record numbers with no stated reason: {unknown}"


def check_all_used(parser, registry, text):
    used = {key for _ln, _tok, _pct, kind, key in parser._occurrences(text) if kind == "r"}
    unused = sorted(set(registry.CITATIONS) - used)
    assert not unused, f"registry keys the Record section does not cite: {unused}"


def test_every_number_in_the_record_section_is_anchored(parser, section):
    check_anchored(parser, section)


def test_every_record_citation_matches_the_record(parser, registry, record, section):
    check_matches(parser, registry, record, section)


def test_every_non_record_number_states_its_reason(parser, registry, section):
    check_reasons(parser, registry, section)


def test_no_citation_in_the_registry_goes_unused(parser, registry, section):
    check_all_used(parser, registry, section)


# Each case plants one error in a copy of the section and names the check that must
# reject it -- not merely some check -- so a case cannot pass because an unrelated rule
# happened to fire.
ANCHORED, MATCHES, REASONS, USED = "anchored", "matches", "reasons", "used"
MUTATIONS = [
    ("transcription error in a mean", MATCHES,
     "−1.2<!--r:ma.m2.t4.i9--> ± 0.6", "−1.3<!--r:ma.m2.t4.i9--> ± 0.6"),
    ("right number, wrong cell", MATCHES,
     "+6.5<!--r:ma.m2.t45.i4-->", "+6.5<!--r:ma.m2.t45.i9-->"),
    ("right cell, wrong method", MATCHES,
     "−3.4<!--r:n2c.m2.t100.i4-->", "−3.4<!--r:ma.m2.t100.i4-->"),
    ("sign flipped", MATCHES, "+0.1<!--r:ma.m2.t9.i4-->", "−0.1<!--r:ma.m2.t9.i4-->"),
    ("an anchored count altered", MATCHES,
     "13<!--r:ma.m2pos.9.4-->/20", "17<!--r:ma.m2pos.9.4-->/20"),
    ("an SD mistyped", MATCHES, "± 0.9<!--r:ma.r4.4.9.sd-->", "± 0.5<!--r:ma.r4.4.9.sd-->"),
    ("anchor removed", ANCHORED, "+2.8<!--r:n2c.r4.hi--> dB", "+2.8 dB"),
    ("a new unsourced number", ANCHORED, "no mechanism is claimed.**",
     "no mechanism is claimed beyond 0.5 dB of it.**"),
    ("non-record number without a reason", REASONS, "3.5<!--n:reg-->", "3.5<!--n:trust-me-->"),
    ("the only citation of a key removed", USED,
     "+7.9<!--r:diag.n2c.minus.ma.4-->", "+7.9<!--n:reg-->"),
]


def _run(check, parser, registry, record, text):
    return {
        ANCHORED: lambda: check_anchored(parser, text),
        MATCHES: lambda: check_matches(parser, registry, record, text),
        REASONS: lambda: check_reasons(parser, registry, text),
        USED: lambda: check_all_used(parser, registry, text),
    }[check]()


@pytest.mark.parametrize("label,check,old,new", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_a_planted_error_is_rejected(parser, registry, record, section, label, check, old, new):
    _run(check, parser, registry, record, section)  # the unmutated section passes
    assert old in section, f"mutation anchor for {label!r} no longer in the section"
    with pytest.raises(AssertionError):
        _run(check, parser, registry, record, section.replace(old, new, 1))
