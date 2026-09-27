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
PAGE = REPO_ROOT / "docs" / "WHEN_TO_TRUST.md"
REGION_OPEN, REGION_CLOSE = "<!-- record:P2-B -->", "<!-- /record:P2-B -->"

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


def page_region(text: str) -> str:
    assert text.count(REGION_OPEN) == 1 and text.count(REGION_CLOSE) == 1, \
        "the page must hold exactly one P2-B region, opened and closed once"
    start, end = text.index(REGION_OPEN), text.index(REGION_CLOSE)
    assert start < end, "the P2-B region closes before it opens"
    return text[start + len(REGION_OPEN):end]


@pytest.fixture(scope="module")
def page():
    return page_region(PAGE.read_text(encoding="utf-8"))


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


def check_all_used(parser, registry, text, page_text=""):
    used = {key for _ln, _tok, _pct, kind, key in parser._occurrences(text + "\n" + page_text)
            if kind == "r"}
    unused = sorted(set(registry.CITATIONS) - used)
    assert not unused, f"registry keys neither the Record section nor the page cites: {unused}"


def test_every_number_in_the_record_section_is_anchored(parser, section):
    check_anchored(parser, section)


def test_every_record_citation_matches_the_record(parser, registry, record, section):
    check_matches(parser, registry, record, section)


def test_every_non_record_number_states_its_reason(parser, registry, section):
    check_reasons(parser, registry, section)


def test_no_citation_in_the_registry_goes_unused(parser, registry, section, page):
    check_all_used(parser, registry, section, page)


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
        USED: lambda: check_all_used(parser, registry, text, page_region(PAGE.read_text(encoding="utf-8"))),
    }[check]()


@pytest.mark.parametrize("label,check,old,new", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_a_planted_error_is_rejected(parser, registry, record, section, label, check, old, new):
    _run(check, parser, registry, record, section)  # the unmutated section passes
    assert old in section, f"mutation anchor for {label!r} no longer in the section"
    with pytest.raises(AssertionError):
        _run(check, parser, registry, record, section.replace(old, new, 1))


# ---------------------------------------------------------------------------------------
# The P2-B region of docs/WHEN_TO_TRUST.md. P2-A's page checks are reused unchanged (they
# take the registry and the text as arguments); the design-value check is P2-B's own,
# because P2-B's design values are integers that P2-A's rule would require written with
# a decimal.
# ---------------------------------------------------------------------------------------


def check_page_design_values(parser, registry, record, text):
    design = registry.page_design_values(record)
    problems = []
    for lineno, token, _pct, kind, key in parser._occurrences(parser._prose(text)):
        if kind != "n":
            continue
        value, _half = parser._quoted_value(token)
        if key != "reg":
            problems.append(f"line {lineno}: {token} is n:{key}; only n:reg is allowed on the page")
        elif not any(abs(abs(value) - d) < 1e-9 for d in design):
            problems.append(f"line {lineno}: {token} is marked n:reg but is not a P2-B design value")
    assert not problems, "\n".join(problems)


PAGE_CHECKS = {
    "anchored": lambda p, reg, rec, t: p.test_every_number_on_the_page_is_anchored(t),
    "matches": lambda p, reg, rec, t: p.test_every_page_citation_matches_the_record(reg, rec, t),
    "cleared": lambda p, reg, rec, t: p.test_the_page_cites_exactly_what_revision_11_cleared(reg, t),
    "design": lambda p, reg, rec, t: check_page_design_values(p, reg, rec, t),
    "digit": lambda p, reg, rec, t: p.test_no_digit_on_the_page_stands_outside_an_anchored_number(t),
    "fifth": lambda p, reg, rec, t: p.test_every_record_number_on_the_page_is_within_a_fifth_of_its_value(
        reg, rec, t),
    "qualifiers": lambda p, reg, rec, t: p.test_the_page_keeps_its_qualifiers(reg, t),
}


@pytest.mark.parametrize("check", list(PAGE_CHECKS))
def test_the_page_region_passes(parser, registry, record, page, check):
    PAGE_CHECKS[check](parser, registry, record, page)


C1S = "Synthetic C 1s spectra"
PAGE_MUTATIONS = [
    ("page: anchor removed", "anchored", "0.6<!--r:ma.R2.worst.loss--> dB", "0.6 dB"),
    ("page: a loss written against the signed key", "matches",
     "0.6<!--r:ma.R2.worst.loss-->", "0.6<!--r:ma.R2.min.m2-->"),
    ("page: a count mistyped", "matches", "16<!--r:ma.m1neg.4.100-->", "12<!--r:ma.m1neg.4.100-->"),
    ("page: right number, wrong key", "matches", "19<!--r:ma.R3.min.k-->", "19<!--r:ma.R4.min.k-->"),
    ("page: sign flipped", "digit", "−1.5<!--r:ma.m1.t4.i100-->", "–1.5<!--r:ma.m1.t4.i100-->"),
    ("page: an uncleared record number, correct and anchored", "cleared", C1S,
     C1S + " (the moving average's diagonal gain at λ = 4 was +9.9<!--r:ma.m1.t4.i4--> dB)"),
    ("page: a cleared statement removed", "cleared",
     ", in at least 18<!--r:ma.R4.min.k--> of\n  20<!--r:n.seeds--> runs", ""),
    ("page: a record value relabelled as a design value", "design", C1S,
     C1S + " (noise2clean gained 17.8<!--n:reg--> dB)"),
    ("page: an unanchored count", "digit", C1S, C1S + " (tested 25 cells)"),
    ("page: rounded until it says something else", "fifth",
     "0.6<!--r:ma.R2.worst.loss-->", "1<!--r:ma.R2.worst.loss-->"),
    ("page: the relative reading of a loss deleted", "qualifiers",
     " than a model trained at that count rate, in any", " in any"),
    ("page: a required qualifier deleted", "qualifiers",
     " None of these is separated from the others, or from\nthe signal-to-noise ratio.", ""),
]


@pytest.mark.parametrize("label,check,old,new", PAGE_MUTATIONS, ids=[m[0] for m in PAGE_MUTATIONS])
def test_a_planted_error_on_the_page_is_rejected(parser, registry, record, page, label, check, old, new):
    PAGE_CHECKS[check](parser, registry, record, page)
    assert old in page, f"mutation anchor for {label!r} no longer on the page"
    with pytest.raises(AssertionError):
        PAGE_CHECKS[check](parser, registry, record, page.replace(old, new, 1))


@pytest.mark.parametrize("broken", [
    lambda t: t.replace(REGION_CLOSE, "", 1),
    lambda t: t.replace(REGION_OPEN, "", 1),
    lambda t: t + "\n" + REGION_OPEN + "\n" + REGION_CLOSE + "\n",
], ids=["unclosed", "unopened", "duplicated"])
def test_a_malformed_region_is_rejected(broken):
    with pytest.raises(AssertionError):
        page_region(broken(PAGE.read_text(encoding="utf-8")))
