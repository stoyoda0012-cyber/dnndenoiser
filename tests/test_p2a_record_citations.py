"""Every number quoted in the P2-A Record section is pinned to a stated source.

See `benchmarks/boundaries/position_shift/record_citations.py` for why. In short:
the Record section is prose, no renderer guard reaches it, and an audit found quoted
figures that were wrong or taken from a different cell than the sentence implied.
Checking that a value merely appears somewhere in the record would not catch the
second kind, so each occurrence carries its own anchor and each anchor names a
metric, a condition, a unit and a derivation.

Rules enforced here, per line of the Record section:
- every decimal or percentage is followed, before the next number, by an anchor
  `<!--r:KEY-->` (from the record) or `<!--n:KEY-->` (deliberately not from it);
- a trailing `<!--n:KEY-row-->` covers the unanchored numbers on that line only, and
  only for non-record keys -- it exists for rows of registered shift values;
- each `r:` value, recomputed from the record, equals the quoted text at the
  precision quoted;
- every `n:` key has a stated reason, and no `r:` key in the registry goes unused.

`docs/WHEN_TO_TRUST.md` quotes the same record to users and is held to the same rules,
over the whole page. It must also cite exactly the record keys Revision 11 cleared for
it, `CLEARED_FOR_WHEN_TO_TRUST` in the registry: no fewer, and none besides.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = REPO_ROOT / "benchmarks" / "boundaries" / "position_shift"
REGISTRY = BOUNDARY / "record_citations.py"
RECORD = BOUNDARY / "results" / "position_shift_boundary.json"
DOCUMENT = REPO_ROOT / "docs" / "preregistration" / "P2A-position-shift-boundary.md"
PAGE = REPO_ROOT / "docs" / "WHEN_TO_TRUST.md"

pytestmark = pytest.mark.skipif(
    not (REGISTRY.is_file() and RECORD.is_file() and DOCUMENT.is_file() and PAGE.is_file()),
    reason="benchmarks/ and docs/ are source-checkout trees, not part of the distribution",
)

NUMBER = re.compile(
    r"(?<![\w.#])(?P<num>[+\-−]?\d+(?:\.\d+)?(?:e[+\-−]?\d+)?)(?P<pct>\s?%)"
    r"|(?<![\w.#])(?P<dec>[+\-−]?\d+\.\d+(?:e[+\-−]?\d+)?)"
)
ANCHOR = re.compile(r"<!--([rn]):([^>]+?)-->")
ANCHORED_INTEGER = re.compile(r"(?<![\w.#])(?P<dec>[+\-−]?\d+)(?=<!--[rn]:)")


@pytest.fixture(scope="module")
def registry():
    spec = importlib.util.spec_from_file_location("_p2a_citations", REGISTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def section():
    text = DOCUMENT.read_text(encoding="utf-8")
    return text[text.index("\n## Record\n"):]


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


def _quoted_value(token: str) -> tuple[float, float]:
    """The quoted number and half a unit of its last quoted digit."""
    token = token.replace("−", "-").strip()
    mantissa, _, exponent = token.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    scale = 10.0 ** int(exponent) if exponent else 1.0
    return float(mantissa) * scale, 0.5 * 10.0 ** (-decimals) * scale


def _occurrences(section: str):
    """(line number, token, is_percent, kind, key) for every number in the section."""
    out = []
    for lineno, line in enumerate(section.splitlines(), start=1):
        row_key = None
        stripped = line.rstrip()
        trailing = re.search(r"<!--n:([^>]+?)-row-->\s*$", stripped)
        if trailing:
            row_key = trailing.group(1)
        blanked = ANCHOR.sub(lambda a: " " * len(a.group(0)), line)
        events = [(m.start(), "num", m) for m in NUMBER.finditer(blanked)]
        spans = [m.span() for m in NUMBER.finditer(blanked)]
        # An integer is a count and is not checked -- unless an anchor follows it directly,
        # which says it is a quoted value (for example a boundary given in bins). Digits
        # inside a number already matched -- the exponent of 5.3e-4 -- are not a second one.
        events += [(m.start(), "num", m) for m in ANCHORED_INTEGER.finditer(line)
                   if not any(a <= m.start() < b for a, b in spans)]
        events += [(m.start(), "anchor", m) for m in ANCHOR.finditer(line)
                   if not m.group(2).endswith("-row")]
        events.sort(key=lambda e: e[0])
        pending = None
        for _pos, kind, match in events:
            if kind == "num":
                if pending is not None:
                    out.append((lineno, pending[0], pending[1], None, None))
                groups = match.groupdict()
                token = groups.get("num") or groups.get("dec")
                pending = (token, bool(groups.get("pct")))
            else:
                if pending is not None:
                    out.append((lineno, pending[0], pending[1], match.group(1), match.group(2)))
                    pending = None
        if pending is not None:
            out.append((lineno, pending[0], pending[1], None, None))
        if row_key is not None:
            out = [(ln, tok, pct, kind or "n", key or f"{row_key}")
                   if ln == lineno and kind is None else (ln, tok, pct, kind, key)
                   for ln, tok, pct, kind, key in out]
    return out


def test_every_number_in_the_record_section_is_anchored(section):
    bare = [(ln, tok) for ln, tok, _pct, kind, _key in _occurrences(section) if kind is None]
    assert not bare, "numbers quoted without a source anchor (line, token): " + repr(bare)


def test_every_record_citation_matches_the_record(registry, record, section):
    problems = []
    for lineno, token, _pct, kind, key in _occurrences(section):
        if kind != "r":
            continue
        if key not in registry.CITATIONS:
            problems.append(f"line {lineno}: unknown citation key {key!r}")
            continue
        description, derive = registry.CITATIONS[key]
        actual = float(derive(record))
        quoted, half_unit = _quoted_value(token)
        if abs(actual - quoted) > half_unit + 1e-12:
            problems.append(f"line {lineno}: quoted {token} for {key!r} ({description}); "
                            f"the record gives {actual!r}")
    assert not problems, "\n".join(problems)


def test_every_non_record_number_states_its_reason(registry, section):
    unknown = sorted({key for _ln, _tok, _pct, kind, key in _occurrences(section)
                      if kind == "n" and key not in registry.NON_RECORD})
    assert not unknown, f"non-record numbers with no stated reason: {unknown}"


def _record_keys(text: str) -> set[str]:
    return {key for _ln, _tok, _pct, kind, key in _occurrences(text) if kind == "r"}


def test_no_citation_in_the_registry_goes_unused(registry, section, page):
    unused = sorted(set(registry.CITATIONS) - _record_keys(section) - _record_keys(page))
    assert not unused, f"registry keys neither the Record section nor the page cites: {unused}"


def test_every_number_on_the_page_is_anchored(page):
    test_every_number_in_the_record_section_is_anchored(page)


def test_every_page_citation_matches_the_record(registry, record, page):
    test_every_record_citation_matches_the_record(registry, record, page)


def test_every_non_record_number_on_the_page_states_its_reason(registry, page):
    test_every_non_record_number_states_its_reason(registry, page)


def test_the_page_cites_exactly_what_revision_11_cleared(registry, page):
    used = _record_keys(page)
    cleared = registry.CLEARED_FOR_WHEN_TO_TRUST
    assert used <= cleared, f"record keys on the page that were never cleared: {sorted(used - cleared)}"
    assert cleared <= used, f"cleared keys the page no longer cites: {sorted(cleared - used)}"


# The checks above are only worth anything if they fail on the errors they exist for.
# Each case plants one such error in a copy of the section and requires at least one
# check to reject it. The first two are the transcription errors this mechanism found
# in the previous draft when it was first run.
MUTATIONS = [
    ("transcription error in an SD", "0.011<!--r:R3.sd-->", "0.012<!--r:R3.sd-->"),
    ("transcription error in a percentage",
     "80 %<!--r:chk6.margin.pct-->", "85 %<!--r:chk6.margin.pct-->"),
    ("right number, wrong shift", "+6.8<!--r:A.gain.+0.25-->", "+6.8<!--r:A.gain.+0.50-->"),
    ("right metric, wrong direction", "**−17.1<!--r:R2.+4-->**", "**−17.1<!--r:R2.-4-->**"),
    ("anchor removed", "+4.5<!--r:R5b.BD--> dB**; **0/20", "+4.5 dB**; **0/20"),
    ("a new unsourced number", "No mechanism is claimed.",
     "No mechanism is claimed, beyond 0.9 of the effect."),
    ("non-record number without a reason", "30.9<!--n:history-->", "30.9<!--n:trust-me-->"),
    ("sign flipped", "−1.1<!--r:L10k.out.+4-->", "+1.1<!--r:L10k.out.+4-->"),
    ("an anchored integer altered", "26<!--r:R6.bins-->", "27<!--r:R6.bins-->"),
]


PAGE_TEXT = PAGE.read_text(encoding="utf-8") if PAGE.is_file() else ""


@pytest.mark.parametrize("label,old,new", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_a_planted_error_is_rejected(registry, record, section, label, old, new):
    assert old in section, f"mutation anchor for {label!r} no longer in the section"
    mutated = section.replace(old, new, 1)
    checks = [
        lambda: test_every_number_in_the_record_section_is_anchored(mutated),
        lambda: test_every_record_citation_matches_the_record(registry, record, mutated),
        lambda: test_every_non_record_number_states_its_reason(registry, mutated),
        lambda: test_no_citation_in_the_registry_goes_unused(registry, mutated, PAGE_TEXT),
    ]
    rejected = 0
    for check in checks:
        try:
            check()
        except AssertionError:
            rejected += 1
    assert rejected, f"planted error not caught: {label}"


# The same, planted in the page. The last two are what the Revision 11 check exists for:
# a number that is correct and anchored, but was never cleared for quotation here.
PAGE_MUTATIONS = [
    ("page: boundary mistyped", "0.47<!--r:R3.pos--> eV toward", "0.52<!--r:R3.pos--> eV toward"),
    ("page: rounded figure outside its precision", "about 0.5<!--r:R3.pos--> eV",
     "about 0.6<!--r:R3.pos--> eV"),
    ("page: anchor removed", "about 1.8<!--r:R6.pos--> eV", "about 1.8 eV"),
    ("page: an uncleared record number, correct and anchored", "No mechanism is claimed.",
     "No mechanism is claimed. At zero shift it gained +11.4<!--r:A.gain.0--> dB."),
    ("page: a cleared sentence removed", " (1.8<!--r:R6.neg--> eV the other way)", ""),
]


@pytest.mark.parametrize("label,old,new", PAGE_MUTATIONS, ids=[m[0] for m in PAGE_MUTATIONS])
def test_a_planted_error_on_the_page_is_rejected(registry, record, section, page, label, old, new):
    assert old in page, f"mutation anchor for {label!r} no longer on the page"
    mutated = page.replace(old, new, 1)
    checks = [
        lambda: test_every_number_on_the_page_is_anchored(mutated),
        lambda: test_every_page_citation_matches_the_record(registry, record, mutated),
        lambda: test_every_non_record_number_on_the_page_states_its_reason(registry, mutated),
        lambda: test_the_page_cites_exactly_what_revision_11_cleared(registry, mutated),
    ]
    rejected = 0
    for check in checks:
        try:
            check()
        except AssertionError:
            rejected += 1
    assert rejected, f"planted error not caught: {label}"
