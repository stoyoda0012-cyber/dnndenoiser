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
over the whole page, and to further rules for the clearance Revision 11 proposes:
- its record keys equal `CLEARED_FOR_WHEN_TO_TRUST`: no fewer, and none besides;
- an `n:` number is `n:reg` and its magnitude is a design value read from the record, so
  an uncleared record value cannot be relabelled as a design value;
- no digit appears outside an anchored number, and no record number is more than 20 %
  from its value however it is rounded;
- the qualifiers in `PAGE_REQUIRED_PHRASES` are present in the visible text.
What passes regardless is listed in Revision 11, item 78: it is a reviewer's.
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


# Digits on the page that are not a quoted number: the core level's name, the metric's
# name and a method's name. Ordered-list markers are handled by `_without_list_markers`.
PAGE_LITERALS = re.compile(r"\bC 1s\b|\bM1\b|\bnoise2noise\b")
# A number counts as anchored only if the anchor follows it directly (a space is allowed
# only before a percent sign) and it is written plainly: no exponent, no leading dot.
ANCHORED_NUMBER = re.compile(r"(?<![\w.])[+\-−]?\d+(?:\.\d+)?(?:\s?%)?<!--[rn]:[^>]*?-->")
LIST_MARKER = re.compile(r"^(\s*)(\d+)\.\s")
# Dashes that render like a minus sign but are not one of the three signs parsed.
LOOKALIKE_SIGN = re.compile(r"[–—‒﹣－]\d")


def _prose(text: str) -> str:
    """The rendered text's source: no HTML comments except the anchors, and no
    link-reference definitions, which Markdown never shows."""
    text = re.sub(r"<!--(?![rn]:).*?-->", "", text, flags=re.S)
    return re.sub(r"^ {0,3}\[[^\]]+\]:.*$", "", text, flags=re.M)


def _without_list_markers(text: str) -> str:
    """Blank the marker of a real ordered list: one that starts at 1 after a blank line and
    counts up by one. A number at the start of a wrapped line inside a paragraph is not a
    list marker -- Markdown renders it as text -- and is left in place to be caught."""
    out, expected = [], None
    for line in text.splitlines():
        m = LIST_MARKER.match(line)
        if not line.strip():
            expected = 1
        elif m and expected is not None and int(m.group(2)) == expected:
            line = line[:m.end(1)] + " " * (m.end() - m.end(1)) + line[m.end():]
            expected += 1
            out.append(line)
            continue
        elif m or (expected == 1):
            expected = None if expected == 1 else expected
        out.append(line)
    return "\n".join(out)


def test_the_page_uses_design_values_only_as_registered(registry, record, page):
    design = registry.page_design_values(record)
    problems = []
    for lineno, token, _pct, kind, key in _occurrences(_prose(page)):
        if kind != "n":
            continue
        value, _half_unit = _quoted_value(token)
        if key != "reg":
            problems.append(f"line {lineno}: {token} is n:{key}; only n:reg is allowed on the page")
        elif "." not in token or not any(abs(abs(value) - d) < 1e-9 for d in design):
            problems.append(f"line {lineno}: {token} is marked n:reg but is not a design value "
                            "written out with its decimal")
    assert not problems, "\n".join(problems)


def test_no_digit_on_the_page_stands_outside_an_anchored_number(page):
    """Catches a signed, sentence-final or unit-glued count, a number with its anchor set off
    by a space, exponent and bare-decimal forms, and a sign written with a dash."""
    prose = _prose(page)
    lookalikes = [(n, line.strip()) for n, line in enumerate(prose.splitlines(), start=1)
                  if LOOKALIKE_SIGN.search(line)]
    assert not lookalikes, "a dash used as a sign before a number (line, text): " + repr(lookalikes)
    rest = PAGE_LITERALS.sub(" ", ANCHORED_NUMBER.sub(" ", _without_list_markers(prose)))
    rest = ANCHOR.sub(" ", rest)
    bare = [(n, line.strip()) for n, line in enumerate(rest.splitlines(), start=1)
            if re.search(r"\d", line)]
    assert not bare, "digits on the page outside an anchored number (line, text): " + repr(bare)


def test_every_record_number_on_the_page_is_within_a_fifth_of_its_value(registry, record, page):
    """Rounding may not change what a number says: '0' for 0.47 passes the precision check.

    A fifth, not a tenth, because an SD quoted to one significant figure (0.03 for 0.034)
    is legitimately 12 % off."""
    problems = []
    for lineno, token, _pct, kind, key in _occurrences(_prose(page)):
        if kind != "r" or key not in registry.CITATIONS:
            continue
        actual = float(registry.CITATIONS[key][1](record))
        quoted, _ = _quoted_value(token)
        if abs(quoted - actual) > 0.2 * abs(actual):
            problems.append(f"line {lineno}: {token} for {key!r} is more than 20 % from {actual!r}")
    assert not problems, "\n".join(problems)


def test_the_page_keeps_its_qualifiers(registry, page):
    visible = " ".join(ANCHOR.sub("", _prose(page)).split())
    missing = [p for p in registry.PAGE_REQUIRED_PHRASES if p not in visible]
    assert not missing, f"qualifiers the clearance requires, missing from the page: {missing}"


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


# The same, planted in the page. Each case names the check that must reject it -- not
# merely some check -- so a case cannot pass because an unrelated rule happened to fire.
# Revision 11, item 78, tabulates these; `test_item_78_table_matches_the_planted_cases`
# keeps that table and this list in step.
ANCHORED = "test_every_number_on_the_page_is_anchored"
MATCHES = "test_every_page_citation_matches_the_record"
CLEARED = "test_the_page_cites_exactly_what_revision_11_cleared"
DESIGN = "test_the_page_uses_design_values_only_as_registered"
DIGIT = "test_no_digit_on_the_page_stands_outside_an_anchored_number"
FIFTH = "test_every_record_number_on_the_page_is_within_a_fifth_of_its_value"
QUALIFIERS = "test_the_page_keeps_its_qualifiers"

M1 = "(M1 is the SNR gain.)"
NO_MECH = " No\n  mechanism is claimed."
PAGE_MUTATIONS = [
    ("page: anchor removed", ANCHORED, "about\n  1.8<!--r:R6.pos--> eV", "about\n  1.8 eV"),
    ("page: boundary mistyped", MATCHES, "0.47<!--r:R3.pos--> eV toward", "0.52<!--r:R3.pos--> eV toward"),
    ("page: rounded figure outside its precision", MATCHES, "about\n  0.5<!--r:R3.pos--> eV",
     "about\n  0.6<!--r:R3.pos--> eV"),
    ("page: an uncleared record number, correct and anchored", CLEARED, M1,
     M1 + " At zero shift it gained +11.4<!--r:A.gain.0--> dB."),
    ("page: a cleared sentence removed", CLEARED, " (1.8<!--r:R6.neg--> eV the other way;", " ("),
    ("page: an uncleared record number under another non-record reason", DESIGN, M1,
     M1 + " B beat D by +4.5<!--n:design--> dB."),
    ("page: an uncleared record number relabelled as a design value", DESIGN, M1,
     M1 + " At zero shift it gained +11.4<!--n:reg--> dB."),
    ("page: an uncleared difference passed off as an integer design value", DESIGN, M1,
     M1 + " B beat D by 4<!--n:reg--> dB."),
    ("page: an unanchored count", DIGIT, M1, M1 + " It gained 11 dB at zero shift."),
    ("page: a signed count", DIGIT, M1, M1 + " It gained +11 dB."),
    ("page: a sentence-final count", DIGIT, M1, M1 + " The gain in dB was 11."),
    ("page: a unit-glued count", DIGIT, M1, M1 + " It gained 11dB."),
    ("page: an anchored number set off by a space", DIGIT, "0.47<!--r:R3.pos--> eV toward",
     "9 <!--r:R3.pos--> eV toward"),
    ("page: an exponent form", DIGIT, M1, M1 + " It gained .13e1 dB."),
    ("page: an integer mantissa with an exponent", DIGIT, "0.47<!--r:R3.pos--> eV toward",
     "9e0<!--r:R3.pos--> eV toward"),
    ("page: a count at the start of a wrapped line", DIGIT, M1,
     M1 + " Its gain in dB was\n  11. Beyond"),
    ("page: a sign flipped with an en dash", DIGIT, "+0.63<!--r:R7.-1-->", "–0.63<!--r:R7.-1-->"),
    ("page: rounded until it says something else", FIFTH, "0.47<!--r:R3.pos--> eV toward",
     "0<!--r:R3.pos--> eV toward"),
    ("page: R7 rounded to full pinning", FIFTH, "−0.67<!--r:R7.+1-->", "−1<!--r:R7.+1-->"),
    ("page: a required qualifier deleted", QUALIFIERS, NO_MECH, ""),
    ("page: a required qualifier hidden in a comment", QUALIFIERS, NO_MECH,
     " <!-- No mechanism is claimed. -->"),
    ("page: a qualifier kept only in a link-reference definition", QUALIFIERS,
     (NO_MECH, "**Record:**"), ("", '[nm]: #record "No mechanism is claimed."\n\n**Record:**')),
]


def _page_check(name, registry, record, text):
    return {
        ANCHORED: lambda: test_every_number_on_the_page_is_anchored(text),
        MATCHES: lambda: test_every_page_citation_matches_the_record(registry, record, text),
        CLEARED: lambda: test_the_page_cites_exactly_what_revision_11_cleared(registry, text),
        DESIGN: lambda: test_the_page_uses_design_values_only_as_registered(registry, record, text),
        DIGIT: lambda: test_no_digit_on_the_page_stands_outside_an_anchored_number(text),
        FIFTH: lambda: test_every_record_number_on_the_page_is_within_a_fifth_of_its_value(
            registry, record, text),
        QUALIFIERS: lambda: test_the_page_keeps_its_qualifiers(registry, text),
    }[name]


@pytest.mark.parametrize("label,check,old,new", PAGE_MUTATIONS, ids=[m[0] for m in PAGE_MUTATIONS])
def test_a_planted_error_on_the_page_is_rejected(registry, record, page, label, check, old, new):
    _page_check(check, registry, record, page)()  # the unmutated page passes this check
    mutated = page
    for o, n in (zip(old, new) if isinstance(old, tuple) else [(old, new)]):
        assert o in mutated, f"mutation anchor for {label!r} no longer on the page"
        mutated = mutated.replace(o, n, 1)
    with pytest.raises(AssertionError):
        _page_check(check, registry, record, mutated)()


def test_item_78_table_matches_the_planted_cases(text=None):
    """Every claim in item 78's table names a check and planted cases that exist here, with
    that check; and every planted case and every page check appears in the table."""
    text = DOCUMENT.read_text(encoding="utf-8") if text is None else text
    table = text[text.index("<!-- item-78-table -->"):text.index("<!-- /item-78-table -->")]
    rows = [r for r in table.splitlines() if r.strip().startswith("|") and "`test_" in r]
    planted = {label: check for label, check, _o, _n in PAGE_MUTATIONS}
    seen_cases, seen_checks = set(), set()
    for row in rows:
        _claim, check_cell, cases_cell = [c.strip() for c in row.strip().strip("|").split("|")]
        check = check_cell.strip("`")
        cases = [c.strip().strip("`") for c in cases_cell.split(";")]
        for case in cases:
            assert planted.get(case) == check, f"item 78 row {row!r}: {case!r} is not a case of {check}"
        seen_cases.update(cases)
        seen_checks.add(check)
    assert seen_cases == set(planted), f"planted cases missing from item 78: {set(planted) - seen_cases}"
    assert seen_checks == {c for _l, c, _o, _n in PAGE_MUTATIONS}


TABLE_MUTATIONS = [
    ("a case renamed", "`page: a signed count`", "`page: a signed number`"),
    ("a case put under the wrong check",
     "| `test_every_number_on_the_page_is_anchored` | `page: anchor removed` |",
     "| `test_every_page_citation_matches_the_record` | `page: anchor removed` |"),
    ("a row dropped", "| a dash other than the three parsed signs used as a sign "
     "| `test_no_digit_on_the_page_stands_outside_an_anchored_number` "
     "| `page: a sign flipped with an en dash` |\n", ""),
]


@pytest.mark.parametrize("label,old,new", TABLE_MUTATIONS, ids=[m[0] for m in TABLE_MUTATIONS])
def test_a_wrong_item_78_table_is_rejected(label, old, new):
    text = DOCUMENT.read_text(encoding="utf-8")
    assert old in text, f"mutation anchor for {label!r} no longer in item 78"
    with pytest.raises(AssertionError):
        test_item_78_table_matches_the_planted_cases(text.replace(old, new, 1))
