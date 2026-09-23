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

pytestmark = pytest.mark.skipif(
    not (REGISTRY.is_file() and RECORD.is_file() and DOCUMENT.is_file()),
    reason="benchmarks/ and docs/ are source-checkout trees, not part of the distribution",
)

NUMBER = re.compile(
    r"(?<![\w.#])(?P<num>[+\-−]?\d+(?:\.\d+)?(?:e[+\-−]?\d+)?)(?P<pct>\s?%)"
    r"|(?<![\w.#])(?P<dec>[+\-−]?\d+\.\d+(?:e[+\-−]?\d+)?)"
)
ANCHOR = re.compile(r"<!--([rn]):([^>]+?)-->")


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
        events = [(m.start(), "num", m) for m in NUMBER.finditer(ANCHOR.sub(lambda a: " " * len(a.group(0)), line))]
        events += [(m.start(), "anchor", m) for m in ANCHOR.finditer(line)
                   if not m.group(2).endswith("-row")]
        events.sort(key=lambda e: e[0])
        pending = None
        for _pos, kind, match in events:
            if kind == "num":
                if pending is not None:
                    out.append((lineno, pending[0], pending[1], None, None))
                token = match.group("num") or match.group("dec")
                pending = (token, bool(match.group("pct")))
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


def test_no_citation_in_the_registry_goes_unused(registry, section):
    used = {key for _ln, _tok, _pct, kind, key in _occurrences(section) if kind == "r"}
    unused = sorted(set(registry.CITATIONS) - used)
    assert not unused, f"registry keys the Record section never cites: {unused}"


# The checks above are only worth anything if they fail on the errors they exist for.
# Each case plants one such error in a copy of the section and requires at least one
# check to reject it. The first two are the transcription errors this mechanism found
# in the previous draft when it was first run.
MUTATIONS = [
    ("transcription error in an SD", "0.011<!--r:R3.sd-->", "0.012<!--r:R3.sd-->"),
    ("transcription error in a percentage",
     "34 %<!--r:D.boundary.nearer-->", "35 %<!--r:D.boundary.nearer-->"),
    ("right number, wrong shift", "+6.83<!--r:A.gain.+0.25-->", "+6.83<!--r:A.gain.+0.50-->"),
    ("right metric, wrong direction", "**−17.10<!--r:R2.+4-->**", "**−17.10<!--r:R2.-4-->**"),
    ("anchor removed", "+4.47<!--r:R5b.BD--> dB**; **0/20", "+4.47 dB**; **0/20"),
    ("a new unsourced number", "No mechanism is claimed.",
     "No mechanism is claimed, beyond 0.9 of the effect."),
    ("non-record number without a reason", "0.0e+00<!--n:history-->", "0.0e+00<!--n:trust-me-->"),
    ("sign flipped", "−1.05<!--r:L10k.out.+4-->", "+1.05<!--r:L10k.out.+4-->"),
]


@pytest.mark.parametrize("label,old,new", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_a_planted_error_is_rejected(registry, record, section, label, old, new):
    assert old in section, f"mutation anchor for {label!r} no longer in the section"
    mutated = section.replace(old, new, 1)
    checks = [
        lambda: test_every_number_in_the_record_section_is_anchored(mutated),
        lambda: test_every_record_citation_matches_the_record(registry, record, mutated),
        lambda: test_every_non_record_number_states_its_reason(registry, mutated),
        lambda: test_no_citation_in_the_registry_goes_unused(registry, mutated),
    ]
    rejected = 0
    for check in checks:
        try:
            check()
        except AssertionError:
            rejected += 1
    assert rejected, f"planted error not caught: {label}"
