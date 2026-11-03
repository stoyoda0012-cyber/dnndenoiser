"""The AI-agent user guide ships with the package and names its version.

The guide is read from the installed package so an agent gets the copy
that matches the code, not the repository's main branch. Its version line
and its links must agree, it must never describe an older version than the
package it ships in, and it must keep the denoiser's scientific limits
(AGENTS.md §5) that it exists to carry.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

import dnndenoiser
from dnndenoiser.guides import AGENT_USER_GUIDE, read_guide


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", text).groups())


def _flat(text: str) -> str:
    """Whitespace collapsed, so a re-wrapped paragraph keeps its phrases."""
    return " ".join(text.split())


def _version_and_links_agree(text: str, package_version: str) -> bool:
    found = re.search(r"Written for: dnndenoiser v(\d+\.\d+\.\d+)", text)
    if not found:
        return False
    stated = found.group(1)
    links = re.findall(r"github\.com/[^>\s]+", text)
    return (bool(links)
            and all(re.search(rf"/(?:blob|tree)/v{re.escape(stated)}/", u) for u in links)
            and _version(stated) >= _version(package_version))


#: A measured number in a unit an evaluation reports in. AGENTS.md §6: such a number
#: belongs to its record, not to a second place.
_MEASURED = re.compile(r"(?i)\d[\d.,]*\s*(dB\b|decibels?|%|per\s?cent)|SNR\s+(of\s+)?[+-]?\d")


def _quotes_a_measured_number(text: str) -> bool:
    body = re.sub(r"https?://\S+|v\d+\.\d+\.\d+|§\d+", "", text)
    return bool(_MEASURED.search(body))


def test_the_guide_is_readable_from_the_package():
    text = read_guide()
    assert text.startswith("# Using dnndenoiser with an AI agent")
    assert read_guide(AGENT_USER_GUIDE) == text


def test_the_version_line_and_the_links_agree():
    assert _version_and_links_agree(read_guide(), dnndenoiser.__version__)


@pytest.mark.parametrize("wrong", [
    "Written for: dnndenoiser v9.9.9. <https://github.com/o/r/blob/main/README.md>",
    "Written for: dnndenoiser v9.9.9. <https://github.com/o/r/blob/v9.9.8/README.md>",
    "Written for: dnndenoiser v0.0.1. <https://github.com/o/r/blob/v0.0.1/README.md>",
    "No version line. <https://github.com/o/r/blob/v9.9.9/README.md>",
])
def test_the_version_check_rejects(wrong):
    assert not _version_and_links_agree(wrong, "0.1.0")


_LIMITS = ("not a measurement", "No pretrained model is shipped", "no reference-free SNR",
           "not an uncertainty of the spectrum", "not a held-out result",
           "worse than the input", "`unknown` is not held out", "--legacy-output")


def test_the_guide_keeps_the_denoiser_limits():
    text = _flat(read_guide())
    for phrase in _LIMITS:
        assert phrase in text, phrase


def test_the_guide_quotes_no_measured_number():
    assert not _quotes_a_measured_number(read_guide())


@pytest.mark.parametrize("planted", [
    "the model gained +16.3 dB", "a gain of 16dB", "16 DB", "16 decibels",
    "within 25 %", "25% better", "25 per cent", "25 percent", "an SNR of 16",
])
def test_the_number_check_rejects(planted):
    assert _quotes_a_measured_number(f"Some text. {planted}. More text.")


def test_only_shipped_guides_can_be_read():
    for name in ("../__init__.py", "/etc/hosts", "nope.md"):
        with pytest.raises(ValueError, match="no shipped guide"):
            read_guide(name)


def _run_cp932(args):
    env = {**os.environ, "PYTHONIOENCODING": "cp932"}
    return subprocess.run([sys.executable, *args], capture_output=True, env=env)


def test_python_m_prints_it_whatever_the_console_encoding():
    """A redirected stdout on a Japanese-locale Windows is cp932, which has no '—'."""
    out = _run_cp932(["-m", "dnndenoiser.guides"])
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    assert out.stdout.decode("utf-8") == read_guide()


def test_a_plain_print_of_the_guide_fails_there():
    """What the test above guards against: printing through the console encoding."""
    out = _run_cp932(["-c", "from dnndenoiser.guides import read_guide; print(read_guide())"])
    assert out.returncode != 0 and b"UnicodeEncodeError" in out.stderr
