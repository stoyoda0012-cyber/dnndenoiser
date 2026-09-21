"""The version is stated in three places, and they must agree.

``pyproject.toml`` is what the wheel is built from, ``dnndenoiser.__version__``
is what a user sees at runtime, and ``CITATION.cff`` is what a citation resolves
to. Nothing made them agree until this test: a release that bumps two of the
three ships a package whose metadata contradicts itself, and the one most likely
to be forgotten is the citation, which is the one a reader relies on.
"""
from __future__ import annotations

import re
from pathlib import Path

import dnndenoiser

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_the_three_declared_versions_agree():
    # Read by regex rather than with tomllib: this project supports Python 3.10
    # and tomllib arrived in 3.11. A test that cannot run on a supported
    # interpreter is not a test of that interpreter.
    pyproject_match = re.search(
        r"^version\s*=\s*[\"']([^\"']+)[\"']\s*$",
        (REPO_ROOT / "pyproject.toml").read_text(), re.MULTILINE,
    )
    assert pyproject_match, "pyproject.toml has no top-level version"
    pyproject = pyproject_match.group(1)

    citation = re.search(
        r"^version:\s*(\S+)\s*$", (REPO_ROOT / "CITATION.cff").read_text(), re.MULTILINE
    )
    assert citation, "CITATION.cff has no top-level 'version:' key"

    assert pyproject == dnndenoiser.__version__ == citation.group(1), (
        f"pyproject {pyproject!r}, __version__ {dnndenoiser.__version__!r}, "
        f"CITATION.cff {citation.group(1)!r}"
    )


def test_the_changelog_has_a_section_for_this_version():
    """A released version that the changelog does not mention is not documented."""
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
    assert f"## [{dnndenoiser.__version__}]" in changelog, (
        f"CHANGELOG.md has no '## [{dnndenoiser.__version__}]' section"
    )
