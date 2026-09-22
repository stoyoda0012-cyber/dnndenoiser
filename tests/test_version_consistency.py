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


def test_packaging_metadata_carries_the_license_and_the_urls():
    """Metadata that a user or an index needs, and that rots silently.

    A distribution with no license field leaves its terms to be guessed, and
    one with no URLs gives an installed copy no route back to the source, the
    changelog or the archive. Asserted against ``pyproject.toml`` rather than
    against a built wheel so the check is cheap enough to always run; the
    wheel's own ``METADATA`` was verified by hand when these were added.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()

    assert re.search(r'^license\s*=\s*["\']MIT["\']', pyproject, re.MULTILINE), (
        "no license declared: the terms would be left to be guessed"
    )
    assert re.search(r"^license-files\s*=", pyproject, re.MULTILINE), (
        "the LICENSE file must travel with the distribution"
    )
    assert "[project.urls]" in pyproject

    for name in ("Homepage", "Repository", "Changelog", "Archive"):
        assert re.search(rf"^{name}\s*=", pyproject, re.MULTILINE), f"no {name} URL"

    # The archive URL is the concept DOI, which resolves to the latest release;
    # a version DOI here would go stale at the next one.
    assert "10.5281/zenodo.22867628" in pyproject, (
        "the Archive URL should be the concept DOI, not a version DOI"
    )
