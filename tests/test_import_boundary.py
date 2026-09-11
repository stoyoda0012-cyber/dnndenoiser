"""Carve-out boundary tests for DNNDenoiser.

DNNDenoiser is the *generic* XPS spectrum denoiser. All depth-profiling and
peak-fitting code lives in sibling repos (``DepthProfiler``, ``toyomacro``).
These tests enforce that boundary so it cannot silently
regress: importing the denoiser must never pull in ``deppro`` / ``toyomacro``,
and the shipped library code must not hardcode sibling-project paths.

The sibling names below are the guard's test data, which makes this file a hit
for any scanner looking for them; it is allowlisted by path for exactly that
reason, and the allowance is keyed to this file's content.
"""
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# The src-layout package is the library, and it is the whole of it: there is one
# surface to scan and this is it. The list is written out rather than derived so
# that adding a second shipped tree is a visible edit here.
LIBRARY_PATHS = [
    REPO_ROOT / "src" / "dnndenoiser",
]


def test_public_api_imports_without_external_solvers():
    pytest.importorskip("torch")
    import dnndenoiser  # noqa: F401
    leaked = [m for m in sys.modules if m.split(".")[0] in ("deppro", "toyomacro")]
    assert not leaked, f"depth-solver / peak-fitter modules leaked in: {leaked}"


def test_public_api_surface():
    pytest.importorskip("torch")
    import dnndenoiser
    for name in (
        "DenoisingNetwork", "build_denoising_network",
        "SyntheticGenerator", "GeneratorConfig", "NoiseConfig", "PEAK_SETS",
    ):
        assert hasattr(dnndenoiser, name), f"missing public symbol: {name}"


def test_no_hardcoded_sibling_paths_in_library():
    """Static guard: shipped library code must not reference sibling projects."""
    forbidden = (
        "DepthProfiler/python", "toyomacro-python",
        "import deppro", "from deppro", "import toyomacro", "from toyomacro",
    )
    offenders = []
    scanned = 0
    for pkg_dir in LIBRARY_PATHS:
        # A missing directory used to be skipped, so a guard pointed at a tree
        # that had been moved or removed went quiet instead of red. Every path
        # in the list must exist, and the scan must actually read files.
        assert pkg_dir.is_dir(), f"library path is missing: {pkg_dir}"
        for py in pkg_dir.rglob("*.py"):
            scanned += 1
            text = py.read_text(encoding="utf-8", errors="ignore")
            if any(token in text for token in forbidden):
                offenders.append(str(py.relative_to(REPO_ROOT)))
    assert scanned, "the guard scanned no files at all"
    assert not offenders, f"library code references sibling projects: {offenders}"
