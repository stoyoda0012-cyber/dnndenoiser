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
    leaked = [
        m for m in sys.modules
        if m.split(".")[0] in ("deppro", "toyomacro", "arhaxpes_denoise")
    ]
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
        # The P1 port is measured against the archived ``arhaxpes_denoise``
        # reference and must not import or vendor it: reproducing it means an
        # independent implementation, not adoption. Registered in
        # docs/preregistration/P1-selfsupervised-moving-average.md.
        "import arhaxpes_denoise", "from arhaxpes_denoise",
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


def test_only_the_golden_generator_may_read_the_p1_reference():
    """The P1 reference may be read to test against, and nowhere else.

    ``docs/preregistration/P1-selfsupervised-moving-average.md`` registers that
    reproducing the archived ``arhaxpes_denoise`` implementation means an
    independent implementation, not adoption: the port may read it to be
    measured against, and may not import or vendor it. The library guard above
    covers ``src/``; this one covers the rest of the repository, so the single
    legitimate reader is named rather than merely left out of scope.

    Imports are found by parsing, not by substring search. A file that merely
    names the package -- this one lists it as a forbidden token, and the port
    cites it in a docstring -- is not importing it, and a guard that could not
    tell the difference would flag itself.
    """
    import ast

    allowed = {pathlib.Path("tests/fixtures/generate_p1_reference_targets.py")}
    skip_dirs = {".git", "build", "dist", "__pycache__", ".ruff_cache", ".pytest_cache"}
    readers = set()
    parsed = 0

    for py in REPO_ROOT.rglob("*.py"):
        rel = py.relative_to(REPO_ROOT)
        if skip_dirs & set(rel.parts):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:  # not ours to judge; the library guard still scans src/
            continue
        parsed += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n.split(".")[0] == "arhaxpes_denoise" for n in names):
                readers.add(rel)

    assert parsed > 10, f"the scan only parsed {parsed} files -- it is not looking"
    assert readers <= allowed, (
        "these files import the P1 reference and are not the golden generator: "
        + ", ".join(sorted(str(p) for p in readers - allowed))
    )
    assert allowed <= readers, (
        "the golden generator no longer imports the reference -- has it been "
        "vendored, or has the guard been left pointing at a file that moved?"
    )
