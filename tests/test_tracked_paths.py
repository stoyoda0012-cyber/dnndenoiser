"""Regression guard: no tracked file carries a developer-specific path.

`AGENTS.md` §10 forbids developer-specific absolute paths in **tracked files** --
all of them, because the repository is public and every tracked file is published.
`tests/test_paths.py` enforces that over the installed package only, which is what
ships in the wheel. The rule is wider than that guard, and a record under
`benchmarks/` carried the developer's home directory in `consistency_anchor.read_from`
for three unpublished commits while `test_paths.py` reported `1 passed`. This file is
the guard at the rule's own width.

The forbidden tokens are read from `tests/test_paths.py` rather than repeated here, so
the two guards cannot drift apart and this file carries no private-looking literal of
its own. `tests/test_paths.py` is exempt by path, because its tokens are its test data.

What this cannot see: history. A path removed from the tree still sits in every
earlier commit that carried it, and a push publishes those too. Checking the commits
a push would expose is part of the pre-push procedure in `docs/VERIFICATION.md`, not
of a test that runs on one tree.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_SOURCE = REPO_ROOT / "tests" / "test_paths.py"
EXEMPT = {"tests/test_paths.py"}


def _tracked_files() -> list[str] | None:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, check=True,
                             capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.decode("utf-8").split("\0") if p]


pytestmark = pytest.mark.skipif(
    _tracked_files() is None,
    reason="not a git checkout (for example an sdist); test_paths.py still guards the package",
)


def forbidden_tokens() -> tuple[str, ...]:
    """The `forbidden` tuple from test_paths.py, read without importing it."""
    tree = ast.parse(TOKEN_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "forbidden"):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("tests/test_paths.py no longer defines `forbidden`")


def offenders_in(paths, read, tokens) -> list[str]:
    found = []
    for path in paths:
        if path in EXEMPT:
            continue
        data = read(path)
        if data is None or b"\0" in data[:8192]:
            continue  # unreadable or binary
        text = data.decode("utf-8", errors="replace")
        for token in tokens:
            if token in text:
                line = text[: text.index(token)].count("\n") + 1
                found.append(f"{path}:{line}: {token!r}")
    return found


def _read_worktree(path: str):
    try:
        return (REPO_ROOT / path).read_bytes()
    except OSError:
        return None


def _a_home_prefix() -> str:
    """A POSIX home-directory prefix from the token list, chosen by shape, not typed."""
    return next(t for t in forbidden_tokens() if t.startswith("/") and t.endswith("/"))


def test_the_token_list_is_read_not_repeated():
    tokens = forbidden_tokens()
    assert sum(t.startswith("/") and t.endswith("/") for t in tokens) >= 2, tokens


def test_no_tracked_file_carries_a_developer_specific_path():
    offenders = offenders_in(_tracked_files(), _read_worktree, forbidden_tokens())
    assert not offenders, "developer-specific paths in tracked files: " + "; ".join(offenders)


def test_the_guard_rejects_a_planted_path():
    """Shown to fail, not only to pass: a planted absolute path in a JSON record, of
    the exact shape the P2-A record carried, is reported."""
    planted = b'{"consistency_anchor": {"read_from": "' + _a_home_prefix().encode() \
        + b'someone/project/benchmarks/reference/results/x.json"}}'
    files = {"benchmarks/example/results/record.json": planted}
    found = offenders_in(files, files.get, forbidden_tokens())
    assert found and found[0].startswith("benchmarks/example/results/record.json:1:")


def test_the_guard_skips_its_own_token_source_but_nothing_else():
    token = _a_home_prefix().encode()
    files = {"tests/test_paths.py": token, "tests/other.py": token}
    found = offenders_in(files, files.get, forbidden_tokens())
    assert [f.split(":")[0] for f in found] == ["tests/other.py"]


def history_offenders(revision_range: str) -> list[str]:
    """Every commit in `revision_range` whose patch adds a forbidden token.

    The pre-push check. A tree can be clean while an earlier commit in the range still
    carries a path, and a push publishes every commit in the range.
    """
    tokens = forbidden_tokens()
    log = subprocess.run(
        ["git", "log", "-p", "--format=COMMIT %h %s", revision_range, "--",
         ".", ":(exclude)tests/test_paths.py"],
        cwd=REPO_ROOT, check=True, capture_output=True).stdout.decode("utf-8", "replace")
    found, commit, path = [], "?", "?"
    for line in log.splitlines():
        if line.startswith("COMMIT "):
            commit = line[7:]
        elif line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            found += [f"{commit} | {path} | {t!r}" for t in tokens if t in line]
    return found


if __name__ == "__main__":
    # python tests/test_tracked_paths.py origin/main..HEAD
    import sys

    revision_range = sys.argv[1] if len(sys.argv) > 1 else "origin/main..HEAD"
    hits = history_offenders(revision_range)
    for hit in hits:
        print(hit)
    print(f"{len(hits)} commit line(s) in {revision_range} add a developer-specific path")
    raise SystemExit(1 if hits else 0)
