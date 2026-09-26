"""Parts shared by the P2 boundary records written after P2-A.

P2-A's script (`position_shift/position_shift_boundary.py`) defines its own copies of
these and is deliberately left as it is: its sha256 is part of its record's provenance,
so editing it to import from here would make the record name a script that is no longer
the one in the tree. New records import from this module instead, and record this
module's own sha256 beside their script's, because it shapes their numbers too.

Every function here is a copy of P2-A's, generalised only where a new record needs a
parameter P2-A fixed; where that is so, the docstring says what was generalised.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from math import comb
from pathlib import Path

import numpy as np
import scipy
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


class SelfCheckFailure(RuntimeError):
    """Raised when a self-check fails. The record is not written."""


# --------------------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------------------


def snr_db(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-spectrum SNR in dB against a clean reference, as in P2-A and the reference
    benchmark: 10 * log10(mean(reference**2) / mean((estimate - reference)**2))."""
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    signal_power = np.mean(reference**2, axis=-1)
    noise_power = np.mean((estimate - reference) ** 2, axis=-1)
    if not np.all(noise_power > 0.0):
        raise ValueError(
            "zero noise power: an estimate is bit-identical to the reference, "
            "which makes SNR undefined rather than infinite"
        )
    if not np.all(signal_power > 0.0):
        raise ValueError("zero signal power: a reference spectrum is all zeros")
    return 10.0 * np.log10(signal_power / noise_power)


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def binomial_one_sided(k: int, n: int) -> float:
    """P(X >= k | n, p = 0.5). Every sign rule is directional, so one-sided."""
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2**n)


def sign_test(values, positive: bool, k: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    favouring = int(np.sum(values > 0) if positive else np.sum(values < 0))
    return {
        "n_seeds": n, "n_favouring": favouring, "k_required": k,
        "one_sided_binomial_p": binomial_one_sided(favouring, n) if n else None,
        "passed": favouring >= k,
    }


def holm(pvalues: list) -> list:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = float(min(1.0, running))
    return adjusted


def threshold_for_family(m: int, n: int, alpha: float = 0.05) -> int:
    """The smallest k whose one-sided p clears Holm's first step for a family of m.

    Not in P2-A, which fixed one threshold. A family in which every test reaches this k
    passes Holm at every step, because each p is at most the first step's bound.
    """
    if m < 1:
        raise ValueError("a family has at least one test")
    for k in range(n + 1):
        if binomial_one_sided(k, n) < alpha / m:
            return k
    raise ValueError(f"no threshold of {n} seeds clears Holm's first step for m = {m}")


# --------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def working_tree_status() -> str:
    return git("status", "--porcelain", "--untracked-files=normal")


def provenance_record(*, quick: bool, preregistration: str, scripts: list) -> dict:
    """What was run, against what, read from git and the filesystem -- never typed.

    Generalised from P2-A's in one respect: `scripts` lists every file whose content
    shapes the numbers (a record's own script and this module), each with its sha256.
    A full run from a tree with uncommitted or untracked changes is refused. Nothing
    here holds an absolute path.
    """
    status = working_tree_status()
    clean = status == ""
    if not clean and not quick:
        raise SelfCheckFailure(
            "refusing a full run from a working tree with uncommitted or untracked "
            "changes; the record would name a commit that is not what ran:\n" + status)
    document = REPO_ROOT / preregistration
    lockfile = REPO_ROOT / "uv.lock"
    touching = git("log", "--format=%H", "--", preregistration).split()
    return {
        "how_recorded": "read from git and the filesystem at run time; nothing here is typed",
        "code_commit": git("rev-parse", "HEAD"),
        "working_tree_clean": clean,
        "working_tree_changes_if_dirty": status.splitlines() if not clean else [],
        "scripts": [
            {"path": Path(s).resolve().relative_to(REPO_ROOT).as_posix(), "sha256": sha256(s)}
            for s in scripts
        ],
        "registration": {
            "document": preregistration,
            "sha256_at_run": sha256(document),
            "first_commit": touching[-1] if touching else None,
            "last_commit_before_run": touching[0] if touching else None,
            "commits_before_run": touching,
        },
        "lockfile": ({"path": "uv.lock", "sha256": sha256(lockfile)}
                     if lockfile.exists() else None),
        "in_virtual_environment": sys.prefix != sys.base_prefix,
    }


def environment_record(device: str) -> dict:
    record = {
        "device_requested_resolved_to": device,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if device == "cuda" and torch.cuda.is_available():
        record["cuda_device_name"] = torch.cuda.get_device_name(0)
    return record
