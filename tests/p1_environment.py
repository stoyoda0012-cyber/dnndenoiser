"""The environment pin shared by the P1 criteria, and the reason for it.

``docs/preregistration/P1-selfsupervised-moving-average.md`` requires C0, C1,
C2, C3 and C7 **only** in one environment and makes them reported, not required,
elsewhere. Two different reasons sit behind that single rule:

- the trained-output criteria depend on floating-point reduction order, which
  differs between backends and builds;
- **C1 depends on ``numpy.argsort``'s tie-breaking**, which numpy does not
  contract. Acquisition indices spaced evenly put neighbours at equal distance
  on both sides, so for odd ``W`` the outermost slot is a genuine tie. The
  default sort is introsort and is not stable, and a different build can resolve
  the tie differently — which is not a defect in the port.

The second was found the hard way: C1 was written without the gate the document
requires, and Linux CI failed exactly the tie-ambiguous cases (W=1, W=5, the
permuted order, the duplicate index) while passing the tie-free ones (W=2,
W=10, the clamp). The document was right and the test was wrong.

:func:`tie_ambiguous_rows` lets a test work out for itself whether a case is
exposed, so the gate follows the mathematics rather than a hand-maintained list.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PINNED = json.loads(
    (Path(__file__).parent / "fixtures" / "p1_reference_targets.json").read_text(encoding="utf-8")
)
ENV = PINNED["_environment"]

_MISMATCH = [
    f"{name}: have {have}, pinned {want}"
    for name, have, want in (
        ("python", ".".join(map(str, sys.version_info[:2])),
         ".".join(ENV["python"].split(".")[:2])),
        ("torch", torch.__version__, ENV["torch"]),
        ("numpy", np.__version__, ENV["numpy"]),
    )
    if have != want
]
IN_PINNED_ENVIRONMENT = not _MISMATCH
MISMATCH_REASON = "; ".join(_MISMATCH)

pinned_environment = pytest.mark.skipif(
    not IN_PINNED_ENVIRONMENT,
    reason="outside the environment these criteria are pinned to — " + MISMATCH_REASON,
)


def tie_ambiguous_rows(frame_indices: np.ndarray, W: int) -> int:
    """How many rows' neighbour sets depend on the sort's tie-breaking.

    Zero means the case is decided by distance alone and holds on any build.
    """
    t = np.asarray(frame_indices, dtype=np.float64)
    distance = np.abs(t[:, None] - t[None, :])
    np.fill_diagonal(distance, np.inf)
    window = min(int(W), len(t) - 1)
    default = np.argsort(distance, axis=1)[:, :window]
    stable = np.argsort(distance, axis=1, kind="stable")[:, :window]
    return sum(1 for i in range(len(t)) if set(default[i]) != set(stable[i]))


def skip_if_tie_ambiguous_outside_pin(frame_indices: np.ndarray, W: int) -> None:
    """Skip a C1 case that the sort's tie-breaking decides, off the pinned build.

    Called by the test rather than applied as a marker, because whether a case is
    exposed is a property of ``(frame_indices, W)`` and is computed, not listed.
    """
    if IN_PINNED_ENVIRONMENT:
        return
    ambiguous = tie_ambiguous_rows(frame_indices, W)
    if ambiguous:
        pytest.skip(
            f"{ambiguous} row(s) here are decided by numpy.argsort's tie-breaking, "
            f"which numpy does not contract; required only in the pinned "
            f"environment ({MISMATCH_REASON})"
        )
