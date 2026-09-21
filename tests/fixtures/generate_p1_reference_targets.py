#!/usr/bin/env python3
"""Regenerate the pinned reference targets for the P1 acceptance tests.

The reference implementation is **not** part of this repository and is not
importable in CI. It is the archived ``arhaxpes_denoise`` package, Zenodo
``10.5281/zenodo.22092109`` version 1.0.0, whose per-file digests are recorded in
``docs/preregistration/P1-selfsupervised-moving-average.md``.

This script reads that package, computes the reference's targets for every case
of criterion C1, and writes their sha256 digests to
``p1_reference_targets.json``. Digests rather than arrays because C1 demands
*exact* equality, for which a digest is a complete test and a 1.6 MB array is
not a better one.

Usage::

    DNND_ARHAXPES_REF=/path/to/software_arhaxpes_denoise/src \
        python3 tests/fixtures/generate_p1_reference_targets.py

The path must be the ``src`` directory of a copy whose ``selfsupervised.py``
matches the digest recorded in the preregistration; this script checks that and
refuses otherwise.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SELFSUPERVISED_SHA256 = (
    "136f2e112430fbb42e72bc6d6c0f2b9b02d3a828f6774c11bd2392eb4a13cb09"
)
DEPOSIT = "10.5281/zenodo.22092109"
DEPOSIT_VERSION = "1.0.0"


def fixture_frames() -> np.ndarray:
    """The preregistration's fixture, built from its constants alone."""
    rng = np.random.default_rng(1)
    energy = np.linspace(0, 1, 256)
    clean = 300 * np.exp(-((energy - 0.5) ** 2) / (2 * 0.04**2)) + 20
    frames = rng.poisson(np.tile(clean, (200, 1))).astype(np.float32)
    g_min, g_max = frames.min(), frames.max()
    return (frames - g_min) / (g_max - g_min)


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:
    ref_src = os.environ.get("DNND_ARHAXPES_REF")
    if not ref_src:
        print(__doc__)
        return 2

    module = Path(ref_src) / "arhaxpes_denoise" / "selfsupervised.py"
    got = hashlib.sha256(module.read_bytes()).hexdigest()
    if got != SELFSUPERVISED_SHA256:
        print(f"refusing: {module} has sha256 {got},")
        print(f"          the deposit's is    {SELFSUPERVISED_SHA256}")
        return 1

    sys.path.insert(0, ref_src)
    from arhaxpes_denoise.selfsupervised import moving_average_targets as reference

    frames = fixture_frames()
    n = len(frames)
    cases: dict[str, dict] = {}

    for W in (1, 2, 5, 10):
        t = reference(frames, np.arange(n), W)
        cases[f"a_arange_W{W}"] = {"sha256": digest(t), "shape": list(t.shape)}

    permuted = np.random.default_rng(7).permutation(n)
    t = reference(frames, permuted, 5)
    cases["b_permuted_W5"] = {"sha256": digest(t), "shape": list(t.shape)}

    t = reference(frames[:3], np.arange(3), 10)
    cases["c_clamp_n3_W10"] = {"sha256": digest(t), "shape": list(t.shape),
                               "values": t.tolist()[:1]}

    duplicated = np.arange(n)
    duplicated[1] = duplicated[0]
    t = reference(frames, duplicated, 5)
    cases["e_duplicate_index_W5"] = {"sha256": digest(t), "shape": list(t.shape)}

    out = {
        "_provenance": {
            "deposit": DEPOSIT,
            "deposit_version": DEPOSIT_VERSION,
            "selfsupervised_sha256": SELFSUPERVISED_SHA256,
            "fixture": "docs/preregistration/P1-selfsupervised-moving-average.md",
            "dtype": "float64",
            "note": "sha256 of the C-contiguous float64 target array's bytes",
        },
        "cases": cases,
    }
    path = Path(__file__).with_name("p1_reference_targets.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path} — {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
