#!/usr/bin/env python3
"""Regenerate the pinned reference values for the P1 acceptance tests.

The reference implementation is **not** part of this repository and is not
importable in CI. It is the archived ``arhaxpes_denoise`` package, Zenodo
``10.5281/zenodo.22092109`` version 1.0.0, whose per-file digests are recorded in
``docs/preregistration/P1-selfsupervised-moving-average.md``.

This script reads that package and writes ``p1_reference_targets.json``: the
reference's targets for every case of C1, its freshly-constructed parameters for
C0, its trained outputs and SNRs for C2 and C3, and its resampled arrays for C6.
Digests rather than arrays, because these criteria demand *exact* or
near-exact equality, for which a digest is a complete test and a megabyte of
committed floats is not a better one.

C0, C2 and C3 are numerical-identity claims pinned to one environment; the
environment this ran in is recorded alongside them so the tests can skip
elsewhere rather than fail.

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
import torch

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

    # --- C0, C2, C3, C6: the environment-pinned criteria -------------------
    import importlib

    ref_ss = importlib.import_module("arhaxpes_denoise.selfsupervised")
    ref_net = importlib.import_module("arhaxpes_denoise.network")
    torch.set_num_threads(1)
    kw = dict(num_features=256, num_hidden_units=100,
              layer_type="ResNet-FCNN", encoder_output_dim=64)

    c0 = {}
    for seed in range(6):
        torch.manual_seed(seed)
        sd = ref_net.DenoisingNetwork(**kw).state_dict()
        flat = np.concatenate([v.detach().numpy().ravel() for v in sd.values()])
        c0[str(seed)] = digest(flat)

    energy = np.linspace(0, 1, 256)
    clean = 300 * np.exp(-((energy - 0.5) ** 2) / (2 * 0.04**2)) + 20
    rng2 = np.random.default_rng(1)
    raw = rng2.poisson(np.tile(clean, (200, 1))).astype(np.float32)
    g_min, g_max = raw.min(), raw.max()
    test = (rng2.poisson(np.tile(clean, (16, 1))).astype(np.float32) - g_min) / (g_max - g_min)
    clean_n = (clean - g_min) / (g_max - g_min)
    targets = reference(frames, np.arange(n), 5)

    def snr(y):
        return float(10 * np.log10(np.mean(clean_n**2) / np.mean((y - clean_n) ** 2)))

    out0 = ref_ss.denoise(ref_ss.train_denoiser(frames, targets, epochs=20, seed=0), test)
    c2 = {"sha256": digest(out0), "max_abs": float(np.abs(out0).max())}
    arrays = {"c2_output_seed0": out0}

    c3 = {}
    for seed in (1, 2, 3, 4, 5):
        y = ref_ss.denoise(ref_ss.train_denoiser(frames, targets, epochs=20, seed=seed), test)
        c3[str(seed)] = {"sha256": digest(y), "snr_db": snr(y)}

    c6 = {str(m): digest(ref_ss.resample(test, m)) for m in (128, 512)}

    # C4: the reference's state_dict manifest, plus the real cross-package load
    # performed here (2.6 MB of weights is not committed; the manifest is what
    # CI can check, and this run is the evidence that the load itself works).
    torch.manual_seed(0)
    ref_model = ref_net.DenoisingNetwork(**kw)
    ref_sd = ref_model.state_dict()
    c4 = {
        "keys": [[k, list(v.shape)] for k, v in ref_sd.items()],
        "parameter_count": int(sum(p.numel() for p in ref_model.parameters())),
    }
    from dnndenoiser.models.network import DenoisingNetwork as PortNet

    torch.manual_seed(0)
    port_model = PortNet(**kw)
    port_model.load_state_dict(ref_sd, strict=True)
    ref_model.load_state_dict(port_model.state_dict(), strict=True)
    ref_model.eval()
    port_model.eval()
    probe = torch.tensor(test, dtype=torch.float32)
    with torch.no_grad():
        a, _ = ref_model(probe)
        b, _ = port_model(probe)
    c4["cross_loaded_outputs_max_abs_diff"] = float((a - b).abs().max())

    c7_out = ref_ss.denoise(ref_ss.train_denoiser(frames, targets, epochs=30, seed=0), test)
    c7 = {"sha256": digest(c7_out), "max_abs": float(np.abs(c7_out).max())}
    arrays["c7_output_epochs30_seed0"] = c7_out

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
        "c0_initial_parameters": c0,
        "c2_trained_output_seed0": c2,
        "c3_trained_output_seeds": c3,
        "c6_resampled": c6,
        "c4_state_dict_manifest": c4,
        "c7_trained_output_epochs30_seed0": c7,
        "_environment": {
            "note": "C0, C2, C3 and C7 are required only in this environment",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "torch_num_threads": 1,
        },
    }
    path = Path(__file__).with_name("p1_reference_targets.json")
    path.write_text(json.dumps(out, indent=2) + "\n")

    # C2 and C7 state a statistic over the reference's own array, so the array
    # is pinned rather than only its digest: 16 KB each, and it lets the test
    # evaluate the registered criterion instead of a proxy for it.
    npz = Path(__file__).with_name("p1_reference_outputs.npz")
    np.savez_compressed(npz, **arrays)
    print(f"wrote {npz} — {', '.join(arrays)}")
    print(f"wrote {path} — C1: {len(cases)} cases; C0/C2/C3/C6/C7 pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
