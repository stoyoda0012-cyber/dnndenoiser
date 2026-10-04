"""Combining models that differ only in their seed (docs/design/MULTI_SEED.md, adopted
2026-10-04).

The ensemble mean is a new estimator and the spread between members a description of these
runs on this input; neither is an uncertainty. The sentences that say so are fixed here and
printed with every ensemble.
"""
from __future__ import annotations

import json
import re

import numpy as np

MEMBERS = "denoised_members"
MEAN = "ensemble_mean_estimate"
SPREAD = "between_run_std_fixed_input"
MEMBERS_GROUP = "members"
ROOT_ATTRS = ("ensemble_k", "members_seeds", "inference_device", "ensemble_notes")
CARRIED_DATASETS = ("noisy", "clean", "energy", "angles", "times", "frame_index")

MEAN_NOTE = (
    "ensemble_mean_estimate is the mean of the members' outputs: a model estimate like any "
    "member's, not a measurement. It can hide one member's failure (every member's output is "
    "in denoised_members), and it is not shown to be closer to the signal than any member; no "
    "comparison with the members is made or implied.")
SPREAD_NOTE = (
    "between_run_std_fixed_input is how much these runs differ from each other on this input "
    "(standard deviation over members, ddof 1). It is not a measurement uncertainty, a "
    "confidence interval or a bound on the difference from the signal, and it is not divided "
    "by the square root of the number of members. It does not reflect a bias the models "
    "share: members can agree closely and all be wrong, inside or outside the training "
    "distribution. It includes no variation from other training data or other acquisitions, "
    "and with two or three members it is itself a poor estimate of the spread between runs.")
FIXED_SENTENCES = (MEAN_NOTE, SPREAD_NOTE)

TREE_NOTE = ("The members were trained from a checkout with uncommitted changes, so the records "
             "cannot show that they ran the same code.")
UNKNOWN_CODE_NOTE = ("The members' code is recorded only by its version string, so the records "
                     "cannot show that the installed code was the same.")
EXECUTION_NOTE = ("The spread includes execution variability: inference ran on a device other "
                  "than the CPU, or members were trained outside Tier 1 (MPS, CUDA).")


def status_note(seed: int, label: str, statuses: dict) -> str:
    recorded = "; ".join(f"{k}: {v}" for k, v in sorted(statuses.items()))
    return (f"The member with seed {seed} ({label}) recorded a status in training ({recorded}); "
            "it is kept, not dropped.")


def non_finite_message(seed: int, label: str) -> str:
    return (f"the output of the member with seed {seed} ({label}) contains non-finite values, so "
            "nothing was combined or written. If you combine the others without it, report the "
            "excluded member and this reason with the result: the new file cannot show the "
            "exclusion.")


def evaluate_message(path) -> str:
    return (f"{path} is an ensemble output (it has '{MEMBERS}' and no 'denoised'). Evaluate each "
            "member on its own: run `dnndenoiser infer -d <data> -m <member.pt> -o <out.h5>` "
            "with one member's checkpoint, evaluate that output, and compare the members "
            "paired by seed. The ensemble mean is a separate estimator and is not evaluated.")


def combine(outputs: list) -> tuple:
    """Mean and ddof-1 standard deviation over members, in float64, stored as float32."""
    stack = np.stack([np.asarray(o, dtype=np.float64) for o in outputs])
    return (stack.mean(axis=0).astype(np.float32),
            stack.std(axis=0, ddof=1).astype(np.float32))


def encode_notes(notes: list) -> str:
    return json.dumps(list(notes))


def unexpected_contents(path) -> list:
    """Datasets and root attributes of an ensemble output that the design does not list;
    root attributes are prefixed with '@'."""
    import h5py

    allowed = set(CARRIED_DATASETS) | {MEMBERS, MEAN, SPREAD}
    member = re.compile(rf"{MEMBERS_GROUP}/\d+/model_provenance")
    found = []
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: found.append(name)
                     if isinstance(obj, h5py.Dataset)
                     and name not in allowed and not member.fullmatch(name) else None)
        found += [f"@{key}" for key in f.attrs if key not in ROOT_ATTRS]
    return found
