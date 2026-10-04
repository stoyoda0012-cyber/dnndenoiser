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
                  "than the CPU, or members were trained on a device other than the CPU.")


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
    """Mean and ddof-1 standard deviation over members, accumulated in float64 one member at a
    time (two passes, no stacked copy), stored as float32."""
    k = len(outputs)
    total = np.zeros(np.shape(outputs[0]), dtype=np.float64)
    for o in outputs:
        total += np.asarray(o, dtype=np.float64)
    mean = total / k
    squares = np.zeros_like(mean)
    for o in outputs:
        squares += (np.asarray(o, dtype=np.float64) - mean) ** 2
    return mean.astype(np.float32), np.sqrt(squares / (k - 1)).astype(np.float32)


def read_members(handle) -> list:
    """Every member's records, verified, in member order; each subgroup's seed must be its
    manifest's and the root's ``members_seeds`` entry at that position, so members that were
    swapped or renamed are refused."""
    from dnndenoiser import provenance as prov

    for key in ("members_seeds", "ensemble_k"):
        if key not in handle.attrs:
            raise prov.MalformedProvenance(f"an ensemble output needs the root attribute '{key}'")
    raw = np.asarray(handle.attrs["members_seeds"])
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise prov.MalformedProvenance(f"members_seeds must be a 1-D integer array, got {raw!r}")
    seeds = [int(s) for s in raw]
    k = handle.attrs["ensemble_k"]
    if not isinstance(k, (int, np.integer)) or int(k) != len(seeds):
        raise prov.MalformedProvenance(f"ensemble_k {k!r} is not the number of members_seeds "
                                       f"({len(seeds)})")
    if MEMBERS_GROUP not in handle:
        raise prov.MalformedProvenance(f"an ensemble output needs the group '{MEMBERS_GROUP}'")
    group = handle[MEMBERS_GROUP]
    if sorted(group) != sorted(str(i) for i in range(len(seeds))):
        raise prov.MalformedProvenance(f"'{MEMBERS_GROUP}' holds {sorted(group)}, not one "
                                       f"subgroup per member of {seeds}")
    records = []
    for i, seed in enumerate(seeds):
        r = prov.read_member_records(group[str(i)])
        if r["manifest"]["command"]["seeds"]["torch"] != seed:
            raise prov.MalformedProvenance(f"member {i} records seed "
                                           f"{r['manifest']['command']['seeds']['torch']}; "
                                           f"members_seeds says {seed}")
        records.append(r)
    return records


def encode_notes(notes: list) -> str:
    return json.dumps(list(notes))


_CARRIED = {"intensity_units", "acquisition_id"}
_ATTRS = {
    "noisy": _CARRIED | {"input_array_digest"},
    MEMBERS: _CARRIED, MEAN: _CARRIED, SPREAD: _CARRIED,
    "clean": {"reference_schema_version", "reference_origin", "reference_lineage",
              "signal_identity"},
    "energy": set(), "times": set(),
    "angles": {"angle_kind", "angle_units"},
    "frame_index": {"order_basis"},
    MEMBERS_GROUP: set(),
}


def unexpected_contents(path) -> list:
    """Datasets, groups and attributes of an ensemble output that the design does not list:
    names of unexpected objects, and ``<object>@<attribute>`` (``@<attribute>`` at the root)
    for unexpected attributes."""
    import h5py

    member = re.compile(rf"{MEMBERS_GROUP}/\d+")
    found = []

    def visit(name, obj):
        if name in _ATTRS:
            allowed = _ATTRS[name]
        elif member.fullmatch(name) and isinstance(obj, h5py.Group):
            allowed = {"model_digest", "model_body_digest", "seed"}
        elif (member.fullmatch(name.rsplit("/", 1)[0]) and name.endswith("/model_provenance")
              and isinstance(obj, h5py.Dataset)):
            allowed = set()
        else:
            found.append(name)
            return
        found.extend(f"{name}@{key}" for key in obj.attrs if key not in allowed)

    with h5py.File(path, "r") as f:
        f.visititems(visit)
        found += [f"@{key}" for key in f.attrs if key not in ROOT_ATTRS]
    return found
