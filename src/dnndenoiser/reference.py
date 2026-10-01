"""What an evaluation reference is, declared and carried with the data.

Implements the declaration part of ``docs/design/EVALUATION_REFERENCE_CONTRACT.md``
(adopted 2026-10-01, phase 1). A reference array (the ``clean`` dataset) carries an
*origin* — where it came from — as two HDF5 attributes, both required together:

- ``reference_schema_version`` — integer, :data:`SCHEMA_VERSION`;
- ``reference_origin`` — a JSON object, validated by :func:`validate_origin`.

A file with neither attribute has the effective origin ``undeclared``. Anything else that
is not a valid bundle is *malformed* and refused, never read as ``undeclared``.

The *relationship* of a reference to the evaluated data and to a model (overlap, use in
model development, signal match) is declared at evaluation and never stored with the
array: the same reference can be independent of one model and not of another.

The metadata cannot prove what a reference is. A noisy array declared as synthetic truth
is evaluated as truth; the protection is that the declaration is explicit, stored, and
visible in every output that depends on it.
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional

import numpy as np

SCHEMA_VERSION = 1
VERSION_ATTR = "reference_schema_version"
ORIGIN_ATTR = "reference_origin"
LINEAGE_ATTR = "reference_lineage"
UNITS_ATTR = "intensity_units"
ACQUISITION_ATTR = "acquisition_id"

ORIGINS = ("synthetic_truth", "estimate", "undeclared")
CONSTRUCTIONS = ("frame_mean", "leave_one_out_mean", "smoothed", "fitted", "other")
UNITS = ("counts", "counts_per_s", "normalised_to_spectrum_max", "normalised_to_global_max",
         "generator_intensity")
OVERLAP = ("overlap", "no_overlap_declared", "unknown")
MODEL_USE = ("yes", "no_declared", "unknown")

CONDITION_KEYS = ("energy_calibration", "channel_or_angle", "exposure_normalisation",
                  "specimen_state")
TRUTH_KEYS = {"origin", "generator", "units"}
ESTIMATE_KEYS = {"origin", "construction", "description", "source", "conditions", "units", "noise"}
LINEAGE_KEYS = {"operation", "from_points", "to_points", "tool", "step"}

UNDECLARED = {"origin": "undeclared"}


class MalformedReference(ValueError):
    """A reference declaration that is present but not valid."""


class ReferenceConflict(ValueError):
    """Declarations that cannot both hold, or an option that does not apply."""


def _reject_duplicates(pairs):
    keys = [k for k, _ in pairs]
    duplicated = sorted({k for k in keys if keys.count(k) > 1})
    if duplicated:
        raise MalformedReference(f"duplicate key(s) {duplicated} in a declaration")
    return dict(pairs)


def _reject_constant(name):
    raise MalformedReference(f"non-finite number {name} in a declaration")


def loads_strict(text: str) -> Any:
    """JSON with no duplicate keys (at any depth) and no NaN or Infinity."""
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicates,
                          parse_constant=_reject_constant)
    except MalformedReference:
        raise
    except (TypeError, ValueError) as exc:
        raise MalformedReference(f"not valid JSON: {exc}") from None


def _strict_int(value: Any, name: str) -> int:
    """An integer stored as an integer: not a float, a string, or a boolean."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise MalformedReference(f"'{name}' must be an integer, got {value!r} "
                                 f"({type(value).__name__})")
    return int(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MalformedReference(f"'{name}' must be a non-empty string, got {value!r}")
    return value


def validate_units(value: Any, name: str = "units") -> str:
    text = _text(value, name)
    if text in UNITS:
        return text
    if text.startswith("other:") and text[len("other:"):].strip():
        return text
    raise MalformedReference(
        f"'{name}' must be one of {', '.join(UNITS)} or 'other:<description>', got {value!r}")


def _frames(value: Any) -> Any:
    if value in ("all", "unrecorded"):
        return value
    if isinstance(value, list) and value and all(isinstance(v, int) and not isinstance(v, bool)
                                                 for v in value):
        return value
    if (isinstance(value, dict) and set(value) == {"range"} and isinstance(value["range"], list)
            and len(value["range"]) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value["range"])
            and value["range"][0] <= value["range"][1]):
        return value
    raise MalformedReference(
        "'source.frames' must be 'all', 'unrecorded', a non-empty list of integer frame "
        f"indices, or {{'range': [first, last]}}, got {value!r}")


def validate_origin(obj: Any) -> dict:
    """Check a ``reference_origin`` object and return it unchanged if valid."""
    if not isinstance(obj, dict):
        raise MalformedReference(f"reference_origin must be a JSON object, got {obj!r}")
    origin = obj.get("origin")
    if origin not in ORIGINS:
        raise MalformedReference(
            f"'origin' must be one of {', '.join(ORIGINS)}, got {origin!r}")
    if origin == "undeclared":
        if set(obj) != {"origin"}:
            raise MalformedReference("an 'undeclared' origin carries no other field")
        return obj
    if origin == "synthetic_truth":
        missing = {"generator", "units"} - set(obj)
        if missing:
            raise MalformedReference(f"synthetic_truth needs {sorted(missing)}")
        extra = set(obj) - TRUTH_KEYS
        if extra:
            raise MalformedReference(f"synthetic_truth has unknown field(s) {sorted(extra)}")
        _text(obj["generator"], "generator")
        validate_units(obj["units"])
        return obj
    # estimate
    missing = {"construction", "source", "conditions", "units", "noise"} - set(obj)
    if missing:
        raise MalformedReference(f"estimate needs {sorted(missing)}")
    extra = set(obj) - ESTIMATE_KEYS
    if extra:
        raise MalformedReference(
            f"estimate has unknown field(s) {sorted(extra)}; relationships to the evaluated "
            "data or the model are declared at evaluation, never stored with the reference")
    if obj["construction"] not in CONSTRUCTIONS:
        raise MalformedReference(
            f"'construction' must be one of {', '.join(CONSTRUCTIONS)}, got {obj['construction']!r}")
    if obj["construction"] == "other":
        _text(obj.get("description"), "description")
    elif "description" in obj:
        _text(obj["description"], "description")
    source = obj["source"]
    if not isinstance(source, dict) or set(source) != {"acquisition_id", "frames"}:
        raise MalformedReference("'source' must be an object with exactly 'acquisition_id' "
                                 "and 'frames'")
    _text(source.get("acquisition_id"), "source.acquisition_id")
    _frames(source.get("frames"))
    conditions = obj["conditions"]
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITION_KEYS):
        raise MalformedReference(
            f"'conditions' must have exactly {', '.join(CONDITION_KEYS)} (values may be "
            f"'unknown'), got {sorted(conditions) if isinstance(conditions, dict) else conditions!r}")
    for key, value in conditions.items():
        _text(value, f"conditions.{key}")
    validate_units(obj["units"])
    noise = obj["noise"]
    if isinstance(noise, dict):
        if not noise:
            raise MalformedReference("'noise' must not be empty; use 'unknown'")
        for key, value in noise.items():
            if isinstance(value, bool) or not (
                    (isinstance(value, str) and value.strip())
                    or (isinstance(value, (int, float)) and math.isfinite(value))):
                raise MalformedReference(f"'noise.{key}' must be a non-empty string or a "
                                         f"finite number, got {value!r}")
    else:
        _text(noise, "noise")
    return obj


def canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_declaration(dataset) -> tuple[Optional[dict], dict]:
    """``(stored, effective)`` for an HDF5 reference dataset.

    ``stored`` is the bundle's origin object as written, or ``None`` when neither bundle
    attribute is present. ``effective`` is the origin in force: the stored one, or
    :data:`UNDECLARED` for a legacy file. Raises :class:`MalformedReference` otherwise.
    """
    attrs = dataset.attrs
    has_version, has_origin = VERSION_ATTR in attrs, ORIGIN_ATTR in attrs
    if not has_version and not has_origin:
        return None, dict(UNDECLARED)
    if has_version != has_origin:
        present = VERSION_ATTR if has_version else ORIGIN_ATTR
        raise MalformedReference(f"'{present}' is present without its pair; the reference "
                                 "declaration needs both attributes")
    version = _strict_int(attrs[VERSION_ATTR], VERSION_ATTR)
    if version != SCHEMA_VERSION:
        raise MalformedReference(f"unsupported {VERSION_ATTR} {version}; this version reads "
                                 f"{SCHEMA_VERSION}")
    raw = attrs[ORIGIN_ATTR]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise MalformedReference(f"'{ORIGIN_ATTR}' must be a JSON string, got {type(raw).__name__}")
    try:
        obj = loads_strict(raw)
    except MalformedReference as exc:
        raise MalformedReference(f"'{ORIGIN_ATTR}': {exc}") from None
    validate_origin(obj)
    return obj, obj


def write_declaration(dataset, origin: dict) -> None:
    validate_origin(origin)
    dataset.attrs[VERSION_ATTR] = SCHEMA_VERSION
    dataset.attrs[ORIGIN_ATTR] = canonical(origin)


def read_lineage(dataset) -> list:
    raw = dataset.attrs.get(LINEAGE_ATTR)
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise MalformedReference(f"'{LINEAGE_ATTR}' must be a JSON string")
    try:
        lineage = loads_strict(raw)
    except MalformedReference as exc:
        raise MalformedReference(f"'{LINEAGE_ATTR}': {exc}") from None
    if not isinstance(lineage, list):
        raise MalformedReference(f"'{LINEAGE_ATTR}' must be a JSON array of records")
    for record in lineage:
        validate_lineage_record(record)
    return lineage


def validate_lineage_record(record: Any) -> dict:
    if not isinstance(record, dict) or set(record) != LINEAGE_KEYS:
        raise MalformedReference(f"a lineage record must have exactly {sorted(LINEAGE_KEYS)}, "
                                 f"got {record!r}")
    for key in ("operation", "tool", "step"):
        _text(record[key], f"lineage.{key}")
    for key in ("from_points", "to_points"):
        if _strict_int(record[key], f"lineage.{key}") < 1:
            raise MalformedReference(f"'lineage.{key}' must be positive")
    return record


def copy_declaration(source, target, append: Optional[dict] = None) -> None:
    """Copy the whole bundle and lineage from one dataset to another, unchanged, and
    append one lineage record when a transform was applied. Relationships are never
    copied: they are not stored."""
    stored, _ = read_declaration(source)
    lineage = read_lineage(source)
    if append is not None:
        validate_lineage_record(append)
    if stored is not None:
        target.attrs[VERSION_ATTR] = source.attrs[VERSION_ATTR]
        target.attrs[ORIGIN_ATTR] = source.attrs[ORIGIN_ATTR]
    if append is not None:
        lineage = lineage + [append]
    if lineage:
        target.attrs[LINEAGE_ATTR] = json.dumps(lineage, sort_keys=True)


def resolve(stored: Optional[dict], stored_effective: dict, cli: Optional[dict],
            selected: str = "the selected reference") -> tuple[dict, str]:
    """The effective declaration and its source (``file``, ``cli`` or ``none``).

    A command-line declaration may fill in an absent or explicitly ``undeclared`` origin;
    against a known stored origin it must be identical, or it is refused.
    """
    if cli is None:
        return stored_effective, ("none" if stored is None else "file")
    validate_origin(cli)
    if stored_effective["origin"] == "undeclared":
        return cli, "cli"
    if canonical(cli) != canonical(stored_effective):
        raise ReferenceConflict(
            f"the command-line declaration differs from the one stored with {selected}: "
            f"stored {canonical(stored_effective)}, given {canonical(cli)}")
    return stored_effective, "file"


def established_overlap(effective: dict, evaluated_acquisition_id: Optional[str],
                        evaluated_frame_index) -> bool:
    """Overlap is established — not inferred from the construction — when the reference's
    source acquisition is the evaluated data's and its frames are all of them or intersect
    the evaluated frames."""
    if effective.get("origin") != "estimate" or evaluated_acquisition_id is None:
        return False
    source = effective["source"]
    if source["acquisition_id"] != evaluated_acquisition_id:
        return False
    frames = source["frames"]
    if frames == "unrecorded":
        return False
    if frames == "all":
        return True
    if evaluated_frame_index is None:
        return False
    evaluated = np.asarray(evaluated_frame_index).astype(np.int64).ravel()
    if isinstance(frames, dict):
        # Compared at the ends, never expanded: a valid range can be very large.
        first, last = frames["range"]
        return bool(np.any((evaluated >= first) & (evaluated <= last)))
    return bool(np.isin(evaluated, np.asarray(frames, dtype=np.int64)).any())


def resolve_relationship(effective: dict, overlap: Optional[str], used: Optional[str],
                         signal_match: Optional[str], established: bool) -> dict:
    if overlap is not None and overlap not in OVERLAP:
        raise ReferenceConflict(f"overlap must be one of {', '.join(OVERLAP)}")
    if used is not None and used not in MODEL_USE:
        raise ReferenceConflict(f"used-in-model-development must be one of {', '.join(MODEL_USE)}")
    if established and overlap == "no_overlap_declared":
        raise ReferenceConflict(
            "no_overlap_declared contradicts an established overlap: the reference's source "
            "acquisition is the evaluated data's and its frames include evaluated frames")
    if established:
        overlap_value, overlap_basis = "overlap", "established"
    elif overlap is not None:
        overlap_value, overlap_basis = overlap, "declared"
    else:
        overlap_value, overlap_basis = "unknown", "default"
    return {
        "overlap_with_evaluated": overlap_value,
        "overlap_basis": overlap_basis,
        "used_in_model_development": used or "unknown",
        "signal_match": signal_match if signal_match and signal_match.strip() else "unknown",
    }
