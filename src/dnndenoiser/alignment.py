"""Alignment before any metric: what ``evaluate`` compares, row by row and point by point.

Implements §5 of ``docs/design/EVALUATION_REFERENCE_CONTRACT.md`` (adopted 2026-10-01;
phase 2). Three things are checked and kept apart:

- comparison coordinates — the shape, the energy grid, and the angle and time axes the
  layout has;
- source identifiers — ``frame_index``, compared only inside one acquisition namespace;
- units — the evaluated arrays' ``intensity_units`` against the reference's declared units.

Detected metadata is always checked. An assertion (``--assert-alignment``) fills in only a
check that could not be made because metadata is absent; it never overrides a detected
mismatch, and it is refused for a check that was made or that does not apply. The result
lists what was verified, what was asserted and what does not apply, separately.

Layout. Spectra are ``(rows, …, energy)``: the first axis is the row (frame or sample)
axis, the last the energy axis, and the ``k = ndim - 2`` axes between them are coordinate
axes. With ``k = 0`` angles and times do not apply. With ``k = 2`` the axes are
``(rows, times, angles, energy)``, as ``generate`` writes them. With ``k = 1`` the one
coordinate axis is named by the coordinate dataset the evaluated file carries (``angles``
or ``times``); a file carrying neither has an unnamed axis, which the user names and
asserts with ``--assert-alignment angles`` or ``--assert-alignment times``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dnndenoiser import reference as ref

CHECKS = ("shape", "energy", "units", "rows", "angles", "times")
ASSERTABLE = ("energy", "units", "rows", "angles", "times")
COORDINATES = ("energy", "angles", "times", "frame_index")
TOLERANCE = 1e-6
ROW_BASES = ("same_file", "identifiers", "asserted")


class AlignmentError(ValueError):
    """The evaluated data and the reference cannot be compared as asked."""


@dataclass
class Side:
    """One side of the comparison: the evaluated arrays, or the reference.

    ``units`` is ``intensity_units`` for the evaluated arrays and the declared ``units``
    for the reference; ``units_note`` says why it is absent when it is. ``acquisition_id``
    is the evaluated data's attribute, or the reference's ``source.acquisition_id``.
    """
    shape: tuple
    energy: Optional[np.ndarray] = None
    angles: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    frame_index: Optional[np.ndarray] = None
    units: Optional[str] = None
    units_note: Optional[str] = None
    acquisition_id: Optional[str] = None


def parse_assertions(text) -> set:
    """``--assert-alignment energy,units,rows,angles,times`` as a set of check names. A
    repeated flag accumulates: a list of values is read as their union."""
    if text is None:
        return set()
    if isinstance(text, (list, tuple)):
        text = ",".join(text)
    names = [part.strip() for part in text.split(",")]
    unknown = [n for n in names if n not in ASSERTABLE]
    if unknown:
        raise AlignmentError(f"--assert-alignment names checks among {', '.join(ASSERTABLE)}, "
                             f"got {text!r}")
    return set(names)


def evaluated_units(noisy_units, denoised_units) -> tuple[Optional[str], Optional[str]]:
    """``(units, note)`` for the evaluated arrays: the one value both carry, or ``None``
    with a note saying what is absent. Different values are a mismatch, not an absence."""
    values = {}
    for name, value in (("noisy", noisy_units), ("denoised", denoised_units)):
        if value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        try:
            values[name] = ref.validate_units(value, f"{name}.{ref.UNITS_ATTR}")
        except ref.MalformedReference as exc:
            raise AlignmentError(str(exc)) from None
    if len(values) == 2:
        if values["noisy"] != values["denoised"]:
            raise AlignmentError(f"units differ between the evaluated arrays: 'noisy' is in "
                                 f"'{values['noisy']}' and 'denoised' in '{values['denoised']}'")
        return values["noisy"], None
    if not values:
        return None, f"the evaluated arrays carry no {ref.UNITS_ATTR}"
    absent = "denoised" if "noisy" in values else "noisy"
    return None, f"'{absent}' carries no {ref.UNITS_ATTR}"


def _axes(evaluated: Side, asserted: set) -> tuple[dict, list]:
    """``({name: axis}, not_applicable)`` for the layout of the evaluated arrays."""
    shape = evaluated.shape
    k = len(shape) - 2
    if k < 0:
        raise AlignmentError(f"spectra need a row axis and an energy axis, got shape {shape}")
    if k > 2:
        raise AlignmentError(f"unsupported layout {shape}: more than two coordinate axes "
                             "between the row axis and the energy axis")
    if k == 0:
        return {}, ["angles", "times"]
    if k == 2:
        return {"times": 1, "angles": 2}, []
    present = [n for n in ("angles", "times") if getattr(evaluated, n) is not None]
    if len(present) == 2:
        raise AlignmentError("a (rows, axis, energy) layout has one coordinate axis, but the "
                             "evaluated file carries both 'angles' and 'times'")
    if present:
        name = present[0]
    else:
        named = [n for n in ("angles", "times") if n in asserted]
        if len(named) != 1:
            raise AlignmentError(
                "the coordinate axis of this (rows, axis, energy) layout is unnamed: the "
                "evaluated file carries neither 'angles' nor 'times'; add the coordinate "
                "dataset, or name and assert the axis with --assert-alignment angles or "
                "--assert-alignment times (not both)")
        name = named[0]
    return {name: 1}, ["times" if name == "angles" else "angles"]


def _vector(value, name: str, which: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.dtype.kind not in "biuf":
        raise AlignmentError(f"'{name}' of {which} must be a one-dimensional numeric array, "
                             f"got shape {arr.shape} and dtype {arr.dtype}")
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise AlignmentError(f"'{name}' of {which} contains non-finite values")
    return arr


def _check_lengths(side: Side, which: str, axes: dict, with_axes: bool) -> None:
    """Metadata that contradicts the arrays it describes is refused outright. Every carried
    coordinate is a finite one-dimensional numeric array, whether or not its axis applies;
    lengths are checked for the axes the layout has."""
    for name in COORDINATES:
        if getattr(side, name) is not None:
            _vector(getattr(side, name), name, which)
    if side.energy is not None:
        energy = _vector(side.energy, "energy", which)
        if len(energy) != side.shape[-1]:
            raise AlignmentError(f"'energy' of {which} has {len(energy)} values but its "
                                 f"spectra have {side.shape[-1]} points")
    if side.frame_index is not None:
        # The row axis exists in every layout, a shared reference's included.
        index = _vector(side.frame_index, "frame_index", which)
        if index.dtype.kind not in "iu":
            raise AlignmentError(f"'frame_index' of {which} must hold integers, got dtype "
                                 f"{index.dtype}")
        if len(index) != side.shape[0]:
            raise AlignmentError(f"'frame_index' of {which} has {len(index)} values but its "
                                 f"row axis has {side.shape[0]}")
    if not with_axes:
        return
    for name, axis in axes.items():
        coord = getattr(side, name)
        if coord is not None:
            coord = _vector(coord, name, which)
            if len(coord) != side.shape[axis]:
                raise AlignmentError(f"'{name}' of {which} has {len(coord)} values but its "
                                     f"{name} axis has {side.shape[axis]}")


def _shape(evaluated: tuple, reference: tuple, shared: bool) -> None:
    if shared:
        points = evaluated[-1]
        if reference[-1] != points or int(np.prod(reference)) != points:
            raise AlignmentError(f"--shared-reference needs a single spectrum of {points} "
                                 f"points, got a reference of shape {reference}")
    elif reference != evaluated:
        raise AlignmentError(f"shapes differ: evaluated {evaluated}, reference {reference}; a "
                             "reference is compared row by row and is never broadcast (a "
                             "single spectrum shared by every row needs --shared-reference)")


def _compare(name: str, a: np.ndarray, b: np.ndarray) -> None:
    # Both sides are finite already: every carried coordinate passed _vector.
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if np.allclose(a, b, rtol=0.0, atol=TOLERANCE):
        return
    reversed_axis = np.allclose(a, b[::-1], rtol=0.0, atol=TOLERANCE)
    detail = ("the reference's axis is reversed" if reversed_axis
              else f"largest difference {float(np.max(np.abs(a - b))):.6g} "
                   f"(tolerance {TOLERANCE:g})")
    raise AlignmentError(f"'{name}' differs between the evaluated data and the reference: "
                         f"{detail}")


def check(evaluated: Side, reference: Side, *, same_file: bool, shared: bool,
          asserted: set) -> dict:
    """Verify alignment, or raise :class:`AlignmentError` naming the check and its reason.

    Returns ``verified``, ``asserted`` and ``not_applicable`` (disjoint lists in the order
    of :data:`CHECKS`) and ``row_correspondence`` (one of :data:`ROW_BASES`).
    """
    axes, not_applicable = _axes(evaluated, asserted)
    _check_lengths(evaluated, "the evaluated file", axes, with_axes=True)
    # Shapes first: the reference's axes are indexed by the evaluated layout only once the
    # shapes are known to agree (or the reference is one spectrum).
    _shape(evaluated.shape, reference.shape, shared)
    if len(axes) == 1 and not shared and reference.angles is not None and reference.times is not None:
        raise AlignmentError("a (rows, axis, energy) layout has one coordinate axis, but the "
                             "reference carries both 'angles' and 'times'")
    _check_lengths(reference, "the reference", axes, with_axes=not shared)
    verified = ["shape"]
    missing: dict = {}

    # Comparison coordinates. In the same file the reference lies on the file's own grid,
    # so a present coordinate is trivially equal; an absent one is still absent.
    for name in ("energy", *axes):
        ours, theirs = getattr(evaluated, name), getattr(reference, name)
        if shared and name != "energy":
            theirs = None
            reason = f"a shared reference carries no {name} axis"
        elif same_file:
            reason = f"the file carries no '{name}'"
        elif ours is None and theirs is None:
            reason = f"neither file carries '{name}'"
        elif ours is None:
            reason = f"the evaluated file carries no '{name}'"
        else:
            reason = f"the reference carries no '{name}'"
        if ours is None or theirs is None:
            missing[name] = reason
            continue
        if not same_file:
            _compare(name, ours, theirs)
        verified.append(name)

    # Units.
    if evaluated.units is not None and reference.units is not None:
        if evaluated.units != reference.units:
            raise AlignmentError(f"units differ: the evaluated arrays are in "
                                 f"'{evaluated.units}', the reference is declared in "
                                 f"'{reference.units}'")
        verified.append("units")
    else:
        notes = [n for n in (evaluated.units_note, reference.units_note) if n]
        missing["units"] = "; ".join(notes) or "units are not recorded"

    # Row correspondence.
    if same_file:
        verified.append("rows")
        basis = "same_file"
    else:
        reason = _identify_rows(evaluated, reference, shared, axes, verified)
        if reason is None:
            verified.append("rows")
            basis = "identifiers"
        else:
            missing["rows"] = reason
            basis = "asserted"

    # Assertions fill in only what is missing.
    for name in sorted(asserted):
        if name in not_applicable:
            raise AlignmentError(f"nothing to assert: '{name}' does not apply to a layout of "
                                 f"shape {evaluated.shape}")
        if name in verified:
            raise AlignmentError(f"the '{name}' check was made from the metadata and verified; "
                                 "an assertion is accepted only for a check that could not "
                                 "be made")
    unasserted = [n for n in CHECKS if n in missing and n not in asserted]
    if unasserted:
        raise AlignmentError(
            "alignment could not be verified: "
            + "; ".join(f"{n} ({missing[n]})" for n in unasserted)
            + ". Add the metadata, or assert what you know with "
            f"--assert-alignment {','.join(unasserted)}")
    return {
        "verified": [n for n in CHECKS if n in verified],
        "asserted": [n for n in CHECKS if n in asserted],
        "not_applicable": [n for n in CHECKS if n in not_applicable],
        "row_correspondence": basis,
    }


def _identify_rows(evaluated: Side, reference: Side, shared: bool, axes: dict,
                   verified: list) -> Optional[str]:
    """``None`` when identifiers cover every non-energy dimension, else why not. A
    detected mismatch of identifiers in one namespace is raised, never returned."""
    if shared:
        return "a shared reference carries no row identifiers"
    if evaluated.frame_index is None or reference.frame_index is None:
        side = "evaluated file" if evaluated.frame_index is None else "reference"
        return f"the {side} carries no 'frame_index'"
    if (evaluated.acquisition_id is None or reference.acquisition_id is None
            or evaluated.acquisition_id != reference.acquisition_id):
        return ("the reference's source acquisition is not the evaluated data's, and equal "
                "frame indices from different acquisitions establish nothing")
    ours = [int(i) for i in np.asarray(evaluated.frame_index).tolist()]
    theirs = [int(i) for i in np.asarray(reference.frame_index).tolist()]
    if ours != theirs:
        first = next(i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b)
        count = sum(a != b for a, b in zip(ours, theirs))
        raise AlignmentError(f"rows do not correspond: 'frame_index' differs at {count} of "
                             f"{len(ours)} rows within acquisition "
                             f"'{evaluated.acquisition_id}' (first at row {first}: evaluated "
                             f"{ours[first]}, reference {theirs[first]})")
    for name in axes:
        if name not in verified:
            return (f"the {name} axis is not verified, so identifiers do not cover every "
                    "non-energy dimension")
    return None


def report_line(alignment: dict) -> str:
    def fmt(names):
        return ", ".join(names) if names else "none"
    basis = alignment["row_correspondence"].replace("_", " ")
    verified = [f"{n} ({basis})" if n == "rows" else n for n in alignment["verified"]]
    return (f"Alignment verified: {fmt(verified)}; asserted: {fmt(alignment['asserted'])}; "
            f"not applicable: {fmt(alignment['not_applicable'])}")
