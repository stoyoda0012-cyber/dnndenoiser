"""HDF5 layout for a stack of repeated acquisitions of one spectrum.

The self-supervised moving-average method
(:mod:`dnndenoiser.training.selfsupervised`) trains from repeated acquisitions
rather than from clean/noisy pairs, so it needs a different file layout from the
``noisy``/``clean`` schema the rest of the CLI uses.

Schema
------

=================  ==============================  ===============================
Dataset            Shape                           Notes
=================  ==============================  ===============================
``frames``         ``(n_frames, energy)`` or       the acquired frames, in any
                   ``(n_frames, n_angles, energy)`` order; energy is the last axis
``energy``         ``(energy,)``                   the energy axis
``frame_index``    ``(n_frames,)``                 acquisition order, integer,
                                                   unique, shared by a frame's
                                                   channels; optional attribute
                                                   ``order_basis``
``angles``         ``(n_angles,)``                 required with a 3-D ``frames``;
                                                   attributes ``angle_kind`` and
                                                   ``angle_units``
=================  ==============================  ===============================

**Channels** (``docs/design/FRAME_STACK_CHANNELS.md``). A 3-D ``frames`` is one frame per
acquisition of every angle channel. ``angles`` gives one finite, non-repeated value per
channel in the order of the channel axis (never sorted), with ``angle_kind`` one of
``emission``, ``analyser`` or ``other:<description>`` and ``angle_units`` ``deg``. A 3-D
stack may not carry ``times``. A 2-D stack is read exactly as before: an ``angles`` or
``times`` dataset there is accepted and not read.

**Order basis.** ``frame_index`` may carry ``order_basis``: ``recorded`` (from the
instrument), ``inferred`` (derived by the user) or ``unknown``; absent reads as
``unknown``. It is a declaration and nothing checks it.

``frame_index`` is the load-bearing part and is why this is a schema rather than
just an array. "Temporally nearest" is measured on it, so a file that omits it,
or that stores frames in acquisition order without saying so, cannot be
distinguished from one whose frames were shuffled — and the targets would be
built from the wrong neighbours without anything going wrong visibly.

**Indices must be unique, and this reader rejects duplicates.** A frame is
excluded from its own neighbourhood by position, not by index value, so two
frames sharing an index become distance-0 "other" frames of each other: each
leaks straight into the other's target and leave-one-out is defeated silently.
At ``W = 1`` the target simply *is* the other frame.

This is the project's own schema, not an instrument format. Reading vendor files
is out of scope (``AGENTS.md`` §1); convert to this layout first.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["FrameStack", "read_frame_stack", "write_frame_stack"]

FRAMES = "frames"
ENERGY = "energy"
FRAME_INDEX = "frame_index"
ANGLES = "angles"
TIMES = "times"
ANGLE_KIND = "angle_kind"
ANGLE_UNITS = "angle_units"
ORDER_BASIS = "order_basis"
ANGLE_KINDS = ("emission", "analyser")
ANGLE_UNIT_VALUES = ("deg",)
ORDER_BASES = ("recorded", "inferred", "unknown")
OTHER_KIND_MAX = 200


@dataclass(frozen=True)
class FrameStack:
    """A validated frame stack.

    Attributes:
        frames: ``(n_frames, n_energy)``.
        energy: ``(n_energy,)``.
        frame_index: ``(n_frames,)`` integer acquisition order, unique and in the
            same row order as ``frames`` — not sorted, because the file records
            the order it was acquired in, not a tidied version of it.
    """

    frames: np.ndarray
    energy: np.ndarray
    frame_index: np.ndarray
    angles: np.ndarray | None = None
    angle_kind: str | None = None
    angle_units: str | None = None
    order_basis: str = "unknown"

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_energy(self) -> int:
        """Energy is the last axis, for 2-D and 3-D stacks alike."""
        return int(self.frames.shape[-1])

    @property
    def n_angles(self) -> int:
        """1 for a 2-D stack."""
        return int(self.frames.shape[1]) if self.frames.ndim == 3 else 1


def attribute_text(value, name: str) -> str | None:
    """A string attribute stored as ``str`` or ``bytes``; anything else is refused."""
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"the attribute '{name}' is not valid UTF-8") from None
    if isinstance(value, (str, np.str_)):
        return str(value)
    raise ValueError(f"the attribute '{name}' must be a string, got {type(value).__name__}")


def validate_order_basis(value) -> str:
    """``order_basis`` as declared, ``unknown`` when absent; outside the vocabulary refused."""
    text = attribute_text(value, ORDER_BASIS)
    if text is None:
        return "unknown"
    if text not in ORDER_BASES:
        raise ValueError(f"'{FRAME_INDEX}' attribute '{ORDER_BASIS}' must be one of "
                         f"{', '.join(ORDER_BASES)}, got {text!r}")
    return text


def validate_angle_kind(value) -> str | None:
    text = attribute_text(value, ANGLE_KIND)
    if text is None:
        return None
    if text in ANGLE_KINDS:
        return text
    if text.startswith("other:"):
        description = text[len("other:"):]
        if description.strip() and len(description) <= OTHER_KIND_MAX:
            return text
    raise ValueError(f"'{ANGLES}' attribute '{ANGLE_KIND}' must be one of "
                     f"{', '.join(ANGLE_KINDS)} or 'other:<description>' (non-empty, at "
                     f"most {OTHER_KIND_MAX} characters), got {text!r}")


def validate_angle_units(value) -> str | None:
    text = attribute_text(value, ANGLE_UNITS)
    if text is None:
        return None
    if text not in ANGLE_UNIT_VALUES:
        raise ValueError(f"'{ANGLES}' attribute '{ANGLE_UNITS}' must be "
                         f"{', '.join(ANGLE_UNIT_VALUES)}, got {text!r}")
    return text


def check_channel_axis(frames_shape: tuple, angles, angle_kind, angle_units,
                       has_times: bool) -> None:
    """The channel-axis rules of a 3-D stack (also applied by ``infer``)."""
    if len(frames_shape) != 3:
        return
    n_angles = frames_shape[1]
    if angles is None:
        raise ValueError(f"a 3-D '{FRAMES}' {tuple(frames_shape)} needs an '{ANGLES}' "
                         "dataset naming its channels")
    arr = np.asarray(angles)
    if arr.ndim != 1 or arr.dtype.kind not in "iuf":
        raise ValueError(f"'{ANGLES}' must be a one-dimensional integer or float array, got "
                         f"shape {arr.shape} and dtype {arr.dtype}")
    if len(arr) != n_angles:
        raise ValueError(f"'{ANGLES}' has {len(arr)} values but '{FRAMES}' has {n_angles} "
                         "channels")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"'{ANGLES}' contains non-finite values")
    if len(np.unique(arr)) != len(arr):
        raise ValueError(f"'{ANGLES}' repeats a value: a channel coordinate must identify "
                         "its channel")
    if validate_angle_kind(angle_kind) is None:
        raise ValueError(f"'{ANGLES}' needs the attribute '{ANGLE_KIND}' "
                         f"({', '.join(ANGLE_KINDS)} or other:<description>)")
    if validate_angle_units(angle_units) is None:
        raise ValueError(f"'{ANGLES}' needs the attribute '{ANGLE_UNITS}' "
                         f"({', '.join(ANGLE_UNIT_VALUES)})")
    if has_times:
        raise ValueError(f"a 3-D frame stack may not carry '{TIMES}': its output would carry "
                         "both angles and times on a one-axis layout")


def _validate(frames: np.ndarray, energy: np.ndarray, frame_index: np.ndarray) -> None:
    if frames.ndim not in (2, 3):
        raise ValueError(f"'{FRAMES}' must be 2-D (n_frames, energy) or 3-D (n_frames, "
                         f"n_angles, energy), got {frames.shape}")
    if energy.ndim != 1:
        raise ValueError(f"'{ENERGY}' must be 1-D, got shape {energy.shape}")
    if frame_index.ndim != 1:
        raise ValueError(f"'{FRAME_INDEX}' must be 1-D, got shape {frame_index.shape}")
    if energy.shape[0] != frames.shape[-1]:
        raise ValueError(
            f"'{ENERGY}' has {energy.shape[0]} points but '{FRAMES}' has "
            f"{frames.shape[-1]} per frame"
        )
    if frame_index.shape[0] != frames.shape[0]:
        raise ValueError(
            f"'{FRAME_INDEX}' has {frame_index.shape[0]} entries but there are "
            f"{frames.shape[0]} frames"
        )
    if not np.issubdtype(frame_index.dtype, np.integer):
        raise ValueError(
            f"'{FRAME_INDEX}' must be an integer dtype, got {frame_index.dtype}. "
            "Acquisition order is a count, and a float index invites ties that "
            "compare equal only approximately."
        )
    if frames.shape[0] < 2:
        raise ValueError(
            f"a frame stack needs at least 2 frames to build leave-one-out "
            f"targets, got {frames.shape[0]}"
        )

    unique, counts = np.unique(frame_index, return_counts=True)
    if unique.size != frame_index.size:
        repeated = unique[counts > 1]
        raise ValueError(
            f"'{FRAME_INDEX}' must be unique; {repeated.size} value(s) repeat "
            f"(first: {repeated[0]}). Two frames sharing an acquisition index "
            "become distance-0 neighbours of each other, so each leaks directly "
            "into the other's self-supervised target."
        )


def read_frame_stack(path: str | Path) -> FrameStack:
    """Read and validate a frame stack.

    Raises:
        KeyError: if a required dataset is missing.
        ValueError: if the shapes disagree, the index is not integral, there are
            fewer than two frames, or acquisition indices repeat.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        missing = [name for name in (FRAMES, ENERGY, FRAME_INDEX) if name not in handle]
        if missing:
            raise KeyError(
                f"{path}: missing dataset(s) {', '.join(missing)}. A frame stack "
                f"needs '{FRAMES}', '{ENERGY}' and '{FRAME_INDEX}'; see "
                "dnndenoiser.data.frame_stack for the schema."
            )
        frames = np.asarray(handle[FRAMES][:])
        energy = np.asarray(handle[ENERGY][:])
        frame_index = np.asarray(handle[FRAME_INDEX][:])
        order_basis = validate_order_basis(handle[FRAME_INDEX].attrs.get(ORDER_BASIS))
        angles = angle_kind = angle_units = None
        if frames.ndim == 3:
            if ANGLES in handle:
                angles = np.asarray(handle[ANGLES][()])
                angle_kind = handle[ANGLES].attrs.get(ANGLE_KIND)
                angle_units = handle[ANGLES].attrs.get(ANGLE_UNITS)
            check_channel_axis(frames.shape, angles, angle_kind, angle_units,
                               TIMES in handle)
            angle_kind = validate_angle_kind(angle_kind)
            angle_units = validate_angle_units(angle_units)

    _validate(frames, energy, frame_index)
    return FrameStack(frames=frames, energy=energy, frame_index=frame_index, angles=angles,
                      angle_kind=angle_kind, angle_units=angle_units, order_basis=order_basis)


def write_frame_stack(
    path: str | Path,
    frames: np.ndarray,
    energy: np.ndarray,
    frame_index: np.ndarray | None = None,
    *,
    angles: np.ndarray | None = None,
    angle_kind: str | None = None,
    angle_units: str | None = None,
    order_basis: str | None = None,
) -> None:
    """Write a frame stack, validating it first.

    Args:
        frame_index: acquisition order. Defaults to ``arange(n_frames)``, which
            asserts that the rows of ``frames`` *are* in acquisition order — pass
            it explicitly whenever they are not.
        angles, angle_kind, angle_units: the channel coordinate of a 3-D ``frames``.
        order_basis: what the order rests on (``recorded``, ``inferred``, ``unknown``).
            Not written unless given, so a stack never claims ``recorded`` by default.
    """
    import h5py

    frames = np.asarray(frames)
    energy = np.asarray(energy)
    if frame_index is None:
        frame_index = np.arange(len(frames))
    frame_index = np.asarray(frame_index)

    _validate(frames, energy, frame_index)
    if order_basis is not None:
        validate_order_basis(order_basis)
    if frames.ndim == 3:
        check_channel_axis(frames.shape, angles, angle_kind, angle_units, False)
    elif angles is not None:
        raise ValueError(f"'{ANGLES}' names the channels of a 3-D '{FRAMES}'; this one is 2-D")

    with h5py.File(path, "w") as handle:
        handle.create_dataset(FRAMES, data=frames)
        handle.create_dataset(ENERGY, data=energy)
        handle.create_dataset(FRAME_INDEX, data=frame_index)
        if order_basis is not None:
            handle[FRAME_INDEX].attrs[ORDER_BASIS] = order_basis
        if angles is not None:
            handle.create_dataset(ANGLES, data=np.asarray(angles))
            handle[ANGLES].attrs[ANGLE_KIND] = angle_kind
            handle[ANGLES].attrs[ANGLE_UNITS] = angle_units
