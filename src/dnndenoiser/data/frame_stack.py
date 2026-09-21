"""HDF5 layout for a stack of repeated acquisitions of one spectrum.

The self-supervised moving-average method
(:mod:`dnndenoiser.training.selfsupervised`) trains from repeated acquisitions
rather than from clean/noisy pairs, so it needs a different file layout from the
``noisy``/``clean`` schema the rest of the CLI uses.

Schema
------

=================  ======================  ====================================
Dataset            Shape                   Notes
=================  ======================  ====================================
``frames``         ``(n_frames, energy)``  the acquired frames, in any order
``energy``         ``(energy,)``           the energy axis
``frame_index``    ``(n_frames,)``         acquisition order, integer, unique
=================  ======================  ====================================

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

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_energy(self) -> int:
        return int(self.frames.shape[1])


def _validate(frames: np.ndarray, energy: np.ndarray, frame_index: np.ndarray) -> None:
    if frames.ndim != 2:
        raise ValueError(f"'{FRAMES}' must be 2-D (n_frames, energy), got {frames.shape}")
    if energy.ndim != 1:
        raise ValueError(f"'{ENERGY}' must be 1-D, got shape {energy.shape}")
    if frame_index.ndim != 1:
        raise ValueError(f"'{FRAME_INDEX}' must be 1-D, got shape {frame_index.shape}")
    if energy.shape[0] != frames.shape[1]:
        raise ValueError(
            f"'{ENERGY}' has {energy.shape[0]} points but '{FRAMES}' has "
            f"{frames.shape[1]} per frame"
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

    _validate(frames, energy, frame_index)
    return FrameStack(frames=frames, energy=energy, frame_index=frame_index)


def write_frame_stack(
    path: str | Path,
    frames: np.ndarray,
    energy: np.ndarray,
    frame_index: np.ndarray | None = None,
) -> None:
    """Write a frame stack, validating it first.

    Args:
        frame_index: acquisition order. Defaults to ``arange(n_frames)``, which
            asserts that the rows of ``frames`` *are* in acquisition order — pass
            it explicitly whenever they are not.
    """
    import h5py

    frames = np.asarray(frames)
    energy = np.asarray(energy)
    if frame_index is None:
        frame_index = np.arange(len(frames))
    frame_index = np.asarray(frame_index)

    _validate(frames, energy, frame_index)

    with h5py.File(path, "w") as handle:
        handle.create_dataset(FRAMES, data=frames)
        handle.create_dataset(ENERGY, data=energy)
        handle.create_dataset(FRAME_INDEX, data=frame_index)
