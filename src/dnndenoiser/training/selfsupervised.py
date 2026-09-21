"""Self-supervised training targets from a stack of repeated acquisitions.

The method trains without a clean reference. For each acquired frame the target
is the mean of its ``W`` temporally nearest **other** frames at the same pixel —
leave-one-out, so that *for independent frames* the target's noise is
independent of the input's. Larger ``W`` corresponds to a longer effective
exposure in the target.

This is a port of the method archived as ``arhaxpes_denoise`` (Zenodo
``10.5281/zenodo.22092109`` v1.0.0). Its acceptance criteria were registered
before the port existed; see ``docs/preregistration/P1-selfsupervised-moving-average.md``.
Nothing here imports or vendors that package.

**Independence is an assumption, not a guarantee.** On a measured stack subject
to drift or charging, temporally adjacent frames are correlated, and temporal
nearness is not statistical independence. The leave-one-out construction cannot
detect that; it is the caller's to establish.
"""
from __future__ import annotations

import numpy as np

__all__ = ["moving_average_targets"]


def moving_average_targets(
    frames: np.ndarray,
    frame_indices: np.ndarray,
    W: int,
) -> np.ndarray:
    """Build leave-one-out moving-average targets for a frame stack.

    For frame ``i`` the target is the mean of the ``W`` frames whose acquisition
    index is nearest to ``frame_indices[i]``, excluding frame ``i`` itself.

    Args:
        frames: ``(n_frames, n_features)``. Upcast to ``float64`` internally, so
            the returned array is ``float64`` whatever the input dtype.
        frame_indices: ``(n_frames,)`` acquisition-order index per frame. This is
            what "temporally nearest" is measured on, and it need not be sorted
            or contiguous.
        W: number of neighbour frames to average. Clamped to ``n_frames - 1``.

    Returns:
        ``(n_frames, n_features)`` float64 targets.

    Raises:
        ValueError: if fewer than two frames are given, or if ``W < 1`` — in
            either case no leave-one-out neighbourhood exists.

    Two behaviours worth knowing, both inherited from the reference and both
    guarded by the acceptance tests:

    **Ties are resolved by ``numpy.argsort``'s default ordering.** Acquisition
    indices spaced evenly put neighbours at equal distance on both sides
    (``1, 1, 2, 2, …``), so for odd ``W`` the outermost slot is a genuine tie.
    The default sort is introsort, which is *not* stable, and numpy does not
    contract its tie-breaking. Selecting neighbours any other way — a stable
    sort, ``argpartition`` — gives a different, equally defensible answer: on a
    200-frame stack the two disagree for 86 rows at ``W=1`` and 107 rows at
    ``W=5``. This function reproduces the reference's choice so that results
    remain comparable, and the tests pin it.

    **A frame is excluded from its own neighbourhood by position, not by index
    value.** Two frames sharing an acquisition index therefore become distance-0
    "other" frames of each other, and each leaks directly into the other's
    target — which defeats leave-one-out silently. Callers reading acquisition
    order from a file must reject duplicates; the HDF5 frame-stack reader does.
    """
    frames = np.asarray(frames, dtype=float)
    frame_indices = np.asarray(frame_indices)

    if frames.ndim != 2:
        raise ValueError(f"frames must be 2-D (n_frames, n_features), got {frames.shape}")
    if frame_indices.ndim != 1:
        raise ValueError(f"frame_indices must be 1-D, got shape {frame_indices.shape}")
    if len(frame_indices) != len(frames):
        raise ValueError(
            f"frames and frame_indices disagree: {len(frames)} frames, "
            f"{len(frame_indices)} indices"
        )

    n = len(frame_indices)
    actual_W = min(int(W), n - 1)
    if actual_W < 1:
        raise ValueError(
            f"no leave-one-out neighbourhood exists: {n} frame(s) with W={W}. "
            "At least two frames and W >= 1 are required."
        )

    t = np.asarray(frame_indices, dtype=np.float64)
    distance = np.abs(t[:, None] - t[None, :])
    np.fill_diagonal(distance, np.inf)  # a frame is not its own neighbour
    neighbours = np.argsort(distance, axis=1)[:, :actual_W]

    return frames[neighbours].mean(axis=1)
