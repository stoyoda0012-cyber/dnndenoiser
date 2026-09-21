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
import torch
import torch.nn as nn

from dnndenoiser.models.network import DenoisingNetwork

__all__ = [
    "TARGET_LENGTH",
    "moving_average_targets",
    "train_selfsupervised",
    "denoise",
    "resample",
]

TARGET_LENGTH = 256
"""Length the network operates on. Stacks of any other length are resampled."""


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


def train_selfsupervised(
    frames: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int | None = None,
) -> DenoisingNetwork:
    """Train a ResNet-FCNN on (frame, self-supervised target) pairs.

    The hyperparameters are those of the archived reference and are **not**
    arguments: Adam(lr=1e-3, weight_decay=1e-9), StepLR(step_size=25,
    gamma=0.5), HuberLoss(delta=1.0), gradient-norm clipping at 4.0. Changing
    any of them makes this something other than the method being reproduced.

    Args:
        frames: ``(n, n_features)`` network inputs, already resampled and
            normalised by the caller.
        targets: ``(n, n_features)`` from :func:`moving_average_targets`.
        epochs, batch_size, device: training controls.
        seed: passed to ``torch.manual_seed`` before the model is constructed.
            **Construction consumes the random stream**, so a caller that builds
            anything else drawing from it first will get a different model from
            the same seed.

    Returns:
        The trained model, in eval mode.

    Three of the components named above are inert on the reference's own
    synthetic fixture, measured rather than assumed: with data normalised to
    [0, 1] the residuals never reach Huber's delta, so the loss is exactly
    ``0.5 * MSE`` there; ``StepLR`` never fires within 20 epochs; and the
    gradient norms stay ~86x below the clip. They are kept because they are part
    of the method, not because that fixture shows them doing anything. See
    ``docs/preregistration/P1-selfsupervised-moving-average.md``.
    """
    frames = np.asarray(frames)
    targets = np.asarray(targets)
    if frames.shape != targets.shape:
        raise ValueError(
            f"frames {frames.shape} and targets {targets.shape} must have the same shape"
        )

    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device(device)
    model = DenoisingNetwork(
        num_features=frames.shape[1],
        num_hidden_units=100,
        layer_type="ResNet-FCNN",
        encoder_output_dim=64,
    ).to(dev)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(frames, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-9)
    schedule = torch.optim.lr_scheduler.StepLR(optimiser, step_size=25, gamma=0.5)
    criterion = nn.HuberLoss(delta=1.0)

    model.train()
    for _ in range(epochs):
        for batch_frames, batch_targets in loader:
            batch_frames = batch_frames.to(dev)
            batch_targets = batch_targets.to(dev)
            optimiser.zero_grad()
            denoised, _ = model(batch_frames)
            loss = criterion(denoised, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimiser.step()
        schedule.step()

    model.eval()
    return model


def denoise(
    model: DenoisingNetwork,
    noisy: np.ndarray,
    device: str = "cpu",
    batch_size: int = 4096,
) -> np.ndarray:
    """Run a trained model over ``(n, n_features)`` spectra.

    The model has two heads; only the first is the denoised spectrum. Batching
    is for memory, not for semantics: the network is a stateless MLP with no
    batch statistics, so the result does not depend on ``batch_size`` beyond
    floating-point blocking in the underlying matrix multiply.
    """
    dev = torch.device(device)
    model.eval()
    out = []
    with torch.no_grad():
        inputs = torch.tensor(np.asarray(noisy), dtype=torch.float32).to(dev)
        for start in range(0, len(inputs), batch_size):
            denoised, _ = model(inputs[start:start + batch_size])
            out.append(denoised.cpu().numpy())
    return np.concatenate(out, axis=0)


def _interpolation_matrix(n_old: int, n_new: int) -> np.ndarray:
    """Linear-interpolation matrix mapping ``n_old`` points onto ``n_new``.

    Built in ``float32``, and the source index is clipped at ``n_old - 2`` so the
    final output point interpolates from the last interval rather than running
    off the end. Both are the reference's choices and both are load-bearing for
    equivalence.
    """
    x_old = np.linspace(0, 1, n_old)
    x_new = np.linspace(0, 1, n_new)
    matrix = np.zeros((n_new, n_old), dtype=np.float32)
    for j, x in enumerate(x_new):
        i = int(np.clip(np.searchsorted(x_old, x, side="right") - 1, 0, n_old - 2))
        frac = (x - x_old[i]) / (x_old[i + 1] - x_old[i])
        matrix[j, i] = 1.0 - frac
        matrix[j, i + 1] = frac
    return matrix


def resample(arr: np.ndarray, n_new: int) -> np.ndarray:
    """Linear-resample the last axis: ``(..., n_old) -> (..., n_new)``.

    The network is fixed at :data:`TARGET_LENGTH`, so every stack of another
    length passes through here.

    Note the no-op case returns the array **without copying** when it is already
    ``float32`` of the right length — the caller shares storage with the result.
    That is the reference's behaviour and is kept for equivalence; do not write
    into a resampled array you did not allocate.
    """
    arr = np.asarray(arr, dtype=np.float32)
    n_old = arr.shape[-1]
    if n_old == n_new:
        return arr
    return (arr @ _interpolation_matrix(n_old, n_new).T).astype(np.float32)
