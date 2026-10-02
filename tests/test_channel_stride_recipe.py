"""QUICK_START's recipe for several channels in one stack (moving-average).

Channel *k*'s frames get the indices k * stride + t. Each frame's value is its channel's
number, so a target that is not exactly that number mixed channels.
"""
from __future__ import annotations

import numpy as np
import pytest

from dnndenoiser.training.selfsupervised import moving_average_targets


def mixed_rows(n, channels, stride, window):
    frames = np.concatenate([np.full((n, 3), float(c)) for c in range(channels)])
    index = np.concatenate([c * stride + np.arange(n) for c in range(channels)])
    own = np.repeat(np.arange(channels), n).astype(float)
    targets = moving_average_targets(frames, index, window)[:, 0]
    return int(np.sum(targets != own))


@pytest.mark.parametrize("n", [3, 10, 50])
def test_a_stride_of_frames_plus_window_keeps_every_target_in_its_channel(n):
    for window in range(1, n):
        assert mixed_rows(n, 3, n + window, window) == 0, window


@pytest.mark.parametrize("n", [10, 50])
def test_a_stride_of_frames_plus_one_mixes_channels_from_a_window_of_three(n):
    """The stride an earlier QUICK_START gave as safe."""
    assert mixed_rows(n, 3, n + 1, 1) == 0
    assert all(mixed_rows(n, 3, n + 1, w) > 0 for w in range(3, n))
