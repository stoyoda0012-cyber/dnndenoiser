"""Criterion C5 of the P1 preregistration: the frame-stack schema.

C5 is a completeness condition, not a numerical one: the layout exists, it
carries acquisition order, it round-trips, and the reader rejects duplicate
acquisition indices.

The duplicate rule is the one with teeth. ``moving_average_targets`` excludes a
frame from its own neighbourhood **by position, not by index value**, so two
frames sharing an index become distance-0 neighbours of each other and each
leaks straight into the other's target — at ``W = 1``, the target simply *is*
the other frame. Nothing about the resulting training run looks wrong. The test
below asserts both that the reader refuses such a file and that the leak is real
if it does not, so the rule is guarded by its consequence rather than by taste.
"""
from __future__ import annotations

import numpy as np
import pytest

from dnndenoiser.data.frame_stack import read_frame_stack, write_frame_stack
from dnndenoiser.training.selfsupervised import moving_average_targets


@pytest.fixture
def stack():
    rng = np.random.default_rng(3)
    energy = np.linspace(280, 300, 64)
    clean = 100 * np.exp(-((energy - 290) ** 2) / (2 * 1.5**2)) + 5
    frames = rng.poisson(np.tile(clean, (12, 1))).astype(np.float32)
    return frames, energy, np.arange(12)


def test_c5_round_trip(tmp_path, stack):
    """What is written is what is read, including the acquisition order."""
    frames, energy, index = stack
    path = tmp_path / "stack.h5"
    write_frame_stack(path, frames, energy, index)

    got = read_frame_stack(path)
    assert np.array_equal(got.frames, frames)
    assert np.array_equal(got.energy, energy)
    assert np.array_equal(got.frame_index, index)
    assert (got.n_frames, got.n_energy) == (12, 64)


def test_c5_acquisition_order_survives_shuffled_rows(tmp_path, stack):
    """Rows need not be in acquisition order, and the index must say so.

    A file whose rows are shuffled but whose index records the true order must
    read back with that order intact — otherwise "temporally nearest" would be
    measured on row position, which is the defect C1(b) exists to catch.
    """
    frames, energy, _ = stack
    shuffled = np.random.default_rng(11).permutation(12)
    path = tmp_path / "shuffled.h5"
    write_frame_stack(path, frames[shuffled], energy, shuffled)

    got = read_frame_stack(path)
    assert np.array_equal(got.frame_index, shuffled)
    assert not np.array_equal(got.frame_index, np.arange(12))
    # and the targets built from it differ from the row-order reading
    by_order = moving_average_targets(got.frames, got.frame_index, 3)
    by_position = moving_average_targets(got.frames, np.arange(12), 3)
    assert not np.array_equal(by_order, by_position)


def test_c5_default_index_asserts_rows_are_in_acquisition_order(tmp_path, stack):
    frames, energy, _ = stack
    path = tmp_path / "default.h5"
    write_frame_stack(path, frames, energy)
    assert np.array_equal(read_frame_stack(path).frame_index, np.arange(12))


def test_c5_duplicate_indices_are_rejected(tmp_path, stack):
    """The rule, and the harm it prevents."""
    frames, energy, index = stack
    duplicated = index.copy()
    duplicated[1] = duplicated[0]

    with pytest.raises(ValueError, match="unique"):
        write_frame_stack(tmp_path / "dup.h5", frames, energy, duplicated)

    # The harm, demonstrated rather than asserted by fiat: with W=1 the target
    # of frame 0 is exactly frame 1, so the "self-supervised" target carries the
    # input's own neighbourhood with no independence at all.
    leaked = moving_average_targets(frames, duplicated, 1)
    assert np.array_equal(leaked[0], frames[1].astype(np.float64))


def test_c5_reader_rejects_a_file_written_around_the_writer(tmp_path, stack):
    """The reader validates too — the writer is not the only gate.

    A file can be produced by anything; the guarantee has to hold at read time.
    """
    h5py = pytest.importorskip("h5py")
    frames, energy, index = stack
    duplicated = index.copy()
    duplicated[1] = duplicated[0]
    path = tmp_path / "hand_written.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("frames", data=frames)
        handle.create_dataset("energy", data=energy)
        handle.create_dataset("frame_index", data=duplicated)

    with pytest.raises(ValueError, match="unique"):
        read_frame_stack(path)


@pytest.mark.parametrize(
    "drop", ["frames", "energy", "frame_index"]
)
def test_c5_missing_dataset_is_named(tmp_path, stack, drop):
    h5py = pytest.importorskip("h5py")
    frames, energy, index = stack
    path = tmp_path / "partial.h5"
    with h5py.File(path, "w") as handle:
        for name, data in (("frames", frames), ("energy", energy), ("frame_index", index)):
            if name != drop:
                handle.create_dataset(name, data=data)

    with pytest.raises(KeyError, match=drop):
        read_frame_stack(path)


def test_c5_float_index_is_rejected(tmp_path, stack):
    """Acquisition order is a count; a float index invites approximate ties."""
    frames, energy, index = stack
    with pytest.raises(ValueError, match="integer"):
        write_frame_stack(tmp_path / "float.h5", frames, energy, index.astype(float))


def test_c5_one_frame_is_rejected(tmp_path, stack):
    frames, energy, _ = stack
    with pytest.raises(ValueError, match="at least 2 frames"):
        write_frame_stack(tmp_path / "one.h5", frames[:1], energy, np.arange(1))


def test_c5_shape_disagreements_are_named(tmp_path, stack):
    frames, energy, index = stack
    with pytest.raises(ValueError, match="energy"):
        write_frame_stack(tmp_path / "bad.h5", frames, energy[:10], index)
    with pytest.raises(ValueError, match="frame_index"):
        write_frame_stack(tmp_path / "bad2.h5", frames, energy, index[:5])
