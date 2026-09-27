"""``dnndenoiser infer`` on a measured frame stack, as a data requester would run it.

Found by the field test of the self-supervised route on measured AR-HAXPES frames: the
file a model was trained on could not be passed to ``infer``. ``train`` read the stack's
``frames`` and resampled a stack that was not 256 points; ``infer`` read only ``noisy`` and
did not resample, so it failed with a ``KeyError`` or, once given a ``noisy`` copy, with a
PyTorch shape error that did not say why.

Also here: the recipe QUICK_START gives for training one model on several angle channels
of one stack, which rests on how ``frame_index`` decides the neighbourhood.
"""
from __future__ import annotations

import sys

import h5py
import numpy as np
import pytest

from dnndenoiser.cli import main
from dnndenoiser.data.frame_stack import write_frame_stack
from dnndenoiser.training.selfsupervised import moving_average_targets, resample


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["dnndenoiser", *argv])
    return main()


def _stack(path, n_energy, n_frames=12, seed=3):
    rng = np.random.default_rng(seed)
    energy = np.linspace(4977.6, 4959.0, n_energy)
    clean = 0.03 * np.exp(-((energy - 4968.0) ** 2) / (2 * 1.5**2)) + 0.002
    frames = (rng.poisson(np.tile(clean * 2000, (n_frames, 1))) / 2000).astype(np.float32)
    write_frame_stack(path, frames, energy)
    return frames, energy


@pytest.fixture
def trained(monkeypatch, tmp_path):
    """A model trained on a 300-point stack, as a measured 1024-channel stack would be."""
    stack = tmp_path / "stack.h5"
    frames, energy = _stack(stack, 300)
    model = tmp_path / "m.pt"
    run(monkeypatch, "train", "-d", str(stack), "-o", str(model),
        "--method", "moving-average", "--window", "1", "--epochs", "1", "--seed", "0",
        "--device", "cpu")
    return stack, model, frames, energy


def test_infer_takes_the_frame_stack_the_model_was_trained_on(monkeypatch, tmp_path, trained, capsys):
    stack, model, frames, energy = trained
    out = tmp_path / "out.h5"
    run(monkeypatch, "infer", "-d", str(stack), "-m", str(model), "-o", str(out), "--device", "cpu")
    printed = capsys.readouterr().out
    assert "Frame stack: denoising its 'frames'" in printed
    assert "Resampling 300 -> 256 points" in printed
    with h5py.File(out) as f:
        assert f["denoised"].shape == (len(frames), 256)
        assert f["noisy"].shape == (len(frames), 256)
        written_energy = f["energy"][:]
    assert len(written_energy) == 256
    assert written_energy[0] == pytest.approx(energy[0]) and written_energy[-1] == pytest.approx(energy[-1])


def test_the_resampled_input_is_what_training_saw(monkeypatch, tmp_path, trained):
    """Passing the 300-point stack gives the same output as passing it already resampled
    with the function training uses -- so inference sees the grid training saw."""
    stack, model, frames, energy = trained
    pre = tmp_path / "pre.h5"
    with h5py.File(pre, "w") as f:
        f.create_dataset("noisy", data=resample(frames, 256))
        f.create_dataset("energy", data=np.linspace(energy[0], energy[-1], 256))
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    run(monkeypatch, "infer", "-d", str(stack), "-m", str(model), "-o", str(a), "--device", "cpu")
    run(monkeypatch, "infer", "-d", str(pre), "-m", str(model), "-o", str(b), "--device", "cpu")
    with h5py.File(a) as fa, h5py.File(b) as fb:
        np.testing.assert_array_equal(fa["denoised"][:], fb["denoised"][:])


def test_a_clean_reference_is_resampled_with_the_input(monkeypatch, tmp_path, trained):
    stack, model, frames, energy = trained
    data = tmp_path / "with_clean.h5"
    with h5py.File(data, "w") as f:
        f.create_dataset("noisy", data=frames)
        f.create_dataset("clean", data=np.repeat(frames.mean(0, keepdims=True), len(frames), 0))
        f.create_dataset("energy", data=energy)
    out = tmp_path / "out.h5"
    run(monkeypatch, "infer", "-d", str(data), "-m", str(model), "-o", str(out), "--device", "cpu")
    with h5py.File(out) as f:
        assert f["clean"].shape == f["denoised"].shape == (len(frames), 256)


def test_a_file_with_neither_noisy_nor_frames_is_refused_by_name(monkeypatch, tmp_path, trained, capsys):
    _stack_path, model, frames, energy = trained
    data = tmp_path / "wrong.h5"
    with h5py.File(data, "w") as f:
        f.create_dataset("spectra", data=frames)
        f.create_dataset("energy", data=energy)
    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, "infer", "-d", str(data), "-m", str(model), "-o", str(tmp_path / "o.h5"),
            "--device", "cpu")
    assert exit_info.value.code == 1
    assert "neither a 'noisy' dataset nor a frame stack's 'frames'" in capsys.readouterr().err


def test_offset_frame_indices_keep_each_channels_neighbours_inside_the_channel():
    """QUICK_START's recipe for several angle channels in one stack: give channel k the
    indices k * stride + t. With the stride larger than any channel's length, every
    frame's W nearest others are in its own channel, for any W below that length."""
    rng = np.random.default_rng(0)
    n_per, n_channels, stride = 7, 3, 1_000_000
    # Each channel's frames carry the channel number, so a target's channel is readable.
    frames = np.concatenate([np.full((n_per, 4), k, dtype=np.float32) + rng.normal(0, 0.01, (n_per, 4))
                             for k in range(n_channels)])
    index = np.concatenate([k * stride + np.arange(n_per) for k in range(n_channels)])
    for W in (1, 3, n_per - 1):
        targets = moving_average_targets(frames, index, W)
        np.testing.assert_allclose(np.round(targets.mean(axis=1)), np.repeat(np.arange(n_channels), n_per))


def test_contiguous_indices_across_channels_would_mix_them():
    """The converse, so the test above can fail: numbering the channels one after another
    makes the last frame of one channel a neighbour of the first frame of the next."""
    n_per, n_channels = 7, 3
    frames = np.concatenate([np.full((n_per, 4), k, dtype=np.float32) for k in range(n_channels)])
    index = np.arange(n_per * n_channels)
    targets = moving_average_targets(frames, index, 1)
    channel_of_target = np.round(targets.mean(axis=1))
    assert not np.array_equal(channel_of_target, np.repeat(np.arange(n_channels), n_per))
