"""Loading a checkpoint must not run code from it unless asked.

``torch.load``'s full unpickler executes whatever the file says to execute, and
a denoiser CLI invites the risky case: trained checkpoints are the natural thing
to pass around, so ``infer -m someone-elses-model.pt`` is a plausible command.
v0.1.0 and v0.1.1 both shipped with the unpickler unconditional, and both are
archived, so the fix cannot be retroactive — it can only be the default from
here, with the unsafe path reachable by naming it.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from dnndenoiser.cli import load_checkpoint, main
from dnndenoiser.data.frame_stack import write_frame_stack


@pytest.fixture
def stack(tmp_path):
    rng = np.random.default_rng(5)
    energy = np.linspace(280, 300, 256)
    clean = 200 * np.exp(-((energy - 290) ** 2) / (2 * 1.2**2)) + 10
    path = tmp_path / "stack.h5"
    write_frame_stack(path, rng.poisson(np.tile(clean, (12, 1))).astype(np.float32), energy)
    return path


@pytest.fixture
def checkpoint(monkeypatch, tmp_path, stack):
    out = tmp_path / "model.pt"
    monkeypatch.setattr(sys, "argv", [
        "dnndenoiser", "train", "-d", str(stack), "-o", str(out),
        "--method", "moving-average", "--epochs", "1",
    ])
    main()
    return out


def test_a_checkpoint_this_version_writes_needs_no_unpickler(checkpoint):
    """The restricted loader must accept our own output, or the default is useless."""
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "model_state_dict" in loaded
    assert isinstance(loaded["energy"], torch.Tensor), (
        "a NumPy array here makes the whole checkpoint unloadable without "
        "unpickling it, which is the thing being avoided"
    )
    assert load_checkpoint(checkpoint)["architecture"] == "ResNet-FCNN"


def test_a_checkpoint_needing_the_unpickler_is_refused_by_default(tmp_path, capsys):
    """A NumPy array stands in for the v0.1.0/v0.1.1 format, and for anything worse.

    The point is not that NumPy is dangerous — it is that the restricted loader
    draws a line, and whatever sits on the far side of it is refused rather than
    executed.
    """
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": {}, "energy": np.arange(4, dtype=np.float32)}, path)

    with pytest.raises(SystemExit) as exit_info:
        load_checkpoint(path)
    assert exit_info.value.code == 1

    err = capsys.readouterr().err
    assert "--trust-checkpoint" in err, "the error must name the way forward"
    assert "runs code" in err, "and say what accepting it means"
    assert "v0.1.1" in err, "and why an older checkpoint hits this"


def test_the_flag_reaches_the_unpickler(tmp_path):
    """The escape hatch has to actually work, or users will reach for something worse."""
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": {}, "energy": np.arange(4, dtype=np.float32)}, path)

    loaded = load_checkpoint(path, trust=True)
    assert isinstance(loaded["energy"], np.ndarray)


def test_infer_refuses_an_unpicklable_checkpoint_without_the_flag(
    monkeypatch, tmp_path, capsys
):
    """End to end: the CLI, not only the helper."""
    model = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": {}, "energy": np.arange(4, dtype=np.float32)}, model)
    data = tmp_path / "data.h5"
    h5py = pytest.importorskip("h5py")
    with h5py.File(data, "w") as handle:
        handle.create_dataset("noisy", data=np.zeros((2, 256), dtype=np.float32))
        handle.create_dataset("energy", data=np.linspace(0, 1, 256))

    monkeypatch.setattr(sys, "argv", [
        "dnndenoiser", "infer", "-d", str(data), "-m", str(model), "-o", str(tmp_path / "o.h5"),
    ])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
    assert "--trust-checkpoint" in capsys.readouterr().err
