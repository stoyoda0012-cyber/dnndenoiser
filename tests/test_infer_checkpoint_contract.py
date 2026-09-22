"""What ``infer`` must read out of a checkpoint, and must not guess.

Two defects, found by the implementation audit and both mine:

- The two training paths wrote the model's shape under different key names —
  ``n_features`` from noise2clean, ``num_features`` from the moving-average
  path added in v0.1.1 — and ``infer`` read only the first set, falling through
  to defaults for the second. The defaults happened to equal the real values,
  so nothing was wrong; ``load_state_dict`` would have caught a shape mismatch
  anyway, as a message about ``encoder.0.weight`` rather than about the file.
- ``infer`` never applied the normalisation the moving-average path records.
  That one **was** wrong, twice: a model trained on data scaled to [0, 1] was
  fed raw counts, and its output was written into a file beside a ``noisy``
  dataset in counts, in a space the file did not name.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from dnndenoiser.cli import checkpoint_model_config, main
from dnndenoiser.data.frame_stack import write_frame_stack


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["dnndenoiser", *argv])
    return main()


@pytest.fixture
def frames():
    rng = np.random.default_rng(5)
    energy = np.linspace(280, 300, 256)
    clean = 200 * np.exp(-((energy - 290) ** 2) / (2 * 1.2**2)) + 10
    return rng.poisson(np.tile(clean, (20, 1))).astype(np.float32), energy


@pytest.fixture
def moving_average_model(monkeypatch, tmp_path, frames):
    data, energy = frames
    stack = tmp_path / "stack.h5"
    write_frame_stack(stack, data, energy)
    out = tmp_path / "ma.pt"
    run(monkeypatch, "train", "-d", str(stack), "-o", str(out),
        "--method", "moving-average", "--epochs", "2")
    return out


@pytest.mark.parametrize(
    "spellings",
    [
        {"num_features": 128, "num_hidden_units": 50, "encoder_output_dim": 32},
        {"n_features": 128, "hidden_units": 50, "encoder_dim": 32},
    ],
    ids=["moving-average-spelling", "noise2clean-spelling"],
)
def test_both_published_key_spellings_are_read(spellings):
    """Both are in archived releases, so both must be readable."""
    config = checkpoint_model_config({"architecture": "FCNN", **spellings}, "x.pt")
    assert config == {
        "architecture": "FCNN",
        "num_features": 128,
        "num_hidden_units": 50,
        "encoder_output_dim": 32,
    }


@pytest.mark.parametrize("missing", ["num_features", "num_hidden_units", "encoder_output_dim"])
def test_a_missing_shape_is_an_error_not_a_default(capsys, missing):
    """Guessing turns "the file does not say" into a shape error deep in torch."""
    checkpoint = {
        "architecture": "FCNN",
        "num_features": 128, "num_hidden_units": 50, "encoder_output_dim": 32,
    }
    del checkpoint[missing]
    with pytest.raises(SystemExit) as exit_info:
        checkpoint_model_config(checkpoint, "broken.pt")
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "broken.pt" in err and missing in err


def test_a_missing_architecture_is_an_error(capsys):
    with pytest.raises(SystemExit):
        checkpoint_model_config(
            {"num_features": 128, "num_hidden_units": 50, "encoder_output_dim": 32},
            "broken.pt",
        )
    assert "architecture" in capsys.readouterr().err


def test_infer_applies_and_inverts_the_recorded_normalisation(
    monkeypatch, tmp_path, frames, moving_average_model, capsys
):
    """The output must come back in the input's units, and say that it did."""
    h5py = pytest.importorskip("h5py")
    data, energy = frames
    probe = data[:4]

    inp = tmp_path / "noisy.h5"
    with h5py.File(inp, "w") as handle:
        handle.create_dataset("noisy", data=probe)
        handle.create_dataset("energy", data=energy)

    out = tmp_path / "denoised.h5"
    run(monkeypatch, "infer", "-d", str(inp), "-m", str(moving_average_model), "-o", str(out))

    printed = capsys.readouterr().out
    assert "Applied the checkpoint's normalisation" in printed
    assert "Inverted the normalisation" in printed
    assert "training stack's, not this file's" in printed, (
        "applying one dataset's constants to another is an assumption and has "
        "to be stated, not buried"
    )

    with h5py.File(out) as handle:
        denoised = handle["denoised"][:]

    # In counts, not in [0, 1]: the failure this replaces wrote normalised
    # numbers beside a `noisy` dataset in counts, with nothing to distinguish
    # them. Two epochs of training is not expected to denoise well, so the test
    # is about the units, not the quality.
    assert denoised.shape == probe.shape
    assert np.ptp(denoised) > 1.0, (
        f"output spans only {np.ptp(denoised):.3g} — it looks like normalised "
        f"space, so the inverse was not applied"
    )


def test_a_checkpoint_without_normalisation_is_left_alone(monkeypatch, tmp_path, capsys):
    """noise2clean checkpoints record none, and their behaviour must not change."""
    h5py = pytest.importorskip("h5py")
    gen = tmp_path / "gen.h5"
    run(monkeypatch, "generate", "-o", str(gen), "-n", "16", "--peak-set", "C1s_single")
    model = tmp_path / "n2c.pt"
    run(monkeypatch, "train", "-d", str(gen), "-o", str(model), "--epochs", "1")

    assert "normalisation" not in torch.load(model, map_location="cpu", weights_only=True)

    out = tmp_path / "o.h5"
    run(monkeypatch, "infer", "-d", str(gen), "-m", str(model), "-o", str(out))
    printed = capsys.readouterr().out
    assert "Applied the checkpoint's normalisation" not in printed
    with h5py.File(out) as handle:
        assert handle["denoised"].shape[-1] == 256
