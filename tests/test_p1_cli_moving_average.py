"""``dnndenoiser train --method moving-average`` end to end (P1, C5).

The CLI path is what makes the method usable on a measured stack, and it is
where the registered constraints have to survive contact with a user: fixed
hyperparameters that must not be silently overridden, a schema whose acquisition
order must actually reach the target construction, and an evaluation caveat that
must be stated rather than assumed read.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from dnndenoiser.cli import main
from dnndenoiser.data.frame_stack import write_frame_stack


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["dnndenoiser", *argv])
    return main()


@pytest.fixture
def stack_file(tmp_path):
    rng = np.random.default_rng(5)
    energy = np.linspace(280, 300, 256)
    clean = 200 * np.exp(-((energy - 290) ** 2) / (2 * 1.2**2)) + 10
    frames = rng.poisson(np.tile(clean, (24, 1))).astype(np.float32)
    path = tmp_path / "stack.h5"
    write_frame_stack(path, frames, energy)
    return path


def test_trains_from_a_frame_stack_with_no_clean_reference(monkeypatch, tmp_path, stack_file, capsys):
    out = tmp_path / "model.pt"
    run(monkeypatch, "train", "-d", str(stack_file), "-o", str(out),
        "--method", "moving-average", "--window", "5", "--epochs", "2", "--seed", "0")

    assert out.exists()
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["training_method"] == "moving-average"
    assert ckpt["window"] == 5
    assert ckpt["architecture"] == "ResNet-FCNN"
    assert ckpt["num_features"] == 256
    assert ckpt["n_frames"] == 24
    assert ckpt["normalisation"]["kind"] == "element-global min-max"
    assert ckpt["normalisation"]["max"] > ckpt["normalisation"]["min"]

    printed = capsys.readouterr().out
    assert "no clean reference" in printed, (
        "the evaluation caveat must be stated: a mean over the same frames is not "
        "independent of the targets built from them"
    )


@pytest.mark.parametrize(
    "flag, value",
    [("--lr", "0.05"), ("--scheduler", "cosine"), ("--arch", "FCNN"), ("--weight-decay", "1e-5")],
)
def test_refuses_flags_that_would_change_the_method(monkeypatch, tmp_path, stack_file, capsys, flag, value):
    """A knob that does not apply must be refused, not ignored.

    Ignoring it would train something that is not the method being reproduced
    while reporting that it is.
    """
    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, "train", "-d", str(stack_file), "-o", str(tmp_path / "m.pt"),
            "--method", "moving-average", "--epochs", "1", flag, value)
    assert exit_info.value.code == 1
    assert "does not take" in capsys.readouterr().err


def test_refuses_a_fixed_flag_even_at_its_default_value(monkeypatch, tmp_path, stack_file, capsys):
    """Presence is what is refused, not a changed value.

    Keying on "differs from the parser default" would let ``--arch`` through in
    silence: its default is ``FCNN`` while this method is always ResNet-FCNN, so
    the untouched value is already the wrong one and agreement with it means
    nothing. The question is whether the flag applies here, and it does not.
    """
    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, "train", "-d", str(stack_file), "-o", str(tmp_path / "m.pt"),
            "--method", "moving-average", "--epochs", "1", "--arch", "ResNet-FCNN")
    assert exit_info.value.code == 1
    assert "--arch" in capsys.readouterr().err


def test_the_default_arch_is_not_what_this_method_uses(monkeypatch, tmp_path, stack_file):
    """The trap the refusal rule exists for, as a fact about the parser.

    The first version of this test asserted only that the checkpoint says
    ResNet-FCNN, which is true however ``--arch`` is defaulted — changing the
    parser default to ResNet-FCNN left all eleven CLI tests passing and quietly
    made the refusal rule's whole justification untestable. It now reads the
    default out of the parser.
    """
    from dnndenoiser.cli import build_parser

    train = build_parser()._subparsers._group_actions[0].choices["train"]
    assert train.get_default("arch") == "FCNN", (
        "the refusal rule keys on a flag being *present* rather than on it "
        "differing from the default, because this default is not what the "
        "method uses. If the default ever becomes ResNet-FCNN, that rationale "
        "needs rewriting rather than silently becoming moot."
    )

    out = tmp_path / "m.pt"
    run(monkeypatch, "train", "-d", str(stack_file), "-o", str(out),
        "--method", "moving-average", "--epochs", "1")
    assert torch.load(out, map_location="cpu", weights_only=False)["architecture"] == "ResNet-FCNN"


@pytest.mark.parametrize(
    "written", ["--lr=0.05", "--arch=FCNN", "--weight-deca", "--grad-clip", "--noise-level"],
    ids=["equals-form", "equals-form-arch", "abbreviation", "unlisted-clip", "unlisted-noise"],
)
def test_refusal_cannot_be_written_around(monkeypatch, tmp_path, stack_file, capsys, written):
    """Every form that once bypassed the rule.

    A literal ``token in sys.argv`` test saw only ``--lr 0.05``. ``--lr=0.05``,
    argparse's unique-prefix abbreviations, and two flags that were simply
    missing from the list all went through silently — ``--grad-clip`` worst of
    all, since it names a component this method fixes at 4.0, so a user could
    ask for a different one and get a checkpoint stamped with the method's name.
    """
    argv = ["train", "-d", str(stack_file), "-o", str(tmp_path / "m.pt"),
            "--method", "moving-average", "--epochs", "1", written]
    if "=" not in written:
        argv.append("1e-5" if written != "--grad-clip" else "0.001")
    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, *argv)
    assert exit_info.value.code == 1
    assert "does not take" in capsys.readouterr().err


def test_rejects_a_stack_with_duplicate_acquisition_indices(monkeypatch, tmp_path, capsys):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(6)
    energy = np.linspace(0, 1, 256)
    frames = rng.poisson(np.full((8, 256), 50.0)).astype(np.float32)
    index = np.arange(8)
    index[3] = index[2]
    path = tmp_path / "dup.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("frames", data=frames)
        handle.create_dataset("energy", data=energy)
        handle.create_dataset("frame_index", data=index)

    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, "train", "-d", str(path), "-o", str(tmp_path / "m.pt"),
            "--method", "moving-average", "--epochs", "1")
    assert exit_info.value.code == 1
    assert "unique" in capsys.readouterr().err


def test_rejects_the_spectra_schema_by_name(monkeypatch, tmp_path, capsys):
    """A ``noisy``/``clean`` file is the wrong layout; say which datasets are missing."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "spectra.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("noisy", data=np.zeros((4, 256), dtype=np.float32))
        handle.create_dataset("energy", data=np.linspace(0, 1, 256))

    with pytest.raises(SystemExit):
        run(monkeypatch, "train", "-d", str(path), "-o", str(tmp_path / "m.pt"),
            "--method", "moving-average", "--epochs", "1")
    err = capsys.readouterr().err
    assert "frames" in err and "frame_index" in err


def test_resamples_a_stack_that_is_not_256_points(monkeypatch, tmp_path, capsys):
    rng = np.random.default_rng(7)
    energy = np.linspace(0, 1, 180)
    frames = rng.poisson(np.full((10, 180), 40.0)).astype(np.float32)
    path = tmp_path / "short.h5"
    write_frame_stack(path, frames, energy)

    out = tmp_path / "m.pt"
    run(monkeypatch, "train", "-d", str(path), "-o", str(out),
        "--method", "moving-average", "--epochs", "1")
    assert "Resampling 180 -> 256" in capsys.readouterr().out
    assert torch.load(out, map_location="cpu", weights_only=False)["num_features"] == 256


def test_window_clamps_to_the_frames_available(monkeypatch, tmp_path, stack_file, capsys):
    out = tmp_path / "m.pt"
    run(monkeypatch, "train", "-d", str(stack_file), "-o", str(out),
        "--method", "moving-average", "--window", "100", "--epochs", "1")
    assert "clamped to 23" in capsys.readouterr().out
    assert torch.load(out, map_location="cpu", weights_only=False)["window"] == 23
