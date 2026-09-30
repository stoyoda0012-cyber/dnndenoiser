"""``train --seed`` must seed every training path, as its help says.

An external audit found that only the moving-average path applied ``--seed``. The
noise2clean and noise2noise paths built the model and shuffled batches from torch's
unseeded stream, and noise2noise synthesized its targets from generators fixed at a
default seed, so ``--seed`` changed nothing on those paths while the help promised
that the same seed gives the same initial weights.

The equality tests here run in one process on CPU. They show the seed reaches the
model construction, the batch order and the synthesized targets; they do not show
that training is reproducible across processes, platforms or devices.
"""
from __future__ import annotations

import sys

import pytest
import torch

from dnndenoiser.cli import main
from dnndenoiser.training import methods


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["dnndenoiser", *argv])
    main()


@pytest.fixture
def dataset(monkeypatch, tmp_path):
    path = tmp_path / "data.h5"
    run_cli(monkeypatch, "generate", "-o", str(path), "-n", "16", "--n-energy", "64",
            "--peak-set", "C1s_single", "--poisson-level", "500", "--seed", "0")
    return path


def train(monkeypatch, dataset, out, method, *extra):
    argv = ["train", "-d", str(dataset), "-o", str(out), "--arch", "FCNN", "--epochs", "1",
            "--batch-size", "4", "--device", "cpu", "--method", method, *extra]
    if method == "noise2noise":
        argv += ["--noise-level", "500"]
    run_cli(monkeypatch, *argv)
    return torch.load(out, map_location="cpu", weights_only=False)["model_state_dict"]


def same_weights(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("method", ["noise2clean", "noise2noise"])
def test_the_same_seed_gives_the_same_trained_weights(monkeypatch, tmp_path, dataset, method):
    a = train(monkeypatch, dataset, tmp_path / "a.pt", method, "--seed", "7")
    b = train(monkeypatch, dataset, tmp_path / "b.pt", method, "--seed", "7")
    assert same_weights(a, b), f"--seed 7 twice gave different {method} weights"


@pytest.mark.parametrize("method", ["noise2clean", "noise2noise"])
def test_a_different_seed_gives_different_weights(monkeypatch, tmp_path, dataset, method):
    """The converse, so the test above cannot pass by the seed being ignored in a way
    that happens to reproduce -- e.g. a constant initialisation."""
    a = train(monkeypatch, dataset, tmp_path / "a.pt", method, "--seed", "7")
    b = train(monkeypatch, dataset, tmp_path / "b.pt", method, "--seed", "8")
    assert not same_weights(a, b)


def test_noise2noise_targets_are_seeded_by_seed(monkeypatch, tmp_path, dataset):
    """The synthesized targets come from NumPy generators inside Noise2Noise, which
    torch.manual_seed does not reach; --seed has to be handed to them."""
    seen = []
    original = methods.create_training_method

    def spy(method_type, noise_fn=None, **kwargs):
        seen.append(kwargs.get("seed"))
        return original(method_type, noise_fn, **kwargs)

    monkeypatch.setattr(methods, "create_training_method", spy)
    train(monkeypatch, dataset, tmp_path / "a.pt", "noise2noise", "--seed", "7")
    assert seen == [7]


def test_without_seed_the_noise2noise_default_is_unchanged(monkeypatch, tmp_path, dataset):
    seen = []
    original = methods.create_training_method

    def spy(method_type, noise_fn=None, **kwargs):
        seen.append(kwargs.get("seed"))
        return original(method_type, noise_fn, **kwargs)

    monkeypatch.setattr(methods, "create_training_method", spy)
    train(monkeypatch, dataset, tmp_path / "a.pt", "noise2noise")
    assert seen == [None]
