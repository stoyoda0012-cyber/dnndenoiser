"""Criteria C0, C2, C3, C6 and C7 of the P1 preregistration.

C1 lives in ``test_p1_moving_average_targets.py``; C4 and C5 in their own files.

``docs/preregistration/P1-selfsupervised-moving-average.md`` scopes C0, C2, C3
and C7 to one environment — CPU, Python 3.12, ``torch`` 2.9.1, ``numpy`` 2.3.3 —
and makes them *reported, not required* anywhere else, because reduction order
differs between backends and builds. These tests therefore **skip** outside that
environment rather than failing there: a skip says "not measured here", which is
the truth, where a failure would say "the port is wrong", which would not be.

The reference values are pinned by ``fixtures/generate_p1_reference_targets.py``
from the archived deposit.

**On why C2 and C3 agree exactly.** They do, to the bit. That is not evidence
that any independent reimplementation would: ``dnndenoiser``'s ResNet-FCNN and
the deposit's ``DenoisingNetwork`` share a code lineage — the deposit vendored
its copy *from this project* — so they consume the random stream identically.
C0 is what makes that shared property explicit and testable instead of a happy
accident, and it is why C0 gates C2 and C3.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dnndenoiser.models.network import DenoisingNetwork
from tests.p1_environment import IN_PINNED_ENVIRONMENT, pinned_environment
from dnndenoiser.training.selfsupervised import (
    denoise,
    moving_average_targets,
    resample,
    train_selfsupervised,
)

_FIXTURES = Path(__file__).parent / "fixtures"
PINNED = json.loads((_FIXTURES / "p1_reference_targets.json").read_text(encoding="utf-8"))
REFERENCE_OUTPUTS = dict(np.load(_FIXTURES / "p1_reference_outputs.npz"))
ENV = PINNED["_environment"]

NET_KW = dict(
    num_features=256, num_hidden_units=100, layer_type="ResNet-FCNN", encoder_output_dim=64
)


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_pinned_values_declare_the_reference_they_came_from():
    """The goldens certify themselves only through the script that writes them.

    ``generate_p1_reference_targets.py`` refuses to run against a copy whose
    digest is wrong — but it is regenerated in the same act as the file it
    certifies, so a regeneration against a modified reference, or against the
    port itself, would produce a green suite and a self-consistent fixture. The
    digest, DOI and version are restated here as literals taken from the
    preregistration, so the fixture has to agree with something it did not
    write.
    """
    provenance = PINNED["_provenance"]
    assert provenance["selfsupervised_sha256"] == (
        "136f2e112430fbb42e72bc6d6c0f2b9b02d3a828f6774c11bd2392eb4a13cb09"
    )
    assert provenance["deposit"] == "10.5281/zenodo.22092109"
    assert provenance["deposit_version"] == "1.0.0"


@pytest.fixture(scope="module")
def fixture():
    """The preregistration's fixture, rebuilt from its constants."""
    rng = np.random.default_rng(1)
    energy = np.linspace(0, 1, 256)
    clean = 300 * np.exp(-((energy - 0.5) ** 2) / (2 * 0.04**2)) + 20
    raw = rng.poisson(np.tile(clean, (200, 1))).astype(np.float32)
    g_min, g_max = raw.min(), raw.max()
    frames = (raw - g_min) / (g_max - g_min)
    test = (rng.poisson(np.tile(clean, (16, 1))).astype(np.float32) - g_min) / (g_max - g_min)
    clean_n = (clean - g_min) / (g_max - g_min)
    targets = moving_average_targets(frames, np.arange(200), 5)
    return frames, targets, test, clean_n


def snr_db(y: np.ndarray, clean_n: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(clean_n**2) / np.mean((y - clean_n) ** 2)))


# --------------------------------------------------------------------------- #
# C0 — RNG-stream alignment. The precondition C2 and C3 rest on.
# --------------------------------------------------------------------------- #
@pinned_environment
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_c0_rng_stream_alignment(seed):
    """Freshly constructed parameters are bit-identical to the reference's.

    C2 and C3 compare trained outputs pairwise by seed, and that pairing cancels
    variance only if both implementations consume the same random stream. One
    extra draw before construction — all it takes is building a DataLoader
    first — and C3 is differencing two effectively independent models.
    """
    torch.manual_seed(seed)
    params = np.concatenate(
        [v.detach().numpy().ravel() for v in DenoisingNetwork(**NET_KW).state_dict().values()]
    )
    assert digest(params) == PINNED["c0_initial_parameters"][str(seed)]


@pinned_environment
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_c0_alignment_holds_inside_the_training_function(fixture, seed):
    """The half of C0 that construction alone does not cover.

    The failure C0 exists to catch is a draw taken *before* the model is built,
    and the likeliest place for one is inside ``train_selfsupervised`` itself --
    building a DataLoader first would do it. Testing a bare ``DenoisingNetwork``
    cannot see that: inserting ``torch.randn(1)`` in the training function
    leaves the construction check passing at all six seeds while C2, C3 and C7
    fail, which is precisely the misattribution C0 is supposed to prevent.

    Training zero epochs returns the initial weights, so this reaches inside the
    function without training anything.
    """
    frames, targets, _, _ = fixture
    model = train_selfsupervised(frames, targets, epochs=0, seed=seed)
    params = np.concatenate(
        [v.detach().numpy().ravel() for v in model.state_dict().values()]
    )
    assert digest(params) == PINNED["c0_initial_parameters"][str(seed)]


# --------------------------------------------------------------------------- #
# C2 / C7 — trained outputs at a fixed seed
# --------------------------------------------------------------------------- #
@pinned_environment
@pytest.mark.parametrize(
    "epochs, key",
    [(20, "c2_output_seed0"), (30, "c7_output_epochs30_seed0")],
    ids=["C2-20-epochs", "C7-30-epochs-scheduler-fires"],
)
def test_c2_and_c7_trained_output(fixture, epochs, key):
    """The registered statistic: ``max|y_port - y_ref| / max|y_ref|`` < 1e-4.

    Evaluated against the reference's own pinned array rather than a digest of
    it, so the criterion under test is the one that was registered. A digest
    would only answer "identical or not", and the registered criterion is a
    tolerance -- reporting the number it failed at is half of what a failure is
    for.

    C7 is the same comparison at 30 epochs, where ``StepLR(step_size=25)`` fires
    once. The 20-epoch fixture never reaches it, so without C7 the scheduler is
    a named component of the method that nothing exercises.
    """
    torch.set_num_threads(1)
    frames, targets, test, _ = fixture
    reference = REFERENCE_OUTPUTS[key]

    got = denoise(train_selfsupervised(frames, targets, epochs=epochs, seed=0), test)
    assert got.shape == reference.shape

    relative = float(np.abs(got - reference).max() / np.abs(reference).max())
    assert relative < 1e-4, f"relative L-inf {relative:.3e} exceeds 1e-4"
    print(f"\n  {key}: relative L-inf = {relative:.3e}")


# --------------------------------------------------------------------------- #
# C3 — five seeds, paired
# --------------------------------------------------------------------------- #
@pinned_environment
def test_c3_paired_snr_across_five_seeds(fixture):
    """|SNR_port - SNR_ref| <= 0.5 dB on each of seeds 1-5, paired by seed.

    The bound is 3.5x tighter than the reference's own seed-to-seed spread
    (sd 1.442 dB), and is defensible only under C0.
    """
    torch.set_num_threads(1)
    frames, targets, test, clean_n = fixture
    deltas = []
    for seed in (1, 2, 3, 4, 5):
        model = train_selfsupervised(frames, targets, epochs=20, seed=seed)
        got = snr_db(denoise(model, test), clean_n)
        want = PINNED["c3_trained_output_seeds"][str(seed)]["snr_db"]
        deltas.append(got - want)
        assert abs(got - want) <= 0.5, (
            f"seed {seed}: port {got:.3f} dB vs reference {want:.3f} dB"
        )
    deltas = np.array(deltas)
    assert np.abs(deltas).max() <= 0.5
    # Reported, not asserted: the paired mean and spread, per AGENTS.md section 6.
    print(f"\n  C3 paired delta: {deltas.mean():+.4f} +/- {deltas.std(ddof=1):.4f} dB")


# --------------------------------------------------------------------------- #
# C6 — the resample path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_new", [128, 512])
def test_c6_resample_matches_the_reference(fixture, n_new):
    """Every stack whose length is not 256 goes through here.

    Checked to a tolerance on every build, and exactly on the pinned one.
    ``resample`` is a float32 matrix multiply and which order the BLAS sums in
    is the build's choice: locally, the same mathematics written three ways
    (``@``, ``einsum``, per-row) gives three results differing by one ULP. So
    an exact digest is a statement about this build, while agreement to a
    tolerance is a statement about the algorithm, and both are worth making.
    """
    _, _, test, _ = fixture
    got = resample(test, n_new)
    reference = REFERENCE_OUTPUTS[f"c6_resampled_{n_new}"]

    assert got.shape == reference.shape
    assert got.dtype == np.float32
    assert np.allclose(got, reference, rtol=0, atol=1e-6), (
        f"max |difference| = {np.abs(got - reference).max():.3e}"
    )

    if IN_PINNED_ENVIRONMENT:
        assert digest(got) == PINNED["c6_resampled"][str(n_new)]


def test_c6_resample_structure_holds_on_any_build(fixture):
    """The parts of ``resample`` that no BLAS gets a say in.

    Shape, dtype, and the clip at ``n_old - 2`` — which is what makes the final
    output point interpolate from the last interval instead of running off the
    end. A port that dropped the clip would produce the right shape and fail
    here rather than only on the pinned build.
    """
    from dnndenoiser.training.selfsupervised import _interpolation_matrix

    matrix = _interpolation_matrix(256, 128)
    assert matrix.dtype == np.float32
    assert matrix.shape == (128, 256)
    assert matrix[-1].nonzero()[0].tolist() == [255], (
        "the last output point must come from the final interval; without the "
        "clip at n_old-2 the source index runs past the end"
    )
    assert np.allclose(matrix.sum(axis=1), 1.0), "each output point is a convex combination"

    _, _, test, _ = fixture
    assert resample(test, 128).shape == (16, 128)


def test_c6_resample_is_a_no_op_without_copying(fixture):
    """The reference returns the input itself when no resampling is needed.

    Documented rather than silently improved: a caller sharing storage with the
    result is a real footgun, and changing it would be a behavioural difference
    in a port whose point is equivalence.
    """
    _, _, test, _ = fixture
    same_length = np.ascontiguousarray(test, dtype=np.float32)
    assert resample(same_length, 256) is same_length
