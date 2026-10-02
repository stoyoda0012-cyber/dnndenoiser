"""Frame stacks with angle channels (docs/design/FRAME_STACK_CHANNELS.md §8, groups 1-10).

Every rejection is pinned to its reason with every other field valid, and has a positive
counterpart. Independent expectations are brute-force means over the same channel, on
index sets with distinct pairwise distances (no ties), so they do not depend on how the
code under test breaks ties; the definition (each channel alone through
``moving_average_targets``) is checked separately, bit for bit.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from dnndenoiser import provenance as prov
from dnndenoiser.data.frame_stack import read_frame_stack, write_frame_stack
from dnndenoiser.training import selfsupervised as ss
from dnndenoiser.training.selfsupervised import channel_targets, moving_average_targets
from tests.test_evaluate_reference import TRUTH, evaluate, refuse, run, write
from tests.test_provenance_manifest import train

ANG = np.array([40.0, 10.0, 25.0])       # interleaved on purpose: never sorted


def frames3(n=6, a=3, e=256, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return (1.0 + rng.normal(0, 0.1, (n, a, e))).astype(dtype)


def stack3(path, n=6, a=3, e=256, seed=0, frame_index=None, order_basis=None, **kw):
    write_frame_stack(path, frames3(n, a, e, seed), np.linspace(0.0, 1.0, e),
                      frame_index=frame_index, angles=kw.pop("angles", ANG[:a]),
                      angle_kind=kw.pop("angle_kind", "emission"),
                      angle_units=kw.pop("angle_units", "deg"), order_basis=order_basis)
    return path


def raw3(path, n=6, a=3, e=16, **attrs):
    """A 3-D stack written by hand, for planting faults the writer would refuse."""
    with h5py.File(path, "w") as f:
        f.create_dataset("frames", data=frames3(n, a, e))
        f.create_dataset("energy", data=np.linspace(0, 1, e))
        f.create_dataset("frame_index", data=np.arange(n))
        if attrs.get("angles", "default") is not None:
            f.create_dataset("angles", data=attrs.get("angles", ANG[:a]))
            for key in ("angle_kind", "angle_units"):
                value = attrs.get(key, {"angle_kind": "emission", "angle_units": "deg"}[key])
                if value is not None:
                    f["angles"].attrs[key] = value
        if "times" in attrs:
            f.create_dataset("times", data=attrs["times"])
        if "order_basis" in attrs:
            f["frame_index"].attrs["order_basis"] = attrs["order_basis"]
        if "energy" in attrs:
            del f["energy"]
            f.create_dataset("energy", data=attrs["energy"])
    return path


# ---------------------------------------------------------------------------------- 1


@pytest.mark.parametrize("attrs, reason", [
    ({"angles": None}, "a 3-D 'frames' (6, 3, 16) needs an 'angles' dataset"),
    ({"angles": np.array([1.0, 2.0])}, "'angles' has 2 values but 'frames' has 3 channels"),
    ({"angles": np.array([True, False, True])}, "'angles' must be a one-dimensional integer or float array"),
    ({"angles": np.array([1.0, np.nan, 3.0])}, "'angles' contains non-finite values"),
    ({"angles": np.array([10.0, 10.0, 20.0])}, "'angles' repeats a value"),
    ({"angle_kind": None}, "'angles' needs the attribute 'angle_kind'"),
    ({"angle_kind": "polar"}, "attribute 'angle_kind' must be one of emission, analyser"),
    ({"angle_kind": "other:  "}, "attribute 'angle_kind' must be one of"),
    ({"angle_kind": "other:" + "x" * 201}, "attribute 'angle_kind' must be one of"),
    ({"angle_units": None}, "'angles' needs the attribute 'angle_units'"),
    ({"angle_units": "rad"}, "attribute 'angle_units' must be deg"),
    ({"times": np.arange(6.0)}, "a 3-D frame stack may not carry 'times'"),
    ({"order_basis": "guessed"}, "attribute 'order_basis' must be one of recorded, inferred, unknown"),
    ({"energy": np.linspace(0, 1, 3)}, "'energy' has 3 points but 'frames' has 16 per frame"),
])
def test_the_reader_refuses_a_malformed_channel_stack(tmp_path, attrs, reason):
    path = raw3(tmp_path / "s.h5", **attrs)
    with pytest.raises(ValueError, match=reason.replace("(", r"\(").replace(")", r"\)")):
        read_frame_stack(path)


def test_a_valid_channel_stack_reads_in_file_order(tmp_path):
    stack = read_frame_stack(raw3(tmp_path / "s.h5"))
    np.testing.assert_array_equal(stack.angles, ANG)                 # interleaved, unsorted
    assert (stack.angle_kind, stack.angle_units, stack.order_basis) == ("emission", "deg", "unknown")
    assert (stack.n_frames, stack.n_angles, stack.n_energy) == (6, 3, 16)
    desc = read_frame_stack(raw3(tmp_path / "d.h5", angles=np.array([30.0, 20.0, 10.0])))
    np.testing.assert_array_equal(desc.angles, [30.0, 20.0, 10.0])
    assert read_frame_stack(raw3(tmp_path / "b.h5", angle_kind=b"analyser",
                                 order_basis=b"recorded")).order_basis == "recorded"


def test_a_two_dimensional_stack_is_read_as_before(tmp_path):
    """Owner's decision: 2-D stacks keep accepting angles and times, unread."""
    path = tmp_path / "s.h5"
    write_frame_stack(path, frames3(n=5, a=1, e=16)[:, 0], np.linspace(0, 1, 16))
    with h5py.File(path, "a") as f:
        f.create_dataset("angles", data=np.array([1.0, 1.0]))       # repeated: not read
        f.create_dataset("times", data=np.arange(5.0))
    stack = read_frame_stack(path)
    assert stack.angles is None and stack.n_angles == 1 and stack.order_basis == "unknown"


def test_the_writer_writes_no_order_basis_unless_given_and_checks_the_channel_axis(tmp_path):
    with h5py.File(stack3(tmp_path / "a.h5"), "r") as f:
        assert "order_basis" not in f["frame_index"].attrs
    with h5py.File(stack3(tmp_path / "b.h5", order_basis="inferred"), "r") as f:
        assert f["frame_index"].attrs["order_basis"] == "inferred"
    with pytest.raises(ValueError, match="'angles' repeats a value"):
        stack3(tmp_path / "c.h5", angles=np.array([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="names the channels of a 3-D 'frames'; this one is 2-D"):
        write_frame_stack(tmp_path / "d.h5", frames3(a=1)[:, 0], np.linspace(0, 1, 256),
                          angles=np.array([1.0]))


# ---------------------------------------------------------------------------------- 2

TIE_FREE = np.array([0, 1, 3, 7, 15, 31, 63])          # distinct pairwise distances
SHUFFLED = np.array([31, 0, 63, 3, 15, 1, 7])


def brute_force(frames, index, W):
    """For each frame, the mean of its W nearest others in the same channel (no ties)."""
    n = len(index)
    out = np.empty(frames.shape, dtype=np.float64)
    for t in range(n):
        others = sorted((abs(int(index[s]) - int(index[t])), s) for s in range(n) if s != t)
        chosen = [s for _d, s in others[:W]]
        out[t] = frames[chosen].astype(np.float64).mean(axis=0)
    return out


@pytest.mark.parametrize("index", [TIE_FREE, SHUFFLED], ids=["sorted", "unsorted"])
def test_targets_are_brute_force_means_inside_the_channel(index):
    f = frames3(n=len(index), a=3, e=8, seed=2)
    for W in range(1, len(index)):
        np.testing.assert_allclose(channel_targets(f, index, W), brute_force(f, index, W),
                                   rtol=0, atol=1e-12)


def test_each_channel_is_moving_average_targets_alone_bit_for_bit():
    """The definition, on float32 frames and an index set with ties."""
    f = frames3(n=9, a=3, e=32, seed=3)
    index = np.arange(9) * 2                                       # evenly spaced: ties
    for W in range(1, 9):
        got = channel_targets(f, index, W)
        for c in range(3):
            np.testing.assert_array_equal(got[:, c], moving_average_targets(f[:, c], index, W))


def test_one_channel_never_changes_another_channels_targets():
    f = frames3(n=7, a=3, e=8, seed=4)
    g = f.copy()
    g[:, 1] += 5.0
    for W in (1, 3, 6):
        a, b = channel_targets(f, TIE_FREE, W), channel_targets(g, TIE_FREE, W)
        np.testing.assert_array_equal(a[:, 0], b[:, 0])
        np.testing.assert_array_equal(a[:, 2], b[:, 2])
        assert not np.array_equal(a[:, 1], b[:, 1])


def pooled(frames, index, W):
    """A planted wrong implementation: neighbours' means pooled over every channel."""
    t = channel_targets(frames, index, W)
    return np.repeat(t.mean(axis=1, keepdims=True), frames.shape[1], axis=1)


def test_the_brute_force_check_rejects_pooled_channels():
    f = frames3(n=7, a=3, e=8, seed=5)
    assert not np.allclose(pooled(f, TIE_FREE, 2), brute_force(f, TIE_FREE, 2), atol=1e-12)


# ---------------------------------------------------------------------------------- 3


def workaround(frames, index, stride):
    n, a, e = frames.shape
    stacked = np.concatenate([frames[:, c] for c in range(a)])
    indices = np.concatenate([c * stride + index for c in range(a)])
    return stacked, indices


def test_the_channel_axis_equals_the_workaround_where_nothing_ties():
    f = frames3(n=len(TIE_FREE), a=3, e=8, seed=6)
    stride = 2 * int(TIE_FREE.max() - TIE_FREE.min()) + 2          # channels cannot interleave
    stacked, indices = workaround(f, TIE_FREE, stride)
    for W in range(1, len(TIE_FREE)):
        via_workaround = moving_average_targets(stacked, indices, W)
        via_channels = channel_targets(f, TIE_FREE, W)
        for c in range(3):
            np.testing.assert_allclose(via_workaround[c * len(TIE_FREE):(c + 1) * len(TIE_FREE)],
                                       via_channels[:, c], rtol=0, atol=1e-12)
    assert (stacked.min(), stacked.max()) == (f.min(), f.max())


def test_on_ties_each_resolution_is_a_valid_one():
    """Evenly spaced indices, W = 1: an inner frame's two nearest others tie. The channel
    axis and the workaround may resolve the tie differently; each must pick one of them."""
    n = 8
    f = frames3(n=n, a=2, e=4, seed=7).astype(np.float64)
    index = np.arange(n)
    stacked, indices = workaround(f, index, 3 * n)
    for got in (channel_targets(f, index, 1),
                np.stack([moving_average_targets(stacked, indices, 1)[c * n:(c + 1) * n]
                          for c in range(2)], axis=1)):
        for t in range(1, n - 1):
            assert any(np.array_equal(got[t], f[s]) for s in (t - 1, t + 1))


# ---------------------------------------------------------------- 4 to 7: through the CLI


def captured_training(monkeypatch, tmp_path, data, *extra):
    seen = {}
    real = ss.train_selfsupervised

    def spy(frames, targets, **kwargs):
        seen["frames"], seen["targets"] = np.array(frames), np.array(targets)
        return real(frames, targets, **kwargs)
    monkeypatch.setattr(ss, "train_selfsupervised", spy)
    ck = train(monkeypatch, data, tmp_path / "m.pt", *extra, method="moving-average")
    monkeypatch.setattr(ss, "train_selfsupervised", real)
    return ck, seen


def test_normalisation_is_over_every_channel(monkeypatch, tmp_path):
    data = stack3(tmp_path / "s.h5", e=256)
    ck, _seen = captured_training(monkeypatch, tmp_path, data)
    with h5py.File(data) as f:
        raw = f["frames"][:].astype(np.float32)
    assert ck["normalisation"]["min"] == float(raw.min())
    assert ck["normalisation"]["max"] == float(raw.max())


def test_rows_are_frame_major_and_inputs_match_their_targets(monkeypatch, tmp_path):
    data = stack3(tmp_path / "s.h5", n=6, a=3, e=256, frame_index=TIE_FREE[:6])
    ck, seen = captured_training(monkeypatch, tmp_path, data, "--window", "2")
    with h5py.File(data) as f:
        raw = f["frames"][:].astype(np.float32)
    lo, hi = ck["normalisation"]["min"], ck["normalisation"]["max"]
    norm = (raw - lo) / (hi - lo)
    expected_targets = brute_force(norm, TIE_FREE[:6], 2)
    for r in range(18):
        t, c = divmod(r, 3)
        np.testing.assert_array_equal(seen["frames"][r], norm[t, c])
        np.testing.assert_allclose(seen["targets"][r], expected_targets[t, c], rtol=0, atol=1e-6)


def pre_change_pipeline(frames2d, index, W, seed):
    """The 2-D moving-average pipeline as it was: resample, normalise, targets, train."""
    from dnndenoiser.training.selfsupervised import TARGET_LENGTH, resample
    x = np.asarray(frames2d, dtype=np.float32)
    if x.shape[-1] != TARGET_LENGTH:
        x = resample(x, TARGET_LENGTH)
    lo, hi = float(x.min()), float(x.max())
    norm = (x - lo) / (hi - lo)
    return ss.train_selfsupervised(norm, moving_average_targets(norm, index, W), epochs=1,
                                   batch_size=8, device="cpu", seed=seed)


@pytest.mark.parametrize("e", [256, 300])
def test_two_d_and_one_channel_train_to_the_pre_change_weights(monkeypatch, tmp_path, e):
    f = frames3(n=10, a=1, e=e, seed=8)
    flat = tmp_path / "flat.h5"
    write_frame_stack(flat, f[:, 0], np.linspace(0, 1, e))
    one = stack3(tmp_path / "one.h5", n=10, a=1, e=e, seed=8, angles=np.array([30.0]))
    expected = pre_change_pipeline(f[:, 0], np.arange(10), 5, seed=0).state_dict()
    for data in (flat, one):
        ck = train(monkeypatch, data, tmp_path / f"{data.stem}.pt", "--seed", "0",
                   method="moving-average")
        for key, value in expected.items():
            assert torch.equal(ck["model_state_dict"][key], value), (data.stem, key)


@pytest.mark.parametrize("e, expected", [(256, None), (300, {"from_points": 300, "to_points": 256})])
def test_resampling_is_recorded_from_the_energy_axis(monkeypatch, tmp_path, e, expected):
    ck = train(monkeypatch, stack3(tmp_path / "s.h5", e=e), tmp_path / "m.pt",
               method="moving-average")
    assert ck["provenance"]["preprocessing"]["resampling"] == expected


def test_the_window_is_clamped_to_the_frames_of_a_channel_stack(monkeypatch, tmp_path):
    ck = train(monkeypatch, stack3(tmp_path / "s.h5", n=4), tmp_path / "m.pt", "--window", "50",
               method="moving-average")
    assert ck["provenance"]["command"]["effective"]["window"] == 3


@pytest.mark.parametrize("basis, warned", [(None, True), ("inferred", True), ("unknown", True),
                                           ("recorded", False)])
def test_the_order_warning_is_for_moving_average_only(monkeypatch, capsys, tmp_path, basis, warned):
    capsys.readouterr()
    train(monkeypatch, stack3(tmp_path / "s.h5", order_basis=basis), tmp_path / "m.pt",
          method="moving-average")
    assert ("Warning: the acquisition order is" in capsys.readouterr().err) is warned
    flat = tmp_path / "flat.h5"
    write_frame_stack(flat, frames3(a=1)[:, 0], np.linspace(0, 1, 256), order_basis=basis)
    train(monkeypatch, flat, tmp_path / "m2.pt", method="moving-average")
    assert ("Warning: the acquisition order is" in capsys.readouterr().err) is warned
    n, c = frames3(n=8, a=1, e=32)[:, 0], frames3(n=8, a=1, e=32, seed=9)[:, 0]
    sup = write(tmp_path / "sup.h5", n, None, c, declaration=TRUTH, frame_index=np.arange(8))
    train(monkeypatch, sup, tmp_path / "m3.pt")
    assert "Warning: the acquisition order is" not in capsys.readouterr().err


# ---------------------------------------------------------------------------------- 8


@pytest.fixture(scope="module")
def ma_model(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("ma")
    try:
        train(mp, stack3(d / "s.h5"), d / "m.pt", method="moving-average")
    finally:
        mp.undo()
    return d / "m.pt"


def infer(monkeypatch, data, model, out):
    run(monkeypatch, "infer", "-d", str(data), "-m", str(model), "-o", str(out), "--device", "cpu")
    return out


@pytest.mark.parametrize("attrs, reason", [
    ({"angles": None}, "needs an 'angles' dataset"),
    ({"angles": np.array([1.0, 1.0, 2.0])}, "'angles' repeats a value"),
    ({"times": np.arange(6.0)}, "a 3-D frame stack may not carry 'times'"),
    ({"angle_kind": "polar"}, "attribute 'angle_kind' must be one of"),
])
def test_infer_refuses_a_malformed_channel_axis(monkeypatch, capsys, tmp_path, ma_model, attrs, reason):
    path = raw3(tmp_path / "s.h5", e=256, **attrs)
    assert reason in refuse(monkeypatch, capsys, "infer", "-d", str(path), "-m", str(ma_model),
                            "-o", str(tmp_path / "o.h5"), "--device", "cpu")


def test_infer_keeps_accepting_what_it_accepted(monkeypatch, tmp_path, ma_model):
    flat = tmp_path / "flat.h5"
    with h5py.File(flat, "w") as f:                    # 2-D, times and angles, no frame_index
        f.create_dataset("frames", data=frames3(n=1, a=1)[:, 0])  # one frame
        f.create_dataset("energy", data=np.linspace(0, 1, 256))
        f.create_dataset("times", data=np.arange(1.0))
        f.create_dataset("angles", data=np.array([3.0]))
    with h5py.File(infer(monkeypatch, flat, ma_model, tmp_path / "o.h5")) as o:
        assert o["denoised"].shape == (1, 256)


@pytest.mark.parametrize("e", [256, 300])
def test_infer_carries_the_declarations(monkeypatch, tmp_path, ma_model, e):
    data = stack3(tmp_path / "s.h5", e=e, order_basis="recorded")
    with h5py.File(data, "a") as f:
        f["angles"].attrs["angle_kind"] = np.bytes_(b"analyser")   # bytes are read too
    with h5py.File(infer(monkeypatch, data, ma_model, tmp_path / "o.h5")) as o:
        assert o["denoised"].shape == (6, 3, 256)
        np.testing.assert_array_equal(o["angles"][:], ANG)
        assert o["angles"].attrs["angle_kind"] == "analyser"
        assert o["angles"].attrs["angle_units"] == "deg"
        assert o["frame_index"].attrs["order_basis"] == "recorded"


def test_evaluate_compares_the_angles_of_a_channel_stack(monkeypatch, capsys, tmp_path, ma_model):
    out = infer(monkeypatch, stack3(tmp_path / "s.h5"), ma_model, tmp_path / "o.h5")
    with h5py.File(out) as o:
        noisy, energy = o["noisy"][:], o["energy"][:]
    mean = np.repeat(noisy.mean(axis=0, keepdims=True), noisy.shape[0], axis=0)
    from tests.test_evaluate_reference import estimate
    for name, angles in (("ref.h5", ANG), ("rev.h5", ANG[::-1])):
        r = write(tmp_path / name, mean, None, mean, declaration=estimate(acquisition="other"),
                  energy=energy)
        with h5py.File(r, "a") as f:
            f.create_dataset("angles", data=angles)
    with h5py.File(out, "a") as f:
        for k in ("noisy", "denoised"):
            f[k].attrs["intensity_units"] = "counts"
    m = evaluate(monkeypatch, tmp_path, out, "--clean", str(tmp_path / "ref.h5"), "--assert-alignment",
                 "rows")
    assert "angles" in m["evaluation_context"]["alignment_verified"]
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(out), "--clean", str(tmp_path / "rev.h5"),
                 "--assert-alignment", "rows")
    assert "the reference's axis is reversed" in err


# ---------------------------------------------------------------------------------- 9


def test_the_manifest_records_the_declarations_for_every_method(monkeypatch, tmp_path):
    ma = train(monkeypatch, stack3(tmp_path / "s.h5", order_basis="inferred"), tmp_path / "ma.pt",
               method="moving-average")["provenance"]
    assert ma["schema"] == "dnd-provenance-2"
    assert ma["training_data"]["angles"] == {"kind": "emission", "units": "deg"}
    assert ma["training_data"]["frame_index_basis"] == "inferred"
    run(monkeypatch, "generate", "-o", str(tmp_path / "g.h5"), "-n", "4", "--n-energy", "32",
        "--peak-set", "C1s_single", "--n-angles", "3")
    sup = train(monkeypatch, tmp_path / "g.h5", tmp_path / "sup.pt")["provenance"]
    assert sup["training_data"]["angles"] == {"kind": None, "units": None}
    assert sup["training_data"]["frame_index_basis"] == "unknown"
    flat_file = write(tmp_path / "f.h5", frames3(n=8, a=1, e=32)[:, 0], None,
                      frames3(n=8, a=1, e=32, seed=1)[:, 0], declaration=TRUTH)
    flat = train(monkeypatch, flat_file, tmp_path / "flat.pt")["provenance"]
    assert flat["training_data"]["angles"] is None


@pytest.mark.parametrize("mutate, reason", [
    (lambda td: td.update(angles={"kind": "emission", "units": "deg"}) or td["layout"].pop("angles", None),
     "'training_data.angles' must be null when the training file has no angles dataset"),
    (lambda td: td.update(angles=None), "'training_data.angles' must be an object when the training file has an angles dataset"),
    (lambda td: td.update(frame_index_basis="guessed"), "'training_data.frame_index_basis' must be one of"),
    (lambda td: td["angles"].update(units="rad"), "'training_data.angles.units'"),
    (lambda td: td["angles"].update(kind="polar"), "'training_data.angles.kind'"),
    (lambda td: td["angles"].update(extra=1), "'training_data.angles' fields do not match"),
])
def test_the_value_rules_of_the_new_fields(monkeypatch, tmp_path, mutate, reason):
    m = train(monkeypatch, stack3(tmp_path / "s.h5"), tmp_path / "m.pt",
              method="moving-average")["provenance"]
    assert prov.validate_manifest(json.loads(json.dumps(m)))
    bad = json.loads(json.dumps(m))
    mutate(bad["training_data"])
    with pytest.raises(prov.MalformedProvenance, match=reason.replace("(", r"\(")):
        prov.validate_manifest(bad)


@pytest.mark.parametrize("method", ["noise2clean", "moving-average"])
def test_out_of_vocabulary_declarations_are_refused_at_train(monkeypatch, capsys, tmp_path, method):
    if method == "moving-average":
        data = stack3(tmp_path / "s.h5")
        with h5py.File(data, "a") as f:
            f["frame_index"].attrs["order_basis"] = "guessed"
    else:
        data = write(tmp_path / "s.h5", frames3(n=8, a=1, e=32)[:, 0], None,
                     frames3(n=8, a=1, e=32, seed=1)[:, 0], declaration=TRUTH)
        with h5py.File(data, "a") as f:
            f.create_dataset("angles", data=np.arange(3.0))
            f["angles"].attrs["angle_kind"] = "polar"
    argv = ["train", "-d", str(data), "-o", str(tmp_path / "m.pt"), "--method", method,
            "--epochs", "1", "--device", "cpu"]
    err = refuse(monkeypatch, capsys, *argv)
    assert ("attribute 'order_basis' must be one of" if method == "moving-average"
            else "attribute 'angle_kind' must be one of") in err


def as_version_1(manifest):
    m = json.loads(json.dumps(manifest))
    m["schema"] = "dnd-provenance-1"
    del m["training_data"]["angles"], m["training_data"]["frame_index_basis"]
    return m


def test_exact_key_sets_per_version(monkeypatch, tmp_path):
    m = train(monkeypatch, stack3(tmp_path / "s.h5"), tmp_path / "m.pt",
              method="moving-average")["provenance"]
    v1 = as_version_1(m)
    assert prov.validate_manifest(v1)
    with_fields = json.loads(json.dumps(v1))
    with_fields["training_data"]["angles"] = None
    with pytest.raises(prov.MalformedProvenance, match="unknown \\['angles'\\]"):
        prov.validate_manifest(with_fields)
    without = json.loads(json.dumps(m))
    del without["training_data"]["frame_index_basis"]
    with pytest.raises(prov.MalformedProvenance, match="missing \\['frame_index_basis'\\]"):
        prov.validate_manifest(without)


def test_a_version_1_checkpoint_still_verifies_and_is_carried_byte_for_byte(monkeypatch, tmp_path):
    run(monkeypatch, "generate", "-o", str(tmp_path / "g.h5"), "-n", "4", "--n-energy", "32",
        "--peak-set", "C1s_single", "--n-angles", "3")
    ck = train(monkeypatch, tmp_path / "g.h5", tmp_path / "m.pt")
    from dnndenoiser.cli import checkpoint_model_config
    v1 = as_version_1(ck["provenance"])
    prov.seal(ck, v1, checkpoint_model_config(ck, "x"))
    old = tmp_path / "v1.pt"
    torch.save(ck, old)
    out = infer(monkeypatch, tmp_path / "g.h5", old, tmp_path / "o.h5")
    with h5py.File(out) as o:
        assert o["model_provenance"][()].decode("utf-8") == prov.canonical(v1)
    model = evaluate(monkeypatch, tmp_path, out)["evaluation_context"]["model"]
    assert model["training_data"]["declared_angles"] == "not recorded"         # not "no angles"
    assert model["training_data"]["declared_frame_index_basis"] == "not recorded"


def test_evaluate_reports_the_declarations_as_declared(monkeypatch, tmp_path, ma_model):
    out = infer(monkeypatch, stack3(tmp_path / "s.h5"), ma_model, tmp_path / "o.h5")
    with h5py.File(out, "a") as f:
        f.create_dataset("clean", data=f["noisy"][:])
        from dnndenoiser import reference as ref
        from tests.test_evaluate_reference import estimate
        ref.write_declaration(f["clean"], estimate(acquisition="other"))
        for k in ("noisy", "denoised"):
            f[k].attrs["intensity_units"] = "counts"
    td = evaluate(monkeypatch, tmp_path, out)["evaluation_context"]["model"]["training_data"]
    assert td["declared_angles"] == {"kind": "emission", "units": "deg"}
    assert td["declared_frame_index_basis"] == "unknown"


# --------------------------------------------------------------------------------- 10


def test_held_out_counts_frames_not_channels(monkeypatch, tmp_path):
    """Trained on one channel, evaluated on all: every frame counts (conservative)."""
    full = stack3(tmp_path / "full.h5")
    with h5py.File(full, "a") as f:
        f["frames"].attrs["acquisition_id"] = "acq-X"
        one = f["frames"][:, 0, :]
    single = tmp_path / "one.h5"
    write_frame_stack(single, one, np.linspace(0, 1, 256))
    with h5py.File(single, "a") as f:
        f["frames"].attrs["acquisition_id"] = "acq-X"
    train(monkeypatch, single, tmp_path / "m.pt", method="moving-average")
    out = infer(monkeypatch, full, tmp_path / "m.pt", tmp_path / "o.h5")
    with h5py.File(out, "a") as f:
        f.create_dataset("clean", data=f["noisy"][:])
        from dnndenoiser import reference as ref
        from tests.test_evaluate_reference import estimate
        ref.write_declaration(f["clean"], estimate(acquisition="other"))
        for k in ("noisy", "denoised"):
            f[k].attrs["intensity_units"] = "counts"
    c = evaluate(monkeypatch, tmp_path, out)["evaluation_context"]
    assert (c["held_out_status"], c["rows_in_training"]) == ("not_held_out", 6)


def test_workaround_indices_with_the_acquisition_id_give_a_documented_false_disjoint(
        monkeypatch, tmp_path):
    """QUICK_START: workaround indices are not acquisition order and must not carry the
    acquisition's acquisition_id. This pins what happens when they do."""
    f = frames3()
    stride = 6 + 5
    stacked, indices = workaround(f[:, 1:], np.arange(6), stride)
    indices = indices + stride                                      # channels 1 and 2
    wa = tmp_path / "wa.h5"
    write_frame_stack(wa, stacked, np.linspace(0, 1, 256), frame_index=indices)
    with h5py.File(wa, "a") as h:
        h["frames"].attrs["acquisition_id"] = "acq-X"
    train(monkeypatch, wa, tmp_path / "m.pt", method="moving-average")
    full = stack3(tmp_path / "full.h5")
    with h5py.File(full, "a") as h:
        h["frames"].attrs["acquisition_id"] = "acq-X"
    out = infer(monkeypatch, full, tmp_path / "m.pt", tmp_path / "o.h5")
    with h5py.File(out, "a") as h:
        h.create_dataset("clean", data=h["noisy"][:])
        from dnndenoiser import reference as ref
        from tests.test_evaluate_reference import estimate
        ref.write_declaration(h["clean"], estimate(acquisition="other"))
        for k in ("noisy", "denoised"):
            h[k].attrs["intensity_units"] = "counts"
    assert evaluate(monkeypatch, tmp_path, out)["evaluation_context"]["held_out_status"] == \
        "disjoint_by_identifiers"
