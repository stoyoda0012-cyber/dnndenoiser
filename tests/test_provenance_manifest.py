"""The provenance manifest, phase A (docs/design/PROVENANCE_MANIFEST.md §8, groups 1-5).

Every rejection is pinned to its reason with every other field valid, and has a positive
counterpart. Expected digests come from an independent transcription of ``dnd-digest-1``
written here (``_digest``, ``_array``), never from ``dnndenoiser.digest`` or
``dnndenoiser.provenance``; two vectors are frozen as literals.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from dnndenoiser import provenance as prov
from dnndenoiser.cli import build_parser
from dnndenoiser.data.frame_stack import write_frame_stack
from tests.test_evaluate_alignment import _array, _digest, _json
from tests.test_evaluate_reference import TRUTH, arrays, evaluate, refuse, run, write

REPO = Path(__file__).resolve().parents[1]


def strict(text):
    def no_constant(name):
        raise AssertionError(f"non-strict JSON constant {name}")
    return json.loads(text, parse_constant=no_constant)


def matching(a):
    """Independent: one component named 'array', the float32 cast in its stored shape."""
    return {"format": "dnd-digest-1",
            "sha256": _digest([("array", _array(np.asarray(a).astype(np.float32)))])}


def supervised_file(path, n=8, e=32, seed=0, **kwargs):
    noisy, _d, clean = arrays(n=n, e=e, seed=seed)
    return write(path, noisy, None, clean, declaration=TRUTH, **kwargs)


def stack_file(path, n=12, e=64, seed=0, dtype=np.float32, frame_index=None):
    rng = np.random.default_rng(seed)
    frames = (1.0 + rng.normal(0, 0.1, (n, e))).astype(dtype)
    write_frame_stack(path, frames, np.linspace(0.0, 1.0, e), frame_index=frame_index)
    return path


def train(monkeypatch, data, out, *extra, method="noise2clean"):
    args = ["train", "-d", str(data), "-o", str(out), "--method", method, "--epochs", "1",
            "--batch-size", "8", "--device", "cpu", *extra]
    if method == "noise2noise" and "--noise-level" not in extra:
        args += ["--noise-level", "100"]
    run(monkeypatch, *args)
    return torch.load(out, map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One checkpoint per method, trained once for the module."""
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("trained")
    out = {}
    try:
        out["noise2clean"] = (supervised_file(d / "sup.h5", acquisition_id="acq-T",
                                              frame_index=np.arange(8)), d / "n2c.pt")
        train(mp, *out["noise2clean"], "--seed", "3")
        out["noise2noise"] = (out["noise2clean"][0], d / "n2n.pt")
        train(mp, *out["noise2noise"], method="noise2noise")
        out["moving-average"] = (stack_file(d / "stack.h5"), d / "ma.pt")
        train(mp, *out["moving-average"], method="moving-average")
    finally:
        mp.undo()
    return out


def manifest_of(path):
    return torch.load(path, map_location="cpu", weights_only=True)["provenance"]


# ---------------------------------------------------------------------------------- 1


@pytest.mark.parametrize("method", ["noise2clean", "noise2noise", "moving-average"])
def test_the_manifest_has_every_field_and_reads_back_restricted(trained, method):
    data, model = trained[method]
    loaded = torch.load(model, map_location="cpu", weights_only=True)
    m = loaded["provenance"]
    assert set(m) == {"schema", "created_utc", "software", "code", "command", "training_data",
                      "targets", "preprocessing", "result", "statuses"}
    assert m["schema"] == "dnd-provenance-1" and m["command"]["method"] == method
    assert prov.validate_manifest(m) is m
    assert m["result"]["epochs"] == 1 and isinstance(m["result"]["final_loss"], float)
    assert set(loaded["model_digest"]) == {"format", "sha256"}
    strict(prov.canonical(m))


@pytest.mark.parametrize("method, input_name", [("noise2clean", "noisy"),
                                                ("moving-average", "frames")])
def test_the_training_digests_equal_an_independent_computation(trained, method, input_name):
    data, model = trained[method]
    td = manifest_of(model)["training_data"]
    with h5py.File(data) as f:
        stored = {k: f[k][:] for k in prov.TRAINING_COMPONENTS if k in f}
    expect = _digest([(k, _array(stored[k]) if k in stored else None)
                      for k in ("noisy", "frames", "clean", "energy", "angles", "times",
                                "frame_index")])
    assert td["digest"] == {"format": "dnd-digest-1", "sha256": expect}
    assert td["array_digests"][input_name] == matching(stored[input_name])
    assert td["layout"][input_name] == {"shape": list(stored[input_name].shape),
                                        "dtype": stored[input_name].dtype.name}
    assert td["rows_used"] == "all"


def test_a_float64_stack_has_the_float32_matching_digest(monkeypatch, tmp_path):
    data = stack_file(tmp_path / "s64.h5", dtype=np.float64)
    m = train(monkeypatch, data, tmp_path / "m.pt", method="moving-average")["provenance"]
    with h5py.File(data) as f:
        frames = f["frames"][:]
    assert frames.dtype == np.float64
    assert m["training_data"]["array_digests"]["frames"] == matching(frames)
    assert m["training_data"]["layout"]["frames"]["dtype"] == "float64"


def test_a_subclass_of_a_json_type_is_refused(trained):
    m = json.loads(prov.canonical(manifest_of(trained["noise2clean"][1])))
    m["software"]["torch"] = torch.__version__                 # a str subclass
    assert type(m["software"]["torch"]) is not str
    with pytest.raises(prov.MalformedProvenance, match="'software.torch' holds a TorchVersion"):
        prov.validate_manifest(m)
    m["software"]["torch"] = str(torch.__version__)
    assert prov.validate_manifest(m) is m


def test_training_records_identifiers_and_units(trained):
    td = manifest_of(trained["noise2clean"][1])["training_data"]
    assert td["acquisition_id"] == "acq-T" and td["intensity_units"] == "counts"
    assert td["frame_index_runs"] == [[0, 7]]
    assert td["reference_declaration"] == TRUTH and td["signal_identity"] is None


@pytest.mark.parametrize("values, runs", [
    ([0, 1, 2, 3], [[0, 3]]),
    ([0, 1, 5, 6, 9], [[0, 1], [5, 6], [9, 9]]),
    ([9, 5, 0, 6, 1], [[0, 1], [5, 6], [9, 9]]),
    ([2, 2, 3, 3], [[2, 3]]),
    ([7], [[7, 7]]),
], ids=["contiguous", "gapped", "unsorted", "duplicated", "single"])
def test_frame_index_runs_encode_the_set(values, runs):
    assert prov.frame_index_runs(np.array(values)) == runs


def test_seeds_are_recorded_as_used(trained):
    assert manifest_of(trained["noise2clean"][1])["command"]["seeds"] == {"torch": 3,
                                                                         "targets": None}
    # noise2noise without --seed still seeds its target generators: 42 and 1042.
    assert manifest_of(trained["noise2noise"][1])["command"]["seeds"] == {"torch": None,
                                                                         "targets": [42, 1042]}


def test_noise2noise_target_seeds_follow_the_seed(monkeypatch, tmp_path, trained):
    m = train(monkeypatch, trained["noise2noise"][0], tmp_path / "m.pt", "--seed", "5",
              method="noise2noise")["provenance"]
    assert m["command"]["seeds"]["targets"] == [5, 1005]
    assert m["targets"] == {"kind": "synthesised_realisation", "noise_level": 100.0}


def check_moving_average_effective(m):
    eff = m["command"]["effective"]
    assert eff["architecture"] == "ResNet-FCNN" and eff["num_hidden_units"] == 100
    assert eff["optimiser"] == {"name": "Adam", "lr": 1e-3, "weight_decay": 1e-9}
    assert eff["scheduler"] == {"name": "StepLR", "step_size": 25, "gamma": 0.5}
    assert eff["loss"] == {"name": "HuberLoss", "delta": 1.0} and eff["grad_clip"] == 4.0


def test_moving_average_records_the_settings_it_used_not_the_parser_defaults(
        monkeypatch, tmp_path):
    data = stack_file(tmp_path / "s.h5", n=4)
    m = train(monkeypatch, data, tmp_path / "m.pt", "--window", "5",
              method="moving-average")["provenance"]
    check_moving_average_effective(m)
    assert m["command"]["effective"]["window"] == 3            # clamped to n_frames - 1
    assert m["targets"] == {"kind": "leave_one_out_window_mean", "window": 3}
    assert m["command"]["arguments"]["arch"] == "FCNN"         # the parser value, kept apart
    assert m["preprocessing"]["resampling"] == {"from_points": 64, "to_points": 256}
    planted = json.loads(json.dumps(m))
    planted["command"]["effective"].update(architecture="FCNN",
                                           optimiser={"name": "Adam", "lr": 0.01,
                                                      "weight_decay": 0.0})
    with pytest.raises(AssertionError):
        check_moving_average_effective(planted)


def test_arguments_record_values_and_flags_and_exclude_paths(monkeypatch, tmp_path, trained):
    data = trained["noise2clean"][0]
    out = tmp_path / "m.pt"
    m = train(monkeypatch, data, out, "--lr=0.05")["provenance"]
    args = m["command"]["arguments"]
    assert args["lr"] == 0.05 and args["batch_size"] == 8 and args["hidden_units"] == 100
    assert "data" not in args and "output" not in args
    assert "--lr" in m["command"]["flags_passed"] and "--data" not in m["command"]["flags_passed"]
    assert m["command"]["effective"]["optimiser"] == {"name": "Adam", "lr": 0.05,
                                                      "weight_decay": 0.0}


def test_every_train_option_is_classified_for_the_record():
    parser = build_parser()
    train_parser = next(a for a in parser._subparsers._group_actions[0].choices.items()
                        if a[0] == "train")[1]
    dests = {a.dest for a in train_parser._actions if a.option_strings and a.dest != "help"}
    assert set(prov.RECORDED_ARGUMENTS).isdisjoint(prov.EXCLUDED_ARGUMENTS)
    assert dests == set(prov.RECORDED_ARGUMENTS) | set(prov.EXCLUDED_ARGUMENTS)


def leaks(text, needles):
    return [n for n in needles if n and len(n) >= 4 and n in text]


def test_no_run_produced_value_names_a_path_a_host_or_a_user(trained, tmp_path):
    data, model = trained["noise2clean"]
    text = prov.canonical(manifest_of(model))
    needles = [str(data), str(model), str(Path(data).parent), socket.gethostname(),
               getpass.getuser(), str(Path.home())]
    assert leaks(text, needles) == []
    planted = text.replace('"device":"cpu"', f'"device":"{data}"')
    assert str(data) in leaks(planted, needles)


def test_the_commit_is_recorded_for_this_checkout_only(tmp_path):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    if head.returncode != 0:
        pytest.skip("not a git checkout")
    mine = prov.code_record(REPO / "src" / "dnndenoiser")
    assert mine["commit"] == head.stdout.strip() and isinstance(mine["tree_clean"], bool)
    # A package inside an ignored environment of another repository: unknown.
    outer = tmp_path / "outer"
    pkg = outer / ".venv" / "site-packages" / "dnndenoiser"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (outer / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    for cmd in (["init", "-q"], ["add", ".gitignore"],
                ["-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-q",
                 "-m", "x"]):
        subprocess.run(["git", *cmd], cwd=outer, check=True, capture_output=True)
    assert prov.code_record(pkg) == {"commit": "unknown", "tree_clean": "unknown"}
    # Even when that repository tracks a src/dnndenoiser of its own: the imported package
    # is not that directory.
    vendored = outer / "src" / "dnndenoiser"
    vendored.mkdir(parents=True)
    (vendored / "__init__.py").write_text("", encoding="utf-8")
    for cmd in (["add", "src/dnndenoiser/__init__.py"],
                ["-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-q",
                 "-m", "y"]):
        subprocess.run(["git", *cmd], cwd=outer, check=True, capture_output=True)
    assert prov.code_record(pkg) == {"commit": "unknown", "tree_clean": "unknown"}
    assert prov.code_record(vendored)["commit"] != "unknown"     # the documented residual
    assert prov.code_record(tmp_path / "nowhere") == {"commit": "unknown", "tree_clean": "unknown"}


def test_a_diverged_loss_is_stored_as_null_with_its_status(trained):
    m = manifest_of(trained["noise2clean"][1])
    args = SimpleNamespace(**{k: m["command"]["arguments"][k] for k in prov.RECORDED_ARGUMENTS})
    rebuilt = prov.build_manifest(
        args=args, flags_passed=[], device="cpu", training_data=m["training_data"],
        targets=m["targets"], effective=m["command"]["effective"], seeds=m["command"]["seeds"],
        preprocessing=m["preprocessing"], epochs=1, final_loss=float("nan"))
    assert rebuilt["result"]["final_loss"] is None
    assert rebuilt["statuses"] == {"result.final_loss": "non-finite value (nan)"}
    strict(prov.canonical(rebuilt))


@pytest.mark.parametrize("case, reason", [
    ("epochs-0", "--epochs must be at least 1, got 0"),
    ("float-index", "'frame_index' of the training file must be a one-dimensional integer array"),
    ("2-d-index", "'frame_index' of the training file must be a one-dimensional integer array"),
    ("short-index", "'frame_index' of the training file has 5 values but its first axis has 8 rows"),
    ("nan-noisy", "the training 'noisy' contains non-finite values"),
])
def test_a_malformed_training_run_is_refused(monkeypatch, capsys, tmp_path, case, reason):
    kwargs = {"frame_index": np.arange(8)}
    extra = []
    if case == "epochs-0":
        extra = ["--epochs", "0"]
    elif case == "float-index":
        kwargs["frame_index"] = np.arange(8.0)
    elif case == "2-d-index":
        kwargs["frame_index"] = np.arange(8).reshape(8, 1)
    elif case == "short-index":
        kwargs["frame_index"] = np.arange(5)
    data = supervised_file(tmp_path / "d.h5", **kwargs)
    if case == "nan-noisy":
        with h5py.File(data, "a") as f:
            f["noisy"][2, 3] = np.nan
    argv = ["train", "-d", str(data), "-o", str(tmp_path / "m.pt"), "--epochs", "1",
            "--device", "cpu", *extra]
    assert reason in refuse(monkeypatch, capsys, *argv)
    assert not (tmp_path / "m.pt").exists()


def test_a_nan_frame_is_refused_before_normalisation(monkeypatch, capsys, tmp_path):
    data = stack_file(tmp_path / "s.h5")
    with h5py.File(data, "a") as f:
        f["frames"][1, 2] = np.nan
    err = refuse(monkeypatch, capsys, "train", "-d", str(data), "-o", str(tmp_path / "m.pt"),
                 "--method", "moving-average", "--epochs", "1", "--device", "cpu")
    assert "the training 'frames' contains non-finite values" in err


# ---------------------------------------------------------------------------------- 2

FROZEN_BODY = "ce3e363ffbbcd7398f07838e5a42126b8d6306da4a5abe6ac9aa539b5e2e82eb"
FROZEN_MODEL = "b4f6bc3ebe185b3ab1e83da3436c36602ef5e7e7d889a8eabdb3d35bd6da9054"
SMALL_STATE = {"b.num_batches_tracked": torch.tensor(3, dtype=torch.int64),
               "a.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
SMALL_CONFIG = {"architecture": "FCNN", "num_features": 3, "num_hidden_units": 2,
                "encoder_output_dim": 2, "training_method": "noise2clean", "normalisation": None}


def test_the_frozen_digest_vectors_reproduce():
    """The vector includes a 0-d buffer (a BatchNorm counter): its header shape is empty."""
    from dnndenoiser import digest as dg
    assert dg.array_payload(np.array(3, dtype=np.int64)).startswith(b"int64|\x00")
    body = prov.body_digest(SMALL_STATE, SMALL_CONFIG)
    assert body == {"format": "dnd-digest-1", "sha256": FROZEN_BODY}
    assert prov.model_digest(body, '{"x":"é"}')["sha256"] == FROZEN_MODEL


def independent_model_digest(checkpoint, resolved):
    state = checkpoint["model_state_dict"]
    config = {"architecture": resolved["architecture"],
              "num_features": resolved["num_features"],
              "num_hidden_units": resolved["num_hidden_units"],
              "encoder_output_dim": resolved["encoder_output_dim"],
              "training_method": checkpoint.get("training_method"),
              "normalisation": checkpoint.get("normalisation")}
    body = _digest([(k, _array(state[k].numpy())) for k in sorted(state)]
                   + [("config", _json(config))])
    text = json.dumps(checkpoint["provenance"], sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    return _digest([("body", _json({"format": "dnd-digest-1", "sha256": body})),
                    ("provenance", text.encode("utf-8"))])


@pytest.mark.parametrize("method", ["noise2clean", "moving-average"])
def test_the_model_digest_equals_an_independent_computation(trained, method):
    ck = torch.load(trained[method][1], map_location="cpu", weights_only=True)
    from dnndenoiser.cli import checkpoint_model_config
    resolved = checkpoint_model_config(ck, "x")
    assert ck["model_digest"]["sha256"] == independent_model_digest(ck, resolved)


def infer(monkeypatch, data, model, out, *extra):
    run(monkeypatch, "infer", "-d", str(data), "-m", str(model), "-o", str(out),
        "--device", "cpu", *extra)
    return out


@pytest.mark.parametrize("edit", ["weight", "buffer", "architecture-key", "normalisation",
                                  "manifest"])
def test_infer_refuses_a_checkpoint_changed_without_its_digest(monkeypatch, capsys, tmp_path,
                                                              trained, edit):
    method = {"buffer": "noise2clean", "normalisation": "moving-average"}.get(edit, "noise2clean")
    data, model = trained[method]
    ck = torch.load(model, map_location="cpu", weights_only=True)
    if edit == "buffer":
        # A ResNet-1DCNN carries BatchNorm buffers; train one and change a running mean.
        model = tmp_path / "cnn.pt"
        ck = train(monkeypatch, data, model, "--arch", "ResNet-1DCNN", "--lr", "0.001")
        key = next(k for k in ck["model_state_dict"] if k.endswith("running_mean"))
        ck["model_state_dict"][key] = ck["model_state_dict"][key] + 1.0
    elif edit == "weight":
        key = sorted(ck["model_state_dict"])[0]
        ck["model_state_dict"][key] = ck["model_state_dict"][key].clone()
        ck["model_state_dict"][key].view(-1)[0] += 1e-3
    elif edit == "architecture-key":
        ck["hidden_units"] = ck["hidden_units"]       # unchanged value...
        ck["training_method"] = "noise2noise"          # ...but a key infer reports changed
    elif edit == "normalisation":
        ck["normalisation"] = {**ck["normalisation"], "max": ck["normalisation"]["max"] * 2}
    elif edit == "manifest":
        ck["provenance"]["training_data"]["acquisition_id"] = "another"
    altered = tmp_path / "altered.pt"
    torch.save(ck, altered)
    err = refuse(monkeypatch, capsys, "infer", "-d", str(data), "-m", str(altered),
                 "-o", str(tmp_path / "o.h5"), "--device", "cpu")
    assert "does not match its weights, configuration and manifest" in err
    assert not (tmp_path / "o.h5").exists()


def test_infer_accepts_an_unaltered_and_a_pre_manifest_checkpoint(monkeypatch, tmp_path, trained):
    data, model = trained["noise2clean"]
    out = infer(monkeypatch, data, model, tmp_path / "o.h5")
    with h5py.File(out) as f:
        assert prov.OUTPUT_MANIFEST in f
    ck = torch.load(model, map_location="cpu", weights_only=True)
    del ck["provenance"], ck["model_digest"]
    old = tmp_path / "old.pt"
    torch.save(ck, old)
    out2 = infer(monkeypatch, data, old, tmp_path / "o2.h5")
    with h5py.File(out2) as f:
        assert prov.OUTPUT_MANIFEST not in f
        assert not {prov.OUTPUT_DIGEST, prov.OUTPUT_BODY_DIGEST} & set(f["denoised"].attrs)
        assert prov.INPUT_ARRAY_DIGEST in f["noisy"].attrs


def test_both_spellings_of_a_shape_key_hash_the_resolved_values(monkeypatch, tmp_path, trained):
    data, model = trained["noise2clean"]
    ck = torch.load(model, map_location="cpu", weights_only=True)
    ck["num_features"] = ck["n_features"]
    both = tmp_path / "both.pt"
    torch.save(ck, both)
    infer(monkeypatch, data, both, tmp_path / "o.h5")         # still verifies


def test_a_state_dict_entry_without_a_digest_is_refused_naming_it():
    with pytest.raises(prov.MalformedProvenance, match="state-dict entry 'x' is a int"):
        prov.body_digest({"x": 3}, SMALL_CONFIG)
    with pytest.raises(prov.MalformedProvenance, match="state-dict entry 'w' has no digest"):
        prov.body_digest({"w": torch.zeros(2, dtype=torch.bfloat16)}, SMALL_CONFIG)


# ---------------------------------------------------------------------------------- 3


@pytest.mark.parametrize("case, reason", [
    ("manifest-only", "has 'provenance' without its pair"),
    ("digest-only", "has 'model_digest' without its pair"),
    ("unknown-schema", "unknown provenance schema 'dnd-provenance-9'"),
    ("missing-field", "'result' fields do not match dnd-provenance-1: missing ['final_loss']"),
    ("unknown-field", "'provenance' fields do not match dnd-provenance-1: unknown ['extra']"),
    ("wrong-type", "'training_data.frame_index_runs' must be null or a list"),
    ("bad-digest", "'model_digest' must be {'format': 'dnd-digest-1'"),
])
def test_a_partial_or_malformed_checkpoint_record_is_refused(monkeypatch, capsys, tmp_path,
                                                             trained, case, reason):
    data, model = trained["noise2clean"]
    ck = torch.load(model, map_location="cpu", weights_only=True)
    if case == "manifest-only":
        del ck["model_digest"]
    elif case == "digest-only":
        del ck["provenance"]
    elif case == "unknown-schema":
        ck["provenance"]["schema"] = "dnd-provenance-9"
    elif case == "missing-field":
        del ck["provenance"]["result"]["final_loss"]
    elif case == "unknown-field":
        ck["provenance"]["extra"] = 1
    elif case == "wrong-type":
        ck["provenance"]["training_data"]["frame_index_runs"] = "0-7"
    elif case == "bad-digest":
        ck["model_digest"] = {"format": "dnd-digest-1", "sha256": "abc"}
    bad = tmp_path / "bad.pt"
    torch.save(ck, bad)
    assert reason in refuse(monkeypatch, capsys, "infer", "-d", str(data), "-m", str(bad),
                            "-o", str(tmp_path / "o.h5"), "--device", "cpu")


def test_a_non_json_value_through_the_full_unpickler_is_refused(monkeypatch, capsys, tmp_path,
                                                               trained):
    data, model = trained["noise2clean"]
    ck = torch.load(model, map_location="cpu", weights_only=True)
    ck["provenance"]["result"]["final_loss"] = np.float64(0.5)
    bad = tmp_path / "bad.pt"
    torch.save(ck, bad)
    err = refuse(monkeypatch, capsys, "infer", "-d", str(data), "-m", str(bad),
                 "-o", str(tmp_path / "o.h5"), "--device", "cpu", "--trust-checkpoint")
    assert "'result.final_loss' holds a float64, not a JSON value" in err


@pytest.fixture
def output(monkeypatch, tmp_path, trained):
    data, model = trained["noise2clean"]
    return infer(monkeypatch, data, model, tmp_path / "o.h5")


@pytest.mark.parametrize("case, reason", [
    ("edited-manifest", "model_digest does not match the stored model_provenance"),
    ("other-model-digest", "model_digest does not match the stored model_provenance"),
    ("missing-body", "the model records are incomplete: missing ['model_body_digest']"),
    ("missing-manifest", "the model records are incomplete: missing ['model_provenance']"),
    ("not-canonical", "the stored model_provenance is not its own canonical serialisation"),
])
def test_evaluate_refuses_inconsistent_model_records(monkeypatch, capsys, tmp_path, output,
                                                     case, reason):
    with h5py.File(output, "a") as f:
        text = f[prov.OUTPUT_MANIFEST][()].decode("utf-8")
        m = json.loads(text)
        if case == "edited-manifest":
            m["training_data"]["acquisition_id"] = "another"
            del f[prov.OUTPUT_MANIFEST]
            f.create_dataset(prov.OUTPUT_MANIFEST, data=prov.canonical(m),
                             dtype=h5py.string_dtype("utf-8"))
        elif case == "other-model-digest":
            f["denoised"].attrs[prov.OUTPUT_DIGEST] = json.dumps(
                {"format": "dnd-digest-1", "sha256": "0" * 64})
        elif case == "missing-body":
            del f["denoised"].attrs[prov.OUTPUT_BODY_DIGEST]
        elif case == "missing-manifest":
            del f[prov.OUTPUT_MANIFEST]
        elif case == "not-canonical":
            del f[prov.OUTPUT_MANIFEST]
            f.create_dataset(prov.OUTPUT_MANIFEST, data=json.dumps(m, indent=1),
                             dtype=h5py.string_dtype("utf-8"))
    assert reason in refuse(monkeypatch, capsys, "evaluate", "-d", str(output))


# ---------------------------------------------------------------------------------- 4


def test_infer_writes_the_records_and_the_input_digest(monkeypatch, tmp_path, trained):
    data, model = trained["noise2clean"]
    ck = torch.load(model, map_location="cpu", weights_only=True)
    src = supervised_file(tmp_path / "wide.h5", e=40)           # resampled 40 -> 32
    with h5py.File(src, "a") as f:
        f["clean"].attrs[prov.SIGNAL_IDENTITY] = "generate-signal:0123456789abcdef"
        noisy_in = f["noisy"][:]
    out = infer(monkeypatch, src, model, tmp_path / "o.h5")
    with h5py.File(out) as f:
        stored = f[prov.OUTPUT_MANIFEST][()].decode("utf-8")
        assert stored == json.dumps(ck["provenance"], sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False)
        assert json.loads(f["denoised"].attrs[prov.OUTPUT_DIGEST]) == ck["model_digest"]
        assert json.loads(f["noisy"].attrs[prov.INPUT_ARRAY_DIGEST]) == matching(noisy_in)
        assert f["noisy"].shape[-1] == 32
        assert f["clean"].attrs[prov.SIGNAL_IDENTITY] == "generate-signal:0123456789abcdef"


def test_infer_does_not_copy_the_input_model_records(monkeypatch, tmp_path, trained, output):
    data, model = trained["noise2clean"]
    ck = torch.load(model, map_location="cpu", weights_only=True)
    del ck["provenance"], ck["model_digest"]
    old = tmp_path / "old.pt"
    torch.save(ck, old)
    again = infer(monkeypatch, output, old, tmp_path / "again.h5")
    with h5py.File(again) as f:
        assert prov.OUTPUT_MANIFEST not in f
        assert prov.OUTPUT_DIGEST not in f["denoised"].attrs


# ---------------------------------------------------------------------------------- 5


def test_evaluate_reports_the_model_identity(monkeypatch, tmp_path, trained, output):
    m = evaluate(monkeypatch, tmp_path, output)
    model = m["evaluation_context"]["model"]
    ck = torch.load(trained["noise2clean"][1], map_location="cpu", weights_only=True)
    text = json.dumps(ck["provenance"], sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    td = ck["provenance"]["training_data"]
    assert model == {
        "model_digest": ck["model_digest"],
        "manifest_digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "method": "noise2clean",
        "training_data": {"digest": td["digest"], "acquisition_id": "acq-T",
                          "intensity_units": "counts"},
        "software": {"dnndenoiser": ck["provenance"]["software"]["dnndenoiser"]},
        "code": ck["provenance"]["code"],
    }
    assert m["evaluation_context"]["held_out_status"] == "unknown"     # phase A


FROZEN_MANIFEST_DIGEST = "97f06f396a709c3a29824e1cc794eeb98e2d1a262d7d455439d286d42803f0fe"   # sha256 of the UTF-8 bytes of {"x":"é"}


def test_the_manifest_digest_of_a_fixed_record():
    records = {"manifest": {"command": {"method": "noise2clean"},
                            "training_data": {"digest": None, "acquisition_id": None,
                                              "intensity_units": None},
                            "software": {"dnndenoiser": "0"},
                            "code": {"commit": "unknown", "tree_clean": "unknown"}},
               "manifest_text": '{"x":"é"}', "model_digest": {}, "body": {}}
    assert prov.model_identity(records)["manifest_digest"] == FROZEN_MANIFEST_DIGEST


def test_evaluate_reports_unknown_without_a_manifest(monkeypatch, tmp_path):
    n, d, c = arrays()
    path = write(tmp_path / "t.h5", n, d, c, declaration=TRUTH)
    assert evaluate(monkeypatch, tmp_path, path)["evaluation_context"]["model"] == "unknown"
