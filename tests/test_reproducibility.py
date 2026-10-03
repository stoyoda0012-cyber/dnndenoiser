"""The reproducibility contract (docs/design/REPRODUCIBILITY.md §4).

Tier 1 runs are compared across **separate processes**, on this machine only: two child
`dnndenoiser train` runs with the same seed, device cpu and thread count must give the
same ``model_body_digest``. Each equality has negative controls -- runs that must differ --
so an equality that holds because the seed was ignored, a file was reused or the wrong
object was digested fails. Tier 2 runs are not tested to agree or to differ.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import dnndenoiser
from dnndenoiser import provenance as prov
from dnndenoiser.data.frame_stack import write_frame_stack

CHILD = ("import sys; from dnndenoiser.cli import main; "
         "sys.argv = ['dnndenoiser'] + sys.argv[1:]; main()")


# The children must run the code under test, not whatever dnndenoiser the environment
# resolves (an editable install of another checkout, say).
PACKAGE_ROOT = str(Path(dnndenoiser.__file__).resolve().parents[1])


def child_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    env["PYTHONPATH"] = os.pathsep.join([PACKAGE_ROOT] + [p for p in env.get("PYTHONPATH", "")
                                                        .split(os.pathsep) if p])
    env.update({k: str(v) for k, v in extra.items()})
    return env


def test_the_children_import_the_code_under_test():
    proc = subprocess.run([sys.executable, "-c", "import dnndenoiser; print(dnndenoiser.__file__)"],
                          env=child_env(), capture_output=True, text=True, encoding="utf-8")
    assert Path(proc.stdout.strip()).resolve() == Path(dnndenoiser.__file__).resolve()


def child_train(tmp_path, name, data, *argv, env=None):
    """Train in a child process; return the checkpoint, read fresh from a new file."""
    out = tmp_path / f"{name}.pt"
    assert not out.exists()
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, "train", "-d", str(data), "-o", str(out),
         "--epochs", "1", "--batch-size", "8", "--device", "cpu", *argv],
        env=env or child_env(), capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert out.exists()
    return torch.load(out, map_location="cpu", weights_only=True)


def body_digest(ck):
    """The checkpoint's model_body_digest, recomputed from the file's own contents."""
    from dnndenoiser.cli import checkpoint_model_config
    return prov.body_digest(ck["model_state_dict"],
                            prov.body_config(ck, checkpoint_model_config(ck, "x")))["sha256"]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("repro")
    rng = np.random.default_rng(0)
    clean = (1.0 + rng.normal(0, 0.1, (16, 32))).astype(np.float32)
    noisy = (clean + rng.normal(0, 0.05, clean.shape)).astype(np.float32)
    sup = d / "sup.h5"
    from tests.test_evaluate_reference import TRUTH, write
    write(sup, noisy, None, clean, declaration=TRUTH)
    flat = d / "flat.h5"
    write_frame_stack(flat, (1 + rng.normal(0, .1, (12, 256))).astype(np.float32),
                      np.linspace(0, 1, 256))
    stack3 = d / "stack3.h5"
    write_frame_stack(stack3, (1 + rng.normal(0, .1, (8, 2, 256))).astype(np.float32),
                      np.linspace(0, 1, 256), angles=np.array([10.0, 20.0]),
                      angle_kind="emission", angle_units="deg")
    return {"sup": sup, "flat": flat, "stack3": stack3}


# ---------------------------------------------------------------------------------- 1

SUPERVISED = [("noise2clean", arch, threads)
              for arch in ("FCNN", "ResNet-FCNN", "1D-CNN", "ResNet-1DCNN", "GRU", "LSTM",
                           "bi-LSTM", "Transformer")
              for threads in (1,)] + [
    ("noise2clean", "Transformer", 2), ("noise2clean", "bi-LSTM", 4),
    ("noise2noise", "FCNN", 1)]


@pytest.mark.parametrize("method, arch, threads", SUPERVISED,
                         ids=[f"{m}-{a}-t{t}" for m, a, t in SUPERVISED])
def test_tier_1_supervised_runs_reproduce_across_processes(tmp_path, data, method, arch, threads):
    extra = ["--method", method, "--arch", arch, "--seed", "3", "--threads", str(threads)]
    if method == "noise2noise":
        extra += ["--noise-level", "100"]
    a = child_train(tmp_path, "a", data["sup"], *extra)
    b = child_train(tmp_path, "b", data["sup"], *extra)
    # Eligible unless this checkout has uncommitted changes, which the run records.
    expected = ("tier-2" if a["provenance"]["code"]["tree_clean"] is False
                else "tier-1-eligible")
    assert a["provenance"]["reproducibility"]["tier"] == expected
    assert body_digest(a) == body_digest(b)


@pytest.mark.parametrize("stack", ["flat", "stack3"])
def test_tier_1_moving_average_runs_reproduce_across_processes(tmp_path, data, stack):
    extra = ["--method", "moving-average", "--seed", "3", "--threads", "1"]
    a = child_train(tmp_path, "a", data[stack], *extra)
    b = child_train(tmp_path, "b", data[stack], *extra)
    assert body_digest(a) == body_digest(b)


# ---------------------------------------------------------------------------------- 2


def test_negative_controls_each_change_the_digest(tmp_path, data):
    base = ["--arch", "FCNN", "--seed", "3", "--threads", "1"]
    reference = body_digest(child_train(tmp_path, "ref", data["sup"], *base))
    other_seed = body_digest(child_train(tmp_path, "seed", data["sup"], "--arch", "FCNN",
                                         "--seed", "4", "--threads", "1"))
    changed = tmp_path / "changed.h5"
    changed.write_bytes(data["sup"].read_bytes())
    with h5py.File(changed, "a") as f:
        f["noisy"][0, 0] = f["noisy"][0, 0] + np.float32(0.25)   # one value of the trained array
    other_byte = body_digest(child_train(tmp_path, "byte", changed, *base))
    other_lr = body_digest(child_train(tmp_path, "lr", data["sup"], *base, "--lr", "0.02"))
    digests = {"seed": other_seed, "input": other_byte, "lr": other_lr}
    assert all(d != reference for d in digests.values()), {k: d == reference for k, d in digests.items()}


# ---------------------------------------------------------------------------------- 3


@pytest.mark.parametrize("argv, env, expected", [
    (["--threads", "1"], {}, 1),
    (["--threads", "2"], {}, 2),
    ([], {"OMP_NUM_THREADS": 3}, 3),
    (["--threads", "1"], {"MKL_NUM_THREADS": 4}, 1),
], ids=["threads-1", "threads-2", "omp-3", "threads-over-mkl"])
@pytest.mark.parametrize("method", ["noise2clean", "moving-average"])
def test_the_recorded_thread_count_is_the_one_in_force(tmp_path, data, argv, env, expected, method):
    stack = data["sup"] if method == "noise2clean" else data["flat"]
    ck = child_train(tmp_path, "t", stack, "--method", method, "--seed", "1", *argv,
                     env=child_env(**env))
    assert ck["provenance"]["software"]["torch_threads"] == expected


@pytest.mark.parametrize("method", ["noise2clean", "moving-average"])
def test_the_thread_count_is_in_force_when_training_starts(monkeypatch, tmp_path, data, method):
    """In process, restoring the count afterwards so later tests are unaffected."""
    from dnndenoiser.training import selfsupervised as ss
    from tests.test_evaluate_reference import run
    seen = {}
    before = torch.get_num_threads()
    real_file = h5py.File

    class SpyFile(real_file):              # the first read of the training data
        def __init__(self, *a, **k):
            seen.setdefault("threads", torch.get_num_threads())
            super().__init__(*a, **k)
    monkeypatch.setattr(h5py, "File", SpyFile)
    del ss
    target = 1 if before != 1 else 2
    try:
        run(monkeypatch, "train", "-d", str(data["sup" if method == "noise2clean" else "flat"]),
            "-o", str(tmp_path / "m.pt"), "--method", method, "--epochs", "1",
            "--device", "cpu", "--threads", str(target))
    finally:
        torch.set_num_threads(before)
    assert seen["threads"] == target           # in force before any data was read


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_thread_count_below_one_is_refused(monkeypatch, capsys, tmp_path, data, value):
    from tests.test_evaluate_reference import refuse
    before = torch.get_num_threads()
    err = refuse(monkeypatch, capsys, "train", "-d", str(data["sup"]), "-o", str(tmp_path / "m.pt"),
                 "--epochs", "1", "--device", "cpu", "--threads", value)
    assert f"--threads must be at least 1, got {value}" in err
    assert torch.get_num_threads() == before


# ---------------------------------------------------------------------------------- 4


@pytest.mark.parametrize("device, seed, tree_clean, expected", [
    ("cpu", 3, True, "tier-1-eligible"), ("cpu", 3, "unknown", "tier-1-eligible"),
    ("cpu", None, True, "tier-2"), ("cpu", 3, False, "tier-2"),
    ("mps", 3, True, "tier-2"), ("cuda", 3, True, "tier-2"),
])
def test_the_contract_label(device, seed, tree_clean, expected):
    assert prov.contract_tier(device, seed, tree_clean) == expected


@pytest.fixture(scope="module")
def manifest(tmp_path_factory, data):
    return child_train(tmp_path_factory.mktemp("m"), "m", data["sup"], "--seed", "3",
                       "--threads", "1")["provenance"]


def test_an_under_claimed_label_is_refused(manifest):
    good = json.loads(json.dumps(manifest))
    good["code"] = {"commit": "0" * 40, "tree_clean": True}
    good["reproducibility"]["tier"] = "tier-2"
    with pytest.raises(prov.MalformedProvenance, match="'reproducibility.tier' must be 'tier-1-eligible'"):
        prov.validate_manifest(good)
    good["reproducibility"]["tier"] = "tier-3"
    with pytest.raises(prov.MalformedProvenance, match="'reproducibility.tier' must be one of"):
        prov.validate_manifest(good)


@pytest.mark.parametrize("field, value, reason", [
    ("software.torch_threads", 0, "'software.torch_threads' must be an integer >= 1"),
    ("command.arguments.threads", -5, "'command.arguments.threads' must be an integer >= 1"),
])
def test_thread_counts_below_one_are_refused_in_a_manifest(manifest, field, value, reason):
    bad = json.loads(json.dumps(manifest))
    head, key = field.rsplit(".", 1)
    node = bad
    for part in head.split("."):
        node = node[part]
    node[key] = value
    with pytest.raises(prov.MalformedProvenance, match=reason):
        prov.validate_manifest(bad)


def test_the_label_uses_the_resolved_device(monkeypatch, manifest):
    from types import SimpleNamespace
    monkeypatch.setattr(prov, "code_record", lambda: {"commit": "0" * 40, "tree_clean": True})
    args = SimpleNamespace(**{k: manifest["command"]["arguments"][k] for k in prov.RECORDED_ARGUMENTS})
    args.device = "auto"
    m = prov.build_manifest(
        args=args, flags_passed=[], device="cpu", training_data=manifest["training_data"],
        targets=manifest["targets"], effective=manifest["command"]["effective"],
        seeds=manifest["command"]["seeds"], preprocessing=manifest["preprocessing"],
        epochs=1, final_loss=0.1, torch_threads=1)
    assert m["software"]["device"] == "cpu"
    assert m["reproducibility"]["tier"] == "tier-1-eligible"
    m2 = prov.build_manifest(
        args=SimpleNamespace(**{**vars(args), "device": "cpu"}), flags_passed=[], device="mps",
        training_data=manifest["training_data"], targets=manifest["targets"],
        effective=manifest["command"]["effective"], seeds=manifest["command"]["seeds"],
        preprocessing=manifest["preprocessing"], epochs=1, final_loss=0.1, torch_threads=1)
    assert m2["reproducibility"]["tier"] == "tier-2"


def test_version_1_and_2_nested_key_errors_name_their_version(manifest):
    from tests.test_frame_stack_channels import as_version_1, as_version_2
    for convert, version in ((as_version_1, "dnd-provenance-1"), (as_version_2, "dnd-provenance-2")):
        bad = convert(manifest)
        bad["command"]["effective"]["optimiser"]["extra"] = 1
        with pytest.raises(prov.MalformedProvenance, match=f"fields do not match {version}"):
            prov.validate_manifest(bad)


@pytest.mark.parametrize("change", ["device-mps", "no-seed", "dirty-tree"])
def test_a_label_that_disagrees_with_its_fields_is_refused(manifest, change):
    bad = json.loads(json.dumps(manifest))
    bad["code"] = {"commit": "0" * 40, "tree_clean": True}       # an eligible starting point
    bad["reproducibility"]["tier"] = "tier-1-eligible"
    assert prov.validate_manifest(json.loads(json.dumps(bad)))
    if change == "device-mps":
        bad["software"]["device"] = "mps"
    elif change == "no-seed":
        bad["command"]["seeds"]["torch"] = None
    else:
        bad["code"] = {"commit": "0" * 40, "tree_clean": False}
    assert bad["reproducibility"]["tier"] == "tier-1-eligible"
    with pytest.raises(prov.MalformedProvenance, match="'reproducibility.tier' must be 'tier-2'"):
        prov.validate_manifest(bad)
    bad["reproducibility"]["tier"] = "tier-2"
    assert prov.validate_manifest(bad)


def test_an_unseeded_run_is_tier_2(tmp_path, data):
    ck = child_train(tmp_path, "u", data["sup"], "--threads", "1")
    assert ck["provenance"]["reproducibility"]["tier"] == "tier-2"


# ---------------------------------------------------------------------------------- 5


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable (local only)")
@pytest.mark.parametrize("method", ["noise2clean", "moving-average"])
def test_a_checkpoint_trained_on_mps_stores_cpu_tensors(monkeypatch, tmp_path, data, method):
    """The model is captured when it is built, and its weights read on MPS after training,
    independently of the code that moves them to the CPU for saving."""
    import hashlib
    from dnndenoiser.models import network
    from tests.test_evaluate_reference import run
    models = []
    real_init = network.DenoisingNetwork.__init__

    def spy_init(self, *a, **k):
        real_init(self, *a, **k)
        models.append(self)
    monkeypatch.setattr(network.DenoisingNetwork, "__init__", spy_init)
    stack = data["sup"] if method == "noise2clean" else data["flat"]
    run(monkeypatch, "train", "-d", str(stack), "-o", str(tmp_path / "m.pt"), "--method", method,
        "--epochs", "1", "--device", "mps", "--seed", "1")
    trained = models[-1].state_dict()
    assert any(v.device.type == "mps" for v in trained.values())
    loaded = torch.load(tmp_path / "m.pt", weights_only=True)    # no map_location
    assert all(v.device.type == "cpu" for v in loaded["model_state_dict"].values())

    def digest(sd):
        h = hashlib.sha256()
        for key in sorted(sd):
            h.update(key.encode())
            h.update(sd[key].detach().to("cpu").numpy().tobytes())
        return h.hexdigest()
    assert digest(loaded["model_state_dict"]) == digest(trained)


# ---------------------------------------------------------------------------------- 6


def test_version_3_key_sets_and_earlier_versions(manifest):
    from tests.test_frame_stack_channels import as_version_1, as_version_2
    assert manifest["schema"] == "dnd-provenance-3"
    assert set(manifest["reproducibility"]) == {"tier"}
    assert type(manifest["software"]["torch_threads"]) is int
    assert manifest["command"]["arguments"]["threads"] == 1
    assert prov.validate_manifest(as_version_2(manifest))
    assert prov.validate_manifest(as_version_1(manifest))
    v2_with = as_version_2(manifest)
    v2_with["software"]["torch_threads"] = 1
    with pytest.raises(prov.MalformedProvenance, match="'software' fields do not match dnd-provenance-2"):
        prov.validate_manifest(v2_with)
    v3_without = json.loads(json.dumps(manifest))
    del v3_without["reproducibility"]
    with pytest.raises(prov.MalformedProvenance, match="'provenance' fields do not match dnd-provenance-3"):
        prov.validate_manifest(v3_without)


def test_evaluate_reports_threads_and_tier_and_keeps_version_2_declarations():
    base = {"command": {"method": "moving-average"},
            "training_data": {"digest": None, "acquisition_id": None, "intensity_units": None,
                              "angles": {"kind": "emission", "units": "deg"},
                              "frame_index_basis": "inferred"},
            "software": {"dnndenoiser": "0", "torch_threads": 2},
            "code": {"commit": "unknown", "tree_clean": "unknown"},
            "reproducibility": {"tier": "tier-1-eligible"}}
    records = lambda m: {"manifest": m, "manifest_text": "{}", "model_digest": {}, "body": {}}  # noqa: E731
    v3 = prov.model_identity(records({**base, "schema": "dnd-provenance-3"}))
    assert (v3["torch_threads"], v3["tier"]) == (2, "tier-1-eligible")
    v2 = prov.model_identity(records({**base, "schema": "dnd-provenance-2"}))
    assert (v2["torch_threads"], v2["tier"]) == ("not recorded", "not recorded")
    assert v2["training_data"]["declared_angles"] == {"kind": "emission", "units": "deg"}
    assert v2["training_data"]["declared_frame_index_basis"] == "inferred"


def test_a_version_2_checkpoint_still_verifies(monkeypatch, tmp_path, data):
    from dnndenoiser.cli import checkpoint_model_config
    from tests.test_evaluate_reference import run
    from tests.test_frame_stack_channels import as_version_2
    ck = child_train(tmp_path, "m", data["sup"], "--seed", "3", "--threads", "1")
    v2 = as_version_2(ck["provenance"])
    prov.seal(ck, v2, checkpoint_model_config(ck, "x"))
    old = tmp_path / "v2.pt"
    torch.save(ck, old)
    out = tmp_path / "o.h5"
    run(monkeypatch, "infer", "-d", str(data["sup"]), "-m", str(old), "-o", str(out), "--device", "cpu")
    with h5py.File(out) as f:
        assert f["model_provenance"][()].decode("utf-8") == prov.canonical(v2)
    assert Path(out).exists()
