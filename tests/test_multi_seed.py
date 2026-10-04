"""Training several seeds and combining their outputs (docs/design/MULTI_SEED.md §6).

Expected values come from outside the code under test: single runs of ``train --seed s`` in
separate processes, stub networks whose outputs are written down here, and arithmetic done
here in float64. Every refusal is pinned to the rule that refused and has an accepted
counterpart; checkpoints that differ are really trained, or re-sealed through
``provenance.seal`` so that they verify and are refused for the planted difference only.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch

from dnndenoiser import cli
from dnndenoiser import ensemble as ens
from dnndenoiser import provenance as prov
from dnndenoiser.data.frame_stack import write_frame_stack
from tests.test_evaluate_reference import FORBIDDEN, refuse, run
from tests.test_provenance_manifest import stack_file, supervised_file, train
from tests.test_reproducibility import CHILD, body_digest, child_env, child_train

# ------------------------------------------------------------------------------ helpers


def child_batch(tmp_path, name, data, seeds, *argv):
    """``train --seeds`` in a child process; the members, read fresh from their files."""
    out = tmp_path / f"{name}.pt"
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, "train", "-d", str(data), "-o", str(out),
         "--epochs", "1", "--batch-size", "8", "--device", "cpu", "--seeds",
         *map(str, seeds), *argv],
        env=child_env(), capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert not out.exists()                      # -o itself is never written
    return {s: torch.load(tmp_path / f"{name}.seed{s}.pt", map_location="cpu",
                          weights_only=True) for s in seeds}


def without(manifest, *paths):
    m = copy.deepcopy(manifest)
    for path in paths:
        *head, last = path.split(".")
        node = m
        for key in head:
            node = node[key]
        node.pop(last, None)
    return m


def reseal(ck, mutate):
    """A copy of a checkpoint whose manifest is changed by ``mutate`` and sealed again, so it
    verifies; the planted difference is the only one."""
    ck = copy.deepcopy(ck)
    manifest = copy.deepcopy(ck[prov.CHECKPOINT_MANIFEST])
    mutate(manifest)
    prov.seal(ck, manifest, cli.checkpoint_model_config(ck, "x"))
    return ck


def save(ck, path):
    torch.save(ck, path)
    return path


def infer_many(monkeypatch, data, models, out, *extra):
    argv = ["infer", "-d", str(data), "-o", str(out), "--device", "cpu", *extra]
    for m in models:
        argv += ["-m", str(m)]
    run(monkeypatch, *argv)
    return out


# ------------------------------------------------------------- 1, 2. batch = single runs


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("ms")
    stack3 = d / "stack3.h5"
    rng = np.random.default_rng(1)
    write_frame_stack(stack3, (1.0 + rng.normal(0, 0.1, (8, 2, 64))).astype(np.float32),
                      np.linspace(0.0, 1.0, 64), angles=np.array([10.0, 30.0]),
                      angle_kind="emission", angle_units="deg")
    return {"sup": supervised_file(d / "sup.h5", n=16, acquisition_id="acq-M",
                                   frame_index=np.arange(16)),
            "stack": stack_file(d / "stack.h5"), "stack3": stack3}


CASES = [
    ("sup", ("--method", "noise2clean", "--arch", "FCNN", "--threads", "1"), (3, 5)),
    ("sup", ("--method", "noise2noise", "--noise-level", "100", "--threads", "1"), (3, 5)),
    ("stack", ("--method", "moving-average", "--threads", "1"), (5, 3)),
    ("stack3", ("--method", "moving-average", "--threads", "1"), (3, 5)),
    ("sup", ("--method", "noise2clean", "--arch", "Transformer", "--threads", "2"), (5, 3)),
]


@pytest.mark.parametrize("data, argv, seeds", CASES)
def test_each_member_equals_the_single_run_of_its_seed(tmp_path, files, data, argv, seeds):
    members = child_batch(tmp_path, "b", files[data], seeds, *argv)
    for s in seeds:
        single = child_train(tmp_path, f"single{s}", files[data], "--seed", str(s), *argv)
        assert body_digest(members[s]) == body_digest(single)
        assert (without(members[s][prov.CHECKPOINT_MANIFEST], "created_utc")
                == without(single[prov.CHECKPOINT_MANIFEST], "created_utc"))


def test_members_differ_and_differ_from_another_seed(tmp_path, files):
    members = child_batch(tmp_path, "b", files["sup"], (3, 5), "--threads", "1")
    other = child_train(tmp_path, "s4", files["sup"], "--seed", "4", "--threads", "1")
    digests = {body_digest(members[3]), body_digest(members[5]), body_digest(other)}
    assert len(digests) == 3


# ------------------------------------------------------------ 3. refusals before training


@pytest.fixture
def no_training(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_train_member", lambda args: calls.append(args.seed))
    return calls


@pytest.mark.parametrize("argv, reason", [
    (("--seeds", "3"), "at least two seeds"),
    (("--seeds", "3", "3"), "repeated"),
    (("--seeds", "3", "-1"), "non-negative"),
    (("--seeds", "3", "5", "--lr", "nan"), "--lr is not finite"),
    (("--seeds", "3", "5", "--method", "noise2noise"), "needs --noise-level"),
    (("--seeds", "3", "5", "--method", "moving-average", "--lr", "0.1"),
     "does not take --lr"),
])
def test_a_batch_is_refused_before_any_member_trains(monkeypatch, capsys, tmp_path, files,
                                                     no_training, argv, reason):
    data = files["stack"] if "moving-average" in argv else files["sup"]
    err = refuse(monkeypatch, capsys, "train", "-d", str(data), "-o", str(tmp_path / "m.pt"),
                 "--epochs", "1", "--device", "cpu", *argv)
    assert reason in err
    assert no_training == []


def test_noise2clean_without_clean_is_refused_before_training(monkeypatch, capsys, tmp_path,
                                                              no_training):
    path = tmp_path / "n.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("noisy", data=np.ones((4, 16), dtype=np.float32))
        f.create_dataset("energy", data=np.arange(16.0))
    err = refuse(monkeypatch, capsys, "train", "-d", str(path), "-o", str(tmp_path / "m.pt"),
                 "--seeds", "3", "5")
    assert "requires clean data" in err and no_training == []


@pytest.mark.parametrize("flag", ["--epochs", "--threads"])
def test_zero_epochs_or_threads_are_refused_before_training(monkeypatch, capsys, tmp_path,
                                                            files, no_training, flag):
    err = refuse(monkeypatch, capsys, "train", "-d", str(files["sup"]), "-o",
                 str(tmp_path / "m.pt"), "--seeds", "3", "5", flag, "0")
    assert "must be at least 1" in err and no_training == []


def test_the_output_name_and_existing_members_are_refused(monkeypatch, capsys, tmp_path, files,
                                                          no_training):
    err = refuse(monkeypatch, capsys, "train", "-d", str(files["sup"]), "-o",
                 str(tmp_path / "m.ckpt"), "--seeds", "3", "5")
    assert "must end in .pt" in err
    (tmp_path / "m.seed5.pt").write_bytes(b"x")
    err = refuse(monkeypatch, capsys, "train", "-d", str(files["sup"]), "-o",
                 str(tmp_path / "m.pt"), "--seeds", "3", "5")
    assert "m.seed5.pt" in err and "already exist" in err
    assert no_training == []


def test_seed_and_seeds_together_are_refused(monkeypatch, capsys, tmp_path, files):
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, "train", "-d", str(files["sup"]), "-o", str(tmp_path / "m.pt"),
            "--seed", "1", "--seeds", "3", "5")
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_see_no_longer_abbreviates_seed(monkeypatch, capsys, tmp_path, files):
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, "train", "-d", str(files["sup"]), "-o", str(tmp_path / "m.pt"),
            "--see", "1")
    assert exc.value.code == 2 and "ambiguous" in capsys.readouterr().err


def test_a_valid_batch_writes_one_member_per_seed(monkeypatch, tmp_path, files):
    run(monkeypatch, "train", "-d", str(files["sup"]), "-o", str(tmp_path / "m.pt"),
        "--epochs", "1", "--batch-size", "8", "--device", "cpu", "--seeds", "7", "2")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.seed2.pt", "m.seed7.pt"]
    for s in (7, 2):
        m = torch.load(tmp_path / f"m.seed{s}.pt", weights_only=True)[prov.CHECKPOINT_MANIFEST]
        assert m["command"]["arguments"]["seed"] == s and m["command"]["seeds"]["torch"] == s
        assert "--seed" in m["command"]["flags_passed"]
        assert "--seeds" not in m["command"]["flags_passed"]
        assert "seeds" not in m["command"]["arguments"]


def test_a_member_failing_mid_batch_keeps_and_lists_the_earlier_ones(monkeypatch, capsys,
                                                                     tmp_path, files):
    real = cli._train_member

    def failing(args):
        if args.seed == 5:
            print("Error: planted failure", file=sys.stderr)
            sys.exit(1)
        return real(args)
    monkeypatch.setattr(cli, "_train_member", failing)
    err = refuse(monkeypatch, capsys, "train", "-d", str(files["sup"]), "-o",
                 str(tmp_path / "m.pt"), "--epochs", "1", "--device", "cpu",
                 "--seeds", "3", "5", "8")
    assert (tmp_path / "m.seed3.pt").exists() and not (tmp_path / "m.seed8.pt").exists()
    assert "seed 5" in err and "m.seed3.pt" in err


# ---------------------------------------------------------------- 4. what a seed varies


def test_noise2noise_members_have_their_own_targets(monkeypatch, tmp_path, files):
    import torch.utils.data as tud
    seen = {}
    real = tud.TensorDataset

    class Spy(real):
        def __init__(self, inputs, targets):
            seen[len(seen)] = targets.clone()
            super().__init__(inputs, targets)
    monkeypatch.setattr(tud, "TensorDataset", Spy)
    run(monkeypatch, "train", "-d", str(files["sup"]), "-o", str(tmp_path / "m.pt"),
        "--epochs", "1", "--batch-size", "8", "--device", "cpu", "--method", "noise2noise",
        "--noise-level", "100", "--seeds", "0", "1000")
    assert not torch.equal(seen[0], seen[1])      # the pair revision 0 would have refused
    for s in (0, 1000):
        m = torch.load(tmp_path / f"m.seed{s}.pt", weights_only=True)[prov.CHECKPOINT_MANIFEST]
        assert m["command"]["seeds"]["targets"] == [s, s + 1000]


def test_moving_average_members_share_their_targets(monkeypatch, tmp_path, files):
    from dnndenoiser.training import selfsupervised as ss
    seen = []
    real = ss.channel_targets

    def spy(*a, **k):
        out = real(*a, **k)
        seen.append(np.array(out))
        return out
    monkeypatch.setattr(ss, "channel_targets", spy)
    run(monkeypatch, "train", "-d", str(files["stack"]), "-o", str(tmp_path / "m.pt"),
        "--epochs", "1", "--device", "cpu", "--method", "moving-average", "--seeds", "3", "5")
    assert len(seen) == 2 and np.array_equal(seen[0], seen[1])


# ----------------------------------------------------------------------- 5. compatibility


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory, files):
    """Trained once: a batch, separate single runs, and checkpoints differing in one
    trained-in respect."""
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("ck")
    out = {}
    try:
        def t(name, data, *extra, method="noise2clean"):
            train(mp, data, d / f"{name}.pt", "--threads", "1", *extra, method=method)
            out[name] = d / f"{name}.pt"
        run(mp, "train", "-d", str(files["sup"]), "-o", str(d / "b.pt"), "--epochs", "1",
            "--batch-size", "8", "--device", "cpu", "--threads", "1", "--seeds", "3", "5")
        out["a"], out["b"] = d / "b.seed3.pt", d / "b.seed5.pt"
        t("single3", files["sup"], "--seed", "3")
        t("single5", files["sup"], "--seed", "5")
        other = supervised_file(d / "other.h5", n=16, seed=9, acquisition_id="acq-M",
                                frame_index=np.arange(16))
        t("other_data", other, "--seed", "5")
        units = supervised_file(d / "units.h5", n=16, acquisition_id="acq-M",
                                frame_index=np.arange(16), units="counts_per_s")
        t("other_units", units, "--seed", "5")
        t("other_lr", files["sup"], "--seed", "5", "--lr", "0.02")
        t("other_arch", files["sup"], "--seed", "5", "--arch", "1D-CNN")
        train(mp, files["sup"], d / "threads2.pt", "--threads", "2", "--seed", "5")
        out["threads2"] = d / "threads2.pt"
        t("unseeded1", files["sup"])
        t("unseeded2", files["sup"])
        longer = supervised_file(d / "long.h5", n=16, e=64, acquisition_id="acq-M",
                                 frame_index=np.arange(16))
        t("long", longer, "--seed", "5")
        t("n2n_a", files["sup"], "--seed", "3", "--noise-level", "100", method="noise2noise")
        t("n2n_b", files["sup"], "--seed", "5", "--noise-level", "200", method="noise2noise")
        t("ma_a", files["stack"], "--seed", "3", "--window", "3", method="moving-average")
        t("ma_b", files["stack"], "--seed", "5", "--window", "4", method="moving-average")
    finally:
        mp.undo()
    return out


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def refused_rule(monkeypatch, capsys, data, models, tmp_path):
    err = refuse(monkeypatch, capsys, "infer", "-d", str(data), "-o", str(tmp_path / "o.h5"),
                 "--device", "cpu", *[x for m in models for x in ("-m", str(m))])
    assert not (tmp_path / "o.h5").exists()
    return err


def code_mutation(code):
    def mutate(m):
        m["code"] = code
        clean = code["tree_clean"]
        m["reproducibility"]["tier"] = prov.contract_tier(m["software"]["device"],
                                                          m["command"]["seeds"]["torch"], clean)
    return mutate


def resealed(tmp_path, ck_path, name, mutate):
    return save(reseal(load(ck_path), mutate), tmp_path / f"{name}.pt")


@pytest.mark.parametrize("partner, rule, field", [
    ("other_data", 3, "training_data"),
    ("other_units", 3, "training_data.intensity_units"),
    ("other_lr", 3, "command.arguments.lr"),
    ("other_arch", 3, "command.arguments.arch"),
    ("threads2", 3, "software.torch_threads"),
    ("long", 3, "command.effective.num_features"),
])
def test_a_trained_in_difference_is_refused_by_rule_3(monkeypatch, capsys, tmp_path, files,
                                                      ckpts, partner, rule, field):
    err = refused_rule(monkeypatch, capsys, files["sup"], [ckpts["a"], ckpts[partner]], tmp_path)
    assert f"rule {rule}" in err and field in err


@pytest.mark.parametrize("a, b, field", [
    ("n2n_a", "n2n_b", "command.arguments.noise_level"),
    ("ma_a", "ma_b", "command.arguments.window"),
])
def test_method_arguments_are_compared(monkeypatch, capsys, tmp_path, files, ckpts, a, b, field):
    data = files["stack"] if a.startswith("ma") else files["sup"]
    err = refused_rule(monkeypatch, capsys, data, [ckpts[a], ckpts[b]], tmp_path)
    assert "rule 3" in err and field in err


@pytest.mark.parametrize("name, mutate, field", [
    ("device", lambda m: (m["software"].update(device="mps"),
                          m["reproducibility"].update(tier="tier-2")), "software.device"),
    ("commit", lambda m: m["code"].update(commit="0" * 40), "code.commit"),
    ("numpy", lambda m: m["software"].update(numpy="0.0.1"), "software.numpy"),
    ("dirty", lambda m: code_mutation({"commit": m["code"]["commit"], "tree_clean":
                                       not m["code"]["tree_clean"]})(m), "code.tree_clean"),
    ("unknown", lambda m: code_mutation({"commit": "unknown", "tree_clean": "unknown"})(m),
     "code"),
])
def test_a_recorded_difference_is_refused_by_rule_3(monkeypatch, capsys, tmp_path, files, ckpts,
                                                    name, mutate, field):
    if name == "unknown" and load(ckpts["a"])[prov.CHECKPOINT_MANIFEST]["code"]["commit"] == "unknown":
        pytest.skip("the checkout's commit is unknown here")
    partner = resealed(tmp_path, ckpts["b"], name, mutate)
    err = refused_rule(monkeypatch, capsys, files["sup"], [ckpts["a"], partner], tmp_path)
    assert "rule 3" in err and field in err


def test_a_version_2_member_is_refused_by_rule_2(monkeypatch, capsys, tmp_path, files, ckpts):
    from tests.test_frame_stack_channels import as_version_2
    ck = load(ckpts["b"])
    prov.seal(ck, as_version_2(ck[prov.CHECKPOINT_MANIFEST]), cli.checkpoint_model_config(ck, "x"))
    err = refused_rule(monkeypatch, capsys, files["sup"], [ckpts["a"], save(ck, tmp_path / "v2.pt")],
                       tmp_path)
    assert "rule 2" in err


def test_a_checkpoint_without_a_manifest_is_refused_by_rule_1(monkeypatch, capsys, tmp_path,
                                                              files, ckpts):
    ck = load(ckpts["b"])
    del ck[prov.CHECKPOINT_MANIFEST], ck[prov.CHECKPOINT_DIGEST]
    err = refused_rule(monkeypatch, capsys, files["sup"], [ckpts["a"], save(ck, tmp_path / "n.pt")],
                       tmp_path)
    assert "rule 1" in err


@pytest.mark.parametrize("models, reason", [(("unseeded1", "unseeded2"), "trained without --seed"),
                                            (("a", "a"), "seed 3 is repeated")])
def test_null_or_repeated_seeds_are_refused_by_rule_4(monkeypatch, capsys, tmp_path, files,
                                                      ckpts, models, reason):
    err = refused_rule(monkeypatch, capsys, files["sup"], [ckpts[m] for m in models], tmp_path)
    assert "rule 4" in err and reason in err


def test_a_copy_with_another_recorded_seed_is_refused_by_rule_5(monkeypatch, capsys, tmp_path,
                                                                files, ckpts):
    def mutate(m):
        m["command"]["arguments"]["seed"] = 99
        m["command"]["seeds"]["torch"] = 99
    copy_ = resealed(tmp_path, ckpts["a"], "copy", mutate)
    err = refused_rule(monkeypatch, capsys, files["sup"], [ckpts["a"], copy_], tmp_path)
    assert "rule 5" in err


@pytest.mark.parametrize("models", [("a", "b"), ("single3", "single5")])
def test_seed_only_members_are_combined(monkeypatch, tmp_path, files, ckpts, models):
    out = infer_many(monkeypatch, files["sup"], [ckpts[m] for m in models], tmp_path / "o.h5")
    with h5py.File(out) as f:
        assert f["denoised_members"].shape[0] == 2


def test_auto_and_cpu_resolving_alike_are_combined(monkeypatch, tmp_path, files, ckpts):
    partner = resealed(tmp_path, ckpts["b"], "auto",
                       lambda m: m["command"]["arguments"].update(device="auto"))
    infer_many(monkeypatch, files["sup"], [ckpts["a"], partner], tmp_path / "o.h5")


def test_a_member_with_a_recorded_status_is_kept_and_named(monkeypatch, capsys, tmp_path, files,
                                                           ckpts):
    def diverged(m):
        m["result"]["final_loss"] = None
        m["statuses"] = {"result.final_loss": "non-finite (nan)"}
    partner = resealed(tmp_path, ckpts["b"], "diverged", diverged)
    capsys.readouterr()
    out = infer_many(monkeypatch, files["sup"], [ckpts["a"], partner], tmp_path / "o.h5")
    printed = capsys.readouterr().out
    with h5py.File(out) as f:
        notes = json.loads(f.attrs["ensemble_notes"])
        assert f["denoised_members"].shape[0] == 2
    assert any("seed 5" in n and "result.final_loss" in n for n in notes)
    assert any(line.startswith("Note:") and "seed 5" in line for line in printed.splitlines())


def test_every_manifest_field_is_compared_or_excluded():
    """Adding a manifest field fails here until it is classified (compared by default)."""
    assert prov.MANIFEST_KEYS == {"schema", "created_utc", "software", "code", "command",
                                  "training_data", "targets", "preprocessing", "result",
                                  "statuses", "reproducibility"}
    assert prov.COMMAND_KEYS == {"method", "arguments", "flags_passed", "effective", "seeds"}
    assert prov.RESULT_KEYS == {"epochs", "final_loss"}
    assert prov.SOFTWARE_KEYS == {"dnndenoiser", "python", "numpy", "torch", "h5py", "torch_cuda",
                                  "platform", "device", "torch_threads"}
    assert set(prov.RECORDED_ARGUMENTS) == {
        "arch", "method", "window", "noise_level", "epochs", "batch_size", "seed", "lr",
        "lr_drop_period", "lr_drop_factor", "scheduler", "warmup_epochs", "weight_decay",
        "grad_clip", "hidden_units", "encoder_dim", "device", "threads"}
    assert prov.ENSEMBLE_EXCLUDED == (
        "created_utc", "command.arguments.seed", "command.seeds", "command.flags_passed",
        "command.arguments.device", "command.arguments.threads", "result.final_loss",
        "statuses")


# ------------------------------------------------------------------------ 6. arithmetic


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, z):
        return torch.full_like(z, self.value)


def stub_by_seed(monkeypatch, values):
    def build(checkpoint, config, device):
        return Constant(values[checkpoint[prov.CHECKPOINT_MANIFEST]["command"]["seeds"]["torch"]])
    monkeypatch.setattr(cli, "_build_network", build)


@pytest.fixture(scope="module")
def trio(tmp_path_factory, files):
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("trio")
    try:
        run(mp, "train", "-d", str(files["sup"]), "-o", str(d / "t.pt"), "--epochs", "1",
            "--batch-size", "8", "--device", "cpu", "--seeds", "1", "2", "3")
    finally:
        mp.undo()
    return [d / f"t.seed{s}.pt" for s in (1, 2, 3)]


def test_the_mean_is_accumulated_in_float64(monkeypatch, tmp_path, files, trio):
    """2**24, 1, 1: float32 accumulation loses the ones; float64 does not."""
    stub_by_seed(monkeypatch, {1: 2.0 ** 24, 2: 1.0, 3: 1.0})
    out = infer_many(monkeypatch, files["sup"], trio, tmp_path / "o.h5")
    expected = np.float32((2.0 ** 24 + 2.0) / 3.0)
    float32_sum = np.float32(np.float32(2.0 ** 24) + np.float32(1.0)) + np.float32(1.0)
    assert np.float32(float32_sum / np.float32(3.0)) != expected
    with h5py.File(out) as f:
        assert np.all(f["ensemble_mean_estimate"][()] == expected)


def test_the_spread_has_ddof_one(monkeypatch, tmp_path, files, trio):
    stub_by_seed(monkeypatch, {1: 1.0, 2: 3.0})
    out = infer_many(monkeypatch, files["sup"], trio[:2], tmp_path / "o.h5")
    with h5py.File(out) as f:
        spread = f["between_run_std_fixed_input"][()]
        members = f["denoised_members"][()]
    assert np.allclose(spread, np.sqrt(2.0)) and not np.allclose(spread, 1.0)
    assert np.all(members[0] == 1.0) and np.all(members[1] == 3.0)


class Double(torch.nn.Module):
    def forward(self, z):
        return 2 * z


def test_each_member_is_applied_with_its_own_normalisation(monkeypatch, tmp_path, ckpts):
    """Below the compatibility check: two moving-average models trained on different stacks
    (different normalisations); the stub doubles its normalised input, so the output
    2x − min depends on whose normalisation is applied."""
    monkeypatch.setattr(prov, "check_ensemble", lambda members: [])
    monkeypatch.setattr(cli, "_build_network", lambda checkpoint, config, device: Double())
    d = tmp_path
    s1 = stack_file(d / "s1.h5", seed=1)
    s2 = d / "s2.h5"
    rng = np.random.default_rng(2)
    write_frame_stack(s2, (5.0 + rng.normal(0, 0.5, (12, 64))).astype(np.float32),
                      np.linspace(0.0, 1.0, 64))
    train(monkeypatch, s1, d / "m1.pt", "--seed", "1", method="moving-average")
    train(monkeypatch, s2, d / "m2.pt", "--seed", "2", method="moving-average")
    out = infer_many(monkeypatch, s1, [d / "m1.pt", d / "m2.pt"], d / "o.h5")
    with h5py.File(out) as f:
        x = f["noisy"][()].astype(np.float64)
        members = f["denoised_members"][()].astype(np.float64)
    for k, path in enumerate((d / "m1.pt", d / "m2.pt")):
        lo = load(path)["normalisation"]["min"]
        assert np.allclose(members[k], 2 * x - lo, rtol=1e-5, atol=1e-5)
    assert not np.allclose(members[0], members[1], rtol=1e-3)


# ------------------------------------------------------------------------- 7. the file


EXPECTED_DATASETS_SUP = {"noisy", "clean", "energy", "frame_index", "denoised_members",
                         "ensemble_mean_estimate", "between_run_std_fixed_input",
                         "members/0/model_provenance", "members/1/model_provenance"}
EXPECTED_ROOT_ATTRS = {"ensemble_k", "members_seeds", "inference_device", "ensemble_notes"}


def test_the_file_holds_exactly_what_the_design_lists(monkeypatch, tmp_path, files, ckpts):
    out = infer_many(monkeypatch, files["sup"], [ckpts["b"], ckpts["a"]], tmp_path / "o.h5")
    assert ens.unexpected_contents(out) == []
    with h5py.File(out) as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        assert set(names) == EXPECTED_DATASETS_SUP
        assert set(f.attrs) == EXPECTED_ROOT_ATTRS
        assert list(f.attrs["members_seeds"]) == [5, 3]           # the order given
        assert int(f.attrs["ensemble_k"]) == 2 and f.attrs["inference_device"] == "cpu"
        assert "denoised" not in f
        for name in ("denoised_members", "ensemble_mean_estimate", "between_run_std_fixed_input"):
            assert f[name].attrs["intensity_units"] == "counts"
            assert f[name].attrs["acquisition_id"] == "acq-M"
            assert f[name].dtype == np.float32
        assert "input_array_digest" in f["noisy"].attrs
        for i, s in enumerate((5, 3)):
            g = f["members"][str(i)]
            assert set(g.attrs) == {"model_digest", "model_body_digest", "seed"}
            assert int(g.attrs["seed"]) == s
            records = prov.read_member_records(g)
            assert records["manifest"]["command"]["seeds"]["torch"] == s


def test_a_planted_scalar_summary_is_caught(monkeypatch, tmp_path, files, ckpts):
    out = infer_many(monkeypatch, files["sup"], [ckpts["a"], ckpts["b"]], tmp_path / "o.h5")
    with h5py.File(out, "a") as f:
        f.create_dataset("between_run_std_max", data=1.0)
        f.attrs["spread_mean"] = 0.1
    assert sorted(ens.unexpected_contents(out)) == ["@spread_mean", "between_run_std_max"]


@pytest.mark.parametrize("tamper", ["manifest", "swap"])
def test_member_records_that_were_tampered_with_are_detected(monkeypatch, tmp_path, files, ckpts,
                                                             tamper):
    out = infer_many(monkeypatch, files["sup"], [ckpts["a"], ckpts["b"]], tmp_path / "o.h5")
    with h5py.File(out, "a") as f:
        g0, g1 = f["members"]["0"], f["members"]["1"]
        if tamper == "manifest":
            text = json.loads(g0["model_provenance"][()].decode("utf-8"))
            text["command"]["arguments"]["lr"] = 0.5
            del g0["model_provenance"]
            g0.create_dataset("model_provenance", data=prov.canonical(text),
                              dtype=h5py.string_dtype("utf-8"))
        else:
            d0, d1 = g0.attrs["model_digest"], g1.attrs["model_digest"]
            g0.attrs["model_digest"], g1.attrs["model_digest"] = d1, d0
    with h5py.File(out) as f:
        with pytest.raises(prov.MalformedProvenance, match="does not match"):
            prov.read_member_records(f["members"]["0"])


@pytest.mark.parametrize("data, shape", [("stack3", (8, 2, 256)), ("stack", (12, 256))])
def test_frame_stacks_and_resampling_keep_their_layout(monkeypatch, tmp_path, files, data, shape):
    run(monkeypatch, "train", "-d", str(files[data]), "-o", str(tmp_path / "m.pt"),
        "--epochs", "1", "--device", "cpu", "--method", "moving-average", "--seeds", "3", "5")
    out = infer_many(monkeypatch, files[data], [tmp_path / "m.seed3.pt", tmp_path / "m.seed5.pt"],
                     tmp_path / "o.h5")
    with h5py.File(out) as f:
        assert f["denoised_members"].shape == (2, *shape)
        assert f["ensemble_mean_estimate"].shape == shape
        assert f["between_run_std_fixed_input"].shape == shape
        assert f["energy"].shape == (256,)


def test_evaluate_refuses_an_ensemble_file_and_gives_the_command(monkeypatch, capsys, tmp_path,
                                                                 files, ckpts):
    out = infer_many(monkeypatch, files["sup"], [ckpts["a"], ckpts["b"]], tmp_path / "o.h5")
    err = refuse(monkeypatch, capsys, "evaluate", "-d", str(out))
    assert "ensemble" in err and "dnndenoiser infer" in err and "paired by seed" in err


def test_a_non_finite_member_output_is_refused_and_nothing_written(monkeypatch, capsys, tmp_path,
                                                                   files, trio):
    stub_by_seed(monkeypatch, {1: 1.0, 2: float("nan")})
    err = refuse(monkeypatch, capsys, "infer", "-d", str(files["sup"]), "-o",
                 str(tmp_path / "o.h5"), "--device", "cpu", "-m", str(trio[0]), "-m", str(trio[1]))
    assert "seed 2" in err and "non-finite" in err and "report" in err
    assert not (tmp_path / "o.h5").exists()


def test_notes_appear_exactly_when_they_apply(monkeypatch, tmp_path, files, ckpts):
    clean = code_mutation({"commit": "0" * 40, "tree_clean": True})
    dirty = code_mutation({"commit": "0" * 40, "tree_clean": False})
    unknown = code_mutation({"commit": "unknown", "tree_clean": "unknown"})
    for name, mutate, expected in (("clean", clean, None), ("dirty", dirty, ens.TREE_NOTE),
                                   ("unknown", unknown, ens.UNKNOWN_CODE_NOTE)):
        pair = [resealed(tmp_path, ckpts[k], f"{name}{k}", mutate) for k in ("a", "b")]
        out = infer_many(monkeypatch, files["sup"], pair, tmp_path / f"{name}.h5")
        with h5py.File(out) as f:
            notes = json.loads(f.attrs["ensemble_notes"])
        tier2 = name == "dirty"
        assert (ens.TREE_NOTE in notes) == (expected == ens.TREE_NOTE)
        assert (ens.UNKNOWN_CODE_NOTE in notes) == (expected == ens.UNKNOWN_CODE_NOTE)
        assert (ens.EXECUTION_NOTE in notes) == tier2


def test_one_model_writes_the_single_layout(monkeypatch, tmp_path, files, ckpts):
    out = tmp_path / "o.h5"
    run(monkeypatch, "infer", "-d", str(files["sup"]), "-m", str(ckpts["a"]), "-o", str(out),
        "--device", "cpu")
    with h5py.File(out) as f:
        assert "denoised" in f and "denoised_members" not in f and "members" not in f


# ------------------------------------------------------------------------------ 8. words

EXTRA = ("uncertainty", "confidence", "interval", "error", "precision", "stable", "stability",
         "robust")


def banned_in(names):
    return [w for n in names for w in FORBIDDEN + EXTRA if w in n.lower()]


def test_no_name_or_label_carries_a_banned_word(monkeypatch, capsys, tmp_path, files, ckpts):
    capsys.readouterr()
    out = infer_many(monkeypatch, files["sup"], [ckpts["a"], ckpts["b"]], tmp_path / "o.h5")
    printed = capsys.readouterr().out.splitlines()
    names = []
    with h5py.File(out) as f:
        names += list(f.attrs)
        f.visititems(lambda n, o: names.extend([n, *o.attrs]))
    fixed = set(ens.FIXED_SENTENCES)
    labels = [line for line in printed
              if line and line.removeprefix("Note: ") not in fixed
              and not line.startswith(("Data:", "Model:", "Saved:", "  Model "))]
    assert banned_in(names) == []
    assert banned_in(labels) == []
    for sentence in ens.FIXED_SENTENCES:
        assert f"Note: {sentence}" in printed


@pytest.mark.parametrize("planted", ["ensemble_uncertainty", "between_run_confidence",
                                     "spread_error", "snr_gain", "stability_score"])
def test_the_word_check_catches_a_planted_name(planted):
    assert banned_in([planted]) != []


# ------------------------------------------------------------- 9. single-seed path unchanged


def test_a_single_seed_run_records_what_it_did_before(monkeypatch, tmp_path, files):
    ck = train(monkeypatch, files["sup"], tmp_path / "m.pt", "--seed", "4")
    m = ck[prov.CHECKPOINT_MANIFEST]
    assert list(m["command"]["arguments"]) == [
        "arch", "method", "window", "noise_level", "epochs", "batch_size", "seed", "lr",
        "lr_drop_period", "lr_drop_factor", "scheduler", "warmup_epochs", "weight_decay",
        "grad_clip", "hidden_units", "encoder_dim", "device", "threads"]
    assert m["command"]["flags_passed"] == ["--batch-size", "--device", "--epochs", "--method",
                                            "--seed", "-d", "-o"]
    assert "seeds" in prov.EXCLUDED_ARGUMENTS


def test_diagnose_still_takes_one_model(monkeypatch, capsys, tmp_path, files, ckpts):
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, "diagnose", "-d", str(files["stack"]), "-m", str(ckpts["ma_a"]),
            "-m", str(ckpts["ma_b"]), "-o", str(tmp_path / "r.json"))
    assert exc.value.code in (1, 2)
