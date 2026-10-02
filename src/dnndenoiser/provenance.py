"""What a trained model records about how it was made: the provenance manifest.

Implements phase A of ``docs/design/PROVENANCE_MANIFEST.md`` (adopted 2026-10-02,
revision 3): the manifest ``train`` writes into a checkpoint, the two digests that bind
it to the weights, and the checks ``infer`` and ``evaluate`` make before they carry or
report it.

- The manifest holds only JSON types and is serialised canonically (sorted keys,
  separators ``,`` and ``:``, non-ASCII kept, UTF-8, no NaN or Infinity). A value the run
  produced that is not finite is stored as ``null`` and its dotted path is listed in
  ``statuses``.
- ``model_body_digest`` covers what ``infer`` applies: the state dict as stored and the
  configuration it rebuilds the network from. ``model_digest`` covers the body digest and
  the manifest, so neither can be edited alone.
- A record that is present is valid or refused; it is never read as absent.

A manifest records what was run. It does not prove where a model came from, and it does
not make training reproducible.
"""
from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dnndenoiser import digest as dg

SCHEMA = "dnd-provenance-1"
CHECKPOINT_MANIFEST = "provenance"
CHECKPOINT_DIGEST = "model_digest"
OUTPUT_MANIFEST = "model_provenance"          # dataset in an infer output
OUTPUT_DIGEST = "model_digest"                # attribute on denoised
OUTPUT_BODY_DIGEST = "model_body_digest"      # attribute on denoised
INPUT_ARRAY_DIGEST = "input_array_digest"     # attribute on the infer output's noisy
SIGNAL_IDENTITY = "signal_identity"           # attribute on clean (written by generate)

# Options of `train` recorded in `command.arguments`, by argparse destination. A test
# classifies every option of the parser, so one added later fails until it is placed in
# one of these two lists.
RECORDED_ARGUMENTS = (
    "arch", "method", "window", "noise_level", "epochs", "batch_size", "seed", "lr",
    "lr_drop_period", "lr_drop_factor", "scheduler", "warmup_epochs", "weight_decay",
    "grad_clip", "hidden_units", "encoder_dim", "device",
)
EXCLUDED_ARGUMENTS = ("data", "output")

TRAINING_COMPONENTS = ("noisy", "frames", "clean", "energy", "angles", "times", "frame_index")

MANIFEST_KEYS = {"schema", "created_utc", "software", "code", "command", "training_data",
                 "targets", "preprocessing", "result", "statuses"}
SOFTWARE_KEYS = {"dnndenoiser", "python", "numpy", "torch", "h5py", "torch_cuda",
                 "platform", "device"}
CODE_KEYS = {"commit", "tree_clean"}
COMMAND_KEYS = {"method", "arguments", "flags_passed", "effective", "seeds"}
TRAINING_DATA_KEYS = {"digest", "array_digests", "layout", "rows_used", "intensity_units",
                      "acquisition_id", "signal_identity", "frame_index_runs",
                      "reference_declaration"}
PREPROCESSING_KEYS = {"resampling", "normalisation"}
RESULT_KEYS = {"epochs", "final_loss"}
TARGET_KINDS = {"clean": {"kind"},
                "synthesised_realisation": {"kind", "noise_level"},
                "leave_one_out_window_mean": {"kind", "window"}}
_HEX64 = re.compile(r"[0-9a-f]{64}")


class MalformedProvenance(ValueError):
    """A provenance record that is present but not valid, or records that disagree."""


# ------------------------------------------------------------------------ serialisation


def canonical(obj: Any) -> str:
    """Step 1's JSON payload profile, strict: no NaN or Infinity."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def finite_or_null(value, path: str, statuses: dict):
    """A run-produced number as a JSON value: ``None`` with a status if not finite."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        statuses[path] = f"non-finite value ({value})"
        return None
    return value


def _check_plain(value, path: str) -> None:
    """Only JSON types, exactly: dict with str keys, list, str, int, float (finite), bool,
    None. Subclasses (a NumPy float64, torch's version string) are refused: the restricted
    checkpoint loader cannot read them back."""
    if value is None or type(value) in (bool, str, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise MalformedProvenance(f"'{path}' is not finite")
        return
    if type(value) is list:
        for i, item in enumerate(value):
            _check_plain(item, f"{path}[{i}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise MalformedProvenance(f"'{path}' has a non-string key {key!r}")
            _check_plain(item, f"{path}.{key}" if path else key)
        return
    raise MalformedProvenance(f"'{path}' holds a {type(value).__name__}, not a JSON value")


def _keys(obj, expected: set, path: str) -> None:
    if not isinstance(obj, dict):
        raise MalformedProvenance(f"'{path}' must be an object, got {type(obj).__name__}")
    if set(obj) != expected:
        missing, unknown = sorted(expected - set(obj)), sorted(set(obj) - expected)
        detail = "; ".join(x for x in (f"missing {missing}" if missing else "",
                                       f"unknown {unknown}" if unknown else "") if x)
        raise MalformedProvenance(f"'{path}' fields do not match {SCHEMA}: {detail}")


_ISO_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_METHODS = ("noise2clean", "noise2noise", "moving-average")
_EFFECTIVE_KEYS = {"architecture", "num_features", "num_hidden_units", "encoder_output_dim",
                   "optimiser", "scheduler", "loss", "grad_clip", "epochs", "batch_size"}


def _is_number(v) -> bool:
    return type(v) in (int, float)


def _require(ok: bool, path: str, what: str, value) -> None:
    if not ok:
        raise MalformedProvenance(f"'{path}' must be {what}, got {value!r}")


def _str(v, path, nullable=False):
    _require(type(v) is str or (nullable and v is None), path,
             "a string" + (" or null" if nullable else ""), v)


def _int(v, path, minimum=None, nullable=False):
    ok = (type(v) is int and (minimum is None or v >= minimum)) or (nullable and v is None)
    _require(ok, path, "an integer" + (f" >= {minimum}" if minimum is not None else "")
             + (" or null" if nullable else ""), v)


def _num(v, path, nullable=False):
    _require(_is_number(v) or (nullable and v is None), path,
             "a number" + (" or null" if nullable else ""), v)


def _validate_effective(eff, method: str) -> None:
    keys = _EFFECTIVE_KEYS | ({"window"} if method == "moving-average" else set())
    _keys(eff, keys, "command.effective")
    _str(eff["architecture"], "command.effective.architecture")
    for k in ("num_features", "num_hidden_units", "encoder_output_dim", "epochs", "batch_size"):
        _int(eff[k], f"command.effective.{k}", minimum=1)
    if "window" in eff:
        _int(eff["window"], "command.effective.window", minimum=1)
    _num(eff["grad_clip"], "command.effective.grad_clip")
    for name in ("optimiser", "scheduler", "loss"):
        value = eff[name]
        _require(type(value) is dict and type(value.get("name")) is str,
                 f"command.effective.{name}", "an object with a string 'name'", value)
        for k, v in value.items():
            if k != "name":
                _require(_is_number(v) or (type(v) is list and all(_is_number(x) for x in v)),
                         f"command.effective.{name}.{k}", "a number or a list of numbers", v)


def validate_manifest(obj) -> dict:
    """Check every field's presence and type (design §2); return it unchanged if valid.
    Anything else -- a missing or unknown field, a wrong type at any depth -- is refused,
    naming the field."""
    from dnndenoiser import reference as ref

    _keys(obj, MANIFEST_KEYS, "provenance")
    _check_plain(obj, "")
    if obj["schema"] != SCHEMA:
        raise MalformedProvenance(f"unknown provenance schema {obj['schema']!r}; this "
                                  f"version reads {SCHEMA}")
    _require(type(obj["created_utc"]) is str and bool(_ISO_UTC.fullmatch(obj["created_utc"])),
             "created_utc", "an ISO 8601 UTC time such as 2026-10-02T12:00:00Z",
             obj["created_utc"])

    sw = obj["software"]
    _keys(sw, SOFTWARE_KEYS, "software")
    for k in ("dnndenoiser", "python", "numpy", "torch", "h5py", "device"):
        _str(sw[k], f"software.{k}")
    _str(sw["torch_cuda"], "software.torch_cuda", nullable=True)
    _keys(sw["platform"], {"system", "machine"}, "software.platform")
    for k in ("system", "machine"):
        _str(sw["platform"][k], f"software.platform.{k}")

    code = obj["code"]
    _keys(code, CODE_KEYS, "code")
    known = (type(code["commit"]) is str and re.fullmatch(r"[0-9a-f]{40,64}", code["commit"])
             and type(code["tree_clean"]) is bool)
    unknown = code["commit"] == "unknown" and code["tree_clean"] == "unknown"
    _require(bool(known or unknown), "code",
             "a commit with a boolean tree_clean, or both 'unknown'", code)

    cmd = obj["command"]
    _keys(cmd, COMMAND_KEYS, "command")
    _require(cmd["method"] in _METHODS, "command.method", f"one of {list(_METHODS)}",
             cmd["method"])
    _keys(cmd["arguments"], set(RECORDED_ARGUMENTS), "command.arguments")
    for k, v in cmd["arguments"].items():
        _require(v is None or type(v) in (bool, int, float, str), f"command.arguments.{k}",
                 "a scalar", v)
    _require(type(cmd["flags_passed"]) is list and all(
        type(f) is str and f.startswith("-") for f in cmd["flags_passed"]),
        "command.flags_passed", "a list of option names", cmd["flags_passed"])
    _validate_effective(cmd["effective"], cmd["method"])
    _keys(cmd["seeds"], {"torch", "targets"}, "command.seeds")
    _int(cmd["seeds"]["torch"], "command.seeds.torch", nullable=True)
    targets_seeds = cmd["seeds"]["targets"]
    _require(targets_seeds is None or (type(targets_seeds) is list and len(targets_seeds) == 2
                                       and all(type(x) is int for x in targets_seeds)),
             "command.seeds.targets", "null or a list of two integers", targets_seeds)

    td = obj["training_data"]
    _keys(td, TRAINING_DATA_KEYS, "training_data")
    validate_digest_object(td["digest"], "training_data.digest")
    _require(type(td["array_digests"]) is dict
             and set(td["array_digests"]) <= {"noisy", "frames", "clean"},
             "training_data.array_digests", "an object keyed by noisy, frames or clean",
             td["array_digests"])
    for name, value in td["array_digests"].items():
        validate_digest_object(value, f"training_data.array_digests.{name}")
    _require(type(td["layout"]) is dict and set(td["layout"]) <= set(TRAINING_COMPONENTS),
             "training_data.layout", "an object keyed by stored components", td["layout"])
    for name, value in td["layout"].items():
        _keys(value, {"shape", "dtype"}, f"training_data.layout.{name}")
        _require(type(value["shape"]) is list and all(type(n) is int and n >= 0
                                                      for n in value["shape"]),
                 f"training_data.layout.{name}.shape", "a list of sizes", value["shape"])
        _str(value["dtype"], f"training_data.layout.{name}.dtype")
    _require(td["rows_used"] == "all", "training_data.rows_used", "'all'", td["rows_used"])
    for k in ("intensity_units", "acquisition_id", "signal_identity"):
        _str(td[k], f"training_data.{k}", nullable=True)
    runs = td["frame_index_runs"]
    ok = runs is None or (type(runs) is list and all(
        type(r) is list and len(r) == 2 and all(type(v) is int for v in r) and r[0] <= r[1]
        for r in runs) and all(runs[i][0] > runs[i - 1][1] + 1 for i in range(1, len(runs))))
    _require(ok, "training_data.frame_index_runs",
             "null or sorted, disjoint, non-adjacent [first, last] integer runs", runs)
    if td["reference_declaration"] is not None:
        try:
            ref.validate_origin(td["reference_declaration"])
        except ref.MalformedReference as exc:
            raise MalformedProvenance(f"'training_data.reference_declaration': {exc}") from None

    targets = obj["targets"]
    _require(type(targets) is dict and type(targets.get("kind")) is str
             and targets["kind"] in TARGET_KINDS, "targets.kind",
             f"one of {sorted(TARGET_KINDS)}", targets.get("kind") if type(targets) is dict
             else targets)
    _keys(targets, TARGET_KINDS[targets["kind"]], "targets")
    if "noise_level" in targets:
        _require(_is_number(targets["noise_level"]) and targets["noise_level"] > 0,
                 "targets.noise_level", "a positive number", targets["noise_level"])
    if "window" in targets:
        _int(targets["window"], "targets.window", minimum=1)

    pre = obj["preprocessing"]
    _keys(pre, PREPROCESSING_KEYS, "preprocessing")
    if pre["resampling"] is not None:
        _keys(pre["resampling"], {"from_points", "to_points"}, "preprocessing.resampling")
        for k in ("from_points", "to_points"):
            _int(pre["resampling"][k], f"preprocessing.resampling.{k}", minimum=1)
    if pre["normalisation"] is not None:
        _keys(pre["normalisation"], {"kind", "min", "max"}, "preprocessing.normalisation")
        _str(pre["normalisation"]["kind"], "preprocessing.normalisation.kind")
        for k in ("min", "max"):
            _num(pre["normalisation"][k], f"preprocessing.normalisation.{k}")

    res = obj["result"]
    _keys(res, RESULT_KEYS, "result")
    _int(res["epochs"], "result.epochs", minimum=1)
    _num(res["final_loss"], "result.final_loss", nullable=True)
    _require(type(obj["statuses"]) is dict and all(
        type(v) is str for v in obj["statuses"].values()),
        "statuses", "an object mapping field paths to strings", obj["statuses"])
    return obj


def validate_digest_object(obj, path: str = "digest") -> dict:
    _keys(obj, {"format", "sha256"}, path)
    if obj["format"] != dg.FORMAT or not (isinstance(obj["sha256"], str)
                                          and _HEX64.fullmatch(obj["sha256"])):
        raise MalformedProvenance(f"'{path}' must be {{'format': '{dg.FORMAT}', 'sha256': "
                                  f"<64 hex digits>}}, got {obj!r}")
    return obj


def _digest_object(components) -> dict:
    return {"format": dg.FORMAT, "sha256": dg.digest(components)["sha256"]}


# ------------------------------------------------------------------------ digests


def matching_digest(array) -> dict:
    """Name- and dtype-independent: one component named ``array``, the array cast to
    float32 in its stored shape -- what ``infer`` writes, whatever the input's name and
    dtype were."""
    arr = np.asarray(array).astype(np.float32)
    return _digest_object([("array", dg.array_payload(arr))])


def body_config(checkpoint: dict, resolved: dict) -> dict:
    """What ``infer`` uses to rebuild and apply the model, under canonical names."""
    return {
        "architecture": resolved["architecture"],
        "num_features": int(resolved["num_features"]),
        "num_hidden_units": int(resolved["num_hidden_units"]),
        "encoder_output_dim": int(resolved["encoder_output_dim"]),
        "training_method": checkpoint.get("training_method"),
        "normalisation": checkpoint.get("normalisation"),
    }


def body_digest(state_dict: dict, config: dict) -> dict:
    """Over the state dict as stored (every entry, buffers included, keys in code-point
    order) and the configuration ``infer`` applies."""
    import torch

    components = []
    bad = [k for k in state_dict if type(k) is not str]
    if bad:
        raise MalformedProvenance(f"state-dict key {bad[0]!r} is not a string")
    for key in sorted(state_dict):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise MalformedProvenance(f"state-dict entry '{key}' is a "
                                      f"{type(value).__name__}, not a tensor")
        try:
            array = value.detach().cpu().contiguous().numpy()
            payload = dg.array_payload(array)
        except (TypeError, RuntimeError) as exc:
            raise MalformedProvenance(f"state-dict entry '{key}' has no digest: "
                                      f"{exc}") from None
        components.append((key, payload))
    try:
        components.append(("config", canonical(config).encode("utf-8")))
    except ValueError as exc:
        raise MalformedProvenance(f"the model configuration is not strict JSON: "
                                  f"{exc}") from None
    return _digest_object(components)


def model_digest(body: dict, manifest_text: str) -> dict:
    """Over the body digest and the manifest's canonical serialisation."""
    return _digest_object([("body", canonical(body).encode("utf-8")),
                           ("provenance", manifest_text.encode("utf-8"))])


# ------------------------------------------------------------------------ the run


def frame_index_runs(frame_index) -> Optional[list]:
    """The set of values as sorted inclusive ``[first, last]`` runs of consecutive
    integers."""
    if frame_index is None:
        return None
    values = sorted({int(v) for v in np.asarray(frame_index).ravel().tolist()})
    runs: list = []
    for v in values:
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return runs


def _git(args, cwd) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                             encoding="utf-8", timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def code_record(package_dir: Optional[Path] = None) -> dict:
    """The commit of this package's own checkout, or ``"unknown"``.

    Recorded only when the package directory is inside a git work tree whose top level
    holds a tracked ``src/dnndenoiser/__init__.py`` resolving to that directory. A
    package installed into an ignored environment inside some other repository gets
    ``"unknown"``: that repository's commit is neither the code's nor public.
    """
    unknown = {"commit": "unknown", "tree_clean": "unknown"}
    if package_dir is None:
        import dnndenoiser
        package_dir = Path(dnndenoiser.__file__).parent
    package_dir = Path(package_dir).resolve()
    top = _git(["rev-parse", "--show-toplevel"], package_dir)
    if not top:
        return unknown
    top = Path(top)
    if (top / "src" / "dnndenoiser").resolve() != package_dir:
        return unknown
    if _git(["ls-files", "--error-unmatch", "src/dnndenoiser/__init__.py"], top) is None:
        return unknown
    commit = _git(["rev-parse", "HEAD"], top)
    tracked = _git(["status", "--porcelain", "--untracked-files=no"], top)
    untracked = _git(["ls-files", "--others", "--exclude-standard", "--", "src/dnndenoiser"],
                     top)
    if not commit or tracked is None or untracked is None:
        return unknown
    return {"commit": commit, "tree_clean": tracked == "" and untracked == ""}


def software_record(device: str) -> dict:
    import h5py
    import torch

    from dnndenoiser import __version__
    return {
        "dnndenoiser": str(__version__),
        "python": str(platform.python_version()),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "h5py": str(h5py.__version__),
        "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "platform": {"system": str(platform.system()), "machine": str(platform.machine())},
        "device": str(device),
    }


def arguments_record(args) -> dict:
    """The recorded options' values as parsed, as JSON values."""
    out = {}
    for name in RECORDED_ARGUMENTS:
        value = getattr(args, name, None)
        if isinstance(value, float) and not math.isfinite(value):
            raise MalformedProvenance(f"--{name.replace('_', '-')} is not finite")
        out[name] = value
    return out


def _attr_text(value, name: str = "attribute") -> Optional[str]:
    """A string attribute's text; anything else is refused, never stringified."""
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            raise MalformedProvenance(f"the {name} is not valid UTF-8") from None
    if isinstance(value, (str, np.str_)):
        return str(value)
    raise MalformedProvenance(f"the {name} must be a string, got {type(value).__name__} "
                              f"{value!r}")


def validate_training_frame_index(frame_index, n_rows: int, name: str) -> None:
    """Step 1's rules for a recorded ``frame_index``: one-dimensional, integer, one value
    per first-axis row."""
    arr = np.asarray(frame_index)
    if arr.ndim != 1 or arr.dtype.kind not in "iu":
        raise MalformedProvenance(f"'frame_index' of {name} must be a one-dimensional "
                                  f"integer array, got shape {arr.shape} and dtype "
                                  f"{arr.dtype}")
    if len(arr) != n_rows:
        raise MalformedProvenance(f"'frame_index' of {name} has {len(arr)} values but "
                                  f"its first axis has {n_rows} rows")


def training_data_record(path, input_name: str) -> dict:
    """§2.3: the training file's content, as stored, before anything is done to it."""
    import h5py

    from dnndenoiser import reference as ref
    with h5py.File(path, "r") as f:
        stored = {}
        for name in TRAINING_COMPONENTS:
            if name not in f:
                continue
            if not isinstance(f[name], h5py.Dataset):
                raise MalformedProvenance(f"the training file's '{name}' is not a dataset")
            arr = np.asarray(f[name][()])
            if arr.ndim < 1 or arr.dtype.kind not in "biuf":
                raise MalformedProvenance(f"the training '{name}' must be a numeric array, "
                                          f"got shape {arr.shape} and dtype {arr.dtype}")
            stored[name] = arr
        attrs = f[input_name].attrs
        units = _attr_text(attrs.get(ref.UNITS_ATTR), f"'{input_name}' {ref.UNITS_ATTR}")
        acquisition = _attr_text(attrs.get(ref.ACQUISITION_ATTR),
                                 f"'{input_name}' {ref.ACQUISITION_ATTR}")
        declaration = signal = None
        if "clean" in f:
            try:
                _, declaration = ref.read_declaration(f["clean"])
            except ref.MalformedReference as exc:
                raise MalformedProvenance(f"the training file's reference declaration is "
                                          f"malformed: {exc}") from None
            signal = _attr_text(f["clean"].attrs.get(SIGNAL_IDENTITY),
                                f"'clean' {SIGNAL_IDENTITY}")
    rows = stored[input_name].shape[0]
    for name, arr in stored.items():
        if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
            raise MalformedProvenance(f"the training '{name}' contains non-finite values")
    index = stored.get("frame_index")
    if index is not None:
        validate_training_frame_index(index, rows, "the training file")
    components = [(name, None if name not in stored else dg.array_payload(stored[name]))
                  for name in TRAINING_COMPONENTS]
    array_digests = {name: matching_digest(stored[name])
                     for name in ("noisy", "frames", "clean") if name in stored}
    return {
        "digest": _digest_object(components),
        "array_digests": array_digests,
        "layout": {name: {"shape": list(arr.shape), "dtype": arr.dtype.name}
                   for name, arr in stored.items()},
        "rows_used": "all",
        "intensity_units": units,
        "acquisition_id": acquisition,
        "signal_identity": signal,
        "frame_index_runs": frame_index_runs(index),
        "reference_declaration": declaration,
    }


def build_manifest(*, args, flags_passed, device, training_data: dict, targets: dict,
                   effective: dict, seeds: dict, preprocessing: dict, epochs: int,
                   final_loss) -> dict:
    statuses: dict = {}
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "software": software_record(device),
        "code": code_record(),
        "command": {"method": args.method, "arguments": arguments_record(args),
                    "flags_passed": sorted(flags_passed), "effective": effective,
                    "seeds": seeds},
        "training_data": training_data,
        "targets": targets,
        "preprocessing": preprocessing,
        "result": {"epochs": int(epochs),
                   "final_loss": finite_or_null(final_loss, "result.final_loss", statuses)},
        "statuses": statuses,
    }
    validate_manifest(manifest)
    canonical(manifest)
    return manifest


def seal(checkpoint: dict, manifest: dict, resolved: dict) -> None:
    """Add the manifest and ``model_digest`` to a checkpoint about to be saved."""
    body = body_digest(checkpoint["model_state_dict"], body_config(checkpoint, resolved))
    checkpoint[CHECKPOINT_MANIFEST] = manifest
    checkpoint[CHECKPOINT_DIGEST] = model_digest(body, canonical(manifest))


# ------------------------------------------------------------------------ reading


def checkpoint_records(checkpoint: dict) -> tuple[Optional[dict], Optional[dict]]:
    """``(manifest, model_digest)``, both ``None`` for a checkpoint written before
    manifests. One without the other, or either malformed, is refused."""
    has_m, has_d = CHECKPOINT_MANIFEST in checkpoint, CHECKPOINT_DIGEST in checkpoint
    if not has_m and not has_d:
        return None, None
    if has_m != has_d:
        present = CHECKPOINT_MANIFEST if has_m else CHECKPOINT_DIGEST
        raise MalformedProvenance(f"the checkpoint has '{present}' without its pair")
    return (validate_manifest(checkpoint[CHECKPOINT_MANIFEST]),
            validate_digest_object(checkpoint[CHECKPOINT_DIGEST], CHECKPOINT_DIGEST))


def verify_checkpoint(checkpoint: dict, resolved: dict) -> Optional[dict]:
    """The checkpoint's records, verified against its weights and configuration; ``None``
    for a checkpoint without them. Returns ``{"manifest_text", "model_digest", "body"}``."""
    manifest, stored = checkpoint_records(checkpoint)
    if manifest is None:
        return None
    text = canonical(manifest)
    body = body_digest(checkpoint["model_state_dict"], body_config(checkpoint, resolved))
    computed = model_digest(body, text)
    if computed["sha256"] != stored["sha256"]:
        raise MalformedProvenance(
            f"the checkpoint's model_digest {stored['sha256'][:16]}... does not match its "
            f"weights, configuration and manifest ({computed['sha256'][:16]}...): the "
            "manifest no longer describes what would be applied")
    return {"manifest_text": text, "model_digest": stored, "body": body}


def read_output_records(handle) -> Optional[dict]:
    """The model records of an ``infer`` output, verified with each other; ``None`` when
    all three are absent."""
    denoised = handle["denoised"].attrs
    present = {OUTPUT_MANIFEST: OUTPUT_MANIFEST in handle,
               OUTPUT_DIGEST: OUTPUT_DIGEST in denoised,
               OUTPUT_BODY_DIGEST: OUTPUT_BODY_DIGEST in denoised}
    if not any(present.values()):
        return None
    if not all(present.values()):
        missing = sorted(k for k, v in present.items() if not v)
        raise MalformedProvenance(f"the model records are incomplete: missing {missing}")
    import h5py
    node = handle[OUTPUT_MANIFEST]
    if not isinstance(node, h5py.Dataset) or node.shape != ():
        raise MalformedProvenance(f"'{OUTPUT_MANIFEST}' must be a scalar string dataset")
    text = _attr_text(node[()], f"'{OUTPUT_MANIFEST}' dataset")
    try:
        manifest = json.loads(text, parse_constant=_no_constant)
        digest_obj = json.loads(_attr_text(denoised[OUTPUT_DIGEST], OUTPUT_DIGEST))
        body = json.loads(_attr_text(denoised[OUTPUT_BODY_DIGEST], OUTPUT_BODY_DIGEST))
    except (ValueError, TypeError) as exc:
        raise MalformedProvenance(f"a model record is not valid JSON: {exc}") from None
    validate_manifest(manifest)
    validate_digest_object(digest_obj, OUTPUT_DIGEST)
    validate_digest_object(body, OUTPUT_BODY_DIGEST)
    if canonical(manifest) != text:
        raise MalformedProvenance("the stored model_provenance is not its own canonical "
                                  "serialisation")
    if model_digest(body, text)["sha256"] != digest_obj["sha256"]:
        raise MalformedProvenance(
            "model_digest does not match the stored model_provenance and model_body_digest: "
            "the manifest was edited, or the records of another model were attached")
    return {"manifest": manifest, "manifest_text": text, "model_digest": digest_obj,
            "body": body}


def _no_constant(name):
    raise ValueError(f"non-finite number {name}")


def model_identity(records: Optional[dict]):
    """``evaluation_context.model``: a summary of a verified manifest, or ``"unknown"``."""
    if records is None:
        return "unknown"
    m = records["manifest"]
    import hashlib
    return {
        "model_digest": records["model_digest"],
        "manifest_digest": hashlib.sha256(records["manifest_text"].encode("utf-8")).hexdigest(),
        "method": m["command"]["method"],
        "training_data": {k: m["training_data"][k]
                          for k in ("digest", "acquisition_id", "intensity_units")},
        "software": {"dnndenoiser": m["software"]["dnndenoiser"]},
        "code": dict(m["code"]),
    }


def flags_from_argv(known_options) -> list:
    """Option names given on the command line (no values)."""
    from dnndenoiser.cli import _flags_passed
    return sorted(_flags_passed(sys.argv[1:], known_options))
