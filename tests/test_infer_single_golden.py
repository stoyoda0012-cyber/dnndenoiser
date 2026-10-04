"""Single-model ``infer`` output, pinned before step 6 changed ``infer``
(docs/design/MULTI_SEED.md §6 test 7).

The golden is a content digest of every dataset and attribute of the output file, computed
with the code before the change. To hold on every platform: the inputs are built from integer
arithmetic only, nothing is resampled, and the network is replaced through the
``_build_network`` seam by an elementwise multiplication, so every value is a chain of
correctly rounded elementwise operations. The model records (which carry a creation time and
a commit) are excluded from the digest and verified instead.
"""
from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

from dnndenoiser import cli
from dnndenoiser import provenance as prov
from dnndenoiser import reference as ref
from dnndenoiser.data.frame_stack import write_frame_stack
from tests.test_evaluate_reference import TRUTH, run
from tests.test_provenance_manifest import train

RECORDS = {prov.OUTPUT_MANIFEST, f"denoised@{prov.OUTPUT_DIGEST}",
           f"denoised@{prov.OUTPUT_BODY_DIGEST}"}


class Diagonal(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = torch.tensor(0.5 + (np.arange(n) % 5) / 10.0, dtype=torch.float32)

    def forward(self, z):
        return z * self.w


def pattern(shape, scale):
    """Values from integer arithmetic only: 1 + ((7i + 13j + ...) mod 17) / scale."""
    idx = np.indices(shape)
    mix = sum((7 + 6 * k) * idx[k] for k in range(len(shape)))
    return 1.0 + (mix % 17) / scale


def content_digest(path) -> str:
    h = hashlib.sha256()

    def value_bytes(v):
        if isinstance(v, str):
            return b"s" + v.encode("utf-8")
        if isinstance(v, bytes):
            return b"b" + v
        a = np.asarray(v)
        return f"{a.dtype.str}{a.shape}".encode() + a.tobytes()

    with h5py.File(path, "r") as f:
        items = []
        f.visititems(lambda name, obj: items.append((name, obj)))
        for name, obj in [("", f)] + sorted(items, key=lambda t: t[0]):
            if isinstance(obj, h5py.Dataset) and name not in RECORDS:
                h.update(f"D{name}".encode() + value_bytes(obj[()]))
            for key in sorted(obj.attrs):
                if f"{name}@{key}" not in RECORDS:
                    h.update(f"A{name}@{key}".encode() + value_bytes(obj.attrs[key]))
    return h.hexdigest()


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("golden")
    try:
        sup = d / "sup.h5"
        with h5py.File(sup, "w") as f:
            f.create_dataset("noisy", data=pattern((8, 32), 50.0).astype(np.float32))
            f["noisy"].attrs["intensity_units"] = "counts"
            f["noisy"].attrs["acquisition_id"] = "acq-G"
            f.create_dataset("clean", data=pattern((8, 32), 60.0).astype(np.float32))
            ref.write_declaration(f["clean"], TRUTH)
            f.create_dataset("energy", data=np.arange(32, dtype=np.float64))
            f.create_dataset("frame_index", data=np.arange(8))
        train(mp, sup, d / "n2c.pt", "--seed", "1")
        stack = d / "stack.h5"
        write_frame_stack(stack, pattern((6, 2, 256), 40.0).astype(np.float32),
                          np.arange(256, dtype=np.float64), angles=np.array([10.0, 30.0]),
                          angle_kind="emission", angle_units="deg", order_basis="recorded")
        train(mp, stack, d / "ma.pt", "--seed", "1", method="moving-average")
    finally:
        mp.undo()
    return {"supervised": (sup, d / "n2c.pt"), "stack": (stack, d / "ma.pt")}


GOLDEN = {
    "supervised": "0ecb1ce2c8969cf37ca2e70b66fa0658bd254a24693c1bd062879c06e5ca967c",
    "stack": "75f419df2fa70d8cfeca35c09a3116c2c3481ce3c2fb44a4611ed930562f89f4",
}


@pytest.mark.parametrize("case", ["supervised", "stack"])
def test_single_model_infer_output_is_unchanged(monkeypatch, tmp_path, models, case):
    data, model = models[case]
    n = 32 if case == "supervised" else 256
    monkeypatch.setattr(cli, "_build_network", lambda checkpoint, config, device: Diagonal(n))
    out = tmp_path / "out.h5"
    run(monkeypatch, "infer", "-d", str(data), "-m", str(model), "-o", str(out),
        "--device", "cpu")
    with h5py.File(out, "r") as f:
        assert prov.read_output_records(f) is not None       # the excluded records verify
    assert content_digest(out) == GOLDEN[case]


def test_the_content_digest_sees_a_changed_value_and_a_changed_attribute(monkeypatch, tmp_path,
                                                                        models):
    data, model = models["supervised"]
    monkeypatch.setattr(cli, "_build_network", lambda checkpoint, config, device: Diagonal(32))
    out = tmp_path / "out.h5"
    run(monkeypatch, "infer", "-d", str(data), "-m", str(model), "-o", str(out),
        "--device", "cpu")
    before = content_digest(out)
    with h5py.File(out, "a") as f:
        f["denoised"][0, 0] += 1e-3
    changed_value = content_digest(out)
    with h5py.File(out, "a") as f:
        f["denoised"][0, 0] -= 1e-3
        f["noisy"].attrs["intensity_units"] = "counts_per_s"
    changed_attr = content_digest(out)
    assert len({before, changed_value, changed_attr}) == 3
    assert json.dumps(GOLDEN)        # the golden is plain data
