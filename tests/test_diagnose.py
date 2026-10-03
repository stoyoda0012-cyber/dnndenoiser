"""How the output depends on the input: ``diagnose`` (docs/design/OUTPUT_CONTRACTION.md §8).

Expected values are computed here independently of ``dnndenoiser.diagnostic``: the
controls of §5 have answers written down from their definitions (matrix products, the
square of a known array), σ is recomputed with a loop over adjacent frame indices, and the
grid test uses a stack whose resampling returns known points. Every refusal has a positive
counterpart.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from dnndenoiser import cli
from dnndenoiser import diagnostic as dx
from dnndenoiser import evaluation as ev
from dnndenoiser.data.frame_stack import write_frame_stack
from tests.test_evaluate_reference import (estimate, evaluate, forbidden_in, load_strict, refuse,
                                           run, write)
from tests.test_provenance_manifest import train

L = 128
ENERGY = np.linspace(280.0, 290.0, L)


def frames2(n=12, e=L, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    signal = 1.0 + 0.5 * np.exp(-0.5 * ((np.arange(e) - e / 2) / (e / 12)) ** 2)
    return signal + rng.normal(0, noise, (n, e))


def band_smoother(n):
    """[1/4, 1/2, 1/4]; at each end the in-range weights renormalised to sum to 1."""
    S = np.zeros((n, n))
    for i in range(n):
        for j, w in ((i - 1, 0.25), (i, 0.5), (i + 1, 0.25)):
            if 0 <= j < n:
                S[i, j] = w
        S[i] /= S[i].sum()
    return S


def gauss(energy, e0, fwhm):
    return np.exp(-4.0 * np.log(2.0) * (energy - e0) ** 2 / fwhm ** 2)


def sigma_by_loop(frames, frame_index):
    """Per channel: median over points of std(adjacent differences)/sqrt(2)."""
    x = frames if frames.ndim == 3 else frames[:, None, :]
    order = sorted(range(len(frame_index)), key=lambda i: frame_index[i])
    diffs = [x[b] - x[a] for a, b in zip(order, order[1:])
             if frame_index[b] - frame_index[a] == 1]
    d = np.stack(diffs)
    return np.array([np.median(d[:, c].std(axis=0) / np.sqrt(2.0)) for c in range(x.shape[1])])


def probe(report, i=0, c=0):
    return report["probes"][i]["channels"][c]


# ------------------------------------------------------------------ 1. controls (§5)


def test_identity_returns_one_for_every_quantity():
    x = frames2()
    r = dx.diagnose(lambda z: z, x, ENERGY, np.arange(len(x)))
    assert r["contraction_ratio"]["channels"] == [pytest.approx(1.0, abs=1e-12)]
    assert len(r["probes"]) == 5
    for i in range(5):
        p = probe(r, i)
        assert p["response_median"] == pytest.approx(1.0, abs=1e-12)
        assert p["area_median"] == pytest.approx(1.0, abs=1e-12)
        assert p["response_iqr"] == [pytest.approx(1.0, abs=1e-12)] * 2


def test_a_constant_returns_zero():
    x = frames2()
    c = x.mean(axis=0)
    r = dx.diagnose(lambda z: np.broadcast_to(c, z.shape).copy(), x, ENERGY, np.arange(len(x)))
    assert r["contraction_ratio"]["channels"][0] == pytest.approx(0.0, abs=1e-12)
    for i in range(5):
        assert probe(r, i)["response_median"] == pytest.approx(0.0, abs=1e-12)
        assert probe(r, i)["area_median"] == pytest.approx(0.0, abs=1e-12)


def test_a_linear_smoother_gives_its_matrix_products():
    x = frames2()
    S = band_smoother(L)
    r = dx.diagnose(lambda z: z @ S.T, x, ENERGY, np.arange(len(x)))
    for i, (e0, fwhm, _k) in enumerate(dx.default_probes(ENERGY)):
        g = gauss(ENERGY, e0, fwhm)
        assert probe(r, i)["response_median"] == pytest.approx(g @ S @ g / (g @ g), abs=1e-12)
        assert probe(r, i)["area_median"] == pytest.approx(S.sum(0) @ g / g.sum(), abs=1e-12)
    xc = x - x.mean(0)
    expected_c = ((xc @ S.T) ** 2).sum() / (xc ** 2).sum()
    assert r["contraction_ratio"]["channels"][0] == pytest.approx(expected_c, abs=1e-12)


def test_c_does_not_separate_a_smoother_from_a_matched_shrink_and_r_does():
    x = frames2()
    S = band_smoother(L)
    xc = x - x.mean(0)
    lam = np.sqrt(((xc @ S.T) ** 2).sum() / (xc ** 2).sum())
    mean = x.mean(0)
    smooth = dx.diagnose(lambda z: z @ S.T, x, ENERGY, np.arange(len(x)))
    shrink = dx.diagnose(lambda z: mean + lam * (z - mean), x, ENERGY, np.arange(len(x)))
    assert (smooth["contraction_ratio"]["channels"][0]
            == pytest.approx(shrink["contraction_ratio"]["channels"][0], abs=1e-12))
    for i in range(5):
        assert probe(shrink, i)["response_median"] == pytest.approx(lam, abs=1e-12)
        assert abs(probe(smooth, i)["response_median"] - lam) > 0.2


def test_the_mirror_has_c_of_one_and_r_of_minus_one():
    x = frames2()
    mean = x.mean(0)
    r = dx.diagnose(lambda z: 2 * mean - z, x, ENERGY, np.arange(len(x)))
    assert r["contraction_ratio"]["channels"][0] == pytest.approx(1.0, abs=1e-12)
    assert probe(r)["response_median"] == pytest.approx(-1.0, abs=1e-12)
    assert probe(r)["area_median"] == pytest.approx(-1.0, abs=1e-12)


def test_a_nonlinear_stub_pins_the_probe_amplitude():
    """f(z) = z**2: the secant R depends on a, so a probe added at another scale fails."""
    x = frames2()
    sigma = sigma_by_loop(x, np.arange(len(x)))[0]
    e0, fwhm, k = dx.default_probes(ENERGY)[2]
    r = dx.diagnose(lambda z: z ** 2, x, ENERGY, np.arange(len(x)))
    g = gauss(ENERGY, e0, fwhm)
    a = k * sigma
    per_frame = [((xi + a * g) ** 2 - xi ** 2) @ g / (a * (g @ g)) for xi in x]
    assert probe(r, 2)["response_median"] == pytest.approx(np.median(per_frame), abs=1e-12)
    assert probe(r, 2)["amplitude"] == pytest.approx(a, abs=1e-12)
    # The wrong amplitude would be caught: the secant at 2a differs by more than tolerance.
    at_2a = [((xi + 2 * a * g) ** 2 - xi ** 2) @ g / (2 * a * (g @ g)) for xi in x]
    assert abs(np.median(at_2a) - np.median(per_frame)) > 1e-3


def test_the_spread_is_the_quartiles_over_frames():
    """Under f(z) = z**2 the per-frame R and A differ, so the quartiles are pinned."""
    x = frames2(n=11, seed=15)
    sigma = sigma_by_loop(x, np.arange(11))[0]
    e0, fwhm, k = dx.default_probes(ENERGY)[2]
    g = gauss(ENERGY, e0, fwhm)
    a = k * sigma
    change = [(xi + a * g) ** 2 - xi ** 2 for xi in x]
    resp = [d @ g / (a * (g @ g)) for d in change]
    area = [d.sum() / (a * g.sum()) for d in change]
    p = probe(dx.diagnose(lambda z: z ** 2, x, ENERGY, np.arange(11)), 2)
    assert p["response_iqr"] == [pytest.approx(np.quantile(resp, q), abs=1e-12) for q in (0.25, 0.75)]
    assert p["area_iqr"] == [pytest.approx(np.quantile(area, q), abs=1e-12) for q in (0.25, 0.75)]
    assert p["area_median"] == pytest.approx(np.median(area), abs=1e-12)
    assert p["response_iqr"][0] < p["response_median"] < p["response_iqr"][1]


def test_the_pooled_summary_is_over_every_frame_and_channel():
    """An asymmetric pool (channel 0 squared, channel 1 halved), so a mean or a per-channel
    summary differs from the pooled median and quartiles."""
    rng = np.random.default_rng(16)
    x = 1.0 + rng.normal(0, 0.05, (9, 2, L))

    def f(z):
        out = z.copy()
        out[:, 0] = z[:, 0] ** 2
        out[:, 1] = 0.5 * z[:, 1]
        return out
    sig = sigma_by_loop(x, np.arange(9))
    e0, fwhm, k = dx.default_probes(ENERGY)[0]
    g = gauss(ENERGY, e0, fwhm)
    resp, area = [], []
    for c in range(2):
        a = k * sig[c]
        for xi in x[:, c]:
            d = (f(np.stack([xi + a * g] * 2)[None])[0, c] - f(np.stack([xi] * 2)[None])[0, c])
            resp.append(d @ g / (a * (g @ g)))
            area.append(d.sum() / (a * g.sum()))
    pooled = dx.diagnose(f, x, ENERGY, np.arange(9))["probes"][0]["pooled"]
    assert pooled["response_median"] == pytest.approx(np.median(resp), abs=1e-12)
    assert pooled["response_iqr"] == [pytest.approx(np.quantile(resp, q), abs=1e-12)
                                      for q in (0.25, 0.75)]
    assert pooled["area_median"] == pytest.approx(np.median(area), abs=1e-12)
    assert pooled["area_iqr"] == [pytest.approx(np.quantile(area, q), abs=1e-12)
                                  for q in (0.25, 0.75)]
    assert abs(np.mean(resp) - np.median(resp)) > 1e-3


def test_f_is_called_once_on_the_frames_and_once_per_probe_with_their_shape():
    x = frames2(n=6)
    shapes = []

    def f(z):
        shapes.append(z.shape)
        return z
    dx.diagnose(f, x, ENERGY, np.arange(6))
    assert shapes == [(6, L)] * 6


# ------------------------------------------------------------- 2. evaluate's claims (§5)


def test_against_the_frame_mean_the_constant_scores_best(monkeypatch, tmp_path):
    x = frames2(n=8, e=64).astype(np.float32)
    mean = np.repeat(x.mean(0, keepdims=True), len(x), 0)
    S = band_smoother(64)
    outputs = {"constant": mean, "smoother": (x.astype(np.float64) @ S.T), "identity": x}
    mse = {}
    for name, denoised in outputs.items():
        path = write(tmp_path / f"{name}.h5", x, denoised, mean,
                     declaration=estimate(acquisition="acq-A"), acquisition_id="acq-A",
                     frame_index=np.arange(len(x)))
        m = evaluate(monkeypatch, tmp_path, path)
        assert ev.SAME_FRAMES_CAVEAT in m["evaluation_context"]["caveats"]
        mse[name] = m["mse_out_mean"]
    assert mse["constant"] == 0.0
    assert mse["constant"] < mse["smoother"] < mse["identity"]


# ------------------------------------------------------- 3 and 4. through the CLI, the grid


class Diagonal(torch.nn.Module):
    """A stand-in network: multiplies its (normalised) input by a fixed vector."""

    def __init__(self, w):
        super().__init__()
        self.w = torch.tensor(w, dtype=torch.float32)

    def forward(self, z):
        return z * self.w


@pytest.fixture(scope="module")
def ma_model(tmp_path_factory):
    """A moving-average model trained on a 256-point stack with an acquisition id."""
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("diag")
    path = stack(d / "train.h5", frames2(n=10, e=256, seed=1), acquisition_id="acq-T")
    try:
        train(mp, path, d / "m.pt", method="moving-average")
    finally:
        mp.undo()
    return d / "m.pt", path


def stack(path, frames, energy=None, frame_index=None, acquisition_id=None, **kw):
    e = frames.shape[-1]
    write_frame_stack(path, frames.astype(np.float32),
                      np.linspace(280.0, 290.0, e) if energy is None else energy,
                      frame_index=frame_index, **kw)
    if acquisition_id is not None:
        with h5py.File(path, "a") as f:
            f["frames"].attrs["acquisition_id"] = acquisition_id
    return path


def diagnose(monkeypatch, tmp_path, data, model, *extra, name="r.json"):
    out = tmp_path / name
    run(monkeypatch, "diagnose", "-d", str(data), "-m", str(model), "-o", str(out), *extra)
    return load_strict(out)


def stub_network(monkeypatch, w):
    monkeypatch.setattr(cli, "_build_network", lambda checkpoint, config, device: Diagonal(w))


def test_the_cli_applies_normalisation_and_its_inverse(monkeypatch, tmp_path, ma_model):
    model, _ = ma_model
    w = np.linspace(0.5, 1.5, 256)
    stub_network(monkeypatch, w)
    x = frames2(n=10, e=256, seed=4)
    r = diagnose(monkeypatch, tmp_path, stack(tmp_path / "s.h5", x), model)
    x32 = x.astype(np.float32).astype(np.float64)
    energy = np.linspace(280.0, 290.0, 256)
    norm = torch.load(model, map_location="cpu", weights_only=True)["normalisation"]
    span = norm["max"] - norm["min"]
    xc = x32 - x32.mean(0)
    expected_c = ((xc * w) ** 2).sum() / (xc ** 2).sum()
    eps = np.finfo(np.float32).eps
    # The network path is float32: each output point carries a few eps32 of relative
    # rounding, so a ratio of sums over 256 points agrees to ~1e-5 relative (≈ 100 eps32).
    assert r["contraction_ratio"]["channels"][0] == pytest.approx(expected_c, rel=1e-5)
    sigma = r["sigma"][0]
    for i, (e0, fwhm, k) in enumerate(dx.default_probes(energy)):
        g = gauss(energy, e0, fwhm)
        tol = 64 * eps * (span + np.abs(x32).max()) / abs(k * sigma)
        assert probe(r, i)["response_median"] == pytest.approx(w @ (g * g) / (g @ g), abs=tol)
        assert probe(r, i)["area_median"] == pytest.approx(w @ g / g.sum(), abs=tol)


@pytest.mark.parametrize("dtype, n_points", [
    (np.float32, 256), (np.float64, 256), (np.int32, 256), (np.float64, 300), (np.float32, 300),
    (np.float16, 256)])
def test_the_model_is_applied_bit_for_bit_as_infer_applies_it(monkeypatch, tmp_path, ma_model,
                                                               dtype, n_points):
    """Large values, where normalising in another dtype than infer's would show."""
    model, _ = ma_model
    scale = (1e3, 1e4) if dtype == np.float16 else (1000, 1e5)     # float16 tops out at 65504
    x = (frames2(n=6, e=n_points, seed=20) * scale[0] + scale[1]).astype(dtype)
    path = tmp_path / "s.h5"
    write_frame_stack(path, x, np.linspace(280.0, 290.0, n_points))
    run(monkeypatch, "infer", "-d", str(path), "-m", str(model), "-o", str(tmp_path / "o.h5"),
        "--device", "cpu")
    with h5py.File(tmp_path / "o.h5") as f:
        denoised = f["denoised"][:].astype(np.float64)
    captured = {}
    real = dx.diagnose

    def spy(f, frames, energy, frame_index, probes=None):
        captured["y"] = np.asarray(f(np.asarray(frames, dtype=np.float64)))
        return real(f, frames, energy, frame_index, probes=probes)
    monkeypatch.setattr(dx, "diagnose", spy)
    diagnose(monkeypatch, tmp_path, path, model)
    assert np.array_equal(captured["y"], denoised)


def test_a_probe_on_an_integer_stack_is_not_truncated(monkeypatch, tmp_path, ma_model):
    """Small counts (σ near 0.5): the probe added to integer frames must stay fractional, as
    infer would normalise them in float64."""
    model, _ = ma_model
    w = np.linspace(0.5, 1.5, 256)
    stub_network(monkeypatch, w)
    counts = np.random.default_rng(22).poisson(0.5, (12, 256)).astype(np.int32)
    r = diagnose(monkeypatch, tmp_path, _int_stack(tmp_path / "i.h5", counts), model)
    energy = np.linspace(280.0, 290.0, 256)
    e0, fwhm, _k = dx.default_probes(energy)[2]
    g = gauss(energy, e0, fwhm)
    assert probe(r, 2)["response_median"] == pytest.approx(w @ (g * g) / (g @ g), abs=1e-4)


def _int_stack(path, counts):
    write_frame_stack(path, counts, np.linspace(280.0, 290.0, counts.shape[-1]))
    return path


class Square(torch.nn.Module):
    """Nonlinear in the normalised space, so a missing normalisation (or inverse) shows:
    for a linear stand-in an affine normalisation cancels."""

    def forward(self, z):
        return z * z


def test_the_cli_feeds_the_network_normalised_input(monkeypatch, tmp_path, ma_model):
    model, _ = ma_model
    monkeypatch.setattr(cli, "_build_network", lambda checkpoint, config, device: Square())
    x = frames2(n=10, e=256, seed=14)
    r = diagnose(monkeypatch, tmp_path, stack(tmp_path / "s.h5", x), model)
    x32 = x.astype(np.float32).astype(np.float64)
    norm = torch.load(model, map_location="cpu", weights_only=True)["normalisation"]
    lo, span = norm["min"], norm["max"] - norm["min"]

    def through(z):
        return span * ((z - lo) / span) ** 2 + lo
    y = through(x32)
    yc, xc = y - y.mean(0), x32 - x32.mean(0)
    assert r["contraction_ratio"]["channels"][0] == pytest.approx((yc ** 2).sum() / (xc ** 2).sum(),
                                                                  rel=1e-4)
    energy = np.linspace(280.0, 290.0, 256)
    e0, fwhm, k = dx.default_probes(energy)[2]
    g = gauss(energy, e0, fwhm)
    a = k * r["sigma"][0]
    per_frame = [(through(xi + a * g) - through(xi)) @ g / (a * (g @ g)) for xi in x32]
    # R is a difference of two float32 outputs over an amplitude of 3σ: rounding of order
    # eps32·|y| per point, divided by a, allows ~1e-3 relative; dropping the normalisation
    # changes R by far more (it is caught).
    assert probe(r, 2)["response_median"] == pytest.approx(np.median(per_frame), rel=1e-3)


def test_the_defaults_give_five_probes_and_no_average(monkeypatch, tmp_path, ma_model):
    model, data = ma_model
    r = diagnose(monkeypatch, tmp_path, data, model)
    assert [p["k"] for p in r["probes"]] == [3.0] * 5
    assert [p["E0"] for p in r["probes"]] == pytest.approx(
        [280.0 + q * 10.0 for q in (0.1, 0.3, 0.5, 0.7, 0.9)])
    assert all(p["fwhm"] == pytest.approx(0.3) for p in r["probes"])
    assert not any("mean" in k or "average" in k for p in r["probes"] for k in p)


def test_the_report_records_device_and_order_basis(monkeypatch, tmp_path, ma_model):
    model, _ = ma_model
    path = stack(tmp_path / "s.h5", frames2(n=6, e=256, seed=19), order_basis="recorded")
    r = diagnose(monkeypatch, tmp_path, path, model)
    assert (r["device"], r["order_basis"]) == ("cpu", "recorded")
    unknown = stack(tmp_path / "u.h5", frames2(n=6, e=256, seed=19))
    assert diagnose(monkeypatch, tmp_path, unknown, model, name="u.json")["order_basis"] == "unknown"


def test_batch_size_bounds_each_forward_call(monkeypatch, tmp_path, ma_model):
    model, data = ma_model
    seen = []

    class Recording(torch.nn.Module):
        def forward(self, z):
            seen.append(z.shape[0])
            return z
    monkeypatch.setattr(cli, "_build_network", lambda checkpoint, config, device: Recording())
    diagnose(monkeypatch, tmp_path, data, model, "--batch-size", "3")
    assert seen and max(seen) == 3 and sum(seen) == 10 * 6     # 10 frames, 1 + 5 passes


def test_two_runs_give_equal_reports(monkeypatch, tmp_path, ma_model):
    model, data = ma_model
    a = diagnose(monkeypatch, tmp_path, data, model, name="a.json")
    b = diagnose(monkeypatch, tmp_path, data, model, name="b.json")
    assert a == b


def test_a_stack_off_the_network_grid_is_resampled_first(monkeypatch, tmp_path, ma_model):
    """511 points made by interleaving midpoints: resampling to 256 returns the originals,
    which are therefore the independent truth for σ and C; the probe is sampled on the
    resampled axis."""
    model, _ = ma_model
    w = np.linspace(0.5, 1.5, 256)
    stub_network(monkeypatch, w)
    x = frames2(n=10, e=256, seed=5).astype(np.float32).astype(np.float64)
    z = np.empty((10, 511))
    z[:, 0::2] = x
    z[:, 1::2] = (x[:, :-1] + x[:, 1:]) / 2
    r = diagnose(monkeypatch, tmp_path, stack(tmp_path / "s.h5", z), model)
    # Resampling is a float32 matrix product whose weights at the original points are
    # 1 and 0 only to float32 rounding; 1e-4 relative allows that and the float32 network.
    assert r["resampled"] == {"from_points": 511, "to_points": 256}
    assert r["sigma"][0] == pytest.approx(sigma_by_loop(x, np.arange(10))[0], rel=1e-4)
    xc = x - x.mean(0)
    assert r["contraction_ratio"]["channels"][0] == pytest.approx(
        ((xc * w) ** 2).sum() / (xc ** 2).sum(), rel=1e-4)
    energy = np.linspace(280.0, 290.0, 256)
    e0, fwhm, _k = dx.default_probes(energy)[1]
    g = gauss(energy, e0, fwhm)
    assert probe(r, 1)["response_median"] == pytest.approx(w @ (g * g) / (g @ g), abs=1e-4)


def test_the_digest_is_of_the_frames_as_stored(monkeypatch, tmp_path, ma_model):
    from dnndenoiser import provenance as prov
    model, _ = ma_model
    z = frames2(n=6, e=300, seed=6)
    path = stack(tmp_path / "s.h5", z)
    r = diagnose(monkeypatch, tmp_path, path, model)
    assert r["input_digest"] == prov.matching_digest(z.astype(np.float32))


# ------------------------------------------------------------------------------- 5. σ


def test_sigma_follows_frame_index_not_row_order():
    x = frames2(n=12, seed=7)
    perm = np.random.default_rng(0).permutation(12)
    shuffled, index = x[perm], np.arange(12)[perm]
    assert dx.noise_scale(shuffled, index)[0] == pytest.approx(
        sigma_by_loop(x, np.arange(12))[0], abs=1e-12)
    assert dx.noise_scale(shuffled, index)[0] != pytest.approx(
        sigma_by_loop(shuffled, np.arange(12))[0], abs=1e-6)


def test_a_gap_between_runs_is_not_differenced():
    x = frames2(n=12, seed=8)
    x[6:] += 1.0                                   # a step between the two runs
    index = np.r_[np.arange(6), np.arange(20, 26)]
    s = dx.noise_scale(x, index)[0]
    assert s == pytest.approx(sigma_by_loop(x, index)[0], abs=1e-12)
    assert s == pytest.approx(0.05, rel=0.3)


def test_slow_drift_does_not_inflate_sigma():
    x = frames2(n=40, seed=9) + 0.05 * np.arange(40)[:, None]
    s = dx.noise_scale(x, np.arange(40))[0]
    assert s == pytest.approx(sigma_by_loop(x, np.arange(40))[0], abs=1e-12)
    assert s == pytest.approx(0.05, rel=0.2)
    assert np.median(x.std(axis=0)) > 5 * s


@pytest.mark.parametrize("frames, index, reason", [
    (np.ones((4, L)), np.arange(4), "no measurable noise"),
    (frames2(n=4), np.array([0, 2, 4, 6]), "fewer than two pairs of adjacent frame indices"),
    (frames2(n=4), np.array([0, 1, 5, 9]), "fewer than two pairs of adjacent frame indices"),
])
def test_sigma_refuses_what_it_cannot_measure(frames, index, reason):
    with pytest.raises(dx.DiagnoseError, match=reason):
        dx.noise_scale(frames, index)


def test_sigma_is_refused_below_a_floor_relative_to_the_frames():
    rng = np.random.default_rng(17)
    noise = rng.normal(0, 1, (6, L))
    with pytest.raises(dx.DiagnoseError, match="no measurable noise"):
        dx.noise_scale(1e3 + 1e-10 * noise, np.arange(6))      # σ ~ 1e-10 < 1e-12 · 1e3
    assert dx.noise_scale(1e3 + 1e-8 * noise, np.arange(6))[0] > 1e-9


def test_sigma_accepts_two_adjacent_pairs():
    assert dx.noise_scale(frames2(n=4), np.array([0, 1, 2, 9]))[0] > 0


# ------------------------------------------------------------------------- 6. channels


def test_channels_are_independent_and_the_overall_c_is_a_ratio_of_sums():
    rng = np.random.default_rng(10)
    x = 1.0 + rng.normal(0, 0.05, (8, 2, L))
    x[:, 1] *= 3.0
    mean = x.mean(0)

    def f(z):
        out = z.copy()
        out[:, 1] = mean[1]
        return out
    r = dx.diagnose(f, x, ENERGY, np.arange(8))
    c = r["contraction_ratio"]
    assert c["channels"] == [pytest.approx(1.0, abs=1e-12), pytest.approx(0.0, abs=1e-12)]
    d0 = ((x[:, 0] - mean[0]) ** 2).sum()
    d1 = ((x[:, 1] - mean[1]) ** 2).sum()
    assert c["overall"] == pytest.approx(d0 / (d0 + d1), abs=1e-12)
    sig = sigma_by_loop(x, np.arange(8))
    assert r["sigma"] == [pytest.approx(s, abs=1e-12) for s in sig]
    p = r["probes"][0]
    assert [ch["amplitude"] for ch in p["channels"]] == [pytest.approx(3 * s, abs=1e-12) for s in sig]
    assert [ch["response_median"] for ch in p["channels"]] == [
        pytest.approx(1.0, abs=1e-12), pytest.approx(0.0, abs=1e-12)]
    assert p["pooled"]["response_median"] == pytest.approx(
        np.median(np.r_[np.ones(8), np.zeros(8)]), abs=1e-12)


def test_a_two_d_stack_has_no_pooled_summary():
    r = dx.diagnose(lambda z: z, frames2(), ENERGY, np.arange(12))
    assert all(p["pooled"] is None for p in r["probes"])


# ------------------------------------------------------------------------- 7. refusals


@pytest.mark.parametrize("make, reason", [
    (lambda x: x[:0], "at least two frames"),
    (lambda x: x[:1], "at least two frames"),
    (lambda x: np.where(np.arange(L) == 3, np.nan, x), "non-finite"),
    (lambda x: np.repeat(x[:1], len(x), 0), "identical frames"),
])
def test_diagnose_refuses_unusable_frames(make, reason):
    x = make(frames2())
    with pytest.raises(dx.DiagnoseError, match=reason):
        dx.diagnose(lambda z: z, x, ENERGY, np.arange(len(x)))


def test_diagnose_refuses_a_non_finite_output():
    x = frames2()
    calls = []

    def f(z):
        calls.append(1)
        return z if len(calls) == 1 else np.full_like(z, np.inf)
    with pytest.raises(dx.DiagnoseError, match="non-finite"):
        dx.diagnose(f, x, ENERGY, np.arange(len(x)))


@pytest.mark.parametrize("probe_, reason", [
    ((280.2, 0.3, 3.0), "closer than 2·FWHM"),
    ((289.9, 0.3, 3.0), "closer than 2·FWHM"),
    ((300.0, 0.3, 3.0), "outside the energy axis"),
    ((285.0, 0.1, 3.0), "below twice the grid spacing"),
    ((285.0, 0.0, 3.0), "FWHM must be positive"),
    ((285.0, -1.0, 3.0), "FWHM must be positive"),
    ((285.0, 0.3, 0.0), "k must be non-zero"),
    ((285.0, float("nan"), 3.0), "finite"),
])
def test_a_probe_is_refused_with_its_reason(probe_, reason):
    with pytest.raises(dx.DiagnoseError, match=reason):
        dx.diagnose(lambda z: z, frames2(), ENERGY, np.arange(12), probes=[probe_])


@pytest.mark.parametrize("e0, fwhm, refused", [
    (280.45, 0.3, True), (280.65, 0.3, False),     # 2·FWHM = 0.6 from the low end
    (281.0, 0.5, False), (289.0, 0.5, False),      # exactly 2·FWHM: not closer, accepted
    (280.99, 0.5, True), (289.01, 0.5, True),
])
def test_the_end_rule_is_two_fwhm(e0, fwhm, refused):
    def go():
        return dx.diagnose(lambda z: z, frames2(), ENERGY, np.arange(12), probes=[(e0, fwhm, 3.0)])
    if refused:
        with pytest.raises(dx.DiagnoseError, match="closer than 2·FWHM"):
            go()
    else:
        assert len(go()["probes"]) == 1


def test_a_non_finite_response_is_refused():
    calls = []

    def f(z):
        calls.append(1)
        return z if len(calls) == 1 else z + 1e308        # finite output, overflowing sum
    with pytest.raises(dx.DiagnoseError, match="response is not finite"):
        dx.diagnose(f, frames2(), ENERGY, np.arange(12))


def test_a_non_finite_contraction_ratio_is_refused():
    with pytest.raises(dx.DiagnoseError, match="contraction ratio is not finite"):
        dx.diagnose(lambda z: z * 1e300, frames2(), ENERGY, np.arange(12))


def test_an_overflowing_denominator_is_refused_not_reported_as_zero():
    """Frames whose variation overflows while σ and the output stay finite: C would be
    finite / inf = 0."""
    x = 5e152 * np.random.default_rng(21).normal(0, 1, (12, L))
    with pytest.raises(dx.DiagnoseError, match="frames' variation overflows"):
        dx.diagnose(lambda z: z * 1e-10, x, ENERGY, np.arange(12))


@pytest.mark.parametrize("index", [np.arange(11), np.arange(13), np.arange(12).reshape(3, 4)])
def test_a_frame_index_of_another_length_is_refused(index):
    with pytest.raises(dx.DiagnoseError, match="frame_index has shape"):
        dx.diagnose(lambda z: z, frames2(), ENERGY, index)
    with pytest.raises(dx.DiagnoseError, match="frame_index has shape"):
        dx.noise_scale(frames2(), index)


def test_an_accepted_probe_replaces_the_defaults_and_a_dip_is_allowed():
    r = dx.diagnose(lambda z: z, frames2(), ENERGY, np.arange(12),
                    probes=[(285.0, 0.3, -2.0), (283.0, 0.5, 3.0)])
    assert [(p["E0"], p["fwhm"], p["k"]) for p in r["probes"]] == [(285.0, 0.3, -2.0),
                                                                   (283.0, 0.5, 3.0)]
    assert probe(r)["response_median"] == pytest.approx(1.0, abs=1e-12)


def test_a_descending_axis_takes_positions_from_its_first_point():
    energy = ENERGY[::-1]
    probes = dx.default_probes(energy)
    assert [p[0] for p in probes] == pytest.approx([289.0, 287.0, 285.0, 283.0, 281.0])
    assert all(p[1] == pytest.approx(0.3) for p in probes)
    r = dx.diagnose(lambda z: z, frames2(), energy, np.arange(12))
    assert len(r["probes"]) == 5


@pytest.mark.parametrize("text", ["285:0.3", "285:0.3:3:1", "a:0.3:3", "285::3", ""])
def test_a_malformed_probe_option_is_refused(text):
    with pytest.raises(dx.DiagnoseError, match="E0:FWHM:k"):
        dx.parse_probe(text)


def test_a_probe_option_parses_including_a_negative_energy():
    assert dx.parse_probe("285:0.3:3") == (285.0, 0.3, 3.0)
    assert dx.parse_probe("-5:0.6:-3") == (-5.0, 0.6, -3.0)


def test_the_cli_takes_a_negative_energy_in_the_equals_form(monkeypatch, tmp_path, ma_model):
    model, _ = ma_model
    x = frames2(n=6, e=256, seed=11)
    path = stack(tmp_path / "s.h5", x, energy=np.linspace(-10.0, 0.0, 256))
    r = diagnose(monkeypatch, tmp_path, path, model, "--probe=-5:0.6:3")
    assert [(p["E0"], p["fwhm"], p["k"]) for p in r["probes"]] == [(-5.0, 0.6, 3.0)]


def test_the_cli_refuses_a_noisy_file_and_names_the_converter(monkeypatch, capsys, tmp_path, ma_model):
    model, _ = ma_model
    x = frames2(n=6, e=256).astype(np.float32)
    path = write(tmp_path / "n.h5", x, None, None)
    err = refuse(monkeypatch, capsys, "diagnose", "-d", str(path), "-m", str(model),
                 "-o", str(tmp_path / "r.json"))
    assert "frame stack" in err and "write_frame_stack" in err


def test_the_cli_refuses_a_malformed_probe(monkeypatch, capsys, tmp_path, ma_model):
    model, data = ma_model
    err = refuse(monkeypatch, capsys, "diagnose", "-d", str(data), "-m", str(model),
                 "-o", str(tmp_path / "r.json"), "--probe", "285:0.3")
    assert "E0:FWHM:k" in err


@pytest.mark.parametrize("target", ["data", "model"])
def test_the_cli_never_writes_onto_its_input(monkeypatch, capsys, tmp_path, ma_model, target):
    model, data = ma_model
    err = refuse(monkeypatch, capsys, "diagnose", "-d", str(data), "-m", str(model),
                 "-o", str({"data": data, "model": model}[target]))
    assert "never writes to its inputs" in err


def test_the_cli_creates_the_report_directory(monkeypatch, tmp_path, ma_model):
    model, data = ma_model
    out = tmp_path / "new" / "dir" / "r.json"
    run(monkeypatch, "diagnose", "-d", str(data), "-m", str(model), "-o", str(out))
    assert load_strict(out)["n_frames"] == 10


# ------------------------------------------------------------------------- 8. held out


def test_the_training_stack_is_not_held_out_and_says_so(monkeypatch, tmp_path):
    """Rule 1 on a 300-point training stack: the digest is of the frames as stored."""
    path = stack(tmp_path / "t.h5", frames2(n=8, e=300, seed=12))
    train(monkeypatch, path, tmp_path / "m.pt", method="moving-average")
    r = diagnose(monkeypatch, tmp_path, path, tmp_path / "m.pt")
    assert (r["held_out_status"], r["held_out_basis"], r["rows_in_training"]) == (
        "not_held_out", "established", 8)
    assert dx.TRAINING_FRAMES_NOTE in r["interpretation"]
    assert r["resampled"] == {"from_points": 300, "to_points": 256}


def test_a_subset_of_the_training_frames_is_counted(monkeypatch, tmp_path, ma_model):
    model, data = ma_model
    with h5py.File(data) as f:
        frames = f["frames"][:]
    sub = stack(tmp_path / "sub.h5", frames[2:7], frame_index=np.arange(2, 7),
                acquisition_id="acq-T")
    r = diagnose(monkeypatch, tmp_path, sub, model)
    assert (r["held_out_status"], r["held_out_basis"], r["rows_in_training"]) == (
        "not_held_out", "established", 5)


def test_new_frames_of_the_same_acquisition_are_disjoint(monkeypatch, capsys, tmp_path, ma_model):
    model, _ = ma_model
    new = stack(tmp_path / "new.h5", frames2(n=6, e=256, seed=13), frame_index=np.arange(10, 16),
                acquisition_id="acq-T")
    capsys.readouterr()
    r = diagnose(monkeypatch, tmp_path, new, model)
    assert (r["held_out_status"], r["held_out_basis"]) == ("disjoint_by_identifiers",
                                                           "established")
    assert r["interpretation"] == dx.INTERPRETATION
    assert dx.TRAINING_FRAMES_NOTE not in capsys.readouterr().out
    assert "rule" not in r


def test_without_a_manifest_the_status_is_unknown(monkeypatch, capsys, tmp_path, ma_model):
    model, data = ma_model
    from dnndenoiser import provenance as prov
    ck = torch.load(model, map_location="cpu", weights_only=True)
    del ck[prov.CHECKPOINT_MANIFEST], ck[prov.CHECKPOINT_DIGEST]
    bare = tmp_path / "bare.pt"
    torch.save(ck, bare)
    capsys.readouterr()
    r = diagnose(monkeypatch, tmp_path, data, bare)
    assert (r["held_out_status"], r["held_out_basis"], r["model"]) == ("unknown", "default",
                                                                       "unknown")
    assert r["interpretation"] == dx.INTERPRETATION
    assert dx.TRAINING_FRAMES_NOTE not in capsys.readouterr().out


# ----------------------------------------------------------------------------- 9. words


def report_words(report):
    keys = []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys.append(k)
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(report)
    return keys


def printed_labels(lines):
    """Every printed line except Note: lines and the input paths, with numbers removed: the
    labels are what remains."""
    import re
    return [re.sub(r"[-+]?\d[\d.e+-]*", " ", line) for line in lines
            if line and not line.startswith(("Note:", "Data:", "Model:", "Report saved:"))]


def test_no_key_or_printed_label_carries_a_forbidden_word(monkeypatch, capsys, tmp_path, ma_model):
    model, data = ma_model
    three = tmp_path / "three.h5"
    write_frame_stack(three, (1.0 + np.random.default_rng(18).normal(0, 0.05, (6, 2, 256))
                              ).astype(np.float32), np.linspace(280.0, 290.0, 256),
                      angles=np.array([10.0, 40.0]), angle_kind="emission", angle_units="deg")
    capsys.readouterr()
    r3 = diagnose(monkeypatch, tmp_path, three, model, name="r3.json")
    lines3 = capsys.readouterr().out.splitlines()
    assert any(line.strip().startswith("pooled") for line in lines3)
    assert any("over channels" in line for line in lines3)
    assert forbidden_in(report_words(r3), printed_labels(lines3)) == []
    capsys.readouterr()
    r = diagnose(monkeypatch, tmp_path, data, model)
    lines = capsys.readouterr().out.splitlines()
    assert forbidden_in(report_words(r), printed_labels(lines)) == []
    notes = " ".join(line[len("Note: "):] for line in lines if line.startswith("Note: "))
    assert dx.INTERPRETATION in notes
    assert r["interpretation"] == f"{dx.INTERPRETATION} {dx.TRAINING_FRAMES_NOTE}"
    assert f"Note: {dx.TRAINING_FRAMES_NOTE}" in lines


@pytest.mark.parametrize("planted", ["snr_estimate", "noise_reduction", "quality"])
def test_the_word_check_catches_a_planted_key(planted):
    assert forbidden_in(report_words({"probes": [{planted: 1}]}), []) != []


@pytest.mark.parametrize("planted", [
    "  channel 0: response 0.91 [0.8, 0.95], area gain 1.0 [1, 1]",
    "  pooled over frames and channels: quality 0.9 [0.8, 1]",
    "Contraction ratio, per channel (noise reduction): 0.2",
])
def test_the_word_check_catches_a_planted_label(planted):
    assert forbidden_in([], printed_labels([planted])) != []


def test_evaluate_points_to_diagnose_exactly_with_the_same_frames_caveat(monkeypatch, capsys, tmp_path):
    x = frames2(n=6, e=64).astype(np.float32)
    mean = np.repeat(x.mean(0, keepdims=True), 6, 0)
    for decl, acq, expected in ((estimate(acquisition="acq-A"), "acq-A", True),
                                (estimate(acquisition="other"), "acq-A", False)):
        path = write(tmp_path / "e.h5", x, x, mean, declaration=decl, acquisition_id=acq,
                     frame_index=np.arange(6))
        capsys.readouterr()
        m = evaluate(monkeypatch, tmp_path, path)
        out = capsys.readouterr().out
        assert (f"Note: {ev.DIAGNOSE_POINTER}" in out) is expected
        assert (ev.SAME_FRAMES_CAVEAT in m["evaluation_context"]["caveats"]) is expected
        assert ev.DIAGNOSE_POINTER not in json.dumps(m)
    assert forbidden_in([ev.DIAGNOSE_POINTER], []) == []
