"""How a model's output depends on its input, on a frame stack.

Implements ``docs/design/OUTPUT_CONTRACTION.md`` (adopted 2026-10-03). Three descriptive
quantities, per channel:

- the **contraction ratio** ``C``: the output's frame-to-frame variation over the input's;
- the **injection response** ``R``: the fraction of a known Gaussian added to the input
  that reaches the output, projected on that Gaussian (a secant at the amplitude used);
- the **area ratio** ``A``: the summed output change over the summed injected change.

None of them is an accuracy, a noise reduction or an SNR. ``C`` counts removed signal
change as removed variation and is blind to bias; ``R`` is reduced by attenuation,
broadening and shift alike, and 1 is what returning the input gives; ``A = 1`` does not
show that the areas of real features are preserved.

:func:`diagnose` takes frames already on the network grid and a callable ``f`` that maps
arrays shaped ``(n_frames, [n_channels,] L)`` to arrays of the same shape, in input units,
acting row by row. Arithmetic is in float64.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

OUTPUT_VERSION = "1"
DEFAULT_POSITIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
DEFAULT_FWHM_FRACTION = 0.03
DEFAULT_K = 3.0
SIGMA_FLOOR = 1e-12
OVERALL_WEIGHTING = ("ratio of the channels' summed numerators to their summed denominators: "
                     "each channel weighted by its frame-to-frame variation")
SPREAD = "[q25, q75] over frames (NumPy linear quantiles); a spread, not an uncertainty"

INTERPRETATION = (
    "These describe how the output depends on the input. They are not measures of accuracy, "
    "of noise reduction or of SNR: the contraction ratio counts removed signal change as well "
    "as removed noise and is blind to bias; the injection response is reduced by attenuation, "
    "broadening and shift alike, and 1 is what returning the input gives; an area ratio of 1 "
    "does not show that the areas of real features are preserved.")
TRAINING_FRAMES_NOTE = ("The frames include the model's training data; these values describe "
                        "the trained model on its training frames.")


class DiagnoseError(ValueError):
    """Input that the diagnostic cannot describe."""


def _channels(frames: np.ndarray) -> np.ndarray:
    """``(n, L)`` or ``(n, c, L)`` as a float64 ``(n, c, L)`` array."""
    x = np.asarray(frames, dtype=np.float64)
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise DiagnoseError(f"frames must be (n_frames, energy) or (n_frames, n_channels, energy), "
                        f"got shape {x.shape}")


def noise_scale(frames, frame_index) -> np.ndarray:
    """σ per channel: the median over grid points of ``std(Δx)/√2``, ``Δx`` the differences
    of frames whose ``frame_index`` values differ by exactly 1 (ddof 0)."""
    x = _channels(frames)
    index = np.asarray(frame_index)
    order = np.argsort(index, kind="stable")
    adjacent = np.flatnonzero(np.diff(index[order]) == 1)
    if adjacent.size < 2:
        raise DiagnoseError(f"σ needs at least two pairs of frames whose frame_index values "
                            f"are adjacent; found {adjacent.size} (fewer than two pairs of "
                            "adjacent frame indices)")
    delta = x[order[adjacent + 1]] - x[order[adjacent]]
    sigma = np.median(delta.std(axis=0) / np.sqrt(2.0), axis=-1)
    for c, s in enumerate(sigma):
        scale = np.abs(x[:, c]).max()
        if not s > SIGMA_FLOOR * scale:
            raise DiagnoseError(f"channel {c}: σ = {s:.3g} is not above {SIGMA_FLOOR:g} × "
                                f"max|x| = {SIGMA_FLOOR * scale:.3g}; there is no measurable "
                                "noise to size the probe")
    return sigma


def default_probes(energy) -> list:
    """Five positions, 10 to 90 % of the way from the first energy point to the last, each
    with FWHM 3 % of |span| and k = 3."""
    e = np.asarray(energy, dtype=np.float64)
    first, span = float(e[0]), float(e[-1] - e[0])
    return [(first + q * span, DEFAULT_FWHM_FRACTION * abs(span), DEFAULT_K)
            for q in DEFAULT_POSITIONS]


def parse_probe(text: str) -> tuple:
    """``"E0:FWHM:k"`` as three floats."""
    parts = str(text).split(":")
    try:
        if len(parts) != 3:
            raise ValueError
        return tuple(float(p) for p in parts)
    except ValueError:
        raise DiagnoseError(f"a probe is given as E0:FWHM:k (three numbers); got {text!r}. "
                            "For a negative E0 use the = form, --probe=-5:0.6:3") from None


def check_probe(energy, e0: float, fwhm: float, k: float) -> None:
    e = np.asarray(energy, dtype=np.float64)
    if not all(np.isfinite(v) for v in (e0, fwhm, k)):
        raise DiagnoseError(f"probe {e0}:{fwhm}:{k}: every value must be finite")
    if fwhm <= 0:
        raise DiagnoseError(f"probe {e0}:{fwhm}:{k}: FWHM must be positive")
    if k == 0:
        raise DiagnoseError(f"probe {e0}:{fwhm}:{k}: k must be non-zero (negative is a dip)")
    lo, hi = float(e.min()), float(e.max())
    if not lo <= e0 <= hi:
        raise DiagnoseError(f"probe {e0}:{fwhm}:{k}: E0 is outside the energy axis "
                            f"[{lo:g}, {hi:g}]")
    spacing = (hi - lo) / (e.size - 1)
    if fwhm < 2 * spacing:
        raise DiagnoseError(f"probe {e0}:{fwhm}:{k}: FWHM is below twice the grid spacing "
                            f"({2 * spacing:.4g}), so the Gaussian is not resolved")
    if min(e0 - lo, hi - e0) < 2 * fwhm:
        raise DiagnoseError(f"probe {e0}:{fwhm}:{k}: E0 is closer than 2·FWHM to an end of "
                            f"the energy axis [{lo:g}, {hi:g}]")


def probe_shape(energy, e0: float, fwhm: float) -> np.ndarray:
    """A Gaussian of unit height at ``e0`` with full width at half maximum ``fwhm``."""
    e = np.asarray(energy, dtype=np.float64)
    return np.exp(-4.0 * np.log(2.0) * (e - e0) ** 2 / fwhm ** 2)


def _call(f: Callable, x: np.ndarray, shape: tuple, what: str) -> np.ndarray:
    y = np.asarray(f(x.reshape(shape)), dtype=np.float64)
    if y.shape != shape:
        raise DiagnoseError(f"the model returned shape {y.shape} for {what} of shape {shape}")
    if not np.all(np.isfinite(y)):
        raise DiagnoseError(f"the model's output for {what} contains non-finite values")
    return y.reshape(x.shape)


def _summary(values: np.ndarray) -> tuple:
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return float(median), [float(q25), float(q75)]


def _summaries(r: np.ndarray, a: np.ndarray) -> dict:
    r_med, r_iqr = _summary(r)
    a_med, a_iqr = _summary(a)
    return {"response_median": r_med, "response_iqr": r_iqr,
            "area_median": a_med, "area_iqr": a_iqr}


def diagnose(f: Callable, frames, energy, frame_index,
             probes: Optional[Sequence[tuple]] = None) -> dict:
    """C, σ, and R and A for each probe, per channel. ``probes`` (``(E0, FWHM, k)``
    tuples) replaces the defaults of :func:`default_probes`."""
    # An overflow is refused below with a message, not reported as a warning first.
    with np.errstate(over="ignore", invalid="ignore"):
        return _diagnose(f, frames, energy, frame_index, probes)


def _diagnose(f, frames, energy, frame_index, probes):
    shape = np.shape(frames)
    x = _channels(frames)
    n, n_channels, length = x.shape
    if n < 2:
        raise DiagnoseError(f"the diagnostic needs at least two frames, got {n}")
    if np.asarray(energy).shape != (length,):
        raise DiagnoseError(f"energy has shape {np.shape(energy)}, frames have {length} points")
    if not np.all(np.isfinite(x)):
        raise DiagnoseError("the frames contain non-finite values")
    if np.shape(frame_index) != (n,):
        raise DiagnoseError(f"frame_index has shape {np.shape(frame_index)}; there are {n} frames")

    mean = x.mean(axis=0)
    denominators = ((x - mean) ** 2).sum(axis=(0, 2))
    for c in range(n_channels):
        # Tested on the frames themselves: a float mean of equal values need not equal
        # them, so the denominator of identical frames is not reliably zero.
        if np.all(x[:, c] == x[0, c]):
            raise DiagnoseError(f"channel {c}: identical frames, so the contraction ratio "
                                "has a zero denominator")
    probes = default_probes(energy) if probes is None else [tuple(map(float, p)) for p in probes]
    for p in probes:
        check_probe(energy, *p)
    sigma = noise_scale(x, frame_index)

    y = _call(f, x, shape, "the frames")
    numerators = ((y - y.mean(axis=0)) ** 2).sum(axis=(0, 2))
    if not np.all(np.isfinite(numerators)):
        raise DiagnoseError("the contraction ratio is not finite (the output's variation "
                            "overflows)")
    report = {
        "diagnose_output_version": OUTPUT_VERSION,
        "n_frames": int(n),
        "n_channels": int(n_channels),
        "contraction_ratio": {
            "channels": [float(v) for v in numerators / denominators],
            "overall": float(numerators.sum() / denominators.sum()),
            "overall_weighting": OVERALL_WEIGHTING,
        },
        "sigma": [float(s) for s in sigma],
        "spread": SPREAD,
        "probes": [],
    }
    for e0, fwhm, k in probes:
        g = probe_shape(energy, e0, fwhm)
        amplitude = k * sigma                                       # per channel
        injected = x + amplitude[None, :, None] * g
        change = _call(f, injected, shape, f"the frames with probe {e0}:{fwhm}:{k}") - y
        r = (change @ g) / (amplitude[None, :] * (g @ g))           # (n, c)
        a = change.sum(axis=-1) / (amplitude[None, :] * g.sum())
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(a))):
            raise DiagnoseError(f"probe {e0}:{fwhm}:{k}: the response is not finite")
        report["probes"].append({
            "E0": float(e0), "fwhm": float(fwhm), "k": float(k),
            "channels": [{"amplitude": float(amplitude[c]), **_summaries(r[:, c], a[:, c])}
                         for c in range(n_channels)],
            "pooled": (_summaries(r.ravel(), a.ravel()) if len(shape) == 3 else None),
        })
    return report
