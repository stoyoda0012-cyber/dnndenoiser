"""What ``evaluate`` computes, named by what it was compared against.

Implements the metric part of ``docs/design/EVALUATION_REFERENCE_CONTRACT.md``
(adopted 2026-10-01, phase 1).

- :func:`evaluate_arrays` is the normal output (``evaluate_output_version`` ``"2"``):
  float64 throughout, strict JSON (no NaN or Infinity; an undefined value is ``None``
  with a status), primary quantities that are mean MSEs against the selected reference,
  and dB quantities only where the reference supports them.
- :func:`evaluate_legacy` reproduces the pre-change arithmetic of ``cmd_evaluate`` at
  ``cb5e000`` exactly, for undeclared references only, and refuses when any historical
  value is not finite.

A reference that is not synthetic truth is an estimate of the signal; agreement with it
is not an established error against the signal. The names below say which.
"""
from __future__ import annotations

import math

import numpy as np

FLOOR = 1e-10
OUTPUT_VERSION = "2"
LEGACY_VERSION = "1-legacy"
LEGACY_BASELINE = "cmd_evaluate at cb5e000"
AGGREGATION_UNIT = ("the flattened spectrum: every non-energy row counts once; flattened rows "
                    "are not independent specimens")

ESTIMATE_CAVEAT = ("This measures agreement with a reference estimate. The reference's noise, "
                   "bias, and dependence on the evaluated data or the model can affect it; it "
                   "is not an established error against the underlying signal.")
SAME_FRAMES_CAVEAT = ("A model returning this mean for every frame has zero discrepancy from it "
                      "by construction. That agreement does not establish accuracy.")
EXPECTATION_CONDITION = ("The difference of mean discrepancies equals the difference of errors "
                         "against the underlying signal in expectation exactly when the cross "
                         "term E<input - output, reference error> is zero. A sufficient "
                         "condition is that, given the matched signal, the reference's error "
                         "has zero mean and is independent of both the input and the output; "
                         "independence alone is not enough.")


class NonFiniteLegacyValue(ValueError):
    """A historical value that cannot be written as strict JSON."""


def case_of(effective: dict, relationship: dict) -> str:
    """``truth``, ``estimate``, ``estimate_same_data`` or ``undeclared``."""
    origin = effective["origin"]
    if origin == "synthetic_truth":
        return "truth"
    if origin == "undeclared":
        return "undeclared"
    if relationship["overlap_with_evaluated"] == "overlap":
        return "estimate_same_data"
    return "estimate"


HEADINGS = {
    "truth": "Error against the synthetic truth",
    "estimate": "Discrepancy from a reference estimate",
    "estimate_same_data": "Discrepancy from a reference built from the evaluated data",
    "undeclared": "Discrepancy from an undeclared reference",
}


def _flat(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, a.shape[-1]) if a.ndim > 2 else a.reshape(1, -1) if a.ndim == 1 else a


def _finite_or_null(value: float, status: str, statuses: dict, key: str):
    if value is None or not math.isfinite(value):
        statuses[key] = status
        return None
    return float(value)


def evaluate_arrays(noisy: np.ndarray, denoised: np.ndarray, reference: np.ndarray,
                    case: str) -> dict:
    """Normal output for arrays of equal shape. Raises ``ValueError`` on empty or
    non-finite inputs."""
    x, y, r = _flat(noisy), _flat(denoised), _flat(reference)
    if x.size == 0 or y.size == 0 or r.size == 0:
        raise ValueError("empty arrays cannot be evaluated")
    for name, arr in (("noisy", x), ("denoised", y), ("reference", r)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"'{name}' contains non-finite values")

    statuses: dict = {}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mse_in = np.mean((x - r) ** 2, axis=-1)
        mse_out = np.mean((y - r) ** 2, axis=-1)
    out: dict = {"n_spectra": int(len(mse_in))}
    out["mse_in_mean"] = _finite_or_null(np.mean(mse_in), "overflow", statuses, "mse_in_mean")
    out["mse_out_mean"] = _finite_or_null(np.mean(mse_out), "overflow", statuses, "mse_out_mean")
    if out["mse_in_mean"] is not None and out["mse_out_mean"] is not None:
        out["mse_difference"] = out["mse_in_mean"] - out["mse_out_mean"]
    else:
        out["mse_difference"] = None
        statuses["mse_difference"] = "an operand is undefined"

    with np.errstate(over="ignore", invalid="ignore"):
        total_in = float(np.sum(mse_in))
        total_out = float(np.sum(mse_out))
    key = "relative_mse_change_aggregate_pct"
    if not (math.isfinite(total_in) and math.isfinite(total_out)
            and np.all(np.isfinite(mse_in)) and np.all(np.isfinite(mse_out))):
        out[key] = None
        statuses[key] = "overflow in the summed MSE"
    elif total_in == 0:
        out[key] = None
        statuses[key] = "the summed input MSE is zero"
    else:
        out[key] = _finite_or_null(100.0 * (1.0 - total_out / total_in), "overflow",
                                   statuses, key)
    positive = mse_in > 0
    out["per_spectrum_excluded_zero_input_mse"] = int(np.sum(~positive))
    key = "mean_relative_mse_change_per_spectrum_pct"
    if not (np.all(np.isfinite(mse_in)) and np.all(np.isfinite(mse_out))):
        # An infinite intermediate can turn into a wrong finite percentage (inf in a
        # denominator gives 100 %); refuse the value rather than report it.
        out[key] = None
        statuses[key] = "overflow in a per-spectrum MSE"
    elif np.any(positive):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = 100.0 * float(np.mean(1.0 - mse_out[positive] / mse_in[positive]))
        out[key] = _finite_or_null(value, "overflow", statuses, key)
    else:
        out["mean_relative_mse_change_per_spectrum_pct"] = None
        statuses["mean_relative_mse_change_per_spectrum_pct"] = "every spectrum has zero input MSE"

    if case in ("truth", "estimate"):
        prefix = {"truth": "snr", "estimate": "agreement_db"}[case]
        change = {"truth": "gain", "estimate": "change"}[case]
        with np.errstate(over="ignore", invalid="ignore"):
            p_ref = np.mean(r ** 2, axis=-1)
        eligible = p_ref > 0
        out["zero_reference_power_count"] = int(np.sum(~eligible))
        p_in = mse_in[eligible]
        p_out = mse_out[eligible]
        out["floor_active_input_count"] = int(np.sum(p_in <= FLOOR))
        out["floor_active_output_count"] = int(np.sum(p_out <= FLOOR))
        keys = (f"{prefix}_input_mean", f"{prefix}_output_mean", f"{prefix}_{change}_mean",
                f"{prefix}_{change}_std")
        if np.any(eligible):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                s_in = 10 * np.log10(p_ref[eligible] / np.maximum(p_in, FLOOR))
                s_out = 10 * np.log10(p_ref[eligible] / np.maximum(p_out, FLOOR))
                delta = s_out - s_in
            values = (np.mean(s_in), np.mean(s_out), np.mean(delta), np.std(delta))
            for key, value in zip(keys, values):
                out[key] = _finite_or_null(value, "overflow", statuses, key)
        else:
            for key in keys:
                out[key] = None
                statuses[key] = "no spectrum has non-zero reference power"
    if statuses:
        out["status"] = statuses
    return out


def evaluate_legacy(noisy: np.ndarray, denoised: np.ndarray, clean: np.ndarray) -> dict:
    """The pre-change ``cmd_evaluate`` arithmetic, verbatim: the arrays' own dtype, the
    same formulas and keys. Refuses, naming the metric, if any value is not finite."""
    from dnndenoiser.cli import compute_mse, compute_snr

    if noisy.ndim > 2:
        noisy = noisy.reshape(-1, noisy.shape[-1])
        denoised = denoised.reshape(-1, denoised.shape[-1])
        clean = clean.reshape(-1, clean.shape[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_input = compute_snr(noisy, clean)
        mse_input = compute_mse(noisy, clean)
        snr_output = compute_snr(denoised, clean)
        mse_output = compute_mse(denoised, clean)
        snr_gain = snr_output - snr_input
        mse_reduction = (mse_input - mse_output) / mse_input * 100
        metrics = {
            'snr_input_mean': float(np.mean(snr_input)),
            'snr_output_mean': float(np.mean(snr_output)),
            'snr_gain_mean': float(np.mean(snr_gain)),
            'snr_gain_std': float(np.std(snr_gain)),
            'mse_input_mean': float(np.mean(mse_input)),
            'mse_output_mean': float(np.mean(mse_output)),
            'mse_reduction_mean': float(np.mean(mse_reduction)),
            'n_samples': len(noisy),
        }
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise NonFiniteLegacyValue(
                f"legacy output refused: the historical '{key}' is {value}, which strict JSON "
                "cannot hold; declare the reference and use the normal output instead")
    return metrics


def caveats(case: str, effective: dict) -> list:
    if case == "truth":
        return []
    notes = [ESTIMATE_CAVEAT, EXPECTATION_CONDITION]
    if case == "estimate_same_data" and effective.get("construction") == "frame_mean":
        notes.append(SAME_FRAMES_CAVEAT)
    return notes


def report_lines(case: str, result: dict) -> list:
    """Printed report. The first line is the heading; names follow the case."""
    n = result["n_spectra"]
    lines = [f"=== {HEADINGS[case]} ===",
             f"Spectra: {n} (flattened rows; not independent specimens)"]
    excluded = result["per_spectrum_excluded_zero_input_mse"]
    lines.append(f"Per-spectrum relative change over {n - excluded} of {n} spectra "
                 f"({excluded} excluded: zero input MSE)")
    if "zero_reference_power_count" in result:
        zero = result["zero_reference_power_count"]
        lines.append(f"dB statistics over {n - zero} of {n} spectra ({zero} excluded: zero "
                     "reference power)")
        lines.append(f"Residual-power floor applied: input {result['floor_active_input_count']}, "
                     f"output {result['floor_active_output_count']} spectra")
    lines.append("")

    def fmt(v, spec):
        return "undefined" if v is None else format(v, spec)

    lines.append(f"{'mean MSE, input':<34} {fmt(result['mse_in_mean'], '.4e')}")
    lines.append(f"{'mean MSE, output':<34} {fmt(result['mse_out_mean'], '.4e')}")
    lines.append(f"{'input minus output':<34} {fmt(result['mse_difference'], '.4e')}")
    lines.append(f"{'relative change, aggregate (%)':<34} "
                 f"{fmt(result['relative_mse_change_aggregate_pct'], '.2f')}")
    lines.append(f"{'relative change, per spectrum (%)':<34} "
                 f"{fmt(result['mean_relative_mse_change_per_spectrum_pct'], '.2f')}")
    if case == "truth":
        lines.append(f"{'SNR input / output (dB)':<34} {fmt(result['snr_input_mean'], '.2f')} / "
                     f"{fmt(result['snr_output_mean'], '.2f')}")
        lines.append(f"{'SNR gain, mean ± std (dB)':<34} {fmt(result['snr_gain_mean'], '+.2f')} ± "
                     f"{fmt(result['snr_gain_std'], '.2f')}")
    elif case == "estimate":
        lines.append(f"{'agreement input / output (dB)':<34} "
                     f"{fmt(result['agreement_db_input_mean'], '.2f')} / "
                     f"{fmt(result['agreement_db_output_mean'], '.2f')}")
        lines.append(f"{'change in agreement (dB)':<34} "
                     f"{fmt(result['agreement_db_change_mean'], '+.2f')} ± "
                     f"{fmt(result['agreement_db_change_std'], '.2f')}")
    for key, reason in result.get("status", {}).items():
        lines.append(f"undefined: {key} -- {reason}")
    return lines
