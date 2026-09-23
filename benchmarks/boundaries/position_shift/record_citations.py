"""Where every number quoted in the P2-A Record section comes from.

The Record section of `docs/preregistration/P2A-position-shift-boundary.md` quotes
numbers in prose, which nothing else checks: `render_report.py`'s guard covers
`report.md`, not that document. An audit found one quoted figure wrong and 48 of 110
quoted decimals printed nowhere a guard reached. Checking that a quoted value merely
*appears* somewhere in the record would not have been enough either -- the same value
can belong to a different arm, level, shift or metric than the sentence implies.

So every decimal or percentage in the Record section carries an anchor at the point
of use, `<!--r:KEY-->` or `<!--n:KEY-->`, and this module says what each key means:

- `r:` keys name a metric, a condition, a unit and how the value is derived from the
  record. `tests/test_p2a_record_citations.py` recomputes each one and requires the
  quoted text to equal it at the precision quoted.
- `n:` keys mark a number that is deliberately NOT from this record -- a registered
  threshold, a value from a discarded run quoted as history, a design-time
  measurement -- with the reason. They are allowed, and they are visible.

What this does not do: it cannot tell whether the prose around a number says the
right thing about it. It pins each number to a stated source; whether the sentence
reads that source correctly is a question for a reviewer, with the source now named.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

PRIMARY = "1000.0"
A, B, C, D = "A_narrow_2304", "B_augmented_2304", "C_narrow_461", "D_narrow_144"


def dkey(delta: float) -> str:
    return f"{delta + 0.0:+.2f}"


def agg(record, arm, level, delta, field):
    return record["aggregates"][arm][level][dkey(delta)][field]


def runs_mean(record, arm, level, delta, field):
    values = [r[field] for r in record["runs"]
              if r["arm"] == arm and str(r["level"]) == level and abs(r["delta"] - delta) < 1e-9]
    return float(np.mean(values))


def boundary(record, arm, direction, field, level=PRIMARY):
    return record["boundaries"][level][arm][direction][field]


def crossings(record, arm, direction, level=PRIMARY):
    return np.array(boundary(record, arm, direction, "per_seed_first_crossing_eV", level), float)


def _abs_deltas(record):
    return sorted({abs(d) for d in record["design"]["manipulated"]["delta_values_eV"]})


def _seed_curve(record, arm, sign, seed, level=PRIMARY):
    by_delta = {}
    for r in record["runs"]:
        if r["arm"] == arm and str(r["level"]) == level and r["seed_index"] == seed:
            by_delta[round(r["delta"], 9)] = r["snr_gain_db_mean"]
    return np.array([by_delta[round(sign * d, 9)] for d in _abs_deltas(record)])


def _dense_first_crossing(xs, f):
    grid = np.linspace(xs[0], xs[-1], 20001)
    ys = f(grid)
    for i in range(len(grid) - 1):
        if ys[i] >= 0.0 > ys[i + 1]:
            return float(grid[i])
    return float("nan")


def interpolated_median(record, arm, direction, kind):
    xs = _abs_deltas(record)
    sign = 1.0 if direction == "positive" else -1.0
    n = len(boundary(record, arm, direction, "per_seed_first_crossing_eV"))
    out = []
    for seed in range(n):
        y = _seed_curve(record, arm, sign, seed)
        f = PchipInterpolator(xs, y) if kind == "pchip" else CubicSpline(xs, y)
        out.append(_dense_first_crossing(xs, f))
    return float(np.median(out))


def displacement_short_of_minus_delta(record, delta):
    """How far the bias-corrected displacement at `delta` falls short of -delta, in eV."""
    return abs(agg(record, A, PRIMARY, delta, "argmax_displacement_bias_corrected_ev_mean") + delta)


def displacement_z(record, delta):
    base = {r["seed_index"]: r["argmax_displacement_ev_mean"] for r in record["runs"]
            if r["arm"] == A and str(r["level"]) == PRIMARY and r["delta"] == 0.0}
    v = np.array([r["argmax_displacement_ev_mean"] - base[r["seed_index"]] for r in record["runs"]
                  if r["arm"] == A and str(r["level"]) == PRIMARY and abs(r["delta"] - delta) < 1e-9])
    residual = v.mean() + delta
    return float(residual / (v.std(ddof=1) / np.sqrt(len(v))))


def grid_step(record):
    return record["self_checks"]["2_and_3_grid_and_test_sweep_rigidity"]["energy_step_eV"]


def smoother_span(record, level, sigma="sigma_1.0eV"):
    table = record["diagnostics"]["m3_comparators_seed_0"][level]
    values = [row[sigma] for row in table.values()]
    return max(values) - min(values)


# key -> (what it is: metric, condition, unit; how it is derived)
CITATIONS = {
    # R1-R7 deciding numbers, primary level
    "A.gain.0": ("SNR gain, arm A, level 1000, delta 0, dB, mean over seeds",
                 lambda r: agg(r, A, PRIMARY, 0.0, "snr_gain_db_mean")),
    "A.gain.0.sd": ("SNR gain, arm A, level 1000, delta 0, dB, SD across seeds",
                    lambda r: agg(r, A, PRIMARY, 0.0, "snr_gain_db_sd_across_seeds")),
    "R2.+4": ("SNR gain, arm A, level 1000, delta +4.0, dB, mean",
              lambda r: r["predictions"]["R2"]["per_direction"]["+4.00"]["stats"]["mean"]),
    "R2.-4": ("SNR gain, arm A, level 1000, delta -4.0, dB, mean",
              lambda r: r["predictions"]["R2"]["per_direction"]["-4.00"]["stats"]["mean"]),
    "R3.pos": ("boundary |delta|*, arm A, level 1000, positive direction, eV, median first crossing",
               lambda r: boundary(r, A, "positive", "median_first_crossing_eV")),
    "R3.neg": ("boundary |delta|*, arm A, level 1000, negative direction, eV, median first crossing",
               lambda r: boundary(r, A, "negative", "median_first_crossing_eV")),
    "R4.+1.5": ("paired SNR gain difference B - A, level 1000, delta +1.5, dB, mean",
                lambda r: r["predictions"]["R4"]["per_direction"]["+1.50"]["stats"]["mean"]),
    "R4.-1.5": ("paired SNR gain difference B - A, level 1000, delta -1.5, dB, mean",
                lambda r: r["predictions"]["R4"]["per_direction"]["-1.50"]["stats"]["mean"]),
    "R5a.AC": ("paired SNR gain difference A - C, level 1000, delta 0, dB, mean",
               lambda r: r["predictions"]["R5a"]["orderings"]["A_over_C"]["stats"]["mean"]),
    "R5a.CD": ("paired SNR gain difference C - D, level 1000, delta 0, dB, mean",
               lambda r: r["predictions"]["R5a"]["orderings"]["C_over_D"]["stats"]["mean"]),
    "R5b.BD": ("paired SNR gain difference B - D, level 1000, delta 0, dB, mean "
               "(the record stores D - B; quoted with the sign flipped)",
               lambda r: -r["predictions"]["R5b"]["stats"]["mean"]),
    "R5b.dz": ("Cohen's dz of D - B, level 1000, delta 0",
               lambda r: r["predictions"]["R5b"]["stats"]["cohens_dz"]),
    "R6.pos": ("boundary |delta|*, arm B, level 1000, positive direction, eV, median",
               lambda r: boundary(r, B, "positive", "median_first_crossing_eV")),
    "R7.+4": ("bias-corrected argmax displacement, arm A, level 1000, delta +4.0, eV, mean",
              lambda r: agg(r, A, PRIMARY, 4.0, "argmax_displacement_bias_corrected_ev_mean")),
    "R7.-4": ("bias-corrected argmax displacement, arm A, level 1000, delta -4.0, eV, mean",
              lambda r: agg(r, A, PRIMARY, -4.0, "argmax_displacement_bias_corrected_ev_mean")),
    "R7.+1": ("bias-corrected argmax displacement, arm A, level 1000, delta +1.0, eV, magnitude",
              lambda r: abs(agg(r, A, PRIMARY, 1.0, "argmax_displacement_bias_corrected_ev_mean"))),

    # boundary precision and direction
    "R3.sd": ("SD across seeds of arm A's first crossing, positive direction, level 1000, eV",
              lambda r: float(np.std(crossings(r, A, "positive"), ddof=1))),
    "R3.linear": ("median first crossing, arm A, positive, linear interpolation (registered), eV",
                  lambda r: boundary(r, A, "positive", "median_first_crossing_eV")),
    "R3.spline": ("median first crossing, arm A, positive, cubic-spline interpolation, eV",
                  lambda r: interpolated_median(r, A, "positive", "spline")),
    "R3.pchip": ("median first crossing, arm A, positive, PCHIP interpolation, eV",
                 lambda r: interpolated_median(r, A, "positive", "pchip")),
    "R3.interp.span": ("span of the three interpolants' medians above, eV",
                       lambda r: (max(boundary(r, A, "positive", "median_first_crossing_eV"),
                                      interpolated_median(r, A, "positive", "spline"),
                                      interpolated_median(r, A, "positive", "pchip"))
                                  - min(boundary(r, A, "positive", "median_first_crossing_eV"),
                                        interpolated_median(r, A, "positive", "spline"),
                                        interpolated_median(r, A, "positive", "pchip")))),
    "R3.dir.mean": ("paired negative-minus-positive first crossing, arm A, level 1000, eV, mean",
                    lambda r: float(np.mean(crossings(r, A, "negative") - crossings(r, A, "positive")))),
    "R3.dir.sd": ("paired negative-minus-positive first crossing, arm A, level 1000, eV, SD",
                  lambda r: float(np.std(crossings(r, A, "negative") - crossings(r, A, "positive"), ddof=1))),
    "R3.range.lo": ("minimum first crossing over both directions, arm A, level 1000, eV",
                    lambda r: float(min(crossings(r, A, "positive").min(), crossings(r, A, "negative").min()))),
    "R3.range.hi": ("maximum first crossing over both directions, arm A, level 1000, eV",
                    lambda r: float(max(crossings(r, A, "positive").max(), crossings(r, A, "negative").max()))),

    # gain tables (descriptive cells)
    **{f"A.gain.{d:+.2f}": (f"SNR gain, arm A, level 1000, delta {d:+.2f}, dB, mean",
                            (lambda d: lambda r: agg(r, A, PRIMARY, d, "snr_gain_db_mean"))(d))
       for d in (0.25, 0.5, 0.75, 1.0, 2.0, 4.0)},
    **{f"B.gain.{d:+.2f}": (f"SNR gain, arm B, level 1000, delta {d:+.2f}, dB, mean",
                            (lambda d: lambda r: agg(r, B, PRIMARY, d, "snr_gain_db_mean"))(d))
       for d in (0.0, 1.0, 1.5, 1.75, 2.0, 2.5, 4.0)},
    **{f"{k}.gain.0": (f"SNR gain, arm {k}, level 1000, delta 0, dB, mean",
                       (lambda arm: lambda r: agg(r, arm, PRIMARY, 0.0, "snr_gain_db_mean"))(arm))
       for k, arm in (("C", C), ("D", D))},
    **{f"{k}.vsA": (f"SNR gain at delta 0, level 1000, arm {k} minus arm A, dB (from the two means)",
                    (lambda arm: lambda r: agg(r, arm, PRIMARY, 0.0, "snr_gain_db_mean")
                     - agg(r, A, PRIMARY, 0.0, "snr_gain_db_mean"))(arm))
       for k, arm in (("B", B), ("C", C), ("D", D))},
    "C.boundary.nearer": ("how much nearer arm C's boundary is than arm A's, positive, level 1000, %",
                          lambda r: 100 * (1 - boundary(r, C, "positive", "median_first_crossing_eV")
                                           / boundary(r, A, "positive", "median_first_crossing_eV"))),
    "D.boundary.nearer": ("how much nearer arm D's boundary is than arm A's, positive, level 1000, %",
                          lambda r: 100 * (1 - boundary(r, D, "positive", "median_first_crossing_eV")
                                           / boundary(r, A, "positive", "median_first_crossing_eV"))),

    # R7 in both directions
    "R7.+4.short": ("|disp(+4) + 4|, arm A, level 1000, eV: shortfall from complete pinning",
                    lambda r: displacement_short_of_minus_delta(r, 4.0)),
    "R7.-4.short": ("|disp(-4) - 4|, arm A, level 1000, eV: shortfall from complete pinning",
                    lambda r: displacement_short_of_minus_delta(r, -4.0)),
    "R7.+4.steps": ("the +4.0 shortfall in units of the realised grid step",
                    lambda r: displacement_short_of_minus_delta(r, 4.0) / grid_step(r)),
    "R7.-4.steps": ("the -4.0 shortfall in units of the realised grid step",
                    lambda r: displacement_short_of_minus_delta(r, -4.0) / grid_step(r)),
    "R7.+4.z": ("the +4.0 shortfall in across-seed standard errors, magnitude",
                lambda r: abs(displacement_z(r, 4.0))),
    "R7.-4.z": ("the -4.0 shortfall in across-seed standard errors, magnitude",
                lambda r: abs(displacement_z(r, -4.0))),
    "R7.-4.pct": ("the -4.0 shortfall as a percentage of |delta|",
                  lambda r: 100 * displacement_short_of_minus_delta(r, -4.0) / 4.0),

    # M3 comparators
    "M3.smoother.span.1000": ("smoother comparator sigma 1.0 eV, span across shifts, level 1000, eV, seed 0",
                              lambda r: smoother_span(r, "1000.0")),
    "M3.smoother.span.10000": ("smoother comparator sigma 1.0 eV, span across shifts, level 10000, eV, seed 0",
                               lambda r: smoother_span(r, "10000.0")),
    **{f"B.disp.{d:+.2f}": (f"bias-corrected argmax displacement, arm B, level 1000, delta {d:+.2f}, eV",
                            (lambda d: lambda r: agg(r, B, PRIMARY, d,
                                                     "argmax_displacement_bias_corrected_ev_mean"))(d))
       for d in (0.25, 0.75, 1.25, 1.5, 1.75, 2.0, 4.0)},

    # the level-10000 cell cited under the descriptive-only exception
    "L10k.gain.+4": ("SNR gain, arm A, level 10000, delta +4.0, dB, mean",
                     lambda r: agg(r, A, "10000.0", 4.0, "snr_gain_db_mean")),
    "L10k.disp.+4": ("bias-corrected argmax displacement, arm A, level 10000, delta +4.0, eV",
                     lambda r: agg(r, A, "10000.0", 4.0, "argmax_displacement_bias_corrected_ev_mean")),
    "L10k.in.+4": ("input SNR, arm A, level 10000, delta +4.0, dB, mean over runs",
                   lambda r: runs_mean(r, A, "10000.0", 4.0, "input_snr_db_mean")),
    "L10k.out.+4": ("output SNR, arm A, level 10000, delta +4.0, dB, mean over runs",
                    lambda r: runs_mean(r, A, "10000.0", 4.0, "output_snr_db_mean")),

    # the run itself
    "run.minutes": ("total wall clock of this run, minutes",
                    lambda r: r["total_wall_clock_seconds"] / 60.0),
    "chk10.sd": ("check 10, arm B realised shift SD, seed 0, eV",
                 lambda r: r["self_checks"]["4_8_10_12_per_seed"][0]["10_augmentation"][B]["sd"]),
    "chk10.sigma": ("check 10, uniform SD of U(-1.5, 1.5), eV",
                    lambda r: r["self_checks"]["4_8_10_12_per_seed"][0]["10_augmentation"][B]
                    ["expected_uniform_sd"]),
    **{f"chk6.{lv}": (f"check 6, worst input-SNR span over 20 seeds, level {lv}, dB",
                      (lambda lv: lambda r: r["self_checks"]["6_input_snr_invariance"]
                       ["measured_span_db_per_level"][lv]["max"])(lv))
       for lv in ("100.0", "1000.0", "10000.0")},
    "chk6.margin.pct": ("check 6 at the primary level: worst span as a percentage of its tolerance",
                        lambda r: 100 * r["self_checks"]["6_input_snr_invariance"]
                        ["measured_span_db_per_level"]["1000.0"]["max"] / 0.2),
    "grid.step": ("the realised float32 energy step, eV",
                  lambda r: grid_step(r)),
    "km.minus.median": ("|KM median - np.median|, arm A, positive, level 1000, eV",
                        lambda r: abs(boundary(r, A, "positive", "kaplan_meier_median_eV")
                                      - boundary(r, A, "positive", "median_first_crossing_eV"))),
}

NON_RECORD = {
    "reg": "a registered design value or threshold from the preregistration, not a measurement",
    "history": "a value from a discarded run, an earlier revision or an earlier write-up, quoted as history",
    "design": "a design-time measurement reported in an earlier revision, not in this record",
    "derived-count": "a ratio or count stated for scale, derived by hand from values cited nearby",
}
