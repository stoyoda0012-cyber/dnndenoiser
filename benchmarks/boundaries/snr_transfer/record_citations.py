"""Where every number quoted in the P2-B Record section comes from.

The Record section of `docs/preregistration/P2B-snr-transfer.md` quotes numbers in
prose, which `render_report.py`'s guard does not reach. The rule is P2-A's
(`benchmarks/boundaries/position_shift/record_citations.py`): every decimal or
percentage carries an anchor at the point of use, and an integer carries one when it is
a quoted value rather than a plain count.

- `r:` keys name a metric, a condition, a unit and how the value is derived from the
  record. `tests/test_p2b_record_citations.py` recomputes each one and requires the
  quoted text to equal it at the precision quoted.
- `n:` keys mark a number that is deliberately NOT from this record, with the reason.

Levels are written as the record writes them ("4.0", "100.0"); a cell is train -> inference.
Keys avoid ">" because an HTML comment cannot hold it safely: `t4.i9` is the cell trained
at lambda = 4 and evaluated at lambda = 9.

What this does not do: it cannot tell whether the prose around a number reads it
correctly. It pins each number to a named source, so that a reviewer can check the reading.
"""
from __future__ import annotations

import numpy as np

LAMBDAS = (4.0, 9.0, 20.0, 45.0, 100.0)
MA, N2C = "moving_average", "noise2clean"


def _cell(t: float, i: float) -> str:
    return f"{float(t)}->{float(i)}"


def m1(record, method, t, i) -> np.ndarray:
    return np.array([s["gains_db"][method][_cell(t, i)] for s in record["seeds"]], dtype=np.float64)


def m2(record, method, t, i) -> np.ndarray:
    return m1(record, method, t, i) - m1(record, method, i, i)


def r4(record, method, a, b) -> np.ndarray:
    """-M2(a -> b) - (-M2(b -> a)), a < b, per seed: the registered R4 statistic."""
    return -m2(record, method, a, b) + m2(record, method, b, a)


def _tag(x: float) -> str:
    return f"{x:g}"


ABOVE = [(t, i) for t in LAMBDAS for i in LAMBDAS if t > i]
BELOW = [(t, i) for t in LAMBDAS for i in LAMBDAS if t < i]
PAIRS = BELOW
ONE_STEP = list(zip(LAMBDAS[:-1], LAMBDAS[1:]))


# Every count below is derived from the per-seed gains, not read from the record's
# `predictions` or `noise2clean_descriptive` trees; the rules are the registered ones.
K_FAMILY_OF_TEN = 17


def r2_count(record, method, t, i):
    return int(np.sum(m2(record, method, t, i) > -1.0))


def r3_count(record, method, t, i):
    return int(np.sum(m2(record, method, t, i) < 0))


def r4_count(record, method, a, b):
    return int(np.sum(r4(record, method, a, b) > 0))


def _met(record, method, family):
    counts = {"R2": [r2_count(record, method, t, i) for t, i in ABOVE],
              "R3": [r3_count(record, method, t, i) for t, i in BELOW],
              "R4": [r4_count(record, method, a, b) for a, b in PAIRS]}[family]
    return sum(c >= K_FAMILY_OF_TEN for c in counts)


def _stamp_ratios(record):
    return [x["lambda_from_variance"] / x["lambda"] for s in record["seeds"]
            for x in s["self_checks"]["2_noise_level"]]


def _all_sds(record):
    return ([float(np.std(m1(record, MA, t, i), ddof=1)) for t in LAMBDAS for i in LAMBDAS]
            + [float(np.std(m2(record, MA, t, i), ddof=1)) for t, i in ABOVE + BELOW])


# key -> (what it is: metric, condition, unit; how it is derived)
CITATIONS = {
    # the run
    "run.minutes": ("total wall clock of this run, minutes", lambda r: r["total_wall_clock_seconds"] / 60),
    "run.hours": ("total wall clock of this run, hours", lambda r: r["total_wall_clock_seconds"] / 3600),

    # self-checks, as stored
    "chk1.worst": ("self-check 1a/1b, worst distance from a whole count over every frame stack "
                   "and pool of every seed, in counts",
                   lambda r: max(x["worst_integer_residual"] for s in r["seeds"]
                                 for k in ("1a_exact_poisson_frames", "1b_exact_poisson_pool")
                                 for x in s["self_checks"][k])),
    "chk2.lo": ("self-check 2, smallest ratio of lambda estimated from the variance to the declared "
                "lambda, over every stack and pool of every seed", lambda r: min(_stamp_ratios(r))),
    "chk2.hi": ("self-check 2, largest such ratio", lambda r: max(_stamp_ratios(r))),
    "chk7.params": ("self-check 7, the one parameter count every trained model had",
                    lambda r: float(np.unique([x["parameters"] for s in r["seeds"]
                                               for x in s["self_checks"]["7_parameter_count"]]).item())),

    # input SNR per level: what each flux means as an operating point
    **{f"in.{_tag(x)}": (f"input SNR at lambda = {_tag(x)}, dB, mean over seeds of each seed's "
                         "mean over its test frames",
                         (lambda x: lambda r: float(np.mean([s["input_snr_db"][str(x)]
                                                             for s in r["seeds"]])))(x))
       for x in LAMBDAS},

    # both methods' diagonal M1 and both M2 grids; dB, mean over seeds. The off-diagonal
    # M1 grid is in report.md, not quoted here
    **{f"{tag}.m1.t{_tag(t)}.i{_tag(i)}": (f"M1 SNR gain, {method}, train lambda {_tag(t)}, "
                                           f"inference lambda {_tag(i)}, dB, mean over seeds",
                                           (lambda meth, t, i: lambda r: float(np.mean(m1(r, meth, t, i))))
                                           (method, t, i))
       for tag, method in (("ma", MA), ("n2c", N2C)) for t in LAMBDAS for i in LAMBDAS if t == i},
    **{f"{tag}.m1.t{_tag(x)}.i{_tag(x)}.sd": (f"M1 SNR gain, {method}, diagonal lambda {_tag(x)}, dB, "
                                              "SD across seeds",
                                              (lambda meth, x: lambda r: float(np.std(m1(r, meth, x, x),
                                                                                      ddof=1)))(method, x))
       for tag, method in (("ma", MA), ("n2c", N2C)) for x in LAMBDAS},
    **{f"{tag}.m2.t{_tag(t)}.i{_tag(i)}": (f"M2 transfer penalty, {method}, train lambda {_tag(t)}, "
                                           f"inference lambda {_tag(i)}, dB, mean over seeds",
                                           (lambda meth, t, i: lambda r: float(np.mean(m2(r, meth, t, i))))
                                           (method, t, i))
       for tag, method in (("ma", MA), ("n2c", N2C)) for t, i in ABOVE + BELOW},
    **{f"ma.m2.t{_tag(t)}.i{_tag(i)}.sd": (f"M2, moving average, train lambda {_tag(t)}, inference "
                                           f"lambda {_tag(i)}, dB, SD across seeds",
                                           (lambda t, i: lambda r: float(np.std(m2(r, MA, t, i), ddof=1)))(t, i))
       for t, i in ((4.0, 9.0), (9.0, 4.0))},
    "ma.r4.onestep.lo": ("R4 statistic, moving average, smallest mean over the four pairs of "
                         "adjacent levels, dB",
                         lambda r: min(float(np.mean(r4(r, MA, a, b))) for a, b in ONE_STEP)),
    "ma.r4.onestep.hi": ("R4 statistic, moving average, largest mean over the four pairs of "
                         "adjacent levels, dB",
                         lambda r: max(float(np.mean(r4(r, MA, a, b))) for a, b in ONE_STEP)),
    "ma.r4.lo": ("R4 statistic, moving average, smallest mean over the ten pairs, dB",
                 lambda r: min(float(np.mean(r4(r, MA, a, b))) for a, b in PAIRS)),
    "ma.r4.hi": ("R4 statistic, moving average, largest mean over the ten pairs, dB",
                 lambda r: max(float(np.mean(r4(r, MA, a, b))) for a, b in PAIRS)),
    "ma.r4.4.9": ("R4 statistic, moving average, pair 4 < 9, dB, mean over seeds",
                  lambda r: float(np.mean(r4(r, MA, 4.0, 9.0)))),
    "ma.r4.4.9.sd": ("R4 statistic, moving average, pair 4 < 9, dB, SD across seeds",
                     lambda r: float(np.std(r4(r, MA, 4.0, 9.0), ddof=1))),

    # sign counts that are not 20 of 20, and summaries of them
    "ma.R1.min.k": ("R1, moving average, smallest count of seeds with diagonal M1 > 0 over "
                    "lambda = 20, 45, 100",
                    lambda r: min(int(np.sum(m1(r, MA, x, x) > 0)) for x in (20.0, 45.0, 100.0))),
    "ma.R3.k.4.9": ("R3, moving average, cell 4 -> 9, seeds with M2 < 0, count",
                    lambda r: r3_count(r, MA, 4.0, 9.0)),
    "ma.R3.min.rest": ("R3, moving average, smallest count of seeds with M2 < 0 over the nine "
                       "cells other than 4 -> 9", lambda r: min(r3_count(r, MA, t, i) for t, i in BELOW
                                                               if (t, i) != (4.0, 9.0))),
    "ma.R4.k.4.9": ("R4, moving average, pair 4 | 9, seeds with the statistic > 0, count",
                    lambda r: r4_count(r, MA, 4.0, 9.0)),
    "ma.R4.min.rest": ("R4, moving average, smallest count of seeds with the statistic > 0 over "
                       "the nine pairs other than 4 | 9", lambda r: min(r4_count(r, MA, a, b) for a, b in PAIRS
                                                                       if (a, b) != (4.0, 9.0))),
    "ma.R3.min.k": ("R3, moving average, smallest count of seeds with M2 < 0 over the ten cells",
                    lambda r: min(r3_count(r, MA, t, i) for t, i in BELOW)),
    "ma.R4.min.k": ("R4, moving average, smallest count of seeds with the statistic > 0 over "
                    "the ten pairs", lambda r: min(r4_count(r, MA, a, b) for a, b in PAIRS)),
    "n.seeds": ("independently seeded runs (the replicates)", lambda r: len(r["seeds"])),
    "n.test": ("test frames per inference level per seed", lambda r: r["design"]["n_test_per_level"]),
    "n.points": ("energy points per spectrum", lambda r: r["design"]["generator_config"]["n_energy_points"]),
    "ma.R2.min.k": ("R2, moving average, smallest count of seeds with M2 > -1 dB over the ten cells",
                    lambda r: min(r2_count(r, MA, t, i) for t, i in ABOVE)),
    "ma.R2.min.m2": ("R2, moving average, most negative single-seed M2 over the ten cells above "
                     "the diagonal (200 seed-cells), dB",
                     lambda r: min(float(np.min(m2(r, MA, t, i))) for t, i in ABOVE)),
    "ma.R2.worst.loss": ("R2, moving average, the largest single-seed loss against the matched "
                         "model over the ten cells above the diagonal, dB, as a positive number "
                         "(minus the most negative M2)",
                         lambda r: -min(float(np.min(m2(r, MA, t, i))) for t, i in ABOVE)),
    "ma.R2.seedcells": ("seed-cells in R2's family: ten cells times the seeds",
                        lambda r: len(ABOVE) * len(r["seeds"])),
    # the one cell whose mean M1 is negative: worse than the input
    "ma.cells": ("cells in the moving average's grid, training level by inference level",
                 lambda r: len(LAMBDAS) ** 2),
    "ma.m1neg.cells": ("moving average, cells of the 25 whose mean M1 is negative, count",
                       lambda r: sum(float(np.mean(m1(r, MA, t, i))) < 0 for t in LAMBDAS for i in LAMBDAS)),
    "ma.m1.t4.i100.sd": ("M1, moving average, train 4, inference 100, dB, SD across seeds",
                         lambda r: float(np.std(m1(r, MA, 4.0, 100.0), ddof=1))),
    "ma.m1.t4.i100.min": ("M1, moving average, train 4, inference 100, dB, smallest over seeds",
                          lambda r: float(np.min(m1(r, MA, 4.0, 100.0)))),
    "ma.m1neg.4.100": ("moving average, train 4, inference 100, seeds with M1 < 0, count",
                       lambda r: int(np.sum(m1(r, MA, 4.0, 100.0) < 0))),
    "ma.m1neg.4.45": ("moving average, train 4, inference 45, seeds with M1 < 0, count",
                      lambda r: int(np.sum(m1(r, MA, 4.0, 45.0) < 0))),
    "ma.m1.t4.i100": ("M1, moving average, train 4, inference 100, dB, mean over seeds",
                      lambda r: float(np.mean(m1(r, MA, 4.0, 100.0)))),
    # the moving average's normalisation constants, as stored by the run (not verified by
    # the renderer's guard)
    **{f"norm.max.{_tag(x)}": (f"moving average, maximum of the lambda = {_tag(x)} training stack "
                               "(the min-max normalisation's upper constant; the clean spectrum's "
                               "maximum is 1), mean over seeds, as stored",
                               (lambda x: lambda r: float(np.mean(
                                   [s["training"][MA][str(x)]["normalisation"]["max"]
                                    for s in r["seeds"]])))(x))
       for x in (4.0, 100.0)},
    "chk6.cells": ("self-check 6, cells checked per seed (the same in every seed)",
                   lambda r: float(np.unique([s["self_checks"]["6_same_test_arrays"]["cells_checked"]
                                              for s in r["seeds"]]).item())),
    "ma.m2pos.9.4": ("moving average, cell 9 -> 4, seeds with M2 > 0, count (not a registered rule)",
                     lambda r: int(np.sum(m2(r, MA, 9.0, 4.0) > 0))),
    "ma.m2pos.min.rest": ("moving average, smallest count of seeds with M2 > 0 over the nine cells "
                          "above the diagonal other than 9 -> 4 (not a registered rule)",
                          lambda r: min(int(np.sum(m2(r, MA, t, i) > 0)) for t, i in ABOVE
                                        if (t, i) != (9.0, 4.0))),
    "ma.m2.rest.lo": ("moving average, smallest mean M2 over those nine cells, dB",
                      lambda r: min(float(np.mean(m2(r, MA, t, i))) for t, i in ABOVE if (t, i) != (9.0, 4.0))),
    "ma.m2.rest.hi": ("moving average, largest mean M2 over those nine cells, dB",
                      lambda r: max(float(np.mean(m2(r, MA, t, i))) for t, i in ABOVE if (t, i) != (9.0, 4.0))),

    # noise2clean, descriptive
    "n2c.R2.met": ("noise2clean, R2-type cells meeting the R2 rule at 17 of 20, count of ten",
                   lambda r: _met(r, N2C, "R2")),
    "n2c.R3.met": ("noise2clean, R3-type cells meeting the R3 rule at 17 of 20, count of ten",
                   lambda r: _met(r, N2C, "R3")),
    "n2c.R4.met": ("noise2clean, R4-type pairs meeting the R4 rule at 17 of 20, count of ten",
                   lambda r: _met(r, N2C, "R4")),
    "n2c.above.lo": ("noise2clean, most negative mean M2 over the ten cells above the diagonal "
                     "(training above inference), dB",
                     lambda r: min(float(np.mean(m2(r, N2C, t, i))) for t, i in ABOVE)),
    "n2c.above.hi": ("noise2clean, least negative mean M2 over those ten cells, dB",
                     lambda r: max(float(np.mean(m2(r, N2C, t, i))) for t, i in ABOVE)),
    "n2c.above.npos": ("noise2clean, cells above the diagonal whose mean M2 is positive, count of ten",
                       lambda r: sum(float(np.mean(m2(r, N2C, t, i))) > 0 for t, i in ABOVE)),
    "n2c.offdiag.npos": ("noise2clean, off-diagonal cells (both directions) whose mean M2 is "
                         "positive, count of twenty",
                         lambda r: sum(float(np.mean(m2(r, N2C, t, i))) > 0 for t, i in ABOVE + BELOW)),
    "n2c.r4.lo": ("noise2clean, smallest mean R4 statistic over the ten pairs, dB",
                  lambda r: min(float(np.mean(r4(r, N2C, a, b))) for a, b in PAIRS)),
    "n2c.r4.hi": ("noise2clean, largest mean R4 statistic over the ten pairs, dB",
                  lambda r: max(float(np.mean(r4(r, N2C, a, b))) for a, b in PAIRS)),

    # precision: the spread across seeds that one decimal is set against
    "ma.sd.min": ("moving average, smallest SD across seeds of any M1 or off-diagonal M2 cell, dB",
                  lambda r: min(_all_sds(r))),
    "ma.sd.max": ("moving average, largest SD across seeds of any M1 or off-diagonal M2 cell, dB",
                  lambda r: max(_all_sds(r))),
    # the two methods' diagonals, both ends quoted
    **{f"diag.n2c.minus.ma.{_tag(x)}": (f"noise2clean diagonal M1 minus moving-average diagonal M1 at "
                                        f"lambda = {_tag(x)}, dB, difference of the two means",
                                        (lambda x: lambda r: float(np.mean(m1(r, N2C, x, x))
                                                                   - np.mean(m1(r, MA, x, x))))(x))
       for x in (4.0, 100.0)},

    # P2-A's operating point, as this run copied it into this record
    "p2a.copied": ("P2-A arm A, level 1000, delta 0, SNR gain in dB, as copied into this record "
                   "at run time (p2a_beside_noise2clean)",
                   lambda r: r["p2a_beside_noise2clean"]["p2a_arm_A_level_1000_delta_0_gain_db"]),
}

NON_RECORD = {
    "reg": "a registered design value, threshold or estimate from the preregistration, not a measurement",
    "impl": "a tolerance fixed in the measurement script at implementation, before the run",
    "external": "a timestamp or fact from GitHub or git, not from this record",
}


# PROPOSED, not in effect: the record keys `docs/WHEN_TO_TRUST.md` may cite in its P2-B
# region (between `<!-- record:P2-B -->` and `<!-- /record:P2-B -->`), as the
# independent review of 51ab3b8 listed them: R1 to R4 in their registered form, the one
# cell whose output was worse than its input and that it was the only one (added at the
# owner's decision after the review of 9758a41), and the conditions a reader needs to
# place them. The clearance takes effect only when the owner confirms it in a Revision of the
# preregistration, after an independent review of the page.
CLEARED_FOR_WHEN_TO_TRUST = frozenset({
    "ma.R1.min.k",
    "ma.R2.seedcells", "ma.R2.worst.loss",
    "ma.R3.min.k",
    "ma.R4.min.k",
    "ma.m1.t4.i100", "ma.m1.t4.i100.sd", "ma.m1neg.4.100",
    "ma.m1neg.cells", "ma.cells",
    "in.4", "in.100",
    "n.seeds", "n.test", "n.points",
})

# Qualifiers that are part of the proposed statements. Deleting one leaves every number
# right, so the number checks cannot see it; this list can.
PAGE_REQUIRED_PHRASES = (
    "than a model trained at that count rate, in any of the",
    "compare a model with one trained at the count rate it was applied to, not with the noisy input",
    "the output still improved on the input",
    "was not measured, and these results should not be assumed to hold for it",
    "cannot be attributed to the signal-to-noise ratio",
    "None of these is separated from the others",
    "the output was further from the clean spectrum than the noisy input was",
    "exactly right by construction",
    "does not measure whether the output varies from frame to frame",
    "was not measured",
)


def page_design_values(record):
    """Registered design values an `n:reg` number on the P2-B region may equal: the flux
    levels and their ratio, frame counts, total exposure, the updates they imply, seeds,
    W, epochs, batch size, the R2 margin and the family thresholds."""
    design = record["design"]
    recipe = design["moving_average_recipe"]
    frames = [int(v) for v in design["frames_per_level"].values()]
    lambdas = [float(x) for x in design["lambdas"]]
    values = set(lambdas) | set(frames) | {
        float(design["total_exposure"]), float(design["W"]), float(recipe["epochs"]),
        float(recipe["batch_size"]), max(lambdas) / min(lambdas), 1.0, 17.0, 19.0}
    values |= {float(-(-n // recipe["batch_size"]) * recipe["epochs"]) for n in frames}
    return values
