"""Regenerate `report.md` FROM the P2-B record, recomputing everything it renders.

Numbers in `report.md` are never typed by hand, and nor is anything else in it: every
line comes out of here, so a paragraph added to that file by hand is dropped the next
time it is regenerated.

Before rendering, `verify` rebuilds from the per-seed gains alone (`seeds[i].gains_db`)
everything downstream of them, and refuses to render if any of it disagrees with the
record. It reports every disagreement together rather than stopping at the first.

What each part establishes, stated so that it cannot be read as more:

- **Independent recomputation** (arithmetic written here, not imported): every
  aggregate -- M1 mean, SD and per-seed list, M2 mean and SD, for both methods and all
  25 cells; every sign count, binomial p, Holm-adjusted p and verdict for R1 to R4 and
  for the noise2clean descriptive families, against thresholds written here as the
  registered literals (19 for R1, 17 for a family of ten); the noise2clean diagonal set
  beside P2-A.
- **Checked against literals written here** from the registration (and, for the
  generator and noise2clean recipe, from P2-A's settings): the design block -- levels,
  Poisson levels, total exposure, frame counts, seeds, test frames, pool size, W, epochs,
  architecture, model configuration, both recipes including the normalisation, the
  generator configuration, the peak set and the noise model -- and the list of
  differences stated beside P2-A.
- **Checked against P2-A's committed record**, when it is present: the P2-A gain this
  record copied.
- **Consistency check** (the measurement script's own `evaluate_predictions` and
  `descriptive_statistics`, deep-diffed against the record): the whole `predictions`
  and `noise2clean_descriptive` trees, including the fields the independent part does not
  rebuild (`evaluated`, `family_size`, `failed_levels`, the cells made descriptive,
  `m1_beside_each_cell`). This catches a record edited, truncated or left stale relative
  to its per-seed gains; it cannot catch an error inside those functions.
- **Not verified here at all**: the per-seed gains themselves (they are the raw data),
  the self-check figures, the input SNRs, the training times and normalisation
  constants, `generated_utc`, the wall clock, the environment and the provenance. They
  are printed as stored, and the report says so where it prints them.

`tests/test_snr_transfer_record.py` tampers with a copy of the record one field at a
time and requires this guard to refuse each edit.

Usage:

    python benchmarks/boundaries/snr_transfer/render_report.py            # to stdout
    python benchmarks/boundaries/snr_transfer/render_report.py --write    # to report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import snr_transfer as measurement  # noqa: E402

DEFAULT_RECORD = Path(__file__).resolve().parent / "results" / "snr_transfer.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "report.md"
TOLERANCE = 1e-9

# The registered design, written out here rather than read from the script, so that a
# record whose design block was altered -- or a script that drifted from the
# registration -- is refused rather than rendered.
REGISTERED_LAMBDAS = (4.0, 9.0, 20.0, 45.0, 100.0)
REGISTERED_FRAMES = {4.0: 12500, 9.0: 5556, 20.0: 2500, 45.0: 1111, 100.0: 500}
REGISTERED_SEEDS = 20
REGISTERED_TEST_FRAMES = 512
REGISTERED_POOL = 2304
REGISTERED_R1_LEVELS = (20.0, 45.0, 100.0)
REGISTERED_R1_K = 19
REGISTERED_FAMILY_OF_TEN_K = 17
REGISTERED_R2_MARGIN_DB = -1.0
REGISTERED_DESIGN = {
    "poisson_levels": [5000.0 / (x / 4.0) ** 0.5 for x in REGISTERED_LAMBDAS],
    "total_exposure": 50000.0,
    "W": 1,
    "epochs": 50,
    "architecture": "ResNet-FCNN",
    "model_config": {"num_features": 256, "num_hidden_units": 100, "encoder_output_dim": 64},
    "peak_set_id": "C1s_adventitious",
    "noise": "exact Poisson (use_gaussian_approx=False) at every level",
    "moving_average_recipe": {
        "optimizer": "Adam", "lr": 0.001, "weight_decay": 1e-09,
        "lr_scheduler": "StepLR(step_size=25, gamma=0.5)", "loss": "HuberLoss(delta=1.0)",
        "gradient_clip_norm": 4.0, "epochs": 50, "batch_size": 32,
        "normalisation": "element-global min-max over the training stack, applied at inference "
                         "with the training constants and inverted on the output"},
    "noise2clean_recipe": {
        "optimizer": "Adam", "lr": 0.001, "weight_decay": 1e-09,
        "lr_scheduler": "StepLR(step_size=10, gamma=0.1)", "loss": "HuberLoss(delta=1.0)",
        "gradient_clip_norm": 4.0, "epochs": 50, "batch_size": 16},
    "generator_config": {
        "n_energy_points": 256, "eta": 0.3, "use_pseudo_voigt": True, "background_type": "linear",
        "background_level": 0.05, "background_slope": 0.001, "intensity_variation": 0.2,
        "position_jitter": 0.3, "width_variation": 0.1, "normalize": True,
        "energy_range": [277.8, 295.5]},
}
REGISTERED_P2A_DIFFERENCES = [
    "noise: exact Poisson here, the Gaussian approximation in P2-A",
    "training: one level here, three in P2-A",
    "test set: 512 noisy frames of ONE clean spectrum per seed here, independently generated "
    "spectra per seed in P2-A, so a seed mean averages over different things and the spread "
    "across seeds means different things",
    "this is not a re-measurement of P2-A under the same conditions",
]
P2A_RECORD = (Path(__file__).resolve().parents[1] / "position_shift" / "results"
              / "position_shift_boundary.json")
METHODS = ("moving_average", "noise2clean")
METHOD_LABELS = {"moving_average": "moving average (primary)",
                 "noise2clean": "noise2clean (baseline, descriptive)"}


class RecordDisagreement(RuntimeError):
    pass


def cell(train: float, infer: float) -> str:
    return f"{float(train)}->{float(infer)}"


def binomial_one_sided(k: int, n: int) -> float:
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2**n)


def holm_adjusted(pvalues: list) -> list:
    """Holm's step-down adjustment, written here: the i-th smallest p times (m - i), made
    monotone and capped at 1. Not `boundary_common.holm`, which produced the record."""
    m = len(pvalues)
    ranked = sorted(range(m), key=lambda j: pvalues[j])
    adjusted, running = [0.0] * m, 0.0
    for i, j in enumerate(ranked):
        running = max(running, min(1.0, (m - i) * pvalues[j]))
        adjusted[j] = running
    return adjusted


# --------------------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------------------


def per_seed(record: dict) -> dict:
    """{method: {cell: array over seeds, in seed order}} from `seeds` alone."""
    seeds = sorted(record["seeds"], key=lambda s: s["seed_index"])
    return {method: {cell(t, i): np.array([s["gains_db"][method][cell(t, i)] for s in seeds],
                                          dtype=np.float64)
                     for t in REGISTERED_LAMBDAS for i in REGISTERED_LAMBDAS}
            for method in METHODS}


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= max(TOLERANCE, abs(float(a)) * 1e-9)


def _check(problems: list, label: str, recomputed, stored) -> None:
    if stored is None or not _close(recomputed, stored):
        problems.append(f"{label}: recomputed {recomputed!r}, record says {stored!r}")


def _check_design(problems: list, record: dict) -> None:
    design = record["design"]
    if [float(x) for x in design["lambdas"]] != list(REGISTERED_LAMBDAS):
        problems.append(f"design.lambdas: {design['lambdas']!r} are not the registered levels")
    frames = {float(k): v for k, v in design["frames_per_level"].items()}
    if frames != REGISTERED_FRAMES:
        problems.append(f"design.frames_per_level: {design['frames_per_level']!r} "
                        "are not the registered frame counts")
    for key, expected in (("n_seeds", REGISTERED_SEEDS), ("n_test_per_level", REGISTERED_TEST_FRAMES),
                          ("noise2clean_pool", REGISTERED_POOL)):
        if design[key] != expected:
            problems.append(f"design.{key}: {design[key]!r}, registered {expected}")
    for key, expected in REGISTERED_DESIGN.items():
        _deep_diff(problems, f"design.{key} (registered)", expected, design.get(key))
    indices = sorted(s["seed_index"] for s in record["seeds"])
    if indices != list(range(REGISTERED_SEEDS)):
        problems.append(f"seeds: indices {indices!r} are not 0..{REGISTERED_SEEDS - 1}, once each")


def _check_aggregates(problems: list, record: dict, series: dict) -> None:
    for method in METHODS:
        stored_method = record["aggregates"][method]
        if set(stored_method) != set(series[method]):
            problems.append(f"aggregates[{method}]: cells differ from the 25 registered")
        for key, values in series[method].items():
            stored = stored_method.get(key, {})
            tag = f"aggregates[{method}][{key}]"
            _check(problems, f"{tag}.m1_mean", np.mean(values), stored.get("m1_mean"))
            _check(problems, f"{tag}.m1_sd", np.std(values, ddof=1), stored.get("m1_sd"))
            listed = stored.get("m1_per_seed") or []
            if len(listed) != len(values) or any(not _close(a, b) for a, b in zip(values, listed)):
                problems.append(f"{tag}.m1_per_seed: differs from the per-seed gains")
            train, infer = (float(x) for x in key.split("->"))
            if train == infer:
                if "m2_mean" in stored or "m2_sd" in stored:
                    problems.append(f"{tag}: a diagonal cell carries an M2")
                continue
            m2 = values - series[method][cell(infer, infer)]
            _check(problems, f"{tag}.m2_mean", np.mean(m2), stored.get("m2_mean"))
            _check(problems, f"{tag}.m2_sd", np.std(m2, ddof=1), stored.get("m2_sd"))


def independent_sign_tests(series: dict, method: str) -> dict:
    """R1 to R4 sign counts for one method, written out here from the registration."""
    lam = REGISTERED_LAMBDAS

    def m2(t, i):
        return series[method][cell(t, i)] - series[method][cell(i, i)]

    return {
        "R1": {str(x): (int(np.sum(series[method][cell(x, x)] > 0)), REGISTERED_R1_K)
               for x in REGISTERED_R1_LEVELS},
        "R2": {cell(t, i): (int(np.sum(m2(t, i) > REGISTERED_R2_MARGIN_DB)), REGISTERED_FAMILY_OF_TEN_K)
               for t in lam for i in lam if t > i},
        "R3": {cell(t, i): (int(np.sum(m2(t, i) < 0)), REGISTERED_FAMILY_OF_TEN_K)
               for t in lam for i in lam if t < i},
        "R4": {f"{a}|{b}": (int(np.sum(-m2(a, b) + m2(b, a) > 0)), REGISTERED_FAMILY_OF_TEN_K)
               for a in lam for b in lam if a < b},
    }


def _check_signs(problems: list, record: dict, series: dict) -> None:
    """Sign counts and verdicts, re-derived from the per-seed gains rather than from the
    stored counts: an edit that changes a count and its p together is still refused."""
    stored_ma = record["predictions"]
    stored_n2c = record["noise2clean_descriptive"]
    n2c_names = {"R1": "R1_cells", "R2": "R2_cells", "R3": "R3_cells", "R4": "R4_pairs"}
    for method, lookup in (("moving_average", lambda f: stored_ma[f]["cells"]),
                           ("noise2clean", lambda f: stored_n2c[n2c_names[f]])):
        for family, cells in independent_sign_tests(series, method).items():
            stored_cells = lookup(family)
            keys = list(cells)
            holm = dict(zip(keys, holm_adjusted([binomial_one_sided(cells[k][0], REGISTERED_SEEDS)
                                                 for k in keys])))
            for key, (count, k) in cells.items():
                stored = stored_cells.get(key, {})
                tag = f"{method}.{family}[{key}]"
                if stored.get("n_favouring") != count:
                    problems.append(f"{tag}.n_favouring: recomputed {count}, "
                                    f"record says {stored.get('n_favouring')!r}")
                if stored.get("k_required") != k:
                    problems.append(f"{tag}.k_required: registered {k}, "
                                    f"record says {stored.get('k_required')!r}")
                if stored.get("passed") is not (count >= k):
                    problems.append(f"{tag}.passed: recomputed {count >= k}, "
                                    f"record says {stored.get('passed')!r}")
                _check(problems, f"{tag}.one_sided_binomial_p",
                       binomial_one_sided(count, REGISTERED_SEEDS), stored.get("one_sided_binomial_p"))
                _check(problems, f"{tag}.holm_adjusted_p", holm[key], stored.get("holm_adjusted_p"))
            if method == "moving_average":
                verdict = all(count >= k for count, k in cells.values())
                if stored_ma[family].get("passed") is not verdict:
                    problems.append(f"predictions.{family}.passed: recomputed {verdict}, "
                                    f"record says {stored_ma[family].get('passed')!r}")


def _deep_diff(problems: list, path: str, expected, stored) -> None:
    """Compare two nested structures, collecting every disagreement rather than raising."""
    if isinstance(expected, dict):
        if not isinstance(stored, dict):
            problems.append(f"{path}: recomputed a mapping, record has {type(stored).__name__}")
            return
        for key in set(expected) | set(stored):
            if key not in expected:
                problems.append(f"{path}.{key}: present in the record, not in the recomputation")
            elif key not in stored:
                problems.append(f"{path}.{key}: recomputed, absent from the record")
            else:
                _deep_diff(problems, f"{path}.{key}", expected[key], stored[key])
    elif isinstance(expected, (list, tuple)):
        if not isinstance(stored, list) or len(stored) != len(expected):
            problems.append(f"{path}: sequence length differs")
            return
        for i, (a, b) in enumerate(zip(expected, stored)):
            _deep_diff(problems, f"{path}[{i}]", a, b)
    elif isinstance(expected, bool) or isinstance(stored, bool):
        if expected is not stored:
            problems.append(f"{path}: recomputed {expected!r}, record says {stored!r}")
    elif isinstance(expected, (int, float)) and isinstance(stored, (int, float)):
        if not _close(expected, stored):
            problems.append(f"{path}: recomputed {expected!r}, record says {stored!r}")
    elif expected != stored:
        problems.append(f"{path}: recomputed {expected!r}, record says {stored!r}")


def verify(record: dict) -> dict:
    """Recompute everything this script renders. Returns the per-seed series."""
    problems = []
    _check_design(problems, record)
    series = per_seed(record)
    _check_aggregates(problems, record, series)
    _check_signs(problems, record, series)
    beside = record["p2a_beside_noise2clean"]
    if beside.get("differences_stated") != REGISTERED_P2A_DIFFERENCES:
        problems.append("p2a_beside_noise2clean.differences_stated: not the registered list")
    if P2A_RECORD.is_file():
        p2a = json.loads(P2A_RECORD.read_text(encoding="utf-8"))
        _check(problems, "p2a_beside_noise2clean.p2a_arm_A_level_1000_delta_0_gain_db (P2-A's record)",
               p2a["aggregates"]["A_narrow_2304"]["1000.0"]["+0.00"]["snr_gain_db_mean"],
               beside.get("p2a_arm_A_level_1000_delta_0_gain_db"))
    _check(problems, "p2a_beside_noise2clean.noise2clean_lambda_100_diagonal_gain_db",
           np.mean(series["noise2clean"][cell(100.0, 100.0)]),
           beside.get("noise2clean_lambda_100_diagonal_gain_db"))

    seeds = sorted(record["seeds"], key=lambda s: s["seed_index"])
    _deep_diff(problems, "predictions", measurement.evaluate_predictions(seeds),
               record["predictions"])
    _deep_diff(problems, "noise2clean_descriptive",
               measurement.descriptive_statistics(seeds, "noise2clean"),
               record["noise2clean_descriptive"])

    if problems:
        raise RecordDisagreement(
            "the record disagrees with itself; nothing was rendered:\n  - "
            + "\n  - ".join(problems))
    return series


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _p(value) -> str:
    return "--" if value is None else f"{value:.3g}"


def _lam(x) -> str:
    return f"{float(x):g}"


def _grid(add, series: dict, method: str, metric: str) -> None:
    lam = REGISTERED_LAMBDAS
    add("| train λ \\ inference λ | " + " | ".join(_lam(i) for i in lam) + " |")
    add("|---|" + "---|" * len(lam))
    for t in lam:
        row = []
        for i in lam:
            values = series[method][cell(t, i)]
            if metric == "m2":
                if t == i:
                    row.append("—")
                    continue
                values = values - series[method][cell(i, i)]
            row.append(f"{np.mean(values):+.1f} ± {np.std(values, ddof=1):.1f}")
        add(f"| **{_lam(t)}** | " + " | ".join(row) + " |")
    add("")


def _family_table(add, cells: dict, value_label: str, values: dict) -> None:
    add(f"| cell | {value_label}, mean ± SD (dB) | seeds meeting the rule | required | p | Holm p | met |")
    add("|---|---|---|---|---|---|---|")
    for key, c in cells.items():
        v = values[key]
        add(f"| {key.replace('->', ' → ').replace('|', ' vs ')} | {np.mean(v):+.1f} ± {np.std(v, ddof=1):.1f} "
            f"| {c['n_favouring']}/{c['n_seeds']} | {c['k_required']} | {_p(c['one_sided_binomial_p'])} "
            f"| {_p(c['holm_adjusted_p'])} | {'yes' if c['passed'] else 'no'} |")
    add("")


def _family_values(series: dict, method: str) -> dict:
    lam = REGISTERED_LAMBDAS

    def m2(t, i):
        return series[method][cell(t, i)] - series[method][cell(i, i)]

    out = {str(x): series[method][cell(x, x)] for x in REGISTERED_R1_LEVELS}
    out.update({cell(t, i): m2(t, i) for t in lam for i in lam if t != i})
    out.update({f"{a}|{b}": -m2(a, b) + m2(b, a) for a in lam for b in lam if a < b})
    return out


def render(record: dict, series: dict) -> str:
    design = record["design"]
    provenance = record["provenance"]
    reg = provenance["registration"]
    n = len(record["seeds"])
    lines: list = []
    add = lines.append

    add("# P2-B — training and inference at different signal-to-noise ratios")
    add("")
    add("<!-- GENERATED FILE. Produced by render_report.py from results/snr_transfer.json.")
    add("     Do not edit by hand: anything written here is dropped the next time it is")
    add("     regenerated. -->")
    add("")
    add(f"Record generated (as stored) {record['generated_utc']} · record version {record['record_version']} · "
        f"{record['total_wall_clock_seconds'] / 60:.1f} min wall clock · "
        f"device `{record['environment']['device_requested_resolved_to']}`")
    add("")
    clean = "clean" if provenance["working_tree_clean"] else "DIRTY"
    add(f"Registered design: `{design['preregistration']}` (first registered "
        f"`{reg['first_commit'][:7]}`, last revised before this run at "
        f"`{reg['last_commit_before_run'][:7]}`; code run from `{provenance['code_commit'][:7]}`, "
        f"working tree {clean}). Resumed seeds: "
        f"{', '.join(map(str, record['resumed_seeds'])) or 'none'}. "
        "What the record means is stated in that document's Record section, not here.")
    add("")
    add("**What this report's guard verifies, and what it does not.** Before rendering, "
        "`render_report.py` recomputes from the per-seed gains, with arithmetic written in that "
        "file, every aggregate, every sign count, binomial p, Holm-adjusted p and verdict against "
        "the registered thresholds, and the noise2clean diagonal set beside P2-A. It checks the "
        "design block and the differences stated beside P2-A against literals written in that "
        "file, and the P2-A gain against P2-A's committed record. It also re-derives the "
        "`predictions` and `noise2clean_descriptive` trees with the measurement script's own "
        "functions and compares them field by field; that part catches an edited or stale "
        "record, not an error inside those functions. **Not verified here at all:** the per-seed "
        "gains themselves, the self-check figures, the input SNRs, the training times and "
        "normalisation constants, the generation time, the wall clock, the environment and the "
        "provenance, which are printed as stored.")
    add("")
    if record.get("quick_mode"):
        add("> **QUICK MODE.** This record was produced by a smoke test. "
            "The numbers are not meaningful and must not be quoted.")
        add("")

    add("## Design, as recorded and checked against the registered literals")
    add("")
    add(f"Synthetic `{design['peak_set_id']}` spectra, {design['generator_config']['n_energy_points']} points; "
        f"{design['noise']}. Architecture {design['architecture']} "
        f"({', '.join(f'{k}={v}' for k, v in design['model_config'].items())}). "
        f"{n} seeds; the seed is the replicate, and every ± below is the SD across seeds. "
        f"{design['n_test_per_level']} test frames per inference level per seed, the same arrays "
        "for both methods and every training level.")
    add("")
    add("| λ (expected count at the maximum) | " + " | ".join(_lam(x) for x in design["lambdas"]) + " |")
    add("|---|" + "---|" * len(design["lambdas"]))
    add("| moving-average training frames | "
        + " | ".join(str(design["frames_per_level"][str(float(x))]) for x in design["lambdas"]) + " |")
    add("| noise2clean training spectra | " + " | ".join(str(design["noise2clean_pool"])
                                                        for _ in design["lambdas"]) + " |")
    snr = [np.mean([s["input_snr_db"][str(float(x))] for s in record["seeds"]]) for x in design["lambdas"]]
    add("| input SNR, dB (as stored) | " + " | ".join(f"{v:.1f}" for v in snr) + " |")
    add("")
    add(f"**Confound, as registered:** {design['confound']}.")
    add("")

    for method in METHODS:
        add(f"## M1 — SNR gain (dB), {METHOD_LABELS[method]}")
        add("")
        _grid(add, series, method, "m1")
    add("M1 at different inference levels is measured from different input SNRs and is not a "
        "like-for-like comparison across columns; M2 compares within a column.")
    add("")
    for method in METHODS:
        add(f"## M2 — transfer penalty (dB), {METHOD_LABELS[method]}")
        add("")
        add("M1 of the cell minus M1 of the diagonal cell in the same column, within seed. "
            "Above the diagonal: training below inference; below it: training above.")
        add("")
        _grid(add, series, method, "m2")

    preds = record["predictions"]
    values = _family_values(series, "moving_average")
    add("## Registered predictions — moving average")
    add("")
    add("| | verdict | family size | seeds required per cell |")
    add("|---|---|---|---|")
    add(f"| R1 positive control at λ = 20, 45, 100 | **{'PASS' if preds['R1']['passed'] else 'FAIL'}** "
        f"| 3 | {preds['R1']['k_required']} |")
    for name, what in (("R2", "training above inference costs little (M2 > −1 dB)"),
                       ("R3", "training below inference costs (M2 < 0)"),
                       ("R4", "the asymmetry, paired within seed")):
        p = preds[name]
        if not p.get("evaluated"):
            add(f"| {name} {what} | not evaluated | — | — |")
        else:
            add(f"| {name} {what} | **{'PASS' if p['passed'] else 'FAIL'}** | {p['family_size']} "
                f"| {p['k_required']} |")
    add("")
    removed = preds.get("levels_removed_by_R1", [])
    add(f"Levels removed by R1's failure rule: {', '.join(map(_lam, removed)) or 'none'}.")
    add("")
    add("### R1 — diagonal M1 at λ = 20, 45, 100")
    add("")
    _family_table(add, preds["R1"]["cells"], "M1", values)
    add("Diagonal M1 at λ = 4 and 9, descriptive (no prediction names them): "
        + "; ".join(f"λ = {_lam(x)}: {np.mean(v):+.1f} ± {np.std(v, ddof=1):.1f} dB, positive in "
                    f"{int(np.sum(np.array(v) > 0))}/{len(v)} seeds"
                    for x, v in preds["R1"]["descriptive_diagonals"].items()) + ".")
    add("")
    add("### R2 — training above inference")
    add("")
    add("The rule is M2 > −1 dB. Each cell's own M1 and the diagonal M1 beside it, as registered: "
        "where the diagonal is weak, a cell can meet the rule by losing little relative to little.")
    add("")
    _family_table(add, preds["R2"]["cells"], "M2", values)
    add("| cell | cell M1 (dB) | diagonal M1 at the same inference λ (dB) |")
    add("|---|---|---|")
    for key, m in preds["R2"]["m1_beside_each_cell"].items():
        add(f"| {key.replace('->', ' → ')} | {m['cell_m1_mean']:+.1f} | {m['diagonal_m1_mean']:+.1f} |")
    add("")
    add("### R3 — training below inference")
    add("")
    _family_table(add, preds["R3"]["cells"], "M2", values)
    add("### R4 — the asymmetry")
    add("")
    add("Per pair a < b: −M2(a → b) − (−M2(b → a)), paired within seed; the rule is that it is positive.")
    add("")
    _family_table(add, preds["R4"]["cells"], "difference", values)

    d = record["noise2clean_descriptive"]
    n2c_values = _family_values(series, "noise2clean")
    add("## noise2clean — the same statistics, descriptive only")
    add("")
    add("None of these is a prediction. They are computed for every cell and pair, with no "
        "exclusion and no stopping rule.")
    add("")
    for label, key in (("R1-type cells", "R1_cells"), ("R2-type cells", "R2_cells"),
                       ("R3-type cells", "R3_cells"), ("R4-type pairs", "R4_pairs")):
        met = sum(c["passed"] for c in d[key].values())
        add(f"### {label} — rule met in {met} of {len(d[key])}")
        add("")
        _family_table(add, d[key], "M1" if key == "R1_cells" else ("difference" if key == "R4_pairs" else "M2"),
                      n2c_values)

    beside = record["p2a_beside_noise2clean"]
    add("## noise2clean at λ = 100 beside P2-A's operating point — descriptive, no tolerance")
    add("")
    if beside.get("available"):
        add(f"noise2clean diagonal M1 at λ = 100: {beside['noise2clean_lambda_100_diagonal_gain_db']:+.1f} dB. "
            f"P2-A, arm A, level 1000, Δ = 0, as copied into this record: "
            f"{beside['p2a_arm_A_level_1000_delta_0_gain_db']:+.1f} dB. They differ in:")
        add("")
        for item in beside["differences_stated"]:
            add(f"- {item}")
    else:
        add("P2-A's record was not available when this record was written.")
    add("")

    add("## Self-checks, as stored (not verified by this report's guard)")
    add("")
    add("The run refuses to write a record if any self-check fails; this record was written, so "
        "all passed. Their stored figures, over all seeds:")
    add("")
    seeds = record["seeds"]
    resid = max(x["worst_integer_residual"] for s in seeds
                for k in ("1a_exact_poisson_frames", "1b_exact_poisson_pool") for x in s["self_checks"][k])
    ratios = [x["lambda_from_variance"] / x["lambda"] for s in seeds for x in s["self_checks"]["2_noise_level"]]
    params = sorted({x["parameters"] for s in seeds for x in s["self_checks"]["7_parameter_count"]})
    leaks = sum(s["self_checks"]["5_no_leakage"]["test_frames_in_training"] for s in seeds)
    pooled = sum(len(s["self_checks"]["5_no_leakage"]["sample_in_pool"]) for s in seeds)
    checked = sorted({s["self_checks"]["6_same_test_arrays"]["cells_checked"] for s in seeds})
    add(f"- 1 exact Poisson: worst distance from a whole count, frames and pools, {resid:.2g}")
    add(f"- 2 noise level: λ from the variance over declared λ, {min(ratios):.3f} to {max(ratios):.3f} "
        f"(tolerance a factor {measurement.LEVEL_TOLERANCE})")
    add("- 3 equal exposure: frames per level "
        + ", ".join(f"λ = {k}: {v}" for k, v in seeds[0]["self_checks"]["3_equal_exposure"]
                    ["frames_per_level"].items()) + " (seed 0; every seed passed)")
    add(f"- 4 targets: every moving-average target row checked against its neighbours "
        f"({sum(v['targets']['rows_checked'] for v in seeds[0]['training']['moving_average'].values())} "
        "rows per seed)")
    add(f"- 5 no leakage: test frames found in training, {leaks}; seed samples found in a pool, {pooled}")
    add(f"- 6 model inputs: cells checked per seed, {', '.join(map(str, checked))}")
    add(f"- 7 architecture: parameter counts, {', '.join(map(str, params))}")
    add("")
    add("## Training time per model (s), as stored, mean over seeds")
    add("")
    add("| method | " + " | ".join(f"λ = {_lam(x)}" for x in design["lambdas"]) + " |")
    add("|---|" + "---|" * len(design["lambdas"]))
    for method in METHODS:
        add(f"| {method} | " + " | ".join(
            f"{np.mean([s['training'][method][str(float(x))]['train_seconds'] for s in seeds]):.0f}"
            for x in design["lambdas"]) + " |")
    add("")
    env = record["environment"]
    add("## Environment, as stored")
    add("")
    add(f"Python {env['python']}, torch {env['torch']}, numpy {env['numpy']}, scipy {env['scipy']}, "
        f"{env['platform']}; lockfile `{provenance['lockfile']['path']}` sha256 "
        f"`{provenance['lockfile']['sha256'][:12]}…`; in a virtual environment: "
        f"{provenance['in_virtual_environment']}.")
    add("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--write", action="store_true", help=f"write {DEFAULT_REPORT.name}")
    args = parser.parse_args(argv)
    record = json.loads(args.record.read_text(encoding="utf-8"))
    series = verify(record)
    text = render(record, series)
    if args.write:
        DEFAULT_REPORT.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
