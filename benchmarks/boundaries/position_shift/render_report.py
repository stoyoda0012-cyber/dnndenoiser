"""Regenerate `report.md` FROM the P2-A record, recomputing everything it renders.

The rule this file exists to enforce is the reference benchmark's: **numbers are never
typed by hand**, and nor is anything else in the generated report. Every line of
`report.md` comes out of here, so a paragraph added to that file by hand is dropped the
next time it is regenerated.

Before rendering anything this script rebuilds, from the record's `runs` array alone,
everything downstream of it -- every aggregate field, the whole `boundaries` tree and the
whole `predictions` tree including each verdict, sign count, test statistic and
Holm-adjusted p -- and refuses to render if any of it disagrees with what the record
stores. It reports every disagreement together rather than stopping at the first, because
stopping at the first hides how much of a record is wrong.

What each half establishes, stated because an earlier version of this docstring claimed
more than the code did. The aggregate half is an INDEPENDENT recomputation: the arithmetic
is written here, not imported. The boundaries and predictions half is a CONSISTENCY check
-- it imports the measurement script's own `boundary_table` and `evaluate_predictions` and
re-derives those trees from `runs`, so it catches a record that has been edited, truncated
or left stale relative to its raw data, but it cannot catch an error inside those
functions. The self-check figures, the environment and the consistency anchor are not
derivable from `runs` at all; they are rendered as stored and are NOT verified here, and
the report says so.

An independent audit tamper-tested the previous guard field by field: it accepted 25 of
30 edits, including flipping a prediction's PASS to FAIL, rewriting a boundary median and
changing a Holm p from 1.3e-44 to 0.9. The verdict table -- the most consequential thing
this file prints -- was copied from the record unverified.
`tests/test_position_shift_boundary_record.py` now performs that tamper test on every
field this guard claims to cover.

Usage:

    python benchmarks/boundaries/position_shift/render_report.py            # to stdout
    python benchmarks/boundaries/position_shift/render_report.py --write    # to report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import position_shift_boundary as measurement  # noqa: E402

DEFAULT_RECORD = Path(__file__).resolve().parent / "results" / "position_shift_boundary.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "report.md"
TOLERANCE = 1e-9

ARM_LABELS = {
    "A_narrow_2304": "A narrow (N=2304)",
    "B_augmented_2304": "B augmented ±1.5 eV (N=2304)",
    "C_narrow_461": "C narrow (N=461)",
    "D_narrow_144": "D narrow (N=144)",
}


class RecordDisagreement(RuntimeError):
    pass


def dkey(delta: float) -> str:
    return f"{delta + 0.0:+.2f}"


def binomial_one_sided(k: int, n: int) -> float:
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2**n)


# --------------------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------------------


def recompute_from_runs(record: dict) -> dict:
    """Rebuild the per-(arm, level, delta) per-seed series from `runs` alone."""
    series = {}
    for entry in record["runs"]:
        key = (entry["arm"], str(entry["level"]), dkey(entry["delta"]))
        series.setdefault(key, []).append(
            (entry["seed_index"], entry["snr_gain_db_mean"],
             entry["argmax_displacement_ev_mean"]))
    return {
        key: {
            "gain": np.array([g for _s, g, _d in sorted(v)], dtype=np.float64),
            "disp": np.array([d for _s, _g, d in sorted(v)], dtype=np.float64),
        }
        for key, v in series.items()
    }


def _disagree(problems, label, recomputed, stored):
    if stored is None and recomputed is None:
        return
    if stored is None or recomputed is None:
        problems.append(f"{label}: recomputed {recomputed!r}, record says {stored!r}")
        return
    if abs(float(recomputed) - float(stored)) > TOLERANCE:
        problems.append(f"{label}: recomputed {recomputed:.10g}, record says {stored:.10g}")


def verify(record: dict) -> dict:
    """Recompute everything this script renders. Returns the recomputed series."""
    problems = []
    series = recompute_from_runs(record)
    design = record["design"]
    deltas = design["manipulated"]["delta_values_eV"]
    levels = design["noise_model"]["levels"]

    for arm, per_level in record["aggregates"].items():
        for level, per_delta in per_level.items():
            baseline = series[(arm, level, dkey(0.0))]
            for key, stored in per_delta.items():
                got = series[(arm, level, key)]
                tag = f"aggregates[{arm}][{level}][{key}]"
                _disagree(problems, f"{tag}.snr_gain_db_mean",
                          float(np.mean(got["gain"])), stored["snr_gain_db_mean"])
                _disagree(problems, f"{tag}.snr_gain_db_sd_across_seeds",
                          float(np.std(got["gain"], ddof=1)) if len(got["gain"]) > 1 else None,
                          stored["snr_gain_db_sd_across_seeds"])
                _disagree(problems, f"{tag}.degradation_db_mean",
                          float(np.mean(got["gain"] - baseline["gain"])),
                          stored["degradation_db_mean"])
                _disagree(problems, f"{tag}.argmax_displacement_bias_corrected_ev_mean",
                          float(np.mean(got["disp"] - baseline["disp"])),
                          stored["argmax_displacement_bias_corrected_ev_mean"])
                if stored["n_seeds"] != len(got["gain"]):
                    problems.append(f"{tag}.n_seeds: recomputed {len(got['gain'])}, "
                                    f"record says {stored['n_seeds']}")

    # Everything downstream of `runs`: the whole boundaries and predictions trees,
    # re-derived with the measurement script's own functions and deep-diffed. This is
    # what catches an edited verdict, a rewritten median or a doctored p-value -- none
    # of which the previous field-by-field guard looked at.
    indexed = measurement.index_runs(record["runs"])
    rebuilt_boundaries = {
        str(level): measurement.boundary_table(indexed["snr_gain_db_mean"], str(level))
        for level in levels
    }
    rebuilt_predictions = measurement.evaluate_predictions(indexed, rebuilt_boundaries)
    for level_table in rebuilt_boundaries.values():
        for arm_table in level_table.values():
            for direction_summary in arm_table.values():
                direction_summary.pop("_per_seed_raw", None)

    _deep_diff(problems, "boundaries", rebuilt_boundaries, record["boundaries"])
    _deep_diff(problems, "predictions", rebuilt_predictions, record["predictions"])
    _deep_diff(problems, "suspect_run_rule",
               measurement.suspect_run_rule(rebuilt_predictions), record["suspect_run_rule"])

    if problems:
        raise RecordDisagreement(
            "the record disagrees with itself; nothing was rendered:\n  - "
            + "\n  - ".join(problems))
    return {"series": series, "levels": levels, "deltas": deltas,
            "positive_abs": sorted({abs(d) for d in deltas})}


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
        if bool(expected) != bool(stored):
            problems.append(f"{path}: recomputed {expected!r}, record says {stored!r}")
    elif isinstance(expected, (int, float)) and isinstance(stored, (int, float)):
        if expected != expected and stored != stored:
            return
        if abs(float(expected) - float(stored)) > max(TOLERANCE, abs(float(expected)) * 1e-9):
            problems.append(f"{path}: recomputed {expected!r}, record says {stored!r}")
    elif expected != stored:
        problems.append(f"{path}: recomputed {expected!r}, record says {stored!r}")


def _sign_holders(prediction: dict):
    if "sign" in prediction:
        yield prediction, ""
    for container in ("per_direction", "orderings", "sign_points"):
        for key, value in (prediction.get(container) or {}).items():
            if isinstance(value, dict) and "sign" in value:
                yield value, f".{container}[{key}]"


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _registration_line(record: dict) -> str:
    provenance = record.get("provenance")
    if provenance:
        reg = provenance["registration"]
        clean = "clean" if provenance["working_tree_clean"] else "DIRTY"
        return (f"(first registered `{(reg['first_commit'] or '?')[:7]}`, last revised before "
                f"this run at `{(reg['last_commit_before_run'] or '?')[:7]}`; code run from "
                f"`{provenance['code_commit'][:7]}`, working tree {clean})")
    commits = record["design"]["preregistration"].get("commits", {})
    return ("(an older record: its registration commits were a hand-typed list, "
            f"{', '.join(commits.values())}, which omits later revisions)")


def _fmt(value, digits=2, dash="--"):
    return dash if value is None else f"{value:.{digits}f}"


def _p(value):
    """p-values to three significant figures. Four fixed decimals printed every p below
    5e-5 as 0.0000, so the exact binomial 9.5e-7 and a t-test's 1.3e-44 looked the same."""
    return "--" if value is None else f"{value:.3g}"


def render(record: dict, computed: dict) -> str:
    design = record["design"]
    level = str(design["inference"]["primary_level"])
    n_seeds = design["inference"]["n_seeds"]
    arms = list(record["aggregates"])
    lines = []
    add = lines.append

    add("# P2-A — the position-shift boundary, and what augmentation does to it")
    add("")
    add("<!-- GENERATED FILE. Produced by render_report.py from")
    add(f"     results/{Path(record['script']).stem}.json. Do not edit by hand:")
    add("     anything written here is dropped the next time it is regenerated. -->")
    add("")
    add(f"Record generated {record['generated_utc']} · "
        f"record version {record['record_version']} · "
        f"{record['total_wall_clock_seconds'] / 60:.1f} min wall clock")
    add("")
    add(f"Registered design: `{design['preregistration']['document']}` "
        + _registration_line(record) + ". "
        "Predictions were fixed before implementation.")
    add("")
    add("**What this report's guard verifies, and what it does not.** Before rendering, "
        "`render_report.py` recomputes every aggregate from the raw runs independently, and "
        "re-derives the `boundaries` and `predictions` trees from the raw runs with the "
        "measurement script's own functions — so an edited or stale record is refused, but an "
        "error *inside* those functions would be reproduced, not caught. **Not verified here "
        "at all:** the self-check figures, the environment, the consistency anchor and the "
        "M3 smoother comparator, which are not derivable from the raw runs and are printed "
        "as stored. Each section below that prints one of those says so.")
    add("")
    if record.get("quick_mode"):
        add("> **QUICK MODE.** This record was produced by a smoke test. "
            "The numbers are not meaningful and must not be quoted.")
        add("")

    add("## What was manipulated")
    add("")
    add("A " + design["manipulated"]["variable"] + ", with the energy grid held fixed.")
    add("")
    add("**Not in scope:** " + design["manipulated"]["not_in_scope"])
    add("")
    add("| Arm | N | Training shift | Per-peak jitter |")
    add("|---|---|---|---|")
    for arm in arms:
        spec = design["arms"][arm]
        shift = ("none" if spec["augmentation_halfwidth_eV"] == 0
                 else f"U(±{spec['augmentation_halfwidth_eV']}) eV, rigid, per sample")
        add(f"| {ARM_LABELS.get(arm, arm)} | {spec['n_train']} | {shift} "
            f"| ±{spec['per_peak_jitter_eV']} eV |")
    add("")
    add(design["why_density_controls_exist"])
    add("")

    add("## The boundary")
    add("")
    add(f"|Δ|\\* is the first crossing of zero mean SNR gain, per seed, "
        f"median over {n_seeds} seeds. A censored seed never crossed inside the tested "
        f"range and enters the order statistics at its bound; it is never dropped.")
    add("")
    add("| Arm | level | direction | median \\|Δ\\|\\* (eV) | censored | sustained (eV) | re-crossing seeds |")
    add("|---|---|---|---|---|---|---|")
    for level_key in record["boundaries"]:
        for arm in arms:
            for direction in ("positive", "negative"):
                summary = record["boundaries"][level_key][arm][direction]
                if not summary.get("defined"):
                    add(f"| {ARM_LABELS.get(arm, arm)} | {level_key} | {direction} "
                        f"| not defined | — | — | — |")
                    continue
                median = summary.get("median_first_crossing_eV")
                shown = _fmt(median) if median is not None else f"> {summary['censoring_bound_eV']}"
                add(f"| {ARM_LABELS.get(arm, arm)} | {level_key} | {direction} | {shown} "
                    f"| {summary['n_censored']}/{summary['n_seeds_defined']} "
                    f"| {_fmt(summary.get('median_sustained_crossing_eV'))} "
                    f"| {summary['n_seeds_re_crossing']} |")
    add("")
    add("Where the mean gain at Δ = 0 is already negative there is no boundary to find, "
        "and the row says `not defined` rather than reporting Δ = 0.")
    add("")

    add(f"## SNR gain against shift, level {level} (primary)")
    add("")
    add("Mean ± across-seed SD in dB. The replicate is the seed.")
    add("")
    add("| Δ (eV) | " + " | ".join(ARM_LABELS.get(a, a) for a in arms) + " |")
    add("|---" * (len(arms) + 1) + "|")
    for delta in computed["deltas"]:
        cells = []
        for arm in arms:
            stored = record["aggregates"][arm][level][dkey(delta)]
            cells.append(f"{stored['snr_gain_db_mean']:+.2f} ± "
                         f"{_fmt(stored['snr_gain_db_sd_across_seeds'])}")
        add(f"| {delta:+.2f} | " + " | ".join(cells) + " |")
    add("")

    add("## Registered predictions")
    add("")
    add(f"All evaluated at level {level} only. Sign rules are one-sided because every "
        f"prediction names its direction in advance; the uniform threshold is "
        f"{design['inference']['sign_rule']['k']}/{design['inference']['sign_rule']['n']} "
        f"(one-sided p = {_p(design['inference']['one_sided_p_of_sign_rule'])}), and the "
        f"positive control's is "
        f"{design['inference']['sign_rule_positive_control']['k']}/"
        f"{design['inference']['sign_rule_positive_control']['n']}.")
    add("")
    add("| | Verdict | Statement |")
    add("|---|---|---|")
    for name, prediction in record["predictions"].items():
        # R6 carries a three-valued verdict. "undecided" -- its augmented arm having no
        # crossing inside the tested range -- is explicitly NOT a falsification in the
        # registered design, and must never render as FAIL.
        three_way = prediction.get("verdict")
        if three_way in ("undecided", "not defined"):
            verdict = f"**{three_way.upper()}**"
        else:
            verdict = "**PASS**" if prediction["passed"] else "**FAIL**"
        add(f"| {name} | {verdict} | {prediction['statement']} |")
    add("")
    add("### Evidence")
    add("")
    for name, prediction in record["predictions"].items():
        add(f"**{name}** — {prediction['statement']}  ")
        for holder, label in _sign_holders(prediction):
            sign = holder["sign"]
            stats = holder.get("stats", {})
            pieces = [f"{sign['n_favouring']}/{sign['n_seeds']} seeds",
                      f"one-sided p = {_p(sign['one_sided_binomial_p'])}"]
            if stats.get("mean") is not None:
                pieces.insert(0, f"mean {stats['mean']:+.3f}")
            if stats.get("cohens_dz") is not None:
                pieces.append(f"d_z = {stats['cohens_dz']:+.2f}")
            if holder.get("holm_adjusted_p") is not None:
                pieces.append(f"Holm p = {_p(holder['holm_adjusted_p'])}")
            add(f"&nbsp;&nbsp;`{label or '.'}` " + " · ".join(pieces) + "  ")
        if name == "R3":
            for direction, value in prediction["per_direction"].items():
                add(f"&nbsp;&nbsp;`{direction}` median |Δ|* = "
                    f"{_fmt(value['median_first_crossing_eV'])} eV vs threshold "
                    f"{prediction['threshold_eV']} eV, {value['n_censored']} censored  ")
        if name == "R6":
            for direction, value in prediction["per_direction"].items():
                if value.get("defined"):
                    add(f"&nbsp;&nbsp;`{direction}` verdict **{value['verdict']}** — "
                        f"arm A {_fmt(value['arm_A_median_eV'])} eV, arm B "
                        f"{_fmt(value['arm_B_median_eV'])} eV, "
                        f"{value['n_b_larger']}/{value['n_compared']} seeds larger  ")
            if any(v.get("verdict") == "beyond range"
                   for v in prediction["per_direction"].values()):
                add(f"&nbsp;&nbsp;*{prediction['three_way_verdict']['beyond range']}*  ")
        if name == "R7":
            for direction, value in prediction["monotonicity"].items():
                add(f"&nbsp;&nbsp;`{direction}` |displacement| at |Δ| = 1, 1.5, 2, 3, 4: "
                    + ", ".join(f"{v:.3f}" for v in value["abs_displacement_eV_at_1_1p5_2_3_4"])
                    + f" eV; largest decrease {max(0.0, value['worst_violation_eV']):.4f} eV "
                    f"against a tolerance of one grid step "
                    f"({value['tolerance_one_grid_step_eV']:.6f} eV)  ")
        add("")

    if record["suspect_run_rule"]["suspect"]:
        add("> **SUSPECT RUN.** R2 and R7 both failed while R1 passed. "
            + record["suspect_run_rule"]["what_it_would_indicate"] + ".")
        add("")

    add("## M3 — argmax displacement, and what it is not")
    add("")
    add(record["design"]["metrics"]["M3_argmax_displacement_ev"]["why_bias_corrected"])
    add("")
    add("### Comparator (iii) — arm B on the identical test arrays")
    add("")
    add(f"Bias-corrected displacement `disp(Δ) − disp(0)` in eV, level {level}, mean over "
        "seeds. Registered in advance: a *structural* window effect would give arms A and B "
        "the same profile; a learned position prior would leave arm B near zero inside its "
        "training range. These values come from `aggregates` and are covered by the guard.")
    add("")
    add("| Δ (eV) | arm A | arm B |")
    add("|---|---|---|")
    for delta in computed["deltas"]:
        a = record["aggregates"]["A_narrow_2304"][level][dkey(delta)]
        b = record["aggregates"]["B_augmented_2304"][level][dkey(delta)]
        add(f"| {delta:+.2f} | {a['argmax_displacement_bias_corrected_ev_mean']:+.4f} "
            f"| {b['argmax_displacement_bias_corrected_ev_mean']:+.4f} |")
    add("")

    comparators = record["diagnostics"]["m3_comparators_seed_0"]
    if comparators:
        add("### Comparator (ii) — a learning-free Gaussian smoother, every level")
        add("")
        add("Mean displacement in eV on this run's own test spectra, **seed 0 only**. A "
            "diagnostic: **not verified by this report's guard**. Read the span across shifts, "
            "not the level: a flat column means the smoother's offset does not depend on the "
            "shift at that noise level.")
        add("")
        for level_key, table in comparators.items():
            sigmas = [k for k in next(iter(table.values())) if k.startswith("sigma")]
            spans = {k: max(r[k] for r in table.values()) - min(r[k] for r in table.values())
                     for k in sigmas + ["noisy_input"]}
            add(f"**Level {level_key}** — span across all shifts: "
                + ", ".join(f"{k} {v:.4f}" for k, v in spans.items()) + " eV")
            add("")
            add("| Δ (eV) | " + " | ".join(sigmas) + " | noisy input |")
            add("|---" * (len(sigmas) + 2) + "|")
            for delta in computed["deltas"]:
                row = table.get(dkey(delta), {})
                add(f"| {delta:+.2f} | "
                    + " | ".join(f"{row.get(sg, float('nan')):+.3f}" for sg in sigmas)
                    + f" | {row.get('noisy_input', float('nan')):+.3f} |")
            add("")

    add("## Self-checks")
    add("")
    add("Thirteen gates, each with a stated failure condition, each voiding the record. "
        "Diagnostics are listed separately below because they cannot fail. The figures in "
        "this section are stored, not recomputed: they are not derivable from the raw runs, "
        "and `render_report.py`'s guard does not cover them.")
    add("")
    checks = record["self_checks"]
    per_seed = checks["4_8_10_12_per_seed"]
    add("| Check | Result |")
    add("|---|---|")
    add(f"| 1 parameter count | {checks['1_parameter_count']['observed']} "
        f"(expected read from the reference record) |")
    add(f"| 2 grid invariance | bit-identical at all "
        f"{checks['2_and_3_grid_and_test_sweep_rigidity']['n_deltas_checked']} shifts |")
    add("| 3 test-sweep peak-set construction | every centre equals its literal plus Δ. "
        "A run-time unit test of the peak-set constructor and of the registry staying "
        "unmutated — it does not inspect generated spectra; check 8b does that |")
    add(f"| 4 training-pool rigidity | "
        f"{per_seed[0]['4_training_pool_rigidity'][arms[0]]['n_reconstructed_and_compared']} "
        f"spectra per arm per seed rebuilt from the literal peaks, the recorded shift and "
        f"the replayed draws, and required bit-identical to the pool |")
    add(f"| 5 truncation | absolute retention at Δ=0 "
        f"{checks['5_truncation']['absolute_retained_fraction_at_delta_zero'] * 100:.3f} %; "
        f"every shift within {checks['5_truncation']['tolerance_total_relative']:.0%} of it, "
        f"each peak within {checks['5_truncation']['tolerance_per_peak_relative']:.0%} |")
    add("| 6 input-SNR invariance | worst span "
        + ", ".join(f"{k} → {v['max']:.3f} dB"
                    for k, v in checks["6_input_snr_invariance"]["measured_span_db_per_level"].items())
        + " |")
    add(f"| 7 translation equivariance | worst residual "
        f"{checks['7_translation_equivariance']['worst_residual']:.5f} of peak height against "
        f"a tolerance of {checks['7_translation_equivariance']['tolerance']}, compared with "
        f"`np.roll` at integer grid offsets |")
    add(f"| 8 pairing integrity | "
        f"{per_seed[0]['8_pairing_integrity']['n_samples_compared']} samples per seed, keyed on "
        f"{per_seed[0]['8_pairing_integrity']['arms_keyed_on']} |")
    add(f"| 8b test-family rigidity | "
        f"{per_seed[0]['8b_test_family_rigidity_spectra_checked']} test spectra per seed "
        f"rebuilt from the shift-independent family seed and required bit-identical |")
    add(f"| 9 argmax well-posedness and reference identity | worst fraction "
        f"{checks['9_argmax_well_posedness_and_reference_identity']['worst_fraction_observed']:.4f} "
        f"against {checks['9_argmax_well_posedness_and_reference_identity']['compared_against']}; "
        f"identity asserted against the array the metric received |")
    aug = per_seed[0]["10_augmentation"]["B_augmented_2304"]
    add(f"| 10 augmentation | narrow arms all-zero; augmented SD {aug['sd']:.4f} against a "
        f"uniform {aug['expected_uniform_sd']:.4f}, "
        f"{aug['deciles_occupied_of_10']}/10 deciles occupied |")
    add(f"| 11 noise-model identity | field by field against the literals; "
        f"{checks['11_noise_model_identity']['note']} |")
    add(f"| 12 leakage | 0 identical spectra across "
        f"{len(per_seed[0]['12_leakage']['arms_hashed'])} arms; near-duplicate ratio "
        f"{per_seed[0]['12_leakage']['near_duplicate_analysis']['ratio_median_test_over_train']:.3f} |")
    add("")
    determinism = record["diagnostics"].get("repeat_run_determinism")
    if determinism:
        add(f"**Diagnostic — run-to-run determinism.** Arm A trained twice on identical "
            f"inputs at one seed: {determinism['absolute_difference_db']:.6f} dB apart. "
            f"Recorded, never asserted.")
        add("")

    anchor = record["consistency_anchor"]
    add("## Consistency anchor")
    add("")
    add("Arm A at Δ = 0 is the reference benchmark's own primary condition for this "
        "architecture, re-drawn with this script's seeding. " + anchor["a_flag_is_not_a_failure"]
        + ". **Printed as stored; not verified by this report's guard.**")
    add("")
    add("| level | reference (dB) | here (dB) | difference | flag threshold | flagged |")
    add("|---|---|---|---|---|---|")
    for level_key, value in anchor["per_level"].items():
        add(f"| {level_key} | {value['reference_snr_gain_db_mean']:+.3f} "
            f"| {value['here_arm_A_delta_zero']:+.3f} | {value['difference_db']:+.3f} "
            f"| {_fmt(value['flag_threshold_db'], 3)} "
            f"| {'yes' if value['flagged'] else 'no'} |")
    add("")

    add("## What this record does not support")
    add("")
    for item in record["claim_scope"]["does_not_support"]:
        add(f"- {item}")
    add("")
    add(f"**Device dependence.** {record['claim_scope']['device_dependence']}")
    add("")
    add(f"**Descriptive-only rule.** {record['claim_scope']['descriptive_only_rule']}")
    add("")
    add(f"**{record['claim_scope']['denoised_output_is_a_model_estimate']}**")
    add("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--write", action="store_true", help=f"write {DEFAULT_REPORT.name}")
    args = parser.parse_args(argv)

    record = json.loads(args.record.read_text(encoding="utf-8"))
    computed = verify(record)
    text = render(record, computed)
    if args.write:
        DEFAULT_REPORT.write_text(text, encoding="utf-8")
        print(f"wrote {DEFAULT_REPORT}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
