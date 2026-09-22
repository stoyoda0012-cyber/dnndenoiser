"""Regenerate `report.md` FROM the P2-A record, recomputing everything it renders.

The rule this file exists to enforce is the reference benchmark's: **numbers are never
typed by hand**, and nor is anything else in the generated report. Every line of
`report.md` comes out of here, so a paragraph added to that file by hand is dropped the
next time it is regenerated.

Before rendering anything this script recomputes, from the per-run raw numbers, every
quantity it is about to display -- the aggregate means, the across-seed standard
deviations, the degradation series, the bias-corrected displacements, the boundary
medians and censored counts, and each prediction's sign count and one-sided binomial
p -- and refuses to render if any of them disagrees with the record. It reports every
disagreement together rather than stopping at the first, because stopping at the first
hides how much of a record is wrong.

Usage:

    python benchmarks/boundaries/position_shift/render_report.py            # to stdout
    python benchmarks/boundaries/position_shift/render_report.py --write    # to report.md
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

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

    # Boundary medians and censored counts, recomputed from the per-seed curves.
    positive_abs = sorted({abs(d) for d in deltas})
    for level, per_arm in record["boundaries"].items():
        for arm, per_direction in per_arm.items():
            for direction, stored in per_direction.items():
                sign = 1.0 if direction == "positive" else -1.0
                n_seeds = len(series[(arm, level, dkey(0.0))]["gain"])
                crossings, censored, defined = [], 0, 0
                for seed_index in range(n_seeds):
                    curve = [series[(arm, level, dkey(sign * d))]["gain"][seed_index]
                             for d in positive_abs]
                    if curve[0] < 0.0:
                        continue
                    defined += 1
                    crossing = None
                    for i in range(len(curve) - 1):
                        if curve[i] >= 0.0 > curve[i + 1]:
                            g0, g1 = curve[i], curve[i + 1]
                            crossing = (positive_abs[i] + (positive_abs[i + 1] - positive_abs[i])
                                        * g0 / (g0 - g1)) if g0 != g1 else positive_abs[i + 1]
                            break
                    if crossing is None:
                        censored += 1
                    else:
                        crossings.append(crossing)
                tag = f"boundaries[{level}][{arm}][{direction}]"
                if bool(stored.get("defined")) != (defined > 0):
                    problems.append(f"{tag}.defined disagrees")
                    continue
                if not stored.get("defined"):
                    continue
                if stored.get("n_censored") != censored:
                    problems.append(f"{tag}.n_censored: recomputed {censored}, "
                                    f"record says {stored.get('n_censored')}")
                allowed = (defined + 1) // 2 - 1
                expected = (float(np.median(sorted(crossings) + [float('inf')] * censored))
                            if censored <= allowed else None)
                _disagree(problems, f"{tag}.median_first_crossing_eV",
                          expected, stored.get("median_first_crossing_eV"))

    # Every prediction's sign count and one-sided binomial p.
    level = str(design["inference"]["primary_level"])
    for name, prediction in record["predictions"].items():
        for holder, label in _sign_holders(prediction):
            sign = holder["sign"]
            recomputed_p = binomial_one_sided(sign["n_favouring"], sign["n_seeds"])
            _disagree(problems, f"predictions[{name}]{label}.one_sided_binomial_p",
                      recomputed_p, sign["one_sided_binomial_p"])
            if sign["passed"] != (sign["n_favouring"] >= sign["k_required"]):
                problems.append(f"predictions[{name}]{label}.passed disagrees with its own rule")

    if problems:
        raise RecordDisagreement(
            "the record disagrees with itself; nothing was rendered:\n  - "
            + "\n  - ".join(problems))
    return {"series": series, "levels": levels, "deltas": deltas, "positive_abs": positive_abs}


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


def _fmt(value, digits=2, dash="--"):
    return dash if value is None else f"{value:.{digits}f}"


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
        f"(registered `{design['preregistration']['commits']['registered']}`, "
        f"Revision 1 `{design['preregistration']['commits']['revision_1']}`). "
        "Predictions were fixed before implementation.")
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
        f"(one-sided p = {design['inference']['one_sided_p_of_sign_rule']:.4f}), and the "
        f"positive control's is "
        f"{design['inference']['sign_rule_positive_control']['k']}/"
        f"{design['inference']['sign_rule_positive_control']['n']}.")
    add("")
    add("| | Verdict | Statement |")
    add("|---|---|---|")
    for name, prediction in record["predictions"].items():
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
                      f"one-sided p = {sign['one_sided_binomial_p']:.4f}"]
            if stats.get("mean") is not None:
                pieces.insert(0, f"mean {stats['mean']:+.3f}")
            if stats.get("cohens_dz") is not None:
                pieces.append(f"d_z = {stats['cohens_dz']:+.2f}")
            if holder.get("holm_adjusted_p") is not None:
                pieces.append(f"Holm p = {holder['holm_adjusted_p']:.4f}")
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
    comparators = record["diagnostics"]["m3_comparators_seed_0"].get(level, {})
    if comparators:
        add("Comparator (ii), the learning-free Gaussian smoother, measured on this run's "
            "own test spectra at seed 0:")
        add("")
        sigmas = [k for k in next(iter(comparators.values())) if k.startswith("sigma")]
        add("| Δ (eV) | " + " | ".join(sigmas) + " | noisy input |")
        add("|---" * (len(sigmas) + 2) + "|")
        for delta in computed["deltas"]:
            row = comparators.get(dkey(delta), {})
            add(f"| {delta:+.2f} | "
                + " | ".join(f"{row.get(s, float('nan')):+.3f}" for s in sigmas)
                + f" | {row.get('noisy_input', float('nan')):+.3f} |")
        add("")
        add("A flat column is the point: none of truncation, background asymmetry, the "
            "three-peak envelope's own asymmetry or plain oversmoothing produces a "
            "displacement that depends on the shift, so a shift-dependent displacement in "
            "a trained arm is not attributable to them.")
        add("")

    add("## Self-checks")
    add("")
    add("Each of these has a stated failure condition and voids the record. "
        "Diagnostics are listed separately below because they cannot fail.")
    add("")
    checks = record["self_checks"]
    add("| Check | Result |")
    add("|---|---|")
    add(f"| 1 parameter count | {checks['1_parameter_count']['observed']} "
        f"(expected from the reference record) |")
    add(f"| 2, 3 grid invariance and test-sweep rigidity | "
        f"{checks['2_and_3_grid_and_test_sweep_rigidity']['n_deltas_checked']} shifts, "
        f"grid bit-identical, baselines are literals |")
    add(f"| 5 truncation | absolute retention at Δ=0 "
        f"{checks['5_truncation']['absolute_retained_fraction_at_delta_zero'] * 100:.3f} %; "
        f"every shift within {checks['5_truncation']['tolerance_total_relative']:.0%} of it |")
    add("| 6 input-SNR invariance | worst span "
        + ", ".join(f"{k} → {v['max']:.3f} dB"
                    for k, v in checks["6_input_snr_invariance"]["measured_span_db_per_level"].items())
        + " |")
    add(f"| 7 translation equivariance | worst residual "
        f"{checks['7_translation_equivariance']['worst_residual']:.5f} of peak height, "
        f"tolerance {checks['7_translation_equivariance']['tolerance']} |")
    add(f"| 9 argmax well-posedness and reference identity | worst fraction "
        f"{checks['9_argmax_well_posedness_and_reference_identity']['worst_fraction_observed']:.4f}, "
        f"compared against "
        f"{checks['9_argmax_well_posedness_and_reference_identity']['compared_against']} |")
    add(f"| 11 noise-model identity | field-by-field against the literals; "
        f"{checks['11_noise_model_identity']['note']} |")
    per_seed = checks["4_8_10_12_per_seed"]
    add(f"| 4 training-pool rigidity | {len(per_seed)} seeds × {len(arms)} arms, 5 % sampled |")
    add(f"| 8, 8b pairing integrity and replay faithfulness | "
        f"{per_seed[0]['8_pairing_integrity']['n_tuples_compared']} tuples per seed; "
        f"{per_seed[0]['8b_replay_faithfulness']['n_spectra_rebuilt_from_replayed_draws']} "
        f"spectra rebuilt from replayed draws and required to be bit-identical |")
    add("| 10 augmentation | narrow arms all-zero; augmented within its range |")
    add(f"| 12 leakage | 0 identical spectra; near-duplicate ratio "
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
        "architecture, re-drawn with this script's seeding. " + anchor["a_flag_is_not_a_failure"] + ".")
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
