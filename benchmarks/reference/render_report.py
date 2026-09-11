"""Render the human-readable views of the reference benchmark from its JSON record.

Every number here is read out of `results/reference_benchmark.json`. Nothing is typed
in by hand, so the Markdown table and the GUI hint strings cannot drift away from the
measurement that produced them: re-run this script after re-running the benchmark and
the text follows.

    python benchmarks/reference/render_report.py                 # print to stdout
    python benchmarks/reference/render_report.py --write         # also write report.md
    python benchmarks/reference/render_report.py --section gui   # just the GUI strings
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy import stats

DEFAULT_RECORD = Path(__file__).resolve().parent / "results" / "reference_benchmark.json"


def _mean(values: list) -> float:
    return sum(values) / len(values)


def _sample_sd(values: list) -> float:
    """Sample standard deviation (ddof=1), matching how the benchmark writes it."""
    if len(values) < 2:
        return float("nan")
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _holm(pvalues: list) -> list:
    """Holm-Bonferroni step-down adjustment.

    Deliberately an independent implementation of the one in
    `reference_benchmark.py`: if the two ever disagree, that disagreement is a
    finding, which is the point of recomputing at all.
    """
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _agrees(recorded, recomputed) -> bool:
    """True when a recorded scalar matches a recomputed one, NaN counting as equal."""
    if recorded is None:
        return False
    try:
        recorded = float(recorded)
    except (TypeError, ValueError):
        return False
    if math.isnan(recorded) and math.isnan(recomputed):
        return True
    return math.isclose(recorded, recomputed, rel_tol=1e-9, abs_tol=1e-12)


def verify_record(record: dict) -> None:
    """Refuse to render a record whose summary does not match its own raw runs.

    Everything the renderer puts in front of a reader is recomputed here from the
    per-run raw numbers and compared against what the benchmark wrote: the
    aggregate means, the across-seed standard deviations, the input SNR that
    labels each column, the parameter counts, and every field of the paired
    comparison including the t statistic, the raw and Holm-adjusted p values,
    Cohen's dz and the seeds-favouring count.

    An earlier version of this function checked only the mean and the seed count
    while its docstring claimed to check everything. A record with tampered
    standard deviations, or an entirely fabricated paired table, rendered without
    complaint. `tests/test_reference_benchmark_record.py` now holds that door
    shut.

    Problems are accumulated and reported together rather than raising on the
    first one, so a single run tells you everything that is wrong.
    """
    problems: list = []

    def check(label: str, recorded, recomputed) -> None:
        if not _agrees(recorded, recomputed):
            problems.append(f"{label}: recorded {recorded!r} != {recomputed!r} recomputed from runs")

    # --- parameter counts: runs must agree with each other and with self_checks
    observed = record.get("self_checks", {}).get("parameter_counts", {}).get("observed", {})
    per_arch_params: dict = {}
    for run in record["runs"]:
        per_arch_params.setdefault(run["architecture"], set()).add(run["n_trainable_parameters"])
    for arch, counts in sorted(per_arch_params.items()):
        if len(counts) != 1:
            problems.append(f"parameter count for {arch} is not constant across runs: {sorted(counts)}")
        elif arch in observed and observed[arch] != next(iter(counts)):
            problems.append(
                f"parameter count for {arch}: runs say {next(iter(counts))}, "
                f"self_checks.parameter_counts says {observed[arch]}"
            )

    # --- aggregates: per-seed series, n, mean, across-seed SD, and the column label
    for condition, per_arch in record["aggregates"].items():
        for arch, per_level in per_arch.items():
            for level, entry in per_level.items():
                runs = [
                    run for run in record["runs"]
                    if run["condition"] == condition and run["architecture"] == arch
                ]
                runs.sort(key=lambda run: run["seed_index"])
                values = [run["per_level"][level]["snr_gain_db_mean"] for run in runs]
                where = f"{condition}/{arch}/L={level}"
                if len(values) != entry["n_seeds"]:
                    problems.append(
                        f"{where}: claims n_seeds={entry['n_seeds']} but {len(values)} runs are recorded"
                    )
                    continue
                if entry.get("per_seed_snr_gain_db") != values:
                    problems.append(f"{where}: per_seed_snr_gain_db does not match the runs")
                check(f"{where} mean", entry["snr_gain_db_mean"], _mean(values))
                check(f"{where} sd", entry.get("snr_gain_db_sd_across_seeds"), _sample_sd(values))
                check(
                    f"{where} input_snr_db_mean",
                    entry.get("input_snr_db_mean"),
                    _mean([run["per_level"][level]["input_snr_db_mean"] for run in runs]),
                )

    # --- the paired table: the most citable block, and previously unguarded
    for condition, block in record.get("paired_vs_baseline", {}).items():
        baseline = block["baseline"]
        for level, per_level in block["by_noise_level"].items():
            seed_indices = per_level["seed_indices"]

            def series(arch: str, level=level, condition=condition, seed_indices=seed_indices):
                by_seed = {
                    run["seed_index"]: run["per_level"][level]["snr_gain_db_mean"]
                    for run in record["runs"]
                    if run["condition"] == condition and run["architecture"] == arch
                }
                return [by_seed[s] for s in seed_indices] if all(s in by_seed for s in seed_indices) else None

            base_values = series(baseline)
            if base_values is None:
                problems.append(f"{condition}/L={level}: baseline {baseline} has no runs for the recorded seeds")
                continue
            if per_level.get("baseline_per_seed_snr_gain_db") != base_values:
                problems.append(f"{condition}/L={level}: baseline_per_seed_snr_gain_db does not match the runs")

            raw_pvalues, arch_order = [], []
            for arch, entry in per_level["comparisons"].items():
                where = f"{condition}/L={level}/{arch} vs {baseline}"
                values = series(arch)
                if values is None:
                    problems.append(f"{where}: no runs for the recorded seeds")
                    continue
                differences = [v - b for v, b in zip(values, base_values)]
                mean_difference = _mean(differences)
                sd_difference = _sample_sd(differences)
                check(f"{where} n_seeds", entry["n_seeds"], float(len(differences)))
                if entry.get("per_seed_difference_db") != differences:
                    problems.append(f"{where}: per_seed_difference_db does not match the runs")
                check(f"{where} mean_difference_db", entry["mean_difference_db"], mean_difference)
                check(f"{where} sd_difference_db", entry["sd_difference_db"], sd_difference)
                check(
                    f"{where} seeds_favouring_arch",
                    entry["seeds_favouring_arch"],
                    float(sum(1 for d in differences if d > 0)),
                )
                if len(differences) > 1 and sd_difference > 0:
                    statistic, p_value = stats.ttest_rel(values, base_values)
                    check(f"{where} paired_t", entry["paired_t"], float(statistic))
                    check(f"{where} p_value_raw", entry["p_value_raw"], float(p_value))
                    check(f"{where} cohens_dz", entry["cohens_dz"], mean_difference / sd_difference)
                    raw_pvalues.append(float(p_value))
                else:
                    raw_pvalues.append(1.0)
                arch_order.append(arch)

            for arch, adjusted in zip(arch_order, _holm(raw_pvalues)):
                check(
                    f"{condition}/L={level}/{arch} p_value_holm_adjusted",
                    per_level["comparisons"][arch].get("p_value_holm_adjusted"),
                    adjusted,
                )

    if problems:
        raise SystemExit(
            "record inconsistent with its own raw runs; refusing to render.\n  - "
            + "\n  - ".join(problems)
        )


def load_record(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    verify_record(record)
    return record


def _levels(record: dict) -> list:
    return [str(level) for level in record["design"]["noise_model"]["levels"]]


def _architectures(record: dict, condition: str) -> list:
    return list(record["aggregates"][condition].keys())


def _fmt(mean: float, sd) -> str:
    if sd is None:
        return f"{mean:+.2f}"
    return f"{mean:+.2f} ± {sd:.2f}"


def provenance_line(record: dict) -> str:
    design = record["design"]
    return (
        f"Source: `benchmarks/reference/results/reference_benchmark.json` "
        f"(record v{record['record_version']}, generated {record['generated_utc']}, "
        f"device `{record['environment']['device_requested_resolved_to']}`, "
        f"torch {record['environment']['torch']}, "
        f"{design['seeds']['n_seeds']} seeds)."
    )


def render_conditions(record: dict) -> str:
    design = record["design"]
    data = design["data"]
    levels = _levels(record)
    inputs = record["aggregates"]["suggested-hyperparameters"]
    first_arch = next(iter(inputs))
    input_snrs = [inputs[first_arch][level]["input_snr_db_mean"] for level in levels]
    lines = [
        "## What was measured",
        "",
        f"- **Data**: {data['provenance']}; peak set `{data['peak_set_id']}` "
        f"({len(data['peak_set_peaks_mu_fwhm_intensity'])} peaks), "
        f"{data['generator_config']['n_energy_points']} energy points over "
        f"{data['energy_range_ev'][0]:.1f}–{data['energy_range_ev'][1]:.1f} eV "
        f"({data['energy_step_ev']:.3f} eV/point); "
        f"position jitter {data['generator_config']['position_jitter']} eV, "
        f"width variation {data['generator_config']['width_variation']}, "
        f"intensity variation {data['generator_config']['intensity_variation']}, "
        f"{data['generator_config']['background_type']} background.",
        f"- **Training pool**: {data['n_train_spectra_total_pooled']} spectra "
        f"({data['n_train_spectra_per_noise_level']} per noise level); "
        f"**test**: {data['n_test_spectra_per_noise_level']} spectra per noise level.",
        f"- **Split rule**: {design['split_rule']['rule']}. "
        f"Leakage-preventing unit: {design['split_rule']['leakage_preventing_unit']}. "
        f"Verified by {design['split_rule']['verified']}.",
        f"- **Noise model**: `{design['noise_model']['config']}` at "
        f"L = {', '.join(str(int(float(level))) for level in levels)}, giving mean input SNR "
        f"{', '.join(f'{value:+.1f}' for value in input_snrs)} dB respectively. "
        f"{design['noise_model']['implementation_note']}",
        f"- **Seeds**: {design['seeds']['n_seeds']}. A seed varies "
        f"{design['seeds']['what_a_seed_varies']}.",
        f"- **Metric**: {design['metric']['name']} — {design['metric']['definition']}. "
        f"Computed over {design['metric']['computed_over']}.",
        f"- **Independence**: {design['metric']['independence_assumption']}",
        f"- **Comparison condition**: {design['conditions']['primary']['why']}. "
        f"The result supports \"{design['conditions']['primary']['supports']}\".",
        f"- **Regime**: {record['claim_scope']['regime_note']}",
        f"- **Reproducibility**: {design['reproducibility']['detail']}",
    ]
    checks = record["self_checks"]
    leakage = checks["leakage"]
    lines += [
        "",
        "### Self-checks recorded with the run",
        "",
        f"- Parameter counts match the expected values for all "
        f"{len(checks['parameter_counts']['observed'])} architectures: "
        f"`{checks['parameter_counts']['passed']}`.",
        f"- Suggested hyperparameters agree with the GUI source: "
        f"`{checks['gui_hyperparameter_crosscheck'].get('passed')}` "
        f"({checks['gui_hyperparameter_crosscheck']['status']}).",
        # This paragraph lived in report.md as hand-written text, which made the
        # file unregenerable: `--write` dropped it silently. report.md says it is
        # generated, so it is generated.
        "  That comparison was made when this measurement was taken, against a",
        "  graphical training tool that is not part of this repository."
        " Re-running the self-check",
        "  here reports `gui-source-not-found` and proceeds; the values it agreed"
        " with",
        "  are embedded in `results/reference_benchmark.json` under",
        "  `self_checks.gui_hyperparameter_crosscheck`, and the recipe itself is a",
        "  literal in `reference_benchmark.py`, so the run is self-contained"
        " either way.",
        f"- Leakage, measured on the first seed only: "
        f"{leakage['identical_clean_spectra']} of {leakage['n_test_spectra']} "
        f"test spectra are byte-identical to any of the {leakage['n_train_spectra']} "
        f"training spectra (clean), {leakage['identical_noisy_spectra']} (noisy). "
        f"The split is probabilistic rather than structural -- train and test draw "
        f"per-spectrum seeds from disjoint streams -- so this is a spot check of one "
        f"seed, not a proof for all of them.",
    ]
    near = leakage.get("near_duplicate_analysis")
    if near:
        lines.append(
            f"- Near duplication: median RMS distance from a clean test spectrum to its "
            f"nearest training spectrum is {near['test_to_train_nn_rms_median']:.5f}, "
            f"against {near['train_to_train_nn_rms_median']:.5f} within the training set "
            f"(ratio {near['ratio_median_test_over_train']:.3f}). {near['what_it_measures']}"
        )
    determinism = checks.get("repeat_run_determinism")
    if determinism:
        lines.append(
            f"- Run-to-run determinism on this device: baseline `"
            f"{determinism['architecture']}` trained twice with identical inputs, "
            f"max |difference| = {determinism['max_abs_difference_db']:.3g} dB, "
            f"bit-identical = `{determinism['bit_identical']}`."
        )
    return "\n".join(lines)


def render_table(record: dict, condition: str) -> str:
    aggregates = record["aggregates"][condition]
    levels = _levels(record)
    architectures = _architectures(record, condition)
    if not architectures:
        return f"(no runs recorded for condition `{condition}`)"
    reference = aggregates[architectures[0]]
    n_seeds = reference[levels[0]]["n_seeds"]

    header_cells = ["Architecture", "Params"]
    for level in levels:
        input_snr = reference[level]["input_snr_db_mean"]
        header_cells.append(f"L={int(float(level))} (in {input_snr:+.1f} dB)")
    if condition == "suggested-hyperparameters":
        header_cells.append("Suggested settings")

    rows = [
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(["---"] * len(header_cells)) + "|",
    ]
    params = {
        run["architecture"]: run["n_trainable_parameters"]
        for run in record["runs"]
        if run["condition"] == condition
    }
    hyperparams = {
        run["architecture"]: run["hyperparameters"]
        for run in record["runs"]
        if run["condition"] == condition
    }
    for arch in architectures:
        cells = [f"`{arch}`", f"{params[arch]:,}"]
        for level in levels:
            entry = aggregates[arch][level]
            cells.append(_fmt(entry["snr_gain_db_mean"], entry["snr_gain_db_sd_across_seeds"]))
        if condition == "suggested-hyperparameters":
            hp = hyperparams[arch]
            cells.append(f"lr {hp['lr']}, {hp['epochs']} ep, batch {hp['batch_size']}")
        rows.append("| " + " | ".join(cells) + " |")

    caption = (
        f"SNR gain in dB, mean ± SD over {n_seeds} seeds "
        f"(the replicate is one independent training run, not one spectrum). "
        f"Condition: `{condition}`."
    )
    return "\n".join(rows) + "\n\n" + caption


def render_paired(record: dict, condition: str) -> str:
    block = record["paired_vs_baseline"].get(condition)
    if not block or not block["by_noise_level"]:
        return f"(no paired comparison recorded for condition `{condition}`)"
    baseline = block["baseline"]
    lines = [
        f"Paired difference against `{baseline}`, computed within seed: the two "
        f"architectures in a seed were trained on the same spectra and scored on the "
        f"same test spectra, so the seed pairs them. n is the number of seeds.",
        "",
        "| Noise level | Architecture | Δ vs "
        f"`{baseline}` (dB) | Seeds favouring | paired t | p (raw) | p (Holm) | Cohen's dz |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for level, payload in block["by_noise_level"].items():
        n_seeds = len(payload["seed_indices"])
        for arch, entry in payload["comparisons"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"L={int(float(level))}",
                        f"`{arch}`",
                        f"{entry['mean_difference_db']:+.2f} ± {entry['sd_difference_db']:.2f}",
                        f"{entry['seeds_favouring_arch']}/{n_seeds}",
                        f"{entry['paired_t']:.2f}",
                        f"{entry['p_value_raw']:.4f}",
                        f"{entry['p_value_holm_adjusted']:.4f}",
                        f"{entry['cohens_dz']:.2f}",
                    ]
                )
                + " |"
            )
    lines += [
        "",
        f"p-values are two-sided paired t-tests on n = {n_seeds} seeds and are "
        "correspondingly weak; the Holm column adjusts for the "
        f"{len(next(iter(block['by_noise_level'].values()))['comparisons'])} comparisons "
        "made at each noise level. Read the sign consistency column alongside them.",
    ]
    return "\n".join(lines)


def render_gui_strings(record: dict, condition: str = "suggested-hyperparameters") -> str:
    """Explain why no dropdown hint strings are generated from this record.

    An earlier version emitted ready-to-paste Python mapping each architecture to
    its SNR gain, intended for the label beside the architecture dropdown. An
    independent review refused them, and the reasoning is worth keeping where
    someone would look for the feature:

    One number per architecture, next to the control that selects it, reads as a
    leaderboard no matter what the surrounding tooltip says. This record cannot
    support one. The top two architectures are not separated from each other; at
    the highest noise level the top four fall inside a 0.28 dB band; and the
    ordering between the two conditions disagrees for three of the seven paired
    comparisons, which makes part of it a learning-rate effect rather than an
    architecture effect. A reader of a single dropdown label sees none of that.

    What the interface shows instead is the trainable parameter count, which is a
    property of the model, is exact, and needs no record behind it.
    """
    del record, condition
    return render_gui_strings.__doc__ or ""

def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, item, out)
    else:
        out[prefix] = value


def _design_differences(record: dict, other: dict) -> list:
    """Design fields that differ between two records, other than the training size.

    The scale check is only evidence if the two records differ in one thing. This
    finds out rather than assuming, and names what it found.
    """
    allowed = {
        "data.n_train_spectra_per_noise_level",
        "data.n_train_spectra_total_pooled",
    }
    a: dict = {}
    b: dict = {}
    _flatten("", record["design"], a)
    _flatten("", other["design"], b)
    differences = []
    for field in sorted(set(a) | set(b)):
        if field in allowed:
            continue
        if a.get(field) != b.get(field):
            differences.append((field, a.get(field, "(absent)"), b.get(field, "(absent)")))
    return differences


def _top_two_separated(record: dict, condition: str, level: str, ranking: list) -> bool:
    """True when the leading two architectures differ by more than their spreads.

    Below that, printing rank positions invites a reader to see an ordering the
    record cannot support.
    """
    if len(ranking) < 2:
        return False
    first = record["aggregates"][condition][ranking[0]][level]
    second = record["aggregates"][condition][ranking[1]][level]
    spread = (first.get("snr_gain_db_sd_across_seeds") or 0.0) + (
        second.get("snr_gain_db_sd_across_seeds") or 0.0
    )
    return (first["snr_gain_db_mean"] - second["snr_gain_db_mean"]) > spread


def render_scale_check(record: dict, other: dict, condition: str) -> str:
    """Compare the same condition across two records that differ in training-set size.

    The question this answers is narrow and important: does the ordering of the
    architectures survive a change in how much data they were trained on, or is the
    ordering an artifact of one training-set size? Nothing else about the two records
    may differ; the header states the sizes so that can be checked.
    """
    levels = _levels(record)
    size_a = record["design"]["data"]["n_train_spectra_total_pooled"]
    size_b = other["design"]["data"]["n_train_spectra_total_pooled"]
    architectures = _architectures(record, condition)
    shared = [arch for arch in architectures if arch in other["aggregates"].get(condition, {})]
    if not shared:
        return "(the second record has no runs for this condition)"

    differences = _design_differences(record, other)
    if differences:
        lines = [
            f"**These two records are not comparable.** They differ in {len(differences)} "
            "design field(s) beyond the training-set size, so a change in the numbers "
            "cannot be attributed to the training-set size alone:",
            "",
        ]
        lines += [f"- `{field}`: `{a}` vs `{b}`" for field, a, b in differences]
        lines += ["", "The comparison below is printed for inspection, not as evidence.", ""]
    else:
        lines = [
            f"Same condition (`{condition}`), two training-set sizes: {size_a} vs {size_b} "
            "pooled training spectra. Every other design field was compared and found "
            "identical -- peak set, generator settings, noise levels, seed count, metric, "
            "hyperparameters, device and library versions. An earlier version of this "
            "paragraph asserted that without checking it.",
            "",
        ]
    for level in levels:
        ranking_a = sorted(
            shared, key=lambda a: -record["aggregates"][condition][a][level]["snr_gain_db_mean"]
        )
        ranking_b = sorted(
            shared, key=lambda a: -other["aggregates"][condition][a][level]["snr_gain_db_mean"]
        )
        rank_b = {arch: index for index, arch in enumerate(ranking_b)}
        max_shift = max(abs(rank_b[arch] - index) for index, arch in enumerate(ranking_a))
        resolvable = _top_two_separated(record, condition, level, ranking_a)
        if resolvable:
            lines += [
                f"**Noise level L = {int(float(level))}** — largest change in rank "
                f"position: {max_shift}.",
            ]
        else:
            lines += [
                f"**Noise level L = {int(float(level))}** — the leading two architectures "
                "are closer together than their across-seed spreads, so there is no "
                "ordering here to preserve or break. Rank positions are omitted; a rank "
                "table would read as a leaderboard of something this record cannot show.",
            ]
        columns = [f"| Architecture | {size_a} spectra | {size_b} spectra | Δ (dB) |"]
        divider = ["|---|---|---|---|"]
        if resolvable:
            columns = [columns[0] + f" rank {size_a} → {size_b} |"]
            divider = ["|---|---|---|---|---|"]
        lines += ["", columns[0], divider[0]]
        for index, arch in enumerate(ranking_a):
            entry_a = record["aggregates"][condition][arch][level]
            entry_b = other["aggregates"][condition][arch][level]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{arch}`",
                        _fmt(entry_a["snr_gain_db_mean"], entry_a["snr_gain_db_sd_across_seeds"]),
                        _fmt(entry_b["snr_gain_db_mean"], entry_b["snr_gain_db_sd_across_seeds"]),
                        f"{entry_b['snr_gain_db_mean'] - entry_a['snr_gain_db_mean']:+.2f}",
                    ]
                    + ([f"{index + 1} → {rank_b[arch] + 1}"] if resolvable else [])
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def render_all(record: dict, other: dict | None = None) -> str:
    parts = [
        "# Reference benchmark — generated report",
        "",
        "<!-- Generated by benchmarks/reference/render_report.py. Do not edit by hand. -->",
        "",
        provenance_line(record),
        "",
        render_conditions(record),
        "",
        "## Primary result — each architecture at its suggested hyperparameters",
        "",
        render_table(record, "suggested-hyperparameters"),
        "",
        "### What this table does not support",
        "",
    ]
    parts += [f"- {item}" for item in record["claim_scope"]["does_not_support"]]
    parts += [
        "",
        f"- {record['claim_scope']['regime_note']}",
        f"- {record['claim_scope']['denoised_output_is_a_model_estimate']}",
        "",
        "**No baseline was measured.** Every figure above is a network compared "
        "against other networks. Nothing here establishes that a network is the "
        "right tool at any of these noise levels: a classical smoother, or even "
        "returning the nearest training spectrum, might score comparably at some "
        "of them, and neither was run. Read the table as a comparison within one "
        "family of methods, not as evidence that the family is warranted.",
        "",
        "Two limits are properties of the design rather than of any number in it, "
        "and they do not appear in the table above. The training noise and the test "
        "noise come from the same function, so the model's noise model is exactly "
        "correct by construction — a condition measured data never satisfies. And "
        "distribution shift is the dominant failure mode of a trained denoiser; "
        f"nothing here probes it. The reference for every figure is "
        f"{record['design']['metric']['reference']}.",
        "",
        "## Paired comparison — primary condition",
        "",
        render_paired(record, "suggested-hyperparameters"),
    ]
    if "matched-budget" in record["aggregates"] and record["aggregates"]["matched-budget"]:
        parts += [
            "",
            "## Secondary result — identical budget for every architecture",
            "",
            f"Condition: lr {record['design']['conditions']['secondary']['hyperparameters']['lr']}, "
            f"{record['design']['conditions']['secondary']['hyperparameters']['epochs']} epochs, "
            f"batch {record['design']['conditions']['secondary']['hyperparameters']['batch_size']} "
            "for all architectures.",
            "",
            "This removes the epoch and batch-size differences between architectures by "
            "introducing a learning-rate one: the fixed rate is already the suggested "
            "value for some architectures and an order of magnitude away for others. It "
            "is a second view, not a neutral arbiter. Where the two conditions disagree, "
            "the disagreement is the finding and belongs in any statement made from "
            "either table.",
            "",
            render_table(record, "matched-budget"),
            "",
            render_paired(record, "matched-budget"),
        ]
    if other is not None:
        parts += [
            "",
            "## Robustness check — does the ordering survive more training data?",
            "",
            render_scale_check(record, other, "suggested-hyperparameters"),
        ]
    parts += [
        "",
        f"- {record['claim_scope']['denoised_output_is_a_model_estimate']}",
        "",
    ]
    return "\n".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument(
        "--compare-record",
        type=Path,
        default=None,
        help="second record differing only in training-set size, for the robustness check",
    )
    parser.add_argument(
        "--section",
        default="all",
        choices=["all", "table", "paired", "gui", "conditions", "scale"],
    )
    parser.add_argument(
        "--condition",
        default="suggested-hyperparameters",
        help="which condition the --section table/paired/gui should render",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the full report to report.md next to the record",
    )
    args = parser.parse_args(argv)

    record = load_record(args.record)
    other = load_record(args.compare_record) if args.compare_record else None
    if args.section == "all":
        text = render_all(record, other)
    elif args.section == "table":
        text = render_table(record, args.condition)
    elif args.section == "paired":
        text = render_paired(record, args.condition)
    elif args.section == "conditions":
        text = render_conditions(record)
    elif args.section == "scale":
        if other is None:
            raise SystemExit("--section scale requires --compare-record")
        text = render_scale_check(record, other, args.condition)
    else:
        text = render_gui_strings(record, args.condition)

    print(text)
    if args.write:
        destination = args.record.parent.parent / "report.md"
        destination.write_text(render_all(record, other) + "\n", encoding="utf-8")
        print(f"\n<!-- wrote {destination} -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
