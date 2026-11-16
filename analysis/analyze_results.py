"""Analyze the evaluation results, organized by research question.

RQ1 (Stability): does each framework give the same score when the same
    sample is evaluated again?
RQ2 (Agreement): do the frameworks agree on the scores of the same answers
    and on how they rank them, and does this depend on the metric?
RQ3 (Dataset influence): does the disagreement between frameworks appear on
    every dataset, or only on some of them?
RQ4 (Metric behavior): which metrics give mostly 0 or 1, and does RAGAS still
    give a Context Recall of 1 to almost every sample outside SQuAD?
RQ5 (Cost): how many judge calls, tokens and seconds does each framework need
    per sample and per metric?

Usage:
    python analyze_results.py
    python analyze_results.py --input-dir datasets/qwen-evaluation
    python analyze_results.py --skip-bootstrap
"""

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

DATASETS = ["squad", "hotpotqa", "msmarco"]
FRAMEWORKS = ["ragas", "deepeval", "opik"]
# Order used for the framework pairs in the RQ2/RQ3 tables and in the bootstrap,
# so that the pairs read DeepEval-RAGAS, DeepEval-Opik, RAGAS-Opik.
PAIR_ORDER = ["deepeval", "ragas", "opik"]
METRICS = [
    "answer_relevancy",
    "faithfulness",
    "context_precision",
    "context_recall",
]
SAMPLE_KEYS = ["dataset", "framework", "id"]
SAMPLE_TEXT_COLUMNS = ["input", "context", "reference", "output"]
EVALUATION_RUN = re.compile(
    r"^(squad|hotpotqa|msmarco)-(ragas|deepeval|opik)-eval-r(\d+)$"
)
# RAGAS returns values such as 0.9999999999 for a perfect score, so scores
# within this distance of 0 or 1 are counted as exactly 0 or 1.
SCORE_TOLERANCE = 1e-6


# ===========================================================================
# Loading and aggregation
# ===========================================================================


def identify_file(path, table):
    """Identify the dataset, framework and run from the file path."""
    text = " ".join([str(path), *path.parts]).lower()

    dataset = next(
        (name for name in DATASETS if name in text),
        None,
    )
    if dataset is None and "dataset" in table.columns:
        values = table["dataset"].dropna().astype(str).str.lower().unique()
        if len(values) == 1 and values[0] in DATASETS:
            dataset = values[0]

    framework = next(
        (name for name in FRAMEWORKS if name in text),
        None,
    )

    # Supports names such as 1_RagasEvaluator.csv and run_1.
    match = re.search(r"(?:run[_-]?)?(\d+)", path.stem.lower())
    run = int(match.group(1)) if match else 1

    if dataset is None or framework is None:
        return None

    return dataset, framework, run


def load_results(input_dir):
    """Load all evaluation CSVs and standardize their metadata."""
    tables = []

    for path in Path(input_dir).rglob("*.csv"):
        table = pd.read_csv(path)
        info = identify_file(path, table)

        if info is None:
            continue

        dataset, framework, run = info
        table["dataset"] = dataset
        table["framework"] = framework
        table["run"] = run
        table["id"] = table["id"].astype(str) if "id" in table else table.index.astype(str)

        tables.append(table)

    if not tables:
        raise ValueError(
            f"No evaluation CSVs identified in {input_dir}. "
            "Check the folder structure and filenames."
        )

    results = pd.concat(tables, ignore_index=True)

    # Older result files may use the metric name without "_score".
    for metric in METRICS:
        score_column = f"{metric}_score"
        if score_column not in results and metric in results:
            results[score_column] = results[metric]

        if score_column in results:
            # Rounding removes floating-point noise such as 1 - 0.8 =
            # 0.19999999999999996, which would otherwise break ties in
            # Kendall's tau-b and make the results depend on the machine.
            results[score_column] = pd.to_numeric(
                results[score_column], errors="coerce"
            ).round(10)
            # Failed evaluations must not be treated as real scores.
            results.loc[results[score_column] < 0, score_column] = float("nan")

    return results


def load_judge_calls(path):
    """Load proxy requests for the configured evaluation runs."""
    if not path.exists():
        raise FileNotFoundError(f"Judge call log not found: {path}")

    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                call = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}"
                ) from error

            run_name = call.get("run_name")
            if not isinstance(run_name, str):
                continue

            match = EVALUATION_RUN.fullmatch(run_name)
            if match is None:
                continue

            dataset, framework, run = match.groups()
            metric = call.get("metric")
            sample_id = call.get("sample_id")
            if metric not in METRICS or sample_id is None:
                raise ValueError(
                    f"Missing or invalid evaluation metadata in "
                    f"{path} on line {line_number}"
                )

            rows.append({
                "dataset": dataset,
                "framework": framework,
                "run": int(run),
                "metric": metric,
                "sample_id": str(sample_id),
                "execution_start": call.get("execution_start"),
                "status": call.get("status"),
                "input_tokens": call.get("input_tokens"),
                "output_tokens": call.get("output_tokens"),
            })

    if not rows:
        raise ValueError(f"No evaluation judge calls found in {path}")

    calls = pd.DataFrame(rows)
    calls["status"] = pd.to_numeric(calls["status"], errors="coerce")
    calls["input_tokens"] = pd.to_numeric(
        calls["input_tokens"], errors="coerce"
    )
    calls["output_tokens"] = pd.to_numeric(
        calls["output_tokens"], errors="coerce"
    )
    return calls


def consolidate_repetitions(results):
    """Merge the repetitions of each evaluation into one row.

    Numeric columns are averaged over the available runs, ignoring missing
    values. Text that does not change between runs (question, context,
    reference, answer) is kept; per-run text such as reasons and metric
    errors is dropped.
    """
    numeric = [
        column
        for column in results.select_dtypes("number").columns
        if column != "run" and not column.endswith("_reason")
    ]
    text = [column for column in SAMPLE_TEXT_COLUMNS if column in results]

    grouped = results.groupby(SAMPLE_KEYS, sort=True)

    consolidated = grouped[text].first().join(grouped[numeric].mean())
    consolidated.insert(0, "n_runs", grouped["run"].nunique())
    consolidated.insert(
        1,
        "runs",
        grouped["run"].agg(lambda runs: ",".join(map(str, sorted(set(runs))))),
    )

    for metric in METRICS:
        column = f"{metric}_score"
        if column in results:
            # Averages of equal values must stay exactly equal (ties).
            consolidated[column] = consolidated[column].round(10)
            consolidated[f"{metric}_n_valid_runs"] = grouped[column].count()

    return consolidated.reset_index()


def reshape_scores(consolidated):
    """Convert consolidated per-sample scores to one row per metric."""
    rows = []

    for metric in METRICS:
        column = f"{metric}_score"
        if column not in consolidated:
            continue

        scores = consolidated[SAMPLE_KEYS + [column]].dropna()
        scores = scores.rename(columns={column: "score"})
        scores["metric"] = metric
        rows.append(scores)

    if not rows:
        raise ValueError("No metric score columns were found in the CSV files.")

    return pd.concat(rows, ignore_index=True)


# ===========================================================================
# Shared helpers
# ===========================================================================


def interval(values):
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return low, high


def framework_gaps(consolidated, dataset, first, second):
    """Absolute score gap between two frameworks: samples x metrics."""
    subset = consolidated[consolidated["dataset"] == dataset]
    gaps = []
    for metric in METRICS:
        wide = subset.pivot(index="id", columns="framework", values=f"{metric}_score")
        gaps.append((wide[first] - wide[second]).abs().rename(metric))
    return pd.concat(gaps, axis=1).to_numpy()


def pair_name(first, second):
    ordered = sorted([first, second], key=PAIR_ORDER.index)
    return f"{ordered[0]}_{ordered[1]}"


# ===========================================================================
# RQ1: Stability
# ===========================================================================


def compare_repetitions(results):
    """Quantify how much the scores of each metric diverge between runs.

    Uses the original (non-consolidated) results. Only samples with a valid
    score in at least two runs are compared.
    """
    rows = []

    for metric in METRICS:
        column = f"{metric}_score"
        if column not in results:
            continue

        scores = results[["dataset", "framework", "id", "run", column]]

        for (dataset, framework), group in scores.groupby(
            ["dataset", "framework"]
        ):
            wide = group.pivot_table(
                index="id", columns="run", values=column, aggfunc="mean"
            )
            wide = wide[wide.notna().sum(axis=1) >= 2]
            if wide.empty:
                continue

            score_range = wide.max(axis=1) - wide.min(axis=1)

            pair_diffs = []
            taus = []
            for first, second in combinations(wide.columns, 2):
                pair_diffs.append((wide[first] - wide[second]).abs())

                pair = wide[[first, second]].dropna()
                if len(pair) >= 2:
                    taus.append(kendalltau(pair[first], pair[second]).statistic)

            rows.append({
                "dataset": dataset,
                "framework": framework,
                "metric": metric,
                "n_runs": wide.shape[1],
                "n_samples": len(wide),
                "mean_abs_diff": pd.concat(pair_diffs, axis=1)
                .mean(axis=1)
                .mean(),
                "mean_std": wide.std(axis=1, ddof=1).mean(),
                "share_identical": (score_range < 1e-9).mean(),
                "mean_kendall_tau_b": pd.Series(taus, dtype=float).mean(),
            })

    return pd.DataFrame(rows)


def summarize_metric_variability(divergence):
    """Summarize score variability across repetitions by framework and metric."""
    if divergence.empty:
        return pd.DataFrame()

    return (
        divergence.groupby(["framework", "metric"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            n_samples=("n_samples", "sum"),
            mean_abs_diff=("mean_abs_diff", "mean"),
            mean_std=("mean_std", "mean"),
            share_identical=("share_identical", "mean"),
            mean_kendall_tau_b=("mean_kendall_tau_b", "mean"),
        )
        .sort_values("mean_abs_diff", ascending=False)
    )


# ===========================================================================
# RQ2: Agreement
# ===========================================================================


def compare_agreement(scores):
    """Compare scores and rankings on samples valid for both frameworks."""
    rows = []

    for (dataset, metric), group in scores.groupby(["dataset", "metric"]):
        wide = group.pivot(
            index="id", columns="framework", values="score"
        )

        for first, second in combinations(FRAMEWORKS, 2):
            if first not in wide or second not in wide:
                continue

            # Only compare the same samples for both frameworks.
            pair = wide[[first, second]].dropna()
            if pair.empty:
                continue

            tau = kendalltau(pair[first], pair[second])

            rows.append({
                "dataset": dataset,
                "metric": metric,
                "framework_1": first,
                "framework_2": second,
                "n_samples": len(pair),
                "mae": (pair[first] - pair[second]).abs().mean(),
                "kendall_tau_b": tau.statistic,
            })

    return pd.DataFrame(rows)


def compare_three_frameworks(scores, results):
    """Per-sample range across the three frameworks (highest minus lowest
    score), per dataset and metric, using samples valid for all three.

    n_full_range counts samples where one framework gave 0 and another 1.
    n_full_range_stable counts those where every framework also gave the
    same score in all of its runs, so the divergence is not run-to-run noise.
    """
    rows = []
    for (dataset, metric), group in scores.groupby(["dataset", "metric"]):
        wide = group.pivot(index="id", columns="framework", values="score")
        wide = wide[[f for f in PAIR_ORDER if f in wide]].dropna()
        if wide.shape[1] < 3:
            continue
        score_range = wide.max(axis=1) - wide.min(axis=1)
        full_range = score_range[score_range >= 1 - SCORE_TOLERANCE].index

        column = f"{metric}_score"
        subset = results[(results["dataset"] == dataset) & results["id"].isin(full_range)]
        run_spread = (
            subset.groupby(["id", "framework"])[column]
            .agg(lambda runs: runs.max() - runs.min())
            .groupby("id")
            .max()
        )
        rows.append({
            "dataset": dataset,
            "metric": metric,
            "n_samples": len(wide),
            "mean_range": score_range.mean(),
            "share_range_at_least_half": (score_range >= 0.5 - SCORE_TOLERANCE).mean(),
            "n_full_range": len(full_range),
            "n_full_range_stable": int((run_spread < 1e-9).sum()),
        })
    return pd.DataFrame(rows)


def bootstrap_kendall(consolidated, rng, n_boot):
    """95% bootstrap interval of Kendall's tau-b for each pair of frameworks,
    per metric and dataset. Undefined values (constant scores) are skipped."""
    rows = []
    for metric in METRICS:
        for dataset in DATASETS:
            wide = consolidated[consolidated["dataset"] == dataset].pivot(
                index="id", columns="framework", values=f"{metric}_score"
            )
            for first, second in combinations(PAIR_ORDER, 2):
                pair = wide[[first, second]].dropna()
                x, y = pair[first].to_numpy(), pair[second].to_numpy()
                observed = kendalltau(x, y).statistic
                if np.isnan(observed):
                    continue
                boot = []
                for _ in range(n_boot):
                    index = rng.integers(0, len(x), len(x))
                    boot.append(kendalltau(x[index], y[index]).statistic)
                low, high = interval(boot)
                rows.append({
                    "metric": metric, "dataset": dataset,
                    "framework_1": first, "framework_2": second,
                    "kendall_tau_b": observed, "ci_low": low, "ci_high": high,
                })
    return pd.DataFrame(rows)


def bootstrap_pair_mae(consolidated, rng, n_boot):
    """MAE of each pair over all metrics and datasets, resampling the samples
    within each dataset."""
    rows = []
    for first, second in combinations(PAIR_ORDER, 2):
        per_dataset = [framework_gaps(consolidated, d, first, second) for d in DATASETS]
        observed = np.nanmean(np.concatenate(per_dataset))
        boot = []
        for _ in range(n_boot):
            resampled = [d[rng.integers(0, len(d), len(d))] for d in per_dataset]
            boot.append(np.nanmean(np.concatenate(resampled)))
        low, high = interval(boot)
        rows.append({
            "framework_1": first, "framework_2": second,
            "mae": observed, "ci_low": low, "ci_high": high,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# RQ3: Dataset influence
# ===========================================================================


def rank_frameworks(scores):
    """Rank frameworks by their mean score for each metric and dataset."""
    means = (
        scores.groupby(["dataset", "metric", "framework"])["score"]
        .mean()
        .reset_index(name="mean_score")
    )

    means["rank"] = (
        means.groupby(["dataset", "metric"])["mean_score"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    return means.sort_values(["metric", "dataset", "rank"])


def summarize_rank_stability(ranking):
    """Show whether each framework keeps the same rank across datasets."""
    rows = []

    for (metric, framework), group in ranking.groupby(
        ["metric", "framework"]
    ):
        ranks = group.set_index("dataset")["rank"].to_dict()

        rows.append({
            "metric": metric,
            "framework": framework,
            "squad_rank": ranks.get("squad"),
            "hotpotqa_rank": ranks.get("hotpotqa"),
            "msmarco_rank": ranks.get("msmarco"),
            "same_rank_all_datasets": (
                len(ranks) == len(DATASETS)
                and len(set(ranks.values())) == 1
            ),
        })

    return pd.DataFrame(rows)


def summarize_dataset_agreement(agreement, divergence):
    """One row per dataset with the MAE of each pair (averaged over metrics),
    the run-to-run variation of each framework (the noise reference), and the
    ratio between the mean MAE and the largest run-to-run variation."""
    agreement = agreement.copy()
    agreement["pair"] = [
        pair_name(a, b) for a, b in zip(agreement["framework_1"], agreement["framework_2"])
    ]
    pair_mae = agreement.pivot_table(index="dataset", columns="pair", values="mae", aggfunc="mean")
    noise = divergence.pivot_table(
        index="dataset", columns="framework", values="mean_abs_diff", aggfunc="mean"
    )
    tau = agreement.groupby("dataset")["kendall_tau_b"].mean()

    rows = []
    for dataset in DATASETS:
        if dataset not in pair_mae.index:
            continue
        row = {"dataset": dataset}
        for first, second in combinations(PAIR_ORDER, 2):
            row[f"mae_{first}_{second}"] = pair_mae.loc[dataset, f"{first}_{second}"]
        row["mae_mean"] = agreement.loc[agreement["dataset"] == dataset, "mae"].mean()
        for framework in PAIR_ORDER:
            row[f"run_variation_{framework}"] = noise.loc[dataset, framework]
        row["mae_mean_over_max_run_variation"] = row["mae_mean"] / noise.loc[dataset].max()
        row["mean_kendall_tau_b"] = tau.loc[dataset]
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_difference(consolidated, rng, n_boot):
    """Difference in mean score for every pair of frameworks, per metric and
    dataset, ordered from the higher to the lower mean. 'adjacent' marks
    frameworks in neighboring positions of the ranking."""
    rows = []
    for metric in METRICS:
        for dataset in DATASETS:
            wide = consolidated[consolidated["dataset"] == dataset].pivot(
                index="id", columns="framework", values=f"{metric}_score"
            )[PAIR_ORDER]
            order = list(wide.mean().sort_values(ascending=False).index)
            for i, j in combinations(range(len(order)), 2):
                higher, lower = order[i], order[j]
                pair = wide[[higher, lower]].dropna()
                diff = (pair[higher] - pair[lower]).to_numpy()
                index = rng.integers(0, len(diff), (n_boot, len(diff)))
                low, high = interval(diff[index].mean(axis=1))
                rows.append({
                    "metric": metric, "dataset": dataset,
                    "higher": higher, "lower": lower, "adjacent": j == i + 1,
                    "mean_difference": diff.mean(), "ci_low": low, "ci_high": high,
                    "excludes_zero": low > 0,
                })
    return pd.DataFrame(rows)


def bootstrap_dataset_mae(consolidated, rng, n_boot):
    """Mean MAE over the three pairs and four metrics, per dataset, and the
    difference between datasets (RQ3). Each dataset is resampled separately
    because the datasets contain different samples."""
    boot = {}
    observed = {}
    rows = []
    for dataset in DATASETS:
        matrix = np.column_stack([
            framework_gaps(consolidated, dataset, first, second)
            for first, second in combinations(PAIR_ORDER, 2)
        ])  # samples x (pairs * metrics)
        observed[dataset] = np.nanmean(matrix)
        boot[dataset] = np.array([
            np.nanmean(matrix[rng.integers(0, len(matrix), len(matrix))])
            for _ in range(n_boot)
        ])
        low, high = interval(boot[dataset])
        rows.append({
            "comparison": dataset, "mean_mae": observed[dataset],
            "ci_low": low, "ci_high": high,
        })
    for first, second in combinations(DATASETS, 2):
        low, high = interval(boot[first] - boot[second])
        rows.append({
            "comparison": f"{first} - {second}",
            "mean_mae": observed[first] - observed[second],
            "ci_low": low, "ci_high": high,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# RQ4: Metric behavior
# ===========================================================================


def snap_extremes(values):
    """Count scores within SCORE_TOLERANCE of 0 or 1 as exactly 0 or 1."""
    values = values.copy()
    values[(values - 0).abs() < SCORE_TOLERANCE] = 0.0
    values[(values - 1).abs() < SCORE_TOLERANCE] = 1.0
    return values


def summarize_score_distribution(scores):
    """Mean, SD, share of samples at 0 and at 1, and number of distinct
    scores, per dataset, metric and framework (averaged scores)."""
    rows = []
    for (dataset, metric, framework), group in scores.groupby(
        ["dataset", "metric", "framework"]
    ):
        values = snap_extremes(group["score"])
        rows.append({
            "dataset": dataset,
            "metric": metric,
            "framework": framework,
            "n_samples": len(values),
            "mean_score": values.mean(),
            "std_score": values.std(ddof=1),
            "share_zero": (values == 0).mean(),
            "share_one": (values == 1).mean(),
            "n_distinct_scores": values.nunique(),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# RQ5: Cost
# ===========================================================================


def summarize_consumption(results, calls):
    """Count actual proxy requests, including retries and failed requests."""
    rows = []
    call_groups = {
        key: group
        for key, group in calls.groupby(
            ["dataset", "framework", "run", "metric"]
        )
    }

    for (dataset, framework, run), group in results.groupby(
        ["dataset", "framework", "run"]
    ):
        for metric in METRICS:
            metric_calls = call_groups.get(
                (dataset, framework, run, metric),
                calls.iloc[0:0],
            )
            input_tokens = metric_calls["input_tokens"]
            output_tokens = metric_calls["output_tokens"]
            score_column = f"{metric}_score"
            scored = (
                group.loc[group[score_column].notna(), "id"].nunique()
                if score_column in group
                else 0
            )
            n_samples = group["id"].nunique()
            n_calls = len(metric_calls)
            n_known_input = int(input_tokens.notna().sum())
            n_known_output = int(output_tokens.notna().sum())
            known_input_tokens = input_tokens.sum(min_count=1)
            known_output_tokens = output_tokens.sum(min_count=1)

            row = {
                "dataset": dataset,
                "framework": framework,
                "repeat": run,
                "metric": metric,
                "n_samples": n_samples,
                "n_samples_scored": scored,
                "n_samples_with_calls": metric_calls["sample_id"].nunique(),
                "n_evaluator_executions": metric_calls[
                    "execution_start"
                ].nunique(),
                "n_judge_calls": n_calls,
                "n_successful_calls": int(
                    metric_calls["status"].eq(200).sum()
                ),
                "n_failed_calls": int(
                    (metric_calls["status"].notna()
                    & metric_calls["status"].ne(200)).sum()
                ),
                "n_calls_with_missing_tokens": int(
                    (input_tokens.isna() | output_tokens.isna()).sum()
                ),
                "reported_input_tokens": known_input_tokens,
                "reported_output_tokens": known_output_tokens,
                "reported_total_tokens": (
                    known_input_tokens + known_output_tokens
                    if pd.notna(known_input_tokens)
                    and pd.notna(known_output_tokens)
                    else float("nan")
                ),
                "mean_calls_per_sample": (
                    n_calls / n_samples if n_samples else float("nan")
                ),
                "mean_calls_per_scored_sample": (
                    n_calls / scored if scored else float("nan")
                ),
                "mean_reported_input_tokens_per_call": (
                    known_input_tokens / n_known_input
                    if n_known_input
                    else float("nan")
                ),
                "mean_reported_output_tokens_per_call": (
                    known_output_tokens / n_known_output
                    if n_known_output
                    else float("nan")
                ),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_runtime(results):
    """Summarize total evaluation time per framework and dataset."""
    if "evaluation_time_seconds" not in results:
        return pd.DataFrame()

    return (
        results.groupby(["dataset", "framework"])
        .agg(
            n_samples=("id", "nunique"),
            mean_time_seconds=("evaluation_time_seconds", "mean"),
            total_time_seconds=("evaluation_time_seconds", "sum"),
        )
        .reset_index()
    )


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze stability, agreement, dataset influence, metric behavior and cost."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/qwen-evaluation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/analysis"),
    )
    parser.add_argument(
        "--consolidated-path",
        type=Path,
        default=Path(
            "datasets/qwen-evaluation-consolidated/evaluation_mean_by_sample.csv"
        ),
        help="where to save the evaluations averaged over repetitions",
    )
    parser.add_argument(
        "--judge-call-log",
        type=Path,
        default=Path("logs/judge_calls/judge_calls.jsonl"),
        help="proxy log used to count actual judge requests",
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="bootstrap seed")
    parser.add_argument(
        "--skip-bootstrap", action="store_true",
        help="skip the bootstrap intervals (the slowest step)",
    )
    args = parser.parse_args()

    results = load_results(args.input_dir)
    calls = load_judge_calls(args.judge_call_log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    consolidated = consolidate_repetitions(results)
    args.consolidated_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(args.consolidated_path, index=False)
    print(f"Saved: {args.consolidated_path}")

    reports = {}

    divergence = compare_repetitions(results)
    reports["stability/repetition_divergence.csv"] = divergence
    reports["stability/variability_by_metric.csv"] = summarize_metric_variability(divergence)

    scores = reshape_scores(consolidated)
    agreement = compare_agreement(scores)
    reports["agreement/agreement.csv"] = agreement
    reports["agreement/three_way_disagreement.csv"] = compare_three_frameworks(scores, results)

    ranking = rank_frameworks(scores)
    reports["dataset_influence/framework_ranking.csv"] = ranking
    reports["dataset_influence/ranking_stability.csv"] = summarize_rank_stability(ranking)
    reports["dataset_influence/dataset_agreement.csv"] = summarize_dataset_agreement(agreement, divergence)

    reports["metric_behavior/score_distribution.csv"] = summarize_score_distribution(scores)

    reports["consumption/consumption_by_metric.csv"] = summarize_consumption(results, calls)
    reports["consumption/runtime_by_dataset.csv"] = summarize_runtime(consolidated)

    if not args.skip_bootstrap:
        # The order of these calls is fixed so the intervals are reproducible.
        rng = np.random.default_rng(args.seed)
        reports["agreement/bootstrap_kendall.csv"] = bootstrap_kendall(consolidated, rng, args.n_boot)
        reports["dataset_influence/bootstrap_mean_difference.csv"] = bootstrap_mean_difference(consolidated, rng, args.n_boot)
        reports["agreement/bootstrap_pair_mae.csv"] = bootstrap_pair_mae(consolidated, rng, args.n_boot)
        reports["dataset_influence/bootstrap_dataset_mae.csv"] = bootstrap_dataset_mae(consolidated, rng, args.n_boot)

    for filename, table in reports.items():
        path = args.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        print(f"Saved: {path}")

    print(f"\nLoaded {len(results)} evaluation rows.")
    print(f"Consolidated samples: {len(consolidated)}")
    print(f"Valid score observations: {len(scores)}")
    print(f"Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
