"""Generate the Kendall tau-b interval figure for the TCC2 manuscript."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "analysis"
OUTPUT_DIR = ROOT / "figures"

DATASETS = [
    ("squad", "SQuAD"),
    ("hotpotqa", "HotpotQA"),
    ("msmarco", "MS MARCO"),
]
METRICS = [
    ("answer_relevancy", "Answer Relevancy", "Answer Rel."),
    ("faithfulness", "Faithfulness", "Faithfulness"),
    ("context_precision", "Context Precision", "Context Prec."),
    ("context_recall", "Context Recall", "Context Recall"),
]
PAIRS = [
    ("deepeval", "ragas", "DeepEval--RAGAS", "#0072B2"),
    ("deepeval", "opik", "DeepEval--Opik", "#D55E00"),
    ("ragas", "opik", "RAGAS--Opik", "#009E73"),
]
DATASET_COLORS = {
    "squad": "#0072B2",
    "hotpotqa": "#D55E00",
    "msmarco": "#009E73",
}
DATASET_MARKERS = {"squad": "o", "hotpotqa": "s", "msmarco": "^"}


def load_reports() -> tuple[pd.DataFrame, pd.DataFrame]:
    agreement = pd.read_csv(REPORTS / "agreement" / "agreement.csv")
    bootstrap_tau = pd.read_csv(REPORTS / "agreement" / "bootstrap_kendall.csv")
    return agreement, bootstrap_tau


def _validate_pair_rows(frame: pd.DataFrame, name: str) -> None:
    expected = {
        frozenset((first, second))
        for first, second, _, _ in PAIRS
    }
    found = {
        frozenset((row.framework_1, row.framework_2))
        for row in frame.itertuples()
    }
    if found != expected:
        raise ValueError(f"{name} does not contain all expected framework pairs")


def plot_kendall_intervals(
    agreement: pd.DataFrame, bootstrap_tau: pd.DataFrame
) -> None:
    """Plot observed Kendall tau-b values and their bootstrap confidence intervals."""
    _validate_pair_rows(bootstrap_tau, "bootstrap_kendall.csv")
    _validate_pair_rows(agreement, "agreement.csv")
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(7.16, 3.05), sharey=True)
    metric_names = [metric[2] for metric in METRICS]
    dataset_offsets = {"squad": -0.18, "hotpotqa": 0.0, "msmarco": 0.18}

    for ax, (first, second, pair_name, _) in zip(axes, PAIRS):
        pair_rows = bootstrap_tau[
            (bootstrap_tau["framework_1"].isin([first, second]))
            & (bootstrap_tau["framework_2"].isin([first, second]))
        ]
        reference_rows = agreement[
            (agreement["framework_1"].isin([first, second]))
            & (agreement["framework_2"].isin([first, second]))
        ]
        for metric_index, (metric, _, _) in enumerate(METRICS):
            for dataset, _ in DATASETS:
                row = pair_rows[
                    (pair_rows["metric"] == metric)
                    & (pair_rows["dataset"] == dataset)
                ]
                reference = reference_rows[
                    (reference_rows["metric"] == metric)
                    & (reference_rows["dataset"] == dataset)
                ]
                if len(row) != 1 or len(reference) != 1:
                    raise ValueError(
                        f"Expected one tau estimate and interval for "
                        f"{metric}/{dataset}/{pair_name}"
                    )
                estimate = float(reference.iloc[0]["kendall_tau_b"])
                low = float(row.iloc[0]["ci_low"])
                high = float(row.iloc[0]["ci_high"])
                position = metric_index + dataset_offsets[dataset]
                ax.errorbar(
                    estimate,
                    position,
                    xerr=[[estimate - low], [high - estimate]],
                    fmt=DATASET_MARKERS[dataset],
                    color=DATASET_COLORS[dataset],
                    markersize=3.1,
                    capsize=1.5,
                    linewidth=0.75,
                    elinewidth=0.75,
                )

        ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(pair_name, fontsize=8)
        ax.set_xlim(-0.5, 0.65)
        ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
        ax.set_yticks(range(len(METRICS)))
        ax.set_yticklabels(metric_names, fontsize=6)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2, linewidth=0.5)
        ax.tick_params(axis="x", labelsize=6, length=2)

    axes[0].set_ylabel("Metric", fontsize=7)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=DATASET_MARKERS[dataset],
            color="none",
            markerfacecolor=DATASET_COLORS[dataset],
            markeredgecolor=DATASET_COLORS[dataset],
            markersize=3.5,
            label=label,
        )
        for dataset, label in DATASETS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(DATASETS),
        frameon=False,
        fontsize=6,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
    fig.savefig(
        OUTPUT_DIR / "kendall_tau_intervals.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    agreement, bootstrap_tau = load_reports()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_kendall_intervals(agreement, bootstrap_tau)


if __name__ == "__main__":
    main()
