"""Generate a sample-level heatmap of scores for each evaluation run."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    ROOT
    / "datasets"
    / "qwen-evaluation"
)
OUTPUT_DIR = ROOT / "figures"

DATASETS = [
    ("squad", "SQuAD"),
    ("hotpotqa", "HotpotQA"),
    ("msmarco", "MS MARCO"),
]
METRICS = [
    ("answer_relevancy_score", "Answer Relevancy"),
    ("faithfulness_score", "Faithfulness"),
    ("context_precision_score", "Context Precision"),
    ("context_recall_score", "Context Recall"),
]
FRAMEWORKS = [
    ("deepeval", "DeepEval"),
    ("ragas", "RAGAS"),
    ("opik", "Opik"),
]
RUNS = [(0, "1"), (1, "2"), (2, "3")]


def load_run_scores(data_path: Path) -> pd.DataFrame:
    """Load raw evaluation runs and return aligned sample scores."""
    run_frames = []
    expected_ids: dict[str, set[str]] = {}
    metric_columns = [metric for metric, _ in METRICS]

    for dataset, _ in DATASETS:
        dataset_ids: set[str] | None = None
        for framework, _ in FRAMEWORKS:
            for run, _ in RUNS:
                path = data_path / f"{dataset}-{framework}-eval-r{run}.csv"
                frame = pd.read_csv(path)
                required_columns = {"id", "dataset", "framework"} | set(
                    metric_columns
                )
                missing_columns = required_columns.difference(frame.columns)
                if missing_columns:
                    raise ValueError(
                        f"{path.name} is missing columns: "
                        f"{sorted(missing_columns)}"
                    )
                if frame["id"].duplicated().any():
                    raise ValueError(f"{path.name} contains duplicate sample IDs")
                if set(frame["dataset"].dropna().unique()) != {dataset}:
                    raise ValueError(f"{path.name} contains unexpected dataset rows")
                if set(frame["framework"].dropna().unique()) != {framework}:
                    raise ValueError(f"{path.name} contains unexpected framework rows")

                current_ids = set(frame["id"].dropna())
                if dataset_ids is None:
                    dataset_ids = current_ids
                elif current_ids != dataset_ids:
                    raise ValueError(
                        f"Sample IDs do not match across runs for {dataset}"
                    )

                values = frame.set_index("id")[metric_columns]
                values.columns = pd.MultiIndex.from_product(
                    [metric_columns, [framework], [run]]
                )
                run_frames.append((dataset, values))

        if dataset_ids is None:
            raise ValueError(f"No evaluation runs were found for {dataset}")
        expected_ids[dataset] = dataset_ids

    aligned = []
    for dataset, _ in DATASETS:
        frames = [
            frame
            for frame_dataset, frame in run_frames
            if frame_dataset == dataset
        ]
        dataset_scores = pd.concat(frames, axis=1).sort_index(axis=1)
        if set(dataset_scores.index) != expected_ids[dataset]:
            raise ValueError(f"Some sample scores are missing for {dataset}")
        aligned.append(
            dataset_scores.reset_index().assign(dataset=dataset)
        )
    return pd.concat(aligned, ignore_index=True)


def build_heatmap_data(scores: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Return individual run scores, sorted within each dataset by mean score."""
    matrix_parts = []
    dataset_boundaries = []
    row_offset = 0
    score_columns = [
        (metric, framework, run)
        for metric, _ in METRICS
        for framework, _ in FRAMEWORKS
        for run, _ in RUNS
    ]

    for dataset, _ in DATASETS:
        dataset_scores = scores.loc[scores["dataset"] == dataset]
        pivoted = dataset_scores.set_index("id")[score_columns]
        mean_score = pivoted.mean(axis=1, skipna=True)
        ordering = pd.DataFrame(
            {"mean": mean_score.to_numpy(), "id": mean_score.index.to_numpy()}
        )
        ordered_ids = ordering.sort_values(
            ["mean", "id"], kind="mergesort", na_position="last"
        )["id"]
        ordered = pivoted.loc[ordered_ids]
        matrix_parts.append(ordered.to_numpy(dtype=float))
        row_offset += len(ordered)
        dataset_boundaries.append(row_offset)

    return np.vstack(matrix_parts), dataset_boundaries[:-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_PATH,
        help="directory containing per-run evaluation CSVs",
    )
    args = parser.parse_args()
    scores = load_run_scores(args.input_dir)
    matrix, dataset_boundaries = build_heatmap_data(scores)
    expected_rows = len(DATASETS) * 50
    expected_columns = len(METRICS) * len(FRAMEWORKS) * len(RUNS)
    if matrix.shape != (expected_rows, expected_columns):
        raise ValueError(
            f"Expected a {expected_rows}x{expected_columns} score matrix, "
            f"got {matrix.shape}"
        )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    fig, ax = plt.subplots(figsize=(7.16, 5.1))
    fig.subplots_adjust(left=0.14, right=0.88, top=0.81, bottom=0.09)
    ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=Normalize(vmin=0, vmax=1),
        origin="upper",
        rasterized=True,
    )

    column_positions = np.arange(expected_columns)
    ax.set_xticks(column_positions)
    ax.set_xticklabels(["" for _ in column_positions])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(column_positions) - 0.5)
    ax.set_ylim(len(matrix) - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for metric_index, (_, metric_name) in enumerate(METRICS):
        columns_per_metric = len(FRAMEWORKS) * len(RUNS)
        center = metric_index * columns_per_metric + (columns_per_metric - 1) / 2
        ax.text(
            center,
            1.105,
            metric_name,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            fontsize=7.5,
            fontweight="bold",
        )
        for framework_index, (_, framework_name) in enumerate(FRAMEWORKS):
            framework_center = (
                metric_index * columns_per_metric
                + framework_index * len(RUNS)
                + (len(RUNS) - 1) / 2
            )
            ax.text(
                framework_center,
                1.045,
                framework_name,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                fontsize=5.8,
            )
            for run_index, (_, run_label) in enumerate(RUNS):
                position = (
                    metric_index * columns_per_metric
                    + framework_index * len(RUNS)
                    + run_index
                )
                ax.text(
                    position,
                    -0.012,
                    run_label,
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=5.2,
                )
        if metric_index:
            ax.axvline(
                metric_index * columns_per_metric - 0.5,
                color="white",
                linewidth=1.4,
                clip_on=False,
            )

    for framework_index in range(1, len(FRAMEWORKS) * len(METRICS)):
        if framework_index % len(FRAMEWORKS):
            ax.axvline(
                framework_index * len(RUNS) - 0.5,
                color="white",
                linewidth=0.75,
                clip_on=False,
            )

    for boundary in dataset_boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=2.2, clip_on=False)

    for index, (_, dataset_name) in enumerate(DATASETS):
        center = index * 50 + 24.5
        ax.text(
            -0.035,
            center,
            dataset_name,
            ha="right",
            va="center",
            rotation=90,
            transform=ax.get_yaxis_transform(),
            fontsize=7,
        )

    ax.text(
        (expected_columns - 1) / 2,
        -0.065,
        "Run",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=6.5,
    )

    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap),
        ax=ax,
        fraction=0.035,
        pad=0.035,
        ticks=np.linspace(0, 1, 6),
    )
    colorbar.set_label("Score", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5, length=2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "score_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
