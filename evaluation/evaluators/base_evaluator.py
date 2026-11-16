import json
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
from config import CURRENT_METRIC_PATH, METRICS, PROXY_URL
from tqdm import tqdm

# Fields each metric needs. If one is empty the metric is not computed; nothing is invented.
REQUIRED_FIELDS = {
    "answer_relevancy": ["input", "output"],
    "faithfulness": ["context", "output"],
    "context_precision": ["input", "context", "reference"],
    "context_recall": ["input", "context", "reference"],
}


class BaseEvaluator(ABC):
    """Shared CSV, error-handling and judge-token logic for all three frameworks."""

    framework = ""
    faithfulness_source = "Faithfulness"  # native metric behind faithfulness_score

    def __init__(self, dataset: pd.DataFrame, output_path: str, run_name: str):
        self.dataset = dataset
        self.checkpoint_path = Path(output_path) / f"{run_name}.csv"
        self.run_name = run_name
        self.execution_start = datetime.now(UTC).isoformat(timespec="microseconds")
        self.set_current_metric({})

    def set_current_metric(self, current: dict) -> None:
        """Associate subsequent proxy calls with this metric and sample."""
        CURRENT_METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        CURRENT_METRIC_PATH.write_text(json.dumps(current), encoding="utf-8")

    def score_row(self, row: pd.Series, metrics: dict) -> dict:
        """Score each available metric and return its CSV row."""
        scores, reasons, errors, tokens = {}, {}, {}, {}
        started = time.perf_counter()

        try:
            for name, score_metric in metrics.items():
                missing = []
                for field in REQUIRED_FIELDS[name]:
                    values = row[field] if isinstance(row[field], list) else [row[field]]
                    if not any(isinstance(value, str) and value.strip() for value in values):
                        missing.append(field)
                if missing:
                    errors[name] = f"not computed: missing {', '.join(missing)}"
                    continue

                self.set_current_metric(
                    {
                        "framework": self.framework,
                        "run_name": self.run_name,
                        "execution_start": self.execution_start,
                        "sample_id": row["id"],
                        "metric": name,
                    }
                )
                try:
                    response = httpx.get(f"{PROXY_URL}/usage", timeout=10)
                    response.raise_for_status()
                    before = response.json()
                except (httpx.HTTPError, ValueError) as error:
                    before = None
                    errors[name] = (
                        f"judge token usage unavailable: "
                        f"{type(error).__name__}: {error}"
                    )

                try:
                    scores[name], reasons[name] = score_metric()
                except Exception as error:
                    errors[name] = f"{type(error).__name__}: {error}"

                if before is None:
                    tokens[name] = None
                    continue

                try:
                    response = httpx.get(f"{PROXY_URL}/usage", timeout=10)
                    response.raise_for_status()
                    after = response.json()
                except (httpx.HTTPError, ValueError) as error:
                    tokens[name] = None
                    errors.setdefault(
                        name,
                        f"judge token usage unavailable: "
                        f"{type(error).__name__}: {error}",
                    )
                    continue

                if after["n_calls_unknown_tokens"] > before["n_calls_unknown_tokens"]:
                    tokens[name] = None
                else:
                    tokens[name] = (
                        after["input_tokens"] - before["input_tokens"],
                        after["output_tokens"] - before["output_tokens"],
                    )
        finally:
            self.set_current_metric({})

        result = {
            "id": row["id"],
            "dataset": row["dataset"],
            "input": row["input"],
            "context": json.dumps(row["context"], ensure_ascii=False),
            "reference": row["reference"],
            "output": row["output"],
            "generation_input_tokens": row["generation_input_tokens"],
            "generation_output_tokens": row["generation_output_tokens"],
            "framework": self.framework,
        }

        for name in METRICS:
            result[f"{name}_score"] = scores.get(name)
            result[f"{name}_reason"] = reasons.get(name)
            result[f"{name}_judge_input_tokens"] = tokens[name][0] if tokens.get(name) is not None else None
            result[f"{name}_judge_output_tokens"] = tokens[name][1] if tokens.get(name) is not None else None

        # Missing metrics made no judge calls; unknown counts on a called metric make the total unknown.
        if any(count is None for count in tokens.values()):
            result["judge_input_tokens"] = None
            result["judge_output_tokens"] = None
        else:
            result["judge_input_tokens"] = sum(count[0] for count in tokens.values())
            result["judge_output_tokens"] = sum(count[1] for count in tokens.values())

        result["faithfulness_source"] = self.faithfulness_source
        result["hallucination_score"] = None  # only Opik fills it
        result["metric_errors"] = json.dumps(errors, ensure_ascii=False)
        result["evaluation_time_seconds"] = round(time.perf_counter() - started, 4)
        return result

    def evaluate(self) -> pd.DataFrame:
        """Evaluate the rows that are not in the CSV yet, saving each one as soon as it is done."""
        done_ids = set()
        if self.checkpoint_path.exists():
            saved = pd.read_csv(self.checkpoint_path, dtype={"id": str})
            if "answer_relevancy_score" not in saved.columns:
                raise ValueError(f"{self.checkpoint_path} has an older column layout; use another repeat number.")
            done_ids = set(saved["id"])

        pending = self.dataset[~self.dataset["id"].isin(done_ids)]
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(pending.iterrows(), total=len(pending), desc=self.run_name):
            result = self._evaluate_row(row)
            has_header = not self.checkpoint_path.exists()
            pd.DataFrame([result]).to_csv(self.checkpoint_path, mode="a", header=has_header, index=False)

        return pd.read_csv(self.checkpoint_path, dtype={"id": str})

    @abstractmethod
    def _evaluate_row(self, row: pd.Series) -> dict:
        """Score one row with the four metrics of the framework by calling score_row."""
