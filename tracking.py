"""Logs experiment runs in MLflow."""

import platform
import sys
from importlib.metadata import distributions, version

import mlflow

MLFLOW_EXPERIMENT = "llm-evaluation-frameworks"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

def environment_snapshot():
    """Returns a snapshot of the current environment, including Python version, platform, and installed packages."""
    packages = "\n".join(
        sorted(
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in distributions()
            if distribution.metadata.get("Name")
        )
    )
    return f"Python: {sys.version}\nPlatform: {platform.platform()}\n\n{packages}"


def latest_run_id(run_name):
    """Returns the run ID of the latest run with the given run name."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs["run_id"].iloc[0]


def log_run(
    dataset,
    stage,
    params=None,
    metrics=None,
    artifacts=(),
    links=None,
):
    """Logs an experiment run in MLflow with the given dataset, stage, parameters, metrics, artifacts, and links."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{dataset}-{stage}"):
        mlflow.set_tags({"dataset": dataset, "stage": stage, **(links or {})})

        mlflow.log_params({
            "python": platform.python_version(),
            **{
                f"lib_{name}": version(name)
                for name in [
                    "pandas", "numpy", "transformers", "ollama",
                    "mlflow", "python-dotenv", "datasets",
                ]
            },
            **(params or {}),
        })

        mlflow.log_metrics(metrics or {})
        mlflow.log_text(environment_snapshot(), "environment.txt")

        for path in artifacts:
            mlflow.log_artifact(str(path))