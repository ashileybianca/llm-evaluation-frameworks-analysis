"""Step 4: Generate answers with Llama 3.2 3B using Ollama."""

import json
import subprocess
import sys
import time
from pathlib import Path

import ollama
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))

from config import (
    BASE_MODEL,
    DATASET_SOURCES,
    GENERATION_MODEL,
    GENERATION_SEED,
    GENERATIONS_DIR,
    LOGS_DIR,
    MODELFILE,
    NUM_CTX,
    NUM_PREDICT,
    PYPROJECT,
    REPORTS_DIR,
    SAMPLES_DIR,
    SAMPLING_SEED,
    TEMPERATURE,
    UV_LOCK,
)
from prompt import CONTENT_TEMPLATE, build_messages
from tracking import latest_run_id, log_run


def load_jsonl(path):
    """Load records from a JSONL file."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def generate_dataset(name, samples_dir, generations_dir):
    """Generate and save answers for one dataset."""
    samples = load_jsonl(samples_dir / f"{name}_sample.jsonl")
    out_path = generations_dir / f"{name}_generations.jsonl"
    token_counts = pd.read_csv(
        samples_dir / f"{name}_sample_tokens.csv",
        dtype={"id": str},
    )
    required = {"id", "llama_context_overflow"}
    missing_columns = required - set(token_counts.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns in {name}_sample_tokens.csv: "
            f"{', '.join(sorted(missing_columns))}"
        )

    token_ids = set(token_counts["id"])
    missing_ids = {sample["id"] for sample in samples} - token_ids
    if missing_ids:
        raise ValueError(
            f"Missing token counts for {len(missing_ids)} samples in "
            f"{name}_sample_tokens.csv"
        )

    # Skip saved IDs so an interrupted run can resume.
    previous = load_jsonl(out_path) if out_path.exists() else []
    completed_ids = {result["id"] for result in previous}
    remaining = [
        sample for sample in samples if sample["id"] not in completed_ids
    ]
    overflow_ids = set(
        token_counts.loc[
            token_counts["llama_context_overflow"].astype(str).str.lower()
            == "true",
            "id",
        ]
    )
    oversized = [
        sample["id"] for sample in remaining if sample["id"] in overflow_ids
    ]
    if oversized:
        raise ValueError(
            f"{name} has {len(oversized)} remaining prompts that exceed "
            f"the context window; see the characterize report."
        )

    with open(out_path, "a", encoding="utf-8") as file:
        for sample in tqdm(remaining, desc=name, unit="question"):
            messages = build_messages(sample)

            # Generate the answer and measure elapsed time.
            start = time.perf_counter()
            response = ollama.chat(
                model=GENERATION_MODEL,
                messages=messages,
                options={
                    "num_predict": NUM_PREDICT,
                    "temperature": TEMPERATURE,
                    "seed": GENERATION_SEED,
                    "num_ctx": NUM_CTX,
                },
                stream=False,
            )
            answer = response.message.content.strip()

            # Save the answer and generation details.
            result = {
                "id": sample["id"],
                "dataset": name,
                "answer": answer,
                "generation_successful": (
                    response.done_reason == "stop" and bool(answer)
                ),
                "finish_reason": response.done_reason,
                "prompt_tokens": response.prompt_eval_count,
                "output_tokens": response.eval_count,
                "generation_wall_time_s": round(
                    time.perf_counter() - start, 3
                ),
                "total_duration_ns": response.total_duration,
                "load_duration_ns": response.load_duration,
                "prompt_eval_duration_ns": response.prompt_eval_duration,
                "eval_duration_ns": response.eval_duration,
                "created_at": response.created_at,
            }

            # Write each result immediately so the run can resume.
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()

            # Warn if the generation did not finish normally.
            if response.done_reason != "stop":
                tqdm.write(
                    f"{name} {sample['id']}: "
                    f"finish_reason={response.done_reason}"
                )

    return out_path


def summarize(name, results):
    """Calculate summary metrics for one dataset."""
    table = pd.DataFrame(results)

    return {
        "dataset": name,
        "n_calls": len(table),
        "n_successful": int(table["generation_successful"].sum()),
        "n_stop": int((table["finish_reason"] == "stop").sum()),
        "n_length_terminated": int(
            (table["finish_reason"] == "length").sum()
        ),
        "mean_prompt_tokens": table["prompt_tokens"].mean(),
        "max_prompt_tokens": int(table["prompt_tokens"].max()),
        "mean_output_tokens": table["output_tokens"].mean(),
        "max_output_tokens": int(table["output_tokens"].max()),
        "total_generation_time_s": (
            table["total_duration_ns"].sum() / 1e9
        ),
        "mean_output_tokens_per_second": (
            table["output_tokens"] / (table["eval_duration_ns"] / 1e9)
        ).mean(),
    }


def run_command(command):
    """Run a command and return its output as text."""
    return subprocess.check_output(
        command, text=True, encoding="utf-8", errors="replace"
    )


def save_system_logs(logs_dir):
    """Save hardware, model, and software details for reproducibility."""
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Record the models currently loaded in Ollama.
    loaded_models = run_command(["ollama", "ps"])
    if len(loaded_models.splitlines()) > 1:
        (logs_dir / "ollama_ps.txt").write_text(
            loaded_models, encoding="utf-8"
        )

    # Record GPU model, driver version, and total memory.
    gpu_info = run_command([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv",
    ])
    (logs_dir / "gpu_info.txt").write_text(
        gpu_info, encoding="utf-8"
    )

    # Record installed Ollama models and their identifiers.
    model_info = run_command(["ollama", "list"])
    (logs_dir / "model_digests.txt").write_text(
        model_info, encoding="utf-8"
    )

    # Record the Ollama version.
    ollama_version = run_command(["ollama", "--version"])
    (logs_dir / "ollama_version.txt").write_text(
        ollama_version, encoding="utf-8"
    )


def model_params():
    """Collect model identifiers and the Ollama version for MLflow."""
    digests = {model.model: model.digest for model in ollama.list().models}

    return {
        "model": GENERATION_MODEL,
        "model_digest": digests[f"{GENERATION_MODEL}:latest"],
        "base_model": BASE_MODEL,
        "base_model_digest": digests[BASE_MODEL],
        "ollama_version": run_command(["ollama", "--version"]).strip(),
    }


def write_table(name, jsonl_path, samples_dir):
    """Save questions, references, answers, and generation details as CSV."""
    questions = pd.DataFrame(
        load_jsonl(samples_dir / f"{name}_sample.jsonl")
    )[["id", "question", "reference"]]

    table = questions.merge(pd.DataFrame(load_jsonl(jsonl_path)), on="id")
    table["total_duration_s"] = table["total_duration_ns"] / 1e9

    first_columns = [
        "id",
        "dataset",
        "question",
        "reference",
        "answer",
        "generation_successful",
        "total_duration_s",
    ]
    table = table[
        first_columns
        + [column for column in table.columns if column not in first_columns]
    ]

    csv_path = jsonl_path.with_suffix(".csv")
    table.to_csv(csv_path, index=False)
    return csv_path


def write_prompt_example(name, samples_dir, reports_dir):
    """Save the exact messages built for one sample."""
    sample = load_jsonl(samples_dir / f"{name}_sample.jsonl")[0]
    messages = build_messages(sample)

    path = reports_dir / f"prompt_example_{name}.txt"
    path.write_text(
        "\n\n".join(
            f"[{message['role']}]\n{message['content']}"
            for message in messages
        ),
        encoding="utf-8",
    )
    return path


def log_results(summary, out_paths, paths):
    """Log each dataset's results, configuration, and artifacts to MLflow."""
    save_system_logs(paths["logs"])

    log_names = [
        "ollama_ps.txt",
        "gpu_info.txt",
        "model_digests.txt",
        "ollama_version.txt",
    ]
    log_files = [
        paths["logs"] / name
        for name in log_names
        if (paths["logs"] / name).exists()
    ]
    params = model_params()

    for _, result in summary.iterrows():
        name = result["dataset"]
        artifacts = [
            out_paths[name],
            write_table(name, out_paths[name], paths["samples"]),
            paths["summary"],
            write_prompt_example(name, paths["samples"], paths["reports"]),
            paths["model_file"],
            paths["pyproject"],
            paths["uv_lock"],
            *log_files,
        ]

        log_run(
            name,
            "generate",
            params={
                **params,
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
                "temperature": TEMPERATURE,
                "generation_seed": GENERATION_SEED,
                "sampling_seed": SAMPLING_SEED,
                "prompt_template": CONTENT_TEMPLATE,
            },
            metrics=result.drop("dataset").astype(float).to_dict(),
            artifacts=artifacts,
            links={
                "sample_run_id": latest_run_id(f"{name}-sample"),
                "characterize_run_id": latest_run_id(f"{name}-characterize"),
            },
        )


def main():
    """Generate answers, save summaries, and track each dataset."""
    paths = {
        "samples": SAMPLES_DIR,
        "generations": GENERATIONS_DIR,
        "reports": REPORTS_DIR,
        "logs": LOGS_DIR,
        "model_file": MODELFILE,
        "pyproject": PYPROJECT,
        "uv_lock": UV_LOCK,
    }

    paths["generations"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)

    datasets = list(DATASET_SOURCES)

    # Generate answers and summarize all datasets.
    out_paths = {
        name: generate_dataset(
            name, paths["samples"], paths["generations"]
        )
        for name in datasets
    }
    summary = pd.DataFrame(
        [
            summarize(name, load_jsonl(path))
            for name, path in out_paths.items()
        ]
    )

    paths["summary"] = paths["reports"] / "generation_summary.csv"
    summary.to_csv(paths["summary"], index=False)
    print(summary.to_string(index=False))

    # Record the environment, outputs, and configuration in MLflow.
    log_results(summary, out_paths, paths)


if __name__ == "__main__":
    main()
