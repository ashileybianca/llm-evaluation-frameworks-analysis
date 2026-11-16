"""Summarize the sampled datasets: passages, token lengths, and generated answers.

Token counts for questions, contexts, and references use the Llama 3.2 tokenizer
without the chat template. Prompt and output tokens are the counts reported by
Ollama during generation.

Usage:
    python analysis/dataset_characteristics.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import ENV_FILE, TOKENIZERS

DATASETS = ["squad", "hotpotqa", "msmarco"]
SAMPLES_DIR = ROOT / "datasets" / "samples"
GENERATIONS_DIR = ROOT / "datasets" / "llama-responses"
OUTPUT_PATH = ROOT / "reports" / "dataset_characteristics.csv"


def describe(values: pd.Series) -> dict:
    return {
        "mean": values.mean(),
        "std": values.std(),
        "median": values.median(),
        "min": values.min(),
        "max": values.max(),
    }


def main() -> None:
    load_dotenv(ENV_FILE)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZERS["llama"], token=os.getenv("HF_TOKEN")
    )

    rows = []
    for dataset in DATASETS:
        samples = pd.read_csv(SAMPLES_DIR / f"{dataset}_sample_tokens.csv")
        generations = pd.read_csv(GENERATIONS_DIR / f"{dataset}_generations.csv")
        if set(samples["id"].astype(str)) != set(generations["id"].astype(str)):
            raise ValueError(f"Sample and generation IDs differ for {dataset}")

        reference_tokens = pd.Series(
            [
                len(tokenizer(str(text), add_special_tokens=False)["input_ids"])
                for text in samples["reference"]
            ],
            dtype="int64",
        )
        variables = {
            "passages": samples["n_passages"],
            "relevant_passages": samples["n_relevant"],
            "question_tokens": samples["llama_question_tokens"],
            "context_tokens": samples["llama_context_tokens"],
            "reference_tokens": reference_tokens,
            "prompt_tokens": generations["prompt_tokens"],
            "output_tokens": generations["output_tokens"],
        }
        for variable, values in variables.items():
            rows.append({
                "dataset": dataset,
                "variable": variable,
                "n_samples": len(values),
                **describe(values),
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
