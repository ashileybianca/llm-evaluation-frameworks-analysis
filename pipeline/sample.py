"""Step 2: Sample 50 questions from each dataset and log them in MLflow."""

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))

from config import DATASET_SOURCES, N_SAMPLES, RAW_DIR, SAMPLES_DIR, SAMPLING_SEED
from tracking import log_run


def read_squad(record):
    """Read a SQuAD record and return a standardized format."""
    answers = record["answers"]["text"]

    return {
        "id": record["id"],
        "question": record["question"],
        "contexts": [record["context"]],
        "relevant": [True],
        "reference": answers[0] if answers else None,
        "references": list(dict.fromkeys(answers)),
    }


def read_hotpotqa(record):
    """Read a HotpotQA record and return a standardized format."""
    titles = record["context"]["title"]
    supporting_titles = set(record["supporting_facts"]["title"])

    return {
        "id": record["id"],
        "question": record["question"],
        "contexts": [
            f"{title}: {' '.join(sentence.strip() for sentence in sentences)}"
            for title, sentences in zip(
                titles, record["context"]["sentences"]
            )
        ],
        "relevant": [title in supporting_titles for title in titles],
        "reference": record["answer"] or None,
        "references": [record["answer"]],
    }


def read_msmarco(record):
    """Read an MS MARCO record and return a standardized format."""
    answers = record["answers"]

    return {
        "id": str(record["query_id"]),
        "question": record["query"],
        "contexts": record["passages"]["passage_text"],
        "relevant": [
            selected == 1
            for selected in record["passages"]["is_selected"]
        ],
        "reference": (
            answers[0]
            if answers and answers[0] != "No Answer Present."
            else None
        ),
        "references": list(dict.fromkeys(answers)),
    }


def to_table(sample):
    """Convert a list of standardized records into a pandas DataFrame."""
    return pd.DataFrame([
        {
            "id": record["id"],
            "question": record["question"],
            "reference": record["reference"],
            "n_passages": len(record["contexts"]),
            "n_relevant": sum(record["relevant"]),
            "relevant_passages": "\n\n".join(
                text
                for text, relevant in zip(
                    record["contexts"], record["relevant"]
                )
                if relevant
            ),
            "contexts": "\n\n".join(record["contexts"]),
        }
        for record in sample
    ])


def main():
    """Sample 50 records from each dataset and save them as JSONL and CSV."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # iterate over each dataset and sample 50 records
    readers = {
        "squad": read_squad,
        "hotpotqa": read_hotpotqa,
        "msmarco": read_msmarco,
    }
    for name in DATASET_SOURCES:
        with open(RAW_DIR / f"{name}.jsonl", encoding="utf-8") as file:
            records = [
                {"dataset": name, **readers[name](json.loads(line))}
                for line in file
            ]

        records = [
            record
            for record in records
            if record["reference"] and any(record["relevant"])
        ]
        sample = random.Random(SAMPLING_SEED).sample(records, N_SAMPLES)

        # save the sampled records as JSONL and CSV
        jsonl_path = SAMPLES_DIR / f"{name}_sample.jsonl"
        csv_path = SAMPLES_DIR / f"{name}_sample.csv"

        with open(jsonl_path, "w", encoding="utf-8") as file:
            file.writelines(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in sample
            )

        to_table(sample).to_csv(csv_path, index=False)

        # just validating if the sample contains 50 unique records and all have relevant passages
        assert len({record["id"] for record in sample}) == N_SAMPLES
        assert all(any(record["relevant"]) for record in sample)

        # logging the sampling process
        log_run(
            name,
            "sample",
            params={
                "dataset": name,
                "split": DATASET_SOURCES[name][2],
                "sampling_seed": SAMPLING_SEED,
                "split_size_after_filter": len(records),
            },
            artifacts=[jsonl_path, csv_path],
        )

        print(f"{name}: {N_SAMPLES} samples from {len(records)} questions")


if __name__ == "__main__":
    main()
