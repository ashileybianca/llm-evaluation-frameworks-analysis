"""Step 1: Download the validation splits of SQuAD, HotpotQA and MS MARCO."""

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from dotenv import load_dotenv

from config import DATASET_REVISIONS, DATASET_SOURCES, ENV_FILE, RAW_DIR
from datasets import load_dataset

load_dotenv(ENV_FILE)

RAW_DIR.mkdir(parents=True, exist_ok=True)

for name, (source, config, split) in DATASET_SOURCES.items():
    dataset = load_dataset(
        source,
        name=config,
        split=split,
        revision=DATASET_REVISIONS[name],
        token=os.getenv("HF_TOKEN"),
    )

    # saving as JSONL in raw datasets directory
    with open(RAW_DIR / f"{name}.jsonl", "w", encoding="utf-8") as file:
        file.writelines(
            json.dumps(row, ensure_ascii=False) + "\n" for row in dataset
        )

    print(f"Saved {name} with {len(dataset)} rows")
