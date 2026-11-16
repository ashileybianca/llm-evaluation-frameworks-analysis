
"""Step 3: Characterize a dataset and summarize its token usage."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parents[1]))

from prompt import CONTENT_TEMPLATE, build_messages

from config import ENV_FILE, NUM_CTX, NUM_PREDICT, TOKENIZER_REVISIONS, TOKENIZERS
from tracking import log_run


def count_tokens(tokenizer, texts):
    """Count tokens in each text."""
    tokens = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return pd.Series([len(ids) for ids in tokens], dtype="int64")


def summarize_tokens(table):
    """Calculate descriptive statistics for complete prompts."""
    rows = []

    for name in TOKENIZERS:
        column = f"{name}_question_context_template_tokens"
        tokens = table[column]

        rows.append({
            "tokenizer": name,
            "samples": len(tokens),
            "mean": tokens.mean(),
            "median": tokens.median(),
            "std": tokens.std(),
            "min": tokens.min(),
            "p90": tokens.quantile(0.90),
            "p95": tokens.quantile(0.95),
            "max": tokens.max(),
            "total_tokens": tokens.sum(),
        })

    return pd.DataFrame(rows)


def characterize(input_path):
    """Add token counts and save dataset statistics."""
    table = pd.read_csv(input_path, dtype=str, keep_default_na=False)

    required = ["question", "contexts"]
    missing = [column for column in required if column not in table]

    if missing:
        raise ValueError(
            f"Missing columns in {input_path.name}: {', '.join(missing)}"
        )

    # Build the complete prompt for each sample.
    messages = [
        build_messages({
            "question": row.question,
            "contexts": [row.contexts],
        })
        for row in table.itertuples(index=False)
    ]

    for name, model in TOKENIZERS.items():
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=TOKENIZER_REVISIONS[name],
            token=os.getenv("HF_TOKEN"),
        )

        # Count question and context tokens separately.
        context = count_tokens(tokenizer, table["contexts"].tolist())
        question = count_tokens(tokenizer, table["question"].tolist())

        table[f"{name}_context_tokens"] = context
        table[f"{name}_question_tokens"] = question
        table[f"{name}_question_context_tokens"] = question + context

        # Count the complete prompt, including the chat template.
        prompts = [
            tokenizer.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        table[f"{name}_question_context_template_tokens"] = [
            len(prompt["input_ids"]) for prompt in prompts
        ]

    # Check whether each Llama prompt fits the context window.
    llama_tokens = table["llama_question_context_template_tokens"]
    context_usage = (llama_tokens + NUM_PREDICT) / NUM_CTX

    table["llama_context_overflow"] = context_usage > 1

    # Save token counts for each sample.
    output_path = input_path.with_name(
        f"{input_path.stem}_tokens.csv"
    )
    table.to_csv(output_path, index=False)

    # Save descriptive statistics for each tokenizer.
    stats = summarize_tokens(table)
    stats_path = input_path.with_name(
        f"{input_path.stem}_token_stats.csv"
    )
    stats.to_csv(stats_path, index=False)

    # Summarize context-window usage for Llama.
    overflow_count = int(table["llama_context_overflow"].sum())
    sample_count = len(table)

    mean_usage = context_usage.mean() * 100
    max_usage = context_usage.max() * 100
    near_limit_count = int((context_usage >= 0.90).sum())

    log_run(
        input_path.stem.removesuffix("_sample"),
        "characterize",
        params={
            "input_file": input_path.name,
            "prompt_template": CONTENT_TEMPLATE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
        artifacts=[output_path, stats_path],
    )

    print(f"Saved token counts to {output_path}")
    print(f"Saved token statistics to {stats_path}")

    if sample_count:
        print(
            f"Context overflow: {overflow_count}/{sample_count} samples "
            f"({overflow_count / sample_count * 100:.1f}%)."
        )
        print(f"Mean context-window usage: {mean_usage:.1f}%.")
        print(f"Maximum context-window usage: {max_usage:.1f}%.")
        print(f"Samples using at least 90% of the window: {near_limit_count}.")

    print("\nPrompt token statistics:")
    print(stats.to_string(index=False))

    return output_path, stats_path


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens and summarize prompt sizes."
    )
    parser.add_argument("input", type=Path, help="Dataset CSV path")
    input_path = parser.parse_args().input.resolve()

    load_dotenv(ENV_FILE)
    characterize(input_path)


if __name__ == "__main__":
    main()
