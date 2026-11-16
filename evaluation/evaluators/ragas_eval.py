import asyncio

import pandas as pd
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from config import (
    EMBEDDINGS_DEVICE,
    EMBEDDINGS_MODEL,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    PROXY_URL,
)
from evaluators.base_evaluator import BaseEvaluator


class RagasEvaluator(BaseEvaluator):
    """Scores with RAGAS. Faithfulness is the share of the output claims that the context supports.

    RAGAS gives no reason for its scores, so the reason columns stay empty.
    """

    framework = "ragas"

    def __init__(self, dataset: pd.DataFrame, output_path: str, run_name: str):
        super().__init__(dataset, output_path, run_name)

        llm = llm_factory(
            JUDGE_MODEL,
            client=AsyncOpenAI(base_url=f"{PROXY_URL}/v1", api_key="ollama"),
            temperature=JUDGE_TEMPERATURE,
        )
        embeddings = HuggingFaceEmbeddings(
            model=EMBEDDINGS_MODEL,
            device=EMBEDDINGS_DEVICE,
        )

        self.answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
        self.faithfulness = Faithfulness(llm=llm)
        self.context_precision = ContextPrecisionWithReference(llm=llm)
        self.context_recall = ContextRecall(llm=llm)

    def _evaluate_row(self, row: pd.Series) -> dict:
        def answer_relevancy():
            result = asyncio.run(self.answer_relevancy.ascore(user_input=row["input"], response=row["output"]))
            return result.value, result.reason

        def faithfulness():
            result = asyncio.run(
                self.faithfulness.ascore(user_input=row["input"], response=row["output"], retrieved_contexts=row["context"])
            )
            return result.value, result.reason

        def context_precision():
            result = asyncio.run(
                self.context_precision.ascore(user_input=row["input"], reference=row["reference"], retrieved_contexts=row["context"])
            )
            return result.value, result.reason

        def context_recall():
            result = asyncio.run(
                self.context_recall.ascore(user_input=row["input"], reference=row["reference"], retrieved_contexts=row["context"])
            )
            return result.value, result.reason

        return self.score_row(
            row,
            {
                "answer_relevancy": answer_relevancy,
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "context_recall": context_recall,
            },
        )
