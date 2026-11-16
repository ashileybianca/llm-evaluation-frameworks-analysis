import pandas as pd
from config import JUDGE_MODEL, JUDGE_TEMPERATURE, PROXY_URL
from opik.evaluation.metrics import (
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
    Hallucination,
)
from opik.evaluation.models import LiteLLMChatModel

from evaluators.base_evaluator import BaseEvaluator


class OpikEvaluator(BaseEvaluator):
    """Scores with Opik, which has no Faithfulness metric, only Hallucination (higher is worse).

    hallucination_score is the original Opik value; faithfulness_score is 1 - hallucination_score.
    """

    framework = "opik"
    faithfulness_source = "Hallucination (1 - score)"

    def __init__(self, dataset: pd.DataFrame, output_path: str, run_name: str):
        super().__init__(dataset, output_path, run_name)

        judge = LiteLLMChatModel(
            f"openai/{JUDGE_MODEL}",
            track=False,
            api_base=f"{PROXY_URL}/v1",
            api_key="ollama",
            temperature=JUDGE_TEMPERATURE,
        )

        self.answer_relevancy = AnswerRelevance(model=judge, track=False)
        self.hallucination = Hallucination(model=judge, track=False)
        self.context_precision = ContextPrecision(model=judge, track=False)
        self.context_recall = ContextRecall(model=judge, track=False)

    def _evaluate_row(self, row: pd.Series) -> dict:
        hallucination_score = None

        def answer_relevancy():
            result = self.answer_relevancy.score(input=row["input"], output=row["output"], context=row["context"])
            return result.value, result.reason

        def faithfulness():
            nonlocal hallucination_score
            result = self.hallucination.score(input=row["input"], output=row["output"], context=row["context"])
            hallucination_score = result.value
            return 1 - result.value, result.reason

        def context_precision():
            result = self.context_precision.score(
                input=row["input"], output=row["output"], expected_output=row["reference"], context=row["context"]
            )
            return result.value, result.reason

        def context_recall():
            result = self.context_recall.score(
                input=row["input"], output=row["output"], expected_output=row["reference"], context=row["context"]
            )
            return result.value, result.reason

        result = self.score_row(
            row,
            {
                "answer_relevancy": answer_relevancy,
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "context_recall": context_recall,
            },
        )
        result["hallucination_score"] = hallucination_score
        return result
