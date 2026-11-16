import pandas as pd
from config import JUDGE_MODEL, JUDGE_TEMPERATURE, PROXY_URL
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from evaluators.base_evaluator import BaseEvaluator


class DeepEvalEvaluator(BaseEvaluator):
    """Scores with DeepEval. Faithfulness is the share of the output claims that the context does not contradict
    (claims the context does not mention still count as faithful).
    """

    framework = "deepeval"

    def __init__(self, dataset: pd.DataFrame, output_path: str, run_name: str):
        super().__init__(dataset, output_path, run_name)

        judge = OllamaModel(model=JUDGE_MODEL, base_url=PROXY_URL, temperature=JUDGE_TEMPERATURE)

        # async_mode=False keeps the judge calls one at a time
        self.answer_relevancy = AnswerRelevancyMetric(model=judge, async_mode=False)
        self.faithfulness = FaithfulnessMetric(model=judge, async_mode=False)
        self.context_precision = ContextualPrecisionMetric(model=judge, async_mode=False)
        self.context_recall = ContextualRecallMetric(model=judge, async_mode=False)

    def _evaluate_row(self, row: pd.Series) -> dict:
        test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["output"],
            expected_output=row["reference"],
            retrieval_context=row["context"],
        )

        def answer_relevancy():
            score = self.answer_relevancy.measure(test_case, _show_indicator=False)
            return score, self.answer_relevancy.reason

        def faithfulness():
            score = self.faithfulness.measure(test_case, _show_indicator=False)
            return score, self.faithfulness.reason

        def context_precision():
            score = self.context_precision.measure(test_case, _show_indicator=False)
            return score, self.context_precision.reason

        def context_recall():
            score = self.context_recall.measure(test_case, _show_indicator=False)
            return score, self.context_recall.reason

        return self.score_row(
            row,
            {
                "answer_relevancy": answer_relevancy,
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "context_recall": context_recall,
            },
        )
