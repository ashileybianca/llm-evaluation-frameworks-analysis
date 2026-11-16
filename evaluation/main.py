"""python main.py {ragas|deepeval|opik} {squad|hotpotqa|msmarco} [--repeat N]

The judge proxy (judge_proxy.py) must be running. If the run stops, run the same command again:
the samples already in the CSV are skipped.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from config import DATASET_SOURCES, EVALUATIONS_DIR, FRAMEWORKS
from dataset_io import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("framework", choices=FRAMEWORKS)
    parser.add_argument("dataset", choices=DATASET_SOURCES)
    parser.add_argument("--repeat", type=int, default=0, help="repetition number, part of the CSV name")
    args = parser.parse_args()

    # Import only the selected framework because its dependencies are heavy
    if args.framework == "ragas":
        from evaluators.ragas_eval import RagasEvaluator as Evaluator
    elif args.framework == "deepeval":
        from evaluators.deepeval_eval import DeepEvalEvaluator as Evaluator
    else:
        from evaluators.opik_eval import OpikEvaluator as Evaluator

    run_name = f"{args.dataset}-{args.framework}-eval-r{args.repeat}"
    results = Evaluator(load_dataset(args.dataset), EVALUATIONS_DIR, run_name).evaluate()
    print(f"{len(results)} rows in {EVALUATIONS_DIR / (run_name + '.csv')}")


if __name__ == "__main__":
    main()
