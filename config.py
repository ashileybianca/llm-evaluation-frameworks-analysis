"""All fixed values of the experiment: folders, datasets, seeds, model settings and MLflow."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

ENV_FILE = ROOT / CONFIG["env_file"]  # HF_TOKEN
DATASETS_DIR = ROOT / CONFIG["datasets"]
RAW_DIR = DATASETS_DIR / "raw"
SAMPLES_DIR = DATASETS_DIR / "samples"
GENERATIONS_DIR = DATASETS_DIR / "llama-responses"
REPORTS_DIR = ROOT / CONFIG["reports"]
LOGS_DIR = ROOT / CONFIG["logs"]
MODELFILE = ROOT / CONFIG["modelfiles"] / "Modelfile.llama"
MODELFILE_JUDGE = ROOT / CONFIG["modelfiles"] / "Modelfile.qwen"
EVALUATIONS_DIR = DATASETS_DIR / "qwen-evaluation"
JUDGE_CALLS_DIR = LOGS_DIR / "judge_calls"
JUDGE_CALLS_PATH = JUDGE_CALLS_DIR / "judge_calls.jsonl"  # written by evaluation/judge_proxy.py
CURRENT_METRIC_PATH = JUDGE_CALLS_DIR / "current_metric.json"  # written by the evaluators, read by the proxy
ANALYSIS_DIR = REPORTS_DIR / "analysis"  # written by analysis/analyze_results.py
PYPROJECT = ROOT / "pyproject.toml"
UV_LOCK = ROOT / "uv.lock"

# Hugging Face revisions keep downloaded datasets and tokenizers stable.
DATASET_REVISIONS = {
    "squad": "7b6d24c440a36b6815f21b70d25016731768db1f",
    "hotpotqa": "1908d6afbbead072334abe2965f91bd2709910ab",
    "msmarco": "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a",
}

# (hf_path, config, split); download.py saves them to RAW_DIR
# MS MARCO stays on v1.1 (the plan said v2.1) so the data and the token stats already computed remain valid
DATASET_SOURCES = {
    "squad": ("rajpurkar/squad", None, "validation"),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "validation"),
    "msmarco": ("microsoft/ms_marco", "v1.1", "validation"),
}

N_SAMPLES = 50
SAMPLING_SEED = 42

FRAMEWORKS = ["ragas", "deepeval", "opik"]
METRICS = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]

GENERATION_MODEL = "llama3.2-3b-ctx4096"  # created from BASE_MODEL with modelfiles/Modelfile.llama
BASE_MODEL = "llama3.2:3b"
NUM_CTX = 4096
NUM_PREDICT = 256
TEMPERATURE = 0
GENERATION_SEED = 42

JUDGE_MODEL = "qwen2.5-7b-ctx8192"  # created from JUDGE_BASE_MODEL with modelfiles/Modelfile.qwen
JUDGE_BASE_MODEL = "qwen2.5:7b"
JUDGE_NUM_CTX = 8192
# Set explicitly for every framework: Opik sends none (Ollama would use 0.8) and RAGAS defaults to 0.01
JUDGE_TEMPERATURE = 0
OLLAMA_URL = "http://127.0.0.1:11434"
PROXY_URL = "http://127.0.0.1:11435"  # judge_proxy.py; every framework calls the judge through it
EMBEDDINGS_MODEL = "BAAI/bge-base-en-v1.5"  # used by RAGAS Answer Relevancy, same as TC1
EMBEDDINGS_DEVICE = "cpu"

TOKENIZERS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}
TOKENIZER_REVISIONS = {
    "llama": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "qwen": "a09a35458c702b33eeacc393d103063234e8bc28",
}

MLFLOW_EXPERIMENT = "llm-evaluation-frameworks"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
