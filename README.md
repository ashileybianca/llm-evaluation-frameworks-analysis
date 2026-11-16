# LLM Evaluation Frameworks Analysis

Code and selected research artifacts for **An Investigation of Metric Divergence Across LLM Evaluation Frameworks**. The study compares DeepEval, RAGAS, and Opik when they evaluate answers generated for question-answering datasets.

## Research at a glance

The experiment asks whether frameworks that offer metrics with the same names assign comparable scores to the same answers. Twenty frameworks were surveyed; DeepEval, RAGAS, and Opik were selected because they were eligible for the experiment and had the greatest overlap among the four metrics studied.

| Component | Experimental setup |
| --- | --- |
| Datasets | SQuAD validation, HotpotQA distractor validation, and MS MARCO v1.1 validation |
| Sample | 50 eligible examples per dataset, sampled with seed 42 |
| Answer generator | Llama 3.2 3B, run locally with Ollama |
| LLM judge | Qwen 2.5 7B, run locally with Ollama |
| Frameworks | DeepEval 4.2.8, RAGAS 0.4.3, and Opik 2.2.99 |
| Metrics | Answer Relevancy, Faithfulness, Context Precision, and Context Recall |
| Repetitions | Three evaluation runs per framework and dataset |
| Core settings | Llama: temperature 0, seed 42, context 4,096, up to 256 output tokens; Qwen judge: temperature 0, context 8,192 |

For each dataset, all three frameworks score the same 50 generated answers with the same judge model and temperature; only the evaluation framework changes. The four metrics broadly assess answer relevance, support from the supplied context, and the precision and recall of that context. Their implementations are framework-specific, so same-named metrics should not be assumed to work identically.

The study measures consistency and agreement between tools; it does **not** establish which framework is correct. It has no human-annotated scores as ground truth. SQuAD provides a limited check for Context Precision and Context Recall because each question has one context passage containing its reference answer.

## Observed findings in this setup

- **Observed repeatability:** Across the three runs, scores were identical for 99.8% of DeepEval samples, 98.2% of RAGAS samples, and 99.0% of Opik samples.
- **Limited cross-framework agreement in this setup:** 30 of the 36 pairwise Kendall rank correlations were at most 0.30. The framework ranking by mean score was consistent across all three datasets only for Faithfulness.
- **Dataset-dependent disagreement:** Disagreement was observed on all datasets and was largest on HotpotQA. RAGAS gave Context Recall a score of 1 for 84% of HotpotQA examples and 90% of MS MARCO examples; in this study, the relevant passages were included in each example's context.
- **Different resource usage profiles:** Opik used the fewest judge calls and the least time. RAGAS used about twice as many judge tokens as DeepEval or Opik, largely because its Context Precision evaluates passages individually.
- **Limited SQuAD structure check:** For the retrieval metrics that can be checked against SQuAD's one-passage structure, RAGAS more often returned the expected score of 1 than DeepEval. This is a dataset-structure check, not a human evaluation.

These are results from this setup, not universal rankings. They use one generator, one judge, three English QA datasets, and 50 examples per dataset; changing these choices may change the results.

## Repository contents

| Path | What it contains |
| --- | --- |
| `pipeline/` | Dataset download, deterministic sampling, prompt/token characterization, and answer generation |
| `evaluation/` | Evaluators for the three frameworks, the command-line entry point, and the local Ollama judge proxy |
| `analysis/` | Scripts for stability, agreement, dataset effects, score distributions, resource use, and figures |
| `datasets/raw/`, `datasets/samples/`, `datasets/llama-responses/` | SQuAD and HotpotQA source splits, selected samples, token counts, and generated answers |
| `datasets/public-results/` | Per-run evaluation CSVs for SQuAD and HotpotQA (MS MARCO per-sample rows are not redistributed here) |
| `logs/judge_calls/judge_calls_public.jsonl` | Filtered judge-call metadata and token counts; request and response bodies are not recorded |
| `figures/` | Selected figures from the analysis |
| `modelfiles/` | Ollama model definitions used for the generator and judge |
| `config.py`, `config.yaml` | Experiment settings and paths |
| `pyproject.toml`, `uv.lock`, `.python-version` | Python dependency declarations, exact dependency lock, and Python version |
| `LICENSE` | MIT license for the repository's original code |

The manuscript, internal notes, full generated reports, MLflow database/artifacts, machine-specific logs, and virtual environments are not included. Runtime outputs are generated locally and ignored by Git.

## Data sources and licensing

### SQuAD and HotpotQA

The SQuAD and HotpotQA dataset sources are marked **CC BY-SA 4.0**. This repository includes material from those sources and data-bearing derivatives (samples, generations, and evaluation rows), reformatted as JSONL and CSV. Follow the datasets' attribution and share-alike terms when reusing those files.

- SQuAD: Rajpurkar et al., [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264/). [Dataset and license](https://huggingface.co/datasets/rajpurkar/squad).
- HotpotQA: Yang et al., [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://aclanthology.org/D18-1259/). [Dataset and license](https://huggingface.co/datasets/hotpotqa/hotpot_qa).

The repository's MIT license covers the original code only. It does not replace or extend the terms for third-party datasets or their contents.

### MS MARCO

The repository does not include MS MARCO questions, passages, references, generated answers, or per-sample evaluation CSVs. MS MARCO per-sample public result files were removed from this repository.

MS MARCO usage follows its own [terms](https://microsoft.github.io/msmarco/). This repository's code and documentation are licensed here, while MS MARCO content remains governed by MS MARCO terms. The filtered public judge-call log contains metadata only, not prompt or response bodies.

`pipeline/download.py` can download MS MARCO for local research. Review and accept its terms before downloading. `.gitignore` excludes local MS MARCO source/sample/generation files and full evaluation outputs; this is a repository safeguard, not a license grant.

## Reproduce the included analysis

The published CSVs and filtered call log can be analyzed without downloading datasets or running the models. Install `uv`, open PowerShell at the repository root, and sync the locked environment:

```powershell
uv sync --locked
uv run --locked python --version
```

Python 3.12.10 is specified in `.python-version`; `uv.lock` pins the direct and transitive Python packages. `uv sync` creates a local `.venv`, which is not committed. Use `uv run --locked` below so commands fail rather than updating an outdated lockfile.

Run the analysis on the redistributed public CSVs:

```powershell
uv run --locked python analysis\analyze_results.py --input-dir datasets\public-results --judge-call-log logs\judge_calls\judge_calls_public.jsonl --skip-bootstrap
```

The analysis script writes reports under `reports/`. Reports and intermediate files are local outputs and are ignored by Git.

## Reproduce the full experiment

Full reproduction requires Python 3.12.10, `uv`, Ollama, access to the gated Llama 3.2 tokenizer on Hugging Face, and local hardware capable of running the models. A supported GPU is recommended to reduce runtime. The commands below are written for Windows PowerShell.

### 1. Configure Hugging Face and Ollama

Create `pipeline/.env` locally with a Hugging Face read token. Do not commit this file:

```text
HF_TOKEN=your_token
```

Create the Ollama models from the included model files:

```powershell
ollama create llama3.2-3b-ctx4096 -f modelfiles/Modelfile.llama
ollama create qwen2.5-7b-ctx8192 -f modelfiles/Modelfile.qwen
ollama list
```

The original run recorded model IDs `19849e529231` for the Llama generator and `99136033e9ba` for the Qwen judge. The Modelfiles use Ollama tags, and the scripts do not enforce those recorded IDs; compare the installed IDs with them if exact model reproduction is important. Hugging Face dataset and tokenizer revisions are pinned in `config.py`; the first RAGAS run downloads the embedding model `BAAI/bge-base-en-v1.5`.

### 2. Download, sample, and prepare the data

Start MLflow in one terminal and leave it running:

```powershell
uv run --locked python -m mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination .\mlartifacts --serve-artifacts --host 127.0.0.1 --port 5000
```

In another terminal, run the pipeline from the repository root:

```powershell
uv run --locked python pipeline\download.py
uv run --locked python pipeline\sample.py
uv run --locked python pipeline\characterize.py datasets\samples\squad_sample.csv
uv run --locked python pipeline\characterize.py datasets\samples\hotpotqa_sample.csv
uv run --locked python pipeline\characterize.py datasets\samples\msmarco_sample.csv
uv run --locked python pipeline\generate.py
```

The sampler filters for examples with a reference answer and at least one relevant passage, then selects 50 examples from each dataset using seed 42. The characterization step counts tokens for the prompts and checks them against the generator's context window before generation. The generation script writes one answer per sample and records token and timing metadata.

### 3. Run the evaluators

Start the local judge proxy in a third terminal and leave it running:

```powershell
uv run --locked python evaluation\judge_proxy.py
```

The proxy forwards judge requests to Ollama and records request metadata and token counts, but not request or response bodies. Run evaluations sequentially. This PowerShell loop runs all three frameworks, all three datasets, and repetitions 0–2:

```powershell
$datasets = @("squad", "hotpotqa", "msmarco")
$frameworks = @("ragas", "deepeval", "opik")
foreach ($repeat in 0..2) {
    foreach ($dataset in $datasets) {
        foreach ($framework in $frameworks) {
            uv run --locked python evaluation\main.py $framework $dataset --repeat $repeat
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $framework / $dataset / repeat $repeat" }
        }
    }
}
```

The same evaluation command can be run individually:

```powershell
uv run --locked python evaluation\main.py ragas squad --repeat 0
```

Each run is written to `datasets/qwen-evaluation/`. If a run is interrupted, rerunning the same command skips samples already present in its CSV. These full local CSVs are excluded from Git.

### 4. Analyze a full local run

After the evaluations finish, analyze the full local CSVs with the runtime judge-call log:

```powershell
uv run --locked python analysis\analyze_results.py --input-dir datasets\qwen-evaluation --judge-call-log logs\judge_calls\judge_calls.jsonl
uv run --locked python analysis\plot_score_heatmap.py --input-dir datasets\qwen-evaluation
uv run --locked python analysis\plot_analysis_figures.py
```

The runtime reports, consolidated CSVs, MLflow files, and full evaluation CSVs remain local and are ignored by Git.
