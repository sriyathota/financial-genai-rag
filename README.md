# VIP-GenAI — Local RAG Evaluation

A retrieval-augmented generation (RAG) pipeline over a corpus of 14 trading books, evaluated with Rouge-1 and BERTScore on a 97-question evaluation set.

## Pipeline components

- **Corpus:** 14 trading-literature PDFs in `trading_books/`
- **Chunking:** `RecursiveCharacterTextSplitter`, chunk_size=750, overlap=100
- **Embedder:** `BAAI/bge-small-en-v1.5` (local, CPU)
- **Vector store:** FAISS (`IndexFlatL2`)
- **Retriever:** top-*k* FAISS retrieval, optionally reranked with `BAAI/bge-reranker-base` cross-encoder
- **LLM:** Ollama `llama3.2` (3B parameters)
- **Metrics:** Rouge-1 F1 and BERTScore F1 (DeBERTa-large-mnli, baseline-rescaled) via DeepEval

## Setup

```bash
# System dependencies
brew install ollama
ollama serve &              # background service
ollama pull llama3.2        # ~2GB, one-time

# Python dependencies
pip install langchain langchain-community faiss-cpu sentence-transformers \
            pypdf pandas openpyxl deepeval rouge-score bert-score
```

## Running the evaluation

The evaluation harness is `eval.py`. It reads `eval_dataset.csv`, runs the RAG pipeline on every question, scores the outputs, prints a summary + threshold sweep, and writes per-case results to `eval_outputs.csv`.

Configuration is controlled via environment variables:

| Variable | Values | Effect |
|---|---|---|
| `USE_HIDDEN_COT` | `0` / `1` | Disable / enable hidden chain-of-thought prompt instruction |
| `USE_RERANKER` | `0` / `1` | Disable / enable cross-encoder reranker |
| `TEMPERATURE` | float | LLM sampling temperature (0 = deterministic) |

Three mandatory macOS env vars prevent OpenMP library conflicts:

```
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
```

### Reproducing the four ablation configurations

```bash
# Row A — Baseline (verbose prompt, T=0.5, CoT on)
# NOTE: requires swapping SYSTEM_PROMPT in eval.py to the verbose version
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  USE_HIDDEN_COT=1 USE_RERANKER=0 TEMPERATURE=0.5 python eval.py

# Row B — Tight+few-shot prompt, T=0, CoT on
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  USE_HIDDEN_COT=1 USE_RERANKER=0 TEMPERATURE=0 python eval.py

# Row C — Tight+few-shot prompt, T=0, CoT OFF (best aggregate without reranker)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  USE_HIDDEN_COT=0 USE_RERANKER=0 TEMPERATURE=0 python eval.py

# Row D — Tight+few-shot prompt, T=0, CoT OFF, + reranker (best overall)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  USE_HIDDEN_COT=0 USE_RERANKER=1 TEMPERATURE=0 python eval.py
```

Each run takes roughly 15–25 minutes depending on whether the reranker is enabled.

## Key results (full details in the paper)

| Config | Rouge Mean | BERT Mean | Rouge Pass@0.5 | BERT Pass@0.5 |
|---|---|---|---|---|
| A (baseline, 3 runs) | 0.284 ± 0.004 | 0.252 ± 0.006 | 10.6% | 7.9% |
| B (tight + few-shot) | 0.281 | 0.248 | 8.2% | 5.2% |
| C (+ CoT off) | 0.293 | 0.280 | 12.4% | 10.3% |
| D (+ reranker) | **0.307** | **0.290** | 9.3% | 11.3% |

Per-case outputs are saved as `eval_outputs_<row>.csv` for each configuration.

## Repo layout

```
.
├── LocalRAG.py                   # Pipeline: loaders, chunker, embedder, retriever, LLM chain
├── eval.py                       # Evaluation harness + summary/threshold sweep printing
├── non_llm_scores.py             # Custom Rouge & BERTScore metric wrappers (with BGE rescaling fix)
├── dataset.py                    # Excel → CSV conversion for the eval dataset
├── custom_llm_for_evaluation.py  # Wraps Gemini as DeepEval judge LLM (unused in current run)
├── trading_books/                # 14 PDFs (source corpus)
├── eval_dataset.csv              # 97 Q/A/reference-context rows
├── eval_outputs_*.csv            # Per-case outputs from each ablation run
└── Dataset for RAG Eval.xlsx     # Original source of eval_dataset.csv
```
