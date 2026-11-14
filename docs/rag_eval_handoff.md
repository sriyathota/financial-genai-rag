# RAG Evaluation — Methodology & Results Handoff

*Prepared by Jacky for the LLM CoT subteam. This document is the new
material to fold into the paper's Methodology and Results sections.
All numbers reproduce from the commands in `README.md` and the per-case
CSVs in the repo root.*

---

## 1. Summary

Four-configuration ablation of the local RAG pipeline isolates the
contributions of (i) prompt engineering, (ii) LLM temperature,
(iii) a hidden Chain-of-Thought instruction, and (iv) a cross-encoder
reranker. Best configuration achieves mean Rouge-1 F1 = **0.307** and
mean BERTScore F1 = **0.290** across 97 questions. Disabling CoT was
the single largest contributor (+0.032 BERT mean, ≈ 5× the baseline
run-to-run σ); adding the reranker contributed a further +0.010 BERT
mean and +8.2 pp at τ = 0.3.

---

## 2. Pipeline (Methodology §)

- **Corpus:** 14 trading PDFs → `RecursiveCharacterTextSplitter`
  (chunk = 750 chars, overlap = 100) → ~10,000 chunks.
- **Embedder:** `BAAI/bge-small-en-v1.5` (33M params, CPU, normalized).
- **Vector store:** FAISS `IndexFlatL2`.
- **Retriever:** FAISS top-5 (default) or FAISS top-20 → BGE-reranker-base
  cross-encoder → top-5 (reranker variant).
- **LLM:** Ollama `llama3.2:3b`; temperature = 0 unless noted.
- **Prompt:** tight system instruction + three few-shot QA exemplars
  (verbatim in `eval.py`). A separate hidden-CoT instruction block is
  appended when `USE_HIDDEN_COT=1`.
- **Metrics:** Rouge-1 F1 and BERTScore F1 via DeepEval's `Scorer`.
  BERTScore uses `microsoft/deberta-large-mnli` with **baseline
  rescaling** — a random-text similarity is subtracted, so negative
  values are possible and scores are *not* directly comparable to
  unrescaled BERTScore in the 0.80–0.95 range.
- **Determinism / variance:** T = 0 produces byte-equivalent outputs
  on repeat runs. Row A was run three times at T = 0.5 to characterize
  noise: σ = 0.004 Rouge, σ = 0.006 BERT.

---

## 3. Experimental design (Results §)

Each row changes exactly one variable from the row above:

| Row | Prompt                | T    | CoT | Reranker | Replicates |
|-----|-----------------------|------|-----|----------|------------|
| A   | Verbose, "be concise" | 0.5  | On  | Off      | 3          |
| B   | Tight + 3-shot        | 0.0  | On  | Off      | 1          |
| C   | Tight + 3-shot        | 0.0  | Off | Off      | 1          |
| D   | Tight + 3-shot        | 0.0  | Off | On       | 1          |

Same 97 eval questions, embedder, vector store, chunk parameters, LLM.
In Row D the reranker picks 5 from an initial 20-candidate FAISS pool.

---

## 4. Results

### 4.1 Aggregate statistics

**Rouge-1 F1**

| # | Config                  | Mean          | Median | Max   |
|---|-------------------------|---------------|--------|-------|
| A | Baseline (N = 3)        | 0.284 ± 0.004 | —      | 0.828 |
| B | Tight + 3-shot, CoT on  | 0.281         | 0.279  | 0.914 |
| C | + CoT off               | 0.293         | 0.273  | 0.800 |
| D | + reranker              | **0.307**     | 0.286  | 0.914 |

**BERTScore F1**

| # | Config                  | Mean          | Median | Max   |
|---|-------------------------|---------------|--------|-------|
| A | Baseline (N = 3)        | 0.252 ± 0.006 | —      | 0.826 |
| B | Tight + 3-shot, CoT on  | 0.248         | 0.241  | 0.871 |
| C | + CoT off               | 0.280         | 0.253  | 0.795 |
| D | + reranker              | **0.290**     | 0.286  | 0.871 |

### 4.2 Threshold sensitivity

Pass rates computed post hoc on the same scores (no re-running).

**Rouge**

| # | Config    | τ ≥ 0.2   | τ ≥ 0.3   | τ ≥ 0.4   | τ ≥ 0.5   |
|---|-----------|-----------|-----------|-----------|-----------|
| A | Baseline  | 70.8%     | 37.1%     | 19.2%     | 10.6%     |
| B | + prompt  | 68.0%     | 42.3%     | 16.5%     |  8.2%     |
| C | + CoT off | 71.1%     | 39.2%     | 21.6%     | **12.4%** |
| D | + rerank  | **75.3%** | **43.3%** | **24.7%** |  9.3%     |

**BERT**

| # | Config    | τ ≥ 0.2 | τ ≥ 0.3   | τ ≥ 0.4   | τ ≥ 0.5   |
|---|-----------|---------|-----------|-----------|-----------|
| A | Baseline  | 57.4%   | 33.7%     | 14.1%     |  7.9%     |
| B | + prompt  | 56.7%   | 33.0%     | 13.4%     |  5.2%     |
| C | + CoT off | 68.0%   | 39.2%     | **19.6%** | 10.3%     |
| D | + rerank  | 68.0%   | **47.4%** | 18.6%     | **11.3%** |

### 4.3 Ablation-by-ablation

- **A → B (prompt + determinism).** Means flat, maxima jump
  (+0.09 Rouge, +0.05 BERT). Few-shot exemplars teach a tight answer
  style that yields near-verbatim matches on stylistically similar
  questions; they also trigger 4–5 honest "I don't know" abstentions
  on retrieval failures. Lifted ceiling + new zeros cancel at the mean.
- **B → C (CoT off) — hero ablation.** BERT mean +0.032 (≈ 5σ above
  baseline noise). BERT τ = 0.5 pass rate **doubles** (5.2 → 10.3%);
  τ = 0.2 jumps +11.3 pp. CoT-on keeps the higher per-case max on
  clean retrieval, but on aggregate the CoT instruction induces
  middle-distribution hedging that dominates for a 3B model.
- **C → D (reranker).** Both means lift modestly (+0.014 Rouge,
  +0.010 BERT). Largest single delta in the table is BERT τ = 0.3
  (+8.2 pp). Rouge τ = 0.5 drops 3.1 pp: the reranker trades some
  near-verbatim extracts for a larger number of semantically correct
  paraphrases that surface-overlap only partially rewards.

### 4.4 Case-level notes

- **List / enumeration questions are the reranker's biggest wins.**
  Turtles' five key questions Rouge 0.22 → 0.74; Jim Rogers's
  principles 0.37 → 0.71; key futures-market advantages 0.23 → 0.55.
- **5 consistent abstentions** across B/C/D represent correct
  behavior (context genuinely insufficient), not pipeline failure,
  but drag the means down near zero.
- **Dataset typo:** "Why did Richard Dennis return to trading in
  1944?" — Dennis was born in 1949; likely 1994. Model correctly
  abstains across all rows. Worth footnoting.

---

## 5. Points for Discussion

1. **Hidden-CoT's benefit on RAG is model-size-dependent.** On a 3B
   model the instruction induces hedging rather than useful silent
   reasoning; net aggregate effect is negative.
2. **Surface-overlap metrics under-credit retrieval-precision gains.**
   Reranker-driven paraphrase substitutions show up most clearly at
   τ = 0.3, not τ = 0.5 — motivating the threshold sweep over a single
   pass-rate number.
3. **BERTScore range is convention-dependent.** Baseline-rescaled
   (0.29 mean here) is not comparable to raw (0.80+). Paper should
   specify.
4. **Reproducibility requires T = 0.** σ ≈ 0.005 on baseline means
   that any reported improvement below ~0.015 at T > 0 is within
   sampling noise.

---

## 6. Inconsistencies to reconcile with the draft

1. **BERTScore F1 = 0.8180** in the abstract ≠ the 0.29 measured here.
   Likely raw vs rescaled. Choose one convention and state it.
2. **BLEU** is emphasized in the abstract but not computed here.
   Reinstate + rerun, or revise to Rouge-1 + BERTScore.
3. **"Llama3, Gemini, Mistral AI"** in the abstract — this ablation
   covers llama3.2:3b only. Narrow the claim if no cross-model results
   exist elsewhere.
4. **Chunk size** — abstract says 500, code uses 750. Update one.

---

## 7. Reproduction

`README.md` has setup and the four run commands. Per-case CSVs
(`eval_outputs_A1…D_reranker.csv`) let teammates recompute any
statistic without re-running the pipeline. Out of scope: DPO/RLHF
training, cross-model comparison, BLEU, chunk-size sweep, live data,
trading backtest.
