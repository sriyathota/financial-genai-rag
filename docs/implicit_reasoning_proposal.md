# Implicit Reasoning (Hidden CoT) Proposal

## Context
The current RAG pipeline retrieves trading textbook chunks and generates
answers directly from retrieved context using an LLM.

## Observation
Answer generation does not explicitly encourage internal reasoning or verification
over retrieved documents. The model outputs a final answer directly.

## Proposal
Introduce an optional prompt-level instruction that encourages internal reasoning
while enforcing final-answer-only output.

This will be gated behind a flag (e.g., USE_HIDDEN_COT) to allow controlled experiments.

## Pseudocode
```text
if USE_HIDDEN_COT:
    system_prompt += "Use careful internal reasoning to answer, but output only the final answer."

answer = LLM(system_prompt + context + question)
