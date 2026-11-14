from urllib import response

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
import pandas as pd
from dataset import *
from custom_llm_for_evaluation import CustomGeminiModel
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric,  AnswerRelevancyMetric,  FaithfulnessMetric
from LocalRAG import create_whole_pipeline
from non_llm_scores import RougeMetric, BERTScore
from pathlib import Path

# SYSTEM_PROMPT = (
#     "You are an expert on trading strategies and quantitative trading. "
#     "Your job is to guide and assist traders by evaluating their strategies, "
#     "determining whether they are profitable, and answering their questions. "
#     "Formulate your answers based on the following context as well as your own "
#     "knowledge about trading and investments. Be concise."
# )

SYSTEM_PROMPT = (
    "You are an expert on trading strategies and quantitative trading. "
    "Answer the user's question using ONLY the provided context below. "
    "Answer in a single sentence. "
    "Do not preamble with phrases like 'Based on the context', "
    "'According to the text', or 'The text states that'. "
    "Start directly with the answer itself. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the provided context.\"\n\n"
    "Examples of the expected answer style:\n\n"
    "Question: What is volume often called in trading analysis?\n"
    "Answer: A secondary indicator that must be placed in the context of other market information to be useful.\n\n"
    "Question: What is the most common method for creating linked-contract charts?\n"
    "Answer: The nearest futures approach, where each contract is plotted until expiration before moving to the next.\n\n"
    "Question: What does the RSI measure?\n"
    "Answer: The relative internal strength of a position compared to itself.\n"
)

BASE_DIR = Path(__file__).resolve().parent
TEXTBOOKS_FILEPATH = BASE_DIR / "trading_books"
EVAL_CSV = BASE_DIR / "eval_dataset.csv"
VECTORSTORE_SAVE_PATH = BASE_DIR
# TEXTBOOKS_FILEPATH = "/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/trading_books"
# EVAL_CSV = "/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/eval_dataset.csv"
# VECTORSTORE_SAVE_PATH = "/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/"

def _print_score_summary(results):
    from collections import defaultdict
    by_metric = defaultdict(list)
    for tr in results.test_results:
        for m in tr.metrics_data:
            by_metric[m.name].append((float(m.score), bool(m.success)))
    print()
    print(f"{'Metric':<15} {'N':<5} {'Mean':<8} {'Median':<8} {'Min':<8} {'Max':<8} {'Pass%':<6}")
    print("-" * 60)
    for name, data in sorted(by_metric.items()):
        scores = sorted(s for s, _ in data)
        n = len(scores)
        passes = sum(1 for _, p in data if p)
        median = scores[n // 2]
        print(f"{name:<15} {n:<5} {sum(scores)/n:<8.3f} {median:<8.3f} {min(scores):<8.3f} {max(scores):<8.3f} {100*passes/n:<5.1f}%")

    # Threshold sweep on the same scores — no re-running.
    thresholds = [0.2, 0.3, 0.4, 0.5]
    print()
    print("Pass rate by threshold (same scores, post-hoc):")
    header = f"{'Metric':<15} " + " ".join(f"τ≥{t}".ljust(8) for t in thresholds)
    print(header)
    print("-" * len(header))
    for name, data in sorted(by_metric.items()):
        scores = [s for s, _ in data]
        n = len(scores)
        cells = " ".join(f"{100*sum(1 for s in scores if s >= t)/n:5.1f}%".ljust(8) for t in thresholds)
        print(f"{name:<15} " + cells)


def _save_per_case_outputs(results, path):
    rows = []
    for tr in results.test_results:
        row = {
            "input": tr.input,
            "actual_output": tr.actual_output,
            "expected_output": tr.expected_output,
        }
        for m in tr.metrics_data:
            row[f"{m.name} score"] = round(float(m.score), 4)
            row[f"{m.name} pass"] = bool(m.success)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Per-case outputs saved to {path}")


def add_actual_outputs_to_dataset(data : pd.DataFrame):
    rag_chain = create_whole_pipeline(
        system_prompt=SYSTEM_PROMPT,
        documents_dir=TEXTBOOKS_FILEPATH,
        vectorstore_save_path=VECTORSTORE_SAVE_PATH
    )

    def call_rag_chain(input_prompt):
        # response = rag_chain.invoke({
        #     "input": input_prompt
        # })
        # return response['answer'], response['context']
        response = rag_chain.invoke({
            "query": input_prompt
        })
        return response['result'], "\n\n".join(d.page_content for d in response['source_documents'])
    data[["actual_output", "actual_retreived_context"]] = data["input"].apply(call_rag_chain).apply(pd.Series)
    return data


def create_dataset_from_csv(csv_filepath):
    dataset = EvaluationDataset()

    dataset.add_goldens_from_csv_file(
        file_path=csv_filepath,
        input_col_name="input",
        expected_output_col_name="expected_output",
        retrieval_context_col_name="retreived_context"
    )

    return dataset

def build_eval_dataset(df: pd.DataFrame) -> EvaluationDataset:
    dataset = EvaluationDataset()

    for _, row in df.iterrows():
        dataset.add_test_case(
            LLMTestCase(
                input=row["input"],
                expected_output=row["expected_output"],
                actual_output=row["actual_output"],
                context=[row["retreived_context"]] if pd.notna(row["retreived_context"]) else [],
                retrieval_context=[row["actual_retreived_context"]],
            )
        )

    return dataset

def run_evaluation():
    df = pd.read_csv(EVAL_CSV)
    df = df.head(5)
    df = add_actual_outputs_to_dataset(df)
    dataset = build_eval_dataset(df)
    metrics = [RougeMetric(threshold=0.5), BERTScore(threshold=0.5)]
    results = evaluate(
        test_cases=dataset.test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=False, show_indicator=True),
    )
    _print_score_summary(results)
    _save_per_case_outputs(results, BASE_DIR / "eval_outputs.csv")


if __name__ == "__main__":
    df = pd.read_csv(EVAL_CSV)
    df = add_actual_outputs_to_dataset(df)

    dataset = build_eval_dataset(df)

    # Use only the non-LLM metrics from non_llm_scores.py
    metrics = [
        RougeMetric(threshold=0.5),
        BERTScore(threshold=0.5),
    ]

    results = evaluate(
        test_cases=dataset.test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=False, show_indicator=True),
    )
    _print_score_summary(results)
    _save_per_case_outputs(results, BASE_DIR / "eval_outputs.csv")

# if __name__ == "__main__":
#     no_actual_outputs_df = pd.read_csv('/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/eval_dataset.csv')
#     actual_outputs_df = add_actual_outputs_to_dataset(no_actual_outputs_df)

#     print(actual_outputs_df.head())


    # dataset = EvaluationDataset(test_cases=[
    #     LLMTestCase(
    #         input="Woah there",
    #         actual_output="Hi there",
    #         expected_output="Woah there",
    #         context=["Woah there hi there"],
    #         retrieval_context=["Woah there hi there"]
    #     )
    # ])
    # x = CustomGeminiModel()
    # faith = ContextualPrecisionMetric(threshold=0.5, model=x)

    # dataset.evaluate(
    #     metrics=[faith]
    # )