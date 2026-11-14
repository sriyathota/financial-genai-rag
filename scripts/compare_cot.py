import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


# scripts/compare_cot.py
from LocalRAG import create_whole_pipeline, USE_HIDDEN_COT
from eval import run_evaluation  # or whatever entrypoint eval.py has

def run(use_cot: bool):
    # if eval.py uses create_whole_pipeline internally, pass through configuration
    USE_HIDDEN_COT = use_cot
    print("=== CoT:", use_cot, "===")
    run_evaluation()

if __name__ == "__main__":
    run(False)
    run(True)