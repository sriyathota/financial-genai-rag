from deepeval.scorer import Scorer
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

import bert_score.utils as _bert_utils

def _safe_sent_encode(tokenizer, sent):
    max_len = getattr(tokenizer, "model_max_length", 512) or 512
    if max_len > 1_000_000:
        max_len = 512
    return tokenizer.encode(sent, add_special_tokens=True, max_length=max_len, truncation=True)

_bert_utils.sent_encode = _safe_sent_encode

class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()
        self.score = 0.0
        self.success = False

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rouge1"
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"



class BERTScore(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()
        self.score = 0.0
        self.success = False

    def measure(self, test_case: LLMTestCase):
        result = self.scorer.bert_score(
            references=test_case.expected_output,
            predictions=test_case.actual_output,
        )
        val = result
        if isinstance(val, dict):
            val = val.get("bert-f1") or val.get("f1") or 0.0
        while isinstance(val, (list, tuple)) and len(val) > 0:
            val = val[-1]
        if hasattr(val, "item"):
            val = val.item()
        self.score = float(val)
        self.success = self.score >= self.threshold
        return self.score
    
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "BERT Score"
    


