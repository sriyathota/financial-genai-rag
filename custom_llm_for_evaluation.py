from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric,  AnswerRelevancyMetric,  FaithfulnessMetric
from deepeval import evaluate
import os

class CustomGeminiModel(DeepEvalBaseLLM):
    def __init__(self, model=None):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key = os.environ["GOOGLE_API_KEY"],
            temperatrue=0.3,
            max_output_tokens=256
        ) if not model else model

    def load_model(self):
        return self.model
        
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content
        
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
        
    def get_model_name(self):
        return "Google Gemini Model"

def calculateMetrics(testcases):  
    # test cases written using LLMTestCase(input = "", actual_output = "", retrieval_context = [""])
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.7,
        top_p=0.85,
        max_output_tokens=2048
    )

    google_gemini = CustomGeminiModel(model=gemini_model)

    contextual_precision = ContextualPrecisionMetric(model = google_gemini)
    contextual_recall = ContextualRecallMetric(model = google_gemini)
    contextual_relevancy = ContextualRelevancyMetric(model = google_gemini)
    answer_relevancy = AnswerRelevancyMetric(model = google_gemini)
    faithfulness = FaithfulnessMetric(model = google_gemini)

    evaluate(
        test_cases=[testcases],
        metrics=[contextual_precision, contextual_recall, contextual_relevancy, answer_relevancy, faithfulness]
    )