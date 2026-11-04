import os
from dotenv import load_dotenv

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

from langchain_groq import ChatGroq

load_dotenv()


class GroqEvalModel(DeepEvalBaseLLM):

    def __init__(self):
        self.model = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Groq GPT-OSS-120B"


# Initialize evaluation model
eval_model = GroqEvalModel()


test_case = LLMTestCase(
    input="What data processing techniques did Ritesh use during his Data Analyst internship?",
    actual_output=(
        "Ritesh used car to build data preprocessing"
    )
)


metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=eval_model
)

metric.measure(test_case)

print("Score:", metric.score)
print("Reason:", metric.reason)
print("Passed:", metric.is_successful())