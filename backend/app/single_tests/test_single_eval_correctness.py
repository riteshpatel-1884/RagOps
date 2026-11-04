
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

    def generate(self, prompt: str, schema=None):
        response = self.model.invoke(prompt)
        output = response.content

        if schema:
            import json
            return schema(**json.loads(output))

        return output

    async def a_generate(self, prompt: str, schema=None):
        response = await self.model.ainvoke(prompt)
        output = response.content

        if schema:
            import json
            return schema(**json.loads(output))

        return output

    def get_model_name(self):
        return "Groq GPT-OSS-120B"


# Initialize evaluation model
eval_model = GroqEvalModel()


# Define test case
test_case = LLMTestCase(
    input="What is Ritesh Patel's educational qualification, university, and CGPA?",
    actual_output=(
        "Ritesh Patel has a B.Tech in Computer Science from KIET Deemed to be University, Ghaziabad, India, with a CGPA of 7.48."
    ),
    expected_output=[
        "Ritesh Patel is pursuing a B.Tech in Computer Science from KIET Deemed to be University, Ghaziabad. His CGPA is 7.48."
    ]
)


# Initialize metric
metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=eval_model
)


# Run evaluation
metric.measure(test_case)

print("Score:", metric.score)
print("Reason:", metric.reason)
print("Passed:", metric.is_successful())
