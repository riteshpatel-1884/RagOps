# It will evaluate all 20 test data slowly. To run all tests in parallel use eval_all_metrics1.fast.py but it will show rate limit reached as it evaluate all test cases paralley so it will cross groq's per min limit of 8000.
import json
from pathlib import Path
from dotenv import load_dotenv

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval
)

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq

load_dotenv()


# ==========================================
# 1. Custom Groq Evaluation Model
# ==========================================


class GroqEvalModel(DeepEvalBaseLLM):

    def __init__(self):
        self.model = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):

        if schema:
            structured_model = self.model.with_structured_output(
                schema,
                method="json_mode"
            )
            return structured_model.invoke(prompt)

        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str, schema=None):

        if schema:
            structured_model = self.model.with_structured_output(
                schema,
                method="json_mode"
            )
            return await structured_model.ainvoke(prompt)

        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Groq GPT-OSS-120B"



# Initialize judge model
eval_model = GroqEvalModel()


# ==========================================
# 2. Dataset Paths
# ==========================================

base_dir = Path(__file__).parent

input_file = base_dir / "versions" / "v1" / "results1.jsonl"

output_file = base_dir / "versions" / "v1" / "evaluated_results1.jsonl"


# ==========================================
# 3. Load Dataset
# ==========================================

with open(input_file, "r", encoding="utf-8") as file:
    results = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

evaluated_results = []


# ==========================================
# 4. Evaluate Every Test Case
# ==========================================

for i, result in enumerate(results, start=1):

    print(f"\nEvaluating test {i}/{len(results)}")

    test_case = LLMTestCase(
        input=result["question"],
        actual_output=result["actual_answer"],
        expected_output=result["expected_answer"],
        retrieval_context=[result["context"]]
    )

    # Initialize all metrics with Groq

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model=eval_model
    )

    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model=eval_model
    )

    context_relevancy = ContextualRelevancyMetric(
        threshold=0.7,
        model=eval_model
    )

    correctness = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output is factually correct "
            "compared with the expected output. Evaluate factual accuracy, "
            "missing information, and incorrect claims."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=0.7,
        model=eval_model
    )

    # Run evaluations

    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)
    context_relevancy.measure(test_case)
    correctness.measure(test_case)

    # Store results

    evaluated_result = {
        **result,

        "answer_relevancy": answer_relevancy.score,
        "faithfulness": faithfulness.score,
        "context_relevancy": context_relevancy.score,
        "correctness": correctness.score,

        "answer_relevancy_passed": answer_relevancy.is_successful(),
        "faithfulness_passed": faithfulness.is_successful(),
        "context_relevancy_passed": context_relevancy.is_successful(),
        "correctness_passed": correctness.is_successful()
    }

    evaluated_results.append(evaluated_result)

    # Print scores

    print("Answer Relevancy:", answer_relevancy.score)
    print("Faithfulness:", faithfulness.score)
    print("Context Relevancy:", context_relevancy.score)
    print("Correctness:", correctness.score)


# ==========================================
# 5. Save Evaluated Results
# ==========================================

with open(output_file, "w", encoding="utf-8") as file:

    for result in evaluated_results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


print("\n================================")
print("Evaluation Completed!")
print(f"Total Test Cases: {len(evaluated_results)}")
print(f"Results Saved: {output_file}")
print("================================")
