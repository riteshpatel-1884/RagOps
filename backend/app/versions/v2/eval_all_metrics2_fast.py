import json
from pathlib import Path
from dotenv import load_dotenv

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval
)
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM
from deepeval.evaluate import CacheConfig
from deepeval import evaluate

from langchain_groq import ChatGroq

load_dotenv()


# ==========================================
# 1. Custom Groq Evaluation Model
# ==========================================

class GroqEvalModel(DeepEvalBaseLLM):

    def __init__(self):
        self.model = ChatGroq(
            model="openai/gpt-oss-20b",
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
# 2. Load Results (all inside v2)
# ==========================================

base_dir = Path(__file__).resolve().parent

input_file = base_dir / "results2.jsonl"
output_file = base_dir / "evaluated_results2.jsonl"

with open(input_file, "r", encoding="utf-8") as file:
    results = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

print(f"Total test cases: {len(results)}")


# ==========================================
# 3. Prepare Test Cases
# ==========================================

questions_tests = []

for result in results:

    context = result["context"]

    if isinstance(context, str):
        retrieval_context = [context]
    else:
        retrieval_context = context

    test_case = LLMTestCase(
        input=result["question"],
        actual_output=result["actual_answer"],
        expected_output=result["expected_answer"],
        retrieval_context=retrieval_context
    )

    questions_tests.append(test_case)


# ==========================================
# 4. Initialize Evaluation Metrics
# ==========================================

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


# ==========================================
# 5. Run Batch Evaluation
# ==========================================

print("\nStarting RAG Evaluation...\n")

evaluation_results = evaluate(
    test_cases=questions_tests,
    metrics=[
        answer_relevancy,
        faithfulness,
        context_relevancy,
        correctness
    ],

    async_config=AsyncConfig(
        run_async=False
    ),

    cache_config=CacheConfig(
        use_cache=True,
        write_cache=True
    )
)


# ==========================================
# 6. Extract Evaluation Results
# ==========================================

evaluated_results = []

for i, (original_result, test_result) in enumerate(
    zip(results, evaluation_results.test_results),
    start=1
):

    print(f"\nTest Case {i}/{len(results)}")

    scores = {
        "answer_relevancy": None,
        "faithfulness": None,
        "context_relevancy": None,
        "correctness": None
    }

    for metric in test_result.metrics_data or []:

        metric_name = metric.name.strip().lower()

        if metric_name == "answer relevancy":
            scores["answer_relevancy"] = metric.score

        elif metric_name == "faithfulness":
            scores["faithfulness"] = metric.score

        elif metric_name == "contextual relevancy":
            scores["context_relevancy"] = metric.score

        elif metric_name.startswith("correctness"):
            scores["correctness"] = metric.score

    evaluated_result = {
        **original_result,
        **scores
    }

    evaluated_results.append(evaluated_result)

    for name, score in scores.items():
        print(f"{name}: {score}")


# ==========================================
# 7. Save Results
# ==========================================

with open(output_file, "w", encoding="utf-8") as file:

    for result in evaluated_results:
        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )


print("\n===================================")
print("RAG Evaluation Completed!")
print(f"Total Test Cases: {len(evaluated_results)}")
print(f"Metrics Evaluated: 4")
print(f"Output: {output_file}")
print("===================================")