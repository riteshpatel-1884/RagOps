"""
Generation layer: turns (question + retrieved context) into an answer.

Only a real LLM generator is offered now — no offline extractive fallback,
no OpenAI, no Anthropic. `get_generator("groq")` wraps LangChain's ChatGroq
behind the same BaseGenerator interface every other module in this project
already expects (pipeline.py, evaluate_generation.py, experiment.py, etc.),
so nothing downstream needs to change based on which generator is active.

Requires: pip install langchain-groq, and a GROQ_API_KEY in your environment
(e.g. via .env — see pipeline.py's load_dotenv() call).
"""
import os
from abc import ABC, abstractmethod
from typing import List
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise assistant that answers questions using ONLY the "
            "provided context. If the context doesn't contain the answer, say so. "
            "Do not add information that isn't in the context.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)


class BaseGenerator(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, question: str, context: List[str]) -> str:
        """Return a generated answer string given the question and retrieved context chunks."""


class LangChainLLMGenerator(BaseGenerator):
    """Wraps any LangChain chat model behind the BaseGenerator interface."""

    name = "langchain-llm"

    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.chain = RAG_PROMPT | self.chat_model

    def generate(self, question: str, context: List[str]) -> str:
        response = self.chain.invoke({"question": question, "context": "\n\n".join(context)})
        return response.content


def get_generator(name: str = "groq", **kwargs) -> BaseGenerator:
    """Factory so pipeline configs can select a generator by name."""
    if name != "groq":
        raise ValueError(f"Unknown generator: '{name}'. Only 'groq' is supported.")

    from langchain_groq import ChatGroq  # requires GROQ_API_KEY + network
    kwargs.setdefault("model", os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
    return LangChainLLMGenerator(ChatGroq(**kwargs))
