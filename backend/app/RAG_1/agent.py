
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage

from .indexing import vector_store
from .objects import Answer

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

structured_llm = llm.with_structured_output(
    Answer,
    method="json_mode"
)

SYSTEM_PROMPT = """
You are a factual RAG assistant.

Answer the user's question using ONLY the retrieved context.

You must return TWO fields:

1. answer:
   - Provide a direct answer to the user's question.
   - Use only information supported by the retrieved context.
   - Do not invent facts.

2. context:
   - Return the relevant information from the retrieved context
     that you actually used to generate the answer.
   - Preserve the original facts, numbers, dates and names.
   - Do not introduce information from outside the retrieved context.

Additional rules:
- Never combine unrelated information.
- If the answer is unavailable, explicitly say so.
- Return both fields as strings.
- Return valid JSON with exactly these keys:
  "answer" and "context".
- Do not return Markdown outside JSON.
"""


def ask(question: str) -> dict:

    # 1. Retrieve relevant chunks
    docs = vector_store.similarity_search(
        question,
        k=4
    )

    # 2. Prepare context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # 3. Generate both answer and context
    response = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"""
            Retrieved Context:
            {context}

            User Question:
            {question}
            """
        )
    ])

    # 4. Return LLM-generated structured response
    return {
        "answer": response.answer,
        "context": response.context
    }


if __name__ == "__main__":

    result = ask(
        "What is Ritesh Patel's educational qualification?"
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nCONTEXT:")
    print(result["context"])
