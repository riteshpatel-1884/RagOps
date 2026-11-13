
from app.versions.toolkit.retrieval import build_pipeline

questions = [
    # Single-fact: q003
    "In which year was Asterion Digital Systems established?",

    # Single-fact: q001
    "What is the document ID of the employee handbook?",

    # Multi-hop: q083
    "Where is Tanvi Deshpande based and who is her reporting manager?",

    # Multi-hop: q084
    "Who handles privacy enquiries, and who does that person report to in the employee directory?",

    # Unanswerable: q086
    "What is the company's dress code?",
]

retriever = build_pipeline("v1")

for index, question in enumerate(questions, start=1):
    print("\n" + "=" * 100)
    print(f"QUESTION {index}: {question}")
    print("=" * 100)

    contexts, timings = retriever.retrieve(
        question,
        debug=True
    )

    print("\nFINAL RETRIEVED CONTEXTS:")
    for i, context in enumerate(contexts, start=1):
        print(f"\n--- Context {i} ---")
        print("Page:", context.get("page"))
        print("Source:", context.get("source"))
        print("Text:", context.get("text"))

    print("\nTIMINGS:", timings)
