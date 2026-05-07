import json
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
LOGS_DIR = PROJECT_ROOT / "logs"


def build_context(results):
    context_blocks = []

    for index, (doc, distance) in enumerate(results, start=1):
        source_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))

        context_blocks.append(
            {
                "source_id": source_id,
                "distance": distance,
                "text": doc.page_content,
            }
        )

    return context_blocks


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    question = input("Ask a question: ")

    results = vectorstore.similarity_search_with_score(question, k=3)
    context_blocks = build_context(results)
    LOGS_DIR.mkdir(exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "retrieved_context": context_blocks,
    }

    log_path = LOGS_DIR / "retrieval_logs.jsonl"
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(log_entry) + "\n")


    print("\nRetrieved evidence:\n")

    if not context_blocks:
        print("I don't know. No relevant context was retrieved.")
        return

    for index, block in enumerate(context_blocks, start=1):
        print(f"[{index}] Source: {block['source_id']}")
        print(f"Distance: {block['distance']:.4f} (lower is better)")
        print(block["text"])
        print()

    print("Instruction for final answer:")
    print("Answer only using the evidence above. Cite the source ID for every claim.")

if __name__ == "__main__":
    main()
