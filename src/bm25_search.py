import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BM25_INDEX_PATH = PROJECT_ROOT / "bm25_index.pkl"


def tokenize(text):
    return text.lower().split()


def main():
    if not BM25_INDEX_PATH.exists():
        print("BM25 index not found.")
        print("Run python src/ingest.py first.")
        return

    with BM25_INDEX_PATH.open("rb") as file:
        index_data = pickle.load(file)

    bm25 = index_data["bm25"]
    documents = index_data["documents"]

    question = input("Ask a keyword question: ")
    tokenized_question = tokenize(question)

    scores = bm25.get_scores(tokenized_question)

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    print("\nTop BM25 results:\n")

    for index, (doc, score) in enumerate(ranked_results, start=1):
        source_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))
        print(f"--- Result {index} ---")
        print(f"Source: {source_id}")
        print(f"BM25 Score: {score:.4f} (higher is better)")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()
