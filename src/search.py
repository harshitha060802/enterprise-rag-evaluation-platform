from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


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

    print("\nTop retrieved chunks:\n")

    for index, (doc, score) in enumerate(results, start=1):
        print(f"--- Result {index} ---")
        source_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))
        print(f"Source: {source_id}")
        print(f"Distance: {score:.4f} (lower is better)")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()
