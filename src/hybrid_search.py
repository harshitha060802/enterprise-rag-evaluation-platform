import pickle
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
BM25_INDEX_PATH = PROJECT_ROOT / "bm25_index.pkl"


def tokenize(text):
    return text.lower().split()


def reciprocal_rank_fusion(result_lists, k=60):
    fused_scores = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {
                    "doc": doc,
                    "score": 0.0,
                }

            fused_scores[chunk_id]["score"] += 1 / (k + rank)

    return sorted(
        fused_scores.values(),
        key=lambda item: item["score"],
        reverse=True,
    )


def get_semantic_results(question, top_k=5):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    return vectorstore.similarity_search(question, k=top_k)


def get_bm25_results(question, top_k=5):
    with BM25_INDEX_PATH.open("rb") as file:
        index_data = pickle.load(file)

    bm25 = index_data["bm25"]
    documents = index_data["documents"]

    scores = bm25.get_scores(tokenize(question))

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [doc for doc, score in ranked_results[:top_k]]


def main():
    question = input("Ask a question: ")

    semantic_results = get_semantic_results(question, top_k=5)
    bm25_results = get_bm25_results(question, top_k=5)

    fused_results = reciprocal_rank_fusion(
        [semantic_results, bm25_results]
    )[:5]

    print("\nTop hybrid results:\n")

    for index, item in enumerate(fused_results, start=1):
        doc = item["doc"]
        source_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))

        print(f"--- Result {index} ---")
        print(f"Source: {source_id}")
        print(f"RRF Score: {item['score']:.4f} (higher is better)")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()
