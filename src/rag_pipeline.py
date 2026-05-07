import json
from datetime import datetime
from pathlib import Path

from sentence_transformers import CrossEncoder

from hybrid_search import get_bm25_results, get_semantic_results, reciprocal_rank_fusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"


def rerank(question, docs):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(question, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    return sorted(
        zip(docs, scores),
        key=lambda item: item[1],
        reverse=True,
    )


def save_trace(question, reranked_results):
    LOGS_DIR.mkdir(exist_ok=True)

    trace = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "retrieval_pipeline": "semantic + bm25 + rrf + cross_encoder_rerank",
        "results": [],
    }

    for rank, (doc, score) in enumerate(reranked_results, start=1):
        source_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))

        trace["results"].append(
            {
                "rank": rank,
                "source_id": source_id,
                "reranker_score": float(score),
                "text": doc.page_content,
            }
        )

    log_path = LOGS_DIR / "rag_pipeline_traces.jsonl"
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(trace) + "\n")


def main():
    question = input("Ask a question: ")

    semantic_results = get_semantic_results(question, top_k=8)
    bm25_results = get_bm25_results(question, top_k=8)

    fused_results = reciprocal_rank_fusion(
        [semantic_results, bm25_results]
    )[:8]

    candidate_docs = [item["doc"] for item in fused_results]
    reranked_results = rerank(question, candidate_docs)[:5]

    save_trace(question, reranked_results)

    print("\nFinal retrieved evidence:\n")

    for index, (doc, score) in enumerate(reranked_results, start=1):
        source_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))

        print(f"[{index}] Source: {source_id}")
        print(f"Reranker Score: {score:.4f} (higher is better)")
        print(doc.page_content)
        print()

    print("Answer policy:")
    print("Answer only using the evidence above. Cite the source ID for every claim.")
    print("Trace saved to logs/rag_pipeline_traces.jsonl")


if __name__ == "__main__":
    main()
