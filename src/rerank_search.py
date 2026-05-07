from sentence_transformers import CrossEncoder

from hybrid_search import get_bm25_results, get_semantic_results, reciprocal_rank_fusion


def rerank(question, docs):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(question, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    reranked = sorted(
        zip(docs, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return reranked


def main():
    question = input("Ask a question: ")

    semantic_results = get_semantic_results(question, top_k=8)
    bm25_results = get_bm25_results(question, top_k=8)

    fused_results = reciprocal_rank_fusion(
        [semantic_results, bm25_results]
    )[:8]

    candidate_docs = [item["doc"] for item in fused_results]
    reranked_results = rerank(question, candidate_docs)[:5]

    print("\nTop reranked results:\n")

    for index, (doc, score) in enumerate(reranked_results, start=1):
        source_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))

        print(f"--- Result {index} ---")
        print(f"Source: {source_id}")
        print(f"Reranker Score: {score:.4f} (higher is better)")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()
