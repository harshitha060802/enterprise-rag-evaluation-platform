import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag_pipeline import rerank
from hybrid_search import get_bm25_results, get_semantic_results, reciprocal_rank_fusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"


SYSTEM_PROMPT = """
You are a retrieval-augmented assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not supported by the context, say "I don't know based on the provided context."
- Cite the source ID immediately after each sentence that uses context.
- Use this citation format: [source_id].
- Do not invent sources.
- Keep the answer concise.
""".strip()


def build_context(reranked_results):
    context_parts = []

    for index, (doc, score) in enumerate(reranked_results, start=1):
        source_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))

        context_parts.append(
            f"[Source ID: {source_id}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def save_answer_trace(question, answer, reranked_results):
    LOGS_DIR.mkdir(exist_ok=True)

    trace = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "model": "gpt-4.1-nano",
        "pipeline": "semantic + bm25 + rrf + cross_encoder_rerank + openai_answer",
        "sources": [],
    }

    for rank, (doc, score) in enumerate(reranked_results, start=1):
        source_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))

        trace["sources"].append(
            {
                "rank": rank,
                "source_id": source_id,
                "reranker_score": float(score),
                "text": doc.page_content,
            }
        )

    log_path = LOGS_DIR / "answer_traces.jsonl"
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(trace) + "\n")


def generate_answer(question, context):
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}",
            },
        ],
        max_output_tokens=180,
    )

    return response.output_text


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    question = input("Ask a question: ")

    semantic_results = get_semantic_results(question, top_k=5)
    bm25_results = get_bm25_results(question, top_k=5)


    fused_results = reciprocal_rank_fusion(
        [semantic_results, bm25_results]
    )[:5]

    candidate_docs = [item["doc"] for item in fused_results]
    reranked_results = rerank(question, candidate_docs)[:3]

    context = build_context(reranked_results)
    answer = generate_answer(question, context)

    save_answer_trace(question, answer, reranked_results)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources used:")
    for doc, score in reranked_results:
        source_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))
        print(f"- {source_id} | reranker_score={score:.4f}")

    print("\nTrace saved to logs/answer_traces.jsonl")


if __name__ == "__main__":
    main()
