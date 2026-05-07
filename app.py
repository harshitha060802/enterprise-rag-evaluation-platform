import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from generate_answer import build_context, generate_answer
from hybrid_search import (
    get_bm25_results,
    get_semantic_results,
    reciprocal_rank_fusion,
)
from rag_pipeline import rerank



def retrieve_and_answer(question):
    semantic_results = get_semantic_results(question, top_k=5)
    bm25_results = get_bm25_results(question, top_k=5)

    fused_results = reciprocal_rank_fusion(
        [semantic_results, bm25_results]
    )[:5]

    candidate_docs = [item["doc"] for item in fused_results]
    reranked_results = rerank(question, candidate_docs)[:3]

    context = build_context(reranked_results)
    answer = generate_answer(question, context)

    sources = []

    for doc, score in reranked_results:
        source_id = doc.metadata.get("chunk_id", doc.metadata.get("source", "unknown"))
        sources.append(
            {
                "source_id": source_id,
                "score": float(score),
                "text": doc.page_content,
            }
        )

    return answer, sources


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    st.set_page_config(
        page_title="Production RAG Chatbot",
        page_icon="",
        layout="wide",
    )

    st.title("Production RAG Chatbot")
    st.caption("Hybrid retrieval + reranking + cited OpenAI answers")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['source_id']}**")
                        st.caption(f"Reranker score: {source['score']:.4f}")
                        st.write(source["text"])

    question = st.chat_input("Ask a question about your documents")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating answer..."):
                answer, sources = retrieve_and_answer(question)

            st.markdown(answer)

            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"**{source['source_id']}**")
                    st.caption(f"Reranker score: {source['score']:.4f}")
                    st.write(source["text"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()
