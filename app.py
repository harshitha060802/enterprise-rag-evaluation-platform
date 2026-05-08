import sys
import os
from hmac import compare_digest
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
from ingest import main as rebuild_indexes


DATA_DIR = PROJECT_ROOT / "data"
ALLOWED_UPLOAD_TYPES = ["pdf", "md"]


def configure_environment():
    load_dotenv(PROJECT_ROOT / ".env")

    for key in ["OPENAI_API_KEY", "APP_PASSWORD"]:
        try:
            secret_value = st.secrets.get(key)
        except FileNotFoundError:
            secret_value = None

        if secret_value:
            os.environ[key] = secret_value


def check_password():
    expected_password = os.getenv("APP_PASSWORD")

    if not expected_password:
        st.warning(
            "APP_PASSWORD is not set. The app is running without password protection."
        )
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title("Production RAG Chatbot")
    st.caption("Enter the app password to use the demo.")

    password = st.text_input("Password", type="password")

    if st.button("Unlock"):
        if compare_digest(password, expected_password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False


def save_uploaded_documents(uploaded_files):
    DATA_DIR.mkdir(exist_ok=True)
    saved_files = []

    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        destination = DATA_DIR / f"uploaded_{safe_name}"
        destination.write_bytes(uploaded_file.getbuffer())
        saved_files.append(destination.name)

    return saved_files


def render_sidebar():
    with st.sidebar:
        st.header("Document Upload")
        st.caption("Upload PDF or Markdown files, then rebuild the retrieval indexes.")

        uploaded_files = st.file_uploader(
            "Add documents",
            type=ALLOWED_UPLOAD_TYPES,
            accept_multiple_files=True,
        )

        if st.button("Save documents and rebuild indexes", disabled=not uploaded_files):
            with st.spinner("Saving documents and rebuilding indexes..."):
                saved_files = save_uploaded_documents(uploaded_files)
                rebuild_indexes()
                st.session_state.messages = []

            st.success(f"Indexed {len(saved_files)} uploaded file(s).")
            for filename in saved_files:
                st.write(f"- {filename}")


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
    configure_environment()

    st.set_page_config(
        page_title="Production RAG Chatbot",
        page_icon="",
        layout="wide",
    )

    if not check_password():
        return

    st.title("Production RAG Chatbot")
    st.caption("Hybrid retrieval + reranking + cited OpenAI answers")
    render_sidebar()

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
