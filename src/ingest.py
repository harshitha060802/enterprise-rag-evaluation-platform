import pickle
import shutil
from pathlib import Path

from rank_bm25 import BM25Okapi

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
BM25_INDEX_PATH = PROJECT_ROOT / "bm25_index.pkl"


def load_documents():
    documents = []

    for file_path in DATA_DIR.glob("*.md"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source"] = file_path.name
            doc.metadata["file_type"] = "markdown"

        documents.extend(loaded_docs)

    for file_path in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(file_path))
        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source"] = file_path.name
            doc.metadata["file_type"] = "pdf"

        documents.extend(loaded_docs)

    return documents
def tokenize(text):
    return text.lower().split()


def save_bm25_index(chunks):
    tokenized_chunks = [tokenize(chunk.page_content) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    index_data = {
        "bm25": bm25,
        "documents": chunks,
    }

    with BM25_INDEX_PATH.open("wb") as file:
        pickle.dump(index_data, file)

def main():
    if CHROMA_DIR.exists():
        print("Removing old ChromaDB database...")
        shutil.rmtree(CHROMA_DIR)

    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s).")

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunk(s).")

    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")

        if page is not None:
            chunk.metadata["chunk_id"] = f"{source}::page-{page + 1}::chunk-{index}"
        else:
            chunk.metadata["chunk_id"] = f"{source}::chunk-{index}"
    print("Saving BM25 keyword index...")
    save_bm25_index(chunks)

    print("Creating local embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Saving chunks to ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    print("Done. Vector database saved to chroma_db.")


if __name__ == "__main__":
    main()
