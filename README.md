# Production RAG Project

This project is a local Retrieval Augmented Generation (RAG) system built in phases.

The goal is to start with a simple document retrieval pipeline and gradually upgrade it into a production-style RAG system with hybrid retrieval, reranking, evaluation, and CI/CD quality gates.

## Current Status

Phase 1 and the core of Phase 2 are implemented.

Implemented so far:

- Load Markdown documents from `data/`
- Load PDF documents from `data/`
- Split documents into overlapping chunks
- Create local embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Store embeddings in ChromaDB
- Search documents using semantic similarity
- Attach source metadata and page-level citations
- Save retrieval logs to `logs/retrieval_logs.jsonl`
- Build a BM25 keyword index
- Search with BM25 keyword retrieval
- Combine semantic and BM25 results with Reciprocal Rank Fusion
- Rerank hybrid candidates with a cross-encoder
- Save main pipeline traces to `logs/rag_pipeline_traces.jsonl`
- Create a small golden retrieval dataset
- Run a retrieval evaluation with a minimum passing score

## Project Structure

```text
production-rag/
  data/              # Source documents
  src/
    ingest.py        # Loads, chunks, embeds, and stores documents
    search.py        # Semantic vector search
    bm25_search.py   # BM25 keyword search
    hybrid_search.py # Semantic + BM25 retrieval with RRF
    rerank_search.py # Hybrid retrieval + cross-encoder reranking
    rag_pipeline.py  # Main pipeline: semantic + BM25 + RRF + reranking + trace logs
    ask.py           # Retrieves evidence and logs results
  evals/             # Future evaluation datasets and scripts
  logs/              # Retrieval logs
  chroma_db/         # Local Chroma vector database
  requirements.txt
  README.md
  ```

## Evaluation

The project includes a small golden dataset for retrieval evaluation:

```text
evals/golden_dataset.csv
```

The current evaluation checks whether the hybrid + reranked retrieval pipeline returns the expected source document for each question.

Run:

```powershell
python evals/eval_retrieval.py
```

The script prints pass/fail results and exits with an error if the retrieval score drops below `0.80`.

## Next Steps

Phase 2:

- Add a final answer-generation script
- Enforce citation-only answers
- Add cleaner config management

Phase 3:

- Create a larger golden Q&A dataset
- Evaluate faithfulness and answer relevancy with Ragas
- Add CI/CD quality gates

