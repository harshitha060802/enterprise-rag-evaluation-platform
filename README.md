# Production RAG Project

This project is a local Retrieval Augmented Generation (RAG) system built in phases.

The goal is to start with a simple document retrieval pipeline and gradually upgrade it into a production-style RAG system with hybrid retrieval, reranking, evaluation, and CI/CD quality gates.

## Current Status

Phase 1, Phase 2, and the foundation of Phase 3 are implemented.

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
- Generate final cited answers using the OpenAI API
- Create a small golden retrieval dataset
- Run a retrieval evaluation with a minimum passing score
- Generate a retrieval evaluation report and chart
- Add a GitHub Actions workflow for retrieval evaluation

## Project Structure

```text
production-rag/
  data/                # Source documents
  src/
    ingest.py          # Loads, chunks, embeds, and stores documents
    search.py          # Semantic vector search
    bm25_search.py     # BM25 keyword search
    hybrid_search.py   # Semantic + BM25 retrieval with RRF
    rerank_search.py   # Hybrid retrieval + cross-encoder reranking
    rag_pipeline.py    # Main pipeline: semantic + BM25 + RRF + reranking + trace logs
    generate_answer.py # OpenAI final answer generation with citations
    ask.py             # Retrieves evidence and logs results
  evals/               # Evaluation dataset, report, chart, and scripts
  logs/                # Local retrieval and answer logs
  chroma_db/           # Local Chroma vector database
  requirements.txt
  README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is ignored by Git and should not be committed.

## Ingestion

Add documents to the `data/` folder.

Supported formats:

- `.md`
- `.pdf`

Run ingestion:

```powershell
python src/ingest.py
```

This creates:

- a local ChromaDB vector database
- a BM25 keyword index

## Retrieval Methods

The project supports several retrieval modes.

### Semantic Search

Semantic search uses embeddings and ChromaDB to find chunks that are similar in meaning to the user question.

Run:

```powershell
python src/search.py
```

### BM25 Keyword Search

BM25 keyword search finds chunks using exact keyword overlap. This is useful for IDs, names, product codes, acronyms, and exact terms.

Run:

```powershell
python src/bm25_search.py
```

### Hybrid Search With RRF

Hybrid search combines semantic search and BM25 search using Reciprocal Rank Fusion.

Flow:

```text
semantic results + BM25 results -> RRF merge -> hybrid ranked chunks
```

Run:

```powershell
python src/hybrid_search.py
```

### Cross-Encoder Reranking

The reranker takes the top hybrid candidates and scores each query/chunk pair more carefully.

Flow:

```text
semantic + BM25 -> RRF top candidates -> cross-encoder reranker -> final ranked chunks
```

Run:

```powershell
python src/rerank_search.py
```

### Main RAG Pipeline

This is the main retrieval pipeline. It runs semantic search, BM25 search, RRF fusion, cross-encoder reranking, and saves a trace log.

Run:

```powershell
python src/rag_pipeline.py
```

Trace logs are saved to:

```text
logs/rag_pipeline_traces.jsonl
```

## Final Answer Generation

This script runs the full retrieval pipeline and then sends the retrieved context to the OpenAI API to generate a concise cited answer.

Run:

```powershell
python src/generate_answer.py
```

The script uses the API key from `.env`:

```text
OPENAI_API_KEY=your_api_key_here
```

For cost control, the script uses a small model, retrieves a limited number of chunks, and caps the answer length.

Answer traces are saved to:

```text
logs/answer_traces.jsonl
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

It also writes a Markdown report:

```text
evals/retrieval_report.md
```

A simple chart can be generated with:

```powershell
python evals/plot_eval_report.py
```

The chart is saved to:

```text
evals/retrieval_score.png
```

## CI/CD

The repository includes a GitHub Actions workflow:

```text
.github/workflows/retrieval-eval.yml
```

The workflow installs dependencies, rebuilds indexes, and runs the retrieval evaluation.

If the retrieval score drops below the threshold, the workflow fails.

## Citations

Each retrieved chunk includes a source ID such as:

```text
example.pdf::page-2::chunk-5
```

The final answer prompt requires citations using the retrieved source IDs.

## Cost Controls

This project minimizes API usage by:

- using a small OpenAI model
- retrieving a limited number of chunks
- reranking before answer generation
- sending only the top retrieved context to the LLM
- capping output length

Recommended account safety settings:

- keep auto recharge off
- use a small initial credit amount
- set project budget alerts
- do not enable data sharing for private documents

## Next Steps

- Add stricter citation validation
- Create a larger golden Q&A dataset
- Add Ragas faithfulness and answer relevancy evaluation
- Add a Streamlit chatbot interface
- Add deployment instructions
