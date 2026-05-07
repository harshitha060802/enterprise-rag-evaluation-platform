# Retrieval Augmented Generation Notes

Retrieval Augmented Generation, also called RAG, is a technique where an AI system retrieves relevant information from external documents before generating an answer.

A basic RAG pipeline has four main steps: document ingestion, chunking, embedding, and retrieval.

Document ingestion means loading raw files such as PDFs, Markdown files, web pages, or text documents into the system.

Chunking means splitting large documents into smaller pieces. This is important because language models and embedding models work better with smaller sections of text.

Embeddings are numerical representations of text. They help the system compare the meaning of a user question with the meaning of stored document chunks.

A vector database stores embeddings and allows semantic search. Semantic search means searching by meaning instead of only exact keywords.

ChromaDB is a lightweight vector database that can run locally on your computer.

Metadata is extra information stored with each chunk, such as the source filename, page number, or section title. Metadata is useful for citations.

A production-grade RAG system should answer only from retrieved context and should cite the source documents used to produce the answer.
