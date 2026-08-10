"""
backend.rag — Retrieval-Augmented Generation helpers.

Modules:
- chroma      : ChromaDB client factory, collection management, and startup init.
- config      : Pipeline constants (paths, chunk settings, model names).
- loader      : PDF document loader (LangChain community PyPDFLoader).
- chunking    : RecursiveCharacterTextSplitter with metadata enrichment.
- embeddings  : OpenAI embeddings factory.
- vectorstore : ChromaDB vector store build / load helpers (langchain-chroma).
- ingest      : End-to-end ingestion pipeline entry point.
- retriever   : LangChain retriever factory (get_retriever).
"""
