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

Fallback behaviour
------------------
Both critical pipeline stages degrade gracefully instead of raising:

Chunking (chunking.split_documents)
    Primary  : RecursiveCharacterTextSplitter (LangChain).
    Fallback : _manual_chunk_documents — pure-Python character-window slicer,
               no third-party dependency.  Triggered when the primary splitter
               raises for any reason (e.g. None page_content, bad separator).
               Fallback chunks carry ``metadata["fallback_chunking"] = True``.

Retrieval (retriever.get_retriever)
    Primary  : OpenAI embeddings + ChromaDB vector-similarity search.
    Fallback : BM25Retriever (rank_bm25) built directly from the on-disk PDFs,
               entirely in-process with no network calls.  Triggered when the
               primary path raises (missing API key, network error, corrupt DB).
               Fallback chunks carry ``metadata["bm25_fallback"] = True``.

Both fallbacks are composable: if the splitter AND the vector store both fail,
manual chunks feed the BM25 index automatically.
"""
