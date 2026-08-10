"""
backend — Retail AI FastAPI application package.

Sub-packages:
- api          : Route handlers / routers grouped by domain.
- core         : Cross-cutting concerns (config, database, logging, security).
- models       : SQLAlchemy ORM model definitions.
- schemas      : Pydantic request/response schemas.
- services     : Business logic layer.
- repositories : Data-access layer (repository pattern).
- agents       : LangChain / LangGraph agent definitions.
- rag          : Retrieval-Augmented Generation helpers (ChromaDB).
- utils        : Shared utility functions.
- tests        : Pytest test suite.
"""
