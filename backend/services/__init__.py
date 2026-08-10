"""
backend.services — Business logic layer.

Service classes orchestrate calls to repositories, external APIs (OpenAI,
ChromaDB), and LangChain agents.  They are the primary home for domain rules
and should remain independent of the HTTP transport layer.
"""
