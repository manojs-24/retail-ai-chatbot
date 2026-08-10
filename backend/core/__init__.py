"""
backend.core — Cross-cutting infrastructure concerns.

Modules:
- config   : Application settings via pydantic-settings.
- database : SQLAlchemy engine, session factory, and FastAPI dependency.
- logging  : Structured rotating-file + console logging setup.
- security : JWT and password-hashing utilities (implementation pending).
"""
