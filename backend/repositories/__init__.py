"""
backend.repositories — Data-access layer (repository pattern).

Each repository wraps a specific ORM model and exposes typed CRUD methods.
Repositories accept a ``Session`` injected via FastAPI's ``Depends(get_db)``
and are the only layer allowed to issue SQL queries directly.
"""
