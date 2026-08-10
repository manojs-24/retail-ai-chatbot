"""
backend.schemas — Pydantic request / response schemas.

Schemas are the public contract between the API layer and its callers.
They handle input validation, output serialization, and OpenAPI doc generation.
Keep schemas decoupled from ORM models — use ``model_validate`` / ``from_orm``
adapters in the service layer when converting between the two.
"""
