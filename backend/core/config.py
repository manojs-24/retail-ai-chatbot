from __future__ import annotations

from functools import lru_cache

__all__ = [
    "Settings",
    "get_settings",
]

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Application
    APP_NAME: str = Field(default="Retail AI System", description="Human-readable application name")
    APP_VERSION: str = Field(default="1.0.0", description="Semantic version string")
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ENVIRONMENT: str = Field(default="development", description="Runtime environment (development / staging / production)")

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./database/retail.db",
        description="SQLAlchemy-compatible database connection URL",
    )

    # OpenAI
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key — never commit to VCS")

    # ChromaDB
    CHROMA_DB_PATH: str = Field(default="./chroma_db", description="Filesystem path for the ChromaDB persistent store")

    # Security
    SECRET_KEY: str = Field(default="", description="HMAC secret used to sign JWT tokens — must be strong in production")
    ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="JWT access-token TTL in minutes")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Root log level (DEBUG / INFO / WARNING / ERROR / CRITICAL)")
    LOG_FILE_PATH: str = Field(default="./logs/retail_ai.log", description="Path to the rotating log file")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
