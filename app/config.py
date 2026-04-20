from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        app_name: str = "SectorSentinel API"
        environment: str = "development"
        api_v1_prefix: str = "/api/v1"
        model_version: str = "v1"
        sql_echo: bool = False
        cache_enabled: bool = True
        cors_origins: list[str] = Field(
            default_factory=lambda: [
                s.strip()
                for s in os.getenv(
                    "CORS_ORIGINS",
                    "http://localhost:5173,http://127.0.0.1:5173,http://frontend,http://frontend:80",
                ).split(",")
                if s.strip()
            ]
        )

        postgres_user: str = "sentinel"
        postgres_password: str = "sentinel123"
        postgres_db: str = "sectorsentinel"
        postgres_host: str = "postgres"
        postgres_port: int = 5432
        database_url: str | None = None
        redis_url: str = "redis://redis:6379"

        page_limit_default: int = 25
        page_limit_max: int = 100

        model_config = SettingsConfigDict(
            env_file=str(Path(__file__).resolve().parents[2] / ".env"),
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

        @property
        def backend_dir(self) -> Path:
            return Path(__file__).resolve().parents[1]

        @property
        def repo_dir(self) -> Path:
            return Path(__file__).resolve().parents[2]

        @property
        def data_dir(self) -> Path:
            return self.backend_dir / "data"

        @property
        def artifacts_dir(self) -> Path:
            return self.backend_dir / "artifacts"

        @property
        def models_dir(self) -> Path:
            return self.backend_dir / "models"

        @property
        def resolved_database_url(self) -> str:
            if self.database_url:
                url = self.database_url
                # Render provides postgres:// but asyncpg needs postgresql+asyncpg://
                if url.startswith("postgres://"):
                    url = url.replace("postgres://", "postgresql+asyncpg://", 1)
                elif url.startswith("postgresql://"):
                    url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
                return url
            return (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )

except ImportError:

    class Settings:
        def __init__(self) -> None:
            self.app_name = os.getenv("APP_NAME", "SectorSentinel API")
            self.environment = os.getenv("ENVIRONMENT", "development")
            self.api_v1_prefix = os.getenv("API_V1_PREFIX", "/api/v1")
            self.model_version = os.getenv("MODEL_VERSION", "v1")
            self.sql_echo = os.getenv("SQL_ECHO", "false").lower() == "true"
            self.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
            self.cors_origins = [
                s.strip()
                for s in os.getenv(
                    "CORS_ORIGINS",
                    "http://localhost:5173,http://127.0.0.1:5173,http://frontend,http://frontend:80",
                ).split(",")
                if s.strip()
            ]
            self.postgres_user = os.getenv("POSTGRES_USER", "sentinel")
            self.postgres_password = os.getenv("POSTGRES_PASSWORD", "sentinel123")
            self.postgres_db = os.getenv("POSTGRES_DB", "sectorsentinel")
            self.postgres_host = os.getenv("POSTGRES_HOST", "postgres")
            self.postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
            self.database_url = os.getenv("DATABASE_URL")
            self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
            self.page_limit_default = int(os.getenv("PAGE_LIMIT_DEFAULT", "25"))
            self.page_limit_max = int(os.getenv("PAGE_LIMIT_MAX", "100"))

        @property
        def backend_dir(self) -> Path:
            return Path(__file__).resolve().parents[1]

        @property
        def repo_dir(self) -> Path:
            return Path(__file__).resolve().parents[2]

        @property
        def data_dir(self) -> Path:
            return self.backend_dir / "data"

        @property
        def artifacts_dir(self) -> Path:
            return self.backend_dir / "artifacts"

        @property
        def models_dir(self) -> Path:
            return self.backend_dir / "models"

        @property
        def resolved_database_url(self) -> str:
            if self.database_url:
                url = self.database_url
                if url.startswith("postgres://"):
                    url = url.replace("postgres://", "postgresql+asyncpg://", 1)
                elif url.startswith("postgresql://"):
                    url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
                return url
            return (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
