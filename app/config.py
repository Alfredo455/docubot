from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCUBOT_", env_file=".env", extra="ignore")

    docs_dir: str = "docs_sample"
    chunk_size: int = 120  # words per chunk
    chunk_overlap: int = 20
    top_k: int = 3
    anthropic_api_key: str | None = None
    model: str = "claude-haiku-4-5-20251001"


settings = Settings()
