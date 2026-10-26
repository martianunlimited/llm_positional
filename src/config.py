"""Configuration settings for LLM Positional Encodings Lab."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for Positional Encodings demo."""

    app_name: str = "LLM Positional Encodings Lab"
    app_version: str = "0.1.0"
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    presets_file: Path = Path(__file__).resolve().parent.parent / "data" / "presets.json"

    default_embed_dim: int = 16
    default_max_pos: int = 64
    default_rope_base: float = 10000.0
    llama3_rope_base: float = 500000.0

    model_config = SettingsConfigDict(
        env_prefix="POSITIONAL_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
