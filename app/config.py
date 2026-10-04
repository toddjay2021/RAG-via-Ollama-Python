"""Application settings loaded from environment variables (.env)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (parent of the app/ package)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for the RAG demo."""

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:1b"
    embedding_model: str = "nomic-embed-text"

    # Chunking / retrieval
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4

    # Limits
    max_upload_size_mb: int = 20
    max_history_messages: int = 4

    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    def ensure_dirs(self) -> None:
        """Create runtime directories if they do not exist."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
