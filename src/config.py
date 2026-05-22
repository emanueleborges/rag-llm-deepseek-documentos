"""
Configuration module for RAG Agent
"""
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Versão do índice vetorial (altere ao mudar o modelo de embeddings)
DEFAULT_INDEX_VERSION = "multilingual-minilm-v2"
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Deepseek API
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_temperature: float = 0.7
    deepseek_max_tokens: int = 2000

    # Vector Store
    vector_store_path: Path = PROJECT_ROOT / "data" / "vector_store"
    chroma_collection_name: str = "rag_documents"

    # Document Processing
    documents_path: Path = PROJECT_ROOT / "data" / "documents"
    max_chunk_size: int = 1000
    chunk_overlap: int = 100

    # RAG retrieval & embeddings
    retrieval_k: int = 20
    rerank_top_k: int = 8
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Streamlit
    streamlit_port: int = 8501

    # Logging
    log_level: str = "INFO"
    log_file: Path = PROJECT_ROOT / "logs" / "rag_agent.log"
    log_answer_max_chars: int = 4000

    # General
    environment: str = "development"
    debug: bool = True

    model_config = {
        "env_file": PROJECT_ROOT / ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @field_validator("vector_store_path", "documents_path", "log_file", mode="before")
    @classmethod
    def resolve_path(cls, value):
        path = Path(value) if not isinstance(value, Path) else value
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def vector_store_dir(self) -> Path:
        return Path(self.vector_store_path)

    @property
    def documents_dir(self) -> Path:
        return Path(self.documents_path)

    @property
    def logs_dir(self) -> Path:
        return Path(self.log_file).parent

    def create_directories(self) -> None:
        for directory in [self.vector_store_dir, self.documents_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
