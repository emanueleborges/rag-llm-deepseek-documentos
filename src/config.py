"""
Configuration module for RAG Agent
"""
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Versão do índice (altere ao mudar embeddings, metadados ou índice BM25)
DEFAULT_INDEX_VERSION = "multilingual-minilm-v2-hybrid-v1"
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Deepseek API (opcional se LLM_PROVIDER=ollama)
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_temperature: float = 0.7
    deepseek_max_tokens: int = 2000

    # Ollama — LLM local gratuito (embeddings: use fastembed no Docker)
    llm_provider: str = "ollama"  # ollama | deepseek
    embeddings_provider: str = "fastembed"  # fastembed | ollama | local
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "ibm/granite4:micro"
    ollama_embedding_model: str = "nomic-embed-text"  # só se EMBEDDINGS_PROVIDER=ollama
    ollama_temperature: float = 0.0
    ollama_num_predict: int = 768  # limite de tokens gerados (menor = mais rápido)
    ollama_num_ctx: int = 4096  # janela de contexto enviada ao modelo
    fastembed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Vector Store
    vector_store_path: Path = PROJECT_ROOT / "data" / "vector_store"
    chroma_collection_name: str = "rag_documents"

    # Document Processing
    documents_path: Path = PROJECT_ROOT / "data" / "documents"
    max_chunk_size: int = 1000
    chunk_overlap: int = 100
    pdf_parser: str = "pypdf"  # pypdf | pymupdf | llamaparse | auto
    llama_cloud_api_key: str = ""

    # Vector Store (Chroma — padrão; Qdrant preparado para Fase 2+)
    vector_store_backend: str = "chroma"  # chroma | qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "rag_documents"

    # RAG retrieval & embeddings
    retrieval_k: int = 10
    rerank_top_k: int = 5
    max_context_chars: int = 7000  # teto de texto enviado ao LLM
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    # Fase 2 — busca híbrida + reranker
    hybrid_search_enabled: bool = True
    hybrid_fetch_k: int = 15
    hybrid_rrf_k: int = 60
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_min_score: float = -4.0
    context_grader_enabled: bool = False
    context_grader_min_score: float = -4.0
    retrieval_metadata_filter: str = ""  # JSON opcional, ex: {"content_type":"text"}

    # Fase 3 — LangGraph (agentic RAG)
    langgraph_enabled: bool = False
    max_retrieval_retries: int = 1
    tavily_api_key: str = ""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Streamlit
    streamlit_port: int = 8501

    # Fase 4 — Chainlit + feedback
    chainlit_port: int = 8502
    feedback_dir: Path = PROJECT_ROOT / "data" / "feedback"

    # Fase 5 — Observabilidade (Langfuse)
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

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

    @field_validator("vector_store_path", "documents_path", "log_file", "feedback_dir", mode="before")
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
        for directory in [
            self.vector_store_dir,
            self.documents_dir,
            self.logs_dir,
            Path(self.feedback_dir),
        ]:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()


def current_index_version() -> str:
    """Versão do índice — muda ao trocar provedor/modelo de embeddings."""
    provider = (settings.embeddings_provider or "").lower()
    if provider == "ollama":
        model = settings.ollama_embedding_model.replace(":", "-").replace("/", "-")
        return f"ollama-{model}-hybrid-v1"
    if provider == "fastembed":
        model = settings.fastembed_model.replace("/", "-")
        return f"fastembed-{model}-hybrid-v1"
    return DEFAULT_INDEX_VERSION
