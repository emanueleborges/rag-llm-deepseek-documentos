"""Factory para LLM — Ollama (local/grátis) ou Deepseek (API)."""
from typing import Any, Iterator, Protocol

from src.config import settings
from src.logger import logger


class RAGLLM(Protocol):
    def invoke(self, input: Any, **kwargs: Any) -> str: ...
    def stream(self, input: Any, **kwargs: Any) -> Iterator[str]: ...


class OllamaRAGLLM:
    """ChatOllama com interface invoke/stream compatível com o RAG chain."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ):
        from langchain_ollama import ChatOllama

        self.model = model or settings.ollama_llm_model
        self.base_url = base_url or settings.ollama_base_url
        self.temperature = (
            settings.ollama_temperature if temperature is None else temperature
        )
        self._llm = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
            num_predict=settings.ollama_num_predict,
            num_ctx=settings.ollama_num_ctx,
            keep_alive="30m",
        )
        logger.info(
            "Ollama LLM: %s @ %s (num_predict=%s, num_ctx=%s)",
            self.model,
            self.base_url,
            settings.ollama_num_predict,
            settings.ollama_num_ctx,
        )

    def invoke(self, input: Any, **kwargs: Any) -> str:
        try:
            response = self._llm.invoke(input, **kwargs)
            return getattr(response, "content", str(response))
        except Exception as exc:
            logger.error("Ollama invoke error: %s", exc)
            return (
                f"Erro ao consultar Ollama ({self.base_url}): {exc}. "
                f"Verifique se o serviço está ativo e se o modelo '{self.model}' "
                "está disponível (`ollama list`)."
            )

    def stream(self, input: Any, **kwargs: Any) -> Iterator[str]:
        try:
            for chunk in self._llm.stream(input, **kwargs):
                content = getattr(chunk, "content", None) or ""
                if content:
                    yield content
        except Exception as exc:
            logger.error("Ollama stream error: %s", exc)
            yield f"Erro ao consultar Ollama: {exc}"


class LLMFactory:
    @staticmethod
    def create_llm() -> RAGLLM:
        provider = (settings.llm_provider or "ollama").lower()
        if provider == "ollama":
            return OllamaRAGLLM()
        from src.modules.rag_chain import DeepseekLLM

        logger.info("Using Deepseek API LLM")
        return DeepseekLLM(
            temperature=settings.deepseek_temperature,
            max_tokens=settings.deepseek_max_tokens,
        )

    @staticmethod
    def active_model_name() -> str:
        if (settings.llm_provider or "").lower() == "ollama":
            return settings.ollama_llm_model
        return settings.deepseek_model

    @staticmethod
    def display_name() -> str:
        if (settings.llm_provider or "ollama").lower() == "ollama":
            return f"Ollama ({settings.ollama_llm_model})"
        return "Deepseek (API)"
