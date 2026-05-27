"""
RAG Chain implementation for RAG Agent
Combines retrieval and generation with Deepseek
"""
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional
import httpx

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.logger import logger
from src.config import settings
from src.modules.vector_store import VectorStoreManager

SYSTEM_PROMPT = """Você é um assistente especializado no Código de Defesa do Consumidor (CDC), Lei nº 8.078/1990, e normas correlatas.

Regras:
- Use PRINCIPALMENTE o contexto abaixo (trechos do CDC e legislação indexada).
- Responda em português do Brasil, com títulos e tópicos organizados.
- Para resumos gerais do CDC: sintetize com os trechos disponíveis.
- Cite artigos somente quando aparecerem no contexto.
- Se o contexto for parcial, responda com o que houver nos trechos.

Contexto:
{context}"""

FALLBACK_SYSTEM_PROMPT = """Você é um assistente especializado em direito do consumidor, CDC (Lei 8.078/1990) e atuação do Procon no Brasil.

A pergunta do usuário NÃO está respondida de forma completa nos trechos do CDC indexados abaixo (podem ser vazios ou só tangenciar o tema).

Instruções:
- Responda em português do Brasil, de forma clara, prática e organizada (títulos, listas e passos quando couber).
- Complemente com conhecimento geral sobre CDC, Procon, reclamações e defesa do consumidor.
- Use os trechos parciais do CDC quando forem úteis; deixe claro o que veio do CDC indexado.
- Não invente número de artigo do CDC que não apareça no contexto.
- Para documentos e procedimentos do Procon: informe documentos usualmente exigidos no Brasil e avise que podem variar por estado/município — recomende confirmar no site do Procon local.
- Seja útil: o usuário espera orientação concreta, não apenas “não consta no material”.

Trechos parciais do CDC indexado (podem ser insuficientes):
{context}"""

_QUERY_STOPWORDS = frozenset({
    "quero", "como", "pode", "sobre", "para", "fazer", "tenho", "essa", "qual",
    "quais", "onde", "quando", "deve", "será", "sera", "posso", "criar", "uma",
    "uns", "das", "dos", "que", "com", "por", "nos", "nas", "the", "and",
})

# Marcadores no texto recuperado que indicam resposta procedimental de verdade
_INTENT_MARKERS = {
    "documentation": (
        "documento", "documentação", "documentacao", "cpf", "rg", "comprovante",
        "nota fiscal", "contrato", "formulário", "formulario", "petição", "peticao",
        "cópia", "copia", "procuração", "procuracao", "identidade", "anexar",
        "juntar", "apresentar documento",
    ),
    "complaint_procedure": (
        "reclamação formal", "reclamacao formal", "protocolo de reclamação",
        "formulário de reclamação", "formulario de reclamacao", "proconsumidor",
        "atendimento presencial", "agendamento", "registro de reclamação",
    ),
}

_TOPIC_SYNONYMS = {
    "reclamação": ("reclama", "reclamação", "reclamacao", "ouvidoria", "denúncia", "denuncia", "atendimento"),
    "reclamacao": ("reclama", "reclamação", "reclamacao", "ouvidoria"),
    "documentação": ("document", "formal", "petição", "peticao", "requerimento", "notificação", "notificacao"),
    "documentacao": ("document", "formal", "petição", "peticao", "requerimento"),
    "arrependimento": ("arrependimento", "desistir", "7 dias", "sete dias"),
    "internet": ("internet", "eletrônico", "eletronico", "comércio eletrônico", "comercio eletronico", "7.962"),
    "compra": ("compra", "contratação", "contratacao", "fornecimento"),
    "prazo": ("prazo", "prescrição", "prescricao", "decadência", "decadencia"),
    "garantia": ("garantia", "vício", "vicio", "defeito"),
    "abusiva": ("abusiva", "abusivas", "cláusula", "clausula", "nulas"),
}


class DeepseekLLM:
    """Custom LLM class for Deepseek API"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        self.api_key = api_key or settings.deepseek_api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = settings.deepseek_base_url

        if not self.api_key:
            logger.warning("Deepseek API key is not configured. Using placeholder responses.")

    def invoke(self, input: Any, config=None, **kwargs) -> str:
        messages = self._to_api_messages(input)
        return self._call_api(messages, **kwargs)

    def _to_api_messages(self, input: Any) -> List[dict]:
        """Convert LangChain messages to Deepseek API format."""
        if isinstance(input, list):
            api_messages = []
            for msg in input:
                role = "user"
                msg_type = getattr(msg, "type", None)
                if msg_type == "system":
                    role = "system"
                elif msg_type in ("human", "user"):
                    role = "user"
                elif msg_type == "ai":
                    role = "assistant"
                elif isinstance(msg, dict):
                    role = msg.get("role", "user")
                content = getattr(msg, "content", None) or str(msg)
                api_messages.append({"role": role, "content": content})
            if api_messages:
                return api_messages

        return [{"role": "user", "content": self._extract_prompt_string(input)}]

    def _extract_prompt_string(self, input: Any) -> str:
        if isinstance(input, str):
            return input

        if hasattr(input, "to_string"):
            return input.to_string()
        if hasattr(input, "text"):
            return input.text

        if isinstance(input, list):
            contents = []
            for msg in input:
                if hasattr(msg, "content"):
                    contents.append(msg.content)
                elif isinstance(msg, dict) and "content" in msg:
                    contents.append(msg["content"])
                elif isinstance(msg, str):
                    contents.append(msg)
            if contents:
                return "\n".join(contents)

        return str(input)

    def _call_api(self, messages: List[dict], **kwargs: Any) -> str:
        try:
            if not self.api_key or self.api_key == "your_deepseek_api_key_here":
                logger.warning("Deepseek API key not configured. Returning placeholder response.")
                return (
                    "Configure sua DEEPSEEK_API_KEY no arquivo .env para obter respostas reais."
                )

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Error calling Deepseek API: {e}")
            return f"Erro ao consultar a API: {e}"

    def stream(self, input: Any, **kwargs: Any) -> Iterator[str]:
        """Stream tokens from Deepseek API (SSE)."""
        if not self.api_key or self.api_key == "your_deepseek_api_key_here":
            yield "Configure sua DEEPSEEK_API_KEY no arquivo .env para obter respostas reais."
            return

        messages = self._to_api_messages(input)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content") or ""
                        if content:
                            yield content
        except Exception as e:
            logger.error(f"Error streaming from Deepseek API: {e}")
            yield f"Erro ao consultar a API: {e}"


class RAGChain:
    """RAG Chain implementation"""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm: Optional[DeepseekLLM] = None,
        streaming: bool = False,
    ):
        self.vector_store_manager = vector_store_manager
        self.streaming = streaming

        if llm is None:
            from src.modules.llm_factory import LLMFactory

            self.llm = LLMFactory.create_llm()
            logger.info("LLM provider: %s", settings.llm_provider)
        else:
            self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Pergunta: {question}\n\nResposta:"),
        ])
        self.fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", FALLBACK_SYSTEM_PROMPT),
            ("human", "Pergunta: {question}\n\nResposta:"),
        ])

        self._langgraph = None
        if settings.langgraph_enabled:
            try:
                from src.agents.graph import build_rag_graph
                self._langgraph = build_rag_graph(self)
                logger.info("LangGraph RAG agent enabled")
            except ImportError as exc:
                logger.warning("LangGraph unavailable (%s) — using linear pipeline", exc)

        logger.info("RAG Chain initialized successfully")

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))

    def _is_overview_question(self, question: str) -> bool:
        q = question.lower()
        overview_terms = (
            "resumo", "o que é", "o que e", "visão geral", "visao geral",
            "conceito", "definição", "definicao", "explicar", "introdução",
            "introducao", "cdc", "código de defesa", "codigo de defesa",
            "direito do consumidor", "defesa do consumidor", "codigo de direito",
        )
        return any(t in q for t in overview_terms)

    def _build_retrieval_queries(self, question: str) -> List[str]:
        queries = [question]
        if self._is_overview_question(question):
            # Uma query extra basta (antes eram 4 — 5× mais lento com Ollama embeddings)
            queries.append(
                "CDC Lei 8078 direitos básicos do consumidor definição fornecedor"
            )
        return queries

    def _retrieve_from_queries(
        self, queries: List[str], question: str
    ) -> List[Document]:
        seen: set = set()
        merged: List[Document] = []
        n_queries = max(len(queries), 1)
        per_query_k = min(settings.hybrid_fetch_k, max(8, settings.hybrid_fetch_k // n_queries))
        overview = self._is_overview_question(question)

        for q in queries:
            for doc in self.vector_store_manager.advanced_search(
                q,
                k=per_query_k,
                rerank_top_k=settings.rerank_top_k + (1 if overview else 0),
            ):
                key = doc.page_content[:300]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        top_k = settings.rerank_top_k + (2 if overview else 0)
        return merged[:top_k]

    def _retrieve_documents(self, question: str) -> List[Document]:
        return self._retrieve_from_queries(
            self._build_retrieval_queries(question), question
        )

    def _source_filenames(self, docs: List[Document]) -> List[str]:
        names = []
        for doc in docs:
            src = doc.metadata.get("filename") or doc.metadata.get("source", "")
            if src:
                name = Path(str(src)).name
                if name not in names:
                    names.append(name)
        return names or ["desconhecido"]

    def _detect_question_intents(self, question: str) -> set:
        """Intenções que exigem detalhe procedimental no contexto recuperado."""
        q = question.lower()
        intents = set()
        if re.search(
            r"\b(documento|documentação|documentacao|anex|juntar|apresentar|comprovante)\b",
            q,
        ):
            intents.add("documentation")
        if re.search(
            r"\b(procon|reclamação|reclamacao|reclamar|ouvidoria|denúncia|denuncia)\b",
            q,
        ):
            intents.add("complaint_procedure")
        return intents

    def _context_supports_intents(self, combined: str, intents: set) -> bool:
        if not intents:
            return True
        for intent in intents:
            markers = _INTENT_MARKERS.get(intent, ())
            if not any(marker in combined for marker in markers):
                return False
        return True

    def _assess_retrieval_mode(
        self, question: str, docs: List[Document]
    ) -> tuple[str, str]:
        """
        Define modo de resposta:
        - strict: só contexto CDC (RAG)
        - fallback: LLM local complementa quando o índice não cobre a pergunta
        """
        llm = self._llm_label()
        if not docs:
            return "fallback", f"Nenhum trecho recuperado — usando {llm}."

        combined = "\n".join(d.page_content.lower() for d in docs)

        cdc_markers = (
            "consumidor", "8.078", "defesa do consumidor",
            "código de defesa", "codigo de defesa", "lei nº 8.078",
        )
        if not any(m in combined for m in cdc_markers):
            return "fallback", f"Trechos não parecem CDC — usando {llm}."

        if self._is_overview_question(question):
            return "strict", "Pergunta de visão geral do CDC."

        intents = self._detect_question_intents(question)
        if intents and not self._context_supports_intents(combined, intents):
            labels = {
                "documentation": "documentação exigida",
                "complaint_procedure": "procedimento de reclamação/Procon",
            }
            missing = ", ".join(labels.get(i, i) for i in sorted(intents))
            return (
                "fallback",
                f"CDC indexado não detalha ({missing}) — complemento {llm}.",
            )

        q_terms = [
            t for t in self._tokenize(question)
            if len(t) >= 4 and t not in _QUERY_STOPWORDS
        ]

        if q_terms:
            hits = sum(1 for t in q_terms if t in combined)
            if hits >= 2:
                return "strict", f"{hits} termo(s) da pergunta no contexto."

            for term in q_terms:
                for syn in _TOPIC_SYNONYMS.get(term, ()):
                    if syn in combined:
                        return "strict", f"Tema relacionado ({syn}) no contexto."

            if hits == 1 and not intents:
                return "strict", "1 termo da pergunta no contexto."

        if intents:
            return "fallback", f"Menção ao tema sem detalhe procedimental — {llm}."

        return (
            "fallback",
            f"Tema não coberto pelos trechos indexados — complemento {llm}.",
        )

    def _llm_label(self) -> str:
        from src.modules.llm_factory import LLMFactory

        return LLMFactory.display_name()

    def _generate_answer(
        self, question: str, docs: List[Document], mode: str
    ) -> str:
        context = (
            self._format_context(docs)
            if docs
            else "(Nenhum trecho relevante recuperado da base indexada.)"
        )
        template = self.prompt if mode == "strict" else self.fallback_prompt
        messages = template.format_messages(context=context, question=question)
        return str(self.llm.invoke(messages))

    def _build_meta(self, docs: List[Document], mode: str, note: str) -> dict:
        files = self._source_filenames(docs)
        return {
            "grounded": mode == "strict",
            "fallback": mode == "fallback",
            "mode": mode,
            "sources_count": len(docs),
            "source_files": files,
            "source_label": ", ".join(files),
            "retrieval_note": note,
            "retrieval": {
                "hybrid": settings.hybrid_search_enabled,
                "reranker": settings.reranker_enabled,
            },
        }

    def _format_context(self, docs: List[Document]) -> str:
        parts = []
        total_chars = 0
        max_chars = settings.max_context_chars

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            source = meta.get("source", "desconhecida")
            page = meta.get("page_label") or meta.get("page")
            section = meta.get("section") or ""
            content_type = meta.get("content_type") or "text"

            header = f"[Trecho {i}"
            if source:
                header += f" | {Path(str(source)).name}"
            if page is not None:
                header += f" | pág. {page}"
            if section:
                header += f" | {section[:80]}"
            header += f" | tipo: {content_type}]"

            body = doc.page_content
            block = f"{header}\n{body}"
            if total_chars + len(block) > max_chars:
                remaining = max_chars - total_chars - len(header) - 20
                if remaining > 200:
                    block = f"{header}\n{body[:remaining]}…"
                    parts.append(block)
                break
            parts.append(block)
            total_chars += len(block)

        return "\n\n---\n\n".join(parts)

    def prepare_retrieval(
        self,
        question: str,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """
        Retrieval + grade alinhado ao LangGraph (sem geração).
        Usado por streaming (WebSocket / Chainlit).
        """
        from src.agents.nodes import REWRITE_PROMPT
        from src.agents.web_search import search_web

        graph_trace: List[str] = ["analyze_query"]
        queries = self._build_retrieval_queries(question)
        attempts = 0
        docs: List[Document] = []
        mode = "fallback"
        note = ""
        web_context = ""

        while True:
            docs = self._retrieve_from_queries(queries, question)
            graph_trace.append("retrieve")
            mode, note = self._assess_retrieval_mode(question, docs)

            if not docs and attempts < settings.max_retrieval_retries:
                graph_trace.append(f"grade:{mode}->rewrite")
                messages = REWRITE_PROMPT.format_messages(question=question)
                rewritten = str(self.llm.invoke(messages)).strip() or question
                queries = [rewritten]
                attempts += 1
                graph_trace.append("rewrite_query")
                continue

            if mode == "fallback" and not docs and settings.tavily_api_key.strip():
                graph_trace.append(f"grade:{mode}->web_search")
                web_context = search_web(question)
                if web_context:
                    note = f"{note} Complemento web (Tavily).".strip()
                graph_trace.append("web_search")

            graph_trace.append(f"grade:{mode}->generate")
            break

        formatted_question = question
        if conversation_context.strip():
            formatted_question = (
                f"Contexto da conversa:\n{conversation_context}\n\n"
                f"Pergunta atual: {question}"
            )

        meta = self._build_meta(docs, mode, note)
        if settings.langgraph_enabled:
            meta["langgraph"] = True
            meta["graph_trace"] = graph_trace
        if web_context:
            meta["web_enriched"] = True

        return {
            "question": formatted_question,
            "raw_question": question,
            "documents": docs,
            "mode": mode,
            "web_context": web_context,
            "meta": meta,
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ],
        }

    def _build_llm_messages_for_context(self, ctx: Dict[str, Any]) -> list:
        docs = ctx.get("documents") or []
        mode = ctx.get("mode") or "fallback"
        web_context = ctx.get("web_context") or ""

        context = self._format_context(docs) if docs else ""
        if web_context:
            context += (
                "\n\n---\n\n[Complemento web — não substitui o CDC indexado]\n"
                + web_context
            )
        if not context.strip():
            context = "(Nenhum trecho relevante recuperado da base indexada.)"

        template = self.prompt if mode == "strict" else self.fallback_prompt
        return template.format_messages(
            context=context,
            question=ctx["question"],
        )

    def stream_answer_from_context(self, ctx: Dict[str, Any]) -> Iterator[str]:
        messages = self._build_llm_messages_for_context(ctx)
        yield from self.llm.stream(messages)

    def query_stream(
        self,
        question: str,
        conversation_context: str = "",
    ) -> Generator[Dict[str, Any], None, None]:
        """Generator síncrono de eventos para clientes streaming."""
        yield {"type": "status", "message": f"Consultando CDC e {self._llm_label()}..."}
        ctx = self.prepare_retrieval(question, conversation_context)
        yield {"type": "meta", "meta": ctx["meta"], "sources": ctx["sources"]}

        full_answer = ""
        for token in self.stream_answer_from_context(ctx):
            full_answer += token
            yield {"type": "token", "content": token}

        yield {
            "type": "done",
            "answer": full_answer,
            "meta": ctx["meta"],
            "sources": ctx["sources"],
        }

    def query(
        self,
        question: str,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """Run a query through the RAG chain"""
        from src.observability.tracer import trace_rag_query

        started = time.perf_counter()
        try:
            logger.info(f"Processing query: {question}")

            if self._langgraph is not None:
                result = self._query_via_graph(question, conversation_context)
            else:
                result = self._query_linear(question, conversation_context)

            duration_ms = (time.perf_counter() - started) * 1000
            trace_rag_query(question, result, duration_ms, conversation_context)
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def _query_linear(
        self,
        question: str,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        docs = self._retrieve_documents(question)
        mode, relevance_note = self._assess_retrieval_mode(question, docs)
        meta = self._build_meta(docs, mode, relevance_note)

        q = question
        if conversation_context.strip():
            q = (
                f"Contexto da conversa:\n{conversation_context}\n\n"
                f"Pergunta atual: {question}"
            )

        answer_str = self._generate_answer(q, docs, mode)

        max_chars = settings.log_answer_max_chars
        logged_answer = answer_str
        if len(logged_answer) > max_chars:
            logged_answer = f"{logged_answer[:max_chars]}... [truncado no log]"

        logger.info(
            "Query processed | sources=%s | mode=%s | %s",
            len(docs),
            mode,
            relevance_note,
        )
        logger.info("Answer:\n%s", logged_answer)

        return {
            "answer": answer_str,
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ],
            "meta": meta,
        }

    def _query_via_graph(
        self,
        question: str,
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        initial = {
            "question": question,
            "conversation_context": conversation_context,
            "retrieval_attempts": 0,
            "graph_trace": [],
            "documents": [],
        }
        final = self._langgraph.invoke(initial)

        docs = final.get("documents") or []
        mode = final.get("mode") or "fallback"
        note = final.get("relevance_note") or ""
        trace = final.get("graph_trace") or []

        meta = self._build_meta(docs, mode, note)
        meta["langgraph"] = True
        meta["graph_trace"] = trace
        if final.get("web_context"):
            meta["web_enriched"] = True

        answer_str = final.get("answer", "")
        max_chars = settings.log_answer_max_chars
        logged = answer_str
        if len(logged) > max_chars:
            logged = f"{logged[:max_chars]}... [truncado no log]"
        logger.info(
            "Query processed (LangGraph) | trace=%s | sources=%s | mode=%s | %s",
            " → ".join(trace),
            len(docs),
            mode,
            note,
        )
        logger.info("Answer:\n%s", logged)

        return {
            "answer": answer_str,
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ],
            "meta": meta,
        }

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        results = []
        for question in questions:
            try:
                results.append(self.query(question))
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                results.append({
                    "answer": "Erro ao processar a pergunta.",
                    "sources": [],
                    "meta": {
                        "grounded": False,
                        "fallback": True,
                        "mode": "fallback",
                        "sources_count": 0,
                        "source_files": [],
                    },
                })
        return results
