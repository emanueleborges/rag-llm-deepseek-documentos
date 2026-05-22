"""
RAG Chain implementation for RAG Agent
Combines retrieval and generation with Deepseek
"""
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
import httpx

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.logger import logger
from src.config import settings
from src.modules.vector_store import VectorStoreManager

SYSTEM_PROMPT = """Você é um assistente especializado no Código de Defesa do Consumidor (CDC), Lei nº 8.078/1990, e normas correlatas.

Regras:
- Use APENAS o contexto abaixo (trechos do CDC e legislação indexada).
- Responda em português do Brasil, com títulos e tópicos organizados.
- Se o contexto NÃO tratar do tema específico da pergunta (ex.: reclamação, compra online, arrependimento), responda EXATAMENTE que não há trechos suficientes no material indexado sobre aquele tema e não invente procedimentos ou artigos.
- Para resumos gerais do CDC: sintetize com os trechos disponíveis (consumidor, fornecedor, direitos, cláusulas abusivas, etc.).
- Cite artigos somente quando aparecerem no contexto.
- Não use conhecimento externo ao contexto. Lacunas específicas: diga o que não consta nos trechos.

Contexto:
{context}"""

_QUERY_STOPWORDS = frozenset({
    "quero", "como", "pode", "sobre", "para", "fazer", "tenho", "essa", "qual",
    "quais", "onde", "quando", "deve", "será", "sera", "posso", "criar", "uma",
    "uns", "das", "dos", "que", "com", "por", "nos", "nas", "the", "and",
})

_TOPIC_SYNONYMS = {
    "reclamação": ("reclama", "reclamação", "reclamacao", "ouvidoria", "denúncia", "denuncia", "procon", "atendimento"),
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


# Terms that boost relevance for consumer-law questions
_CDC_BOOST_TERMS = frozenset({
    "consumidor", "consumidores", "fornecedor", "cdc", "defesa",
    "contrato", "contratos", "cláusula", "cláusulas", "abusiva",
    "responsabilidade", "produto", "serviço", "servicos", "direito",
    "arrependimento", "internet", "eletrônico", "eletronico", "comércio",
    "comercio", "publicidade", "garantia", "cobrança", "cobranca",
    "crédito", "credito", "coletiva", "coletivo", "sanção", "sancao",
    "8078", "7962", "8078/1990",
})

# Annex topics often retrieved incorrectly for general CDC questions
_IRRELEVANT_FOR_OVERVIEW = (
    "biossegurança", "biosseguranca", "ctnbio", "transgênico", "transgenico",
    "água potável", "agua potavel", "conta mensal de água",
)


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
            logger.info("Creating default Deepseek LLM")
            self.llm = DeepseekLLM(
                temperature=settings.deepseek_temperature,
                max_tokens=settings.deepseek_max_tokens,
            )
        else:
            self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Pergunta: {question}\n\nResposta:"),
        ])

        logger.info("RAG Chain initialized successfully")

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))

    def _rerank_documents(
        self, question: str, docs: List[Document], top_k: int = None
    ) -> List[Document]:
        """Re-rank by keyword overlap and CDC-related terms."""
        if not docs:
            return []

        top_k = top_k or settings.rerank_top_k
        q_terms = self._tokenize(question)
        wants_overview = bool(
            q_terms
            & {"resumo", "visão", "visao", "geral", "overview", "direito", "cdc", "consumidor"}
        )

        def score(doc: Document) -> float:
            text = doc.page_content.lower()
            doc_terms = self._tokenize(text)

            overlap = len(q_terms & doc_terms)
            cdc_boost = sum(3 for t in _CDC_BOOST_TERMS if t in text)
            penalty = 0.0

            if wants_overview and any(term in text for term in _IRRELEVANT_FOR_OVERVIEW):
                penalty -= 8.0

            # Prefer chunks with structural CDC markers
            if re.search(r"art\.\s*\d+", text, re.IGNORECASE):
                cdc_boost += 2
            if "código de defesa do consumidor" in text or "codigo de defesa do consumidor" in text:
                cdc_boost += 4
            if "título" in text or "capítulo" in text or "capitulo" in text:
                cdc_boost += 1

            return overlap + cdc_boost + penalty

        ranked = sorted(docs, key=score, reverse=True)
        return ranked[:top_k]

    def _is_overview_question(self, question: str) -> bool:
        q = question.lower()
        overview_terms = (
            "resumo", "o que é", "o que e", "visão geral", "visao geral",
            "conceito", "definição", "definicao", "explicar", "introdução",
            "introducao", "cdc", "código de defesa", "codigo de defesa",
            "direito do consumidor", "defesa do consumidor", "codigo de direito",
        )
        return any(t in q for t in overview_terms)

    def _retrieve_documents(self, question: str) -> List[Document]:
        queries = [question]
        if self._is_overview_question(question):
            queries.extend([
                "Código de Defesa do Consumidor Lei 8078 objetivo política nacional relações de consumo",
                "definição consumidor fornecedor produto serviço artigo 2 artigo 3",
                "direitos básicos do consumidor artigo 6 CDC",
                "CÓDIGO DE PROTEÇÃO E DEFESA DO CONSUMIDOR",
            ])

        seen: set = set()
        merged: List[Document] = []
        per_query_k = max(8, settings.retrieval_k // len(queries))

        for q in queries:
            for doc in self.vector_store_manager.similarity_search(q, k=per_query_k):
                key = doc.page_content[:300]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        top_k = settings.rerank_top_k + (4 if self._is_overview_question(question) else 0)
        return self._rerank_documents(question, merged, top_k=top_k)

    def _source_filenames(self, docs: List[Document]) -> List[str]:
        names = []
        for doc in docs:
            src = doc.metadata.get("filename") or doc.metadata.get("source", "")
            if src:
                name = Path(str(src)).name
                if name not in names:
                    names.append(name)
        return names or ["desconhecido"]

    def _assess_context_relevance(
        self, question: str, docs: List[Document]
    ) -> tuple[bool, str]:
        """Check if retrieved chunks likely support answering the question."""
        if not docs:
            return False, "Nenhum trecho recuperado da base."

        combined = "\n".join(d.page_content.lower() for d in docs)

        cdc_markers = (
            "consumidor", "8.078", "defesa do consumidor",
            "código de defesa", "codigo de defesa", "lei nº 8.078",
        )
        if not any(m in combined for m in cdc_markers):
            return False, "Trechos recuperados não parecem ser do CDC."

        q_terms = [
            t for t in self._tokenize(question)
            if len(t) >= 4 and t not in _QUERY_STOPWORDS
        ]

        if self._is_overview_question(question):
            return True, "Pergunta de visão geral do CDC."

        if q_terms:
            hits = sum(1 for t in q_terms if t in combined)
            if hits >= 1:
                return True, f"{hits} termo(s) da pergunta no contexto."

            for term in q_terms:
                for syn in _TOPIC_SYNONYMS.get(term, ()):
                    if syn in combined:
                        return True, f"Tema relacionado ({syn}) no contexto."

        return (
            False,
            "Nenhum trecho indexado aborda diretamente o tema da pergunta.",
        )

    def _build_refusal_answer(
        self, question: str, docs: List[Document], reason: str
    ) -> str:
        files = ", ".join(self._source_filenames(docs)) if docs else "nenhum"
        return (
            "Não encontrei trechos suficientes no **CDC indexado** para responder com "
            f"segurança sobre: *{question.strip()}*.\n\n"
            f"**Motivo:** {reason}\n\n"
            f"**Trechos consultados:** {len(docs)} (arquivo(s): {files}).\n\n"
            "Sugestões:\n"
            "- Reformule a pergunta com termos do CDC (ex.: artigo, direito de arrependimento, reclamação);\n"
            "- Use **Reindexar Documentos** na barra lateral;\n"
            "- Verifique se o PDF está em `data/documents/`."
        )

    def _build_meta(self, docs: List[Document], grounded: bool, note: str) -> dict:
        files = self._source_filenames(docs)
        return {
            "grounded": grounded,
            "sources_count": len(docs),
            "source_files": files,
            "source_label": ", ".join(files),
            "retrieval_note": note,
        }

    def _format_context(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "desconhecida")
            page = doc.metadata.get("page")
            header = f"[Trecho {i}"
            if source:
                header += f" | {Path(str(source)).name}"
            if page is not None:
                header += f" | pág. {page}"
            header += "]"
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def query(self, question: str) -> Dict[str, Any]:
        """Run a query through the RAG chain"""
        try:
            logger.info(f"Processing query: {question}")

            docs = self._retrieve_documents(question)
            grounded, relevance_note = self._assess_context_relevance(question, docs)
            meta = self._build_meta(docs, grounded, relevance_note)

            if not grounded:
                answer_str = self._build_refusal_answer(question, docs, relevance_note)
                logger.info(
                    "Query refused (insufficient context) | sources=%s | %s",
                    len(docs),
                    relevance_note,
                )
                logger.info("Answer:\n%s", answer_str)
                return {
                    "answer": answer_str,
                    "sources": [
                        {"content": doc.page_content, "metadata": doc.metadata}
                        for doc in docs
                    ],
                    "meta": meta,
                }

            context = self._format_context(docs)
            messages = self.prompt.format_messages(context=context, question=question)
            answer = self.llm.invoke(messages)
            answer_str = str(answer)

            max_chars = settings.log_answer_max_chars
            logged_answer = answer_str
            if len(logged_answer) > max_chars:
                logged_answer = f"{logged_answer[:max_chars]}... [truncado no log]"

            logger.info(
                "Query processed successfully | sources=%s | grounded=true | %s",
                len(docs),
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

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

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
                    "meta": {"grounded": False, "sources_count": 0, "source_files": []},
                })
        return results
