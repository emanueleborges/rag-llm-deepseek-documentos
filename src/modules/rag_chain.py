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
        self.fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", FALLBACK_SYSTEM_PROMPT),
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
        - fallback: Deepseek complementa quando o índice não cobre a pergunta
        """
        if not docs:
            return "fallback", "Nenhum trecho recuperado — usando Deepseek."

        combined = "\n".join(d.page_content.lower() for d in docs)

        cdc_markers = (
            "consumidor", "8.078", "defesa do consumidor",
            "código de defesa", "codigo de defesa", "lei nº 8.078",
        )
        if not any(m in combined for m in cdc_markers):
            return "fallback", "Trechos não parecem CDC — usando Deepseek."

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
                f"CDC indexado não detalha ({missing}) — complemento Deepseek.",
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
            return "fallback", "Menção ao tema sem detalhe procedimental — Deepseek."

        return (
            "fallback",
            "Tema não coberto pelos trechos indexados — complemento Deepseek.",
        )

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
            mode, relevance_note = self._assess_retrieval_mode(question, docs)
            meta = self._build_meta(docs, mode, relevance_note)

            answer_str = self._generate_answer(question, docs, mode)

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
                    "meta": {
                        "grounded": False,
                        "fallback": True,
                        "mode": "fallback",
                        "sources_count": 0,
                        "source_files": [],
                    },
                })
        return results
