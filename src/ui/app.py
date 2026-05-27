"""
Streamlit interface for RAG Agent — UI estilo Nova AI (glass + roxo)
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import importlib

from src.logger import logger

APP_NAME = "RAG Agent"
APP_TAGLINE = "Assistente CDC com recuperação de documentos (RAG)"


def llm_display_name() -> str:
    from src.modules.llm_factory import LLMFactory

    return LLMFactory.display_name()


UI_DIR = Path(__file__).parent
AVATAR_USER = UI_DIR / "assets" / "avatar_user.png"
AVATAR_ASSISTANT = UI_DIR / "assets" / "avatar_assistant.png"


def chat_avatar(role: str) -> str:
    """Avatares coloridos do chat (imagens em src/ui/assets/)."""
    path = AVATAR_USER if role == "user" else AVATAR_ASSISTANT
    if path.is_file():
        return str(path)
    return "🧑" if role == "user" else "🤖"

SUGGESTION_CARDS = [
    (
        "Direitos básicos do consumidor",
        "Entenda os direitos fundamentais previstos no CDC.",
        "Quais são os direitos básicos do consumidor previstos no CDC?",
    ),
    (
        "Compra online e arrependimento",
        "Saiba como o CDC protege compras pela internet.",
        "O que diz o CDC sobre compras pela internet e direito de arrependimento?",
    ),
    (
        "Cláusulas abusivas",
        "Veja o que a lei define e como se proteger.",
        "O que são cláusulas abusivas e como o CDC as trata?",
    ),
    (
        "Reclamação e Procon",
        "Passo a passo para formalizar uma reclamação.",
        "Como o consumidor pode formalizar reclamação segundo o CDC?",
    ),
]

NOVA_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-top: #080512;
    --bg-mid: #1a1040;
    --bg-bottom: #0c0618;
    --accent: #9d5cff;
    --accent-dark: #7b3fe4;
    --glass: rgba(72, 48, 120, 0.35);
    --glass-hover: rgba(100, 70, 160, 0.45);
    --glass-border: rgba(157, 92, 255, 0.22);
    --text: #ffffff;
    --text-muted: #b8aed4;
    --online: #3dff9a;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.stApp {
    background: linear-gradient(
        180deg,
        var(--bg-top) 0%,
        var(--bg-mid) 42%,
        var(--bg-bottom) 100%
    ) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.main .block-container {
    max-width: min(1280px, 94vw) !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    padding-top: 0.5rem !important;
    padding-bottom: 6rem !important;
}

.main p, .main span, .main label, .main .stMarkdown {
    color: var(--text);
}

/* —— Topbar —— */
.rag-topbar {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.75rem 0 1.25rem;
    position: relative;
}
.rag-brand {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}
.rag-badge-online {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    margin-left: 0.65rem;
    padding: 0.2rem 0.55rem 0.2rem 0.45rem;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--online) !important;
    background: rgba(61, 255, 154, 0.1);
    border: 1px solid rgba(61, 255, 154, 0.25);
    border-radius: 999px;
    vertical-align: middle;
}
.rag-badge-online .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--online);
    box-shadow: 0 0 8px var(--online);
}

/* —— Hero —— */
.rag-hero {
    text-align: center;
    padding: 0.5rem 0 1.75rem;
}
.rag-hero-icon {
    width: 72px;
    height: 72px;
    margin: 0 auto 1.25rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(145deg, rgba(157, 92, 255, 0.35), rgba(60, 30, 100, 0.5));
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(157, 92, 255, 0.2);
    font-size: 2rem;
}
.rag-hero h1 {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    margin: 0 0 0.65rem !important;
    line-height: 1.25 !important;
    border: none !important;
    padding: 0 !important;
}
.rag-hero-sub {
    color: var(--text-muted) !important;
    font-size: 0.92rem;
    line-height: 1.55;
    max-width: 560px;
    margin: 0 auto;
}

/* —— Cards de sugestão —— */
.rag-suggestions {
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
    margin: 0.5rem 0 1.5rem;
}
.rag-card {
    background: var(--glass) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--glass-border) !important;
    border-radius: 18px;
    padding: 1rem 1.2rem;
    cursor: default;
    transition: background 0.2s, border-color 0.2s, transform 0.15s;
}
.rag-card-title {
    color: var(--text) !important;
    font-weight: 600;
    font-size: 0.98rem;
    margin-bottom: 0.3rem;
}
.rag-card-desc {
    color: var(--text-muted) !important;
    font-size: 0.84rem;
    line-height: 1.45;
    margin: 0;
}

/* Botões-card na área principal */
section.main .stButton > button {
    width: 100% !important;
    min-height: 4.2rem !important;
    padding: 1rem 1.2rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    white-space: pre-wrap !important;
    line-height: 1.45 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    background: var(--glass) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 18px !important;
    color: var(--text) !important;
    box-shadow: none !important;
}
section.main .stButton > button:hover {
    background: var(--glass-hover) !important;
    border-color: rgba(157, 92, 255, 0.45) !important;
    transform: translateY(-1px);
}
section.main .stButton > button p {
    text-align: left !important;
    line-height: 1.45 !important;
}
section.main .btn-nova-conversa > button {
    min-height: auto !important;
    padding: 0.4rem 0.9rem !important;
    font-size: 0.8rem !important;
    text-align: center !important;
    justify-content: center !important;
    border-radius: 999px !important;
}

/* Badge RAG */
.rag-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 0.75rem;
    border-radius: 12px;
    font-size: 0.82rem;
    margin-bottom: 0.65rem;
    backdrop-filter: blur(8px);
}
.rag-pill-ok {
    background: rgba(61, 255, 154, 0.12);
    border: 1px solid rgba(61, 255, 154, 0.28);
    color: #b8ffe0 !important;
}
.rag-pill-warn {
    background: rgba(255, 180, 80, 0.12);
    border: 1px solid rgba(255, 180, 80, 0.28);
    color: #ffe4b8 !important;
}
.rag-pill-fallback {
    background: rgba(157, 92, 255, 0.15);
    border: 1px solid rgba(157, 92, 255, 0.35);
    color: #e8dcff !important;
}

/* Chat */
.stChatMessage {
    background: var(--glass) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--glass-border) !important;
    border-radius: 18px !important;
    padding: 1rem 1.1rem !important;
}
.stChatMessage p, .stChatMessage div, .stChatMessage span {
    color: var(--text) !important;
}

/* Avatares coloridos (pessoa / assistente) */
[data-testid="stChatMessageAvatar"] {
    width: 2.75rem !important;
    height: 2.75rem !important;
    min-width: 2.75rem !important;
    min-height: 2.75rem !important;
    border-radius: 50% !important;
    overflow: hidden !important;
    padding: 0 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
}
[data-testid="stChatMessageAvatar"] img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    border-radius: 50% !important;
}
[data-testid="stChatMessageAvatarIcon-user"],
[data-testid="stChatMessage"]:has(img[alt="user"]) [data-testid="stChatMessageAvatar"] {
    background: linear-gradient(145deg, #ffc107, #ff9800) !important;
    border: 2px solid #ffeb3b !important;
    box-shadow: 0 4px 16px rgba(255, 193, 7, 0.5) !important;
}
[data-testid="stChatMessageAvatarIcon-assistant"],
[data-testid="stChatMessage"]:has(img[alt="assistant"]) [data-testid="stChatMessageAvatar"] {
    background: linear-gradient(145deg, #7ec8ff, #4a9eff) !important;
    border: 2px solid #b8e4ff !important;
    box-shadow: 0 4px 16px rgba(74, 158, 255, 0.45) !important;
}

/* Input fixo estilo Nova — mesma largura do chat */
[data-testid="stBottomBlock"] {
    background: linear-gradient(0deg, var(--bg-bottom) 30%, transparent) !important;
    padding-bottom: 0.75rem;
}
[data-testid="stBottomBlock"] > div {
    max-width: min(1280px, 94vw) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
[data-testid="stChatInput"] > div {
    background: rgba(28, 18, 52, 0.75) !important;
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border) !important;
    border-radius: 26px !important;
    padding: 0.35rem 0.5rem 0.35rem 1rem !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    background: transparent !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, var(--accent), var(--accent-dark)) !important;
    border: none !important;
    border-radius: 50% !important;
    color: white !important;
    width: 2.5rem !important;
    height: 2.5rem !important;
    min-height: 2.5rem !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    box-shadow: 0 4px 20px rgba(157, 92, 255, 0.45) !important;
}

.rag-input-hint {
    text-align: center;
    color: var(--text-muted) !important;
    font-size: 0.72rem;
    margin-top: -0.25rem;
    padding-bottom: 0.5rem;
}
.rag-input-hint .status-dot {
    display: inline-block;
    width: 5px;
    height: 5px;
    background: var(--online);
    border-radius: 50%;
    margin-right: 0.25rem;
    vertical-align: middle;
    box-shadow: 0 0 6px var(--online);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e0818, #1a1035) !important;
    border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
[data-testid="stSidebar"] .stButton button {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: var(--glass-hover) !important;
    border-color: rgba(157, 92, 255, 0.45) !important;
}
[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-dark)) !important;
    border: none !important;
}

.stSpinner > div {
    border-top-color: var(--accent) !important;
}
/* Spinner só na área do chat, não no topo da página */
[data-testid="stChatMessage"] .stSpinner {
    margin: 0.25rem 0 !important;
}
[data-testid="stChatMessage"] .stSpinner > label {
    color: var(--text-muted) !important;
    font-size: 0.88rem !important;
}

.stAlert, [data-testid="stNotification"] {
    background: var(--glass) !important;
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
}

.streamlit-expanderHeader {
    background: var(--glass) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}
.streamlit-expanderContent {
    background: rgba(10, 6, 20, 0.5) !important;
    border: 1px solid var(--glass-border);
    border-radius: 0 0 12px 12px;
}

hr {
    border-color: var(--glass-border) !important;
    opacity: 0.5;
}

h1, h2, h3, .stHeading {
    color: var(--text) !important;
}

/* Esconde títulos padrão redundantes na área principal */
.rag-hide-default-title h1:first-child {
    display: none;
}
</style>
"""


def _reload_rag_modules():
    """Recarrega módulos para evitar código em cache do Streamlit."""
    import src.config
    import src.modules.embeddings
    import src.modules.vector_store
    import src.modules.document_processor
    import src.modules.rag_chain
    import src.indexing

    for module in (
        src.config,
        src.modules.embeddings,
        src.modules.vector_store,
        src.modules.document_processor,
        src.modules.rag_chain,
        src.indexing,
    ):
        importlib.reload(module)

    from src.config import settings
    from src.indexing import needs_reindex, reindex_documents
    from src.modules.document_processor import DocumentProcessor
    from src.modules.vector_store import VectorStoreManager
    from src.modules.rag_chain import RAGChain

    return settings, needs_reindex, reindex_documents, DocumentProcessor, VectorStoreManager, RAGChain


def setup_page():
    """Configura página e tema visual."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(NOVA_THEME_CSS, unsafe_allow_html=True)


def render_topbar(online: bool = True) -> None:
    """Barra superior com nome do app e status."""
    status = (
        '<span class="rag-badge-online"><span class="dot"></span> ONLINE</span>'
        if online
        else '<span class="rag-badge-online" style="color:#ff9f6b;border-color:rgba(255,159,107,.3)">'
        '<span class="dot" style="background:#ff9f6b;box-shadow:none"></span> OFFLINE</span>'
    )
    st.markdown(
        f'<div class="rag-topbar"><span class="rag-brand">{APP_NAME}{status}</span></div>',
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Área central de boas-vindas."""
    st.markdown(
        """
        <div class="rag-hero">
            <div class="rag-hero-icon">🖥️</div>
            <h1>Como posso ajudar hoje?</h1>
            <p class="rag-hero-sub">
                Faça uma pergunta sobre o CDC ou escolha uma sugestão
                para começar sua conversa com o assistente.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggestion_cards() -> None:
    """Cards clicáveis com fundo semi-transparente."""
    for i, (title, desc, question) in enumerate(SUGGESTION_CARDS):
        label = f"{title}\n{desc}"
        if st.button(label, key=f"suggest_{i}", use_container_width=True):
            st.session_state.pending_query = question
            st.rerun()


def render_rag_badge(meta: dict | None) -> None:
    """Exibe se a resposta foi fundamentada nos trechos do CDC."""
    if not meta:
        return

    count = meta.get("sources_count", 0)
    files = meta.get("source_label") or ", ".join(meta.get("source_files", []))
    note = meta.get("retrieval_note")
    trace = meta.get("graph_trace")
    trace_html = ""
    if trace:
        trace_html = f'<br><small style="opacity:.6">Agente: {" → ".join(trace)}</small>'

    if meta.get("grounded"):
        extra = f" — <em>{files}</em>" if files else ""
        note_html = f'<br><small style="opacity:.75">Recuperação: {note}</small>' if note else ""
        st.markdown(
            f'<div class="rag-pill rag-pill-ok">📎 Baseado em <strong>{count}</strong> '
            f"trecho(s) do CDC{extra}{note_html}{trace_html}</div>",
            unsafe_allow_html=True,
        )
    elif meta.get("fallback"):
        extra = f" — <em>{files}</em>" if files else ""
        note_html = f'<br><small style="opacity:.75">{note}</small>' if note else ""
        st.markdown(
            f'<div class="rag-pill rag-pill-fallback">✨ Complemento <strong>{llm_display_name()}</strong> '
            f"({count} trecho(s) CDC consultado(s){extra}){note_html}{trace_html}</div>",
            unsafe_allow_html=True,
        )
    else:
        extra = f" em <em>{files}</em>" if files else ""
        note_html = f'<br><small style="opacity:.75">Motivo: {note}</small>' if note else ""
        st.markdown(
            f'<div class="rag-pill rag-pill-warn">⚠️ Sem base suficiente no CDC '
            f"({count} trecho(s){extra}){note_html}</div>",
            unsafe_allow_html=True,
        )


def render_input_hint() -> None:
    """Dica abaixo do campo de mensagem."""
    st.markdown(
        """
        <p class="rag-input-hint">
            Enter para enviar · Shift+Enter quebra linha &nbsp;&nbsp;
            <span class="status-dot"></span> Sistema otimizado para usabilidade
        </p>
        """,
        unsafe_allow_html=True,
    )


def _already_answered(user_input: str) -> bool:
    """Evita processar a mesma pergunta duas vezes (rerun do Streamlit)."""
    hist = st.session_state.chat_history
    return (
        len(hist) >= 2
        and hist[-2]["role"] == "user"
        and hist[-2]["content"] == user_input
        and hist[-1]["role"] == "assistant"
    )


def _append_assistant_reply(result: dict) -> None:
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "meta": result.get("meta", {}),
        }
    )


def _append_assistant_error(error: Exception) -> None:
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": f"Erro ao processar pergunta: {error}",
            "sources": [],
            "meta": {},
        }
    )
    logger.error(f"Query error: {error}")


def _ensure_user_message_in_history(user_input: str) -> None:
    hist = st.session_state.chat_history
    if hist and hist[-1]["role"] == "user" and hist[-1]["content"] == user_input:
        return
    st.session_state.chat_history.append({"role": "user", "content": user_input})


def _conversation_context(max_turns: int = 4) -> str:
    """Memória curta: últimos turnos para o grafo LangGraph."""
    hist = st.session_state.chat_history
    if not hist:
        return ""
    lines = []
    for msg in hist[-max_turns * 2 :]:
        role = "Usuário" if msg["role"] == "user" else "Assistente"
        content = (msg.get("content") or "")[:300]
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def render_assistant_typing(user_input: str) -> None:
    """Busca no CDC e gera resposta com streaming (menos espera percebida)."""
    with st.chat_message("assistant", avatar=chat_avatar("assistant")):
        rag = st.session_state.rag_chain
        conv = _conversation_context()

        with st.spinner("Buscando trechos no CDC..."):
            try:
                ctx = rag.prepare_retrieval(user_input, conv)
            except Exception as e:
                _append_assistant_error(e)
                st.error(f"Erro ao processar pergunta: {e}")
                return

        meta = ctx.get("meta", {})
        sources = ctx.get("sources", [])
        render_rag_badge(meta)

        placeholder = st.empty()
        full_answer = ""
        try:
            for token in rag.stream_answer_from_context(ctx):
                full_answer += token
                placeholder.markdown(full_answer + "▌")
            placeholder.markdown(full_answer)
        except Exception as e:
            _append_assistant_error(e)
            st.error(f"Erro ao gerar resposta: {e}")
            return

        result = {"answer": full_answer, "sources": sources, "meta": meta}
        _append_assistant_reply(result)
        render_sources_expander(sources, preview_len=200)


def render_sources_expander(sources: list, preview_len: int = 300) -> None:
    """Lista trechos recuperados usados na resposta."""
    if not sources:
        return
    with st.expander("📚 Documentos de origem (trechos do RAG)"):
        st.caption(
            "Texto enviado ao modelo junto com sua pergunta. "
            f"A redação final é feita pelo **{llm_display_name()}** com base nesses trechos."
        )
        for i, source in enumerate(sources, 1):
            meta = source.get("metadata", {})
            fname = meta.get("filename") or Path(str(meta.get("source", ""))).name
            page = meta.get("page_label") or meta.get("page")
            section = meta.get("section") or ""
            ctype = meta.get("content_type") or ""
            header = f"**Fonte {i}**"
            if fname:
                header += f" — `{fname}`"
            if page is not None:
                header += f" (pág. {page})"
            if section:
                header += f" · _{section[:60]}_"
            if ctype:
                header += f" · [{ctype}]"
            st.markdown(header)
            st.markdown(source.get("content", "N/A")[:preview_len] + "...")


def initialize_session_state():
    """Initialize session state variables"""
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None
    if "document_processor" not in st.session_state:
        st.session_state.document_processor = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "init_error" not in st.session_state:
        st.session_state.init_error = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "last_processed_query" not in st.session_state:
        st.session_state.last_processed_query = None


def initialize_components(show_feedback: bool = False) -> bool:
    """Initialize RAG components. Returns True on success."""
    if st.session_state.initialized:
        return True

    try:
        (
            settings,
            needs_reindex,
            reindex_documents,
            DocumentProcessor,
            VectorStoreManager,
            RAGChain,
        ) = _reload_rag_modules()

        with st.spinner("Inicializando sistema RAG..."):
            st.session_state.vector_store_manager = VectorStoreManager()
            st.session_state.document_processor = DocumentProcessor()

            if needs_reindex():
                with st.spinner(
                    "Indexando documentos (embeddings — pode levar 1–2 min)..."
                ):
                    count = reindex_documents(
                        st.session_state.vector_store_manager,
                        st.session_state.document_processor,
                    )
                    logger.info(f"Auto-reindexed {count} chunks on startup")

            st.session_state.rag_chain = RAGChain(
                st.session_state.vector_store_manager
            )
            st.session_state.initialized = True
            st.session_state.init_error = None
            logger.info("RAG system initialized in Streamlit")

        if show_feedback:
            st.success("Sistema RAG inicializado com sucesso!")
        return True

    except Exception as e:
        st.session_state.initialized = False
        st.session_state.init_error = str(e)
        logger.error(f"Initialization error: {e}")
        return False


def render_sidebar(settings, reindex_documents, RAGChain):
    """Painel lateral de configurações."""
    with st.sidebar:
        st.markdown(f"### ⚙️ {APP_NAME}")
        st.caption(APP_TAGLINE)

        if st.button(
            "Reinicializar Sistema RAG",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.initialized = False
            st.session_state.rag_chain = None
            st.session_state.vector_store_manager = None
            st.session_state.document_processor = None
            initialize_components(show_feedback=True)

        st.divider()
        st.markdown("#### 📄 Documentos")

        if st.session_state.initialized:
            if st.button("Ver informações da coleção", use_container_width=True):
                try:
                    info = st.session_state.vector_store_manager.get_collection_info()
                    st.info(
                        f"**Coleção:** {info.get('name', 'N/A')}\n\n"
                        f"**Chunks:** {info.get('count', 0)}\n\n"
                        f"**Busca híbrida:** {'sim' if info.get('hybrid_search') else 'não'} "
                        f"(BM25: {'ok' if info.get('bm25_ready') else 'pendente reindex'})\n\n"
                        f"**Reranker:** {info.get('reranker_model') or 'desativado'}\n\n"
                        f"**Metadados:** {', '.join(info.get('metadata_fields', [])) or 'N/A'}\n\n"
                        f"**Pasta:** `{info.get('persist_directory', 'N/A')}`"
                    )
                except Exception as e:
                    st.error(f"Erro: {e}")

            if st.button("Reindexar documentos", use_container_width=True):
                try:
                    with st.spinner("Reindexando..."):
                        count = reindex_documents(
                            st.session_state.vector_store_manager,
                            st.session_state.document_processor,
                        )
                        if count:
                            st.session_state.rag_chain = RAGChain(
                                st.session_state.vector_store_manager
                            )
                            st.success(f"✅ {count} chunks indexados!")
                        else:
                            st.warning(
                                f"Nenhum documento em `{settings.documents_dir}`"
                            )
                except Exception as e:
                    st.error(f"Erro: {e}")

            if st.button("Limpar coleção", use_container_width=True):
                try:
                    st.session_state.vector_store_manager.delete_collection()
                    st.session_state.rag_chain = RAGChain(
                        st.session_state.vector_store_manager
                    )
                    st.success("✅ Coleção limpa!")
                except Exception as e:
                    st.error(f"Erro: {e}")

        st.divider()
        st.markdown(
            f"""
            **Stack**
            - LLM: {llm_display_name()}
            - Vetores: ChromaDB
            - Chunk: {settings.max_chunk_size}
            """
        )


def main():
    """Main Streamlit application"""
    setup_page()
    initialize_session_state()

    (
        settings,
        _needs_reindex,
        reindex_documents,
        _DocumentProcessor,
        _VectorStoreManager,
        RAGChain,
    ) = _reload_rag_modules()

    initialize_components()
    render_sidebar(settings, reindex_documents, RAGChain)

    online = st.session_state.initialized and not st.session_state.init_error
    render_topbar(online=online)

    if st.session_state.init_error:
        st.error(
            f"Não foi possível iniciar: {st.session_state.init_error}. "
            "Abra o menu ☰ → Reinicializar Sistema RAG."
        )
        return

    if not st.session_state.initialized:
        st.info("Inicializando sistema RAG...")
        return

    # Pergunta nova: grava só a mensagem do usuário; resposta vem depois do histórico
    query_processing = None
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        if (
            query != st.session_state.last_processed_query
            and not _already_answered(query)
        ):
            _ensure_user_message_in_history(query)
            query_processing = query

    has_chat = bool(st.session_state.chat_history)

    if not has_chat:
        render_hero()
        render_suggestion_cards()
    else:
        st.markdown(
            '<p style="text-align:center;color:#b8aed4;font-size:0.85rem;margin-bottom:1rem;">'
            "Conversa sobre o CDC · respostas com base nos documentos indexados"
            "</p>",
            unsafe_allow_html=True,
        )

    hist = st.session_state.chat_history
    last_is_open_user = (
        query_processing
        and hist
        and hist[-1]["role"] == "user"
        and hist[-1]["content"] == query_processing
    )
    messages_to_render = hist[:-1] if last_is_open_user else hist

    for message in messages_to_render:
        with st.chat_message(message["role"], avatar=chat_avatar(message["role"])):
            if message["role"] == "assistant":
                render_rag_badge(message.get("meta"))
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources_expander(message["sources"], preview_len=200)

    if query_processing:
        if last_is_open_user:
            with st.chat_message("user", avatar=chat_avatar("user")):
                st.markdown(hist[-1]["content"])
        render_assistant_typing(query_processing)
        st.session_state.last_processed_query = query_processing

    if has_chat:
        st.markdown('<div class="btn-nova-conversa">', unsafe_allow_html=True)
        if st.button("↩ Nova conversa"):
            st.session_state.chat_history = []
            st.session_state.pending_query = None
            st.session_state.last_processed_query = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Envie uma mensagem...")
    render_input_hint()

    if user_input and user_input != st.session_state.last_processed_query:
        st.session_state.pending_query = user_input
        st.rerun()


if __name__ == "__main__":
    main()
