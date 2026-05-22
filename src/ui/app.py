"""
Streamlit interface for RAG Agent
"""
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import importlib

from src.logger import logger


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
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="RAG Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS para texto branco em fundo preto
    st.markdown(
        """
        <style>
        /* Fundo preto para toda a aplicação */
        .stApp {
            background-color: #000000 !important;
        }
        
        /* Força texto branco em todos os elementos principais */
        .main, .stApp, .stMarkdown, .stText, p, div, span, label {
            color: #ffffff !important;
        }
        
        /* Títulos em branco */
        h1, h2, h3, h4, h5, h6, .stHeading {
            color: #ffffff !important;
        }
        
        /* Estilo específico para mensagens do chat */
        .stChatMessage {
            background-color: #1e1e1e !important;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            color: #ffffff !important;
        }
        
        /* Força texto branco dentro das mensagens do chat */
        .stChatMessage p, .stChatMessage div, .stChatMessage span {
            color: #ffffff !important;
        }
        
        /* Sidebar com fundo preto e texto branco */
        [data-testid="stSidebar"] {
            background-color: #0e0e0e !important;
            color: #ffffff !important;
        }
        
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] div, 
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }
        
        /* Input do chat com fundo escuro e texto branco */
        .stChatInput input, .stTextInput input {
            color: #ffffff !important;
            background-color: #2d2d2d !important;
            border: 1px solid #404040 !important;
        }
        
        /* Placeholder do input */
        .stChatInput input::placeholder, .stTextInput input::placeholder {
            color: #888888 !important;
        }
        
        /* Botões */
        .stButton button {
            color: #ffffff !important;
            background-color: #2d2d2d !important;
            border: 1px solid #404040 !important;
        }
        
        .stButton button:hover {
            background-color: #404040 !important;
        }
        
        /* Alertas e info boxes */
        .stAlert, .stInfo, .stWarning, .stSuccess, .stError {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            color: #ffffff !important;
            background-color: #1e1e1e !important;
        }
        
        .streamlit-expanderContent {
            background-color: #0e0e0e !important;
            color: #ffffff !important;
        }
        
        /* Código */
        code, pre {
            color: #ffffff !important;
            background-color: #2d2d2d !important;
        }
        
        /* Mensagens de sucesso/erro/info */
        .stSuccess, .stInfo, .stWarning, .stError {
            color: #ffffff !important;
        }
        
        /* Select boxes e outros widgets */
        .stSelectbox div, .stSlider div {
            color: #ffffff !important;
        }
        
        /* Links */
        a {
            color: #4da6ff !important;
        }
        
        /* Divisor */
        hr {
            border-color: #404040 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_rag_badge(meta: dict | None) -> None:
    """Exibe se a resposta foi fundamentada nos trechos do CDC."""
    if not meta:
        return

    count = meta.get("sources_count", 0)
    files = meta.get("source_label") or ", ".join(meta.get("source_files", []))

    if meta.get("grounded"):
        st.success(
            f"📎 Resposta baseada em **{count} trecho(s)** do CDC"
            + (f" — _{files}_" if files else "")
        )
        note = meta.get("retrieval_note")
        if note:
            st.caption(f"Recuperação: {note}")
    else:
        st.warning(
            f"⚠️ Sem base suficiente no CDC indexado "
            f"({count} trecho(s) consultado(s)"
            + (f" em _{files}_" if files else "")
            + ")"
        )
        note = meta.get("retrieval_note")
        if note:
            st.caption(f"Motivo: {note}")


def render_sources_expander(sources: list, preview_len: int = 300) -> None:
    """Lista trechos recuperados usados na resposta."""
    if not sources:
        return
    with st.expander("📚 Documentos de origem (trechos do RAG)"):
        st.caption(
            "Texto enviado ao modelo junto com sua pergunta. "
            "A redação final é feita pelo **Deepseek** com base nesses trechos."
        )
        for i, source in enumerate(sources, 1):
            meta = source.get("metadata", {})
            fname = meta.get("filename") or Path(str(meta.get("source", ""))).name
            page = meta.get("page")
            header = f"**Fonte {i}**"
            if fname:
                header += f" — `{fname}`"
            if page is not None:
                header += f" (pág. {page})"
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
                    "Indexando documentos (embeddings semânticos — pode levar 1–2 min)..."
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
        st.error(f"Erro ao inicializar o sistema RAG: {e}")
        logger.error(f"Initialization error: {e}")
        return False


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

    # Header
    st.title("🤖 RAG Agent - Assistente Inteligente")
    st.markdown(
        "Sistema de Perguntas e Respostas com Recuperação de Documentos (RAG)"
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")

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

        # Document management section
        st.subheader("📄 Gerenciar Documentos")

        if st.session_state.initialized:
            # Display collection info
            if st.button("Ver Informações da Coleção", use_container_width=True):
                try:
                    info = st.session_state.vector_store_manager.get_collection_info()
                    st.info(
                        f"**Coleção:** {info.get('name', 'N/A')}\n\n"
                        f"**Total de chunks:** {info.get('count', 0)}\n\n"
                        f"**Diretório:** {info.get('persist_directory', 'N/A')}"
                    )
                except Exception as e:
                    st.error(f"Erro ao obter informações: {e}")

            # Reindex documents (replaces entire collection)
            if st.button("Reindexar Documentos", use_container_width=True):
                try:
                    with st.spinner("Processando e reindexando documentos..."):
                        count = reindex_documents(
                            st.session_state.vector_store_manager,
                            st.session_state.document_processor,
                        )

                        if count:
                            st.session_state.rag_chain = RAGChain(
                                st.session_state.vector_store_manager
                            )
                            st.success(
                                f"✅ Base reindexada com {count} chunks!"
                            )
                        else:
                            st.warning(
                                "Nenhum documento encontrado em "
                                f"{settings.documents_dir}"
                            )

                except Exception as e:
                    st.error(f"Erro ao ingerir documentos: {e}")

            # Clear collection
            if st.button("Limpar Coleção", use_container_width=True):
                try:
                    st.session_state.vector_store_manager.delete_collection()

                    # Reinitialize RAG chain
                    st.session_state.rag_chain = RAGChain(
                        st.session_state.vector_store_manager
                    )

                    st.success("✅ Coleção limpa com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao limpar coleção: {e}")

        st.divider()

        # System information
        st.subheader("ℹ️ Informações do Sistema")
        st.markdown(
            f"""
            **Configuração:**
            - LLM: Deepseek
            - Vector Store: ChromaDB
            - Chunk Size: {settings.max_chunk_size}
            - Documents Path: `{settings.documents_dir}`
            - Vector Store Path: `{settings.vector_store_dir}`
            """
        )

    # Main content area
    if st.session_state.init_error:
        st.error(
            f"Não foi possível iniciar o sistema: {st.session_state.init_error}. "
            "Use 'Reinicializar Sistema RAG' na barra lateral após corrigir o problema."
        )
    elif not st.session_state.initialized:
        st.info("Inicializando sistema RAG...")
    else:
        # Chat interface
        st.subheader("💬 Chat")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    render_rag_badge(message.get("meta"))
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("sources"):
                    render_sources_expander(message["sources"], preview_len=200)

        # Input
        user_input = st.chat_input("Digite sua pergunta...")

        if user_input:
            # Add user message to history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Get response
            try:
                with st.spinner("Processando pergunta..."):
                    result = st.session_state.rag_chain.query(user_input)

                    # Add assistant message to history
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "meta": result.get("meta", {}),
                        }
                    )

                    with st.chat_message("assistant"):
                        render_rag_badge(result.get("meta"))
                        st.markdown(result["answer"])
                        render_sources_expander(result["sources"])

            except Exception as e:
                st.error(f"Erro ao processar pergunta: {e}")
                logger.error(f"Query error: {e}")


if __name__ == "__main__":
    main()