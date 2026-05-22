# RAG Agent - Retrieval-Augmented Generation System

## 📋 Descrição

Um sistema inteligente de Perguntas e Respostas (Q&A) que combina **Retrieval-Augmented Generation (RAG)**, **LangChain**, **ChromaDB** e **Deepseek API** para fornecer respostas precisas e contextualizadas baseadas em seus documentos.

### 🎯 Características

- ✅ **RAG Completo**: Recuperação de documentos + Geração com IA
- ✅ **Deepseek Integration**: Utiliza a API do Deepseek para LLM
- ✅ **Vector Store**: ChromaDB para armazenamento e busca eficiente
- ✅ **API FastAPI**: Endpoints RESTful para integração
- ✅ **Interface Streamlit**: UI intuitiva e interativa
- ✅ **Processamento de Documentos**: Suporte a PDF, TXT e outros formatos
- ✅ **Logging Completo**: Rastreamento detalhado de operações
- ✅ **Modular**: Arquitetura bem organizada e extensível

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────┐
│          Frontend (Streamlit UI)            │
│  Interface web interativa                   │
└──────────────────┬──────────────────────────┘
                   │ HTTP
┌──────────────────▼──────────────────────────┐
│        API Layer (FastAPI)                  │
│  Endpoints para queries e gerenciamento     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│      RAG Chain (LangChain)                  │
│  Orquestração de retrieval + generation     │
└───────┬──────────────────────┬──────────────┘
        │                      │
┌───────▼────────┐    ┌────────▼──────────┐
│  Vector Store  │    │  LLM Provider     │
│   ChromaDB     │    │  Deepseek API     │
└───────▲────────┘    └───────────────────┘
        │
┌───────┴──────────────────┐
│ Document Processor       │
│ (PDF, TXT, etc)          │
└──────────────────────────┘
```

## 🚀 Início Rápido

### 1. Pré-requisitos

- Python 3.8+
- pip ou conda
- API Key do Deepseek (obtenha em https://www.deepseek.com)

### 2. Instalação

```bash
# Clone o repositório (se aplicável)
cd rag_agent

# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 3. Configuração

```bash
# Copie o arquivo .env.example para .env
cp .env.example .env

# Edite o arquivo .env e adicione sua API key do Deepseek
# DEEPSEEK_API_KEY=sua_chave_api_aqui
```

**Variáveis de ambiente importantes:**

```env
# Deepseek API
DEEPSEEK_API_KEY=sua_chave_aqui
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Vector Store
VECTOR_STORE_PATH=./data/vector_store
CHROMA_COLLECTION_NAME=rag_documents

# Documentos
DOCUMENTS_PATH=./data/documents
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=100

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
DEBUG=True
```

### 4. Adicione Seus Documentos

Coloque seus documentos na pasta `data/documents/`:

```bash
# Exemplos de estrutura
data/
├── documents/
│   ├── arquivo1.pdf
│   ├── arquivo2.txt
│   ├── subpasta/
│   │   └── arquivo3.pdf
```

Documentos suportados:
- 📄 PDF (`.pdf`)
- 📝 Texto puro (`.txt`)
- 📊 Word (`.docx`) - com instalação adicional

## 💻 Como Usar

### Opção 1: Interface Streamlit (Recomendado)

```bash
streamlit run src/ui/app.py
```

Abra seu navegador em: http://localhost:8501

**Funcionalidades:**
1. Inicializar o sistema RAG
2. Ingerir documentos
3. Fazer perguntas em tempo real
4. Ver documentos de origem
5. Gerenciar coleções

### Opção 2: API FastAPI

```bash
# Inicie o servidor
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints disponíveis:**

```bash
# Health Check
curl http://localhost:8000/health

# Fazer uma query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "O que é RAG?", "streaming": false}'python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload


# Ingerir documentos
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'

# Obter informações da coleção
curl http://localhost:8000/collection-info

# Limpar coleção
curl -X DELETE http://localhost:8000/collection
```

**Documentação Interativa:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Opção 3: Script Python

```bash
# Execute exemplos
python examples.py
```

**Exemplo de código Python:**

```python
from src.modules import DocumentProcessor, VectorStoreManager, RAGChain
from src.config import settings

# Inicializar
vector_store_manager = VectorStoreManager()
document_processor = DocumentProcessor()
rag_chain = RAGChain(vector_store_manager)

# Processar documentos
documents = document_processor.process_documents()
vector_store_manager.add_documents(documents)

# Reinicializar RAG chain
rag_chain = RAGChain(vector_store_manager)

# Fazer query
result = rag_chain.query("Sua pergunta aqui?")
print("Resposta:", result["answer"])
print("Fontes:", result["sources"])
```

## 📁 Estrutura do Projeto

```
rag_agent/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configurações da aplicação
│   ├── logger.py              # Sistema de logging
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── document_processor.py   # Processamento de documentos
│   │   ├── embeddings.py           # Embeddings com Deepseek
│   │   ├── vector_store.py         # Gerenciamento do ChromaDB
│   │   └── rag_chain.py            # Cadeia RAG principal
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py            # API FastAPI
│   └── ui/
│       ├── __init__.py
│       └── app.py             # Interface Streamlit
├── data/
│   ├── documents/             # Seus documentos aqui
│   └── vector_store/          # Armazenamento de vetores
├── logs/                      # Arquivos de log
├── requirements.txt           # Dependências Python
├── .env.example              # Exemplo de variáveis de ambiente
├── .env                      # Variáveis de ambiente (não commitar)
├── examples.py               # Scripts de exemplo
└── README.md                 # Este arquivo
```

## 🔧 Componentes Principais

### DocumentProcessor
Carrega e processa documentos, dividindo-os em chunks.

```python
from src.modules import DocumentProcessor

processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)
documents = processor.process_documents("path/to/docs")
```

### VectorStoreManager
Gerencia o ChromaDB para armazenamento e recuperação de vetores.

```python
from src.modules import VectorStoreManager

manager = VectorStoreManager()
manager.add_documents(documents)
retriever = manager.get_retriever(search_kwargs={"k": 4})
```

### RAGChain
Combina retrieval e generation para responder perguntas.

```python
from src.modules import RAGChain

rag = RAGChain(vector_store_manager)
result = rag.query("Sua pergunta?")
# result = {"answer": "...", "sources": [...]}
```

## 🌐 Deepseek API

### Obtendo Créditos

1. Acesse https://www.deepseek.com
2. Crie uma conta
3. Obtenha sua API key em Settings
4. Adicione créditos à sua conta

### Modelos Disponíveis

- `deepseek-chat`: Melhor para chat conversacional
- `deepseek-coder`: Otimizado para código
- `deepseek-reasoning`: Para tarefas complexas (premium)

## 📊 Exemplo de Workflow Completo

```python
# 1. Inicializar
from src.modules import DocumentProcessor, VectorStoreManager, RAGChain
from src.config import settings

# 2. Processar documentos
processor = DocumentProcessor()
documents = processor.process_documents()
print(f"Carregados {len(documents)} chunks")

# 3. Criar vector store
store = VectorStoreManager()
store.add_documents(documents)
print("Documentos indexados")

# 4. Criar RAG chain
rag = RAGChain(store)

# 5. Fazer queries
questions = [
    "O que é RAG?",
    "Como funciona o LangChain?",
    "Quais são as vantagens?"
]

for q in questions:
    result = rag.query(q)
    print(f"\nPergunta: {q}")
    print(f"Resposta: {result['answer']}")
    print(f"Fontes: {len(result['sources'])} documentos")
```

## 🐳 Docker

### Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/) e Docker Compose v2
- Arquivo `.env` com `DEEPSEEK_API_KEY` (copie de `.env.example`)

### Subir a aplicação

```bash
# Na raiz do projeto
cp .env.example .env   # se ainda não tiver .env
# Edite .env e defina DEEPSEEK_API_KEY

# Interface Streamlit + API FastAPI
docker compose up --build -d

# Apenas a interface (build da imagem uma vez via serviço api)
docker compose build api && docker compose up -d ui

# Apenas a API
docker compose up --build -d api
```

**URLs:**

| Serviço | URL |
|---------|-----|
| Streamlit (UI) | http://localhost:8501 |
| API | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |

Documentos em `data/documents/` e o índice vetorial em `data/vector_store/` ficam no host (volumes bind). O modelo de embeddings é incluído na imagem no build.

### Comandos úteis

```bash
docker compose logs -f ui      # logs da interface
docker compose logs -f api     # logs da API
docker compose down            # parar e remover containers
```

A primeira subida pode levar alguns minutos (build com PyTorch e sentence-transformers). Recomenda-se **4 GB+ de RAM** para os containers.

## 🐛 Troubleshooting

### "API key not provided"
- Verifique se a variável `DEEPSEEK_API_KEY` está configurada em `.env`
- Copie seu arquivo `.env.example` para `.env` e adicione sua chave

### "No documents found"
- Verifique se há documentos em `data/documents/`
- Formatos suportados: PDF, TXT
- Verifique as permissões de leitura

### "Vector store not found"
- A primeira vez que você executa, ele cria automaticamente
- Se problemas persistirem, delete `data/vector_store/` e reinicie

### Erros de conexão com Deepseek
- Verifique sua conexão com a internet
- Confirme que a API key é válida
- Verifique se você tem créditos suficientes na conta

### Docker: build lento ou container reinicia
- O build baixa PyTorch e o modelo de embeddings; aguarde a conclusão
- Confirme que `.env` existe e contém `DEEPSEEK_API_KEY`
- Verifique logs: `docker compose logs api` ou `docker compose logs ui`
- Em máquinas com pouca RAM, suba só a UI: `docker compose build api && docker compose up -d ui`
- Erro `rag-agent:latest: already exists` ao buildar: remova builds duplicados (só `api` faz build) e rode `docker compose build api`

## 📚 Recursos Adicionais

- [LangChain Documentation](https://python.langchain.com/)
- [Deepseek API Docs](https://www.deepseek.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://docs.streamlit.io/)

## 🤝 Contribuindo

Contribuições são bem-vindas! Para mudanças significativas, abra uma issue primeiro para discutir.

## 📝 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo LICENSE para detalhes.

## 📧 Suporte

Para suporte e dúvidas, crie uma issue ou entre em contato.

---

**Versão:** 1.0.0  
**Última atualização:** Maio 2026  
**Mantido por:** RAG Agent Team
