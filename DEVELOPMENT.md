# Guia de Desenvolvimento - RAG Agent

## 🎯 Estendendo o Projeto

Este guia mostra como adicionar novas funcionalidades ao RAG Agent.

## 📚 Adicionando Novos Document Loaders

Para suportar novos formatos de documento (CSV, Word, etc):

```python
# src/modules/document_processor.py

from langchain_community.document_loaders import CSVLoader

class DocumentProcessor:
    def load_documents(self, documents_path: str = None) -> List[Document]:
        # ... código existente ...
        
        # Adicione novo loader
        csv_loader = DirectoryLoader(
            str(docs_path),
            glob="**/*.csv",
            loader_cls=CSVLoader,
        )
        try:
            csv_docs = csv_loader.load()
            documents.extend(csv_docs)
            logger.info(f"Loaded {len(csv_docs)} CSV documents")
        except Exception as e:
            logger.warning(f"Error loading CSV documents: {e}")
        
        return documents
```

## 🧠 Usando Diferentes LLMs

Para trocar para outro LLM (OpenAI, Anthropic, etc):

```python
# src/modules/rag_chain.py

from langchain.llms import OpenAI, Anthropic

def create_llm(provider: str = "deepseek"):
    """Factory para criar LLMs"""
    
    if provider == "openai":
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo"
        )
    elif provider == "anthropic":
        return Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    else:
        return DeepseekLLM()

# Uso
rag = RAGChain(vector_store_manager, llm=create_llm("openai"))
```

## 🎨 Customizando Prompts

Para criar prompts especializados:

```python
# src/modules/prompts.py

from langchain.prompts import PromptTemplate

# Prompt para Suporte ao Cliente
SUPPORT_PROMPT = PromptTemplate(
    template="""Você é um agente de suporte ao cliente prestativo.
    
Context:
{context}

Customer Question: {question}

Response:""",
    input_variables=["context", "question"]
)

# Prompt para Análise Técnica
TECHNICAL_PROMPT = PromptTemplate(
    template="""Você é um especialista técnico analisando documentação.
    
Documentation:
{context}

Technical Question: {question}

Detailed Analysis:""",
    input_variables=["context", "question"]
)

# Prompt para Resumo
SUMMARY_PROMPT = PromptTemplate(
    template="""Resuma o seguinte conteúdo em 3 pontos principais:

Content:
{context}

Summary:""",
    input_variables=["context"]
)
```

## 💾 Integrando Novos Vector Stores

Para usar FAISS em vez de ChromaDB:

```python
# src/modules/vector_store.py

from langchain.vectorstores import FAISS

class FAISSVectorStoreManager:
    def __init__(self, embeddings, persist_directory: str = None):
        self.embeddings = embeddings
        self.persist_directory = persist_directory or "./data/vector_store"
        self.index = None

    def add_documents(self, documents: List[Document]) -> None:
        if self.index is None:
            self.index = FAISS.from_documents(
                documents,
                self.embeddings
            )
        else:
            self.index.add_documents(documents)
        
        self.index.save_local(self.persist_directory)

    def get_retriever(self, search_kwargs: dict = None):
        if self.index is None:
            self.index = FAISS.load_local(
                self.persist_directory,
                self.embeddings
            )
        return self.index.as_retriever(search_kwargs=search_kwargs)
```

## 🔄 Adicionando Cache de Queries

Para melhorar performance com queries repetidas:

```python
# src/modules/cache.py

import hashlib
from functools import wraps

class QueryCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds
        import time
        self.time = time.time

    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str):
        key = self._hash_query(query)
        if key in self.cache:
            entry, timestamp = self.cache[key]
            if self.time() - timestamp < self.ttl:
                return entry
            else:
                del self.cache[key]
        return None

    def set(self, query: str, result):
        key = self._hash_query(query)
        self.cache[key] = (result, self.time())

# Uso
cache = QueryCache()

def query_with_cache(question: str):
    cached = cache.get(question)
    if cached:
        return cached
    
    result = rag_chain.query(question)
    cache.set(question, result)
    return result
```

## 📊 Adicionando Monitoramento e Métricas

```python
# src/modules/monitoring.py

from datetime import datetime
from typing import Dict, List

class QueryMonitor:
    def __init__(self):
        self.queries: List[Dict] = []

    def log_query(self, question: str, answer: str, 
                  sources_count: int, response_time: float):
        self.queries.append({
            "timestamp": datetime.now(),
            "question": question,
            "answer_length": len(answer),
            "sources": sources_count,
            "response_time": response_time,
        })

    def get_statistics(self) -> Dict:
        if not self.queries:
            return {}
        
        return {
            "total_queries": len(self.queries),
            "avg_response_time": sum(q["response_time"] for q in self.queries) / len(self.queries),
            "avg_sources": sum(q["sources"] for q in self.queries) / len(self.queries),
        }

# Uso
monitor = QueryMonitor()

@timeit
def query_with_monitoring(question: str):
    import time
    start = time.time()
    result = rag_chain.query(question)
    elapsed = time.time() - start
    
    monitor.log_query(
        question,
        result["answer"],
        len(result["sources"]),
        elapsed
    )
    
    return result
```

## 🧪 Implementando Testes Customizados

```python
# tests/test_custom.py

import pytest
from src.modules import RAGChain, VectorStoreManager

class TestCustomRAG:
    @pytest.fixture
    def rag_chain(self):
        manager = VectorStoreManager(collection_name="test")
        return RAGChain(manager)
    
    def test_query_response_format(self, rag_chain):
        result = rag_chain.query("Test question?")
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
    
    def test_empty_query(self, rag_chain):
        with pytest.raises(ValueError):
            rag_chain.query("")
    
    def test_batch_processing(self, rag_chain):
        questions = ["Q1?", "Q2?", "Q3?"]
        results = rag_chain.batch_query(questions)
        assert len(results) == 3
```

## 🚀 Implantando em Produção

### Usando Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build e run:
```bash
docker build -t rag-agent .
docker run -e DEEPSEEK_API_KEY=your_key -p 8000:8000 rag-agent
```

### Usando Gunicorn (Para produção)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 src.api.main:app
```

## 📈 Otimizações de Performance

### 1. Batch Processing de Documents
```python
# Processar documentos em lotes
def batch_add_documents(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        vector_store.add_documents(batch)
        logger.info(f"Added batch {i//batch_size + 1}")
```

### 2. Reranking de Resultados
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/mmarco-MiniLMv2-L12-H384-v1')

def rerank_results(query, retrieved_docs, top_k=3):
    pairs = [[query, doc.page_content] for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
```

### 3. Índices Otimizados
```python
# Para ChromaDB com índices melhores
vector_store = Chroma(
    collection_name="optimized",
    embedding_function=embeddings,
    persist_directory="./data/vector_store",
    collection_metadata={"hnsw:space": "cosine"}  # Usa métrica cosine
)
```

## 🔐 Segurança

### Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: QueryRequest):
    # ...
```

### Autenticação
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401)
    return credentials.credentials

@app.post("/query")
async def query(request: QueryRequest, token = Depends(verify_token)):
    # ...
```

## 📚 Recursos

- [LangChain Advanced](https://python.langchain.com/docs/advanced/)
- [ChromaDB Advanced](https://docs.trychroma.com/)
- [FastAPI Production](https://fastapi.tiangolo.com/deployment/)

---

Para mais informações, consulte o README.md principal.
