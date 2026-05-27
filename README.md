# RAG Agent - Retrieval-Augmented Generation System

## 📋 Descrição

Aplicação de Perguntas e Respostas (Q&A) usando **Retrieval-Augmented Generation (RAG)**, **LangChain**, **ChromaDB** e provedores de LLM como **Ollama** e **Deepseek**.

## 🚀 Instruções de uso

### 1. Clone do repositório

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd rag-llm-deepseek-documentos
```

### 2. Crie e ative o ambiente virtual

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instale dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o arquivo `.env` e ajuste os valores de acordo com seu ambiente.

### 5. Adicione documentos

Coloque seus arquivos em `data/documents/`:

```bash
mkdir -p data/documents
cp /caminho/para/seu/documento.pdf data/documents/
```

Documentos normalmente suportados:
- PDF (`.pdf`)
- Texto (`.txt`)
- Outros formatos via parser adicional

## ▶️ Executando a aplicação

### Iniciar a API FastAPI

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Executar a interface Streamlit

```bash
streamlit run src/ui/app.py
```

### Rodar com Docker

```bash
docker compose up --build
```

## 📡 Endpoints da API

### Health check

```http
GET http://127.0.0.1:8000/health
```

### Consulta RAG

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "O que é RAG?", "conversation_context": "", "streaming": false}'
```

### Ingerir documentos

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents_path": "./data/documents"}'
```

### Enviar feedback

```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"rating": 1, "question": "Qual é a capital do Brasil?", "answer": "Brasília", "session_id": "sess1", "message_id": "msg1", "meta": {}}'
```

### Listar feedback

```http
GET http://127.0.0.1:8000/feedback?limit=20
```

### WebSocket de chat

```http
ws://127.0.0.1:8000/ws/chat
```

Envie JSON com `type` igual a `query` ou `feedback`.

Exemplo de query WebSocket:

```json
{
  "type": "query",
  "question": "O que é RAG?",
  "conversation_context": ""
}
```

## 🧩 Dependências extras

Se algum módulo estiver faltando, instale as dependências adicionais:

```bash
pip install langgraph chainlit langfuse ragas sentence-transformers pymupdf pytest black
```

> Se aparecer `No module named 'langgraph'`, instale `langgraph` manualmente ou use o `requirements-docker.txt` para um runtime mais completo.

## 📝 Observações

- Use `requirements.txt` para o setup local básico e completo.
- Configure `.env` antes de iniciar a API.
- Ajuste `API_HOST` e `API_PORT` no `.env` se necessário.
