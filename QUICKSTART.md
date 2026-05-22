# Quick Start Script para RAG Agent

## Windows (PowerShell)

```powershell
# 1. Criar ambiente virtual
python -m venv venv

# 2. Ativar ambiente virtual
.\venv\Scripts\Activate.ps1

# 3. Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# 4. Criar arquivo .env
Copy-Item .env.example .env

# 5. Editar .env com sua API key do Deepseek
# Use seu editor favorito para abrir .env
notepad .env

# 6. Testar instalação
python -c "from src.modules import RAGChain; print('✓ Installation successful!')"

# 7. Executar a interface Streamlit
streamlit run src/ui/app.py
```

## Linux/Mac (Bash)

```bash
# 1. Criar ambiente virtual
python3 -m venv venv

# 2. Ativar ambiente virtual
source venv/bin/activate

# 3. Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# 4. Criar arquivo .env
cp .env.example .env

# 5. Editar .env com sua API key do Deepseek
nano .env  # ou vim, ou seu editor favorito

# 6. Testar instalação
python -c "from src.modules import RAGChain; print('✓ Installation successful!')"

# 7. Executar a interface Streamlit
streamlit run src/ui/app.py
```

## Próximos Passos

1. **Adicione seus documentos**: Coloque arquivos PDF ou TXT em `data/documents/`

2. **Abra a interface**: O Streamlit abrirá automaticamente em `http://localhost:8501`

3. **Clique em "Inicializar Sistema RAG"**

4. **Clique em "Ingerir Documentos"** para indexar seus arquivos

5. **Faça suas perguntas** no chat

## Troubleshooting

**"Module not found" error?**
```bash
# Certifique-se de estar no diretório correto
cd rag_agent

# Verifique se o ambiente virtual está ativo
# Windows: procure (venv) no prompt
# Linux/Mac: procure (venv) no prompt

# Reinstale as dependências
pip install -r requirements.txt
```

**"DEEPSEEK_API_KEY not found"?**
```bash
# Abra seu arquivo .env
nano .env  # Linux/Mac
notepad .env  # Windows

# Adicione sua chave:
# DEEPSEEK_API_KEY=sk_xxxxxxxxxxxxx
```

**Porta 8501 já em uso?**
```bash
# Use uma porta diferente
streamlit run src/ui/app.py --server.port 8502
```

## Comando Rápido (Todos de uma vez)

### Windows PowerShell
```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt; Copy-Item .env.example .env; streamlit run src/ui/app.py
```

### Linux/Mac Bash
```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cp .env.example .env && streamlit run src/ui/app.py
```

## Modo API (Alternativa ao Streamlit)

Se preferir usar a API REST:

```bash
# 1. Ativar ambiente virtual (se ainda não estiver ativo)
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\Activate.ps1  # Windows

# 2. Iniciar servidor FastAPI
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Acessar documentação
# Abra http://localhost:8000/docs no navegador

# 4. Fazer uma query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Sua pergunta?", "streaming": false}'
```

## Modo Script Python

```bash
# 1. Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\Activate.ps1  # Windows

# 2. Executar exemplos
python examples.py

# 3. Usar em seu próprio script
python
>>> from src.modules import RAGChain, VectorStoreManager
>>> manager = VectorStoreManager()
>>> rag = RAGChain(manager)
>>> result = rag.query("Sua pergunta?")
>>> print(result["answer"])
```

## Estrutura de Diretórios Criada

Depois de executar o setup, você terá:

```
rag_agent/
├── src/                    # Código fonte
├── data/
│   ├── documents/         # Seus documentos aqui
│   └── vector_store/      # Base de dados vetorial
├── logs/                  # Arquivos de log
├── venv/                  # Ambiente virtual
├── .env                   # Suas configurações
├── requirements.txt       # Dependências
├── README.md             # Documentação
├── DEVELOPMENT.md        # Guia de desenvolvimento
└── examples.py           # Scripts de exemplo
```

## Suporte

Algum problema? Verifique:
- [README.md](README.md) - Documentação completa
- [DEVELOPMENT.md](DEVELOPMENT.md) - Guia de desenvolvimento
- Logs em `logs/rag_agent.log`

---

**Primeira execução?** Siga o passo a passo acima. É tudo que você precisa! 🚀
