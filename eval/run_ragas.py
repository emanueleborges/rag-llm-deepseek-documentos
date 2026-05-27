#!/usr/bin/env python3
"""
Avaliação RAGAS do pipeline CDC.

Métricas: faithfulness, answer_relevancy, context_recall.

Uso:
  pip install -r requirements.txt
  python eval/run_ragas.py
  python eval/run_ragas.py --dataset eval/datasets/cdc_baseline.jsonl --limit 3
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET = PROJECT_ROOT / "eval" / "datasets" / "cdc_baseline.jsonl"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"


def load_dataset(path: Path, limit: int | None = None) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def run_pipeline(dataset_path: Path, limit: int | None, skip_ragas: bool) -> dict:
    from src.config import settings
    from src.logger import logger
    from src.modules.document_processor import DocumentProcessor
    from src.modules.rag_chain import RAGChain
    from src.modules.vector_store import VectorStoreManager
    from src.indexing import needs_reindex, reindex_documents

    settings.create_directories()
    items = load_dataset(dataset_path, limit=limit)
    if not items:
        raise SystemExit(f"Dataset vazio: {dataset_path}")

    logger.info("Carregando RAG para avaliação (%d perguntas)...", len(items))
    vsm = VectorStoreManager()
    if needs_reindex():
        logger.info("Reindexando documentos antes da avaliação...")
        reindex_documents(vsm, DocumentProcessor())

    rag = RAGChain(vsm)

    eval_rows = []
    raw_results = []

    for i, item in enumerate(items, 1):
        question = item["question"]
        logger.info("[%d/%d] %s", i, len(items), question[:80])
        result = rag.query(question)
        contexts = [s.get("content", "") for s in result.get("sources", [])]
        row = {
            "question": question,
            "answer": result.get("answer", ""),
            "contexts": contexts,
            "ground_truth": item.get("ground_truth", ""),
        }
        eval_rows.append(row)
        raw_results.append(
            {
                "id": item.get("id"),
                "tags": item.get("tags", []),
                "question": question,
                "answer": result.get("answer", ""),
                "meta": result.get("meta", {}),
                "sources_count": len(contexts),
            }
        )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "samples": len(eval_rows),
        "ragas": None,
        "raw": raw_results,
    }

    if skip_ragas:
        logger.info("RAGAS ignorado (--skip-ragas)")
        return report

    try:
        from datasets import Dataset
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_recall, faithfulness
    except ImportError as exc:
        raise SystemExit(
            "Dependências de avaliação ausentes. Execute:\n"
            "  pip install -r requirements.txt\n"
            f"Detalhe: {exc}"
        ) from exc

    if not settings.deepseek_api_key.strip() and settings.llm_provider.lower() != "ollama":
        raise SystemExit("DEEPSEEK_API_KEY necessária para métricas RAGAS (LLM juiz).")

    logger.info("Executando RAGAS (faithfulness, answer_relevancy, context_recall)...")

    if settings.llm_provider.lower() == "ollama":
        from langchain_ollama import ChatOllama

        evaluator_llm = LangchainLLMWrapper(
            ChatOllama(
                model=settings.ollama_llm_model,
                temperature=0,
                base_url=settings.ollama_base_url,
            )
        )
    else:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                base_url=f"{settings.deepseek_base_url.rstrip('/')}/v1",
                temperature=0,
            )
        )

    if settings.embeddings_provider.lower() == "ollama":
        from langchain_ollama import OllamaEmbeddings

        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(
                model=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url,
            )
        )
    else:
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=settings.embedding_model)
        )

    ds = Dataset.from_list(eval_rows)
    scores = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    metrics_dict = scores.to_pandas().mean(numeric_only=True).to_dict()
    report["ragas"] = {k: round(float(v), 4) for k, v in metrics_dict.items()}
    logger.info("RAGAS scores: %s", report["ragas"])
    return report


def save_report(report: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"ragas_{ts}.json"
    md_path = REPORTS_DIR / f"ragas_{ts}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Relatório RAGAS — CDC",
        "",
        f"- **Data:** {report['timestamp']}",
        f"- **Dataset:** `{report['dataset']}`",
        f"- **Amostras:** {report['samples']}",
        "",
    ]
    if report.get("ragas"):
        lines.append("## Métricas")
        lines.append("")
        lines.append("| Métrica | Score |")
        lines.append("|---------|-------|")
        for name, score in report["ragas"].items():
            lines.append(f"| {name} | {score:.4f} |")
        lines.append("")
    else:
        lines.append("_RAGAS não executado._")
        lines.append("")

    lines.append("## Amostras")
    lines.append("")
    for item in report.get("raw", []):
        mode = (item.get("meta") or {}).get("mode", "?")
        lines.append(f"### {item.get('id', '?')} — `{mode}`")
        lines.append("")
        lines.append(f"**P:** {item['question']}")
        lines.append("")
        lines.append(f"**Trechos:** {item.get('sources_count', 0)}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path


def main():
    parser = argparse.ArgumentParser(description="Avaliação RAGAS do RAG CDC")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Caminho do JSONL com question + ground_truth",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limitar N perguntas")
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Só executar pipeline e salvar respostas (sem métricas)",
    )
    args = parser.parse_args()

    report = run_pipeline(args.dataset, args.limit, args.skip_ragas)
    out = save_report(report)
    print(f"Relatório salvo em: {out}")


if __name__ == "__main__":
    main()
