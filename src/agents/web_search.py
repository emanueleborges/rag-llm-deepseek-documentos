"""Busca web opcional via Tavily (quando CDC indexado não cobre)."""
from typing import List

import httpx

from src.config import settings
from src.logger import logger


def search_web(query: str, max_results: int = 3) -> str:
    api_key = (settings.tavily_api_key or "").strip()
    if not api_key:
        return ""

    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        data = response.json()
        parts: List[str] = []
        for item in data.get("results", [])[:max_results]:
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")
            if content:
                parts.append(f"- **{title}** ({url})\n{content[:500]}")
        return "\n\n".join(parts)
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return ""
