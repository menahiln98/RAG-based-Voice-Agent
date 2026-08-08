"""
Cross-encoder reranking via the Cohere Rerank API (same provider as
embeddings.py -- no separate Hugging Face dependency needed).

Why rerank at all, given we already did semantic search: embedding
similarity (what Pinecone used) is fast but approximate -- it compares
two independently-computed vectors. A cross-encoder-style reranker
instead reads the query and each candidate passage together, so it can
pick up on interactions between them that two separate embeddings miss.
It's more accurate but too slow to run over an entire corpus -- hence
the two-stage design: cheap approximate search narrows thousands of
chunks down to a handful (TOP_K), then the accurate-but-slower reranker
picks the best few of *those* (TOP_K_RERANKED).
"""
import requests

from config import COHERE_API_KEY, COHERE_RERANK_MODEL, TOP_K_RERANKED

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"

_HEADERS = {
    "Authorization": f"Bearer {COHERE_API_KEY}",
    "Content-Type": "application/json",
}


def rerank(query: str, candidates: list[dict], top_n: int = TOP_K_RERANKED) -> list[dict]:
    """
    candidates: list of dicts with at least a "text" key (as returned by
    vector_store.semantic_search's metadata).
    Returns the same dicts, re-sorted by Cohere's relevance score,
    truncated to top_n, each with a "rerank_score" key added.
    """
    if not candidates:
        return []

    payload = {
        "model": COHERE_RERANK_MODEL,
        "query": query,
        "documents": [c["text"] for c in candidates],
        "top_n": top_n,
    }
    response = requests.post(COHERE_RERANK_URL, headers=_HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    results = response.json()["results"]

    # results give back {"index": original_position, "relevance_score": float},
    # already sorted most-to-least relevant -- map back to the original dicts.
    ranked = []
    for r in results:
        candidate = candidates[r["index"]]
        candidate["rerank_score"] = r["relevance_score"]
        ranked.append(candidate)

    return ranked