"""
Generates embeddings via the Cohere API -- no model weights are
downloaded or loaded into local process memory, and no card is required
for the Trial key (unlike Hugging Face Spaces' Docker/compute tier,
which does require one -- this is a different product entirely).

Cohere's embed-v4.0 is used because:
  - The Trial API key is genuinely free with no credit card, confirmed
    across independent sources -- and it also covers reranking, so one
    provider handles both steps instead of two.
  - It distinguishes query embeddings from document embeddings via the
    `input_type` parameter ("search_query" vs "search_document"), which
    is Cohere's equivalent of BGE's manual instruction-prefix convention
    -- same benefit, handled natively by the API instead of by us.
  - Matryoshka-style output lets us pick the embedding dimension
    (256/512/1024/1536) explicitly; we use 1024 as a quality/storage
    balance -- see config.py for the full justification.

Rate limit note: the Trial key allows 1,000 calls/month shared across
Cohere's Chat, Embed, and Rerank endpoints combined (20 requests/minute).
Fine for development and demoing; not intended for production traffic.
"""
import requests

from config import COHERE_API_KEY, COHERE_EMBED_MODEL, EMBEDDING_DIMENSION

COHERE_EMBED_URL = "https://api.cohere.com/v2/embed"

_HEADERS = {
    "Authorization": f"Bearer {COHERE_API_KEY}",
    "Content-Type": "application/json",
}


def _call_cohere_embed(texts: list[str], input_type: str) -> list[list[float]]:
    payload = {
        "model": COHERE_EMBED_MODEL,
        "texts": texts,
        "input_type": input_type,
        "embedding_types": ["float"],
        "output_dimension": EMBEDDING_DIMENSION,
    }
    response = requests.post(COHERE_EMBED_URL, headers=_HEADERS, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["embeddings"]["float"]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document chunks at ingestion time."""
    return _call_cohere_embed(texts, input_type="search_document")


def embed_query(query: str) -> list[float]:
    """Embed a single user query. Cohere's `search_query` input_type
    handles the asymmetric-search optimization natively -- no manual
    prefix string needed, unlike BGE."""
    return _call_cohere_embed([query], input_type="search_query")[0]