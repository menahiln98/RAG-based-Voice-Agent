"""
Pinecone wrapper: index creation, upsert, semantic search, and
delete-by-source (so a whole source PDF can be removed in one call, per
the mentor's requirement).

Why Pinecone over self-managing a local index:
  - Fully managed, no server process for us to run alongside the
    FastAPI app on the free Koyeb instance -- avoids the RAM contention
    problem entirely.
  - No local storage involved at any point (satisfies the "no local
    storage" constraint outright, since it was never an option here).
"""
from pinecone import Pinecone, ServerlessSpec

from config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION,
    EMBEDDING_DIMENSION, TOP_K,
)

_pc = None


def _get_client():
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def get_or_create_index():
    pc = _get_client()
    existing = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,   # MUST match the embedding model's output dim exactly
            metric="cosine",                 # cosine similarity: standard choice for normalized
                                              # sentence-embedding models like BGE -- it measures
                                              # directional similarity and ignores magnitude, which
                                              # matters more than raw distance for semantic search.
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(PINECONE_INDEX_NAME)


def upsert_chunks(vectors: list[dict]):
    """
    vectors: list of dicts shaped like
      {"id": str, "values": list[float], "metadata": {...}}
    metadata should include at minimum: source_file, category, text
    """
    index = get_or_create_index()
    # Pinecone recommends batching upserts; 100 per call is a safe default
    # for the free tier's write-unit limits.
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])


def semantic_search(query_vector: list[float], top_k: int = TOP_K, category: str | None = None):
    """
    Returns Pinecone's top_k nearest neighbors by cosine similarity.
    Optionally filters by `category` metadata (e.g. "backend", "hr_policy")
    so a query can be scoped to one knowledge area when the router decides
    that's appropriate.
    """
    index = get_or_create_index()
    query_kwargs = {"vector": query_vector, "top_k": top_k, "include_metadata": True}
    if category:
        query_kwargs["filter"] = {"category": {"$eq": category}}
    result = index.query(**query_kwargs)
    return result.get("matches", [])


def delete_by_source(source_file: str):
    """
    Deletes every vector belonging to a given source file in one call --
    this is how "delete the whole PDF" is implemented: source_file is
    stored as metadata on every chunk at ingestion time, so we can filter
    and delete by it directly rather than tracking individual chunk IDs.
    """
    index = get_or_create_index()
    index.delete(filter={"source_file": {"$eq": source_file}})


def list_sources() -> set[str]:
    """
    Pinecone doesn't offer a native 'list distinct metadata values' call,
    so this does a lightweight dummy query with a large top_k and dedupes
    source_file values client-side. Fine at this project's small scale;
    would need a separate metadata index (e.g. a small manifest) if the
    corpus grew much larger.
    """
    index = get_or_create_index()
    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", 0)
    if total == 0:
        return set()
    dummy_vector = [0.0] * EMBEDDING_DIMENSION
    matches = index.query(vector=dummy_vector, top_k=min(total, 10000), include_metadata=True)
    return {m["metadata"]["source_file"] for m in matches.get("matches", []) if m.get("metadata")}