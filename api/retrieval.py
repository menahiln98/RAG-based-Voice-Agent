"""
Ties embedding + Pinecone search + reranking into one retrieval step.

Design choice: we do NOT try to pre-classify which category (hr_policy,
frontend, backend, etc.) a query belongs to with keyword matching before
searching. That's brittle -- "what should I expect for a backend role"
and "how many leaves do backend employees get" both mention "backend"
but need different categories. Instead we search the WHOLE index
(no category filter) with a slightly larger top_k, and let the
reranker's relevance scoring sort out which chunks are actually most
relevant to the query -- semantic relevance naturally handles this
better than keyword routing would.
"""
from embeddings import embed_query
from vector_store import semantic_search
from reranker import rerank
from config import TOP_K

# Minimum Cohere relevance score (0-1 range) to trust a chunk as
# "actually relevant" rather than just "the least-irrelevant of a bad
# batch". Cohere's rerank scores tend to sit lower than raw cosine
# similarity even for genuinely relevant matches, so this threshold is
# deliberately conservative -- tune it once you have a real test set
# (see DESIGN_NOTES.md section 7). It's what makes the "I don't have
# information on that" refusal behavior work.
GROUNDING_THRESHOLD = 0.10


def retrieve(query: str) -> list[dict]:
    query_vector = embed_query(query)
    candidates = semantic_search(query_vector, top_k=TOP_K)

    # Reshape Pinecone's match objects into the flat {"text": ..., ...}
    # shape the reranker expects.
    flat_candidates = [
        {
            "text": m["metadata"]["text"],
            "source_file": m["metadata"]["source_file"],
            "category": m["metadata"]["category"],
            "similarity_score": m["score"],
        }
        for m in candidates
    ]

    reranked = rerank(query, flat_candidates)
    return reranked


def retrieve_grounded(query: str) -> tuple[list[dict], bool]:
    """
    Returns (chunks, is_grounded). is_grounded is False when nothing
    retrieved clears GROUNDING_THRESHOLD -- the caller should then have
    the LLM refuse rather than answer from its own general knowledge.
    """
    chunks = retrieve(query)
    is_grounded = any(c["rerank_score"] >= GROUNDING_THRESHOLD for c in chunks)
    return chunks, is_grounded