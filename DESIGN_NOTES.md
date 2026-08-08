# Design Notes — Pakistan Tech Career &amp; HR Voice Assistant

Answers to the conceptual questions your mentor expects you to justify,
covering both the ingestion stage and the retrieval/generation stage now
that both are built.

## 1. Why RAG, and what are the alternatives?

Three realistic ways to get an LLM to answer questions about your own
documents:

- **Fine-tuning an LLM on the documents.** Expensive, slow to update
  (any policy change means re-training), and prone to the model
  "memorizing" facts imprecisely rather than citing them exactly — bad
  for something like an exact leave-day count.
- **Long-context stuffing** (paste all documents into every prompt).
  Works only while the corpus is small; cost and latency scale with
  corpus size regardless of what's actually relevant to the question.
- **RAG.** Only the relevant chunks are pulled in per query, so
  cost/latency stay flat as the corpus grows, updates are instant (edit
  the source, re-embed, done), and the answer can be traced back to a
  specific chunk — which is what "grounded, not hallucinated" means in
  practice.

**Use case fit:** this project needs exactly-correct answers (leave day
counts, interview-question specifics) from a corpus that will keep
growing (more roles, more policy documents). RAG stays accurate and
cheap as that happens; fine-tuning would need to be redone every time a
policy changes.

## 2. Delete / update handling

Every chunk's metadata stores `source_file` at ingestion time.
`vector_store.delete_by_source()` deletes every vector matching that
filename in one call — the whole PDF disappears from the index without
tracking individual chunk IDs by hand.

To update a document: re-run ingestion on the new version. Chunk IDs are
deterministic (`sha256(source_file + chunk_index)`), so re-ingesting the
same filename overwrites the old vectors at those IDs rather than
duplicating them. If the new version has fewer chunks, run
`delete_by_source()` first to avoid orphaned old chunks past the new
chunk count.

## 3. Embedding model choice and why dimension matters

**Model:** `embed-v4.0`, called via the Cohere API (not loaded locally — see
point 5). Originally planned as Hugging Face's BGE, switched to Cohere
because their Trial API key requires no credit card (confirmed
independently) and covers both embeddings and reranking in one provider.

**Why this model:** the Trial key needs no credit card and is confirmed
free across independent sources, and it has a native `input_type`
parameter (`"search_query"` vs `"search_document"`) that handles the same
asymmetric-search optimization BGE needed a manual prefix string for —
Cohere does it natively.

**Why dimension matters:** embed-v4.0 is a Matryoshka model supporting
256/512/1024/1536-dim output via the `output_dimension` request
parameter; this project uses 1024. Pinecone's index has a fixed
`dimension` that every vector must match exactly — a mismatch fails
every upsert outright, not silently degrades. This also locks in the
embedding model/dimension choice once real data is ingested, since an
existing index's dimension can't be changed in place; switching means
recreating the index.

## 4. No local storage

Nothing in this pipeline writes vectors, embeddings, or extracted text to
local disk at any stage. PDFs are read directly at ingestion time; text
lives in memory only for that run; vectors go straight to Pinecone
(fully managed, hosted). The deployed API (Vercel) never loads a model
or stores anything itself — every call (embedding, reranking,
generation) goes out to a hosted API and comes back with a result. This
satisfies the constraint by construction, not as an afterthought.

## 5. Semantic search, reranking, and the retrieval algorithm

**Stage 1 — approximate search:** the query is embedded with Cohere
embed-v4.0, then Pinecone returns the `top_k` nearest chunks by
**cosine similarity** — the standard metric for sentence embeddings
since it measures direction (semantic similarity) rather than
magnitude, which isn't meaningful for relevance here.

**Stage 2 — precise reranking:** Cohere's rerank-v3.5 re-scores those
`top_k` candidates by reading the query and each passage together, catching interactions that two independently-computed
embeddings miss. This is more accurate but too slow to run over the
whole corpus — hence retrieve-then-rerank rather than reranking
everything.

**No keyword-based category routing.** Rather than guessing which
knowledge area (hr_policy, frontend, backend, etc.) a query belongs to
before searching, the whole index is searched with no filter and the
reranker's relevance scoring sorts out what's actually useful. A query
like "what should I expect for a backend role" and "how many leaves do
backend employees get" both mention "backend" but need different
categories — semantic reranking handles this more reliably than keyword
matching would.

**Grounding check:** if nothing retrieved clears a minimum reranker score
(`GROUNDING_THRESHOLD` in `retrieval.py`), the query is treated as
ungrounded and the LLM is told explicitly that no relevant information
was found — this is what produces the "I don't have information on
that" refusal instead of a hallucinated guess.

## 6. Temperature

Set low (`0.15` in `config.py`) because this is a factual, grounded-answer
assistant, not a creative one. Higher temperature increases the chance
the model paraphrases loosely or adds small unstated details — exactly
wrong when the honest answer to "how many casual leaves do I get" is a
specific number. Not set to `0.0` because a little variability keeps
phrasing natural across similar queries for a voice response, rather
than robotically identical every time.

## 7. Checking quality and improving it

Planned approach: build a small labeled test set — a list of
(question, expected source file, expected key fact) triples covering all
7 document categories, including a few deliberately out-of-scope
questions (like the maternity-leave example used earlier) to test the
refusal path. Run each question through the pipeline and check two
things separately: **retrieval accuracy** (did the expected source file
end up in the reranked top results?) and **answer groundedness** (does
the generated answer's claim actually appear in the retrieved text, or
did the model add something not present?). If retrieval is failing,
the fix is usually chunking/embedding-side (chunk size, overlap); if
retrieval is fine but answers still drift, the fix is prompt-side
(tightening the grounding instruction) or lowering temperature further.

## 8. Hosting: why Vercel, and how the pieces fit together

Originally ChromaDB (local file storage) ruled out Vercel's stateless
serverless functions. Switching to Pinecone (fully hosted, no local
storage) removed that blocker entirely — nothing in this stack loads a
model or persists data on the compute instance itself; every step
(embedding, reranking, vector search, generation) is an outbound API
call. That fits Vercel's request-response serverless model cleanly.
The one constraint to watch: Vercel's function execution timeout (10s
on the free Hobby tier) — the full retrieve → rerank → generate chain
needs to comfortably finish inside that window.

## 9. How Vapi connects to this backend

Vapi handles STT (caller's speech → text) and TTS (our reply → spoken
audio) itself. In between, it calls our deployed endpoint
(`/chat/completions`) as an OpenAI-compatible "Custom LLM" provider,
sending the conversation so far in standard OpenAI chat-completions
format. Our endpoint runs retrieval + reranking + generation and returns
a standard OpenAI-shaped response, which Vapi converts back to speech.
This is Vapi's documented, officially supported integration pattern —
no custom protocol needed on either side.