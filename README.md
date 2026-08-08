# Pakistan Tech Career &amp; HR Voice Assistant

A real-time voice-based RAG assistant that answers two kinds of questions: company HR policy queries (leave, benefits, notice period, performance reviews, etc.) and Pakistan tech-industry interview preparation across five roles — Frontend, Backend, Project Manager, DevOps, and AI/ML Engineer — including a shared HR/behavioral interview track.

Built as a real-time voice pipeline using [Vapi.ai](https://vapi.ai) for speech I/O, with a fully from-scratch Retrieval-Augmented Generation backend (no LangChain) grounding every answer in a curated document set rather than the LLM's own general knowledge.

---

## How It Works

```
User speaks
    │
    ▼
Vapi.ai  ──  Speech-to-Text
    │
    │  POST /chat/completions (OpenAI-compatible)
    ▼
FastAPI backend (Vercel)
    │
    ├── Embed query          →  Cohere embed-v4.0
    ├── Semantic search      →  Pinecone (cosine similarity, top-k)
    ├── Rerank candidates    →  Cohere rerank-v3.5
    ├── Grounding check      →  refuse if nothing clears the relevance threshold
    └── Generate answer      →  Groq (openai/gpt-oss-120b), low temperature
    │
    │  OpenAI-compatible chat completion response
    ▼
Vapi.ai  ──  Text-to-Speech
    │
    ▼
User hears the spoken answer
```

If a question falls outside the document set (e.g. a policy that was never written into the corpus), the assistant says so directly instead of guessing — this grounding boundary is a deliberate design choice, not a limitation to work around.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Voice (STT/TTS/streaming) | [Vapi.ai](https://vapi.ai) | Real-time speech input/output |
| Backend | Python, FastAPI | OpenAI-compatible Custom LLM endpoint |
| RAG pipeline | Built from scratch | No LangChain — manual chunking, retrieval, reranking |
| Embeddings | Cohere (`embed-v4.0`) | Via Cohere API — no card required, no local model loading |
| Vector Database | Pinecone | Fully managed — no local/persistent storage |
| Reranking | Cohere Rerank (`rerank-v3.5`) | Same Cohere API as embeddings |
| LLM | Groq — `openai/gpt-oss-120b` | Grounded answer generation |
| Deployment | Vercel (serverless functions) | Stateless — fits a stack with no local storage |

---

## Project Structure

```
RAG-based Voice-Agent/
├── data/
│   └── raw_pdfs/                 # Source documents for the knowledge base
│       ├── hr_policy.pdf
│       ├── frontend_interview_prep.pdf
│       ├── backend_interview_prep.pdf
│       ├── pm_interview_prep.pdf
│       ├── devops_interview_prep.pdf
│       ├── aiml_interview_prep.pdf
│       └── hr_behavioral_interview_prep.pdf
│
├── ingestion/                    # Run locally, once, to populate Pinecone
│   ├── config.py
│   ├── extraction.py             # PDF text extraction, with OCR fallback
│   ├── chunking.py               # From-scratch recursive text chunker
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── ingest.py                 # CLI: add / delete / list / rebuild
│   ├── requirements.txt
│   └── .env.example
│
├── api/                          # Deployed to Vercel as the live backend
│   ├── index.py                  # /chat/completions endpoint (Vapi calls this)
│   ├── config.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── reranker.py
│   ├── retrieval.py
│   ├── generation.py
│   └── requirements.txt
│
├── vercel.json
├── .gitignore
├── DESIGN_NOTES.md                # Justifications: RAG vs. alternatives, dimension
│                                  # choice, temperature, quality evaluation, etc.
└── README.md
```

`ingestion/` and `api/` intentionally hold separate copies of a few shared modules — they're two independent programs (one-off local script vs. live deployed server), and Vercel's Python runtime requires every file a deployed function imports to live in the same folder as that function.

---

## Local Setup

### Prerequisites
- Python 3.10+
- A [Cohere](https://dashboard.cohere.com) Trial API key (free, no credit card required)
- A [Pinecone](https://www.pinecone.io) API key (free tier)
- A [Groq](https://console.groq.com) API key (free tier)
- A [Vapi.ai](https://vapi.ai) account

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd "RAG-based Voice-Agent"
```

### 2. Set up environment variables

Copy the example file and fill in your keys:

```bash
cp ingestion/.env.example .env
```

```env
COHERE_API_KEY=your_cohere_trial_key
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

### 3. Ingest the knowledge base

```bash
cd ingestion
pip install -r requirements.txt
python ingest.py rebuild ../data/raw_pdfs
```

This extracts, chunks, embeds, and upserts all 7 documents into Pinecone.

### 4. Run the backend locally

```bash
cd ../api
pip install -r requirements.txt
uvicorn index:app --reload
```

Test it directly:

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "How many casual leaves do I get per year?"}]}'
```

---

## Deployment

### Backend (Vercel)

```bash
vercel deploy
```

Add `COHERE_API_KEY`, `PINECONE_API_KEY`, and `GROQ_API_KEY` under Project → Settings → Environment Variables on Vercel before deploying to production.

### Voice Agent (Vapi.ai)

1. In the Vapi dashboard, create a new assistant.
2. Under Model, select **Custom LLM**.
3. Set the endpoint URL to your deployed Vercel URL (e.g. `https://your-project.vercel.app`).
4. Vapi will call `/chat/completions` on that URL automatically, following the standard OpenAI request/response format.

---

## Document Management (CLI)

```bash
python ingest.py add <path_to_pdf> <category>   # add or update a document
python ingest.py delete <source_filename>        # remove a whole document
python ingest.py list                             # list all ingested documents
```

---

## Knowledge Base Categories

| Category | Covers |
|---|---|
| `hr_policy` | Recruitment, onboarding, leave, performance management, compensation, workplace conduct, training, offboarding |
| `frontend` | JavaScript/TypeScript, React, CSS/layout, performance, scenario questions, difficulty tiers (Intern → Principal) |
| `backend` | API design, databases, auth/security, system design, scenario questions, difficulty tiers |
| `project_manager` | Methodology, stakeholder/risk management, tooling, scenario questions, difficulty tiers |
| `devops` | CI/CD, containers/orchestration, infrastructure as code, scenario questions, difficulty tiers |
| `aiml` | Core ML, deep learning/NLP, applied focus, scenario questions, difficulty tiers |
| `hr_behavioral` | Shared behavioral/HR-round questions across all roles, difficulty tiers |

---

## Design Rationale

See [`DESIGN_NOTES.md`](./DESIGN_NOTES.md) for detailed justifications covering:
- Why RAG over fine-tuning or long-context prompting
- Delete/update handling for source documents
- Embedding model choice and why its dimension matters
- No-local-storage compliance across the whole pipeline
- Semantic search + reranking algorithm design
- Temperature choice for grounded generation
- Quality evaluation approach

---

## Disclaimer

The HR policy and interview-preparation content in this knowledge base is original material written for this project, grounded in general publicly known norms (e.g. typical Pakistan leave-law ranges). It does not represent any specific real company's actual policies and should not be treated as authoritative for real employment decisions.

---

## Author

**Menahil** — AI Student, GCU Lahore
Built as an internship task: a real-time RAG-powered voice assistant using Vapi.ai.

---

## License

MIT License — feel free to use, modify, and distribute.