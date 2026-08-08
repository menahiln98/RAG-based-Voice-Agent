"""
Central configuration. All secrets/config come from environment variables
(loaded via a .env file locally, or set directly on Vercel) -- nothing is
hardcoded, and nothing is persisted to local disk beyond this process's
runtime, in line with the "no local storage" requirement.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Cohere (embeddings + reranking, one provider, no card required) ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")

# --- Embedding dimension ---
# Cohere's embed-v4.0 is a Matryoshka model supporting 256/512/1024/1536-dim
# output via the `output_dimension` request parameter. We use 1024 -- a
# strong quality/storage balance -- and this MUST match the Pinecone
# index's `dimension` setting exactly, or every upsert call fails outright.
# If this is ever changed, the Pinecone index must be recreated (you
# cannot change an existing index's dimension in place).
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# --- Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pak-tech-career-assistant")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")  # free tier serverless region

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))      # target characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80")) # overlap characters between chunks

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))          # candidates pulled from Pinecone
TOP_K_RERANKED = int(os.getenv("TOP_K_RERANKED", "3"))  # kept after reranking

# --- Generation ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Low temperature deliberately: this is a grounded Q&A assistant, not a
# creative one. We want the model to stick closely to retrieved context
# rather than improvise, so 0.1-0.2 is appropriate. See DESIGN_NOTES.md
# section 6 for the full justification.
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.15"))


def validate():
    missing = [name for name, val in [
        ("COHERE_API_KEY", COHERE_API_KEY),
        ("PINECONE_API_KEY", PINECONE_API_KEY),
        ("GROQ_API_KEY", GROQ_API_KEY),
    ] if not val]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill in your keys."
        )