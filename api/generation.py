"""
Generation step: retrieved chunks + query -> Groq LLM -> grounded answer.

Temperature justification (mentor requirement): set low (0.15, see
config.py) because this is a factual, grounded-answer assistant, not a
creative one. Higher temperature increases the chance the model
paraphrases loosely or introduces small unstated details -- exactly what
we don't want when someone asks "how many casual leaves do I get" and
the honest answer is a specific number. Not set to 0.0 because a small
amount of variability keeps phrasing natural for a voice response rather
than robotic/repetitive across similar queries.
"""
import requests

from config import GROQ_API_KEY, GROQ_MODEL, GENERATION_TEMPERATURE

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are a voice assistant for a Pakistan-based tech company, answering \
two kinds of questions: (1) company HR policy questions (leave, benefits, notice period, \
performance reviews, etc.) and (2) tech career / interview-preparation questions for \
Frontend, Backend, Project Manager, DevOps, and AI/ML roles, including general HR/behavioral \
interview questions.

Rules:
- Answer ONLY using the provided context below. Do not use outside knowledge.
- If the context does not contain the answer, say clearly that you don't have that \
information in your available documents and suggest checking with HR directly -- do NOT \
guess or make something up.
- Keep answers conversational and concise, since this response will be spoken aloud by a \
voice agent, not read as text. Avoid bullet points, headers, or markdown -- speak in \
natural sentences.
- Do not mention "the context", "the documents", or "chunks" in your answer -- just answer \
naturally as if you know this information."""


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[Source: {c['source_file']}]\n{c['text']}" for c in chunks)


def generate_answer(query: str, chunks: list[dict], is_grounded: bool) -> str:
    if not is_grounded:
        context = "No sufficiently relevant information was found for this query."
    else:
        context = build_context(chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": GENERATION_TEMPERATURE,
            "max_tokens": 300,  # keep responses short -- this gets spoken aloud
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()