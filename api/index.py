"""
FastAPI app exposing an OpenAI-compatible /chat/completions endpoint that
Vapi calls directly as a "Custom LLM" provider.

How Vapi uses this: Vapi handles STT (user's speech -> text) and TTS
(our text reply -> spoken audio) itself. In between, it sends the
conversation so far as a standard OpenAI chat-completions request to
this endpoint, and expects a standard OpenAI-shaped response back. Vapi
sends {"stream": true} by default but is documented to handle a
non-streaming JSON response fine too -- we return non-streaming here to
keep the implementation simple and reliable within Vercel's function
timeout; streaming (SSE) is a documented future improvement.

Deploy target: Vercel serverless function (see /api/index.py + vercel.json).
This file contains the actual app; api/index.py just imports it so Vercel's
Python runtime can find it in the expected location.
"""
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from retrieval import retrieve_grounded
from generation import generate_answer

app = FastAPI(title="Pak Tech Career & HR Voice Assistant - Custom LLM Backend")


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # The most recent user message is the actual question to answer --
    # Vapi sends the full running conversation each time, but our RAG
    # step only needs the latest turn.
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return JSONResponse(status_code=400, content={"error": "No user message found"})
    query = user_messages[-1]["content"]

    chunks, is_grounded = retrieve_grounded(query)
    answer = generate_answer(query, chunks, is_grounded)

    # Standard OpenAI chat-completions response shape -- this is what
    # makes the endpoint "OpenAI-compatible" and lets Vapi parse it
    # without any custom integration code on their side.
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "pak-tech-rag-assistant"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
    }