"""
FastAPI app exposing an OpenAI-compatible /chat/completions endpoint that
Vapi calls directly as a "Custom LLM" provider.

How Vapi uses this: Vapi handles STT (user's speech -> text) and TTS
(our text reply -> spoken audio) itself. In between, it sends the
conversation so far as a standard OpenAI chat-completions request to
this endpoint, and expects a standard OpenAI-shaped response back.

Streaming: Vapi sends {"stream": true} by default. We now honor that and
stream Server-Sent Events back as tokens arrive from Groq, instead of
waiting for the full answer -- this lets Vapi start speaking the first
words while the rest of the RAG pipeline's answer is still generating,
which matters a lot here since retrieve->rerank->generate takes a few
real seconds and a caller sitting in total silence that long tends to
just hang up. Non-streaming requests (stream absent or false) still get
a single JSON response, unchanged from before.

Deploy target: Vercel serverless function (see /api/index.py + vercel.json).
This file contains the actual app; api/index.py just imports it so Vercel's
Python runtime can find it in the expected location.
"""
import sys
import os

# Vercel's Python runtime doesn't always add this file's own directory to
# sys.path before importing it, which breaks sibling imports like
# `from retrieval import retrieve_grounded` below even though this exact
# same code runs fine locally with uvicorn. Adding it explicitly here
# fixes that regardless of how Vercel's runtime is configured.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from retrieval import retrieve_grounded
from generation import generate_answer, generate_answer_stream

app = FastAPI(title="Pak Tech Career & HR Voice Assistant - Custom LLM Backend")


@app.get("/")
@app.get("/api")
@app.get("/api/")
def health_check():
    return {"status": "ok"}


async def _handle_chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    wants_stream = body.get("stream", False)
    model_name = body.get("model", "pak-tech-rag-assistant")

    # The most recent user message is the actual question to answer --
    # Vapi sends the full running conversation each time, but our RAG
    # step only needs the latest turn.
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return JSONResponse(status_code=400, content={"error": "No user message found"})
    query = user_messages[-1]["content"]

    chunks, is_grounded = retrieve_grounded(query)

    if wants_stream:
        return StreamingResponse(
            _stream_sse(query, chunks, is_grounded, model_name),
            media_type="text/event-stream",
        )

    answer = generate_answer(query, chunks, is_grounded)

    # Standard OpenAI chat-completions response shape -- this is what
    # makes the endpoint "OpenAI-compatible" and lets Vapi parse it
    # without any custom integration code on their side.
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
    }


def _stream_sse(query: str, chunks: list, is_grounded: bool, model_name: str):
    """Generator yielding OpenAI-format SSE chunks as Groq streams tokens back."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    for delta in generate_answer_stream(query, chunks, is_grounded):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# Registered under every path variant we've seen Vercel actually route to
# in practice (bare, with /api prefix, with/without trailing slash) --
# Vercel's Python zero-config routing behavior didn't match either the
# documented convention or our explicit vercel.json rewrite cleanly, so
# rather than keep guessing, all plausible real paths point at the same
# handler. Whichever one turns out to be live, it works.
app.post("/chat/completions")(_handle_chat_completions)
app.post("/api/chat/completions")(_handle_chat_completions)
app.post("/api/chat/completions/")(_handle_chat_completions)