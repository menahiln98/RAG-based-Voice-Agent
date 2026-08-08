"""
From-scratch recursive text chunker (no LangChain).

Strategy: try to split on the largest structural boundary first
(paragraph breaks), and only fall back to smaller boundaries (sentences,
then raw characters) if a piece is still too large. This keeps chunks
semantically coherent -- a chunk is far more likely to be "one complete
idea" than if we just sliced every N characters blindly.

Overlap is added between consecutive chunks so a fact split across a
chunk boundary (e.g. a sentence continuing into the next chunk) still has
a chance of being retrieved in full by at least one of the two chunks.
"""
import re
from dataclasses import dataclass

from config import CHUNK_SIZE, CHUNK_OVERLAP

SEPARATORS = ["\n\n", "\n", ". ", " "]  # largest to smallest structural boundary


@dataclass
class Chunk:
    text: str
    chunk_index: int


def _split_on_separator(text: str, separator: str) -> list[str]:
    if separator == " ":
        return text.split(" ")
    parts = text.split(separator)
    # Re-attach the separator to keep sentence punctuation / paragraph
    # structure intact for readability and better embeddings.
    return [p + separator for p in parts[:-1]] + [parts[-1]]


def _recursive_split(text: str, separators: list[str], max_size: int) -> list[str]:
    if len(text) <= max_size:
        return [text]

    if not separators:
        # No separators left -- hard-cut at max_size as a last resort.
        return [text[i:i + max_size] for i in range(0, len(text), max_size)]

    sep, rest_separators = separators[0], separators[1:]
    pieces = _split_on_separator(text, sep)

    chunks, current = [], ""
    for piece in pieces:
        if len(current) + len(piece) <= max_size:
            current += piece
        else:
            if current:
                chunks.append(current)
            if len(piece) > max_size:
                # Piece itself is too big -- recurse with a smaller separator.
                chunks.extend(_recursive_split(piece, rest_separators, max_size))
                current = ""
            else:
                current = piece
    if current:
        chunks.append(current)

    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + chunks[i])
    return overlapped


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    raw_chunks = _recursive_split(text, SEPARATORS, chunk_size)
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
    raw_chunks = _add_overlap(raw_chunks, overlap)
    return [Chunk(text=c, chunk_index=i) for i, c in enumerate(raw_chunks)]