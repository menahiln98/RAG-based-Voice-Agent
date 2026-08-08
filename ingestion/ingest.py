"""
Data ingestion CLI for the Pakistan Tech Career & HR Voice Assistant.

Usage:
  python ingest.py add <path_to_pdf> <category>
  python ingest.py delete <source_filename>
  python ingest.py list
  python ingest.py rebuild <folder_of_pdfs>   # convenience: ingest every
                                               # pdf in a folder using its
                                               # filename prefix as category

Categories used by this project:
  hr_policy, frontend, backend, project_manager, devops, aiml, hr_behavioral
"""
import argparse
import hashlib
import os
import sys

import config
from extraction import extract_full_text
from chunking import chunk_text
from embeddings import embed_documents
from vector_store import upsert_chunks, delete_by_source, list_sources


def _chunk_id(source_file: str, chunk_index: int) -> str:
    # Deterministic ID so re-ingesting the same file+chunk overwrites
    # rather than duplicates.
    raw = f"{source_file}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def add_document(pdf_path: str, category: str):
    source_file = os.path.basename(pdf_path)
    print(f"Extracting text from {source_file} ...")
    full_text = extract_full_text(pdf_path)
    if not full_text.strip():
        print(f"  WARNING: no extractable text found in {source_file} (even after OCR fallback). Skipping.")
        return

    print("Chunking ...")
    chunks = chunk_text(full_text)
    print(f"  {len(chunks)} chunks produced")

    print("Embedding (Cohere API) ...")
    texts = [c.text for c in chunks]
    vectors_raw = embed_documents(texts)

    print("Upserting to Pinecone ...")
    vectors = [
        {
            "id": _chunk_id(source_file, c.chunk_index),
            "values": vectors_raw[i],
            "metadata": {
                "source_file": source_file,
                "category": category,
                "chunk_index": c.chunk_index,
                "text": c.text,   # stored so retrieval can return the original text directly
            },
        }
        for i, c in enumerate(chunks)
    ]
    upsert_chunks(vectors)
    print(f"Done: {source_file} ingested under category '{category}' ({len(vectors)} chunks).")


def delete_document(source_file: str):
    print(f"Deleting all chunks for source file '{source_file}' ...")
    delete_by_source(source_file)
    print("Done.")


def list_documents():
    sources = list_sources()
    if not sources:
        print("No documents currently ingested.")
        return
    print(f"{len(sources)} document(s) currently ingested:")
    for s in sorted(sources):
        print(f"  - {s}")


def rebuild_folder(folder: str):
    # Convenience for this project's 7-document corpus: infer category
    # from filename prefix. Adjust CATEGORY_MAP if filenames change.
    category_map = {
        "hr_policy.pdf": "hr_policy",
        "frontend_interview_prep.pdf": "frontend",
        "backend_interview_prep.pdf": "backend",
        "pm_interview_prep.pdf": "project_manager",
        "devops_interview_prep.pdf": "devops",
        "aiml_interview_prep.pdf": "aiml",
        "hr_behavioral_interview_prep.pdf": "hr_behavioral",
    }
    for filename, category in category_map.items():
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            add_document(path, category)
        else:
            print(f"  SKIPPED (not found): {filename}")


def main():
    config.validate()

    parser = argparse.ArgumentParser(description="Ingestion CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add/update a single PDF")
    p_add.add_argument("pdf_path")
    p_add.add_argument("category")

    p_del = sub.add_parser("delete", help="Delete a whole PDF's chunks by filename")
    p_del.add_argument("source_filename")

    sub.add_parser("list", help="List all ingested source documents")

    p_rebuild = sub.add_parser("rebuild", help="Ingest the standard 7-document corpus from a folder")
    p_rebuild.add_argument("folder")

    args = parser.parse_args()

    if args.command == "add":
        add_document(args.pdf_path, args.category)
    elif args.command == "delete":
        delete_document(args.source_filename)
    elif args.command == "list":
        list_documents()
    elif args.command == "rebuild":
        rebuild_folder(args.folder)


if __name__ == "__main__":
    sys.exit(main())