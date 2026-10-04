import os
import re

from pypdf import PdfReader

from src.semantic_search import (
    create_embeddings,
    get_embedding_model,
    EMBEDDING_MODEL
)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_document_text(text):

    if not text:
        return ""

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(file_path):

    reader = PdfReader(
        file_path
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = (
            page.extract_text()
            or ""
        )

        page_text = clean_document_text(
            page_text
        )

        if page_text:

            pages.append({
                "page": page_number,
                "text": page_text
            })

    return pages


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt_text(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        text = file.read()

    text = clean_document_text(
        text
    )

    if not text:
        return []

    return [{
        "page": 1,
        "text": text
    }]


# ============================================================
# DOCUMENT LOADER
# ============================================================

def load_document(file_path):

    if not file_path:

        return {
            "error":
                "Please upload a document."
        }

    extension = os.path.splitext(
        file_path
    )[1].lower()

    if extension == ".pdf":

        pages = extract_pdf_text(
            file_path
        )

        file_type = "PDF"

    elif extension == ".txt":

        pages = extract_txt_text(
            file_path
        )

        file_type = "TXT"

    else:

        return {
            "error":
                "Unsupported file type. "
                "Please upload PDF or TXT."
        }

    if not pages:

        return {
            "error":
                "No readable text was found. "
                "Scanned PDFs may require OCR."
        }

    full_text = "\n\n".join(
        page["text"]
        for page in pages
    )

    return {
        "pages": pages,
        "text": full_text,
        "file_type": file_type,
        "page_count": len(pages),
        "word_count": len(
            full_text.split()
        ),
        "character_count": len(
            full_text
        )
    }


# ============================================================
# PAGE-AWARE CHUNKING
# ============================================================

def create_document_chunks(
    pages,
    chunk_size=100,
    overlap=20
):

    chunks = []

    chunk_id = 1

    chunk_size = max(
        int(chunk_size),
        1
    )

    overlap = max(
        int(overlap),
        0
    )

    if overlap >= chunk_size:
        overlap = chunk_size - 1

    step = (
        chunk_size
        - overlap
    )

    for page_data in pages:

        page_number = (
            page_data["page"]
        )

        words = (
            page_data["text"]
            .split()
        )

        start = 0

        while start < len(words):

            end = (
                start
                + chunk_size
            )

            chunk_words = words[
                start:end
            ]

            chunk_text = " ".join(
                chunk_words
            )

            if chunk_text.strip():

                chunks.append({
                    "chunk_id":
                        chunk_id,

                    "page":
                        page_number,

                    "text":
                        chunk_text
                })

                chunk_id += 1

            if end >= len(words):
                break

            start += step

    return chunks


# ============================================================
# DOCUMENT SEMANTIC SEARCH
# ============================================================

def search_document(
    file_path,
    query,
    top_k=3
):

    if not query or not query.strip():

        return {
            "error":
                "Please enter a search query."
        }

    start_time = (
        time.perf_counter()
    )

    document = load_document(
        file_path
    )

    if "error" in document:
        return document

    chunks = create_document_chunks(
        document["pages"]
    )

    if not chunks:

        return {
            "error":
                "No searchable text "
                "was found."
        }

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    chunk_embeddings = (
        create_embeddings(
            chunk_texts
        )
    )

    query_embedding = (
        create_embeddings(
            [query]
        )
    )

    similarities = (
        cosine_similarity(
            query_embedding,
            chunk_embeddings
        )[0]
    )

    top_k = max(
        int(top_k),
        1
    )

    top_k = min(
        top_k,
        len(chunks)
    )

    ranked_indices = (
        np.argsort(
            similarities
        )[::-1][:top_k]
    )

    results = []

    for rank, index in enumerate(
        ranked_indices,
        start=1
    ):

        chunk = chunks[index]

        results.append({
            "rank":
                rank,

            "page":
                chunk["page"],

            "chunk_id":
                chunk["chunk_id"],

            "similarity":
                round(
                    float(
                        similarities[index]
                    ),
                    4
                ),

            "text":
                chunk["text"]
        })

    elapsed = (
        time.perf_counter()
        - start_time
    ) * 1000

    return {
        "task":
            "Document Semantic Search",

        "file_type":
            document[
                "file_type"
            ],

        "pages":
            document[
                "page_count"
            ],

        "word_count":
            document[
                "word_count"
            ],

        "chunks":
            len(chunks),

        "results":
            results,

        "model":
            EMBEDDING_MODEL,

        "search_time_ms":
            round(
                elapsed,
                2
            )
    }


# ============================================================
# UI RESULT FORMATTER
# ============================================================

def format_document_results(
    result
):

    if "error" in result:
        return result["error"]

    sections = []

    for item in result[
        "results"
    ]:

        sections.append(
            f"RANK {item['rank']}\n"
            f"Page: {item['page']}\n"
            f"Chunk: {item['chunk_id']}\n"
            f"Similarity: "
            f"{item['similarity']}\n\n"
            f"{item['text']}"
        )

    return (
        "\n\n"
        + ("=" * 60)
        + "\n\n"
    ).join(sections)
