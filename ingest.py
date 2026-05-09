"""
ingest.py — Seed Pinecone with sample documents to test HyDE retrieval.

Run: python ingest.py
"""

import os
import requests

API_URL = "http://localhost:8000"

# ─── Sample documents about RAG and embeddings ────────────────────────────────
# These are written in "document style" (the kind HyDE mimics)
SAMPLE_DOCS = [
    {
        "id": "doc_001",
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "a retrieval system with a generative language model. Instead of relying "
            "solely on parametric knowledge, RAG fetches relevant passages from an "
            "external knowledge base at inference time, grounding the model's output "
            "in factual, up-to-date information."
        ),
        "metadata": {"topic": "RAG", "source": "overview"}
    },
    {
        "id": "doc_002",
        "text": (
            "Vector databases store dense numerical representations of text, images, "
            "or other data. Similarity search in vector databases works by computing "
            "cosine similarity or dot product between query vectors and stored vectors, "
            "returning the closest matches. Pinecone is a managed vector database "
            "optimized for low-latency approximate nearest-neighbor (ANN) search."
        ),
        "metadata": {"topic": "vector_db", "source": "overview"}
    },
    {
        "id": "doc_003",
        "text": (
            "Hypothetical Document Embedding (HyDE) addresses the semantic mismatch "
            "between short user queries and longer, information-dense documents. "
            "When a user asks a question, HyDE prompts an LLM to generate a hypothetical "
            "answer passage. The embedding of this passage is used for retrieval instead "
            "of embedding the raw query, resulting in a vector that better aligns with "
            "the document embedding space."
        ),
        "metadata": {"topic": "HyDE", "source": "technical"}
    },
    {
        "id": "doc_004",
        "text": (
            "Sentence transformers and dense retrieval models are trained using "
            "contrastive learning objectives, where semantically similar text pairs "
            "are pushed together in the embedding space while dissimilar pairs are "
            "pushed apart. Models like text-embedding-004 from Google produce 768- "
            "dimensional vectors that capture semantic meaning across many languages."
        ),
        "metadata": {"topic": "embeddings", "source": "technical"}
    },
    {
        "id": "doc_005",
        "text": (
            "Chunking strategy significantly impacts RAG performance. Documents are "
            "typically split into overlapping windows of 256–512 tokens before embedding. "
            "Larger chunks preserve more context but risk diluting relevance; smaller "
            "chunks improve precision but may lose surrounding context needed for "
            "coherent answers."
        ),
        "metadata": {"topic": "RAG", "source": "best_practices"}
    },
    {
        "id": "doc_006",
        "text": (
            "Re-ranking is a post-retrieval step in RAG pipelines where a cross-encoder "
            "model scores each retrieved document against the query and reorders them. "
            "Unlike bi-encoders used in first-stage retrieval, cross-encoders jointly "
            "encode query and document together, producing more accurate relevance scores "
            "at the cost of higher latency."
        ),
        "metadata": {"topic": "RAG", "source": "advanced"}
    },
    {
        "id": "doc_007",
        "text": (
            "The Groq LPU (Language Processing Unit) is purpose-built for low-latency "
            "LLM inference. Unlike GPU-based serving, LPUs use a deterministic, "
            "memory-bandwidth-optimized architecture that achieves significantly higher "
            "tokens-per-second throughput, making them ideal for real-time RAG pipelines "
            "where the retrieval step must complete within milliseconds."
        ),
        "metadata": {"topic": "groq", "source": "hardware"}
    },
    {
        "id": "doc_008",
        "text": (
            "Query expansion is a family of techniques that enrich an original query "
            "before retrieval. Methods include pseudo-relevance feedback, synonym "
            "injection, and LLM-based reformulation. HyDE can be seen as a form of "
            "query expansion where the expansion is a full hypothetical answer rather "
            "than additional keywords."
        ),
        "metadata": {"topic": "retrieval", "source": "techniques"}
    }
]


def main():
    print("🚀 Ingesting documents into Pinecone via /ingest endpoint...")
    print(f"   API: {API_URL}")
    print(f"   Documents: {len(SAMPLE_DOCS)}\n")

    response = requests.post(
        f"{API_URL}/ingest",
        json={"documents": SAMPLE_DOCS, "namespace": ""}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Upserted {result['upserted']} documents.")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    main()
