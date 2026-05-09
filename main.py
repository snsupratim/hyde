"""
HyDE RAG Application
====================
Demonstrates Hypothetical Document Embedding (HyDE) in RAG retrieval.

HyDE fixes the semantic gap:
  Standard : embed(raw short query)          → Pinecone search
  HyDE     : LLM generates hypothetical doc
             → embed(hypothetical doc)       → Pinecone search

Stack: FastAPI + Groq (llama-3.3-70b-versatile) + Gemini embedding-001 + Pinecone
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ─── Config ──────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "hyde-rag-index")

GROQ_MODEL       = "llama-3.3-70b-versatile"
EMBEDDING_MODEL  = "models/gemini-embedding-001"   # outputs 3072-dim vectors
TOP_K            = 5

# ─── Lazy client init (avoids import-time crash if keys are missing) ─────────
_groq_client  = None
_embedder     = None
_pc_index     = None


def get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def get_embedder():
    global _embedder
    if _embedder is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        _embedder = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _embedder


def get_index():
    global _pc_index
    if _pc_index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pc_index = pc.Index(PINECONE_INDEX)
    return _pc_index


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="HyDE RAG API",
    description="Compare Standard vs HyDE retrieval side-by-side",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K
    namespace: Optional[str] = ""


class CompareRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K
    namespace: Optional[str] = ""


class IngestRequest(BaseModel):
    documents: list[dict]   # [{"id": "...", "text": "...", "metadata": {...}}]
    namespace: Optional[str] = ""


# ─── Core helpers ─────────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """Embed text using Gemini gemini-embedding-001 (3072 dims)."""
    return get_embedder().embed_query(text)


def generate_hypothetical_doc(query: str) -> str:
    """
    HyDE Step 1: Ask Groq to write a hypothetical document-style passage
    that *would* answer the query. Its embedding aligns with real doc embeddings.
    """
    system_prompt = (
        "You are an expert in NLP, machine learning, and Retrieval-Augmented Generation (RAG) systems. "
        "Given a user question about AI/ML topics, write a detailed factual passage (3-5 sentences) "
        "that directly answers the question as if extracted from a technical NLP research paper or documentation. "
        "Write in third-person encyclopedic style. Do NOT say 'I' or 'Here is'. "
        "Note: 'HyDE' refers to Hypothetical Document Embedding, a RAG retrieval technique — "
        "not the Jekyll static site generator."
    )
    response = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ],
        max_tokens=250,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def pinecone_search(vector: list[float], top_k: int, namespace: str) -> list[dict]:
    """Similarity search in Pinecone; returns ranked results."""
    results = get_index().query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    return [
        {
            "id":       m["id"],
            "score":    round(m["score"], 4),
            "text":     m["metadata"].get("text", ""),
            "metadata": {k: v for k, v in m["metadata"].items() if k != "text"},
        }
        for m in results["matches"]
    ]


def generate_final_answer(query: str, docs: list[dict]) -> str:
    """RAG answer generation — uses retrieved docs as context."""
    context = "\n\n".join(f"[Doc {i+1}] {d['text']}" for i, d in enumerate(docs))
    response = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question using ONLY "
                    "the provided context documents. Be concise and accurate."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        max_tokens=400,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "HyDE RAG API is running ✅",
        "model":   GROQ_MODEL,
        "embedding": EMBEDDING_MODEL,
        "endpoints": {
            "POST /query/standard": "Standard RAG — embed raw query",
            "POST /query/hyde":     "HyDE RAG — embed hypothetical doc",
            "POST /query/compare":  "Side-by-side comparison",
            "POST /ingest":         "Upsert documents into Pinecone",
            "GET  /health":         "Health check",
        },
    }


@app.get("/health")
def health():
    try:
        stats = get_index().describe_index_stats()
        return {
            "status": "ok",
            "pinecone_index":   PINECONE_INDEX,
            "total_vectors":    stats.get("total_vector_count", 0),
            "embedding_model":  EMBEDDING_MODEL,
            "groq_model":       GROQ_MODEL,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/query/standard")
def standard_query(req: QueryRequest):
    """
    Standard RAG:
      1. Embed the raw user query
      2. Search Pinecone
      3. Generate answer
    """
    query_vector = embed_text(req.query)
    docs         = pinecone_search(query_vector, req.top_k, req.namespace)
    answer       = generate_final_answer(req.query, docs)
    return {
        "method":         "standard",
        "query":          req.query,
        "retrieved_docs": docs,
        "answer":         answer,
    }


@app.post("/query/hyde")
def hyde_query(req: QueryRequest):
    """
    HyDE RAG:
      1. Groq generates a hypothetical document answer
      2. Embed the hypothetical doc (NOT the raw query)
      3. Search Pinecone with richer embedding
      4. Generate final answer from real retrieved docs
    """
    hypothetical_doc = generate_hypothetical_doc(req.query)
    hyde_vector      = embed_text(hypothetical_doc)
    docs             = pinecone_search(hyde_vector, req.top_k, req.namespace)
    answer           = generate_final_answer(req.query, docs)
    return {
        "method":           "hyde",
        "query":            req.query,
        "hypothetical_doc": hypothetical_doc,   # ← what HyDE imagined
        "retrieved_docs":   docs,
        "answer":           answer,
    }


@app.post("/query/compare")
def compare_query(req: CompareRequest):
    """
    Side-by-side comparison of Standard vs HyDE.
    The analysis.unique_to_hyde field shows what HyDE found that standard missed.
    """
    # Standard
    query_vector    = embed_text(req.query)
    standard_docs   = pinecone_search(query_vector, req.top_k, req.namespace)
    standard_answer = generate_final_answer(req.query, standard_docs)

    # HyDE
    hypothetical_doc = generate_hypothetical_doc(req.query)
    hyde_vector      = embed_text(hypothetical_doc)
    hyde_docs        = pinecone_search(hyde_vector, req.top_k, req.namespace)
    hyde_answer      = generate_final_answer(req.query, hyde_docs)

    # Overlap
    std_ids  = {d["id"] for d in standard_docs}
    hyde_ids = {d["id"] for d in hyde_docs}

    return {
        "query": req.query,
        "standard": {
            "retrieved_docs": standard_docs,
            "answer":         standard_answer,
        },
        "hyde": {
            "hypothetical_doc": hypothetical_doc,
            "retrieved_docs":   hyde_docs,
            "answer":           hyde_answer,
        },
        "analysis": {
            "overlap_count":       len(std_ids & hyde_ids),
            "overlapping_doc_ids": list(std_ids & hyde_ids),
            "unique_to_standard":  list(std_ids - hyde_ids),
            "unique_to_hyde":      list(hyde_ids - std_ids),   # ← key insight
            "total_docs_compared": req.top_k,
        },
    }


@app.post("/ingest")
def ingest_documents(req: IngestRequest):
    """
    Embed and upsert documents into Pinecone.
    Payload: {"documents": [{"id": "...", "text": "...", "metadata": {...}}]}
    """
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    vectors = []
    for doc in req.documents:
        if "id" not in doc or "text" not in doc:
            raise HTTPException(status_code=400, detail="Each doc needs 'id' and 'text'.")
        meta = {**doc.get("metadata", {}), "text": doc["text"]}
        vectors.append({
            "id":       doc["id"],
            "values":   embed_text(doc["text"]),
            "metadata": meta,
        })

    # Batch upsert
    batch = 100
    for i in range(0, len(vectors), batch):
        get_index().upsert(vectors=vectors[i:i+batch], namespace=req.namespace)

    return {
        "status":    "success",
        "upserted":  len(vectors),
        "namespace": req.namespace or "default",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)