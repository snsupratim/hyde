"""
backend/main.py
═══════════════
Ecommerce HyDE RAG — FastAPI backend

Endpoints:
  GET  /health            → system status
  GET  /products          → list all products (with optional category filter)
  GET  /categories        → list all unique categories
  POST /search/standard   → standard RAG product search
  POST /search/hyde       → HyDE RAG product search
  POST /search/compare    → side-by-side comparison
  GET  /analytics         → recent search logs with score comparison
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
SUPABASE_URL     = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY     = os.getenv("SUPABASE_SERVICE_KEY", "")

GROQ_MODEL       = "llama-3.3-70b-versatile"
EMBEDDING_MODEL  = "models/gemini-embedding-001"
DEFAULT_TOP_K    = 5

# ── Lazy clients ──────────────────────────────────────────────────────────────
_groq      = None
_embedder  = None
_supabase  = None


def get_groq():
    global _groq
    if _groq is None:
        from groq import Groq
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


def get_embedder():
    global _embedder
    if _embedder is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        _embedder = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _embedder


def get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ecommerce HyDE RAG API",
    description="Real ecommerce product search using HyDE retrieval",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = DEFAULT_TOP_K
    category: Optional[str] = None          # optional category filter


class CompareRequest(BaseModel):
    query: str
    top_k: Optional[int] = DEFAULT_TOP_K
    category: Optional[str] = None


# ── Core helpers ──────────────────────────────────────────────────────────────

def embed_text(text: str) -> str:
    """
    Returns the Gemini embedding as a bracketed string '[0.1,0.2,...]'.
    Supabase REST API requires halfvec passed as a string — it auto-casts.
    """
    vector = get_embedder().embed_query(text)
    return "[" + ",".join(str(round(v, 8)) for v in vector) + "]"


def generate_hypothetical_product(query: str, category: Optional[str]) -> str:
    """
    HyDE Step: ask Groq to write a hypothetical product description
    that would match the query — in the same style as product catalog text.
    This is the key difference from standard RAG.
    """
    cat_hint = f" in the {category} category" if category else ""
    system_prompt = (
        "You are an ecommerce product catalog writer for an Indian shopping platform. "
        "Given a user's shopping query, write a detailed product description (3-5 sentences) "
        "exactly as it would appear in a product listing — including product type, key features, "
        "materials, use case, and who it's suitable for. "
        "Include price range context if relevant. "
        "Write in third person. Do NOT use 'I' or 'Here is'. "
        "Do NOT invent brand names — just describe the product generically."
    )
    user_msg = f"Write a product description for{cat_hint}: {query}"
    response = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=200,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def vector_search(
    vector: list[float],
    top_k: int,
    category: Optional[str]
) -> list[dict]:
    """
    Call the Supabase match_products RPC (pgvector cosine similarity).
    Returns a ranked list of products with similarity scores.
    """
    params = {
        "query_embedding": vector,
        "match_count": top_k,
        "filter_category": category,
    }
    result = get_supabase().rpc("match_products", params).execute()
    return [
        {
            "product_id":   r["product_id"],
            "name":         r["name"],
            "category":     r["category"],
            "price":        float(r["price"]),
            "description":  r["description"],
            "image_url":    r.get("image_url", ""),
            "brand":        r.get("brand", ""),
            "tags":         r.get("tags", []),
            "stock":        r.get("stock", 0),
            "similarity":   round(float(r["similarity"]), 4),
        }
        for r in result.data
    ]


def generate_recommendation(query: str, products: list[dict]) -> str:
    """Generate a short shopping recommendation using retrieved products."""
    context = "\n".join(
        f"- {p['name']} (₹{p['price']}) — {p['description'][:120]}..."
        for p in products[:3]
    )
    response = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful Indian ecommerce shopping assistant. "
                    "Based on the product listings provided, give a short friendly recommendation "
                    "(2-3 sentences) to the user. Mention the product names and why they fit. "
                    "Use ₹ symbol for prices."
                ),
            },
            {
                "role": "user",
                "content": f"I'm looking for: {query}\n\nAvailable products:\n{context}",
            },
        ],
        max_tokens=200,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def log_search(
    query: str,
    method: str,
    products: list[dict],
    hypothetical_doc: Optional[str] = None,
):
    """Save search to Supabase search_logs table for analytics."""
    try:
        get_supabase().table("search_logs").insert({
            "query":            query,
            "method":           method,
            "hypothetical_doc": hypothetical_doc,
            "result_ids":       [p["product_id"] for p in products],
            "result_scores":    [p["similarity"] for p in products],
        }).execute()
    except Exception:
        pass  # don't crash the search if logging fails


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message":   "Ecommerce HyDE RAG API ✅",
        "model":     GROQ_MODEL,
        "embedding": EMBEDDING_MODEL,
        "docs":      "/docs",
    }


@app.get("/health")
def health():
    try:
        count = get_supabase().table("products").select("id", count="exact").execute()
        return {
            "status":          "ok",
            "total_products":  count.count,
            "embedding_model": EMBEDDING_MODEL,
            "groq_model":      GROQ_MODEL,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/products")
def list_products(
    category: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
):
    """List all products, with optional category filter."""
    q = get_supabase().table("products").select("*").limit(limit)
    if category:
        q = q.eq("category", category)
    result = q.execute()
    return {"products": result.data, "count": len(result.data)}


@app.get("/categories")
def list_categories():
    """Return all unique product categories."""
    result = get_supabase().table("products").select("category").execute()
    cats = sorted({r["category"] for r in result.data})
    return {"categories": cats}


@app.post("/search/standard")
def standard_search(req: SearchRequest):
    """
    Standard RAG:
      1. Embed the raw user query
      2. Similarity search in Supabase pgvector
      3. Generate recommendation from retrieved products
    """
    query_vector = embed_text(req.query)
    products     = vector_search(query_vector, req.top_k, req.category)
    reco         = generate_recommendation(req.query, products)
    log_search(req.query, "standard", products)

    return {
        "method":         "standard",
        "query":          req.query,
        "products":       products,
        "recommendation": reco,
    }


@app.post("/search/hyde")
def hyde_search(req: SearchRequest):
    """
    HyDE RAG:
      1. Groq generates a hypothetical product description for the query
      2. Embed THAT description (not the raw query)
      3. Similarity search with the richer embedding
      4. Generate recommendation from real retrieved products
    """
    hypothetical_doc = generate_hypothetical_product(req.query, req.category)
    hyde_vector      = embed_text(hypothetical_doc)
    products         = vector_search(hyde_vector, req.top_k, req.category)
    reco             = generate_recommendation(req.query, products)
    log_search(req.query, "hyde", products, hypothetical_doc)

    return {
        "method":           "hyde",
        "query":            req.query,
        "hypothetical_doc": hypothetical_doc,   # what HyDE imagined
        "products":         products,
        "recommendation":   reco,
    }


@app.post("/search/compare")
def compare_search(req: CompareRequest):
    """
    Side-by-side Standard vs HyDE comparison.
    The response includes both result sets, both recommendations,
    and an analysis block showing which products differ.
    """
    # ── Standard ──
    query_vector    = embed_text(req.query)
    std_products    = vector_search(query_vector, req.top_k, req.category)
    std_reco        = generate_recommendation(req.query, std_products)
    log_search(req.query, "standard", std_products)

    # ── HyDE ──
    hypothetical_doc = generate_hypothetical_product(req.query, req.category)
    hyde_vector      = embed_text(hypothetical_doc)
    hyde_products    = vector_search(hyde_vector, req.top_k, req.category)
    hyde_reco        = generate_recommendation(req.query, hyde_products)
    log_search(req.query, "hyde", hyde_products, hypothetical_doc)

    # ── Analysis ──
    std_ids  = {p["product_id"] for p in std_products}
    hyde_ids = {p["product_id"] for p in hyde_products}

    # Average similarity score comparison
    std_avg  = round(sum(p["similarity"] for p in std_products) / len(std_products), 4) if std_products else 0
    hyde_avg = round(sum(p["similarity"] for p in hyde_products) / len(hyde_products), 4) if hyde_products else 0

    return {
        "query": req.query,

        "standard": {
            "products":       std_products,
            "recommendation": std_reco,
            "avg_score":      std_avg,
        },

        "hyde": {
            "hypothetical_doc": hypothetical_doc,
            "products":         hyde_products,
            "recommendation":   hyde_reco,
            "avg_score":        hyde_avg,
        },

        "analysis": {
            "score_improvement":   round(hyde_avg - std_avg, 4),
            "overlap_count":       len(std_ids & hyde_ids),
            "unique_to_standard":  list(std_ids - hyde_ids),
            "unique_to_hyde":      list(hyde_ids - std_ids),
            "total_compared":      req.top_k,
        },
    }


@app.get("/analytics")
def analytics(limit: int = Query(20, le=100)):
    """Recent search logs — useful for comparing standard vs HyDE over time."""
    result = (
        get_supabase()
        .table("search_logs")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"logs": result.data, "count": len(result.data)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)