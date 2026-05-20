# 🛒 Ecommerce HyDE RAG System

Real-time ecommerce product search using **Hypothetical Document Embedding (HyDE)**.
Users search with natural language — HyDE bridges the gap between casual queries and product catalog text.

## Stack
| Layer | Tech |
|---|---|
| UI | Streamlit |
| API | FastAPI |
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | Gemini embedding-001 (3072 dims) |
| Database + VectorDB | Supabase (PostgreSQL + pgvector) |

---

## Project Structure
```
ecom_hyde_rag/
├── backend/
│   └── main.py          ← FastAPI backend
├── frontend/
│   └── app.py           ← Streamlit UI
├── scripts/
│   ├── schema.sql        ← Supabase tables + pgvector function
│   └── seed_products.py  ← 30 sample Indian products with embeddings
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup (step by step)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Fill in: GROQ_API_KEY, GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
```

### 3. Supabase setup
- Go to [supabase.com](https://supabase.com) → create a new project
- In **SQL Editor**, run `scripts/schema.sql` (enables pgvector, creates tables + RPC function)
- In **Settings → API**, copy your `URL` and `service_role` key into `.env`

### 4. Seed products
```bash
python scripts/seed_products.py
```
Inserts 30 products and generates Gemini embeddings for each.

### 5. Start FastAPI backend
```bash
uvicorn backend.main:app --reload
# Runs at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### 6. Start Streamlit frontend
```bash
streamlit run frontend/app.py
# Opens at http://localhost:8501
```

---

## Try these queries (in the Compare tab)

| Query | Why it's a good HyDE test |
|---|---|
| `something warm for winter under 500` | Vague, conversational — no product keywords |
| `gift for dad who likes fitness` | Zero product vocabulary |
| `budget earphones for metro commute` | Context-heavy, no brand names |
| `kitchen item for making tea quickly` | Indirect reference to kettle |
| `shoes for someone who walks a lot` | No technical spec mentioned |

These are the queries where HyDE scores **significantly higher** than standard RAG
because the hypothetical product description uses catalog-style vocabulary.

---

## API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System status + product count |
| GET | `/products` | List products (with category filter) |
| GET | `/categories` | Unique categories |
| POST | `/search/standard` | Standard RAG search |
| POST | `/search/hyde` | HyDE RAG search |
| POST | `/search/compare` | Side-by-side comparison |
| GET | `/analytics` | Recent search logs |
