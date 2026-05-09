# HyDE RAG — FastAPI + Groq + Gemini Embeddings + Pinecone

Demonstrates **Hypothetical Document Embedding (HyDE)** vs standard RAG retrieval.

## Why HyDE?

| Problem | Solution |
|---|---|
| Short query `"how does hyde work?"` has a thin embedding → misses document-style vectors | Generate a full hypothetical answer → embed THAT instead |

```
User Query
   │
   ├─► [Standard]  embed("how does hyde work?")   → Pinecone search
   │
   └─► [HyDE]      Groq generates hypothetical doc
                       "Hypothetical Document Embedding (HyDE) is..."
                             │
                        embed(hypothetical_doc)    → Pinecone search
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Fill in GROQ_API_KEY, GOOGLE_API_KEY, PINECONE_API_KEY
```

Load them in your shell:
```bash
export $(cat .env | xargs)
```

### 3. Create Pinecone index
Go to [pinecone.io](https://pinecone.io) → Create index:
- **Name**: `hyde-rag-index`
- **Dimensions**: `768`  (Gemini text-embedding-004 output size)
- **Metric**: `cosine`

### 4. Start the API
```bash
uvicorn main:app --reload
```

### 5. Ingest sample documents
```bash
python ingest.py
```

### 6. Run the comparison test
```bash
python test_hyde.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/`                | Overview |
| `GET`  | `/health`          | Check Pinecone connection |
| `POST` | `/ingest`          | Upsert documents |
| `POST` | `/query/standard`  | Standard RAG |
| `POST` | `/query/hyde`      | HyDE RAG |
| `POST` | `/query/compare`   | Side-by-side comparison |

## Interactive Docs
Open `http://localhost:8000/docs` to explore the API with Swagger UI.

## Example: Compare endpoint

```bash
curl -X POST http://localhost:8000/query/compare \
  -H "Content-Type: application/json" \
  -d '{"query": "how does hyde work?", "top_k": 3}'
```

Response includes:
- `standard.retrieved_docs` — docs found by embedding raw query
- `hyde.hypothetical_doc` — what Groq imagined the answer looks like
- `hyde.retrieved_docs` — docs found by embedding hypothetical doc
- `analysis.unique_to_hyde` — docs HyDE found that standard missed ← the key insight
