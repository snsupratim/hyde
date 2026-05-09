"""
test_hyde.py — Test Standard vs HyDE retrieval side-by-side.

This script visually demonstrates WHY HyDE solves the semantic gap problem
by showing what each method retrieves for the same queries.

Run: python test_hyde.py
"""

import requests
import json

API_URL = "http://localhost:8000"

# ─── Test queries (short, conversational — typical user input) ─────────────────
TEST_QUERIES = [
    "how does hyde work?",
    "why is pinecone fast?",
    "best practices for splitting text in rag?",
    "what is groq?",
]


def separator(char="─", width=70):
    print(char * width)


def print_docs(docs: list[dict], label: str):
    print(f"\n  📄 {label} Retrieved Docs:")
    for i, doc in enumerate(docs, 1):
        print(f"     [{i}] score={doc['score']:.4f} | id={doc['id']}")
        print(f"         {doc['text'][:100]}...")


def run_compare(query: str):
    separator("═")
    print(f"  🔍 Query: \"{query}\"")
    separator("═")

    resp = requests.post(
        f"{API_URL}/query/compare",
        json={"query": query, "top_k": 3}
    )

    if resp.status_code != 200:
        print(f"  ❌ Error: {resp.status_code} — {resp.text}")
        return

    data = resp.json()

    # ── HyDE hypothetical doc ──
    print(f"\n  🤖 HyDE Hypothetical Doc (what Groq imagines the answer looks like):")
    hypo = data["hyde"]["hypothetical_doc"]
    # wrap text at 70 chars
    words = hypo.split()
    line, lines = [], []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > 65:
            lines.append("     " + " ".join(line))
            line = []
    if line:
        lines.append("     " + " ".join(line))
    print("\n".join(lines))

    # ── Docs retrieved ──
    print_docs(data["standard"]["retrieved_docs"], "Standard (raw query embedded)")
    print_docs(data["hyde"]["retrieved_docs"],     "HyDE    (hypothetical doc embedded)")

    # ── Overlap analysis ──
    analysis = data["analysis"]
    print(f"\n  📊 Overlap Analysis:")
    print(f"     Shared docs       : {analysis['overlap_count']} / {analysis['total_docs_compared']}")
    print(f"     Unique to Standard: {analysis['unique_to_standard']}")
    print(f"     Unique to HyDE    : {analysis['unique_to_hyde']}")

    # ── Answers ──
    print(f"\n  💬 Standard Answer:")
    print(f"     {data['standard']['answer'][:300]}...")
    print(f"\n  ✨ HyDE Answer:")
    print(f"     {data['hyde']['answer'][:300]}...")

    print()


def main():
    separator("═")
    print("  HyDE vs Standard RAG — Side-by-Side Test")
    separator("═")
    print()

    # Health check
    health = requests.get(f"{API_URL}/health")
    if health.status_code != 200:
        print("❌ API not running. Start with: uvicorn main:app --reload")
        return

    h = health.json()
    print(f"  ✅ API healthy | Pinecone vectors: {h.get('pinecone_vectors', 0)}")
    print()

    if h.get("pinecone_vectors", 0) == 0:
        print("  ⚠️  No vectors found. Run: python ingest.py first.")
        return

    for query in TEST_QUERIES:
        run_compare(query)
        separator()
        print()


if __name__ == "__main__":
    main()
