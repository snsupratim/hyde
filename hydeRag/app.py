"""
frontend/app.py
═══════════════
Streamlit UI for Ecommerce HyDE RAG system

Features:
  ● Product search with Standard vs HyDE toggle
  ● Side-by-side compare mode showing score differences
  ● Product cards with image, price, score badge
  ● Hypothetical doc viewer (what HyDE imagined)
  ● Analytics tab with search history
"""

import streamlit as st
import requests
from typing import Optional

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopSmart — HyDE RAG",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.product-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    background: white;
    position: relative;
}
.score-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    color: white;
    margin-bottom: 8px;
}
.score-high   { background: #2e7d32; }
.score-medium { background: #f57c00; }
.score-low    { background: #c62828; }
.price-tag {
    font-size: 22px;
    font-weight: 700;
    color: #1a237e;
}
.product-name {
    font-size: 16px;
    font-weight: 600;
    color: #212121;
    margin: 6px 0 4px;
}
.product-meta {
    font-size: 12px;
    color: #757575;
    margin-bottom: 8px;
}
.product-desc {
    font-size: 13px;
    color: #424242;
    line-height: 1.5;
}
.tag-chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    background: #e3f2fd;
    color: #1565c0;
    font-size: 11px;
    margin: 2px;
}
.hyde-box {
    background: #f3e5f5;
    border-left: 4px solid #7b1fa2;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    color: #4a148c;
    margin-bottom: 16px;
}
.score-improve-positive { color: #2e7d32; font-weight: 700; }
.score-improve-negative { color: #c62828; font-weight: 700; }
.method-header-standard { color: #1565c0; font-weight: 700; }
.method-header-hyde     { color: #6a1b9a; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=30)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=60)
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def score_badge(score: float) -> str:
    css = "score-high" if score >= 0.75 else ("score-medium" if score >= 0.55 else "score-low")
    return f'<span class="score-badge {css}">score: {score:.3f}</span>'


def render_product_card(p: dict):
    tags_html = " ".join(f'<span class="tag-chip">{t}</span>' for t in (p.get("tags") or [])[:4])
    stock_text = f"✅ In stock ({p.get('stock',0)})" if p.get("stock", 0) > 0 else "❌ Out of stock"
    st.markdown(f"""
    <div class="product-card">
        {score_badge(p["similarity"])}
        <div class="price-tag">₹{p['price']:,.0f}</div>
        <div class="product-name">{p['name']}</div>
        <div class="product-meta">
            {p.get('brand','')} &nbsp;·&nbsp; {p['category'].replace('_',' ').title()} &nbsp;·&nbsp; {stock_text}
        </div>
        <div class="product-desc">{p['description']}</div>
        <div style="margin-top:8px">{tags_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_product_grid(products: list[dict], cols: int = 2):
    for i in range(0, len(products), cols):
        row = st.columns(cols)
        for j, col in enumerate(row):
            if i + j < len(products):
                with col:
                    render_product_card(products[i + j])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 ShopSmart")
    st.caption("Powered by HyDE RAG")

    health = api_get("/health")
    if health and health.get("status") == "ok":
        st.success(f"✅ API connected  \n📦 {health.get('total_products', 0)} products")
    else:
        st.error("❌ API not reachable  \nStart: `uvicorn backend.main:app --reload`")

    st.divider()

    # Category filter
    cats_resp = api_get("/categories")
    categories = ["All"] + (cats_resp.get("categories", []) if cats_resp else [])
    selected_cat = st.selectbox("Filter by category", categories)
    category_param = None if selected_cat == "All" else selected_cat

    top_k = st.slider("Results to show", 3, 10, 5)

    st.divider()
    st.markdown("""
    **How HyDE works here:**
    1. You type a vague query like *"warm jacket under 500"*
    2. Groq writes a hypothetical product listing
    3. That listing gets embedded (not your query)
    4. Supabase finds products matching that embedding
    5. Score is higher → better match
    """)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_search, tab_compare, tab_analytics = st.tabs([
    "🔍 Search", "⚖️ Compare Standard vs HyDE", "📊 Analytics"
])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — Search
# ══════════════════════════════════════════════════════════════
with tab_search:
    st.header("Product Search")

    method = st.radio(
        "Retrieval method",
        ["Standard RAG", "HyDE RAG"],
        horizontal=True,
        help="HyDE generates a hypothetical product description first, then searches — usually more accurate for vague queries"
    )

    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g.  something warm for winter under ₹500  /  running shoes for flat feet  /  gift for dad",
    )

    if st.button("🔍 Search", type="primary", use_container_width=True) and query:
        endpoint = "/search/hyde" if method == "HyDE RAG" else "/search/standard"

        with st.spinner("Searching products..."):
            result = api_post(endpoint, {
                "query":    query,
                "top_k":    top_k,
                "category": category_param,
            })

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            # HyDE hypothetical doc
            if result.get("hypothetical_doc"):
                st.markdown(f"""
                <div class="hyde-box">
                <strong>🤖 What HyDE imagined the product looks like:</strong><br><br>
                {result['hypothetical_doc']}
                </div>
                """, unsafe_allow_html=True)

            # Recommendation
            st.info(f"💡 **Shopping assistant says:** {result.get('recommendation', '')}")

            st.subheader(f"Found {len(result['products'])} products")
            render_product_grid(result["products"])


# ══════════════════════════════════════════════════════════════
#  TAB 2 — Compare
# ══════════════════════════════════════════════════════════════
with tab_compare:
    st.header("Standard vs HyDE — Side-by-Side")
    st.caption("Run both retrieval methods on the same query and see what's different")

    compare_query = st.text_input(
        "Enter a query to compare",
        placeholder="e.g.  something warm for winter  /  budget earphones  /  yoga equipment",
        key="compare_input",
    )

    if st.button("⚖️ Compare", type="primary", use_container_width=True) and compare_query:
        with st.spinner("Running both methods... (takes ~10 seconds)"):
            result = api_post("/search/compare", {
                "query":    compare_query,
                "top_k":    top_k,
                "category": category_param,
            })

        if "error" in result:
            st.error(result["error"])
        else:
            analysis = result["analysis"]
            std      = result["standard"]
            hyde     = result["hyde"]

            # ── Score summary ──────────────────────────────────────────
            st.subheader("📊 Score Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Standard avg score", f"{std['avg_score']:.3f}")
            m2.metric("HyDE avg score",     f"{hyde['avg_score']:.3f}",
                      delta=f"+{analysis['score_improvement']:.3f}" if analysis['score_improvement'] > 0
                            else f"{analysis['score_improvement']:.3f}")
            m3.metric("Overlapping products", f"{analysis['overlap_count']} / {analysis['total_compared']}")
            m4.metric("Unique to HyDE",       len(analysis['unique_to_hyde']))

            # ── HyDE hypothetical doc ──────────────────────────────────
            with st.expander("🤖 See what HyDE imagined", expanded=True):
                st.markdown(f"""
                <div class="hyde-box">
                {hyde['hypothetical_doc']}
                </div>
                """, unsafe_allow_html=True)

            # ── Side by side products ──────────────────────────────────
            col_std, col_hyde = st.columns(2)

            with col_std:
                st.markdown('<p class="method-header-standard">🔵 Standard RAG</p>', unsafe_allow_html=True)
                st.caption(f"Avg similarity: {std['avg_score']:.3f}")
                st.info(f"💬 {std['recommendation']}")
                for p in std["products"]:
                    render_product_card(p)

            with col_hyde:
                st.markdown('<p class="method-header-hyde">🟣 HyDE RAG</p>', unsafe_allow_html=True)
                st.caption(f"Avg similarity: {hyde['avg_score']:.3f}")
                st.info(f"💬 {hyde['recommendation']}")
                for p in hyde["products"]:
                    render_product_card(p)

            # ── Product-level score diff ───────────────────────────────
            st.subheader("📈 Per-product Score Comparison")
            std_map  = {p["name"]: p["similarity"] for p in std["products"]}
            hyde_map = {p["name"]: p["similarity"] for p in hyde["products"]}
            all_names = list(dict.fromkeys(
                [p["name"] for p in std["products"]] + [p["name"] for p in hyde["products"]]
            ))

            for name in all_names:
                s = std_map.get(name)
                h = hyde_map.get(name)
                if s and h:
                    diff = h - s
                    col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
                    col_a.text(name[:45])
                    col_b.metric("Standard", f"{s:.3f}")
                    col_c.metric("HyDE", f"{h:.3f}", delta=f"{diff:+.3f}")
                    only = ""
                elif s:
                    col_a, col_b = st.columns([3, 2])
                    col_a.text(name[:45])
                    col_b.caption("Only in Standard")
                else:
                    col_a, col_b = st.columns([3, 2])
                    col_a.text(name[:45])
                    col_b.caption("✨ Only in HyDE")


# ══════════════════════════════════════════════════════════════
#  TAB 3 — Analytics
# ══════════════════════════════════════════════════════════════
with tab_analytics:
    st.header("📊 Search Analytics")
    st.caption("Recent searches logged from the system")

    if st.button("🔄 Refresh"):
        st.rerun()

    logs_resp = api_get("/analytics", {"limit": 30})
    if not logs_resp or not logs_resp.get("logs"):
        st.info("No searches yet. Run some searches first.")
    else:
        logs = logs_resp["logs"]

        # Summary
        total   = len(logs)
        std_cnt = sum(1 for l in logs if l["method"] == "standard")
        hyd_cnt = sum(1 for l 
                      in logs if l["method"] == "hyde")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total searches", total)
        c2.metric("Standard searches", std_cnt)
        c3.metric("HyDE searches", hyd_cnt)

        st.divider()

        # Log table
        for log in logs[:20]:
            with st.expander(
                f"{'🔵' if log['method']=='standard' else '🟣'} "
                f"[{log['method'].upper()}]  \"{log['query']}\"  "
                f"— {log['created_at'][:16]}"
            ):
                scores = log.get("result_scores") or []
                if scores:
                    avg = sum(scores) / len(scores)
                    st.metric("Avg similarity", f"{avg:.3f}")

                if log.get("hypothetical_doc"):
                    st.markdown("**HyDE hypothetical doc:**")
                    st.markdown(f"""
                    <div class="hyde-box">{log['hypothetical_doc']}</div>
                    """, unsafe_allow_html=True)

                if scores:
                    st.write("Individual scores:", scores)
