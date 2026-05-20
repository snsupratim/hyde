"""
seed_products.py
────────────────
Seeds 30 sample Indian ecommerce products into Supabase,
then generates and stores Gemini embeddings for each.

Run: python scripts/seed_products.py
"""

import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

PRODUCTS = [
    # ── Winter / Clothing ──────────────────────────────────────────────────
    {"name": "Woodland Men's Winter Jacket",        "category": "clothing",
     "price": 1899, "brand": "Woodland", "stock": 45, "tags": ["winter","jacket","men"],
     "image_url": "https://via.placeholder.com/300x300?text=Woodland+Jacket",
     "description": "Thick polyester winter jacket with fleece lining, wind-resistant outer shell, and two side pockets. Ideal for cold weather in hills and plains."},

    {"name": "Mufti Quilted Puffer Jacket",         "category": "clothing",
     "price": 2499, "brand": "Mufti", "stock": 30, "tags": ["winter","puffer","men"],
     "image_url": "https://via.placeholder.com/300x300?text=Puffer+Jacket",
     "description": "Lightweight quilted puffer jacket with down-like fill, water-repellent coating, and packable design. Perfect for travel in winter months."},

    {"name": "Allen Solly Women's Wool Sweater",    "category": "clothing",
     "price": 899, "brand": "Allen Solly", "stock": 60, "tags": ["winter","sweater","women","warm"],
     "image_url": "https://via.placeholder.com/300x300?text=Wool+Sweater",
     "description": "100% merino wool round-neck sweater for women. Extremely warm, soft against skin, and suitable for office and casual wear in winter."},

    {"name": "FabIndia Cotton Shawl",               "category": "clothing",
     "price": 499, "brand": "FabIndia", "stock": 80, "tags": ["shawl","warm","winter","unisex"],
     "image_url": "https://via.placeholder.com/300x300?text=Cotton+Shawl",
     "description": "Hand-woven cotton shawl with traditional block print. Provides moderate warmth and is lightweight enough to carry daily."},

    {"name": "Biba Women's Fleece Hoodie",          "category": "clothing",
     "price": 649, "brand": "Biba", "stock": 55, "tags": ["hoodie","fleece","women","winter"],
     "image_url": "https://via.placeholder.com/300x300?text=Fleece+Hoodie",
     "description": "Soft fleece hoodie with kangaroo pocket and drawstring hood. Great layering piece for cold evenings and morning jogs."},

    {"name": "Peter England Men's Thermal Set",     "category": "clothing",
     "price": 449, "brand": "Peter England", "stock": 100, "tags": ["thermal","inner","men","winter"],
     "image_url": "https://via.placeholder.com/300x300?text=Thermal+Set",
     "description": "Top and bottom thermal innerwear set with ribbed knit fabric. Retains body heat effectively under any outfit."},

    # ── Footwear ───────────────────────────────────────────────────────────
    {"name": "Woodland Men's Snow Boots",           "category": "footwear",
     "price": 2199, "brand": "Woodland", "stock": 25, "tags": ["boots","snow","men","winter"],
     "image_url": "https://via.placeholder.com/300x300?text=Snow+Boots",
     "description": "Waterproof leather snow boots with thick rubber sole, ankle support, and anti-slip grip. Suitable for hill stations and rainy seasons."},

    {"name": "Bata Women's Winter Ankle Boots",     "category": "footwear",
     "price": 1299, "brand": "Bata", "stock": 40, "tags": ["boots","women","ankle","warm"],
     "image_url": "https://via.placeholder.com/300x300?text=Ankle+Boots",
     "description": "Stylish ankle boots with faux-fur lining for warmth. Block heel and side zipper. Works well with jeans and kurtas."},

    {"name": "Puma Men's Running Shoes",            "category": "footwear",
     "price": 2799, "brand": "Puma", "stock": 70, "tags": ["running","sports","men","shoes"],
     "image_url": "https://via.placeholder.com/300x300?text=Running+Shoes",
     "description": "Lightweight mesh running shoes with EVA midsole cushioning and breathable upper. Designed for road running and gym workouts."},

    {"name": "Relaxo Sparx Casual Sneakers",        "category": "footwear",
     "price": 699, "brand": "Relaxo", "stock": 90, "tags": ["sneakers","casual","men","budget"],
     "image_url": "https://via.placeholder.com/300x300?text=Sneakers",
     "description": "Budget-friendly canvas sneakers with rubber sole. Comfortable for daily college and office use."},

    # ── Electronics ────────────────────────────────────────────────────────
    {"name": "boAt Rockerz 450 Bluetooth Headphones","category": "electronics",
     "price": 1299, "brand": "boAt", "stock": 120, "tags": ["headphones","bluetooth","music","budget"],
     "image_url": "https://via.placeholder.com/300x300?text=Headphones",
     "description": "Wireless over-ear headphones with 15-hour battery, 40mm drivers, and foldable design. Comes with aux cable for wired use."},

    {"name": "Realme Buds Air 3 TWS Earbuds",      "category": "electronics",
     "price": 1799, "brand": "Realme", "stock": 85, "tags": ["earbuds","tws","noise-cancelling"],
     "image_url": "https://via.placeholder.com/300x300?text=Earbuds",
     "description": "True wireless earbuds with active noise cancellation, 30-hour total battery, and IPX5 water resistance. Ideal for commuting."},

    {"name": "Mi 10000mAh Power Bank",              "category": "electronics",
     "price": 899, "brand": "Mi", "stock": 200, "tags": ["powerbank","charger","budget","travel"],
     "image_url": "https://via.placeholder.com/300x300?text=Power+Bank",
     "description": "Slim 10000mAh power bank with dual USB output, 18W fast charging, and LED indicator. Lightweight and TSA-approved for travel."},

    {"name": "Philips Smart LED Bulb 9W",           "category": "electronics",
     "price": 349, "brand": "Philips", "stock": 300, "tags": ["led","bulb","smart","home"],
     "image_url": "https://via.placeholder.com/300x300?text=LED+Bulb",
     "description": "WiFi-enabled smart LED bulb with 16 million colors, voice control support, and 25000-hour lifespan."},

    {"name": "Syska LED Strip Lights 5m",           "category": "electronics",
     "price": 599, "brand": "Syska", "stock": 150, "tags": ["led","strip","rgb","room"],
     "image_url": "https://via.placeholder.com/300x300?text=LED+Strip",
     "description": "5-meter RGB LED strip with remote control, self-adhesive backing, and waterproof coating. Great for room decoration."},

    # ── Kitchen & Home ─────────────────────────────────────────────────────
    {"name": "Prestige Electric Kettle 1.5L",       "category": "kitchen",
     "price": 799, "brand": "Prestige", "stock": 110, "tags": ["kettle","electric","kitchen","tea"],
     "image_url": "https://via.placeholder.com/300x300?text=Kettle",
     "description": "1500W electric kettle with auto-shutoff, stainless steel interior, and 360-degree swivel base. Boils 1.5L in under 3 minutes."},

    {"name": "Hawkins Pressure Cooker 5L",          "category": "kitchen",
     "price": 1299, "brand": "Hawkins", "stock": 65, "tags": ["cooker","pressure","kitchen","cooking"],
     "image_url": "https://via.placeholder.com/300x300?text=Pressure+Cooker",
     "description": "Hard anodised aluminium pressure cooker with safety valve and non-stick inner. Suitable for all stovetops."},

    {"name": "Milton Thermosteel Flask 500ml",      "category": "kitchen",
     "price": 549, "brand": "Milton", "stock": 180, "tags": ["flask","hot","cold","travel"],
     "image_url": "https://via.placeholder.com/300x300?text=Flask",
     "description": "Double-walled vacuum insulated flask that keeps beverages hot for 24 hours and cold for 12 hours. BPA-free lid."},

    {"name": "Pigeon Non-stick Cookware Set 3pc",   "category": "kitchen",
     "price": 999, "brand": "Pigeon", "stock": 75, "tags": ["cookware","non-stick","kitchen","set"],
     "image_url": "https://via.placeholder.com/300x300?text=Cookware+Set",
     "description": "3-piece non-stick cookware set including fry pan, kadhai, and sauce pan. Induction and gas compatible."},

    # ── Books ──────────────────────────────────────────────────────────────
    {"name": "Atomic Habits by James Clear",        "category": "books",
     "price": 349, "brand": "Penguin", "stock": 500, "tags": ["self-help","habits","productivity"],
     "image_url": "https://via.placeholder.com/300x300?text=Atomic+Habits",
     "description": "Bestselling self-help book covering the science of habit formation, tiny improvements, and compound growth for personal development."},

    {"name": "The Psychology of Money",             "category": "books",
     "price": 299, "brand": "HarperCollins", "stock": 400, "tags": ["finance","money","psychology"],
     "image_url": "https://via.placeholder.com/300x300?text=Psychology+Money",
     "description": "Explores how people think about money — timeless lessons on wealth, greed, and happiness by Morgan Housel."},

    {"name": "Deep Work by Cal Newport",            "category": "books",
     "price": 329, "brand": "Piatkus", "stock": 350, "tags": ["productivity","focus","work"],
     "image_url": "https://via.placeholder.com/300x300?text=Deep+Work",
     "description": "Argues that the ability to focus without distraction is the superpower of the 21st century. Practical strategies for deep concentration."},

    # ── Sports & Fitness ───────────────────────────────────────────────────
    {"name": "Boldfit Resistance Bands Set",        "category": "fitness",
     "price": 399, "brand": "Boldfit", "stock": 200, "tags": ["resistance","bands","workout","home-gym"],
     "image_url": "https://via.placeholder.com/300x300?text=Resistance+Bands",
     "description": "Set of 5 latex resistance bands in different strengths. Ideal for home workouts, stretching, and physiotherapy."},

    {"name": "Cosco Yoga Mat 6mm",                 "category": "fitness",
     "price": 549, "brand": "Cosco", "stock": 160, "tags": ["yoga","mat","exercise","gym"],
     "image_url": "https://via.placeholder.com/300x300?text=Yoga+Mat",
     "description": "Anti-slip 6mm thick yoga mat with carrying strap. Made from NBR foam, suitable for yoga, Pilates, and floor exercises."},

    {"name": "Nivia Football Size 5",               "category": "fitness",
     "price": 699, "brand": "Nivia", "stock": 90, "tags": ["football","sports","outdoor","kids"],
     "image_url": "https://via.placeholder.com/300x300?text=Football",
     "description": "Standard size 5 rubber football suitable for training and casual play. Durable and performs well on grass and concrete."},

    # ── Personal Care ──────────────────────────────────────────────────────
    {"name": "Biotique Bio Walnut Bark Shampoo",   "category": "personal_care",
     "price": 199, "brand": "Biotique", "stock": 250, "tags": ["shampoo","hair","natural","ayurvedic"],
     "image_url": "https://via.placeholder.com/300x300?text=Shampoo",
     "description": "Ayurvedic shampoo with walnut bark, bhringraj, and soap nut for strengthening hair. Suitable for all hair types."},

    {"name": "Nivea Men Face Wash 100ml",           "category": "personal_care",
     "price": 149, "brand": "Nivea", "stock": 400, "tags": ["face-wash","men","skin","daily"],
     "image_url": "https://via.placeholder.com/300x300?text=Face+Wash",
     "description": "Oil-control face wash with micro-granules that deep cleans pores and removes excess oil. Leaves skin fresh and matte."},

    {"name": "Bombay Shaving Company Shaving Kit",  "category": "personal_care",
     "price": 499, "brand": "BSC", "stock": 130, "tags": ["shaving","grooming","men","gift"],
     "image_url": "https://via.placeholder.com/300x300?text=Shaving+Kit",
     "description": "Complete shaving kit with pre-shave scrub, shaving cream, and post-shave balm. Gives a smooth, irritation-free shave."},

    # ── Accessories ────────────────────────────────────────────────────────
    {"name": "Titan Analog Watch for Men",          "category": "accessories",
     "price": 1999, "brand": "Titan", "stock": 50, "tags": ["watch","men","analog","formal"],
     "image_url": "https://via.placeholder.com/300x300?text=Titan+Watch",
     "description": "Classic analog wristwatch with stainless steel case, leather strap, and sapphire crystal glass. Water resistant up to 50m."},

    {"name": "Fastrack Backpack 25L",               "category": "accessories",
     "price": 1499, "brand": "Fastrack", "stock": 70, "tags": ["backpack","travel","college","bag"],
     "image_url": "https://via.placeholder.com/300x300?text=Backpack",
     "description": "25-litre polyester backpack with laptop sleeve, multiple pockets, and padded shoulder straps. Ideal for college, travel, and office."},

    {"name": "Da Milano Leather Wallet",            "category": "accessories",
     "price": 799, "brand": "Da Milano", "stock": 100, "tags": ["wallet","leather","men","slim"],
     "image_url": "https://via.placeholder.com/300x300?text=Wallet",
     "description": "Slim genuine leather bifold wallet with 6 card slots, 2 currency compartments, and RFID blocking layer."},
]


def embed_text(text: str) -> str:
    """
    Returns embedding as a string like '[0.1, 0.2, ...]'
    Supabase REST API needs halfvec passed as a string — it auto-casts.
    """
    vector = embedder.embed_query(text)
    return "[" + ",".join(str(round(v, 8)) for v in vector) + "]"


def seed():
    print(f"🛒 Seeding {len(PRODUCTS)} products into Supabase...\n")

    for i, p in enumerate(PRODUCTS, 1):
        # 1. Insert product
        res = supabase.table("products").insert({
            "name":        p["name"],
            "category":    p["category"],
            "price":       p["price"],
            "description": p["description"],
            "image_url":   p.get("image_url", ""),
            "stock":       p.get("stock", 0),
            "brand":       p.get("brand", ""),
            "tags":        p.get("tags", []),
        }).execute()

        product_id = res.data[0]["id"]

        # 2. Generate embedding — rich text combining all product fields
        embed_text_content = (
            f"Product: {p['name']}. "
            f"Category: {p['category']}. "
            f"Brand: {p.get('brand','')}. "
            f"Price: ₹{p['price']}. "
            f"{p['description']}"
        )
        vector_str = embed_text(embed_text_content)

        # 3. Store embedding as string — Supabase casts to halfvec automatically
        supabase.table("product_embeddings").insert({
            "product_id": product_id,
            "embedding":  vector_str,
        }).execute()

        print(f"  [{i:02d}/{len(PRODUCTS)}] ✅ {p['name']} (₹{p['price']})")

        # Rate limit buffer
        if i % 5 == 0:
            time.sleep(1)

    print(f"\n✅ Done! {len(PRODUCTS)} products seeded with embeddings.")


if __name__ == "__main__":
    seed()