-- ══════════════════════════════════════════════════════
--  Ecommerce HyDE RAG — Supabase Schema (fixed)
--  pgvector fix: halfvec(3072) supports ivfflat index
--  Run this in Supabase → SQL Editor
-- ══════════════════════════════════════════════════════

-- Enable pgvector extension
create extension if not exists vector;

-- ── Products table ──────────────────────────────────────
create table if not exists products (
  id          uuid primary key default gen_random_uuid(),
  name        text not null,
  category    text not null,
  price       numeric(10,2) not null,
  description text not null,
  image_url   text,
  stock       integer default 0,
  brand       text,
  tags        text[],
  created_at  timestamptz default now()
);

-- ── Embeddings table ────────────────────────────────────
-- halfvec instead of vector — supports ivfflat up to 4000 dims
create table if not exists product_embeddings (
  id          uuid primary key default gen_random_uuid(),
  product_id  uuid references products(id) on delete cascade,
  embedding   halfvec(3072) not null,
  created_at  timestamptz default now()
);

-- ── Search logs table ───────────────────────────────────
create table if not exists search_logs (
  id               uuid primary key default gen_random_uuid(),
  query            text not null,
  method           text not null,
  hypothetical_doc text,
  result_ids       uuid[],
  result_scores    float[],
  created_at       timestamptz default now()
);

-- ── Vector similarity search function ──────────────────
-- Accepts halfvec, uses cosine distance operator <=>
create or replace function match_products(
  query_embedding halfvec(3072),
  match_count     int default 5,
  filter_category text default null
)
returns table (
  product_id   uuid,
  name         text,
  category     text,
  price        numeric,
  description  text,
  image_url    text,
  brand        text,
  tags         text[],
  stock        integer,
  similarity   float
)
language sql stable
as $$
  select
    p.id        as product_id,
    p.name,
    p.category,
    p.price,
    p.description,
    p.image_url,
    p.brand,
    p.tags,
    p.stock,
    1 - (pe.embedding <=> query_embedding) as similarity
  from product_embeddings pe
  join products p on p.id = pe.product_id
  where (filter_category is null or p.category = filter_category)
  order by pe.embedding <=> query_embedding
  limit match_count;
$$;

-- ── Indexes ─────────────────────────────────────────────
-- halfvec supports ivfflat up to 4000 dims — this works now
create index if not exists idx_product_embeddings_vector
  on product_embeddings using ivfflat (embedding halfvec_cosine_ops)
  with (lists = 100);

create index if not exists idx_products_category on products(category);
create index if not exists idx_search_logs_created on search_logs(created_at desc);