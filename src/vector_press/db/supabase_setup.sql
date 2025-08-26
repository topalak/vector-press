  -- Enable the pgvector extension
  create extension if not exists vector;

  -- Table 1: Guardian Articles Metadata
  create table guardian_articles (
      id bigserial primary key,                  -- Auto-incrementing primary key
      article_id varchar not null unique,       -- Custom article identifier (e.g., technology/2025/aug/05/google-step-artificial-general-intelligence-deepmind-agi)
      title varchar not null,
      headline varchar not null,
      section varchar not null,
      publication_date timestamp with time zone not null,
      url varchar not null,
      summary text,                              -- standfirst
      body_text text,                            -- Full article body text
      trail_text text,
      word_count integer not null,
      char_count integer not null,
      fetch_time timestamp with time zone not null, -- When article was fetched from API
      search_metadata jsonb default '{}'::jsonb, -- Track which queries found this
      created_at timestamp with time zone default timezone('utc'::text, now()) not null
  );

  -- Table 2: Article Chunks with Embeddings
  create table article_chunks (
      id bigserial primary key,                  -- Auto-incrementing primary key
      article_id varchar not null references guardian_articles(article_id) on delete cascade,  -- Custom article identifier (e.g., technology/2025/aug/05/google-step-artificial-general-intelligence-deepmind-agi)
      chunk_number integer not null,
      content text not null,                     -- Chunk of full_text
      embedding vector(768) not null,          -- Nomic embedding
      created_at timestamp with time zone default timezone('utc'::text, now()) not null,

      -- Prevent duplicate chunks for same article
      unique(article_id, chunk_number)
  );

  -- Function to search chunks with metadata
  -- or just create function
  create or replace function match_article_chunks (
    query_embedding vector(768),
    match_count int default 10,
    section_filter varchar default null
  ) returns table (
    chunk_id bigint,
    article_id varchar,
    chunk_number integer,
    content text,
    title varchar,
    section varchar,
    url varchar,
    publication_date timestamp with time zone,
    similarity float
  )
  language plpgsql
  as $$
  begin
    return query
    select
      ac.id as chunk_id,
      ac.article_id,
      ac.chunk_number,
      ac.content,
      ga.title,
      ga.section,
      ga.url,
      ga.publication_date,
      1 - (ac.embedding <=> query_embedding) as similarity
    from article_chunks ac
    join guardian_articles ga on ac.article_id = ga.article_id
    where (section_filter is null or ga.section = section_filter)
    order by ac.embedding <=> query_embedding
    limit match_count;
  end;
  $$;
  -- Enable RLS
  alter table guardian_articles enable row level security;
  alter table article_chunks enable row level security;

  -- Public read access policies
  create policy "Allow public read access on articles"
    on guardian_articles for select to public using (true);

  create policy "Allow public read access on chunks"
    on article_chunks for select to public using (true);

  -- Insert policies (adjust based on your auth needs)
  create policy "Allow public insert on articles"
    on guardian_articles for insert to public with check (true);

  create policy "Allow public insert on chunks"
    on article_chunks for insert to public with check (true);
