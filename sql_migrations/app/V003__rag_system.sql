-- Enable the pgvector extension to work with embeddings in the public schema
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

-- Embedding Profiles: Stores configuration for computing embeddings
CREATE TABLE embedding_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    api_key TEXT,
    base_url TEXT,
    credentials JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON COLUMN embedding_profiles.provider IS 'Provider name (e.g., openai, google, cohere)';
COMMENT ON COLUMN embedding_profiles.model_name IS 'Model name (e.g., text-embedding-3-small, text-embedding-004)';

-- RAG Index: Stores data and its embeddings for retrieval
CREATE TABLE rag_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    embedding_profile_id UUID REFERENCES embedding_profiles(id) ON DELETE CASCADE,

    -- pgvector columns of different sizes.
    -- Common embedding sizes: 384 (all-MiniLM-L6-v2), 768 (nomic-embed-text), 1536 (text-embedding-3-small), 3072 (text-embedding-3-large)
    embedding_384 public.vector(384),
    embedding_768 public.vector(768),
    embedding_1536 public.vector(1536),
    embedding_3072 public.vector(3072),

    mime_type TEXT NOT NULL,
    text_content TEXT,
    json_content JSONB,
    binary_content BYTEA,

    metadata JSONB NOT NULL DEFAULT '{}',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_rag_index_embedding_profile_id ON rag_index(embedding_profile_id);
