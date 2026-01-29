CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;


-- Table for storing document chunks and embeddings
CREATE TABLE rag_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    embedding_profile_id UUID REFERENCES embedding_profiles(id) ON DELETE SET NULL,
    collection_name TEXT NOT NULL, -- Logical grouping (e.g., "kb_finance", "user_123_docs")
    source_id TEXT NOT NULL, -- This represents for the unique identifier of the source document.
    content TEXT,
    embedding_size INTEGER NOT NULL,
    embedding VECTOR,
    metadata JSONB, -- Metadata for filtering (source URL, page number, document_id, etc.)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for standard metadata/collection filtering
CREATE INDEX idx_rag_index_embedding_profile_id ON rag_index(embedding_profile_id);
CREATE INDEX idx_rag_index_collection ON rag_index (collection_name);
CREATE INDEX idx_rag_index_metadata ON rag_index USING gin (metadata);
CREATE INDEX idx_rag_source_id ON rag_index (source_id);

-- Create indexes for popular embedding sizes.
CREATE INDEX idx_rag_embedding_768 ON rag_index
USING hnsw ((embedding::vector(768)) vector_cosine_ops)
WHERE (embedding_size = 768);

CREATE INDEX idx_rag_embedding_1024 ON rag_index
USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
WHERE (embedding_size = 1024);

CREATE INDEX idx_rag_embedding_1536 ON rag_index
USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WHERE (embedding_size = 1536);
