-- Add profile_name and llm_profile_name columns to rag_index
ALTER TABLE rag_index ADD COLUMN embedding_profile_name TEXT;

-- Add indices for pgvector columns using HNSW
-- Note: using vector_cosine_ops which is common for RAG.
CREATE INDEX idx_rag_index_embedding_384 ON rag_index USING hnsw (embedding_384 public.vector_cosine_ops);
CREATE INDEX idx_rag_index_embedding_768 ON rag_index USING hnsw (embedding_768 public.vector_cosine_ops);
CREATE INDEX idx_rag_index_embedding_1536 ON rag_index USING hnsw (embedding_1536 public.vector_cosine_ops);

-- Not supported.
-- CREATE INDEX idx_rag_index_embedding_3072 ON rag_index USING hnsw (embedding_3072 public.vector_cosine_ops);
