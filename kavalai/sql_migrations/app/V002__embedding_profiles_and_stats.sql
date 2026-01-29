-- Embedding profiles denote embedding providers and models.
CREATE TABLE embedding_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    base_url TEXT,
    embedding_size INTEGER,
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);


CREATE TABLE embedding_call_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    embedding_profile_id UUID REFERENCES embedding_profiles(id) ON DELETE SET NULL,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    request_data JSONB,
    response_data JSONB,
    response_code INTEGER,
    batch_size INTEGER,
    total_tokens INTEGER,
    duration_seconds NUMERIC(10, 6),
    cost NUMERIC(10, 6),
    currency CHAR(3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_embedding_call_stats_profile_id ON embedding_call_stats(embedding_profile_id);
CREATE INDEX idx_embedding_call_stats_agent_id ON embedding_call_stats(agent_id);
