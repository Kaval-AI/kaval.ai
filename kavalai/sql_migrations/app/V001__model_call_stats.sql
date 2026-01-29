CREATE TABLE model_call_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- What was called
    call_type TEXT NOT NULL CHECK (call_type IN ('llm', 'embedding')),
    model TEXT NOT NULL,

    -- Optional association (kept as a plain UUID to avoid FK dependency)
    agent_id UUID,

    -- Payloads / result
    request_data JSONB,
    response_data JSONB,
    response_code INTEGER,

    -- Token accounting (LLM calls typically fill prompt/completion; embeddings may only fill total_tokens)
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,

    -- Embedding-specific
    batch_size INTEGER,

    -- Common metrics
    duration_seconds NUMERIC(10, 6),
    cost NUMERIC(10, 6),
    currency CHAR(3),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_model_call_stats_call_type ON model_call_stats(call_type);
CREATE INDEX idx_model_call_stats_model     ON model_call_stats(model);
CREATE INDEX idx_model_call_stats_agent_id  ON model_call_stats(agent_id);
CREATE INDEX idx_model_call_stats_created_at ON model_call_stats(created_at);
