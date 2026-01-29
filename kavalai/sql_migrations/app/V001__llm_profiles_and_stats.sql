-- LLM profile denotes the provider, model, api_key and
CREATE TABLE llm_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    api_key TEXT,
    base_url TEXT,
    config JSONB, -- Additional config parameters like mode, temperature etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);


-- LLM call stats capture the metrics and costs of llm calls.
CREATE TABLE llm_call_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    llm_profile_id UUID REFERENCES llm_profiles(id) ON DELETE SET NULL,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    request_data JSONB,
    response_data JSONB,
    response_code INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    duration_seconds NUMERIC(10, 6),
    cost NUMERIC(10, 6),
    currency CHAR(3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_llm_call_stats_profile_id ON llm_call_stats(llm_profile_id);
CREATE INDEX idx_llm_call_stats_agent_id ON llm_call_stats(agent_id);
