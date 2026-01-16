-- LLM Profiles: Stores credentials and configuration for different LLM providers
CREATE TABLE llm_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    api_key TEXT,
    base_url TEXT,
    default_mode TEXT,
    credentials JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON COLUMN llm_profiles.provider IS 'Provider name (e.g., openai, anthropic, google, groq)';
COMMENT ON COLUMN llm_profiles.model_name IS 'Model name (e.g., gpt-4, claude-3-sonnet-20240229)';
COMMENT ON COLUMN llm_profiles.default_mode IS 'Default instructor mode (e.g., TOOLS, JSON, MD_JSON, etc.)';

-- LLM Call Stats: Records statistics for individual LLM calls
CREATE TABLE llm_call_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    llm_profile_id UUID REFERENCES llm_profiles(id) ON DELETE SET NULL ,
    name TEXT,
    response_code INTEGER,
    cost NUMERIC(10, 6),
    currency TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
