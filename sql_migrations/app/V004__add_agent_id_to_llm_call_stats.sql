-- Add agent_id to llm_call_stats for better filtering
ALTER TABLE llm_call_stats ADD COLUMN agent_id UUID REFERENCES agents(id) ON DELETE CASCADE;
CREATE INDEX idx_llm_call_stats_agent_id ON llm_call_stats(agent_id);
