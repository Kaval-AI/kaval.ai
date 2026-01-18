-- Add metrics fields to llm_call_stats
ALTER TABLE llm_call_stats ADD COLUMN prompt_tokens INTEGER;
ALTER TABLE llm_call_stats ADD COLUMN completion_tokens INTEGER;
ALTER TABLE llm_call_stats ADD COLUMN total_tokens INTEGER;
ALTER TABLE llm_call_stats ADD COLUMN duration_ms INTEGER;
ALTER TABLE llm_call_stats ADD COLUMN request_data JSONB;
ALTER TABLE llm_call_stats ADD COLUMN response_data JSONB;
