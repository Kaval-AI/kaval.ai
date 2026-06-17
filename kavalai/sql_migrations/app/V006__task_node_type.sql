-- Add node_type to tasks so v2 workflow node executions record their node kind
-- (start/end/llm/agent/function/if/switch).
ALTER TABLE tasks ADD COLUMN node_type TEXT;
