-- Add prompt and errors columns to tasks table.
ALTER TABLE tasks ADD COLUMN prompt TEXT;
ALTER TABLE tasks ADD COLUMN errors JSONB;
