-- Multi-purpose cache on backend side for storing various results.
CREATE TABLE project_cache
(
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects (id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    value      TEXT -- the cached value
);

CREATE INDEX idx_project_cache_name ON project_cache (name);
