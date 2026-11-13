-- Public GitHub files and pull requests. Code chunks live in github_chunks,
-- never in document_chunks, so no existing query, index, or ranking sees them.
-- The commit SHA is the tree that was indexed. File digests skip unchanged files.

CREATE TABLE IF NOT EXISTS github_sync_state (
    repo         TEXT        PRIMARY KEY,
    commit_sha   TEXT        NOT NULL,
    synced_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT github_sync_state_sha_nonempty CHECK (length(btrim(commit_sha)) > 0)
);

CREATE TABLE IF NOT EXISTS github_file_state (
    repo         TEXT        NOT NULL,
    path         TEXT        NOT NULL,
    source_file  TEXT        NOT NULL,
    blob_sha     TEXT        NOT NULL,
    synced_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (repo, path),
    CONSTRAINT github_file_state_sha_nonempty CHECK (length(btrim(blob_sha)) > 0)
);

-- Ids start at 1e9 so a code chunk id never equals a document_chunks id.
CREATE SEQUENCE IF NOT EXISTS github_chunks_id_seq START WITH 1000000000;

CREATE TABLE IF NOT EXISTS github_chunks (
    id             BIGINT       PRIMARY KEY DEFAULT nextval('github_chunks_id_seq'),
    repo           TEXT         NOT NULL,
    path           TEXT         NOT NULL,
    kind           TEXT         NOT NULL CHECK (kind IN ('code', 'config', 'workflow', 'doc', 'pull_request')),
    chunk_index    INTEGER      NOT NULL,
    symbol         TEXT         NOT NULL DEFAULT '',
    header_words   TEXT         NOT NULL,
    start_line     INTEGER      NOT NULL,
    end_line       INTEGER      NOT NULL,
    chunk_text     TEXT         NOT NULL,
    content_sha256 CHAR(64)     NOT NULL,
    token_count    INTEGER      NOT NULL,
    embedding      vector(1536) NOT NULL,
    metadata       JSONB        NOT NULL DEFAULT '{}'::jsonb,
    header_vector  tsvector     GENERATED ALWAYS AS (to_tsvector('english', header_words)) STORED,
    search_vector  tsvector     GENERATED ALWAYS AS (
        setweight(to_tsvector('english', header_words), 'A') ||
        to_tsvector('english', regexp_replace(chunk_text, '[^[:alnum:]]+', ' ', 'g'))
    ) STORED,

    UNIQUE (repo, path, chunk_index)
);

CREATE INDEX IF NOT EXISTS github_chunks_embedding_hnsw
    ON github_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS github_chunks_search_gin ON github_chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS github_chunks_header_gin ON github_chunks USING gin (header_vector);
