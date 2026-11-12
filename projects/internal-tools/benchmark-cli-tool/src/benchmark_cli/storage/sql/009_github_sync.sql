-- Public GitHub files and pull requests indexed into document_chunks.
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
