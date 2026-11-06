-- Fathom meetings indexed into document_chunks.
-- Rows remember each recording so a later run can skip unchanged transcripts
-- and delete meetings that leave the API key's accessible set.

CREATE TABLE IF NOT EXISTS fathom_sync_state (
    recording_id    TEXT        PRIMARY KEY,
    source_file     TEXT        NOT NULL,
    content_sha256  TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    fathom_url      TEXT        NOT NULL,
    join_url        TEXT        NOT NULL DEFAULT '',
    attendees       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    recorded_at     TIMESTAMPTZ,
    revised_at      TIMESTAMPTZ,
    last_synced_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT fathom_sync_state_hash_nonempty CHECK (length(btrim(content_sha256)) > 0)
);

CREATE INDEX IF NOT EXISTS fathom_sync_state_source_idx
    ON fathom_sync_state (source_file);
