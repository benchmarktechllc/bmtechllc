-- Fireflies meetings indexed into document_chunks.
-- Rows remember each transcript so a later run can skip unchanged meetings
-- and delete meetings that leave the API key's accessible set.
-- recording_id stores the Fireflies transcript id.

CREATE TABLE IF NOT EXISTS fireflies_sync_state (
    recording_id      TEXT        PRIMARY KEY,
    source_file       TEXT        NOT NULL,
    content_sha256    TEXT        NOT NULL,
    title             TEXT        NOT NULL,
    fireflies_url     TEXT        NOT NULL,
    join_url          TEXT        NOT NULL DEFAULT '',
    attendees         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    recorded_at       TIMESTAMPTZ,
    duration_minutes  DOUBLE PRECISION,
    host              TEXT        NOT NULL DEFAULT '',
    summary_status    TEXT        NOT NULL DEFAULT '',
    last_synced_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT fireflies_sync_state_hash_nonempty CHECK (length(btrim(content_sha256)) > 0)
);

CREATE INDEX IF NOT EXISTS fireflies_sync_state_source_idx
    ON fireflies_sync_state (source_file);
