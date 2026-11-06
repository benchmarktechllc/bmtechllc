CREATE TABLE IF NOT EXISTS discord_source_messages (
    channel_id          TEXT        NOT NULL,
    message_id          TEXT        NOT NULL,
    channel_name        TEXT        NOT NULL,
    author_name         TEXT        NOT NULL,
    message_text        TEXT        NOT NULL,
    sent_at             TIMESTAMPTZ NOT NULL,
    edited_at           TIMESTAMPTZ,
    reply_to_message_id TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (channel_id, message_id),
    CONSTRAINT discord_source_messages_text_nonempty CHECK (length(btrim(message_text)) > 0)
);

CREATE INDEX IF NOT EXISTS discord_source_messages_channel_sent_idx
    ON discord_source_messages (channel_id, sent_at, message_id);

CREATE TABLE IF NOT EXISTS discord_sync_state (
    channel_id      TEXT        PRIMARY KEY,
    channel_name    TEXT        NOT NULL,
    last_message_id TEXT,
    reindex_required BOOLEAN     NOT NULL DEFAULT FALSE,
    last_synced_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE discord_sync_state
    ADD COLUMN IF NOT EXISTS reindex_required BOOLEAN NOT NULL DEFAULT FALSE;
