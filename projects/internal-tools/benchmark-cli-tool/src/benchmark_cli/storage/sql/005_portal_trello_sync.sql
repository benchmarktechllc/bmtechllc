-- Portal pages and Trello cards indexed into document_chunks.
-- These tables remember what was published so a later run can replace one
-- source and delete pages or cards that are no longer eligible.

CREATE TABLE IF NOT EXISTS portal_sync_state (
    source_file     TEXT        PRIMARY KEY,
    content_sha256  TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    section         TEXT        NOT NULL,
    updated_on      DATE,
    last_synced_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT portal_sync_state_hash_nonempty CHECK (length(btrim(content_sha256)) > 0)
);

CREATE TABLE IF NOT EXISTS trello_sync_state (
    board_id         TEXT        PRIMARY KEY,
    board_name       TEXT        NOT NULL,
    short_link       TEXT        NOT NULL,
    last_activity_at TIMESTAMPTZ,
    last_synced_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS trello_source_cards (
    board_id          TEXT        NOT NULL,
    card_id           TEXT        NOT NULL,
    source_file       TEXT        NOT NULL,
    list_name         TEXT        NOT NULL,
    card_name         TEXT        NOT NULL,
    card_url          TEXT        NOT NULL,
    labels            JSONB       NOT NULL DEFAULT '[]'::jsonb,
    due_at            TIMESTAMPTZ,
    content_sha256    TEXT        NOT NULL,
    last_activity_at  TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (board_id, card_id),
    CONSTRAINT trello_source_cards_hash_nonempty CHECK (length(btrim(content_sha256)) > 0)
);

CREATE INDEX IF NOT EXISTS trello_source_cards_source_idx
    ON trello_source_cards (source_file);
