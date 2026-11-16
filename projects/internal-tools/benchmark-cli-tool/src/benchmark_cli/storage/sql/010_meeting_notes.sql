CREATE TABLE IF NOT EXISTS meeting_notes_posts (
    transcript_id TEXT PRIMARY KEY,
    posted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    channel_id TEXT NOT NULL,
    message_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    title TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS meeting_notes_cursor (
    id INT PRIMARY KEY CHECK (id = 1),
    watermark_at TIMESTAMPTZ NOT NULL,
    initialized_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
