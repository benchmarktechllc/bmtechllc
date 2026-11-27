-- 001: sessions (SPEC.md section 3). One row per /ask interaction.
-- Classifier columns stay NULL until Slice 2 fills them.

CREATE TABLE sessions (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id   text NOT NULL UNIQUE,
    guild_id         text NOT NULL,
    channel_id       text NOT NULL,
    asker_user_id    text NOT NULL,
    question         text NOT NULL,
    concerns         jsonb,
    more_concerns    boolean,
    primary_category text,
    categories       text[],
    confidence       double precision,
    status           text NOT NULL DEFAULT 'running'
                     CHECK (status IN ('running', 'awaiting_approval', 'completed', 'failed')),
    iterations       integer NOT NULL DEFAULT 0,
    heartbeat_at     timestamptz NOT NULL DEFAULT now(),
    reply_error      text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);

-- The app connects over Postgres only; the anon/PostgREST API sees nothing.
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
