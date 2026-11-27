-- 002: the agent conversation (messages), per-call audit (traces) and the synthetic
-- support data the read tools query (SPEC.md section 3).

-- Append-only: the exact API content blocks of every turn, in order. The app never
-- UPDATEs or DELETEs these rows; resume (Slice 3) reloads them by seq.
CREATE TABLE messages (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES sessions (id),
    seq        integer NOT NULL CHECK (seq >= 0),
    role       text NOT NULL CHECK (role IN ('user', 'assistant')),
    content    jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, seq)
);

-- One row per tool call, or one per model call that made no tool call.
-- iteration 0 is the classifier; 1..8 are agent loop calls.
-- latency_ms is the model call's latency. error is set when the call raised
-- (API error, timeout), in which case stop_reason and tokens are NULL.
CREATE TABLE traces (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id    uuid NOT NULL REFERENCES sessions (id),
    iteration     integer NOT NULL CHECK (iteration >= 0),
    model         text NOT NULL,
    stop_reason   text,
    input_tokens  integer,
    output_tokens integer,
    tool_use_id   text,
    tool_name     text,
    tool_input    jsonb,
    tool_result   jsonb,
    hook_decision text CHECK (hook_decision IN ('allow', 'require_approval', 'deny')),
    latency_ms    integer,
    error         text,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX traces_session_id_idx ON traces (session_id);

-- Synthetic support data. Nothing here is real.
CREATE TABLE tickets (
    id       text PRIMARY KEY,
    subject  text NOT NULL,
    status   text NOT NULL CHECK (status IN ('open', 'pending', 'closed')),
    customer text NOT NULL,
    history  jsonb NOT NULL DEFAULT '[]'
);

CREATE TABLE kb_articles (
    id    text PRIMARY KEY,
    title text NOT NULL,
    body  text NOT NULL
);

CREATE TABLE teammates (
    discord_user_id text PRIMARY KEY,
    name            text NOT NULL,
    is_approver     boolean NOT NULL DEFAULT false
);

-- The app connects over Postgres only; the anon/PostgREST API sees nothing.
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE traces ENABLE ROW LEVEL SECURITY;
ALTER TABLE tickets ENABLE ROW LEVEL SECURITY;
ALTER TABLE kb_articles ENABLE ROW LEVEL SECURITY;
ALTER TABLE teammates ENABLE ROW LEVEL SECURITY;
