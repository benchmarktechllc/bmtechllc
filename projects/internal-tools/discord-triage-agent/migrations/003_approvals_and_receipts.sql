-- 003: approvals for gated tool calls and receipts for executed ones (SPEC.md sections 3, 5).

-- One row per blocked tool call. `turn` is the loop iteration that made the call; all
-- requests of a turn must be decided before the session resumes.
CREATE TABLE approval_requests (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id         uuid NOT NULL REFERENCES sessions (id),
    turn               integer NOT NULL CHECK (turn >= 1),
    tool_use_id        text NOT NULL UNIQUE,
    tool_name          text NOT NULL,
    tool_input         jsonb NOT NULL,
    status             text NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending', 'approved', 'denied', 'expired')),
    requested_by       text NOT NULL,
    -- NULL for an expiry; otherwise the Discord user who clicked.
    decided_by         text,
    self_approved      boolean NOT NULL DEFAULT false,
    decided_at         timestamptz,
    expires_at         timestamptz NOT NULL,
    discord_message_id text,
    created_at         timestamptz NOT NULL DEFAULT now(),
    CHECK ((status = 'pending') = (decided_at IS NULL))
);

CREATE INDEX approval_requests_session_turn_idx ON approval_requests (session_id, turn);
CREATE INDEX approval_requests_pending_expiry_idx ON approval_requests (expires_at)
    WHERE status = 'pending';

-- Written in the same transaction as the ticket change, so a receipt exists exactly when
-- the close ran. UNIQUE on approval_request_id makes a repeated run return the stored one.
CREATE TABLE receipts (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    approval_request_id uuid NOT NULL UNIQUE REFERENCES approval_requests (id),
    session_id          uuid NOT NULL REFERENCES sessions (id),
    tool_name           text NOT NULL,
    tool_input          jsonb NOT NULL,
    outcome             text NOT NULL CHECK (outcome IN ('succeeded', 'failed')),
    result              jsonb NOT NULL,
    executed_at         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX receipts_session_id_idx ON receipts (session_id);

-- The app connects over Postgres only; the anon/PostgREST API sees nothing.
ALTER TABLE approval_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE receipts ENABLE ROW LEVEL SECURITY;
