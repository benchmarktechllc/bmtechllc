# Discord Triage Agent — Spec

A Discord `/ask` command that classifies a support question, gathers facts with read-only lookups, and replies. The one irreversible action (`close_ticket`) waits for a human to approve it in Discord. Every session, model turn, tool call and approval is stored in Supabase for audit.

**Confirmed assumptions (2026-11-27):**
- **Discord server:** one Discord server (guild), used only for testing. `/ask` is registered for that server only.
- **Approvers:** approvals go to one private channel (`APPROVER_CHANNEL_ID`), and only members with one role (`APPROVER_ROLE_ID`) can approve. Both are environment variables; changing them means a restart.
- **Classifier model:** `claude-haiku-5-5`, the same model used for the fixture checks.
- **Defaults:** approvals expire after 24 h (`APPROVAL_TTL`, set to a few minutes for local tests); a classification below 0.6 confidence becomes `other`.
- **Restarts:** if the app stops mid-answer, the session is marked `failed` on startup and the asker gets a fixed apology (no job queue). Sessions waiting for approval are unaffected and resume normally.
- **Data:** all data is synthetic and lives in a new Supabase project created for this tool.
- **Hosting:** v1 runs on a developer machine, and Discord reaches it through a tunnel URL. The bot works only while the app and the tunnel are running. Deploying to Fly.io comes after v1 is complete.

## 1. Users
- **Asker:** any guild member who runs `/ask <question>`.
- **Approver:** a member with `APPROVER_ROLE_ID`. Approves or denies blocked actions with buttons in the approvals channel. An asker who holds the role may approve their own request; the decision is flagged `self_approved` for audit.
- **Auditor:** reads the Supabase tables with SQL. There is no UI.

## 2. Core loop
1. Discord POSTs the `/ask` interaction to `/interactions`. Plain code verifies the Ed25519 signature (invalid → 401), inserts a `sessions` row, and ACKs with type 5 (deferred) in under 3 s.
2. A background task **classifies** the question with one structured-output call (`claude-haiku-5-5`, `max_tokens` 256). The result is `{primary, categories, confidence}`: `categories` lists 1–3 labels from `billing|account|bug|how_to|other`, one per concern, and `primary` must be one of them. Plain code validates it. Invalid output, or confidence below 0.6, becomes `primary = other`, `categories = [other]`. Both are stored on the session.
3. **Agent loop** (`claude-sonnet-5-5`, effort `low`, `max_tokens` 1024, tools `get_ticket`, `search_kb`, `close_ticket`):
   - While `stop_reason == "tool_use"`, every `tool_use` block goes through the gate (section 5). All `tool_result`s for that turn go back together in one user message.
   - **The one exception:** if any call in the turn needs approval, the loop pauses during the `tool_use` turn (section 5). Plain code edits the deferred reply to a fixed note: "Waiting for approval on N action(s) — session <id>". This is a status message, not a model reply. The model's reply still comes only on `end_turn`.
   - After each model call, the loop appends a `messages` row and one `traces` row per tool call (or one for the turn if there was no tool call).
   - It replies in Discord **only on `end_turn`**, with exactly one reply per session. The system prompt tells the model to address every concern in that reply. Live fixture checks verify that it does (section 8).
   - `max_tokens`, `refusal`, an API error, or hitting the **8-iteration cap** stops the loop and sends a fixed fallback reply. The session is marked `failed`.
4. **Reply routing** is plain code. Within 15 min of the interaction, the reply edits the deferred original response. Later (after an approval), it goes out by bot-token REST `POST /channels/{channel_id}/messages`, because interaction tokens expire after 15 min.

## 3. Data model (Supabase Postgres)
| Table | Key columns |
|---|---|
| `sessions` | `id`, `interaction_id` (unique), `guild_id`, `channel_id`, `asker_user_id`, `question`, `primary_category`, `categories` text[], `confidence`, `status` (running \| awaiting_approval \| completed \| failed), `iterations`, timestamps |
| `messages` | `id`, `session_id`, `seq` (unique per session), `role`, `content` jsonb (the exact API content blocks), `created_at`. **Append-only.** |
| `traces` | `id`, `session_id`, `iteration`, `model`, `stop_reason`, `input_tokens`, `output_tokens`, `tool_use_id`, `tool_name`, `tool_input` jsonb, `tool_result` jsonb, `hook_decision` (allow \| require_approval \| deny), `latency_ms`, `created_at` |
| `approval_requests` | `id`, `session_id`, `turn` (the loop iteration that made the call), `tool_use_id` (unique), `tool_name`, `tool_input` jsonb, `status` (pending \| approved \| denied \| expired), `requested_by`, `decided_by`, `self_approved` (bool, `decided_by = requested_by`), `decided_at`, `expires_at`, `discord_message_id` |
| `receipts` | `id`, `approval_request_id` (**unique**), `session_id`, `tool_name`, `tool_input`, `result` jsonb, `executed_at` |
| `Synthetic` | `tickets` (id, subject, status, customer, history), `kb_articles` (id, title, body), `teammates` (discord_user_id, name, is_approver) |

RLS is enabled with no policies. The app connects server-side over Postgres only, and the anon/PostgREST API sees nothing.

**Project:** a new, dedicated Supabase project used only by this tool and holding only synthetic data. It shares nothing with other BMTech projects.
- **Region:** close to where the app runs: the developer's location for v1, and later the Fly.io region.
- **Schema:** applied only through `migrations/`, never edited by hand in the dashboard.
- **Credentials:** during local operation, `DATABASE_URL` lives in a local `.env` that Git ignores. After the deploy it lives only in `fly secrets`.
- **Tests** never touch this project. They run against a local Postgres: the Supabase CLI stack (`supabase start`) or a Postgres container.
- **To confirm when the project is created:** the plan, whether a free project pauses when idle (which would break the bot), and what backups the plan includes. Record all three in the project README.

## 4. Endpoints
- `POST /interactions`: handles PING (→ type 1), the `/ask` command, and button components `approval:<id>:approve|deny`.
- `GET /healthz`: liveness check, on the tunnel URL in v1 and on the Fly.io URL later.
- **In-app sweep** every 5 min:
  - expires approvals still `pending` past `expires_at`, applying the same decision transaction as a click (section 5), and resumes a session once its turn has no pending requests;
  - on startup, runs the expiry pass at once, so approvals that expired while the app was stopped are processed;
  - on startup, marks sessions stuck in `running` for > 10 min as `failed` and posts a fixed apology.

## 5. Enforcement rules for irreversible actions (plain code, never the model)
- **Policy table:** `{get_ticket: allow, search_kb: allow, close_ticket: require_approval}`. Any other tool name → `deny` (an `is_error` tool_result).
- **Block (the PreToolUse gate):** the gate checks every `tool_use` block in a turn, in order:
  - `allow` calls run right away, and their results are saved to `traces`;
  - `deny` calls get an `is_error` result, also saved;
  - each `require_approval` call is **not executed**. It gets its own `pending` approval request (same `turn`) and its own Approve/Deny message in the approvals channel, showing the ticket and tool input.
  If the turn has any pending requests, plain code sets the session to `awaiting_approval`, posts the waiting note (section 2), and stops the loop.
  The gate's decision depends only on the tool name and the database. Text in the question, the ticket history or the model's output (for example "ignore policy" or "already approved") cannot change it.
- **Click:** plain code checks that the clicker's roles include `APPROVER_ROLE_ID` (the asker is allowed if they hold it); otherwise it sends an ephemeral "not authorized". It then runs one transaction:
  1. lock the session row (`SELECT … FOR UPDATE`);
  2. `UPDATE … SET status=… WHERE id=… AND status='pending' RETURNING`, recording `decided_by` and `self_approved` (zero rows → already decided → no-op, so a double click cannot act twice);
  3. if no request from that `turn` is still pending, flip the session from `awaiting_approval` to `running`.
  Only the transaction that does the flip triggers the resume, so two approvals decided at the same moment resume exactly once.
- **Resume (same session, never a new one):** reload `messages` for the same `session_id`. Build **one** user message holding exactly one `tool_result` for **every** `tool_use_id` in the paused turn, in the original order: the saved results of allowed and denied calls, plus one result per approval request. Append it with the next `seq`, then continue the section 2 loop under the same iteration count. No new `sessions` row is created. Each approval's result is:
  - on approve: write the `receipts` row, then execute `close_ticket` and return its real result;
  - on deny: `is_error` "denied by <approver>";
  - on expiry: `is_error` "approval expired".
- **Defense in depth:** the `close_ticket` executor refuses to run unless an `approved` request and its receipt exist for that `tool_use_id`. Nothing in the prompt tells the model it may bypass approval.

## 6. Why an agent instead of a fixed workflow
Questions vary, and which lookups each one needs (ticket then KB, KB only, none) can't be fixed in advance. A small tool loop handles that with little code. The error tolerance fits:
- A wrong read is cheap: the reply is only text, and the trace shows why it went wrong.
- The one costly error, closing the wrong ticket, is taken away from the model by the deterministic gate above.
- The audit need is met by append-only messages, per-iteration traces with `stop_reason`, and receipts.

If traces show that each category always maps to the same lookups, replace the loop with a fixed workflow: it is cheaper and easier to audit.

## 7. Non-goals
- No real ticket system: tickets are synthetic rows.
- No web UI or dashboard.
- No MCP, subagents or multi-agent research.
- No multi-turn follow-ups: one `/ask` is one session.
- No write tools besides `close_ticket`.
- No production or customer data.

## 8. Budget and limits
- **Total model spend:** about $5, with a hard abort at $4 in `scripts/live_smoke.py`.
- **Models:** `claude-sonnet-5-5` for the agent under test, run only in a handful of short manual runs. `claude-haiku-5-5` for classification and fixture checks.
- **Caps:** `max_tokens` 1024 for the loop and 256 for the classifier; 8 iterations per session.
- **Live fixture checks:** Haiku grades each live Sonnet reply against the fixture's list of expected concerns (all covered: yes or no). Live runs also assert that no `close_ticket` executes without an approval click.
- **CI** uses a scripted fake Anthropic client ($0). No unsupervised or scheduled live loops.
- **Hosting, v1:** a developer machine plus a free Cloudflare quick tunnel (`cloudflared`), $0. The bot is unavailable whenever the app or the tunnel stops.
- **Hosting, later:** one always-on Fly.io machine, about $2–5/mo.
- **Database:** the new Supabase project for both stages (plan to be confirmed, section 3).
