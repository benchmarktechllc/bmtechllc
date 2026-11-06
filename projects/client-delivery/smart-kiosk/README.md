# Smart Kiosk

Local edge-first kiosk prototype. The kiosk browser talks only to the local
Fastify edge service, which owns SQLite persistence, server-side pricing,
orders, and deterministic mock payment, kitchen, and POS adapters.

The current build is a demo foundation. It does not process cards or send
orders to live payment, kitchen, or POS systems.

## Prerequisites

- Node.js 22+
- pnpm 10+

## Run locally

```bash
pnpm install
pnpm dev
```

Open `http://localhost:5173`. The edge service runs at
`http://127.0.0.1:3001` and creates its seeded SQLite database at
`apps/edge-server/data/smart-kiosk.sqlite` by default.

Copy `.env.example` to `.env` to override edge host, port, database path, kiosk
origin, pricing, and staff access. During development, Vite proxies `/v1`,
`/health`, and `/ready` to the local
edge service, so the browser uses the same origin as the kiosk UI. Set
`VITE_EDGE_URL` only when the browser must connect directly to a separately
hosted edge address.

## Verification

```bash
pnpm typecheck
pnpm build
pnpm lint
pnpm test
```

## Current implementation

- Seeded breakfast and lunch schedules, categories, menu items, and a modifier.
- SQLite migration ledger (`schema_migrations`) and foreign-key enforcement.
- Server-authoritative pricing, order creation, transaction idempotency, and
  transaction-time snapshots, including configurable tax and promotions.
- Clearly labelled mock payment outcomes and persisted mock kitchen/POS
  dispatch outcomes.
- Kiosk cart recovery within the browser session, offline messaging, timeout
  reset, checkout recovery, and a configurable promo-code field.
- Protected staff console at `http://localhost:5173/?staff=1` for availability
  changes and operational status. Use `dev-staff-token` only for local
  development, or set `STAFF_ACCESS_TOKEN`.
- Health (`/health`) and readiness (`/ready`) endpoints plus API security
  response headers and structured Fastify logs.

## Operations notes

- Set a long random `STAFF_ACCESS_TOKEN` before production use. The API fails
  closed for staff operations when it is missing in production.
- `TAX_RATE_BPS` uses basis points: `825` represents an 8.25% tax rate.
- `KIOSK_PROMOTIONS_JSON` is an array of `{ code, type, value }` entries, where
  `type` is `PERCENT_BPS` or `AMOUNT_MINOR`. For example,
  `WELCOME10` with `PERCENT_BPS` / `1000` applies 10% off before tax.
- Back up the SQLite database and monitor `/ready` in the deployment platform.

## Deliberately deferred

Real payment terminals, live POS/KDS/printer connections, encrypted payment
store-and-forward, cloud menu synchronization, and client checkout while the
Store Edge Node is unreachable require separately selected and verified
integrations.
