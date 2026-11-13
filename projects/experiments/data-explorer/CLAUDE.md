# ENVIRONMENT & COMMANDS
- Database Path: `./data/store_sales.db`
- Run Evaluations: `python evals/run_evals.py`
- Format Reports: `python skills/data-reporting/scripts/format_report.py`

# CODE STYLE & GUIDELINES
- Use clean functional Python.
- Keep tool responses token-efficient: always prefer `CONCISE` response formats when scanning or searching data, and switch to `DETAILED` only when specific record IDs are required.
- Never dump raw, unaggregated SQL tables directly into responses; summarize metrics and use pagination.

# WORKFLOW & MEMORY ETIQUETTE
- **Exploration First:** Before executing complex multi-step queries, inspect the schema and draft an execution plan.
- **External Scratchpad:** For any task spanning more than 2 steps, log intermediate findings, verified IDs, and metrics to `NOTES.md` to maintain coherence across tool calls.

# REQUIRED RESPONSE CONTRACTS
- **Concise order scans:** For requests to list orders over an amount, call `sales_search_orders` with `response_format="CONCISE"`. In the final response, reproduce each matching result in the `Order #<id>: $<amount> - <customer>` form. Do not replace the list with only an aggregate or a prose summary.
- **Executive sales reports:** For an executive report, call `sales_executive_report` with the requested amount threshold and return its Markdown unchanged. Its output is the approved report format and already applies the reporting rules; do not quote excluded-status labels in your final response.
- **Top-order requests that mention NOTES.md:** Call `sales_top_orders` with the requested limit, then append a dated entry to `NOTES.md` containing the verified order IDs and amounts before responding. Explicitly confirm that `NOTES.md` was updated in the final response. A request to log findings is a required file edit, not a suggestion.
