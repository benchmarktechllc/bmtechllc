---
name: data-reporting
description: Generates formatted sales and customer summary reports from the store_sales SQLite database. Trigger when the user asks for a sales report, executive summary, or formatted table of order metrics.
---

# Sales Report Generation Workflow

When asked to generate a sales report, follow these steps in order:

1. **Gather Data:** Use the `sales_search_orders` tool with `response_format="CONCISE"` to scan relevant records. Switch to `DETAILED` only if you need specific customer or transaction IDs.
2. **Verify Edge Cases:** Before finalizing totals, read `schema_guide.md` in this skill directory to verify rules on voided orders, refunds, and fiscal date cutoffs.
3. **Log Progress:** For multi-step reports, write intermediate totals and verified IDs to `NOTES.md`.
4. **Format Deterministically:** Pass the aggregated JSON data into `scripts/format_report.py` via Bash rather than manually building Markdown tables.