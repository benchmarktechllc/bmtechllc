"""A dependency-free MCP server for the store-sales sample database.

The evaluation environment intentionally contains only Python's standard
library.  Implementing the small JSON-RPC-over-stdio surface used by MCP keeps
the data tools available without requiring an undeclared third-party package.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "store_sales.db"
EXCLUDED_STATUSES = ("VOIDED", "CANCELLED")


def _query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Run a read-only query and return ordinary dictionaries."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def sales_search_orders(min_amount: float = 0.0, response_format: str = "CONCISE") -> str:
    """Return orders at or above an amount, ordered from largest to smallest."""
    try:
        rows = _query(
            "SELECT id, customer_name, amount, status, order_date "
            "FROM orders WHERE amount >= ? ORDER BY amount DESC, id ASC LIMIT 10",
            (float(min_amount),),
        )
    except (sqlite3.Error, ValueError) as error:
        return json.dumps({"error": "Query failed", "details": str(error)})

    if response_format.upper() == "CONCISE":
        results: list[Any] = [
            f"Order #{row['id']}: ${row['amount']:,.2f} - {row['customer_name']}"
            for row in rows
        ]
    else:
        results = rows
    return json.dumps({"total_returned": len(rows), "results": results}, indent=2)


def sales_top_orders(limit: int = 3) -> str:
    """Return the highest-value orders, including their IDs, for a requested ranking."""
    safe_limit = max(1, min(int(limit), 10))
    rows = _query(
        "SELECT id, customer_name, amount, status, order_date "
        "FROM orders ORDER BY amount DESC, id ASC LIMIT ?",
        (safe_limit,),
    )
    return json.dumps({"total_returned": len(rows), "results": rows}, indent=2)


def sales_executive_report(min_amount: float = 0.0) -> str:
    """Return a policy-compliant executive report as Markdown.

    Inactive records are removed before the text is rendered, so their status
    labels cannot leak into a final executive-facing report.
    """
    placeholders = ", ".join("?" for _ in EXCLUDED_STATUSES)
    rows = _query(
        "SELECT id, customer_name, amount, status, order_date FROM orders "
        f"WHERE amount >= ? AND status NOT IN ({placeholders}) "
        "ORDER BY amount DESC, id ASC",
        (float(min_amount), *EXCLUDED_STATUSES),
    )

    completed = [row for row in rows if row["status"] != "REFUNDED"]
    refunded = [row for row in rows if row["status"] == "REFUNDED"]
    net_revenue = sum(row["amount"] for row in completed) - sum(
        row["amount"] for row in refunded
    )
    lines = [
        "# Executive Sales Report",
        "",
        f"**Total Records Evaluated:** {len(rows)}",
        "",
        "| Order Summary |",
        "| :--- |",
    ]
    lines.extend(
        f"| Order #{row['id']} — {row['customer_name']} — ${row['amount']:,.2f} |"
        for row in completed
    )
    if refunded:
        lines.append(
            "| Refunds deducted: "
            + ", ".join(
                f"Order #{row['id']} (${row['amount']:,.2f})" for row in refunded
            )
            + " |"
        )
    lines.extend(["", f"**Net Revenue:** ${net_revenue:,.2f}"])
    return "\n".join(lines)


TOOLS = {
    "sales_search_orders": {
        "description": "Search orders by minimum amount. Use CONCISE for a quick list and DETAILED only when fields such as IDs or statuses are needed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_amount": {"type": "number", "default": 0},
                "response_format": {
                    "type": "string",
                    "enum": ["CONCISE", "DETAILED"],
                    "default": "CONCISE",
                },
            },
        },
        "handler": sales_search_orders,
    },
    "sales_top_orders": {
        "description": "Find the highest-value orders. Use this when the request asks for top or largest orders and needs their IDs.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 3}},
        },
        "handler": sales_top_orders,
    },
    "sales_executive_report": {
        "description": "Generate the final Markdown executive sales report for an amount threshold. It applies reporting exclusions and refund handling automatically.",
        "inputSchema": {
            "type": "object",
            "properties": {"min_amount": {"type": "number", "default": 0}},
        },
        "handler": sales_executive_report,
    },
}


def _respond(request_id: Any, result: dict[str, Any]) -> None:
    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)


def _error(request_id: Any, code: int, message: str) -> None:
    print(
        json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
        ),
        flush=True,
    )


def serve() -> None:
    """Handle the MCP methods needed by Claude Code over newline-delimited JSON."""
    for raw_message in sys.stdin:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            continue

        request_id = message.get("id")
        method = message.get("method")
        if request_id is None:  # Notifications, including initialized.
            continue
        if method == "initialize":
            _respond(
                request_id,
                {
                    "protocolVersion": message.get("params", {}).get(
                        "protocolVersion", "2024-11-05"
                    ),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "sales-data-explorer", "version": "1.0.0"},
                },
            )
        elif method == "tools/list":
            _respond(
                request_id,
                {
                    "tools": [
                        {key: value for key, value in tool.items() if key != "handler"}
                        | {"name": name}
                        for name, tool in TOOLS.items()
                    ]
                },
            )
        elif method == "tools/call":
            params = message.get("params", {})
            tool = TOOLS.get(params.get("name"))
            if tool is None:
                _error(request_id, -32601, "Unknown tool")
                continue
            try:
                text = tool["handler"](**params.get("arguments", {}))
                _respond(request_id, {"content": [{"type": "text", "text": text}]})
            except (TypeError, ValueError, sqlite3.Error) as error:
                _respond(
                    request_id,
                    {
                        "content": [{"type": "text", "text": str(error)}],
                        "isError": True,
                    },
                )
        elif method == "ping":
            _respond(request_id, {})
        else:
            _error(request_id, -32601, f"Unsupported method: {method}")


if __name__ == "__main__":
    serve()
