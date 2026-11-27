"""Tool definitions sent to the model. Strict schemas, so tool_use inputs match exactly."""

from __future__ import annotations

from typing import Final

from anthropic.types import ToolParam

GET_TICKET: Final[ToolParam] = {
    "name": "get_ticket",
    "description": (
        "Look up one support ticket by its id (for example T-1002). Returns the subject, "
        "status (open, pending or closed), customer and the ticket's history notes. "
        "Use it whenever the question mentions a ticket id."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string", "description": "Ticket id, like T-1002."},
        },
        "required": ["ticket_id"],
        "additionalProperties": False,
    },
    "strict": True,
}

SEARCH_KB: Final[ToolParam] = {
    "name": "search_kb",
    "description": (
        "Search the help-center knowledge base and return up to 3 matching articles "
        "(id, title, body). Use it for how-to questions and for policies such as refunds."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A few keywords, like 'reset password'."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

TOOLS: Final[list[ToolParam]] = [GET_TICKET, SEARCH_KB]
