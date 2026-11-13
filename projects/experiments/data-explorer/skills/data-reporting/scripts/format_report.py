import sys
import json
from datetime import datetime

def generate_markdown_report(raw_json: str) -> str:
    data = json.loads(raw_json)
    orders = data.get("results", [])

    lines = [
        f"# Executive Sales Report ({datetime.now().strftime('%Y-%m-%d')})",
        "",
        f"**Total Records Evaluated:** {len(orders)}",
        "",
        "| Order Summary |",
        "| :--- |",
    ]

    for item in orders:
        lines.append(f"| {item} |")

    return "\n".join(lines)

if __name__ == "__main__":
    # Reads JSON string passed as the first CLI argument
    if len(sys.argv) < 2:
        print("Usage: python format_report.py '<json_payload>'")
        sys.exit(1)

    print(generate_markdown_report(sys.argv[1]))
