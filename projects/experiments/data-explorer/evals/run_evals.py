import subprocess
import json
from pathlib import Path

# Define realistic, multi-step evaluation scenarios
EVAL_CASES = [
    {
        "id": "eval_01_concise_scan",
        "prompt": "List orders over $100. Keep it concise.",
        "expected_substrings": ["Order #", "$"],
        "forbidden_substrings": ["Traceback", "Error"],
    },
    {
        "id": "eval_02_executive_report",
        "prompt": "Generate an executive sales report for all orders over $50.",
        "expected_substrings": ["# Executive Sales Report", "| Order Summary |"],
        "forbidden_substrings": ["VOIDED", "CANCELLED"],
    },
    {
        "id": "eval_03_notes_persistence",
        "prompt": "Find the top 3 orders by amount, log their IDs to NOTES.md, and summarize them.",
        "expected_substrings": ["NOTES.md"],
        "check_notes_updated": True,
    },
]

def run_agent_prompt(prompt: str) -> str:
    """
    Executes a headless Claude Code CLI run against the current workspace.
    Swap or adapt the CLI command if using a different agent runner.
    """
    result = subprocess.run(
        [
            "claude",
            "-p",
            # Headless runs have no interactive permission dialog. Allow only
            # the local sales tools and the workspace file tools required by
            # the scenarios; without this, Claude receives a denied tool call
            # instead of the data it needs to answer.
            "--allowedTools="
            "mcp__sales-db__sales_search_orders,"
            "mcp__sales-db__sales_top_orders,"
            "mcp__sales-db__sales_executive_report,"
            "Read,Edit,Write,Bash",
            prompt,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + "\n" + result.stderr

def evaluate() -> None:
    notes_path = Path("NOTES.md")
    initial_notes = notes_path.read_text() if notes_path.exists() else ""

    passed = 0
    for case in EVAL_CASES:
        print(f"Running {case['id']}...")
        output = run_agent_prompt(case["prompt"])

        missing = [s for s in case.get("expected_substrings", []) if s not in output]
        forbidden = [s for s in case.get("forbidden_substrings", []) if s in output]
        notes_ok = True

        if case.get("check_notes_updated"):
            current_notes = notes_path.read_text() if notes_path.exists() else ""
            notes_ok = current_notes != initial_notes

        if not missing and not forbidden and notes_ok:
            print(f"  [PASS] {case['id']}")
            passed += 1
        else:
            print(f"  [FAIL] {case['id']}")
            if missing:
                print(f"    Missing expected text: {missing}")
            if forbidden:
                print(f"    Found forbidden text: {forbidden}")
            if not notes_ok:
                print("    NOTES.md was not updated.")

    print(f"\nSummary: {passed}/{len(EVAL_CASES)} evaluations passed.")

if __name__ == "__main__":
    evaluate()
