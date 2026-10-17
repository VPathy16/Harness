"""Compare measured reports without silently treating unknown costs as zero."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("harness")
    p.add_argument("baseline")
    args = p.parse_args()
    h, b = [json.loads(Path(f).read_text()) for f in [args.harness, args.baseline]]
    if not h.get("task_spec_sha256") or h["task_spec_sha256"] != b.get("task_spec_sha256"):
        raise SystemExit("Task specification fingerprints are missing or differ; refuse comparison")
    if [r["task_id"] for r in h["rows"]] != [r["task_id"] for r in b["rows"]]:
        raise SystemExit("Task IDs or order differ; refuse comparison")
    if h["configuration"]["seconds"] != b["configuration"]["seconds"]:
        raise SystemExit("Per-task time budgets differ; refuse comparison")
    costs_known = all(x.get("estimated_execution_usd") is not None for x in [h, b])
    result = {"scope": "descriptive only; no statistical/general intelligence claim",
              "harness_accuracy": h["successes"] / h["tasks"], "baseline_accuracy": b["successes"] / b["tasks"],
              "higher_accuracy": h["successes"] > b["successes"],
              "lower_warm_p50": h["p50_seconds"] < b["p50_seconds"],
              "lower_warm_p95": h["p95_seconds"] < b["p95_seconds"],
              "lower_execution_cost": h["estimated_execution_usd"] < b["estimated_execution_usd"] if costs_known else None,
              "costs_complete": False,
              "notes": "Training/amortization and service hardware are not automatically measured; inspect cold loading separately."}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
