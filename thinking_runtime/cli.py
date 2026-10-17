from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import time
from pathlib import Path

from .domain import Task, verify
from .runtime import Budget, solve
from .workspace import Workspace


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def run_case(task, args, generator, ranker):
    start = time.perf_counter()
    feedback, trace = [], []
    calls = input_tokens = output_tokens = 0
    usage_known = True
    status, expression = "no_verified_solution", None
    expanded = generated = rank_calls = 0
    if args.mode != "symbolic":
        for attempt in range(1 if args.mode in {"direct", "hybrid"} else args.max_calls):
            remaining = args.seconds - (time.perf_counter() - start)
            if remaining <= 0:
                status = "budget_exhausted"
                break
            calls += 1
            try:
                response = generator.generate(task.public(), feedback, remaining)
                if response["input_tokens"] is None or response["output_tokens"] is None:
                    usage_known = False
                else:
                    input_tokens += response["input_tokens"]
                    output_tokens += response["output_tokens"]
                candidate = response["expression"]
                ok, reason = verify(task, candidate)
                late = time.perf_counter() - start > args.seconds
                feedback.append({"expression": candidate, "feedback": reason})
                trace.append({"event": "model_proposal", "valid": ok, "late": late, **response, "reason": reason})
                if late:
                    status = "budget_exhausted"
                    break
                if ok:
                    status, expression = "verified", candidate
                    break
            except Exception as exc:
                usage_known = False
                trace.append({"event": "model_error", "error": type(exc).__name__, "message": str(exc)[:300]})
                status = "model_error"
                break
    remaining = args.seconds - (time.perf_counter() - start)
    if args.mode in {"symbolic", "hybrid"} and status != "verified" and remaining > 0:
        result = solve(task, Budget(args.expansions, remaining, args.ranker_calls), ranker=ranker)
        status, expression = result.status, result.expression
        expanded, generated, rank_calls = result.expanded, result.generated, result.ranker_calls
        trace += result.trace
    elapsed = time.perf_counter() - start
    if elapsed > args.seconds:
        status, expression = "budget_exhausted", None
    cost = None
    if args.usd_per_hour is not None:
        cost = elapsed * args.usd_per_hour / 3600
    if calls and args.backend == "http":
        if args.input_usd_per_million is None or args.output_usd_per_million is None or not usage_known:
            cost = None
        else:
            cost = (cost or 0) + (input_tokens * args.input_usd_per_million + output_tokens * args.output_usd_per_million) / 1e6
    return {"task_id": task.id, "status": status, "expression": expression, "seconds": elapsed,
            "model_calls": calls, "ranker_calls": rank_calls, "expanded": expanded, "generated": generated,
            "input_tokens": input_tokens if usage_known else None,
            "output_tokens": output_tokens if usage_known else None,
            "estimated_execution_usd": cost, "trace": trace}


def bench(args):
    from .adapters import GemmaGenerator, HTTPGenerator, LayaRanker
    if args.mode != "symbolic" and args.backend is None:
        raise ValueError("Choose --backend gemma or http for model-based modes")
    if args.backend == "http" and (not args.base_url or not args.model):
        raise ValueError("HTTP backend needs --base-url and --model")
    suite = json.loads(Path(args.tasks).read_text())
    tasks = [Task(t["id"], tuple(t["numbers"]), t.get("target", 24)) for t in suite]
    if not tasks or len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Need nonempty tasks with unique IDs")
    started = time.perf_counter()
    generator = None
    if args.mode != "symbolic":
        generator = (HTTPGenerator(args.base_url, args.model, args.max_tokens, args.seed, args.llama_no_thinking) if args.backend == "http"
                     else GemmaGenerator(Path(args.models) / "gemma", args.max_tokens, args.device))
    ranker = LayaRanker(Path(args.models) / "laya", args.device) if args.ranker == "laya" else None
    load_seconds = time.perf_counter() - started
    workspace = Workspace(args.workspace)
    rows = []
    try:
        for task in tasks:
            row = run_case(task, args, generator, ranker)
            workspace.record(task, row)
            rows.append(row)
            print(f"{task.id}: {row['status']} {row['seconds']:.4f}s", flush=True)
    finally:
        workspace.close()
    latencies = sorted(r["seconds"] for r in rows)
    report = {
        "schema_version": 1, "scope": "arithmetic smoke suite; not general intelligence evidence",
        "task_spec_sha256": hashlib.sha256(json.dumps([t.public() for t in tasks], sort_keys=True).encode()).hexdigest(),
        "configuration": vars(args), "platform": platform.platform(),
        "cold_load_seconds": load_seconds, "total_wall_seconds": time.perf_counter() - started,
        "successes": sum(r["status"] == "verified" for r in rows), "tasks": len(rows),
        "p50_seconds": statistics.median(latencies), "p95_seconds": latencies[math.ceil(.95 * len(rows)) - 1],
        "client_process_peak_rss_bytes": rss_bytes(), "remote_server_memory_bytes": None,
        "estimated_execution_usd": sum(r["estimated_execution_usd"] for r in rows) if all(r["estimated_execution_usd"] is not None for r in rows) else None,
        "estimated_cold_load_usd": load_seconds * args.usd_per_hour / 3600 if args.usd_per_hour is not None else None,
        "training_cost_usd": None, "rows": rows,
        "notes": ["No cost is assumed zero when a tariff is missing.",
                  "No exact-result cache is read during benchmarking.",
                  "Budgets are cooperative: an in-flight kernel may finish late; late answers count as failures.",
                  "Peak RSS is process memory, not GPU VRAM or remote server memory."]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {report['successes']}/{len(rows)}; report: {output}")


def main():
    p = argparse.ArgumentParser(description="Harness research prototype")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--tasks", default="examples/arithmetic.json")
    b.add_argument("--mode", choices=["symbolic", "direct", "feedback", "hybrid"], default="symbolic")
    b.add_argument("--backend", choices=["gemma", "http"])
    b.add_argument("--base-url")
    b.add_argument("--model")
    b.add_argument("--seed", type=int)
    b.add_argument("--llama-no-thinking", action="store_true", help="llama.cpp-specific template and prompt-cache options")
    b.add_argument("--models", default="models")
    b.add_argument("--device", default="cpu")
    b.add_argument("--ranker", choices=["none", "laya"], default="none")
    b.add_argument("--ranker-calls", type=int, default=0)
    b.add_argument("--max-calls", type=int, default=8)
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument("--seconds", type=float, default=10)
    b.add_argument("--expansions", type=int, default=10000)
    b.add_argument("--usd-per-hour", type=float)
    b.add_argument("--input-usd-per-million", type=float)
    b.add_argument("--output-usd-per-million", type=float)
    b.add_argument("--workspace", default="workspace")
    b.add_argument("--output", default="reports/benchmark.json")
    args = p.parse_args()
    Budget(args.expansions, args.seconds, args.ranker_calls)
    if args.max_calls <= 0 or args.max_tokens <= 0:
        p.error("Call and token limits must be positive")
    for name in ("usd_per_hour", "input_usd_per_million", "output_usd_per_million"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            p.error("Rates must be finite and nonnegative")
    bench(args)


if __name__ == "__main__":
    main()
