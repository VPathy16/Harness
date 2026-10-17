"""Run river planning and optionally a real local Gemma proposal plus repair.

The hybrid reuses the exact direct-model proposal, not a second random draw.
Load time is separate. Published larger-model results are not local measurements.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thinking_runtime.river import RiverTask, solve_river, verify_river


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--capacity", type=int, default=3)
    p.add_argument("--server")
    p.add_argument("--gguf")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--output", default="reports/river-run.json")
    args = p.parse_args()
    if bool(args.server) != bool(args.gguf):
        p.error("Supply both --server and --gguf, or neither")
    task = RiverTask(args.pairs, args.capacity)
    spec = task.public()
    search = solve_river(task, max_seconds=args.seconds)
    report = {"task": spec, "task_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
              "configuration": vars(args), "platform": platform.platform(),
              "symbolic": search, "verification": verify_river(task, search["moves"]) if search["moves"] else None,
              "larger_model_run_locally": False, "cost_usd": None,
              "notes": ["BFS and verifier use independently implemented safety checks.",
                        "No cached or embedded solution sequence; solver enumerates passenger combinations.",
                        "Domain rules were manually implemented, not inferred by a model.",
                        "Hybrid reuses the direct proposal, then searches only if verification fails.",
                        "Laya, EmbeddingGemma and TimesFM are not used."]}
    print(f"Search: {search['status']}, {len(search['moves'] or [])} moves, {search['seconds']:.6f}s", flush=True)
    if args.server:
        (ROOT / "workspace").mkdir(exist_ok=True)
        command = [args.server, "-m", args.gguf, "-c", "4096", "-t", "4", "--parallel", "1",
                   "--host", "127.0.0.1", "--port", str(args.port)]
        report["worker_command"] = command
        with (ROOT / "workspace/river-server.log").open("w") as log:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            started = time.perf_counter()
            try:
                while time.perf_counter() - started < 120:
                    if worker.poll() is not None:
                        raise RuntimeError(f"Worker exited: {worker.returncode}")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    time.sleep(.25)
                else:
                    raise TimeoutError("Model loading exceeded 120 seconds")
                report["cold_load_seconds"] = time.perf_counter() - started
                prompt = ("Solve this river-crossing planning task. Return a JSON object with key moves: "
                          "a list of passenger lists, one list for each crossing. Include the complete "
                          "solution, not an algorithm. The first crossing is left to right; directions "
                          "alternate. Check boat safety and both banks after every move. " + json.dumps(spec))
                payload = {"model": "gemma-e2b-q4", "messages": [{"role": "user", "content": prompt}],
                           "temperature": .7, "seed": 7, "max_tokens": args.max_tokens,
                           "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False,
                           "response_format": {"type": "json_schema", "json_schema": {"name": "crossings", "strict": True,
                               "schema": {"type": "object", "properties": {"moves": {"type": "array", "minItems": 1,
                                   "maxItems": 127, "items": {"type": "array", "minItems": 1, "maxItems": task.capacity,
                                       "items": {"type": "string", "enum": list(task.people)}}}},
                                   "required": ["moves"], "additionalProperties": False}}}}
                report["request"] = payload
                request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                           json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                model_started = time.perf_counter()
                try:
                    with urllib.request.urlopen(request, timeout=args.seconds) as response:
                        data = json.load(response)
                    report["model_response"] = data
                    content = data["choices"][0]["message"]["content"]
                    try:
                        proposed = json.loads(content)["moves"]
                        verification = verify_river(task, proposed)
                    except (ValueError, KeyError, TypeError) as exc:
                        proposed = None
                        verification = {"valid": False, "reason": f"Invalid JSON output: {type(exc).__name__}"}
                    elapsed = time.perf_counter() - model_started
                    valid = verification["valid"] and elapsed <= args.seconds
                    report["direct"] = {"status": "verified" if valid else "no_verified_solution",
                                        "seconds": elapsed, "moves": proposed, "verification": verification}
                except Exception as exc:
                    elapsed = time.perf_counter() - model_started
                    valid = False
                    proposed = None
                    report["direct"] = {"status": "model_error", "seconds": elapsed,
                                        "error": f"{type(exc).__name__}: {exc}"}
                print(f"Gemma direct: {report['direct']['status']}, {elapsed:.3f}s", flush=True)
                if valid:
                    report["hybrid"] = {"status": "verified", "seconds": elapsed,
                                        "moves": proposed, "search_used": False, "shared_direct_proposal": True}
                else:
                    remaining = args.seconds - (time.perf_counter() - model_started)
                    repair = solve_river(task, max_seconds=remaining) if remaining > 0 else {"status": "budget_exhausted", "moves": None}
                    elapsed_total = time.perf_counter() - model_started
                    if elapsed_total > args.seconds:
                        repair = {"status": "budget_exhausted", "moves": None}
                    report["hybrid"] = {"status": repair["status"], "seconds": elapsed_total,
                                        "moves": repair["moves"], "search_used": True,
                                        "shared_direct_proposal": True, "search": repair}
                print(f"Hybrid: {report['hybrid']['status']}, {report['hybrid']['seconds']:.3f}s", flush=True)
            except Exception as exc:
                report["worker_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved: {output}", flush=True)
    if "worker_error" in report:
        raise SystemExit(report["worker_error"])


if __name__ == "__main__":
    main()
