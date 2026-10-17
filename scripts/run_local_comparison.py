"""Run matched direct/hybrid Gemma trials using a local llama.cpp worker.

The worker is stopped on completion; loading time and worker RSS are measured
separately from per-task client measurements. No external inference API is used.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--port", type=int, default=8087)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tasks", default="examples/arithmetic.json")
    p.add_argument("--report-prefix", default="run")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--expansions", type=int, default=10000)
    args = p.parse_args()
    task_count = len(json.loads((ROOT / args.tasks).read_text()))
    command = [args.server, "-m", args.gguf, "-c", "2048", "-t", str(args.threads),
               "--parallel", "1", "--host", "127.0.0.1", "--port", str(args.port)]
    report = {"worker_command": command, "scope": f"{task_count} arithmetic tasks, no 120B baseline",
              "worker_peak_rss_bytes": 0, "runs": [], "cost_usd": None}
    done = threading.Event()
    start = time.perf_counter()
    (ROOT / "workspace").mkdir(exist_ok=True)
    log = (ROOT / "workspace/llama-server.log").open("w")
    worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

    def monitor():
        proc = psutil.Process(worker.pid)
        while not done.wait(.1):
            try:
                # Some launchers fork the actual inference process.
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try: rss += child.memory_info().rss
                    except psutil.NoSuchProcess: pass
                report["worker_peak_rss_bytes"] = max(report["worker_peak_rss_bytes"], rss)
            except psutil.NoSuchProcess:
                break

    monitoring = threading.Thread(target=monitor, daemon=True)
    monitoring.start()
    try:
        while time.perf_counter() - start < 120:
            if worker.poll() is not None:
                raise RuntimeError(f"Model worker exited {worker.returncode}; inspect workspace/llama-server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1) as r:
                    if r.status == 200: break
            except OSError:
                pass
            time.sleep(.25)
        else:
            raise TimeoutError("Worker was not ready within 120 seconds")
        report["worker_ready_seconds"] = time.perf_counter() - start
        print(f"Gemma worker ready in {report['worker_ready_seconds']:.2f}s", flush=True)
        for mode in ["direct", "hybrid"]:
            output = f"reports/{args.report_prefix}-gemma-{mode}.json"
            cmd = [sys.executable, "-m", "thinking_runtime", "bench", "--mode", mode,
                   "--backend", "http", "--base-url", f"http://127.0.0.1:{args.port}/v1",
                   "--model", "gemma-e2b-q4", "--seed", "7", "--llama-no-thinking",
                   "--tasks", args.tasks, "--expansions", str(args.expansions),
                   "--max-tokens", str(args.max_tokens), "--seconds", str(args.seconds), "--output", output]
            result = subprocess.run(cmd, cwd=ROOT, timeout=task_count * args.seconds + 60)
            if result.returncode: raise RuntimeError(f"Benchmark {mode} exited {result.returncode}")
            report["runs"].append(output)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        done.set()
        monitoring.join(timeout=2)
        worker.terminate()
        try: worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        log.close()
        report["total_seconds"] = time.perf_counter() - start
        report["worker_peak_rss_observed_bytes"] = report["worker_peak_rss_bytes"]
        if report["worker_peak_rss_bytes"] < 64 * 1024 * 1024:
            report["worker_peak_rss_bytes"] = None
            report["memory_note"] = "Process inspection did not expose a credible inference-worker RSS; memory is unmeasured."
        (ROOT / f"reports/{args.report_prefix}-local-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
