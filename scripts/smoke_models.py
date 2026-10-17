"""Run one real component per process, with bounded wall time and recorded errors."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def component(name, device):
    os.environ.setdefault("USE_TF", "0")
    from thinking_runtime import adapters
    from thinking_runtime.cli import rss_bytes
    from thinking_runtime.domain import Task, verify
    start = time.perf_counter()
    path = ROOT / "models" / name
    if name == "gemma":
        model = adapters.GemmaGenerator(path, max_tokens=64, device=device)
    elif name == "laya":
        model = adapters.LayaRanker(path, device=device)
    elif name == "embedding":
        model = adapters.EmbeddingMemory(path, device=device)
    else:
        model = adapters.TimesFMForecaster(path)
    loaded = time.perf_counter()
    if name == "gemma":
        task = Task("gemma-smoke", (1, 2, 3, 4))
        output = model.generate(task.public(), [], 30)
        output["verified"] = verify(task, output["expression"])[0]
    elif name == "laya":
        output = model.rank(Task("laya-smoke", (1, 2, 3, 4)).public(), [["1*2", "3*4"], ["1+2", "3+4"]])
    elif name == "embedding":
        output = model.retrieve("How do I fix a database query?", ["SQL queries retrieve database records.", "Rain falls from clouds."])
    else:
        output = model.forecast([float(i) for i in range(1, 65)], 4)
    return {"component": name, "status": "inference_completed", "load_seconds": loaded - start,
            "inference_seconds": time.perf_counter() - loaded, "peak_rss_bytes": rss_bytes(), "output": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="*", choices=["gemma", "laya", "embedding", "timesfm"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--child", action="store_true")
    p.add_argument("--output", default="reports/model-smoke.json")
    args = p.parse_args()
    names = args.names or ["laya", "embedding", "timesfm", "gemma"]
    if args.child:
        try:
            result = component(names[0], args.device)
        except Exception as exc:
            result = {"component": names[0], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print("RESULT_JSON=" + json.dumps(result), flush=True)
        return
    results = []
    for name in names:
        try:
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()), name, "--child", "--device", args.device],
                                 capture_output=True, text=True, timeout=args.timeout, cwd=ROOT)
            lines = [s for s in run.stdout.splitlines() if s.startswith("RESULT_JSON=")]
            if lines:
                result = json.loads(lines[-1][len("RESULT_JSON="):])
            else:
                result = {"component": name, "status": "failed", "returncode": run.returncode, "error": run.stderr[-1500:]}
        except subprocess.TimeoutExpired:
            result = {"component": name, "status": "timeout", "timeout_seconds": args.timeout}
        results.append(result)
        print(json.dumps(result), flush=True)
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"scope": "component load/inference checks, not a quality benchmark", "results": results}, indent=2) + "\n")
    raise SystemExit(0 if all(r["status"] == "inference_completed" for r in results) else 1)


if __name__ == "__main__":
    main()
