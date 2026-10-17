"""Download pinned public weights and record file sizes/checksums; never invent success."""
from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name, spec, destination, metadata_only=False):
    from huggingface_hub import HfApi, hf_hub_download
    result = {"name": name, "repo_id": spec["repo_id"], "revision": spec["revision"],
              "status": "pending", "files": [], "errors": []}
    start = time.monotonic()
    try:
        info = HfApi().model_info(spec["repo_id"], revision=spec["revision"], files_metadata=True)
        if info.sha != spec["revision"]:
            raise ValueError("Resolved revision differs from lock")
        for entry in info.siblings:
            filename = entry.rfilename
            if not any(fnmatch.fnmatch(filename, p) for p in spec["allow_patterns"]):
                continue
            if metadata_only and filename.endswith((".safetensors", ".bin", ".pt")):
                continue
            print(f"{name}: downloading {filename} ({entry.size} bytes)", flush=True)
            try:
                path = Path(hf_hub_download(spec["repo_id"], filename,
                            revision=spec["revision"], local_dir=destination / name))
                actual_size = path.stat().st_size
                if entry.size is not None and actual_size != entry.size:
                    raise ValueError("Downloaded byte count mismatch")
                digest = sha256(path)
                expected = getattr(entry.lfs, "sha256", None) if entry.lfs else None
                if expected and digest != expected:
                    raise ValueError("SHA256 mismatch")
                result["files"].append({"path": f"{name}/{filename}", "bytes": actual_size,
                                        "sha256": digest})
            except Exception as exc:
                result["errors"].append({"file": filename, "error": f"{type(exc).__name__}: {exc}"})
                print(f"{name}: FAILED {filename}: {type(exc).__name__}", flush=True)
        has_weights = any(f["path"].endswith(".safetensors") for f in result["files"])
        result["status"] = ("metadata_only" if metadata_only else "downloaded") if not result["errors"] and (metadata_only or has_weights) else "incomplete"
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append({"error": f"{type(exc).__name__}: {exc}"})
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="*", default=[])
    p.add_argument("--destination", type=Path, default=ROOT / "models")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    specs = json.loads((ROOT / "config/models.json").read_text())
    names = args.names or list(specs)
    if set(names) - set(specs):
        p.error("Unknown model name")
    args.destination.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "models": [], "weights_in_git": False}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(download, n, specs[n], args.destination, args.metadata_only): n for n in names}
        for job in concurrent.futures.as_completed(jobs):
            report["models"].append(job.result())
            (ROOT / "reports/downloads.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({m["name"]: m["status"] for m in report["models"]}, indent=2))
    raise SystemExit(0 if all(m["status"] in {"downloaded", "metadata_only"} for m in report["models"]) else 1)


if __name__ == "__main__":
    main()
