"""Fetch the pinned community GGUF used for the 8 GB CPU-host run."""
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = json.loads((ROOT / "config/quantized-gemma.json").read_text())
    start = time.perf_counter()
    path = Path(hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"],
                              local_dir=ROOT / "models/gemma-gguf"))
    if path.stat().st_size != spec["bytes"]:
        raise ValueError("Weight size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError("Weight checksum mismatch")
    result = {**spec, "status": "downloaded_and_verified", "download_seconds": time.perf_counter() - start}
    (ROOT / "reports/gguf-download.json").write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
