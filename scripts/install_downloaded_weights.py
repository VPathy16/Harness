"""Attach separately delivered weight files to the metadata included in the source ZIP."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"gemma": "Thinking-Runtime-Gemma-E2B.safetensors",
         "laya": "Thinking-Runtime-Laya.safetensors",
         "embedding": "Thinking-Runtime-EmbeddingGemma-2.safetensors",
         "timesfm": "Thinking-Runtime-TimesFM-2.5.safetensors"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    report = json.loads((ROOT / "reports/downloads.json").read_text())
    for model in report["models"]:
        name = model["name"]
        expected = next(f for f in model["files"] if f["path"] == name + "/model.safetensors")
        source = args.artifacts / NAMES[name]
        if not source.exists():
            print(f"{name}: missing {source.name}; use download_models.py {name}")
            continue
        if source.stat().st_size != expected["bytes"]:
            raise ValueError(f"{name}: byte size mismatch")
        digest = hashlib.sha256()
        with source.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""): digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"{name}: SHA256 mismatch")
        destination = ROOT / "models" / name / "model.safetensors"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if os.path.samefile(source, destination):
                print(f"{name}: already attached")
                continue
            raise FileExistsError(f"Refusing to overwrite {destination}; move it aside first")
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
        print(f"{name}: verified and attached")


if __name__ == "__main__":
    main()
