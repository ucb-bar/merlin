"""Download and authenticate the exact checkpoint used by the model bringup."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from loader import MODEL_ID, REVISION

WEIGHTS_SHA256 = "58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945"


def identify(checkpoint):
    records = {}
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = checkpoint / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        records[name] = dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)
    if records["model.safetensors"]["sha256"] != WEIGHTS_SHA256:
        raise RuntimeError(
            "Checkpoint weights differ from the pinned pretrained snapshot"
        )
    return dict(model=MODEL_ID, revision=REVISION, files=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID,
        revision=REVISION,
        local_dir=args.out,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        max_workers=3,
    )
    identity = identify(args.out)
    (args.out / "checkpoint-identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    print(f"Verified pretrained checkpoint: {MODEL_ID}@{REVISION}", flush=True)


if __name__ == "__main__":
    main()
