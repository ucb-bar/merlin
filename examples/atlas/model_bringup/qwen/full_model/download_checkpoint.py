"""Download observations; reject use until all snapshot files have trusted pins."""

import argparse
import json
import os
from pathlib import Path

from checkpoint_identity import (
    FILES, MODEL_ID, REVISION, TRUSTED_SHA256, authenticate, identify,
)

WEIGHTS_SHA256 = TRUSTED_SHA256["model.safetensors"]


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
        allow_patterns=list(FILES),
        max_workers=3,
    )
    identity = identify(args.out)
    (args.out / "checkpoint-identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    authenticate(args.out)
    print(f"Verified pretrained checkpoint: {MODEL_ID}@{REVISION}", flush=True)


if __name__ == "__main__":
    main()
