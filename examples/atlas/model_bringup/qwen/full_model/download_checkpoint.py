"""Download a commit-pinned snapshot and receipt its authenticated file hashes."""

import argparse
import json
import os
from pathlib import Path

from checkpoint_identity import (
    FILES, MODEL_ID, REVISION, TRUSTED_SHA256, authenticate,
)

WEIGHTS_SHA256 = TRUSTED_SHA256["model.safetensors"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download, snapshot_download

    receipt = args.out / "checkpoint-identity.json"
    receipt.unlink(missing_ok=True)

    snapshot_download(
        MODEL_ID,
        revision=REVISION,
        local_dir=args.out,
        allow_patterns=list(FILES),
        max_workers=3,
    )
    # snapshot_download may return an existing local directory when offline.
    # Force each unpinned metadata file through the commit-specific online API
    # before giving its bytes authenticated download provenance. Weights already
    # have an independent digest and need no forced multi-gigabyte redownload.
    for name in FILES:
        if name not in TRUSTED_SHA256:
            hf_hub_download(
                MODEL_ID, filename=name, revision=REVISION, local_dir=args.out,
                force_download=True, local_files_only=False,
            )
    # The successful commit-specific download supplies metadata provenance here.
    # Local readers do not reuse this receipt as an independent trust anchor.
    identity = authenticate(args.out, downloaded_revision=REVISION)
    receipt.write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    print(f"Verified pretrained checkpoint: {MODEL_ID}@{REVISION}", flush=True)


if __name__ == "__main__":
    main()
