"""Checkpoint observations and independent trust policy, without model imports.

The weights have an independent reference digest. A successful Hugging Face
download at the full revision commit also authenticates the downloaded metadata.
Later local reads observe metadata bytes without trusting a local receipt.
"""

import hashlib
from pathlib import Path

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
TRUSTED_SHA256 = {
    "model.safetensors": "58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945",
}


def identify(checkpoint, *, downloaded_revision=None):
    """Check trusted hashes and observe the five requested checkpoint files.

    Only the downloader may supply downloaded_revision, immediately after a
    successful forced online metadata downloads at REVISION. A directory name or a local JSON
    manifest cannot establish that provenance for subsequent local reads.
    """
    if downloaded_revision is not None and downloaded_revision != REVISION:
        raise ValueError("Download must use the full pinned Hugging Face revision")
    records = {}
    for name in FILES:
        path = Path(checkpoint) / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        observed = digest.hexdigest()
        trusted = TRUSTED_SHA256.get(name)
        if trusted is not None and observed != trusted:
            raise RuntimeError(f"Checkpoint file differs from trusted digest: {name}")
        source = ("independent_sha256" if trusted is not None else
                  f"huggingface:{MODEL_ID}@{REVISION}" if downloaded_revision else None)
        records[name] = dict(sha256=observed, bytes=path.stat().st_size,
                             authenticated=source is not None,
                             authentication_source=source)
    result = dict(requested_model=MODEL_ID, requested_revision=REVISION,
                  authentication_scope="listed_files_only",
                  checkpoint_authenticated=all(row["authenticated"] for row in records.values()),
                  files=records)
    if downloaded_revision is not None:
        result["downloaded_revision"] = downloaded_revision
    return result


def authenticate(checkpoint, *, downloaded_revision=None):
    """Require authentic weights; metadata without a trust anchor stays observed.

    Additional independently trusted hashes, when supplied by this source, are
    enforced too. Never promote checkpoint-identity.json to a trust anchor.
    """
    if "model.safetensors" not in TRUSTED_SHA256:
        raise RuntimeError("Required authenticated weights digest is missing")
    return identify(checkpoint, downloaded_revision=downloaded_revision)
