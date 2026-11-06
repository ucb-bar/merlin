"""Checkpoint observations and independent trust policy, without model imports.

Only the weights digest has a reviewed reference. Observed local JSON hashes
cannot authenticate a snapshot. Pretrained execution fails closed until all
consumed files have independently reviewed digests in TRUSTED_SHA256.
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


def identify(checkpoint):
    """Observe local bytes; never infer snapshot identity from local metadata."""
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
        records[name] = dict(sha256=observed, bytes=path.stat().st_size,
                             authenticated=trusted is not None)
    return dict(requested_model=MODEL_ID, requested_revision=REVISION,
                checkpoint_authenticated=all(row["authenticated"] for row in records.values()),
                files=records)


def authenticate(checkpoint):
    """Reject incomplete trust before model/config/tokenizer deserialization."""
    missing = [name for name in FILES if name not in TRUSTED_SHA256]
    if missing:
        raise RuntimeError(
            "Pretrained Qwen execution is disabled: independently trusted SHA-256 "
            "pins are missing for " + ", ".join(missing)
            + ". Observed hashes and checkpoint-identity.json are not trust anchors."
        )
    return identify(checkpoint)
