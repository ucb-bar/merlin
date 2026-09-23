"""A content-addressed store of per-layer measurement receipts.

A receipt is a JSON object that carries its own key fields and a ``receipt_sha256`` over everything
else. It is stored at ``<root>/<digest[:2]>/<digest>.json`` where ``digest`` is the ``LayerKey``
digest. On read the self-hash is re-checked and the stored key must equal the requested one; a
receipt that fails either check is an error, never a cache miss that silently re-measures.

ELF bytes are not stored here: a receipt names the ELF by digest, and the ELF can be evicted.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .key import LayerKey

SELF_HASH_FIELD = "receipt_sha256"


class ReceiptError(ValueError):
    pass


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def seal_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` with its self-hash set. ``payload`` must carry ``key``."""
    if "key" not in payload:
        raise ReceiptError("a receipt must carry its key fields")
    body = {k: v for k, v in payload.items() if k != SELF_HASH_FIELD}
    return {**body, SELF_HASH_FIELD: hashlib.sha256(_canonical(body)).hexdigest()}


def verify_receipt(receipt: Mapping[str, Any]) -> None:
    body = {k: v for k, v in receipt.items() if k != SELF_HASH_FIELD}
    if receipt.get(SELF_HASH_FIELD) != hashlib.sha256(_canonical(body)).hexdigest():
        raise ReceiptError("receipt self-hash does not match its contents")


class ReceiptCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, key: LayerKey) -> Path:
        digest = key.digest()
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: LayerKey) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        receipt = json.loads(path.read_text(encoding="utf-8"))
        verify_receipt(receipt)
        if receipt.get("key") != key.to_dict():
            raise ReceiptError(f"receipt at {path} records a different key")
        return receipt

    def put(self, key: LayerKey, payload: Mapping[str, Any]) -> dict[str, Any]:
        receipt = seal_receipt({**payload, "key": key.to_dict()})
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(receipt, sort_keys=True, indent=1) + "\n")
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return receipt
