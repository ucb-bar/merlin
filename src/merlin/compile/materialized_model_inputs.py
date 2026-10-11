"""Exact input binding for a trusted materializer, separate from M2M receipts.

Only the trusted caller selects this binding after its own source/ABI validator
has completed. It establishes unchanged materialized bytes, never compiler or
numerical qualification and never impersonates the original capture issuer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from merlin.compile.model_execution_inputs import file_sha256


@dataclass(frozen=True)
class MaterializedModelInputs:
    root: Path
    identity_bytes: bytes
    validator_identity_bytes: bytes
    source_identity_bytes: bytes

    @classmethod
    def bind(cls, root, *, artifacts, validator_identity, source_identity):
        root = Path(root).resolve(strict=True)
        if not isinstance(validator_identity, dict) or not validator_identity.get("worker_sha256"):
            raise ValueError("trusted model materializer needs its selected validator identity")
        if not isinstance(source_identity, dict) or not source_identity.get("content_sha256") or not source_identity.get("artifacts"):
            raise ValueError("trusted model materializer needs exact frozen source identities")
        if not isinstance(artifacts, dict) or not {
            "model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
            "inputs.npz", "input_order.json", "golden.npy", "goldens.npz", "output_order.json",
        } <= set(artifacts):
            raise ValueError("materialized model needs complete source, ABI, and output artifacts")
        bound = cls(root, json.dumps(artifacts, sort_keys=True).encode(), json.dumps(validator_identity, sort_keys=True).encode(), json.dumps(source_identity, sort_keys=True).encode())
        bound.verify(root)
        return bound

    def verify(self, root):
        if type(self) is not MaterializedModelInputs or Path(root).resolve(strict=True) != self.root:
            raise ValueError("materialized model binding names another capture")
        artifacts = json.loads(self.identity_bytes)
        for name, record in artifacts.items():
            if Path(name).name != name or not isinstance(record, dict):
                raise ValueError("invalid materialized model artifact")
            path = self.root / name
            if path.is_symlink() or not path.is_file() or file_sha256(path) != record.get("sha256"):
                raise ValueError(f"materialized model artifact changed: {name}")
            if path.stat().st_size != record.get("size_bytes"):
                raise ValueError(f"materialized model artifact extent changed: {name}")
        return artifacts

    def record(self):
        return {"scope": "trusted runtime materialization; no M2M issuer or compiled qualification claim",
                "artifacts": json.loads(self.identity_bytes), "validator": json.loads(self.validator_identity_bytes), "source": json.loads(self.source_identity_bytes)}
