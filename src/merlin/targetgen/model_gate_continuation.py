"""Trusted operator-only admission of a model's deeper native continuation.

This token changes scheduling only. Numerical/source/dispatch/full-tier gates
still execute normally; request flags and public ledgers cannot construct it.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class ModelGateContinuation:
    capsule: str
    capsule_sha256: str
    target: str
    source_digest: str
    voucher_sha256: str


def admits(capsule, continuations, *, target, adapters):
    if set(adapters or {}) != {"L2", "L3"} or capsule.get("kind") != "model":
        return False
    for token in continuations or ():
        if type(token) is not ModelGateContinuation:
            raise ValueError("model continuation requires an operator-verified token")
        if token.capsule == capsule.get("name") and token.target == target:
            path = Path(capsule["__dir__"]) / "capsule.yaml"
            return sha256(path.read_bytes()).hexdigest() == token.capsule_sha256
    return False
