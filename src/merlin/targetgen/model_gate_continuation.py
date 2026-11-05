"""Trusted operator-only admission of a model's deeper native continuation.

This token changes scheduling only. Numerical/source/dispatch/full-tier gates
still execute normally; request flags and public ledgers cannot construct it.
"""

from dataclasses import dataclass, field
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


@dataclass
class ModelGatePrerequisites:
    """Host-issued scheduling evidence; never a model/numerical certificate.

    The trusted completion owner updates qualified only after reopening original
    products. The fixed denominator comes from the selected correctness policy.
    """
    target: str
    models: dict[str, str]
    denominator: frozenset[str]
    qualified: set[str]
    scope_sha256: str
    declaration_sources: dict[str, str] = field(default_factory=dict)

    def verify_declarations(self):
        # Original policy declarations bind the fixed denominator; public
        # projections bind what is actually graded. Neither can drift after
        # issuance, even when a worker grades an identical private copy.
        for filename, digest in self.declaration_sources.items():
            path = Path(filename)
            if (path.is_symlink() or not path.is_file()
                    or sha256(path.read_bytes()).hexdigest() != digest):
                raise ValueError("rolling model gate source declaration changed")

    def fraction(self, capsule, *, target):
        name = capsule.get("name")
        if name not in self.models:
            return None
        if target != self.target or capsule.get("kind") != "model":
            raise ValueError("rolling model gate target/kind changed")
        self.verify_declarations()
        path = Path(capsule["__dir__"]) / "capsule.yaml"
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != self.models[name]:
            raise ValueError("rolling model gate declaration changed")
        return len(self.qualified & self.denominator) / len(self.denominator) if self.denominator else 0.0


def prerequisite_fraction(capsule, prerequisite, *, target):
    if prerequisite is None:
        return None
    if type(prerequisite) is not ModelGatePrerequisites:
        raise ValueError("rolling model gate requires host-issued prerequisites")
    return prerequisite.fraction(capsule, target=target)
