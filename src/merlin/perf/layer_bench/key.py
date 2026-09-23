"""The identity of one per-layer measurement.

A cycle count is a fact about a specific program on a specific engine modelling a specific design,
built by a specific harness. Every one of those is a field here, and the digest covers all of them, so
a cached receipt is reused only when nothing that could change the number has changed. Change the
schedule, the numerics contract, the emitter, the engine binary or the harness and the key changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

PROTOCOLS = ("warm_then_measured", "cold_single")


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class LayerKey:
    target: str
    #: Hardware-pin artifact name (``merlin/contract/hardware_pins.yaml``) of the modelled design.
    design_pin: str
    #: sha256 of the engine binary that produced the cycles.
    engine_sha256: str
    #: Canonical signature of the fused group: op chain, shapes, dtypes, layouts.
    group_signature: str
    #: ``NumericsContract.digest()`` of the contract the group's output is graded under.
    contract_digest: str
    #: Digest of the canonical schedule that lowered the group.
    schedule_digest: str
    #: Digest of the emitter / instruction-library code the group's lowering reached.
    emitter_digest: str
    harness_version: str
    protocol: str

    def __post_init__(self) -> None:
        for name in ("engine_sha256", "contract_digest", "schedule_digest", "emitter_digest"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"LayerKey.{name} must be a lowercase sha256 hex digest")
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol {self.protocol!r}; known {PROTOCOLS}")
        for name in ("target", "design_pin", "group_signature", "harness_version"):
            if not getattr(self, name):
                raise ValueError(f"LayerKey.{name} must be non-empty")

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
