"""Finite source-proposal discovery claims; no legality or execution authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from merlin.common.digest import is_sha256


@dataclass(frozen=True)
class ProposalBudget:
    """Independent output and work-session bounds for trusted in-process hooks.

    Sessions bound matcher/checker invocations, not their instruction count or
    wall time. Process-backed untrusted compilers still need runner limits.
    """

    max_candidates: int = 256
    max_match_visits: int = 128
    max_proof_sessions: int = 384
    max_region_operations: int = 16
    max_source_operations: int = 10000
    max_tensor_elements: int = 1_000_000

    def __post_init__(self):
        limits = {"max_candidates": (1, 1024), "max_match_visits": (1, 512),
                  "max_proof_sessions": (1, 1536), "max_region_operations": (1, 64),
                  "max_source_operations": (1, 10000), "max_tensor_elements": (1, (1 << 63) - 1)}
        for name, (low, high) in limits.items():
            if type(getattr(self, name)) is not int or not low <= getattr(self, name) <= high:
                raise ValueError(f"proposal discovery requires bounded {name}")


@dataclass(frozen=True)
class SourcePattern:
    id: str
    root_operations: tuple[str, ...]
    pipeline: str
    alternatives_per_anchor: int

    def __post_init__(self):
        if (any(type(v) is not str or not v for v in (self.id, self.pipeline))
                or type(self.root_operations) is not tuple or not self.root_operations
                or len(set(self.root_operations)) != len(self.root_operations)
                or any(type(v) is not str or not v for v in self.root_operations)
                or type(self.alternatives_per_anchor) is not int or not 1 <= self.alternatives_per_anchor <= 16):
            raise ValueError("source pattern requires explicit identities and finite anchor alternatives")


@dataclass(frozen=True)
class ProposalContext:
    source_sha256: str
    selected_contract_sha256: str
    numerical_policy_sha256: str
    preparation_sha256: str
    topology_sha256: str
    provider: str
    implementation_sha256: str
    patterns: tuple[SourcePattern, ...]
    resource: str
    memory_domain: str
    budget: ProposalBudget

    def __post_init__(self):
        for name in ("source_sha256", "selected_contract_sha256", "numerical_policy_sha256",
                     "preparation_sha256", "topology_sha256", "implementation_sha256"):
            if not is_sha256(getattr(self, name)):
                raise ValueError(f"proposal discovery requires pinned {name}")
        if (any(type(v) is not str or not v for v in (self.provider, self.resource, self.memory_domain))
                or type(self.budget) is not ProposalBudget or type(self.patterns) is not tuple
                or not self.patterns or len(self.patterns) > 64
                or any(type(p) is not SourcePattern for p in self.patterns)
                or len({p.id for p in self.patterns}) != len(self.patterns)):
            raise ValueError("proposal discovery requires an immutable bounded provider context")

    def record(self):
        return asdict(self)


@dataclass(frozen=True)
class ProposalPage:
    proposals: tuple
    discovery: dict

    def __post_init__(self):
        from merlin.compiler.backend.sourceproposal import SourceProposal

        if (type(self.proposals) is not tuple or len(self.proposals) > 1024
                or any(type(p) is not SourceProposal for p in self.proposals)
                or type(self.discovery) is not dict):
            raise ValueError("proposal page requires typed immutable proposals and discovery trace")
