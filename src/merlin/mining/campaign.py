"""Holdout-safe, non-agentic acceptance policy for compiler-optimization mining.

The search mechanism may be a beam, a grid, or an agent-authored pass proposal.  Acceptance is always
deterministic: paper models are forbidden during development; generic capsules are assigned to fixed
train/validation/heldout partitions by content-stable hash; correctness, emitted-code change, noise
margin, and cross-family validation all gate promotion. Two consecutive empty sweeps permit freezing.
"""
from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

from merlin.common.artifacts import utc_stamp
from merlin.common.schemas import validate_or_raise


@dataclass(frozen=True)
class PartitionPolicy:
    modulus: int = 5
    train: tuple[int, ...] = (0, 1, 2)
    validation: tuple[int, ...] = (3,)
    heldout: tuple[int, ...] = (4,)

    def __post_init__(self) -> None:
        assigned = set(self.train) | set(self.validation) | set(self.heldout)
        if self.modulus < 3 or assigned != set(range(self.modulus)):
            raise ValueError("partition buckets must cover range(modulus) exactly")
        if set(self.train) & set(self.validation) or set(self.train) & set(self.heldout) or set(
                self.validation) & set(self.heldout):
            raise ValueError("train, validation, and heldout buckets must be disjoint")

    def bucket(self, capsule_id: str) -> int:
        digest = hashlib.sha256(capsule_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.modulus

    def split(self, capsule_id: str) -> str:
        bucket = self.bucket(capsule_id)
        if bucket in self.train:
            return "train"
        if bucket in self.validation:
            return "validation"
        return "heldout"

    def to_dict(self) -> dict[str, Any]:
        return {"algorithm": "sha256_mod", "modulus": self.modulus,
                "train": list(self.train), "validation": list(self.validation),
                "heldout": list(self.heldout)}


@dataclass(frozen=True)
class CandidateObservation:
    candidate: str
    action_class: str
    capsule_id: str
    family: str
    workload: str
    baseline_ns: int
    candidate_ns: int
    correctness_ok: bool
    baseline_code_digest: str
    candidate_code_digest: str

    @property
    def speedup(self) -> float:
        return self.baseline_ns / self.candidate_ns if self.candidate_ns > 0 else 0.0


@dataclass(frozen=True)
class CandidateDecision:
    candidate: str
    accepted: bool
    reasons: tuple[str, ...]
    train_median_speedup: float | None
    validation_median_speedup: float | None
    families: tuple[str, ...]
    action_classes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"candidate": self.candidate, "accepted": self.accepted,
                "reasons": list(self.reasons), "train_median_speedup": self.train_median_speedup,
                "validation_median_speedup": self.validation_median_speedup,
                "families": list(self.families), "action_classes": list(self.action_classes)}


@dataclass
class Campaign:
    excluded_models: frozenset[str]
    partition: PartitionPolicy = field(default_factory=PartitionPolicy)
    noise_margin: float = 0.02
    min_families: int = 3
    required_empty_sweeps: int = 2
    decisions: list[CandidateDecision] = field(default_factory=list)
    empty_sweeps: int = 0

    def _development_rows(self, observations: Iterable[CandidateObservation]) -> list[CandidateObservation]:
        rows = list(observations)
        leaked = sorted({row.workload for row in rows if row.workload in self.excluded_models})
        if leaked:
            raise ValueError(f"paper-model leakage into compiler development: {leaked}")
        used_heldout = sorted(row.capsule_id for row in rows
                              if self.partition.split(row.capsule_id) == "heldout")
        if used_heldout:
            raise ValueError(f"heldout capsules were supplied to candidate selection: {used_heldout}")
        return rows

    def decide(self, observations: Iterable[CandidateObservation]) -> CandidateDecision:
        rows = self._development_rows(observations)
        names = {row.candidate for row in rows}
        if len(names) != 1:
            raise ValueError("one decision requires observations for exactly one candidate")
        name = next(iter(names))
        reasons: list[str] = []
        train = [row for row in rows if self.partition.split(row.capsule_id) == "train"]
        validation = [row for row in rows if self.partition.split(row.capsule_id) == "validation"]
        if not train or not validation:
            reasons.append("both train and validation observations are required")
        if any(not row.correctness_ok for row in rows):
            reasons.append("correctness gate failed")
        if any(row.baseline_code_digest == row.candidate_code_digest for row in rows):
            reasons.append("candidate emitted byte-equivalent code for at least one capsule")
        families = tuple(sorted({row.family for row in rows}))
        if len(families) < self.min_families:
            reasons.append(f"observed {len(families)} families; need at least {self.min_families}")
        train_speedup = statistics.median(row.speedup for row in train) if train else None
        validation_speedup = statistics.median(row.speedup for row in validation) if validation else None
        threshold = 1.0 + self.noise_margin
        if train_speedup is not None and train_speedup <= threshold:
            reasons.append("training median does not clear the noise margin")
        if validation_speedup is not None and validation_speedup <= threshold:
            reasons.append("validation median does not clear the noise margin")
        if any(row.speedup < 1.0 - self.noise_margin for row in validation):
            reasons.append("candidate materially regresses a validation capsule")
        decision = CandidateDecision(
            candidate=name, accepted=not reasons, reasons=tuple(reasons),
            train_median_speedup=train_speedup, validation_median_speedup=validation_speedup,
            families=families, action_classes=tuple(sorted({row.action_class for row in rows})))
        self.decisions.append(decision)
        return decision

    def finish_sweep(self, decisions: Iterable[CandidateDecision]) -> None:
        self.empty_sweeps = 0 if any(d.accepted for d in decisions) else self.empty_sweeps + 1

    @property
    def converged(self) -> bool:
        return self.empty_sweeps >= self.required_empty_sweeps

    def freeze(self, *, development_corpus_sha256: str, policy_sha256: str,
               runtime_sha256: str) -> dict[str, Any]:
        if not self.converged:
            raise ValueError(
                f"campaign has {self.empty_sweeps} empty sweeps; needs {self.required_empty_sweeps}")
        record = {
            "version": 1, "status": "frozen", "frozen_at": utc_stamp(),
            "development_corpus_sha256": development_corpus_sha256,
            "partition": self.partition.to_dict(), "excluded_models": sorted(self.excluded_models),
            "accepted_candidates": [d.to_dict() for d in self.decisions if d.accepted],
            "rejected_candidates": [d.to_dict() for d in self.decisions if not d.accepted],
            "convergence": {"required_empty_sweeps": self.required_empty_sweeps,
                            "observed_empty_sweeps": self.empty_sweeps},
            "policy_sha256": policy_sha256, "runtime_sha256": runtime_sha256,
        }
        validate_or_raise(record, "compiler_freeze")
        return record
