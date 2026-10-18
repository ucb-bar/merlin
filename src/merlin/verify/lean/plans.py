"""Authored boundary rosters, resolved before execution and independent of observed records.

A matching roster establishes recording completeness at its stated granularity. Semantic checking
and byte-bound chain replay remain separate requirements; a plan never grants a pass permission.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import stages


class PlanMismatch(ValueError):
    """The capture does not contain exactly the independently expected ordered occurrences."""


@dataclass(frozen=True)
class Boundary:
    name: str
    pass_entry: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.pass_entry, str)
            or not self.pass_entry
        ):
            raise ValueError("a planned boundary requires a name and pass entry")


@dataclass(frozen=True)
class Plan:
    scope: str
    boundaries: tuple[Boundary, ...]
    granularity: str = "individual-boundary-occurrences"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scope, str)
            or not self.scope
            or not isinstance(self.granularity, str)
            or not self.granularity
        ):
            raise ValueError("a boundary plan requires its scope and granularity")
        if type(self.boundaries) is not tuple or not self.boundaries:
            raise ValueError("a boundary plan requires an immutable nonempty tuple")
        if any(type(b) is not Boundary for b in self.boundaries):
            raise ValueError("a boundary plan requires Boundary occurrences")

    def canonical_bytes(self) -> bytes:
        payload = {
            "schema": "merlin.lean.boundary_plan.v1",
            "scope": self.scope,
            "granularity": self.granularity,
            "boundaries": [{"boundary": b.name, "pass": b.pass_entry} for b in self.boundaries],
        }
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validate(self, published: list[Path]) -> None:
        """Require every occurrence, including unchanged ones, with exact order and pass identity."""
        if len(published) != len(self.boundaries):
            raise PlanMismatch(f"expected {len(self.boundaries)} boundary occurrences, recorded {len(published)}")
        for ordinal, (path, expected) in enumerate(zip(published, self.boundaries, strict=True)):
            meta = json.loads((path / "meta.json").read_bytes())
            if path.name != f"{ordinal:04d}_{expected.name}":
                raise PlanMismatch(f"boundary occurrence {ordinal}: unexpected directory {path.name}")
            if (meta.get("boundary"), meta.get("pass")) != (expected.name, expected.pass_entry):
                raise PlanMismatch(f"boundary occurrence {ordinal}: expected {expected.name} / {expected.pass_entry}")


def replay(expected: Plan, manifest: Path, root: Path) -> None:
    """Check a retained roster against a caller's trusted expected plan, then check actual records.

    Loading expectations from the recorded boundaries, or trusting a recorded plan hash alone,
    would let an omitted boundary define its own denominator. The caller supplies the independently
    resolved plan. This replays only the roster obligation, not the semantic chain or source proofs.
    """
    if manifest.read_bytes() != expected.canonical_bytes():
        raise PlanMismatch("retained boundary plan differs from the independently expected plan")
    published = stages.completed(root)
    reserved = root / ".reserved"
    if not reserved.is_dir() or sorted(p.name for p in reserved.iterdir()) != [
        f"{n:06d}" for n in range(len(published))
    ]:
        raise PlanMismatch("recording has incomplete or unexpected reservations")
    published_set = set(published)
    if any(p.name.startswith(".writing-") for p in root.iterdir()) or any(
        p.is_dir() and not p.name.startswith(".") and p not in published_set for p in root.iterdir()
    ):
        raise PlanMismatch("recording has unpublished boundary directories")
    expected.validate(published)


def preprocessing() -> Plan:
    """Resolve the structural preprocessing roster before running its parser or transformations.

    The authored rewrite table supplies one occurrence per slot, even when it changes no bytes.
    The optional reassociation occurrence is included exactly where preprocessing selects it. A
    changed selection during execution causes a roster mismatch rather than silently widening scope.
    """
    from merlin.llvmlower import passes_xdsl

    selected = passes_xdsl._layer_norm_chunked_sums_selected()
    if selected and sum(statistic == "quant_ext_lowered" for statistic, _, _ in passes_xdsl._PREPROCESS) != 1:
        raise ValueError("selected chunked sums requires exactly one quantization preprocessing slot")
    prefix = passes_xdsl.__name__
    boundaries = [Boundary("xdsl-parse", f"{prefix}.parse_mlir_text")]
    for statistic, stage, name in passes_xdsl._PREPROCESS:
        boundaries.append(Boundary(stage, f"{prefix}.{name}"))
        if statistic == "quant_ext_lowered" and selected:
            boundaries.append(
                Boundary(
                    "xdsl-layer-norm-chunked-sums",
                    "merlin.llvmlower.normalization_reassociation.chunk_layer_norm_sums",
                )
            )
    return Plan("lower_model structural preprocessing only", tuple(boundaries), "individual-preprocessing-boundaries")
