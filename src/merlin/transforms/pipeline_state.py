"""Shared pass state and invalidation, independent of transform execution brokers.

These are source/context custody checks, not proof or execution authority.
The structured transform driver reexports the same types and owns its verifier.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .certificates import ir_digest


@dataclass(frozen=True)
class PipelineContext:
    """Identities of the selected contracts, not declarations of hardware support.

    Missing identities remain unknown. They cannot authorize an optimization
    requiring hardware, numerical or execution facts. Hosts supply their own
    reviewed contracts; an author cannot obtain permissions by naming a digest.
    """

    hardware: str | None = None
    numerical_policy: str | None = None
    resources: str | None = None
    execution: str | None = None
    library: str | None = None

    def __post_init__(self):
        for identity in vars(self).values():
            if identity is not None and (not isinstance(identity, str) or not identity.strip()):
                raise ValueError("pipeline contract identities must be nonempty strings or unknown")


@dataclass(frozen=True)
class AnalysisStamp:
    source_sha256: str
    context: PipelineContext
    ir_level: str


@dataclass(frozen=True)
class PassSpec:
    """Small execution contract shared by structured and ordinary model passes."""

    name: str
    stage: int
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    mutates_ir: bool = True
    ir_level: str = "structured"

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.ir_level, str)
            or not self.ir_level
            or type(self.stage) is not int
            or self.stage < 0
        ):
            raise ValueError("passes require a name, IR level and nonnegative stage")
        if type(self.mutates_ir) is not bool:
            raise ValueError("pass mutation declaration must be Boolean")
        for names in (self.requires, self.produces, self.invalidates):
            if (
                type(names) is not tuple
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError("analysis declarations require unique nonempty names")


@dataclass(frozen=True)
class PassRecord:
    name: str
    before: AnalysisStamp
    after: AnalysisStamp
    invalidated: tuple[str, ...]
    produced: tuple[str, ...]
    repair_round: int


def _dependent_invalidations(roots, dependencies):
    invalidated = set(roots)
    while True:
        following = invalidated | {name for name, inputs in dependencies.items() if inputs & invalidated}
        if following == invalidated:
            return invalidated
        invalidated = following


def validate_pass_order(specs: tuple[PassSpec, ...]) -> None:
    """Reject explicit ordering errors before executing any transformation."""
    available: set[str] = set()
    seen: set[str] = set()
    dependencies: dict[str, set[str]] = {}
    stage = -1
    level = specs[0].ir_level if specs else "structured"
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"duplicate pass: {spec.name}")
        if spec.ir_level != level:
            raise ValueError(f"pass {spec.name} requires a different IR level")
        if spec.stage < stage:
            raise ValueError(f"invalid explicit pass ordering before {spec.name}")
        missing = set(spec.requires) - available
        if missing:
            raise ValueError(f"pass {spec.name} lacks prerequisites: {', '.join(sorted(missing))}")
        # These analyses describe the current IR. A potentially mutating pass
        # cannot carry them into an explicitly ordered consumer unless it
        # produces fresh facts itself, even if its declaration omitted '*'.
        roots = (
            available
            if spec.mutates_ir or "*" in spec.invalidates
            else set(spec.invalidates) | (set(spec.produces) & available)
        )
        invalidated = _dependent_invalidations(roots, dependencies)
        available.difference_update(invalidated)
        dependencies = {name: inputs for name, inputs in dependencies.items() if name not in invalidated}
        inputs = set(spec.requires) & available
        dependencies.update((name, inputs | (set(spec.produces) - {name})) for name in spec.produces)
        available.update(spec.produces)
        seen.add(spec.name)
        stage = spec.stage


class PipelineSession:
    """Current-IR analysis custody and bounded repair, without proof authority.

    Consumers still validate their analyses independently. Fingerprints detect
    out-of-band mutations, including changes made without this runner. A pass
    failure poisons this private session; its partially rewritten IR cannot be
    consumed or reported as a successful compilation.
    """

    def __init__(
        self,
        module,
        *,
        context=None,
        ir_level="structured",
        max_repairs=2,
        fingerprint=None,
        verify_ir=None,
    ):
        if type(max_repairs) is not int or max_repairs < 0:
            raise ValueError("repair budget must be a nonnegative integer")
        self.module = module
        self.context = context if context is not None else PipelineContext()
        if type(self.context) is not PipelineContext or not ir_level:
            raise ValueError("explicit pipeline context and IR level required")
        self.ir_level = ir_level
        if (fingerprint is None) != (verify_ir is None) or (
            fingerprint is not None and (not callable(fingerprint) or not callable(verify_ir))
        ):
            raise ValueError("an alternate IR requires both explicit fingerprint and structural verifier callbacks")
        self._fingerprint = ir_digest if fingerprint is None else fingerprint
        self._verify_ir = (lambda value: value.verify()) if verify_ir is None else verify_ir
        self.max_repairs = max_repairs
        self.repair_round = 0
        self.records: list[PassRecord] = []
        self._analyses: dict[str, tuple[AnalysisStamp, object]] = {}
        self._dependencies: dict[str, set[str]] = {}
        self._stage = -1
        self._failed = False

    def stamp(self):
        if self._failed:
            raise ValueError("pipeline session failed; discard its private IR")
        digest = self._fingerprint(self.module)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("pipeline IR fingerprint must be a SHA-256 hex identity")
        return AnalysisStamp(digest, self.context, self.ir_level)

    def analysis(self, name):
        stamp, value = self._analyses.get(name, (None, None))
        if stamp != self.stamp():
            raise ValueError(f"missing or stale analysis: {name}")
        return value

    def run(self, spec: PassSpec, action: Callable[[], Mapping[str, object]]):
        before = self.stamp()
        if spec.ir_level != self.ir_level or spec.stage < self._stage:
            raise ValueError(f"pass {spec.name} has invalid IR level or ordering")
        for name in spec.requires:
            self.analysis(name)
        try:
            products = action()
            if not isinstance(products, Mapping):
                raise ValueError(f"pass {spec.name} did not return a mapping of produced analyses")
            products = dict(products)
            if self._verify_ir(self.module) is not None:
                raise ValueError("pipeline structural verifier must raise on failure and return None on success")
            after = self.stamp()
            if before.context != after.context:
                raise ValueError(f"pass {spec.name} changed the selected contracts")
            if before.ir_level != after.ir_level:
                raise ValueError(f"pass {spec.name} changed IR level without an explicit lowering boundary")
            if not spec.mutates_ir and before != after:
                raise ValueError(f"analysis pass {spec.name} changed IR or context")
            if set(products) != set(spec.produces):
                raise ValueError(f"pass {spec.name} produced a different analysis set")
        except BaseException:
            self._failed = True
            self._analyses.clear()
            self._dependencies.clear()
            raise
        # Every IR change invalidates old stamps, even if a pass forgot to
        # declare an invalidation. Context changes likewise cannot retain facts.
        invalidated = set(self._analyses) if before != after or "*" in spec.invalidates else set(spec.invalidates)
        # Replacing an analysis (for example, a new encoded byte image) also
        # invalidates analyses derived from its previous version, even when
        # the owning schedule IR itself did not change.
        invalidated.update(set(products) & self._analyses.keys())
        invalidated = _dependent_invalidations(invalidated, self._dependencies)
        invalidated.intersection_update(self._analyses)
        for name in invalidated:
            del self._analyses[name]
            self._dependencies.pop(name, None)
        inputs = set(spec.requires) & self._analyses.keys()
        self._dependencies.update((name, inputs | (set(products) - {name})) for name in products)
        self._analyses.update((name, (after, value)) for name, value in products.items())
        self.records.append(
            PassRecord(spec.name, before, after, tuple(sorted(invalidated)), spec.produces, self.repair_round)
        )
        self._stage = spec.stage

    def repair(self, *, stage: int, reason: str):
        """Invalidate facts before replanning an already selected legal fallback.

        This does not manufacture a fallback, undo a failed pass, or approve a
        changed program. The caller must restore/check its own private candidate.
        """
        self.stamp()
        if type(stage) is not int or stage < 0 or not reason:
            raise ValueError("repair requires a nonnegative restart stage and reason")
        if self.repair_round >= self.max_repairs:
            raise ValueError(f"pipeline repair budget exhausted: {reason}")
        self.repair_round += 1
        self._analyses.clear()
        self._dependencies.clear()
        self._stage = stage
