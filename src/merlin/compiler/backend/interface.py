"""Inspectable compiler declarations over the existing source proposal protocol.

These are implementation claims. They neither admit a source program nor supply
hardware facts, numerical policy, equivalence, or observed execution evidence.
Original proposal admission and selected lowering tickets remain authoritative.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from merlin.compiler.backend.protocol import (
    CRegionABI,
    TensorPort,
    LoweringRecipe,
    ProposalBudget,
    ProposalContext,
    SourcePattern,
    validator_identity as _validator_identity,
)
from merlin.compiler.records import Artifact, Compilation, CompileOptions, Deployment, Source


def _names(values, *, what, empty=False):
    if (
        type(values) is not tuple
        or not empty
        and not values
        or any(type(value) is not str or not value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"compiler description requires immutable unique {what}")


def _pairs(values, *, what, empty=False):
    if (
        type(values) is not tuple
        or not empty
        and not values
        or any(
            type(row) is not tuple or len(row) != 2 or any(type(v) is not str or not v.strip() for v in row)
            for row in values
        )
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"compiler description requires immutable named {what}")


@dataclass(frozen=True)
class SourceDomain:
    """The IR, operations, types and named premises a compiler claims to accept."""

    ir: str
    operations: tuple[str, ...]
    types: tuple[str, ...]
    constraints: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        _names((self.ir,), what="source IR identity")
        _names(self.operations, what="source operations")
        _names(self.types, what="source types")
        _pairs(self.constraints, what="source constraints", empty=True)
        if len(dict(self.constraints)) != len(self.constraints):
            raise ValueError("source constraints require distinct names")


@dataclass(frozen=True)
class TargetPipeline:
    """A named input-IR to target-IR to artifact route, independent of LLVM."""

    id: str
    input_ir: str
    target_ir: str
    artifacts: tuple[tuple[str, str], ...]
    abi: str
    passes: tuple[str, ...]

    def __post_init__(self):
        for value in (self.id, self.input_ir, self.target_ir, self.abi):
            _names((value,), what="pipeline identities")
        _pairs(self.artifacts, what="artifact kind/role pairs")
        # Repeated passes are meaningful; pipeline order is preserved exactly.
        if (
            type(self.passes) is not tuple
            or not self.passes
            or any(type(value) is not str or not value.strip() for value in self.passes)
        ):
            raise ValueError("compiler pipeline requires ordered declared passes")


@dataclass(frozen=True)
class ImplementationFamily:
    """Named alternatives reuse source patterns and state their fusion premises.

    Several families may share one pattern. Preconditions describe obligations;
    they are not evaluated as predicates or treated as proofs by this record.
    """

    id: str
    patterns: tuple[str, ...]
    pipeline: str
    fusion_preconditions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        _names((self.id,), what="implementation family identity")
        _names((self.pipeline,), what="family pipeline identity")
        _names(self.patterns, what="family source pattern identities")
        _pairs(self.fusion_preconditions, what="fusion preconditions", empty=True)
        if len(dict(self.fusion_preconditions)) != len(self.fusion_preconditions):
            raise ValueError("fusion preconditions require distinct names")


@dataclass(frozen=True)
class CompilerDescription:
    provider: str
    source_domain: SourceDomain
    patterns: tuple[SourcePattern, ...]
    families: tuple[ImplementationFamily, ...]
    pipelines: tuple[TargetPipeline, ...]

    def __post_init__(self):
        _names((self.provider,), what="provider identity")
        if type(self.source_domain) is not SourceDomain:
            raise ValueError("compiler description needs a typed source domain")
        self.source_domain.__post_init__()
        for records, record_type, label in (
            (self.patterns, SourcePattern, "patterns"),
            (self.families, ImplementationFamily, "families"),
            (self.pipelines, TargetPipeline, "pipelines"),
        ):
            if type(records) is not tuple or not records or any(type(row) is not record_type for row in records):
                raise ValueError(f"compiler description needs immutable typed {label}")
            _names(tuple(row.id for row in records), what=label)
            for row in records:
                row.__post_init__()
        patterns = {row.id for row in self.patterns}
        pipelines = {row.id for row in self.pipelines}
        if any(
            pattern.pipeline not in pipelines or not set(pattern.root_operations) <= set(self.source_domain.operations)
            for pattern in self.patterns
        ):
            raise ValueError("compiler patterns differ from declared source operations or pipelines")
        if any(pipeline.input_ir != self.source_domain.ir for pipeline in self.pipelines):
            raise ValueError("compiler pipeline input differs from its source domain")
        if any(family.pipeline not in pipelines or not set(family.patterns) <= patterns for family in self.families):
            raise ValueError("compiler family names an undeclared pattern or pipeline")
        if set().union(*(set(family.patterns) for family in self.families)) != patterns:
            raise ValueError("compiler families must account for every described source pattern")

    def record(self):
        self.__post_init__()
        return asdict(self)

    def validate_context(self, context: ProposalContext):
        if type(context) is not ProposalContext or (context.provider, context.patterns) != (
            self.provider,
            self.patterns,
        ):
            raise ValueError("proposal context differs from its compiler description")

    def validate_recipe(self, recipe: LoweringRecipe):
        if type(recipe) is not LoweringRecipe:
            raise ValueError("described compiler offers require an existing typed LoweringRecipe")
        recipe.__post_init__()
        pipeline = next((row for row in self.pipelines if row.id == recipe.pipeline), None)
        if (
            pipeline is None
            or not any(family.pipeline == recipe.pipeline for family in self.families)
            or recipe.abi != pipeline.abi
            or not set(recipe.artifacts) <= set(pipeline.artifacts)
        ):
            raise ValueError("lowering recipe differs from its declared compiler pipeline, artifacts or ABI")


class CompilerInterface:
    """Snapshot an optional provider declaration without replacing admission."""

    def __init__(self, provider, name):
        self.provider = provider
        self.name = name
        hook = getattr(provider, "describe_compiler", None)
        if hook is not None and not callable(hook):
            raise ValueError("describe_compiler must be callable when supplied")
        self._hook = _validator_identity(hook) if hook is not None else None
        self.description = hook() if hook is not None else None
        if hook is not None and type(self.description) is not CompilerDescription:
            raise ValueError("describe_compiler must return a typed CompilerDescription")
        if self.description is not None and self.description.provider != name:
            raise ValueError("compiler description changed its selected provider identity")
        self._snapshot = self._record_description(self.description)
        self.revalidate()

    @staticmethod
    def _record_description(description):
        return json.dumps(description.record(), sort_keys=True, allow_nan=False) if description is not None else None

    def revalidate(self, *, patterns=None):
        hook = getattr(self.provider, "describe_compiler", None)
        current = _validator_identity(hook) if callable(hook) else None
        if (
            hook is not None
            and not callable(hook)
            or (current is None) != (self._hook is None)
            or current is not None
            and any(a is not b for a, b in zip(current, self._hook, strict=True))
            or self._record_description(self.description) != self._snapshot
        ):
            raise ValueError("compiler description hook or original declaration changed")
        described = hook() if hook is not None else None
        if type(described) is not type(self.description) or self._record_description(described) != self._snapshot:
            raise ValueError("compiler declaration changed during discovery; rediscover proposals")
        if self.description is not None:
            describe = getattr(self.provider, "describe", None)
            actual = describe() if callable(describe) else patterns
            if actual is not None and actual != self.description.patterns:
                raise ValueError("legacy pattern description differs from the compiler interface")
            if patterns is not None and patterns != self.description.patterns:
                raise ValueError("bound source patterns differ from the compiler interface")

    def validate_context(self, context):
        self.revalidate(patterns=context.patterns)
        if self.description is not None:
            self.description.validate_context(context)

    def validate_recipe(self, recipe):
        self.revalidate()
        if self.description is not None:
            self.description.validate_recipe(recipe)

    def record(self):
        self.revalidate()
        return {
            "schema": "merlin.compiler_interface.v1",
            "provider": self.name,
            "status": "described" if self.description is not None else "legacy_no_compiler_description",
            "description": self.description.record() if self.description is not None else None,
            "authority": "provider implementation claims; original source, numerical and hardware admission still required",
        }


def _source_members():
    import sys

    owner = sys.modules[__name__]
    return (
        (owner, "_source_members"),
        (owner, "_names"),
        (owner, "_pairs"),
        *(
            (record, "__post_init__")
            for record in (SourceDomain, TargetPipeline, ImplementationFamily, CompilerDescription)
        ),
        *((CompilerDescription, name) for name in ("record", "validate_context", "validate_recipe")),
        *(
            (CompilerInterface, name)
            for name in (
                "__init__",
                "_record_description",
                "revalidate",
                "validate_context",
                "validate_recipe",
                "record",
            )
        ),
    )


__all__ = [
    "CRegionABI",
    "TensorPort",
    "Artifact",
    "Compilation",
    "CompileOptions",
    "Deployment",
    "Source",
    "LoweringRecipe",
    "ProposalBudget",
    "ProposalContext",
    "SourcePattern",
    "SourceDomain",
    "TargetPipeline",
    "ImplementationFamily",
    "CompilerDescription",
    "CompilerInterface",
]
