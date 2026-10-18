"""Authored ordinary tensor-prefix phases and an explicit native lowering barrier.

The plan is resolved from effective selections before execution. It never recovers
boundaries by parsing a rendered pipeline, and grants no semantic pass permission.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from .plans import Boundary, Plan

# Independent expected occurrences: an edited production roster must be reviewed,
# rather than defining its own proof denominator from observed native snapshots.
_TENSOR_PHASES = (
    Boundary("native-canonicalize", "canonicalize"),
    Boundary("native-cse", "cse"),
    Boundary("native-guarded-fusion", "func.func(linalg-fuse-elementwise-ops)"),
    Boundary("native-generalize", "func.func(linalg-generalize-named-ops)"),
)
_PARSE = Boundary("native-parse", "torch_mlir.ir.Module.parse")
_BUFFERIZE = "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}"


@dataclass(frozen=True)
class NativePlan:
    """Immutable complete execution order, with one explicit tensor/native boundary."""

    prefix: tuple[Boundary, ...]
    suffix: tuple[str, ...]
    index_bits: int
    data_layout: str
    fusion_guard: bool

    def __post_init__(self) -> None:
        if self.prefix != _TENSOR_PHASES:
            raise ValueError("native tensor phases differ from the authored ordinary roster")
        if type(self.suffix) is not tuple or not self.suffix or self.suffix[0] != _BUFFERIZE:
            raise ValueError("native suffix must start at the explicit bufferization barrier")
        if any(type(p) is not str or not p for p in self.suffix):
            raise ValueError("native suffix requires immutable nonempty pass entries")
        if self.suffix.count(_BUFFERIZE) != 1:
            raise ValueError("native plan requires exactly one bufferization barrier")
        if type(self.index_bits) is not int or self.index_bits not in (32, 64):
            raise ValueError("initial native verification supports explicit 32/64-bit index domains")
        if not isinstance(self.data_layout, str) or not self.data_layout:
            raise ValueError("native verification requires the actual compiler data layout")
        if self.fusion_guard is not True:
            raise ValueError("initial native verification requires ordinary guarded fusion")

    @property
    def boundaries(self) -> Plan:
        return Plan(
            "ordinary native tensor prefix before bufferization",
            (_PARSE, *self.prefix),
            "native-parse-and-authored-pass-occurrences;fusion-includes-control-hooks",
        )

    def canonical_bytes(self) -> bytes:
        payload = {
            "schema": "merlin.lean.native_plan.v1",
            "boundaries": json.loads(self.boundaries.canonical_bytes()),
            "suffix": list(self.suffix),
            "barrier": "before-one-shot-bufferize",
            "index_bits": self.index_bits,
            "data_layout": self.data_layout,
            "fusion_guard": self.fusion_guard,
        }
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def pipeline(self) -> str:
        """Render already-authored entries; the renderer does not infer any boundaries."""
        return ",".join((*[b.pass_entry for b in self.prefix], *self.suffix))


def ordinary(
    *,
    features: frozenset[str],
    index_bits: int,
    data_layout: str,
    custom_pipeline: str | None = None,
    vectorize: bool = False,
    parallel: bool = False,
    parallel_harts: int | None = None,
    parallel_chunks=None,
    transform_schedule: str | None = None,
    source_scalar_carrier=None,
    source_observation_effects=None,
    masked_contraction_effects=None,
) -> NativePlan:
    """Admit the bounded ordinary route using effective, already-resolved features.

    Production passes still determine their existing suffix. The independently
    authored tensor roster and explicit next pass prevent a changed or omitted
    shared phase from silently defining its own denominator.
    """
    from merlin.llvmlower import fusion_guard, pipeline

    if type(features) is not frozenset or features:
        raise ValueError("initial native verification requires no effective features")
    if (
        custom_pipeline is not None
        or vectorize
        or parallel
        or parallel_harts is not None
        or parallel_chunks is not None
        or transform_schedule is not None
        or source_scalar_carrier is not None
        or source_observation_effects is not None
        or masked_contraction_effects is not None
    ):
        raise ValueError("initial native verification requires ordinary serial lowering without callbacks")
    if (
        pipeline._generalize_before_fuse()
        or pipeline._fuse_post()
        or pipeline._sink_deallocs()
        or os.environ.get("MERLIN_NO_DEALLOC")
        or not fusion_guard.enabled()
    ):
        raise ValueError("initial native verification refuses nonordinary pipeline switches")
    actual = tuple(pipeline._UPSTREAM_PASSES)
    expected = tuple(b.pass_entry for b in _TENSOR_PHASES)
    if len(actual) <= len(expected) or actual[: len(expected)] != expected or actual[len(expected)] != _BUFFERIZE:
        raise ValueError("production tensor roster differs from independently authored native phases")
    suffix = []
    for entry in actual[len(expected) :]:
        suffix.extend(pipeline._dealloc_passes(sink=False) if entry == "__DEALLOC__" else [entry])
    # Bind the actual conversion entries structurally before rendering them.
    for name in pipeline._INDEX_CONVERSION_PASSES:
        if suffix.count(name) != 1:
            raise ValueError(f"native suffix has no unique bare {name} entry")
        suffix[suffix.index(name)] = f"{name}{{index-bitwidth={index_bits}}}"
    return NativePlan(_TENSOR_PHASES, tuple(suffix), index_bits, data_layout, True)
