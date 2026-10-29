"""Common callable transformations; verification authority belongs to the host."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from xdsl.dialects import builtin

from .certificates import TRANSFORM_SCOPES, TransformationChange, Witness, ir_digest, ir_text, make_change
from .loops import segment_loops
from .normalization import normalize_shapes, simplify_integers
from .pipeline_state import (
    AnalysisStamp,
    PassRecord,
    PassSpec,
    PipelineContext,
    PipelineSession,
    validate_pass_order,
)
from .planning import DependencyPlan, StoragePlan, apply_order, plan_dependencies, plan_storage, reuse_storage
from .validation import validate_change


@dataclass(frozen=True)
class PipelineOptions:
    index_bits: int
    tile_iterations: tuple[int, ...] = ()
    normalize_shapes: bool = True
    simplify_integers: bool = True
    reuse_buffers: bool = True
    dependency_orders: tuple[tuple[int, tuple[int, ...]], ...] = ()
    eliminate_redundancy: bool = False
    hoist_invariants: bool = False
    unroll_factors: tuple[int, ...] = ()
    max_added_operations: int = 4096
    pass_order: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    module: builtin.ModuleOp
    changes: tuple[TransformationChange, ...]
    storage_plan: StoragePlan
    dependency_plan: DependencyPlan
    plan_only: bool
    analyses_stamp: AnalysisStamp | None = None
    passes: tuple[PassRecord, ...] = ()


def apply_common_pipeline(
    module: builtin.ModuleOp,
    options: PipelineOptions,
    *,
    verifier: Callable[[TransformationChange], object] | None = None,
    plan_only: bool = False,
    context: PipelineContext | None = None,
) -> PipelineResult:
    """Transform a private clone; dispatch every covered change to verification.

    ``verifier`` is supplied by the host's trusted broker. This API creates no
    formal receipts and cannot turn a callback return value into proof authority.
    The broker must supply live checker completion. Without
    that hook, only explicit ``plan_only`` invocations can produce changed IR.
    No callback, successful numerical test or structural certificate proves a
    transformation by itself.
    """
    if type(plan_only) is not bool:
        raise ValueError("plan_only must be an explicit boolean; data values cannot disable the proof gate")
    if options.index_bits <= 1 or any(tile <= 0 for tile in options.tile_iterations):
        raise ValueError("positive tiles and an explicit signed index width are required")
    module.verify()
    current = module.clone()
    changes: list[TransformationChange] = []
    session = PipelineSession(current, context=context)
    actions: dict[str, tuple[PassSpec, Callable[[], Mapping[str, object]]]] = {}

    def register(name, stage, action, *, requires=(), produces=(), mutates=True):
        actions[name] = (PassSpec(name, stage, requires, produces, ("*",) if mutates else (), mutates), action)

    def run(kind: str, action: Callable[[], tuple[Witness, ...]], parameters: dict[str, Any], scope: tuple[str, ...]):
        before = ir_text(current)
        witnesses = action()
        if not witnesses:
            if before != ir_text(current):
                raise ValueError("transformation changed IR without a structural witness")
            return {}
        if scope != TRANSFORM_SCOPES[kind]:
            raise ValueError("internal transformation coverage differs from its registry")
        change = make_change(kind, before, current, parameters, witnesses, scope)
        validate_change(change)
        if verifier is None and not plan_only:
            raise ValueError("covered transformation requires the host verifier; use plan_only for unverified planning")
        if verifier is not None:
            receipt = verifier(change)
            if not plan_only:
                from merlin.verify.lean_checker import accepted_pipeline_receipt

                if not accepted_pipeline_receipt(receipt, change):
                    raise ValueError("covered transformation lacks fresh whole-program Lean completion for this exact change")
        changes.append(change)
        return {}

    def transform(kind, stage, action, parameters=None, requires=()):
        register(kind, stage, lambda: run(kind, action, parameters or {}, TRANSFORM_SCOPES[kind]), requires=requires)

    if options.normalize_shapes:
        transform("shape_view_normalization", 0, lambda: normalize_shapes(current))
    if options.simplify_integers:
        transform("exact_integer_simplification", 0, lambda: simplify_integers(current))
    if options.eliminate_redundancy:
        from .pure_scalar import eliminate_scalar_redundancy

        transform("scalar_redundancy_elimination", 0, lambda: eliminate_scalar_redundancy(current))
    if options.hoist_invariants:
        from .pure_scalar import hoist_scalar_invariants

        transform("scalar_loop_invariant_motion", 0, lambda: hoist_scalar_invariants(current))
    if options.tile_iterations:
        transform(
            "ordered_loop_segmentation",
            2,
            lambda: segment_loops(current, options.tile_iterations, index_bits=options.index_bits),
            {"tile_iterations": options.tile_iterations, "index_bits": options.index_bits},
        )
    if options.unroll_factors:
        from .unrolling import unroll_ordered_loops

        transform(
            "ordered_loop_unrolling",
            2,
            lambda: unroll_ordered_loops(
                current,
                options.unroll_factors,
                index_bits=options.index_bits,
                max_added_operations=options.max_added_operations,
            ),
            {
                "factors": options.unroll_factors,
                "index_bits": options.index_bits,
                "max_added_operations": options.max_added_operations,
            },
        )

    if options.reuse_buffers:
        # This narrow source rewrite needs the original rectangular initializer.
        # Physical allocation belongs after tiling; this is normalization of
        # already explicit, disjoint source lifetimes, not device placement.
        register(
            "storage_analysis", 1, lambda: {"storage": plan_storage(current)}, produces=("storage",), mutates=False
        )
        transform("buffer_reuse", 1, lambda: reuse_storage(current, session.analysis("storage")), requires=("storage",))
    if options.dependency_orders:
        register(
            "dependency_analysis",
            4,
            lambda: {"dependencies": plan_dependencies(current)},
            produces=("dependencies",),
            mutates=False,
        )

        def schedule():
            # Each requested order is separately witnessed against the IR at
            # that point. apply_order independently reconstructs dependencies.
            session.analysis("dependencies")
            for block_index, order in options.dependency_orders:
                run(
                    "sequential_dependency_schedule",
                    lambda: (apply_order(current, block_index, order),),
                    {"block_index": block_index, "order": order},
                    TRANSFORM_SCOPES["sequential_dependency_schedule"],
                )
            return {}

        register("sequential_dependency_schedule", 4, schedule, requires=("dependencies",))

    def final_analyses():
        # Keep the historical standalone source closure intact. Rich alias and
        # liveness analyses remain separately callable; this fact records the
        # verified current structure, not a new semantic proof.
        current.verify()
        return {
            "structure": ir_digest(current),
            "storage": plan_storage(current),
            "dependencies": plan_dependencies(current),
        }

    register("final_analyses", 5, final_analyses, produces=("structure", "storage", "dependencies"), mutates=False)
    register("final_verification", 5, lambda: {}, requires=("structure", "storage", "dependencies"), mutates=False)
    names = options.pass_order or tuple(sorted(actions, key=lambda name: actions[name][0].stage))
    if type(names) is not tuple or len(names) != len(actions) or set(names) != set(actions):
        raise ValueError("explicit pass order must name every selected pass exactly once")
    specs = tuple(actions[name][0] for name in names)
    validate_pass_order(specs)
    for name in names:
        session.run(*actions[name])
    return PipelineResult(
        current,
        tuple(changes),
        session.analysis("storage"),
        session.analysis("dependencies"),
        plan_only,
        session.stamp(),
        tuple(session.records),
    )
