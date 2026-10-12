"""Common callable transformations; verification authority belongs to the host."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from xdsl.dialects import builtin

from .certificates import TRANSFORM_SCOPES, TransformationChange, Witness, ir_text, make_change
from .loops import segment_loops
from .normalization import normalize_shapes, simplify_integers
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


@dataclass(frozen=True)
class PipelineResult:
    module: builtin.ModuleOp
    changes: tuple[TransformationChange, ...]
    storage_plan: StoragePlan
    dependency_plan: DependencyPlan
    plan_only: bool


def apply_common_pipeline(
    module: builtin.ModuleOp,
    options: PipelineOptions,
    *,
    verifier: Callable[[TransformationChange], object] | None = None,
    plan_only: bool = False,
) -> PipelineResult:
    """Transform a private clone; dispatch every covered change to verification.

    ``verifier`` is supplied by the host's trusted broker. This API creates no
    formal receipts and cannot turn a callback return value into proof authority.
    The broker must supply live checker completion. Without
    that hook, only explicit ``plan_only`` invocations can produce changed IR.
    No callback, successful numerical test or structural certificate proves a
    transformation by itself.
    """
    if options.index_bits <= 1 or any(tile <= 0 for tile in options.tile_iterations):
        raise ValueError("positive tiles and an explicit signed index width are required")
    module.verify()
    current = module.clone()
    changes: list[TransformationChange] = []

    def run(kind: str, action: Callable[[], tuple[Witness, ...]], parameters: dict[str, Any], scope: tuple[str, ...]):
        before = ir_text(current)
        witnesses = action()
        if not witnesses:
            return
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
                    raise ValueError("covered transformation lacks fresh host Lean completion for this exact change")
        changes.append(change)

    if options.normalize_shapes:
        run(
            "shape_view_normalization",
            lambda: normalize_shapes(current),
            {},
            ("identical_memref_cast", "static_dimensions"),
        )
    if options.simplify_integers:
        run(
            "exact_integer_simplification",
            lambda: simplify_integers(current),
            {},
            ("scalar_signless_wrapping_integers",),
        )
    if options.eliminate_redundancy:
        from .pure_scalar import eliminate_scalar_redundancy

        run(
            "scalar_redundancy_elimination",
            lambda: eliminate_scalar_redundancy(current),
            {},
            TRANSFORM_SCOPES["scalar_redundancy_elimination"],
        )
    if options.hoist_invariants:
        from .pure_scalar import hoist_scalar_invariants

        run(
            "scalar_loop_invariant_motion",
            lambda: hoist_scalar_invariants(current),
            {},
            TRANSFORM_SCOPES["scalar_loop_invariant_motion"],
        )
    storage = plan_storage(current)
    if options.reuse_buffers:
        run(
            "buffer_reuse",
            lambda: reuse_storage(current, storage),
            {},
            ("static_contiguous_initialized_integer_buffers", "sequential_nonoverlapping_lifetimes"),
        )
    if options.tile_iterations:
        run(
            "ordered_loop_segmentation",
            lambda: segment_loops(current, options.tile_iterations, index_bits=options.index_bits),
            {"tile_iterations": options.tile_iterations, "index_bits": options.index_bits},
            (
                "static_nonnegative_index_bounds",
                "positive_step",
                "exact_iteration_order",
                "forwarded_loop_carried_state",
            ),
        )
    if options.unroll_factors:
        from .unrolling import unroll_ordered_loops

        run(
            "ordered_loop_unrolling",
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
            TRANSFORM_SCOPES["ordered_loop_unrolling"],
        )
    for block_index, order in options.dependency_orders:
        run(
            "sequential_dependency_schedule",
            lambda: (apply_order(current, block_index, order),),
            {"block_index": block_index, "order": order},
            ("ssa_dependencies", "sequential_effect_order"),
        )
    current.verify()
    return PipelineResult(current, tuple(changes), storage, plan_dependencies(current), plan_only)
