"""Subgraph partitioner — closes Step 5 (#101) and Phase 3a/b.

Walks a parsed MLIR module containing multiple anchor ops (the ones any
of the v2 recognizers in `qnn_emit_recognizers/` know how to lower) and
splits it into a list of `Island` records. Each island is a maximal
connected subgraph claimable by one recognizer, with explicit boundary
SSA inputs and outputs.

Compared to the v2 dispatcher (which matches a single recognizer per
whole-func module), the partitioner is the bridge from real per-model
IR (one big func with many dispatches) to per-island artifacts that
each compile to a separate `.qnn-ctx`. The partitioner is itself
**bindings-only** (no regex) — same strict rule as the v2 emitter.

Phase 3a focuses on partitioner *correctness* on hand-curated fixtures.
Phase 3b wires the partitioner into `./merlin compile` via the
`--qnn-partition` flag and exercises it on real yolov8 IR.
"""

from __future__ import annotations

import dataclasses
from typing import Any

# Anchor ops claimed by each recognizer module. Order is significant:
# it controls the partitioner's preference when multiple anchors are
# adjacent (most-specific first). Single-op anchors come last so they
# don't steal ops from the multi-op conv DAG.
_ANCHOR_OP_TO_RECOGNIZER: tuple[tuple[str, str, str], ...] = (
    # (anchor_op_name, recognizer_module, default_target)
    ("linalg.conv_2d_nchw_fchw_q", "nchw_int8_conv", "qnn-hta"),
    ("linalg.pooling_nchw_max", "nchw_int8_pool", "qnn-gpu"),
    ("tensor.concat", "nchw_int8_concat", "qnn-gpu"),
    ("linalg.transpose", "nchw_int8_transpose", "qnn-gpu"),
    ("tensor.collapse_shape", "nchw_int8_reshape", "qnn-gpu"),
    ("tensor.expand_shape", "nchw_int8_reshape", "qnn-gpu"),
)


@dataclasses.dataclass(frozen=True)
class BoundaryValue:
    """One SSA value crossing an island boundary.

    `ssa_name` is the bindings' canonical name (`%argN` for block args,
    op-result names for SSA defs). `shape`/`dtype` are the static
    tensor type information needed to declare the boundary in the
    eventual per-island manifest entry.
    """

    ssa_name: str
    shape: tuple[int, ...]
    dtype: str
    # For boundary outputs only: name of the op (and result index) that
    # produces this value inside the island. None for boundary inputs.
    producing_op: str | None = None


@dataclasses.dataclass(frozen=True)
class Island:
    """One partitioned subgraph + its routing decision.

    The partitioner emits one Island per anchor op found in the input
    module. `op_names` lists the ops claimed by this island in source
    order; `boundary_inputs` are SSA values defined outside the island
    but consumed inside (will become QNN graph inputs); `boundary_outputs`
    are SSA values defined inside but used outside (QNN graph outputs).

    `target` selects the runtime backend; `recognizer_name` records
    which recognizer module is expected to lower this island.
    """

    name: str
    recognizer_name: str
    target: str
    op_names: tuple[str, ...]
    boundary_inputs: tuple[BoundaryValue, ...]
    boundary_outputs: tuple[BoundaryValue, ...]


def _is_ranked_tensor(value: Any) -> bool:
    ty = value.type
    return hasattr(ty, "shape") and hasattr(ty, "element_type")


def _tensor_metadata(value: Any) -> tuple[tuple[int, ...], str]:
    return tuple(value.type.shape), str(value.type.element_type)


_FUNC_OP_NAMES = ("func.func", "util.func")
_RETURN_OP_NAMES = ("func.return", "util.return")


def _walk_inner_ops(func: Any):
    for region in func.operation.regions:
        for block in region.blocks:
            yield from block.operations


def _find_func(module: Any) -> Any | None:
    """Return the first func-like op in `module`. Both `func.func` (the
    standard MLIR shape) and `util.func` (IREE's wrapper after ONNX
    import / `iree-import-onnx` produces this) qualify."""
    for op in module.body.operations:
        if op.operation.name in _FUNC_OP_NAMES:
            return op
    return None


def _find_anchors(func: Any) -> list[tuple[Any, str, str]]:
    """Return [(op, recognizer_name, default_target), ...] for every
    anchor op in `func`, in source order."""
    out: list[tuple[Any, str, str]] = []
    for op in _walk_inner_ops(func):
        for anchor_name, recognizer, target in _ANCHOR_OP_TO_RECOGNIZER:
            if op.operation.name == anchor_name:
                out.append((op, recognizer, target))
                break
    return out


def _producer_op(value: Any) -> Any | None:
    """Op that defines `value` (None for block arguments)."""
    owner = value.owner
    return owner if hasattr(owner, "operation") else None


def _build_op_index(func: Any) -> tuple[list[Any], dict[str, int]]:
    """Walk inner ops once, assigning each a stable integer index.
    Returns (ops_in_source_order, ssa_name_to_producer_index).

    The MLIR bindings return new Python proxy objects on each access,
    so `id(op)` is not stable across iterations. We use the op's
    canonical SSA-name (its first-result `get_name()`) as a stable
    string key, and a positional index as the partition tag.
    """
    ops: list[Any] = []
    ssa_to_index: dict[str, int] = {}
    for op in _walk_inner_ops(func):
        idx = len(ops)
        ops.append(op)
        # Map every result's SSA name back to this op's index. Some ops
        # (e.g. func.return) have no results.
        for r in op.results:
            ssa_to_index[r.get_name()] = idx
    return ops, ssa_to_index


def _claim_set_via_def_use(
    anchor_index: int,
    ops: list[Any],
    ssa_to_index: dict[str, int],
    anchor_indices: set[int],
    claimed_so_far: set[int],
) -> set[int]:
    """Greedy SSA-def-use closure for an anchor: walk transitively via
    operand-defining ops (upstream) and result-consumer ops (downstream),
    stopping at:
      - block arguments (= func inputs; not in `ssa_to_index`)
      - `func.return` (= func output)
      - ops already claimed by a previous island
      - other anchor ops (boundary between islands)

    The partitioner is intentionally heuristic at Phase 3a — it doesn't
    consult the recognizer's exact pattern. For the fixtures Phase 3a
    targets (one anchor per claimable region, no shared intermediate
    consumers), this closure produces the right partition. Phase 3b
    refines with per-recognizer claim queries when needed.
    """
    claimed: set[int] = {anchor_index}
    other_anchors = anchor_indices - {anchor_index}

    # BFS upstream: every operand's defining-op index gets claimed,
    # transitively, unless it's another anchor or already claimed.
    frontier: list[int] = [anchor_index]
    while frontier:
        idx = frontier.pop()
        op = ops[idx]
        for operand in op.operands:
            ssa = operand.get_name()
            src_idx = ssa_to_index.get(ssa)
            if src_idx is None:
                continue  # block arg
            if src_idx in claimed or src_idx in claimed_so_far:
                continue
            if src_idx in other_anchors:
                continue
            claimed.add(src_idx)
            frontier.append(src_idx)

    # BFS downstream: every result's consumer ops, until we hit
    # func.return / another anchor / already claimed.
    frontier = [anchor_index]
    visited: set[int] = set()
    while frontier:
        idx = frontier.pop()
        if idx in visited:
            continue
        visited.add(idx)
        op = ops[idx]
        for r in op.results:
            for use in r.uses:
                user = use.owner
                if user is None or not hasattr(user, "operation"):
                    continue
                if user.operation.name in _RETURN_OP_NAMES:
                    continue
                # Find user's index by SSA name of its first result
                # (or by linear scan for ops with no results).
                user_idx: int | None = None
                if user.results:
                    user_idx = ssa_to_index.get(user.results[0].get_name())
                if user_idx is None:
                    # Linear fallback for op without results — should
                    # be rare (only func.return-style ops).
                    continue
                if user_idx in claimed or user_idx in claimed_so_far:
                    continue
                if user_idx in other_anchors:
                    continue
                claimed.add(user_idx)
                frontier.append(user_idx)
    return claimed


def _compute_boundary(
    claimed_indices: set[int],
    ops: list[Any],
    ssa_to_index: dict[str, int],
) -> tuple[list[BoundaryValue], list[BoundaryValue]]:
    """Given a set of claimed op indices, compute the boundary SSA values:

      inputs  : SSA values consumed inside the island that are defined
                outside (block args or non-claimed ops).
      outputs : SSA values defined inside the island that are consumed
                outside (by non-claimed ops or `func.return`).

    Each boundary value is reported once; duplicates from multiple uses
    inside the island collapse to a single entry.
    """
    inside_ops = [ops[i] for i in sorted(claimed_indices)]

    # Collect all SSA values DEFINED inside (each result of each inside op).
    inside_defs: set[str] = set()
    for op in inside_ops:
        for r in op.results:
            inside_defs.add(r.get_name())

    # Inputs: any operand of an inside op whose def is not in inside_defs
    # AND is a tensor type (we only track tensor boundaries).
    seen_inputs: set[str] = set()
    inputs: list[BoundaryValue] = []
    for op in inside_ops:
        for operand in op.operands:
            ssa = operand.get_name()
            if ssa in inside_defs:
                continue
            if not _is_ranked_tensor(operand):
                continue
            if ssa in seen_inputs:
                continue
            seen_inputs.add(ssa)
            shape, dtype = _tensor_metadata(operand)
            inputs.append(BoundaryValue(ssa_name=ssa, shape=shape, dtype=dtype))

    # Outputs: any inside-def whose uses include at least one op that
    # is NOT in claimed_indices (or `func.return`).
    seen_outputs: set[str] = set()
    outputs: list[BoundaryValue] = []
    for op in inside_ops:
        for r in op.results:
            if not _is_ranked_tensor(r):
                continue
            ssa = r.get_name()
            if ssa in seen_outputs:
                continue
            crosses_boundary = False
            for use in r.uses:
                user = use.owner
                if user is None or not hasattr(user, "operation"):
                    continue
                if user.operation.name in _RETURN_OP_NAMES:
                    crosses_boundary = True
                    break
                user_idx = ssa_to_index.get(user.results[0].get_name()) if user.results else None
                if user_idx is None or user_idx not in claimed_indices:
                    crosses_boundary = True
                    break
            if not crosses_boundary:
                continue
            seen_outputs.add(ssa)
            shape, dtype = _tensor_metadata(r)
            outputs.append(
                BoundaryValue(
                    ssa_name=ssa,
                    shape=shape,
                    dtype=dtype,
                    producing_op=op.operation.name,
                )
            )
    return inputs, outputs


def partition(module: Any, *, target_router=None) -> list[Island]:
    """Top-level partitioner. Walks the first `func.func` in `module`,
    finds all anchor ops in source order, computes a claim set for
    each, and emits one `Island` record per anchor.

    `target_router(anchor_op_name, op) -> str` is an optional callback
    that overrides the default backend selection (e.g. to route large
    convs to HTA and small ones to GPU). When None, every anchor uses
    its `_ANCHOR_OP_TO_RECOGNIZER` default target.

    The partitioner is **deterministic**: same input → same output,
    same island ordering. Compile-determinism gate (Phase 1's
    `assert_compile_deterministic`) depends on this.
    """
    func = _find_func(module)
    if func is None:
        return []

    ops, ssa_to_index = _build_op_index(func)
    # Find anchor indices in source order.
    anchor_descriptors: list[tuple[int, str, str]] = []  # (index, recog, target)
    for idx, op in enumerate(ops):
        for anchor_name, recognizer, target in _ANCHOR_OP_TO_RECOGNIZER:
            if op.operation.name == anchor_name:
                anchor_descriptors.append((idx, recognizer, target))
                break
    anchor_indices = {d[0] for d in anchor_descriptors}

    islands: list[Island] = []
    claimed_so_far: set[int] = set()
    for index, (anchor_idx, recognizer, default_target) in enumerate(anchor_descriptors):
        anchor_op = ops[anchor_idx]
        target = target_router(anchor_op.operation.name, anchor_op) if target_router is not None else default_target
        claimed = _claim_set_via_def_use(anchor_idx, ops, ssa_to_index, anchor_indices, claimed_so_far)
        op_names_in_island = tuple(ops[i].operation.name for i in sorted(claimed))
        boundary_inputs, boundary_outputs = _compute_boundary(claimed, ops, ssa_to_index)
        islands.append(
            Island(
                name=f"island_{index}_{recognizer}",
                recognizer_name=recognizer,
                target=target,
                op_names=op_names_in_island,
                boundary_inputs=tuple(boundary_inputs),
                boundary_outputs=tuple(boundary_outputs),
            )
        )
        claimed_so_far |= claimed
    return islands


def parse_and_partition(text: str, *, target_router=None) -> list[Island]:
    """Convenience wrapper: parse `text` via `iree.compiler.ir` then
    partition. Used by tests and by `tools/compile.py` when the
    `--qnn-partition` flag is set."""
    from iree.compiler import ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(text, ctx)
    return partition(module, target_router=target_router)


# ---------------------------------------------------------------------
# Phase 5 — per-island slice MLIR emission + claim-aware partition.
#
# `partition_with_claims` returns the same Islands plus the integer
# indices of their claimed ops in source order; `emit_island_slice_mlir`
# uses those indices to assemble a standalone MLIR module containing
# only the island's ops, with boundary IO as the func signature.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IslandWithClaims:
    """Island augmented with the integer indices of its claimed ops in
    the source func — needed for slice emission. Returned by
    `partition_with_claims` (kept separate from `Island` so the basic
    case stays lightweight)."""

    island: Island
    claimed_indices: tuple[int, ...]


def partition_with_claims(module: Any, *, target_router=None) -> list[IslandWithClaims]:
    """Variant of `partition()` that also returns each island's
    claimed op indices (drives `emit_island_slice_mlir`)."""
    func = _find_func(module)
    if func is None:
        return []

    ops, ssa_to_index = _build_op_index(func)
    anchor_descriptors: list[tuple[int, str, str]] = []
    for idx, op in enumerate(ops):
        for anchor_name, recognizer, target in _ANCHOR_OP_TO_RECOGNIZER:
            if op.operation.name == anchor_name:
                anchor_descriptors.append((idx, recognizer, target))
                break
    anchor_indices = {d[0] for d in anchor_descriptors}

    out: list[IslandWithClaims] = []
    claimed_so_far: set[int] = set()
    for index, (anchor_idx, recognizer, default_target) in enumerate(anchor_descriptors):
        anchor_op = ops[anchor_idx]
        target = target_router(anchor_op.operation.name, anchor_op) if target_router is not None else default_target
        claimed = _claim_set_via_def_use(anchor_idx, ops, ssa_to_index, anchor_indices, claimed_so_far)
        op_names_in_island = tuple(ops[i].operation.name for i in sorted(claimed))
        boundary_inputs, boundary_outputs = _compute_boundary(claimed, ops, ssa_to_index)
        island = Island(
            name=f"island_{index}_{recognizer}",
            recognizer_name=recognizer,
            target=target,
            op_names=op_names_in_island,
            boundary_inputs=tuple(boundary_inputs),
            boundary_outputs=tuple(boundary_outputs),
        )
        out.append(IslandWithClaims(island=island, claimed_indices=tuple(sorted(claimed))))
        claimed_so_far |= claimed
    return out


def emit_island_slice_mlir(
    module: Any,
    island_with_claims: IslandWithClaims,
) -> str:
    """Emit a standalone MLIR module containing only the island's ops
    as a `func.func`. Boundary inputs become func arguments, boundary
    outputs become the func return.

    Each op is emitted via the bindings' string representation
    (`str(op)`). The resulting slice is a structural artifact useful
    for: (a) per-island manifest entry shape; (b) downstream slice-MLIR
    inspection; (c) recognizer test fixtures derived from real models.

    NOTE: the slice is *not always re-parseable* as-is — IREE-emitted
    IR can carry op-attributes that reference SSA values upstream of
    the claim set. When that happens, the v2 emitter's recognizers
    don't re-parse the slice; instead they re-walk the *original*
    module restricted to `island_with_claims.claimed_indices`. The
    slice is the inspection artifact; the recognizer drives off the
    bindings module directly.
    """
    func = _find_func(module)
    if func is None:
        raise ValueError("module has no func.func / util.func to slice")

    isl = island_with_claims.island
    claimed = set(island_with_claims.claimed_indices)
    op_texts: list[str] = []
    for idx, op in enumerate(_walk_inner_ops(func)):
        if idx not in claimed:
            continue
        if op.operation.name in _RETURN_OP_NAMES:
            continue
        try:
            op_texts.append(str(op))
        except Exception:  # noqa: BLE001
            continue

    def shape_str(bv: BoundaryValue) -> str:
        return "x".join(str(d) for d in bv.shape)

    args_str = ", ".join(f"{bv.ssa_name}: tensor<{shape_str(bv)}x{bv.dtype}>" for bv in isl.boundary_inputs)
    if not isl.boundary_outputs:
        ret_ty = "()"
        ret_ssa = ""
    elif len(isl.boundary_outputs) == 1:
        bv = isl.boundary_outputs[0]
        ret_ty = f"tensor<{shape_str(bv)}x{bv.dtype}>"
        ret_ssa = bv.ssa_name
    else:
        types = ", ".join(f"tensor<{shape_str(bv)}x{bv.dtype}>" for bv in isl.boundary_outputs)
        ret_ty = f"({types})"
        ret_ssa = ", ".join(bv.ssa_name for bv in isl.boundary_outputs)

    body = "\n    ".join(op_texts)
    return (
        "module {\n"
        f"  func.func @{isl.name}({args_str}) -> {ret_ty} {{\n"
        f"    {body}\n"
        f"    return {ret_ssa} : {ret_ty}\n"
        "  }\n"
        "}\n"
    )


def parse_and_partition_with_claims(text: str, *, target_router=None) -> tuple[Any, list[IslandWithClaims]]:
    """Like `parse_and_partition` but returns the parsed module
    alongside the islands+claims (so callers can call
    `emit_island_slice_mlir` against the same module)."""
    from iree.compiler import ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(text, ctx)
    return module, partition_with_claims(module, target_router=target_router)
