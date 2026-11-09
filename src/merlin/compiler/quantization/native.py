"""Explicit diagnostic graph preparation through a selected OOT implementation.

The selected source/policy is intent, not an acceptance threshold or evidence of
numerical quality, correct device execution or hardware qualification.
"""

from __future__ import annotations

import json
from pathlib import Path


def prepare_graph_request(source, *, entry, policy, binding, target_identity, mode, original_source_sha256=None):
    """Preserve the original selected graph hook and its diagnostic boundary."""
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.semantic_compiler.model import KernelRequest

    if mode != "diagnostic" or binding is None:
        raise LinalgBridgeError("graph precision policy requires diagnostic mode and target support")
    prepare = getattr(binding, "prepare_precision_graph", None)
    if not callable(prepare):
        raise LinalgBridgeError("selected target support has no graph precision lowering")
    if original_source_sha256 is None:
        plan = prepare(source, entry=entry, policy=policy)
    else:
        # This is an explicit source-binding contract. A provider without the
        # staged-source keyword refuses; never retry after dropping the premise.
        plan = prepare(source, entry=entry, policy=policy, original_source_sha256=original_source_sha256)
    request = getattr(plan, "request", None)
    operations = getattr(plan, "source_operations", None)
    if (
        not isinstance(request, KernelRequest)
        or request.target_identity != target_identity
        or request.lowering_policy != "diagnostic"
        or not isinstance(operations, tuple)
        or not all(isinstance(operation, str) for operation in operations)
        or not callable(getattr(plan, "record", None))
    ):
        raise LinalgBridgeError("target graph precision lowering returned an invalid request")
    return request, operations, plan


def precision_plan_bytes(plan) -> bytes:
    """Canonical original plan serialization for publication and replay."""
    return (json.dumps(plan.record(), sort_keys=True, indent=2) + "\n").encode()


def prepare_tile_request(
    staged, region_id, *, binding, target_identity, mode, tile_index=None, whole_contraction=False
):
    """Admit one explicitly selected source/policy-bound diagnostic tile.

    Target arithmetic and geometry are offered by the OOT hook. Its returned
    correspondence must still describe the caller's actual staged program.
    """
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.semantic_compiler.precision_contract import DiagnosticMatmulTile

    if mode != "diagnostic" or binding is None:
        raise LinalgBridgeError(
            "source-bound precision tiles require staged compiler inputs, target support, and diagnostic mode"
        )
    if staged.precision_policy_document is None:
        raise LinalgBridgeError("source-bound precision tile has no selected policy document")
    prepare = getattr(binding, "prepare_precision_tile", None)
    if not callable(prepare):
        raise LinalgBridgeError("selected target support has no source-bound precision tile")
    # The current provider hook does not implement this historical optional
    # scope. Do not retry without the keyword or silently compile one tile.
    if whole_contraction:
        raise LinalgBridgeError("selected target support has no whole-contraction precision lowering")
    plan = prepare(staged, region_id) if tile_index is None else prepare(staged, region_id, tile_index=tile_index)
    if not isinstance(plan, DiagnosticMatmulTile) or plan.request.target_identity != target_identity:
        raise LinalgBridgeError("target precision tile returned an invalid request or target identity")
    if plan.request.lowering_policy != "diagnostic":
        raise LinalgBridgeError("source-bound precision tile must stay diagnostic")
    if plan.whole_contraction:
        raise LinalgBridgeError("target precision lowering changed the requested contraction scope")
    if (
        plan.region_id != region_id
        or plan.original_source_sha256 != staged.original_source_sha256
        or plan.staged_program_sha256 != staged.program_sha256
        or plan.precision_policy_sha256 != staged.precision_policy_sha256
    ):
        raise LinalgBridgeError("target precision tile differs from selected staged source or policy")
    source_op = "linalg.generic" if plan.batch_index is not None else "linalg.matmul"
    return plan.request, "source_bound_diagnostic_tile", (source_op,), plan


def prepared_static_inputs(binding, plan, staged_root: Path, execution_plan: dict):
    """Check the target's source-bound constant payloads before publication."""
    from merlin.semantic_compiler.precision_constants import PreparedStaticInput

    prepare = getattr(binding, "prepare_precision_constants", None)
    if not callable(prepare):
        raise ValueError("target cannot prepare source-bound static inputs")
    prepared = prepare(plan, staged_root, execution_plan)
    if not isinstance(prepared, tuple) or any(not isinstance(item, PreparedStaticInput) for item in prepared):
        raise ValueError("target returned malformed source-bound static inputs")
    roles = [item.role for item in prepared]
    allowed_roles = {conversion.source_role for conversion in plan.inputs}
    if len(set(roles)) != len(roles) or any(role not in allowed_roles for role in roles):
        raise ValueError("target returned repeated or unknown source-bound input roles")
    return prepared
