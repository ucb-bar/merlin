"""Read-only evidence before claiming a captured model compiled for an OOT target.

Routing at a requested deployment format is a capability question. Grouping the
captured IR is a different question: it sees the operand formats and quantization
operations that actually exist. Keep both counts visible until a compiler pass
bridges them and emits an executable target program.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path


def preflight_model(capture_bundle: str | Path, *, target: str, deployment_dtype: str) -> dict:
    """Compare declared-format routes with groups present in a real capture.

    This inspects inputs only. It does not run a model, generate a capsule, build a
    backend, or claim a target binary. An explicit capture path avoids silently
    resolving another model revision by name.
    """
    from merlin.common import mlir_query as mq
    from merlin.targetgen import capsule_source, group_capsule_entries, routing
    from merlin.xdsl_dialects.lowering import compute_groups

    bundle = Path(capture_bundle)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError(f"capture bundle must be an existing non-symlink directory: {bundle}")
    model = bundle / "model.mlir"
    if model.is_symlink() or not model.is_file():
        raise ValueError(f"capture bundle has no regular, non-symlink model.mlir: {bundle}")
    if not deployment_dtype:
        raise ValueError("deployment_dtype must be an exact target format name")

    required = (
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "input_order.json",
        "golden.npy",
    )
    missing = [name for name in required if not (bundle / name).is_file() or (bundle / name).is_symlink()]
    text = model.read_text(encoding="utf-8")
    blockers: list[dict[str, object]] = []
    try:
        demands = capsule_source.model_op_demands_checked(text, deployment_dtype)
        inventory_verified = True
    except capsule_source.ModelDemandIncomplete as exc:
        # Keep the legacy tag inventory for a useful comparison, but make its
        # incompleteness an explicit blocker, never a successful route ledger.
        demands = capsule_source.model_op_demands(text, deployment_dtype)
        inventory_verified = False
        blockers.append({"kind": "route_inventory_incomplete", "detail": str(exc)})
    route = routing.route_plan(demands, target)
    module = mq.parse(model)
    groups = compute_groups.form_groups(module, target)
    contractions = [group for group in groups if group.root is not None]
    on_device = [group for group in contractions if group.placement != compute_groups.HOST]
    tagged_contractions = [demand for demand in demands if demand.family == "contraction"]
    declared_mesh = [result for result in route["mesh"] if result.demand.family == "contraction"]
    stated = group_capsule_entries.entries(target, module, with_raw=False)

    captured_dtypes = Counter(str(group.in_dtype or "unknown") for group in contractions)
    refusals = Counter(
        str(group.refusal or "unknown") for group in contractions if group.placement == compute_groups.HOST
    )
    if missing:
        blockers.append({"kind": "incomplete_capture", "files": missing})
    # A raw count mismatch is not proof of missing operations: some structurally
    # contracting attention regions are classified under a different family.
    # The strict identity-and-shape check above is the authority for completeness.
    if len(declared_mesh) > len(on_device):
        blockers.append(
            {
                "kind": "declared_route_not_materialized",
                "declared_routes": len(declared_mesh),
                "captured_accelerator_groups": len(on_device),
                "explanation": (
                    "Routing at the requested deployment format is not an executable placement "
                    "of the captured IR. A proven dtype/quantization lowering may be required."
                ),
            }
        )
    incompatible = sum(count for dtype, count in captured_dtypes.items() if dtype != deployment_dtype)
    if incompatible:
        blockers.append(
            {
                "kind": "capture_dtype_needs_bridge",
                "contractions": incompatible,
                "requested": deployment_dtype,
                "explanation": "No target-native quantization bridge is established by this preflight.",
            }
        )
    if stated["unstated"]:
        blockers.append({"kind": "accelerator_group_has_no_capsule_form", "reasons": stated["unstated"]})
    if not on_device:
        blockers.append({"kind": "no_accelerator_groups_in_capture"})

    return {
        "schema": "whole_model_preflight_v1",
        "status": "blocked" if blockers else "analyzed_not_compiled",
        "reason": (
            f"{len(blockers)} readiness blocker(s); see blockers and contractions"
            if blockers
            else "capture analysis passed, but no target model binary was built or run"
        ),
        "target": target,
        "capture_bundle": str(bundle.resolve()),
        "deployment_dtype": deployment_dtype,
        "target_binary_emitted": False,
        "scope": "static capture and capability analysis; no model execution or target build",
        "route_inventory_verified": inventory_verified,
        "contractions": {
            "captured": len(contractions),
            "tagged_demands": len(tagged_contractions),
            "declared_format_routes": len(declared_mesh),
            "captured_accelerator_groups": len(on_device),
            "capsule_form_groups": stated["stated"],
            "captured_operand_dtypes": dict(sorted(captured_dtypes.items())),
            "host_refusals": dict(sorted(refusals.items())),
        },
        "blockers": blockers,
    }
