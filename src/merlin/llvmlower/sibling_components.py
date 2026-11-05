"""Independent shared-input components; original unmatched source stays intact.

Only existing exact source outlining and borrowed writer checks authorize each
rewrite. This supplies neither placement nor target effects or execution proof.
"""

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class SiblingComponent:
    core: object
    writer: object
    receipt: dict


@dataclass(frozen=True)
class SiblingComponents:
    module: object
    components: tuple
    receipt: dict


def prepare_sibling_components(module, *, effects, max_members=8, max_components=256, output_path=None, values=None):
    """Select disjoint same-SSA-LHS clusters using current semantic proofs.

    Unsupported/nonzero roots and clusters crossing unknown effects remain
    untouched for ordinary placement. Every member keeps its original host
    observers, including distinct per-column quantizers and external outputs.
    """
    from xdsl.dialects import func

    from merlin.compile.scheduling.dense_source import read_zero_seeded_integer_matmul
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.fresh_tensor_writer import rewrite_fresh_tensor_writers
    from merlin.llvmlower.source_region import _apply_sibling_contractions, _plan_sibling_contractions
    from merlin.llvmlower.source_region_host import _prepare_sibling_writer
    from merlin.transforms.tensor_views import canonicalize_tensor_views
    from merlin.xdsl_dialects._common import text

    if (
        type(effects) is not BorrowedBufferEffects
        or type(max_members) is not int
        or not 2 <= max_members <= 8
        or type(max_components) is not int
        or max_components < 1
    ):
        raise ValueError("source components need typed effects and finite explicit limits")
    effects.__post_init__()
    module.verify()
    before = text(module, generic=True)
    current = module.clone()
    canonicalize_tensor_views(current)
    # All components are one transaction on this private clone. Ordinals and
    # source identity refer to the same normalized source, rather than dozens
    # of whole-graph snapshots of successive intermediate rewrites.
    normalized_sha256 = sha256(text(current, generic=True).encode()).hexdigest()
    source_positions = {op: i for owner in current.body.block.ops if isinstance(owner, func.FuncOp)
                        for block in owner.body.blocks for i, op in enumerate(block.ops)}
    # Roots keep their SSA identities on this one normalized private clone.
    # Retain the shared shape analysis while disjoint components are replaced;
    # no new contraction becomes eligible from an earlier component rewrite.
    readout_extents = None
    if output_path is not None:
        from merlin.compile.analysis.contraction_shapes import contraction_extents
        readout_extents = contraction_extents(current)
    components, attempts = [], []
    while len(components) < max_components:
        functions = [
            op for op in current.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"
        ]
        if len(functions) != 1 or len(functions[0].body.blocks) != 1:
            raise ValueError("source components require one original counted forward block")
        runs = {}
        for op in functions[0].body.block.ops:
            if op.name not in {"linalg.generic", "linalg.matmul"} or not op.results:
                continue
            try:
                shape = read_zero_seeded_integer_matmul(op)
            except ValueError:
                continue
            runs.setdefault((op.operands[0], shape[:2]), []).append(op)
        success = False
        for roots in runs.values():
            for start in range(0, len(roots), max_members):
                selected = tuple(roots[start : start + max_members])
                if len(selected) < 2:
                    continue
                symbol = "source_shared_component_" + str(len(components))
                try:
                    plan = _plan_sibling_contractions(current, selected, symbol=symbol)
                    if output_path is not None:
                        from .sibling_readouts import extend_sibling_readouts
                        from merlin.xdsl_dialects.lowering.output_fusion import Values
                        plan = extend_sibling_readouts(plan, path=output_path, values=values or Values(),
                                                       extents=readout_extents)
                except ValueError as error:
                    attempts.append({"selected": False, "reason": str(error)})
                    continue
                # No expected refusal occurs after committing the checked
                # plan. Any writer/graph failure aborts this entire clone.
                _apply_sibling_contractions(plan)
                writer = _prepare_sibling_writer(current, symbol=symbol, effects=effects)
                components.append(SiblingComponent(plan.core, writer, {
                    "schema": "merlin.original_sibling_source_host.v2",
                    "view_cse": [],
                    "readouts": plan.receipt.get("readouts", []),
                    "outline": {
                        "schema": "merlin.source_sibling_contraction_batch.v1",
                        "source_ir_sha256": normalized_sha256,
                        "source_root_ordinals": [source_positions[op] for op in selected],
                        "source_geometries": plan.receipt["source_geometries"],
                        "source_hoisted_preparation_ordinals": sorted(source_positions[op] for op in plan.hoisted),
                        "core_ir_sha256": sha256(text(plan.core, generic=True).encode()).hexdigest(),
                        "source_operation_coordinate_system": "normalized_batch_source",
                        "same_lhs_ssa": True,
                        "original_epilogues_and_observers_preserved": True,
                        "original_full_source_unchanged": True,
                        "target_implementation_or_effect_admission": False,
                        "target_execution": False,
                    },
                    "source_unchanged": True,
                    "original_epilogues_and_all_live_outputs_preserved": True,
                    "physical_effects_and_target_execution_established": False,
                }))
                success = True
                break
            if success:
                break
        if not success:
            break
    rewrite_fresh_tensor_writers(current, tuple(c.writer for c in components))
    current.verify()
    final_sha256 = sha256(text(current, generic=True).encode()).hexdigest()
    for component in components:
        component.receipt["outline"]["host_ir_sha256"] = final_sha256
        component.receipt["outline"]["host_uses_explicit_borrowed_writer"] = True
    if text(module, generic=True) != before:
        raise ValueError("source component selection changed original source")
    return SiblingComponents(
        current,
        tuple(components),
        {
            "schema": "merlin.original_sibling_components.v1",
            "normalized_source_ir_sha256": normalized_sha256,
            "host_ir_sha256": final_sha256,
            "components": [c.receipt for c in components],
            "refused_proposals": attempts,
            "unmatched_source_retained": True,
            "all_original_epilogues_and_outputs_preserved": True,
            "effect_or_placement_grant": False,
            "component_limit_reached": len(components) == max_components,
            "max_members": max_members,
            "max_components": max_components,
        },
    )
