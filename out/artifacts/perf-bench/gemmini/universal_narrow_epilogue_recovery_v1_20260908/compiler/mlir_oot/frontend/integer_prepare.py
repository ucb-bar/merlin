"""Target-neutral preparation of a captured quantized model for target placement.

Static PT2E captures deliberately retain quantize/dequantize boundaries in the
frontend IR.  Those boundaries must be consumed by Merlin's registered integer
passes *before* the residual ``quant_ext`` operations are expanded to floating
point linalg.  Target readers then receive ordinary upstream linalg with real
integer contractions; they do not need to understand TorchAO or model2MLIR.
"""
from __future__ import annotations

import io

from xdsl.printer import Printer

from .parse import parse_module
from .integer_passes import lower_contraction_int8, lower_conv_int8
from xdsl.transforms.dead_code_elimination import dce


def _module_text(module) -> str:
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue() + "\n"


def _splat_constant(value):
    owner = getattr(value, "owner", None)
    if getattr(owner, "name", None) != "tensor.splat":
        return None
    constant = getattr(owner.operands[0], "owner", None)
    if getattr(constant, "name", None) != "arith.constant":
        return None
    # IntegerAttr(0) is false-y in xdsl.  Select by presence so a symmetric
    # quantizer's perfectly valid zero point is not mistaken for no constant.
    attr = constant.properties.get("value")
    if attr is None:
        attr = constant.attributes.get("value")
    return getattr(getattr(attr, "value", None), "data", None)


def static_input_prologue(source_text: str) -> dict | None:
    """Describe a static PT2E input quantizer rooted at an entry argument.

    The description is target-neutral and sufficient for a deployment harness
    to perform the permitted input conversion before its timed model entry.
    Anything ambiguous returns ``None`` rather than moving graph work outside
    the measurement boundary.
    """
    from xdsl.dialects.builtin import StringAttr, TensorType
    from xdsl.ir import Block

    module = parse_module(source_text)
    funcs = [op for op in module.walk() if op.name == "func.func"]
    if len(funcs) != 1 or not funcs[0].regions or not funcs[0].regions[0].blocks:
        return None
    block = funcs[0].regions[0].blocks[0]
    candidates = []
    for op in module.walk():
        name = getattr(getattr(op, "op_name", None), "data", "")
        if name != "quant_ext.quantize_per_tensor" or len(op.operands) != 3:
            continue
        source = op.operands[0]
        if not isinstance(source.owner, Block) or source.owner is not block:
            continue
        scale, zero_point = _splat_constant(op.operands[1]), _splat_constant(op.operands[2])
        out_type = op.results[0].type
        if scale is None or zero_point is None or not isinstance(out_type, TensorType):
            continue
        candidates.append({
            "kind": "static_pt2e_per_tensor_input_quantize",
            "region_id": getattr(op.attributes.get("prov.region_id"), "data", ""),
            "input_arg_index": block.args.index(source),
            "shape": [int(d) for d in out_type.get_shape()],
            "output_dtype": str(out_type.element_type),
            "scale": float(scale),
            "zero_point": int(zero_point),
            "quant_min": int(op.properties["quant_min"].value.data),
            "quant_max": int(op.properties["quant_max"].value.data),
            "rounding": "float32_multiply_by_float32_reciprocal_then_round_nearest_even",
        })
    return candidates[0] if len(candidates) == 1 else None


def prepare_int8_text(source_text: str) -> tuple[str, dict]:
    """Return upstream MLIR plus an auditable, per-pass preparation report.

    This is model- and target-independent: the same registered integer passes
    are used by Merlin's host interpreter and compiled whole-model backends.
    Any unconsumed quantization boundary is subsequently lowered by the common
    upstream fallback, so mixed integer/float graphs remain representable.
    """
    module = parse_module(source_text)

    def quant_census() -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in module.walk():
            name = getattr(getattr(op, "op_name", None), "data", "")
            if op.name == "builtin.unregistered" and name.startswith("quant_ext."):
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    before = quant_census()
    contraction_report: dict[str, int] = {}
    conv_report: dict[str, int] = {}
    # Projection contractions and direct/windowed convolutions are disjoint structural forms.  Run
    # both over the same parsed module so they preserve captured i8 tensors/scales and only the
    # admitted contraction itself becomes i8xi8->i32.
    contractions = lower_contraction_int8(
        module, named_contraction=False, report_out=contraction_report)
    convolutions = lower_conv_int8(module, report_out=conv_report)
    # Unregistered quant_ext ops conservatively have no Pure trait, so xDSL's DCE cannot remove
    # even a dequantize whose only contraction user was replaced. Erase only structurally dead
    # quant boundaries, then let ordinary trait-based DCE collect their now-unused pure furniture.
    dead_quant = []
    while True:
        dead = []
        for op in module.walk():
            name = getattr(getattr(op, "op_name", None), "data", "")
            pure_tensor_op = (op.name.startswith(("arith.", "tensor.", "linalg."))
                              or (op.name == "builtin.unregistered"
                                  and name.startswith("quant_ext.")))
            if (pure_tensor_op and op.results
                    and all(not any(use.operation.parent_block() is not None
                                    for use in result.uses)
                            for result in op.results)):
                dead.append(op)
                if op.name == "builtin.unregistered" and name.startswith("quant_ext."):
                    dead_quant.append(op)
        if not dead:
            break
        for op in reversed(dead):
            op.detach()
            op.erase(safe_erase=False)
        dce(module)
    module.verify()
    after = quant_census()

    # A remaining dequant is legal only when it serves a non-contraction host operation.  If one still
    # feeds a linalg reduction/matmul, placement would silently route an eligible operation as f32.
    admitted_qdq_failures = []
    for op in module.walk():
        name = getattr(getattr(op, "op_name", None), "data", "")
        if op.name != "builtin.unregistered" or not name.startswith("quant_ext.dequantize"):
            continue
        for result in op.results:
            for use in result.uses:
                user = use.operation
                if user.parent_block() is None or not user.name.startswith("linalg."):
                    continue
                body = ([child.name for child in user.regions[0].blocks[0].ops]
                        if user.regions and user.regions[0].blocks else [])
                if (user.name in ("linalg.matmul", "linalg.batch_matmul")
                        or any(item in body for item in ("arith.addf", "arith.addi"))):
                    admitted_qdq_failures.append({
                        "quant_op": name,
                        "consumer": user.name,
                        "region_id": getattr(user.attributes.get("prov.region_id"), "data", ""),
                    })

    pass_counts = {"contraction_int8": contractions, "conv_int8": convolutions}
    pass_report = {"contraction_int8": contraction_report, "conv_int8": conv_report}
    return _module_text(module), {
        "integer_pass_counts": pass_counts,
        "integer_pass_report": pass_report,
        "quant_ext_before": before,
        "quant_ext_after": after,
        "quant_ext_preserved_for_generated_host_lowering": sum(after.values()),
        "dead_quant_ext_removed": len(dead_quant),
        "admitted_qdq_failures": admitted_qdq_failures,
        "integer_preparation_complete": not admitted_qdq_failures,
        "self_contained_preprocess": "captured_static_qdq_to_i8xi8_i32_v1",
    }
