"""Target-neutral preparation of a captured quantized model for target placement.

Static PT2E captures deliberately retain quantize/dequantize boundaries in the
frontend IR.  Those boundaries must be consumed by Merlin's registered integer
passes *before* the residual ``quant_ext`` operations are expanded to floating
point linalg.  Target readers then receive ordinary upstream linalg with real
integer contractions; they do not need to understand TorchAO or model2MLIR.
"""
from __future__ import annotations

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import passes_xdsl
from merlin.llvmlower.quant_passes import apply_quant
from merlin.xdsl_dialects._common import text as module_to_text


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

    module = parse_mlir_text(source_text)
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
    module = parse_mlir_text(source_text)
    quant_report: dict[str, dict] = {}
    quant_counts = apply_quant(module, report_out=quant_report)
    upstream_text, common_stats = passes_xdsl.preprocess_text(module_to_text(module))
    return upstream_text, {
        "integer_pass_counts": quant_counts,
        "integer_pass_report": quant_report,
        **common_stats,
    }
