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
from merlin.llvmlower import passes_quant_int as canonical_quant_impl
from merlin.llvmlower.quant_passes import apply_quant
from merlin.xdsl_dialects._common import text as module_to_text

from . import gemmini_friendly_quant


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
    # Keep the model-independent pass local to this compiler snapshot. The repository may contain
    # a newer canonical quantizer; only the contraction implementation is replaced for this call,
    # and it is restored immediately so other compilation flows cannot inherit artifact policy.
    canonical = canonical_quant_impl.lower_contraction_int8
    canonical_quant_impl.lower_contraction_int8 = gemmini_friendly_quant.lower_contraction_int8
    try:
        quant_counts = apply_quant(module, report_out=quant_report)
    finally:
        canonical_quant_impl.lower_contraction_int8 = canonical
    upstream_text, common_stats = passes_xdsl.preprocess_text(module_to_text(module))
    return upstream_text, {
        "integer_pass_counts": quant_counts,
        "integer_pass_report": quant_report,
        **common_stats,
    }


def _quant_op_name(op) -> str:
    """Return the registered or unregistered operation name without provenance."""
    if getattr(op, "name", None) != "builtin.unregistered":
        return str(getattr(op, "name", ""))
    return str(getattr(getattr(op, "op_name", None), "data", ""))


def _integer_property(op, name: str) -> int | None:
    attr = op.properties.get(name)
    if attr is None:
        attr = op.attributes.get(name)
    raw = getattr(getattr(attr, "value", None), "data", None)
    return int(raw) if isinstance(raw, int) else None


def _transpose_source(value):
    """Return ``(value-before-transpose, output-axis-map)`` for a rank-2 swap."""
    owner = getattr(value, "owner", None)
    if getattr(owner, "name", None) != "linalg.transpose":
        return value, (0, 1)
    try:
        permutation = tuple(int(v) for v in owner.permutation.get_values())
    except (AttributeError, TypeError, ValueError):
        return None
    if permutation != (1, 0):
        return None
    # output[j, i] reads input[i, j]: an input scale axis is moved to the
    # output position whose permutation entry names it.
    return owner.inputs[0], tuple(permutation.index(i) for i in range(2))


def _symmetric_per_channel_i8(value) -> tuple[dict | None, str]:
    """Recognise a calibrated i8 per-channel tensor consumed as f32.

    The check deliberately proves all facts needed to factor the scale out of
    the integer reduction.  Provenance, symbol names, model names and concrete
    layer sizes are not consulted.
    """
    from xdsl.dialects.builtin import TensorType, f32, i8

    unwrapped = _transpose_source(value)
    if unwrapped is None:
        return None, "unsupported_weight_transpose"
    candidate, axis_map = unwrapped
    dequant = getattr(candidate, "owner", None)
    if _quant_op_name(dequant) != "quant_ext.dequantize_per_channel":
        return None, "not_per_channel_i8_dequant"
    if len(dequant.operands) < 3:
        return None, "malformed_per_channel_dequant"
    qvalue, scale, zero_point = dequant.operands[:3]
    qtype, stype, ztype, outtype = qvalue.type, scale.type, zero_point.type, candidate.type
    if not (isinstance(qtype, TensorType) and qtype.element_type == i8
            and isinstance(stype, TensorType) and stype.element_type == f32
            and isinstance(outtype, TensorType) and outtype.element_type == f32):
        return None, "unsupported_weight_dequant_types"
    qshape, sshape = list(qtype.get_shape()), list(stype.get_shape())
    if list(outtype.get_shape()) != qshape:
        return None, "weight_dequant_shape_mismatch"
    if not isinstance(ztype, TensorType) or list(ztype.get_shape()) != sshape:
        return None, "weight_zero_point_shape_mismatch"
    axis = _integer_property(dequant, "axis")
    if axis is None or not qshape:
        return None, "missing_weight_scale_axis"
    if axis < 0:
        axis += len(qshape)
    if not 0 <= axis < len(qshape):
        return None, "invalid_weight_scale_axis"
    if len(sshape) != 1 or int(sshape[0]) != int(qshape[axis]):
        return None, "weight_scale_shape_mismatch"
    if _splat_constant(zero_point) != 0:
        return None, "weight_zero_point_not_proven_zero"
    effective_axis = axis_map[axis] if len(qshape) == 2 else axis
    return {"axis": effective_axis, "scale_extent": int(sshape[0])}, ""


def _contraction_input_maps(op) -> tuple[list[list[int]], list[bool]] | None:
    """Return operand iterator dimensions and reduction flags for a contraction."""
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir.affine import AffineDimExpr

    if op.name == "linalg.matmul":
        return [[0, 2], [2, 1]], [False, False, True]
    if op.name != "linalg.generic" or len(op.inputs) != 2:
        return None
    try:
        maps = list(op.indexing_maps)
        iters = [getattr(a, "data", a) for a in op.iterator_types]
    except AttributeError:
        return None
    if len(maps) < 3 or not any(i == L.IteratorType.REDUCTION for i in iters):
        return None
    input_maps: list[list[int]] = []
    for mapping in maps[:2]:
        if any(not isinstance(expr, AffineDimExpr) for expr in mapping.data.results):
            return None
        input_maps.append([expr.position for expr in mapping.data.results])
    return input_maps, [i == L.IteratorType.REDUCTION for i in iters]


def _is_symmetric_per_tensor_i8(value) -> tuple[bool, str]:
    """Recognise a static/dynamic activation QDQ that can be reused exactly."""
    from xdsl.dialects.builtin import TensorType, f32, i8

    dequant = getattr(value, "owner", None)
    if _quant_op_name(dequant) != "quant_ext.dequantize_per_tensor":
        return False, "plain_f32_activation"
    if len(dequant.operands) < 3:
        return False, "malformed_activation_dequant"
    qtype, stype, ztype, outtype = (
        dequant.operands[0].type, dequant.operands[1].type,
        dequant.operands[2].type, value.type)
    if not (isinstance(qtype, TensorType) and qtype.element_type == i8
            and isinstance(stype, TensorType) and stype.element_type == f32
            and not list(stype.get_shape())
            and isinstance(ztype, TensorType) and not list(ztype.get_shape())
            and isinstance(outtype, TensorType) and outtype.element_type == f32):
        return False, "unsupported_activation_dequant_types"
    if list(qtype.get_shape()) != list(outtype.get_shape()):
        return False, "activation_dequant_shape_mismatch"
    if _splat_constant(dequant.operands[2]) != 0:
        return False, "activation_zero_point_not_proven_zero"
    return True, ""


def _dynamic_weight_only_selector(report: dict):
    """Build the structural reach predicate for the scoped contraction pass."""
    from xdsl.dialects.builtin import TensorType, f32

    refused = report.setdefault("refused", {})

    def refuse(reason: str) -> bool:
        refused[reason] = refused.get(reason, 0) + 1
        return False

    def select(op) -> bool:
        report["candidate_f32_contractions"] = report.get("candidate_f32_contractions", 0) + 1
        view = _contraction_input_maps(op)
        if view is None:
            return refuse("noncanonical_contraction_maps")
        input_maps, reduction_flags = view
        weight_matches: list[tuple[int, dict]] = []
        weight_reasons: list[str] = []
        for operand_index, operand in enumerate(op.inputs):
            info, reason = _symmetric_per_channel_i8(operand)
            if info is not None:
                weight_matches.append((operand_index, info))
            else:
                weight_reasons.append(reason)
        if len(weight_matches) != 1:
            if len(weight_matches) > 1:
                return refuse("requires_exactly_one_per_channel_weight")
            # Surface the safety-relevant reason rather than the generic absence.
            safety = next((r for r in weight_reasons if r not in (
                "not_per_channel_i8_dequant",)), "no_per_channel_i8_weight")
            return refuse(safety)

        weight_index, weight = weight_matches[0]
        axis = int(weight["axis"])
        if axis >= len(input_maps[weight_index]):
            return refuse("weight_scale_axis_not_in_contraction_map")
        iterator_dim = input_maps[weight_index][axis]
        if iterator_dim >= len(reduction_flags) or reduction_flags[iterator_dim]:
            # Per-K scales cannot be multiplied into the accumulator after the
            # sum. Refusing here is required for arithmetic correctness.
            return refuse("weight_scale_varies_along_reduction")

        activation = op.inputs[1 - weight_index]
        if not (isinstance(activation.type, TensorType)
                and activation.type.element_type == f32):
            return refuse("activation_is_not_f32")
        static_activation, activation_reason = _is_symmetric_per_tensor_i8(activation)
        if activation_reason not in ("", "plain_f32_activation"):
            return refuse(activation_reason)
        key = "eligible_static_activation_qdq" if static_activation else "eligible_dynamic_activation"
        report[key] = report.get(key, 0) + 1
        report["eligible"] = report.get("eligible", 0) + 1
        return True

    return select


def prepare_dynamic_weight_only_text(source_text: str) -> tuple[str, dict]:
    """Bridge canonical weight-only/QDQ contractions into explicit integer IR.

    This is an opt-in numeric contract, not a claim that dynamically quantizing
    a float activation preserves the original float matmul bit-for-bit. Only
    structurally proven symmetric per-channel i8 weights whose scale varies
    along an output dimension are admitted. Everything else remains on the
    ordinary host path and is counted in ``selection.refused``.
    """
    # Use the backend's own grammar route here. It accepts the same custom
    # multi-result linalg spelling that the normal whole-program path accepts;
    # the analysis-only frontend parser is intentionally narrower. Once the
    # selected rewrite has printed generic/upstream IR, common preprocessing
    # can consume it through its ordinary route.
    from mlir_oot.frontend.parse import parse_module

    module = parse_module(source_text)
    selection: dict = {"candidate_f32_contractions": 0, "eligible": 0, "refused": {}}
    quant_report: dict[str, dict] = {}
    quant_counts = apply_quant(
        module,
        passes=["contraction_int8"],
        select=_dynamic_weight_only_selector(selection),
        report_out=quant_report,
    )
    if quant_counts["contraction_int8"] == 0:
        # A scoped bridge that selects nothing must be a true no-op. Besides
        # avoiding needless whole-module churn, this retains custom linalg
        # spellings accepted by the backend but not by the narrower common
        # preprocessing parser (notably multi-result reductions).
        return source_text, {
            "integer_pass_counts": quant_counts,
            "integer_pass_report": quant_report,
            "selection": selection,
            "common_preprocess": "skipped_no_selected_rewrite",
        }
    upstream_text, common_stats = passes_xdsl.preprocess_text(module_to_text(module))
    return upstream_text, {
        "integer_pass_counts": quant_counts,
        "integer_pass_report": quant_report,
        "selection": selection,
        **common_stats,
    }
