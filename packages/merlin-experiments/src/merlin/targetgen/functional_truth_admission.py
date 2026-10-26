"""Must-accelerate admission of a source operation, decided by the selected software spec.

The functional rules classify what an overload computes: the native construction a case reduces to
(a contraction, an additive reduction, an affine map, movement or a literal fill) or why it has
none. They do not decide acceleration. A native construction becomes a device obligation only when
:func:`merlin.targetgen.software_spec.admit_operation` admits its observed signature against the
selected, fact-resolved spec; the declarations there (ranks, layouts, tails, aliasing, placement,
arithmetic, operand formats) are the single authority. Nothing here names a target or an op
spelling: observations are read from the case's schema-bound tensors.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from merlin.targetgen.access_observations import broadcast_axes
from merlin.targetgen.admission_conditions import binding_declaration

from .functional_truth_types import dclass

#: Native construction (functional truth family code) -> (semantic family, arithmetic class).
#: These are the codes of :data:`functional_truth_types.FAM`, a semantic vocabulary.
CONSTRUCTIONS = {
    "N1": ("contraction", "integer_multiply_accumulate"),
    "N1t": ("contraction", "integer_multiply_accumulate"),
    "N2": ("reduction", "integer_sum"),
    "N2e": ("reduction", "integer_sum"),
    "N3": ("elementwise_map", "integer_affine"),
    "N3c": ("elementwise_map", "integer_affine"),
    "N4": ("movement", "none"),
    "N5": ("movement", "none"),
    "N6": ("movement", "none"),
}
#: Framework storage names in the element spelling software declarations use.
_ELEMENT = {
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float64": "f64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "ui8",
    "bool": "i1",
    "complex32": "complex<f16>",
    "complex64": "complex<f32>",
    "complex128": "complex<f64>",
}
_ROW_MAJOR = "row_major_contiguous"
_STATIC_TAILS = "zero_pad_valid_window"


def element(dtype: str | None) -> str | None:
    return _ELEMENT.get(dtype) if dtype is not None else None


def _layout(tensors) -> str | None:
    """The layout of the operands the construction reads: dense row-major tensor values.

    A captured call passes immutable tensor values; the caller's strides and offsets are bound at
    the call boundary (``core_aten_batch.caller_input_storage``), never by the device operation.
    A tensor without a known shape leaves the layout unobserved.
    """
    return _ROW_MAJOR if all(isinstance(tensor.shape, list) for tensor in tensors) else None


def _broadcasting(inputs, result) -> str:
    """``none`` for equal shapes, ``outer``/``inner`` for a right-aligned broadcast, else ``none``.

    A shape change that is not a broadcast (a pad, a repeat, a selection) is movement's own form,
    not operand broadcasting.
    """
    if result is None:
        return "none"
    observed = "none"
    for tensor in inputs:
        if tensor.shape == result.shape or len(tensor.shape) > len(result.shape):
            continue
        aligned = [1] * (len(result.shape) - len(tensor.shape)) + list(tensor.shape)
        if any(extent not in (1, out) for extent, out in zip(aligned, result.shape, strict=True)):
            continue
        expanded = [i for i, (extent, out) in enumerate(zip(aligned, result.shape, strict=True)) if extent != out]
        broadcasting = broadcast_axes(expanded, len(result.shape))
        if observed != "inner" and broadcasting != "none":
            observed = broadcasting
    return observed


def _aliasing(raw: Mapping) -> str:
    if raw.get("mutated_arguments"):
        return "in_place"
    if raw.get("output_input_aliases"):
        return "result_views_input"
    return "disjoint_inputs_outputs"


def _representation(case, arithmetic: str, numerics: Mapping, *, components=False) -> dict:
    """The operand formats the construction computes in, from the spec's declared numeric policy.

    Movement keeps the storage bytes. Arithmetic over floating storage is the declared quantized
    model when the spec selects it; integer storage wider than the operand format is the declared
    radix component representation. An undeclared policy leaves the storage format, which the
    declarations then screen as it is.
    """
    storage = (case.ins or case.outs)[0].dtype if (case.ins or case.outs) else None
    if arithmetic == "none":
        return {"operand_dtype": element(storage), "numerical_family": "F1"}
    operand = numerics.get("operand_dtype")
    fields = {"accum_dtype": numerics.get("accumulator_dtype"), "readout_dtype": numerics.get("readout_dtype")}
    domain = numerics.get("component_domain") or {}
    classes = {dclass(tensor.dtype) for tensor in case.ins} or {dclass(storage)}
    storage_rule = numerics.get("quantized_native_fallback") or {}
    quantized = (
        classes <= {"float"}
        and domain.get("floating_storage_arithmetic") == "declared_quantized_model"
        and all(element(tensor.dtype) in storage_rule.get("operand_storage", []) for tensor in case.ins)
    )
    if quantized and components and domain.get("native_admission") == "structural_per_dtype_semantic_proof":
        return {**fields, "operand_dtype": operand, "arithmetic": "integer_components", "numerical_family": "F2"}
    if quantized:
        return {**fields, "operand_dtype": operand}
    if classes <= {"int", "bool"} and storage is not None:
        if not components and element(storage) == element(_storage_name(operand)):
            return {**fields, "operand_dtype": operand}
        if domain.get("native_admission") == "structural_per_dtype_semantic_proof":
            # rewrite_integer_components emits balanced radix-256 signed bytes
            # with full i32 partial sums, independently of an operation declaration.
            return {
                "operand_dtype": "i8",
                "accum_dtype": "i32",
                "readout_dtype": "i32",
                "arithmetic": "integer_components",
                "numerical_family": "F5" if "bool" in classes else "F2",
            }
        if element(storage) == element(_storage_name(operand)):
            return {**fields, "operand_dtype": operand}
    return {**fields, "operand_dtype": element(storage)}


def _storage_name(operand: str | None) -> str | None:
    """The framework storage name of a declared operand format (``int8``/``i8`` -> ``int8``)."""
    if operand is None:
        return None
    for name, spelling in _ELEMENT.items():
        if operand in (name, spelling):
            return name
    return None


def source_signature(
    case, raw: Mapping, family_code: str, spec: Mapping, *, components=False, contraction_rank=None
) -> tuple[str, dict] | None:
    """The observed admission signature of a native construction, or ``None`` when it has none.

    ``contraction_rank`` is the result rank of the contraction the captured source performs,
    when one was observed; it replaces the operator's result rank for a contraction.
    """
    construction = CONSTRUCTIONS.get(family_code)
    if construction is None:
        return None
    family, arithmetic = construction
    result = case.outs[0] if case.outs else None
    rank = len(result.shape) if result is not None else 0
    if family == "contraction" and contraction_rank is not None:
        rank = contraction_rank
    signature = {
        "family": family,
        "arithmetic": arithmetic,
        "rank": rank,
        "layout": _layout([*case.ins, *case.outs]),
        "aliasing": _aliasing(raw),
        **_representation(case, arithmetic, spec.get("numerical_semantics") or {}, components=components),
    }
    if signature.get("arithmetic") == "integer_components":
        form = "bilinear" if family == "contraction" else "linear"
        signature["numeric"] = {"family": signature["numerical_family"], "form": form}
        signature["numerical_contract"] = {
            "representation": "signed_radix_256_components",
            "producer": f"closed_static_{form}_integer",
        }
    if family in ("contraction", "reduction"):
        # Captured cases have static extents; a padded element of a sum is its additive identity.
        signature["tails"] = _STATIC_TAILS
    if family == "contraction":
        signature["broadcasting"] = "independent_batches" if rank > 2 else "none"
    elif family == "reduction":
        signature["kind"] = "sum"
    elif family == "elementwise_map":
        signature["expression"] = "affine"
        signature["tensor_operands"] = len(case.ins)
        signature["broadcasting"] = _broadcasting(case.ins, result)
    else:
        signature["broadcasting"] = _broadcasting(case.ins[:1], result)
    return family, signature


def _component_refusal(signature: Mapping, spec: Mapping) -> dict | None:
    """Require the source-derived component form's own compatible declaration."""
    if signature.get("arithmetic") != "integer_components":
        return None
    identity = binding_declaration(signature)
    declaration = next((row for row in spec.get("operations", []) if row["id"] == identity), None)
    if declaration is None:
        return {"status": "unsupported", "reason": f"no SW operation declaration {identity}"}
    contract = declaration.get("numerical_contract") or {}
    for field, observed in signature["numerical_contract"].items():
        if contract.get(field) != observed:
            return {
                "status": "unsupported",
                "declaration": identity,
                "reason": f"observed {field} {observed!r} differs from declared {contract.get(field)!r}",
            }
    return None


def _storage_only(module, *, arithmetic=()) -> bool:
    """A normalized source containing only storage selectors and index arithmetic."""
    from xdsl.dialects.builtin import IndexType

    structural = {
        "builtin.module",
        "func.func",
        "func.return",
        "linalg.generic",
        "linalg.yield",
        "linalg.index",
        "arith.constant",
        "tensor.empty",
        "tensor.cast",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.extract_slice",
        "tensor.insert_slice",
        "tensor.concat",
        "tensor.splat",
    }
    for op in module.walk():
        if op.name in structural or op.name in arithmetic:
            continue
        if op.name.startswith("arith.") and op.operands and all(isinstance(v.type, IndexType) for v in op.operands):
            continue
        if op.name == "arith.select":
            condition = op.operands[0].owner
            if getattr(condition, "name", None) == "arith.cmpi" and all(
                isinstance(v.type, IndexType) for v in condition.operands
            ):
                continue
        return False
    return True


def _reads_arguments(module) -> bool:
    """Some operation of the (normalized) source reads a function argument.

    Returning an argument unchanged (an input boundary the caller observes) reads no data.
    """
    return any(
        use.operation.name != "func.return"
        for op in module.walk()
        if op.name == "func.func"
        for argument in op.body.block.args
        for use in argument.uses
    )


def _nonfinite_literals(module) -> bool:
    """Some floating literal of the source is NaN or infinite."""
    import math

    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, FloatAttr

    for op in module.walk():
        if op.name != "arith.constant":
            continue
        value = op.properties.get("value") or op.attributes.get("value")
        if isinstance(value, FloatAttr):
            words = [value.value.data]
        elif isinstance(value, DenseIntOrFPElementsAttr):
            words = [word for word in value.get_values() if isinstance(word, float)]
        else:
            continue
        if any(not math.isfinite(word) for word in words):
            return True
    return False


def _data_comparisons(module) -> bool:
    """The source orders a data word against another value, as a threshold or compare/select does.

    Min/max against a different word also orders data. Loop-index comparisons are addressing.
    A word compared with itself is an IEEE special-value test (``uno``/``ord``), which typed
    component proofs carry; it orders nothing.
    """
    from xdsl.dialects.builtin import IndexType

    return any(
        op.name
        in {
            "arith.cmpi",
            "arith.cmpf",
            "arith.maxsi",
            "arith.minsi",
            "arith.maxui",
            "arith.minui",
            "arith.maximumf",
            "arith.minimumf",
            "arith.maxnumf",
            "arith.minnumf",
        }
        and not all(isinstance(v.type, IndexType) for v in op.operands)
        and op.operands[0] is not op.operands[1]
        for op in module.walk()
    )


def _contraction_rank(module) -> int | None:
    """The largest classified contraction rank, or ``None`` if a product reduction is unclassified.

    An incomplete observation cannot replace the operator's result rank.
    """
    from xdsl.dialects.builtin import IndexType

    from merlin.kernels.shapes import observe_contractions

    observed = observe_contractions(module)
    classified = {op for op, _shape in observed}
    for op in module.walk():
        if op.name != "linalg.generic" or op in classified:
            continue
        if not any(it.data.value == "reduction" for it in op.get_iterator_types()):
            continue
        if any(
            child.name in {"arith.muli", "arith.mulf"}
            and not all(isinstance(value.type, IndexType) for value in child.operands)
            and all(getattr(value.owner, "name", None) != "arith.constant" for value in child.operands)
            for child in op.walk()
        ):
            return None
    ranks = [len(op.results[0].type.get_shape()) for op, _shape in observed]
    return max(ranks) if ranks else None


def _normalized(source_module):
    from merlin.llvmlower.entry_weight_projection import _pure_dce
    from merlin.targetgen.core_aten_numeric import normalize_closed_identities

    normalized = source_module.clone()
    normalize_closed_identities(normalized)
    for function in normalized.walk():
        if function.name == "func.func":
            _pure_dce(function)
    return normalized


def admit(case, raw: Mapping, family_code: str, admission: Mapping, *, source_module=None, descriptor_folds=()) -> dict:
    """Screen one native construction against the selected spec at accelerator placement.

    With the captured source, the projection reads what that source computes:

    * ``descriptor_folds`` are the capture's proofs that the result folds from the inputs'
      static descriptors (:func:`core_aten_capture.static_descriptor_folds`). With a source
      that reads no argument, such a result has no tensor data for any unit to place.
    * A source of storage selectors alone is movement, unless it materializes a nonfinite
      literal: exact literal producers cover finite words only.
    * A contraction's rank is the rank of the contraction in the source (an im2col
      convolution contracts a matrix). An unclassified product reduction leaves the
      operator's result rank in place.
    * Radix components are a closed linear producer. A source that compares data words
      (a threshold, a compare/select, min/max) is not linear in them, so the component form does
      not carry it.
    """
    from merlin.targetgen.software_spec import admit_operation

    spec = admission["spec"]
    normalized = contraction_rank = None
    if source_module is not None:
        normalized = _normalized(source_module)
        if descriptor_folds and not _reads_arguments(normalized):
            return {
                "status": "not_native",
                "reason": "result folds from static input descriptors and reads no tensor data",
            }
        if _storage_only(normalized) and not _nonfinite_literals(normalized):
            from merlin.targetgen.native_unit_policy import construction_evidence

            excluded = (spec.get("numerical_semantics", {}).get("native_unit_contract") or {}).get(
                "excluded_constructions"
            )
            if construction_evidence(normalized, excluded=excluded)["status"] == "eligible":
                family_code = "N4"
        elif (
            {dclass(tensor.dtype) for tensor in case.ins} <= {"int", "bool"}
            and _storage_only(
                source_module, arithmetic={"arith.addi", "arith.subi", "arith.extsi", "arith.extui", "arith.trunci"}
            )
            and any(
                op.name == "linalg.generic" and any(it.data.value == "reduction" for it in op.get_iterator_types())
                for op in source_module.walk()
            )
        ):
            # A masked additive scan is linear in its inputs. Its hypothetical
            # triangular-matrix construction does not impose a bilinear rank.
            family_code = "N2"
        contraction_rank = _contraction_rank(normalized)
    observed = source_signature(case, raw, family_code, spec, contraction_rank=contraction_rank)
    if observed is None:
        return {"status": "not_native", "reason": f"{family_code} is not a native construction"}
    _family, signature = observed
    decision = _component_refusal(signature, spec) or admit_operation(spec, case.ov, signature, "accelerator")
    if source_module is not None and decision["status"] != "admitted":
        alternative = source_signature(
            case, raw, family_code, spec, components=True, contraction_rank=contraction_rank
        )[1]
        if alternative.get("arithmetic") == "integer_components" and alternative != signature:
            represented = _component_refusal(alternative, spec) or admit_operation(
                spec, case.ov, alternative, "accelerator"
            )
            if represented["status"] == "admitted":
                signature, decision = alternative, represented
    if (
        decision["status"] == "admitted"
        and signature.get("arithmetic") == "integer_components"
        and decision.get("declaration") != binding_declaration(signature)
    ):
        decision = {"status": "unsupported", "reason": "source component form requires its own declaration"}
    if (
        decision["status"] == "admitted"
        and signature.get("arithmetic") == "integer_components"
        and normalized is not None
        and _data_comparisons(normalized)
    ):
        decision = {
            "status": "unsupported",
            "declaration": decision.get("declaration"),
            "reason": "the source compares data words; a linear component producer does not carry the comparison",
        }
    return {**decision, "observed_signature": signature}


def select_admission(target: str, *, rtl_facts: Path, software_spec: Path, host_capabilities: Path) -> dict:
    """The fact-resolved spec the Phase 0 selection screens with (``core_aten_spec_int8.select_spec``)."""
    import json

    from merlin.targetgen.core_aten_spec_int8 import select_spec

    facts = json.loads(Path(rtl_facts).read_text())
    selection = select_spec(target, facts, Path(software_spec), Path(host_capabilities))
    return {
        "spec": selection["spec"],
        "identities": selection["identities"],
        "target": target,
        "rtl_facts": facts,
        "capability_contract": selection["capability_contract"],
        "host_capabilities": selection["host"],
    }
