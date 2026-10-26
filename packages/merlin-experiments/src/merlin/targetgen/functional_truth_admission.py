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


def _representation(case, arithmetic: str, numerics: Mapping) -> dict:
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
    if classes <= {"float", "complex"} and domain.get("floating_storage_arithmetic") == "declared_quantized_model":
        return {**fields, "operand_dtype": operand}
    if classes <= {"int", "bool"} and storage is not None:
        if element(storage) == element(_storage_name(operand)):
            return {**fields, "operand_dtype": operand}
        if domain.get("native_admission") == "structural_per_dtype_semantic_proof":
            return {**fields, "operand_dtype": operand, "numerical_family": "F5" if "bool" in classes else "F2"}
    return {**fields, "operand_dtype": element(storage)}


def _storage_name(operand: str | None) -> str | None:
    """The framework storage name of a declared operand format (``int8``/``i8`` -> ``int8``)."""
    if operand is None:
        return None
    for name, spelling in _ELEMENT.items():
        if operand in (name, spelling):
            return name
    return None


def source_signature(case, raw: Mapping, family_code: str, spec: Mapping) -> tuple[str, dict] | None:
    """The observed admission signature of a native construction, or ``None`` when it has none."""
    construction = CONSTRUCTIONS.get(family_code)
    if construction is None:
        return None
    family, arithmetic = construction
    result = case.outs[0] if case.outs else None
    rank = len(result.shape) if result is not None else 0
    signature = {
        "family": family,
        "arithmetic": arithmetic,
        "rank": rank,
        "layout": _layout([*case.ins, *case.outs]),
        "aliasing": _aliasing(raw),
        **_representation(case, arithmetic, spec.get("numerical_semantics") or {}),
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


def admit(case, raw: Mapping, family_code: str, admission: Mapping) -> dict:
    """Screen one native construction against the selected spec at accelerator placement."""
    from merlin.targetgen.software_spec import admit_operation

    spec = admission["spec"]
    observed = source_signature(case, raw, family_code, spec)
    if observed is None:
        return {"status": "not_native", "reason": f"{family_code} is not a native construction"}
    _family, signature = observed
    decision = admit_operation(spec, case.ov, signature, "accelerator")
    return {**decision, "observed_signature": signature}


def select_admission(target: str, *, rtl_facts: Path, software_spec: Path, host_capabilities: Path) -> dict:
    """The fact-resolved spec the Phase 0 selection screens with (``core_aten_spec_int8.select_spec``)."""
    import json

    from merlin.targetgen.core_aten_spec_int8 import select_spec

    facts = json.loads(Path(rtl_facts).read_text())
    selection = select_spec(target, facts, Path(software_spec), Path(host_capabilities))
    return {"spec": selection["spec"], "identities": selection["identities"], "target": target}
