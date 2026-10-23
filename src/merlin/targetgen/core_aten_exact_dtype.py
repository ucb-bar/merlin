"""Source-bound native dtype admission, independent of quantized recipes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def bitwise_catalog_coverage(module, manifest):
    """Require complete selector lineage, not a token device copy beside host math."""
    if manifest.get("source_sha256") != hashlib.sha256(str(module).encode()).hexdigest():
        raise ValueError("bitwise provider proof is for different source bytes")
    operations = list(module.walk())
    functions = [op for op in operations if op.name == "func.func" and op.sym_name.data == "forward"]
    if len(functions) != 1:
        raise ValueError("bitwise admission requires one forward definition")
    function = functions[0]
    returns = [op for op in function.body.block.ops if op.name == "func.return"]
    if len(returns) != 1:
        raise ValueError("bitwise admission requires one terminal return")
    bindings = {}
    for binding in manifest.get("bindings", []):
        numeric = binding.get("numeric", {})
        if numeric.get("family") != "F1":
            raise ValueError("native movement catalog includes a non-bitwise binding")
        kernel = next((k for k in manifest["kernels"] if k["symbol"] == binding["symbol"]), None)
        if kernel is None or (
            numeric.get("quantization") != "none"
            or numeric.get("exact_arithmetic") is not True
            or numeric.get("byte_order") != "little"
            or kernel["meta"].get("proof") != "closed_static_element_selectors"
            or kernel["meta"].get("readout") != "scratchpad_bytes"
        ):
            raise ValueError("native movement lacks an exact byte-selector kernel proof")
        ordinal = binding["root_ordinal"]
        if type(ordinal) is not int or not 0 <= ordinal < len(operations) or ordinal in bindings:
            raise ValueError("invalid or duplicate movement root ordinal")
        bindings[ordinal] = binding
    if not bindings:
        raise ValueError("no source-bound bitwise movement kernel")
    visited = set()
    word_epilogues = set()

    def covered(value):
        if value in function.body.block.args:
            return
        owner = value.owner
        ordinal = operations.index(owner)
        if ordinal in visited:
            return
        if ordinal not in bindings:
            from merlin.targetgen.native_nan_boundary import prove

            boundary = prove(owner)
            if boundary is not None:
                if ordinal not in word_epilogues:
                    word_epilogues.add(ordinal)
                    covered(owner.inputs[boundary["runtime_input"]])
                return
            raise ValueError(f"native result depends on unbound {owner.name} at ordinal {ordinal}")
        visited.add(ordinal)
        for source in bindings[ordinal]["operands"].values():
            if "func_arg" in source:
                continue
            producer = operations[source["op_ordinal"]]
            # Static literals carry their declared exact storage encoding.
            if producer.name == "arith.constant":
                continue
            covered(producer.results[source["result"]])

    for value in returns[0].operands:
        covered(value)
    return sorted(visited)


def _expose_literal_movement(module):
    """Expose literal storage bytes, including NaN payloads, as selectors."""
    from math import prod

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerType, TensorType
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import copy, point

    for op in list(module.walk()):
        if op.name != "arith.constant" or not isinstance(op.results[0].type, TensorType):
            continue
        value = op.properties.get("value", op.attributes.get("value"))
        if not isinstance(value, DenseIntOrFPElementsAttr):
            continue
        typ = op.results[0].type
        if not prod(typ.get_shape()):
            continue  # Empty shape metadata has no storage element to splat.
        widths = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8}
        if str(typ.element_type) not in widths:
            if value.is_splat():
                scalar = arith.ConstantOp(value.get_attrs()[0])
                splat = tensor.SplatOp(scalar.result, [], typ)
                Rewriter.replace_op(op, [scalar, splat])
            continue
        width = widths[str(typ.element_type)]
        raw = value.data.data
        count = prod(typ.get_shape())
        if not count:
            continue
        if len(raw) == width:
            raw *= count
        if len(raw) != width * count:
            raise ValueError("dense literal bytes do not cover their declared tensor")
        element = IntegerType(8 * width)
        integer_type = TensorType(element, typ.get_shape())
        bits = [int.from_bytes(raw[i : i + width], "little", signed=True) for i in range(0, len(raw), width)]
        literal = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(integer_type, bits))
        operations = [literal]
        copied = copy(literal.result, operations)
        result = point([copied], typ, lambda args, emit: emit(arith.BitcastOp(args[0], typ.element_type)), operations)
        Rewriter.replace_op(op, operations, [result])


def prepare_bitwise(directory, report, selection, provider, package, module):
    """Try a complete F1 program; unsupported arithmetic retains other routes."""
    from merlin.targetgen.software_spec import admit_operation
    from merlin.targetgen.transfer_contracts import screen_transfer_contract

    if not callable(getattr(provider, "core_aten_catalog", None)):
        return None
    if any(k.startswith("prov.quant.") for op in module.walk() for k in op.attributes):
        return None
    from merlin.llvmlower import image_folds as image
    from merlin.llvmlower import structural_folds as structural

    for fold in (
        image.fold_boolean_unit_image,
        image.fold_boolean_image_predicates,
        image.fold_boolean_domain,
        structural.fold_singleton_reductions,
        structural.fold_singleton_selectors,
        structural.fold_singleton_carriers,
        structural.expose_uniform_literal_carriers,
    ):
        fold(module)
    from merlin.llvmlower.static_reshape_movement import expose_static_reshape_movement

    expose_static_reshape_movement(module)
    from merlin.targetgen.native_nan_boundary import expose_device_transport

    nan_boundaries = expose_device_transport(module)
    # These operations cannot be byte selectors even when a later copy matches.
    if any(
        op.name in {"linalg.matmul", "linalg.batch_matmul", "linalg.reduce"}
        or op.name
        in {
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.negf",
            "arith.extf",
            "arith.truncf",
            "arith.sitofp",
            "arith.fptosi",
        }
        for op in module.walk()
    ):
        return None
    _expose_literal_movement(module)
    from merlin.targetgen.entry_output_movement import expose_argument_results
    from merlin.targetgen.materialized_movement import expose_alias_results

    expose_argument_results(module, report)
    materialized_aliases = expose_alias_results(module, report)
    module.verify()
    source = str(module)
    stage = directory / "dtype-preflight"
    stage.mkdir(exist_ok=True)
    path = stage / "model.mlir"
    path.write_text(source)
    try:
        manifest_path, _ = provider.core_aten_catalog(path, stage, package=package)
        manifest = json.loads(Path(manifest_path).read_text())
        roots = bitwise_catalog_coverage(module, manifest)
    except (ValueError, RuntimeError) as exc:
        (stage / "refusal.json").write_text(json.dumps({"reason": str(exc)}) + "\n")
        return None
    sites = []
    operations = list(module.walk())
    for ordinal in roots:
        typ = operations[ordinal].results[0].type
        signature = dict(
            family="movement",
            numerical_family="F1",
            operand_dtype=str(typ.element_type),
            rank=len(typ.get_shape()),
            layout="row_major_contiguous",
            aliasing="disjoint_inputs_outputs",
        )
        decision = admit_operation(selection["spec"], "movement", signature, "accelerator")
        host = admit_operation(selection["host"], "movement", signature, "host")
        if decision["status"] != "admitted" or host["status"] != "admitted":
            return None
        sites.append(dict(source_operation_ordinal=ordinal, family="movement", status="admitted", software=decision))
    for origin, destination in (("host", "accelerator"), ("accelerator", "host")):
        transfer = screen_transfer_contract(
            selection["spec"],
            source_placement=origin,
            destination_placement=destination,
            operand_dtype="i8",
            result_dtype="i8",
            operand_layout="row_major_contiguous",
            result_layout="row_major_contiguous",
        )
        if transfer["status"] != "admitted":
            return None
    plan = dict(
        status="admitted",
        numerical_family="F1",
        quantized_contractions=0,
        native_contractions=0,
        quantized_non_contraction_sites=0,
        exact_dtype_sites=len(sites),
        sites=sites,
        provider_catalog=manifest,
        native_nan_storage_boundaries=nan_boundaries,
        host_floor=dict(status="required", comparison="native_eager_bits_every_boundary_result"),
    )
    if materialized_aliases:
        plan["materialized_output_aliases"] = dict(
            method="complete_byte_selector_with_static_borrowed_copy_out",
            source_sha256=manifest["source_sha256"],
            output_indices=materialized_aliases,
        )
    (directory / "model.mlir").write_text(source + "\n")
    return publish_plan(directory, report["cases"], selection, plan)


def publish_plan(directory, records, selection, plan):
    """Write spec-int8-admission.json (selection first) and bind plan to each record."""
    document = dict(selection=selection, **plan)
    (directory / "spec-int8-admission.json").write_text(json.dumps(document, indent=2) + "\n")
    for record in records:
        record.update(numeric_admission=plan, numeric_selection=selection["identities"])
    return document


def _host_plan(directory, records, selection, code, reason, **details):
    """A complete host placement with no device sites."""
    plan = dict(status="host", refusal_code=code, **details, host_reason=reason)
    plan.update(quantized_contractions=0, native_contractions=0, quantized_non_contraction_sites=0, sites=[])
    return publish_plan(directory, records, selection, plan)


def empty_host_plan(directory, report, selection):
    """No accelerator work exists for a complete empty tensor boundary."""
    from math import prod

    cases = [r for r in report["cases"] if r.get("status") == "bundled"]
    if not cases:
        return None
    for record in cases:
        outputs = record.get("output_abi", [])
        empty_outputs = bool(outputs) and all(prod(abi["shape"]) == 0 for abi in outputs)
        if not empty_outputs:
            return None
    code = "empty_tensor_no_device_work"
    return _host_plan(directory, cases, selection, code, code)


def native_dtype_request(report):
    """A quantized f32 oracle cannot qualify native integer/half/complex results."""
    records = [record for record in report["cases"] if record.get("status") == "bundled"]
    inputs = [abi["dtype"] for record in records for abi in record.get("input_abi", [])]
    outputs = [abi["dtype"] for record in records for abi in record.get("output_abi", [])]
    return (
        not inputs or any(dtype != "f32" for dtype in inputs) or any(dtype.startswith("complex") for dtype in outputs)
    )


def retain_native_host(directory, report, selection):
    """Explicit native host fallback grants no lossy int8 recapture."""
    reason = dict(family="native", reason="no closed exact component producer for the native source arithmetic")
    path = directory / "exact-dtype-refusal.json"
    if path.exists():
        reason = json.loads(path.read_text())
    code = "native_dtype_exact_components_unavailable"
    return _host_plan(directory, report["cases"], selection, code, reason["reason"], refusal_details=reason)
