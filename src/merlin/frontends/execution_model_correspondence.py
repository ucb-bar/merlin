"""Independent complete-SSA reconstruction of executed model call binding.

The consumer owns this reader. No capture frontend, producer analysis, sample
inputs or reference arrays are imported. Correspondence proves that ordinary
parent computation and retained kernel bodies/guards were actually preserved;
it grants no TTIR numeric/effect equivalence or downstream lifetime permission.
"""

import hashlib
from io import StringIO

from xdsl.dialects import builtin, func
from xdsl.printer import Printer

from .derived_runtime_arguments import checked_source_layout_call
from .execution_call_contracts import CallContracts, scalar_attribute
from .execution_call_views import FIELDS as VIEW_FIELDS
from .execution_call_views import mark, source_views
from .execution_call_views import validate as validate_views
from .kernel_shape_abi import validate_dimensions
from .linalg_mlir import parse_mlir_text

SCHEMA = "t2m.model-kernel-binding.v4"
PROFILE = "t2m.execution-checked-calls.v1"
_CALL_FIELDS = {
    "id",
    "parent_call_ordinal",
    "callee",
    "kernel",
    "schema_operands",
    "model_call_attributes",
    "source_operand_bindings",
    "source_result_order",
    "captured_scalar_arguments",
    "runtime_scalar_arguments",
    "runtime_operand_arguments",
    "source_layout_arguments",
    "implementation_sha256",
    "source_sha256",
}


def input_contract(entry):
    """Bind only actual public input axes; original executable guards remain.

    This does not infer result extents or recover runtime bounds from samples.
    Complete expansion checking preserves the original source-domain guards.
    """
    inputs, axes = [], {}
    for ordinal, typ in enumerate(entry.function_type.inputs):
        if isinstance(typ, builtin.TensorType):
            shape = []
            for axis, size in enumerate(typ.get_shape()):
                if size == builtin.DYNAMIC_INDEX:
                    name = f"d{ordinal}_{axis}"
                    axes[name] = {"argument": ordinal, "axis": axis}
                    shape.append(name)
                else:
                    shape.append(size)
            inputs.append({"kind": "tensor", "dtype": str(typ.element_type), "shape": shape})
        else:
            inputs.append({"kind": "scalar", "dtype": str(typ), "shape": []})
    return {"inputs": inputs, "dimension_bindings": axes}


def _interface(spec, function):
    buffers, scalars = spec["buffers"], spec.get("runtime_scalars", [])
    if any(buffer["effect"] not in ("read", "write", "readwrite") for buffer in buffers) or any(
        scalar["dtype"] not in ("i1", "i8", "i16", "i32", "i64", "f32", "f64") for scalar in scalars
    ):
        raise ValueError("executed kernel has an unsupported storage/effect/scalar declaration")
    validate_dimensions(spec, buffers, scalars)
    names = [item["name"] for item in (*buffers, *scalars)]
    if len(set(names)) != len(names) or not function.body.blocks or len(function.body.blocks) != 1:
        raise ValueError("executed model needs one unambiguous typed kernel body")
    inputs, outputs = [b for b in buffers if b["effect"] != "write"], [b for b in buffers if b["effect"] != "read"]
    if (
        len(function.function_type.inputs) != len(inputs) + len(scalars)
        or len(function.function_type.outputs) != len(outputs)
        or not isinstance(function.body.block.last_op, func.ReturnOp)
    ):
        raise ValueError("executed kernel formal/result roster differs from its ABI")
    for buffer, typ in zip(
        (*inputs, *outputs),
        (*list(function.function_type.inputs)[: len(inputs)], *function.function_type.outputs),
        strict=True,
    ):
        if (
            not isinstance(typ, builtin.TensorType)
            or not isinstance(typ.encoding, builtin.NoneAttr)
            or str(typ.element_type) != buffer["dtype"]
            or len(typ.get_shape()) != len(buffer["shape"])
            or any(
                actual != (builtin.DYNAMIC_INDEX if isinstance(n, str) else n)
                for n, actual in zip(buffer["shape"], typ.get_shape(), strict=True)
            )
        ):
            raise ValueError("executed kernel tensor types differ from its declared storage/shapes")
    return inputs, outputs


def checked_execution_model(parent, bound, bindings, interfaces):
    """Check full SSA expansion after retained artifact custody is verified."""
    try:
        return _checked_execution_model(parent, bound, bindings, interfaces)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError("malformed executed model correspondence") from error


def _checked_execution_model(parent, bound, bindings, interfaces):
    """Reconstruct every original call expansion and compare the entire model."""
    if (
        bindings.get("schema") != SCHEMA
        or bindings.get("binding_profile") != PROFILE
        or bindings.get("result_mode") != "execution-owned"
        or bindings.get("resource_scope") != "execution-allocator-accounted"
        or any(
            key in bindings
            for key in (
                "runtime_shape_contract",
                "model_shape_profile",
                "guard_storage_bytes",
                "argument_binding_profile",
                "tensor_view_binding_profile",
                "source_tensor_view_binding_profile",
            )
        )
    ):
        raise ValueError("executed model requires its explicit call/result/resource profile")
    records = bindings.get("calls")
    if (
        set(interfaces) != set(bindings.get("kernels", {}))
        or not isinstance(records, list)
        or {record.get("implementation_sha256") for record in records} != set(interfaces)
    ):
        raise ValueError("executed correspondence has an incomplete implementation inventory")
    located = bindings.get("located_implementations")
    if not isinstance(located, dict) or set(located) != set(interfaces):
        raise ValueError("executed correspondence lacks its complete located kernel inventory")
    validate_views(bindings)
    layout_profile = bindings.get("source_layout_binding_profile")
    specialized = any("source_layout_specialization" in kernel for kernel in bindings["kernels"].values())
    if (
        specialized
        and layout_profile != "t2m.captured-source-layout.v1"
        or not specialized
        and "source_layout_binding_profile" in bindings
    ):
        raise ValueError("executed source layouts require their explicit captured source-layout profile")
    checked_interfaces = {}
    for digest, (spec, source) in interfaces.items():
        raw = source.encode()
        if located[digest] != {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}:
            raise ValueError("executed located kernel differs from its retained artifact")
        kernel = parse_mlir_text(source)
        kernel.verify()
        functions = list(kernel.ops)
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(kernel)
        if hashlib.sha256((stream.getvalue() + "\n").encode()).hexdigest() != digest:
            raise ValueError("executed located kernel changes its retained unlocated implementation")
        if len(functions) != 1 or not isinstance(functions[0], func.FuncOp):
            raise ValueError("executed located kernel needs one defined implementation")
        checked_interfaces[digest] = (spec, functions[0])
    expected = parent.clone()
    calls = [op for op in expected.walk() if isinstance(op, func.CallOp)]
    if not isinstance(records, list) or len(calls) != len(records):
        raise ValueError("executed correspondence does not cover every original call")
    declarations = {op.sym_name.data: op for op in expected.ops if isinstance(op, func.FuncOp)}
    used = set()
    for ordinal, (call, record) in enumerate(zip(calls, records, strict=True)):
        if not isinstance(record, dict) or set(record) - VIEW_FIELDS - {"source_layout_bindings"} != _CALL_FIELDS:
            raise ValueError("executed call has an unknown or incomplete correspondence record")
        symbol = call.callee.root_reference.data
        declaration = declarations.get(symbol)
        identity = call.attributes.get("prov.kernel")
        names = call.attributes.get("prov.kernel_operands")
        attributes = call.attributes.get("prov.kernel_args", builtin.DictionaryAttr({}))
        if (
            type(record["parent_call_ordinal"]) is not int
            or record["parent_call_ordinal"] != ordinal
            or record["id"] != f"kernel_call.{ordinal:04d}"
            or record["callee"] != symbol
            or not isinstance(identity, builtin.StringAttr)
            or record["kernel"] != identity.data
            or declaration is None
            or declaration.body.blocks
            or declaration.attributes.get("prov.kernel") != identity
            or not isinstance(names, builtin.ArrayAttr)
            or any(not isinstance(name, builtin.StringAttr) for name in names)
            or len(names) != len(call.operands)
            or len({n.data for n in names}) != len(names)
            or record["schema_operands"] != [n.data for n in names]
            or record["model_call_attributes"] != {name: str(value) for name, value in call.attributes.items()}
            or not isinstance(attributes, builtin.DictionaryAttr)
            or builtin.DictionaryAttr(
                {key: scalar_attribute(value) for key, value in record["captured_scalar_arguments"].items()}
            )
            != attributes
        ):
            raise ValueError("executed correspondence changes the original typed model call")
        spec, function = checked_interfaces[record["implementation_sha256"]]
        retained = bindings["kernels"][record["implementation_sha256"]]
        if spec != retained.get("spec") or record["source_sha256"] != retained.get("source_sha256"):
            raise ValueError("executed call changes the retained kernel source or ABI")
        layout_arguments = record["source_layout_arguments"]
        layout_bindings = record.get("source_layout_bindings")
        if (
            not isinstance(layout_arguments, dict)
            or "source_layout_specialization" in retained
            and (
                not isinstance(layout_bindings, dict)
                or any(not isinstance(item, dict) for item in layout_bindings.values())
                or layout_arguments != {key: item.get("schema_argument") for key, item in layout_bindings.items()}
            )
            or "source_layout_specialization" not in retained
            and (layout_arguments or "source_layout_bindings" in record)
        ):
            raise ValueError("executed source layout arguments differ from their checked captured bindings")
        checked_source_layout_call(parent, retained, record, layout_profile)
        input_buffers, output_buffers = _interface(spec, function)
        mapping, live = record["source_operand_bindings"], record["runtime_operand_arguments"]
        views = record.get("source_tensor_views", {})
        if (
            set(mapping) != {b["name"] for b in input_buffers}
            or len(set(mapping.values())) != len(mapping)
            or set(mapping.values()) | set(live.values()) | set(views) != {n.data for n in names}
            or len(record["source_result_order"]) != len(output_buffers)
            or set(record["source_result_order"]) != {b["name"] for b in output_buffers}
            or len(call.results) != len(output_buffers)
        ):
            raise ValueError("executed correspondence loses original operands or results")
        emitted = []
        callsite = record["id"]

        def append(op, role, *, call=call, record=record, identity=identity, callsite=callsite, emitted=emitted):
            op.location = call.location
            op.attributes.update(
                {
                    "prov.family": builtin.StringAttr("binding"),
                    "prov.op": builtin.StringAttr(role),
                    "prov.kernel_origin": identity,
                    "prov.kernel_callsite": builtin.StringAttr(callsite),
                    "prov.region_id": builtin.StringAttr(callsite + "/abi"),
                    "prov.emitted_id": builtin.StringAttr(f"{callsite}/abi.{len(emitted):04d}"),
                    "prov.kernel_source_sha256": builtin.StringAttr(record["source_sha256"]),
                }
            )
            emitted.append(op)
            return op.results[0] if op.results else None

        contracts = CallContracts(append)
        arguments, dimensions = contracts.arguments(
            spec, function, record, dict(zip((n.data for n in names), call.operands, strict=True)), attributes
        )
        checked_views = source_views(
            record, dict(zip((n.data for n in names), call.operands, strict=True)), attributes, retained,
            scalars={scalar["name"]: value for scalar, value in
                     zip(spec.get("runtime_scalars", []), arguments[len(input_buffers):], strict=True)}, append=append
        )
        mapper = dict(zip(function.body.block.args, arguments, strict=True))
        for source in function.body.block.ops:
            if isinstance(source, func.ReturnOp):
                returned = [mapper[value] for value in source.arguments]
                break
            cloned = source.clone(value_mapper=mapper)
            mark(cloned, checked_views, record.get("source_view_policy"))
            for op in cloned.walk():
                op.attributes.update(
                    {
                        "prov.kernel_origin": identity,
                        "prov.kernel_callsite": builtin.StringAttr(callsite),
                        "prov.kernel_source_sha256": builtin.StringAttr(record["source_sha256"]),
                    }
                )
                for key in ("prov.region_id", "prov.emitted_id"):
                    if isinstance(prior := op.attributes.get(key), builtin.StringAttr):
                        op.attributes[key] = builtin.StringAttr(callsite + "/" + prior.data)
            emitted.append(cloned)
        results = {}
        for buffer, value in zip(output_buffers, returned, strict=True):
            contracts.dimensions(value, buffer, dimensions)
            results[buffer["name"]] = value
        replacement = [
            contracts.view(
                results[name],
                old.type,
                next(b for b in output_buffers if b["name"] == name),
                record,
                attributes,
                role="kernel-to-caller-view",
                dimensions=dimensions,
            )
            for name, old in zip(record["source_result_order"], call.results, strict=True)
        ]
        call.parent.insert_ops_before(emitted, call)
        for old, value in zip(call.results, replacement, strict=True):
            old.replace_all_uses_with(value)
        call.parent.erase_op(call)
        used.add(symbol)
    for symbol, declaration in declarations.items():
        if "prov.kernel" in declaration.attributes:
            if symbol not in used or declaration.body.blocks:
                raise ValueError("executed correspondence retains an unresolved kernel declaration")
            expected.body.block.erase_op(declaration)
    expected.attributes["prov.kernel_binding"] = builtin.StringAttr(SCHEMA)
    weight = bound.attributes.get("prov.weights_file")
    if weight not in (parent.attributes.get("prov.weights_file"), builtin.StringAttr("weights.safetensors")):
        raise ValueError("executed correspondence changes the original weight reference")
    if weight is not None:
        expected.attributes["prov.weights_file"] = weight
    if any(
        isinstance(op, func.CallOp) or op.name == "builtin.unregistered" or "prov.kernel" in op.attributes
        for op in expected.walk()
    ):
        raise ValueError("executed expansion retains an unresolved source dependency")
    expected.verify()
    if not expected.is_structurally_equivalent(bound):
        raise ValueError("executed model body differs from original SSA, kernel bodies or executable call guards")
    if any(
        tuple(value.type for value in a.results) != tuple(value.type for value in b.results)
        for a, b in zip(expected.walk(), bound.walk(), strict=True)
    ):
        raise ValueError("executed model changes an original operation result type")
    if any(a.location != b.location for a, b in zip(expected.walk(), bound.walk(), strict=True)):
        raise ValueError("executed model source locations differ from original parent and kernel bodies")
    return {
        "schema": "merlin.executed-model-correspondence.v1",
        "calls": len(calls),
        "scope": "complete-SSA-call-expansion; TTIR-equivalence-and-target-admission-unverified",
    }
