"""Source-only selection of execution-produced result descriptors.

Public ranks, storage and static dimensions come from actual MLIR types.
Existing invocation input constraints remain restrictions, never output-size
predictions or permission to change arithmetic, effects or ownership.
"""

import math
from copy import deepcopy
from io import StringIO

from xdsl.dialects import builtin, func
from xdsl.printer import Printer

MODE = "execution-owned"
PROFILE = "merlin.execution-owned-results.v1"
SCHEMA = "merlin.compile-signature.v6"
_SCALARS = {"i1", "i8", "i16", "i32", "i64", "f32", "f64"}
_STORAGE = _SCALARS | {"f16", "bf16", "f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ"}
LIMITS_PROFILE = "merlin.owned-result-limits.v2"


def checked_allocation_policy(value=None):
    """Validate an explicit finite resource selection, separately from semantics.

    Legacy v1 is the exact original default. V2 permits independently selected
    byte/output/work limits without removing any runtime allocation check.
    These are operator resource bounds, never source or evaluator shape facts.
    """
    default = {
        "schema": "merlin.owned-result-limits.v1",
        "max_bytes": 64 << 20,
        "max_allocations": 4096,
        "max_output_bytes": 64 << 20,
        "index_bits": 64,
    }
    if value is None:
        return default
    if not isinstance(value, dict) or set(value) != set(default):
        raise ValueError("execution allocation policy needs its complete explicit limits")
    if value.get("schema") == default["schema"]:
        if value != default or any(type(value[key]) is not int for key in default if key != "schema"):
            raise ValueError("legacy execution allocation policy has fixed limits")
        return default
    if (
        value.get("schema") != LIMITS_PROFILE
        or type(value["index_bits"]) is not int
        or value["index_bits"] != 64
        or any(
            type(value[key]) is not int or not 0 < value[key] < 1 << 63
            for key in ("max_bytes", "max_allocations", "max_output_bytes")
        )
    ):
        raise ValueError("execution allocation policy requires finite positive signed-i64 limits and supported indices")
    return dict(value)


def validate_mode(mode):
    if mode is not None and mode != MODE:
        raise ValueError("unknown explicitly selected result mode")


def program_text(module, mode):
    validate_mode(mode)
    if mode is None:
        from merlin.compiler.dialects.xdsl_context import text

        return text(module, generic=True)
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(module)
    return stream.getvalue()


def _types(module, entry):
    functions = [op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == entry]
    if len(functions) != 1 or not functions[0].body.blocks:
        raise ValueError("executed results require one defined original entry")
    function = functions[0]
    if function.arg_attrs is not None:
        for typ, attributes in zip(function.function_type.inputs, function.arg_attrs, strict=True):
            writable = attributes.data.get("bufferization.writable")
            if (
                isinstance(typ, builtin.TensorType)
                and writable is not None
                and (not isinstance(writable, builtin.IntegerAttr) or writable.value.data != 0)
            ):
                raise ValueError("execution result source inputs must be borrowed immutable tensor values")

    def slot(typ, *, output=False):
        if isinstance(typ, builtin.TensorType):
            if not isinstance(typ.encoding, builtin.NoneAttr) or len(typ.get_shape()) > 16:
                raise ValueError("executed results require bounded ranked unencoded tensor storage")
            dtype = str(typ.element_type)
            if dtype not in _STORAGE:
                raise ValueError("unsupported execution-result tensor storage")
            return {
                "kind": "tensor",
                "dtype": dtype,
                "shape": [None if n == builtin.DYNAMIC_INDEX else n for n in typ.get_shape()],
            }
        if output or str(typ) not in _SCALARS:
            raise ValueError("executed results require tensor outputs and supported native scalar inputs")
        return {"kind": "scalar", "dtype": str(typ), "shape": []}

    inputs = [slot(t) for t in function.function_type.inputs]
    outputs = [slot(t, output=True) for t in function.function_type.outputs]
    if len(inputs) > 64 or not 1 <= len(outputs) <= 64:
        raise ValueError("executed result argument/result roster exceeds its explicit limits")
    return inputs, outputs


def _constraints(inputs, rows, bindings):
    if not isinstance(rows, list) or len(rows) != len(inputs) or not isinstance(bindings, dict):
        raise ValueError("executed result inputs need their complete original constraints")
    names, scalar_names = set(), set()
    for original, actual in zip(rows, inputs, strict=True):
        if (
            not isinstance(original, dict)
            or set(original) - {"kind", "dtype", "shape", "scalar", "bounds"}
            or original.get("dtype") != actual["dtype"]
            or original.get("kind", "tensor") != actual["kind"]
            or not isinstance(original.get("shape"), list)
            or len(original["shape"]) != len(actual["shape"])
        ):
            raise ValueError("executed input constraints change original storage/rank/kind")
        for size, wanted in zip(original["shape"], actual["shape"], strict=True):
            if wanted is not None and size != wanted:
                raise ValueError("executed input constraint changes a static source extent")
            if size is not None and not (type(size) is int and 0 <= size < 1 << 63 or type(size) is str and size):
                raise ValueError("unsupported executed input dimension constraint")
            if isinstance(size, str):
                names.add(size)
        if "bounds" in original:
            bounds = original["bounds"]
            if (
                actual["kind"] != "scalar"
                or not isinstance(bounds, list)
                or len(bounds) != 2
                or any(type(n) not in (int, float) or not math.isfinite(n) for n in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError("invalid original native scalar input bounds")
        if "scalar" in original:
            if actual["kind"] != "scalar" or type(original["scalar"]) is not str or not original["scalar"]:
                raise ValueError("invalid original scalar dimension binding")
            scalar_names.add(original["scalar"])
    for name, coordinate in bindings.items():
        if (
            type(name) is not str
            or not name
            or not isinstance(coordinate, dict)
            or set(coordinate) != {"argument", "axis"}
            or type(coordinate["argument"]) is not int
            or type(coordinate["axis"]) is not int
        ):
            raise ValueError("invalid execution input-axis binding")
        argument, axis = coordinate["argument"], coordinate["axis"]
        if not 0 <= argument < len(inputs) or not 0 <= axis < len(inputs[argument]["shape"]):
            raise ValueError("execution input-axis binding is outside the source signature")
        if rows[argument]["shape"][axis] != name or inputs[argument]["shape"][axis] is not None:
            raise ValueError("execution input-axis binding changes its actual source coordinate")
    if names - set(bindings) - scalar_names:
        raise ValueError("execution input dimension has no original axis/scalar binding")


def select_signature(module, signature, mode, *, allocation_policy=None):
    validate_mode(mode)
    if mode is None:
        if allocation_policy is not None:
            raise ValueError("execution allocation policy requires explicit execution-owned results")
        return signature
    module.verify()
    entry = signature.get("entry")
    if type(entry) is not str or not entry.isascii() or not entry.isidentifier():
        raise ValueError("executed result entry needs its original portable symbol")
    inputs, outputs = _types(module, entry)
    constraints = {
        "inputs": deepcopy(signature.get("inputs")),
        "dimension_bindings": deepcopy(signature.get("dimension_bindings", {})),
    }
    _constraints(inputs, **{"rows": constraints["inputs"], "bindings": constraints["dimension_bindings"]})
    return {
        "schema": SCHEMA,
        "entry": entry,
        "inputs": inputs,
        "outputs": outputs,
        "shape_scope": "execution-produced-result-descriptors",
        "result_abi": PROFILE,
        "input_constraints": constraints,
        "input_mutability": "borrowed-immutable-tensors",
        "allocation_policy": checked_allocation_policy(allocation_policy),
    }


def validate_signature(module, signature):
    if not isinstance(signature, dict) or signature.get("schema") != SCHEMA:
        raise ValueError("execution result invocation requires its explicit signature v6")
    constraints = signature.get("input_constraints")
    if not isinstance(constraints, dict) or set(constraints) != {"inputs", "dimension_bindings"}:
        raise ValueError("execution result signature lacks its original input constraints")
    if "allocation_policy" not in signature:
        raise ValueError("execution result signature lacks its explicit allocation policy")
    expected = select_signature(
        module, {"entry": signature.get("entry"), **constraints}, MODE, allocation_policy=signature["allocation_policy"]
    )
    if signature != expected:
        raise ValueError("execution result signature differs from the actual source and selected resource profile")
    return expected


def selected_manifest(manifest, mode):
    if mode is not None:
        manifest["result_mode"] = PROFILE
        manifest["scope"] = "compiler_only_execution_sized_results; source_closure_and_numerical_admission_unverified"
    return manifest
