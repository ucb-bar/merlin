"""Bind captured immutable tensors into source-preserving MLIR constants.

Compiler-staged parameters and constants enter this transform. Captured buffers
require explicit original-caller freeze consent and source-bound observations. Raw tensor
storage stays authoritative; this does not establish importer fidelity or
authorize a new precision scheme.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from io import StringIO
from math import prod

from xdsl.dialects import arith, builtin, func
from xdsl.ir import ParametrizedAttribute

from merlin.common.dense_float_ir import DenseFloatPrinter
from merlin.compiler.frontend.inputs import (
    StagedCompileInputs,
    load_compile_inputs,
    load_lifted_argument_bytes,
)
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text


@dataclass(frozen=True)
class BoundCapture:
    program: str
    record: dict
    _staged_identity: str = field(repr=False)
    _record_identity: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_record_identity", json.dumps(self.record, sort_keys=True, allow_nan=False))

    def revalidate(self, directory, staged: StagedCompileInputs | None = None):
        """Recheck unchanged transform and original compiler stage around a consumer."""
        if (
            json.dumps(self.record, sort_keys=True, allow_nan=False) != self._record_identity
            or _sha(self.program.encode()) != self.record["bound_program_sha256"]
        ):
            raise ValueError("captured immutable-binding result changed after transformation")
        current = load_compile_inputs(directory)
        if json.dumps(asdict(current), sort_keys=True, allow_nan=False, default=_bytes) != self._staged_identity:
            raise ValueError("captured compiler stage changed after immutable argument binding")
        if (
            staged is not None
            and json.dumps(asdict(staged), sort_keys=True, allow_nan=False, default=_bytes) != self._staged_identity
        ):
            raise ValueError("captured immutable binding differs from the originally selected stage")
        if "quantization_legalization" in self.record:
            from merlin.compiler.quantization.legalize import revalidate

            revalidate(self.record["quantization_legalization"])


_WIDTHS = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "i8": 1, "ui8": 1, "i16": 2, "i32": 4, "i64": 8}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _bytes(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    raise TypeError("captured stage identity contains an unsupported value")


def _symbols(attribute):
    if isinstance(attribute, builtin.SymbolRefAttr):
        return True
    if isinstance(attribute, builtin.ArrayAttr):
        return any(_symbols(value) for value in attribute)
    if isinstance(attribute, builtin.DictionaryAttr):
        return any(_symbols(value) for value in attribute.data.values())
    if isinstance(attribute, ParametrizedAttribute):
        return any(_symbols(value) for value in attribute.parameters)
    return False


def _entry(module, name):
    functions = tuple(module.body.block.ops)
    if (
        len(functions) != 1
        or type(functions[0]) is not func.FuncOp
        or functions[0].sym_name.data != name
        or len(functions[0].body.blocks) != 1
    ):
        raise ValueError("captured argument binding requires one complete entry function and one entry block")
    function = functions[0]
    if any(
        _symbols(value)
        for operation in (module, function)
        for value in (*operation.attributes.values(), *operation.properties.values())
    ):
        raise ValueError("captured binding cannot retain unresolved symbol references")
    if set(function.properties) - {"sym_name", "function_type", "sym_visibility", "arg_attrs", "res_attrs"}:
        raise ValueError("captured function has unsupported signature properties")
    for operation in function.body.walk():
        if operation.name not in {"func.return"} and operation.name.partition(".")[0] not in {
            "arith",
            "math",
            "tensor",
            "linalg",
            "scf",
            "cf",
        }:
            raise ValueError("captured binding requires tensor/arithmetic control without calls or state effects")
        if isinstance(operation, builtin.UnregisteredOp) or any(
            _symbols(value) for value in (*operation.attributes.values(), *operation.properties.values())
        ):
            raise ValueError("captured binding cannot retain unresolved operations, calls or symbol references")
    return function


def bind_lifted_arguments(
    directory, staged: StagedCompileInputs, *, max_initialization_bytes: int, freeze_readonly_buffers: bool = False
) -> BoundCapture:
    """Return transformed IR plus exact original-to-runtime/initializer bindings.

    The byte budget counts unique embedded tensors, before loading payloads.
    The original staged directory and its capture correspondence remain intact.
    Every old operation is checked structurally under only the declared argument
    substitutions; runtime ports are ordered by their recorded input positions.
    """
    if type(staged) is not StagedCompileInputs:
        raise ValueError("captured binding requires the original typed compiler stage")
    if type(max_initialization_bytes) is not int or max_initialization_bytes < 0:
        raise ValueError("captured initialization byte budget must be a nonnegative integer")
    current = load_compile_inputs(directory)
    if current != staged:
        raise ValueError("captured binding differs from the originally selected compiler stage")
    stage_identity = json.dumps(asdict(current), sort_keys=True, allow_nan=False, default=_bytes)
    # Freeze the reachable dictionaries too: a frozen dataclass alone does not
    # prevent role or input-order substitution during payload extraction.
    original = json.dumps(staged.argument_bindings, sort_keys=True, allow_nan=False)
    rows = json.loads(original)
    if type(freeze_readonly_buffers) is not bool:
        raise ValueError("freeze_readonly_buffers requires an explicit Boolean")
    if not freeze_readonly_buffers and any(row["kind"] == "buffer" for row in rows):
        raise ValueError("captured buffer binding requires an admitted immutable-state or state ABI contract")
    module = parse_mlir_text(staged.program)
    module.verify()
    legalization = None
    if freeze_readonly_buffers:
        from merlin.compiler.frontend.archives import _read_member
        from merlin.compiler.quantization.legalize import legalize

        module, legalization = legalize(module, trace=json.loads(_read_member(directory, "frontend-trace.json")))
    function = _entry(module, staged.entry)
    from merlin.compiler.frontend.state import authorize_freeze

    freeze = authorize_freeze(
        directory, staged, function, consent=freeze_readonly_buffers, max_bytes=max_initialization_bytes
    )
    # Legacy parameter-free stages have no explicit argument roster.
    if not rows:
        rows = [
            {
                "kind": "input",
                "name": f"arg{index}",
                "input_index": index,
                "shape": list(argument.type.get_shape()),
                "dtype": str(argument.type.element_type),
            }
            for index, argument in enumerate(function.args)
        ]
    if len(rows) != len(function.args):
        raise ValueError("captured bindings do not cover the complete original entry signature")
    for argument, row in zip(function.args, rows, strict=True):
        typ = argument.type
        if (
            type(typ) is not builtin.TensorType
            or type(typ.encoding) is not builtin.NoneAttr
            or not typ.has_static_shape()
            or list(typ.get_shape()) != row["shape"]
            or str(typ.element_type) != row["dtype"]
            or row["kind"] not in {"input", "param", "constant", "buffer"}
        ):
            raise ValueError("captured binding requires exact static unencoded tensor arguments")
        if row["kind"] != "input" and row["dtype"] not in _WIDTHS:
            raise ValueError(f"captured initializer has no admitted exact-storage encoding: {row['dtype']}")
    runtime = sorted(
        (index for index, row in enumerate(rows) if row["kind"] == "input"),
        key=lambda index: rows[index]["input_index"],
    )
    if [rows[index]["input_index"] for index in runtime] != list(range(len(runtime))):
        raise ValueError("captured runtime bindings need distinct complete input positions")
    payloads = load_lifted_argument_bytes(
        directory, staged, kinds=("param", "constant", "buffer"), max_bytes=max_initialization_bytes
    )
    if set(payloads) != {index for index, row in enumerate(rows) if row["kind"] != "input"}:
        raise ValueError("captured initializer payloads do not exactly cover lifted immutable arguments")
    transformed = module.clone()
    target = transformed.body.block.first_op
    block = target.body.block
    old_args, old_ops = tuple(block.args), tuple(block.ops)
    substitutions, constants, records = {}, {}, []
    for index, (argument, row) in enumerate(zip(old_args, rows, strict=True)):
        if row["kind"] == "input":
            continue
        data = payloads[index]
        if len(data) != prod(row["shape"]) * _WIDTHS[row["dtype"]]:
            raise ValueError("captured initializer payload differs from exact tensor storage extent")
        key = (row.get("file", "weights.safetensors"), row["weight"], row["dtype"], tuple(row["shape"]))
        if key not in constants:
            value = builtin.DenseIntOrFPElementsAttr(argument.type, builtin.BytesAttr(data))
            constant = arith.ConstantOp(value, argument.type)
            block.insert_op_before(constant, old_ops[0])
            constants[key] = constant
        constant = constants[key]
        argument.replace_all_uses_with(constant.result)
        substitutions[index] = constant.result
        records.append(
            {
                "argument_index": index,
                "kind": row["kind"],
                "file": key[0],
                "weight": row["weight"],
                "shape": row["shape"],
                "dtype": row["dtype"],
                "storage_bytes": len(data),
                "storage_sha256": _sha(data),
                "initializer_index": tuple(constants).index(key),
            }
        )
        if row["kind"] == "buffer":
            records[-1]["binding"] = "frozen_snapshot"
    # Build new arguments in user input order, then erase all original slots.
    for index in runtime:
        argument = block.insert_arg(old_args[index].type, len(block.args), location=old_args[index].location)
        old_args[index].replace_all_uses_with(argument)
        substitutions[index] = argument
    for argument in old_args:
        block.erase_arg(argument)
    if target.arg_attrs is not None:
        values = target.arg_attrs.data
        target.properties["arg_attrs"] = builtin.ArrayAttr(values[index] for index in runtime)
    target.update_function_type()
    context = {function.body.block: block}
    context.update((argument, substitutions[index]) for index, argument in enumerate(function.args))
    if not all(
        left.is_structurally_equivalent(right, context)
        for left, right in zip(function.body.block.ops, old_ops, strict=True)
    ):
        raise ValueError("captured argument substitution changed an original operation or dependency")
    transformed.verify()
    text = StringIO()
    DenseFloatPrinter(stream=text, print_generic_format=True).print_op(transformed)
    program = text.getvalue()
    reparsed = parse_mlir_text(program)
    reparsed.verify()
    if not transformed.is_structurally_equivalent(reparsed):
        raise ValueError("captured initializer serialization changed original operations or exact tensor storage")
    if (
        load_compile_inputs(directory) != current
        or json.dumps(staged.argument_bindings, sort_keys=True, allow_nan=False) != original
    ):
        raise ValueError("captured compiler inputs changed during immutable argument binding")
    record = {
        "schema": "merlin.captured-bindings.v1",
        "entry": staged.entry,
        "captured_source_sha256": staged.original_source_sha256,
        "staged_program_sha256": staged.program_sha256,
        "bound_program_sha256": _sha(program.encode()),
        "weights_sha256": staged.weights_sha256,
        "constants_sha256": staged.constants_sha256,
        "initializers": records,
        "runtime_inputs": [
            {
                "argument_index": index,
                "input_index": rows[index]["input_index"],
                "name": rows[index]["name"],
                "shape": rows[index]["shape"],
                "dtype": rows[index]["dtype"],
            }
            for index in runtime
        ],
        "unique_initialization_bytes": sum(len(constant.value.data.data) for constant in constants.values()),
        "initialization_byte_budget": max_initialization_bytes,
        "scope": "immutable exact-storage substitution; original operations retained; importer fidelity unverified",
    }
    if freeze is not None:
        record["buffer_freeze"] = freeze
    if legalization is not None:
        record["quantization_legalization"] = legalization
    result = BoundCapture(program, record, stage_identity)
    result.revalidate(directory, staged)
    return result
