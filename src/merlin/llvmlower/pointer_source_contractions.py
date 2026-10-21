"""Bind closed signed contractions to their positional pointer-source ABI.

These are source/storage names, not an execution, arithmetic, alias-disjointness
or scheduling capability. Original owners must still check current inputs,
complete emitted products, host effects, outputs and target arithmetic.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from xdsl.dialects import arith, builtin, func, tensor

from merlin.common import mlir_query as mq
from merlin.transforms.dense_ir import recognize_signed_integer_matmul


def _shape(value, element):
    typ = value.type
    if (
        type(typ) is not builtin.TensorType
        or typ.element_type != element
        or type(typ.encoding) is not builtin.NoneAttr
        or len(typ.get_shape()) != 2
        or any(d <= 0 for d in typ.get_shape())
    ):
        raise ValueError("pointer contractions require unencoded positive rank-two signed integer storage")
    return list(typ.get_shape())


def _metadata(operation, properties=()):
    if set(operation.properties) != set(properties) or any(
        not name.startswith("prov.") for name in operation.attributes
    ):
        raise ValueError("pointer contractions have unknown source properties or effects")


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _interval(value):
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(n) is not int for n in value)
        or not -128 <= value[0] <= value[1] <= 127
    ):
        raise ValueError("pointer contractions require original complete signed-i8 input intervals")
    return value


def bind_pointer_source_contractions(source: Path, command_buffer: dict, proof: dict) -> tuple[dict, ...]:
    """Return every exact dot in source order, naming outputs by return position.

    The narrow grammar is immutable i8 entries, scalar i32 zero constants,
    zero splats, signed i8/i32 rank-two dots, and distinct complete returns.
    Proof ordinals and SSA owners must cover every operation; no global plan is
    inferred. Naming, shapes or this helper cannot establish physical ownership.
    """
    source = Path(source)
    if source.is_symlink() or not source.is_file() or type(command_buffer) is not dict or type(proof) is not dict:
        raise ValueError("pointer contractions need an ordinary current source and explicit ABI/proof")
    raw = source.read_bytes()
    before = copy.deepcopy((command_buffer, proof))
    if (
        proof.get("schema") != "merlin.connected_integer_source_bounds.v1"
        or proof.get("status") != "source_arithmetic_proven"
        or hashlib.sha256(raw).hexdigest() != proof.get("interface_sha256")
    ):
        raise ValueError("pointer contractions differ from their current complete typed proof")
    module = mq.parse(raw.decode())
    module.verify()
    functions = list(mq.walk(module, "func.func"))
    if len(functions) != 1 or list(module.body.block.ops) != functions or len(functions[0].regions[0].blocks) != 1:
        raise ValueError("pointer contractions require one closed single-block source function")
    _metadata(module)
    _metadata(functions[0], ("sym_name", "function_type"))
    block = functions[0].regions[0].block
    ops = list(block.ops)
    if not block.args or not ops or type(ops[-1]) is not func.ReturnOp:
        raise ValueError("pointer contractions require immutable entries and complete returns")
    returned = tuple(ops[-1].operands)
    _metadata(ops[-1])
    if not returned or len(set(returned)) != len(returned):
        raise ValueError("pointer contractions require distinct complete returned dots")
    input_shapes = [_shape(v, builtin.i8) for v in block.args]
    output_shapes = [_shape(v, builtin.i32) for v in returned]
    if (
        not _same(proof.get("input_abi"), [dict(shape=s, bits=8) for s in input_shapes])
        or not _same(proof.get("output_abi"), [dict(shape=s, bits=32) for s in output_shapes])
        or type(proof.get("input_intervals")) is not list
        or len(proof["input_intervals"]) != len(block.args)
        or type(proof.get("operations")) is not list
        or len(proof["operations"]) != len(ops) - 1
    ):
        raise ValueError("pointer contractions lack complete source/proof signature or operation coverage")
    intervals = [_interval(v) for v in proof["input_intervals"]]
    abi, tensors = command_buffer.get("compiler_pointer_abi"), command_buffer.get("tensors")
    if (
        type(abi) is not dict
        or set(abi) != {"version", "arguments"}
        or type(abi["version"]) is not int
        or abi["version"] != 1
        or type(abi["arguments"]) is not list
        or len(abi["arguments"]) != len(block.args) + len(returned)
        or any(type(name) is not str or not name for name in abi["arguments"])
        or len(set(abi["arguments"])) != len(abi["arguments"])
        or type(tensors) is not dict
        or set(tensors) != set(abi["arguments"])
        or command_buffer.get("kernel_abi") is not None
        or command_buffer.get("params")
        or command_buffer.get("outputs") is not None
    ):
        raise ValueError("pointer contractions have no complete distinct positional pointer ABI v1")
    for i, (name, shape) in enumerate(zip(abi["arguments"], input_shapes + output_shapes, strict=True)):
        row, is_input = tensors[name], i < len(block.args)
        if (
            type(row) is not dict
            or not _same(row.get("shape"), shape)
            or row.get("dtype") != ("i8" if is_input else "i32")
            or row.get("role") != ("input" if is_input else "output")
            or set(row) - {"shape", "dtype", "role", "layout"}
            or row.get("layout", "dense_row_major") not in {"row_major", "dense_row_major"}
        ):
            raise ValueError("pointer contraction positional ABI storage differs from the typed input/return")
    names = dict(zip((*block.args, *returned), abi["arguments"], strict=True))
    owners = {value: dict(argument=i) for i, value in enumerate(block.args)}
    bounds = dict(zip(block.args, intervals, strict=True))
    constants, splats, dots, pairs = set(), set(), [], set()
    for ordinal, (op, record) in enumerate(zip(ops[:-1], proof["operations"], strict=True)):
        if len(op.results) != 1 or any(v not in owners for v in op.operands):
            raise ValueError("pointer contractions read derived, unknown or incomplete source owners")
        expected = dict(ordinal=ordinal, operation=op.name, inputs=[owners[v] for v in op.operands], interval=[0, 0])
        if type(op) is arith.ConstantOp:
            _metadata(op, ("value",))
            if (
                not isinstance(op.value, builtin.IntegerAttr)
                or op.value.type != builtin.i32
                or op.value.value.data != 0
            ):
                raise ValueError("pointer contractions require scalar i32 zero initialization")
            constants.add(op.results[0])
        elif type(op) is tensor.SplatOp:
            _metadata(op)
            _shape(op.results[0], builtin.i32)
            if len(op.operands) != 1 or op.operands[0] not in constants:
                raise ValueError("pointer contractions require an exact scalar-zero splat")
            splats.add(op.results[0])
        else:
            detached = op.clone()
            for nested in detached.walk():
                for name in tuple(nested.attributes):
                    if name.startswith("prov."):
                        del nested.attributes[name]
            try:
                recognize_signed_integer_matmul(detached, ("i8", "i8", "i32"))
            finally:
                detached.drop_all_references()
            lhs, rhs, initial = op.operands
            if lhs not in block.args or rhs not in block.args or initial not in splats or (lhs, rhs) in pairs:
                raise ValueError(
                    "pointer contractions require distinct dots of immutable entries and exact zero splats"
                )
            pairs.add((lhs, rhs))
            m, k = _shape(lhs, builtin.i8)
            rhs_k, n = _shape(rhs, builtin.i8)
            if rhs_k != k or _shape(op.results[0], builtin.i32) != [m, n]:
                raise ValueError("pointer contraction matrix extents disagree")
            products = [a * b for a in bounds[lhs] for b in bounds[rhs]]
            product = [min(products), max(products)]
            interval = [k * product[0], k * product[1]]
            prefix = [min(0, interval[0]), max(0, interval[1])]
            if prefix[0] < -(1 << 31) or prefix[1] >= 1 << 31:
                raise ValueError("pointer contraction source proof exceeds signed-i32 prefixes")
            expected.update(
                kind="multiply_accumulate",
                result_bits=32,
                input_shapes=[[m, k], [k, n]],
                output_shape=[m, n],
                scalar_operations=["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"],
                scalar_literals=[],
                reduction_extent=k,
                initial_interval=[0, 0],
                addend_interval=product,
                prefix_interval=prefix,
                product_intervals=[product],
                operand_bits=[8, 8],
                interval=interval,
            )
            dots.append((ordinal, op))
        if type(record) is not dict or type(record.get("ordinal")) is not int or not _same(record, expected):
            raise ValueError(
                "pointer contraction proof differs from complete source ordinal, SSA ownership or arithmetic"
            )
        owners[op.results[0]] = dict(operation=ordinal, result=0)
        bounds[op.results[0]] = expected["interval"]
    if not dots or {op.results[0] for _, op in dots} != set(returned):
        raise ValueError("pointer contractions must return every distinct dot exactly once")
    if any(any(use.operation.name != "tensor.splat" for use in v.uses) or not tuple(v.uses) for v in constants):
        raise ValueError("pointer contractions have unused or foreign scalar initialization effects")
    if any(any(use.operation not in [op for _, op in dots] for use in v.uses) or not tuple(v.uses) for v in splats):
        raise ValueError("pointer contractions have unused or foreign zero storage effects")
    if (
        not _same(proof.get("output_owners"), [owners[v] for v in returned])
        or not _same(proof.get("output_intervals"), [bounds[v] for v in returned])
        or source.is_symlink()
        or source.read_bytes() != raw
        or before != (command_buffer, proof)
    ):
        raise ValueError("pointer contraction complete returns or current source/ABI/proof changed")
    return tuple(
        dict(
            ordinal=i,
            lhs=names[op.operands[0]],
            rhs=names[op.operands[1]],
            output=names[op.results[0]],
            m=_shape(op.operands[0], builtin.i8)[0],
            k=_shape(op.operands[0], builtin.i8)[1],
            n=_shape(op.operands[1], builtin.i8)[1],
        )
        for i, op in dots
    )
