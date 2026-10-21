"""Bind exact signed dots and their final signed-byte saturation to source ABI.

This prospective source grammar preserves the original raw-only binder. It
recognizes complete immutable-entry dots, optionally followed by the explicit
pointwise max(-128), min(127), truncation sequence. No target schedule or
numerical execution is provided; original source/packet/numerical owners remain
responsible for current values, every product, physical aliases and readout.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from xdsl.dialects import arith, builtin, func, tensor
from xdsl.ir.affine import AffineMap

from merlin.common import mlir_query as mq
from merlin.llvmlower.pointer_source_contractions import _interval, _metadata, _same, _shape
from merlin.transforms.dense_ir import recognize_signed_integer_matmul

PROTOCOL = "merlin.pointer_source_saturation.v1"


def _pointwise(op, scalar_name, operands):
    if (
        len(op.regions) != 1
        or len(op.regions[0].blocks) != 1
        or len(op.results) != 1
        or len(op.operands) != len(operands) + 1
        or tuple(op.operands[:-1]) != tuple(operands)
        or op.operands[-1].owner.name != "tensor.empty"
        or op.operands[-1].type != op.results[0].type
        or any(_shape(v, v.type.element_type) != list(op.results[0].type.get_shape()) for v in op.operands)
    ):
        raise ValueError("native saturation requires complete same-shape pointwise storage")
    _metadata(op, ("indexing_maps", "iterator_types", "operandSegmentSizes"))
    maps = list(op.properties["indexing_maps"])
    if (
        any(m.data != AffineMap.identity(2) for m in maps)
        or len(maps) != len(op.operands)
        or [a.data.value for a in op.properties["iterator_types"]] != ["parallel", "parallel"]
        or list(op.properties["operandSegmentSizes"].get_values()) != [len(operands), 1]
    ):
        raise ValueError("native saturation has nonidentity or reduction iteration geometry")
    block = op.regions[0].block
    body = list(block.ops)
    if (
        len(block.args) != len(op.operands)
        or [v.type for v in block.args] != [v.type.element_type for v in op.operands]
        or len(body) not in {2, 3}
        or body[-1].name != "linalg.yield"
        or len(body[-2].results) != 1
        or body[-2].results[0].type != op.results[0].type.element_type
        or list(body[-1].operands) != list(body[-2].results)
    ):
        raise ValueError("native saturation scalar body differs from exact final clamp/cast")
    if len(body) == 2:
        if body[0].name != scalar_name or tuple(body[0].operands) != tuple(block.args[:-1]):
            raise ValueError("native saturation scalar body differs from exact final clamp/cast")
        _metadata(body[0])
    else:
        cmp, select, _yield = body
        if (
            scalar_name not in {"arith.maxsi", "arith.minsi"}
            or cmp.name != "arith.cmpi" or select.name != "arith.select"
            or tuple(cmp.operands) != tuple(block.args[:2]) or len(cmp.results) != 1
            or select.operands[0] is not cmp.results[0]
        ):
            raise ValueError("native saturation cmp/select lacks exact signed operand ownership")
        _metadata(cmp, ("predicate",))
        _metadata(select)
        same_order = tuple(select.operands[1:]) == tuple(block.args[:2])
        reverse_order = tuple(select.operands[1:]) == tuple(reversed(block.args[:2]))
        predicate = cmp.properties["predicate"].value.data
        expected = 4 if scalar_name == "arith.maxsi" else 2
        if not (same_order and predicate == expected or reverse_order and predicate == 6 - expected):
            raise ValueError("native saturation cmp/select changes signed min/max semantics")
    _metadata(body[-1])
    return [scalar.name for scalar in body]


def bind_pointer_source_saturation(source: Path, command_buffer: dict, proof: dict) -> tuple[dict, ...]:
    source = Path(source)
    if source.is_symlink() or not source.is_file() or type(command_buffer) is not dict or type(proof) is not dict:
        raise ValueError("native source binding requires current source, ABI and proof")
    raw = source.read_bytes()
    before = copy.deepcopy((command_buffer, proof))
    if (
        proof.get("schema") != "merlin.connected_integer_source_bounds.v1"
        or proof.get("status") != "source_arithmetic_proven"
        or hashlib.sha256(raw).hexdigest() != proof.get("interface_sha256")
    ):
        raise ValueError("native source binding differs from current complete source proof")
    module = mq.parse(raw.decode())
    module.verify()
    functions = list(mq.walk(module, "func.func"))
    if len(functions) != 1 or list(module.body.block.ops) != functions or len(functions[0].regions[0].blocks) != 1:
        raise ValueError("native source binding requires one complete closed block")
    _metadata(module)
    _metadata(functions[0], ("sym_name", "function_type"))
    block = functions[0].regions[0].block
    ops = list(block.ops)
    if not block.args or not ops or type(ops[-1]) is not func.ReturnOp:
        raise ValueError("native source binding requires entries and complete returns")
    _metadata(ops[-1])
    returned = tuple(ops[-1].operands)
    if not returned or len(set(returned)) != len(returned):
        raise ValueError("native source binding requires distinct complete returned dots")
    input_shapes = [_shape(v, builtin.i8) for v in block.args]
    if any(type(v.type) is not builtin.TensorType or type(v.type.element_type) is not builtin.IntegerType for v in returned):
        raise ValueError("native source outputs require ranked signed integer tensors")
    output_shapes = [_shape(v, v.type.element_type) for v in returned]
    output_bits = [v.type.element_type.width.data for v in returned]
    if any(bits not in {8, 32} for bits in output_bits) or (
        not _same(proof.get("input_abi"), [dict(shape=s, bits=8) for s in input_shapes])
        or not _same(proof.get("output_abi"), [dict(shape=s, bits=b) for s, b in zip(output_shapes, output_bits, strict=True)])
        or type(proof.get("input_intervals")) is not list
        or len(proof["input_intervals"]) != len(block.args)
        or type(proof.get("operations")) is not list
        or len(proof["operations"]) != len(ops) - 1
    ):
        raise ValueError("native source proof lacks complete typed ABI/operation coverage")
    intervals = [_interval(v) for v in proof["input_intervals"]]
    abi, tensors = command_buffer.get("compiler_pointer_abi"), command_buffer.get("tensors")
    if (
        type(abi) is not dict or set(abi) != {"version", "arguments"}
        or type(abi["version"]) is not int or abi["version"] != 1
        or type(abi["arguments"]) is not list
        or len(abi["arguments"]) != len(block.args) + len(returned)
        or any(type(name) is not str or not name for name in abi["arguments"])
        or len(set(abi["arguments"])) != len(abi["arguments"])
        or type(tensors) is not dict or set(tensors) != set(abi["arguments"])
        or command_buffer.get("kernel_abi") is not None or command_buffer.get("params")
        or command_buffer.get("outputs") is not None
    ):
        raise ValueError("native source binding lacks distinct complete positional ABI v1")
    shapes = input_shapes + output_shapes
    bits = [8] * len(block.args) + output_bits
    for i, (name, shape, width) in enumerate(zip(abi["arguments"], shapes, bits, strict=True)):
        row = tensors[name]
        if (
            type(row) is not dict or not _same(row.get("shape"), shape)
            or row.get("dtype") != f"i{width}"
            or row.get("role") != ("input" if i < len(block.args) else "output")
            or set(row) - {"shape", "dtype", "role", "layout"}
            or row.get("layout", "dense_row_major") not in {"row_major", "dense_row_major"}
        ):
            raise ValueError("native source ABI differs from complete typed storage")
    names = dict(zip((*block.args, *returned), abi["arguments"], strict=True))
    owners = {v: dict(argument=i) for i, v in enumerate(block.args)}
    bounds = dict(zip(block.args, intervals, strict=True))
    constants, splats, dots, stages, pairs = {}, {}, {}, {}, set()
    consumed = set()
    for ordinal, (op, record) in enumerate(zip(ops[:-1], proof["operations"], strict=True)):
        if len(op.results) != 1 or any(v not in owners for v in op.operands):
            raise ValueError("native source reads unsupported or incomplete SSA owners")
        result = op.results[0]
        expected = dict(ordinal=ordinal, operation=op.name, inputs=[owners[v] for v in op.operands])
        if type(op) is arith.ConstantOp:
            _metadata(op, ("value",))
            if not isinstance(op.value, builtin.IntegerAttr) or op.value.type != builtin.i32 or op.value.value.data not in {0, -128, 127}:
                raise ValueError("native source has unsupported clamp/initializer constants")
            constants[result] = op.value.value.data
            interval = [constants[result]] * 2
        elif type(op) is tensor.SplatOp:
            _metadata(op)
            _shape(result, builtin.i32)
            if len(op.operands) != 1 or op.operands[0] not in constants:
                raise ValueError("native source splat lacks exact signed scalar ownership")
            splats[result] = constants[op.operands[0]]
            interval = [splats[result]] * 2
        elif type(op) is tensor.EmptyOp:
            _metadata(op)
            _shape(result, result.type.element_type)
            if op.operands:
                raise ValueError("native source has dynamic empty storage")
            interval = None
        elif op.name == "linalg.generic" and len(op.operands) == 3 and op.operands[0] in block.args and op.operands[1] in block.args:
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
            if splats.get(initial) != 0 or (lhs, rhs) in pairs:
                raise ValueError("native dot lacks distinct immutable operands and exact zero seed")
            pairs.add((lhs, rhs))
            m, k = _shape(lhs, builtin.i8)
            rhs_k, n = _shape(rhs, builtin.i8)
            if rhs_k != k or _shape(result, builtin.i32) != [m, n]:
                raise ValueError("native dot geometry disagrees")
            products = [a * b for a in bounds[lhs] for b in bounds[rhs]]
            product = [min(products), max(products)]
            interval = [k * product[0], k * product[1]]
            prefix = [min(0, interval[0]), max(0, interval[1])]
            if prefix[0] < -(1 << 31) or prefix[1] >= 1 << 31:
                raise ValueError("native source dot prefixes exceed signed-i32")
            expected.update(kind="multiply_accumulate", result_bits=32,
                input_shapes=[[m, k], [k, n]], output_shape=[m, n],
                scalar_operations=["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"],
                scalar_literals=[], reduction_extent=k, initial_interval=[0, 0], addend_interval=product,
                prefix_interval=prefix, product_intervals=[product], operand_bits=[8, 8])
            dots[ordinal] = dict(ordinal=ordinal, lhs=names[lhs], rhs=names[rhs], m=m, k=k, n=n)
            stages[result] = ordinal, 0
        elif op.name == "linalg.generic" and op.operands and op.operands[0] in stages:
            previous = op.operands[0]
            dot, stage = stages[previous]
            scalar = {0: "arith.maxsi", 1: "arith.minsi", 2: "arith.trunci"}.get(stage)
            if scalar is None or len(tuple(previous.uses)) != 1:
                raise ValueError("native saturation has early/multiple readout or unsupported stage")
            operands = tuple(op.operands[:-1])
            if stage < 2:
                if len(operands) != 2 or splats.get(operands[1]) != (-128 if stage == 0 else 127):
                    raise ValueError("native saturation differs from exact signed-byte clamp")
                _shape(result, builtin.i32)
                fn = max if stage == 0 else min
                interval = [fn(v, -128 if stage == 0 else 127) for v in bounds[previous]]
            else:
                if len(operands) != 1:
                    raise ValueError("native truncation has extra source effects")
                _shape(result, builtin.i8)
                interval = bounds[previous]
            scalar_operations = _pointwise(op, scalar, operands)
            expected.update(kind="pointwise", result_bits=8 if stage == 2 else 32,
                input_shapes=[list(v.type.get_shape()) for v in operands],
                output_shape=list(result.type.get_shape()), scalar_operations=scalar_operations, scalar_literals=[])
            stages[result] = dot, stage + 1
            consumed.add(previous)
        else:
            raise ValueError("native source has effects outside exact dot and final saturation")
        expected["interval"] = interval
        if type(record) is not dict or type(record.get("ordinal")) is not int or not _same(record, expected):
            raise ValueError("native source proof differs from exact SSA/geometry/arithmetic")
        owners[result] = dict(operation=ordinal, result=0)
        bounds[result] = interval
    selected = []
    seen = set()
    for result in returned:
        if result not in stages:
            raise ValueError("native source returns an unsupported noncontraction value")
        dot, stage = stages[result]
        if stage not in {0, 3} or dot in seen or result in consumed:
            raise ValueError("native source returns incomplete/repeated dot saturation")
        seen.add(dot)
        selected.append({**dots[dot], "output": names[result], "output_dtype": "i8" if stage == 3 else "i32"})
    if seen != set(dots) or any(v not in consumed and v not in returned for v in stages):
        raise ValueError("native source drops or duplicates a dot/clamp stage")
    if any(not tuple(v.uses) for v in (*constants, *splats)):
        raise ValueError("native source contains unused scalar/splat effects")
    if (
        not _same(proof.get("output_owners"), [owners[v] for v in returned])
        or not _same(proof.get("output_intervals"), [bounds[v] for v in returned])
        or source.is_symlink() or source.read_bytes() != raw or before != (command_buffer, proof)
    ):
        raise ValueError("native source complete returns/current inputs changed")
    return tuple(selected)
