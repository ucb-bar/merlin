"""Fail-closed NumPy execution for structurally proven host regions.

Provenance names are hints, not executable semantics.  This lane admits a
region only after checking its linalg iteration space, affine operand maps,
scalar body, and tensor dtypes.  In particular, ``select`` provenance may mean
either ``aten.where`` or an index/slice operation; only the former has the
pointwise scalar signature implemented here.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import MutableMapping

import numpy as np
from xdsl.dialects.builtin import TensorType
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

from .frontend import _str_attr
from .full_graph import _tensor_dtype, _tensor_shape


_SEMANTIC_ROOTS = {
    "dtype_cast": frozenset({"arith.extf", "arith.truncf", "arith.sitofp", "arith.fptosi"}),
    "add": frozenset({"arith.addf", "arith.addi"}),
    "sub": frozenset({"arith.subf", "arith.subi"}),
    "mul": frozenset({"arith.mulf", "arith.muli"}),
    "div": frozenset({"arith.divf"}),
    "compare": frozenset({"arith.cmpf", "arith.cmpi"}),
    "select": frozenset({"arith.select"}),
}
_COMPOSITE_PATTERNS = {
    "pow": frozenset({("arith.constant", "math.powf")}),
    "rsqrt": frozenset({("math.rsqrt",)}),
    "sin": frozenset({("math.sin",)}),
    "cos": frozenset({("math.cos",)}),
    "sigmoid": frozenset({(
        "arith.constant", "arith.negf", "math.exp", "arith.addf", "arith.divf",
    )}),
    "gelu": frozenset({(
        "arith.constant", "arith.constant", "arith.constant", "arith.mulf",
        "math.erf", "arith.addf", "arith.mulf", "arith.mulf",
    )}),
    "elementwise": frozenset({
        ("arith.sitofp", "arith.constant", "arith.divf"),
        ("arith.constant", "arith.divf"),
    }),
    "minmax": frozenset({("arith.constant", "arith.minimumf")}),
}
_CONSTRUCTOR_PATTERNS = {
    "arange": frozenset({
        (
            "linalg.index", "arith.index_cast", "arith.sitofp",
            "arith.constant", "arith.mulf", "arith.constant", "arith.addf",
        ),
        (
            "linalg.index", "arith.index_cast", "arith.constant",
            "arith.muli", "arith.constant", "arith.addi",
        ),
    }),
}
_CAST_OPS = frozenset({
    "arith.extf", "arith.truncf", "arith.sitofp", "arith.fptosi",
    "arith.index_cast",
})
_SCALAR_OPS = frozenset().union(*_SEMANTIC_ROOTS.values(), {
    "arith.constant", "arith.index_cast", "arith.negf", "arith.minimumf",
    "linalg.index", "math.cos", "math.erf", "math.exp", "math.powf",
    "math.rsqrt", "math.sin",
})
_REGION_SCAFFOLD = frozenset({"arith.constant", "tensor.splat", "tensor.empty", "linalg.generic"})


class UnsupportedHostRegion(ValueError):
    """A region has no exact implementation in this deliberately small lane."""


def _dtype_name(value) -> str:
    dtype = _tensor_dtype(value)
    if dtype is not None:
        return dtype
    if isinstance(value.type, TensorType):
        return str(value.type.element_type)
    return str(value.type)


def _numpy_dtype(dtype: str):
    try:
        return {
            "i1": np.bool_, "i8": np.int8, "i16": np.int16,
            "i32": np.int32, "i64": np.int64, "index": np.int64,
            "f16": np.float16, "f32": np.float32, "f64": np.float64,
            # NumPy has no native BF16.  Values are kept as rounded f32.
            "bf16": np.float32,
        }[dtype]
    except KeyError as error:
        raise UnsupportedHostRegion(f"unsupported host dtype {dtype}") from error


def _bf16_rne(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _cast(value: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return _bf16_rne(value)
    if dtype == "i1":
        # The captured semantic is a tensor dtype conversion to bool.  Use the
        # source-language truth conversion rather than depending on NumPy's
        # integer-width casting details.
        return np.not_equal(value, 0)
    return np.asarray(value).astype(_numpy_dtype(dtype), casting="unsafe", copy=False)


def _constant_value(op):
    value = op.properties.get("value")
    raw = getattr(getattr(value, "value", None), "data", None)
    if raw is None:
        raise UnsupportedHostRegion("arith.constant has no scalar numeric value")
    return np.asarray(raw, dtype=_numpy_dtype(str(op.results[0].type)))


def _constant_signature(op) -> dict:
    """Retain the exact printed attribute, including infinities and signed zeros."""
    _constant_value(op)
    return {
        "result_type": str(op.results[0].type),
        "attribute": str(op.properties["value"]),
    }


def _index_dimension(op) -> int:
    value = op.properties.get("dim")
    result = getattr(getattr(value, "value", None), "data", None)
    if result is None:
        raise UnsupportedHostRegion("linalg.index has no dimension")
    return int(result)


def _affine_map_signature(mapping, input_shape: tuple[int, ...], output_shape: tuple[int, ...]):
    affine = mapping.data
    if affine.num_symbols != 0 or affine.num_dims != len(output_shape):
        raise UnsupportedHostRegion("affine operand map has symbols or the wrong loop rank")
    if len(affine.results) != len(input_shape):
        raise UnsupportedHostRegion("affine operand map rank differs from its tensor")
    seen = set()
    result = []
    for axis, (expr, extent) in enumerate(zip(affine.results, input_shape)):
        if isinstance(expr, AffineDimExpr):
            dim = int(expr.position)
            if dim in seen or dim >= len(output_shape):
                raise UnsupportedHostRegion("affine operand map repeats an output dimension")
            if int(extent) != int(output_shape[dim]):
                raise UnsupportedHostRegion("affine operand extent does not match its output dimension")
            seen.add(dim)
            result.append({"kind": "dim", "position": dim})
        elif isinstance(expr, AffineConstantExpr) and int(expr.value) == 0:
            if int(extent) < 1:
                raise UnsupportedHostRegion("zero-index broadcast has an empty input extent")
            result.append({"kind": "constant", "value": 0})
        else:
            raise UnsupportedHostRegion(
                f"operand axis {axis} is not a dimension or constant-zero broadcast"
            )
    return result


def _broadcast_operand(value: np.ndarray, mapping: list[dict], output_shape: tuple[int, ...]):
    array = np.asarray(value)
    if array.ndim != len(mapping):
        raise ValueError(f"runtime input rank {array.ndim} differs from extracted map rank {len(mapping)}")
    index = tuple(0 if item["kind"] == "constant" else slice(None) for item in mapping)
    reduced = array[index]
    mapped_dims = [item["position"] for item in mapping if item["kind"] == "dim"]
    if mapped_dims:
        order = np.argsort(mapped_dims)
        reduced = np.transpose(reduced, axes=tuple(int(v) for v in order))
    reshape = [1] * len(output_shape)
    for axis, dim in enumerate(sorted(mapped_dims)):
        reshape[dim] = reduced.shape[axis]
    return np.broadcast_to(reduced.reshape(reshape), output_shape)


def _integer_predicate(lhs, rhs, predicate: int):
    if predicate >= 6:
        if lhs.dtype.kind != "u":
            lhs = lhs.view(np.dtype(f"u{lhs.dtype.itemsize}"))
            rhs = rhs.view(np.dtype(f"u{rhs.dtype.itemsize}"))
        predicate -= 4
    operations = {
        0: np.equal, 1: np.not_equal, 2: np.less, 3: np.less_equal,
        4: np.greater, 5: np.greater_equal,
    }
    if predicate not in operations:
        raise UnsupportedHostRegion(f"unsupported integer comparison predicate {predicate}")
    return operations[predicate](lhs, rhs)


def _float_predicate(lhs, rhs, predicate: int):
    unordered = np.isnan(lhs) | np.isnan(rhs)
    ordered = ~unordered
    table = {
        0: np.zeros_like(ordered),
        1: ordered & np.equal(lhs, rhs), 2: ordered & np.greater(lhs, rhs),
        3: ordered & np.greater_equal(lhs, rhs), 4: ordered & np.less(lhs, rhs),
        5: ordered & np.less_equal(lhs, rhs), 6: ordered & np.not_equal(lhs, rhs),
        7: ordered,
        8: unordered | np.equal(lhs, rhs), 9: unordered | np.greater(lhs, rhs),
        10: unordered | np.greater_equal(lhs, rhs), 11: unordered | np.less(lhs, rhs),
        12: unordered | np.less_equal(lhs, rhs), 13: unordered | np.not_equal(lhs, rhs),
        14: unordered, 15: np.ones_like(ordered),
    }
    if predicate not in table:
        raise UnsupportedHostRegion(f"unsupported float comparison predicate {predicate}")
    return table[predicate]


def _predicate(op) -> int:
    value = op.properties.get("predicate")
    result = getattr(getattr(value, "value", None), "data", None)
    if result is None:
        raise UnsupportedHostRegion("comparison predicate is missing")
    return int(result)


def _execute_scalar(op, arguments: list[np.ndarray]):
    name = op.name
    if name == "arith.constant":
        return _constant_value(op)
    if name in _CAST_OPS:
        return _cast(arguments[0], str(op.results[0].type))
    if name in {"arith.addf", "arith.addi"}:
        result = np.add(arguments[0], arguments[1])
    elif name in {"arith.subf", "arith.subi"}:
        result = np.subtract(arguments[0], arguments[1])
    elif name in {"arith.mulf", "arith.muli"}:
        result = np.multiply(arguments[0], arguments[1])
    elif name == "arith.divf":
        result = np.divide(arguments[0], arguments[1])
    elif name == "arith.negf":
        result = np.negative(arguments[0])
    elif name == "arith.minimumf":
        result = np.minimum(arguments[0], arguments[1])
    elif name == "math.sin":
        result = np.sin(arguments[0])
    elif name == "math.cos":
        result = np.cos(arguments[0])
    elif name == "math.exp":
        result = np.exp(arguments[0])
    elif name == "math.powf":
        result = np.power(arguments[0], arguments[1])
    elif name == "math.rsqrt":
        result = np.reciprocal(np.sqrt(arguments[0]))
    elif name == "math.erf":
        source = np.asarray(arguments[0])
        result = np.fromiter(
            (math.erf(float(value)) for value in source.flat),
            dtype=np.float64,
            count=source.size,
        ).reshape(source.shape)
    elif name == "arith.cmpi":
        return _integer_predicate(arguments[0], arguments[1], _predicate(op))
    elif name == "arith.cmpf":
        return _float_predicate(arguments[0], arguments[1], _predicate(op))
    elif name == "arith.select":
        return np.where(arguments[0], arguments[1], arguments[2])
    else:
        raise UnsupportedHostRegion(f"unsupported scalar operation {name}")
    return _cast(result, str(op.results[0].type))


@dataclass(frozen=True)
class HostRegionProgram:
    region_id: str
    semantic: str
    aten: str
    operations: tuple
    generic: object | None
    input_values: tuple
    output_value: object
    signature: dict


def _extract_program(region_id: str, operations: tuple) -> HostRegionProgram:
    semantic = _str_attr(operations[0], "prov.op")
    aten = _str_attr(operations[0], "prov.aten")
    if semantic == "fill":
        if tuple(op.name for op in operations) != ("arith.constant", "tensor.splat"):
            raise UnsupportedHostRegion("fill is not one scalar constant followed by tensor.splat")
        constant, splat = operations
        if tuple(splat.operands) != (constant.results[0],):
            raise UnsupportedHostRegion("fill splat does not consume its captured constant")
        output_shape = _tensor_shape(splat.results[0])
        if output_shape is None or any(extent < 0 for extent in output_shape):
            raise UnsupportedHostRegion("fill output is not a static ranked tensor")
        output_dtype = _dtype_name(splat.results[0])
        _numpy_dtype(output_dtype)
        constant_record = _constant_signature(constant)
        signature = {
            "schema": "atlas_host_constructor_signature_v1",
            "semantic": semantic,
            "aten": aten,
            "input_shapes": [],
            "input_dtypes": [],
            "output_shape": list(output_shape),
            "output_dtype": output_dtype,
            "operand_maps": [],
            "scalar_ops": ["arith.constant", "tensor.splat"],
            "scalar_constants": [constant_record],
            "materialization_rule": "exact scalar constant splatted to the static output shape",
        }
        return HostRegionProgram(
            region_id, semantic, aten, operations, None, (), splat.results[0], signature
        )
    if (semantic not in _SEMANTIC_ROOTS and semantic not in _COMPOSITE_PATTERNS
            and semantic not in _CONSTRUCTOR_PATTERNS):
        raise UnsupportedHostRegion(f"semantic {semantic} has no host pointwise implementation")
    if any(op.name not in _REGION_SCAFFOLD for op in operations):
        raise UnsupportedHostRegion("region contains non-pointwise scaffold operations")
    generics = [op for op in operations if op.name == "linalg.generic"]
    if len(generics) != 1 or operations[-1] is not generics[0]:
        raise UnsupportedHostRegion("region is not one terminal linalg.generic")
    generic = generics[0]
    allowed_arities = {0} if semantic == "arange" else {1, 2, 3}
    if (len(generic.inputs) not in allowed_arities or len(generic.outputs) != 1
            or len(generic.results) != 1):
        raise UnsupportedHostRegion("generic operand/result arity is unsupported")
    output_shape = _tensor_shape(generic.results[0])
    if output_shape is None or any(extent < 0 for extent in output_shape):
        raise UnsupportedHostRegion("host pointwise output is not a static ranked tensor")
    iterator_types = [getattr(item.data, "value", str(item.data)) for item in generic.iterator_types]
    if iterator_types != ["parallel"] * len(output_shape):
        raise UnsupportedHostRegion("host pointwise region has non-parallel iterators")
    maps = list(generic.indexing_maps)
    if len(maps) != len(generic.inputs) + 1:
        raise UnsupportedHostRegion("generic indexing-map arity is inconsistent")
    output_map = _affine_map_signature(maps[-1], output_shape, output_shape)
    if output_map != [{"kind": "dim", "position": i} for i in range(len(output_shape))]:
        raise UnsupportedHostRegion("generic output map is not identity")
    input_maps = []
    for value, mapping in zip(generic.inputs, maps[:-1]):
        shape = _tensor_shape(value)
        if shape is None:
            raise UnsupportedHostRegion("generic input is not a ranked tensor")
        _numpy_dtype(_dtype_name(value))
        input_maps.append(_affine_map_signature(mapping, shape, output_shape))
    _numpy_dtype(_dtype_name(generic.results[0]))

    block = generic.body.blocks[0]
    scalar_ops = list(block.ops)
    if not scalar_ops or scalar_ops[-1].name != "linalg.yield":
        raise UnsupportedHostRegion("generic scalar body has no terminal yield")
    compute = scalar_ops[:-1]
    if not compute or any(op.name not in _SCALAR_OPS for op in compute):
        raise UnsupportedHostRegion("generic scalar body contains an unsupported operation")
    root = getattr(scalar_ops[-1].operands[0], "owner", None)
    if root is not compute[-1]:
        raise UnsupportedHostRegion("scalar yield does not consume the terminal operation")
    names = tuple(op.name for op in compute)
    if semantic in _SEMANTIC_ROOTS:
        if root.name not in _SEMANTIC_ROOTS[semantic]:
            raise UnsupportedHostRegion("scalar root does not implement the declared semantic")
        if any(op.name not in _CAST_OPS for op in compute[:-1]):
            raise UnsupportedHostRegion("only widening/narrowing casts may precede the scalar root")
    elif semantic in _COMPOSITE_PATTERNS and names not in _COMPOSITE_PATTERNS[semantic]:
        raise UnsupportedHostRegion("scalar DAG does not match a captured semantic pattern")
    elif semantic == "arange":
        if names not in _CONSTRUCTOR_PATTERNS[semantic]:
            raise UnsupportedHostRegion("scalar DAG does not match a captured constructor pattern")
        indices = [op for op in compute if op.name == "linalg.index"]
        if len(indices) != 1 or _index_dimension(indices[0]) != 0 or len(output_shape) != 1:
            raise UnsupportedHostRegion("arange must use dimension zero of a rank-one output")
    # The output accumulator block argument must not influence a pure pointwise result.
    allowed_values = set(block.args[:len(generic.inputs)])
    for op in compute:
        if any(value not in allowed_values for value in op.operands):
            raise UnsupportedHostRegion("scalar body reads its output initializer or an unknown value")
        allowed_values.update(op.results)

    signature = {
        "schema": (
            "atlas_host_constructor_signature_v1" if semantic == "arange"
            else "atlas_host_pointwise_signature_v1"
        ),
        "semantic": semantic,
        "aten": aten,
        "input_shapes": [list(_tensor_shape(value) or ()) for value in generic.inputs],
        "input_dtypes": [_dtype_name(value) for value in generic.inputs],
        "output_shape": list(output_shape),
        "output_dtype": _dtype_name(generic.results[0]),
        "operand_maps": input_maps,
        "scalar_ops": [op.name for op in compute],
        "scalar_constants": [
            _constant_signature(op) for op in compute if op.name == "arith.constant"
        ],
        "comparison_predicate": (
            _predicate(root) if root.name in {"arith.cmpi", "arith.cmpf"} else None
        ),
        "broadcast_rule": "affine dimensions plus exact constant-zero axes",
    }
    if semantic == "arange":
        signature["materialization_rule"] = (
            "rank-one dimension-zero index, cast as captured, multiplied by exact "
            "step constant, then added to exact start constant"
        )
    return HostRegionProgram(
        region_id, semantic, aten, operations, generic,
        tuple(generic.inputs), generic.results[0], signature,
    )


class HostSemanticLane:
    """Extract and execute only host regions whose complete signature is supported."""

    def __init__(self, workload):
        funcs = [op for op in workload.module.walk() if op.name == "func.func"]
        if len(funcs) != 1:
            raise ValueError(f"expected one func.func, found {len(funcs)}")
        self.block = funcs[0].body.blocks[0]
        grouped: OrderedDict[str, list] = OrderedDict()
        for op in self.block.ops:
            region_id = _str_attr(op, "prov.region_id")
            if region_id:
                grouped.setdefault(region_id, []).append(op)
        self.programs: OrderedDict[str, HostRegionProgram] = OrderedDict()
        self.rejections: OrderedDict[str, str] = OrderedDict()
        for region_id, operations in grouped.items():
            try:
                self.programs[region_id] = _extract_program(region_id, tuple(operations))
            except UnsupportedHostRegion as error:
                self.rejections[region_id] = str(error)

    def signature_for(self, region_id: str) -> dict | None:
        program = self.programs.get(region_id)
        return program.signature if program is not None else None

    def execute(self, region_id: str, values: MutableMapping) -> np.ndarray:
        if region_id not in self.programs:
            raise UnsupportedHostRegion(self.rejections.get(region_id, f"unknown region {region_id}"))
        program = self.programs[region_id]
        for op in program.operations:
            if op.name == "arith.constant":
                values[op.results[0]] = _constant_value(op)
            elif op.name == "tensor.splat":
                shape = _tensor_shape(op.results[0])
                if shape is None or op.operands[0] not in values:
                    raise ValueError("tensor.splat input or shape is unavailable")
                dtype = _dtype_name(op.results[0])
                values[op.results[0]] = _cast(
                    np.full(shape, values[op.operands[0]], dtype=_numpy_dtype(dtype)), dtype
                )
            elif op.name == "tensor.empty":
                # The output initializer is intentionally unread by qualified
                # bodies, but retain a correctly typed value for block mapping.
                shape = _tensor_shape(op.results[0])
                values[op.results[0]] = np.empty(shape, dtype=_numpy_dtype(_dtype_name(op.results[0])))
            elif op.name == "linalg.generic":
                output_shape = tuple(program.signature["output_shape"])
                block = op.body.blocks[0]
                scalar_values = {}
                for argument, operand, mapping in zip(block.args, op.inputs,
                                                       program.signature["operand_maps"]):
                    if operand not in values:
                        raise ValueError(f"missing runtime value for {region_id} input")
                    scalar_values[argument] = _broadcast_operand(values[operand], mapping, output_shape)
                output_init = values.get(op.outputs[0])
                if output_init is None:
                    # tensor.empty is intentionally unannotated in much of the
                    # capture and therefore is not grouped into the semantic
                    # region.  Qualified scalar bodies cannot read this block
                    # argument, so a typed placeholder is sufficient.
                    output_init = np.empty(
                        output_shape, dtype=_numpy_dtype(program.signature["output_dtype"])
                    )
                scalar_values[block.args[len(op.inputs)]] = output_init
                for scalar_op in block.ops:
                    if scalar_op.name == "linalg.yield":
                        result = scalar_values[scalar_op.operands[0]]
                        result = _cast(result, program.signature["output_dtype"])
                        values[op.results[0]] = np.ascontiguousarray(result)
                    else:
                        if scalar_op.name == "linalg.index":
                            dimension = _index_dimension(scalar_op)
                            scalar_values[scalar_op.results[0]] = np.indices(
                                output_shape, dtype=np.int64
                            )[dimension]
                        else:
                            scalar_values[scalar_op.results[0]] = _execute_scalar(
                                scalar_op, [scalar_values[value] for value in scalar_op.operands]
                            )
        return values[program.output_value]

    def seed_external_values(self, region_ids: list[str]) -> dict:
        """Create deterministic fresh inputs for a real captured region chain."""
        selected_ops = {
            op for region_id in region_ids for op in self.programs[region_id].operations
        }
        values = {}
        seed_index = 0
        for region_id in region_ids:
            program = self.programs[region_id]
            for value in program.input_values:
                if getattr(value, "owner", None) in selected_ops or value in values:
                    continue
                shape = _tensor_shape(value)
                dtype = _dtype_name(value)
                if shape is None:
                    raise ValueError("external pointwise input is not a ranked tensor")
                count = int(np.prod(shape, dtype=np.int64))
                if dtype == "i1":
                    array = ((np.arange(count, dtype=np.int64) + seed_index) % 3) != 0
                elif dtype.startswith("i") or dtype == "index":
                    array = (
                        (np.arange(count, dtype=np.int64) * (seed_index + 1) + seed_index)
                        % 17
                    ) + 1
                else:
                    array = (
                        (np.arange(count, dtype=np.float32) * (seed_index + 1) + seed_index)
                        % 29 + 3
                    ) / np.float32(11)
                values[value] = _cast(array.reshape(shape), dtype)
                seed_index += 1
        return values

    def discover_contiguous_runs(self, *, max_external_elements: int = 1_000_000) -> list[dict]:
        """Return capture-order supported runs carrying at least one SSA edge."""
        ordered = []
        seen = set()
        for op in self.block.ops:
            region_id = _str_attr(op, "prov.region_id")
            if region_id and region_id not in seen:
                seen.add(region_id)
                ordered.append(region_id)
        candidates = []
        start = 0
        while start < len(ordered):
            if ordered[start] not in self.programs:
                start += 1
                continue
            stop = start
            while stop < len(ordered) and ordered[stop] in self.programs:
                stop += 1
            run = ordered[start:stop]
            selected_ops = {
                op for region_id in run for op in self.programs[region_id].operations
            }
            produced = set()
            dependencies = 0
            external = set()
            for region_id in run:
                program = self.programs[region_id]
                for value in program.input_values:
                    if value in produced:
                        dependencies += 1
                    elif getattr(value, "owner", None) not in selected_ops:
                        external.add(value)
                produced.update(value for op in program.operations for value in op.results)
            elements = sum(int(np.prod(_tensor_shape(value) or (), dtype=np.int64))
                           for value in external)
            if dependencies and elements <= max_external_elements:
                candidates.append({
                    "region_ids": run,
                    "region_count": len(run),
                    "dependency_edges": dependencies,
                    "external_elements": elements,
                    "semantics": [self.programs[region_id].semantic for region_id in run],
                })
            start = stop
        if not candidates:
            raise ValueError("capture has no bounded contiguous supported host chain")
        return sorted(candidates, key=lambda row: (
            -row["region_count"], -row["dependency_edges"], row["external_elements"],
            row["region_ids"],
        ))

    def discover_contiguous_chain(self, *, max_external_elements: int = 1_000_000) -> list[str]:
        """Choose the longest capture-order run with a true SSA dependency."""
        return self.discover_contiguous_runs(
            max_external_elements=max_external_elements
        )[0]["region_ids"]


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def signature_sha256(signature: dict) -> str:
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
