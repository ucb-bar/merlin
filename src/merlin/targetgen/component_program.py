"""Closed independent tensor programs, emitted as standard Linalg MLIR.

This generic builder describes observable tensor semantics. Logical aliases and
epoch updates are functionalized into SSA values; physical buffer aliasing,
allocation lifetime and host/device synchronization need execution witnesses.
It contains no target schedule, instruction, golden or workload information.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

SCHEMA_V2 = "merlin.component_program.v2"
SCHEMA_V3 = "merlin.component_program.v3"
LOGICAL_IO_V1 = "merlin.logical_io.v1"


@dataclass(frozen=True)
class ValueType:
    shape: tuple[int, ...]
    dtype: str

    @property
    def mlir(self):
        return "tensor<" + "x".join(str(value) for value in self.shape) + "x" + self.dtype + ">"


def _name(value):
    return isinstance(value, str) and value and value[0].isalpha() and all(c.isalnum() or c == "_" for c in value)


def _integer_dtype(value, *, operand_dtype, accumulator_dtype):
    dtype = {"operand": operand_dtype, "accumulator": accumulator_dtype}.get(value, value)
    if not isinstance(dtype, str) or not dtype.startswith("i") or not dtype[1:].isdigit() or int(dtype[1:]) < 1:
        raise ValueError("independent component DAG requires signed integer types")
    return dtype


def analyze(program, *, operand_dtype, accumulator_dtype):
    """Check a finite typed DAG and derive source effects, uses and live outputs."""
    v3 = isinstance(program, dict) and program.get("schema") == SCHEMA_V3
    v2 = isinstance(program, dict) and program.get("schema") in {SCHEMA_V2, SCHEMA_V3}
    logical_io = isinstance(program, dict) and program.get("logical_io") == LOGICAL_IO_V1
    if not isinstance(program, dict) or set(program) != {"inputs", "nodes", "outputs"} | ({"schema"} if v2 else set()) | ({"logical_io"} if logical_io else set()):
        raise ValueError("component program requires exactly inputs, nodes and outputs")
    _integer_dtype(accumulator_dtype, operand_dtype=operand_dtype, accumulator_dtype=accumulator_dtype)
    if not isinstance(program["inputs"], list) or not program["inputs"]:
        raise ValueError("component program requires explicit inputs")
    types, inputs, aliases, storage, current, uses = {}, [], {}, {}, {}, {}
    for row in program["inputs"]:
        if not isinstance(row, dict) or set(row) != {"name", "role", "shape", "dtype"}:
            raise ValueError("component input requires exactly name, role, shape and dtype")
        name, shape = row["name"], row["shape"]
        dtype = _integer_dtype(row["dtype"], operand_dtype=operand_dtype, accumulator_dtype=accumulator_dtype)
        if not _name(name) or name in types or row["role"] not in {"input", "weight", "bias"}:
            raise ValueError("component input name/role is invalid or duplicated")
        if not isinstance(shape, list) or len(shape) != 2 or any(type(dim) is not int or dim < 1 for dim in shape):
            raise ValueError("component programs require positive rank-two input shapes")
        types[name] = ValueType(tuple(shape), dtype)
        storage[name] = name
        current[name] = name
        uses[name] = 0
        inputs.append(dict(row, dtype=dtype))
    nodes, effects, versions = [], set(), {}
    if not isinstance(program["nodes"], list):
        raise ValueError("component nodes must be an explicit ordered list")
    for ordinal, node in enumerate(program["nodes"]):
        if not isinstance(node, dict) or not _name(node.get("name")) or node["name"] in types:
            raise ValueError("component node needs a fresh safe name")
        name, op, args = node["name"], node.get("op"), node.get("inputs")
        extra = (
            {"dtype"}
            if op == "cast"
            else {"axis", "dtype"}
            if op == "reduce_sum"
            else {"shape"}
            if op == "reshape"
            else {"offsets", "sizes", "strides"}
            if op == "extract_slice"
            else {"axis"}
            if op == "concat"
            else set()
        )
        supported = {"matmul", "add", "copy", "transpose", "alias", "update"} | (
            {"cast", "reduce_sum", "reshape"} if v2 else set()
        ) | ({"extract_slice", "concat"} if v3 else set())
        if set(node) != {"name", "op", "inputs"} | extra or op not in supported:
            raise ValueError("unsupported component node; use explicit supported generic tensor operations")
        arity = 2 if op in {"matmul", "add", "update"} else 1
        if (not isinstance(args, list) or (len(args) < 2 if op == "concat" else len(args) != arity)
                or any(not isinstance(arg, str) or arg not in types for arg in args)):
            raise ValueError("component node operand is absent or its arity differs")
        actual = [current[storage[arg]] for arg in args]
        selected = [types[arg] for arg in args]
        for arg in actual:
            uses[arg] = uses.get(arg, 0) + 1
        lhs = selected[0]
        if op == "matmul":
            rhs = selected[1]
            if lhs.shape[1] != rhs.shape[0]:
                raise ValueError("component matmul reduction extents differ")
            if max(int(lhs.dtype[1:]), int(rhs.dtype[1:])) > int(accumulator_dtype[1:]):
                raise ValueError("component matmul operands exceed selected accumulator width")
            result = ValueType((lhs.shape[0], rhs.shape[1]), accumulator_dtype)
        elif op in {"add", "update"}:
            if selected[1] != lhs:
                raise ValueError("component add/update requires matching shapes and dtypes")
            result = lhs
        elif op == "transpose":
            result = ValueType((lhs.shape[1], lhs.shape[0]), lhs.dtype)
        elif op == "extract_slice":
            offsets, sizes, strides = (node[key] for key in ("offsets", "sizes", "strides"))
            if (any(not isinstance(items, list) or len(items) != 2 for items in (offsets, sizes, strides))
                    or any(type(x) is not int or x < 0 for x in offsets)
                    or any(type(x) is not int or x < 1 for x in sizes + strides)
                    or any(offset + (size - 1) * stride >= extent
                           for offset, size, stride, extent in zip(offsets, sizes, strides, lhs.shape, strict=True))):
                raise ValueError("component slice requires bounded positive static rank-two geometry")
            result = ValueType(tuple(sizes), lhs.dtype)
        elif op == "concat":
            axis = node["axis"]
            if (type(axis) is not int or axis not in {0, 1}
                    or any(value.dtype != lhs.dtype or value.shape[1 - axis] != lhs.shape[1 - axis]
                           for value in selected)):
                raise ValueError("component concatenation requires one rank-two axis and matching storage")
            shape = list(lhs.shape)
            shape[axis] = sum(value.shape[axis] for value in selected)
            result = ValueType(tuple(shape), lhs.dtype)
        elif op == "reshape":
            # A row-major reinterpretation: the same elements in the same order under a new
            # rank-two extent. Moves data, computes nothing, and preserves the element count.
            shape = node["shape"]
            if not isinstance(shape, list) or len(shape) != 2 or any(type(dim) is not int or dim < 1 for dim in shape):
                raise ValueError("component reshape requires an explicit positive rank-two result shape")
            if shape[0] * shape[1] != lhs.shape[0] * lhs.shape[1]:
                raise ValueError("component reshape must preserve the element count")
            result = ValueType(tuple(shape), lhs.dtype)
        elif op in {"cast", "reduce_sum"}:
            dtype = _integer_dtype(node["dtype"], operand_dtype=operand_dtype, accumulator_dtype=accumulator_dtype)
            shape = list(lhs.shape)
            if op == "reduce_sum":
                if type(node["axis"]) is not int or node["axis"] not in {0, 1}:
                    raise ValueError("component sum requires an explicit rank-two axis 0 or 1")
                shape[node["axis"]] = 1
                if int(lhs.dtype[1:]) > int(dtype[1:]):
                    raise ValueError("component sum requires a result at least as wide as its input")
            result = ValueType(tuple(shape), dtype)
        else:
            result = lhs
        types[name] = result
        uses[name] = 0
        if op in {"alias", "update"}:
            owner = storage[args[0]]
            storage[name] = owner
            if op == "alias":
                aliases[name] = args[0]
                effects.add("alias")
            else:
                current[owner] = name
                versions[owner] = versions.get(owner, 0) + 1
                effects.add("epoch_mutation")
        else:
            storage[name] = name
            current[name] = name
        nodes.append(
            {**node, "ordinal": ordinal, "actual_inputs": actual, "shape": list(result.shape), "dtype": result.dtype}
        )
    outputs = program["outputs"]
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("component program needs at least one published output")
    output_names, out = set(), []
    for row in outputs:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "value"}
            or not _name(row["name"])
            or row["name"] in output_names
            or row["value"] not in types
        ):
            raise ValueError("component publication needs a unique safe name and a known value")
        output_names.add(row["name"])
        name = current[storage[row["value"]]]
        value_type = types[name]
        out.append({**row, "actual_value": name, "shape": list(value_type.shape), "dtype": value_type.dtype})
        if uses.get(name, 0) and name not in {input["name"] for input in inputs}:
            effects.add("escaped_use")
    if any(count > 1 for name, count in uses.items()):
        effects.add("multiple_consumers")
    if any(count > 1 for name, count in uses.items() if name not in {input["name"] for input in inputs}):
        effects.add("shared_producer")
    if any(uses[row["name"]] > 1 and row["name"] not in versions for row in inputs if row["role"] == "weight"):
        effects.add("immutable_reuse")
    effects.add("output_publication")
    result = {
        "inputs": inputs,
        "nodes": nodes,
        "outputs": out,
        "effects": sorted(effects),
        "uses": uses,
        "logical_epochs": versions,
        "logical_aliases": aliases,
        "selected_storage": {"operand": operand_dtype, "accumulator": accumulator_dtype},
        "scope": "functionalized tensor semantics; physical alias, lifetime and synchronization unverified",
    }
    if v2:
        result["schema"] = SCHEMA_V3 if v3 else SCHEMA_V2
    if logical_io:
        if output_names & {row["name"] for row in inputs}:
            raise ValueError("logical IO requires distinct input and output names")
        result["logical_io"] = LOGICAL_IO_V1
    return result


def render(program, *, operand_dtype, accumulator_dtype):
    """Emit the original generic source without corpus, data or target geometry.

    This fixed frontend construction establishes typed source semantics only.
    Numerical domains, target legality and candidate lowering remain separate.
    """
    accumulator = accumulator_dtype
    if not isinstance(accumulator, str) or not accumulator.startswith("i") or not accumulator[1:].isdigit():
        raise ValueError("component DAG requires a signed integer accumulator")
    program = analyze(program, operand_dtype=operand_dtype, accumulator_dtype=accumulator)
    types = {row["name"]: ValueType(tuple(row["shape"]), row["dtype"]) for row in program["inputs"] + program["nodes"]}
    values = {row["name"]: "%" + row["name"] for row in program["inputs"]}
    args = ", ".join(values[row["name"]] + ": " + types[row["name"]].mlir for row in program["inputs"])
    returns = ", ".join(ValueType(tuple(row["shape"]), row["dtype"]).mlir for row in program["outputs"])
    # Opt-in preserves historical source identities. These are public logical
    # names, not physical buffers, placements or target execution authority.
    attributes = ""
    if program.get("logical_io") == LOGICAL_IO_V1:
        fields = [f'merlin.logical_io = "{LOGICAL_IO_V1}"']
        fields.extend(
            f"merlin.logical_{key} = " + json.dumps([row["name"] for row in program[key]], ensure_ascii=False)
            for key in ("inputs", "outputs")
        )
        attributes = " attributes {" + ", ".join(fields) + "}"
    lines = [
        'builtin.module attributes {prov.level = "linalg-on-tensors"} {',
        f"  func.func @forward({args}) -> ({returns}){attributes} {{",
    ]
    identity = "affine_map<(d0, d1) -> (d0, d1)>"
    for node in program["nodes"]:
        name, op = node["name"], node["op"]
        # V2 reserves underscore-prefixed SSA names for builder temporaries.
        # Public source names start with a letter, so they cannot shadow a
        # scalar argument/product or another node's initialization value.
        prefix = f"_component_{node['ordinal']}_" if program.get("schema") in {SCHEMA_V2, SCHEMA_V3} else ""

        def scalar(token):
            return "%" + prefix + token

        def temporary(token):
            return scalar(token) if prefix else f"%{name}_{token}"

        actual = node["actual_inputs"]
        result = types[name]
        if op == "alias":
            values[name] = values[actual[0]]
            continue
        values[name] = "%" + name
        if op == "extract_slice":
            source = types[actual[0]]
            geometry = "".join("[" + ", ".join(map(str, node[key])) + "]"
                               for key in ("offsets", "sizes", "strides"))
            lines.append(f"    %{name} = tensor.extract_slice {values[actual[0]]}{geometry} "
                         f": {source.mlir} to {result.mlir}")
            continue
        if op == "concat":
            operands = ", ".join(values[arg] for arg in actual)
            operands_type = ", ".join(types[arg].mlir for arg in actual)
            lines.append(f"    %{name} = tensor.concat dim({node['axis']}) {operands} "
                         f": ({operands_type}) -> {result.mlir}")
            continue
        if op == "reshape":
            source = types[actual[0]]
            flat = ValueType((source.shape[0] * source.shape[1],), source.dtype)
            lines.append(
                f"    {temporary('flat')} = tensor.collapse_shape {values[actual[0]]} [[0, 1]] "
                f": {source.mlir} into {flat.mlir}"
            )
            lines.append(
                f"    %{name} = tensor.expand_shape {temporary('flat')} [[0, 1]] "
                f"output_shape [{result.shape[0]}, {result.shape[1]}] : {flat.mlir} into {result.mlir}"
            )
            continue
        lines.append(f"    {temporary('empty')} = tensor.empty() : {result.mlir}")
        if op in {"copy", "transpose"}:
            source = types[actual[0]]
            if op == "copy":
                lines.append(
                    f"    %{name} = linalg.copy ins({values[actual[0]]} : {source.mlir}) "
                    f"outs({temporary('empty')} : {result.mlir}) -> {result.mlir}"
                )
            else:
                lines.append(
                    f"    %{name} = linalg.transpose ins({values[actual[0]]} : {source.mlir}) "
                    f"outs({temporary('empty')} : {result.mlir}) permutation = [1, 0]"
                )
            continue
        if op in {"cast", "reduce_sum"}:
            source = types[actual[0]]
            reduction = op == "reduce_sum"
            axis = node.get("axis")
            inner = ValueType((source.shape[1 - axis],), result.dtype) if reduction else result
            destination = temporary("empty")
            if reduction:
                lines.append(f"    {temporary('reduced_empty')} = tensor.empty() : {inner.mlir}")
                lines.append(f"    {temporary('zero')} = arith.constant 0 : {result.dtype}")
                lines.append(
                    f"    {temporary('init')} = linalg.fill ins({temporary('zero')} : {result.dtype}) "
                    f"outs({temporary('reduced_empty')} : {inner.mlir}) -> {inner.mlir}"
                )
                destination = temporary("init")
            output_map = f"affine_map<(d0, d1) -> (d{1 - axis})>" if reduction else identity
            iterators = (
                ('["reduction", "parallel"]' if axis == 0 else '["parallel", "reduction"]')
                if reduction
                else '["parallel", "parallel"]'
            )
            output_name = temporary("reduced") if reduction else f"%{name}"
            lines.append(
                f"    {output_name} = linalg.generic {{indexing_maps = [{identity}, {output_map}], "
                f"iterator_types = {iterators}}} "
                f"ins({values[actual[0]]} : {source.mlir}) outs({destination} : {inner.mlir}) "
                f'attrs = {{prov.region_id = "component_{node["ordinal"]}", prov.op = "{op}", '
                f'prov.family = "{"reduction" if reduction else "cast"}"}} {{'
            )
            lines.append(
                f"      ^bb{node['ordinal']}({scalar('element')}: {source.dtype}, {scalar('old')}: {result.dtype}):"
            )
            element = scalar("element")
            if source.dtype != result.dtype:
                conversion = "extsi" if int(source.dtype[1:]) < int(result.dtype[1:]) else "trunci"
                lines.append(
                    f"        {scalar('converted')} = arith.{conversion} {element} : {source.dtype} to {result.dtype}"
                )
                element = scalar("converted")
            if reduction:
                lines.append(f"        {scalar('sum')} = arith.addi {scalar('old')}, {element} : {result.dtype}")
                element = scalar("sum")
            lines.extend([f"        linalg.yield {element} : {result.dtype}", f"    }} -> {inner.mlir}"])
            if reduction:
                lines.append(
                    f"    %{name} = tensor.expand_shape {output_name} [[0, 1]] "
                    f"output_shape [{result.shape[0]}, {result.shape[1]}] : {inner.mlir} into {result.mlir}"
                )
            continue
        if op == "matmul":
            maps = [
                "affine_map<(d0, d1, d2) -> (d0, d2)>",
                "affine_map<(d0, d1, d2) -> (d2, d1)>",
                "affine_map<(d0, d1, d2) -> (d0, d1)>",
            ]
            iterators = '["parallel", "parallel", "reduction"]'
            # Keep initialization symbolic: parsers may expand tensor splats
            # into one host value per element even in source-only compilation.
            lines.append(f"    {temporary('zero')} = arith.constant 0 : {result.dtype}")
            lines.append(
                f"    {temporary('init')} = linalg.fill ins({temporary('zero')} : {result.dtype}) "
                f"outs({temporary('empty')} : {result.mlir}) -> {result.mlir}"
            )
            initial = temporary("init")
            tagged_op, family = "matmul", "contraction"
        else:
            maps, iterators, initial = [identity] * 3, '["parallel", "parallel"]', temporary("empty")
            tagged_op, family = "add", "elementwise"
        lhs, rhs = (types[arg] for arg in actual)
        lines.append(
            f"    %{name} = linalg.generic {{indexing_maps = [{', '.join(maps)}], iterator_types = {iterators}}} "
            f"ins({values[actual[0]]}, {values[actual[1]]} : {lhs.mlir}, {rhs.mlir}) outs({initial} : {result.mlir}) "
            f'attrs = {{prov.region_id = "component_{node["ordinal"]}", prov.op = "{tagged_op}", '
            f'prov.family = "{family}"}} {{'
        )
        lines.append(
            f"      ^bb{node['ordinal']}({scalar('lhs')}: {lhs.dtype}, {scalar('rhs')}: {rhs.dtype}, "
            f"{scalar('old')}: {result.dtype}):"
        )
        if op == "matmul":
            operands = []
            for side, value_type in (("lhs", lhs), ("rhs", rhs)):
                if value_type.dtype != result.dtype:
                    lines.append(
                        f"        {scalar(side + '_wide')} = arith.extsi {scalar(side)} : "
                        f"{value_type.dtype} to {result.dtype}"
                    )
                    operands.append(scalar(side + "_wide"))
                else:
                    operands.append(scalar(side))
            lines.append(f"        {scalar('product')} = arith.muli {operands[0]}, {operands[1]} : {result.dtype}")
            lines.append(f"        {scalar('sum')} = arith.addi {scalar('old')}, {scalar('product')} : {result.dtype}")
        else:
            lines.append(f"        {scalar('sum')} = arith.addi {scalar('lhs')}, {scalar('rhs')} : {result.dtype}")
        lines.extend([f"        linalg.yield {scalar('sum')} : {result.dtype}", f"    }} -> {result.mlir}"])
    lines.extend(
        [
            "    func.return " + ", ".join(values[row["actual_value"]] for row in program["outputs"]) + " : " + returns,
            "  }",
            "}",
        ]
    )
    return program, "\n".join(lines) + "\n"


def build(entry, binding):
    from . import corpus_spec

    program, source = render(
        entry.get("program"),
        operand_dtype=binding.mlir_dtype(binding.operand_dtype),
        accumulator_dtype=binding.mlir_dtype(binding.accum_dtype),
    )
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "linalg_mlir": "capsule.interface.mlir",
        "inputs": program["inputs"],
        "operation": {
            "op": "component_program",
            "attributes": {
                "program": copy.deepcopy(entry["program"]),
                "out": program["outputs"][0]["name"],
                "outs": [row["name"] for row in program["outputs"]],
                "arg_order": [row["name"] for row in program["inputs"] + program["outputs"]],
            },
        },
        "component_program": program,
        "numeric_policy": corpus_spec._numeric_policy(binding, binding.accum_dtype, None),
        "expected": {
            "instruction_classes": binding.classes_for(op="matmul", output_dtype=binding.cap_dtype(binding.accum_dtype))
            if any(node["op"] == "matmul" for node in program["nodes"])
            else [],
            "modes": {},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    return cap, source
