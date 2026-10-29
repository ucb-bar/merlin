"""Closed public source cuts and exact dimension equalities for pressure probes.

Unsupported calls remain explicit. These equalities describe tensor semantics,
not physical buffers, schedules, allocation concurrency or framework aliasing.
"""

from __future__ import annotations

import copy
from math import prod

from merlin.targetgen.frontend_use_def import original_use_def_semantics

CONTRACTIONS = {"aten.linear.default", "aten.mm.default", "aten.matmul.default"}
SUPPORTED = CONTRACTIONS | {
    "aten.clone.default",
    "aten.relu.default",
    "aten.add.Tensor",
    "aten.transpose.int",
    "aten.mean.dim",
    "aten.sum.dim_IntList",
}
TYPED_SCHEMA = "merlin.public_connected_source.v2"


def contractions(program):
    if program.get("schema") == TYPED_SCHEMA:
        from .public_typed_semantics import CONTRACTIONS as typed

        return typed
    return CONTRACTIONS


def references(value):
    if isinstance(value, dict):
        if set(value) == {"node_id", "value_id"}:
            yield value["value_id"]
        else:
            for child in value.values():
                yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def cuts(trace, *, max_nodes, typed=False, shared_inputs=False, parameter_inputs=False):
    """Retain maximal connected supported calls, with every cut input explicit."""
    original_use_def_semantics(trace)
    nodes = trace["graphs"]["original"]["nodes"]
    schemas = trace["graphs"]["original"].get("operator_schemas") or {}
    values = {r["id"]: r for n in nodes for r in n["results"]}
    owners = {r["id"]: n for n in nodes for r in n["results"]}
    selected, missing = {}, []
    for node in nodes:
        if not node["op"].startswith("call"):
            continue
        try:
            if not schemas.get(node["target"]):
                raise ValueError("original operator schema identity is unavailable")
            _shape_relations(node, values, typed=typed)
        except ValueError as exc:
            missing.append({"node": node["id"], "operation": node["target"], "status": "UNKNOWN", "reason": str(exc)})
        else:
            selected[node["id"]] = node
    adjacent = {name: set() for name in selected}
    for name, node in selected.items():
        for ref in references([node["args"], node["kwargs"]]):
            parent = owners[ref]["id"]
            if parent in selected:
                adjacent[name].add(parent)
                adjacent[parent].add(name)
    if shared_inputs:
        # Preserve source-level sharing at a cut boundary. This connects the
        # complete original call graph, never a claim about physical DMA reuse.
        consumers = {}
        for name, node in selected.items():
            for ref in references([node["args"], node["kwargs"]]):
                if owners[ref]["id"] not in selected:
                    consumers.setdefault(ref, []).append(name)
        for names in consumers.values():
            for name in names[1:]:
                adjacent[names[0]].add(name)
                adjacent[name].add(names[0])
    regions = []
    while adjacent:
        start = next(iter(adjacent))
        pending, component = [start], set()
        while pending:
            current = pending.pop()
            if current not in component:
                component.add(current)
                pending.extend(adjacent[current] - component)
        for name in component:
            del adjacent[name]
        region = [n for n in nodes if n["id"] in component]
        operations = {n["target"] for n in region}
        from .public_typed_semantics import CONTRACTIONS as typed_contractions

        selected_contractions = typed_contractions if typed else CONTRACTIONS
        if len(region) > max_nodes or (
            not typed and (not operations & selected_contractions or not operations - selected_contractions)
        ):
            missing.append(
                {
                    "nodes": [n["id"] for n in region],
                    "status": "UNKNOWN",
                    "reason": "cut lacks connected non-contraction work or exceeds node budget",
                }
            )
            continue
        live_outs = {
            ref
            for outside in nodes
            if outside["id"] not in component
            for ref in references([outside["args"], outside["kwargs"]])
            if owners[ref]["id"] in component
        }
        if not live_outs:
            missing.append(
                {
                    "nodes": [n["id"] for n in region],
                    "status": "UNKNOWN",
                    "reason": "original source cut has no externally observed live-out",
                }
            )
            continue
        try:
            regions.append(
                _parameterize(
                    region, values, schemas, live_outs, typed=typed, owners=owners if parameter_inputs else None
                )
            )
        except ValueError as exc:
            missing.append({"nodes": [n["id"] for n in region], "status": "UNKNOWN", "reason": str(exc)})
    return regions, missing


def _shape_relations(node, values, *, typed=False):
    if typed:
        from .public_typed_semantics import shape_relations

        return shape_relations(node, values)
    target, args, kwargs = node["target"], node["args"], node["kwargs"]
    if target not in SUPPORTED or len(node["results"]) != 1:
        raise ValueError("original call has no implemented exact shape constructor")
    result = node["results"][0]
    refs = list(references(args))
    rows = [values[ref] for ref in refs]
    if any(
        r.get("kind") != "tensor"
        or r.get("dtype") != "float32"
        or not isinstance(r.get("shape"), list)
        or not r["shape"]
        or any(type(d) is not int or d < 1 for d in r["shape"])
        for r in [*rows, result]
    ):
        raise ValueError("constructor requires positive static original float32 tensors")
    if not refs or not isinstance(args[0], dict) or args[0].get("value_id") != refs[0]:
        raise ValueError("original ordered Tensor arguments are unavailable")
    source, output = rows[0]["shape"], result["shape"]
    relations, units = [], []

    def same(a, b, pairs):
        for i, j in pairs:
            if values[a]["shape"][i] != values[b]["shape"][j]:
                raise ValueError("original types disagree with the operation shape rule")
            relations.append(((a, i), (b, j)))

    out = result["id"]
    if target in {"aten.clone.default", "aten.relu.default", "aten.add.Tensor"}:
        allowed = (
            {"memory_format"} if target == "aten.clone.default" else {"alpha"} if target == "aten.add.Tensor" else set()
        )
        if (
            set(kwargs) - allowed
            or any(v is not None and v != 1 for v in kwargs.values())
            or len(args) != (2 if target == "aten.add.Tensor" else 1)
        ):
            raise ValueError("constructor does not implement these original call defaults")
        if output != source or any(r["shape"] != source for r in rows):
            raise ValueError("constructor requires no original broadcasting")
        for ref in refs:
            same(ref, out, enumerate(range(len(source))))
    elif target == "aten.transpose.int":
        if len(args) != 3 or kwargs or any(type(d) is not int or not -len(source) <= d < len(source) for d in args[1:]):
            raise ValueError("transpose needs exact original literal axes")
        permutation = list(range(len(source)))
        a, b = (d % len(source) for d in args[1:])
        permutation[a], permutation[b] = permutation[b], permutation[a]
        if output != [source[i] for i in permutation]:
            raise ValueError("original transpose result shape differs")
        same(refs[0], out, [(i, j) for j, i in enumerate(permutation)])
    elif target in {"aten.mean.dim", "aten.sum.dim_IntList"}:
        if len(args) not in {2, 3} or set(kwargs) - {"keepdim", "dtype"} or kwargs.get("dtype") is not None:
            raise ValueError("reduction needs exact original dimensions and unchanged dtype")
        dims = args[1]
        if (
            not isinstance(dims, list)
            or not dims
            or any(type(d) is not int or not -len(source) <= d < len(source) for d in dims)
        ):
            raise ValueError("reduction axes are unavailable")
        dims = {d % len(source) for d in dims}
        keep = args[2] if len(args) == 3 else kwargs.get("keepdim", False)
        if type(keep) is not bool or (len(args) == 3 and "keepdim" in kwargs):
            raise ValueError("reduction keepdim is ambiguous")
        expected = [1 if i in dims else d for i, d in enumerate(source) if keep or i not in dims]
        if output != expected:
            raise ValueError("original reduction result shape differs")
        j = 0
        for i in range(len(source)):
            if i in dims:
                if keep:
                    units.append((out, j))
                    j += 1
            else:
                same(refs[0], out, [(i, j)])
                j += 1
    else:
        linear = target == "aten.linear.default"
        if kwargs or len(args) not in ({2, 3} if linear else {2}) or len(rows) not in ({2, 3} if linear else {2}):
            raise ValueError("contraction ordered operands/defaults are unavailable")
        weight = rows[1]["shape"]
        if (
            len(weight) != 2
            or (not linear and len(source) != 2)
            or output != [*source[:-1], weight[0 if linear else 1]]
        ):
            raise ValueError("constructor supports linear or unbatched rank-two matmul")
        same(refs[0], refs[1], [(len(source) - 1, 1 if linear else 0)])
        same(refs[0], out, [(i, i) for i in range(len(source) - 1)])
        same(refs[1], out, [(0 if linear else 1, len(output) - 1)])
        if len(rows) == 3:
            if rows[2]["shape"] != [output[-1]]:
                raise ValueError("linear bias differs from the original output extent")
            same(refs[2], out, [(0, len(output) - 1)])
    return relations, units


def _parameterize(nodes, values, schemas, live_outs, *, typed=False, owners=None):
    produced = {n["results"][0]["id"] for n in nodes}
    leaves = list(dict.fromkeys(r for n in nodes for r in references([n["args"], n["kwargs"]]) if r not in produced))
    order = [*leaves, *[n["results"][0]["id"] for n in nodes]]
    names = {value: f"v{i}" for i, value in enumerate(order)}
    parents = {(value, i): (value, i) for value in order for i in range(len(values[value]["shape"]))}

    def root(key):
        while parents[key] != key:
            key = parents[key]
        return key

    units = set()
    for node in nodes:
        relations, fixed = _shape_relations(node, values, typed=typed)
        units.update(fixed)
        for a, b in relations:
            parents[root(b)] = root(a)
    # Preserve source singleton axes (including documented batch-one bindings)
    # and keepdim results. Only nonunit source dimensions become probe knobs.
    fixed_roots = {root(key) for key in units} | {
        root((value, i)) for value in order for i, extent in enumerate(values[value]["shape"]) if extent == 1
    }
    dimensions = {}

    def shape(value):
        result = []
        for i in range(len(values[value]["shape"])):
            key = root((value, i))
            if key in fixed_roots:
                result.append(1)
            else:
                dimensions.setdefault(key, f"D{len(dimensions)}")
                result.append(dimensions[key])
        return result

    def argument(value):
        if isinstance(value, dict):
            if set(value) == {"node_id", "value_id"}:
                return {"value": names[value["value_id"]]}
            if typed:
                from .public_typed_semantics import literal

                literal(value)
                return copy.deepcopy(value)
            raise ValueError("constructor does not implement opaque original literal objects")
        if isinstance(value, list):
            return [argument(v) for v in value]
        if value is not None and type(value) not in {int, float, bool, str}:
            raise ValueError("constructor does not implement this original literal")
        return value

    inputs = [{"name": names[v], "shape": shape(v), "dtype": values[v]["dtype"]} for v in leaves]
    if owners is not None:
        from .public_boundary_parameters import source_role

        for row, value in zip(inputs, leaves, strict=True):
            if role := source_role(owners[value]):
                row["source_parameter"] = role
    if typed:
        from .public_typed_semantics import layout_order

        for row, value in zip(inputs, leaves, strict=True):
            order = layout_order(values[value])
            if order is not None:
                row["layout_order"] = order
    calls = [
        {
            "name": names[n["results"][0]["id"]],
            "target": n["target"],
            "args": argument(n["args"]),
            "kwargs": {k: argument(v) for k, v in n["kwargs"].items()},
            "shape": shape(n["results"][0]["id"]),
            "dtype": n["results"][0]["dtype"],
            "operator_schema": schemas[n["target"]],
        }
        for n in nodes
    ]
    return {
        **({"schema": TYPED_SCHEMA} if typed else {}),
        "inputs": inputs,
        "nodes": calls,
        "outputs": [names[n["results"][0]["id"]] for n in nodes if n["results"][0]["id"] in live_outs],
        "dimensions": list(dimensions.values()),
        "original_nodes": [n["id"] for n in nodes],
    }


def instantiate(program, dimensions):
    result = copy.deepcopy(program)
    for row in [*result["inputs"], *result["nodes"]]:
        row["shape"] = [dimensions[d] if isinstance(d, str) else d for d in row["shape"]]
        if any(type(d) is not int or d < 1 for d in row["shape"]):
            raise ValueError("public source shape binding is not positive")
    result["dimension_values"] = dict(dimensions)
    return result


def logical_elements(program):
    return sum(prod(row["shape"]) for row in [*program["inputs"], *program["nodes"]])
