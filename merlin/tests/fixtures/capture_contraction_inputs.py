"""Independent small original graphs and actual typed reduction inputs; no captured data."""

import hashlib
import json

import yaml

from merlin.capture import contraction_formats as F
from merlin.common import mlir_query as Q
from merlin.common.ir_lock import IR_LOCK
from merlin.common.pinned_files import PinnedFile


def _pin(path):
    return PinnedFile(path.resolve(), hashlib.sha256(path.read_bytes()).hexdigest())


def _save(path, doc):
    path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    return _pin(path)


def _graph(geometries):
    nodes = []
    for index, (m, n, k) in enumerate(geometries):
        ids = [f"a{index}", f"b{index}", f"mm{index}"]
        for identity, shape in zip(ids, ([m, k], [k, n], [m, n]), strict=True):
            nodes.append(
                {
                    "id": identity,
                    "graph_id": "g:original:root",
                    "ordinal": len(nodes),
                    "op": "call_function" if identity == ids[2] else "placeholder",
                    "target": "aten.mm.default" if identity == ids[2] else identity,
                    "args": [{"node_id": value, "value_id": value + ":v0"} for value in ids[:2]]
                    if identity == ids[2]
                    else [],
                    "kwargs": {},
                    "results": [{"id": identity + ":v0", "kind": "tensor", "dtype": "float32", "shape": shape}],
                }
            )
    nodes.append(
        {
            "id": "output",
            "op": "output",
            "graph_id": "g:original:root",
            "ordinal": len(nodes),
            "args": [{"node_id": f"mm{index}", "value_id": f"mm{index}:v0"} for index in range(len(geometries))],
            "results": [],
        }
    )
    doc = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "counting_unit": "static_captured_call_sites",
        "nodes": nodes,
        "call_count": len(geometries),
        "by_target": {"aten.mm.default": len(geometries)},
    }
    doc["sha256"] = F._digest(doc)
    return doc


def _module(geometries, formats):
    bodies, arguments, results = [], [], []
    for index, ((m, n, k), form) in enumerate(zip(geometries, formats, strict=True)):
        dtype = "i8" if form == "integer" else {"float": "f32", "bf16": "bf16", "f16": "f16"}[form]
        out = "i32" if form == "integer" else dtype
        arguments.extend([f"%a{index}: tensor<{m}x{k}x{dtype}>", f"%b{index}: tensor<{k}x{n}x{dtype}>"])
        results.append(f"tensor<{m}x{n}x{out}>")
        scalar = (
            """%ex = "arith.extsi"(%x) : (i8) -> i32
          %ey = "arith.extsi"(%y) : (i8) -> i32
          %mul = "arith.muli"(%ex, %ey) : (i32, i32) -> i32
          %add = "arith.addi"(%acc, %mul) : (i32, i32) -> i32"""
            if form == "integer"
            else f"""%mul = "arith.mulf"(%x, %y) : ({dtype}, {dtype}) -> {dtype}
          %add = "arith.addf"(%acc, %mul) : ({dtype}, {dtype}) -> {dtype}"""
        )
        bodies.append(f"""%z{index} = tensor.empty() : tensor<{m}x{n}x{out}>
          %r{index} = "linalg.generic"(%a{index}, %b{index}, %z{index}) <{{
          indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>,
                          affine_map<(d0,d1,d2)->(d0,d1)>],
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>,
                            #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}}> ({{
          ^bb0(%x: {dtype}, %y: {dtype}, %acc: {out}):
            {scalar}
            "linalg.yield"(%add) : ({out}) -> ()
          }}) {{prov.op = "{"int_matmul" if form == "integer" else "matmul"}",
                prov.origin_node_ids = ["mm{index}"], prov.source_node_ids = ["p{index}"]}} :
          (tensor<{m}x{k}x{dtype}>, tensor<{k}x{n}x{dtype}>, tensor<{m}x{n}x{out}>) -> tensor<{m}x{n}x{out}>""")
    return (
        "builtin.module { func.func @forward("
        + ", ".join(arguments)
        + ") -> ("
        + ", ".join(results)
        + ") {\n"
        + "\n".join(bodies)
        + "\nfunc.return "
        + ", ".join(f"%r{i}" for i in range(len(results)))
        + " : "
        + ", ".join(results)
        + "\n} }"
    )


def _trace(graph, text):
    from xdsl.dialects.builtin import ArrayAttr

    with IR_LOCK:
        operations = list(Q.parse(text).walk())
        records = [
            {
                "ordinal": i,
                "operation": Q.op_name(op),
                "source_node_ids": [value.data for value in op.attributes.get("prov.source_node_ids", ArrayAttr([]))],
                "origin_node_ids": [value.data for value in op.attributes.get("prov.origin_node_ids", ArrayAttr([]))],
                "operand_types": [str(value.type) for value in op.operands],
                "result_types": [str(value.type) for value in op.results],
            }
            for i, op in enumerate(operations)
        ]
    return {
        "schema": "m2m.frontend_trace.v1",
        "graphs": {"original": graph},
        "mlir": {
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "bytes": len(text.encode()),
            "operations": records,
        },
    }


def _selection(root, *, geometries=((2, 3, 4),), formats=("integer",)):
    root.mkdir(parents=True, exist_ok=True)
    graph = _graph(geometries)
    text = _module(geometries, formats)
    mlir = root / "model.mlir"
    mlir.write_text(text)
    original = _save(root / "original.json", graph)
    trace = _save(root / "frontend-trace.json", _trace(graph, text))
    contract = root / "session_contract.yaml"
    contract.write_text(yaml.safe_dump({"version": 1}))
    return F.CaptureContractionSelection(
        _pin(contract), (F.ProgramContractionInput("forward", original, trace, _pin(mlir)),)
    )
