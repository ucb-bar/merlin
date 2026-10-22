import copy
import hashlib
import json

import pytest

from merlin_experiments.corpus.projection_shapes import contractions, write_sources


def graph():
    nodes, edges = [], []
    for name in ("gate", "up"):
        nodes.append({"id": name, "target": "aten._int_mm.default",
                      "results": [{"dtype": "int32", "shape": [3, 19]}]})
        for index, shape in enumerate(([3, 7], [7, 19])):
            edges.append({"consumer_node_id": name, "argument_path": f"args/{index}",
                          "dtype": "int8", "shape": shape})
    return {"source_sha256": "a" * 64, "model": "public_graph", "program": "prefill",
            "graphs": [{"stage": "prepared", "nodes": nodes, "edges": edges}]}


def test_exact_shapes_retain_all_shared_geometry_occurrences():
    source = graph()
    before = copy.deepcopy(source)
    [row] = contractions(source)
    assert row["shape_mkn"] == [3, 7, 19]
    assert row["source_node_ids"] == ["gate", "up"]
    assert row["trace_sha256"] == source["source_sha256"]
    assert source == before


def test_inconsistent_prepared_semantics_refuse():
    source = graph()
    source["graphs"][0]["edges"][1]["shape"] = [8, 19]
    with pytest.raises(ValueError, match="exact integer"):
        contractions(source)


def test_ambiguous_source_argument_refuses():
    source = graph()
    source["graphs"][0]["edges"].append(copy.deepcopy(source["graphs"][0]["edges"][0]))
    with pytest.raises(ValueError, match="ambiguous"):
        contractions(source)


def test_native_inferred_output_tracks_selected_shape(tmp_path):
    import yaml

    template = tmp_path / "template"
    template.mkdir()
    cap = {"name": "source", "interface_mlir": "source.mlir",
           "inputs": [{"name": "A", "dtype": "i8", "shape": [1, 2]},
                      {"name": "W", "dtype": "i8", "shape": [2, 4]}],
           "operation": {"op": "matmul", "attributes": {"lhs": "A", "weight": "W", "out": "Y",
                                                               "epilogue": [], "output_dtype": "i32"}},
           "numeric_policy": {"compare": "exact_int", "dtype": "i32"}}
    (template / "capsule.yaml").write_text(yaml.safe_dump(cap))
    (template / "source.mlir").write_text('''module attributes {merlin_iface.version = "0.1", merlin_iface.target = "mock", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<2x4xi8>
  %A = merlin_iface.tensor {name = "A", role = "input"} : tensor<1x2xi8>
  %r = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<2x4xi8>) -> !merlin_iface.resident
  %acc = merlin_iface.matmul %A, %r : (tensor<1x2xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y = merlin_iface.commit %acc {name = "Y", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<1x4xi32>
  merlin_iface.evict %r : (!merlin_iface.resident) -> ()
}''')
    projection = tmp_path / "projection.json"
    projection.write_text(json.dumps(graph()))
    [result] = write_sources(projection, sha256=hashlib.sha256(projection.read_bytes()).hexdigest(),
                             shapes=[(3, 7, 19)], template=template, destination=tmp_path / "new",
                             name_prefix="probe")
    assert "tensor<3x19xi32>" in (result / "source.mlir").read_text()
    with pytest.raises(ValueError, match="observed"):
        write_sources(projection, sha256=hashlib.sha256(projection.read_bytes()).hexdigest(),
                      shapes=[(3, 7, 20)], template=template, destination=tmp_path / "invented",
                      name_prefix="probe")
