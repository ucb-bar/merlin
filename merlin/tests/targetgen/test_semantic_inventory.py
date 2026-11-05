"""The linalg reader preserves enough structure for exact semantic matching."""

from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir

_CAPSULES = repo_root() / "merlin" / "contract" / "capsules" / "radiance" / "model_slices"


def _capsule(name: str) -> dict:
    path = _CAPSULES / name / "capsule.interface.mlir"
    return parse_linalg_mlir(path.read_text(encoding="utf-8"))


def test_matmul_bias_preserves_broadcast_map_scalar_dag_and_provenance():
    inventory = _capsule("RP15_fused_matmul_bias_bf16_pt")
    matmul, bias = inventory["ops"]
    assert matmul["operation"] == "linalg.matmul"
    assert bias["ins"][0]["source"] == ("op", matmul["id"])
    assert bias["ins"][0]["result_index"] == 0
    assert bias["indexing_maps"] == [
        "affine_map<(d0, d1) -> (d0, d1)>",
        "affine_map<(d0, d1) -> (d1)>",
        "affine_map<(d0, d1) -> (d0, d1)>",
    ]
    assert bias["indexing_maps_explicit"] is True
    assert bias["iterator_types"] == ["parallel", "parallel"]
    assert bias["scalar_body"] == {
        "arguments": ["bf16", "bf16", "bf16"],
        "captures": [],
        "operations": [
            {
                "op": "arith.addf",
                "operands": ["arg:0", "arg:1"],
                "results": ["bf16"],
                "attributes": {"fastmath": "#arith.fastmath<none>"},
                "effects": [],
                "regions": 0,
            }
        ],
        "yields": ["op:0:0"],
        "effects": [],
    }
    assert bias["source_provenance"] == {
        "entry": "forward",
        "payload_index": bias["id"],
        "tags": bias["prov"],
        "location": None,
    }
    assert bias["prov"]["aten"] == "aten.add.Tensor"


def test_reduce_and_elementwise_body_edges_are_distinct():
    inventory = _capsule("RP4_softmax_fp32_pt")
    reduce = next(op for op in inventory["ops"] if op["kind"] == "linalg.reduce")
    assert reduce["indexing_maps"] == [
        "affine_map<(d0, d1) -> (d0, d1)>",
        "affine_map<(d0, d1) -> (d0)>",
    ]
    assert reduce["indexing_maps_explicit"] is False
    assert reduce["iterator_types"] == ["parallel", "reduction"]
    assert len(reduce["ins"]) == len(reduce["outs"]) == 1
    assert reduce["ins"][0]["source"] == ("arg", 0)
    assert reduce["outs"][0]["source"][0] == "init"
    assert reduce["scalar_body"]["arguments"] == ["f32", "f32"]
    assert reduce["scalar_body"]["operations"][0]["op"] == "arith.maximumf"
    assert reduce["scalar_body"]["operations"][0]["operands"] == ["arg:0", "arg:1"]
    assert reduce["scalar_body"]["yields"] == ["op:0:0"]
    assert reduce["attributes"]["dimensions"] == "array<i64: 1>"

    exp = next(op for op in inventory["ops"] if "math.exp" in op["body_ops"])
    assert exp["scalar_body"]["operations"][0]["op"] == "math.exp"
    assert exp["scalar_body"]["operations"][0]["operands"] == ["arg:0"]


_TWO_RESULTS = """
module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%a: tensor<4xi32>, %b: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
    %v, %i = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%a : tensor<4xi32>) outs(%b, %b : tensor<4xi32>, tensor<4xi32>) {
      ^bb0(%x: i32, %old_v: i32, %old_i: i32):
        %sum = arith.addi %x, %old_v : i32
        %idx = arith.subi %old_i, %x : i32
        linalg.yield %sum, %idx : i32, i32
      } -> (tensor<4xi32>, tensor<4xi32>)
    func.return %v, %i : tensor<4xi32>, tensor<4xi32>
  }
}
"""


def test_multi_result_body_has_separate_yield_edges():
    generic = parse_linalg_mlir(_TWO_RESULTS)["ops"][0]
    body = generic["scalar_body"]
    assert len(generic["results"]) == 2
    assert [op["operands"] for op in body["operations"]] == [
        ["arg:0", "arg:1"],
        ["arg:2", "arg:0"],
    ]
    assert body["yields"] == ["op:0:0", "op:1:0"]
    assert body["effects"] == []


_CAPTURE_AND_UNKNOWN = """
module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%a: tensor<4xi32>, %out: tensor<4xi32>) -> tensor<4xi32> {
    %c = arith.constant 2 : i32
    %v = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%a : tensor<4xi32>) outs(%out : tensor<4xi32>) {
      ^bb0(%x: i32, %old: i32):
        %sum = arith.addi %x, %c : i32
        %opaque = "unknown.opaque"(%sum) : (i32) -> i32
        linalg.yield %opaque : i32
      } -> tensor<4xi32>
    func.return %v : tensor<4xi32>
  }
}
"""


def test_outer_capture_and_unknown_effects_remain_explicit():
    body = parse_linalg_mlir(_CAPTURE_AND_UNKNOWN)["ops"][0]["scalar_body"]
    assert body["captures"] == [{"source": ("const", None), "shape": [], "dtype": "i32", "const_value": 2.0}]
    assert body["operations"][0]["operands"] == ["arg:0", "capture:0"]
    assert body["operations"][1]["op"] == "unknown.opaque"
    assert body["operations"][1]["operands"] == ["op:0:0"]
    assert body["operations"][1]["effects"] is None
    assert body["effects"] is None


_MAP = """
module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%a: tensor<4xi32>, %out: tensor<4xi32>) -> tensor<4xi32> {
    %v = "linalg.map"(%a, %out) ({
      ^bb0(%x: i32, %old: i32):
        %sum = arith.addi %x, %old : i32
        "linalg.yield"(%sum) : (i32) -> ()
    }) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    func.return %v : tensor<4xi32>
  }
}
"""


def test_named_map_infers_parallel_iterators_and_keeps_output_separate():
    map_op = parse_linalg_mlir(_MAP)["ops"][0]
    assert map_op["kind"] == "linalg.map"
    assert map_op["indexing_maps"] == ["affine_map<(d0) -> (d0)>"] * 2
    assert map_op["indexing_maps_explicit"] is False
    assert map_op["iterator_types"] == ["parallel"]
    assert map_op["ins"][0]["source"] == ("arg", 0)
    assert map_op["outs"][0]["source"] == ("arg", 1)
    assert map_op["scalar_body"]["yields"] == ["op:0:0"]


def test_external_declarations_before_forward_are_not_selected_as_entry():
    mlir = """
module attributes {prov.level = "linalg-on-tensors"} {
  func.func private @external_helper(tensor<4xf32>) -> tensor<4xf32>
  func.func @forward(%arg: tensor<4xf32>) -> tensor<4xf32> {
    func.return %arg : tensor<4xf32>
  }
}
"""
    inventory = parse_linalg_mlir(mlir)
    assert inventory["entry"] == "forward"
    assert inventory["args"] == [{"index": 0, "shape": [4], "dtype": "f32"}]
    assert inventory["results"] == [{"shape": [4], "dtype": "f32"}]


def test_external_declarations_alone_report_missing_definition():
    mlir = """
module attributes {prov.level = "linalg-on-tensors"} {
  func.func private @external_helper(tensor<4xf32>) -> tensor<4xf32>
}
"""
    with pytest.raises(ValueError, match="no func.func definition with a body"):
        parse_linalg_mlir(mlir)


def test_checked_in_tinyllama_recapture_skips_external_declarations():
    """This checked-in recapture is a parser fixture, not hardware provenance or execution evidence."""
    path = repo_root() / "merlin" / "benchmarks" / "dse_guidance" / "recaptures_loop" / "tiny_llama" / "model.mlir"
    inventory = parse_linalg_mlir(path.read_text(encoding="utf-8"))
    assert inventory["entry"] == "forward"
    assert inventory["args"]
    assert inventory["ops"]
    # Observable tensor constants and splats are payloads too, so the first op need not be a generic.
    assert "linalg.generic" in {op["kind"] for op in inventory["ops"]}
