# ruff: noqa: E501 -- the IR fixture keeps the printer's one-line operation form
"""Public case files for the composition-route tests: chain declarations and a small provenance-tagged IR.

The IR is a two-product program sharing one input (one weight transposed in the program), a residual
add of both products feeding a pool, and a batched head product. Region tags use the frontend's generic
provenance vocabulary (``prov.region_id``, ``prov.family``, ``prov.op``); nothing names a target.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROGRAM_IR = """\
builtin.module {
  func.func @forward(%X: tensor<4x8xi8>, %W0: tensor<8x4xi8>, %W1: tensor<4x8xi8>, %Q: tensor<2x4x8xi8>, %K: tensor<2x8x4xi8>) -> (tensor<4x4xi32>, tensor<2x4x4xi32>) {
    %zero = arith.constant 0 : i32
    %e0 = tensor.empty() : tensor<8x4xi8>
    %w1t = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%W1 : tensor<4x8xi8>) outs(%e0 : tensor<8x4xi8>) attrs = {prov.region_id = "permute_0", prov.op = "permute", prov.family = "layout"} {
      ^bb0(%a: i8, %b: i8):
        linalg.yield %a : i8
    } -> tensor<8x4xi8>
    %e1 = tensor.empty() : tensor<4x4xi32>
    %i1 = linalg.fill ins(%zero : i32) outs(%e1 : tensor<4x4xi32>) -> tensor<4x4xi32>
    %p0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%X, %W0 : tensor<4x8xi8>, tensor<8x4xi8>) outs(%i1 : tensor<4x4xi32>) attrs = {prov.region_id = "matmul_0", prov.op = "int_matmul", prov.family = "contraction"} {
      ^bb1(%l: i8, %r: i8, %o: i32):
        %lw = arith.extsi %l : i8 to i32
        %rw = arith.extsi %r : i8 to i32
        %m = arith.muli %lw, %rw : i32
        %s = arith.addi %o, %m : i32
        linalg.yield %s : i32
    } -> tensor<4x4xi32>
    %p1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%X, %w1t : tensor<4x8xi8>, tensor<8x4xi8>) outs(%i1 : tensor<4x4xi32>) attrs = {prov.region_id = "matmul_1", prov.op = "int_matmul", prov.family = "contraction"} {
      ^bb2(%l: i8, %r: i8, %o: i32):
        %lw = arith.extsi %l : i8 to i32
        %rw = arith.extsi %r : i8 to i32
        %m = arith.muli %lw, %rw : i32
        %s = arith.addi %o, %m : i32
        linalg.yield %s : i32
    } -> tensor<4x4xi32>
    %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%p0, %p1 : tensor<4x4xi32>, tensor<4x4xi32>) outs(%e1 : tensor<4x4xi32>) attrs = {prov.region_id = "add_0", prov.op = "add", prov.family = "elementwise"} {
      ^bb3(%a: i32, %b: i32, %o: i32):
        %t = arith.addi %a, %b : i32
        linalg.yield %t : i32
    } -> tensor<4x4xi32>
    %e2 = tensor.empty() : tensor<4x4xi32>
    %pool = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum : tensor<4x4xi32>) outs(%e2 : tensor<4x4xi32>) attrs = {prov.region_id = "max_pool2d_0", prov.op = "max_pool2d", prov.family = "pool"} {
      ^bb4(%a: i32, %o: i32):
        linalg.yield %a : i32
    } -> tensor<4x4xi32>
    %e3 = tensor.empty() : tensor<2x4x4xi32>
    %i3 = linalg.fill ins(%zero : i32) outs(%e3 : tensor<2x4x4xi32>) -> tensor<2x4x4xi32>
    %h = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%Q, %K : tensor<2x4x8xi8>, tensor<2x8x4xi8>) outs(%i3 : tensor<2x4x4xi32>) attrs = {prov.region_id = "matmul_2", prov.op = "batch_matmul", prov.family = "contraction"} {
      ^bb5(%l: i8, %r: i8, %o: i32):
        %lw = arith.extsi %l : i8 to i32
        %rw = arith.extsi %r : i8 to i32
        %m = arith.muli %lw, %rw : i32
        %s = arith.addi %o, %m : i32
        linalg.yield %s : i32
    } -> tensor<2x4x4xi32>
    func.return %pool, %h : tensor<4x4xi32>, tensor<2x4x4xi32>
  }
}
"""


def weights(*shapes):
    return [{"name": f"W{i}", "role": "weight", "shape": list(shape), "dtype": "i8"} for i, shape in enumerate(shapes)]


def matmul(name="mm", **attrs):
    return {
        "name": name,
        "kind": "layer",
        "label": "public",
        "semantic": {"semantic_family": "contraction"},
        "inputs": [{"name": "A0", "role": "input", "shape": [16, 32], "dtype": "i8"}, *weights((32, 16))],
        "operation": {"op": "matmul", "attributes": {"lhs": "A0", "weight": "W0", "out": "Y0", **attrs}},
    }


def conv_pool(name="conv_pool"):
    return matmul(name, epilogue=["bias_add", "acc_scale", "relu", "maxpool"], acc_scale=0.0625, bias="B",
                  pool_size=[2, 2], pool_stride=[2, 2])


def residual(name="residual"):
    return {
        "name": name,
        "kind": "layer",
        "label": "public",
        "semantic": {"semantic_family": "elementwise_map"},
        "inputs": [{"name": "X0", "role": "input", "shape": [9, 24], "dtype": "i8"},
                   {"name": "X1", "role": "input", "shape": [9, 24], "dtype": "i8"}],
        "operation": {"op": "residual_add", "attributes": {"lhs": "X0", "rhs": "X1", "epilogue": ["relu"]}},
    }


def siblings(name="siblings"):
    return {
        "name": name,
        "kind": "model_slice",
        "label": "public",
        "inputs": [{"name": "A", "role": "input", "shape": [16, 32], "dtype": "i8"}, *weights((32, 16), (32, 24))],
        "operation": {"op": "component_program", "attributes": {"program": {"nodes": [
            {"name": "p0", "op": "matmul", "inputs": ["A", "W0"]},
            {"name": "p1", "op": "matmul", "inputs": ["A", "W1"]},
        ]}}},
    }


def model(name="model_case", group=None):
    declaration = {
        "name": name,
        "kind": "model",
        "label": "public",
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": "I0", "role": "input", "shape": [4, 8], "dtype": "f32"}],
        "operation": {"op": "model", "attributes": {"weights": "capsule.weights.safetensors", "out": "Y0"}},
        "software_screen": {"decisions": [
            {"operation": "relu", "placement": "host", "classification": "host_required"},
            {"operation": "matmul", "placement": "accelerator", "classification": "accelerator_candidate"},
        ]},
    }
    if group:
        declaration["model_qualification"] = {"group_capsules": [{"capsule": group, "count": 1}]}
    return declaration


def declare(workspace: Path, group: str, declaration: dict, *, ir: str | None = None) -> Path:
    folder = Path(workspace) / group / declaration["name"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "capsule.yaml").write_text(yaml.safe_dump(declaration))
    if ir is not None:
        (folder / "capsule.interface.mlir").write_text(ir)
    return folder


def program(workspace: Path, name="PG_program", ir: str = PROGRAM_IR) -> Path:
    """A program case staged with only its interface IR (no declaration), as the program suite stages it."""
    folder = Path(workspace) / "public_program_sources" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "capsule.interface.mlir").write_text(ir)
    return folder
