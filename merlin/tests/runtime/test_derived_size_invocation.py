"""Derived allocation needs actual typed computation, independently of receipts."""

import json

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.frontends.model_shape_facts import EXPRESSION_PROFILE, ShapeFacts
from merlin.frontends.shape_expressions import Program
from merlin.llvmlower.runtime_invocation import resolve

SOURCE = """module {func.func @forward(%x:tensor<?xi32>)->tensor<?xi32> {
  %zero=arith.constant 0:index
  %n=tensor.dim %x,%zero:tensor<?xi32>
  %word=arith.index_castui %n:index to i64
  %back=arith.index_castui %word:i64 to index
  %same=arith.cmpi eq,%n,%back:index
  cf.assert %same,"source round trip"
  %high=arith.constant 31:i64
  %bounded=arith.cmpi ule,%word,%high:i64
  cf.assert %bounded,"source range"
  %two=arith.constant 2:i64
  %half=arith.divui %word,%two:i64
  %size=arith.index_castui %half:i64 to index
  %out="tensor.extract_slice"(%x,%size) <{static_offsets=array<i64:0>,
    static_sizes=array<i64:-9223372036854775808>,static_strides=array<i64:1>,
    operandSegmentSizes=array<i32:1,0,1,0>}>:(tensor<?xi32>,index)->tensor<?xi32>
  return %out:tensor<?xi32>
}}"""


def example(root, n=7):
    root.mkdir()
    (root / "model.mlir").write_text(SOURCE)
    function = next(iter(parse_mlir_text(SOURCE).ops))
    contract = ShapeFacts.entry(function, expressions=True).contract(function)
    signature = {
        **contract,
        "schema": "merlin.compile-signature.v4",
        "entry": "forward",
        "model_shape_profile": EXPRESSION_PROFILE,
        "shape_scope": "source-proved-derived-dimensions",
    }
    (root / "signature.json").write_text(json.dumps(signature))
    np.savez(root / "inputs.npz", in0=np.arange(n, dtype=np.int32))
    return root, signature


@pytest.mark.parametrize("n", (0, 1, 7, 31))
def test_source_derived_sizes_allocate_without_extra_public_arguments(tmp_path, n):
    root, _signature = example(tmp_path / "source", n)
    plan = resolve(root / "model.mlir", root / "signature.json", root / "inputs.npz")
    assert plan["inputs"] == [{"shape": [n], "dtype": "i32", "kind": "tensor"}]
    assert plan["outputs"] == [{"shape": [n // 2], "dtype": "i32"}]
    assert plan["input_axis_dimensions"] == {"d0_0": n}
    assert plan["mlir_argument_specialization"] == "not-performed"


@pytest.mark.parametrize("damage", ("profile", "downgrade", "program", "valid-program", "output", "body", "guard"))
def test_rehashed_declarations_cannot_change_actual_source_size_computation(tmp_path, damage):
    root, signature = example(tmp_path / "source")
    if damage == "profile":
        signature.pop("model_shape_profile")
    elif damage == "downgrade":
        signature["schema"] = "merlin.compile-signature.v3"
    elif damage == "program":
        next(row for row in signature["shape_program"]["nodes"].values() if row["op"] == "const")["value"] ^= 1
    elif damage == "valid-program":
        program = Program()
        axis = program.dimension("d0_0")
        word = program.intern("index_castui", type_=64, args=[axis])
        three = program.intern("const", type_=64, value=3)
        wrong = program.intern("divui", type_=64, args=[word, three])
        root_id = program.intern("index_castui", type_="index", args=[wrong])
        signature["outputs"][0]["shape"] = [root_id]
        signature["shape_program"] = program.export([root_id])
    elif damage == "output":
        signature["outputs"][0]["shape"] = [1]
    elif damage == "body":
        (root / "model.mlir").write_text(SOURCE.replace("constant 2:i64", "constant 3:i64"))
    else:
        (root / "model.mlir").write_text(SOURCE.replace('cf.assert %bounded,"source range"', ""))
    (root / "signature.json").write_text(json.dumps(signature))
    with pytest.raises(ValueError):
        resolve(root / "model.mlir", root / "signature.json", root / "inputs.npz")


@pytest.mark.parametrize("nx,ny", ((0, 1), (1, 7), (3, 1), (3, 3)))
def test_conditional_index_sizes_retain_actual_source_selection(tmp_path, nx, ny):
    source = """module {func.func @forward(%x:tensor<?xi32>,%y:tensor<?xi32>)->tensor<?xi32> {
      %zero=arith.constant 0:index
      %one=arith.constant 1:index
      %n=tensor.dim %x,%zero:tensor<?xi32>
      %m=tensor.dim %y,%zero:tensor<?xi32>
      %singleton=arith.cmpi eq,%n,%one:index
      %size=arith.select %singleton,%m,%n:index
      %out=tensor.empty(%size):tensor<?xi32>
      return %out:tensor<?xi32>
    }}"""
    function = next(iter(parse_mlir_text(source).ops))
    contract = ShapeFacts.entry(function, expressions=True).contract(function)
    signature = {
        **contract,
        "schema": "merlin.compile-signature.v4",
        "entry": "forward",
        "model_shape_profile": EXPRESSION_PROFILE,
        "shape_scope": "source-proved-derived-dimensions",
    }
    (tmp_path / "model.mlir").write_text(source)
    (tmp_path / "signature.json").write_text(json.dumps(signature))
    np.savez(tmp_path / "inputs.npz", in0=np.zeros(nx, np.int32), in1=np.zeros(ny, np.int32))
    plan = resolve(tmp_path / "model.mlir", tmp_path / "signature.json", tmp_path / "inputs.npz")
    assert plan["outputs"] == [{"shape": [ny if nx == 1 else nx], "dtype": "i32"}]
