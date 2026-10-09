"""Observable tensor constants retain source identity, values and return order.

This inventories standard source IR only. It provides no lowering, numerical,
semantic-owner, resource, device-effect or runtime qualification.
"""

import pytest
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
from xdsl.parser import Parser

from merlin.targetgen.contract.linalg_iface import make_linalg_context, parse_linalg_mlir
from merlin.targetgen.oot_starterkit.iface import parse_linalg


def _constants(dtype="f32", first=(1.25, -2.0), second=(3.0, 4.0), returned="%b, %a, %b"):
    tensor = f"tensor<2x{dtype}>"
    return f"""module {{
      func.func @forward() -> ({tensor}, {tensor}, {tensor}) {{
        %a = arith.constant dense<{list(first)}> : {tensor}
        %b = arith.constant dense<{list(second)}> : {tensor}
        func.return {returned} : {tensor}, {tensor}, {tensor}
      }}
    }}"""


@pytest.mark.parametrize(
    "dtype,first,second",
    [
        ("f32", (1.25, -2.0), (3.0, 4.0)),
        ("i64", ((1 << 53) + 1, -(1 << 63)), ((1 << 63) - 1, -7)),
        ("i8", (-128, 7), (0, 127)),
    ],
)
def test_returned_tensor_constants_are_distinct_payloads_with_exact_typed_values(dtype, first, second):
    text = _constants(dtype, first, second)
    inventory = parse_linalg_mlir(text)
    assert inventory["args"] == []
    assert [op["operation"] for op in inventory["ops"]] == ["arith.constant", "arith.constant"]
    assert [op["id"] for op in inventory["ops"]] == [0, 1]
    for op, expected in zip(inventory["ops"], (first, second), strict=True):
        assert op["ins"] == op["outs"] == []
        assert op["results"] == [{"shape": [2], "dtype": dtype}]
        value = Parser(make_linalg_context(), op["attributes"]["value"]).parse_attribute()
        assert isinstance(value, DenseIntOrFPElementsAttr)
        assert tuple(value.get_values()) == expected
        if dtype.startswith("i"):
            assert all(type(number) is int for number in value.get_values())
    assert [value["source"] for value in inventory["returns"]] == [("op", 1), ("op", 0), ("op", 1)]
    assert [value["result_index"] for value in inventory["returns"]] == [0, 0, 0]
    assert len(inventory["results"]) == len(inventory["returns"]) == 3
    assert parse_linalg(text) == inventory


def test_same_type_swapped_constants_change_the_observed_original_return_binding():
    original = parse_linalg_mlir(_constants())
    swapped = parse_linalg_mlir(_constants(returned="%a, %b, %b"))
    assert original["args"] == swapped["args"] and original["results"] == swapped["results"]
    assert original["ops"] == swapped["ops"]
    assert original["returns"] != swapped["returns"]
    assert [value["source"] for value in swapped["returns"]] == [("op", 0), ("op", 1), ("op", 1)]


def test_equal_constant_values_do_not_collapse_distinct_original_ssa_identities():
    inventory = parse_linalg_mlir(_constants(second=(1.25, -2.0)))
    assert inventory["ops"][0]["attributes"] == inventory["ops"][1]["attributes"]
    assert [value["source"] for value in inventory["returns"]] == [("op", 1), ("op", 0), ("op", 1)]


def test_returned_tensor_splats_keep_their_exact_scalar_operands():
    text = """module {
      func.func @forward() -> (tensor<3xi64>, tensor<3xi64>) {
        %first = arith.constant 9007199254740993 : i64
        %second = arith.constant -7 : i64
        %a = tensor.splat %first : tensor<3xi64>
        %b = tensor.splat %second : tensor<3xi64>
        func.return %b, %a : tensor<3xi64>, tensor<3xi64>
      }
    }"""
    inventory = parse_linalg_mlir(text)
    assert [op["operation"] for op in inventory["ops"]] == ["tensor.splat", "tensor.splat"]
    assert [op["ins"][0]["const_value"] for op in inventory["ops"]] == [9007199254740993, -7]
    assert all(type(op["ins"][0]["const_value"]) is int for op in inventory["ops"])
    assert [value["source"] for value in inventory["returns"]] == [("op", 1), ("op", 0)]


def test_data_consumed_tensor_constant_is_not_hidden_initialization():
    text = """module {
      func.func @forward() -> tensor<2xi32> {
        %data = arith.constant dense<[-7, 13]> : tensor<2xi32>
        %empty = tensor.empty() : tensor<2xi32>
        %copied = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
          ins(%data : tensor<2xi32>) outs(%empty : tensor<2xi32>) {
          ^bb0(%x: i32, %unused: i32):
            linalg.yield %x : i32
          } -> tensor<2xi32>
        func.return %copied : tensor<2xi32>
      }
    }"""
    inventory = parse_linalg_mlir(text)
    assert [op["operation"] for op in inventory["ops"]] == ["arith.constant", "linalg.generic"]
    assert inventory["ops"][1]["ins"][0]["source"] == ("op", 0)
    assert inventory["ops"][1]["ins"][0]["result_index"] == 0
    assert inventory["returns"][0]["source"] == ("op", 1)


@pytest.mark.parametrize("initializer", ["constant", "splat"])
def test_destination_only_tensor_initializer_retains_its_existing_representation(initializer):
    value = (
        "%seed = arith.constant dense<0> : tensor<2x4xi32>"
        if initializer == "constant"
        else "%zero = arith.constant 0 : i32\n %seed = tensor.splat %zero : tensor<2x4xi32>"
    )
    text = f"""module {{
      func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi32> {{
        {value}
        %product = linalg.matmul ins(%a, %b : tensor<2x3xi8>, tensor<3x4xi8>)
          outs(%seed : tensor<2x4xi32>) -> tensor<2x4xi32>
        func.return %product : tensor<2x4xi32>
      }}
    }}"""
    inventory = parse_linalg_mlir(text)
    assert [op["operation"] for op in inventory["ops"]] == ["linalg.matmul"]
    assert inventory["ops"][0]["outs"][0]["source"] == (
        ("const", None) if initializer == "constant" else ("init", "splat")
    )
    assert inventory["returns"][0]["source"] == ("op", 0)
