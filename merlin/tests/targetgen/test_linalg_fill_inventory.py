"""Inventory actual fill uses and ordered returned values, without lowering grants."""

import pytest

from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
from merlin.targetgen.oot_starterkit.iface import parse_linalg


def _constant_fill(value=7):
    return f"""module {{
      func.func @forward() -> tensor<5xi64> {{
        %value = arith.constant {value} : i64
        %empty = tensor.empty() : tensor<5xi64>
        %filled = linalg.fill ins(%value : i64) outs(%empty : tensor<5xi64>) -> tensor<5xi64>
        func.return %filled : tensor<5xi64>
      }}
    }}"""


@pytest.mark.parametrize("value", [0, -7, (1 << 53) + 1, -(1 << 63)])
def test_returned_fill_is_payload_with_exact_integer_constant(value):
    inventory = parse_linalg_mlir(_constant_fill(value))
    assert inventory["args"] == []
    assert inventory["results"] == [{"shape": [5], "dtype": "i64"}]
    assert len(inventory["ops"]) == 1
    fill = inventory["ops"][0]
    assert fill["operation"] == "linalg.fill" and fill["id"] == 0
    assert fill["ins"] == [{"source": ("const", None), "shape": [], "dtype": "i64", "const_value": value}]
    assert type(fill["ins"][0]["const_value"]) is int
    assert fill["outs"] == [{"source": ("init", "empty"), "shape": [5], "dtype": "i64"}]
    assert fill["results"] == inventory["results"]
    assert inventory["returns"] == [{"source": ("op", 0), "result_index": 0, "shape": [5], "dtype": "i64"}]
    assert parse_linalg(_constant_fill(value)) == inventory


def test_runtime_scalar_and_original_destination_arguments_are_preserved():
    text = """module {
      func.func @forward(%value: i32, %destination: tensor<3xi32>) -> tensor<3xi32> {
        %filled = linalg.fill ins(%value : i32) outs(%destination : tensor<3xi32>) -> tensor<3xi32>
        func.return %filled : tensor<3xi32>
      }
    }"""
    inventory = parse_linalg_mlir(text)
    assert inventory["args"] == [
        {"index": 0, "shape": [], "dtype": "i32"},
        {"index": 1, "shape": [3], "dtype": "i32"},
    ]
    fill = inventory["ops"][0]
    assert fill["ins"] == [{"source": ("arg", 0), "shape": [], "dtype": "i32"}]
    assert fill["outs"] == [{"source": ("arg", 1), "shape": [3], "dtype": "i32"}]


def _contraction(*, publish_initialization):
    result = "(tensor<2x4xi32>, tensor<2x4xi32>)" if publish_initialization else "tensor<2x4xi32>"
    returned = (
        "%filled, %product : tensor<2x4xi32>, tensor<2x4xi32>"
        if publish_initialization
        else "%product : tensor<2x4xi32>"
    )
    return f"""module {{
      func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> {result} {{
        %zero = arith.constant 0 : i32
        %empty = tensor.empty() : tensor<2x4xi32>
        %filled = linalg.fill ins(%zero : i32) outs(%empty : tensor<2x4xi32>) -> tensor<2x4xi32>
        %product = linalg.matmul ins(%a, %b : tensor<2x3xi8>, tensor<3x4xi8>)
          outs(%filled : tensor<2x4xi32>) -> tensor<2x4xi32>
        func.return {returned}
      }}
    }}"""


def test_destination_only_contraction_fill_keeps_existing_initialization_inventory():
    inventory = parse_linalg_mlir(_contraction(publish_initialization=False))
    assert [op["operation"] for op in inventory["ops"]] == ["linalg.matmul"]
    matmul = inventory["ops"][0]
    assert matmul["id"] == 0 and matmul["outs"][0]["source"] == ("init", "fill")
    assert matmul["extents"] == {"m": 2, "k": 3, "n": 4}
    assert inventory["returns"][0]["source"] == ("op", 0)


def test_escaped_initialization_and_contraction_have_two_distinct_original_outputs():
    inventory = parse_linalg_mlir(_contraction(publish_initialization=True))
    assert [op["operation"] for op in inventory["ops"]] == ["linalg.fill", "linalg.matmul"]
    assert inventory["ops"][1]["outs"][0]["source"] == ("op", 0)
    assert inventory["ops"][1]["outs"][0]["result_index"] == 0
    assert [value["source"] for value in inventory["returns"]] == [("op", 0), ("op", 1)]
    assert len(inventory["results"]) == len(inventory["returns"]) == 2


def test_fill_consumed_as_data_is_a_producer_even_when_not_directly_returned():
    text = _constant_fill().replace(
        "func.return %filled : tensor<5xi64>",
        """
        %destination = tensor.empty() : tensor<5xi64>
        %copied = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
          ins(%filled : tensor<5xi64>) outs(%destination : tensor<5xi64>) {
          ^bb0(%x: i64, %unused: i64):
            linalg.yield %x : i64
          } -> tensor<5xi64>
        func.return %copied : tensor<5xi64>""",
    )
    inventory = parse_linalg_mlir(text)
    assert [op["operation"] for op in inventory["ops"]] == ["linalg.fill", "linalg.generic"]
    assert inventory["ops"][1]["ins"][0]["source"] == ("op", 0)
    assert inventory["returns"][0]["source"] == ("op", 1)


def test_repeated_published_values_are_not_deduplicated():
    text = (
        _constant_fill()
        .replace("-> tensor<5xi64> {", "-> (tensor<5xi64>, tensor<5xi64>) {")
        .replace(
            "func.return %filled : tensor<5xi64>",
            "func.return %filled, %filled : tensor<5xi64>, tensor<5xi64>",
        )
    )
    inventory = parse_linalg_mlir(text)
    assert len(inventory["results"]) == len(inventory["returns"]) == 2
    assert inventory["returns"][0] == inventory["returns"][1]


def test_returned_input_and_fill_preserve_actual_output_order():
    text = (
        _constant_fill()
        .replace("@forward() -> tensor<5xi64>", "@forward(%original: tensor<5xi64>) -> (tensor<5xi64>, tensor<5xi64>)")
        .replace(
            "func.return %filled : tensor<5xi64>",
            "func.return %original, %filled : tensor<5xi64>, tensor<5xi64>",
        )
    )
    inventory = parse_linalg_mlir(text)
    assert inventory["args"] == [{"index": 0, "shape": [5], "dtype": "i64"}]
    assert [value["source"] for value in inventory["returns"]] == [("arg", 0), ("op", 0)]
    swapped = parse_linalg_mlir(text.replace("func.return %original, %filled", "func.return %filled, %original"))
    assert swapped["results"] == inventory["results"]
    assert [value["source"] for value in swapped["returns"]] == [("op", 0), ("arg", 0)]


def test_return_join_retains_result_index_of_a_multi_result_producer():
    text = """module {
      func.func @forward(%value: tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>) {
        %empty = tensor.empty() : tensor<3xi32>
        %first, %second = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
          ins(%value : tensor<3xi32>) outs(%empty, %empty : tensor<3xi32>, tensor<3xi32>) {
          ^bb0(%x: i32, %unused1: i32, %unused2: i32):
            linalg.yield %x, %x : i32, i32
          } -> (tensor<3xi32>, tensor<3xi32>)
        func.return %second, %first : tensor<3xi32>, tensor<3xi32>
      }
    }"""
    inventory = parse_linalg_mlir(text)
    assert [value["source"] for value in inventory["returns"]] == [("op", 0), ("op", 0)]
    assert [value["result_index"] for value in inventory["returns"]] == [1, 0]


@pytest.mark.parametrize("declaration", ["tensor<4xi64>", "tensor<5xi32>", "(tensor<5xi64>, tensor<5xi64>)"])
def test_return_shape_dtype_and_missing_output_disagreement_refuse(declaration):
    text = _constant_fill().replace("@forward() -> tensor<5xi64>", "@forward() -> " + declaration)
    with pytest.raises(ValueError, match="complete ordered result signature"):
        parse_linalg_mlir(text)


def test_multiblock_return_join_is_explicitly_unavailable():
    text = """module {
      func.func @forward(%value: tensor<3xi32>) -> tensor<3xi32> {
        cf.br ^exit
      ^exit:
        func.return %value : tensor<3xi32>
      }
    }"""
    inventory = parse_linalg_mlir(text)
    assert inventory["results"] == [{"shape": [3], "dtype": "i32"}]
    assert inventory["returns"] is None
