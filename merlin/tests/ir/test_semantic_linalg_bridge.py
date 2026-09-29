"""Parsed Linalg-to-native graph boundary checks for the admitted integer subset."""

from __future__ import annotations

import os
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.semantic_compiler.allocate import StorageBank
from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text
from merlin.semantic_compiler.model import KernelRequest, TensorType
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
from merlin.semantic_compiler.rules import InstructionDescriptor
from merlin.semantic_compiler.search import select_and_allocate


@pytest.fixture(scope="module")
def bridge(tmp_path_factory: pytest.TempPathFactory) -> Path:
    manifest = repo_root() / "src/merlin/semantic_compiler/egg_bridge/Cargo.toml"
    target = tmp_path_factory.mktemp("linalg-native-egg")
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(target)
    subprocess.run(["cargo", "build", "--locked", "--release", "--manifest-path", str(manifest)],
                   check=True, env=environment)
    return target / "release/merlin-egg-bridge"


def _generic_source(*, init: str = "argument", body: str = "signed", swapped_map: bool = False,
                    two_outputs: bool = False) -> str:
    function_args = "%a: tensor<2x3xi8>, %b: tensor<3x2xi8>"
    prefix = ""
    if init == "argument":
        function_args += ", %c: tensor<2x2xi32>"
    else:
        prefix = "%empty = tensor.empty() : tensor<2x2xi32>\n"
        if init == "filled":
            prefix += (
                "%seven = arith.constant 7 : i32\n"
                "%c = linalg.fill ins(%seven : i32) outs(%empty : tensor<2x2xi32>) -> tensor<2x2xi32>\n"
            )
        else:
            prefix += "%c = tensor.empty() : tensor<2x2xi32>\n"
    left_map = "(k,m)" if swapped_map else "(m,k)"
    cast = "extui" if body == "unsigned" else "extsi"
    if body == "dropped_accumulator":
        yield_operand = "%p"
    else:
        yield_operand = "%v"
    result_types = "(tensor<2x2xi32>, tensor<2x2xi32>)" if two_outputs else "tensor<2x2xi32>"
    returned = "%r, %c : tensor<2x2xi32>, tensor<2x2xi32>" if two_outputs else "%r : tensor<2x2xi32>"
    return f"""module {{
func.func @work({function_args}) -> {result_types} {{
{prefix}%r = linalg.generic {{indexing_maps = [affine_map<(m,n,k)->{left_map}>,
  affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
  iterator_types = ["parallel", "parallel", "reduction"]}}
  ins(%a, %b : tensor<2x3xi8>, tensor<3x2xi8>) outs(%c : tensor<2x2xi32>) {{
  ^bb0(%x: i8, %y: i8, %acc: i32):
  %xx = arith.{cast} %x : i8 to i32
  %yy = arith.extsi %y : i8 to i32
  %p = arith.muli %xx, %yy : i32
  %v = arith.addi %p, %acc : i32
  linalg.yield {yield_operand} : i32
  }} -> tensor<2x2xi32>
  func.return {returned}
}}
}}"""


def _inputs(*, init: int = 10) -> dict[str, TensorValue]:
    return {
        "arg0": TensorValue(TensorType((2, 3), "i8", "signed-i8"), (1, 2, 3, 4, 5, 6)),
        "arg1": TensorValue(TensorType((3, 2), "i8", "signed-i8"), (1, 2, 3, 4, 5, 6)),
        "arg2": TensorValue(TensorType((2, 2), "i32", "i32-wrap-k-ascending"), (init,) * 4),
    }


def test_parsed_generic_preserves_init_multiple_outputs_and_source_identity() -> None:
    source = _generic_source(two_outputs=True)
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-linalg-v1")
    assert translated.request.source_identity == sha256(source.encode()).hexdigest()
    assert translated.request.outputs == ("op0r0", "arg2")
    assert translated.source_operations == ("linalg.generic", "func.return")
    assert KernelRequest.from_record(translated.request.record()).record() == translated.request.record()
    result = evaluate_graph(translated.request, _inputs())
    assert [value.elements for value in result] == [(32, 38, 59, 74), (10, 10, 10, 10)]
    overflow = evaluate_graph(translated.request, _inputs(init=(1 << 31) - 1))
    assert overflow[0].elements[0] == -(1 << 31) + 21


def test_parsed_fill_materializes_declared_constant_and_rejects_uninitialized_init() -> None:
    translated = translate_linalg_text(_generic_source(init="filled"), entry="work", target_identity="synthetic")
    assert translated.source_operations == (
        "tensor.empty", "arith.constant", "linalg.fill", "linalg.generic", "func.return",
    )
    assert len(translated.constants) == 1
    assert translated.constants["op2r0"].elements == (7, 7, 7, 7)
    inputs = {key: value for key, value in _inputs().items() if key != "arg2"}
    assert evaluate_graph(translated.request, inputs, constants=translated.constants)[0].elements == (
        29, 35, 56, 71,
    )
    with pytest.raises(LinalgBridgeError, match="uninitialized"):
        translate_linalg_text(_generic_source(init="empty"), entry="work", target_identity="synthetic")


@pytest.mark.parametrize("mutation", ["unsigned", "dropped_accumulator"])
def test_parsed_generic_rejects_changed_scalar_semantics(mutation: str) -> None:
    with pytest.raises(LinalgBridgeError, match="body|widen|yield"):
        translate_linalg_text(_generic_source(body=mutation), entry="work", target_identity="synthetic")


def test_parsed_generic_rejects_changed_indexing_map() -> None:
    with pytest.raises(LinalgBridgeError, match="indexing maps"):
        translate_linalg_text(_generic_source(swapped_map=True), entry="work", target_identity="synthetic")


def test_parsed_return_must_match_declared_output_types() -> None:
    source = _generic_source(two_outputs=True).replace(
        "-> (tensor<2x2xi32>, tensor<2x2xi32>)",
        "-> (tensor<2x2xi32>, tensor<1x2xi32>)",
    )
    with pytest.raises(LinalgBridgeError, match="entry signature"):
        translate_linalg_text(source, entry="work", target_identity="synthetic")


def test_named_i32_matmul_uses_same_initialized_graph_primitive() -> None:
    source = """module { func.func @work(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>,
      %c: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %r = linalg.matmul ins(%a, %b : tensor<2x2xi32>, tensor<2x2xi32>)
        outs(%c : tensor<2x2xi32>) -> tensor<2x2xi32>
      func.return %r : tensor<2x2xi32>
    } }"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    tensor = TensorType((2, 2), "i32", "i32-wrap-k-ascending")
    values = {
        "arg0": TensorValue(tensor, (1, 2, 3, 4)),
        "arg1": TensorValue(tensor, (5, 6, 7, 8)),
        "arg2": TensorValue(tensor, (10, 20, 30, 40)),
    }
    assert evaluate_graph(translated.request, values)[0].elements == (29, 42, 73, 90)


def test_parsed_region_reaches_native_rule_extraction_and_allocation(bridge: Path) -> None:
    translated = translate_linalg_text(_generic_source(), entry="work", target_identity="synthetic")
    maps = translated.request.nodes[-1].index_maps
    descriptor = InstructionDescriptor(
        "synthetic_contract", "matmul_accumulate", ("external",) * 3, "external",
        "i32", "i32-wrap-k-ascending", (2,), input_dtypes=("i8", "i8", "i32"),
        input_numerical_policies=("signed-i8", "signed-i8", "i32-wrap-k-ascending"),
        input_ranks=(2, 2, 2), index_maps=maps,
    )
    selected = select_and_allocate(
        translated.request, (descriptor,), (StorageBank("external", "dram", 4, "word"),),
        bridge=bridge, fixed_inputs={"arg0": 0, "arg1": 1, "arg2": 2},
    )
    assert selected.status == "selected", selected.reason
    assert selected.engine == "merlin_native" and selected.check_fingerprint
    assert selected.graph is not None and selected.graph.value(selected.graph.outputs[0]).symbol.startswith("i_")
