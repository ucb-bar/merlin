"""Wrapping Core ATen contractions may widen only with an exact integer readout."""

import numpy as np
import pytest

from merlin.common.mlir_query import parse
from merlin.common.paths import repo_root
from merlin.kernels.shapes import observe_contractions
from merlin.llvmlower.modular_contractions import prepare_bundle, widen_module
from merlin.system.offload import device_dtype_triples
from merlin.targetgen.core_aten_device import selected_facts

SOURCE = """builtin.module {
func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi8> {
 %z = arith.constant 0 : i8
 %init = tensor.splat %z : tensor<2x4xi8>
 %r = linalg.matmul {prov.region_id = "source_mm", prov.aten = "aten.mm.default"}
      ins(%a, %b : tensor<2x3xi8>, tensor<3x4xi8>) outs(%init : tensor<2x4xi8>) -> tensor<2x4xi8>
 func.return %r : tensor<2x4xi8>
}
}"""


def test_proved_widening_keeps_original_abi_and_wrap_readout(tmp_path):
    source = tmp_path / "model.mlir"
    source.write_text(SOURCE)
    facts = repo_root() / "merlin/tests/fixtures/core_aten_device/facts.json"
    with selected_facts("synthetic_core_aten", facts):
        receipt = prepare_bundle(tmp_path, device_dtype_triples("synthetic_core_aten"))
    assert receipt["widened_contractions"] == 1
    assert (tmp_path / "model.before_modular_widening.mlir").read_text() == SOURCE
    module = parse(source)
    module.verify()
    assert [shape.dtypes for _, shape in observe_contractions(module)] == [("i8", "i8", "i32")]
    assert "arith.trunci" in str(module) and 'prov.aten = "aten.mm.default"' in str(module)
    forward = next(op for op in module.walk() if op.name == "func.func")
    assert str(forward.function_type.outputs.data[0]) == "tensor<2x4xi8>"
    assert widen_module(module, [("i8", "i8", "i32")]) == 0


@pytest.mark.parametrize("change", ["nonzero", "overflow", "wrong_body", "no_datapath", "ambiguous"])
def test_near_misses_are_not_widened(change):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import StringAttr

    source = SOURCE.replace("constant 0", "constant 1") if change == "nonzero" else SOURCE
    module = parse(source)
    mul = next(op for op in module.walk() if op.name == "arith.muli")
    if change == "overflow":
        mul.properties["overflowFlags"] = arith.IntegerOverflowAttr([arith.IntegerOverflowFlag.NSW])
    if change == "wrong_body":
        mul.attributes["unknown_numeric_policy"] = StringAttr("other")
    triples = [] if change == "no_datapath" else [("i8", "i8", "i32")]
    if change == "ambiguous":
        triples.append(("i8", "i8", "i64"))
    before = str(module)
    assert widen_module(module, triples) == 0 and str(module) == before


def test_wider_accumulator_overflow_still_preserves_original_wrapping_bits():
    # Exhaustive source bytes and a reduction long enough to overflow the wider
    # accumulator: independent arithmetic, no corpus outputs or sampled answer key.
    values = np.arange(-128, 128, dtype=np.int64)
    product = values[:, None] * values[None, :]
    for length in (1, 31, 32, 262145):
        exact = product * length
        narrow = ((exact + 128) % 256 - 128).astype(np.int8)
        wrapped_wide = ((exact + (1 << 31)) % (1 << 32) - (1 << 31)).astype(np.int32)
        np.testing.assert_array_equal(narrow, wrapped_wide.astype(np.int8))


def test_batched_and_partial_shapes_preserve_source_maps():
    source = """builtin.module {
func.func @forward(%a: tensor<2x2x3xi8>, %b: tensor<2x3x4xi8>) -> tensor<2x2x4xi8> {
 %z = arith.constant 0 : i8
 %init = tensor.splat %z : tensor<2x2x4xi8>
 %r = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3) -> (d0,d1,d3)>,
    affine_map<(d0,d1,d2,d3) -> (d0,d3,d2)>, affine_map<(d0,d1,d2,d3) -> (d0,d1,d2)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<2x2x3xi8>, tensor<2x3x4xi8>) outs(%init : tensor<2x2x4xi8>) {
 ^bb0(%a0: i8, %b0: i8, %acc: i8):
   %product = arith.muli %a0, %b0 : i8
   %sum = arith.addi %product, %acc : i8
   linalg.yield %sum : i8
 } -> tensor<2x2x4xi8>
 func.return %r : tensor<2x2x4xi8>
}
}"""
    module = parse(source)
    before = next(op for op, _ in observe_contractions(module)).get_indexing_maps()
    assert widen_module(module, [("i8", "i8", "i32")]) == 1
    after = next(op for op, _ in observe_contractions(module))
    assert after.get_indexing_maps() == before
    assert str(after.results[0].type) == "tensor<2x2x4xi32>"


def test_wrapping_result_matches_pytorch_when_available():
    torch = pytest.importorskip("torch")
    for reduction in (3, 31, 32):
        a = np.arange(2 * reduction).astype(np.int8).reshape(2, reduction)
        b = (np.arange(reduction * 4) * 127).astype(np.int8).reshape(reduction, 4)
        expected = torch.ops.aten.mm.default(torch.from_numpy(a), torch.from_numpy(b)).numpy()
        wide = a.astype(np.int32) @ b.astype(np.int32)
        np.testing.assert_array_equal(wide.astype(np.int8), expected)
