"""Scalar storage wrappers may be views only when every coordinate is proved."""

import pytest
from mlir_fixtures import assert_unchanged, returned

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.structural_folds import fold_singleton_carriers


@pytest.mark.parametrize("dtype", ["i1", "i8", "i32", "i64", "f16", "bf16", "f32", "complex<f32>"])
def test_one_extracted_storage_word_reassembles_without_numeric_conversion(dtype):
    module = parse_mlir_text(f"""module {{func.func @forward(%x:tensor<{dtype}>) -> tensor<1x{dtype}> {{
      %v=tensor.extract %x[] : tensor<{dtype}>
      %r=tensor.from_elements %v : tensor<1x{dtype}>
      func.return %r : tensor<1x{dtype}>
    }} }}""")
    assert fold_singleton_carriers(module) == 1
    module.verify()
    assert returned(module).owner.name == "tensor.reshape"
    assert fold_singleton_carriers(module) == 0


def test_selecting_one_of_multiple_words_is_not_a_reshape():
    module = parse_mlir_text("""module {func.func @forward(%x:tensor<2xi64>) -> tensor<1xi64> {
      %i=arith.constant 0 : index
      %v=tensor.extract %x[%i] : tensor<2xi64>
      %r=tensor.from_elements %v : tensor<1xi64>
      func.return %r : tensor<1xi64>
    }}""")
    assert_unchanged(module, fold_singleton_carriers)
