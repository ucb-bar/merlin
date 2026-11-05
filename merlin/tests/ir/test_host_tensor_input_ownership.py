"""Functional tensor updates must not overwrite caller-owned host inputs."""

import ctypes

import pytest

from merlin.llvmlower import toolchain

pytestmark = pytest.mark.skipif(not toolchain.available(), reason="select the native LLVM toolchain")

SOURCE = """
module {
  func.func @forward(%cache: tensor<4xf32>, %value: tensor<1xf32>,
                     %position: tensor<1xi64>) -> (tensor<4xf32>, tensor<1xf32>) {
    %zero = arith.constant 0 : index
    %raw = tensor.extract %position[%zero] : tensor<1xi64>
    %index = arith.index_cast %raw : i64 to index
    %updated = "tensor.insert_slice"(%value, %cache, %index) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>,
      static_offsets = array<i64: -9223372036854775808>,
      static_sizes = array<i64: 1>, static_strides = array<i64: 1>
    }> : (tensor<1xf32>, tensor<4xf32>, index) -> tensor<4xf32>
    return %updated, %value : tensor<4xf32>, tensor<1xf32>
  }
}
"""


@pytest.mark.parametrize("textual", [False, True])
@pytest.mark.parametrize("features", [(), ("vectorized_transcendental_activation",)])
def test_functional_update_preserves_inputs_and_all_results(tmp_path, textual, features):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    built = lower_model(SOURCE, tmp_path, targets=("host",), textual=textual, features=features)
    call = HostModel.load(str(built.host_so))
    for position, value in ((2, -1.25), (0, 7.5)):
        cache = (ctypes.c_float * 6)(91, 10, 20, 30, 40, 93)
        update = (ctypes.c_float * 3)(95, value, 97)
        index = (ctypes.c_int64 * 1)(position)
        output = (ctypes.c_float * 6)(*[101] * 6)
        echoed = (ctypes.c_float * 3)(*[103] * 3)
        before = bytes(cache), bytes(update), bytes(index)
        call(
            [
                (ctypes.addressof(cache) + 4, (4,)),
                (ctypes.addressof(update) + 4, (1,)),
                (ctypes.addressof(index), (1,)),
                (ctypes.addressof(output) + 4, (4,)),
                (ctypes.addressof(echoed) + 4, (1,)),
            ]
        )
        assert (bytes(cache), bytes(update), bytes(index)) == before
        expected = [10, 20, 30, 40]
        expected[position] = value
        assert list(output) == [101, *expected, 101]
        assert list(echoed) == [103, value, 103]


def test_explicit_memref_store_keeps_its_mutable_abi(tmp_path):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    source = """module {
      func.func @forward(%buffer: memref<4xf32>) {
        %index = arith.constant 1 : index
        %value = arith.constant 3.25 : f32
        memref.store %value, %buffer[%index] : memref<4xf32>
        return
      }
    }"""
    built = lower_model(source, tmp_path, targets=("host",))
    values = (ctypes.c_float * 4)(10, 20, 30, 40)
    HostModel.load(str(built.host_so))([(ctypes.addressof(values), (4,))])
    assert list(values) == [10, 3.25, 30, 40]
