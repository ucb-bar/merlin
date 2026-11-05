"""Real scalar lowering must remove a named transpose without changing outputs."""

import numpy as np

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.optional_passes import Selection, applied

SOURCE = """
module {
 func.func @forward(%a: tensor<3x5xf32>, %b: tensor<5x3xf32>) -> tensor<5x3xf32> {
  %e = tensor.empty() : tensor<5x3xf32>
  %t = linalg.transpose ins(%a : tensor<3x5xf32>) outs(%e : tensor<5x3xf32>) permutation = [1, 0]
  %e2 = tensor.empty() : tensor<5x3xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%t, %b : tensor<5x3xf32>, tensor<5x3xf32>) outs(%e2 : tensor<5x3xf32>) {
   ^bb0(%x: f32, %y: f32, %old: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<5x3xf32>
  return %r : tensor<5x3xf32>
 }
}
"""


def test_second_fusion_eliminates_materialization_with_exact_guarded_native_results(tmp_path):
    a = (np.arange(15, dtype=np.float32).reshape(3, 5) - 7) / 8
    b = np.linspace(-2, 2, 15, dtype=np.float32).reshape(5, 3)
    expected = a.T + b
    store_loops = []
    for name, selected in (("control", "-host-elementwise-fusion"), ("fusion", "host-elementwise-fusion")):
        # Isolate the existing upstream fusion mechanism from the independent
        # automatic layout planner, which can also remove this small transpose.
        with applied(Selection.parse("-automatic-host-permutations," + selected)):
            result = lower_model(SOURCE, tmp_path / name)
        llvm = result.ll_path.read_text()
        store_loops.append(llvm.count("store float "))
        invoke = HostModel.load(str(result.host_so))
        guarded = np.full(15 + 16, 93, dtype=np.float32)
        output = guarded[8:-8].reshape(5, 3)
        for _ in range(3):
            invoke([(a.ctypes.data, a.shape), (b.ctypes.data, b.shape), (output.ctypes.data, output.shape)])
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
            np.testing.assert_array_equal(guarded[:8], np.full(8, 93, dtype=np.float32))
            np.testing.assert_array_equal(guarded[-8:], np.full(8, 93, dtype=np.float32))
    # Bufferization can reuse the output for the first transpose. The waste is
    # the extra complete store/read loop even when no extra malloc is emitted.
    assert store_loops == [2, 1]
