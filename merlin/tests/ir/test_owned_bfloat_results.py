"""Owned-result feature composition retains initial per-operation BF16 rounding."""

import numpy as np
import pytest

from merlin.llvmlower import toolchain
from merlin.llvmlower.owned_results import compile_owned_results


@pytest.mark.skipif(not toolchain.available(), reason="existing native compiler required")
def test_owned_result_grid_normalization_follows_bfloat_source_legalization(tmp_path):
    source = """module {func.func @forward(%x:tensor<4xbf16>)->tensor<4xbf16> {
      %empty=tensor.empty():tensor<4xbf16>
      %out=linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}
        ins(%x:tensor<4xbf16>) outs(%empty:tensor<4xbf16>) {
        ^bb0(%value:bf16,%unused:bf16):
          %one=arith.constant 1.0:bf16
          %first=arith.addf %value,%one:bf16
          %second=arith.addf %first,%one:bf16
          linalg.yield %second:bf16
      } -> tensor<4xbf16>
      return %out:tensor<4xbf16>
    }}"""
    program = compile_owned_results(
        source,
        workdir=tmp_path / "native",
        timeout=30,
        features=frozenset(
            {"emulate_bfloat_elementwise", "preserve_bfloat_transport", "reuse_loop_tensor_destinations"}
        ),
    )
    words = np.asarray([0x3F80, 0xBF80, 0x0000, 0x4380], np.uint16)
    (actual,), observation = program.execute((words,))
    # 256 + 1 rounds to 256 twice; postponing rounding would produce 258.
    assert actual.dtype == np.uint16 and actual.tolist() == [0x4040, 0x3F80, 0x4000, 0x4380]
    assert words.tolist() == [0x3F80, 0xBF80, 0x0000, 0x4380] and observation["source_calls"] == 1
