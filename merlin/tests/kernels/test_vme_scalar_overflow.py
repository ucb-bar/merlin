"""The VME scalar twin must match the hardware's wrapping i32 accumulator."""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from merlin.kernels.vme_kernel import ROLES, OperandLayout, VmeKernelSpec, emit_kernel


def test_vme_scalar_twin_wraps_signed_endpoint_dot_without_undefined_behavior(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C compiler unavailable")

    spec = VmeKernelSpec(
        roles={role: role for role in ROLES},
        operand_type=(8, 4),
        readout_type=(32, 1),
        kstep=2,
        layout=OperandLayout(0, 3, 8, 3, 0, 27, 4, 24, 0, 29, 2),
    )
    encoding = SimpleNamespace(match=0, mask=0, fill_mask=0, insn_r=lambda *args: ".word 0")
    source = emit_kernel({role: encoding for role in ROLES}, spec, derivation_ok=True)
    # The scalar build excludes all dummy instruction words. K=131073 makes
    # (-128 * -128) * K exceed INT32_MAX by 16385, and UBSan catches signed
    # overflow in a twin that accidentally uses an int32 accumulator.
    program = (
        source
        + """
#include <stdio.h>
int main(void) {
  enum { K = 131073 };
  static int8_t a[K], b[K];
  int32_t result = 0;
  for (size_t i = 0; i < K; ++i) a[i] = b[i] = -128;
  merlin_vme_gemm_i8(&result, a, b, NULL, 1, 1, K);
  printf("%u\\n", (uint32_t)result);
  return 0;
}
"""
    )
    c_file = tmp_path / "vme_scalar.c"
    binary = tmp_path / "vme_scalar"
    c_file.write_text(program, encoding="utf-8")
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-fsanitize=signed-integer-overflow",
            "-fno-sanitize-recover=all",
            "-DMERLIN_MATRIX_SCALAR_TILE",
            "-DMERLIN_MATRIX_TILE_EDGE=64",
            str(c_file),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == str((128 * 128 * 131073) & 0xFFFFFFFF)
