"""The C runtime's memref descriptors must be laid out the way MLIR reads them FOR THEIR RANK.

MLIR's C interface reads a rank-r descriptor as the packed ``{allocated, aligned, offset, sizes[r],
strides[r]}``. ``merlin_descriptor_t`` reserves ``MERLIN_MAX_RANK`` slots for each, so a field-wise
``d->strides[i]`` for r < MAX lands where MLIR reads ``sizes[r + i]``... and MLIR reads the strides out
of the unused size slots: zeros. Generated code with static shapes never looks, which is how it went
unnoticed; an external kernel's shim does (MEASURED on ``mlp_basic_triton``: x arrived as sizes [4, 16],
strides [0, 0]), and so does ``memrefCopy`` in ``abi/mlir_runtime.c`` on any runtime-built descriptor.

This compiles ``merlin_model.c`` on the host with a ``merlin_invoke`` that reads each descriptor exactly
as MLIR would, so no simulator is needed.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir

_PROBE = r"""
#include <stdio.h>
#include "merlin_model.h"

static const merlin_arg_t ARGS[3] = {
  {MERLIN_INPUT, 0L, 2, {4, 16}, 4},
  {MERLIN_INPUT, 1L, 1, {32}, 4},
  {MERLIN_OUTPUT, 0L, 3, {2, 3, 5}, 4},
};
static const int RANK[3] = {2, 1, 3};

/* Read each descriptor the way MLIR's C interface does for its rank: packed sizes then strides. */
void merlin_invoke(void **d) {
  for (int a = 0; a < 3; a++) {
    const int64_t *rest = (const int64_t *)((char *)d[a] + 2 * sizeof(void *));
    int r = RANK[a];
    printf("arg%d sizes", a);
    for (int i = 0; i < r; i++) printf(" %lld", (long long)rest[1 + i]);
    printf(" strides");
    for (int i = 0; i < r; i++) printf(" %lld", (long long)rest[1 + r + i]);
    printf("\n");
  }
}

int main(void) {
  static float in0[64], in1[32], out[30];
  void *ptrs[3] = {in0, in1, 0};
  merlin_descriptor_t descs[3];
  merlin_run(ARGS, 3, 0, ptrs, out, descs);
  return 0;
}
"""


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")
def test_descriptors_are_packed_for_their_rank(tmp_path):
    rt = runtime_dir() / "c"
    src = tmp_path / "probe.c"
    src.write_text(_PROBE, encoding="utf-8")
    exe = tmp_path / "probe"
    subprocess.run(
        ["cc", "-O1", "-I", str(rt), str(src), str(rt / "merlin_model.c"), "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.splitlines()
    assert lines == ["arg0 sizes 4 16 strides 16 1", "arg1 sizes 32 strides 1", "arg2 sizes 2 3 5 strides 15 5 1"]
