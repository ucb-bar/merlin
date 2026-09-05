"""The C runtime hands the model the descriptor MLIR's C interface actually reads.

`_mlir_ciface_<fn>` loads ``{ptr, ptr, i64, [rank x i64], [rank x i64]}`` -- packed to the
argument's OWN rank. ``merlin_descriptor_t`` reserves ``MERLIN_MAX_RANK`` slots per array so one
struct fits any rank, so writing ``d->strides[i]`` puts the strides where a rank-8 argument would
read them and every lower-rank argument read uninitialized stack instead. It stayed invisible for
models whose kernels recompute their own strides from static shapes, and faulted the moment one
materialized a descriptor (an unranked ``memrefCopy``, i.e. copying an input into a buffer).
"""
from __future__ import annotations

import subprocess

import pytest

from merlin.common.paths import repo_root

_FIXTURE = r"""
#include <stdint.h>
#include <stdio.h>
#include "merlin_model.h"

/* The generated trampoline, stubbed: record what the runtime handed us. */
static const int64_t *SEEN[2];
void merlin_invoke(void **descriptor_ptrs) {
  SEEN[0] = (const int64_t *)descriptor_ptrs[0];
  SEEN[1] = (const int64_t *)descriptor_ptrs[1];
}

int main(void) {
  static const merlin_arg_t args[2] = {
    {MERLIN_WEIGHT, 0L, 2, {3, 5}, 4},
    {MERLIN_OUTPUT, 0L, 1, {7}, 4},
  };
  char blob[64] = {0};
  float out[7];
  merlin_descriptor_t descs[2];
  merlin_run(args, 2, blob, 0, out, descs);

  /* rank 2: sizes at words 3,4 and strides at words 5,6 -- packed, not at 3+MERLIN_MAX_RANK. */
  const int64_t *w = SEEN[0];
  if (w[3] != 3 || w[4] != 5) { printf("sizes %lld %lld\n", (long long)w[3], (long long)w[4]); return 1; }
  if (w[5] != 5 || w[6] != 1) { printf("strides %lld %lld\n", (long long)w[5], (long long)w[6]); return 2; }
  /* rank 1: size at word 3, stride at word 4. */
  const int64_t *v = SEEN[1];
  if (v[3] != 7 || v[4] != 1) { printf("rank1 %lld %lld\n", (long long)v[3], (long long)v[4]); return 3; }
  printf("ok\n");
  return 0;
}
"""


@pytest.mark.timeout(120)
def test_descriptors_are_packed_to_each_argument_rank(tmp_path):
    rt = repo_root() / "merlin" / "runtime" / "c"
    src = tmp_path / "fixture.c"
    src.write_text(_FIXTURE, encoding="utf-8")
    exe = tmp_path / "fixture"
    build = subprocess.run(
        ["cc", "-std=c11", "-O1", f"-I{rt}", str(src), str(rt / "merlin_model.c"), "-o", str(exe)],
        capture_output=True, text=True)
    if build.returncode != 0 and "cc: not found" in (build.stderr or ""):
        pytest.skip("no host C compiler")
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"rc={run.returncode} {run.stdout}{run.stderr}"
    assert run.stdout.strip() == "ok"
