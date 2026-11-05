"""Execute a fused source graph through checked outlining and bounded C dispatch."""

import shutil
import subprocess

import pytest

from merlin.compile.dispatch_c import DispatchCall, emit_program_c
from merlin.compile.semantic_regions import outline_semantic_regions
from merlin.runtime.program import build_program
from merlin.tests.ir.test_semantic_regions import graph, plan_for
from merlin.xdsl_dialects.lowering.arena_plan import ArenaRequirements
from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest


def test_fused_multioutput_calls_use_distinct_contexts_and_preserve_buffers(tmp_path):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler required")
    source = graph()
    outlined = outline_semantic_regions(source, entry="forward", plan=plan_for(source, ((0, 2), (1,), (3,))))

    def requirements(dispatch):
        return ArenaRequirements(
            80, (0, 1, 2), (0, 1, 2), alignment=16, reserved=((0, 16),), dispatch_digest=dispatch_digest(dispatch)
        )

    program = build_program(
        outlined.logical, global_plan=outlined.plan, global_plan_emitter=outlined, memory_requirements=requirements
    )
    report, files = emit_program_c(
        program, "application", {i: DispatchCall(f"kernel_{i}", "device", context_index=i) for i in range(3)}
    )
    assert report["inputs"] == ["z", "x"]
    assert report["outputs"] == ["d", "b", "d", "x"]
    assert report["global_plan_emission"] == outlined.emission.receipt()
    assert [row["id"] for row in report["buffers"]] == ["z", "x", "one", "d", "b"]
    for filename, text in files.items():
        (tmp_path / filename).write_text(text)
    (tmp_path / "check.c").write_text(r"""
#include "application.h"
#include <assert.h>
#include <string.h>
static int keys[3], fail;
int kernel_0(void *ctx, const void *const *in, void *const *out) {
  assert(ctx == keys);
  for (int i=0;i<4;++i) {
    ((int *)out[0])[i]=((const int *)in[0])[i]+((const int *)in[2])[i];
    ((int *)out[1])[i]=((const int *)in[1])[i]+((const int *)in[2])[i];
  }
  return 0;
}
int kernel_1(void *ctx, const void *const *in, void *const *out) {
  assert(ctx == keys+1);
  if (fail) return 29;
  for (int i=0;i<4;++i) ((int *)out[0])[i]=((const int *)in[0])[i]*((const int *)in[0])[i];
  return 0;
}
int kernel_2(void *ctx, const void *const *in, void *const *out) {
  assert(ctx == keys+2);
  for (int i=0;i<4;++i) ((int *)out[0])[i]=((const int *)in[0])[i]+((const int *)in[1])[i];
  return 0;
}
int main(void) {
  _Alignas(16) unsigned char arena[APPLICATION_ARENA_BYTES+16];
  int x[4], z[4], one[4]={1,1,1,1}, d[6], b[6];
  void *items[]={keys,keys+1,keys+2}; application_contexts contexts={items,3};
  application_state state={0};
  application_buffer buffers[]={{z,16},{x,16},{one,16},{d+1,16},{b+1,16}};
  for (int trial=0;trial<3;++trial) {
    for (int i=0;i<4;++i) { x[i]=trial+i; z[i]=7-i; }
    memset(arena,0xcd,sizeof(arena)); memset(d,0xee,sizeof(d)); memset(b,0xee,sizeof(b));
    assert(application_run(&state,arena,APPLICATION_ARENA_BYTES,buffers,5,0)==APPLICATION_INVALID);
    contexts.count=2;
    assert(application_run(&state,arena,APPLICATION_ARENA_BYTES,buffers,5,&contexts)==APPLICATION_INVALID);
    assert(!state.active); contexts.count=3;
    assert(application_run(&state,arena,APPLICATION_ARENA_BYTES,buffers,5,&contexts)==APPLICATION_OK);
    assert(!state.active && state.completed==3);
    for (int i=0;i<4;++i) {
      assert(x[i]==trial+i && z[i]==7-i && one[i]==1);
      assert(b[i+1]==(x[i]+1)*(x[i]+1) && d[i+1]==b[i+1]+z[i]+1);
    }
    for (unsigned i=0;i<16;++i) {
      assert(arena[i]==0xcd && arena[APPLICATION_ARENA_BYTES+i]==0xcd);
    }
    assert((unsigned)d[0]==0xeeeeeeeeu && (unsigned)d[5]==0xeeeeeeeeu);
    assert((unsigned)b[0]==0xeeeeeeeeu && (unsigned)b[5]==0xeeeeeeeeu);
  }
  fail=1;
  assert(application_run(&state,arena,APPLICATION_ARENA_BYTES,buffers,5,&contexts)==APPLICATION_KERNEL_ERROR);
  assert(state.active && state.completed==1 && state.kernel_status==29);
  assert(application_run(&state,arena,APPLICATION_ARENA_BYTES,buffers,5,&contexts)==APPLICATION_BUSY);
  return 0;
}
""")
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "application.c", "check.c", "-o", "check"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)
