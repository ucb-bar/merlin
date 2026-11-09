"""Compile and execute C dispatch with real arena reuse and error handling."""

import json
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.compile.dispatch_c import DispatchCall, emit_program_c
from merlin.compiler.schedule.arena import ArenaRequirements
from merlin.runtime.program import build_program
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node
from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest


def program():
    graph = DispatchProgram(
        "forward",
        [0],
        {
            key: Buffer(key, [4], "i8", "arg" if key == "x" else "intermediate", arg_index=0 if key == "x" else None)
            for key in ("x", "a", "y", "c", "z")
        },
        [
            Node("dispatch", op, ins, [out])
            for op, ins, out in (
                ("cpu", ["x"], "a"),
                ("device", ["a"], "y"),
                ("cpu", ["x"], "c"),
                ("device", ["c"], "z"),
            )
        ],
        ["y", "z"],
    )
    return build_program(
        graph,
        memory_requirements=ArenaRequirements(
            8,
            (0, 1, 2, 3),
            (0, 1, 2, 3),
            alignment=1,
            reserved=((0, 4),),
            dispatch_digest=dispatch_digest(graph),
        ),
    )


def calls():
    return {
        i: DispatchCall("add" if i % 2 == 0 else "double_value", "cpu" if i % 2 == 0 else "device") for i in range(4)
    }


def test_compiled_dispatch_consumes_offsets_and_preserves_guards(tmp_path):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    graph = program()
    report, files = emit_program_c(graph, "model", calls())
    assert graph.memory_plan.offsets == {"a": 4, "c": 4}
    assert report["outputs"] == ["y", "z"]
    assert all("buffer_alignment" not in call for call in report["calls"])
    assert all("alignment" not in buffer for buffer in report["buffers"])
    assert "arena_alignment" not in report
    json.dumps(report)
    for member, text in files.items():
        (tmp_path / member).write_text(text)
    (tmp_path / "check.c").write_text(r"""
#include "model.h"
#include <assert.h>
#include <string.h>
static unsigned char *expected;
static int failure, adds, doubles;
int add(void *ctx, const void *const *in, void *const *out) {
  (void)ctx; ++adds; assert(out[0] == expected);
  for (unsigned i = 0; i < 4; ++i) ((unsigned char *)out[0])[i] = ((const unsigned char *)in[0])[i] + 1;
  return 0;
}
int double_value(void *ctx, const void *const *in, void *const *out) {
  (void)ctx; ++doubles; assert(in[0] == expected);
  if (failure) return 37;
  for (unsigned i = 0; i < 4; ++i) ((unsigned char *)out[0])[i] = 2 * ((const unsigned char *)in[0])[i];
  return 0;
}
int main(void) {
  unsigned char arena[10], input[4], y[6], z[6];
  model_buffer buffers[] = {{input,4}, {y+1,4}, {z+1,4}};
  model_state state = {0}; expected = arena + 5;
  memset(arena, 0xcd, sizeof(arena));
  for (unsigned trial = 0; trial < 2; ++trial) {
    memset(input, trial, 4); memset(y,0xee,6); memset(z,0xee,6);
    assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_OK);
    assert(!state.active && state.completed == 4 && !state.kernel_status);
    for (unsigned i = 0; i < 4; ++i) assert(input[i] == trial && y[i+1] == 2*(trial+1) && z[i+1] == y[i+1]);
    assert(y[0] == 0xee && y[5] == 0xee && z[0] == 0xee && z[5] == 0xee);
    for (unsigned i = 0; i < 5; ++i) assert(arena[i] == 0xcd);
    assert(arena[9] == 0xcd);
  }
  assert(model_run(&state, arena+1, 7, buffers, 3, 0) == MODEL_INVALID);
  buffers[2].data = y+1;
  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_INVALID);
  buffers[2].data = z+1;
  model_state before; memcpy(&before, &state, sizeof(state));
  buffers[0].data = &state.active;
  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_INVALID);
  assert(memcmp(&state, &before, sizeof(state)) == 0 && adds == 4);
  buffers[0].data = input;
  buffers[2].data = &state.completed;
  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_INVALID);
  assert(memcmp(&state, &before, sizeof(state)) == 0 && doubles == 4);
  buffers[2].data = z+1;
  union { model_state state; unsigned char arena[sizeof(model_state) + 8]; } shared = {0};
  assert(model_run(&shared.state, shared.arena, 8, buffers, 3, 0) == MODEL_INVALID);
  assert(!shared.state.active && adds == 4);
  failure = 1;
  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_KERNEL_ERROR);
  assert(state.active && state.completed == 1 && state.kernel_status == 37);
  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_BUSY);
  assert(adds == 5 && doubles == 5);
  return 0;
}
""")
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "model.c", "check.c", "-o", "check"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "offset",
        "size",
        "extent",
        "graph",
        "binding",
        "async",
        "context_negative",
        "context_bool",
        "alignment_zero",
        "alignment_bool",
        "alignment_nonpower",
    ],
)
def test_export_rejects_changed_plan_or_missing_implementation(mutation):
    graph, bindings = program(), calls()
    if mutation == "offset":
        graph.memory_plan.offsets["a"] = 0  # overlaps the reserved area
    elif mutation == "size":
        graph.memory_plan.sizes["a"] = 2
    elif mutation == "extent":
        graph.memory_plan.arena_bytes = 7
    elif mutation == "graph":
        graph.dispatch.nodes[0].op = "different"
    elif mutation == "binding":
        bindings.pop(2)
    elif mutation.startswith("context"):
        bindings[0] = replace(bindings[0], context_index=-1 if mutation == "context_negative" else True)
    elif mutation.startswith("alignment"):
        alignment = {"alignment_zero": 0, "alignment_bool": True, "alignment_nonpower": 3}[mutation]
        bindings[0] = replace(bindings[0], buffer_alignment=alignment)
    else:
        bindings[0] = replace(bindings[0], completion="launch_only")
    with pytest.raises(ValueError):
        emit_program_c(graph, "model", bindings)


def test_view_alias_requires_an_explicit_safe_binding():
    graph = program()
    graph.dispatch.nodes[1] = Node("view", "slice", ["a"], ["c"])
    graph.dispatch.nodes = graph.dispatch.nodes[:2] + [Node("dispatch", "copy", ["c"], ["y"])]
    graph.dispatch.results = ["y"]
    del graph.dispatch.buffers["z"]
    req = ArenaRequirements(
        8, (0, 1, 2), (0, 1, 2), alignment=1, aliases=(("c", "a", 0),), dispatch_digest=dispatch_digest(graph.dispatch)
    )
    graph = build_program(graph.dispatch, memory_requirements=req)
    bindings = {i: DispatchCall("compiled", "cpu") for i in range(3)}
    with pytest.raises(ValueError, match="alias-safe"):
        emit_program_c(graph, "model", bindings)
    bindings[1] = replace(bindings[1], alias_safe=True)
    emit_program_c(graph, "model", bindings)


def test_large_signature_keeps_validation_compact_and_checks_late_buffers(tmp_path):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    inputs = [f"x{i}" for i in range(700)]
    buffers = {key: Buffer(key, [4], "i8", "arg", arg_index=i) for i, key in enumerate(inputs)}
    buffers["y"] = Buffer("y", [4], "i8", "intermediate")
    graph = DispatchProgram("forward", list(range(700)), buffers, [Node("dispatch", "copy", inputs, ["y"])], ["y"])
    bounded = build_program(
        graph,
        memory_requirements=ArenaRequirements(1, (0,), (1,), alignment=1, dispatch_digest=dispatch_digest(graph)),
    )
    _, files = emit_program_c(
        bounded, "large", {0: DispatchCall("copy_last", "cpu")}, read_only_inputs=tuple(inputs[:500])
    )
    assert len(files["large.c"]) < 100_000
    for member, text in files.items():
        (tmp_path / member).write_text(text)
    (tmp_path / "check.c").write_text(r"""
#include "large.h"
#include <assert.h>
#include <string.h>
static unsigned calls;
int copy_last(void *context, const void *const *in, void *const *out) {
  (void)context; ++calls; memcpy(out[0],in[699],4); return 0;
}
int main(void) {
  unsigned char input[700][4], output[4]={0};
  large_buffer buffers[701]; large_state state={0};
  memset(input,0x29,sizeof(input));
  for (unsigned i=0;i<700;++i) buffers[i]=(large_buffer){input[i],4};
  buffers[700]=(large_buffer){output,4};
  buffers[1].data=input[0]; /* Explicitly read-only inputs may overlap. */
  assert(large_run(&state,0,0,buffers,701,0)==LARGE_OK && calls==1);
  assert(!memcmp(output,input[699],4));
  large_state before=state;
  buffers[500].data=input[0];
  assert(large_run(&state,0,0,buffers,701,0)==LARGE_INVALID);
  buffers[500].data=input[500]; buffers[700].data=input[499];
  assert(large_run(&state,0,0,buffers,701,0)==LARGE_INVALID);
  buffers[700].data=output; buffers[699].data=(void *)(UINTPTR_MAX-1);
  assert(large_run(&state,0,0,buffers,701,0)==LARGE_INVALID);
  buffers[699].data=input[699]; buffers[699].bytes=3;
  assert(large_run(&state,0,0,buffers,701,0)==LARGE_INVALID);
  assert(calls==1 && !memcmp(&state,&before,sizeof(state)));
  buffers[699].bytes=4;
  assert(large_run(&state,0,0,buffers,701,0)==LARGE_OK && calls==2);
  return 0;
}
""")
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "large.c", "check.c", "-o", "check"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True, timeout=10)


def test_compiled_dispatch_checks_all_pointer_alignments_before_calls(tmp_path):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    original = program()
    bounded = build_program(
        original.dispatch,
        memory_requirements=replace(original.memory_requirements, reserved=()),
    )
    bindings = {
        i: replace(call, buffer_alignment=alignment)
        for (i, call), alignment in zip(calls().items(), (4, 16, 8, 4), strict=True)
    }
    report, files = emit_program_c(bounded, "aligned", bindings)
    assert bounded.memory_plan.offsets == {"a": 0, "c": 0}
    assert [(row["id"], row["alignment"]) for row in report["buffers"]] == [("x", 8), ("y", 16), ("z", 4)]
    assert report["arena_alignment"] == 16
    # A valid arena placement can still be incompatible with a compiled wrapper.
    with pytest.raises(ValueError, match="arena offset"):
        emit_program_c(original, "bad", bindings)
    for member, text in files.items():
        (tmp_path / member).write_text(text)
    (tmp_path / "check.c").write_text(r"""
#include "aligned.h"
#include <assert.h>
#include <string.h>
static unsigned calls;
int add(void *context, const void *const *in, void *const *out) {
  (void)context; ++calls;
  assert((uintptr_t)in[0]%8u==0 && (uintptr_t)out[0]%8u==0);
  memcpy(out[0],in[0],4); return 0;
}
int double_value(void *context, const void *const *in, void *const *out) {
  (void)context; ++calls;
  assert((uintptr_t)in[0]%4u==0 && (uintptr_t)out[0]%4u==0);
  memcpy(out[0],in[0],4); return 0;
}
int main(void) {
  _Alignas(16) unsigned char arena[32], input[32], y[32], z[32];
  memset(input,0x5a,sizeof input); memset(y,0xa5,sizeof y); memset(z,0xa5,sizeof z);
  aligned_buffer buffers[]={{input,4},{y,4},{z,4}};
  aligned_state state={0};
  assert(ALIGNED_ARENA_ALIGNMENT==16);
  assert(aligned_run(&state,arena,sizeof arena,buffers,3,0)==ALIGNED_OK);
  assert(calls==4 && !memcmp(input,y,4) && !memcmp(input,z,4));
  aligned_state before=state;
  unsigned char before_y[32], before_z[32]; memcpy(before_y,y,32); memcpy(before_z,z,32);
  buffers[0].data=input+4; /* Used by align4 and align8 calls: enforce max before either. */
  assert(aligned_run(&state,arena,sizeof arena,buffers,3,0)==ALIGNED_INVALID);
  buffers[0].data=input; buffers[1].data=y+8;
  assert(aligned_run(&state,arena,sizeof arena,buffers,3,0)==ALIGNED_INVALID);
  buffers[1].data=y; buffers[2].data=z+2;
  assert(aligned_run(&state,arena,sizeof arena,buffers,3,0)==ALIGNED_INVALID);
  buffers[2].data=z;
  assert(aligned_run(&state,arena+8,16,buffers,3,0)==ALIGNED_INVALID);
  assert(calls==4 && !memcmp(&state,&before,sizeof state));
  assert(!memcmp(y,before_y,32) && !memcmp(z,before_z,32));
  assert(aligned_run(&state,arena,sizeof arena,buffers,3,0)==ALIGNED_OK && calls==8);
  return 0;
}
""")
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "aligned.c", "check.c", "-o", "check"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True, timeout=10)
