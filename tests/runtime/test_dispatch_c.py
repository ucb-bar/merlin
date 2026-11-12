import json
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.compiler.lower.c.dispatch import DispatchCall, emit_program_c
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
            8, (0, 1, 2, 3), (0, 1, 2, 3), alignment=1, reserved=((0, 4),), dispatch_digest=dispatch_digest(graph)
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
    assert all(("buffer_alignment" not in call for call in report["calls"]))
    assert all(("alignment" not in buffer for buffer in report["buffers"]))
    assert "arena_alignment" not in report
    json.dumps(report)
    for member, text in files.items():
        (tmp_path / member).write_text(text)
    (tmp_path / "check.c").write_text(
        '\n#include "model.h"\n#include <assert.h>\n#include <string.h>\nstatic unsigned char *expected;\nstatic int failure, adds, doubles;\nint add(void *ctx, const void *const *in, void *const *out) {\n  (void)ctx; ++adds; assert(out[0] == expected);\n  for (unsigned i = 0; i < 4; ++i) ((unsigned char *)out[0])[i] = ((const unsigned char *)in[0])[i] + 1;\n  return 0;\n}\nint double_value(void *ctx, const void *const *in, void *const *out) {\n  (void)ctx; ++doubles; assert(in[0] == expected);\n  if (failure) return 37;\n  for (unsigned i = 0; i < 4; ++i) ((unsigned char *)out[0])[i] = 2 * ((const unsigned char *)in[0])[i];\n  return 0;\n}\nint main(void) {\n  unsigned char arena[10], input[4], y[6], z[6];\n  model_buffer buffers[] = {{input,4}, {y+1,4}, {z+1,4}};\n  model_state state = {0}; expected = arena + 5;\n  memset(arena, 0xcd, sizeof(arena));\n  for (unsigned trial = 0; trial < 2; ++trial) {\n    memset(input, trial, 4); memset(y,0xee,6); memset(z,0xee,6);\n    assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_OK);\n    assert(!state.active && state.completed == 4 && !state.kernel_status);\n    for (unsigned i = 0; i < 4; ++i) assert(input[i] == trial && y[i+1] == 2*(trial+1) && z[i+1] == y[i+1]);\n    assert(y[0] == 0xee && y[5] == 0xee && z[0] == 0xee && z[5] == 0xee);\n    for (unsigned i = 0; i < 5; ++i) assert(arena[i] == 0xcd);\n    assert(arena[9] == 0xcd);\n  }\n  assert(model_run(&state, arena+1, 7, buffers, 3, 0) == MODEL_INVALID);\n  buffers[2].data = y+1;\n  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_INVALID);\n  buffers[2].data = z+1;\n  model_state before; memcpy(&before, &state, sizeof(state));\n  buffers[0].data = &state.active;\n  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_INVALID);\n  assert(memcmp(&state, &before, sizeof(state)) == 0 && adds == 4);\n  buffers[0].data = input;\n  buffers[2].data = &state.completed;\n  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_INVALID);\n  assert(memcmp(&state, &before, sizeof(state)) == 0 && doubles == 4);\n  buffers[2].data = z+1;\n  union { model_state state; unsigned char arena[sizeof(model_state) + 8]; } shared = {0};\n  assert(model_run(&shared.state, shared.arena, 8, buffers, 3, 0) == MODEL_INVALID);\n  assert(!shared.state.active && adds == 4);\n  failure = 1;\n  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_KERNEL_ERROR);\n  assert(state.active && state.completed == 1 && state.kernel_status == 37);\n  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_BUSY);\n  assert(adds == 5 && doubles == 5);\n  return 0;\n}\n'
    )
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "model.c", "check.c", "-o", "check"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)


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
