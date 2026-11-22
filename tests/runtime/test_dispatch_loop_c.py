"""Execute a two-iteration generic C schedule against its unrolled authority."""
from dataclasses import replace
import shutil
import subprocess

import pytest

from merlin.compiler.lower.c.dispatch import DispatchCall, build_program, emit_program_c, prepare_dispatch_loop
from merlin.compiler.schedule.storage.arena import ArenaRequirements
from merlin.compiler.partition.dispatch_program import Buffer, DispatchProgram, Node
from merlin.compiler.partition.global_plan_emission import dispatch_digest


def fixture(count=6, capacity=64):
    buffers = {"x": Buffer("x", [4], "i8", "arg", 0),
               "constant": Buffer("constant", [4], "i8", "const")}
    nodes = []
    previous = "x"
    for i in range(count):
        key = f"v{i}"
        buffers[key] = Buffer(key, [4], "i8", "intermediate")
        nodes.append(Node("dispatch", f"op{i}", [previous, "constant"], [key]))
        previous = key
    graph = DispatchProgram("forward", [0], buffers, nodes, [previous])
    program = build_program(graph, memory_requirements=ArenaRequirements(
        capacity, tuple(2*i for i in range(count)), tuple(2*i+1 for i in range(count)),
        alignment=1, dispatch_digest=dispatch_digest(graph)))
    calls = {i: DispatchCall(f"variant{i}", "device" if i % 2 else "host", context_index=i)
             for i in range(count)}
    return program, calls


def test_two_iteration_compiled_loop_exact_pointers_variants_and_failure(tmp_path):
    compiler = shutil.which("cc")
    assert compiler, "a C compiler is required"
    program, calls = fixture()
    schedule = prepare_dispatch_loop(program, calls, offset=1, period=2, trip_count=2)
    flat, flat_files = emit_program_c(program, "flat", calls)
    loop, loop_files = emit_program_c(program, "counted", calls, loop=schedule)
    assert flat["calls"] == loop["calls"]
    assert flat["memory_plan"] == loop["memory_plan"]
    assert loop["loop"]["trip_count"] == 2
    for member, content in {**flat_files, **loop_files}.items():
        (tmp_path / member).write_text(content)
    # The call table is static storage: a board host declares no heap, so the loop must not allocate.
    assert not any(word in loop_files["counted.c"] for word in ("malloc", "free(", "stdlib"))
    (tmp_path / "check.c").write_text(r'''
#include "flat.h"
#include "counted.h"
#include <assert.h>
#include <string.h>
typedef struct { int variant; } Context;
typedef struct { int variant; void *context; const void *input, *constant; void *output; } Log;
static Context ctx[6]={{0},{1},{2},{3},{4},{5}};
static Log logs[6], expected[6];
static int used, fail=-1, admissions, decision=1;
static int execute(int i, void *c, const void *const *in, void *const *out) {
  assert(c == &ctx[i] && ((Context *)c)->variant == i);
  logs[used++]=(Log){i,c,in[0],in[1],out[0]};
  if (i==fail) return 37;
  for(unsigned j=0;j<4;++j) ((unsigned char *)out[0])[j]=
    ((const unsigned char *)in[0])[j]+((const unsigned char *)in[1])[j]+i;
  return 0;
}
#define WRAPPER(i) int variant##i(void *c,const void *const *in,void *const *out) { return execute(i,c,in,out); }
WRAPPER(0) WRAPPER(1) WRAPPER(2) WRAPPER(3) WRAPPER(4) WRAPPER(5)
static int admit(void *opaque, const counted_loop *loop) {
  assert(opaque==ctx && used==0);
  assert(loop->count==6 && loop->offset==1 && loop->period==2 && loop->trip_count==2);
  for (unsigned i=0;i<6;++i) {
    assert(loop->calls[i].context==&ctx[i]);
    assert(loop->calls[i].input_count==2 && loop->calls[i].output_count==1);
    assert(loop->calls[i].inputs[0]==expected[i].input);
    assert(loop->calls[i].inputs[1]==expected[i].constant);
    assert(loop->calls[i].outputs[0]==expected[i].output);
  }
  ++admissions; return decision;
}
int main(void) {
  unsigned char arena[64], input[4]={1,2,3,4}, constant[4]={7,6,5,4}, result[4], reference[4];
  void *contexts[6]; for(unsigned i=0;i<6;++i) contexts[i]=&ctx[i];
  flat_contexts flat_ctx={contexts,6}; counted_contexts loop_ctx={contexts,6};
  flat_buffer a[]={{input,4},{constant,4},{result,4}};
  counted_buffer b[]={{input,4},{constant,4},{result,4}};
  flat_state fs={0}; counted_state ls={0};
  memset(arena,0xcd,64);
  assert(flat_run(&fs,arena,64,a,3,&flat_ctx)==FLAT_OK);
  memcpy(expected,logs,sizeof(logs)); memcpy(reference,result,4);
  assert(fs.completed==6 && !fs.active);
  for (unsigned j=0;j<4;++j) assert(reference[j]==input[j]+6*constant[j]+15);
  for(int mode=0;mode<3;++mode) {
    used=0; memset(arena,0xcd,64); memset(result,0,4); ls=(counted_state){0};
    decision=mode==0?0:1;
    assert(counted_run_loop(&ls,arena,64,b,3,&loop_ctx,mode==2?0:admit,ctx)==COUNTED_OK);
    assert(used==6 && ls.completed==6 && !ls.active);
    assert(!memcmp(reference,result,4));
    for(int i=0;i<6;++i) {
      assert(logs[i].variant==expected[i].variant && logs[i].context==expected[i].context);
      assert(logs[i].input==expected[i].input && logs[i].constant==expected[i].constant && logs[i].output==expected[i].output);
    }
    for(unsigned j=0;j<4;++j) assert(input[j]==j+1 && constant[j]==7-j);
  }
  assert(admissions==2);
  used=0; decision=2; counted_state before=ls;
  assert(counted_run_loop(&ls,arena,64,b,3,&loop_ctx,admit,ctx)==COUNTED_INVALID);
  assert(!used && !memcmp(&before,&ls,sizeof(ls)));
  for(int mode=0;mode<2;++mode) {
    used=0; fail=4; decision=mode; ls=(counted_state){0};
    assert(counted_run_loop(&ls,arena,64,b,3,&loop_ctx,admit,ctx)==COUNTED_KERNEL_ERROR);
    assert(ls.active && ls.completed==4 && ls.kernel_status==37 && used==5);
    assert(counted_run_loop(&ls,arena,64,b,3,&loop_ctx,admit,ctx)==COUNTED_BUSY && used==5);
  }
  return 0;
}
''')
    subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "flat.c", "counted.c", "check.c", "-o", "check"],
                   cwd=tmp_path, check=True, capture_output=True, timeout=60)
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True, timeout=10)


def test_large_loop_has_bounded_stack_frame(tmp_path):
    compiler = shutil.which("cc")
    assert compiler, "a C compiler is required"
    program, calls = fixture(count=2048, capacity=2048*64)
    schedule = prepare_dispatch_loop(program, calls, offset=0, period=1, trip_count=2048)
    _, files = emit_program_c(program, "counted", calls, loop=schedule)
    for member, content in files.items():
        (tmp_path / member).write_text(content)
    # The original stack table alone exceeded 96 KiB at this scale. Check the
    # unoptimized frame so scalar replacement cannot hide an automatic table.
    subprocess.run([compiler, "-std=c11", "-O0", "-Wall", "-Wextra", "-Werror",
                    "-Wframe-larger-than=4096", "-c", "counted.c", "-o", "counted.o"],
                   cwd=tmp_path, check=True, capture_output=True, timeout=60)


def planned():
    """Two repeated regions emitted from a selected plan, as the application render builds them."""
    from merlin.compiler.partition.semantic_regions import outline_semantic_regions
    from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
    from semantic_application_fixtures import plan_for

    tensor = TensorType((1,), "f32", "raw-bit-pattern")
    nodes = [SemanticNode(key, "input", (), tensor, effect="input") for key in ("x", "w0", "w1")]
    nodes += [SemanticNode(f"r{i}", "f32_add", (("x", "r0")[i], f"w{i}"), tensor,
                           (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
                           index_maps=pointwise_identity_maps(1, 2)) for i in range(2)]
    source = KernelRequest(tuple(nodes), ("r1",), ("external",), tuple((key, "external") for key in ("x", "w0", "w1")),
                           "synthetic")
    plan = plan_for(source, ((0,), (1,)))
    plan = replace(plan, selected=(replace(plan.selected[0], placement="host", implementation="shared_kernel"),
                                   replace(plan.selected[1], placement="host")))
    outlined = outline_semantic_regions(source, entry="test", plan=plan)
    program = build_program(outlined.logical, global_plan=plan, global_plan_emitter=outlined,
                            memory_requirements=lambda graph: ArenaRequirements(
                                256, (0, 2), (1, 3), dispatch_digest=dispatch_digest(graph)))
    return program, {i: DispatchCall(f"variant{i}", "host", context_index=i) for i in range(2)}


def test_stale_schedule_rejected():
    for field in ("source", "placement", "symbol", "context", "binding", "memory", "authority", "plan"):
        program, calls = planned() if field == "plan" else fixture()
        if field == "plan":
            loop = prepare_dispatch_loop(program, calls, offset=0, period=1, trip_count=2)
            emit_program_c(program, "test", calls, loop=loop)
            # The same dispatch under a different selected implementation, with an emission updated to match.
            plan = program.global_plan
            program.global_plan = replace(plan, selected=(
                replace(plan.selected[0], implementation="different_kernel"), *plan.selected[1:]))
            program.global_plan_emission = replace(program.global_plan_emission,
                                                   plan_digest=program.global_plan.digest)
            with pytest.raises(ValueError, match="loop differs"):
                emit_program_c(program, "test", calls, loop=loop)
            continue
        loop = prepare_dispatch_loop(program, calls, offset=1, period=2, trip_count=2)
        if field == "source":
            program.dispatch.nodes[0].op = "changed"
        elif field == "placement":
            calls[3] = replace(calls[3], placement="other")
        elif field == "symbol":
            calls[3] = replace(calls[3], symbol="other")
        elif field == "context":
            calls[3] = replace(calls[3], context_index=0)
        elif field == "binding":
            program.dispatch.nodes[3].inputs.reverse()
        elif field == "memory":
            program.memory_plan.offsets["v0"] += 1
        else:
            loop = replace(loop, authority_sha256="stale")
        with pytest.raises(ValueError):
            emit_program_c(program, "test", calls, loop=loop)


def test_invalid_bounds_refused():
    program, calls = fixture()
    for bounds in ((-1, 2, 2), (True, 2, 2), (1, 0, 2), (1, 2, 1), (1, 3, 2)):
        with pytest.raises(ValueError, match="bounds"):
            prepare_dispatch_loop(program, calls, offset=bounds[0], period=bounds[1], trip_count=bounds[2])


def test_loop_entry_symbol_collision_rejected():
    program, calls = fixture()
    calls[0] = replace(calls[0], symbol="test_run_loop")
    loop = prepare_dispatch_loop(program, calls, offset=1, period=2, trip_count=2)
    with pytest.raises(ValueError, match="conflicts"):
        emit_program_c(program, "test", calls, loop=loop)


def test_a_counted_loop_calls_each_kernel_directly_and_takes_no_kernel_address():
    """A static stack bound follows a function pointer to every address-taken function, so the table holds none."""
    import re

    program, calls = fixture()
    schedule = prepare_dispatch_loop(program, calls, offset=1, period=2, trip_count=2)
    _, files = emit_program_c(program, "counted", calls, loop=schedule)
    source = files["counted.c"]
    for call in calls.values():
        uses = re.findall(r"\b" + call.symbol + r"\b.{0,24}", source)
        assert uses and all(use.startswith((call.symbol + "(void *", call.symbol + "(call->context")) for use in uses), uses
    assert "unsigned body;" in files["counted.h"] and not re.search(r"\(\*\w+\)\s*\(", source)
