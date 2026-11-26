"""Independent generated applications keep each invocation's buffer bindings."""
from dataclasses import replace
import shutil
import subprocess
from merlin.compiler.lower.c.application import emit_application_c
from merlin.compiler.lower.c.dispatch import DispatchCall, emit_program_c, build_program
from merlin.compiler.partition.semantic_regions import outline_semantic_regions
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType
from merlin.compiler.schedule.storage.arena import ArenaRequirements
from merlin.compiler.partition.global_plan_emission import dispatch_digest
from semantic_application_fixtures import plan_for

def _two_call_application(directory, name):
    """y = first(x), out = second(y, x): two host calls, the second reading the input through the call table."""
    x = SemanticNode("x", "input", (), TensorType((4,), "f32", "raw-bit-pattern"), effect="input")
    y = SemanticNode("y", "reshape", ("x",), TensorType((4,), "f32", "raw-bit-pattern"))
    out = SemanticNode("out", "concat", ("y", "x"), TensorType((8,), "f32", "raw-bit-pattern"), (("axis", 0),))
    source = KernelRequest((x, y, out), ("out",), ("external",), (("x", "external"),), "synthetic")
    plan = plan_for(source, ((0,), (1,)))
    plan = replace(plan, selected=tuple(replace(row, placement="host") for row in plan.selected))
    outlined = outline_semantic_regions(source, entry=name, plan=plan)
    program = build_program(outlined.logical, global_plan=plan, global_plan_emitter=outlined,
                            memory_requirements=lambda graph: ArenaRequirements(
                                1024, (0, 2), (1, 3), dispatch_digest=dispatch_digest(graph)))
    driver, driver_files = emit_program_c(program, name + "_driver", {
        0: DispatchCall(name + "_first", "host"), 1: DispatchCall(name + "_second", "host")})
    assert "loop" not in driver or driver["loop"] is None
    _, files = emit_application_c(source, driver, driver_files, name, capacity=4096)
    for filename, text in {**driver_files, **files}.items():
        (directory / filename).write_text(text)
    return [name + ".c", name + "_driver.c"]


def test_each_invocation_owns_its_call_table(tmp_path):
    compiler = shutil.which("cc")
    assert compiler, "C compiler is required for generated application execution"
    sources = _two_call_application(tmp_path, "alpha") + _two_call_application(tmp_path, "beta")
    # alpha's first call invokes a second alpha instance, then beta, before the outer second call reads its table.
    (tmp_path / "main.c").write_text(r'''
#include "alpha.h"
#include "beta.h"
#include <assert.h>
#include <string.h>
static int depth;
static const void *second_inputs[2];
#define READ(x) (second_inputs[0]==(x) || second_inputs[1]==(x))
static _Alignas(64) unsigned char inner_workspace[ALPHA_WORKSPACE_BYTES];
static _Alignas(64) unsigned char beta_workspace[BETA_WORKSPACE_BYTES];
static void run_nested(void) {
  float x[4]={5,6,7,8},y[8];const float *inputs[]={x};float *outputs[]={y};
  alpha_driver_state state={0};
  assert(alpha_invoke(&state,inner_workspace,sizeof(inner_workspace),inputs,outputs,0)==0);
  const float expected[8]={5,6,7,8,5,6,7,8};
  assert(memcmp(y,expected,sizeof(y))==0 && READ(x));
  float b[4]={9,10,11,12},c[8];const float *beta_inputs[]={b};float *beta_outputs[]={c};
  beta_driver_state other={0};
  assert(beta_invoke(&other,beta_workspace,sizeof(beta_workspace),beta_inputs,beta_outputs,0)==0);
  const float beta_expected[8]={9,10,11,12,9,10,11,12};
  assert(memcmp(c,beta_expected,sizeof(c))==0 && READ(b));
}
static int first(const void *const *inputs,void *const *outputs) {
  memcpy(outputs[0],inputs[0],16);
  if (!depth++) run_nested();
  return 0;
}
static int second(const void *const *inputs,void *const *outputs) {
  /* Both inputs hold the same bytes (y copies x), so the output does not depend on their order. */
  second_inputs[0]=inputs[0];second_inputs[1]=inputs[1];
  memcpy(outputs[0],inputs[0],16);memcpy((unsigned char *)outputs[0]+16,inputs[1],16);
  return 0;
}
int alpha_first(void *c,const void *const *i,void *const *o) { (void)c; return first(i,o); }
int alpha_second(void *c,const void *const *i,void *const *o) { (void)c; return second(i,o); }
int beta_first(void *c,const void *const *i,void *const *o) { (void)c; memcpy(o[0],i[0],16); return 0; }
int beta_second(void *c,const void *const *i,void *const *o) { (void)c; return second(i,o); }
int main(void) {
  static _Alignas(64) unsigned char workspace[ALPHA_WORKSPACE_BYTES];
  float x[4]={1,2,3,4},y[8];const float *inputs[]={x};float *outputs[]={y};
  alpha_driver_state state={0};
  assert(alpha_invoke(&state,workspace,sizeof(workspace),inputs,outputs,0)==0);
  /* The outer second call still reads the outer input and writes the outer output staging. */
  const float expected[8]={1,2,3,4,1,2,3,4};
  assert(depth==2 && READ(x) && memcmp(y,expected,sizeof(y))==0);
  return 0;
}
''')
    built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", *sources, "main.c",
                            "-o", "check"], cwd=tmp_path, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(tmp_path / "check")], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
