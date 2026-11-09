"""Source effects and returned flags survive outlined liveness and execution."""

import shutil
import subprocess

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from merlin.common.paths import runtime_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_outline import plan_deployable_chain
from merlin.runtime.outlined_chain import render_outlined_chain
from merlin.runtime.staged_dispatch import StagedRoles

GUARDED = """module {
func.func @forward(%a:tensor<4xf32>)->tensor<4xf32> {
 %e=tensor.empty():tensor<i1>
 %true=arith.constant true
 %init=linalg.fill ins(%true:i1) outs(%e:tensor<i1>)->tensor<i1>
 %valid=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]}
 ins(%a:tensor<4xf32>) outs(%init:tensor<i1>) {
 ^bb0(%x:f32,%acc:i1):
  %abs=math.absf %x:f32
  %max=arith.constant 3.4028234663852886e38:f32
  %finite=arith.cmpf ole,%abs,%max:f32
  %ok=arith.andi %finite,%acc:i1
  linalg.yield %ok:i1
 }->tensor<i1>
 %flag=tensor.extract %valid[]:tensor<i1>
 cf.assert %flag,"finite source input required"
 %empty=tensor.empty():tensor<4xf32>
 %out=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
 ins(%a:tensor<4xf32>) outs(%empty:tensor<4xf32>) {
 ^bb0(%x:f32,%old:f32):
  %one=arith.constant 1.0:f32
  %y=arith.addf %x,%one:f32
  linalg.yield %y:f32
 }->tensor<4xf32>
 return %out:tensor<4xf32>
}}"""


def plan(text=GUARDED, slots=(0, 1)):
    return plan_deployable_chain(
        parse_mlir_text(text), worker_slots=slots, worker_count=2, workspace_budget_bytes=4096, alignment_bytes=64
    )


def test_returned_flag_guard_is_ordered_live_and_keeps_its_producer():
    p = plan()
    assert [n.kind for n in p.program.nodes] == ["dispatch", "guard", "dispatch"]
    guard = p.dispatches[1]["views_before"]
    assert len(guard) == 1 and guard[0]["kind"] == "assert_flag"
    assert guard[0]["message"] == "finite source input required"
    assert guard[0]["input"] == p.dispatches[0]["outputs"][0]
    assert all(buffer.dtype in {"f32", "i1"} for buffer in p.buffers)
    assert not any(n.op == "tensor.extract" for n in p.program.nodes)


def test_unknown_no_result_effect_is_not_silently_pruned():
    text = GUARDED.replace('cf.assert %flag,"finite source input required"', '"unknown.effect"() : () -> ()')
    with pytest.raises(ValueError, match="unsupported deployable driver/view"):
        plan(text)


def test_unsupported_scalar_guard_is_not_discarded():
    text = GUARDED.replace("%flag=tensor.extract %valid[]:tensor<i1>", "%flag=arith.constant true")
    with pytest.raises(ValueError, match="rank-zero I1"):
        plan(text)


def test_exact_original_argument_provenance_is_retained_not_redistributed():
    module = parse_mlir_text(GUARDED)
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    function.attributes["prov.argument_source_node_ids"] = ArrayAttr([StringAttr("original-input")])
    p = plan_deployable_chain(
        module, worker_slots=(0, 1), worker_count=2, workspace_budget_bytes=4096, alignment_bytes=64
    )
    assert p.to_dict()["source_argument_ids"] == ["original-input"]
    for fn in p.outlined.module.body.block.ops:
        if fn.name == "func.func" and "$kernel_" in fn.sym_name.data:
            assert "prov.argument_source_node_ids" not in fn.attributes
    function.attributes["prov.argument_source_node_ids"] = ArrayAttr([])
    with pytest.raises(ValueError, match="provenance.*arity"):
        plan_deployable_chain(
            module, worker_slots=(0, 1), worker_count=2, workspace_budget_bytes=4096, alignment_bytes=64
        )


def test_partial_section_cannot_drop_a_source_guard():
    from merlin.xdsl_dialects.lowering.dispatch_program import slice_program

    p = plan().program
    dispatches = [node for node in p.nodes if node.kind == "dispatch"]
    for i, node in enumerate(dispatches):
        node.prov["prov.region_id"] = str(i)
    with pytest.raises(ValueError, match="cannot remove source guards"):
        slice_program(p, ["1"])
    assert [node.kind for node in p.nodes] == ["dispatch", "guard", "dispatch"]


def test_whole_section_selection_retains_the_complete_guarded_program():
    from merlin.xdsl_dialects.lowering.dispatch_program import slice_program, verify_program

    p = plan().program
    dispatches = [node for node in p.nodes if node.kind == "dispatch"]
    for i, node in enumerate(dispatches):
        node.prov["prov.region_id"] = str(i)
    selected = slice_program(p, ["0", "1"], entry_suffix="_whole")
    assert verify_program(selected) == []
    assert selected.entry == p.entry + "_whole"
    assert selected.to_dict() | {"entry": p.entry} == p.to_dict()
    assert all(left is not right for left, right in zip(selected.nodes, p.nodes, strict=True))


@pytest.mark.parametrize("slots", [(0, 1), (1, 0)])
def test_actual_compiled_finite_flags_block_later_math_for_nan_and_repeat(tmp_path, slots):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.deployable_ir import checked_generic_text
    from merlin.llvmlower.kernel_backend import extract_kernel
    from merlin.llvmlower.lower import lower_model

    if not toolchain.available() or shutil.which("cc") is None:
        pytest.skip("selected LLVM compiler and host C runtime required")
    p = plan(slots=slots)
    objects = []
    for index, step in enumerate(p.dispatches):
        kernel = extract_kernel(p.outlined.module, step["symbol"], entry="actual_" + str(index))
        built = lower_model(
            checked_generic_text(kernel),
            tmp_path / ("lower" + str(index)),
            targets=(),
            vectorize=False,
            features=frozenset(),
            static_arena=False,
        )
        obj = tmp_path / ("kernel" + str(index) + ".o")
        subprocess.run(
            [str(toolchain.clang()), "-O2", "-fPIC", "-ffp-contract=off", "-c", str(built.ll_path), "-o", str(obj)],
            check=True,
            capture_output=True,
            timeout=120,
        )
        objects.append(obj)
    roles = StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))
    emitted = render_outlined_chain(p, ("checked_0", "checked_1"), roles=roles, prefix="guarded")
    (tmp_path / "guarded.h").write_text(emitted["header"])
    (tmp_path / "guarded.c").write_text(emitted["source"])
    (tmp_path / "main.c").write_text(
        r"""
#include "guarded.h"
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
static _Thread_local unsigned hart;
static unsigned actual_calls[2];
void _mlir_ciface_actual_0(void*,void*);void _mlir_ciface_actual_1(void*,void*);
void _mlir_ciface_checked_0(void*a,void*b){assert(hart==@S0@);actual_calls[0]++;_mlir_ciface_actual_0(a,b);}
void _mlir_ciface_checked_1(void*a,void*b){assert(hart==@S1@);actual_calls[1]++;_mlir_ciface_actual_1(a,b);}
static unsigned current(void*x){(void)x;return hart;}
static uint64_t clock_read(void*x){(void)x;assert(hart<2);return 0;}
static void relax(void*x){(void)x;sched_yield();}
static pthread_t threads[2];static merlin_staged_team*team;
static void*worker(void*x){unsigned slot=(unsigned)(uintptr_t)x;hart=slot;
 int r=merlin_staged_worker(team,slot);assert(!r||team->report.status);return 0;}
static void launch(void*x,merlin_staged_team*t){(void)x;assert(hart==2);team=t;
 for(unsigned s=0;s<2;s++)assert(!pthread_create(&threads[s],0,worker,(void*)(uintptr_t)s));}
static int join(void*x,merlin_staged_team*t){(void)x;assert(hart==2&&t->report.quiescent);
 for(unsigned s=0;s<2;s++)assert(!pthread_join(threads[s],0));return 0;}
static _Alignas(64) unsigned char input[144],output[144],workspace[GUARDED_WORKSPACE_BYTES+128];
static guarded_context c;
int main(void){hart=2;merlin_staged_hooks hooks={.current_hart=current,.global_ticks=clock_read,
 .worker_cycles=clock_read,
 .worker_instret=clock_read,.relax=relax};
 merlin_staged_config config={.sequence=1,.controller_hart=2,.worker_count=2,.worker_harts={0,1},
 .max_wait_polls=10000000};
 float values[4]={-2,-1,0,4},expected[4]={-1,0,1,5};
 for(unsigned mode=0;mode<2;mode++){
  memset(&c,0,sizeof c);memset(input,0xa5,sizeof input);memset(output,0xa5,sizeof output);
  memset(workspace,0xa5,sizeof workspace);memcpy(input+64,values,16);actual_calls[0]=actual_calls[1]=0;
  if(mode){uint32_t nan=0x7fc00000;memcpy(input+64,&nan,4);}
  unsigned char saved[sizeof input];memcpy(saved,input,sizeof input);
  c.inputs[0]=input+64;c.input_capacity[0]=16;c.outputs[0]=output+64;c.output_capacity[0]=16;
  c.workspace=workspace+64;c.workspace_capacity=GUARDED_WORKSPACE_BYTES;
  config.sequence=mode?100:1;
  for(unsigned repeat=0;repeat<(mode?1:2);repeat++){
   int status=guarded_run(&c,&config,&hooks,launch,join,0);
   assert(c.team.report.quiescent);
   if(mode){assert(status&&actual_calls[0]==1&&actual_calls[1]==0&&c.completed_dispatches==1);
    for(unsigned i=64;i<80;i++)assert(output[i]==0xa5);}
   else{assert(!status&&c.completed_dispatches==2);assert(!memcmp(output+64,expected,16));}
   assert(!memcmp(input,saved,sizeof input));
   for(unsigned i=0;i<64;i++){assert(output[i]==0xa5&&output[80+i]==0xa5);
    assert(workspace[i]==0xa5&&workspace[GUARDED_WORKSPACE_BYTES+64+i]==0xa5);}
   config.sequence+=2;
  }
 }
 return 0;
}
""".replace("@S0@", str(slots[0])).replace("@S1@", str(slots[1]))
    )
    runtime = runtime_dir() / "c"
    exe = tmp_path / "guard-test"
    subprocess.run(
        [
            str(toolchain.clang()),
            "-std=c11",
            "-O2",
            "-pthread",
            "-I" + str(tmp_path),
            "-I" + str(runtime),
            str(tmp_path / "main.c"),
            str(tmp_path / "guarded.c"),
            str(runtime / "merlin_staged.c"),
            str(runtime / "merlin_descriptors.c"),
            *map(str, objects),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=20)
