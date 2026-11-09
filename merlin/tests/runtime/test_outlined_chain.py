"""Deployable outlined arithmetic stays on workers with explicit buffer lifetimes."""

import json
import shutil
import subprocess

import numpy as np
import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.utils.exceptions import VerifyException

from merlin.common.paths import runtime_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_outline import canonical_kernel, kernel_code_key, plan_deployable_chain
from merlin.runtime.outlined_chain import render_outlined_chain
from merlin.runtime.staged_dispatch import StagedRoles

CHAIN = r"""builtin.module {
func.func @forward(%a:tensor<4x8xi8>,%b:tensor<8x8xi8>,%bias:tensor<4x8xi32>)->tensor<21xi32> {
  %e=tensor.empty():tensor<4x8xi32>
  %z=arith.constant 0:i32
  %init=linalg.fill ins(%z:i32) outs(%e:tensor<4x8xi32>)->tensor<4x8xi32>
  %m=linalg.generic {
    indexing_maps=[affine_map<(d0,d1,d2)->(d0,d2)>,affine_map<(d0,d1,d2)->(d2,d1)>,
                   affine_map<(d0,d1,d2)->(d0,d1)>],
    iterator_types=["parallel","parallel","reduction"]}
    ins(%a,%b:tensor<4x8xi8>,tensor<8x8xi8>) outs(%init:tensor<4x8xi32>) {
    ^bb0(%aa:i8,%bb:i8,%acc:i32):
      %ae=arith.extsi %aa:i8 to i32
      %be=arith.extsi %bb:i8 to i32
      %prod=arith.muli %ae,%be:i32
      %total=arith.addi %acc,%prod:i32
      linalg.yield %total:i32
  }->tensor<4x8xi32>
  %e1=tensor.empty():tensor<4x8xi32>
  %sum=linalg.generic {
    indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>],
    iterator_types=["parallel","parallel"]}
    ins(%m,%bias:tensor<4x8xi32>,tensor<4x8xi32>) outs(%e1:tensor<4x8xi32>) {
    ^bb0(%x:i32,%y:i32,%o:i32):
      %v=arith.addi %x,%y:i32
      linalg.yield %v:i32
  }->tensor<4x8xi32>
  %flat=tensor.collapse_shape %sum [[0,1]]:tensor<4x8xi32> into tensor<32xi32>
  %slice="tensor.extract_slice"(%flat) <{static_offsets=array<i64:1>,static_sizes=array<i64:21>,
    static_strides=array<i64:1>,operandSegmentSizes=array<i32:1,0,0,0>}>
    : (tensor<32xi32>) -> tensor<21xi32>
  %e2=tensor.empty():tensor<21xi32>
  %out=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
    iterator_types=["parallel"]} ins(%slice:tensor<21xi32>) outs(%e2:tensor<21xi32>) {
    ^bb0(%x:i32,%o:i32):
      %two=arith.constant 2:i32
      %v=arith.muli %x,%two:i32
      linalg.yield %v:i32
  }->tensor<21xi32>
  func.return %out:tensor<21xi32>
}}
"""


def policy():
    return StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))


def plan(source=CHAIN, **kwargs):
    return plan_deployable_chain(
        parse_mlir_text(source),
        worker_slots=kwargs.pop("worker_slots", (0, 1, 0)),
        worker_count=2,
        workspace_budget_bytes=kwargs.pop("workspace_budget_bytes", 4096),
        alignment_bytes=64,
        **kwargs,
    )


def test_closed_plan_keeps_arithmetic_and_alias_lifetimes_explicit():
    result = plan()
    assert len(result.dispatches) == 3
    assert [step["worker_slot"] for step in result.dispatches] == [0, 1, 0]
    assert [view["kind"] for view in result.dispatches[2]["views_before"]] == ["alias", "pack_slice"]
    assert result.workspace_bytes == 512
    assert result.to_dict()["controller_arithmetic"] is False
    assert result.to_dict()["full_model_admission"] is False
    assert len({buffer.workspace_offset for buffer in result.buffers if buffer.workspace_offset is not None}) == 4


@pytest.mark.parametrize(
    "kwargs", [{"worker_slots": (0, 1)}, {"worker_slots": (0, 2, 0)}, {"workspace_budget_bytes": 511}]
)
def test_bad_assignments_or_incomplete_memory_budget_refuse(kwargs):
    with pytest.raises(ValueError):
        plan(**kwargs)


def test_input_used_as_writable_kernel_accumulator_refuses():
    with pytest.raises(ValueError, match="writable output"):
        plan(CHAIN.replace("outs(%init:tensor<4x8xi32>)", "outs(%bias:tensor<4x8xi32>)"))


def test_input_alias_cannot_hide_writable_kernel_accumulator():
    source = CHAIN.replace(
        "%be=arith.extsi %bb:i8 to i32",
        """%be=arith.extsi %bb:i8 to i32
      %alias=tensor.cast %a:tensor<4x8xi8> to tensor<4x8xi8>
      %zero8=arith.constant 0:i8
      %nested=linalg.fill ins(%zero8:i8) outs(%alias:tensor<4x8xi8>)->tensor<4x8xi8>""",
    )
    with pytest.raises(ValueError, match="writable output"):
        plan(source)


def test_unknown_nested_effect_operation_is_not_assumed_pure():
    source = CHAIN.replace(
        "%be=arith.extsi %bb:i8 to i32",
        """%be=arith.extsi %bb:i8 to i32
      "unknown.effect"(%a):(tensor<4x8xi8>)->()""",
    )
    with pytest.raises(ValueError, match="unsupported closed tensor kernel"):
        plan(source)


def test_parent_function_contract_is_not_silently_lost_when_outlining():
    module = parse_mlir_text(CHAIN)
    next(op for op in module.walk() if op.name == "func.func").attributes["unknown.function.contract"] = StringAttr("x")
    with pytest.raises(ValueError, match="function-level contracts"):
        plan_deployable_chain(
            module, worker_slots=(0, 1, 0), worker_count=2, workspace_budget_bytes=4096, alignment_bytes=64
        )


def test_unsupported_live_scalar_driver_never_runs_on_controller():
    source = CHAIN.replace(
        "func.return %out:tensor<21xi32>", "%scalar=arith.constant 2:i32\n func.return %scalar:i32"
    ).replace("->tensor<21xi32> {", "->i32 {", 1)
    with pytest.raises(ValueError, match="ranked tensor"):
        plan(source)


def test_slice_bounds_are_not_silently_clamped():
    with pytest.raises((ValueError, VerifyException)):
        plan(CHAIN.replace("static_offsets=array<i64:1>", "static_offsets=array<i64:20>"))


def test_code_key_uses_exact_semantics_not_signature_or_provenance_guessing():
    p = plan()
    original = canonical_kernel(p.outlined.module, p.dispatches[2]["symbol"])
    other_plan = plan(CHAIN.replace("%two=arith.constant 2", "%two=arith.constant 3"))
    changed = canonical_kernel(other_plan.outlined.module, other_plan.dispatches[2]["symbol"])
    assert kernel_code_key(original, {}) != kernel_code_key(changed, {})
    same = original.clone()
    same.attributes["unknown.policy"] = StringAttr("must remain")
    assert kernel_code_key(original, {}) != kernel_code_key(same, {})
    same.attributes.clear()
    for op in same.walk():
        for result in op.results:
            result.name_hint = "different_name"
    fn = next(op for op in same.body.block.ops if op.name == "func.func")
    normalized = canonical_kernel(same, fn.sym_name.data)
    assert kernel_code_key(original, {}) == kernel_code_key(normalized, {})
    assert kernel_code_key(original, {"flags": ["first"]}) != kernel_code_key(original, {"flags": ["second"]})


def test_render_binds_role_policy_and_only_compute_invokes_arithmetic():
    result = render_outlined_chain(plan(), ("k0", "k1", "k2"), roles=policy())
    source = result["source"]
    compute = source.split("static int compute(", 1)[1].split("static int finish(", 1)[0]
    assert "_mlir_ciface_k0(c->descriptor_ptrs" in compute
    for section in (
        source.split("static int model_init", 1)[1].split("static int prepare", 1)[0],
        source.split("static int prepare", 1)[1].split("static int compute", 1)[0],
        source.split("static int finish", 1)[1],
    ):
        assert "_mlir_ciface_" not in section
    assert "config->controller_hart!=2" in source
    assert "config->worker_harts[1]!=1" in source
    with pytest.raises(ValueError, match="different typed ABI"):
        render_outlined_chain(plan(), ("same", "same", "same"), roles=policy())


def c_compiler():
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")
    return cc


def test_shared_descriptor_preparation_is_rank_exact_and_transactional(tmp_path):
    source = tmp_path / "descriptor.c"
    source.write_text(r"""
#include "merlin_model.h"
#include <assert.h>
#include <string.h>
int main(void){
 int data[8]={0};void*ptrs[2]={data,data};size_t capacities[2]={24,16};
 merlin_arg_t args[2]={{MERLIN_INPUT,0,2,{2,3},4},{MERLIN_OUTPUT,0,1,{4},4}};
 merlin_descriptor_t descriptors[2];void*d[2];
 assert(!merlin_prepare_descriptors(args,2,ptrs,capacities,descriptors,d));
 int64_t*w=(int64_t*)&descriptors[0];assert(w[3]==2&&w[4]==3&&w[5]==3&&w[6]==1);
 w=(int64_t*)&descriptors[1];assert(w[3]==4&&w[4]==1);
 memset(descriptors,0xa5,sizeof descriptors);unsigned char old[sizeof descriptors];
 memcpy(old,descriptors,sizeof old);capacities[1]=15;
 assert(merlin_prepare_descriptors(args,2,ptrs,capacities,descriptors,d)==-1);
 assert(!memcmp(old,descriptors,sizeof old));
 capacities[1]=16;args[0].dims[0]=INT64_MAX;args[0].dims[1]=2;
 assert(merlin_prepare_descriptors(args,2,ptrs,capacities,descriptors,d)==-1);
 args[0].dims[0]=0;
 assert(merlin_prepare_descriptors(args,2,ptrs,capacities,descriptors,d)==-1);
 return 0;
}""")
    exe = tmp_path / "descriptor"
    runtime = runtime_dir() / "c"
    subprocess.run(
        [
            c_compiler(),
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(runtime),
            str(source),
            str(runtime / "merlin_descriptors.c"),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=10)


def test_real_compiler_cli_builds_no_parent_and_retains_failure(tmp_path):
    from merlin.compile.outlined_program import main
    from merlin.llvmlower import toolchain

    if not toolchain.available():
        pytest.skip("MLIR compiler unavailable")
    source = tmp_path / "model.mlir"
    source.write_text(CHAIN)
    config = {
        "schema": "merlin.outlined_program.v1",
        "roles": {
            "cpu_hart_ids": [0, 1, 2],
            "model_init_hart": 2,
            "pack_hart": 2,
            "dispatch_hart": 2,
            "finish_hart": 2,
            "compute_harts": [0, 1],
            "profile_harts": [0, 1],
        },
        "worker_slots": [0, 1, 0],
        "workspace_budget_bytes": 4096,
        "alignment_bytes": 64,
        "compute_cflags": ["-march=rv64gcv", "-fno-vectorize", "-fno-slp-vectorize"],
    }
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    out = tmp_path / "built"
    assert main(["--source", str(source), "--config", str(cfg), "--output", str(out)]) == 0
    built = json.loads((out / "build.json").read_text())
    assert built["dispatches"] == 3 and built["parent_forward_compiled"] is False
    assert all(item["argument_ranks"] for item in built["kernel_builds"])
    assert all("@forward(" not in open(item["llvm"]).read() for item in built["kernel_builds"])
    config["workspace_budget_bytes"] = 1
    cfg.write_text(json.dumps(config))
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="unique-storage bytes"):
        main(["--source", str(source), "--config", str(cfg), "--output", str(failed)])
    assert json.loads((failed / "build.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("stride", [1, 2])
def test_actual_persistent_workers_execute_full_chain_and_reject_reuse_errors(tmp_path, stride):
    source = CHAIN
    if stride == 2:
        source = (
            source.replace("tensor<21xi32>", "tensor<16xi32>")
            .replace("static_sizes=array<i64:21>", "static_sizes=array<i64:16>")
            .replace("static_offsets=array<i64:1>", "static_offsets=array<i64:0>")
        )
        source = source.replace("static_strides=array<i64:1>", "static_strides=array<i64:2>")
    planned = plan(source)
    emitted = render_outlined_chain(planned, ("k0", "k1", "k2"), roles=policy())
    (tmp_path / "merlin_chain.h").write_text(emitted["header"])
    (tmp_path / "merlin_chain.c").write_text(emitted["source"])
    a = np.arange(32, dtype=np.uint8).reshape(4, 8).view(np.int8) - np.int8(16)
    b = np.arange(64, dtype=np.uint8).reshape(8, 8).view(np.int8) - np.int8(32)
    bias = np.resize(np.asarray([-(1 << 31), (1 << 31) - 1, -7, 8], dtype=np.int32), (4, 8))
    mat = (a.astype(np.int64) @ b.astype(np.int64)).astype("<i4")
    added = (mat.astype(np.int64) + bias.astype(np.int64)).astype("<i4")
    selected = added.reshape(-1)[1:22] if stride == 1 else added.reshape(-1)[::2]
    expected = (selected.astype(np.int64) * 2).astype("<i4")
    declarations = []
    for name, values in (("a", a), ("b", b), ("bias", bias), ("expected", expected)):
        raw = values.tobytes()
        declarations.append(f"static const unsigned char {name}_data[]={{{','.join(map(str, raw))}}};")
    fixture = r"""
#include "merlin_chain.h"
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
@DATA@
static _Thread_local unsigned self;
static atomic_uint_fast64_t clock_tick,cycles_seen[2],retired_seen[2];
static unsigned hart(void*x){(void)x;return self;}
static int ticks(void*x,uint64_t*v){(void)x;assert(self<2);*v=atomic_fetch_add(&clock_tick,1);return 0;}
static uint64_t cycles(void*x){(void)x;assert(self<2);return atomic_fetch_add(&cycles_seen[self],1);}
static uint64_t retired(void*x){(void)x;assert(self<2);return atomic_fetch_add(&retired_seen[self],1);}
static void relax(void*x){(void)x;sched_yield();}
static void* data(void*d){return ((merlin_descriptor_t*)d)->aligned;}
void _mlir_ciface_k0(void*a,void*b,void*out){
 assert(self==0);const int8_t*A=data(a),*B=data(b);uint32_t*C=data(out);
 for(unsigned i=0;i<4;i++)for(unsigned j=0;j<8;j++){
  uint32_t sum=0;for(unsigned k=0;k<8;k++)sum+=(uint32_t)((int32_t)A[i*8+k]*(int32_t)B[k*8+j]);
  C[i*8+j]=sum;
 }
}
void _mlir_ciface_k1(void*a,void*b,void*out){
 assert(self==1);const uint32_t*A=data(a),*B=data(b);uint32_t*C=data(out);
 for(unsigned i=0;i<32;i++)C[i]=A[i]+B[i];
}
void _mlir_ciface_k2(void*a,void*out){
 assert(self==0);const uint32_t*A=data(a);uint32_t*C=data(out);
 for(int64_t i=0;i<((int64_t*)out)[3];i++)C[i]=A[i]*UINT32_C(2);
}
struct pool {pthread_mutex_t mutex;pthread_cond_t cond;unsigned ready,generation,finished,stop,join_error;
             merlin_staged_team*team;};
static struct pool pool={.mutex=PTHREAD_MUTEX_INITIALIZER,.cond=PTHREAD_COND_INITIALIZER};
static void*worker(void*arg){
 self=(unsigned)(uintptr_t)arg;unsigned last=0;
 pthread_mutex_lock(&pool.mutex);pool.ready++;pthread_cond_broadcast(&pool.cond);
 for(;;){
  while(!pool.stop&&pool.generation==last)pthread_cond_wait(&pool.cond,&pool.mutex);
  if(pool.stop)break;
  last=pool.generation;merlin_staged_team*team=pool.team;pthread_mutex_unlock(&pool.mutex);
  assert(merlin_staged_worker(team,self)==0||team->report.status!=0);
  pthread_mutex_lock(&pool.mutex);pool.finished++;pthread_cond_broadcast(&pool.cond);
 }
 pthread_mutex_unlock(&pool.mutex);return NULL;
}
static void launch(void*x,merlin_staged_team*team){
 (void)x;assert(self==2);pthread_mutex_lock(&pool.mutex);assert(pool.finished==0||pool.finished==2);
 pool.finished=0;pool.team=team;pool.generation++;pthread_cond_broadcast(&pool.cond);pthread_mutex_unlock(&pool.mutex);
}
static int join(void*x,merlin_staged_team*team){
 (void)x;assert(self==2&&team->report.quiescent);pthread_mutex_lock(&pool.mutex);
 while(pool.finished!=2)pthread_cond_wait(&pool.cond,&pool.mutex);
 pthread_mutex_unlock(&pool.mutex);return pool.join_error?777:0;
}
static _Alignas(64) unsigned char a_store[sizeof(a_data)+128],b_store[sizeof(b_data)+128];
static _Alignas(64) unsigned char bias_store[sizeof(bias_data)+128],out_store[sizeof(expected_data)+128];
static _Alignas(64) unsigned char workspace[MERLIN_CHAIN_WORKSPACE_BYTES+128];
static merlin_chain_context c;
static void guards(unsigned char*p,size_t size){
 for(unsigned i=0;i<64;i++)assert(p[i]==0x5a&&p[64+size+i]==0x5a);
}
int main(void){
 self=2;memset(a_store,0x5a,sizeof a_store);memset(b_store,0x5a,sizeof b_store);
 memset(bias_store,0x5a,sizeof bias_store);memset(out_store,0x5a,sizeof out_store);
 memset(workspace,0x5a,sizeof workspace);
 memcpy(a_store+64,a_data,sizeof a_data);memcpy(b_store+64,b_data,sizeof b_data);
 memcpy(bias_store+64,bias_data,sizeof bias_data);
 c.inputs[0]=a_store+64;c.inputs[1]=b_store+64;c.inputs[2]=bias_store+64;
 c.input_capacity[0]=sizeof a_data;c.input_capacity[1]=sizeof b_data;c.input_capacity[2]=sizeof bias_data;
 c.outputs[0]=out_store+64;c.output_capacity[0]=sizeof expected_data;
 c.workspace=workspace+64;c.workspace_capacity=MERLIN_CHAIN_WORKSPACE_BYTES;
 pthread_t threads[2];assert(!pthread_create(&threads[0],NULL,worker,(void*)0));
 assert(!pthread_create(&threads[1],NULL,worker,(void*)1));
 pthread_mutex_lock(&pool.mutex);while(pool.ready!=2)pthread_cond_wait(&pool.cond,&pool.mutex);
 pthread_mutex_unlock(&pool.mutex);
 merlin_staged_hooks hooks={.current_hart=hart,.worker_cycles=cycles,.worker_instret=retired,
   .relax=relax,.shared_window_ticks=ticks};
 merlin_staged_config cfg={.controller_hart=2,.worker_count=2,.worker_harts={0,1},.sequence=1,
   .max_wait_polls=10000000,.enable_worker_windows=1};
 for(unsigned run=0;run<2;run++){
  cfg.sequence=1+3*run;assert(merlin_chain_run(&c,&cfg,&hooks,launch,join,NULL)==0);
  assert(c.completed_dispatches==3&&c.observed_count==3&&c.last_sequence==cfg.sequence+2);
  assert(c.kernel_calls[0]==2&&c.kernel_calls[1]==1);
  assert(!memcmp(out_store+64,expected_data,sizeof expected_data));
  assert(!memcmp(a_store+64,a_data,sizeof a_data)&&!memcmp(b_store+64,b_data,sizeof b_data));
  assert(!memcmp(bias_store+64,bias_data,sizeof bias_data));
  guards(a_store,sizeof a_data);guards(b_store,sizeof b_data);guards(bias_store,sizeof bias_data);
  guards(out_store,sizeof expected_data);guards(workspace,MERLIN_CHAIN_WORKSPACE_BYTES);
  for(unsigned i=0;i<3;i++)assert(c.observations[i].quiescent&&c.observations[i].worker_clock_windows_valid);
 }
 assert(merlin_chain_run(&c,&cfg,&hooks,launch,join,NULL)==-125);
 cfg.sequence=7;self=0;assert(merlin_chain_run(&c,&cfg,&hooks,launch,join,NULL)==-121);self=2;
 pool.join_error=1;assert(merlin_chain_run(&c,&cfg,&hooks,launch,join,NULL)==777&&c.poisoned);
 cfg.sequence=11;assert(merlin_chain_run(&c,&cfg,&hooks,launch,join,NULL)==-123);
 pthread_mutex_lock(&pool.mutex);pool.stop=1;pthread_cond_broadcast(&pool.cond);pthread_mutex_unlock(&pool.mutex);
 assert(!pthread_join(threads[0],NULL)&&!pthread_join(threads[1],NULL));return 0;
}
""".replace("@DATA@", "\n".join(declarations))
    (tmp_path / "main.c").write_text(fixture)
    runtime = runtime_dir() / "c"
    executable = tmp_path / "chain"
    subprocess.run(
        [
            c_compiler(),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            "-I" + str(tmp_path),
            "-I" + str(runtime),
            str(tmp_path / "main.c"),
            str(tmp_path / "merlin_chain.c"),
            str(runtime / "merlin_descriptors.c"),
            str(runtime / "merlin_staged.c"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=20)
