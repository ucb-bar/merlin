"""Rank-zero tensors stay descriptors; i1 storage is checked canonical bytes."""

import pytest
from xdsl.dialects.builtin import StringAttr, TensorType, f32, i1

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_outline import plan_deployable_chain, static_tensor_contract
from merlin.runtime.outlined_chain import render_outlined_chain
from merlin.runtime.staged_dispatch import StagedRoles

SOURCE = r"""module {
func.func @forward(%a:tensor<7xf32>,%threshold:tensor<f32>,%flag:tensor<i1>)->(tensor<7xi1>,tensor<i32>) {
 %e=tensor.empty():tensor<7xi1>
 %m=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>,
  affine_map<(d0)->()>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
 ins(%a,%threshold,%flag:tensor<7xf32>,tensor<f32>,tensor<i1>) outs(%e:tensor<7xi1>) {
 ^bb0(%x:f32,%t:f32,%f:i1,%old:i1):
  %cmp=arith.cmpf ogt,%x,%t:f32
  %v=arith.andi %cmp,%f:i1
  linalg.yield %v:i1
 }->tensor<7xi1>
 %z=arith.constant 0:i32
 %init=tensor.splat %z:tensor<i32>
 %count=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]}
 ins(%m:tensor<7xi1>) outs(%init:tensor<i32>) {
 ^bb0(%v:i1,%acc:i32):
  %wide=arith.extui %v:i1 to i32
  %sum=arith.addi %wide,%acc:i32
  linalg.yield %sum:i32
 }->tensor<i32>
 func.return %m,%count:tensor<7xi1>,tensor<i32>
}}
"""


def plan(source=SOURCE):
    return plan_deployable_chain(
        parse_mlir_text(source),
        worker_slots=(0, 1),
        worker_count=2,
        workspace_budget_bytes=4096,
        alignment_bytes=64,
        data_preparation=True,
    )


def test_rank_zero_has_one_element_and_no_c_scalar_argument():
    assert static_tensor_contract(TensorType(f32, [])) == ((), "f32", 4)
    assert static_tensor_contract(TensorType(i1, [])) == ((), "i1", 1)
    record = plan().to_dict()
    assert record["rank_zero_tensor_abi"] == {"elements": 1, "descriptor_words": 3, "C_scalar_arguments": False}
    assert record["boolean_storage_abi"]["canonical_values"] == [0, 1]
    assert record["boolean_storage_abi"]["container_bytes_per_element"] == 1
    assert record["controller_arithmetic"] is False


@pytest.mark.parametrize("shape", [(0,), (-1,), (2, 0)])
def test_empty_or_dynamic_tensor_is_not_a_rank_zero_scalar(shape):
    with pytest.raises(ValueError, match="static positive"):
        static_tensor_contract(TensorType(i1, shape))


def test_unknown_tensor_layout_does_not_gain_boolean_storage_permission():
    with pytest.raises(ValueError, match="dense tensor encoding"):
        static_tensor_contract(TensorType(i1, (3,), StringAttr("unknown")))


def test_generated_boolean_checks_keep_arithmetic_on_assigned_workers():
    roles = StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))
    result = render_outlined_chain(plan(), ("compare", "count"), roles=roles)
    source = result["source"]
    init = source.split("static int model_init", 1)[1].split("static int prepare", 1)[0]
    compute = source.split("static int compute", 1)[1].split("static int finish", 1)[0]
    assert "return -126" in init and "return -127" in compute
    assert "_mlir_ciface_compare" not in init and "_mlir_ciface_count" not in init
    assert "_mlir_ciface_compare" in compute and "_mlir_ciface_count" in compute
    assert "{MERLIN_INPUT,0,0,{0},4}" in source
    assert "{MERLIN_INPUT,0,0,{0},1}" in source
    assert "{MERLIN_OUTPUT,0,0,{0},4}" in source


def test_boolean_transpose_remains_compute_not_unqualified_raw_byte_packing():
    source = """module {
func.func @forward(%x:tensor<3x5xi1>)->tensor<5x3xi1> {
 %e=tensor.empty():tensor<5x3xi1>
 %t=linalg.transpose ins(%x:tensor<3x5xi1>) outs(%e:tensor<5x3xi1>) permutation=[1,0]
 func.return %t:tensor<5x3xi1>
}}
"""
    result = plan_deployable_chain(
        parse_mlir_text(source),
        worker_slots=(0,),
        worker_count=1,
        workspace_budget_bytes=1024,
        alignment_bytes=64,
        data_preparation=True,
    )
    assert len(result.dispatches) == 1
    assert result.data_preparation["controller_permutations"] == []


def test_rank_zero_descriptor_has_three_words_and_one_element_capacity(tmp_path):
    import shutil
    import subprocess

    from merlin.common.paths import runtime_dir

    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    fixture = tmp_path / "descriptor.c"
    fixture.write_text(r"""
#include "merlin_model.h"
#include <assert.h>
#include <string.h>
int main(void){
 float scalar=1.25f;unsigned char flag=1;
 merlin_arg_t args[2]={{MERLIN_INPUT,0,0,{999},4},{MERLIN_INPUT,0,0,{999},1}};
 void*data[2]={&scalar,&flag};size_t capacities[2]={4,1};
 merlin_descriptor_t d[2];void*ptrs[2];memset(d,0xa5,sizeof d);
 assert(!merlin_prepare_descriptors(args,2,data,capacities,d,ptrs));
 for(unsigned i=0;i<2;i++){
  assert(ptrs[i]==&d[i]&&d[i].allocated==data[i]&&d[i].aligned==data[i]&&d[i].offset==0);
  const unsigned char*p=(const unsigned char*)&d[i];
  for(size_t k=3*sizeof(int64_t);k<sizeof(d[i]);k++)assert(p[k]==0xa5);
 }
 unsigned char saved[sizeof d];memcpy(saved,d,sizeof d);capacities[0]=3;
 assert(merlin_prepare_descriptors(args,2,data,capacities,d,ptrs)==-1);
 assert(!memcmp(saved,d,sizeof d));
 return 0;
}
""")
    runtime = runtime_dir() / "c"
    exe = tmp_path / "descriptor"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(runtime),
            str(fixture),
            str(runtime / "merlin_descriptors.c"),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=10)


def test_generated_callbacks_refuse_noncanonical_input_and_output_before_publication(tmp_path):
    import shutil
    import subprocess

    from merlin.common.paths import runtime_dir

    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    roles = StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))
    emitted = render_outlined_chain(plan(), ("compare", "count"), roles=roles)
    (tmp_path / "merlin_chain.h").write_text(emitted["header"])
    (tmp_path / "merlin_chain.c").write_text(emitted["source"])
    fixture = tmp_path / "guard.c"
    fixture.write_text(r"""
#include "merlin_chain.c"
#include <assert.h>
static unsigned self,calls;
static unsigned hart(void*p){(void)p;return self;}
void _mlir_ciface_compare(void*a,void*t,void*f,void*out){
 (void)a;(void)t;(void)f;calls++;
 memset(((merlin_descriptor_t*)out)->aligned,2,7);
}
void _mlir_ciface_count(void*input,void*out){(void)input;(void)out;assert(0);}
static _Alignas(64) unsigned char arena[4096],values[28],threshold[4],flag[1],mask[7],count[4];
static merlin_chain_context good,bad;
static void init(merlin_chain_context*c){
 c->workspace=arena;c->workspace_capacity=sizeof arena;
 c->inputs[0]=values;c->inputs[1]=threshold;c->inputs[2]=flag;
 c->input_capacity[0]=28;c->input_capacity[1]=4;c->input_capacity[2]=1;
 c->outputs[0]=mask;c->outputs[1]=count;c->output_capacity[0]=7;c->output_capacity[1]=4;
 c->team.config.controller_hart=2;c->team.config.worker_harts[0]=0;c->team.config.worker_harts[1]=1;
 c->team.config.sequence=1;c->platform_hooks.current_hart=hart;
}
int main(void){
 memset(mask,0x5a,sizeof mask);memset(count,0x5a,sizeof count);
 self=2;flag[0]=2;init(&bad);assert(model_init(&bad)==-126);assert(calls==0);
 flag[0]=1;init(&good);assert(model_init(&good)==0);assert(prepare(&good)==0);
 self=0;assert(compute(&good,0)==-127);assert(calls==1&&good.kernel_calls[0]==1);
 for(unsigned i=0;i<sizeof mask;i++)assert(mask[i]==0x5a);
 for(unsigned i=0;i<sizeof count;i++)assert(count[i]==0x5a);
 return 0;
}
""")
    runtime = runtime_dir() / "c"
    exe = tmp_path / "guard"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(runtime),
            "-I" + str(tmp_path),
            str(fixture),
            str(runtime / "merlin_descriptors.c"),
            str(runtime / "merlin_staged.c"),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=10)
