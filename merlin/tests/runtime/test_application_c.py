"""Execute application constants, ordered outputs and atomic result staging."""

import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.compile.application_c import emit_application_c
from merlin.compile.dispatch_c import DispatchCall, emit_program_c
from merlin.compile.semantic_regions import outline_semantic_regions
from merlin.runtime.program import build_program
from merlin.semantic_compiler.model import (
    ConstantBinding,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.tests.ir.test_semantic_regions import plan_for
from merlin.xdsl_dialects.lowering.arena_plan import ArenaRequirements
from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest


def test_compiled_application_embeds_constants_and_copies_all_results_only_on_success(tmp_path):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler required")
    type_ = TensorType((4,), "f32", "raw-bit-pattern")
    source = KernelRequest(
        (
            SemanticNode("x", "input", (), type_, effect="input"),
            SemanticNode("bias", "constant", (), type_, effect="constant"),
            SemanticNode(
                "y",
                "f32_add",
                ("x", "bias"),
                type_,
                (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
                index_maps=pointwise_identity_maps(1, 2),
            ),
            SemanticNode(
                "z",
                "f32_mul",
                ("y", "y"),
                type_,
                (("numeric_semantics", "ieee754-f32-mul-rne-v1"),),
                index_maps=pointwise_identity_maps(1, 2),
            ),
        ),
        ("z", "y", "z", "x", "bias"),
        ("external",) * 5,
        (("x", "external"),),
        "synthetic",
        constants=(ConstantBinding("bias", "f32-le-raw", "0000803f" * 4),),
    )
    outlined = outline_semantic_regions(source, entry="model", plan=plan_for(source, ((0,), (1,))))
    program = build_program(
        outlined.logical,
        global_plan=outlined.plan,
        global_plan_emitter=outlined,
        memory_requirements=lambda graph: ArenaRequirements(
            256, (0, 2), (1, 3), dispatch_digest=dispatch_digest(graph)
        ),
    )
    driver, driver_files = emit_program_c(
        program, "model_driver", {i: DispatchCall(f"kernel_{i}", "device") for i in range(2)}
    )
    interface, files = emit_application_c(source, driver, driver_files, "model", capacity=256)
    assert interface["outputs"] == list(source.outputs) and interface["inputs"] == ["x"]
    for filename, text in {**driver_files, **files}.items():
        (tmp_path / filename).write_text(text)
    (tmp_path / "check.c").write_text(r"""
#include "model.h"
#include <assert.h>
#include <string.h>
static int fail;
int kernel_0(void *ctx,const void *const *in,void *const *out) {
  (void)ctx;
  for (int i=0;i<4;++i) ((float *)out[0])[i]=((const float *)in[0])[i]+((const float *)in[1])[i];
  return 0;
}
int kernel_1(void *ctx,const void *const *in,void *const *out) {
  (void)ctx;
  if (fail) return 19;
  for (int i=0;i<4;++i) ((float *)out[0])[i]=((const float *)in[0])[i]*((const float *)in[0])[i];
  return 0;
}
int main(void) {
  _Alignas(64) unsigned char memory[MODEL_WORKSPACE_BYTES+128];
  model_driver_state state={0}; float x[4],result[5][4];
  const float *inputs[]={x}; float *outputs[]={result[0],result[1],result[2],result[3],result[4]};
  memset(memory,0xcd,sizeof(memory));
  for (int panel=0;panel<3;++panel) {
    for (int i=0;i<4;++i) x[i]=panel+i;
    memset(result,0xa7,sizeof(result));
    assert(model_invoke(&state,memory+64,MODEL_WORKSPACE_BYTES,inputs,outputs,0)==0);
    for (int i=0;i<4;++i) {
      assert(result[0][i]==(x[i]+1)*(x[i]+1) && result[1][i]==x[i]+1);
      assert(result[2][i]==result[0][i] && result[3][i]==x[i] && result[4][i]==1);
    }
    for (unsigned i=0;i<64;++i) assert(memory[i]==0xcd && memory[MODEL_WORKSPACE_BYTES+64+i]==0xcd);
  }
  const float *invalid_inputs[]={(const float *)&state};
  assert(model_invoke(&state,memory+64,MODEL_WORKSPACE_BYTES,invalid_inputs,outputs,0)==MODEL_DRIVER_INVALID);
  assert(!state.active);
  fail=1; memset(result,0xa7,sizeof(result));
  assert(model_invoke(&state,memory+64,MODEL_WORKSPACE_BYTES,inputs,outputs,0)==MODEL_DRIVER_KERNEL_ERROR);
  assert(state.active && state.completed==1 && state.kernel_status==19);
  const unsigned char *bytes=(const unsigned char *)result;
  for (unsigned i=0;i<sizeof(result);++i) assert(bytes[i]==0xa7);
  assert(model_invoke(&state,memory+64,MODEL_WORKSPACE_BYTES,inputs,outputs,0)==MODEL_DRIVER_BUSY);
  return 0;
}
""")
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
            "model.c",
            "model_driver.c",
            "check.c",
            "-o",
            "check",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)


def test_mixed_integer_application_executes_embedding_and_preserves_i64_outputs(tmp_path):
    import numpy as np

    from merlin.compile.layout_c import emit_layout_c

    index_type = TensorType((2,), "i64", "signed-i64")
    table_type = TensorType((5, 3), "f32", "raw-bit-pattern")
    result_type = TensorType((2, 3), "f32", "raw-bit-pattern")
    extrema_type = TensorType((3,), "i64", "signed-i64")
    table = np.array(
        [
            0x3F800000,
            0x80000000,
            0x7FC12345,
            0xFF812345,
            0,
            0x7F800000,
            0xBF800000,
            0x00800000,
            0x7F7FFFFF,
            7,
            0x12345678,
            0xDEADBEEF,
            0,
            0x3F000000,
            0xC0000000,
        ],
        dtype="<u4",
    )
    extrema = np.array([-(1 << 63), (1 << 63) - 1, -17], dtype="<i8")
    indices = SemanticNode("indices", "input", (), index_type, effect="input")
    weights = SemanticNode("table", "constant", (), table_type, effect="constant")
    integers = SemanticNode("extrema", "constant", (), extrema_type, effect="constant")
    out = SemanticNode(
        "result",
        "embedding_lookup",
        (indices.id, weights.id),
        result_type,
        (("index_domain", "nonnegative-below-table-rows"), ("minimum_index_bits", 4)),
    )
    source = KernelRequest(
        (indices, weights, integers, out),
        (out.id, indices.id, integers.id),
        ("external",) * 3,
        ((indices.id, "external"),),
        "synthetic",
        constants=(
            ConstantBinding(weights.id, "f32-le-raw", table.tobytes().hex()),
            ConstantBinding(integers.id, "i64-le", extrema.tobytes().hex()),
        ),
    )
    outlined = outline_semantic_regions(source, entry="model", plan=plan_for(source, ((0,),)))
    program = build_program(
        outlined.logical,
        global_plan=outlined.plan,
        global_plan_emitter=outlined,
        memory_requirements=lambda graph: ArenaRequirements(256, (0,), (1,), dispatch_digest=dispatch_digest(graph)),
    )
    driver, driver_files = emit_program_c(program, "model_driver", {0: DispatchCall("lookup", "device")})
    interface, files = emit_application_c(source, driver, driver_files, "model", capacity=256)
    assert interface["pointer_abi"] == "opaque_native_dense_buffers"
    assert interface["input_types"][0]["dtype"] == "i64"
    assert [row["dtype"] for row in interface["output_types"]] == ["f32", "i64", "i64"]
    assert next(row for row in driver["buffers"] if row["id"] == "indices")["bytes"] == 16
    from merlin.compile.semantic_regions import extract_semantic_region

    _, lookup_files = emit_layout_c(extract_semantic_region(source, (out.id,)), "lookup")
    for filename, content in {**driver_files, **files, **lookup_files}.items():
        (tmp_path / filename).write_text(content)
    (tmp_path / "check.c").write_text(r"""
#include "model.h"
#include <assert.h>
#include <string.h>
#include <stdint.h>
int main(void) {
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES+128];
  unsigned char input[18],output[40]; int64_t returned[2],extrema[3];
  int64_t indices[2]={2,0}; const void *inputs[]={input+1};
  void *outputs[]={output+8,returned,extrema};
  memcpy(input+1,indices,16); memset(workspace,0xcd,sizeof(workspace)); memset(output,0xa7,sizeof(output));
  model_driver_state state={0};
  assert(model_invoke(&state,workspace+64,MODEL_WORKSPACE_BYTES,inputs,outputs,0)==0);
  uint32_t expected[]={0xbf800000,0x00800000,0x7f7fffff,0x3f800000,0x80000000,0x7fc12345};
  assert(memcmp(output+8,expected,24)==0 && memcmp(returned,indices,16)==0);
  assert(extrema[0]==INT64_MIN && extrema[1]==INT64_MAX && extrema[2]==-17);
  indices[0]=1; indices[1]=1; memcpy(input+1,indices,16);
  assert(model_invoke(&state,workspace+64,MODEL_WORKSPACE_BYTES,inputs,outputs,0)==0);
  uint32_t repeated[]={0xff812345,0,0x7f800000,0xff812345,0,0x7f800000};
  assert(memcmp(output+8,repeated,24)==0 && memcmp(returned,indices,16)==0);
  for (int i=0;i<8;++i) assert(output[i]==0xa7 && output[32+i]==0xa7);
  for (unsigned i=0;i<64;++i) assert(workspace[i]==0xcd && workspace[MODEL_WORKSPACE_BYTES+64+i]==0xcd);
  const int64_t invalid[]={-1,5,INT64_MIN,INT64_MAX,INT64_C(4294967296)};
  for (unsigned k=0;k<sizeof(invalid)/sizeof(*invalid);++k) {
    state=(model_driver_state){0}; indices[1]=invalid[k]; memcpy(input+1,indices,16);
    memset(output,0xa7,sizeof(output)); memset(returned,0xa7,sizeof(returned)); memset(extrema,0xa7,sizeof(extrema));
    assert(model_invoke(&state,workspace+64,MODEL_WORKSPACE_BYTES,inputs,outputs,0)==MODEL_DRIVER_KERNEL_ERROR);
    assert(state.kernel_status==-2 && state.completed==0);
    for (unsigned i=0;i<sizeof(output);++i) assert(output[i]==0xa7);
    for (unsigned i=0;i<sizeof(returned);++i) assert(((unsigned char *)returned)[i]==0xa7);
    for (unsigned i=0;i<sizeof(extrema);++i) assert(((unsigned char *)extrema)[i]==0xa7);
  }
  return 0;
}
""")
    subprocess.run(
        [
            shutil.which("cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "model.c",
            "model_driver.c",
            "lookup.c",
            "check.c",
            "-o",
            "check",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)


@pytest.mark.parametrize(
    "dtype,width,encoding,policy",
    [
        ("bf16", 2, "bf16-le-raw", "raw-bit-pattern"),
        ("fp8_e4m3", 1, "fp8_e4m3-raw", "raw-bit-pattern"),
        ("f64", 8, "f64-le-raw", "raw-bit-pattern"),
        ("i32", 4, "i32-le", "integer-modulo-2pow32"),
        ("i8", 1, "i8", "integer-modulo-2pow8"),
        ("i1", 1, "i1-raw", "boolean"),
    ],
)
def test_application_transports_typed_bits_through_compiled_layouts(tmp_path, dtype, width, encoding, policy):
    from merlin.compile.layout_c import emit_layout_c

    type_ = TensorType((2, 3), dtype, policy)
    # Include all-one and sign bits without converting through a host float.
    payload = bytes([0, 1, 1, 0, 1, 0]) if dtype == "i1" else bytes(range(width * 6))
    if dtype != "i1":
        payload = bytes([0xFF]) * width + bytes([0x80]) * width + payload[2 * width :]
    x = SemanticNode("x", "input", (), type_, effect="input")
    c = SemanticNode("c", "constant", (), type_, effect="constant")
    result = SemanticNode("out", "concat", (x.id, c.id), TensorType((4, 3), dtype, policy), (("axis", 0),))
    request = KernelRequest(
        (x, c, result),
        (result.id, x.id, c.id, result.id),
        ("external",) * 4,
        ((x.id, "external"),),
        "synthetic",
        constants=(ConstantBinding(c.id, encoding, payload.hex()),),
    )
    plan = plan_for(request, ((0,),))
    plan = replace(plan, selected=tuple(replace(region, placement="host") for region in plan.selected))
    outlined = outline_semantic_regions(request, entry="model", plan=plan)
    program = build_program(
        outlined.logical,
        global_plan=outlined.plan,
        global_plan_emitter=outlined,
        memory_requirements=lambda graph: ArenaRequirements(2048, (0,), (1,), dispatch_digest=dispatch_digest(graph)),
    )
    driver, driver_files = emit_program_c(program, "model_driver", {0: DispatchCall("layout", "host")})
    interface, files = emit_application_c(request, driver, driver_files, "model", capacity=2048)
    from merlin.compile.semantic_regions import extract_semantic_region

    _, layout_files = emit_layout_c(extract_semantic_region(request, (result.id,)), "layout")
    for name, text in {**driver_files, **files, **layout_files}.items():
        (tmp_path / name).write_text(text)
    assert interface["pointer_abi"] == "opaque_native_dense_buffers"
    assert interface["input_types"][0]["dtype"] == dtype
    assert [row["bytes"] for row in interface["constants"]] == [len(payload)]
    (tmp_path / "check.c").write_text(f"""
#include "model.h"
#include <assert.h>
#include <string.h>
int main(void) {{
  const unsigned char expected[]={{{",".join(map(str, payload))}}};
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES+128];
  unsigned char input[sizeof(expected)], a[2*sizeof(expected)+2],b[sizeof(expected)+2];
  unsigned char c[sizeof(expected)+2],d[2*sizeof(expected)+2];
  const void *inputs[]={{input}}; void *outputs[]={{a+1,b+1,c+1,d+1}};
  model_driver_state state={{0}};
  memset(workspace,0xcd,sizeof(workspace));
  for (int panel=0;panel<3;++panel) {{
    for (unsigned i=0;i<sizeof(input);++i) input[i]=(unsigned char)(expected[i]^{1 if dtype == "i1" else 0x71}u*panel);
    memset(a,0xa7,sizeof(a)); memset(b,0xa7,sizeof(b)); memset(c,0xa7,sizeof(c)); memset(d,0xa7,sizeof(d));
    assert(model_invoke(&state,workspace+64,MODEL_WORKSPACE_BYTES,inputs,outputs,0)==0);
    assert(memcmp(a+1,input,sizeof(input))==0 && memcmp(a+1+sizeof(input),expected,sizeof(expected))==0);
    assert(memcmp(b+1,input,sizeof(input))==0 && memcmp(c+1,expected,sizeof(expected))==0);
    assert(memcmp(a,d,sizeof(a))==0);
    assert(a[0]==0xa7 && a[sizeof(a)-1]==0xa7 && b[0]==0xa7 && b[sizeof(b)-1]==0xa7);
    assert(c[0]==0xa7 && c[sizeof(c)-1]==0xa7);
    for (unsigned i=0;i<64;++i) assert(workspace[i]==0xcd && workspace[MODEL_WORKSPACE_BYTES+64+i]==0xcd);
  }}
  outputs[0]=input;
  assert(model_invoke(&state,workspace+64,MODEL_WORKSPACE_BYTES,inputs,outputs,0)!=0);
  return 0;
}}
""")
    subprocess.run(
        [
            shutil.which("cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "model.c",
            "model_driver.c",
            "layout.c",
            "check.c",
            "-o",
            "check",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)
