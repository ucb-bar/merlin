"""Execute session orchestration without framework inputs or reference data."""

import json
import shutil
import subprocess

import pytest
import yaml

from merlin.compile.session_dispatch import SessionImplementation, emit_session_dispatch
from merlin.llvmlower.session_bundle import load


def session_source(root):
    root.mkdir()
    programs = []
    for name, steps in (("encode", 1), ("advance", 4), ("decode", 1)):
        stage = root / name
        stage.mkdir()
        body = {
            "encode": "%c = arith.constant dense<2.0> : tensor<2xf32>\n%y = arith.mulf %x, %c : tensor<2xf32>",
            "advance": "%c = arith.constant dense<3.0> : tensor<2xf32>\n%y = arith.addf %x, %c : tensor<2xf32>",
            "decode": "",
        }[name]
        value = "%y" if body else "%x"
        (stage / "model.mlir").write_text(
            "module { func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {\n"
            + body + f"\nreturn {value} : tensor<2xf32>\n}} }}\n"
        )
        (stage / "weights.safetensors.manifest.json").write_text(json.dumps({"entries": []}))
        if steps > 1:
            (stage / "session_contract.yaml").write_text(yaml.safe_dump({
                "version": 1, "stages": [name], "steps": steps,
                "states": [{"name": "state", "input_arg": 0, "output_index": 0}],
                "quality": {"output_index": 0},
            }))
        programs.append({"name": name, "bundle": name, "steps": steps})
    raw = {
        "version": 2, "stages": [p["name"] for p in programs], "programs": programs,
        "stage_schedule": [{"name": p["name"], "steps": p["steps"], "execution": "compiled", "timed": True}
                           for p in programs],
        "bindings": [
            {"from": {"program": a, "output_index": 0}, "to": {"program": b, "input_arg": 0}}
            for a, b in (("encode", "advance"), ("advance", "decode"))
        ],
        "states": [{"name": "state"}], "quality": {"program": "advance"},
    }
    (root / "session_contract.yaml").write_text(yaml.safe_dump(raw))
    return root


def emit(root):
    session = load(root, runtime_inputs=False)
    return emit_session_dispatch(session, {
        "encode": SessionImplementation("encode_call", (0,), "compiled_cpu"),
        "advance": SessionImplementation("advance_call", (0,), "compiled_cpu"),
    }, name="pipeline", capacity=1024)


def test_compiler_only_intake_and_recurrent_c_execution(tmp_path):
    root = session_source(tmp_path / "source")
    with pytest.raises(ValueError, match="inputs.npz"):
        load(root)
    description, files = emit(root)
    assert len(description["program"]["dispatch"]["nodes"]) == 5
    assert len(description["contexts"]) == 2
    assert description["stage_outputs"]["decode"] == description["stage_outputs"]["advance"]
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    fixture = tmp_path / "fixture.c"
    rows = description["driver"]["buffers"]
    assert len(rows) == 2
    fixture.write_text(r'''
#include "pipeline.h"
#include <assert.h>
static int calls;
int encode_call(void *ctx,const void *const *in,void *const *out) {
  assert(ctx && *(int *)ctx==10 && calls++%5==0);
  const float *a=in[0]; float *b=out[0];
  for (int i=0;i<2;++i) b[i]=2*a[i];
  return 0;
}
int advance_call(void *ctx,const void *const *in,void *const *out) {
  assert(ctx && *(int *)ctx==20 && calls%5!=0); ++calls;
  const float *a=in[0]; float *b=out[0];
  for (int i=0;i<2;++i) b[i]=a[i]+3;
  return 0;
}
int main(void) {
  _Alignas(64) unsigned char arena[1024];
  struct { float before,input[2],after; } a={91,{2,-3},92};
  struct { float before,output[2],after; } b={93,{0,0},94};
  int ca=10,cb=20; void *items[]={&ca,&cb};
  pipeline_contexts contexts={items,2}; pipeline_state state={0};
  pipeline_buffer buffers[]={{a.input,sizeof(a.input)},{b.output,sizeof(b.output)}};
  for(int repeat=0;repeat<2;++repeat) {
    assert(pipeline_run(&state,arena,sizeof(arena),buffers,2,&contexts)==0);
    assert(b.output[0]==16 && b.output[1]==6);
    assert(a.input[0]==2 && a.input[1]==-3);
    assert(a.before==91 && a.after==92 && b.before==93 && b.after==94);
  }
  assert(calls==10);
  return 0;
}
''')
    subprocess.run([shutil.which("cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                    str(tmp_path / "pipeline.c"), str(fixture), "-o", str(tmp_path / "run")], check=True)
    subprocess.run([str(tmp_path / "run")], check=True)


def test_missing_computation_and_changed_state_edges_are_rejected(tmp_path):
    root = session_source(tmp_path / "source")
    (root / "decode/model.mlir").write_text("""module {
      func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {
        %y = arith.addf %x, %x : tensor<2xf32>
        return %y : tensor<2xf32>
      }
    }""")
    with pytest.raises(ValueError, match="only a return"):
        emit(root)
    path = root / "advance/session_contract.yaml"
    child = yaml.safe_load(path.read_text())
    child["states"][0]["output_index"] = 1
    path.write_text(yaml.safe_dump(child))
    with pytest.raises(ValueError, match="matching input and output"):
        emit(root)


def test_session_invocation_budget_and_omitted_recurrent_ports(tmp_path):
    session = load(session_source(tmp_path / "source"), runtime_inputs=False)
    with pytest.raises(ValueError, match="invocation budget"):
        emit_session_dispatch(session, {}, name="pipeline", capacity=1024, max_invocations=5)
    implementations = {
        "encode": SessionImplementation("encode_call", (0,), "compiled_cpu"),
        "advance": SessionImplementation("advance_call", (), "compiled_cpu"),
    }
    with pytest.raises(ValueError, match="omits a recurrent input"):
        emit_session_dispatch(session, implementations, name="pipeline", capacity=1024)


def test_multiple_outputs_and_nonzero_state_port_keep_their_order(tmp_path):
    root = session_source(tmp_path / "source")
    for name in ("advance", "decode"):
        body = "%y = arith.addf %a, %b : tensor<2xf32>\n" if name == "advance" else ""
        values = "%y, %a" if body else "%b, %a"
        (root / name / "model.mlir").write_text(
            "module { func.func @forward(%a: tensor<2xf32>, %b: tensor<2xf32>) "
            "-> (tensor<2xf32>, tensor<2xf32>) {\n" + body
            + f"return {values} : tensor<2xf32>, tensor<2xf32>\n}} }}"
        )
    path = root / "advance/session_contract.yaml"
    child = yaml.safe_load(path.read_text())
    child["states"][0].update(input_arg=1, output_index=1)
    path.write_text(yaml.safe_dump(child))
    path = root / "session_contract.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["bindings"].append({"from": {"program": "advance", "output_index": 1},
                            "to": {"program": "decode", "input_arg": 1}})
    path.write_text(yaml.safe_dump(raw))
    description, _ = emit_session_dispatch(load(root, runtime_inputs=False), {
        "encode": SessionImplementation("encode_call", (0,), "compiled_cpu"),
        "advance": SessionImplementation("advance_call", (1, 0), "compiled_cpu"),
    }, name="pipeline", capacity=1024)
    nodes = description["program"]["dispatch"]["nodes"]
    assert nodes[1]["inputs"] == ["stage_1_arg_1", "stage_0_step_0_out_0"]
    assert nodes[2]["inputs"] == ["stage_1_step_0_out_1", "stage_0_step_0_out_0"]
    assert description["stage_outputs"]["decode"] == ("stage_1_step_3_out_1", "stage_1_step_3_out_0")
    assert description["states"]["state"] == "stage_1_step_3_out_1"


def test_forwarded_stage_does_not_reserve_a_compiled_context(tmp_path):
    root = session_source(tmp_path / "source")
    (root / "encode/model.mlir").write_text(
        "module { func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> { return %x : tensor<2xf32> } }"
    )
    description, _ = emit_session_dispatch(load(root, runtime_inputs=False), {
        "advance": SessionImplementation("advance_call", (0,), "compiled_cpu"),
    }, name="pipeline", capacity=1024)
    assert description["contexts"] == [{"index": 0, "program": "advance", "symbol": "advance_call"}]
    assert len(description["program"]["dispatch"]["nodes"]) == 4
