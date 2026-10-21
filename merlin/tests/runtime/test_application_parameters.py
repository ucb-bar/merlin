"""External immutable parameters retain bytes, shared ports and input order."""

import json
import os
import shutil
import struct
import subprocess
from dataclasses import replace
from hashlib import sha256

import pytest

from merlin.compile.application_c import emit_application_c
from merlin.compile.c_parameters import bind_application_parameters
from merlin.compile.dispatch_c import DispatchCall, emit_program_c
from merlin.compile.layout_c import emit_layout_c
from merlin.compile.semantic_regions import extract_semantic_region, outline_semantic_regions
from merlin.frontends.compile_inputs import load_compile_inputs, stage_compile_inputs
from merlin.llvmlower.weights_pack import checked_payload
from merlin.runtime.program import build_program
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.tests.ir.test_compile_inputs import _write_weighted_capture
from merlin.tests.ir.test_semantic_regions import plan_for
from merlin.xdsl_dialects.lowering.arena_plan import ArenaRequirements
from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest


def setup(tmp_path):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _write_weighted_capture(capture, shared_weight=True)
    file = capture / "weights.safetensors"
    raw = file.read_bytes()[:-16] + struct.pack("<4I", 0x80000000, 1, 0x7FC12345, 0x7F7FFFFF)
    file.write_bytes(raw)
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    receipt["artifacts"][file.name] = {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
    (capture / "capture_receipt.json").write_text(json.dumps(receipt))
    stage_compile_inputs(capture, stage)
    staged = load_compile_inputs(stage)
    type_ = TensorType((2, 2), "f32", "raw-bit-pattern")
    inputs = tuple(SemanticNode(f"arg{i}", "input", (), type_, effect="input") for i in range(3))
    result = SemanticNode("out", "concat", tuple(node.id for node in inputs),
                          replace(type_, shape=(6, 2)), (("axis", 0),))
    # This composed graph is a public ABI unit fixture, not capture equivalence evidence.
    source = KernelRequest((*inputs, result), (result.id, "arg0", "arg2"), ("external",) * 3,
                           tuple((node.id, "external") for node in inputs), "synthetic")
    parameters = bind_application_parameters(stage, staged, source)
    return source, parameters


def test_external_parameter_c_runs_with_one_runtime_port_and_shared_weights(tmp_path):
    source, parameters = setup(tmp_path)
    plan = plan_for(source, ((0,),))
    plan = replace(plan, selected=tuple(replace(row, placement="host") for row in plan.selected))
    outlined = outline_semantic_regions(source, entry="model", plan=plan)
    program = build_program(outlined.logical, global_plan=plan, global_plan_emitter=outlined,
                            memory_requirements=lambda graph: ArenaRequirements(
                                1024, (0,), (1,), dispatch_digest=dispatch_digest(graph)))
    driver, driver_files = emit_program_c(program, "model_driver", {0: DispatchCall("layout", "host")},
                                         read_only_inputs=parameters.node_ids)
    interface, files = emit_application_c(source, driver, driver_files, "model", capacity=1024,
                                           parameters=parameters)
    assert interface["inputs"] == ["arg2"] and parameters.node_ids == ("arg0", "arg1")
    assert interface["model_parameters"]["payload_bytes"] == 16
    _, layout = emit_layout_c(extract_semantic_region(source, ("out",)), "layout")
    for name, text in {**driver_files, **files, **layout}.items():
        (tmp_path / name).write_text(text)
    parameters.write_payload(tmp_path / "parameters.bin")
    assert sha256((tmp_path / "parameters.bin").read_bytes()).hexdigest() == parameters.payload_sha256
    (tmp_path / "main.c").write_text(r"""
#include "model.h"
#include <assert.h>
#include <string.h>
int main(void) {
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES];
  _Alignas(8) uint32_t weights[]={0x80000000,1,0x7fc12345,0x7f7fffff};
  float input[4],joined[12],returned[4],copied[4];
  const float *inputs[]={input};float *outputs[]={joined,returned,copied};
  model_parameters parameters={0},before;model_driver_state state={0};
  memset(joined,0xcd,sizeof(joined));
  assert(model_invoke(&state,workspace,sizeof(workspace),&parameters,inputs,outputs,0)==MODEL_DRIVER_INVALID);
  assert(!state.active && ((unsigned char*)joined)[0]==0xcd);
  assert(model_parameters_bind(&parameters,weights,sizeof(weights)-1)==-1);
  assert(model_parameters_bind(&parameters,weights,sizeof(weights))==0);
  before=parameters;weights[3]^=1;
  assert(model_parameters_bind(&parameters,weights,sizeof(weights))==-2);
  assert(memcmp(&before,&parameters,sizeof(before))==0);weights[3]^=1;
  for(unsigned panel=0;panel<3;++panel) {
    for(unsigned i=0;i<4;++i) input[i]=(float)(panel+i);
    assert(model_invoke(&state,workspace,sizeof(workspace),&parameters,inputs,outputs,0)==0);
    assert(memcmp(joined,weights,16)==0 && memcmp(joined+4,weights,16)==0);
    assert(memcmp(joined+8,input,16)==0 && memcmp(returned,weights,16)==0 && memcmp(copied,input,16)==0);
  }
  parameters.data=(const unsigned char*)weights+4;
  assert(model_invoke(&state,workspace,sizeof(workspace),&parameters,inputs,outputs,0)==MODEL_DRIVER_INVALID);
  parameters=before;
  outputs[1]=(float*)weights;
  assert(model_invoke(&state,workspace,sizeof(workspace),&parameters,inputs,outputs,0)==MODEL_DRIVER_INVALID);
  assert(!state.active);
  return 0;
}
""")
    command = [shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
               "-fsanitize=undefined", "-fno-sanitize-recover=all", "model.c", "model_driver.c", "layout.c",
               "main.c", "-o", "check"]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)
    if cross := os.environ.get("MERLIN_RISCV_GCC"):
        for name in ("model", "model_driver", "layout"):
            subprocess.run([cross, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-march=rv64gcv",
                            "-mabi=lp64d", "-mstrict-align", "-c", name + ".c", "-o", name + ".o"],
                           cwd=tmp_path, check=True, capture_output=True)


def test_parameter_table_and_payload_mutations_refuse(tmp_path):
    source, parameters = setup(tmp_path)
    for entry in (replace(parameters.entries[0], offset=4), replace(parameters.entries[0], dtype="BF16")):
        with pytest.raises(ValueError):
            replace(parameters, entries=(entry, parameters.entries[1])).check(source)
    with parameters.archive_path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x01")
    destination = tmp_path / "bad.bin"
    with pytest.raises(ValueError, match="archive changed"):
        parameters.write_payload(destination)
    assert not destination.exists()


def test_checked_payload_does_not_replace_existing_output(tmp_path):
    _, parameters = setup(tmp_path)
    path = tmp_path / "existing.bin"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        checked_payload(parameters.archive_path, archive_sha256=parameters.archive_sha256, out_blob=path)
    assert path.read_bytes() == b"keep"
