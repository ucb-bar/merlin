"""Compile and freshly replay compact residency using a synthetic provider.

The native image is an identity fixture, not executed device code. Generated C
executes with an explicit resident-data context to test application publication.
"""
import json
import shutil
import subprocess
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest

from merlin.compiler.build.regions import (
    CompiledCRegion, compile_semantic_application, verify_semantic_application,
)
from merlin.compiler.semantic.codegen import CodegenOptions, codegen_options, current_codegen_options
from merlin.compiler.semantic.model import SemanticNode, TensorType
from merlin.compiler.semantic.target_binding import verify_native_publication
from semantic_application_fixtures import plan_for
from test_compact_parameters import fixture


def setup(tmp_path):
    source, original = fixture(tmp_path)
    resident_result = SemanticNode("resident_result", "reshape", ("arg2",), source.node("arg2").type)
    result = SemanticNode("out", "concat", ("arg0", "arg1", "resident_result", "arg3"),
                          TensorType((5,), "f32", "raw-bit-pattern"),
                          (("axis", 0),))
    source = replace(source, nodes=(*source.nodes, resident_result, result), outputs=("out",))
    parameters = replace(original, source_request_digest=source.digest())
    plan = plan_for(source, ((0,), (1,)))
    plan = replace(plan, selected=tuple(replace(row, placement=placement)
                 for row, placement in zip(plan.selected, ("accelerator", "host"))))

    class Adapter:
        compiles = 0
        exports = 0

        def plan(self, request):
            assert request == source
            return plan

        def resident_memory_geometry(self):
            return {"dma_base": 0, "dma_capacity": 65536, "alignment": 4}

        def inventory_resident_inputs(self, region, alternative):
            return [{"node_id": "arg2", "bytes": 8, "alignment": 4,
                     "layout": {"schema": "test.dense-resident.v1"}}]

        def pack_resident_input(self, region, node_id, dense, alternative):
            assert node_id == "arg2" and len(dense) == 8
            return dense

        def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
            self.compiles += 1
            if alternative.placement == "accelerator":
                root = segments / name
                root.mkdir()
                binary = b"synthetic identity fixture; not device code\n"
                (root / "program.bin").write_bytes(binary)
                manifest = {"engine": "merlin_native", "request_digest": region.digest(),
                            "target_identity": source.target_identity,
                            "binary_sha256": sha256(binary).hexdigest()}
                (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
            return self.export_region(region, alternative, name, segments, target_source)

        def export_region(self, region, alternative, name, segments, target_source):
            self.exports += 1
            device = alternative.placement == "accelerator"
            description, programs = {}, ()
            if device:
                root = segments / name
                manifest = json.loads((root / "manifest.json").read_bytes())
                verify_native_publication(root, manifest, engine="merlin_native", request_digest=region.digest(),
                                          target_identity=source.target_identity)
                binding = current_codegen_options().resident_bindings[region.digest()]["arg2"]
                description = {"resident_bindings": [{"node_id": "arg2", **binding}]}
                programs = (f"segments/{name}",)
            header = f"int {name}(void *,const void *const *,void *const *);\n"
            inputs = tuple(key for key, _ in region.input_storages)
            resident_index = inputs.index("resident_result") if not device else None
            tail_index = inputs.index("arg3") if not device else None
            body = f'''#include <assert.h>
#include <string.h>
int {name}(void *context,const void *const *in,void *const *out) {{
  (void)context;
  assert(in[0] && in[0]==in[1] && in[2] && in[3]);
  unsigned char *result=out[0];
  memcpy(result,in[0],4);memcpy(result+4,in[1],4);
  memcpy(result+8,in[{resident_index}],8);memcpy(result+16,in[{tail_index}],4);
  return 0;
}}
'''
            if device:
                body = f'''#include <assert.h>
#include <string.h>
int {name}(void *context,const void *const *in,void *const *out) {{
  assert(context && !in[0]);memcpy(out[0],context,8);return 0;
}}
'''
            return CompiledCRegion(name, name + ".h", inputs,
                                   region.outputs, {name + ".h": header.encode(), name + ".c": body.encode()},
                                   (), programs, description)

    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_text("{}\n")
    options = CodegenOptions(application_residency=True, io_memory_base=4096, io_memory_bytes=256,
                             io_memory_capacity_base=4096, io_memory_capacity_bytes=4096)
    return source, parameters, Adapter(), SimpleNamespace(root=snapshot_root), options


def compile_fixture(tmp_path):
    source, parameters, adapter, snapshot, options = setup(tmp_path)
    destination = tmp_path / "application"
    with codegen_options(options):
        manifest = compile_semantic_application(source, adapter=adapter, snapshot=snapshot,
            target_source=tmp_path, destination=destination, limits=None, name="model", capacity=1024,
            parameters=parameters)
    return source, parameters, adapter, snapshot, options, destination, manifest


def test_compilation_internal_replay_and_fresh_replay_publish_compact_bytes(tmp_path):
    source, parameters, adapter, snapshot, options, destination, manifest = compile_fixture(tmp_path)
    assert adapter.compiles == 2 and adapter.exports >= 4
    assert (destination / "handoff/parameters.bin").read_bytes() == bytes(range(4)) + bytes(range(12, 16))
    published = manifest["application_interface"]["model_parameters"]
    assert published["schema"] == "merlin.compact_application_parameters.v1"
    assert published["resident_only"] == ["arg2"]
    assert published["source_parameters"]["payload_sha256"] == parameters.payload_sha256
    assert manifest["request_digest"] == source.digest() == parameters.source_request_digest
    assert json.loads((destination / "dispatch-c.json").read_bytes())["resident_only_inputs"] == ["arg2"]
    residency = json.loads((destination / "residency.json").read_bytes())
    load, = residency["loads"]
    assert (destination / load["path"]).read_bytes() == bytes(range(4, 12))
    with codegen_options(options):
        checked = verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
            target_source=tmp_path, artifact=destination, parameters=parameters)
    assert checked["status"] == "checked"
    assert adapter.exports >= 6
    package = destination / "c-package"
    (tmp_path / "main.c").write_text(r'''
#include "combined/model_combined.h"
#include <assert.h>
#include <string.h>
int main(void) {
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES];
  _Alignas(8) unsigned char host[]={0,1,2,3,12,13,14,15};
  unsigned char resident[]={4,5,6,7,8,9,10,11};void *items[]={resident,0};
  model_driver_contexts contexts={items,2};
  model_driver_state state={0};model_parameters parameters={0};
  float output[5];float *outputs[]={output};
  assert(model_parameters_bind(&parameters,host,sizeof(host))==0);
  assert(model_invoke(&state,workspace,sizeof(workspace),&parameters,0,outputs,&contexts)==0);
  const unsigned char expected[]={0,1,2,3,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  assert(memcmp(output,expected,sizeof(expected))==0 && state.completed==2);
  resident[7]^=1;
  assert(model_invoke(&state,workspace,sizeof(workspace),&parameters,0,outputs,&contexts)==0);
  assert(memcmp(output,expected,sizeof(expected))!=0);
  return 0;
}
''')
    compiler = shutil.which("cc")
    assert compiler, "C compiler is required for generated publication execution"
    built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                            "combined/model_combined.c", str(tmp_path / "main.c"), "-I", str(package),
                            "-o", str(tmp_path / "check")], cwd=package, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)


def test_compact_and_original_source_mutations_both_refuse_fresh_replay(tmp_path):
    for index, changed in enumerate(["compact", "resident_source"]):
        case = tmp_path / f"case_{index}"
        case.mkdir()
        _compact_and_original_source_mutations_both_refuse_fresh_replay(case, changed)


def _compact_and_original_source_mutations_both_refuse_fresh_replay(tmp_path, changed):
    source, parameters, adapter, snapshot, options, destination, _ = compile_fixture(tmp_path)
    if changed == "compact":
        path = destination / "handoff/parameters.bin"
        raw = bytearray(path.read_bytes())
        raw[0] ^= 1
    else:
        path = parameters.archive_path
        raw = bytearray(path.read_bytes())
        # Target a removed resident-only byte; compact host backing is unchanged.
        raw[-5] ^= 1
    path.write_bytes(raw)
    with codegen_options(options), pytest.raises(ValueError):
        verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
            target_source=tmp_path, artifact=destination, parameters=parameters)
