"""Publish, replay and execute every recorded trip through the normal C package."""
from dataclasses import replace
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from merlin.compile.semantic_application import (
    CompiledCRegion, compile_semantic_application, verify_semantic_application,
)
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.tests.ir.test_semantic_regions import plan_for


def fixture():
    tensor = TensorType((1,), "f32", "raw-bit-pattern")
    nodes = [SemanticNode("x", "input", (), tensor, effect="input")]
    previous = "x"
    for i in range(22):
        name = f"value_{i}"
        nodes.append(SemanticNode(name, "f32_add", (previous, previous), tensor,
                     (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
                     index_maps=pointwise_identity_maps(1, 2)))
        previous = name
    source = KernelRequest(tuple(nodes), (previous,), ("external",), (("x", "external"),), "synthetic")
    plan = plan_for(source, tuple((i,) for i in range(22)))
    plan = replace(plan, selected=tuple(replace(row, placement="host") for row in plan.selected))

    class Adapter:
        region_loop_parameters = ()

        def plan(self, source):
            return plan

        def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
            return self.export_region(region, alternative, name, segments, target_source)

        def export_region(self, region, alternative, name, segments, target_source):
            header = f"int {name}(void *,const void *const *,void *const *);\n"
            body = (f"int {name}(void *c,const void *const *in,void *const *out) {{\n"
                    "  (void)c; *(float *)out[0]=2.0f*(*(const float *)in[0]); return 0;\n}\n")
            return CompiledCRegion(name, name + ".h", tuple(k for k, _ in region.input_storages),
                                   region.outputs, {name + ".h": header.encode(), name + ".c": body.encode()},
                                   (), (), {})
    return source, Adapter()


def test_published_loop_replays_and_normal_invoke_executes_all_22_trips(tmp_path):
    source, adapter = fixture()
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_text("{}\n")
    snapshot = SimpleNamespace(root=snapshot_root)
    destination = tmp_path / "application"
    manifest = compile_semantic_application(source, adapter=adapter, snapshot=snapshot,
        target_source=tmp_path, destination=destination, limits=None, name="model", capacity=4096,
        allow_host_only=True, emit_region_loop=True)
    assert manifest["dispatch_loop"] is True
    loop = json.loads((destination / "dispatch-c.json").read_bytes())["loop"]
    assert (loop["offset"], loop["period"], loop["trip_count"]) == (0, 1, 22)
    assert manifest["application_interface"]["dispatch_loop"]["trip_count"] == 22
    assert (destination / "handoff/model_combined.c").read_bytes() == (
        destination / "c-package/combined/model_combined.c").read_bytes()
    verified = verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
        target_source=tmp_path, artifact=destination, allow_host_only=True)
    assert verified["status"] == "checked"
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    package = destination / "c-package"
    (tmp_path / "check.c").write_text(r'''
#include "combined/model_combined.h"
#include <assert.h>
int main(void) {
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES];
  model_driver_state state={0}; float input=1.0f, output=0;
  const float *inputs[]={&input}; float *outputs[]={&output};
  void *items[22]={0}; model_driver_contexts contexts={items,22};
  assert(model_invoke(&state,workspace,sizeof(workspace),inputs,outputs,&contexts)==0);
  assert(output==4194304.0f && input==1.0f && state.completed==22 && !state.active);
  return 0;
}
''')
    subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "combined/model_combined.c", str(tmp_path / "check.c"), "-I", str(package),
                    "-o", str(tmp_path / "check")],
                   cwd=package, check=True, capture_output=True, timeout=60)
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True, timeout=10)
    # A changed option cannot silently replay a different C package.
    manifest["dispatch_loop"] = False
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from replayed"):
        verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
            target_source=tmp_path, artifact=destination, allow_host_only=True)
