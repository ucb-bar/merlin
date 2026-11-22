"""Execute one shared C body with distinct weights and contexts per trip."""

import json
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.compiler.build.regions import (
    CompiledCRegion,
    _bind_normal_region,
    _finalize_loop_regions,
    _normal_loop,
    _render,
)
from merlin.compiler.build.staging import LoopBound
from merlin.compiler.partition.semantic_regions import outline_semantic_regions
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from semantic_application_fixtures import plan_for


def fixture(trips, *, decline=False, bad_ports=False):
    tensor = TensorType((1,), "f32", "raw-bit-pattern")
    nodes = [SemanticNode("input", "input", (), tensor, effect="input")]
    nodes += [SemanticNode(f"weight_{i}", "input", (), tensor, effect="input") for i in range(trips)]
    previous = "input"
    for i in range(trips):
        result = f"result_{i}"
        nodes.append(
            SemanticNode(
                result,
                "f32_add",
                (previous, f"weight_{i}"),
                tensor,
                (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
                index_maps=pointwise_identity_maps(1, 2),
            )
        )
        previous = result
    source = KernelRequest(
        tuple(nodes),
        (previous,),
        ("external",),
        tuple((node.id, "external") for node in nodes if node.effect == "input"),
        "synthetic",
    )
    plan = plan_for(source, tuple((i,) for i in range(trips)))
    plan = replace(
        plan, selected=tuple(replace(row, placement="host", implementation="shared_kernel") for row in plan.selected)
    )

    class Adapter:
        def plan(self, request):
            return plan

        def bind_loop_region(self, representative, region, alternative, nodes, segments, target_source):
            if decline:
                return None
            inputs = tuple(nodes[key] for key in representative.inputs)
            return replace(
                representative,
                inputs=inputs[::-1] if bad_ports else inputs,
                outputs=tuple(nodes[key] for key in representative.outputs),
            )

    return source, plan, Adapter()


def body(region, name):
    header = f"int {name}(void *,const void *const *,void *const *);\n"
    text = (
        f"int {name}(void *context,const void *const *in,void *const *out) {{\n"
        "  if (context) ++*(unsigned *)context;\n"
        "  *(float *)out[0]=*(const float *)in[0]+*(const float *)in[1]; return 0;\n}\n"
    )
    return CompiledCRegion(
        name,
        name + ".h",
        tuple(key for key, _ in region.input_storages),
        region.outputs,
        {name + ".h": header.encode(), name + ".c": text.encode()},
        (),
        (),
        {},
    )


def shared(source, plan, adapter):
    outlined = outline_semantic_regions(source, entry="model", plan=plan)
    loop, bound = _normal_loop(outlined, plan, adapter)
    selected = {row.id: row for row in plan.selected}
    compiled = []
    for index, (key, region) in enumerate(outlined.regions):
        package = None
        if index in bound:
            package = _bind_normal_region(adapter, bound[index], compiled, region, selected[key], None, None)
        compiled.append(package or body(region, f"region_{index:03d}"))
    return loop, bound, compiled


def test_shared_body_executes_each_weight_binding_and_catches_a_later_mutation(tmp_path):
    for index, trips in enumerate([2, 22]):
        case = tmp_path / f"case_{index}"
        case.mkdir()
        _shared_body_executes_each_weight_binding_and_catches_a_later_mutation(case, trips)


def _shared_body_executes_each_weight_binding_and_catches_a_later_mutation(tmp_path, trips):
    source, plan, adapter = fixture(trips)
    loop, bound, compiled = shared(source, plan, adapter)
    assert loop["trip_count"] == trips and tuple(bound) == tuple(range(1, trips))
    _, summaries, interface, _, files = _render(
        source, adapter, "model", 4096, compiled, plan=plan, emit_region_loop=True
    )
    assert [row.get("compiled_body_index") for row in summaries] == [None] + [0] * (trips - 1)
    assert interface["dispatch_loop"]["trip_count"] == trips
    assert json.loads(files["application-loop-plan.json"])["region_loop"]["trip_count"] == trips
    assert not any(name.startswith("c-package/regions/region_001/") for name in files)
    package = tmp_path / "package"
    for name, data in files.items():
        if name.startswith("c-package/"):
            path = package / name.removeprefix("c-package/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    (tmp_path / "check.c").write_text("""
#include "combined/model_combined.h"
#include <assert.h>
int main(int argc, char **argv) {
  (void)argv;
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES];
  model_driver_state state={0}; float input=1.0f, output=0;
  float weights[TRIPS]; unsigned visits[TRIPS]={0}; void *items[TRIPS];
  const float *inputs[TRIPS+1]; float *outputs[]={&output};
  inputs[0]=&input;
  for(unsigned i=0;i<TRIPS;++i) { weights[i]=(float)(i+1); inputs[i+1]=&weights[i]; items[i]=&visits[i]; }
  if(argc>1) weights[TRIPS-1]+=1.0f;
  model_driver_contexts contexts={items,TRIPS};
  assert(model_invoke(&state,workspace,sizeof(workspace),inputs,outputs,&contexts)==0);
  assert(state.completed==TRIPS && !state.active);
  for(unsigned i=0;i<TRIPS;++i) assert(visits[i]==1);
  return output==(float)(1+TRIPS*(TRIPS+1)/2) ? 0 : 1;
}
""")
    compiler = shutil.which("cc")
    assert compiler, "a C compiler is required; this execution gate cannot skip"
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", f"-DTRIPS={trips}",
         "combined/model_combined.c", str(tmp_path / "check.c"), "-I", str(package), "-o", str(tmp_path / "check")],
        cwd=package, check=True, capture_output=True, timeout=60,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True, timeout=10)
    negative = subprocess.run([str(tmp_path / "check"), "mutate-last-weight"], capture_output=True, timeout=10)
    assert negative.returncode == 1


def test_changed_calibration_attribute_is_not_bound():
    source, plan, adapter = fixture(2)
    nodes = list(source.nodes)
    nodes[-1] = replace(nodes[-1], attrs=(*nodes[-1].attrs, ("static_exp", 3)))
    source = replace(source, nodes=tuple(nodes))
    adapter.region_loop_parameters = ("static_exp",)
    outlined = outline_semantic_regions(source, entry="model", plan=plan)
    loop, bound = _normal_loop(outlined, plan, adapter)
    assert loop["trip_count"] == 2 and bound == {}


def test_provider_can_decline_a_body_with_baked_bindings():
    source, plan, adapter = fixture(2, decline=True)
    _, _, compiled = shared(source, plan, adapter)
    assert [row.body_index for row in compiled] == [None, None]


def test_provider_cannot_reorder_corresponding_ports():
    source, plan, adapter = fixture(2, bad_ports=True)
    with pytest.raises(ValueError, match="ports differ"):
        shared(source, plan, adapter)


def test_shared_body_must_equal_its_representative():
    source, plan, adapter = fixture(2)
    _, _, compiled = shared(source, plan, adapter)
    compiled[1] = replace(compiled[1], files={**compiled[1].files, "extra.c": b"int changed;\n"})
    with pytest.raises(ValueError, match="differs from its representative"):
        _render(source, adapter, "model", 4096, compiled, plan=plan, emit_region_loop=True)


def test_finalization_cannot_change_ordered_runtime_ports():
    source, plan, adapter = fixture(2)
    _, _, compiled = shared(source, plan, adapter)
    adapter.finalize_loop_regions = lambda rows, *, target_source: (
        replace(rows[0], inputs=rows[0].inputs[::-1]), *rows[1:]
    )
    with pytest.raises(ValueError, match="finalization changed region binding"):
        _finalize_loop_regions(adapter, compiled, None)


def test_deployment_discovery_requires_the_provider_capability(monkeypatch):
    source, plan, adapter = fixture(2)
    outlined = outline_semantic_regions(source, entry="model", plan=plan)
    monkeypatch.setattr("merlin.compiler.build.deployment.selected", lambda: True)
    assert _normal_loop(outlined, plan, adapter) == (None, {})
    adapter.supports_deployment_loop_sharing = True
    loop, bound = _normal_loop(outlined, plan, adapter)
    assert loop["trip_count"] == 2 and isinstance(bound[1], LoopBound)
