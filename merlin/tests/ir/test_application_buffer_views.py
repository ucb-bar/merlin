"""Dense views preserve ownership, source proofs and generated C execution."""

import json
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from merlin.compile.application_plan import application_plan
from merlin.compile.application_views import IMPLEMENTATION, checked_view_aliases, select_dense_buffer_views
from merlin.compile.semantic_application import (
    CompiledCRegion,
    compile_semantic_application,
    verify_semantic_application,
)
from merlin.compile.semantic_regions import outline_semantic_regions
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options, options_from_args
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.tests.ir.test_semantic_regions import plan_for


def fixture(count=33):
    tensor = TensorType((count,), "i32", "signed-i32")
    nodes = (
        SemanticNode("x", "input", (), tensor, effect="input"),
        SemanticNode("a", "xor", ("x",), tensor),
        SemanticNode("b", "reshape", ("a",), replace(tensor, shape=(1, count))),
        SemanticNode("c", "copy", ("b",), tensor),
        SemanticNode("temporary", "xor", ("x",), tensor),
        SemanticNode("later", "xor", ("temporary",), tensor),
        SemanticNode("out", "xor", ("c", "later"), tensor),
    )
    source = KernelRequest(nodes, ("out",), ("external",), (("x", "external"),), "synthetic")
    plan = plan_for(source, tuple((i,) for i in range(len(nodes) - 1)))
    return source, replace(plan, selected=tuple(replace(row, placement="host") for row in plan.selected))


def test_selection_is_explicit_and_identity_records_the_option():
    source, plan = fixture()
    adapter = SimpleNamespace(plan=lambda _: plan)
    assert application_plan(source, adapter) == plan
    with codegen_options(CodegenOptions(application_buffer_views=True)):
        selected = application_plan(source, adapter)
    assert checked_view_aliases(source, selected) == (("b", "a", 0), ("c", "b", 0))
    assert "application_buffer_views" not in CodegenOptions().identity_record()
    assert options_from_args(SimpleNamespace(application_buffer_views=True)).identity_record()[
        "application_buffer_views"
    ]
    with pytest.raises(ValueError, match="must be Boolean"):
        CodegenOptions(application_buffer_views=1)


@pytest.mark.parametrize(
    "change", ["public", "external", "dtype", "policy", "count", "map", "attribute", "transpose", "representation"]
)
def test_unproved_views_keep_the_original_implementation(change):
    source, _ = fixture()
    node = source.node("b")
    if change == "public":
        source = replace(source, outputs=("out", "b"), output_storages=("external",) * 2)
    elif change == "external":
        node = replace(node, inputs=("x",))
        source = replace(
            source, nodes=tuple(replace(n, inputs=("a",)) if n.id == "temporary" else n for n in source.nodes)
        )
    elif change == "dtype":
        node = replace(node, type=replace(node.type, dtype="u32"))
    elif change == "policy":
        node = replace(node, type=replace(node.type, numerical_policy="raw-bit-pattern"))
    elif change == "count":
        node = replace(node, type=replace(node.type, shape=(1, 34)))
    elif change == "map":
        node = replace(node, index_maps=(IndexMap(2, ((0, 1),), (0,)), pointwise_identity_maps(2, 1)[0]))
    elif change == "attribute":
        node = replace(node, attrs=(("unknown", "meaning"),))
    elif change == "transpose":
        node = replace(node, op="transpose")
    source = replace(source, nodes=tuple(node if n.id == "b" else n for n in source.nodes))
    plan = plan_for(source, tuple((i,) for i in range(6)))
    if change == "representation":
        item = plan.selected[1]
        output = item.outputs[0]
        item = replace(
            item, outputs=(replace(output, representation=replace(output.representation, layout="blocked")),)
        )
        plan = replace(plan, selected=tuple(item if row.id == item.id else row for row in plan.selected))
    selected = select_dense_buffer_views(source, plan)
    assert selected.selected[1] == plan.selected[1]


def test_aliases_cannot_overlap_distinct_consumer_ports_or_bypass_source_replay():
    source, plan = fixture()
    node = replace(source.node("out"), inputs=("a", "c", "later"))
    source = replace(source, nodes=(*source.nodes[:-1], node))
    plan = plan_for(source, tuple((i,) for i in range(6)))
    selected = select_dense_buffer_views(source, plan)
    # The first view can share a; the second must materialize a separate c.
    assert checked_view_aliases(source, selected) == (("b", "a", 0),)
    forged = replace(
        selected.selected[2], implementation=IMPLEMENTATION, placement="host", metadata=(("kind", "dense_buffer_view"),)
    )
    forged_plan = replace(selected, selected=(*selected.selected[:2], forged, *selected.selected[3:]))
    with pytest.raises(ValueError, match="distinct region input ports"):
        outline_semantic_regions(source, entry="test", plan=forged_plan)
    changed = replace(source, nodes=tuple(replace(n, op="transpose") if n.id == "b" else n for n in source.nodes))
    with pytest.raises(ValueError, match="source proof"):
        outline_semantic_regions(changed, entry="test", plan=selected)


@pytest.mark.parametrize("count", [1, 33, 132])
def test_published_c_views_keep_roots_live_and_replay_exactly(tmp_path, count):
    source, plan = fixture(count)

    class Adapter:
        def plan(self, source):
            return plan

        def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
            return self.export_region(region, alternative, name, segments, target_source)

        def export_region(self, region, alternative, name, segments, target_source):
            node = next(n for n in region.nodes if n.effect == "pure")
            assert node.op == "xor", "Merlin must own view compilation and replay"
            expression = "((const unsigned *)inputs[0])[i] ^ " + (
                "((const unsigned *)inputs[1])[i]" if len(node.inputs) == 2 else "0x80000001u"
            )
            body = (
                f"int {name}(void *context,const void *const *inputs,void *const *outputs) {{\n"
                f"  (void)context; for (unsigned i=0;i<{count};++i) ((unsigned *)outputs[0])[i]={expression};\n"
                "  return 0;\n}\n"
            )
            return CompiledCRegion(
                name,
                name + ".h",
                node.inputs,
                region.outputs,
                {
                    name + ".h": f"int {name}(void *,const void *const *,void *const *);\n".encode(),
                    name + ".c": body.encode(),
                },
                (),
                (),
                {},
            )

    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_text("{}\n")
    snapshot = SimpleNamespace(root=snapshot_root)
    destination = tmp_path / "application"
    adapter = Adapter()
    with codegen_options(CodegenOptions(application_buffer_views=True)):
        manifest = compile_semantic_application(
            source,
            adapter=adapter,
            snapshot=snapshot,
            target_source=tmp_path,
            destination=destination,
            limits=None,
            name="model",
            capacity=8192,
            allow_host_only=True,
        )
        memory = json.loads((destination / "program.json").read_text())["memory_plan"]
        offsets = memory["offsets"]
        assert offsets["a"] == offsets["b"] == offsets["c"]
        assert offsets["a"] != offsets["temporary"] and offsets["a"] != offsets["later"]
        assert sum(r["description"].get("kind") == "dense_buffer_view" for r in manifest["regions"]) == 2
        assert (
            verify_semantic_application(
                source,
                adapter=adapter,
                snapshot=snapshot,
                target_source=tmp_path,
                artifact=destination,
                allow_host_only=True,
            )["status"]
            == "checked"
        )
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    package = destination / "c-package"
    (tmp_path / "check.c").write_text(f"""
#include "combined/model_combined.h"
#include <assert.h>
int main(void) {{
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES];
  model_driver_state state={{0}};
  unsigned input[{count}], output[{count}];
  for (unsigned i=0;i<{count};++i) input[i]=i*65537u;
  const void *inputs[]={{input}}; void *outputs[]={{output}};
  void *items[6]={{0}}; model_driver_contexts contexts={{items,6}};
  assert(model_invoke(&state,workspace,sizeof(workspace),inputs,outputs,&contexts)==0);
  for (unsigned i=0;i<{count};++i) {{assert(output[i]==0x80000001u); assert(input[i]==i*65537u);}}
  assert(state.completed==6 && !state.active);
  assert(region_001(0,inputs,outputs)!=0);
  return 0;
}}
""")
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "combined/model_combined.c",
            str(tmp_path / "check.c"),
            "-I",
            str(package),
            "-o",
            str(tmp_path / "check"),
        ],
        cwd=package,
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)
    # Even rehashing a forged allocation cannot grant an unproved alias.
    from hashlib import sha256

    program = json.loads((destination / "program.json").read_text())
    program["memory_plan"]["offsets"]["c"] += 64
    (destination / "program.json").write_text(json.dumps(program))
    manifest["stage_artifacts"]["program.json"] = sha256((destination / "program.json").read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with codegen_options(CodegenOptions(application_buffer_views=True)), pytest.raises(ValueError):
        verify_semantic_application(
            source,
            adapter=adapter,
            snapshot=snapshot,
            target_source=tmp_path,
            artifact=destination,
            allow_host_only=True,
        )
