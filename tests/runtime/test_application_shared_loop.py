"""Compile, replay, and execute one shared C body with distinct weights and contexts per trip.

The accelerator region publishes a one-word synthetic program, not device code;
the loop bodies are host C that executes here.
"""

import json
import shutil
import subprocess
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest

from merlin.compiler.build.regions import CompiledCRegion, compile_semantic_application, verify_semantic_application
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.compiler.semantic.target_binding import NativeCompilationError
from semantic_application_fixtures import plan_for

PROGRAM = b"\x73\0\0\0"
WINDOW_BASE, WINDOW_BYTES = 0x10000, 4096
F32_ADD = (("numeric_semantics", "ieee754-f32-add-rne-v1"),)


def _tensor(shape):
    return TensorType(shape, "f32", "raw-bit-pattern")


def _source(trips, wide=(), tail_first=False, body=(1,)):
    """`trips` chained host additions, then one accelerator addition over `wide` (default two elements)."""
    one = _tensor(body)
    nodes = [SemanticNode("input", "input", (), one, effect="input")]
    nodes += [SemanticNode(f"weight_{i}", "input", (), one, effect="input") for i in range(trips)]
    tail = _tensor(wide or (2,))
    nodes += [SemanticNode(name, "input", (), tail, effect="input") for name in ("x", "y")]
    previous = "input"
    for i in range(trips):
        nodes.append(SemanticNode(f"result_{i}", "f32_add", (previous, f"weight_{i}"), one, F32_ADD,
                                  index_maps=pointwise_identity_maps(len(body), 2)))
        previous = f"result_{i}"
    total = SemanticNode("sum", "f32_add", ("x", "y"), tail, F32_ADD,
                         index_maps=pointwise_identity_maps(len(tail.shape), 2))
    first = next(i for i, node in enumerate(nodes) if node.effect == "pure")
    nodes.insert(first if tail_first else len(nodes), total)
    inputs = tuple((node.id, "external") for node in nodes if node.effect == "input")
    source = KernelRequest(tuple(nodes), (previous, "sum"), ("external", "external"), inputs, "synthetic")
    plan = plan_for(source, tuple((i,) for i in range(trips + 1)))
    placements = ("accelerator",) + ("host",) * trips if tail_first else ("host",) * trips + ("accelerator",)
    plan = replace(plan, selected=tuple(replace(row, placement=placement, implementation="shared_kernel")
                                        for row, placement in zip(plan.selected, placements, strict=True)))
    return source, plan


class Provider:
    """Host bodies count their context; the accelerator region publishes a synthetic one-word program."""

    def __init__(self, plan, *, decline=False, bad_ports=False, finalize=False, bad_finalize=False,
                 whole_limit=None, writes_on_bind=False, refuse_label=None, changed_body=False, bind_raises=None):
        self._plan, self.compiles, self.exports, self.binds = plan, 0, 0, 0
        self.refuse_label, self.changed_body, self.bind_raises = refuse_label, changed_body, bind_raises
        self.decline, self.bad_ports, self.whole_limit, self.writes_on_bind = (
            decline, bad_ports, whole_limit, writes_on_bind)
        if finalize or bad_finalize:
            def finalize_regions(compiled, *, target_source):
                # Future bindings are unavailable when the representative exports, so one table is published last.
                table = json.dumps([list(row.inputs) for row in compiled if row.symbol == compiled[0].symbol]).encode()
                rows = tuple(replace(row, files={**row.files, "bindings.json": table})
                             if row.symbol == compiled[0].symbol else row for row in compiled)
                if bad_finalize:
                    rows = (replace(rows[0], inputs=rows[0].inputs[::-1]), *rows[1:])
                return rows
            self.finalize_loop_regions = finalize_regions

    def plan(self, request):
        return self._plan

    def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
        self.compiles += 1
        if name == self.refuse_label:
            raise NativeCompilationError("resource_limit", "the body's own compile exceeds the selected I/O window")
        if alternative.placement == "accelerator":
            if self.whole_limit and region.node("x").type.shape == self.whole_limit:
                raise NativeCompilationError("resource_limit", "complete row loop exceeds the selected I/O window")
            (segments / name).mkdir(parents=True)
            (segments / name / "program.bin").write_bytes(PROGRAM)
            (segments / name / "manifest.json").write_text(json.dumps(dict(
                engine="merlin_native", request_digest=region.digest(), binary_sha256=sha256(PROGRAM).hexdigest(),
                target_identity=region.target_identity)))
        return self.export_region(region, alternative, name, segments, target_source)

    def export_region(self, region, alternative, name, segments, target_source):
        self.exports += 1
        inputs = tuple(key for key, _ in region.input_storages)
        header = f"int {name}(void *,const void *const *,void *const *);\n"
        if alternative.placement == "accelerator":
            text = f"int {name}(void *c,const void *const *in,void *const *out) {{ (void)c;(void)in;(void)out; return 0; }}\n"
            return CompiledCRegion(name, name + ".h", inputs, region.outputs,
                                   {name + ".h": header.encode(), name + ".c": text.encode()}, (),
                                   (f"segments/{name}",), {})
        text = (
            f"int {name}(void *context,const void *const *in,void *const *out) {{\n"
            "  if (context) ++*(unsigned *)context;\n"
            "  *(float *)out[0]=*(const float *)in[0]+*(const float *)in[1]; return 0;\n}\n"
        )
        return CompiledCRegion(name, name + ".h", inputs, region.outputs,
                               {name + ".h": header.encode(), name + ".c": text.encode()}, (), (), {})

    def bind_loop_region(self, representative, region, alternative, nodes, segments, target_source):
        self.binds += 1
        if self.decline:
            return None
        if self.bind_raises:
            raise NativeCompilationError(self.bind_raises, "binding replay exceeds the selected I/O window")
        if self.writes_on_bind:
            (segments / f"region_{self.binds:03d}").mkdir(parents=True)
        inputs = tuple(nodes[key] for key in representative.inputs)
        files = {**representative.files, "extra.c": b"int changed;\n"} if self.changed_body else representative.files
        return replace(representative, inputs=inputs[::-1] if self.bad_ports else inputs, files=files,
                       outputs=tuple(nodes[key] for key in representative.outputs))


class OnchipProvider(Provider):
    """The same provider for an on-chip deployment that declares loop sharing."""

    supports_deployment_loop_sharing = True

    def application_binding(self, name, regions, interface):
        from merlin.compiler.build.regions import ApplicationCBinding

        return ApplicationCBinding("", "", {"invoke": name + "_run", "completion": "synchronous"})

    def deployment_requirements(self, regions, native_root):
        (region,) = regions
        host = region["placement"] == "host"
        images = [dict(path=program + "/program.bin", sha256=sha256(PROGRAM).hexdigest(), bytes=4,
                       imem_offset_bytes=0, capacity_bytes=1024, upload_convention="counted_ensure_loaded")
                  for program in region["native_programs"]]
        return dict(
            schema="merlin.provider_deployment.v1", window=dict(address=WINDOW_BASE, bytes=WINDOW_BYTES),
            images=images, window_section=".device_window", image_section=".device_images",
            reservations=[] if host else [dict(name="frame", address=WINDOW_BASE, bytes=64, role="input",
                                               lifetime=dict(scope="region_call", context_index=0),
                                               promotion_receipts=[])],
            regions=[dict(context_index=0, placement=region["placement"], launches_per_invocation=0 if host else 1)],
            allocation_alignment=64, host_context_reservation_bytes=64, max_polls=1,
            host_map=dict(imem_address=0, imem_bytes=1024, csr_address=0),
            runtime=dict(host_type="h", io_type="io", start="s", wait="w", image_completion="c", reload_policy="r"),
            host_allocation=dict(address=0, bytes=1024, reserved_bytes=64, measurement="declared_reservations",
                                 declared_reservations=[dict(name="context", bytes=64)]),
            raw_scratch_bytes=1, conversion_scratch_bytes=1)


def _snapshot(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text("{}\n")
    return SimpleNamespace(root=root)


def _onchip():
    from merlin.compiler.semantic.codegen import CodegenOptions, codegen_options

    class Onchip(CodegenOptions):
        application_deployment = "onchip"

    return codegen_options(Onchip(io_memory_base=WINDOW_BASE, io_memory_bytes=WINDOW_BYTES,
                                  io_memory_capacity_base=WINDOW_BASE, io_memory_capacity_bytes=WINDOW_BYTES))


def compile_fixture(tmp_path, trips, **options):
    source, plan = _source(trips)
    adapter = Provider(plan, **options)
    snapshot = _snapshot(tmp_path / "snapshot")
    destination = tmp_path / "application"
    manifest = compile_semantic_application(source, adapter=adapter, snapshot=snapshot, target_source=tmp_path,
                                            destination=destination, limits=None, name="model", capacity=4096)
    return source, adapter, snapshot, destination, manifest


def _verify(source, adapter, snapshot, tmp_path, destination):
    return verify_semantic_application(source, adapter=adapter, snapshot=snapshot, target_source=tmp_path,
                                       artifact=destination)


CHECK = r"""
#include "combined/model_combined.h"
#include <assert.h>
int main(int argc, char **argv) {
  (void)argv;
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES];
  model_driver_state state={0}; float input=1.0f, output=0, x[2]={0}, y[2]={0}, sum[2]={0};
  float weights[TRIPS]; unsigned visits[TRIPS+1]={0}; void *items[TRIPS+1];
  const float *inputs[TRIPS+3]; float *outputs[]={&output,sum};
  inputs[0]=&input;
  for(unsigned i=0;i<TRIPS;++i) { weights[i]=(float)(i+1); inputs[i+1]=&weights[i]; }
  for(unsigned i=0;i<=TRIPS;++i) items[i]=&visits[i];
  inputs[TRIPS+1]=x; inputs[TRIPS+2]=y;
  if(argc>1) weights[TRIPS-1]+=1.0f;
  model_driver_contexts contexts={items,TRIPS+1};
  assert(model_invoke(&state,workspace,sizeof(workspace),inputs,outputs,&contexts)==0);
  assert(input==1.0f && state.completed==TRIPS+1 && !state.active);
  for(unsigned i=0;i<TRIPS;++i) assert(visits[i]==1);
  return output==(float)(1+TRIPS*(TRIPS+1)/2) ? 0 : 1;
}
"""


def test_automatic_shared_body_executes_weight_bindings_and_later_mutation(tmp_path):
    for trips in (2, 22):
        _shared_body_executes(tmp_path / f"trips_{trips}", trips)


def _shared_body_executes(tmp_path, trips):
    tmp_path.mkdir()
    source, adapter, snapshot, destination, manifest = compile_fixture(tmp_path, trips)
    assert adapter.compiles == 2 and adapter.binds == 2 * (trips - 1)  # compile, then its own replay
    assert manifest["dispatch_loop"] is True
    assert manifest["application_interface"]["dispatch_loop"]["trip_count"] == trips
    rows = json.loads((destination / "compiled-regions.json").read_bytes())
    assert [row.get("compiled_body_index") for row in rows] == [None] + [0] * (trips - 1) + [None]
    assert sorted(path.name for path in (destination / "c-package/regions").iterdir()) == [
        "region_000", f"region_{trips:03d}"]
    exports = adapter.exports
    assert _verify(source, adapter, snapshot, tmp_path, destination)["status"] == "checked"
    assert adapter.exports - exports == 2
    package = destination / "c-package"
    (tmp_path / "check.c").write_text(CHECK)
    compiler = shutil.which("cc")
    assert compiler, "a C compiler is required; this execution gate cannot skip"
    subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", f"-DTRIPS={trips}",
                    "combined/model_combined.c", str(tmp_path / "check.c"), "-I", str(package),
                    "-o", str(tmp_path / "check")], cwd=package, check=True, capture_output=True, timeout=60)
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True, timeout=10)
    negative = subprocess.run([str(tmp_path / "check"), "mutate-last-weight"], capture_output=True, timeout=10)
    assert negative.returncode == 1
    manifest["dispatch_loop"] = False
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from replayed"):
        _verify(source, adapter, snapshot, tmp_path, destination)


def test_provider_without_binding_or_declining_compiles_every_region_flat(tmp_path):
    for case in ("decline", "no_hook"):
        root = tmp_path / case
        root.mkdir()
        source, plan = _source(2)
        adapter = Provider(plan, decline=case == "decline")
        if case == "no_hook":
            adapter.bind_loop_region = None
        manifest = compile_semantic_application(source, adapter=adapter, snapshot=_snapshot(root / "snapshot"),
                                                target_source=root, destination=root / "application", limits=None,
                                                name="model", capacity=4096)
        assert adapter.compiles == 3
        assert all("compiled_body_index" not in row for row in manifest["regions"])
        if case == "no_hook":
            assert "dispatch_loop" not in manifest
            assert not (root / "application/application-loop-plan.json").exists()


def test_provider_cannot_reorder_ports_or_change_the_shared_body(tmp_path):
    with pytest.raises(ValueError, match="ports differ"):
        compile_fixture(tmp_path / "ports", 2, bad_ports=True)
    with pytest.raises(ValueError, match="differs from its representative"):
        compile_fixture(tmp_path / "body", 2, changed_body=True)


def test_complete_provider_binding_table_is_finalized_and_replayed(tmp_path):
    source, adapter, snapshot, destination, _ = compile_fixture(tmp_path, 3, finalize=True)
    table_path = destination / "c-package/regions/region_000/bindings.json"
    table = json.loads(table_path.read_bytes())
    assert len(table) == 3 and len({tuple(ports) for ports in table}) == 3
    assert adapter.compiles == 2
    assert _verify(source, adapter, snapshot, tmp_path, destination)["status"] == "checked"
    table[-1] = table[0]
    table_path.write_text(json.dumps(table))
    with pytest.raises(ValueError):
        _verify(source, adapter, snapshot, tmp_path, destination)


def test_provider_finalization_cannot_change_ordered_runtime_ports(tmp_path):
    with pytest.raises(ValueError, match="finalization changed region binding"):
        compile_fixture(tmp_path, 2, bad_finalize=True)


def test_changed_calibration_attribute_is_not_bound(tmp_path):
    from merlin.compiler.build.regions import _normal_loop
    from merlin.compiler.partition.semantic_regions import outline_semantic_regions

    source, plan = _source(2)
    nodes = list(source.nodes)
    index = next(i for i, node in enumerate(nodes) if node.id == "result_1")
    nodes[index] = replace(nodes[index], attrs=(*nodes[index].attrs, ("static_exp", 3)))
    source = replace(source, nodes=tuple(nodes))
    adapter = Provider(plan)
    adapter.region_loop_parameters = ("static_exp",)
    loop, bound = _normal_loop(outline_semantic_regions(source, entry="model", plan=plan), plan, adapter)
    assert loop["trip_count"] == 2 and bound == {}


def test_deployment_loop_discovery_requires_explicit_provider_capability(monkeypatch):
    from merlin.compiler.build.regions import _normal_loop
    from merlin.compiler.build.staging import LoopBound
    from merlin.compiler.partition.semantic_regions import outline_semantic_regions

    source, plan = _source(2)
    adapter = Provider(plan)
    outlined = outline_semantic_regions(source, entry="model", plan=plan)
    monkeypatch.setattr("merlin.compiler.build.deployment.selected", lambda: True)
    assert _normal_loop(outlined, plan, adapter) == (None, {})
    adapter.supports_deployment_loop_sharing = True
    loop, bound = _normal_loop(outlined, plan, adapter)
    assert loop["trip_count"] == 2 and tuple(bound) == (1,) and isinstance(bound[1], LoopBound)


# ---- A loop whose staging is found after binding publishes exactly what an unbound compile publishes ----

WIDE = (1, 4, 8, 4)  # fits the window, so only the provider can refuse it whole


def _staged(root, *, bind, trigger, tail_first=False, trips=3, **options):
    """Compile on-chip; `trigger` is "region" (a later region refuses) or "render" (the flat arena overflows)."""
    source, plan = _source(trips, wide=WIDE if trigger == "region" else None, tail_first=tail_first,
                           body=options.pop("body", (1,)))
    adapter = OnchipProvider(plan, whole_limit=WIDE if trigger == "region" else None, **options)
    if not bind:
        adapter.bind_loop_region = None
    destination = root / "application"
    with _onchip():
        manifest = compile_semantic_application(
            source, adapter=adapter, snapshot=_snapshot(root / "snapshot"), target_source=root,
            destination=destination, limits=None, name="model", capacity=8 if trigger == "render" else 4096)
        checked = verify_semantic_application(source, adapter=adapter, snapshot=_snapshot(root / "snapshot"),
                                              target_source=root, artifact=destination)
    tree = {path.relative_to(destination).as_posix(): path.read_bytes()
            for path in sorted(destination.rglob("*")) if path.is_file()}
    return adapter, manifest, checked, tree


@pytest.mark.parametrize("trips", [2, 3])
@pytest.mark.parametrize("trigger", ["region", "render", "first_region", "bound_body", "bind_raises"])
def test_staging_found_after_binding_publishes_the_unbound_application(tmp_path, trigger, trips):
    """`bound_body`: a bound body also refuses when it compiles itself, so the first refusal names the reason.

    `bind_raises`: the provider refuses the binding itself for resource_limit, so that region compiles itself.
    """
    case = dict(trigger="render" if trigger in ("render", "bind_raises") else "region",
                tail_first=trigger == "first_region", trips=trips)
    if trigger == "bound_body":
        case.update(refuse_label=f"region_{trips - 1:03d}", body=WIDE)
    if trigger == "bind_raises":
        case.update(bind_raises="resource_limit")
    bound, manifest, checked, tree = _staged(tmp_path / "bound", bind=True, **case)
    unbound, expected_manifest, expected_checked, expected = _staged(tmp_path / "unbound", bind=False, **case)
    # A region refused before the loop stops binding: those exports would be discarded.
    assert (bound.binds == 0) if trigger == "first_region" else (bound.binds >= 1)
    assert unbound.binds == 0
    assert manifest["staged_deployment"] is True and "dispatch_loop" not in manifest
    assert all("compiled_body_index" not in row for row in manifest["regions"])
    assert checked["status"] == expected_checked["status"] == "checked"
    assert manifest == expected_manifest
    if trigger == "bound_body":
        assert json.loads(tree["deployment.json"])["staging_reason"].startswith("the body's own compile")
    assert sorted(tree) == sorted(expected)
    assert {name for name in tree if tree[name] != expected[name]} == set()


def test_a_provider_that_writes_while_binding_is_refused_when_staging_is_found(tmp_path):
    with pytest.raises(ValueError, match="while binding a shared body"):
        _staged(tmp_path, bind=True, trigger="render", writes_on_bind=True)


def test_only_an_onchip_resource_limit_binding_refusal_is_a_decline(tmp_path):
    with pytest.raises(NativeCompilationError) as onchip:
        _staged(tmp_path / "onchip", bind=True, trigger="render", bind_raises="unsupported_semantics")
    assert onchip.value.status == "unsupported_semantics"
    with pytest.raises(NativeCompilationError) as offchip:
        compile_fixture(tmp_path / "offchip", 2, bind_raises="resource_limit")
    assert offchip.value.status == "resource_limit"


def test_staging_known_before_compilation_keeps_its_reason(tmp_path, monkeypatch):
    """A reason found before region compilation is published even when a region refuses later."""
    monkeypatch.setattr("merlin.compiler.build.region_fit.staging_reason", lambda *args: "pre-compile bound")
    for bind in (False, True):
        adapter, manifest, checked, tree = _staged(tmp_path / str(bind), bind=bind, trigger="region", trips=2)
        assert adapter.binds == 0 and manifest["staged_deployment"] is True and checked["status"] == "checked"
        assert json.loads(tree["deployment.json"])["staging_reason"] == "pre-compile bound"
