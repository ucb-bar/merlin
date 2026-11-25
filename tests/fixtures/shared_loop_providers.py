"""Shared-loop and on-chip deployment providers: host bodies count their context, and the accelerator
region publishes a one-word synthetic program, not device code."""

import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

from merlin.compiler.backend.protocol import LoopBodyBinding, PlatformDeployment, ResidentParameters
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


class FlatProvider:
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



class Provider(FlatProvider, LoopBodyBinding):
    """The same provider with a counted-loop body binding."""

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


class _OnchipBinding(PlatformDeployment):
    """The platform C binding and deployment record of the fixture's on-chip map, which it requires."""

    def deployment_required(self):
        return True

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


class OnchipProvider(_OnchipBinding, Provider):
    """The same provider for an on-chip deployment that declares loop sharing."""

    supports_deployment_loop_sharing = True


class OnchipFlatProvider(_OnchipBinding, FlatProvider):
    """An on-chip provider without a loop body binding."""


def _snapshot(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text("{}\n")
    return SimpleNamespace(root=root)


def _onchip(**options):
    from merlin.compiler.semantic.codegen import CodegenOptions, codegen_options

    return codegen_options(CodegenOptions(io_memory_base=WINDOW_BASE, io_memory_bytes=WINDOW_BYTES,
                                          io_memory_capacity_base=WINDOW_BASE, io_memory_capacity_bytes=WINDOW_BYTES,
                                          **options))


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


class DeployedProvider(OnchipProvider):
    """The on-chip provider for a whole application of several regions, without sharing."""

    supports_deployment_loop_sharing = False

    def deployment_requirements(self, regions, native_root):
        rows = [OnchipProvider.deployment_requirements(self, [region], native_root) for region in regions]
        return {**rows[0], "images": [image for row in rows for image in row["images"]],
                "reservations": [dict(row, lifetime=dict(scope="region_call", context_index=index))
                                 for index, item in enumerate(rows) for row in item["reservations"]],
                "regions": [dict(row, context_index=index) for index, item in enumerate(rows) for row in item["regions"]]}


class AdmittingProvider(DeployedProvider, ResidentParameters):
    """A deployed provider whose resident admission is the callback ``admit``; it packs nothing itself."""

    admit = None

    def admit_resident_deployment(self, *args, **kwargs):
        return self.admit(*args, **kwargs)

    def resident_memory_geometry(self, *args):
        raise AssertionError("this fixture keeps nothing resident")

    inventory_resident_inputs = pack_resident_input = resident_memory_geometry


def _deployed(root, adapter=None, **options):
    """Compile and replay the two-trip fixture as an on-chip deployment; return its manifest and published tree."""
    source, plan = _source(2)
    adapter = adapter or DeployedProvider(plan)
    destination = root / "application"
    with _onchip(**options):
        manifest = compile_semantic_application(
            source, adapter=adapter, snapshot=_snapshot(root / "snapshot"), target_source=root,
            destination=destination, limits=None, name="model", capacity=4096)
        checked = verify_semantic_application(source, adapter=adapter, snapshot=_snapshot(root / "snapshot"),
                                              target_source=root, artifact=destination)
    assert checked["status"] == "checked"
    return manifest, {path.relative_to(destination).as_posix(): path.read_bytes()
                      for path in sorted(destination.rglob("*")) if path.is_file()}


class UntypedBindingProvider(FlatProvider):
    """Every loop hook by name, without declaring LoopBodyBinding, so Merlin calls none of them."""

    region_loop_parameters = ()
    supports_deployment_loop_sharing = True
    bind_loop_region = Provider.bind_loop_region
    finalize_loop_regions = LoopBodyBinding.finalize_loop_regions
