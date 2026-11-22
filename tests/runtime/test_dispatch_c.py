import json
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.compiler.lower.c.dispatch import DispatchCall, build_program, emit_program_c
from merlin.compiler.partition.dispatch_program import Buffer, DispatchProgram, Node
from merlin.compiler.partition.global_plan_emission import dispatch_digest
from merlin.compiler.schedule.storage.arena import ArenaRequirements


def program():
    graph = DispatchProgram(
        "forward",
        [0],
        {
            key: Buffer(key, [4], "i8", "arg" if key == "x" else "intermediate", arg_index=0 if key == "x" else None)
            for key in ("x", "a", "y", "c", "z")
        },
        [
            Node("dispatch", op, ins, [out])
            for op, ins, out in (
                ("cpu", ["x"], "a"),
                ("device", ["a"], "y"),
                ("cpu", ["x"], "c"),
                ("device", ["c"], "z"),
            )
        ],
        ["y", "z"],
    )
    return build_program(
        graph,
        memory_requirements=ArenaRequirements(
            8, (0, 1, 2, 3), (0, 1, 2, 3), alignment=1, reserved=((0, 4),), dispatch_digest=dispatch_digest(graph)
        ),
    )


def calls():
    return {
        i: DispatchCall("add" if i % 2 == 0 else "double_value", "cpu" if i % 2 == 0 else "device") for i in range(4)
    }


def test_compiled_dispatch_consumes_offsets_and_preserves_guards(tmp_path):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    graph = program()
    report, files = emit_program_c(graph, "model", calls())
    assert graph.memory_plan.offsets == {"a": 4, "c": 4}
    assert report["outputs"] == ["y", "z"]
    assert all(("buffer_alignment" not in call for call in report["calls"]))
    assert all(("alignment" not in buffer for buffer in report["buffers"]))
    assert "arena_alignment" not in report
    json.dumps(report)
    for member, text in files.items():
        (tmp_path / member).write_text(text)
    (tmp_path / "check.c").write_text(
        '\n#include "model.h"\n#include <assert.h>\n#include <string.h>\nstatic unsigned char *expected;\nstatic int failure, adds, doubles;\nint add(void *ctx, const void *const *in, void *const *out) {\n  (void)ctx; ++adds; assert(out[0] == expected);\n  for (unsigned i = 0; i < 4; ++i) ((unsigned char *)out[0])[i] = ((const unsigned char *)in[0])[i] + 1;\n  return 0;\n}\nint double_value(void *ctx, const void *const *in, void *const *out) {\n  (void)ctx; ++doubles; assert(in[0] == expected);\n  if (failure) return 37;\n  for (unsigned i = 0; i < 4; ++i) ((unsigned char *)out[0])[i] = 2 * ((const unsigned char *)in[0])[i];\n  return 0;\n}\nint main(void) {\n  unsigned char arena[10], input[4], y[6], z[6];\n  model_buffer buffers[] = {{input,4}, {y+1,4}, {z+1,4}};\n  model_state state = {0}; expected = arena + 5;\n  memset(arena, 0xcd, sizeof(arena));\n  for (unsigned trial = 0; trial < 2; ++trial) {\n    memset(input, trial, 4); memset(y,0xee,6); memset(z,0xee,6);\n    assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_OK);\n    assert(!state.active && state.completed == 4 && !state.kernel_status);\n    for (unsigned i = 0; i < 4; ++i) assert(input[i] == trial && y[i+1] == 2*(trial+1) && z[i+1] == y[i+1]);\n    assert(y[0] == 0xee && y[5] == 0xee && z[0] == 0xee && z[5] == 0xee);\n    for (unsigned i = 0; i < 5; ++i) assert(arena[i] == 0xcd);\n    assert(arena[9] == 0xcd);\n  }\n  assert(model_run(&state, arena+1, 7, buffers, 3, 0) == MODEL_INVALID);\n  buffers[2].data = y+1;\n  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_INVALID);\n  buffers[2].data = z+1;\n  model_state before; memcpy(&before, &state, sizeof(state));\n  buffers[0].data = &state.active;\n  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_INVALID);\n  assert(memcmp(&state, &before, sizeof(state)) == 0 && adds == 4);\n  buffers[0].data = input;\n  buffers[2].data = &state.completed;\n  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_INVALID);\n  assert(memcmp(&state, &before, sizeof(state)) == 0 && doubles == 4);\n  buffers[2].data = z+1;\n  union { model_state state; unsigned char arena[sizeof(model_state) + 8]; } shared = {0};\n  assert(model_run(&shared.state, shared.arena, 8, buffers, 3, 0) == MODEL_INVALID);\n  assert(!shared.state.active && adds == 4);\n  failure = 1;\n  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_KERNEL_ERROR);\n  assert(state.active && state.completed == 1 && state.kernel_status == 37);\n  assert(model_run(&state, arena+1, 8, buffers, 3, 0) == MODEL_BUSY);\n  assert(adds == 5 && doubles == 5);\n  return 0;\n}\n'
    )
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "model.c", "check.c", "-o", "check"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)


def test_view_alias_requires_an_explicit_safe_binding():
    graph = program()
    graph.dispatch.nodes[1] = Node("view", "slice", ["a"], ["c"])
    graph.dispatch.nodes = graph.dispatch.nodes[:2] + [Node("dispatch", "copy", ["c"], ["y"])]
    graph.dispatch.results = ["y"]
    del graph.dispatch.buffers["z"]
    req = ArenaRequirements(
        8, (0, 1, 2), (0, 1, 2), alignment=1, aliases=(("c", "a", 0),), dispatch_digest=dispatch_digest(graph.dispatch)
    )
    graph = build_program(graph.dispatch, memory_requirements=req)
    bindings = {i: DispatchCall("compiled", "cpu") for i in range(3)}
    with pytest.raises(ValueError, match="alias-safe"):
        emit_program_c(graph, "model", bindings)
    bindings[1] = replace(bindings[1], alias_safe=True)
    emit_program_c(graph, "model", bindings)


# ---- Staged deployment: row, tile, and quantizer-group rules, whole-refusal replay, and direct calls ----

STAGE_BASE, STAGE_WINDOW = 0x10000, 4096


def _tensor(shape, dtype="f32"):
    from merlin.compiler.semantic.model import TensorType

    return TensorType(shape, dtype, "raw-bit-pattern")


def _onchip():
    from merlin.compiler.semantic.codegen import CodegenOptions, codegen_options

    class Onchip(CodegenOptions):
        application_deployment = "onchip"

    return codegen_options(Onchip(io_memory_base=STAGE_BASE, io_memory_bytes=STAGE_WINDOW,
                                  io_memory_capacity_base=STAGE_BASE, io_memory_capacity_bytes=STAGE_WINDOW))


def _outlined(nodes, outputs, groups=None, placement="accelerator"):
    from merlin.compiler.partition.global_plan import (BufferRepresentation, CycleInterval, GlobalPlan,
                                                       RegionAlternative, ValueRepresentation)
    from merlin.compiler.partition.semantic_regions import extract_semantic_region, outline_semantic_regions
    from merlin.compiler.semantic.model import KernelRequest

    def plan_for(source, groups):
        pure = [node for node in source.nodes if node.effect == "pure"]
        dense = ValueRepresentation("memory", "dense", pure[0].type.dtype)
        rows = []
        for index, indices in enumerate(groups):
            region = extract_semantic_region(source, tuple(pure[i].id for i in indices))
            ports = tuple(node.id for node in region.nodes if node.effect in {"input", "constant"})
            buffers = [tuple(BufferRepresentation(key, replace(dense, dtype=source.node(key).type.dtype)) for key in keys)
                       for keys in (ports, region.outputs)]
            rows.append(RegionAlternative(f"region_{index}", tuple(indices), f"kernel_{index}", "device",
                                          CycleInterval.unknown("uncalibrated"), *buffers))
        return GlobalPlan(tuple(rows), (), CycleInterval.unknown("uncalibrated"))

    inputs = tuple((node.id, "external") for node in nodes if node.effect == "input")
    source = KernelRequest(nodes, outputs, ("external",) * len(outputs), inputs, "synthetic")
    pure = sum(node.effect == "pure" for node in nodes)
    plan = plan_for(source, groups or (tuple(range(pure)),))
    plan = replace(plan, selected=(replace(plan.selected[0], placement=placement), *(
        replace(row, placement="host") for row in plan.selected[1:])))
    return source, plan, outline_semantic_regions(source, entry="application", plan=plan)


class _Provider:
    """One-word programs, a 64-byte frame reservation, and one launch per device step."""

    def application_binding(self, name, regions, interface):
        from merlin.compiler.build.regions import ApplicationCBinding

        return ApplicationCBinding("", "", {"invoke": name + "_run", "completion": "synchronous"})

    def deployment_requirements(self, regions, native_root):
        from hashlib import sha256

        (region,) = regions
        host = region["placement"] == "host"
        images = [dict(path=program + "/program.bin", sha256=sha256(b"\x73\0\0\0").hexdigest(), bytes=4,
                       imem_offset_bytes=0, capacity_bytes=1024, upload_convention="counted_ensure_loaded")
                  for program in region["native_programs"]]
        return dict(
            schema="merlin.provider_deployment.v1", window=dict(address=STAGE_BASE, bytes=STAGE_WINDOW), images=images,
            window_section=".device_window", image_section=".device_images",
            reservations=[] if host else [dict(name="frame", address=STAGE_BASE, bytes=64, role="input",
                                               lifetime=dict(scope="region_call", context_index=0),
                                               promotion_receipts=[])],
            regions=[dict(context_index=0, placement=region["placement"], launches_per_invocation=0 if host else 1)],
            allocation_alignment=64, host_context_reservation_bytes=64, max_polls=1,
            host_map=dict(imem_address=0, imem_bytes=1024, csr_address=0),
            runtime=dict(host_type="h", io_type="io", start="s", wait="w", image_completion="c", reload_policy="r"),
            host_allocation=dict(address=0, bytes=1024, reserved_bytes=64, measurement="declared_reservations",
                                 declared_reservations=[dict(name="context", bytes=64)]),
            raw_scratch_bytes=1, conversion_scratch_bytes=1)


def _publish(segments, request, label):
    from hashlib import sha256

    (segments / label).mkdir(parents=True, exist_ok=True)
    (segments / label / "program.bin").write_bytes(b"\x73\0\0\0")
    (segments / label / "manifest.json").write_text(json.dumps(dict(
        engine="merlin_native", request_digest=request.digest(), binary_sha256=sha256(b"\x73\0\0\0").hexdigest(),
        target_identity=request.target_identity)))
    return (f"segments/{label}",)


def _package(region, programs=()):
    from merlin.compiler.build.regions import CompiledCRegion

    inputs = tuple(node.id for node in region.nodes if node.effect in {"input", "constant"})
    return CompiledCRegion("call", "call.h", inputs, region.outputs, {"call.c": b"int call;\n", "call.h": b"#pragma once\n"},
                           (), programs, {})


def test_staging_splits_a_concatenation_only_by_rows_before_its_joined_axis():
    from merlin.compiler.build.staging import leading_rows, row_slice
    from merlin.compiler.semantic.model import SemanticNode

    def region(axis):
        half, joined = (1, 4, 32, 8), tuple(e * 2 if i == axis else e for i, e in enumerate((1, 4, 32, 8)))
        nodes = (SemanticNode("a", "input", (), _tensor(half), effect="input"),
                 SemanticNode("b", "input", (), _tensor(half), effect="input"),
                 SemanticNode("joined", "concat", ("a", "b"), _tensor(joined), (("axis", axis),)))
        return _outlined(nodes, ("joined",))[2].regions[0][1]

    rows = region(3)
    assert leading_rows(rows, 4) == dict(joined="split", a="split", b="split")
    assert row_slice(rows, 4).node("joined").type.shape == (1, 1, 32, 16)
    # Rows of the joined axis itself come from different operands.
    assert all(leading_rows(region(1), count) is None for count in (2, 4, 8, 16, 64))


@pytest.mark.parametrize("reduce, size, admitted, refused", [
    ("block", 3, (), 4),  # four 32-row chunks would cut three-row groups
    ("block", 4, (2, 4, 8, 32), 64),
    ("head_block", 3, (2, 4), 8),  # eight 16-row chunks would cut each head's groups at row 16
    ("token", 32, (4, 128), 256),
])
def test_staging_splits_a_runtime_quantizer_only_at_declared_group_bounds(reduce, size, admitted, refused):
    from merlin.compiler.build.staging import _quantizer_groups, leading_row_counts, leading_rows
    from merlin.compiler.semantic.model import SemanticNode

    shape = (1, 4, 32, 16)
    attrs = (("mode", "dynamic"), ("reduce", reduce), ("rows", size), ("semantic_kind", "dynamic_activation_quantizer"))
    nodes = (SemanticNode("x", "input", (), _tensor(shape, "bf16"), effect="input"),
             SemanticNode("codes", "atlas.quantize_act.codes", ("x",), _tensor(shape, "i8"), attrs),
             SemanticNode("exponent", "atlas.quantize_act.exponent", ("x",), _tensor((*shape[:-1], 1), "i32"), attrs))
    region = _outlined(nodes, ("codes", "exponent"))[2].regions[0][1]
    groups = _quantizer_groups(shape, dict(attrs))
    assert sum(groups) == 128 and (len(groups) > 1) == (reduce != "tensor")
    for count in admitted:
        assert leading_rows(region, count) == dict(x="split", codes="split", exponent="split")
    assert leading_rows(region, refused) is None
    assert set(leading_row_counts(region, 1 << 20)) >= set(admitted)


def test_staging_tiles_a_contraction_when_no_single_axis_option_places(tmp_path):
    import numpy as np

    from merlin.compiler.build.staging import band_rules, split_rule, stage
    from merlin.compiler.semantic.model import IndexMap, SemanticNode

    m = k = n = 64
    identity = IndexMap(2, ((1, 0), (0, 1)), (0, 0))
    nodes = (
        SemanticNode("x", "input", (), _tensor((m, k), "i8"), effect="input"),
        SemanticNode("xe", "input", (), _tensor((), "i32"), effect="input"),
        SemanticNode("w", "input", (), _tensor((n, k), "i8"), effect="input"),
        SemanticNode("we", "input", (), _tensor((n, 1), "i32"), effect="input"),
        SemanticNode("y", "synthetic.qlinear", ("x", "xe", "w", "we"), _tensor((m, n), "bf16"),
                     (("rhs_transposed", "true"), ("semantic_kind", "prequantized_contraction"))),
        SemanticNode("z", "synthetic.copy", ("y",), _tensor((m, n), "bf16"), index_maps=(identity, identity)),
    )
    source, plan, outlined = _outlined(nodes, ("z",), groups=((0,), (1,)))
    region = outlined.regions[0][1]
    assert split_rule(region)[0] == "output_columns" and [rule.name for rule in band_rules(region)] == ["rows"]
    segments = tmp_path / "segments"
    compiled = [_package(region, _publish(segments, region, "region_000")), _package(outlined.regions[1][1])]

    def compile_slice(index, key, region, request, label, root):
        return _package(request, _publish(root / "segments", request, label))

    with _onchip():
        steps, regions, packages, *_ = stage(source, _Provider(), outlined, plan, compiled, capacity=STAGE_WINDOW,
                                             native_root=tmp_path, compile_slice=compile_slice)
    split = regions[0]
    # The whole activation and the whole weight each fill the window, so every column slice and every row band
    # refuses; a tile reads only its weight rows and its activation rows.
    assert split["staging"] == "tiles" and split["slices"] == n // split["columns"] * (m // split["rows"]) > 1
    assert all(row["window_required_bytes"] <= STAGE_WINDOW for row in steps)
    rng = np.random.default_rng(11)
    x, w = rng.integers(1, 120, (m, k), dtype=np.int8), rng.integers(1, 120, (n, k), dtype=np.int8)
    reference = (x.astype(np.int64) @ w.astype(np.int64).T).astype("<u2")

    def gather(store, row, key):
        items = [load["source"] for load in row["loads"]] + [read["destination"] for read in row["reads"]]
        return b"".join(store[item["offset"] + j * item["stride"]:][:item["bytes"]]
                        for item in items if item["value"] == key for j in range(item["count"]))

    tiles, output = steps[:split["slices"]], bytearray(reference.nbytes)
    for row in tiles:
        box = (np.frombuffer(gather(x.tobytes(), row, "x"), np.int8).reshape(-1, k).astype(np.int64)
               @ np.frombuffer(gather(w.tobytes(), row, "w"), np.int8).reshape(-1, k).astype(np.int64).T)
        box = box.astype("<u2").tobytes()
        (read,) = row["reads"]
        target, width = read["destination"], read["destination"]["bytes"]
        for j in range(target["count"]):
            start = target["offset"] + j * target["stride"]
            output[start:start + width] = box[j * width:(j + 1) * width]
    assert bytes(output) == reference.tobytes() and any(reference.ravel())
    assert sum(read["destination"]["bytes"] * read["destination"]["count"] for row in tiles
               for read in row["reads"]) == reference.nbytes


def test_replay_progress_is_silent_within_its_interval():
    import io

    from merlin.compiler.build.regions import _ReplayProgress

    stream = io.StringIO()
    progress = _ReplayProgress(7, interval=60.0, clock=iter([0.0, 59.0, 60.0, 61.0, 125.0]).__next__, stream=stream)
    for _ in range(4):
        progress()
    assert stream.getvalue().splitlines() == [
        "merlin verify: 2 region exports replayed over 7 source regions",
        "merlin verify: 4 region exports replayed over 7 source regions",
    ]


def test_whole_refused_region_replays_from_its_published_slices_and_calls_matching_ports_directly(
        tmp_path, monkeypatch, capfd):
    """The whole compile overflows; verification replays the slices only, reports progress, and calls directly."""
    from types import SimpleNamespace

    import merlin.compiler.build.regions as regions
    from merlin.compiler.semantic.model import IndexMap, SemanticNode
    from merlin.compiler.semantic.search.exploration import SearchLimits
    from merlin.compiler.semantic.target_binding import NativeCompilationError

    full = (1, 4, 32, 16)
    maps = tuple(IndexMap(4, tuple(tuple(int(i == a) for i in range(4)) for a in range(4)), (0,) * 4) for _ in "xyz")
    nodes = (SemanticNode("x", "input", (), _tensor(full), effect="input"),
             SemanticNode("y", "input", (), _tensor(full), effect="input"),
             SemanticNode("sum", "f32_add", ("x", "y"), _tensor(full),
                          (("numeric_semantics", "ieee754-f32-add-rne-v1"),), index_maps=maps))
    source, plan, _ = _outlined(nodes, ("sum",))
    exported = []

    class FrontDoor(_Provider):
        def plan(self, request):
            return plan

        def export_region(self, region, alternative, label, segments, target_source):
            exported.append(label)  # verification replays only through export
            return self._package(region, label, segments)

        def _package(self, region, label, segments):
            from merlin.compiler.build.regions import CompiledCRegion

            json.loads((segments / label / "manifest.json").read_bytes())  # reads its compiled publication
            inputs = tuple(node.id for node in region.nodes if node.effect == "input")
            files = {label + ".c": b"int x;\n", label + ".h": b"#pragma once\n"}
            return CompiledCRegion(label, label + ".h", inputs, region.outputs, files, (), (f"segments/{label}",), {})

        def compile_region(self, snapshot, region, alternative, label, segments, target_source, limits):
            if region.node("x").type.shape == full:
                raise NativeCompilationError("resource_limit", "complete row loop exceeds the selected I/O window")
            _publish(segments, region, label)
            return self._package(region, label, segments)

    monkeypatch.setattr(regions, "REPLAY_PROGRESS_SECONDS", 0.0)
    (tmp_path / "snapshot").mkdir()
    (tmp_path / "snapshot/manifest.json").write_text("{}")
    with _onchip():
        manifest = regions.compile_semantic_application(
            source, adapter=FrontDoor(), snapshot=SimpleNamespace(root=tmp_path / "snapshot"), target_source=tmp_path,
            destination=tmp_path / "front", limits=SearchLimits(), name="rows", capacity=STAGE_WINDOW)
    record = json.loads((tmp_path / "front/deployment.json").read_bytes())
    (split,) = record["regions"]
    assert manifest["staged_deployment"] is True and "slices" in split
    assert "complete row loop exceeds" in split["unsplit_refusal"]
    assert sorted(row["package"] for row in manifest["regions"]) == [split["shared_package"]]
    # The whole region was never exported in verification: only the published slice package was.
    assert set(exported) == {split["shared_package"]}
    lines = [line for line in capfd.readouterr().err.splitlines() if line.startswith("merlin verify:")]
    assert lines and lines[-1].endswith("region exports replayed over 1 source regions")
    # Ports in dispatch order: the step application calls its region symbol, not a reordering wrapper.
    (calls,) = (tmp_path / "front/steps").rglob("*_calls.c")
    (driver,) = (tmp_path / "front/steps").rglob("*_driver.c")
    package = split["shared_package"]
    assert "_call_0(" not in calls.read_text() and f"= {package}(" in driver.read_text()


# ---- On-chip view placement: the counted solver, then first fit once its budget is spent ----

def _view_application(tmp_path, capacity, deployment="onchip", measured_host=None, *, affine_view=False):
    """Five device regions and one dense view; first fit places this arena in 1024 bytes and the solver in 704."""
    from types import SimpleNamespace

    import merlin.compiler.build.regions as regions
    from merlin.compiler.partition.global_plan import (BufferRepresentation, CycleInterval, GlobalPlan,
                                                       RegionAlternative, ValueRepresentation)
    from merlin.compiler.partition.semantic_regions import extract_semantic_region
    from merlin.compiler.semantic.codegen import CodegenOptions, codegen_options
    from merlin.compiler.semantic.model import IndexMap, KernelRequest, SemanticNode, pointwise_identity_maps
    from merlin.compiler.semantic.search.exploration import SearchLimits

    rows = (("t0", ("x",), (96,)), ("t1", ("x", "t0"), (64,)), ("v", ("t1",), (8, 8)), ("t2", ("x", "t0", "t1"), (16,)),
            ("t3", ("x", "t1", "t2"), (80,)), ("y", ("t3", "v"), (16,)))
    nodes = (SemanticNode("x", "input", (), _tensor((4,)), effect="input"), *(
        SemanticNode(key, "reshape" if key == "v" else "synthetic.mix", inputs, _tensor(shape))
        for key, inputs, shape in rows))
    if affine_view:
        nodes = tuple(replace(node, op="gather", index_maps=(
            IndexMap(2, ((8, 1),), (0,)), pointwise_identity_maps(2, 1)[0]))
            if node.id == "v" else node for node in nodes)
    source = KernelRequest(nodes, ("y",), ("external",), (("x", "external"),), "synthetic")
    dense = ValueRepresentation("memory", "dense", "f32")
    alternatives = []
    for index, (key, inputs, _) in enumerate(rows):
        region = extract_semantic_region(source, (key,))
        ports = tuple(node.id for node in region.nodes if node.effect == "input")
        alternatives.append(RegionAlternative(
            f"region_{index}", (index,), f"kernel_{index}", "device", CycleInterval.unknown("uncalibrated"),
            tuple(BufferRepresentation(port, dense) for port in ports), (BufferRepresentation(key, dense),)))
    plan = GlobalPlan(tuple(alternatives), (), CycleInterval.unknown("uncalibrated"))

    class Regions(_Provider):
        def plan(self, request):
            return plan

        def export_region(self, region, alternative, label, segments, target_source):
            inputs = tuple(node.id for node in region.nodes if node.effect == "input")
            files = {label + ".c": b"int x;\n", label + ".h": b"#pragma once\n"}
            return regions.CompiledCRegion(label, label + ".h", inputs, region.outputs, files, (),
                                           (f"segments/{label}",), {})

        def compile_region(self, snapshot, region, alternative, label, segments, target_source, limits):
            _publish(segments, region, label)
            return self.export_region(region, alternative, label, segments, target_source)

        def deployment_requirements(self, rows, native_root):
            record = super().deployment_requirements(rows[:1], native_root)
            record["images"] = [image for row in rows for image in
                                super(Regions, self).deployment_requirements([row], native_root)["images"]]
            record["reservations"] = [dict(record["reservations"][0], name=f"frame_{row['context_index']}",
                                           lifetime=dict(scope="region_call", context_index=row["context_index"]))
                                      for row in rows if row["placement"] != "host"]
            record["regions"] = [dict(context_index=row["context_index"], placement=row["placement"],
                                      launches_per_invocation=0 if row["placement"] == "host" else 1) for row in rows]
            return record

    class Options(CodegenOptions):
        application_deployment = deployment

    (tmp_path / "snapshot").mkdir(exist_ok=True)
    (tmp_path / "snapshot/manifest.json").write_text("{}")
    out = tmp_path / f"application-{capacity}-{deployment}-{measured_host}"
    with codegen_options(Options(io_memory_base=STAGE_BASE, io_memory_bytes=STAGE_WINDOW,
                                 io_memory_capacity_base=STAGE_BASE, io_memory_capacity_bytes=STAGE_WINDOW,
                                 application_buffer_views=True, measured_host_bytes=measured_host)):
        manifest = regions.compile_semantic_application(
            source, adapter=Regions(), snapshot=SimpleNamespace(root=tmp_path / "snapshot"), target_source=tmp_path,
            destination=out, limits=SearchLimits(), name="views", capacity=capacity)
    return manifest, out


def _memory(out):
    return json.loads((out / "program.json").read_bytes())["memory_plan"]


def test_view_arena_takes_the_solver_placement_while_its_budget_lasts(tmp_path):
    for capacity in (2048, 1024):
        for deployment in ("onchip", None):
            manifest, out = _view_application(tmp_path, capacity, deployment, affine_view=capacity == 1024)
            assert not manifest.get("staged_deployment") and _memory(out)["arena_bytes"] == 704
            assert _memory(out)["offsets"] == {"t0": 0, "t1": 384, "t2": 640, "t3": 0, "v": 384}
    assert json.loads((tmp_path / "application-1024-onchip-None/deployment.json").read_bytes())["schema"] == (
        "merlin.onchip_deployment.v1")


def test_view_arena_that_no_placement_fits_still_stages_on_chip_and_refuses_otherwise(tmp_path):
    from merlin.compiler.semantic.target_binding import NativeCompilationError

    manifest, out = _view_application(tmp_path, 704)
    assert manifest["staged_deployment"] is True
    assert "output staging exceed" in json.loads((out / "deployment.json").read_bytes())["staging_reason"]
    with pytest.raises(NativeCompilationError, match="output staging exceed"):
        _view_application(tmp_path, 704, deployment=None)


def test_view_arena_places_by_first_fit_once_the_solver_work_is_spent(tmp_path, monkeypatch):
    from merlin.compiler.schedule.storage import arena
    from merlin.compiler.semantic.target_binding import NativeCompilationError

    monkeypatch.setattr(arena, "ARENA_SOLVER_WORK", 1)
    # First fit's 1024-byte placement stands where the output still fits after it.
    manifest, out = _view_application(tmp_path, 2048)
    assert not manifest.get("staged_deployment") and _memory(out)["arena_bytes"] == 1024
    # Where it does not, on chip stages and a compile without a deployment refuses.
    manifest, out = _view_application(tmp_path, 1024)
    reason = json.loads((out / "deployment.json").read_bytes())["staging_reason"]
    assert manifest["staged_deployment"] is True and "output staging exceed" in reason
    with pytest.raises(NativeCompilationError, match="output staging exceed"):
        _view_application(tmp_path, 1024, deployment=None)


def test_measured_application_host_over_the_provider_host_stages_on_chip(tmp_path):
    # The test provider backs 1024 host bytes; a host measured at that size stays on chip.
    manifest, _ = _view_application(tmp_path, 2048, measured_host=1024)
    assert not manifest.get("staged_deployment")
    manifest, out = _view_application(tmp_path, 2048, measured_host=1025)
    reason = json.loads((out / "deployment.json").read_bytes())["staging_reason"]
    assert manifest["staged_deployment"] is True
    assert reason == "deployment_host_overflow: measured application host 1025 bytes exceeds 1024 backed host bytes"
    # Without an on-chip deployment there is no provider host to bound.
    manifest, _ = _view_application(tmp_path, 2048, deployment=None, measured_host=1025)
    assert not manifest.get("staged_deployment")


def _view_arena(first_fit=False):
    graph = DispatchProgram("forward", [0], {
        name: Buffer(name, [4], "i8", "arg" if name == "x" else "intermediate", arg_index=0 if name == "x" else None)
        for name in ("x", "a", "v", "b", "y")}, [
            Node("dispatch", "first", ["x"], ["a"]), Node("view", "reshape", ["a"], ["v"]),
            Node("dispatch", "second", ["x"], ["b"]), Node("dispatch", "last", ["v", "b"], ["y"])], ["y"])
    return graph, ArenaRequirements(64, (0, 2, 4, 6), (1, 3, 5, 7), alignment=4, aliases=(("v", "a", 0),),
                                    dispatch_digest=dispatch_digest(graph), first_fit=first_fit)


def test_view_arena_uses_first_fit_only_on_request_and_when_the_solver_cannot_decide():
    from unittest.mock import patch

    from merlin.compiler.partition.memory_plan import ArenaPlanError
    from merlin.compiler.schedule.storage.arena import plan_bounded_arena
    from merlin.compiler.semantic.search.buffers import BufferAllocation

    solved = plan_bounded_arena(*_view_arena())
    assert plan_bounded_arena(*_view_arena(first_fit=True)).offsets == solved.offsets
    assert solved.offsets["v"] == solved.offsets["a"]
    undecided = BufferAllocation("unknown", reason="canceled", work=1)
    with patch("merlin.compiler.semantic.search.buffers.allocate_buffers", return_value=undecided):
        first, second = (plan_bounded_arena(*_view_arena(first_fit=True)) for _ in range(2))
        with pytest.raises(ArenaPlanError, match="bounded arena allocation unknown: canceled"):
            plan_bounded_arena(*_view_arena())
    assert first.offsets == second.offsets
    assert first.offsets["v"] == first.offsets["a"] != first.offsets["b"]


def test_arena_solver_work_is_one_counted_budget_over_queries_and_plans():
    from merlin.compiler.partition.memory_plan import ArenaPlanError
    from merlin.compiler.schedule.storage.arena import _bounded_arena_inputs, arena_solver_work, plan_bounded_arena
    from merlin.compiler.semantic.search.buffers import allocate_buffers

    graph, requirements = _view_arena()
    _, requests, banks, _ = _bounded_arena_inputs(graph, requirements)
    full = allocate_buffers(requests, banks, timeout_ms=None, rlimit=10**9)
    assert full.status == "sat" and full.work > 0
    assert allocate_buffers(requests, banks, timeout_ms=None, rlimit=10**9) == full
    # The limit covers every minimization query of one call, not each query separately.
    partial = allocate_buffers(requests, banks, timeout_ms=None, rlimit=full.work // 2)
    assert partial.status == "unknown" and partial.work <= full.work
    # One compile's budget is shared: a second identical solve finds it spent and refuses without solving.
    with arena_solver_work(full.work) as budget:
        assert dict(plan_bounded_arena(graph, requirements).offsets) == dict(full.addresses)
        assert budget == [0]
        with pytest.raises(ArenaPlanError, match="solver work budget exhausted"):
            plan_bounded_arena(graph, requirements)
