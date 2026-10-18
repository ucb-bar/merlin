"""Ordinary whole-model image reuse must bind pass selection and emitted bytes.

The linked-image builder is stubbed: these tests check cache transactions, not
compiler numerical behavior or target support. Source-row execution is tested
independently through ordinary native lowering.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.llvmlower import optional_passes
from merlin.model_check import full_build as FB


def _admit(*, elf, evidence_root):
    evidence_root.mkdir(parents=True, exist_ok=True)
    path = evidence_root / "admission.json"
    result = {"status": "accepted", "elf_sha256": FB.sha256(elf)}
    path.write_text(json.dumps(result))
    return {**result, "report_path": str(path), "report_sha256": FB.sha256(path)}


def _service(target, evaluator=_admit):
    import inspect

    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    owner = Path(inspect.getsourcefile(evaluator)).resolve()
    return LinkedElfAdmissionService(target, evaluator, ((str(owner), FB.sha256(owner)),))


@pytest.fixture
def builds(tmp_path, monkeypatch):
    from merlin.compile import baremetal_model

    source = tmp_path / "pass.py"
    source.write_text("default = False\n")
    monkeypatch.setattr(FB, "_lowering_sources", lambda: {"selected.pass": source})
    monkeypatch.setattr(FB, "elf_sections", lambda path: {".text": path.stat().st_size})
    monkeypatch.setattr(FB, "_routing", lambda *args: "admitted-device-route")
    calls = []
    changed = []

    def compiler(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output"])
        cgen = output / "cgen"
        cgen.mkdir(parents=True)
        (cgen / "weights.bin").write_bytes(b"unchanged weights")
        elf = output / "model.elf"
        elf.write_bytes(optional_passes.active().spell().encode() or b"default")
        for action in changed:
            action()
        result = {"output": {"elf": str(elf), "elf_sha256": FB.sha256(elf), "build_hash": "observed"}}
        if kwargs.get("op_profile"):
            table = output / "op_profile_table.json"
            table.write_text('[{"id":0,"mlir_op":"arith.constant"}]')
            result["output"]["op_profile_table"] = {"path": str(table), "sha256": FB.sha256(table)}
        return result

    monkeypatch.setattr(baremetal_model, "compile_saved_model", compiler)
    for name in ("capture", "host", "device"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "source").write_text("input1")
    (tmp_path / "boards").write_text("board1")
    (tmp_path / "cpu.dts").write_text("host01")
    settings = FB.BuildSettings(
        "selected-board", tmp_path / "boards", tmp_path / "cpu.dts", "typed", 1, tmp_path / "host")

    def build(*, target="neutral", offload=True, op_profile=False, admission=None, weights_source=None):
        return FB.build(
            tmp_path / "capture", target=target, label="program", submission=tmp_path / "device",
            submission_sha256="source", settings=replace(settings, op_profile=op_profile),
            directory=tmp_path / "image-cache",
            admission=admission or _service(target), offload=offload,
            weights_source=weights_source,
        )

    return build, calls, source, changed


@pytest.mark.parametrize("relative", ["capture/source", "host/source", "device/source", "boards", "cpu.dts"])
def test_current_input_bytes_invalidate_even_with_same_size_and_mtime(builds, relative):
    build, calls, source, _ = builds
    first = build()
    assert build()["reused"]
    path = source.parent / relative
    stat = path.stat()
    path.write_text("input2")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns
    second = build()
    assert not second["reused"] and build()["reused"]
    assert first["key"] != second["key"] and len(calls) == 2


@pytest.mark.parametrize("relative", ["capture/source", "host/source", "device/source", "boards", "cpu.dts"])
@pytest.mark.parametrize("during", ["compilation", "cache lookup"])
def test_inputs_changed_during_transaction_are_not_published_or_reused(builds, monkeypatch, relative, during):
    build, calls, source, changed = builds
    path = source.parent / relative
    if during == "compilation":
        changed.append(lambda: path.write_text("input2"))
    else:
        first = build()
        original_sha = FB.sha256

        def hashing(member):
            result = original_sha(member)
            if Path(member) == Path(first["elf"]):
                path.write_text("input2")
            return result

        monkeypatch.setattr(FB, "sha256", hashing)
    with pytest.raises(FB.FullBuildError, match="inputs changed during " + during):
        build()
    assert len(calls) == 1
    if during == "compilation":
        assert not (source.parent / "image-cache/build.json").exists()


@pytest.mark.parametrize("relative", ["capture/source", "host/source", "device/source", "boards", "cpu.dts"])
@pytest.mark.parametrize("invalid", ["missing", "symlink"])
def test_invalid_current_inputs_cannot_use_old_image(builds, relative, invalid):
    build, calls, source, _ = builds
    build()
    member = source.parent / relative
    member.unlink()
    if invalid == "symlink":
        member.symlink_to(source)
    with pytest.raises(FB.FullBuildError, match="current full-model build inputs|input contains a symlink"):
        build()
    assert len(calls) == 1


def test_unknown_policy_state_and_renamed_capture_files_are_part_of_identity(builds):
    build, calls, source, _ = builds
    first = build()
    root = source.parent / "capture"
    policy = root / "numerical-policy.json"
    policy.write_text('{"rounding":"exact"}')
    assert not build()["reused"]
    policy.rename(root / "state-contract.json")
    assert not build()["reused"]
    (root / "state-contract.json").unlink()
    final = build()
    assert not final["reused"] and final["key"] == first["key"] and len(calls) == 4


def test_host_twin_does_not_depend_on_submission_bytes_or_existence(builds):
    build, calls, source, _ = builds
    first = build(offload=False)
    package = source.parent / "device"
    (package / "source").unlink()
    package.rmdir()
    assert build(offload=False)["reused"] and len(calls) == 1
    assert "device_package" not in first["key"]["inputs"]


def test_input_cache_overlap_is_refused_before_compiling(builds):
    _, calls, source, _ = builds
    stage = source.parent / "capture"
    with pytest.raises(FB.FullBuildError, match="contains the image cache"):
        FB.build(stage, target="neutral", label="program", submission=source.parent / "device",
                 submission_sha256="source", settings=FB.BuildSettings(
                     "board", source.parent / "boards", source.parent / "cpu.dts", "typed", 1, source.parent / "host"),
                 directory=stage / "cache", offload=False)
    assert not calls


@pytest.mark.parametrize("geometry", [(3, 8, 4096), (16, 4, 2048)])
def test_scoped_rtl_facts_without_artifact_bind_current_hardware(builds, geometry):
    from merlin.targetgen.rtl.facts import observed_facts

    build, calls, _, _ = builds
    rows, cols, capacity = geometry
    with observed_facts("neutral", {"facts": {"primitive": [rows, cols], "bytes": capacity}}):
        first = build()
        assert build()["reused"]
        with observed_facts("neutral", {"facts": {"primitive": [cols, rows], "bytes": capacity // 2}}):
            second = build()
            assert not second["reused"] and build()["reused"]
            assert first["key"]["inputs"] != second["key"]["inputs"]
        assert not build()["reused"]
    assert len(calls) == 3


def test_explicit_rtl_file_changes_and_disappears_without_extraction(builds, monkeypatch):
    from merlin.targetgen.rtl import facts

    build, calls, source, _ = builds
    path = source.parent / "facts.json"
    path.write_text('{"facts":{"capacity":2048}}')
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(path))
    monkeypatch.setattr(facts, "ensure_facts", lambda *a, **kw: pytest.fail("cache lookup extracted RTL"))
    build()
    assert build()["reused"]
    path.write_text('{"facts":{"capacity":4096}}')
    assert not build()["reused"]
    path.unlink()
    with pytest.raises(FB.FullBuildError, match="current full-model build inputs"):
        build()
    assert len(calls) == 2


def test_scoped_software_semantics_bind_current_selection_even_without_source_path(builds):
    from merlin.targetgen.target_registry import observed_contract

    build, calls, _, _ = builds
    with observed_contract("neutral", {"name": "neutral", "rounding": "nearest-even"}):
        first = build()
        assert build()["reused"]
        with observed_contract("neutral", {"name": "neutral", "rounding": "toward-zero"}):
            second = build()
            assert not second["reused"] and build()["reused"]
            assert first["key"]["inputs"]["selected_capability_sha256"] != second[
                "key"]["inputs"]["selected_capability_sha256"]
        assert not build()["reused"]
    assert len(calls) == 3


@pytest.mark.parametrize("relative", ["target_contract.yaml", "dialect_plan.yaml", "rtl_facts/facts.json"])
def test_selected_provider_metadata_changes_without_importing_provider_code(builds, monkeypatch, relative):
    build, calls, source, _ = builds
    provider = source.parent / "provider"
    contracts = provider / "contracts"
    (contracts / "rtl_facts").mkdir(parents=True)
    (contracts / "target_contract.yaml").write_text("name: neutral\nplugin: {backend: never.import.this}\n")
    (contracts / "dialect_plan.yaml").write_text("operations: []\n")
    (contracts / "rtl_facts/facts.json").write_text('{"facts":{"target":"neutral","capacity":4096}}')
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider))
    first = build()
    assert build()["reused"]
    member = contracts / relative
    member.write_text(member.read_text() + "\n")
    second = build()
    assert not second["reused"] and build()["reused"]
    assert first["key"] != second["key"] and len(calls) == 2


@pytest.mark.parametrize("during_build", [False, True])
def test_reused_weights_source_is_rechecked_before_reuse_or_publication(builds, during_build):
    build, calls, source, changed = builds
    weights = source.parent / "weights.bin"
    weights.write_bytes(b"original")
    if during_build:
        changed.append(lambda: weights.write_bytes(b"modified"))
        with pytest.raises(FB.FullBuildError, match="inputs changed during compilation"):
            build(weights_source=weights)
        assert not (source.parent / "image-cache/build.json").exists()
    else:
        build(weights_source=weights)
        assert build(weights_source=weights)["reused"]
        weights.write_bytes(b"modified")
        assert not build(weights_source=weights)["reused"] and len(calls) == 2


def test_elf_changed_during_final_input_validation_is_not_returned(builds, monkeypatch):
    build, calls, _, _ = builds
    first = build(offload=False)
    reader = FB._build_inputs
    reads = []

    def inputs(*args, **kwargs):
        result = reader(*args, **kwargs)
        reads.append(result)
        if len(reads) == 2:
            elf = Path(first["elf"])
            elf.write_bytes(b"x" * elf.stat().st_size)
        return result

    monkeypatch.setattr(FB, "_build_inputs", inputs)
    with pytest.raises(FB.FullBuildError, match="ELF changed during cache lookup"):
        build(offload=False)
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["source", "controls", "profile"])
def test_late_validation_changes_cannot_escape_final_rechecks(builds, monkeypatch, mutation):
    build, calls, source, _ = builds
    first = build(op_profile=True)
    reader, reads = FB._build_inputs, []

    def inputs(*args, **kwargs):
        result = reader(*args, **kwargs)
        reads.append(result)
        if len(reads) == 2:
            if mutation == "source":
                source.write_text("different source")
            elif mutation == "controls":
                monkeypatch.setenv("MERLIN_NEW_CONTROL", "changed during input validation")
            else:
                Path(first["op_profile_table"]["path"]).write_text("[]")
        return result

    monkeypatch.setattr(FB, "_build_inputs", inputs)
    with pytest.raises(FB.FullBuildError, match="changed during cache lookup"):
        build(op_profile=True)
    assert len(calls) == 1


def test_scoped_software_backing_bytes_are_bound_alongside_observed_document(builds):
    from merlin.targetgen.target_registry import observed_contract

    build, calls, source, _ = builds
    path = source.parent / "software.yaml"
    path.write_text("name: neutral\n")
    with observed_contract("neutral", {"name": "neutral"}, source_path=path):
        build()
        assert build()["reused"]
        path.write_text("name: neutral\n\n")
        assert not build()["reused"]
        path.unlink()
        with pytest.raises(FB.FullBuildError, match="current full-model build inputs"):
            build()
    assert len(calls) == 2


def _refuse(*, elf, evidence_root):
    result = _admit(elf=elf, evidence_root=evidence_root)
    result["status"] = "refused"
    report = Path(result["report_path"])
    report.write_text(json.dumps({"status": "refused", "elf_sha256": result["elf_sha256"]}))
    return {**result, "report_sha256": FB.sha256(report)}


def test_cache_hit_applies_current_admission_instead_of_old_acceptance(builds):
    build, calls, _, _ = builds
    build()
    reused = build()
    assert reused["cache_admission"]["status"] == "accepted"
    assert Path(reused["cache_admission"]["report_path"]).is_file()
    assert reused["cache_lookup_seconds"] >= 0
    with pytest.raises(FB.FullBuildError, match="current linked ELF admission refused"):
        build(admission=_service("neutral", _refuse))
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["elf", "report", "source"])
def test_admission_cannot_change_its_artifact_report_or_source_selection(builds, mutation):
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    build, calls, source, _ = builds
    first = build()
    service = _service("neutral")
    original = service.revalidate

    def revalidate(self, *, elf, result, target):
        if mutation == "elf":
            Path(elf).write_bytes(b"x" * Path(elf).stat().st_size)
        elif mutation == "report":
            Path(result["report_path"]).write_text("{}")
        else:
            object.__setattr__(self, "source_pins", ((str(source), "0" * 64),))
        return original(elf=elf, result=result, target=target)

    # Actual service checks are retained; inject drift at its verification boundary.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LinkedElfAdmissionService, "revalidate", revalidate)
        with pytest.raises(ValueError, match="changed|unchanged actual"):
            build(admission=service)
    assert len(calls) == 1 and Path(first["elf"]).is_file()


@pytest.mark.parametrize("offload", [False, True])
def test_same_directory_reuses_identical_selection_and_rebuilds_pass_ablation(builds, offload):
    build, calls, _, _ = builds
    for choice in ("-automatic-host-rows", "+automatic-host-rows", "+automatic-host-numerics"):
        with optional_passes.applied(optional_passes.Selection.parse(choice)):
            first = build(offload=offload)
            assert not first["reused"]
            assert build(offload=offload)["reused"]
            assert Path(first["elf"]).read_text() == optional_passes.active().spell()
    assert len(calls) == 3
    assert calls[0]["device"] == ("admitted-device-route" if offload else None)


@pytest.mark.parametrize("name", ["MERLIN_AUTOMATIC_HOST_ROWS", "MERLIN_FUTURE_SCHEDULING_CONTROL"])
def test_switches_and_future_controls_invalidate_without_explicit_pass_selection(builds, monkeypatch, name):
    build, calls, _, _ = builds
    monkeypatch.delenv(optional_passes.ENV, raising=False)
    monkeypatch.delenv(name, raising=False)
    first = build()
    monkeypatch.setenv(name, "")
    assert not build()["reused"]
    monkeypatch.setenv(name, "0")
    assert not build()["reused"]
    assert build()["reused"]
    assert len(calls) == 3
    assert first["key"]["lowering"]["project_controls_sha256"] != build()["key"]["lowering"]["project_controls_sha256"]


def test_same_size_source_and_elf_mutations_and_foreign_target_force_rebuild(builds):
    build, calls, source, _ = builds
    first = build()
    assert build()["reused"]
    original = source.stat()
    source.write_text("default = True \n")
    assert source.stat().st_size == original.st_size
    assert not build()["reused"]
    elf = Path(first["elf"])
    elf.write_bytes(b"x" * elf.stat().st_size)
    assert not build()["reused"]
    assert not build(target="other-neutral")["reused"]
    assert len(calls) == 4


@pytest.mark.parametrize("during_build", [False, True])
def test_native_abi_source_changes_invalidate_image_reuse(builds, tmp_path, monkeypatch, during_build):
    from merlin.common import paths

    build, calls, _, changed = builds
    runtime = tmp_path / "runtime"
    (runtime / "abi").mkdir(parents=True)
    source = runtime / "abi/mlir_runtime.c"
    source.write_text("body1")
    monkeypatch.setattr(paths, "runtime_dir", lambda: runtime)
    if during_build:
        changed.append(lambda: source.write_text("body2"))
        with pytest.raises(FB.FullBuildError, match="changed during compilation"):
            build()
        assert not (tmp_path / "image-cache/build.json").exists()
        return
    first = build()
    assert build()["reused"]
    source.write_text("body2")
    second = build()
    assert not second["reused"] and build()["reused"] and len(calls) == 2
    assert first["key"]["lowering"]["native_abi_source_sha256"] != second["key"]["lowering"]["native_abi_source_sha256"]


@pytest.mark.parametrize("change", ["selection", "source"])
def test_changed_selection_or_source_during_compilation_is_never_published(builds, monkeypatch, change):
    build, calls, source, changed = builds
    if change == "selection":
        changed.append(lambda: monkeypatch.setenv("MERLIN_FUTURE_CONTROL", "different"))
    else:
        changed.append(lambda: source.write_text("changed while compiling\n"))
    with pytest.raises(FB.FullBuildError, match="changed during compilation"):
        build()
    assert len(calls) == 1
    assert not (source.parent / "image-cache/build.json").exists()


def test_legacy_record_missing_selection_cannot_hit(builds):
    build, calls, _, _ = builds
    first = build()
    record_path = Path(first["elf"]).parent.parent / "build.json"
    record = json.loads(record_path.read_text())
    record["key"].pop("lowering")
    record_path.write_text(json.dumps(record))
    assert not build()["reused"]
    assert len(calls) == 2


def test_controls_changed_during_cache_lookup_cannot_authorize_a_hit(builds, monkeypatch):
    build, calls, _, _ = builds
    first = build()
    original_sha = FB.sha256

    def hashing(path):
        result = original_sha(path)
        if Path(path) == Path(first["elf"]):
            monkeypatch.setenv("MERLIN_NEW_CONTROL", "changed during hash")
        return result

    monkeypatch.setattr(FB, "sha256", hashing)
    with pytest.raises(FB.FullBuildError, match="changed during cache lookup"):
        build()
    assert len(calls) == 1


def test_default_source_inventory_contains_actual_row_and_numerical_pass_owners():
    sources = FB._lowering_sources()
    assert sources["merlin.llvmlower/automatic_rows.py"].is_file()
    assert sources["merlin.transforms/row_normalization_check.py"].is_file()
    assert sources["merlin.llvmlower/automatic_host.py"].is_file()
    assert sources["merlin.llvmlower/optional_passes.py"].is_file()
    assert all(path.suffix == ".py" for path in sources.values())


@pytest.mark.parametrize("mutation", ["content", "missing", "symlink", "receipt"])
def test_profile_selection_and_table_mutation_prevent_image_cache_reuse(builds, mutation):
    build, calls, _, _ = builds
    plain = build()
    assert "op_profile" not in calls[0] and "op_profile_table" not in plain
    profiled = build(op_profile=True)
    assert not profiled["reused"] and calls[1]["op_profile"] is True
    assert build(op_profile=True)["reused"]
    table = Path(profiled["op_profile_table"]["path"])
    if mutation == "content":
        table.write_text("[]")
    elif mutation == "missing":
        table.unlink()
    elif mutation == "symlink":
        content = table.read_bytes()
        table.unlink()
        other = table.with_suffix(".other")
        other.write_bytes(content)
        table.symlink_to(other)
    else:
        receipt = table.parent.parent / "build.json"
        record = json.loads(receipt.read_text())
        record.pop("op_profile_table")
        receipt.write_text(json.dumps(record))
    assert not build(op_profile=True)["reused"]
    assert len(calls) == 3
    assert not build()["reused"]


@pytest.mark.parametrize("storage", ["materialized", "streamed"])
def test_actual_whole_model_image_reuse_and_changed_source_execute_complete_state(tmp_path, monkeypatch, storage):
    """Real ordinary build/relink/Spike path; no backend or compiler stubs."""
    import struct

    import numpy as np
    import yaml

    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike, spike_model
    from merlin.targetgen.rtl.facts import observed_facts

    if not toolchain.available() or not spike.available():
        pytest.skip("selected native lowering, cross toolchain and functional simulator required")
    root = tmp_path / "out"
    root.mkdir()
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    monkeypatch.setenv("MERLIN_WEIGHT_PAYLOAD_STORAGE", storage)
    capture, host = tmp_path / "capture", tmp_path / "host"
    capture.mkdir()
    host.mkdir()
    (host / "manifest.yaml").write_text(yaml.safe_dump({
        "target": "host", "run_id": "cache-test", "family": "scalar_linalg",
        "authoring": {"mode": "deterministic_generated_from_spec", "generated_by_agent": False},
        "outputs": {"knobs": "knobs.yaml"},
    }))
    (host / "knobs.yaml").write_text(yaml.safe_dump({
        "backend": "scalar", "dtype_strategy": "fp32",
        "cflags": ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
    }))
    dts, catalog = tmp_path / "cpu.dts", tmp_path / "boards.yaml"
    dts.write_text('cpu@0 { device_type = "cpu"; riscv,isa = "rv64imafdc_zicsr_zifencei"; };\n')
    catalog.write_text(yaml.safe_dump({"schema_version": 1, "boards": {"selected": {
        "target": "neutral", "dram_bytes": 64 << 20, "dram_base": spike_model.DRAM_BASE,
        "harts": 1, "console": "htif", "flow": "baremetal", "code_reserve": 16 << 20,
        "host_dts_sha256": FB.sha256(dts),
    }}}))
    settings = FB.BuildSettings("selected", catalog, dts, "fp32", 1, host)
    weight = np.array([2, -3], np.float32)
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    (capture / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + weight.tobytes())
    (capture / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "param", "weight": "w"}, "1": {"kind": "input", "name": "x"},
        "2": {"kind": "input", "name": "state"},
    }))
    initial = np.array([1, 4], np.float32)
    np.savez(capture / "inputs.npz", in0=initial, in1=np.zeros(2, np.float32))
    np.save(capture / "golden.npy", initial + weight)
    (capture / "input_order.json").write_text('{"x":0,"state":1}')

    def select(operation):
        np.save(capture / "golden.npy", (weight + initial) if operation == "addf" else (weight * initial))
        (capture / "model.mlir").write_text('''module {
func.func @forward(%w: tensor<2xf32>, %x: tensor<2xf32>, %s: tensor<2xf32>)
    -> (tensor<2xf32>, tensor<2xf32>) {
  %e = tensor.empty() : tensor<2xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
    iterator_types = ["parallel"]} ins(%w, %x : tensor<2xf32>, tensor<2xf32>) outs(%e : tensor<2xf32>) {
  ^bb0(%w0: f32, %x0: f32, %o0: f32):
    %v = arith.OP %w0, %x0 : f32
    linalg.yield %v : f32
  } -> tensor<2xf32>
  %n = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
    iterator_types = ["parallel"]} ins(%s, %r : tensor<2xf32>, tensor<2xf32>) outs(%e : tensor<2xf32>) {
  ^bb0(%s0: f32, %r0: f32, %o0: f32):
    %v = arith.addf %s0, %r0 : f32
    linalg.yield %v : f32
  } -> tensor<2xf32>
  return %r, %n : tensor<2xf32>, tensor<2xf32>
}}'''.replace("arith.OP", "arith." + operation))
        artifacts = {p.name: {"bytes": p.stat().st_size, "sha256": FB.sha256(p)}
                     for p in capture.iterdir() if p.name != "capture_receipt.json"}
        (capture / "capture_receipt.json").write_text(json.dumps({
            "schema": "m2m.capture-receipt.v1", "materialized_abi": {"complete": True}, "artifacts": artifacts,
        }))

    def build():
        return FB.build(capture, target="neutral", label="stateful", submission=tmp_path / "unused",
                        submission_sha256="unused", settings=settings, directory=root / "image-cache", offload=False)

    evidence = []
    with observed_facts("neutral", {"facts": {}}):
        for op, arithmetic in (("addf", np.add), ("mulf", np.multiply)):
            select(op)
            fresh = build()
            from merlin.llvmlower.compilation_recipe import verify_completed_recipe

            recipe = Path(fresh["build_dir"]) / "compilation_recipe.json"
            verify_completed_recipe(recipe, executable=Path(fresh["elf"]), expected_recipe_sha256=FB.sha256(recipe))
            reused = build()
            assert not fresh["reused"] and reused["reused"]
            (root / f"{op}.build.json").write_text(json.dumps(reused, indent=2))
            (root / f"{op}.mlir").write_bytes((capture / "model.mlir").read_bytes())
            (root / f"{op}.recipe.json").write_bytes(
                (Path(reused["build_dir"]) / "compilation_recipe.json").read_bytes())
            state = np.zeros(2, np.float32)
            for step in range(3):
                bound = {"in0": initial + step, "in1": state}
                elf = FB.relink(reused, capture, bound, root / f"{op}-{step}")
                relink_recipe = elf.parent / "compilation_recipe.json"
                actual = verify_completed_recipe(
                    relink_recipe, executable=elf, expected_recipe_sha256=FB.sha256(relink_recipe))
                assert len(actual["commands"]) == 2
                assert any(Path(item["path"]) == elf.parent / "cgen/model_io.h"
                           for item in actual["commands"][0]["inputs"])
                console, wall = FB.run_host(elf, timeout_s=30)
                (root / f"{op}-{step}.console.bin").write_bytes(console)
                outputs, _ = FB.read(console, capture, reused["build_hash"])
                expected = arithmetic(weight, bound["in0"])
                np.testing.assert_array_equal(outputs[0], expected)
                np.testing.assert_array_equal(outputs[1], state + expected)
                state = outputs[1].copy()
                evidence.append({"operation": op, "step": step, "outputs": [a.tolist() for a in outputs],
                                 "elf_sha256": FB.sha256(elf), "sections": FB.elf_sections(elf),
                                 "functional_simulator_wall_seconds": wall, "key": reused["key"]})
            original_source = (capture / "model.mlir").read_bytes()
            (capture / "model.mlir").write_bytes(original_source.replace(b"arith.addf", b"arith.subf", 1))
            with pytest.raises(FB.FullBuildError, match="relink build inputs changed"):
                FB.relink(reused, capture, bound, root / f"{op}-stale-source")
            (capture / "model.mlir").write_bytes(original_source)
            object_path = Path(reused["build_dir"]) / "model.o"
            original_object = object_path.read_bytes()
            object_path.write_bytes(original_object[:-1] + bytes([original_object[-1] ^ 1]))
            with pytest.raises(FB.FullBuildError, match="changed|rebuild"):
                FB.relink(reused, capture, bound, root / f"{op}-stale-object")
            object_path.write_bytes(original_object)
        # Hardware drift must re-enter normal admission, which rejects a broken DTS pin.
        dts.write_text(dts.read_text() + "\n")
        with pytest.raises(ValueError, match="DTS bytes differ"):
            build()
    (root / "execution-evidence.json").write_text(json.dumps(evidence, indent=2))
