"""Ordinary whole-model image reuse must bind pass selection and emitted bytes.

The linked-image builder is stubbed: these tests check cache transactions, not
compiler numerical behavior or target support. Source-row execution is tested
independently through ordinary native lowering.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.llvmlower import optional_passes
from merlin.model_check import full_build as FB


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
    settings = FB.BuildSettings("selected-board", tmp_path / "boards", tmp_path / "cpu.dts", "typed", 1, tmp_path)

    def build(*, target="neutral", offload=True, op_profile=False):
        return FB.build(
            tmp_path, target=target, label="program", submission=tmp_path,
            submission_sha256="source", settings=replace(settings, op_profile=op_profile),
            directory=tmp_path / "image-cache",
            admission=object(), offload=offload,
        )

    return build, calls, source, changed


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
