"""File-only receipt joins; synthetic records issue no compiler/engine verdict."""

import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import candidate_observations as O
from merlin_experiments.phase2 import ordinary_calibration as CAL
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST

from merlin.common import invocation_record as I
from merlin.targetgen import capsule_runner as CR


def _pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def _elf():
    names, strings = b"\0.names\0.code\0.strings\0.symbols\0", b"\0entry\0"
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHIQQQIHHHHHH", header, 16, 2, 0, 1, 1024, 0, 64, 0, 64, 0, 0, 64, 5, 1)
    start = 64 + 5 * 64
    sections = [bytes(64), struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, start, len(names), 0, 0, 1, 0)]
    start += len(names)
    sections.append(struct.pack("<IIQQQQIIQQ", 8, 1, 4, 1024, start, 8, 0, 0, 1, 0))
    start += 8
    sections.append(struct.pack("<IIQQQQIIQQ", 14, 3, 0, 0, start, len(strings), 0, 0, 1, 0))
    start += len(strings)
    sections.append(struct.pack("<IIQQQQIIQQ", 23, 2, 0, 0, start, 24, 3, 0, 8, 24))
    return (bytes(header) + b"".join(sections) + names + b"opaque00" + strings
            + struct.pack("<IBBHQQ", 1, 2, 0, 2, 1024, 8))


def _console():
    return "synthetic original console; no native execution\n"


def _record(root, tool, stage, inputs, outputs):
    with I.observe(root, stage=stage, argv=[str(tool)], inputs=inputs, outputs=outputs, env={}) as record:
        record.complete(subprocess.CompletedProcess([], 0, b"", b""))
    return record.path


@pytest.fixture
def native_products(tmp_path, monkeypatch):
    assert Path(O.__file__).resolve().is_relative_to(Path.cwd() / "packages/merlin-experiments/src")
    assert Path(I.__file__).resolve().is_relative_to(Path.cwd() / "src")
    root = tmp_path / "native-owner"
    root.mkdir()
    source = tmp_path / "public-interface"
    source.write_text("module {}")
    paths = {role: root / (role + ".actual") for role in ("emitted", "command_buffer", "copy", "ll", "object", "elf")}
    paths["emitted"].write_text("module { llvm.func @entry() { llvm.return } }")
    paths["command_buffer"].write_text(json.dumps({"commands": [], "tensors": {}}))
    paths["copy"].write_bytes(paths["emitted"].read_bytes())
    paths["ll"].write_text("synthetic translated program")
    paths["object"].write_bytes(b"synthetic object")
    paths["elf"].write_bytes(_elf())
    tool = root / "selected-build-tool"
    tool.write_bytes(b"synthetic pinned executable; never executed")
    stages = [_record(root, tool, "llvm_translation", (paths["copy"],), (paths["ll"],)),
              _record(root, tool, "object", (paths["ll"],), (paths["object"],)),
              _record(root, tool, "elf", (paths["object"],), (paths["elf"],))]
    emission = {"source_interface": _pin(source), "lowered_mlir": _pin(paths["emitted"]),
                "command_buffer": _pin(paths["command_buffer"])}
    grade = {"candidate_emission": emission, "candidate_native_execution": {
        "elf": _pin(paths["elf"]), "source": {"interface": _pin(source)},
        "build_artifacts": {role: _pin(paths[role]) for role in ("copy", "ll", "object", "elf")},
        "candidate": {"lowered_mlir_sha256": emission["lowered_mlir"]["sha256"],
                      "command_buffer_sha256": hashlib.sha256(json.dumps(
                          json.loads(paths["command_buffer"].read_text()), sort_keys=True).encode()).hexdigest()}}}
    frame = SimpleNamespace(entry_symbol="entry")
    recipe = SimpleNamespace(require_kernel_stack_frame=lambda: frame)
    monkeypatch.setattr(FAST, "get_backend", lambda target: SimpleNamespace(harness_build_recipe=lambda: recipe))
    def unsupported(**kwargs):
        raise ValueError("old scalar-site grammar unsupported")
    monkeypatch.setattr(FAST.S, "derive_compiled_static_features", unsupported)
    return SimpleNamespace(root=root, source=source, paths=paths, grade=grade, stages=stages)


def test_actual_native_roles_keep_rich_features_when_old_site_grammar_is_unknown(native_products):
    fixture = native_products
    # Decoys from the old naming convention cannot replace explicit native roles.
    (fixture.root / "lowered.llvm.mlir").write_text("wrong product")
    (fixture.root / FAST.PACKAGE_ELF_NAME).write_bytes(b"wrong product")
    result = FAST.compiled_observation(fixture.root, target="synthetic", original_grade=fixture.grade)
    assert result["status"] == "observed"
    assert result["static_status"] == "UNKNOWN"
    assert all(value is None for value in result["features"].values())
    assert result["elf_sha256"] == _pin(fixture.paths["elf"])["sha256"]
    assert result["llvm_mlir_sha256"] == _pin(fixture.paths["emitted"])["sha256"]
    diagnostics = result["candidate_diagnostics"]
    assert diagnostics["product_collection"]["consumption_join"] == "reopened_original_native_products"
    assert diagnostics["candidate_features"]["features"]["linked_executable_bytes"] == 8
    assert str(fixture.root) not in json.dumps(result)


def _transport_source(tmp_path, monkeypatch):
    from merlin.common import content_store

    source = tmp_path / "original-capsule"
    source.mkdir()
    (source / "capsule.yaml").write_text("kind: model\n")
    (source / "capsule.interface.mlir").write_text("module {}")
    for name in ("capsule.yaml", "capsule.interface.mlir"):
        os.link(source / name, tmp_path / (name + ".original-alias"))
    monkeypatch.setattr(content_store, "store_root", lambda: tmp_path / "store")
    cap = CR._pin_model_capsule({"kind": "model", "__dir__": str(source)}, tmp_path / "consumed")
    return source, cap


def test_legacy_model_without_interface_keeps_its_existing_transport(tmp_path, monkeypatch):
    from merlin.common import content_store

    source = tmp_path / "legacy"
    source.mkdir()
    (source / "capsule.yaml").write_text("kind: model\nmodel: original-loader\n")
    monkeypatch.setattr(content_store, "store_root", lambda: tmp_path / "store")
    cap = CR._pin_model_capsule({"kind": "model", "__dir__": str(source)}, tmp_path / "consumed")
    assert set(cap["__model_source_transport"]["inputs"]) == {"capsule_declaration"}
    assert not (Path(cap["__dir__"]) / "capsule.interface.mlir").exists()


def test_source_drift_during_transport_is_refused(tmp_path, monkeypatch):
    import shutil

    original = shutil.copyfile

    def changed(source, destination, **kwargs):
        value = original(source, destination, **kwargs)
        if Path(source).name == "capsule.interface.mlir":
            Path(destination).write_text("foreign source during copy")
        return value

    monkeypatch.setattr(shutil, "copyfile", changed)
    with pytest.raises(ValueError, match="changed during private input transport"):
        _transport_source(tmp_path, monkeypatch)


def test_runner_keeps_patchpoint_and_delegates_to_source_identity(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_source_identity as identity

    capsule, destination, expected = {"kind": "model"}, tmp_path / "private", object()
    calls = []

    def pin(actual_capsule, actual_destination):
        calls.append((actual_capsule, actual_destination))
        return expected

    monkeypatch.setattr(identity, "_pin_model_capsule", pin)
    assert CR._pin_model_capsule(capsule, destination) is expected
    assert calls == [(capsule, destination)]


def test_sealed_child_closure_reopens_existing_source_identity_module(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import compiler_profile_child as child

    from merlin.common import paths
    from merlin.targetgen import capsule_source_identity as identity

    source = Path(identity.__file__).resolve()
    module = "merlin.targetgen.capsule_source_identity"
    assert str(source) in {pin["path"] for pin in child._closure()}
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    original = paths.module_source_path
    monkeypatch.setattr(paths, "module_source_path", lambda name: copied if name == module else original(name))
    document = {"schema": child.SCHEMA, "host_sources": child._closure()}
    assert child._selected_closure(document) == document["host_sources"]
    copied.write_bytes(copied.read_bytes() + b"\n# source custody implementation changed\n")
    assert child._selected_closure(document) != document["host_sources"]


def test_hardlinked_original_sources_are_copied_before_original_product_join(native_products, tmp_path, monkeypatch):
    fixture = native_products
    source, cap = _transport_source(tmp_path, monkeypatch)
    transport = cap["__model_source_transport"]
    for role, row in transport["inputs"].items():
        original, consumed = Path(row["original"]["path"]), Path(row["consumed"]["path"])
        assert original.stat().st_nlink >= 2
        assert consumed.stat().st_nlink == 1
        assert original.stat().st_ino != consumed.stat().st_ino
        assert original.read_bytes() == consumed.read_bytes()
        assert not consumed.stat().st_mode & 0o222
        fixture.grade["candidate_emission"][role] = _pin(consumed)
    fixture.grade["candidate_native_execution"]["source"]["interface"] = _pin(
        Path(cap["__dir__"]) / "capsule.interface.mlir")
    result = FAST.compiled_observation(fixture.root, target="synthetic", original_grade=fixture.grade)
    assert result["status"] == "observed"
    # An independent later source change cannot mutate the actual consumed bytes.
    (source / "capsule.interface.mlir").write_text("changed original after transport")
    assert (Path(cap["__dir__"]) / "capsule.interface.mlir").read_text() == "module {}"


@pytest.mark.parametrize("damage", ["candidate_hardlink", "foreign_source_hardlink", "source_drift"])
def test_private_source_transport_never_relaxes_original_product_checks(
        native_products, tmp_path, monkeypatch, damage):
    fixture = native_products
    source, cap = _transport_source(tmp_path, monkeypatch)
    consumed = Path(cap["__dir__"]) / "capsule.interface.mlir"
    fixture.grade["candidate_emission"]["source_interface"] = _pin(consumed)
    fixture.grade["candidate_emission"]["capsule_declaration"] = _pin(Path(cap["__dir__"]) / "capsule.yaml")
    fixture.grade["candidate_native_execution"]["source"]["interface"] = _pin(consumed)
    if damage == "candidate_hardlink":
        os.link(fixture.paths["emitted"], tmp_path / "foreign-product")
    elif damage == "foreign_source_hardlink":
        fixture.grade["candidate_emission"]["source_interface"] = _pin(source / "capsule.interface.mlir")
    else:
        consumed.chmod(0o600)
        consumed.write_text("changed consumed source")
    result = FAST.compiled_observation(fixture.root, target="synthetic", original_grade=fixture.grade)
    assert result["status"] == "UNKNOWN"


@pytest.mark.parametrize("drift", ["source", "emitted", "copy", "object", "elf", "receipt", "role"])
def test_native_roles_or_actual_producer_drift_cannot_supply_cheap_features(native_products, drift):
    fixture = native_products
    if drift == "source":
        fixture.source.write_text("changed public source")
    elif drift == "receipt":
        row = json.loads(fixture.stages[-1].read_text())
        row["stage"] = "object"
        fixture.stages[-1].write_text(json.dumps(row))
    elif drift == "role":
        fixture.grade["candidate_emission"]["lowered_mlir"] = _pin(fixture.paths["ll"])
    else:
        fixture.paths[drift].write_bytes(b"changed actual product")
    result = FAST.compiled_observation(fixture.root, target="synthetic", original_grade=fixture.grade)
    assert result["status"] == "UNKNOWN"
    assert all(value is None for value in result["features"].values())
    assert result["candidate_diagnostics"]["candidate_features"]["features"]["linked_executable_bytes"] is None


def test_native_calibration_reopens_same_program_engine_and_full_numeric_console(native_products):
    fixture = native_products
    compiled = FAST.compiled_observation(fixture.root, target="synthetic", original_grade=fixture.grade)
    engine = fixture.root / "selected-engine"
    engine.write_bytes(b"synthetic engine identity; never executed")
    console = fixture.root / "actual-console"
    console.write_text(_console())
    with I.observe_call(fixture.root, stage="execution", function=_console, arguments={"simulator": "gsim"},
                        inputs=(fixture.paths["elf"],), dependencies=(engine,)) as recorded:
        recorded.returned(stdout=_console())
    native = fixture.grade["candidate_native_execution"]
    observed = {"elf": str(fixture.paths["elf"]), "elf_sha256": native["elf"]["sha256"],
                "binary_sha256": _pin(engine)["sha256"], "actual_console": _pin(console),
                "execution_invocations": {"L3": _pin(recorded.path)},
                "actual_candidate": fixture.grade["candidate_emission"], "actual_source": native["source"]}
    attempts = fixture.root / "candidate.attempts.json"
    document = {"package_sha256": "a" * 64, "attempts": [{"workspace": str(fixture.root), "raw": {
        "execution": {"family": "public", "capsule": "connected", "arm": "candidate"},
        "measurement": {"numeric": "pass", "gsim_qualification": {"admitted": True},
                        "gsim_execution": {"gsim": observed}}}}]}
    attempts.write_text(json.dumps(document))
    joined = CAL._join(fixture.root, {"family": "public", "capsule": "connected"}, "candidate", compiled, "a" * 64)
    assert joined[0] == observed["elf_sha256"]
    assert str(recorded.path) in {path for path, _ in joined[2]}
    observed["binary_sha256"] = "b" * 64
    attempts.write_text(json.dumps(document))
    with pytest.raises(CAL.CalibrationUnavailable, match="fresh joined"):
        CAL._join(fixture.root, {"family": "public", "capsule": "connected"}, "candidate", compiled, "a" * 64)
    observed["binary_sha256"] = _pin(engine)["sha256"]
    attempts.write_text(json.dumps(document))
    console.write_text("changed numerical console")
    with pytest.raises(ValueError, match="changed"):
        CAL._join(fixture.root, {"family": "public", "capsule": "connected"}, "candidate", compiled, "a" * 64)


def test_shape_static_count_note_preserves_separate_coverage_and_numerical_meaning(tmp_path, monkeypatch, capsys):
    from merlin.targetgen import lowering_coverage as LC

    monkeypatch.setattr(LC, "sweep", lambda *args, **kwargs: {"all_covered": True})
    assert O._shape_coverage(tmp_path, "", context=SimpleNamespace(target="fixture", repo=tmp_path)) == 0
    note = json.loads(capsys.readouterr().out)["note"]
    assert "static instructions" in note
    assert "not evidence of numerical refusal or profitability" in note
    assert "original numerical gates" in note
