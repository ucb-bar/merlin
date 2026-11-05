from __future__ import annotations

import copy
import hashlib
import json

import pytest
import yaml
from merlin_experiments.phase1 import author_input_views as views
from merlin_experiments.phase1 import public_software_declaration as software


def _write(root, name, raw):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw.encode() if isinstance(raw, str) else raw)
    return path


def _capsule():
    return {
        "inputs": [{"name": "A", "shape": [16, 32], "dtype": "i8"}],
        "operation": {"op": "matmul", "attributes": {"epilogue": ["relu"], "output_dtype": "i32"}},
        "numeric_policy": {"compare": "bounded_int", "atol": 1},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        "semantic": {"must_accelerate": True, "generalization_axis": "capacity"},
        "source_role": "derived_sweep",
        "source_reference": "1024 hardware rows",
        "integer_partial_sum_bound": {"mac_result_bits": 20},
        "software_screen": {"reason": "reviewed hardware admission"},
        "pass_requirements": ["tile-schedule"],
        "expected": {"instruction_classes": ["LOAD_CMD"], "modes": {"relu": True}, "outputs": ["Y"]},
        "mathematical_modes": {"relu": True},
    }


def test_projection_preserves_math_and_byte_custody(tmp_path):
    source = tmp_path / "source"
    original = _capsule()
    capsule = _write(source, "case/capsule.yaml", yaml.safe_dump(original))
    mlir = _write(source, "case/capsule.interface.mlir", "module { /* numerical source */ }\n")
    tensor = _write(source, "case/inputs.npz", b"\x00\x01numerical fixture")
    _write(source, "case/README.md", "hardware derived prose")
    _write(source, "case/expected_instruction_coverage.yaml", "instruction_classes: [LOAD_CMD]\n")
    _write(source, "case/frontend-trace.json", '{"transformations": ["derived lowering"]}')
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    destination = tmp_path / "view"
    receipt = views.project_tree(source, destination)
    actual = yaml.safe_load((destination / "case/capsule.yaml").read_text())
    expected = copy.deepcopy(original)
    for key in views._REMOVED_KEYS:
        expected.pop(key, None)
    expected["semantic"].pop("generalization_axis")
    expected["expected"].pop("instruction_classes")
    expected["expected"].pop("modes")
    assert actual == expected
    assert (destination / mlir.relative_to(source)).read_bytes() == mlir.read_bytes()
    assert (destination / tensor.relative_to(source)).read_bytes() == tensor.read_bytes()
    assert sorted(p.name for p in (destination / "case").iterdir()) == [
        "capsule.interface.mlir",
        "capsule.yaml",
        "inputs.npz",
    ]
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert all(not p.stat().st_mode & 0o222 for p in [destination, *destination.rglob("*")])
    row = next(row for row in receipt["files"] if row["path"] == "case/capsule.yaml")
    assert row["source_sha256"] == hashlib.sha256(capsule.read_bytes()).hexdigest()
    assert "semantic/generalization_axis" in row["removed_fields"]
    assert "expected/instruction_classes" in row["removed_fields"]
    second = views.project_tree(source, tmp_path / "second")
    assert receipt["sha256"] == second["sha256"]
    assert receipt["files"] == second["files"]


def test_nested_metadata(tmp_path):
    source = tmp_path / "source"
    value = {
        "cases": [{"source_reference": "private", "semantic": {"generalization_axis": "hint", "must_accelerate": True}}]
    }
    _write(source, "suite.json", json.dumps(value))
    destination = tmp_path / "view"
    views.project_tree(source, destination)
    assert json.loads((destination / "suite.json").read_text()) == {"cases": [{"semantic": {"must_accelerate": True}}]}


@pytest.mark.parametrize("name", ["unknown.yaml", "notes.md", "facts.json", "optimized.elf", "helper.py"])
def test_unclassified_input_fails_before_publication(tmp_path, name):
    source = tmp_path / "source"
    _write(source, name, "unreviewed")
    destination = tmp_path / "view"
    with pytest.raises(ValueError, match="unclassified"):
        views.project_tree(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("name", ["README.md", "capsule.yaml", "directory"])
def test_any_symlink_fails_before_publication(tmp_path, name):
    source = tmp_path / "source"
    source.mkdir()
    target = _write(tmp_path, "private", "private")
    (source / name).symlink_to(target)
    destination = tmp_path / "view"
    with pytest.raises(ValueError, match="nonordinary"):
        views.project_tree(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "name,raw",
    [
        ("capsule.yaml", "numeric_policy: {}\nnumeric_policy: {compare: wrong}\n"),
        ("suite.json", '{"cases": [], "cases": [1]}'),
    ],
)
def test_duplicate_metadata_keys_rejected(tmp_path, name, raw):
    source = tmp_path / "source"
    _write(source, name, raw)
    with pytest.raises(ValueError):
        views.project_tree(source, tmp_path / "view")
    assert not (tmp_path / "view").exists()


def test_source_annotations_are_not_silently_rewritten(tmp_path):
    source = tmp_path / "source"
    _write(source, "capsule.interface.mlir", "module { /* rtl_facts secret */ }")
    with pytest.raises(ValueError, match="annotation"):
        views.project_tree(source, tmp_path / "view")
    assert not (tmp_path / "view").exists()


def test_destination_must_be_fresh_and_disjoint(tmp_path):
    source = tmp_path / "source"
    _write(source, "capsule.yaml", "inputs: []")
    with pytest.raises(ValueError):
        views.project_tree(source, source / "inside")
    destination = tmp_path / "view"
    destination.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        views.project_tree(source, destination)


@pytest.mark.parametrize(
    "name",
    [
        "golden.yaml",
        "golden.npz",
        "case.golden.bin",
        "expected_command_buffer.json",
        "expected_command_buffer_notes.txt",
        "expected_instruction_coverage.yaml",
        "capsule.weights.safetensors",
        "capsule.weights.safetensors.manifest.json",
        "hidden/capsule.yaml",
    ],
)
def test_private_answers_absent_even_at_canonical_view_path(tmp_path, name):
    source = tmp_path / "source"
    _write(source, name, "private answer")
    destination = tmp_path / "view"
    receipt = views.project_tree(source, destination)
    assert not (destination.resolve() / name).exists()
    assert not any(p.is_file() for p in destination.rglob("*"))
    assert receipt["files"][0]["view_path"] is None
    assert receipt["files"][0]["kind"] == "omitted"


def test_explicit_transform_replaces_source_and_rewrites_capsule_before_readonly_publication(tmp_path):
    source = tmp_path / "source"
    _write(source, "case/capsule.interface.mlir", "module { /* numerical source */ }")
    _write(source, "case/capsule.yaml", "interface_mlir: capsule.interface.mlir\nsource_reference: private\n")
    _write(source, "case/golden.yaml", "private")
    seen = []

    def transform(relative, raw):
        seen.append(relative)
        if relative.endswith(".mlir"):
            return "case/capsule.program.json", b'{"operation": "matmul"}\n'
        return relative, b"program_input: capsule.program.json\nsource_reference: private\n"

    destination = tmp_path / "view"
    receipt = views.project_tree(source, destination, file_transform=transform)
    assert "case/golden.yaml" not in seen
    assert not (destination / "case/capsule.interface.mlir").exists()
    assert json.loads((destination / "case/capsule.program.json").read_text())["operation"] == "matmul"
    assert yaml.safe_load((destination / "case/capsule.yaml").read_text()) == {"program_input": "capsule.program.json"}
    row = next(row for row in receipt["files"] if row["path"].endswith(".mlir"))
    assert row["view_path"] == "case/capsule.program.json"
    assert row["transformed"] and row["view_sha256"] != row["source_sha256"]


@pytest.mark.parametrize(
    "newname",
    [
        "../leak.program.json",
        "/absolute.program.json",
        "a/./b.program.json",
        "golden.program.json",
        "expected_command_buffer.json",
        "unknown.elf",
    ],
)
def test_transform_rejects_escaping_unreviewed_and_answer_paths(tmp_path, newname):
    source = tmp_path / "source"
    _write(source, "capsule.interface.mlir", "module {}")
    destination = tmp_path / "view"
    with pytest.raises(ValueError):
        views.project_tree(source, destination, file_transform=lambda path, raw: (newname, b"payload"))
    assert not destination.exists()


def test_transform_omissions_and_collisions(tmp_path):
    source = tmp_path / "source"
    _write(source, "a.mlir", "module {}")
    _write(source, "b.mlir", "module {}")
    receipt = views.project_tree(source, tmp_path / "omitted", file_transform=lambda path, raw: None)
    assert all(row["view_path"] is None and row["transformed"] for row in receipt["files"])
    with pytest.raises(ValueError, match="colliding"):
        views.project_tree(source, tmp_path / "collision", file_transform=lambda path, raw: ("same.program.json", raw))
    assert not (tmp_path / "collision").exists()


def _withheld():
    return {
        "author_profile": {"disclosure": {"software_spec": False}},
        "compiler_tool_profile": {"semantic_inputs": {"software_spec": {"path": "/nonexistent/private.yaml"}}},
    }


def test_withheld_software_never_reads_or_advertises_private_selection(tmp_path, monkeypatch):
    bundle = _withheld()
    original = copy.deepcopy(bundle)
    monkeypatch.setattr(software.BW, "snapshot_input_paths", lambda *a, **k: pytest.fail("private semantics was read"))
    record = software.stage(tmp_path, bundle, repo=tmp_path)
    assert record is None
    software.verify(record, tmp_path, bundle, repo=tmp_path)
    software.append_guidance(record, tmp_path, tmp_path)
    assert not (tmp_path / "TASK.md").exists()
    assert not (tmp_path / "public_declarations").exists()
    assert bundle == original


@pytest.mark.parametrize("stale", ["file", "symlink", "parent_symlink", "record"])
def test_withheld_software_rejects_existing_public_alias_or_record(tmp_path, stale):
    alias = tmp_path / "public_declarations/software-spec.yaml"
    if stale in {"file", "symlink"}:
        alias.parent.mkdir()
        if stale == "file":
            alias.write_text("private")
        else:
            alias.symlink_to(tmp_path / "missing")
    elif stale == "parent_symlink":
        alias.parent.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        if stale == "record":
            software.verify({"schema": "stale"}, tmp_path, _withheld(), repo=tmp_path)
        else:
            software.stage(tmp_path, _withheld(), repo=tmp_path)


@pytest.mark.parametrize("value", [None, "false", 0, [], {}])
def test_malformed_software_flag_fails_closed(value):
    with pytest.raises(ValueError):
        software.public_software_enabled({"author_profile": {"disclosure": {"software_spec": value}}})


def test_historical_absence_and_explicit_true_preserve_selection(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(software, "_selection", lambda *args: calls.append(args) or None)
    assert software.public_software_enabled({})
    assert software.public_software_enabled({"author_profile": {"disclosure": {}}})
    for bundle in [{}, {"author_profile": {"disclosure": {"software_spec": True}}}]:
        assert software.stage(tmp_path, bundle, repo=tmp_path) is None
    assert len(calls) == 2
