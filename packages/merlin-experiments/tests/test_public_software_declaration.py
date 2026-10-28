"""Selected public data remains readable without unmasking backend implementation."""

import hashlib
import importlib
import shutil
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase1 import public_software_declaration as P

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


@pytest.fixture
def selected(tmp_path, monkeypatch):
    surfaces = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")

    repo, ws = tmp_path / "repo", tmp_path / "run/workspace"
    backend = repo / "backend"
    backend.mkdir(parents=True)
    ws.mkdir(parents=True)
    template = Path(__file__).resolve().parents[3] / "examples/gemmini/target/software-spec.yaml"
    source = backend / "software.yaml"
    source.write_bytes(template.read_bytes())
    (backend / "implementation.py").write_text("PRIVATE_IMPLEMENTATION = True\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    bundle = dict(
        allowed=[dict(path=str(source), mode="ro", note="admitted software semantics")],
        denied=[],
        host_inputs=[],
        compiler_tool_profile=dict(semantic_inputs=dict(software_spec=dict(path=str(source), sha256=digest))),
    )
    monkeypatch.setattr(surfaces, "_support_package_dirs", lambda: [backend])
    BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    yield repo, ws, bundle, source
    BW.remove_bundle_snapshot(ws)


def test_exact_frozen_data_alias_is_independent_and_custody_is_reopened(selected):
    repo, ws, bundle, source = selected
    original = source.read_bytes()
    record = P.stage(ws, bundle, repo=repo)
    alias = Path(record["path"])
    assert alias.read_bytes() == original
    assert alias.stat().st_ino != source.stat().st_ino
    assert not alias.stat().st_mode & 0o222
    source.write_text("later private source change cannot replace frozen public bytes")
    P.verify(record, ws, bundle, repo=repo)
    (ws / "TASK.md").write_text("Original task\n")
    run = ws.parent / "records"
    run.mkdir()
    P.append_guidance(record, ws, run)
    assert record["path"] in (ws / "TASK.md").read_text()
    assert (run / "TASK.md").read_bytes() == (ws / "TASK.md").read_bytes()


@pytest.mark.parametrize("damage", ["alias", "writable", "symlink", "record", "selected_source"])
def test_changed_alias_or_selected_input_refuses(selected, damage):
    repo, ws, bundle, _ = selected
    record = P.stage(ws, bundle, repo=repo)
    alias = Path(record["path"])
    if damage == "alias":
        alias.chmod(0o600)
        alias.write_text("changed")
        alias.chmod(0o400)
    elif damage == "writable":
        alias.chmod(0o600)
    elif damage == "symlink":
        alias.unlink()
        alias.symlink_to(ws / "absent")
    elif damage == "record":
        record["sha256"] = "0" * 64
    else:
        [frozen] = BW.snapshot_input_paths(ws, bundle, [Path(record["source"]["path"])], repo=repo)
        frozen.chmod(0o600)
        frozen.write_text("changed")
        frozen.chmod(0o400)
    with pytest.raises((ValueError, RuntimeError), match="changed|differs"):
        P.verify(record, ws, bundle, repo=repo)


def test_unselected_or_ambiguous_source_cannot_create_alias(selected):
    repo, ws, bundle, _ = selected
    assert P.stage(ws, {}, repo=repo) is None
    P.verify(None, ws, bundle, repo=repo)
    bundle["allowed"].append(dict(bundle["allowed"][0]))
    with pytest.raises(ValueError, match="exact admitted"):
        P.stage(ws, bundle, repo=repo)
    assert not (ws / "public_declarations").exists()


def test_real_bwrap_keeps_backend_hidden_and_exact_public_alias_readable(selected):
    binary = shutil.which("bwrap")
    if binary is None:
        pytest.skip("requires local bwrap")
    repo, ws, bundle, source = selected
    record = P.stage(ws, bundle, repo=repo)
    argv = BW.base_argv(ws, bundle, repo=repo, include_claude_home=False)
    frozen = BW._frozen_support_surfaces(ws, bundle, [], repo)
    argv = BW.apply_answer_masks(argv, [AnswerSurface("backend", source.parent, "dir", "backend"), *frozen])
    code = (
        "from pathlib import Path; import hashlib,json; "
        f"p=Path({record['path']!r}); assert hashlib.sha256(p.read_bytes()).hexdigest()=={record['sha256']!r}; "
        f"assert not Path({str(source.parent / 'implementation.py')!r}).is_file(); "
        f"p=Path({str(source)!r}); assert not p.exists() or p.stat().st_size==0; print('EXACT_PUBLIC_DATA')"
    )
    result = subprocess.run(
        [*argv, "/usr/bin/python3", "-I", "-B", "-c", code], capture_output=True, text=True, timeout=15
    )
    if result.returncode and "Operation not permitted" in result.stderr:
        pytest.skip("kernel user namespaces unavailable")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "EXACT_PUBLIC_DATA"


def _guidance(selected_fixture, monkeypatch, tmp_path, *, joint):
    from merlin_experiments.phase1 import joint_control

    repo, ws, bundle, _ = selected_fixture
    record = P.stage(ws, bundle, repo=repo)
    (ws / "TASK.md").write_text("Original task\n")
    run = ws.parent / "records"
    run.mkdir(exist_ok=True)
    monkeypatch.setattr(joint_control, "selected", lambda: {"policy": {}} if joint else None)
    P.append_guidance(record, ws, run)
    text = (ws / "TASK.md").read_text()
    assert (run / "TASK.md").read_bytes() == text.encode()
    return record, text


def test_ordinary_run_does_not_name_the_joint_only_scheduling_tool(selected, monkeypatch, tmp_path):
    record, text = _guidance(selected, monkeypatch, tmp_path, joint=False)
    assert "read_scheduling_contracts" not in text
    assert f"Use this readable data leaf; the original backend tree remains withheld." in text
    assert record["path"] in text


def test_joint_task_text_is_byte_identical_to_the_historical_text(selected, monkeypatch, tmp_path):
    record, text = _guidance(selected, monkeypatch, tmp_path, joint=True)
    historical = (
        "Original task\n"
        "\n## Selected public software declaration\n\n"
        + f"Read `{record['path']}` for the original selected software semantics "
        + f"(SHA-256 `{record['sha256']}`). Use this readable data leaf with "
        + "`read_scheduling_contracts`; the original backend tree remains withheld. "
        + "This identical-byte alias grants no additional implementation or qualification.\n"
    )
    assert text == historical
