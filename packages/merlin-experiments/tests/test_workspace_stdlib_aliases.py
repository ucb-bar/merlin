"""Friendly grant names must not replace stdlib modules at cold Python startup."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase1 import workspace_transport as W
from merlin_experiments.phase1.context import InvocationContext


@pytest.fixture
def context(tmp_path, monkeypatch):
    repo = tmp_path / "repository"
    repo.mkdir()
    corpus = repo / "corpus"
    corpus.mkdir()
    descriptor = repo / "descriptor.yaml"
    descriptor.write_text(f"target: fixture\ncapsule_corpus: {corpus}\n")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    return InvocationContext(
        repo, descriptor, tmp_path, "fixture", tmp_path / "runs", tmp_path / "reports", tmp_path / "bundles", ()
    )


@pytest.mark.parametrize("kind", ["file", "directory", "bare_file"])
def test_populated_workspace_cold_python_preserves_stdlib_types(context, tmp_path, kind):
    relative = "types.py" if kind == "bare_file" else "curated/types.py" if kind == "file" else "curated/types"
    source = context.repo / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    if kind == "directory":
        source.mkdir()
        payload = source / "__init__.py"
    else:
        payload = source
    content = b'raise RuntimeError("curated source is not the stdlib types module")\n'
    payload.write_bytes(content)
    workspace = tmp_path / "workspace"
    bundle = {"allowed": [{"path": relative, "mode": "ro"}]}
    try:
        W.assemble(bundle, workspace, "bwrap", context=context)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import types,json; print(json.dumps({'module':types.__file__,"
                "'value':types.SimpleNamespace(value=7).value}))",
            ],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        observed = json.loads(result.stdout)
        assert observed["value"] == 7
        assert not Path(observed["module"]).is_relative_to(workspace)
        assert not (workspace / "types.py").exists() and not (workspace / "types").exists()
        aliases = [path for path in workspace.iterdir() if path.is_symlink()]
        assert len(aliases) == 1 and aliases[0].resolve() == source
        retained = aliases[0] / "__init__.py" if source.is_dir() else aliases[0]
        assert retained.read_bytes() == content
        assert payload.read_bytes() == content
        assert W.BW.snapshot_input_paths(workspace, bundle, [source], repo=context.repo)[0] != source
    finally:
        W.BW.remove_bundle_snapshot(workspace)


def test_noncolliding_and_explicit_aliases_keep_their_original_names(context, tmp_path):
    sources = []
    for relative in ("curated/helper.py", "curated/types.py", "other/helper.py"):
        path = context.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
        sources.append(path)
    workspace = tmp_path / "workspace"
    bundle = {
        "allowed": [
            {"path": "curated/helper.py"},
            {"path": "curated/types.py", "as": "explicit/types.py"},
            {"path": "other/helper.py"},
        ]
    }
    try:
        W.assemble(bundle, workspace, "bwrap", context=context)
        expected = {
            "helper.py": sources[0],
            "explicit/types.py": sources[1],
            "other_helper.py": sources[2],
        }
        for name, source in expected.items():
            assert (workspace / name).is_symlink()
            assert (workspace / name).resolve() == source
            assert (workspace / name).read_bytes() == source.read_bytes()
    finally:
        W.BW.remove_bundle_snapshot(workspace)


def test_explicit_flat_alias_remains_the_selected_contract_name(context, tmp_path):
    source = context.repo / "api.py"
    source.write_text("selected explicit alias\n")
    workspace = tmp_path / "workspace"
    bundle = {"allowed": [{"path": "api.py", "as": "types.py"}]}
    try:
        W.assemble(bundle, workspace, "bwrap", context=context)
        assert (workspace / "types.py").is_symlink()
        assert (workspace / "types.py").resolve() == source
    finally:
        W.BW.remove_bundle_snapshot(workspace)
