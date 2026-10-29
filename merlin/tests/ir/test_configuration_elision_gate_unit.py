"""SDK-free admission/authority controls; no mocked positive kernel completion."""

import copy
import json
import pickle
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.verify import configuration_elision_gate as gate
from merlin.verify.configuration_elision_intake import prepare


def forbidden(*args, **kwargs):
    pytest.fail("native execution or consumer reached in SDK-free refusal control")


@pytest.fixture(scope="module")
def candidate():
    base = repo_root() / "out/artifacts/verification/merlin-passes/loop-scope-20261028/NEXT-CONFIGURATION/F"
    folder = base / "corpus-v1/positives/multifield"
    return prepare(
        (folder / "stage0.mlir").read_text(),
        (folder / "stage1.mlir").read_text(),
        reference=repo_root().parent / "merlin-graph-proof-intake",
        facts=base / "fixture-input-v1/facts.json",
        software=base / "fixture-input-v1/software.yaml",
        target="configuration_fixture",
    )


def test_changed_child_cleanup_source_invalidates_request(candidate, monkeypatch):
    from merlin.verify.configuration_elision_intake import revalidate

    cleanup = (repo_root() / "src/merlin/common/owned_children.py").resolve()
    read_bytes = Path.read_bytes

    def changed(path):
        data = read_bytes(path)
        return data + b"\n" if path.resolve() == cleanup else data

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="source changed"):
        revalidate(candidate)


@pytest.mark.parametrize("kind", ["none", "saved", "query", "roots", "domain", "relation", "source", "final"])
def test_changed_requests_and_saved_status_refused_before_native(candidate, kind, tmp_path, monkeypatch):
    values = {
        "none": None,
        "saved": {"status": "checked"},
        "query": replace(candidate, query=candidate.query + "\n"),
        "roots": replace(candidate, roots=candidate.roots[:-1]),
        "domain": replace(candidate, domain="other"),
        "relation": replace(candidate, relation="possible-success"),
        "source": replace(candidate, source=""),
        "final": replace(candidate, final=""),
    }
    monkeypatch.setattr(gate, "_run", forbidden)
    result = gate.check(values[kind], lean=Path("/never-opened/lean"), destination=tmp_path / "new")
    assert result.status == "unavailable" and result.completion is None
    assert not (tmp_path / "new").exists()


def test_complete_current_source_closure_includes_actual_local_imports(candidate):
    import sys

    refs = {row["path"] for row in json.loads(candidate.sources_json)}
    for name, module in tuple(sys.modules.items()):
        if name.startswith("merlin.verify.configuration_elision_") or name in {
            "merlin",
            "merlin.verify",
            "merlin.common",
            "merlin.common.owned_process",
            "merlin.common.paths",
        }:
            assert str(Path(module.__file__).resolve()) in refs
    assert len(json.loads(candidate.proof_sources_json)) == 7


@pytest.mark.parametrize("kind", ["constructed", "copy", "deepcopy", "saved"])
def test_no_nonlive_completion_authorizes_consumer(candidate, kind):
    token = gate.Completion(candidate.identity, "/saved.json")
    value = {
        "constructed": token,
        "copy": copy.copy(token),
        "deepcopy": copy.deepcopy(token),
        "saved": {"status": "checked"},
    }[kind]
    with pytest.raises(gate.Unavailable):
        gate.deliver(value, candidate, forbidden)
    with pytest.raises(TypeError):
        pickle.dumps(token)


@pytest.mark.parametrize(
    "tail", ["", "Q.lean:9:1: error: unknown identifier x\n", "Q.lean:9:1: error(lean.X): anything\n"]
)
def test_closed_false_is_distinct_from_mixed_unknown(tail):
    stdout = "Q.lean:1:1: error: Tactic `decide` proved that the proposition\n False\nis false\n" + tail
    error = gate.Unavailable if tail else gate.NotEstablished
    with pytest.raises(error):
        gate.require_query_success(subprocess.CompletedProcess([], 1, stdout, ""))
    assert not gate._LIVE


@pytest.mark.parametrize("kind", ["timeout", "signal"])
def test_first_process_failure_is_retained(tmp_path, monkeypatch, kind):
    def run(*args, **kwargs):
        if kind == "timeout":
            raise subprocess.TimeoutExpired(args[0], 1, output=b"first stdout", stderr=b"first stderr")
        return subprocess.CompletedProcess(args[0], -11, "first stdout", "first stderr")

    monkeypatch.setattr(gate, "run_owned", run)
    with pytest.raises(gate.Unavailable):
        gate._run(["/synthetic-not-executed"], tmp_path, {}, tmp_path, "query", 1)
    assert (tmp_path / "query.stdout").read_text() == "first stdout"
    assert (tmp_path / "query.stderr").read_text() == "first stderr"


def test_saved_green_json_is_not_replay_authority(tmp_path, monkeypatch):
    path = tmp_path / "saved.json"
    path.write_text('{"status":"checked"}')
    monkeypatch.setattr(gate, "_run", forbidden)
    assert gate.replay(path, lean=Path("/never-opened/lean"), destination=tmp_path / "new").completion is None


def test_project_compile_uses_canonical_source_root(candidate, tmp_path, monkeypatch):
    tool = tmp_path / "never-executed-tool"
    tool.write_text("SDK-free path-binding test data")
    calls = []

    def inspect_only(command, cwd, env, directory, label, timeout):
        assert label == "import-0"
        assert Path(command[-1]).is_relative_to(cwd)
        assert cwd == repo_root() / "src/merlin/verify/lean"
        assert str(directory / "lib") == env["LEAN_PATH"]
        calls.append(command)
        raise gate.Unavailable("SDK-free stop after checking first project command")

    monkeypatch.setattr(gate, "_run", inspect_only)
    result = gate.check(candidate, lean=tool, destination=tmp_path / "new")
    assert result.status == "unavailable" and result.completion is None and len(calls) == 1
    assert not gate._LIVE
