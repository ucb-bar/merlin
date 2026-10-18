"""Optional leaves use one authenticated query without dropping membership guards."""

from types import SimpleNamespace

import pytest

from merlin.common import bundle_snapshot
from merlin.targetgen import compiler_profile_admission as A
from merlin.targetgen import tool_registry
from merlin.targetgen.sandbox import bwrap


@pytest.fixture
def selected(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    sources = [repo / "merlin/A.py", repo / "merlin/B.py"]
    sources[0].parent.mkdir()
    for source in sources:
        source.write_text("original selected source")
    ws = tmp_path / "workspace"
    ws.mkdir()
    bundle = {"allowed": [{"path": str(s), "mode": "ro"} for s in sources], "host_inputs": [], "denied": []}
    bwrap.materialize_bundle_inputs(ws, bundle, repo=repo)
    monkeypatch.setattr(
        tool_registry,
        "spec",
        lambda _: SimpleNamespace(bundle_paths=("merlin/python/merlin/A.py", "merlin/python/merlin/B.py")),
    )
    owner = SimpleNamespace(
        repo=repo, workspace=ws, bundle=bundle, profile={"treatment_tools": ["isa_dialect_generation"]}
    )
    return owner, sources


def test_optional_leaves_keep_order_and_query_snapshot_once(selected, monkeypatch):
    owner, sources = selected
    original = bundle_snapshot.snapshot_input_paths
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(bundle_snapshot, "snapshot_input_paths", observed)
    assert list(A._optional_modules(owner)) == ["merlin.A", "merlin.B"]
    assert calls == [sources]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign", "mutation", "symlink"])
def test_optional_batch_refuses_changed_or_indirect_selection(selected, fault):
    owner, sources = selected
    frozen = bundle_snapshot.snapshot_path(
        bundle_snapshot.bundle_snapshot_root(owner.workspace), sources[1], owner.repo
    )
    if fault == "missing":
        owner.bundle["allowed"].pop()
    elif fault == "duplicate":
        owner.bundle["allowed"].append(dict(owner.bundle["allowed"][0]))
    elif fault == "foreign":
        owner.bundle["allowed"][1]["path"] = str(owner.repo / "foreign/merlin/B.py")
    elif fault == "mutation":
        frozen.chmod(0o600)
        frozen.write_text("changed selected payload")
    else:
        frozen.parent.chmod(0o700)
        frozen.unlink()
        frozen.symlink_to(sources[1])
    with pytest.raises((ValueError, RuntimeError)):
        list(A._optional_modules(owner))
