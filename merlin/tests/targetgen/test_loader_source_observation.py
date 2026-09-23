"""Observe loader imports without importing frameworks or running a capture."""

import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from merlin.targetgen import _m2m_capture_worker as worker
from merlin.targetgen import capture_cache


def test_worker_observes_top_level_and_input_builder_imports(tmp_path, monkeypatch):
    top = tmp_path / "synthetic_top_level_dependency.py"
    call = tmp_path / "synthetic_input_dependency.py"
    top.write_text("value = 1\n")
    call.write_text("value = 2\n")
    loader = tmp_path / "loader.py"
    loader.write_text(
        "import synthetic_top_level_dependency\n"
        "def get_model_and_inputs():\n"
        "    import synthetic_input_dependency\n"
        "    return None, ()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in (top.stem, call.stem):
        monkeypatch.delitem(sys.modules, name, raising=False)
    # Keep imports local to this test, including entries added by executing the loader.
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    torch = ModuleType("torch")
    torch.manual_seed = lambda seed: None
    coverage = ModuleType("m2m.coverage")
    coverage.opaque_report = lambda *args: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "m2m", ModuleType("m2m"))
    monkeypatch.setitem(sys.modules, "m2m.coverage", coverage)
    collect = worker._loader_dependency_sources

    class Observed(Exception):
        pass

    def observe(before, path):
        records = {row["module"]: row for row in collect(before, path)}
        for source in (top, call):
            assert records[source.stem] == {
                "module": source.stem,
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        raise Observed  # Stop the real worker before any model conversion or tensor work.

    monkeypatch.setattr(worker, "_loader_dependency_sources", observe)
    with pytest.raises(Observed):
        worker.main(["--loader", str(loader), "--dtype", "f32", "--out", str(tmp_path / "capture")])


@pytest.mark.parametrize("change", ["none", "before", "during", "missing"])
def test_loader_dependency_staging_checks_observed_bytes(tmp_path, monkeypatch, change):
    import shutil

    from merlin.capture import bundle
    from merlin.targetgen import capsule_source

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    source = upstream / "dependency.py"
    source.write_text("original source\n")
    (upstream / "LICENSE").write_text("synthetic fixture license\n")
    monkeypatch.setattr(bundle, "capture_config", lambda _: {"upstream": str(upstream)})
    meta = {
        "loader_dependency_sources": [
            {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ]
    }
    if change == "before":
        source.write_text("changed source\n")
    elif change == "missing":
        source.unlink()
    elif change == "during":
        copyfile = shutil.copyfile

        def changed_copy(src, dst):
            if src == source:
                source.write_text("changed during copy\n")
            return copyfile(src, dst)

        monkeypatch.setattr(shutil, "copyfile", changed_copy)
    destination = tmp_path / "capsule"
    if change == "none":
        assert capsule_source.freeze_model_loader_dependencies("fixture", destination, meta) == "capsule.loader_deps"
        assert (destination / "capsule.loader_deps/dependency.py").read_bytes() == source.read_bytes()
    else:
        with pytest.raises(capsule_source.M2MUnavailable, match="loader dependency changed"):
            capsule_source.freeze_model_loader_dependencies("fixture", destination, meta)


@pytest.mark.parametrize("change", ["none", "modified", "missing", "legacy", "malformed"])
def test_cache_rechecks_observed_loader_sources_before_reuse(tmp_path, monkeypatch, change):
    from merlin.targetgen import capsule_source as source

    dependency = tmp_path / "dependency.py"
    dependency.write_text("observed bytes\n")
    records = [{"path": str(dependency), "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest()}]
    slot = tmp_path / "slot"
    attempt = slot / "attempt-original"
    attempt.mkdir(parents=True)
    weights_manifest = attempt / "weights.manifest.json"
    weights_manifest.write_text("{}")
    meta = {
        "ok": True,
        "opaque": 0,
        "loader_provenance_status": "synthetic",
        "capture_abi_version": source._MODEL_CAPTURE_ABI_VERSION,
        "output_abi": [],
        "weights_manifest": str(weights_manifest),
        "loader_dependency_sources": records,
    }
    if change == "legacy":
        del meta["loader_dependency_sources"]
    elif change == "malformed":
        meta["loader_dependency_sources"] = [None]
    (attempt / "meta.json").write_text(json.dumps(meta))
    (attempt / "linalg.mlir").write_text("cached module")
    (attempt / "inputs.json").write_text("[]")
    (attempt / "golden.json").write_text("[]")
    capture_cache.commit(slot, attempt)
    if change == "modified":
        dependency.write_text("changed bytes\n")
    elif change == "missing":
        dependency.unlink()

    class FreshCaptureRequested(Exception):
        pass

    def worker(*args, **kwargs):
        raise FreshCaptureRequested

    monkeypatch.setattr(source.subprocess, "run", worker)
    capture = source.PytorchRefSource(m2m_dir=tmp_path, python=tmp_path / "python")
    kwargs = dict(
        workdir=tmp_path / "work",
        src="loader",
        scheme=None,
        declared_env={},
        interpreter=tmp_path / "python",
        recipe=None,
        recipe_sha256="",
        slot=slot,
    )
    if change == "none":
        artifact, selected = capture._run_capture(tmp_path / "loader.py", "model", "f32", **kwargs)
        assert artifact.linalg_mlir == "cached module" and selected == attempt
    else:
        with pytest.raises(FreshCaptureRequested):
            capture._run_capture(tmp_path / "loader.py", "model", "f32", **kwargs)
        assert capture_cache.committed_attempt(slot) is None
        assert (attempt / "linalg.mlir").read_text() == "cached module"  # historical bytes retained


def test_dependency_changes_during_capture_refuse_parent_publication(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_source as source

    dependency = tmp_path / "dependency.py"
    dependency.write_text("observed bytes\n")
    meta = {
        "loader_dependency_sources": [
            {
                "path": str(dependency),
                "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest(),
            }
        ]
    }
    capture = source.PytorchRefSource(m2m_dir=tmp_path, python=tmp_path / "python")
    slot = tmp_path / "slot"
    monkeypatch.setattr(capture, "_cache_slot", lambda *args: slot)

    def changed_capture(*args, **kwargs):
        dependency.write_text("changed after observation\n")
        return SimpleNamespace(meta=meta), slot / "attempt-test"

    monkeypatch.setattr(capture, "_run_capture", changed_capture)
    monkeypatch.setattr(capture_cache, "commit", lambda *args: pytest.fail("published changed dependencies"))
    with pytest.raises(source.M2MUnavailable, match="observed loader dependencies changed"):
        capture._run(tmp_path / "loader.py", "model", "f32", src="loader", workdir=tmp_path / "work")
