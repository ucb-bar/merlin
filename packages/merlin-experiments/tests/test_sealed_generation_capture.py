"""Generation-time captures go through the admitted sealed runner, sharing one stored runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from merlin_experiments.capture_execution import runtime_store, sealed_m2m
from merlin_experiments.phase0 import sealed_generation as SG

from merlin.targetgen.capsule_source import M2MUnavailable


def _runtime_plan(tmp_path):
    venv = tmp_path / "venv"
    (venv / "lib/site").mkdir(parents=True)
    (venv / "lib/site/mod.py").write_text("VALUE = 1\n")
    (venv / "pyvenv.cfg").write_text("home = x\n")
    base = tmp_path / "base"
    (base / "bin").mkdir(parents=True)
    (base / "bin/python3.12").write_bytes(b"\x7fELF fixture")
    (base / "bin").chmod(0o755)
    library = tmp_path / "libfixture.so"
    library.write_bytes(b"library bytes")
    return {
        "venv": str(venv),
        "base": str(base),
        "system_libs": [str(library)],
        "selected_trees": {
            "venv": sealed_m2m._source_tree(venv, skip_lib64=True),
            "base": sealed_m2m._source_tree(base),
        },
    }


def test_the_runtime_is_stored_once_and_hard_linked_into_each_private_root(tmp_path, monkeypatch):
    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    plan = _runtime_plan(tmp_path)
    first, second = tmp_path / "run1/guest-root", tmp_path / "run2/guest-root"
    for root in (first, second):
        root.mkdir(parents=True)
        runtime_store.link_runtime(plan, root)
    module = Path("opt/capture-venv/lib/site/mod.py")
    assert os.stat(first / module).st_ino == os.stat(second / module).st_ino
    assert sealed_m2m._snapshot_tree(first / "opt/capture-venv") == plan["selected_trees"]["venv"]
    base = first / Path(plan["base"]).relative_to("/")
    assert sealed_m2m._snapshot_tree(base) == plan["selected_trees"]["base"]
    assert (first / Path(plan["system_libs"][0]).relative_to("/")).read_bytes() == b"library bytes"
    # One stored copy per runtime identity, published only once complete.
    assert len([path for path in (tmp_path / "store").iterdir() if path.is_dir()]) == 1


def test_a_store_that_differs_from_the_selection_is_never_published(tmp_path, monkeypatch):
    monkeypatch.setenv(runtime_store.STORE_ENV, str(tmp_path / "store"))
    plan = _runtime_plan(tmp_path)
    plan["selected_trees"]["venv"] = {"members": 0, "bytes": 0, "sha256": "0" * 64}
    with pytest.raises(sealed_m2m.SealedM2MError, match="differs"):
        runtime_store.store_entry(plan)
    assert not [path for path in (tmp_path / "store").iterdir() if path.is_dir()]


def test_worker_options_are_bound_into_the_command_and_absent_ones_keep_it():
    output = Path("/capture-out")
    plain = sealed_m2m._command_v2(output, dtype="int8", recipe=True)
    assert sealed_m2m._command_v2(output, dtype="int8", recipe=True, options=None) == plain
    tolerant = sealed_m2m._command_v2(output, dtype="int8", recipe=True, options={"agreement_tolerance": [0.5, 0.25]})
    assert "'--agreement-atol', '0.5', '--agreement-rtol', '0.25'" in tolerant[-1]
    assert "--dtype', 'f32'" in sealed_m2m._command_v2(output, dtype="f32", recipe=False)[-1]
    staged = sealed_m2m._command_v2(output, dtype="fp32", recipe=False, options={"stage_fp32": True})
    assert "'--stage-fp32'" in staged[-1]
    assert staged != plain
    assert (
        "'--stage-fp32'"
        not in sealed_m2m._command_v2(output, dtype="fp32", recipe=False, options={"stage_fp32": False})[-1]
    )
    for bad in ({"scheme": "x"}, {"agreement_tolerance": [1.0]}, {"agreement_tolerance": [-1.0, 0.0]}):
        with pytest.raises(sealed_m2m.SealedM2MError):
            sealed_m2m._command_v2(output, dtype="int8", recipe=True, options=bad)
    for bad in (1, "true", None):
        with pytest.raises(sealed_m2m.SealedM2MError):
            sealed_m2m._command_v2(output, dtype="fp32", recipe=False, options={"stage_fp32": bad})
    assert (
        "'--stage-fp32'" in sealed_m2m._command_v2(output, dtype="int8", recipe=True, options={"stage_fp32": True})[-1]
    )
    with pytest.raises(sealed_m2m.SealedM2MError):
        sealed_m2m._command_v2(output, dtype="int8", recipe=False, options={"stage_fp32": True})


def _source(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    return SG.SealedCaptureSource({"m2m_root": str(tmp_path), "venv": str(venv), "runs_root": str(tmp_path / "runs")})


def test_requests_the_seal_cannot_express_fail_closed(tmp_path):
    source = _source(tmp_path)
    base = {
        "loader_py": tmp_path / "loader.py",
        "op": "op",
        "dtype": "f32",
        "workdir": tmp_path,
        "interpreter": source.python,
    }
    for extra, match in (
        ({"scheme": "w8a8"}, "scheme"),
        ({"already_quantized": True}, "scheme"),
        ({"declared_env": {"DATA": "x"}}, "environment"),
        ({"stage_fp32": "true"}, "stage_fp32"),
        ({"stage_fp32": True, "dtype": "int8"}, "FP32 staging"),
        ({"interpreter": tmp_path / "other/python"}, "interpreter"),
    ):
        with pytest.raises(M2MUnavailable, match=match):
            source._launch_worker([], env={}, **{**base, **extra})
    assert source._cache_slot("op") is None


@pytest.mark.parametrize("timeout", [None, 900])
def test_a_sealed_capture_is_copied_with_its_attestation(tmp_path, monkeypatch, timeout):
    from merlin_experiments.phase0 import capture_execution_attestation as attestation_module
    from merlin_experiments.phase0 import capture_selection

    source = _source(tmp_path)
    if timeout is not None:
        source.config["execution_timeout_seconds"] = timeout
    loader = tmp_path / "loader.py"
    loader.write_text("def get_model_and_inputs():\n    pass\n")
    calls = {}

    def fake_select(**kwargs):
        calls["select"] = kwargs
        capture = kwargs["run_dir"] / "capture"
        capture.mkdir(parents=True)
        for name in ("linalg.mlir", "model.mlir"):
            (capture / name).write_text("module {}\n")
        (capture / "meta.json").write_text(json.dumps({"ok": True, "frontend_trace": {"path": "frontend-trace.json"}}))
        (capture / "frontend-trace.json").write_text("{}")
        (capture / "integer-reference.json").write_text("{}")
        (capture / "capture_receipt.json").write_text("{}")
        selection = kwargs["output_dir"] / "capture-selection.json"
        selection.parent.mkdir()
        selection.write_text("{}")
        return {"path": str(selection), "sha256": "a" * 64}

    monkeypatch.setattr(capture_selection, "select", fake_select)
    monkeypatch.setattr(capture_selection, "issue", lambda *a, **k: None)
    monkeypatch.setattr(capture_selection, "verify", lambda *a, **k: {"replay": True})
    monkeypatch.setattr(attestation_module, "attest_sealed_m2m", lambda *a, **k: {"issuer": "sealed"})
    workdir = tmp_path / "work"
    workdir.mkdir()
    result = source._launch_worker(
        [],
        env={},
        loader_py=loader,
        op="mul",
        dtype="f32",
        workdir=workdir,
        interpreter=source.python,
        agreement_tolerance=(0.1, 0.2),
        stage_fp32=True,
    )
    assert result.returncode == 0
    assert json.loads((workdir / "meta.json").read_text())["capture_execution_attestation"] == {"issuer": "sealed"}
    assert source.attestations == [{"issuer": "sealed"}]
    meta = json.loads((workdir / "meta.json").read_text())
    # Bundle-relative members resolve in the copy; bundle-only files stay with the sealed run.
    assert meta["frontend_trace"]["path"] == str(workdir / "frontend-trace.json")
    assert (workdir / "integer-reference.json").is_file() and not (workdir / "capture_receipt.json").exists()
    assert calls["select"]["worker_options"] == {
        "agreement_tolerance": [0.1, 0.2],
        "stage_fp32": True,
    }
    assert (Path(calls["select"]["workload_root"]) / "loader.py").read_text() == loader.read_text()
    # Every generation capture is selected with the run's frozen timeout; none keeps the historical 120 s.
    assert calls["select"]["execution_timeout_seconds"] == timeout


@pytest.mark.parametrize("timeout", [119, 14_401, "900", True])
def test_an_unbounded_generation_capture_timeout_is_refused(monkeypatch, timeout):
    base = {"m2m_root": "/m2m", "venv": "/venv", "runs_root": "/runs"}
    monkeypatch.setenv(SG.CONFIG_ENV, json.dumps({**base, "execution_timeout_seconds": timeout}))
    with pytest.raises(ValueError, match="between 120 and 14400"):
        SG.configured()
    monkeypatch.setenv(SG.CONFIG_ENV, json.dumps({**base, "execution_timeout_seconds": 3600}))
    assert SG.configured()["execution_timeout_seconds"] == 3600


def test_verified_generation_requires_an_attested_capture(monkeypatch):
    from merlin_experiments.phase0 import capture_execution_attestation as attestation_module

    probe = {"pytorch_ref": {"op": "mul"}, "frontend_trace": {"capture_mlir_sha256": "c" * 64}}
    assert "no sealed-runner attestation" in SG.verified_capture_failure(probe)
    assert SG.verified_capture_failure({"materialized_capture": {"capture_sha256": "c" * 64}}) is None
    assert SG.verified_capture_failure({"interface_mlir": "capsule.interface.mlir"}) is None
    monkeypatch.setattr(attestation_module, "require_verified_execution", lambda document: None)
    attested = {**probe, "capture_execution_attestations": [{"capture": {"model_sha256": "c" * 64}}]}
    assert SG.verified_capture_failure(attested) is None
    other = {**probe, "capture_execution_attestations": [{"capture": {"model_sha256": "d" * 64}}]}
    assert "not the attested capture" in SG.verified_capture_failure(other)


def test_verified_frontend_generation_binds_one_sealed_source_before_writing(tmp_path, monkeypatch):
    monkeypatch.delenv(SG.CONFIG_ENV, raising=False)
    entries = [{"kind": "model", "name": "derived"}, {"source": "pytorch", "name": "operation"}]
    with pytest.raises(ValueError, match="selected sealed Model2MLIR runtime before capsule writes"):
        SG.bind_source(entries, verified=True)
    source = _source(tmp_path)
    monkeypatch.setattr(SG, "capture_source", lambda: source)
    with pytest.raises(ValueError, match="runtime that is unavailable"):
        SG.bind_source(entries, verified=True)
    monkeypatch.setattr(source, "available", lambda: True)
    assert SG.bind_source(entries, verified=True) is source
    assert SG.bind_source([{"kind": "model", "materialized_capture": {"receipt": "selected"}}], verified=True) is None


def test_a_read_only_package_is_staged_writable_with_identical_bytes(tmp_path):
    package, schemas = tmp_path / "frozen/src/merlin", tmp_path / "frozen/merlin/schemas"
    (package / "targetgen").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "targetgen/_m2m_capture_worker.py").write_text("print('worker')\n")
    schemas.mkdir(parents=True)
    (schemas / "registry.yaml").write_text("formats: {}\n")
    for root in (package, schemas):
        for path in sorted([root, *root.rglob("*")], key=lambda item: len(item.parts), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
    source = _source(tmp_path)
    staged_package, staged_schemas = source._staged_merlin(package, schemas)
    assert (staged_package / "targetgen/_m2m_capture_worker.py").read_text() == "print('worker')\n"
    assert os.access(staged_package, os.W_OK) and os.access(staged_schemas / "registry.yaml", os.W_OK)
    assert staged_schemas.parent.parent == staged_package.parent.parent
    for root in (package, schemas):
        for path in [root, *root.rglob("*")]:
            path.chmod(0o755 if path.is_dir() else 0o644)


def test_a_selected_temporary_root_keeps_capture_work_off_the_host_tmp(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", None)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    SG.SealedCaptureSource(
        {
            "m2m_root": str(tmp_path),
            "venv": str(venv),
            "runs_root": str(tmp_path / "runs"),
            "tmp_root": str(tmp_path / "t"),
        }
    )
    assert Path(tempfile.mkdtemp()).parent == tmp_path / "t"
