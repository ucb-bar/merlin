"""The chip examples' single-operator helpers (examples/lib/chip_flow.py), without native work."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def _load(name: str):
    path = repo_root() / "examples" / "lib" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"examples_lib_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_builtin_operator_has_its_loader():
    flow = _load("chip_flow")
    for op in flow.OPS:
        assert (flow.WORKLOADS / op / "loader.py").is_file(), op


def test_an_operator_shape_must_match_its_loader(monkeypatch, capsys):
    flow = _load("chip_flow")
    monkeypatch.setattr(flow, "capture", lambda *a, **k: pytest.fail("captured despite a bad shape"))
    with pytest.raises(SystemExit, match="takes 3 dims"):
        flow.main(["op", "gemm", "64", "64", "--board", "b"])
    with pytest.raises(SystemExit):
        flow.main(["op", "gemm", "--board", "b", "--matrix-unit", "u"])  # matrix flags come together
    assert "given together" in capsys.readouterr().err


def _matrix_result(tmp_path):
    return {
        "binary": str(tmp_path / "ship.elf"),
        "twin_binary": str(tmp_path / "twin.elf"),
        "matrix": {"spike_harts": 2, "ship_ram_bytes": 1 << 20, "twin_ram_bytes": 1 << 20},
    }


def _args():
    from types import SimpleNamespace

    return SimpleNamespace(matrix_support_target="chip_a", board="chip_board")


def test_the_shipped_image_leg_is_skipped_when_the_target_declares_no_model(tmp_path, monkeypatch):
    from merlin.runtime.backends import zephyr_model as zm
    from merlin.targetgen import spike_extension as SE

    flow = _load("chip_flow")
    monkeypatch.setattr(SE, "declared_extension", lambda target: None)
    monkeypatch.setattr(zm, "run_on_spike", lambda *a, **k: pytest.fail("ran without a declared model"))
    got = flow.matrix_on_spike("op_x", _args(), _matrix_result(tmp_path))
    assert got["status"] == "skipped" and "runner.spike_extension" in got["reason"]


def test_the_shipped_image_leg_fails_closed_on_a_declared_model_it_cannot_load(tmp_path, monkeypatch):
    from merlin.runtime.backends import zephyr_model as zm
    from merlin.targetgen import spike_extension as SE

    flow = _load("chip_flow")
    monkeypatch.setattr(SE, "declared_extension", lambda target: {"extension_name": "m"})

    def refuse(target, **kw):
        raise SE.SpikeExtensionError("digest mismatch")

    monkeypatch.setattr(SE, "resolve", refuse)
    monkeypatch.setattr(zm, "run_on_spike", lambda *a, **k: pytest.fail("ran on an unverified model"))
    got = flow.matrix_on_spike("op_x", _args(), _matrix_result(tmp_path))
    assert got["status"] == "error" and "digest mismatch" in got["error"]


def test_the_shipped_image_runs_on_the_declared_model_and_must_match_the_twin(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from merlin.runtime.backends import zephyr_model as zm
    from merlin.targetgen import spike_extension as SE

    flow = _load("chip_flow")
    ext = SE.SpikeExtension(
        target="chip_a",
        extension_name="unit_model",
        extlib=tmp_path / "libm.so",
        library_dir=tmp_path,
        sha256="0" * 64,
        declared=True,
    )
    monkeypatch.setattr(SE, "declared_extension", lambda target: {"extension_name": "unit_model"})
    monkeypatch.setattr(SE, "resolve", lambda target, **kw: ext)
    monkeypatch.setattr("merlin.runtime.boards.board", lambda name: SimpleNamespace(vlen=512, dram_base=0x80000000))
    monkeypatch.setattr(flow, "_references", lambda name: {"w8a8": np.ones(4)})
    monkeypatch.setattr(zm, "_gate", lambda got, refs: {"ok": True, "w8a8_cos": 1.0})
    calls = []
    outputs = {"ship.elf": np.arange(4.0), "twin.elf": np.arange(4.0)}

    def fake_spike(elf, **kw):
        calls.append((Path(elf).name, kw.get("extension")))
        return {"prefix": outputs[Path(elf).name], "metrics": {"cycles": 7}, "out_hash": {"value": 1}}

    monkeypatch.setattr(zm, "run_on_spike", fake_spike)
    got = flow.matrix_on_spike("op_x", _args(), _matrix_result(tmp_path))
    assert calls == [("ship.elf", ext), ("twin.elf", None)]  # the model loads for the shipped image only
    assert got["status"] == "verified" and got["bit_exact_vs_twin"]
    assert got["extension"]["sha256"] == "0" * 64

    outputs["twin.elf"] = np.arange(4.0) + 1  # one wrong element anywhere is a mismatch, cos notwithstanding
    got = flow.matrix_on_spike("op_x", _args(), _matrix_result(tmp_path))
    assert got["status"] == "run_mismatch" and not got["bit_exact_vs_twin"]


def test_the_shipped_image_leg_merlin_compile_already_ran_is_reported_not_rerun(tmp_path, monkeypatch):
    from merlin.runtime.backends import zephyr_model as zm

    flow = _load("chip_flow")
    monkeypatch.setattr(zm, "run_on_spike", lambda *a, **k: pytest.fail("simulated the same images twice"))
    result = _matrix_result(tmp_path)
    result["matrix"]["shipped_on_simulator"] = {
        "via": "spike_extension", "status": "verified", "elf": str(tmp_path / "ship.elf"), "spike_cycles": 5,
        "extension": {"name": "m", "extlib": "/m.so", "sha256": "0" * 64}, "bit_exact_vs_twin": True,
        "verify": {"gate_ok": True, "w8a8_cos": 1.0},
    }  # fmt: skip
    got = flow.matrix_on_spike("op_x", _args(), result)
    assert got["status"] == "verified" and got["bit_exact_vs_twin"] and got["extension"]["name"] == "m"


def test_the_twin_is_not_simulated_again_when_its_answer_was_recorded(tmp_path, monkeypatch):
    """merlin-compile records the twin's output identity; comparing against it must not cost a second
    whole-model twin simulation, and a differing identity is still a mismatch."""
    from types import SimpleNamespace

    import numpy as np

    from merlin.runtime.backends import output_identity
    from merlin.runtime.backends import zephyr_model as zm
    from merlin.targetgen import spike_extension as SE

    flow = _load("chip_flow")
    ext = SE.SpikeExtension(
        target="chip_a",
        extension_name="m",
        extlib=tmp_path / "l.so",
        library_dir=tmp_path,
        sha256="0" * 64,
        declared=True,
    )
    monkeypatch.setattr(SE, "declared_extension", lambda target: {"extension_name": "m"})
    monkeypatch.setattr(SE, "resolve", lambda target, **kw: ext)
    monkeypatch.setattr("merlin.runtime.boards.board", lambda name: SimpleNamespace(vlen=512, dram_base=0x80000000))
    monkeypatch.setattr(flow, "_references", lambda name: {"w8a8": np.ones(4)})
    monkeypatch.setattr(zm, "_gate", lambda got, refs: {"ok": True, "w8a8_cos": 1.0})
    ran = []

    def fake_spike(elf, **kw):
        ran.append(Path(elf).name)
        return {"prefix": np.arange(4.0), "metrics": {"cycles": 7}, "out_hash": {"value": 5}}

    monkeypatch.setattr(zm, "run_on_spike", fake_spike)
    result = _matrix_result(tmp_path)
    twin = {"prefix": np.arange(4.0), "out_hash": {"value": 5}}
    result["matrix"]["twin_output_identity"] = output_identity.output_identity(twin)
    got = flow.matrix_on_spike("op_x", _args(), result)
    assert ran == ["ship.elf"] and got["status"] == "verified" and got["bit_exact_vs_twin"]
    assert got["twin_compare"]["twin_rerun"] is False

    twin["out_hash"] = {"value": 6}  # same dumped prefix, different bits further into the tensor
    result["matrix"]["twin_output_identity"] = output_identity.output_identity(twin)
    got = flow.matrix_on_spike("op_x", _args(), result)
    assert got["status"] == "run_mismatch" and got["twin_compare"]["out_hash_equal"] is False


def test_a_workload_declares_its_capture_deps_beside_its_loader(tmp_path):
    flow = _load("chip_flow")
    (tmp_path / "capture.toml").write_text('deps = ["libone", "libtwo>=2"]\n', encoding="utf-8")
    assert flow.declared_deps(tmp_path) == ["libone", "libtwo>=2"]
    assert flow.declared_deps(tmp_path / "absent") == []


def _probe_says(monkeypatch, verdict):
    import json
    import subprocess

    def probe(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(verdict), stderr="")

    monkeypatch.setattr(subprocess, "run", probe)


def test_a_missing_declared_dep_is_installed_into_the_capture_venv(tmp_path, monkeypatch):
    """A roster model whose own venv is absent falls back to the shared capture venv, which carries
    model2MLIR's stack and not every model's libraries: TinyLlama failed there on `import transformers`."""
    import shutil
    import subprocess

    flow = _load("chip_flow")
    _probe_says(monkeypatch, {"ok": [], "missing": ["libone"], "conflict": []})
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
    installs = []

    def run(cmd, **kw):
        installs.append([str(c) for c in cmd])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(flow, "_run", run)
    flow.ensure_capture_deps(tmp_path / "py", ["libone"], workload="w")
    assert installs == [["uv", "pip", "install", "--python", str(tmp_path / "py"), "libone"]]


def test_a_declared_dep_that_would_replace_an_installed_version_is_refused(tmp_path, monkeypatch):
    flow = _load("chip_flow")
    _probe_says(monkeypatch, {"ok": [], "missing": [], "conflict": ["torch==1.0 (have 2.9)"]})
    monkeypatch.setattr(flow, "_run", lambda *a, **k: pytest.fail("changed the shared venv"))
    with pytest.raises(SystemExit, match="its own venv"):
        flow.ensure_capture_deps(tmp_path / "py", ["torch==1.0"], workload="w")


def test_a_roster_capture_uses_the_configured_capture_interpreter(tmp_path, monkeypatch):
    """`run.sh setup` builds the capture venv OUTSIDE the model2MLIR checkout and records it as
    MERLIN_M2M_PYTHON; a roster capture that ignored it ran under merlin's own interpreter (no torch)."""
    from merlin.compile.bundles import _capture_python

    (tmp_path / "workloads" / "somemodel").mkdir(parents=True)
    py = tmp_path / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("MERLIN_M2M_PYTHON", str(py))
    assert _capture_python(tmp_path, "somemodel") == py
