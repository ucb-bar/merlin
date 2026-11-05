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
