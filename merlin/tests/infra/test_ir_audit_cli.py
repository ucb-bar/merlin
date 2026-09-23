"""CLI audit flags reach real lowering; only the Triton frontend is synthetic."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import python_import_roots
from merlin.triton import bridge, cli, source
from merlin.xdsl_dialects._common import text
from merlin.xdsl_dialects.lowering.input_workload import build_input_module


@pytest.fixture
def frontend(monkeypatch):
    module = build_input_module(reuse=1, m=2, k=2, n=2)
    monkeypatch.setattr(cli, "load_kernel", lambda name: lambda: None)
    monkeypatch.setattr(
        source,
        "make_ttir",
        lambda spec: SimpleNamespace(text="synthetic frontend", digest="fixture", triton_version="fixture"),
    )
    monkeypatch.setattr(
        bridge,
        "to_linalg",
        lambda ttir, spec: SimpleNamespace(
            module=module, text=text(module), report=SimpleNamespace(as_dict=lambda: {})
        ),
    )
    return ["fixture.py:kernel", "--target", "toy_npu", "--arg", "out=*fp32:2x2:write", "--emit", "report"]


def test_help_discovers_exact_stage_and_sidecar_flags():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in python_import_roots()))
    result = subprocess.run(
        [sys.executable, "-m", "merlin.triton.cli", "--help"], env=env, text=True, capture_output=True, check=True
    )
    assert "--ir-audit" in result.stdout
    assert "--audit-sidecar" in result.stdout
    assert "may be large" in " ".join(result.stdout.split())


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_controls_actual_stage_snapshots_and_sidecar_binding(frontend, tmp_path, enabled):
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"unchanged opaque bytes")
    out = tmp_path / "run"
    args = frontend + ["--out", str(out)]
    if enabled:
        args += ["--ir-audit", "--audit-sidecar", str(weights)]
    assert cli.main(args) == 0
    indexes = list(out.glob("ir-audit-*/index.json"))
    assert len(indexes) == int(enabled)
    if enabled:
        index = json.loads(indexes[0].read_text())
        assert index["outcome"] == "completed"
        assert [stage["name"] for stage in index["stages"]] == [
            "input",
            "contract",
            "schedule",
            "interface",
            "target",
            "runtime",
        ]
        assert index["sidecars"][0]["path"] == str(weights.resolve())
    assert weights.read_bytes() == b"unchanged opaque bytes"


def test_sidecar_flag_without_audit_refuses_before_frontend():
    with pytest.raises(SystemExit, match="requires --ir-audit"):
        cli.main(["absent.py:kernel", "--audit-sidecar", "absent"])


def test_missing_sidecar_refuses_actual_lowering(frontend, tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.main(frontend + ["--out", str(tmp_path / "run"), "--ir-audit", "--audit-sidecar", str(tmp_path / "absent")])
    assert not list((tmp_path / "run").glob("ir-audit-*"))


def test_route_only_audit_flag_still_writes_nothing(frontend, tmp_path):
    assert cli.main(frontend + ["--out", str(tmp_path / "run"), "--ir-audit", "--route-only"]) == 0
    assert not (tmp_path / "run").exists()


def test_compact_cli_records_staged_inspection_views(frontend, tmp_path):
    assert cli.main(frontend + ["--out", str(tmp_path), "--ir-audit", "compact"]) == 0
    index_path = next(tmp_path.glob("ir-audit-*/index.json"))
    index = json.loads(index_path.read_text())
    assert index["mode"] == "compact"
    assert len(index["stages"]) == 6
    assert all((index_path.parent / stage["inspection"]["file"]).is_file() for stage in index["stages"])
    assert not list(index_path.parent.glob("*.mlir"))


def test_cli_rejects_unknown_mode_before_importing_kernel():
    with pytest.raises(SystemExit):
        cli.main(["absent.py:kernel", "--ir-audit", "typo"])


def test_llvm_route_threads_flags_to_shared_lowering(frontend, monkeypatch, tmp_path):
    from merlin import compile_core

    route = SimpleNamespace(kind="llvm", reason="synthetic route", as_dict=lambda: {})
    monkeypatch.setattr(compile_core, "choose_route", lambda *args, **kwargs: route)
    calls = []

    def lower(module, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(staged=None)

    monkeypatch.setattr(compile_core, "compile_core_mlir", lower)
    sidecar = tmp_path / "declared"
    assert cli.main(frontend + ["--out", str(tmp_path), "--ir-audit", "--audit-sidecar", str(sidecar)]) == 0
    assert calls[0]["workdir"] == tmp_path / "llvm"
    assert calls[0]["ir_audit"] == "exact"
    assert calls[0]["audit_sidecars"] == (str(sidecar),)
