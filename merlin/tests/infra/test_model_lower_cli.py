"""Model inspection CLI contracts without importing or executing a compiler."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    source = tmp_path / "model.mlir"
    source.write_text("module {}\n")
    output = tmp_path / "lowered"
    observed = []
    backend = ModuleType("merlin.llvmlower.lower")

    def lower(path, workdir, **kwargs):
        observed.append((Path(path), Path(workdir), kwargs))
        Path(workdir).mkdir(exist_ok=True)
        ll = Path(workdir) / "model.ll"
        ll.write_text("; synthetic LLVM\n")
        return SimpleNamespace(
            workdir=Path(workdir),
            ll_path=ll,
            host_so=None,
            riscv_obj=None,
            stats={"synthetic": 1},
            audit_index=Path(workdir) / "audit/index.json" if kwargs["ir_audit"] else None,
        )

    backend.lower_model_file = lower
    monkeypatch.setitem(sys.modules, "merlin.llvmlower.lower", backend)
    return SimpleNamespace(source=source, output=output, observed=observed, backend=backend)


def cli():
    return importlib.import_module("merlin.llvmlower.cli")


def test_default_is_llvm_only_and_reports_outputs(invocation, capsys):
    item = invocation
    assert cli().main([str(item.source), "--out", str(item.output)]) == 0
    path, output, options = item.observed[0]
    assert path == item.source and output == item.output
    assert options["targets"] == ()
    assert options["features"] is None
    assert options["ir_audit"] is False
    assert options["audit_sidecars"] == ()
    result = json.loads(capsys.readouterr().out)
    assert result["ll_path"] == str(item.output / "model.ll")
    assert result["host_so"] is None and result["riscv_obj"] is None
    assert result["audit_index"] is None
    assert result["stats"] == {"synthetic": 1}


def test_explicit_targets_audit_and_features_forward_without_copying_sidecars(invocation, capsys):
    item = invocation
    sidecars = [item.source.parent / "weights.safetensors", item.source.parent / "weights.manifest.json"]
    for path in sidecars:
        path.write_bytes(b"synthetic sidecar")
    argv = [
        str(item.source),
        "--out",
        str(item.output),
        "--target",
        "host",
        "--target",
        "riscv",
        "--textual",
        "--ir-audit",
        "compact",
        "--feature",
        "first",
        "--feature",
        "second",
    ]
    for path in sidecars:
        argv.extend(["--audit-sidecar", str(path)])
    assert cli().main(argv) == 0
    options = item.observed[0][2]
    assert options["targets"] == ("host", "riscv")
    assert options["textual"] is True
    assert options["ir_audit"] == "compact"
    assert options["features"] == frozenset({"first", "second"})
    assert tuple(map(Path, options["audit_sidecars"])) == tuple(sidecars)
    assert not list(item.output.glob("*.safetensors"))
    assert json.loads(capsys.readouterr().out)["audit_index"] == str(item.output / "audit/index.json")


def test_top_level_lower_route_and_bare_audit_flag(invocation):
    from merlin import cli as entry

    item = invocation
    assert entry.main(["lower", str(item.source), "--out", str(item.output), "--ir-audit"]) == 0
    assert item.observed[0][2]["ir_audit"] == "exact"


@pytest.mark.parametrize("problem", ["missing-input", "existing-output", "sidecar-without-audit", "missing-out"])
def test_invalid_preflight_never_calls_lowering(invocation, problem):
    item = invocation
    argv = [str(item.source), "--out", str(item.output)]
    if problem == "missing-input":
        argv[0] = str(item.source.parent / "absent.mlir")
    elif problem == "existing-output":
        item.output.mkdir()
        (item.output / "keep").write_bytes(b"retained")
    elif problem == "sidecar-without-audit":
        argv.extend(["--audit-sidecar", str(item.source)])
    else:
        argv = [str(item.source)]
    with pytest.raises(SystemExit) as exc:
        cli().main(argv)
    assert exc.value.code == 2
    assert item.observed == []
    if problem == "existing-output":
        assert (item.output / "keep").read_bytes() == b"retained"
    else:
        assert not item.output.exists()


def test_failed_lowering_retains_evidence_and_returns_failure(invocation, capsys):
    item = invocation

    def fail(path, workdir, **kwargs):
        Path(workdir).mkdir(exist_ok=True)
        (Path(workdir) / "failure.log").write_text("retained failure")
        raise RuntimeError("synthetic lowering refusal")

    item.backend.lower_model_file = fail
    assert cli().main([str(item.source), "--out", str(item.output)]) == 1
    assert "synthetic lowering refusal" in capsys.readouterr().err
    assert (item.output / "failure.log").read_text() == "retained failure"


def test_help_in_fresh_process_does_not_import_lowering_or_frameworks():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'merlin.llvmlower.lower' or fullname.split('.')[0] in {'torch', 'triton', 'xdsl'}:
            raise AssertionError('help imported heavy dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from merlin.cli import main
raise SystemExit(main(['lower', '--help']))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--ir-audit" in result.stdout
    assert "--audit-sidecar" in result.stdout
