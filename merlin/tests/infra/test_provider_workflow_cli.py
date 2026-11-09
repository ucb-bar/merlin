"""Explicit provider choices do not change ordinary lowering or hide refusal."""

import json
from types import SimpleNamespace

import pytest

from merlin.llvmlower import cli
from merlin.llvmlower import provider_workflow as workflow


def config(**kwargs):
    return {
        "schema": "merlin.provider_workflow.v1",
        "target": "test_support",
        "vlen_bits": 256,
        "elen_bits": 64,
        "operand_bits": 8,
        "accumulator_bits": 32,
        "operand_lmul": 1,
        "workspace_bytes": 8192,
        **kwargs,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("copy_loops", 1),
        ("normalize_windows", "yes"),
        ("stream_im2col", None),
        ("workspace_bytes", True),
        ("timeout_seconds", 0),
        ("copy_max_rank", 9),
    ],
)
def test_invalid_explicit_options_refuse(tmp_path, field, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config(**{field: value})))
    with pytest.raises(ValueError):
        workflow.ProviderConfig.load(path)


def test_options_default_off_and_unknown_fields_refuse(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()))
    selected = workflow.ProviderConfig.load(path)
    assert not selected.copy_loops and not selected.stream_im2col and not selected.normalize_windows
    path.write_text(json.dumps(config(unrecognized=True)))
    with pytest.raises(TypeError):
        workflow.ProviderConfig.load(path)


def test_missing_provider_api_never_falls_back():
    with pytest.raises(ValueError, match="no fallback"):
        workflow._public_api(SimpleNamespace(), "copy_loops_api")


def test_selected_cli_uses_the_provider_entry_and_reports_receipt(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.mlir"
    source.write_text("module {}")
    selected = tmp_path / "provider.json"
    selected.write_text(json.dumps(config(copy_loops=True)))
    output = tmp_path / "out"
    calls = []

    def compile_file(path, destination, options, *, targets):
        calls.append((path, destination, options, targets))
        return SimpleNamespace(
            workdir=destination,
            ll_path=destination / "model.ll",
            host_so=None,
            riscv_obj=None,
            audit_index=None,
            stats={"provider_workflow": "receipt"},
        )

    monkeypatch.setattr(workflow, "compile_file", compile_file)
    assert cli.main([str(source), "--out", str(output), "--provider-config", str(selected)]) == 0
    assert calls[0][0:2] == (source, output) and calls[0][2].copy_loops
    assert json.loads(capsys.readouterr().out)["stats"]["provider_workflow"] == "receipt"


@pytest.mark.parametrize("options", [["--feature", "old_runner_hook"], ["--textual"], ["--ir-audit"]])
def test_closed_provider_path_cannot_silently_drop_runner_features(tmp_path, options):
    source = tmp_path / "source.mlir"
    source.write_text("module {}")
    selected = tmp_path / "provider.json"
    selected.write_text(json.dumps(config()))
    with pytest.raises(SystemExit) as error:
        cli.main([str(source), "--out", str(tmp_path / "out"), "--provider-config", str(selected), *options])
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_execution_requires_selected_provider_and_riscv_object(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text("module {}")
    with pytest.raises(SystemExit):
        cli.main([str(source), "--out", str(tmp_path / "out"), "--execution-plan", "plan.json"])
    assert not (tmp_path / "out").exists()


def test_changed_config_bytes_refuse_before_provider_resolution(tmp_path):
    source, selected = tmp_path / "model.mlir", tmp_path / "provider.json"
    source.write_text("module {}")
    selected.write_text(json.dumps(config()))
    options = workflow.ProviderConfig.load(selected)
    selected.write_text(json.dumps(config(vlen_bits=512)))
    with pytest.raises(ValueError, match="changed after selection"):
        workflow.compile_file(source, tmp_path / "build", options)
    assert json.loads((tmp_path / "build/provider-workflow.json").read_text())["status"] == "failed"


def test_closed_native_path_does_not_bypass_floating_effect_policies():
    from merlin.common import mlir_query as mq

    module = mq.parse("module { func.func @f(%x:f32)->f32 {func.return %x:f32} }")
    with pytest.raises(ValueError, match="ordinary lowering"):
        workflow._integer_source(module)


def test_mixed_merlin_namespace_roots_refuse(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    module = ModuleType("merlin.unexpected_external")
    module.__file__ = str(tmp_path / "unexpected.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="mixed Merlin"):
        workflow._core_sources(())
