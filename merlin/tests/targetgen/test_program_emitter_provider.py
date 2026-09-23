"""Selected helper ownership and subprocess protocol, without executing model code."""

import json
from types import SimpleNamespace

import pytest
import yaml

from merlin.targetgen import program_oracle as oracle


def test_target_specific_emitter_is_not_shipped_by_core():
    from merlin.common.paths import repo_root

    assert not (repo_root() / "src/merlin/targetgen/oracle_helpers/npu_emit.py").exists()


@pytest.fixture
def provider(tmp_path, monkeypatch):
    from merlin.targetgen import target_registry

    root = tmp_path / "support"
    (root / "contracts").mkdir(parents=True)
    (root / "provider.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.provider.v1",
                "id": "fixture-support",
                "target": "fixture_device",
                "role": "support",
            }
        )
    )
    helper = root / "emitter.py"
    helper.write_text("raise AssertionError('must never import this helper')\n")
    contract = root / "contracts/target_contract.yaml"
    declaration = {"path": "emitter.py", "args": ["--provider-option"]}
    contract.write_text(yaml.safe_dump({"name": "fixture_device", "runner": {"program_emitter": declaration}}))
    info = target_registry.TargetInfo(
        "fixture_device",
        "external",
        root,
        contract,
        root / "dialect.yaml",
        root / "facts.json",
        "fixture",
        root,
    )
    monkeypatch.setattr(target_registry, "resolve", lambda target: info)
    return info, helper, declaration


def test_resolves_only_selected_helper_without_importing(provider):
    _, helper, _ = provider
    assert oracle._program_emitter("fixture_device") == (helper, ["--provider-option"])


@pytest.mark.parametrize("role", ["candidate_compiler", "host_schedule"])
def test_non_support_provider_is_refused(provider, role):
    info, _, _ = provider
    (info.base / "provider.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.provider.v1",
                "id": "fixture",
                "target": "fixture_device",
                "role": role,
            }
        )
    )
    with pytest.raises(oracle.OracleUnavailable, match="support provider is required"):
        oracle._program_emitter("fixture_device")


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        {},
        {"path": "missing.py"},
        {"path": "../outside.py"},
        {"path": "/tmp/foreign.py"},
        {"path": "emitter.py", "args": "--flag"},
        {"path": "emitter.py", "args": [1]},
        {"path": "emitter.py", "typo": True},
    ],
)
def test_missing_or_invalid_declaration_is_unavailable(provider, declaration):
    info, _, _ = provider
    info.contract_path.write_text(
        yaml.safe_dump(
            {
                "name": "fixture_device",
                "runner": {"program_emitter": declaration},
            }
        )
    )
    with pytest.raises(oracle.OracleUnavailable, match="program emitter unavailable"):
        oracle._program_emitter("fixture_device")


def test_helper_symlink_cannot_escape_provider(provider, tmp_path):
    _, helper, _ = provider
    outside = tmp_path / "outside.py"
    outside.write_text("# outside")
    helper.unlink()
    helper.symlink_to(outside)
    with pytest.raises(oracle.OracleUnavailable, match="escapes provider"):
        oracle._program_emitter("fixture_device")


@pytest.mark.parametrize("mode", ["program", "inputs"])
def test_bundle_uses_provider_options_and_model_environment(provider, tmp_path, monkeypatch, mode):
    _, helper, _ = provider
    monkeypatch.setattr(oracle, "_model_venv_python", lambda model: tmp_path / "python")
    monkeypatch.setattr(oracle, "ext_path", lambda model: tmp_path / "model")
    monkeypatch.setattr(oracle, "_assemble_kernel_words", lambda *args: [9])
    calls = []
    bundle = {"words": [7], "inputs": [{"base": 10, "b64": "AA=="}], "output": None, "golden": None}

    def run(command, **kwargs):
        calls.append((command, kwargs))
        from pathlib import Path

        Path(command[command.index("--out") + 1]).write_text(json.dumps(bundle))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(oracle.subprocess, "run", run)
    request = (
        {"program": "Example"}
        if mode == "program"
        else {
            "kernel_s": tmp_path / "kernel.S",
            "inputs": [{"base": 10, "dtype": "i8", "values": [0]}],
        }
    )
    result = oracle.emit_bundle(target="fixture_device", model_ext="model", workdir=tmp_path, timeout=12, **request)
    command, options = calls[0]
    assert command[:3] == [str(tmp_path / "python"), str(helper), "--provider-option"]
    assert ("--program" if mode == "program" else "--inputs") in command
    assert options == {"cwd": str(tmp_path / "model"), "capture_output": True, "text": True, "timeout": 12}
    assert result["words"] == ([7] if mode == "program" else [9])
    assert result["inputs"] == bundle["inputs"]


def test_kernel_without_layout_needs_no_emitter(tmp_path, monkeypatch):
    monkeypatch.setattr(oracle, "_assemble_kernel_words", lambda *args: [9])
    monkeypatch.setattr(oracle, "_program_emitter", lambda target: pytest.fail("no emitter is needed"))
    assert oracle.emit_bundle(
        target="fixture_device", model_ext="model", kernel_s=tmp_path / "kernel.S", workdir=tmp_path, timeout=1
    )["words"] == [9]


def test_emitter_and_provider_membership_bind_verdict_cache(provider, tmp_path, monkeypatch):
    from merlin.runtime.backends import base
    from merlin.targetgen import tier_cache

    info, helper, _ = provider
    monkeypatch.setattr(tier_cache, "_GRADING_MODULES", ())
    monkeypatch.setattr(tier_cache, "_INSTRUMENT_MEMO", {})
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(__file__=str(helper)))
    assert info.contract_path in tier_cache.grading_path("fixture_device")
    before = tier_cache.instrument_digest("fixture_device", "L2", rtl_tier=False)
    assert before is not None
    helper.write_text("# changed helper\n")
    tier_cache._INSTRUMENT_MEMO.clear()  # same reset as the next grading invocation
    assert tier_cache.instrument_digest("fixture_device", "L2", rtl_tier=False) != before
    before = tier_cache.instrument_digest("fixture_device", "L2", rtl_tier=False)
    (info.base / "sibling.py").write_text("# helper dependency\n")
    tier_cache._INSTRUMENT_MEMO.clear()
    assert tier_cache.instrument_digest("fixture_device", "L2", rtl_tier=False) != before
    before = tier_cache.instrument_digest("fixture_device", "L2", rtl_tier=False)
    contract = yaml.safe_load(info.contract_path.read_text())
    contract["runner"]["program_emitter"]["args"] = ["--different-policy"]
    info.contract_path.write_text(yaml.safe_dump(contract))
    tier_cache._INSTRUMENT_MEMO.clear()
    assert tier_cache.instrument_digest("fixture_device", "L2", rtl_tier=False) != before
    helper.unlink()
    tier_cache._INSTRUMENT_MEMO.clear()
    assert tier_cache.instrument_digest("fixture_device", "L2", rtl_tier=False) is None
