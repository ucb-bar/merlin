"""Offline final coverage and frozen-image barriers; never invoke an FPGA wrapper."""
import json

import pytest

from merlin.common.tree_hash import hash_tree
from merlin_experiments.post_freeze import firesim, joint_firesim as F


@pytest.fixture
def selection(tmp_path):
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    (compiler / "manifest.yaml").write_text("target: fixture\n")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"schema": "merlin.joint_campaign_completion.v1",
        "frozen_compiler": str(compiler), "compiler_sha256": hash_tree(compiler)["sha256"],
        "completion_reason": "freeze_drain"}))
    launcher = tmp_path / "fpga-run-elf"
    launcher.write_text("#!/bin/sh\nexit 99\n")
    launcher.chmod(0o700)
    source = tmp_path / "source.json"
    source.write_text('{"name":"small","kind":"layer"}')
    value = {"schema": F.SCHEMA, "freeze_record": firesim.pin(freeze), "launcher": firesim.pin(launcher),
        "hardware": {"configuration_class": "selected.original.Config", "source_firrtl_sha256": "a" * 64,
                     "rtl_facts_sha256": "b" * 64, "agfi": "selected-image", "target_clock_hz": 123},
        "roster": {"capsules": [{"id": "small", "source": firesim.pin(source)}],
                   "models": [{"id": "full", "source": firesim.pin(source)}]}, "jobs": []}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(value))
    return path, value, tmp_path


def test_complete_roster_remains_pending_and_dispatch_refuses(selection):
    path, _value, root = selection
    result = F.prepare(path)
    assert result["status"] == "blocked"
    assert "candidate: missing capsules/small" in result["blockers"]
    assert "candidate: missing models/full" in result["blockers"]
    assert "baseline: missing models/full" in result["blockers"]
    assert result["final_validation_complete"] is False
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="prerequisites"):
        F.dispatch(manifest, job_id="anything", output_dir=root / "job")
    assert not (root / "job").exists()


def test_partial_model_and_modified_frozen_compiler_refuse(selection):
    path, value, root = selection
    value["roster"]["models"] = []
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="full models"):
        F.prepare(path)
    (root / "compiler/manifest.yaml").write_text("target: changed\n")
    with pytest.raises(ValueError, match="unchanged final compiler"):
        F.prepare(path)


def test_alternate_hardware_cannot_satisfy_selected_configuration(selection):
    path, value, root = selection
    observed = root / "hardware.json"
    observed.write_text(json.dumps({"selected_hardware": {**value["hardware"], "agfi": "different"},
                                    "build_artifacts": [value["roster"]["models"][0]["source"]]}))
    value["hardware_correspondence"] = firesim.pin(observed)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exact selected configuration"):
        F.prepare(path)


def test_capsule_adapter_prepares_native_image_without_hardware(selection, monkeypatch):
    from merlin.targetgen.capsule_runner import OracleUnavailable
    from merlin.targetgen.contract import compile as compiler
    from merlin_experiments.post_freeze.capsule_firesim import adapter
    _path, value, root = selection
    elf = root / "capsule.elf"
    elf.write_bytes(b"\x7fELFfixture")
    calls = []
    def compile_image(*args, **kwargs):
        calls.append((args, kwargs))
        return elf
    monkeypatch.setattr(compiler, "compile_lowered_to_elf", compile_image)
    with pytest.raises(OracleUnavailable, match="observation pending"):
        adapter(target="fixture", hardware=value["hardware"])({}, "module {}", root, 30)
    assert calls[0][1]["target"] == "fixture"
    saved = json.loads((root / "joint_firesim_image.json").read_text())
    assert saved["elf"] == firesim.pin(elf)
    assert saved["paid_jobs_launched"] == 0


def test_deployed_profile_keeps_unknown_development_equivalence_explicit(selection):
    path, value, root = selection
    value['schema'] = F.DEPLOYED_SCHEMA
    value['hardware'] = {'configuration_class': 'firechip.chip.FireSim.FireSimGemminiRocketConfig',
        'platform_configuration': 'WithAutoILA_FRFCFS16GBQuadRankLLC4MB_BaseF2Config',
        'agfi': 'agfi-existing', 'target_clock_hz': 1_000_000_000}
    profile = root / 'development.json'
    profile.write_text(json.dumps({'execution_clock_hz': 30_000_000,
        'source_firrtl_sha256': 'a' * 64, 'rtl_facts_sha256': 'b' * 64}))
    value['development_profile'] = firesim.pin(profile)
    evidence = root / 'deployed.json'
    observed = {'schema': 'merlin.firesim_deployment_correspondence.v1',
        'selected_hardware': value['hardware'], 'deployment_matches': True,
        'build_artifacts': [value['roster']['models'][0]['source']],
        'development_source_equivalence': {'status': 'unknown',
            'reason': 'Build identity and DTS match; deployed FIRRTL digest unavailable'}}
    evidence.write_text(json.dumps(observed))
    value['hardware_correspondence'] = firesim.pin(evidence)
    path.write_text(json.dumps(value))
    result = F.prepare(path)
    assert result['hardware']['target_clock_hz'] == 1_000_000_000
    assert result['development_source_equivalence']['status'] == 'unknown'
    assert result['development_timing_transfer_qualified'] is False
    assert result['scientific_qualification'] is False
    assert all('correspondence' not in b for b in result['blockers'])
    assert any('active FASED' in b for b in result['blockers'])
    observed['active_fased_runtime'] = {'selected_hardware': value['hardware'],
        'configuration': firesim.pin(profile), 'effective_parameters': {'mm_llc_setBits_0': 12}}
    evidence.write_text(json.dumps(observed))
    value['hardware_correspondence'] = firesim.pin(evidence)
    path.write_text(json.dumps(value))
    assert not any('active FASED' in b for b in F.prepare(path)['blockers'])
    observed['deployment_matches'] = False
    evidence.write_text(json.dumps(observed))
    value['hardware_correspondence'] = firesim.pin(evidence)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='reviewed deployment'):
        F.prepare(path)
