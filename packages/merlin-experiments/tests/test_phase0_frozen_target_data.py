"""Data-bound Phase 0 keeps its contract/facts through guarded bootstrap."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments import frozen_python, source_snapshot
from merlin_experiments.phase0 import evidence as evidence_owner
from merlin_experiments.phase0 import freeze


def test_data_only_stage_bootstrap_and_verification(tmp_path, monkeypatch):
    contract = {"name": "gemmini", "runner": {"backend": "chipyard_rocc"}}
    inputs = {}
    for name, document in {
        "descriptor": {"target": "gemmini"},
        "recipe": {},
        "performance_template": {},
        "software_spec": {},
        "capability_contract": contract,
        "rtl_facts": {"facts": {"selected": True}},
    }.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(document))
        inputs[name] = str(path)
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("raise AssertionError('not the probe entrypoint')\n")
    paths = {f"phase0:operator:{name}": value for name, value in inputs.items()}
    paths["phase0:source:entry"] = str(entrypoint)
    argv = [sys.executable, str(entrypoint)]
    for name, value in inputs.items():
        argv += ["--" + name.replace("_", "-"), value]
    observed = SimpleNamespace(
        target="gemmini", contract=contract, software_spec={}, source_paths=(), source_snapshots=(), views_json=b"{}"
    )
    monkeypatch.setattr(freeze, "selection", lambda *_: observed)
    monkeypatch.setattr(evidence_owner, "load_exported_evidence", lambda *_: observed)

    def export(_evidence, destination):
        destination.mkdir(parents=True)
        return {"sources": []}

    monkeypatch.setattr(freeze, "export_evidence", export)
    monkeypatch.setattr(
        source_snapshot, "selected_provider", lambda *_: pytest.fail("data-only stage must not resolve support")
    )
    plan = freeze.stage(
        {
            "target": "gemmini",
            "run_dir": str(tmp_path / "run"),
            "storage_root": str(tmp_path / "out"),
            "input_paths": paths,
            "inputs": {name: {"path": path, "sha256": freeze.fingerprint(path)} for name, path in paths.items()},
            "phases": {
                "0": {
                    "inputs": inputs,
                    "argv": argv,
                    "entrypoint": str(entrypoint),
                    "env": {},
                    "phase0_evidence": {"status": "diagnostic"},
                }
            },
        }
    )
    snapshot = Path(plan["phase0_source_snapshot"])
    receipt = freeze._verify_frozen_sources(plan)
    command = plan["phases"]["0"]
    assert receipt["selected_provider"] is None
    assert not (snapshot / source_snapshot.PROVIDER_ROOT).exists()
    for variable, field in (("MERLIN_TARGET_CONTRACT", "capability_contract"), ("MERLIN_RTL_FACTS", "rtl_facts")):
        selected = command["inputs"][field]
        assert command["env"][variable] == selected
        assert snapshot / receipt["target_data"][variable] == Path(selected)
        changed = copy.deepcopy(plan)
        changed["phases"]["0"]["env"][variable] = inputs[field]
        with pytest.raises(ValueError, match="provider selection changed"):
            freeze._verify_frozen_sources(changed)
    for path in inputs.values():
        Path(path).unlink()
    # Ambient selections must be overwritten from the receipt at every bootstrap.
    probe = """
import json, os, subprocess, sys
from pathlib import Path
from merlin.targetgen.target_registry import load_contract
from merlin_experiments.frozen_python import inherited_python_command
assert os.environ['MERLIN_TARGET_PATH'] == ''
assert load_contract('gemmini') == {'name': 'gemmini', 'runner': {'backend': 'chipyard_rocc'}}
assert json.loads(Path(os.environ['MERLIN_RTL_FACTS']).read_bytes()) == {'facts': {'selected': True}}
child = ("from merlin.targetgen.target_registry import load_contract; "
         "assert load_contract('gemmini')['runner']['backend'] == 'chipyard_rocc'")
os.environ.update(MERLIN_TARGET_PATH='/poison', MERLIN_TARGET_CONTRACT='/poison', MERLIN_RTL_FACTS='/poison')
subprocess.run(inherited_python_command([sys.executable, '-c', child]), check=True, timeout=20)
"""
    launch = frozen_python.python_command(
        snapshot,
        [sys.executable, "-c", probe],
        verifier_source=Path(source_snapshot.__file__),
        instrumentation_from_snapshot=True,
    )
    env = dict(os.environ, MERLIN_TARGET_PATH="/poison", MERLIN_TARGET_CONTRACT="/poison", MERLIN_RTL_FACTS="/poison")
    result = subprocess.run(launch, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    freeze._verify_frozen_sources(plan)
    selected = snapshot / receipt["target_data"]["MERLIN_TARGET_CONTRACT"]
    selected.chmod(0o600)
    selected.write_text('{"name": "drift"}')
    selected.chmod(0o400)
    with pytest.raises(source_snapshot.SnapshotError, match="source changed"):
        freeze._verify_frozen_sources(plan)


@pytest.mark.parametrize("value", ["/live/contract.json", "../contract.json", "missing.json"])
def test_target_data_must_name_sealed_declared_input(tmp_path, value):
    source = tmp_path / "source"
    (source / "python/toy").mkdir(parents=True)
    (source / "python/toy/__init__.py").write_text("")
    snapshot = tmp_path / "snapshot"
    source_snapshot.create(
        source,
        snapshot,
        output_root=tmp_path / "out",
        source_roots=("python",),
        python_roots=("python",),
        legacy_roots=(),
        runtime_links=False,
    )
    receipt = source_snapshot.verify(snapshot)
    receipt["target_data"] = {"MERLIN_TARGET_CONTRACT": value}
    with pytest.raises(source_snapshot.SnapshotError):
        source_snapshot.import_layout(snapshot, receipt)
