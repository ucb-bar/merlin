"""Selected runtime reuse binds ordinary runner and CLI inputs without executing captures."""

import json
from copy import deepcopy

import pytest
from merlin_experiments import SpecError, load_spec, runner
from test_phase0_explicit_plan import authored as authored


def test_capture_runtime_reuse_is_selected_verified_and_file_pinned(authored, monkeypatch, tmp_path):
    from merlin_experiments.phase0 import freeze as phase0_freeze
    from merlin_experiments.phase0 import m2m_runtime
    from merlin_experiments.phase0.sealed_generation import CONFIG_ENV

    make, root, config = authored
    (root / "software.yaml").write_text("{}\n")
    monkeypatch.setattr(m2m_runtime, "observe", lambda *args, **kwargs: {"base": str(tmp_path / "base")})
    monkeypatch.setattr(phase0_freeze, "selected_inputs", lambda *args, **kwargs: ({}, {}))
    observed = []

    def checked_config(*args, runtime_reuse=None, **kwargs):
        observed.append(runtime_reuse)
        return {} if runtime_reuse is None else {"runtime_reuse": deepcopy(runtime_reuse)}

    monkeypatch.setattr(m2m_runtime, "sealed_capture_config", checked_config)
    make({**config, "software_spec": "software.yaml", "evidence_mode": "verified"})
    selection = root / "runtime.json"
    selected = {"schema": "fixture", "entry": str(tmp_path / "readonly-runtime")}
    selection.write_text(json.dumps(selected))
    default = runner.resolve_plan(
        load_spec(root / "experiment.yaml"), phase="0", run_dir=tmp_path / "default-run",
        phase0_m2m_root=tmp_path / "m2m", phase0_m2m_python=tmp_path / "venv/bin/python",
    )
    assert "phase0_capture_runtime_reuse" not in default["phases"]["0"]
    assert "phase0:capture-runtime-reuse-selection" not in default["input_paths"]
    plan = runner.resolve_plan(
        load_spec(root / "experiment.yaml"), phase="0", run_dir=tmp_path / "run",
        phase0_m2m_root=tmp_path / "m2m", phase0_m2m_python=tmp_path / "venv/bin/python",
        phase0_capture_runtime_reuse=selection,
    )
    command = plan["phases"]["0"]
    assert observed == [None, selected]
    assert command["phase0_capture_runtime_reuse"] == selected
    assert json.loads(command["env"][CONFIG_ENV])["runtime_reuse"] == selected
    assert selected["entry"] in command["input_owner_roots"]
    assert plan["input_paths"]["phase0:capture-runtime-reuse-selection"] == str(selection)
    # Reopening frozen input pins detects changes to the original selection;
    # framework/source verification is independently exercised by freeze tests.
    monkeypatch.setattr(runner, "_verify_phase0_sources", lambda plan: None)
    frozen = {**plan, "inputs": {
        name: {"path": path, "sha256": runner.fingerprint(path)} for name, path in plan["input_paths"].items()
    }}
    runner._verify_inputs(frozen)
    selection.write_text(json.dumps({**selected, "entry": str(tmp_path / "different-runtime")}))
    with pytest.raises(SpecError, match="frozen input changed: phase0:capture-runtime-reuse-selection"):
        runner._verify_inputs(frozen)


@pytest.mark.parametrize("raw", ['[]', '{"entry":"a","entry":"b"}', '{"entry":NaN}'])
def test_capture_runtime_reuse_rejects_ambiguous_records(authored, raw):
    make, root, _ = authored
    make()
    selection = root / "runtime.json"
    selection.write_text(raw)
    with pytest.raises(SpecError, match="invalid Phase 0 capture runtime reuse"):
        runner.resolve_plan(load_spec(root / "experiment.yaml"), phase="0", phase0_capture_runtime_reuse=selection)


def test_capture_runtime_reuse_requires_ordinary_sealed_runtime_selection(authored):
    make, root, _ = authored
    make()
    selection = root / "runtime.json"
    selection.write_text('{}')
    with pytest.raises(SpecError, match="requires --phase 0"):
        runner.resolve_plan(load_spec(root / "experiment.yaml"), phase0_capture_runtime_reuse=selection)
    with pytest.raises(SpecError, match="requires a selected sealed generation runtime"):
        runner.resolve_plan(load_spec(root / "experiment.yaml"), phase="0", phase0_capture_runtime_reuse=selection)
    alias = root / "alias.json"
    alias.symlink_to(selection)
    with pytest.raises(SpecError, match="ordinary file"):
        runner.resolve_plan(load_spec(root / "experiment.yaml"), phase="0", phase0_capture_runtime_reuse=alias)


def test_cli_forwards_explicit_capture_runtime_selection(authored, monkeypatch):
    from merlin_experiments import cli

    make, root, _ = authored
    make()
    selection = root / "runtime.json"
    observed = []
    monkeypatch.setattr(runner, "resolve_plan", lambda spec, **kwargs: observed.append(kwargs) or {})
    assert cli.main([
        'inspect', str(root / 'experiment.yaml'), '--phase', '0',
        '--phase0-capture-runtime-reuse', str(selection),
        '--phase0-rtl-source-relocations', str(root / 'rtl-relocations.json'),
    ]) == 0
    assert observed[0]['phase0_capture_runtime_reuse'] == selection
    assert observed[0]['phase0_rtl_source_relocations'] == root / 'rtl-relocations.json'


def test_rtl_source_relocations_reach_evidence_and_keep_original_file_pin(authored, monkeypatch, tmp_path):
    from merlin_experiments.phase0 import freeze as phase0_freeze

    make, root, config = authored
    (root / 'software.yaml').write_text('{}\n')
    selection = root / 'relocations.json'
    copied = root / 'original-source.txt'
    copied.write_text('exact originally recorded source\n')
    mapping = {'/original/reader.py': {'path': str(copied), 'sha256': runner.fingerprint(copied)}}
    selection.write_text(json.dumps(mapping))
    seen = []

    def evidence(command, target):
        if 'phase0_rtl_source_relocations' in command:
            seen.append(command['phase0_rtl_source_relocations'])
        return {'phase0:source:recorded-reader': str(copied)}, {'status': 'verified'}

    monkeypatch.setattr(phase0_freeze, 'selected_inputs', evidence)
    make({**config, 'software_spec': 'software.yaml', 'evidence_mode': 'verified'})
    plan = runner.resolve_plan(
        load_spec(root / 'experiment.yaml'), phase='0', run_dir=tmp_path / 'run',
        phase0_rtl_source_relocations=selection,
    )
    assert seen == [mapping]
    assert plan['phases']['0']['phase0_rtl_source_relocations'] == mapping
    assert plan['input_paths']['phase0:rtl-source-relocations-selection'] == str(selection)
    assert plan['input_paths']['phase0:source:recorded-reader'] == str(copied)


def test_rtl_source_mapping_cannot_be_silently_ignored(authored):
    make, root, _ = authored
    make()
    selection = root / 'relocations.json'
    selection.write_text('{}')
    with pytest.raises(SpecError, match='requires --phase 0'):
        runner.resolve_plan(load_spec(root / 'experiment.yaml'), phase0_rtl_source_relocations=selection)
    with pytest.raises(SpecError, match='requires selected Phase 0 RTL evidence'):
        runner.resolve_plan(load_spec(root / 'experiment.yaml'), phase='0', phase0_rtl_source_relocations=selection)
    selection.write_text('{"/original":{},"/original":{}}')
    with pytest.raises(SpecError, match='duplicate JSON key'):
        runner.resolve_plan(load_spec(root / 'experiment.yaml'), phase='0', phase0_rtl_source_relocations=selection)
