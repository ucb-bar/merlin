"""Run-owned host inputs reach operator children without entering author grants."""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from merlin_experiments.phase1 import options, session
from merlin_experiments.phase1.feedback import public_owned_host_inputs as inputs
from merlin_experiments.phase1.feedback import public_owned_host_selection as transport

from test_public_owned_host_inputs import pin, selected


def test_private_freeze_child_inheritance_and_exact_resume(selected, tmp_path, monkeypatch):
    request = selected.write()
    run = tmp_path / "run"
    native, record = inputs.freeze_for_run(request["path"], request["sha256"], target="fixture",
                                           descriptor=selected.descriptor, run_dir=run)
    assert set(record["inputs"]) == {"descriptor", "board_catalog", "dts", "rtl_facts"}
    assert all(Path(row["path"]).stat().st_mode & 0o777 == 0o400 for row in record["inputs"].values())
    (run / "environment.yaml").write_text(yaml.safe_dump({inputs.KEY: record, "run_config": {inputs.KEY: request}}))
    assert inputs.from_run(run, target="fixture", descriptor=selected.descriptor) == native
    monkeypatch.setenv(transport.ENV, "untrusted inherited selection")
    assert inputs.install_from_run(run, target="fixture", descriptor=selected.descriptor) == native
    assert json.loads(os.environ[transport.ENV]) == native
    assert inputs.freeze_for_run(request["path"], request["sha256"], target="fixture",
                                  descriptor=selected.descriptor, run_dir=run, prior=record, resuming=True) == (native, record)
    monkeypatch.setenv(transport.ENV, json.dumps(native))
    script = "import json,os,pathlib,hashlib;r=json.loads(os.environ['MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION']);p=pathlib.Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];s=json.loads(p.read_bytes());assert set(s)=={'board','arena_mb','dts','board_catalog','rtl_facts'};print(s['board'])"
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "native_board"
    with pytest.raises(ValueError, match="removed"):
        inputs.freeze_for_run("", "", target="fixture", descriptor=selected.descriptor, run_dir=run, prior=record, resuming=True)
    copied = Path(record["inputs"]["dts"]["path"])
    copied.chmod(0o600);copied.write_text("changed private copy")
    with pytest.raises(ValueError, match="changed"):
        inputs.install_from_run(run, target="fixture", descriptor=selected.descriptor)
    assert transport.ENV not in os.environ


def test_formal_child_never_uses_inherited_selection_without_native_record(tmp_path, monkeypatch):
    monkeypatch.setenv(transport.ENV, "untrusted inherited selection")
    assert inputs.install_from_run(tmp_path, target="fixture", descriptor=tmp_path / "absent") is None
    assert transport.ENV not in os.environ


@pytest.mark.parametrize("flags", [
    ["--public-owned-host-inputs", "/operator/inputs.json"],
    ["--public-owned-host-inputs-sha256", "a" * 64],
    ["--public-owned-host-inputs", "/operator/inputs.json", "--public-owned-host-inputs-sha256", "bad"],
])
def test_native_flags_require_the_exact_pair_before_preparation(flags):
    with pytest.raises(ValueError, match="public host"):
        session.validate_options(options.parse_options(["--run-id", "host-test", *flags]))


def test_independent_input_pair_does_not_enable_private_models():
    parsed = options.parse_options(["--run-id", "host-test", "--public-owned-host-inputs", "/operator/inputs.json",
                                    "--public-owned-host-inputs-sha256", "a" * 64])
    assert parsed.private_full_model_spec == "" and parsed.instruction_selection == ""
    session.validate_options(parsed)


def test_actual_native_prepare_installs_inputs_before_continuation(monkeypatch, tmp_path):
    # Reuse the existing real native admission fixture, which substitutes only
    # external execution. Workspace freezing and visibility remain original.
    from merlin_experiments.phase1 import session as S
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.runtime import modelinputs as model_execution_inputs
    from merlin_experiments.phase1.feedback import private_instruction_declaration

    path = Path(__file__).parent / "test_phase1_session.py"
    if not path.is_file():
        from merlin.common.paths import module_source_path
        path = module_source_path("merlin_experiments.phase1.session").parents[3] / "tests/test_phase1_session.py"
    spec = importlib.util.spec_from_file_location("host_inputs_original_session_fixture", path)
    fixture = importlib.util.module_from_spec(spec);sys.modules[spec.name] = fixture;spec.loader.exec_module(fixture)
    project = fixture.project.__wrapped__(tmp_path, monkeypatch)
    fixture._external_substitutes(monkeypatch)
    dts = project / "native.dts";dts.write_text("selected DTS")
    facts = project / "private-facts.json";facts.write_text("selected protected facts")
    catalog = project / "boards.yaml"
    catalog.write_text(yaml.safe_dump({"schema_version": 1, "boards": {"native": {
        "target": "fixture", "rtl_sim_config": "native_cfg", "host_dts_sha256": pin(dts)["sha256"],
        "dram_bytes": "256 MiB", "dram_base": 2147483648, "harts": 1, "console": "htif", "flow": "baremetal", "code_reserve": "64 MiB"}}}))
    descriptor = project / "target_experiment.yaml"
    doc = yaml.safe_load(descriptor.read_bytes());doc["host"] = {"board": "native", "catalog": "boards.yaml"};descriptor.write_text(yaml.safe_dump(doc))
    request_file = project / "operator-host-inputs.json"
    declaration = project / "operator-instruction.json";declaration.write_text("selected instruction sources")
    monkeypatch.setattr(private_instruction_declaration, "read", lambda path, **kw: SimpleNamespace(verify=lambda: None))
    request_file.write_text(json.dumps(dict(schema=inputs.SCHEMA, target="fixture", descriptor=pin(descriptor),
                                           board_catalog=pin(catalog), dts=pin(dts), rtl_facts=pin(facts),
                                           instruction_declaration=pin(declaration), board="native", arena_mb=32)))
    monkeypatch.setattr(model_execution_inputs, "selected_firrtl", lambda *a, **kw: {})
    request = fixture._request(project)
    request = dataclasses.replace(request, options=dataclasses.replace(request.options,
        public_owned_host_inputs=str(request_file), public_owned_host_inputs_sha256=pin(request_file)["sha256"]))
    observed = []

    def continuation(prepared):
        prepared.verify_inputs()
        record = prepared.environment[inputs.KEY]
        assert prepared.private_full_model_spec is None
        assert json.loads(os.environ[transport.ENV]) == record["selection"]
        assert inputs.from_run(prepared.run_dir, target="fixture", descriptor=descriptor) == record["selection"]
        script = "import json,os,pathlib,hashlib;r=json.loads(os.environ['MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION']);p=pathlib.Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];s=json.loads(p.read_bytes());assert set(s)=={'board','arena_mb','dts','board_catalog','rtl_facts','instruction_declaration'};print(s['board'])"
        child = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, check=True)
        assert child.stdout.strip() == "native"
        argv = BW.base_argv(prepared.workspace, prepared.bundle, repo=project)
        for row in [record["request"], record["selection"], *record["inputs"].values()]:
            assert not BW.is_exposed(argv, Path(row["path"]))
        assert not BW.is_exposed(argv, request_file)
        observed.append(record)
        return 0

    assert fixture._run(request, continuation) == 0
    assert len(observed) == 1
