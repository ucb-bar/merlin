"""Exercise process execution and resume using harmless local Python programs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from merlin_experiments import SpecError, load_spec
from merlin_experiments.adapters import ADAPTERS, Adapter, Option
from merlin_experiments.cli import main
from merlin_experiments.runner import preflight, resolve_plan, resume, run, status


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    engine = tmp_path / "engine.py"
    engine.write_text(
        "import argparse, pathlib, sys\n"
        "p=argparse.ArgumentParser(); p.add_argument('--receipt'); p.add_argument('--number')\n"
        "a=p.parse_args(); f=pathlib.Path(a.receipt)/('calls-'+a.number)\n"
        "n=int(f.read_text())+1 if f.exists() else 1; f.write_text(str(n))\n"
        "print('executed', a.number, n)\n"
        "sys.exit(7 if a.number=='1' and n==1 else 0)\n"
    )
    receipt = tmp_path / "receipt"
    receipt.mkdir()
    control = tmp_path / "immutable.txt"
    control.write_text("original")
    phases = {}
    for number in ("0", "1", "2"):
        name = f"fixture{number}"
        monkeypatch.setitem(
            ADAPTERS,
            name,
            Adapter(
                name,
                number,
                "engine.py",
                {"receipt": Option("workspace", True), "number": Option(required=True)},
            ),
        )
        phases[number] = {"adapter": name, "config": {"receipt": "receipt", "number": number}}
    definition = tmp_path / "workflow.yaml"
    document = {
        "schema_version": 1,
        "id": "test-workflow",
        "target": "test-target",
        "inputs": {"control": "immutable.txt"},
        "phases": phases,
    }
    definition.write_text(yaml.safe_dump(document))
    return definition, tmp_path / "run", control, engine


def test_inspect_preflight_run_status_resume_processes(workflow, capsys):
    definition, destination, _, _ = workflow
    assert main(["inspect", str(definition), "--run-dir", str(destination)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["phases"]["1"]["argv"][0] == sys.executable
    assert not destination.exists()
    assert main(["preflight", str(definition)]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["configuration_ready"] and checked["engine_readiness"] == "not_executed"
    assert main(["run", str(definition), "--run-dir", str(destination)]) == 7
    failed = json.loads(capsys.readouterr().out)
    assert failed["state"] == "execution_failed"
    assert [row["phase"] for row in failed["attempts"]] == ["0", "1"]
    assert main(["resume", str(destination)]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["state"] == "execution_succeeded"
    assert [row["phase"] for row in resumed["attempts"]] == ["0", "1", "1", "2"]
    assert "scientific verdict" in resumed["evidence_authority"]
    assert (definition.parent / "receipt/calls-0").read_text() == "1"
    assert main(["status", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out) == resumed
    assert resume(destination) == 0
    assert len(status(destination)["attempts"]) == 4


def test_phase1_launch_override_rejects_multiphase_plan(workflow):
    definition, destination, _, _ = workflow
    with pytest.raises(SpecError, match="Phase 1-only"):
        resolve_plan(load_spec(definition), run_dir=destination, phase1_driver="codex")


def test_completed_portfolio_segment_can_resume_from_explicit_checkpoint(tmp_path, monkeypatch):
    from merlin_experiments import portfolio_catalog

    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(portfolio_catalog, "verify_plan", lambda _plan: None)
    engine = tmp_path / "segment_engine.py"
    engine.write_text(
        "import argparse, hashlib, json, pathlib\n"
        "from merlin.benchharness import hash_tree\n"
        "p=argparse.ArgumentParser(); p.add_argument('--output'); "
        "p.add_argument('--resume-checkpoint')\n"
        "a=p.parse_args(); out=pathlib.Path(a.output); out.mkdir(parents=True)\n"
        "records=out/'global_iterations'; records.mkdir()\n"
        "candidate=records/'blocked_submission'; candidate.mkdir()\n"
        "(candidate/'compiler.py').write_text('synthetic compiler for '+out.name+'\\n')\n"
        "candidate_sha=hash_tree(candidate)['sha256']\n"
        "checkpoint=records/'round_0000_authoring.json'\n"
        "raw=json.dumps({'schema':'global_authoring_checkpoint_v1',"
        "'candidate_path':str(candidate),'candidate_sha256':candidate_sha}).encode()\n"
        "checkpoint.write_bytes(raw)\n"
        "sequence={'schema':'global_agent_sequence_v1','status':'budget_complete',"
        "'promotion_ready':False,'last_good_checkpoint':{'path':str(checkpoint),"
        "'sha256':hashlib.sha256(raw).hexdigest(),'candidate_sha256':candidate_sha}}\n"
        "(records/'agent_sequence.json').write_text(json.dumps(sequence))\n"
        "(out/'launch.json').write_text('{}')\n"
        "(out/'host_resource_telemetry.json').write_text(json.dumps("
        "{'status':'completed','worker_returncode':0}))\n"
    )
    destination = tmp_path / "run"
    first = destination / "phase2/segment-0001"
    monkeypatch.setitem(
        ADAPTERS,
        "fixture_segment",
        Adapter(
            "fixture_segment",
            "2",
            "segment_engine.py",
            {"output": Option(required=True)},
            resume="checkpoint_segment",
        ),
    )
    definition = tmp_path / "definition.yaml"
    definition.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "segment-fixture",
                "target": "synthetic",
                "phases": {"2": {"adapter": "fixture_segment", "config": {"output": str(first)}}},
            }
        )
    )
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    plan["phases"]["2"].update(adapter="model_portfolio", module="synthetic.installed.segment")
    assert run(plan) == 0
    assert status(destination)["phases"]["2"]["engine_output"] == str(first)
    assert resume(destination) == 0
    assert len(status(destination)["attempts"]) == 1
    # A selected checkpoint may be a separately retained copy; the native
    # portfolio verifier owns its scientific/lineage admission.
    checkpoint = tmp_path / "selected-checkpoint.json"
    checkpoint.write_bytes((first / "global_iterations/round_0000_authoring.json").read_bytes())
    assert resume(destination, checkpoint=checkpoint) == 0
    second = destination / "phase2/segment-0002"
    attempts = status(destination)["attempts"]
    assert [entry["engine_output"] for entry in attempts] == [str(first), str(second)]
    assert all("terminal_outputs" in entry for entry in attempts)
    assert attempts[1]["resume_checkpoint"]["path"] == str(checkpoint)
    assert resume(destination) == 0
    # The native worker does not write a final review seal when a recovered
    # round left failures, even if its last checkpoint is promotion-ready.
    from merlin_experiments.runner import _installed_phase2_output

    sequence_path = first / "global_iterations/agent_sequence.json"
    sequence_bytes = sequence_path.read_bytes()
    recovered = json.loads(sequence_bytes)
    recovered.update(promotion_ready=True, failures=[{"recovery": "next_budgeted_round_from_consumed_checkpoint"}])
    sequence_path.write_text(json.dumps(recovered))
    assert "global_candidate" not in _installed_phase2_output(plan["phases"]["2"], attempts[0])
    recovered["failures"] = []
    sequence_path.write_text(json.dumps(recovered))
    with pytest.raises(SpecError, match="global_candidate"):
        _installed_phase2_output(plan["phases"]["2"], attempts[0])
    sequence_path.write_bytes(sequence_bytes)
    candidate = first / "global_iterations/blocked_submission/compiler.py"
    original = candidate.read_bytes()
    candidate.write_text("changed after segment 2\n")
    with pytest.raises(SpecError, match="completed portfolio candidate differs|terminal bytes changed"):
        resume(destination)
    candidate.write_bytes(original)
    # The native scientific digest skips build/, but the orchestration pin
    # must still detect any later bytes under the selected snapshot.
    ignored = candidate.parent / "build"
    ignored.mkdir()
    (ignored / "codegen.py").write_text("changed executable state\n")
    with pytest.raises(SpecError, match="terminal bytes changed"):
        resume(destination)
    (ignored / "codegen.py").unlink()
    ignored.rmdir()
    linked = candidate.parent / "linked.py"
    linked.symlink_to(candidate)
    with pytest.raises(SpecError, match="terminal tree contains a link"):
        resume(destination)
    linked.unlink()
    checkpoint.write_text("changed checkpoint bytes\n")
    with pytest.raises(SpecError, match="previous portfolio resume checkpoint bytes changed"):
        resume(destination)
    original_checkpoint = (first / "global_iterations/round_0000_authoring.json").read_bytes()
    checkpoint.write_bytes(original_checkpoint)
    from merlin_experiments import runner as runner_module

    original_fingerprint = runner_module.fingerprint

    def unavailable(path):
        if Path(path) == checkpoint:
            raise OSError("selected checkpoint cannot be read")
        return original_fingerprint(path)

    monkeypatch.setattr(runner_module, "fingerprint", unavailable)
    with pytest.raises(SpecError, match="resume checkpoint is unavailable or changed"):
        resume(destination)
    assert len(status(destination)["attempts"]) == 2


@pytest.mark.parametrize("mutation", ["changed_input", "missing_input", "changed_script", "changed_definition"])
def test_resume_rejects_drift(workflow, mutation):
    definition, destination, control, engine = workflow
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 7
    if mutation == "missing_input":
        control.unlink()
    else:
        changed = {"changed_input": control, "changed_script": engine, "changed_definition": definition}[mutation]
        changed.write_text(changed.read_text() + "\nchanged")
    with pytest.raises(SpecError, match="changed|absent"):
        resume(destination)
    assert len(status(destination)["attempts"]) == 2


def test_frozen_plan_cannot_be_changed(workflow):
    definition, destination, _, _ = workflow
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 7
    frozen = destination / "resolved-plan.json"
    frozen.write_text(frozen.read_text() + " ")
    with pytest.raises(SpecError, match="plan changed"):
        resume(destination)


def test_missing_input_refuses_before_creating_run(workflow):
    definition, destination, control, _ = workflow
    control.unlink()
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    assert not preflight(plan)["configuration_ready"]
    with pytest.raises(SpecError, match="preflight failed"):
        run(plan)
    assert not destination.exists()


def test_definition_resolution_is_independent_of_cwd(workflow, monkeypatch, tmp_path):
    definition, destination, control, _ = workflow
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    plan = resolve_plan(load_spec(definition), phase="0", run_dir=destination)
    assert plan["input_paths"]["declared:control"] == str(control)
    assert list(plan["phases"]) == ["0"]


def test_editable_workspace_cannot_overlap_a_declared_frozen_input(workflow):
    definition, destination, _, _ = workflow
    document = yaml.safe_load(definition.read_text())
    document["inputs"]["must_remain_immutable"] = "receipt"
    definition.write_text(yaml.safe_dump(document))
    with pytest.raises(SpecError, match="mutable output/workspace.*overlaps frozen input"):
        resolve_plan(load_spec(definition), run_dir=destination)


def test_native_engine_output_cannot_overlap_frozen_input_even_with_external_run_dir(workflow, monkeypatch):
    from merlin_experiments import runner

    definition, destination, _, _ = workflow
    protected = definition.parent / "native-output"
    protected.mkdir()
    document = yaml.safe_load(definition.read_text())
    document["inputs"]["must_remain_immutable"] = protected.name
    definition.write_text(yaml.safe_dump(document))
    original = ADAPTERS["fixture0"]

    class NativeOutput:
        name = original.name

        def resolve(self, *args):
            command = original.resolve(*args)
            command["engine_output"] = str(protected / "native-run")
            return command

    spec = load_spec(definition)
    monkeypatch.setitem(runner.ADAPTERS, "fixture0", NativeOutput())
    with pytest.raises(SpecError, match="mutable output/workspace.*overlaps frozen input"):
        resolve_plan(spec, phase="0", run_dir=destination)


def test_schema_rejects_unknown_commands_modes_and_options(workflow):
    definition, _, _, _ = workflow
    original = yaml.safe_load(definition.read_text())
    for mutate in (
        lambda doc: doc.update(command="echo no"),
        lambda doc: doc.update(schema_version=2),
        lambda doc: doc["phases"]["0"].update(mode="measured_claims"),
        lambda doc: doc["phases"]["0"]["config"].update(shell=True),
        lambda doc: doc["phases"]["0"].update(adapter="model_portfolio"),
    ):
        document = json.loads(json.dumps(original))
        mutate(document)
        definition.write_text(yaml.safe_dump(document))
        with pytest.raises(SpecError):
            load_spec(definition)


def test_catalog_resolves_paths_from_catalog(workflow, capsys):
    definition, _, _, _ = workflow
    index = definition.parent / "catalog.yaml"
    index.write_text(yaml.safe_dump({"schema_version": 1, "experiments": {"test-workflow": definition.name}}))
    assert main(["--catalog", str(index), "list"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == "test-workflow"


def test_phase1_and_phase2_real_adapter_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    from merlin_experiments.adapters import _legacy_entrypoints

    for name in ("capsule_bench", "model_portfolio"):
        for relative in _legacy_entrypoints()[name].values():
            entrypoint = tmp_path / relative
            entrypoint.parent.mkdir(parents=True, exist_ok=True)
            entrypoint.write_text("# transport-only fixture; not executed\n")
            if name == "capsule_bench":
                for helper in ("_common.py", "sandbox_toolchain.py"):
                    (entrypoint.parent / helper).write_text("# pinned transport-only startup fixture; not executed\n")
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: sample\ncapsule_corpus: corpus/isa\n")
    (tmp_path / "corpus/isa").mkdir(parents=True)
    spec_path = tmp_path / "experiment.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "sample-functional",
                "kind": "template",
                "target": "sample",
                "phases": {
                    1: {
                        "adapter": "capsule_bench",
                        "config": {
                            "descriptor": "target.yaml",
                            "bundle": "merlin_assisted_rtlchecks_fixture",
                            "bundle_manifest": "bundle/input_bundle_manifest.yaml",
                            "oracle_timing": "timing.json",
                            "arm": "merlin_assisted",
                            "treatment": "rtlchecks",
                            "model": "declared-model",
                            "effort": "high",
                            "max_wall_s": 60,
                            "round_timeout": 60,
                        },
                    },
                    2: {
                        "adapter": "model_portfolio",
                        "mode": "model_portfolio",
                        "config": {
                            "campaign_config": "campaign.json",
                            "deployment": "deployment.json",
                            "candidate": "candidate",
                            "round_seconds": 60,
                            "max_tool_calls": 5,
                            "max_rounds": 1,
                            "total_authoring_seconds": 60,
                        },
                    },
                },
            }
        )
    )
    clang = tmp_path / "clang-23"
    clang.write_text("#!/bin/sh\nexit 0\n")
    clang.chmod(0o755)
    monkeypatch.setenv("MERLIN_CLANG", str(clang))
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(tmp_path / "unselected-support"))
    monkeypatch.setenv("MERLIN_CAPSULE_TIER_FLOOR", "L3")
    monkeypatch.setenv("MERLIN_GRADE_WORKERS", "16")
    monkeypatch.setenv("MERLIN_ORACLE_JAIL_ROOTS", str(tmp_path / "toolchain"))
    (tmp_path / "toolchain").mkdir()
    (tmp_path / "toolchain" / "spike").write_text("frozen")
    monkeypatch.setenv("MERLIN_ORACLE_TOOLCHAIN", str(tmp_path / "toolchain"))
    target_simulator = f"MERLIN_{load_spec(spec_path).target.upper()}_SIMV"
    monkeypatch.setenv(target_simulator, str(tmp_path / "toolchain" / "simv"))
    plan = resolve_plan(load_spec(spec_path), run_dir=tmp_path / "run")
    functional = plan["phases"]["1"]
    assert functional["argv"][1:3] == ["-m", "merlin_experiments.phase1"]
    assert functional["argv"][functional["argv"].index("--treatment") + 1] == "rtlchecks"
    assert "--continuous" not in functional["argv"]
    assert functional["argv"][functional["argv"].index("--schedule") + 1] == "continuous"
    assert functional["env"]["MERLIN_TARGET_EXPERIMENT"] == str(descriptor)
    assert functional["env"]["MERLIN_CLANG"] == str(clang)
    assert functional["env"]["MERLIN_TARGET_PATH"] == str(tmp_path / "unselected-support")
    assert functional["env"]["MERLIN_CAPSULE_TIER_FLOOR"] == "L3"  # a resume cannot drop the floor
    assert functional["env"]["MERLIN_GRADE_WORKERS"] == "16"  # nor the selected grade parallelism
    assert functional["env"]["MERLIN_ORACLE_JAIL_ROOTS"] == str(tmp_path / "toolchain")  # nor the simulator jail
    assert functional["env"][target_simulator] == str(tmp_path / "toolchain" / "simv")  # nor the target's tools
    assert plan["input_paths"]["phase1:oracle_toolchain"] == str(tmp_path / "toolchain")  # bytes re-verified on resume
    monkeypatch.setenv("MERLIN_CAPSULE_TIER_FLOOR", "RTL")
    with pytest.raises(SpecError, match="MERLIN_CAPSULE_TIER_FLOOR"):
        resolve_plan(load_spec(spec_path), run_dir=tmp_path / "run-invalid-floor")
    assert plan["input_paths"]["phase1:operator:clang"] == str(clang)
    assert plan["phases"]["2"]["resume_policy"] == "checkpoint_segment"
    assert plan["phases"]["2"]["mode"] == "model_portfolio"
    assert "--root" not in plan["phases"]["2"]["argv"]


def test_resume_refuses_a_surviving_engine(workflow):
    definition, destination, _, _ = workflow
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 7
    path = destination / "orchestration.json"
    record = json.loads(path.read_text())
    record["attempts"][-1].update(state="running", pid=os.getpid())
    path.write_text(json.dumps(record))
    with pytest.raises(SpecError, match="still running"):
        resume(destination)


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError])
def test_post_spawn_bookkeeping_failure_retains_storage_ownership(workflow, monkeypatch, failure_type):
    from merlin_experiments import runner

    from merlin.common.storage_lifecycle import inventory

    definition, destination, _, engine = workflow
    engine.write_text("import time\ntime.sleep(30)\n")
    original_write = runner._write_json
    observed_pids = []

    def fail_after_spawn(path, value):
        attempts = value.get("attempts", [])
        if attempts and attempts[-1].get("pid"):
            observed_pids.append(attempts[-1]["pid"])
            raise failure_type("injected post-spawn state failure")
        original_write(path, value)

    monkeypatch.setattr(runner, "_write_json", fail_after_spawn)
    with pytest.raises((SpecError, RuntimeError), match="retained|injected"):
        run(resolve_plan(load_spec(definition), phase="0", run_dir=destination))
    assert observed_pids, "negative control must actually spawn a process"
    assert not runner._process_active(observed_pids[0]), "best-effort teardown should stop the known child"
    row = next(row for row in inventory()["paths"] if row["path"] == str(destination))
    assert row["leases"] == 1 and row["state"] == "running"
    with pytest.raises(SpecError, match="unresolved engine ownership"):
        resume(destination)


def test_resume_rejects_a_changed_output_registry(workflow, monkeypatch):
    from merlin.common.storage_lifecycle import inventory

    definition, destination, _, _ = workflow
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 7
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(definition.parent / "different-out"))
    with pytest.raises(SpecError, match="frozen storage root"):
        resume(destination)
    assert inventory()["paths"] == []


def test_driver_exit_does_not_release_a_live_worker_group(workflow):
    from merlin_experiments import runner

    from merlin.common.storage_lifecycle import inventory

    definition, destination, _, engine = workflow
    engine.write_text(
        "import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
    )
    with pytest.raises(SpecError, match="workers remain; storage lease retained"):
        run(resolve_plan(load_spec(definition), phase="0", run_dir=destination))
    row = next(row for row in inventory()["paths"] if row["path"] == str(destination))
    assert row["leases"] == 1
    pid = status(destination)["attempts"][-1]["pid"]
    assert not runner._process_active(pid)


@pytest.mark.parametrize("policy", ["native_flag", "native_chain"])
def test_native_resume_policies_use_real_processes(tmp_path, monkeypatch, policy):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    engine = tmp_path / "engine.py"
    engine.write_text(
        "import argparse, pathlib, sys, json\n"
        "p=argparse.ArgumentParser(); p.add_argument('--receipt'); p.add_argument('--output')\n"
        "p.add_argument('--resume',action='store_true'); p.add_argument('--resume-checkpoint')\n"
        "a=p.parse_args(); r=pathlib.Path(a.receipt); seen=r/'seen.json'\n"
        "first=not seen.exists(); seen.write_text(json.dumps(vars(a)))\n"
        "sys.exit(7 if first else 0)\n"
    )
    receipt = tmp_path / "receipt"
    receipt.mkdir()
    name = "fixture"
    monkeypatch.setitem(
        ADAPTERS,
        name,
        Adapter(
            name,
            "2",
            "engine.py",
            {"receipt": Option("workspace", True)},
            resume=policy,
        ),
    )
    phase = {"adapter": name, "config": {"receipt": "receipt"}}
    definition = tmp_path / "experiment.yaml"
    definition.write_text(
        yaml.safe_dump({"schema_version": 1, "id": "resume-policies", "target": "sample", "phases": {2: phase}})
    )
    destination = tmp_path / "run"
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 7
    assert resume(destination) == 0
    observed = json.loads((receipt / "seen.json").read_text())
    assert observed["resume"] == (policy == "native_flag")


def test_production_flags_exist_in_legacy_argparse_contract(tmp_path):
    import ast

    from merlin_experiments.spec import ExperimentSpec

    from merlin.common.paths import module_source_path, repo_root

    root = repo_root()
    spec = ExperimentSpec(tmp_path / "spec.yaml", {"id": "adapter-contract", "target": "sample"})
    for adapter in ADAPTERS.values():
        config = {}
        for name, option in adapter.options.items():
            if not option.required:
                continue
            value = 1 if option.kind == "positive" else "value"
            if option.choices:
                value = option.choices[0]
            if name.endswith("sha256"):
                value = "0" * 64
            config[name] = value
        if adapter.name == "capsule_bench":
            config.update(
                bundle="raw_baseline_fixture", bundle_manifest="input_bundle_manifest.yaml", oracle_timing="timing.json"
            )
        adapter.validate(config)
        if adapter.name == "capsule_derivation":
            config["descriptor"] = str(root / "merlin/experiments/capsule_bench/targets/sample/target_experiment.yaml")
            config["profiles_root"] = str(tmp_path / "profiles")
        if adapter.name == "measured_claims":
            config.update(
                core_package_root=str(module_source_path("merlin").parent),
                experiments_package_root=str(module_source_path("merlin_experiments").parent),
                experiments_namespace_root=str(module_source_path("merlin.targetgen.capsule_runner").parent.parent),
            )
        if adapter.name == "model_portfolio":
            from merlin_experiments.phase2.portfolio_options import build_parser as portfolio_parser

            flags = {flag for action in portfolio_parser()._actions for flag in action.option_strings}
            declared = {
                option.flag or "--" + name.replace("_", "-")
                for name, option in adapter.options.items()
                if option.flag != ""
            }
            assert declared <= flags
            continue
        command = adapter.resolve(spec, config, root, tmp_path / "run")
        if adapter.name == "capsule_bench":
            from merlin_experiments.phase1.options import build_parser

            flags = {flag for action in build_parser()._actions for flag in action.option_strings}
            # Additional explicit-input flags belong to the installed command, not native options.
            tree = ast.parse(Path(command["entrypoint"]).read_text())
            flags.update(
                arg.value
                for call in ast.walk(tree)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_argument"
                for arg in call.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            )
        elif adapter.name == "model_portfolio":
            from merlin_experiments.phase2.portfolio_options import build_parser as portfolio_parser

            flags = {flag for action in portfolio_parser()._actions for flag in action.option_strings}
        else:
            entrypoint = Path(command["entrypoint"])
            if adapter.name == "whole_model_measured":
                # The mode's __main__ only dispatches; its parser lives in its cli module.
                entrypoint = module_source_path("merlin_experiments.phase2.whole_model_measured.cli")
            tree = ast.parse(entrypoint.read_text())
            flags = {
                arg.value
                for call in ast.walk(tree)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_argument"
                for arg in call.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            }
        if adapter.name == "measured_claims":
            for module in ("chia_envelope", "checkpoint_cli"):
                tree = ast.parse(module_source_path(f"merlin_experiments.phase2.{module}").read_text())
                flags.update(
                    arg.value
                    for call in ast.walk(tree)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "add_argument"
                    for arg in call.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                )
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "resource_names" for target in node.targets
                    ):
                        flags.update("--" + name.replace("_", "-") for name in ast.literal_eval(node.value))
        emitted = {arg for arg in command["argv"] if arg.startswith("--") and arg != "--"}
        assert emitted <= flags, (adapter.name, emitted - flags)


def test_templates_are_discoverable_but_cannot_execute(workflow):
    definition, destination, _, _ = workflow
    document = yaml.safe_load(definition.read_text())
    document["kind"] = "template"
    definition.write_text(yaml.safe_dump(document))
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    with pytest.raises(SpecError, match="template cannot execute"):
        run(plan)
    assert not destination.exists()


@pytest.mark.parametrize(
    "name", ["measured-claims-template", "model-portfolio-template", "whole-model-measured-template"]
)
def test_phase2_templates_require_operator_target_and_refuse_execution(name, tmp_path, monkeypatch):
    from merlin_experiments import runner

    from merlin.common.paths import repo_root

    definition = repo_root() / "experiments" / "definitions" / f"{name}.yaml"
    spec = load_spec(definition)
    assert spec.target == "OPERATOR_TARGET"
    assert spec.document["kind"] == "template"
    if name == "measured-claims-template":
        assert spec.document["phases"]["2"]["config"]["descriptor"] == ("../operator-inputs/target_experiment.yaml")
    destination = tmp_path / "must-not-exist"
    plan = resolve_plan(spec, phase="2", run_dir=destination)
    monkeypatch.setattr(runner, "preflight", lambda *_: pytest.fail("template reached preflight"))
    with pytest.raises(SpecError, match="template cannot execute"):
        run(plan)
    assert not destination.exists()


def test_target_catalog_definitions_have_one_home_in_examples():
    from merlin_experiments.cli import catalog

    from merlin.common.paths import repo_root

    root = repo_root()
    target_specs = [path for path in catalog().values() if load_spec(path).document.get("kind") != "template"]
    assert set(target_specs) == set((root / "examples").glob("*/experiment.yaml"))
    for path in target_specs:
        assert path.parent.parent == root / "examples"
        assert path.name == "experiment.yaml"
        spec = load_spec(path)
        assert not (root / "experiments" / "definitions" / f"{spec.id}.yaml").exists()


def test_catalog_examples_declare_operator_prerequisites_not_ready_runs():
    from merlin_experiments.cli import catalog
    from merlin_experiments.phase0.declarations import from_definition
    from merlin_experiments.phase1.source_inputs import fingerprint

    for path in catalog().values():
        spec = load_spec(path)
        if spec.document.get("kind") == "template":
            assert not preflight(resolve_plan(spec))["configuration_ready"]
        else:
            if "0" in spec.document["phases"]:
                assert from_definition(path).recipe.is_file()
                phase0_readiness = preflight(resolve_plan(spec, phase="0"))
                assert phase0_readiness["configuration_ready"] or (
                    phase0_readiness["errors"]
                    and all("selected synthesis is unverified_legacy" in error for error in phase0_readiness["errors"])
                )
            else:
                with pytest.raises(SpecError, match="not a concrete capsule derivation definition"):
                    from_definition(path)
            config = spec.document["phases"]["1"]["config"]
            assert config["treatment"] == "rtlchecks"
            assert config["bundle"] == "merlin_assisted_rtlchecks_public_v0"
            try:
                functional = resolve_plan(spec, phase="1")
            except SpecError as exc:
                assert str(exc) in {
                    "phase-1 selected target contract is not agreed and resolvable: none",
                    "invalid Phase 1 capability selection: declared capability contract is absent; "
                    "prepare a new release",
                }
                continue
            command = functional["phases"]["1"]
            assert command["module"] == "merlin_experiments.phase1"
            assert command["argv"][1:3] == ["-m", "merlin_experiments.phase1"]
            assert command["argv"][command["argv"].index("--treatment") + 1] == "rtlchecks"
            assert command["argv"][command["argv"].index("--bundle") + 1] == config["bundle"]
            assert not any(name.startswith("phase1:native:") for name in functional["input_paths"])
            expected_errors = set()
            if config.get("require_reviewed_corpus"):
                expected_errors.add(
                    "this functional experiment requires a reviewed Phase 0 release; "
                    "select its seal and release descriptor before starting Phase 1"
                )
            declared_inputs = set(command["inputs"].values())
            for name, value in functional["input_paths"].items():
                if name.startswith("phase1:operator:") or value in declared_inputs:
                    try:
                        fingerprint(value)
                    except (SpecError, OSError) as exc:
                        expected_errors.add(str(exc))
                else:
                    # Implementation and other nonoperator pins must still be valid.
                    fingerprint(value)
            from merlin.targetgen.plugins import resolve_support

            try:
                if not resolve_support(spec.target).plugin().get("backend"):
                    expected_errors.add(
                        f"phase 1 EL4 target {spec.target!r} has no selected support plugin.backend; "
                        "RTL checks cannot run from the metadata-only example. Select a reviewed OOT support provider."
                    )
            except (OSError, ValueError) as exc:
                expected_errors.add(f"phase 1 EL4 support for {spec.target!r} cannot be resolved: {exc}")
            readiness = preflight(functional)
            assert set(readiness["errors"]) == expected_errors
            assert readiness["configuration_ready"] is (not expected_errors)
            assert not preflight(resolve_plan(spec))["configuration_ready"]


def test_phase0_consumes_explicit_out_of_tree_descriptor(tmp_path):
    import argparse
    import ast
    from pathlib import Path

    from merlin.common.paths import module_source_path

    source = module_source_path("merlin_experiments.phase0.generation")
    tree = ast.parse(source.read_text())
    selected = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    descriptor = tmp_path / "external-target.yaml"
    descriptor.write_text("target: external\n")
    from merlin_experiments.phase0.profiles import validate_profile_inputs

    namespace = {"argparse": argparse, "Path": Path, "validate_profile_inputs": validate_profile_inputs}
    seen = []

    class SelectedDescriptor(Exception):
        pass

    def capture_path(path):
        seen.append(path)
        raise SelectedDescriptor

    namespace["_descriptor_for"] = lambda target: tmp_path / "legacy.yaml"
    namespace["_ensure_contract_on_path"] = capture_path
    function = ast.Module(body=[selected["generate_target"]], type_ignores=[])
    exec(compile(function, str(source), "exec"), namespace)
    with pytest.raises(SelectedDescriptor):
        namespace["generate_target"](
            "shared-profile", descriptor=descriptor, output_root=tmp_path / "artifacts", profiles_root=tmp_path
        )
    assert seen == [descriptor]
    with pytest.raises(SelectedDescriptor):
        namespace["generate_target"]("shared-profile", output_root=tmp_path / "artifacts", profiles_root=tmp_path)
    assert seen[-1] == tmp_path / "legacy.yaml"

    calls = []
    namespace["generate_target"] = lambda target, **kw: calls.append((target, kw)) or []
    namespace["profile_targets"] = lambda **kwargs: ["shared-profile"]
    main_source = module_source_path("merlin_experiments.phase0.__main__")
    main_node = next(
        node
        for node in ast.parse(main_source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_function = ast.Module(body=[main_node], type_ignores=[])
    exec(compile(main_function, str(source), "exec"), namespace)
    output = tmp_path / "artifacts"
    profiles = tmp_path / "profiles"
    assert (
        namespace["main"](
            [
                "--target",
                "shared-profile",
                "--descriptor",
                str(descriptor),
                "--output-root",
                str(output),
                "--profiles-root",
                str(profiles),
            ]
        )
        == 0
    )
    assert calls == [("shared-profile", {"descriptor": descriptor, "output_root": output, "profiles_root": profiles})]
    with pytest.raises(SystemExit) as exc:
        namespace["main"](["--descriptor", str(descriptor)])
    assert exc.value.code == 2


def test_phase0_output_and_manifest_are_run_owned(tmp_path):
    import ast
    import copy
    from types import SimpleNamespace

    from merlin_experiments.phase0.generation import _require_distinct_corpus_destinations
    from merlin_experiments.phase0.profiles import validate_profile_inputs
    from merlin_experiments.spec import ExperimentSpec

    from merlin.common.paths import module_source_path, python_source_dir, repo_root

    source = module_source_path("merlin_experiments.phase0.generation")
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_target")
    canonical = tmp_path / "canonical"
    output = tmp_path / "run" / "phase0" / "capsules"
    recorded = []
    hardware_targets = []
    namespace = {
        "__name__": "merlin_experiments.phase0.generation",
        "__package__": "merlin_experiments.phase0",
        "Path": Path,
        "copy": copy,
        "validate_profile_inputs": validate_profile_inputs,
        "_descriptor_for": lambda _: tmp_path / "descriptor.yaml",
        "_ensure_contract_on_path": lambda _: None,
        "load_target_experiment": lambda _: SimpleNamespace(
            capsule_corpus=canonical / "isa", target="external-hardware"
        ),
        "_require_distinct_corpus_destinations": _require_distinct_corpus_destinations,
        "load_profile": lambda _, **kwargs: {},
        "CS": SimpleNamespace(derive_binding=lambda *args: None),
        "_performance_facts": lambda target: hardware_targets.append(target) or {"sha256": "0" * 64},
        "expand_sweeps": lambda *args, **kwargs: [],
        "assert_no_claim_capsules": lambda *args, **kwargs: None,
        "held_out_models": lambda te: [],
        "validate_roles": lambda roles: [],
        "_instruction_policy": lambda *args: {"status": "none_declared"},
        "_with_candidate_policy": lambda entry, policy: entry,
        "_prune_superseded_synth": lambda *args, **kwargs: [],
        "update_provenance_manifest": lambda *args, **kwargs: recorded.append(kwargs),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["generate_target"]("profile", output_root=output, profiles_root=tmp_path) == []
    assert recorded[-1]["cap_root"] == output
    assert output.is_dir()
    assert not canonical.exists()
    assert hardware_targets == ["profile"]
    namespace["generate_target"](
        "profile", descriptor=tmp_path / "descriptor.yaml", output_root=output, profiles_root=tmp_path
    )
    assert hardware_targets[-1] == "external-hardware"
    assert recorded[-1]["target"] == "external-hardware"
    spec = ExperimentSpec(tmp_path / "spec.yaml", {"id": "example", "target": "sample"})
    command = ADAPTERS["capsule_derivation"].resolve(
        spec, {"descriptor": "descriptor.yaml", "profiles_root": "profiles"}, repo_root(), tmp_path / "run"
    )
    assert command["argv"][command["argv"].index("--output-root") + 1] == str(output)
    assert command["engine_output"] == str(output)
    assert str(python_source_dir()) in command["env"]["PYTHONPATH"].split(os.pathsep)


def test_phase0_requires_explicit_inputs_even_in_a_checkout(tmp_path, monkeypatch):
    from merlin_experiments.spec import ExperimentSpec

    from merlin.common.paths import checkout_root

    checkout = checkout_root()
    assert checkout is not None
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: sample\n")
    config = {"descriptor": str(descriptor), "profile": "shared-profile"}
    spec = ExperimentSpec(
        tmp_path / "definition.yaml",
        {
            "id": "profile-default",
            "target": "sample",
            "phases": {"0": {"adapter": "capsule_derivation", "config": config}},
        },
    )
    for explicit in (False, True):
        if not explicit:
            with pytest.raises(SpecError, match="explicit recipe inputs"):
                resolve_plan(spec, phase="0", run_dir=tmp_path / "run")
            continue
        expected = tmp_path / "external-profiles"
        if explicit:
            expected = tmp_path / "external-profiles"
            config["profiles_root"] = str(expected)
        plan = resolve_plan(spec, phase="0", run_dir=tmp_path / "run")
        command = plan["phases"]["0"]
        assert command["argv"][command["argv"].index("--profiles-root") + 1] == str(expected)
        assert plan["input_paths"]["phase0:profiles"] == str(expected)
        assert plan["input_paths"]["phase0:target_profile"] == str(expected / "shared-profile.yaml")


def test_unsealed_phase0_to_phase1_handoff_cannot_run(tmp_path):
    from merlin_experiments.cli import catalog

    definition = load_spec(
        next(path for path in catalog().values() if load_spec(path).document.get("kind") != "template")
    )
    destination = tmp_path / "unsealed-run"
    plan = resolve_plan(definition, run_dir=destination)
    with pytest.raises(SpecError, match="handoff is not sealed"):
        run(plan)
    assert not destination.exists()


def _without_resolvable_clang(monkeypatch, tmp_path):
    for name in ("MERLIN_CLANG", "MERLIN_IREE_BIN", "MERLIN_EXT_MERLIN_IREE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))


def test_unresolvable_clang_refuses_grading_phases_before_any_engine_starts(workflow, monkeypatch, tmp_path):
    # toolchain.clang() ends its fallback chain in a bare name, so a checkout without an install
    # used to reach grading and fail every capsule as a candidate compile error.
    definition, destination, _, _ = workflow
    _without_resolvable_clang(monkeypatch, tmp_path)
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    errors = [error for error in preflight(plan)["errors"] if "toolchain clang" in error]
    assert [error.split()[1] for error in errors] == ["1", "2"]
    assert all("'clang-23'" in error and str(tmp_path) in error for error in errors)
    with pytest.raises(SpecError, match="toolchain clang does not resolve"):
        run(plan)
    assert not destination.exists()
    assert not list((tmp_path / "receipt").iterdir())


def test_clang_resolves_from_the_engine_checkout_install(workflow, monkeypatch, tmp_path):
    definition, destination, _, _ = workflow
    _without_resolvable_clang(monkeypatch, tmp_path)
    compiler = tmp_path / "third_party" / "llvm-install" / "bin" / "clang-23"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("#!/bin/sh\nexit 0\n")
    compiler.chmod(0o644)
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    assert any("toolchain clang" in error for error in preflight(plan)["errors"])  # present, not executable
    compiler.chmod(0o755)
    assert preflight(plan)["configuration_ready"]
