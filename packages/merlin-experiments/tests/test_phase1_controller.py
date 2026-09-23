"""Real package controller/copy transport admission; explicitly untrusted diagnostics.

Copy this test into an installed core+experiments environment: no checkout fixture
or native controller is imported. The only executable substitute is a local,
fixed-output provider CLI. Oracle availability is declared false by fixture support.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import data_path, python_import_roots

DRIVER = r"""
import json, os, socket, subprocess
from pathlib import Path
def forbidden(*args, **kwargs):
    raise AssertionError("synthetic provider cannot launch processes or listeners")
subprocess.Popen=forbidden
socket.socket.bind=forbidden
root=Path(os.environ["CONTROLLER_FIXTURE_ROOT"])
calls=root/"provider_calls.jsonl"
first=not calls.exists()
workspace=Path.cwd()
with calls.open("a") as stream:
    stream.write(json.dumps({"workspace":str(workspace),"pid":os.getpid()})+"\n")
partial=workspace/"submission/partial.txt"
if first:
    partial.write_text("retained partial candidate\n")
    print(json.dumps({"type":"rate_limit_event","rate_limit_info":{"rateLimitType":"seven_day","status":"rejected"}}))
else:
    assert partial.read_text()=="retained partial candidate\n"
    print(json.dumps({"type":"assistant","message":{"model":"claude-fixture",
          "content":[{"type":"text","text":"Incomplete local fixture."}],
          "usage":{"input_tokens":1,"output_tokens":1}}}))
    print(json.dumps({"type":"result","subtype":"success","is_error":False,"result":"No compiler delivered."}))
"""

RUN = r"""
import importlib.abc,json,os,sys,socket,subprocess
from pathlib import Path
root=Path(sys.argv[1])
real_popen=subprocess.Popen
def forbidden(*args, **kwargs):
    raise AssertionError("controller diagnostic cannot launch native tools or listeners")
def local_provider_only(command, *args, **kwargs):
    if not isinstance(command,(list,tuple)) or not command or kwargs.get("shell"):
        return forbidden()
    cwd=kwargs.get("cwd")
    if cwd is None:
        return forbidden()
    workspace=Path(cwd)
    if not workspace.is_absolute() or not workspace.resolve().is_relative_to(root.resolve()):
        return forbidden()
    expected=(f"claude --print --model claude-fixture --effort high "
              f"--permission-mode bypassPermissions --add-dir {workspace} "
              f"--output-format stream-json --verbose < {workspace/'TASK.md'}")
    if list(command) != ["bash","-c",expected]:
        return forbidden()
    # Validate the exact production wrapper, then run ONLY our inert provider.
    # Shell parsing/redirection and the real provider transport are NOT qualified.
    # The synthetic script emits fixed events and does not consume task stdin.
    return real_popen([sys.executable,str(root/"bin/claude")],*args,**kwargs)
subprocess.Popen=local_provider_only
socket.socket.bind=forbidden
class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"_common","run_agent_experiment","run_baseline_qa_loop","sandbox_toolchain"}:
            raise AssertionError("native import forbidden: "+fullname)
sys.meta_path.insert(0,DenyNative())
from merlin_experiments.phase1.context import load_context
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.controller import run
root=Path(sys.argv[1])
argv=["--run-id","diagnostic","--arm","raw_baseline","--experiment","realistic",
      "--driver","claudecode","--model","claude-fixture","--schedule","rounds",
      "--max-rounds","1","--round-timeout","10","--qa-timeout","1",
      "--sandbox","none","--allow-unsandboxed","--no-oracle","--skip-hidden"]+sys.argv[2:]
interface=os.environ.get("CONTROLLER_FIXTURE_INTERFACE")
if interface in {"cli", "rtlchecks", "catalog-rtlchecks"}:
    import runpy
    if interface == "catalog-rtlchecks":
        from merlin_experiments.spec import catalog,load_spec
        from merlin_experiments.adapters import ADAPTERS,PHASE1_MODULE
        spec=load_spec(catalog(root/"catalog.yaml")["fixture-rtl"])
        phase=spec.document["phases"]["1"]
        command=ADAPTERS[phase["adapter"]].resolve(spec,phase["config"],root,root/"diagnostic")
        assert command["module"]==PHASE1_MODULE
        assert command["argv"][1:3]==["-m",PHASE1_MODULE]
        resolved=command["argv"][3:]
        assert resolved[resolved.index("--treatment")+1]=="rtlchecks"
        assert resolved[resolved.index("--bundle-manifest")+1]==str(root/"bundle/input_bundle_manifest.yaml")
        assert resolved[resolved.index("--oracle-timing")+1]==str(root/"timing.json")
        assert resolved[resolved.index("--sandbox")+1]=="bwrap"
        receipt=root/("catalog-resumed.json" if "--resume" in sys.argv else "catalog-fresh.json")
        receipt.write_text(json.dumps(command,sort_keys=True))
        # The real catalog command is retained above. Only this explicitly untrusted
        # copy-mode diagnostic replaces OS isolation and bounds the synthetic round.
        resolved[resolved.index("--sandbox")+1]="none"
        resolved[resolved.index("--schedule")+1]="rounds"
        resolved += ["--allow-unsandboxed","--no-oracle","--skip-hidden","--max-rounds","1",
                     "--public-root",str(root/"corpus/isa"),*sys.argv[2:]]
        os.environ.update(command["env"])
        sys.argv=[PHASE1_MODULE,*resolved]
        runpy.run_module(PHASE1_MODULE,run_name="__main__")
        raise AssertionError("catalog CLI must return through SystemExit")
    bundle="synthetic"
    if interface == "rtlchecks":
        argv[argv.index("--arm")+1]="merlin_assisted"
        argv += ["--treatment", "rtlchecks"]
        bundle="merlin_assisted_rtlchecks_fixture"
    sys.argv=["merlin_experiments.phase1", "--descriptor",str(root/"custom-descriptor.yaml"),
              "--repo",str(root),"--bundle-manifest",str(root/"bundle/input_bundle_manifest.yaml"),
              "--bundle",bundle,"--oracle-timing",str(root/"timing.json"),
              "--public-root",str(root/"corpus/isa"),*argv]
    runpy.run_module("merlin_experiments.phase1",run_name="__main__")
    raise AssertionError("CLI must return through SystemExit")
context=load_context(root/"custom-descriptor.yaml",repo=root)
options=parse_options(argv)
before=dict(os.environ)
try:
    result=run(context,options,bundle_manifest=root/"bundle/input_bundle_manifest.yaml",
               bundle_id="synthetic",oracle_timing=root/"timing.json",public_root=root/"corpus/isa")
finally:
    assert dict(os.environ)==before,"controller environment escaped invocation"
assert not {"_common","run_agent_experiment","run_baseline_qa_loop"}.intersection(sys.modules)
raise SystemExit(result)
"""


def _yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value))


def _project(root: Path):
    root.mkdir()
    shutil.copytree(data_path("contract", "schemas"), root / "merlin/contract/schemas")
    shutil.copytree(data_path("schemas"), root / "merlin/schemas")
    for label, name in (("public", "public_member"), ("hidden", "hidden_member")):
        capsule = root / "corpus" / ("isa" if label == "public" else "hidden") / name
        _yaml(
            capsule / "capsule.yaml",
            {
                "name": name,
                "kind": "isa",
                "source_role": "handauthored_compiler_test",
                "label": label,
                "operation": {"op": "matmul", "attributes": {}},
                "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
                "expected": {"instruction_classes": [], "modes": {}},
                "required_oracle_tiers": ["L0", "L2"],
                "interface_mlir": "capsule.interface.mlir",
            },
        )
        (capsule / "capsule.interface.mlir").write_text("module {}\n")
        _yaml(capsule / "golden.yaml", {"outputs": {"Y": [[6]]}})
    _yaml(
        root / "custom-descriptor.yaml",
        {
            "target": "fixture",
            "capsule_corpus": str(root / "corpus/isa"),
            "toolchain": {"sim_via": "fixture-offline"},
        },
    )
    _yaml(
        root / "support/contracts/target_contract.yaml",
        {
            "name": "fixture",
            "plugin": {"sim_oracle": "oracle.py"},
            "runner": {"sim_via": "fixture-offline"},
        },
    )
    (root / "support/oracle.py").write_text(
        "from merlin.targetgen.oracle_policy import register_sim_oracle\n"
        "def unavailable(*a, **k): raise RuntimeError('no fixture numerical oracle')\n"
        "register_sim_oracle('fixture-offline', adapters=lambda t: {'L2': unavailable}, "
        "available=lambda t: (False, 'explicit offline fixture'), exclusive=True)\n"
    )
    _yaml(
        root / "bundle/input_bundle_manifest.yaml",
        {
            "bundle_id": "synthetic",
            "allowed": [{"path": str(root / "corpus/isa"), "as": "capsules"}],
            "denied": [],
        },
    )
    for name in ("tools.txt", "allowed_files.txt", "denied_files.txt"):
        (root / "bundle" / name).write_text("")
    (root / "task").mkdir()
    (root / "task/TASK_realistic.md").write_text("Local diagnostic fixture. No paid provider or hardware.\n")
    (root / "bin").mkdir()
    cli = root / "bin/claude"
    cli.write_text(f"#!{sys.executable}\n" + DRIVER)
    cli.chmod(0o700)
    environment = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("MERLIN_", "ANTHROPIC_", "CLAUDE_", "AWS_", "CODEX_", "OPENAI_"))
    }
    environment.update(
        {
            "CONTROLLER_FIXTURE_ROOT": str(root),
            "MERLIN_OUT_ROOT": str(root / "generated"),
            "MERLIN_TARGET_PATH": str(root / "support"),
            "MERLIN_AET_SINK": "0",
            "MERLIN_BUNDLE_CAS": "",
            "MERLIN_MLC_DIR": str(root / "unavailable-modelir"),
            "PYTHONPATH": os.pathsep.join(map(str, python_import_roots())),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
        }
    )
    return environment


def _invoke(root, environment, *extra):
    return subprocess.run(
        [sys.executable, "-c", RUN, str(root), *extra],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("interface", ["python", "cli", "rtlchecks", "catalog-rtlchecks"])
def test_real_package_controller_copy_fresh_and_resume(tmp_path, interface):
    root = tmp_path / "operator"
    environment = _project(root)
    environment["CONTROLLER_FIXTURE_INTERFACE"] = interface
    if interface in {"rtlchecks", "catalog-rtlchecks"}:
        manifest_path = root / "bundle/input_bundle_manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest.update(bundle_id="merlin_assisted_rtlchecks_fixture", arm="merlin_rtlchecks")
        _yaml(manifest_path, manifest)
    if interface == "catalog-rtlchecks":
        _yaml(root / "catalog.yaml", {"schema_version": 1, "experiments": {"fixture-rtl": "experiment.yaml"}})
        _yaml(
            root / "experiment.yaml",
            {
                "schema_version": 1,
                "id": "fixture-rtl",
                "target": "fixture",
                "phases": {
                    "1": {
                        "adapter": "capsule_bench",
                        "config": {
                            "descriptor": "custom-descriptor.yaml",
                            "arm": "merlin_assisted",
                            "treatment": "rtlchecks",
                            "model": "claude-fixture",
                            "effort": "high",
                            "driver": "claudecode",
                            "max_wall_s": 30,
                            "round_timeout": 10,
                            "qa_timeout": 1,
                            "experiment": "realistic",
                            "bundle": "merlin_assisted_rtlchecks_fixture",
                            "bundle_manifest": "bundle/input_bundle_manifest.yaml",
                            "oracle_timing": "timing.json",
                        },
                    }
                },
            },
        )
    fresh = _invoke(root, environment)
    assert fresh.returncode == 42, fresh.stdout + fresh.stderr
    [checkpoint] = list((root / "generated").rglob("qa_loop_state.yaml"))
    run = checkpoint.parent
    original_environment = (run / "environment.yaml").read_bytes()
    evidence = yaml.safe_load(original_environment)
    if interface in {"rtlchecks", "catalog-rtlchecks"}:
        assert evidence["bundle_id"] == "merlin_assisted_rtlchecks_fixture"
        assert evidence["invocation_treatment"]["name"] == "rtlchecks"
        assert evidence["invocation_treatment"]["callbacks"]["qa_runner"]["module"] == (
            "merlin_experiments.phase1.feedback.rtlchecks"
        )
    assert evidence["sandbox"] == "none"
    assert evidence["bundle_input_snapshot"] is None
    assert evidence["golden_mask_selftest"]["pilot_golden_visible_to_agent"] == "OK"
    assert evidence["workspace_copy_report"]["answer_files_dropped"] == 1
    for owner in ("controller.py", "task_staging.py", "workspace_transport.py", "authoring.py"):
        assert "phase1:source:" + owner in evidence["implementation_sources"]["inputs"]
    workspace = Path(evidence["workspace_path"])
    assert (workspace / "TASK.md").read_bytes() == (run / "TASK.md").read_bytes()
    assert not list(workspace.rglob("golden.yaml"))
    resumed = _invoke(root, environment, "--resume")
    assert resumed.returncode == 1, resumed.stdout + resumed.stderr
    assert (run / "qa_loop_summary.yaml").is_file(), resumed.stdout + resumed.stderr
    assert (run / "environment.yaml").read_bytes() == original_environment
    summary = yaml.safe_load((run / "qa_loop_summary.yaml").read_text())
    assert summary["formal_complete"] is False
    calls = [json.loads(line) for line in (root / "provider_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2 and calls[0]["workspace"] == calls[1]["workspace"]
    assert (workspace / "submission/partial.txt").read_text() == "retained partial candidate\n"
    if interface == "catalog-rtlchecks":
        fresh_command = json.loads((root / "catalog-fresh.json").read_text())
        assert json.loads((root / "catalog-resumed.json").read_text()) == fresh_command
        assert fresh_command["resume_policy"] == "native_flag"


@pytest.mark.parametrize("damage", ["task", "source-record"])
def test_package_controller_resume_refuses_drift_before_provider(tmp_path, damage):
    root = tmp_path / "operator"
    environment = _project(root)
    fresh = _invoke(root, environment)
    assert fresh.returncode == 42, fresh.stdout + fresh.stderr
    [record] = list((root / "generated").rglob("environment.yaml"))
    evidence = yaml.safe_load(record.read_text())
    if damage == "task":
        (Path(evidence["workspace_path"]) / "TASK.md").write_text("tampered task")
    else:
        sources = evidence["implementation_sources"]["inputs"]
        sources["phase1:source:task_staging.py"]["sha256"] = "0" * 64
        record.write_text(yaml.safe_dump(evidence))
    before = (root / "provider_calls.jsonl").read_bytes()
    resumed = _invoke(root, environment, "--resume")
    assert resumed.returncode != 0
    output = resumed.stdout + resumed.stderr
    if damage == "task":
        assert "experiment treatment drifted after setup" in output
    else:
        assert "source identity changed" in output
    assert (root / "provider_calls.jsonl").read_bytes() == before
