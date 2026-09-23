"""Explicit operator review and actual native snapshot admission, without agents/hardware."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from merlin_experiments.adapters import ADAPTERS
from merlin_experiments.cli import main
from merlin_experiments.corpus import release as corpus_release
from merlin_experiments.runner import fingerprint


def _member(root: Path, category: str, name: str, label: str) -> None:
    path = root / category / name
    path.mkdir(parents=True)
    capsule = {
        "name": name,
        "kind": "isa",
        "source_role": "handauthored_compiler_test",
        "label": label,
        "operation": {"op": "matmul", "attributes": {}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "expected": {"instruction_classes": [], "modes": {}},
        "required_oracle_tiers": ["L0"],
        "interface_mlir": "capsule.interface.mlir",
    }
    (path / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    (path / "capsule.interface.mlir").write_text("module {}\n")
    (path / "golden.yaml").write_text("outputs: {}\n")


@pytest.fixture
def release_fixture(tmp_path, monkeypatch):
    from merlin.common.paths import data_path

    contract = data_path("contract")
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(contract))
    monkeypatch.setenv("MERLIN_SCHEMAS_DIR", str(contract.parent / "schemas"))
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    profiles = tmp_path / "merlin/contract/capsules/profiles"
    profiles.mkdir(parents=True)
    (profiles / "fixture-device.yaml").write_text("capsules: []\n")
    public_contract = (
        "VERSION",
        "command_buffer_abi.yaml",
        "interface_dialect_contract.yaml",
        "interface_grammar.md",
        "integrity_policy.md",
        "mlir_oot_backend_contract.yaml",
        "target_dialect_contract.yaml",
    )
    for name in public_contract:
        (tmp_path / "merlin/contract" / name).write_bytes((contract / name).read_bytes())
    import shutil

    shutil.copytree(contract / "schemas", tmp_path / "merlin/contract/schemas")
    (tmp_path / "third_party/llvm-install").mkdir(parents=True)
    baseline = tmp_path / "baseline"
    _member(baseline, "isa", "generated_member", "public")
    _member(baseline, "layers", "retained_member", "public")
    _member(baseline, "hidden", "private_member_identity", "hidden")
    (baseline / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "generated": ["isa/generated_member"],
                "hand_authored": ["layers/retained_member"],
                "held_out": {"n_generated": 0, "n_hand_authored": 1},
            }
        )
    )
    experiment = tmp_path / "source-experiment"
    (experiment / "task").mkdir(parents=True)
    (experiment / "scripts").mkdir()
    (experiment / "task/TASK_full.md").write_text("Fixture compiler task\n")
    (experiment / "task/TASK_realistic.md").write_text("Fixture compiler task\n")
    (experiment / "scripts/agent_selfcheck.py").write_text("# public fixture\n")
    descriptor = experiment / "target_experiment.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "target": "fixture-device",
                "capsule_corpus": "baseline/isa",
                "grading": {
                    "expected_cohort": {"source_capsules": 2, "admitted_capsules": 2},
                    "hidden_capability_admission": {"source_capsules": 1, "admitted_capsules": 1},
                },
            }
        )
    )
    derivation = tmp_path / "derive.py"
    derivation.write_text(
        "import argparse, pathlib, shutil, yaml\n"
        "p=argparse.ArgumentParser();p.add_argument('--target');p.add_argument('--output-root');"
        "p.add_argument('--descriptor');a=p.parse_args()\n"
        "root=pathlib.Path(a.descriptor).parent.parent;output=pathlib.Path(a.output_root)\n"
        "shutil.copytree(root/'baseline/isa/generated_member',output/'isa/generated_member')\n"
        "(output/'isa/generated_member/README.md').write_text('new derived member bytes\\n')\n"
        "(output/'MANIFEST.yaml').write_text(yaml.safe_dump({'generated_by':'derive.py','generated':['isa/generated_member'],"
        "'held_out':{'n_generated':0}}))\n"
    )
    native = tmp_path / "native.py"
    native.write_text(
        "import argparse,os,pathlib,yaml\n"
        "from merlin.targetgen.sandbox.bwrap import materialize_bundle_inputs\n"
        "from merlin_experiments.corpus.release import verify_snapshot\n"
        "descriptor=pathlib.Path(os.environ['MERLIN_TARGET_EXPERIMENT'])\n"
        "p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);"
        "p.add_argument('--bundle-manifest',required=True);p.add_argument('--oracle-timing',required=True)\n"
        "a,_=p.parse_known_args();bundle=yaml.safe_load(pathlib.Path(a.bundle_manifest).read_text())\n"
        "assert bundle['bundle_id']==a.bundle;assert yaml.safe_load(pathlib.Path(a.oracle_timing).read_text())=={}\n"
        "ws=pathlib.Path(os.environ['MERLIN_OUT_ROOT'])/'build/native-probe/ws';ws.mkdir(parents=True)\n"
        "materialize_bundle_inputs(ws,bundle)\n"
        "verify_snapshot(pathlib.Path(os.environ['MERLIN_CORPUS_SEAL']),descriptor,ws,bundle)\n"
        "(pathlib.Path(os.environ['MERLIN_OUT_ROOT'])/'native-receipt').write_text('verified before agent')\n"
    )
    monkeypatch.setitem(
        ADAPTERS, "capsule_derivation", replace(ADAPTERS["capsule_derivation"], script=derivation.name, module=None)
    )
    monkeypatch.setitem(ADAPTERS, "capsule_bench", replace(ADAPTERS["capsule_bench"], script=native.name, module=None))
    definition = tmp_path / "phase0.yaml"
    definition.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "derive-fixture",
                "target": "fixture-device",
                "phases": {"0": {"adapter": "capsule_derivation", "config": {"descriptor": str(descriptor)}}},
            }
        )
    )
    return {
        "root": tmp_path,
        "definition": definition,
        "baseline": baseline,
        "run": tmp_path / "out/runs/derivation",
        "release": tmp_path / "out/artifacts/protocols/review-fixture",
    }


def _prepare(fixture, capsys):
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    code = main(["corpus", "prepare", str(fixture["run"]), "--output", str(fixture["release"])])
    output = capsys.readouterr()
    if code:
        diagnostic = fixture["release"] / "private/failure.json"
        pytest.fail(output.err + (diagnostic.read_text() if diagnostic.exists() else ""))
    assert "private_member_identity" not in output.out + output.err
    return json.loads(output.out)


def _seal(fixture, report, capsys):
    assert (
        main(
            [
                "corpus",
                "seal",
                str(fixture["release"]),
                "--expected-digest",
                report["review_digest"],
                "--reviewed-by",
                "synthetic-test-operator",
                "--review-note",
                "fixture-only explicit review",
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert "private_member_identity" not in output.out + output.err
    return json.loads(output.out)


def _phase1_definition(fixture, sealed):
    definition = fixture["root"] / "phase1.yaml"
    timing = fixture["root"] / "oracle-timing.yaml"
    timing.write_text("{}\n")
    bundle = "raw_baseline_public_v0"
    manifest = Path(sealed["descriptor"]).parent / "input_bundles" / bundle / "input_bundle_manifest.yaml"
    definition.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "functional-fixture",
                "target": "fixture-device",
                "phases": {
                    "1": {
                        "adapter": "capsule_bench",
                        "config": {
                            "descriptor": sealed["descriptor"],
                            "corpus_seal": sealed["seal"],
                            "bundle": bundle,
                            "bundle_manifest": str(manifest),
                            "oracle_timing": str(timing),
                            "arm": "raw_baseline",
                            "model": "fixture-only",
                            "effort": "high",
                            "max_wall_s": 60,
                            "round_timeout": 60,
                        },
                    }
                },
            }
        )
    )
    return definition


def test_public_prepare_inspect_explicit_seal_and_native_phase1(release_fixture, capsys):
    fixture = release_fixture
    original = fingerprint(fixture["baseline"])
    report = _prepare(fixture, capsys)
    assert report["state"] == "awaiting_operator_review"
    assert not (fixture["release"] / "private/seal.json").exists()
    assert fingerprint(fixture["baseline"]) == original
    assert (fixture["release"] / "payload/corpus/layers/retained_member/capsule.yaml").is_file()
    assert (fixture["release"] / "payload/corpus/isa/generated_member/README.md").is_file()
    assert main(["corpus", "inspect", str(fixture["release"])]) == 0
    assert json.loads(capsys.readouterr().out)["review_digest"] == report["review_digest"]
    sealed = _seal(fixture, report, capsys)
    definition = _phase1_definition(fixture, sealed)
    run = fixture["root"] / "out/runs/functional"
    assert main(["run", str(definition), "--phase", "1", "--run-dir", str(run)]) == 0
    assert (fixture["root"] / "out/native-receipt").read_text() == "verified before agent"
    assert fingerprint(fixture["baseline"]) == original
    assert fixture["release"].stat().st_mode & 0o077 == 0
    assert Path(sealed["seal"]).stat().st_mode & 0o077 == 0


def test_example_style_phase1_selects_release_and_bundle_without_editing_definition(release_fixture, capsys):
    fixture = release_fixture
    report = _prepare(fixture, capsys)
    sealed = _seal(fixture, report, capsys)
    definition = _phase1_definition(fixture, sealed)
    document = yaml.safe_load(definition.read_text())
    config = document["phases"]["1"]["config"]
    retained_bundle = Path(config["bundle_manifest"])
    config["descriptor"] = str(fixture["root"] / "source-experiment/target_experiment.yaml")
    config.pop("corpus_seal")
    config["require_reviewed_corpus"] = True
    definition.write_text(yaml.safe_dump(document))

    assert main(["preflight", str(definition), "--phase", "1"]) == 2
    assert "requires a reviewed Phase 0 release" in capsys.readouterr().out

    selected_bundle = fixture["root"] / "reviewed-bundle/input_bundle_manifest.yaml"
    selected_bundle.parent.mkdir()
    selected_bundle.write_bytes(retained_bundle.read_bytes())
    args = [
        str(definition),
        "--phase",
        "1",
        "--corpus-seal",
        sealed["seal"],
        "--bundle-manifest",
        str(selected_bundle),
    ]
    assert main(["inspect", *args]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["phases"]["1"]["inputs"]["descriptor"] == sealed["descriptor"]
    assert plan["phases"]["1"]["inputs"]["corpus_seal"] == sealed["seal"]
    assert plan["phases"]["1"]["requires_reviewed_corpus"] is True
    assert main(["preflight", *args]) == 0
    assert json.loads(capsys.readouterr().out)["configuration_ready"] is True
    unsafe = yaml.safe_load(selected_bundle.read_text())
    unsafe["allowed"].append({"path": "merlin/contract/", "mode": "ro"})
    selected_bundle.write_text(yaml.safe_dump(unsafe))
    assert main(["preflight", *args]) == 2
    assert "still grants the historical in-tree capsule corpus" in capsys.readouterr().out


@pytest.mark.parametrize("task_only", [False, True])
def test_explicit_source_resources_become_self_contained_release_inputs(release_fixture, capsys, task_only):
    fixture = release_fixture
    source = fixture["root"] / "source-experiment"
    resources = fixture["root"] / "authored-resources"
    resources.mkdir()
    (source / "task").rename(resources / "task")
    (source / "task").mkdir()
    (source / "task/TASK_full.md").write_text("Wrong descriptor-sibling task\n")
    descriptor = source / "target_experiment.yaml"
    document = yaml.safe_load(descriptor.read_text())
    document["task_root" if task_only else "resources_root"] = (
        "authored-resources/task" if task_only else "authored-resources"
    )
    descriptor.write_text(yaml.safe_dump(document))

    report = _prepare(fixture, capsys)
    prepared = Path(report["descriptor"])
    assert "resources_root" not in yaml.safe_load(prepared.read_text())
    assert "task_root" not in yaml.safe_load(prepared.read_text())
    assert (prepared.parent / "task/TASK_full.md").read_bytes() == (resources / "task/TASK_full.md").read_bytes()
    sealed = _seal(fixture, report, capsys)
    # Moving/changing the source cannot redirect the approved release's task resources.
    (resources / "task/TASK_full.md").write_text("Changed after preparation\n")
    definition = _phase1_definition(fixture, sealed)
    assert main(["run", str(definition), "--phase", "1", "--run-dir", str(fixture["root"] / "out/runs/explicit")]) == 0


def test_contracts_root_harness_is_staged_without_live_source_pointer(release_fixture, capsys):
    from merlin.targetgen.sandbox.toolchain import curated_harness_dir
    from merlin.targetgen.target_experiment import load_target_experiment

    fixture = release_fixture
    selected = fixture["root"] / "authored-contracts/harness"
    selected.mkdir(parents=True)
    (selected / "runtime.h").write_text("/* selected public harness */\n")
    source = fixture["root"] / "source-experiment"
    decoy = source / "contracts/harness"
    decoy.mkdir(parents=True)
    (decoy / "runtime.h").write_text("/* wrong legacy harness */\n")
    descriptor = source / "target_experiment.yaml"
    document = yaml.safe_load(descriptor.read_text())
    document["contracts_root"] = "authored-contracts"
    document["hardware_spec"] = {"curated_harness": "contracts/harness"}
    descriptor.write_text(yaml.safe_dump(document))

    report = _prepare(fixture, capsys)
    prepared = Path(report["descriptor"])
    assert "contracts_root" not in yaml.safe_load(prepared.read_text())
    copied = prepared.parent / "contracts/harness"
    assert (copied / "runtime.h").read_bytes() == (selected / "runtime.h").read_bytes()
    sealed = _seal(fixture, report, capsys)
    (selected / "runtime.h").write_text("/* changed after preparation */\n")
    assert curated_harness_dir(load_target_experiment(prepared)) == str(copied)
    assert (copied / "runtime.h").read_text() == "/* selected public harness */\n"
    definition = _phase1_definition(fixture, sealed)
    assert main(["run", str(definition), "--phase", "1", "--run-dir", str(fixture["root"] / "out/runs/contracts")]) == 0


def test_generated_prompt_release_needs_no_authored_task_directory(release_fixture, capsys):
    fixture = release_fixture
    source = fixture["root"] / "source-experiment/task"
    source.rename(fixture["root"] / "unused-authored-task")
    report = _prepare(fixture, capsys)
    prepared_tasks = Path(report["descriptor"]).parent / "task"
    assert prepared_tasks.is_dir() and list(prepared_tasks.iterdir()) == []
    preparation = json.loads((fixture["release"] / "private/preparation.json").read_text())
    assert preparation["scaffolding"]["task"] == {"path": str(source), "present": False, "sha256": None}
    sealed = _seal(fixture, report, capsys)
    definition = _phase1_definition(fixture, sealed)
    assert main(["run", str(definition), "--phase", "1", "--run-dir", str(fixture["root"] / "out/runs/generated")]) == 0


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling_symlink"])
def test_invalid_task_resource_is_not_treated_as_generated_prompt_absence(release_fixture, capsys, kind):
    fixture = release_fixture
    source = fixture["root"] / "source-experiment/task"
    retained = fixture["root"] / "unused-authored-task"
    source.rename(retained)
    if kind == "file":
        source.write_text("not a task directory\n")
    else:
        source.symlink_to(retained if kind == "symlink" else fixture["root"] / "absent")
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    assert main(["corpus", "prepare", str(fixture["run"]), "--output", str(fixture["release"])]) != 0
    assert not (fixture["release"] / "private/seal.json").exists()


def test_operator_review_requires_exact_observed_digest(release_fixture, capsys):
    report = _prepare(release_fixture, capsys)
    assert (
        main(
            [
                "corpus",
                "seal",
                str(release_fixture["release"]),
                "--expected-digest",
                "0" * 64,
                "--reviewed-by",
                "fixture",
                "--review-note",
                "fixture",
            ]
        )
        == 2
    )
    assert "digest" in capsys.readouterr().err
    assert not (release_fixture["release"] / "private/seal.json").exists()
    sealed = _seal(release_fixture, report, capsys)
    assert (
        main(
            [
                "corpus",
                "seal",
                str(release_fixture["release"]),
                "--expected-digest",
                report["review_digest"],
                "--reviewed-by",
                "fixture",
                "--review-note",
                "overwrite",
            ]
        )
        == 2
    )
    capsys.readouterr()
    assert corpus_release.verify(Path(sealed["seal"]), Path(sealed["descriptor"]))


@pytest.mark.parametrize("relative", ["isa/generated_member/README.md", "hidden/private_member_identity/golden.yaml"])
def test_sealed_byte_drift_refuses_before_any_native_process(release_fixture, capsys, relative):
    report = _prepare(release_fixture, capsys)
    sealed = _seal(release_fixture, report, capsys)
    path = release_fixture["release"] / "payload/corpus" / relative
    path.chmod(0o600)  # deliberate operator tampering, not a normal writable release
    path.write_text("changed fixture bytes")
    definition = _phase1_definition(release_fixture, sealed)
    assert main(["run", str(definition), "--phase", "1"]) == 2
    output = capsys.readouterr()
    assert "private_member_identity" not in output.out + output.err
    assert not (release_fixture["root"] / "out/native-receipt").exists()


def test_snapshot_mismatch_cannot_borrow_reviewed_seal(release_fixture, capsys):
    from merlin.targetgen.sandbox.bwrap import materialize_bundle_inputs

    report = _prepare(release_fixture, capsys)
    sealed = _seal(release_fixture, report, capsys)
    descriptor = Path(sealed["descriptor"])
    bundle = yaml.safe_load(
        (descriptor.parent / "input_bundles/raw_baseline_hwbringup_v0/input_bundle_manifest.yaml").read_text()
    )
    member = release_fixture["release"] / "payload/corpus/hidden/private_member_identity/golden.yaml"
    original = member.read_bytes()
    member.chmod(0o600)
    member.write_text("snapshot differs from reviewed source")
    ws = release_fixture["root"] / "out/build/frozen-probe/ws"
    ws.mkdir(parents=True)
    materialize_bundle_inputs(ws, bundle)
    member.write_bytes(original)
    member.chmod(0o400)
    with pytest.raises(ValueError, match="snapshot differs"):
        corpus_release.verify_snapshot(Path(sealed["seal"]), descriptor, ws, bundle)


def test_failed_generation_has_no_prepare_or_implicit_review(release_fixture, capsys):
    script = release_fixture["root"] / "derive.py"
    script.write_text("raise SystemExit(3)\n")
    assert main(["run", str(release_fixture["definition"]), "--run-dir", str(release_fixture["run"])]) == 3
    capsys.readouterr()
    assert main(["corpus", "prepare", str(release_fixture["run"]), "--output", str(release_fixture["release"])]) == 2
    assert not release_fixture["release"].exists()


def test_native_harness_binds_seal_before_any_agent_launch():
    import ast

    from merlin.common.paths import module_source_path, repo_root

    harness = repo_root() / "merlin/experiments/capsule_bench/harness/run_baseline_qa_loop.py"
    if not harness.is_file():
        pytest.skip("historical native harness source is not part of an installed wheel")
    tree = ast.parse(harness.read_text())
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert ast.unparse(entry.body[-1].value.func) == "controller.run"
    controller = ast.parse(module_source_path("merlin_experiments.phase1.controller").read_text())
    run = next(node for node in controller.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    lifecycle = next(node for node in run.body if isinstance(node, ast.With))
    preparation = next(
        node
        for node in lifecycle.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prepared" for t in node.targets)
    )
    assert ast.unparse(preparation.value.func) == "session.prepare"
    execute = lifecycle.body[-1]
    assert isinstance(execute, ast.Return)
    assert ast.unparse(execute.value.func) == "authoring.execute"
    assert ast.unparse(execute.value.args[0]) == "prepared"
    assert preparation.lineno < execute.lineno
    refusal = lifecycle.body[lifecycle.body.index(preparation) + 1]
    assert isinstance(refusal, ast.If)
    assert ast.unparse(refusal.test) == "isinstance(prepared, int)"
    assert isinstance(refusal.body[0], ast.Return)
    assert ast.unparse(refusal.body[0].value) == "prepared"
    authoring = ast.parse(module_source_path("merlin_experiments.phase1.authoring").read_text())
    continuation = next(node for node in authoring.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    session = ast.parse(module_source_path("merlin_experiments.phase1.session").read_text())
    admission = next(node for node in session.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
    calls = [node for node in ast.walk(admission) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    verification = [node.lineno for node in calls if node.func.id == "verify_snapshot"]
    assert not any(node.func.id == "_launch" for node in calls)
    launches = [
        node
        for node in ast.walk(continuation)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_launch"
    ]
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_launch" for t in node.targets)
        and ast.unparse(node.value)
        == (
            "T.timed(partial(EX.launch, config=execution, capsules_root=_public_root, "
            "policy_root=_policy_root, contract=_contract_root), "
            "'agent', treatment.on_duration)"
        )
        for node in continuation.body
    ), "the measured callback must wrap the real agent launch"
    assert verification and launches
    assert max(verification) < admission.body[-1].lineno
    assert isinstance(admission.body[-1], ast.Return)
    assert ast.unparse(admission.body[-1].value.func) == "PreparedRun"


def test_derivation_bytes_changed_after_receipt_cannot_be_prepared(release_fixture, capsys):
    fixture = release_fixture
    assert main(["run", str(fixture["definition"]), "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    (fixture["run"] / "phase0/capsules/isa/generated_member/README.md").write_text("post-run drift")
    assert main(["corpus", "prepare", str(fixture["run"]), "--output", str(fixture["release"])]) == 2
    assert "output changed" in capsys.readouterr().err
    assert not fixture["release"].exists()


@pytest.mark.parametrize("damage", ["unclassified", "unaccounted_removal", "symlink", "hardlink", "changed_visibility"])
def test_prepare_refuses_ambiguous_or_unsafe_assembly(release_fixture, capsys, damage):
    fixture = release_fixture
    baseline = fixture["baseline"]
    if damage == "unclassified":
        manifest = yaml.safe_load((baseline / "MANIFEST.yaml").read_text())
        manifest["hand_authored"] = []
        (baseline / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    elif damage == "unaccounted_removal":
        manifest = yaml.safe_load((baseline / "MANIFEST.yaml").read_text())
        manifest["generated"].append("layers/retained_member")
        manifest["hand_authored"] = []
        (baseline / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    elif damage == "symlink":
        (baseline / "layers/retained_member/private-link").symlink_to(baseline / "hidden")
    elif damage == "hardlink":
        os.link(
            baseline / "hidden/private_member_identity/golden.yaml",
            baseline / "layers/retained_member/private-alias.yaml",
        )
    else:
        script = fixture["root"] / "derive.py"
        script.write_text(
            script.read_text()
            + (
                "cap=output/'isa/generated_member/capsule.yaml';doc=yaml.safe_load(cap.read_text());"
                "doc['label']='hidden';cap.write_text(yaml.safe_dump(doc))\n"
            )
        )
    original = (baseline / "MANIFEST.yaml").read_bytes()
    assert main(["run", str(fixture["definition"]), "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    assert main(["corpus", "prepare", str(fixture["run"]), "--output", str(fixture["release"])]) == 2
    output = capsys.readouterr()
    assert "private_member_identity" not in output.out + output.err
    assert (baseline / "MANIFEST.yaml").read_bytes() == original
    assert not (fixture["release"] / "private/seal.json").exists()


@pytest.mark.parametrize("damage", ["metadata_permissions", "private_symlink", "receipt_identity"])
def test_private_seal_and_metadata_tampering_is_refused(release_fixture, capsys, damage):
    fixture = release_fixture
    report = _prepare(fixture, capsys)
    sealed = _seal(fixture, report, capsys)
    private = fixture["release"] / "private"
    if damage == "metadata_permissions":
        (private / "preparation.json").chmod(0o644)
    elif damage == "private_symlink":
        moved = fixture["release"] / "other-private"
        private.rename(moved)
        private.symlink_to(moved, target_is_directory=True)
    else:
        document = json.loads(Path(sealed["seal"]).read_text())
        document["review_digest"] = "0" * 64
        Path(sealed["seal"]).write_text(json.dumps(document))
    assert main(["corpus", "inspect", str(fixture["release"])]) == 2
    output = capsys.readouterr()
    assert "private_member_identity" not in output.out + output.err


def test_seal_cannot_authorize_original_descriptor_or_implicit_approval(release_fixture, capsys):
    fixture = release_fixture
    report = _prepare(fixture, capsys)
    assert (
        main(
            [
                "corpus",
                "seal",
                str(fixture["release"]),
                "--expected-digest",
                report["review_digest"],
                "--reviewed-by",
                "",
                "--review-note",
                "",
            ]
        )
        == 2
    )
    capsys.readouterr()
    sealed = _seal(fixture, report, capsys)
    with pytest.raises(ValueError, match="descriptor is not"):
        corpus_release.verify(Path(sealed["seal"]), fixture["root"] / "source-experiment/target_experiment.yaml")


def test_retention_pin_and_readonly_release_do_not_change_sources(release_fixture, capsys):
    from merlin.common.storage_lifecycle import inventory

    fixture = release_fixture
    original_mode = (fixture["baseline"] / "isa/generated_member/capsule.yaml").stat().st_mode
    report = _prepare(fixture, capsys)
    _seal(fixture, report, capsys)
    payload = fixture["release"] / "payload"
    assert all(not path.stat().st_mode & 0o222 for path in [payload, *payload.rglob("*")])
    assert (fixture["baseline"] / "isa/generated_member/capsule.yaml").stat().st_mode == original_mode
    rows = inventory()["paths"]
    assert next(row for row in rows if row["path"] == str(fixture["release"]))["pins"]


def test_prepared_default_bundle_and_selfcheck_are_relocatable(release_fixture, capsys):
    report = _prepare(release_fixture, capsys)
    experiment = Path(report["descriptor"]).parent
    for variant in ("public_v0", "realistic_v0", "hwbringup_v0"):
        assert (experiment / f"input_bundles/raw_baseline_{variant}/input_bundle_manifest.yaml").is_file()
    probe = subprocess.run(
        [sys.executable, "-I", str(experiment / "scripts/agent_selfcheck.py"), "--help"],
        cwd=release_fixture["root"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert "driver-side broker" in probe.stdout


def test_reviewed_handoff_uses_shared_content_store_without_mutating_sources(release_fixture, capsys, monkeypatch):
    from merlin.targetgen.sandbox import bwrap as BW

    fixture = release_fixture
    store = fixture["root"] / "out/artifacts/cache/bundle-inputs-cas"
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(store))
    original = fingerprint(fixture["baseline"])
    report = _prepare(fixture, capsys)
    sealed = _seal(fixture, report, capsys)
    definition = _phase1_definition(fixture, sealed)
    run = fixture["root"] / "out/runs/functional-cas"
    assert main(["run", str(definition), "--phase", "1", "--run-dir", str(run)]) == 0
    assert fingerprint(fixture["baseline"]) == original
    assert store.is_dir()
    corpus = fixture["release"] / "payload/corpus"
    public = corpus / "isa/generated_member/capsule.interface.mlir"
    private = corpus / "hidden/private_member_identity/capsule.interface.mlir"
    assert public.read_bytes() == private.read_bytes()
    assert public.stat().st_ino != private.stat().st_ino
    bundle = yaml.safe_load(
        (
            Path(sealed["descriptor"]).parent / "input_bundles/raw_baseline_public_v0/input_bundle_manifest.yaml"
        ).read_text()
    )
    ws = fixture["root"] / "out/build/native-probe/ws"
    argv = BW.base_argv(ws, bundle, repo=fixture["root"])
    assert BW.is_exposed(argv, public)
    assert not BW.is_exposed(argv, private)


@pytest.mark.parametrize("location", ["live", "frozen"])
def test_release_review_metadata_never_enters_candidate_grants(release_fixture, capsys, monkeypatch, location):
    from types import SimpleNamespace

    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.sandbox import toolchain as TC

    fixture = release_fixture
    sealed = _seal(fixture, _prepare(fixture, capsys), capsys)
    descriptor = Path(sealed["descriptor"])
    bundle = yaml.safe_load(
        (descriptor.parent / "input_bundles/raw_baseline_public_v0/input_bundle_manifest.yaml").read_text()
    )
    private = fixture["release"] / "private"
    assert str(private) in {entry["path"] for entry in bundle["host_inputs"]}
    assert str(private) not in {entry["path"] for entry in bundle["allowed"]}
    ws = fixture["root"] / "snapshot-ws"
    ws.mkdir()
    BW.materialize_bundle_inputs(ws, bundle, repo=fixture["root"])
    [frozen] = BW.snapshot_input_paths(ws, bundle, [private], repo=fixture["root"])
    corpus_release.verify_snapshot(Path(sealed["seal"]), descriptor, ws, bundle, repo=fixture["root"])
    source = private if location == "live" else frozen
    alias = fixture["root"] / "runtime-alias"
    extra = ["--ro-bind", str(source.parent), str(alias)]
    monkeypatch.setattr(BW, "repo_root", lambda: fixture["root"])
    monkeypatch.setattr(BW, "claude_runtime_binds", lambda: [])
    monkeypatch.setattr(TC, "toolchain_binds", lambda te: extra)
    monkeypatch.setattr(BW, "answer_surfaces", lambda te: [])
    exposed = alias / source.name / "preparation.json"
    assert BW.is_exposed(extra, exposed)  # Negative control: mode700 alone is not isolation.
    argv = BW.full_argv(SimpleNamespace(target="fixture-device"), ws, bundle)
    assert not BW.is_exposed(argv, exposed)
    public = json.dumps(BW.snapshot_record(ws))
    assert "private_member_identity" not in public
    assert "fixture-only explicit review" not in public
    BW.remove_bundle_snapshot(ws)


def test_missing_private_snapshot_cannot_fall_back_to_live_review(release_fixture, capsys):
    from merlin.targetgen.sandbox import bwrap as BW

    fixture = release_fixture
    sealed = _seal(fixture, _prepare(fixture, capsys), capsys)
    descriptor = Path(sealed["descriptor"])
    bundle = yaml.safe_load(
        (descriptor.parent / "input_bundles/raw_baseline_public_v0/input_bundle_manifest.yaml").read_text()
    )
    # An older snapshot or a declaration stripped of the private review directory
    # cannot use the still-valid live seal as a replacement for missing evidence.
    bundle["host_inputs"] = [
        entry for entry in bundle["host_inputs"] if entry["path"] != str(fixture["release"] / "private")
    ]
    ws = fixture["root"] / "missing-review-snapshot"
    ws.mkdir()
    BW.materialize_bundle_inputs(ws, bundle, repo=fixture["root"])
    with pytest.raises(RuntimeError, match="snapshot"):
        corpus_release.verify_snapshot(Path(sealed["seal"]), descriptor, ws, bundle, repo=fixture["root"])
    BW.remove_bundle_snapshot(ws)
