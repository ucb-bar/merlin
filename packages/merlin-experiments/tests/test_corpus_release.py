"""Explicit operator review and actual native snapshot admission, without agents/hardware."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.adapters import ADAPTERS
from merlin_experiments.cli import main
from merlin_experiments.corpus import release as corpus_release
from merlin_experiments.corpus.coverage import _public_category_roots
from merlin_experiments.corpus.preparation import assemble, generation_lineage
from merlin_experiments.runner import fingerprint
from merlin_experiments.spec import SpecError


def test_release_uses_phase0_readiness_without_promoting_phase1(monkeypatch, tmp_path):
    from merlin_experiments.phase0 import coverage_commitment as commitment

    from merlin.targetgen import target_experiment

    root = tmp_path / "release"
    private = root / "private"
    private.mkdir(parents=True)
    # Compiler-owned support-lowering remains incomplete in the original
    # certificate. This wiring test supplies a synthetic policy verdict only;
    # real readiness still requires a verified capture issuer.
    report = {
        "schema": commitment.SCHEMA,
        "phase": "phase1",
        "status": "incomplete",
        "blockers": [{"component": "support_lowering", "reason": "pending compiler evidence"}],
    }
    coverage = private / "workload-coverage.json"
    coverage.write_text(json.dumps(report))
    coverage.chmod(0o600)
    readiness = {
        "schema": commitment.READINESS_SCHEMA,
        "status": "ready",
        "inputs_sha256": "a" * 64,
        "cohort_sha256": "b" * 64,
        "blockers": [],
        "deferred_phase1": report["blockers"],
    }
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda _: SimpleNamespace(capsule_corpus=root))
    monkeypatch.setattr(commitment, "read_inputs", lambda _: {"selected": True})
    monkeypatch.setattr(commitment, "requires_workload_coverage", lambda *_: True)
    monkeypatch.setattr(commitment, "build_phase0_readiness", lambda observed: readiness if observed == report else {})
    summary = {"required": True, "status": "incomplete", "report_sha256": corpus_release._digest(report)}
    prepared = {
        "admission": {
            "whole_workload_phase1": summary,
            "phase0_readiness": commitment.phase0_readiness_identity(readiness, required=True),
        }
    }
    assert corpus_release._verify_workload_coverage(root, prepared) == summary
    prepared["admission"]["phase0_readiness"]["report_sha256"] = "0" * 64
    with pytest.raises(SpecError, match="readiness differs"):
        corpus_release._verify_workload_coverage(root, prepared)


def test_generation_lineage_reconciles_selected_inputs_and_exact_capsules(monkeypatch, tmp_path):
    from merlin_experiments.phase0 import evidence
    from merlin_experiments.phase0.coverage_commitment import INPUT_PATH, INPUT_SCHEMA

    generated = tmp_path / "generated"
    capsule = generated / "isa" / "one"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("name: one\n")
    requirement = tmp_path / "requirements.yaml"
    requirement.write_text("scope: {}\n")
    bundle = tmp_path / "phase0"
    (bundle / "coverage").mkdir(parents=True)
    (bundle / "evidence-manifest.json").write_text("{}\n")
    inputs_path = generated / INPUT_PATH
    inputs_path.parent.mkdir()
    inputs_path.write_text(json.dumps({"schema": INPUT_SCHEMA, "conformance": {"scope": {}}}) + "\n")
    coverage_input = {"path": INPUT_PATH.as_posix(), "sha256": fingerprint(inputs_path)}
    receipt_path = bundle / "coverage/generation.json"
    (generated / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "generated": ["isa/one"],
                "coverage_inputs": coverage_input,
                "phase0_evidence": {
                    "generation_receipt": str(receipt_path),
                    "manifest": str(bundle / "evidence-manifest.json"),
                },
            }
        )
    )
    cohorts = {}
    for phase in ("phase1", "phase2"):
        path = bundle / "coverage" / f"{phase}-capsule-coverage.json"
        path.write_text(json.dumps({"status": "incomplete", "cohort": {"n_capsules": 1}}))
        cohorts[phase] = {"path": str(path), "sha256": fingerprint(path), "status": "incomplete", "n_capsules": 1}
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "merlin.phase0_generation.v1",
                "target": "fixture-device",
                "evidence_status": "verified",
                "corpus_manifest": str(generated / "MANIFEST.yaml"),
                "coverage_inputs": coverage_input,
                "capsules_written": 1,
                "capsule_commitments": [{"member": "isa/one", "sha256": fingerprint(capsule)}],
                "cohort_coverage": cohorts,
                "omitted": [{"reason": "fixture omission"}],
            }
        )
    )
    monkeypatch.setattr(
        evidence,
        "load_exported_evidence",
        lambda _root: SimpleNamespace(
            target="fixture-device",
            status="verified",
            source_snapshots=(SimpleNamespace(role="conformance-spec", sha256=fingerprint(requirement)),),
        ),
    )
    plan = {
        "target": "fixture-device",
        "phase0_evidence_bundle": str(bundle),
        "phases": {"0": {"inputs": {"conformance_spec": str(requirement)}}},
    }
    lineage = generation_lineage(plan, generated)
    assert lineage["capsules"] == 1
    assert lineage["generation_receipt_sha256"] == fingerprint(receipt_path)
    (capsule / "capsule.yaml").write_text("name: one\nchanged: true\n")
    with pytest.raises(SpecError, match="emitted capsule bytes"):
        generation_lineage(plan, generated)


def test_selected_instruction_model_is_private_and_bound_to_release(monkeypatch, tmp_path):
    from merlin_experiments.phase0 import evidence

    bundle = tmp_path / "selected-evidence"
    member = bundle / "software/instruction-semantics.json"
    member.parent.mkdir(parents=True)
    member.write_text('{"schema":"merlin.instruction_semantics.v1","target":"fixture-device","status":"UNKNOWN"}\n')
    (bundle / "evidence-manifest.json").write_text('{"schema":"phase0_evidence_v1"}\n')
    monkeypatch.setattr(
        evidence,
        "load_exported_evidence",
        lambda _path: SimpleNamespace(
            target="fixture-device",
            archived_artifacts=(
                ("software/instruction-semantics.json", member.read_bytes()),
                ("evidence-manifest.json", (bundle / "evidence-manifest.json").read_bytes()),
            ),
        ),
    )
    root = tmp_path / "release"
    private = root / "private"
    private.mkdir(parents=True)
    (root / "payload").mkdir()
    (root / "payload/source.txt").write_text("frozen corpus\n")
    commitment = corpus_release._stage_instruction_model(
        {"phase0_evidence_bundle": str(bundle), "target": "fixture-device"}, private
    )
    copied = private / "instruction-semantics.json"
    assert copied.read_bytes() == member.read_bytes()
    assert copied.stat().st_mode & 0o077 == 0
    prepared = {"payload_sha256": fingerprint(root / "payload"), "instruction_semantics": commitment}
    corpus_release._content(root, prepared)
    prepared["instruction_semantics"] = {**commitment, "sha256": "0" * 64}
    with pytest.raises(SpecError, match="private instruction model changed"):
        corpus_release._content(root, prepared)
    copied.chmod(0o644)
    with pytest.raises(SpecError, match="owner-only"):
        corpus_release._content(
            root, {"payload_sha256": fingerprint(root / "payload"), "instruction_semantics": commitment}
        )
    monkeypatch.setattr(
        evidence,
        "load_exported_evidence",
        lambda _path: SimpleNamespace(
            target="fixture-device", archived_artifacts=(("software/instruction-semantics.json", b"changed"),)
        ),
    )
    with pytest.raises(SpecError, match="changed after evidence verification"):
        corpus_release._stage_instruction_model(
            {"phase0_evidence_bundle": str(bundle), "target": "fixture-device"}, tmp_path / "other-private"
        )


def test_selected_rtl_facts_come_from_verified_evidence_not_capsules(monkeypatch, tmp_path):
    from merlin_experiments.phase0 import evidence

    bundle = tmp_path / "phase0"
    member = bundle / "hardware/effective-views/loaded-facts.json"
    member.parent.mkdir(parents=True)
    member.write_text('{"facts":{"target":"fixture-device","selection_marker":"verified"}}')
    generated = bundle / "capsules"
    stale = generated / "hardware/effective-views/loaded-facts.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"facts":{"target":"fixture-device","selection_marker":"stale"}}')
    verified_bytes = member.read_bytes()
    monkeypatch.setattr(
        evidence,
        "load_exported_evidence",
        lambda _path: SimpleNamespace(
            target="fixture-device",
            archived_artifacts=(("hardware/effective-views/loaded-facts.json", verified_bytes),),
        ),
    )
    plan = {"phase0_evidence_bundle": str(bundle), "target": "fixture-device"}
    assert corpus_release._selected_rtl_facts(plan, generated) == member
    member.write_text('{"facts":{"target":"fixture-device","selection_marker":"changed"}}')
    with pytest.raises(SpecError, match="differ from verified evidence"):
        corpus_release._selected_rtl_facts(plan, generated)


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
        # The frozen Phase 0 placement every public member must carry.
        "software_screen": {"status": "admitted", "placements": {"accelerator": 1}},
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
    clang = tmp_path / "third_party/llvm-install/bin/clang-23"
    clang.parent.mkdir(parents=True)
    clang.write_text("#!/bin/sh\nexit 0\n")  # Fixture-only executable for the toolchain presence preflight.
    clang.chmod(0o755)
    # Select THIS toolchain. The package conftest points MERLIN_CLANG at the host's clang-23 when the
    # checkout has one, and a selected install outside the fixture checkout is granted to the bundle
    # -- which, with the content store disabled above, deep-copies the whole LLVM install (~5 GB)
    # into tmp_path for every test that prepares a release.
    monkeypatch.setenv("MERLIN_CLANG", str(clang))
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
        "release": tmp_path / "out/artifacts/protocols/fixture-device/review-fixture",
    }


def test_release_derives_admission_from_staged_members(release_fixture, capsys, monkeypatch):
    from merlin.targetgen import eligibility

    descriptor = release_fixture["root"] / "source-experiment/target_experiment.yaml"
    authored = yaml.safe_load(descriptor.read_text())
    authored["grading"] = {
        "release_admission": "derive_from_corpus_v1",
        "resource_bound": {"policy": "fixture_review", "exclude_capsules": [], "required_admitted_models": []},
    }
    descriptor.write_text(yaml.safe_dump(authored))
    monkeypatch.setattr(eligibility, "capability_map_for_target", lambda _target: {"fixture": object()})

    report = _prepare(release_fixture, capsys)
    promoted = yaml.safe_load(Path(report["descriptor"]).read_text())
    assert "expected_cohort" not in authored["grading"]
    assert "release_admission" not in promoted["grading"]
    assert promoted["grading"]["expected_cohort"] == {"source_capsules": 2, "admitted_capsules": 2}
    assert promoted["grading"]["hidden_capability_admission"] == {"source_capsules": 1, "admitted_capsules": 1}
    assert report["counts"]["public_source"] == 2
    assert report["counts"]["hidden_source"] == 1


def test_release_rejects_absent_selected_llvm(release_fixture, capsys):
    llvm = release_fixture["root"] / "third_party/llvm-install"
    llvm.rename(llvm.with_name("llvm-unselected"))
    assert (
        main(["run", str(release_fixture["definition"]), "--phase", "0", "--run-dir", str(release_fixture["run"])]) == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "corpus",
                "prepare",
                "--historical-overlay",
                str(release_fixture["run"]),
                "--output",
                str(release_fixture["release"]),
            ]
        )
        == 2
    )
    failure = json.loads((release_fixture["release"] / "private/failure.json").read_text())
    assert "selected LLVM/MLIR toolchain is absent" in failure["error"]
    assert not (release_fixture["release"] / "private/seal.json").exists()


def test_generated_only_admission_reports_model_policy_and_private_cohort_gaps(release_fixture, capsys, monkeypatch):
    from merlin.targetgen import eligibility

    fixture = release_fixture
    generated_member = fixture["baseline"] / "isa/generated_member/capsule.yaml"
    model = yaml.safe_load(generated_member.read_text())
    model["kind"] = "model"
    generated_member.write_text(yaml.safe_dump(model))
    descriptor = fixture["root"] / "source-experiment/target_experiment.yaml"
    authored = yaml.safe_load(descriptor.read_text())
    authored["grading"] = {
        "release_admission": "derive_from_corpus_v1",
        "resource_bound": {
            "policy": "fixture_review",
            "exclude_capsules": [],
            "required_admitted_models": ["stale_model"],
        },
    }
    descriptor.write_text(yaml.safe_dump(authored))
    monkeypatch.setattr(eligibility, "capability_map_for_target", lambda _target: {"fixture": object()})

    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "corpus",
                "prepare",
                str(fixture["run"]),
                "--output",
                str(fixture["release"]),
                "--generated-only",
            ]
        )
        == 2
    )
    failure = json.loads((fixture["release"] / "private/failure.json").read_text())
    assert "unclassified public models: generated_member" in failure["error"]
    assert "policy names absent from staged public models: stale_model" in failure["error"]
    assert "hidden grading cohort is empty" in failure["error"]
    assert not (fixture["release"] / "private/seal.json").exists()


def _prepare(fixture, capsys):
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    code = main(["corpus", "prepare", "--historical-overlay", str(fixture["run"]), "--output", str(fixture["release"])])
    output = capsys.readouterr()
    if code:
        diagnostic = fixture["release"] / "private/failure.json"
        pytest.fail(output.err + (diagnostic.read_text() if diagnostic.exists() else ""))
    assert "private_member_identity" not in output.out + output.err
    return json.loads(output.out)


def _phase1_policy(fixture):
    source = fixture["root"] / "source-experiment/target_experiment.yaml"
    document = yaml.safe_load(source.read_text())
    document["phase1_gates"] = {"private_full_models": {"required": True, "models": ["fixture-model"]}}
    selected = fixture["root"] / "phase1-policy-descriptor.yaml"
    selected.write_text(yaml.safe_dump(document, sort_keys=False))
    return selected


def test_explicit_phase1_policy_is_additive_retained_and_review_bound(release_fixture, capsys):
    fixture = release_fixture
    selected = _phase1_policy(fixture)
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    source_output = fixture["run"] / "phase0/capsules"
    original_output_sha = fingerprint(source_output)
    assert (
        main(
            [
                "corpus",
                "prepare",
                "--historical-overlay",
                str(fixture["run"]),
                "--output",
                str(fixture["release"]),
                "--phase1-policy-descriptor",
                str(selected),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["state"] == "awaiting_operator_review"
    assert fingerprint(source_output) == original_output_sha
    promoted = yaml.safe_load((fixture["release"] / "payload/experiment/target_experiment.yaml").read_text())
    assert promoted["phase1_gates"] == yaml.safe_load(selected.read_text())["phase1_gates"]
    retained = fixture["release"] / "private/phase1-policy-descriptor.yaml"
    preparation = json.loads((fixture["release"] / "private/preparation.json").read_text())
    assert preparation["phase1_policy"] == {"source_path": str(selected), "sha256": fingerprint(retained)}
    assert retained.stat().st_mode & 0o077 == 0
    sealed = _seal(fixture, report, capsys)
    selected.write_text("changed after preparation\n")
    assert corpus_release.verify(Path(sealed["seal"]), Path(sealed["descriptor"]))


def test_phase1_policy_refuses_any_non_gate_descriptor_change(release_fixture, capsys):
    fixture = release_fixture
    selected = _phase1_policy(fixture)
    policy = yaml.safe_load(selected.read_text())
    policy["target"] = "wrong-target"
    selected.write_text(yaml.safe_dump(policy))
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "corpus",
                "prepare",
                "--historical-overlay",
                str(fixture["run"]),
                "--output",
                str(fixture["release"]),
                "--phase1-policy-descriptor",
                str(selected),
            ]
        )
        == 2
    )
    assert "only in phase1_gates" in (fixture["release"] / "private/failure.json").read_text()
    assert not (fixture["release"] / "private/seal.json").exists()


def test_phase1_policy_retained_projection_is_checked_before_review(release_fixture, capsys):
    fixture = release_fixture
    selected = _phase1_policy(fixture)
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "corpus",
                "prepare",
                "--historical-overlay",
                str(fixture["run"]),
                "--output",
                str(fixture["release"]),
                "--phase1-policy-descriptor",
                str(selected),
            ]
        )
        == 0
    )
    capsys.readouterr()
    retained = fixture["release"] / "private/phase1-policy-descriptor.yaml"
    retained.chmod(0o600)
    policy = yaml.safe_load(retained.read_text())
    policy["phase1_gates"]["private_full_models"]["models"] = ["different"]
    retained.write_text(yaml.safe_dump(policy))
    retained.chmod(0o400)
    preparation_path = fixture["release"] / "private/preparation.json"
    preparation = json.loads(preparation_path.read_text())
    preparation["phase1_policy"]["sha256"] = fingerprint(retained)
    preparation_path.write_text(json.dumps(preparation))
    with pytest.raises(SpecError, match="promoted Phase 1 gates differ"):
        corpus_release.inspect_release(fixture["release"])


def test_external_private_baseline_is_explicit_and_receipted(release_fixture, capsys):
    fixture = release_fixture
    private_source = fixture["root"] / "operator-private-corpus"
    (fixture["baseline"] / "hidden").rename(private_source)
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()

    absent = fixture["root"] / "out/artifacts/protocols/fixture-device/missing-private"
    assert main(["corpus", "prepare", "--historical-overlay", str(fixture["run"]), "--output", str(absent)]) == 2
    assert "private corpus" in (absent / "private/failure.json").read_text()
    capsys.readouterr()

    assert (
        main(
            [
                "corpus",
                "prepare",
                "--historical-overlay",
                str(fixture["run"]),
                "--output",
                str(fixture["release"]),
                "--private-baseline",
                str(private_source),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    preparation = json.loads((fixture["release"] / "private/preparation.json").read_text())
    assert report["counts"]["hidden_source"] == 1
    assert preparation["assembly"]["baseline"]["hidden"] == {
        "path": str(private_source),
        "sha256": fingerprint(private_source),
    }
    assert (fixture["release"] / "payload/corpus/hidden/private_member_identity/capsule.yaml").is_file()
    assert private_source.is_dir()


def _place(directory: Path, placements: dict | None) -> None:
    path = directory / "capsule.yaml"
    document = yaml.safe_load(path.read_text())
    if placements is None:
        document.pop("software_screen", None)
    else:
        document["software_screen"] = {"status": "admitted", "placements": placements}
    path.write_text(yaml.safe_dump(document))


@pytest.mark.parametrize(
    ("placements", "reason"),
    [
        ({"host": 3}, "no operation of the written program is placed on the accelerator (host=3)"),
        ({}, "no placed operation"),
        (None, "no frozen per-operation placement record"),
    ],
)
def test_assembly_refuses_a_public_member_without_accelerator_placement(release_fixture, tmp_path, placements, reason):
    import shutil

    from merlin.targetgen.target_experiment import load_target_experiment

    fixture = release_fixture
    te = load_target_experiment(fixture["root"] / "source-experiment/target_experiment.yaml")
    generated = tmp_path / "generated"
    _member(generated, "isa", "generated_member", "public")
    _member(generated, "model_slices", "host_member", "public")
    _place(generated / "model_slices/host_member", placements)
    (generated / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {"generated": ["isa/generated_member", "model_slices/host_member"], "held_out": {"n_generated": 0}}
        )
    )
    private_source = tmp_path / "operator-private"
    (fixture["baseline"] / "hidden").rename(private_source)
    shutil.rmtree(fixture["baseline"])
    with pytest.raises(SpecError, match="1 public capsule\\(s\\) carry no accelerator placement") as refused:
        assemble(te, generated, tmp_path / "release-corpus", private_baseline=private_source, generated_only=True)
    detail = str(refused.value).split("host_member: ", 1)[1]
    assert reason in detail
    assert "generated_member" not in str(refused.value)


def test_prepare_refuses_a_public_capsule_without_accelerator_placement(release_fixture, capsys):
    fixture = release_fixture
    _place(fixture["baseline"] / "isa/generated_member", {"host": 2})
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    command = ["corpus", "prepare", str(fixture["run"]), "--output", str(fixture["release"]), "--generated-only"]
    assert main(command) == 2
    failure = json.loads((fixture["release"] / "private/failure.json").read_text())
    assert "generated_member: no operation of the written program is placed on the accelerator" in failure["error"]
    assert "host placement is Merlin-supplied" in failure["error"]
    assert not (fixture["release"] / "private/preparation.json").exists()


def test_admission_rechecks_accelerator_placement_of_the_reviewed_payload(tmp_path):
    from merlin_experiments.corpus.publication import require_accelerator_placement

    _member(tmp_path, "isa", "accelerated", "public")
    _member(tmp_path, "hidden", "withheld", "hidden")
    _place(tmp_path / "hidden/withheld", {"host": 1})
    # Only the openly graded cohort is judged; the count names what was checked.
    assert require_accelerator_placement([tmp_path / "isa", tmp_path / "hidden"]) == 1
    _place(tmp_path / "isa/accelerated", {"accelerator": 0, "host": 4})
    with pytest.raises(SpecError, match="accelerated: no operation"):
        require_accelerator_placement([tmp_path / "isa"])


def test_generated_only_assembly_needs_no_legacy_corpus(release_fixture, tmp_path):
    import shutil

    from merlin.targetgen.target_experiment import load_target_experiment

    fixture = release_fixture
    te = load_target_experiment(fixture["root"] / "source-experiment/target_experiment.yaml")
    generated = tmp_path / "generated"
    _member(generated, "isa", "generated_member", "public")
    (generated / "MANIFEST.yaml").write_text(
        yaml.safe_dump({"generated": ["isa/generated_member"], "held_out": {"n_generated": 0}})
    )
    private_source = tmp_path / "operator-private"
    (fixture["baseline"] / "hidden").rename(private_source)
    shutil.rmtree(fixture["baseline"])

    destination = tmp_path / "release-corpus"
    receipt = assemble(te, generated, destination, private_baseline=private_source, generated_only=True)
    assert receipt["mode"] == "generated_only"
    assert set(receipt["baseline"]) == {"hidden"}
    assert sorted(path.parent.relative_to(destination).as_posix() for path in destination.glob("*/*/capsule.yaml")) == [
        "hidden/private_member_identity",
        "isa/generated_member",
    ]
    promoted = yaml.safe_load((destination / "MANIFEST.yaml").read_text())
    assert promoted["hand_authored"] == []
    assert promoted["held_out"] == {"n_generated": 0, "n_hand_authored": 1}


def test_generated_only_release_accepts_no_historical_corpus_selection(release_fixture, capsys):
    descriptor = release_fixture["root"] / "source-experiment/target_experiment.yaml"
    authored = yaml.safe_load(descriptor.read_text())
    authored.pop("capsule_corpus")
    authored["grading"]["expected_cohort"] = {"source_capsules": 1, "admitted_capsules": 1}
    descriptor.write_text(yaml.safe_dump(authored))
    assert (
        main(["run", str(release_fixture["definition"]), "--phase", "0", "--run-dir", str(release_fixture["run"])]) == 0
    )
    capsys.readouterr()
    report = corpus_release.prepare(
        release_fixture["run"],
        release_fixture["release"],
        generated_only=True,
        private_baseline=release_fixture["baseline"] / "hidden",
    )
    assert report["counts"]["public_source"] == report["counts"]["public_admitted"] == 1
    assert report["counts"]["hidden_source"] == report["counts"]["hidden_admitted"] == 1
    promoted = yaml.safe_load(Path(report["descriptor"]).read_text())
    assert Path(promoted["capsule_corpus"]) == release_fixture["release"] / "payload/corpus/isa"
    assert "capsule_corpus" not in yaml.safe_load(descriptor.read_text())


def test_generated_corpus_anchor_preserves_all_public_categories_and_private_boundary(tmp_path):
    from merlin_experiments.corpus.generated_source import bind_generated_corpus

    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text("target: fixture-device\n")
    te = load_target_experiment(descriptor)
    generated = tmp_path / "generated"
    for category, label in (("z_public", "public"), ("a_public", "public"), ("hidden", "hidden"), ("_feedback", "dev")):
        _member(generated, category, category + "_member", label)
    bound = bind_generated_corpus(te, generated, generated_only=True)
    assert {path.name for path in bound.graded_roots()} == {"a_public", "z_public"}
    assert bound.hidden_roots() == [generated / "hidden"]
    assert te.capsule_corpus is None
    with pytest.raises(SpecError, match="requires generated-only"):
        bind_generated_corpus(te, generated, generated_only=False)
    private_only = tmp_path / "private-only"
    _member(private_only, "hidden", "secret", "hidden")
    with pytest.raises(SpecError, match="no public capsule category"):
        bind_generated_corpus(te, private_only, generated_only=True)


def test_default_release_cli_ignores_removed_legacy_public_corpus(release_fixture, capsys, monkeypatch):
    import shutil

    from merlin_experiments.phase1 import corpus_inputs

    from merlin.common.paths import data_path
    from merlin.targetgen import capsule_runner
    from merlin.targetgen.sandbox import bwrap
    from merlin.targetgen.target_experiment import load_target_experiment

    fixture = release_fixture
    descriptor = fixture["root"] / "source-experiment/target_experiment.yaml"
    authored = yaml.safe_load(descriptor.read_text())
    authored["grading"]["expected_cohort"] = {"source_capsules": 1, "admitted_capsules": 1}
    descriptor.write_text(yaml.safe_dump(authored))
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    private_source = fixture["root"] / "operator-private-corpus"
    (fixture["baseline"] / "hidden").rename(private_source)
    shutil.rmtree(fixture["baseline"])

    assert (
        main(
            [
                "corpus",
                "prepare",
                str(fixture["run"]),
                "--output",
                str(fixture["release"]),
                "--private-baseline",
                str(private_source),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["counts"]["public_source"] == 1
    assert report["counts"]["hidden_source"] == 1
    sealed = corpus_release.seal(
        fixture["release"],
        expected_digest=report["review_digest"],
        reviewed_by="fixture operator",
        review_note="synthetic release handoff test",
    )
    assert sealed["state"] == "sealed"
    preparation = json.loads((fixture["release"] / "private/preparation.json").read_text())
    assert preparation["assembly"]["mode"] == "generated_only"
    assert set(preparation["assembly"]["baseline"]) == {"hidden"}
    promoted = load_target_experiment(Path(report["descriptor"]))
    monkeypatch.setattr(capsule_runner, "qa_loop_adapters", lambda *args, **kwargs: {"L0": object()})
    monkeypatch.setattr(capsule_runner, "oracle_adapters", lambda *args, **kwargs: {"L0": object()})
    phase1_run = fixture["root"] / "phase1-run"
    phase1_run.mkdir()
    bundle = yaml.safe_load(
        (
            Path(report["descriptor"]).parent / "input_bundles/raw_baseline_public_v0/input_bundle_manifest.yaml"
        ).read_text()
    )
    effective, record = corpus_inputs.stage(phase1_run, promoted, bundle, contract=data_path("contract"))
    workspace = fixture["root"] / "phase1-workspace"
    bwrap.materialize_bundle_inputs(workspace, effective, repo=fixture["root"])
    corpus_release.verify_snapshot(
        fixture["release"] / "private/seal.json", Path(report["descriptor"]), workspace, effective, repo=fixture["root"]
    )
    view = corpus_inputs.resolve(
        workspace, effective, record, repo=fixture["root"], reviewed_roots=tuple(promoted.graded_roots())
    )
    assert (view.public / "generated_member/capsule.yaml").is_file()
    assert (view.policy / "0/generated_member/capsule.yaml").is_file()


def test_generated_only_release_refuses_implicit_legacy_hidden_corpus(release_fixture, capsys):
    fixture = release_fixture
    descriptor = fixture["root"] / "source-experiment/target_experiment.yaml"
    authored = yaml.safe_load(descriptor.read_text())
    authored["grading"]["expected_cohort"] = {"source_capsules": 1, "admitted_capsules": 1}
    descriptor.write_text(yaml.safe_dump(authored))
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "corpus",
                "prepare",
                str(fixture["run"]),
                "--output",
                str(fixture["release"]),
                "--generated-only",
            ]
        )
        == 2
    )
    failure = json.loads((fixture["release"] / "private/failure.json").read_text())
    assert "private corpus" in failure["error"] or "hidden grading cohorts" in failure["error"]


def test_retired_generated_member_requires_exact_review_and_is_removed_from_copy(release_fixture, tmp_path):
    from merlin.targetgen.target_experiment import load_target_experiment

    fixture = release_fixture
    baseline = fixture["baseline"]
    manifest = yaml.safe_load((baseline / "MANIFEST.yaml").read_text())
    manifest["generated"] = ["layers/retained_member"]
    manifest["hand_authored"] = ["isa/generated_member"]
    (baseline / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    generated = tmp_path / "generated"
    _member(generated, "isa", "generated_member", "public")
    (generated / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "generated": ["isa/generated_member"],
                "held_out": {"n_generated": 0},
            }
        )
    )
    te = load_target_experiment(fixture["root"] / "source-experiment/target_experiment.yaml")
    with pytest.raises(SpecError, match="retirement review must account exactly"):
        assemble(te, generated, tmp_path / "unreviewed", generated_only=False)
    review = tmp_path / "retirements.yaml"
    review.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "retired": {"layers/retained_member": "No longer derived by this profile"},
            }
        )
    )
    destination = tmp_path / "reviewed"
    receipt = assemble(te, generated, destination, retirements=review, generated_only=False)
    assert not (destination / "layers/retained_member").exists()
    assert (baseline / "layers/retained_member/capsule.yaml").exists()
    assert receipt["retirements"]["sha256"] == fingerprint(review)
    promoted = yaml.safe_load((destination / "MANIFEST.yaml").read_text())
    assert promoted["retired_generated"]["count"] == 1
    assert "retained_member" not in str(promoted["retired_generated"])


def test_reviewed_hand_authored_retirement_is_private_and_release_local(release_fixture, tmp_path):
    from merlin.targetgen.target_experiment import load_target_experiment

    fixture = release_fixture
    baseline = fixture["baseline"]
    manifest = yaml.safe_load((baseline / "MANIFEST.yaml").read_text())
    manifest["generated"] = ["isa/generated_member"]
    manifest["hand_authored"] = ["layers/retained_member"]
    (baseline / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    generated = tmp_path / "generated"
    _member(generated, "isa", "generated_member", "public")
    (generated / "MANIFEST.yaml").write_text(
        yaml.safe_dump({"generated": ["isa/generated_member"], "held_out": {"n_generated": 0}})
    )
    te = load_target_experiment(fixture["root"] / "source-experiment/target_experiment.yaml")
    review = tmp_path / "retirements.yaml"
    review.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "retired": {"layers/retained_member": "Move claim behind owner-only validation"}}
        )
    )
    destination = tmp_path / "reviewed"
    receipt = assemble(te, generated, destination, retirements=review, generated_only=False)
    assert not (destination / "layers/retained_member").exists()
    assert (baseline / "layers/retained_member/capsule.yaml").exists()
    assert [row["member"] for row in receipt["retirements"]["members"]] == ["layers/retained_member"]
    promoted = yaml.safe_load((destination / "MANIFEST.yaml").read_text())
    assert promoted["retired_hand_authored"]["count"] == 1
    assert "retained_member" not in str(promoted)


@pytest.mark.parametrize("member", ["isa/generated_member", "layers/absent_member"])
def test_hand_authored_retirement_refuses_fresh_or_unknown_member(release_fixture, tmp_path, member):
    from merlin.targetgen.target_experiment import load_target_experiment

    fixture = release_fixture
    baseline = fixture["baseline"]
    manifest = yaml.safe_load((baseline / "MANIFEST.yaml").read_text())
    manifest["generated"] = ["isa/generated_member"]
    manifest["hand_authored"] = ["layers/retained_member"]
    (baseline / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    generated = tmp_path / "generated"
    _member(generated, "isa", "generated_member", "public")
    (generated / "MANIFEST.yaml").write_text(
        yaml.safe_dump({"generated": ["isa/generated_member"], "held_out": {"n_generated": 0}})
    )
    review = tmp_path / "retirements.yaml"
    review.write_text(yaml.safe_dump({"schema_version": 1, "retired": {member: "Invalid retirement"}}))
    te = load_target_experiment(fixture["root"] / "source-experiment/target_experiment.yaml")
    with pytest.raises(SpecError, match="retirement review must account exactly"):
        assemble(te, generated, tmp_path / "refused", retirements=review, generated_only=False)


@pytest.mark.parametrize("damage", ["missing_member", "public_generated_alias"])
def test_external_private_baseline_rejects_incomplete_or_aliased_source(release_fixture, capsys, damage):
    fixture = release_fixture
    private_source = fixture["root"] / "operator-private-corpus"
    (fixture["baseline"] / "hidden").rename(private_source)
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    private_member = private_source / "private_member_identity"
    if damage == "missing_member":
        (private_member / "capsule.yaml").unlink()
    else:
        (private_member / "golden.yaml").unlink()
        os.link(fixture["run"] / "phase0/capsules/isa/generated_member/golden.yaml", private_member / "golden.yaml")
    assert (
        main(
            [
                "corpus",
                "prepare",
                "--historical-overlay",
                str(fixture["run"]),
                "--output",
                str(fixture["release"]),
                "--private-baseline",
                str(private_source),
            ]
        )
        == 2
    )
    assert not (fixture["release"] / "private/seal.json").exists()


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


def test_prepared_release_carries_generated_manifest_not_live_checkout(release_fixture, capsys):
    fixture = release_fixture
    _prepare(fixture, capsys)
    promoted = fixture["release"] / "payload/corpus/MANIFEST.yaml"
    manifest = yaml.safe_load(promoted.read_text())
    assert manifest["generated"] == ["isa/generated_member"]
    assert manifest["hand_authored"] == ["layers/retained_member"]
    assert manifest["generated_by"] == "derive.py"
    assert promoted.stat().st_ino != (fixture["baseline"] / "MANIFEST.yaml").stat().st_ino


def test_prepared_release_binds_exact_generated_facts_to_rtl_arm(release_fixture, capsys):
    fixture = release_fixture
    derivation = fixture["root"] / "derive.py"
    derivation.write_text(
        derivation.read_text()
        + "facts=output/'hardware/effective-views/loaded-facts.json'\n"
        + "facts.parent.mkdir(parents=True)\n"
        + 'facts.write_text(\'{"facts":{"target":"fixture-device","selection_marker":"phase0"}}\')\n'
    )
    _prepare(fixture, capsys)
    experiment = fixture["release"] / "payload/experiment"
    selected = experiment / "rtl_facts/facts.json"
    assert json.loads(selected.read_text())["facts"]["selection_marker"] == "phase0"
    bundle = yaml.safe_load(
        (experiment / "input_bundles/merlin_assisted_rtlchecks_public_v0/input_bundle_manifest.yaml").read_text()
    )
    assert bundle["selected_rtl_facts_file"] == str(selected)
    assert str(selected.parent) + "/" in {row["path"] for row in bundle["allowed"]}
    denied = yaml.safe_load(
        (experiment / "input_bundles/merlin_assisted_public_v0/input_bundle_manifest.yaml").read_text()
    )
    assert "selected_rtl_facts_file" not in denied


def test_prepared_operand_observer_keeps_public_facts_and_private_source_closure(release_fixture, capsys):
    import hashlib
    import runpy

    from merlin_experiments.phase0.coverage_commitment import INPUT_SCHEMA, read_inputs
    from merlin_experiments.phase1.corpus_inputs import require_reviewed_bundle

    from merlin.targetgen.rocc.selected_operand_protocol import selected_inputs
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.target_experiment import load_target_experiment

    fixture = release_fixture
    root = fixture["root"]
    checkout = Path(__file__).resolve().parents[3]
    observer = runpy.run_path(str(checkout / "merlin/tests/targetgen/test_selected_operand_protocol.py"))
    _, facts, _, sidecar, firrtl, isa = observer["selected"](root)
    runtime = runpy.run_path(str(checkout / "merlin/tests/runtime/test_bound_chipyard_rocc_backend.py"))
    contract, native_facts = runtime["selection"].__wrapped__(root)
    contract["name"] = "fixture-device"
    contract["rocc_operand_roles"]["instructions"] = [{"funct": 4, "class": "INPUT", "operands": {}}]
    native_facts["inputs"].update(facts["inputs"])
    native_facts["inputs"]["target"] = "fixture-device"
    native_facts["facts"]["target"] = "fixture-device"
    native_facts["facts"]["interfaces"] = facts["facts"]["interfaces"]
    fact_bytes = json.dumps(native_facts).encode()
    sidecar["target"] = "fixture-device"
    sidecar["original_facts_sha256"] = hashlib.sha256(fact_bytes).hexdigest()
    original_sidecar = root / "operand-protocol.json"
    original_sidecar.write_text(json.dumps(sidecar))
    coverage_inputs = {
        "schema": INPUT_SCHEMA,
        "capability_contract": contract,
        "raw_facts_utf8": fact_bytes.decode(),
        "evidence": {"raw_facts_sha256": hashlib.sha256(fact_bytes).hexdigest()},
    }
    derivation = root / "derive.py"
    derivation.write_text(
        derivation.read_text()
        + "facts=output/'hardware/effective-views/loaded-facts.json'\n"
        + "facts.parent.mkdir(parents=True)\n"
        + f"facts.write_bytes({fact_bytes!r})\n"
        + "contract=output/'software/contract.json'\n"
        + "contract.parent.mkdir(parents=True)\n"
        + f"contract.write_text({json.dumps(contract)!r})\n"
        + "from merlin_experiments.phase0.coverage_commitment import write_inputs\n"
        + "manifest=yaml.safe_load((output/'MANIFEST.yaml').read_bytes())\n"
        + f"manifest['coverage_inputs']=write_inputs(output,{coverage_inputs!r})\n"
        + "(output/'MANIFEST.yaml').write_text(yaml.safe_dump(manifest))\n"
    )
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    corpus_release.prepare(
        fixture["run"],
        fixture["release"],
        generated_only=False,
        operand_protocol_selection=(original_sidecar, hashlib.sha256(original_sidecar.read_bytes()).hexdigest()),
    )
    experiment = fixture["release"] / "payload/experiment"
    te = load_target_experiment(experiment / "target_experiment.yaml")
    assert read_inputs(te.capsule_corpus.parent) == coverage_inputs
    public_facts = experiment / "rtl_facts/facts.json"
    private_facts = experiment / "contracts/operand-facts.json"
    assert public_facts.read_bytes() == private_facts.read_bytes() == fact_bytes
    assert public_facts.stat().st_ino != private_facts.stat().st_ino
    selected_contract = yaml.safe_load((experiment / "contracts/target_contract.yaml").read_text())
    selection = selected_contract["rocc_operand_selection"]
    assert selection["original_facts"] == {
        "path": str(private_facts),
        "sha256": hashlib.sha256(fact_bytes).hexdigest(),
    }
    checker_inputs = set(selected_inputs(selection))
    assert checker_inputs == {
        str(private_facts),
        str(experiment / "contracts/operand-protocol.json"),
        str(isa),
        str(firrtl),
    }
    for arm in ("merlin_assisted_rtlchecks_public_v0", "raw_baseline_public_v0"):
        authored = experiment / "input_bundles" / arm / "input_bundle_manifest.yaml"
        bundle = yaml.safe_load(authored.read_text())
        require_reviewed_bundle(te, authored, bundle)
        incomplete = {**bundle, "host_inputs": [row for row in bundle["host_inputs"] if row["path"] != str(firrtl)]}
        with pytest.raises(ValueError, match="complete original-only frozen source roster"):
            require_reviewed_bundle(te, authored, incomplete)
        hosts = {row["path"] for row in bundle["host_inputs"]}
        assert checker_inputs <= hosts
        workspace = root / arm / "workspace"
        workspace.mkdir(parents=True)
        try:
            BW.materialize_bundle_inputs(workspace, bundle, repo=root)
            policy = BW.base_argv(workspace, bundle, repo=root)
            assert not BW.is_exposed(policy, private_facts)
            assert not BW.is_exposed(policy, isa)
            assert not BW.is_exposed(policy, firrtl)
            if arm == "merlin_assisted_rtlchecks_public_v0":
                assert all(not public_facts.is_relative_to(Path(path)) for path in hosts)
                assert bundle["selected_rtl_facts_file"] == str(public_facts)
                assert BW.is_exposed(policy, public_facts)
            else:
                assert "selected_rtl_facts_file" not in bundle
                assert not BW.is_exposed(policy, public_facts)
        finally:
            BW.remove_bundle_snapshot(workspace)
        original_mode = public_facts.stat().st_mode
        public_facts.chmod(original_mode | 0o200)
        public_facts.write_bytes(fact_bytes + b"\n")
        with pytest.raises(ValueError, match="selected byte pin"):
            require_reviewed_bundle(te, authored, bundle)
        public_facts.write_bytes(fact_bytes)
        public_facts.chmod(original_mode)
        require_reviewed_bundle(te, authored, bundle)
    for source in (firrtl, isa, private_facts):
        original = source.read_bytes()
        original_mode = source.stat().st_mode
        source.chmod(original_mode | 0o200)
        source.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="changed|pin"):
            selected_inputs(selection)
        with pytest.raises(ValueError):
            require_reviewed_bundle(te, authored, bundle)
        assert public_facts.read_bytes() == fact_bytes
        source.write_bytes(original)
        source.chmod(original_mode)
        assert set(selected_inputs(selection)) == checker_inputs
        require_reviewed_bundle(te, authored, bundle)
    contract_path = te.declared_contract_path()
    original_contract = contract_path.read_bytes()
    original_mode = contract_path.stat().st_mode
    contract_path.chmod(original_mode | 0o200)
    selected_contract["rocc_operand_selection"]["original_facts"]["path"] = str(
        experiment / "contracts/other-facts.json"
    )
    contract_path.write_text(yaml.safe_dump(selected_contract))
    with pytest.raises(ValueError, match="exact release-local facts and sidecar"):
        require_reviewed_bundle(te, authored, bundle)
    contract_path.write_bytes(original_contract)
    contract_path.chmod(original_mode)
    require_reviewed_bundle(te, authored, bundle)


def test_release_hands_selected_capability_view_to_native_phase1(release_fixture, capsys, monkeypatch):
    from merlin_experiments.phase1 import source_inputs
    from merlin_experiments.phase1.context import load_context
    from merlin_experiments.runner import _phase1_source_inputs

    from merlin.targetgen.target_experiment import declared_vs_resolved_contract, load_target_experiment
    from merlin.targetgen.target_registry import resolve

    monkeypatch.setattr(os, "environ", dict(os.environ))
    fixture = release_fixture
    root = fixture["root"]
    support = root / "support"
    (support / "contracts").mkdir(parents=True)
    provider_contract = support / "contracts/target_contract.yaml"
    provider_contract.write_text("name: fixture-device\nplugin: {backend: provider_backend}\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(support))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    selected = {"name": "fixture-device", "selection_marker": "phase0-effective", "eligibility": {"compute": []}}
    raw = json.dumps(selected, sort_keys=True, indent=2) + "\n"
    derivation = root / "derive.py"
    derivation.write_text(
        derivation.read_text()
        + "contract=output/'software/contract.json'\n"
        + "contract.parent.mkdir(parents=True)\n"
        + f"contract.write_text({raw!r})\n"
    )
    report = _prepare(fixture, capsys)
    descriptor = Path(report["descriptor"])
    te = load_target_experiment(descriptor)
    copied = te.declared_contract_path()
    assert copied is not None and copied.is_relative_to(descriptor.parent)
    assert copied.read_text() == raw
    assert "MERLIN_TARGET_CONTRACT" not in os.environ  # preparation does not leak its selection
    for manifest in descriptor.parent.glob("input_bundles/*/input_bundle_manifest.yaml"):
        bundle = yaml.safe_load(manifest.read_text())
        assert str(copied) in {row["path"] for row in bundle["allowed"]}
    prepared = json.loads((fixture["release"] / "private/preparation.json").read_text())
    assert prepared["scaffolding"]["capability_contract"]["sha256"] == fingerprint(copied)

    command = {
        "env": {"MERLIN_REPO_ROOT": str(root), "MERLIN_TARGET_CONTRACT": str(copied)},
        "entrypoint": str(root / "installed.py"),
        "inputs": {"descriptor": str(descriptor)},
    }
    frozen = _phase1_source_inputs(command)
    assert frozen["phase1:startup:target_contract"] == str(copied)
    assert frozen["phase1:startup:provider:contract"] == str(provider_contract)
    assert "MERLIN_TARGET_CONTRACT" not in os.environ

    load_context(descriptor, repo=root)
    assert os.environ["MERLIN_TARGET_CONTRACT"] == str(copied)
    assert resolve(te.target).load_contract() == selected
    assert resolve(te.target).plugin()["backend"] == "provider_backend"
    assert declared_vs_resolved_contract(te) == (copied, copied, "agree")
    arguments = {"repo": root, "entrypoint": root / "installed.py", "descriptor": descriptor}
    record = source_inputs.record(**arguments)
    source_inputs.verify(record, **arguments)
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(provider_contract))
    with pytest.raises(ValueError, match="capability contract.*differs"):
        load_context(descriptor, repo=root)
    assert copied.stat().st_nlink == 1
    copied.chmod(0o600)  # Deliberate tampering with this fixture's independent frozen copy.
    copied.write_text(raw + "\n")
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(copied))
    with pytest.raises(SpecError, match="source identity changed"):
        source_inputs.verify(record, **arguments)


def test_selected_capability_view_uses_verified_phase0_export(monkeypatch, tmp_path):
    from merlin_experiments.phase0 import evidence

    bundle = tmp_path / "phase0"
    member = bundle / "software/contract.json"
    member.parent.mkdir(parents=True)
    member.write_text('{"name":"fixture-device","selection_marker":"verified"}\n')
    generated = bundle / "capsules"
    stale = generated / "software/contract.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"name":"fixture-device","selection_marker":"stale"}\n')
    raw = member.read_bytes()
    monkeypatch.setattr(
        evidence,
        "load_exported_evidence",
        lambda _: SimpleNamespace(archived_artifacts=(("software/contract.json", raw),)),
    )
    plan = {"phase0_evidence_bundle": str(bundle), "target": "fixture-device"}
    selected = corpus_release._selected_capability_contract(plan, generated)
    assert selected == (member, fingerprint(member))
    member.write_text(stale.read_text())
    from merlin_experiments.corpus.preparation import copy_input

    with pytest.raises(SpecError, match="differs from verified evidence"):
        copy_input(selected[0], tmp_path / "staged.yaml", expected_sha256=selected[1])
    with pytest.raises(SpecError, match="capability contract.*verified evidence"):
        corpus_release._selected_capability_contract(plan, generated)


def test_reviewed_phase1_refuses_missing_or_different_phase0_contract(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import coverage_commitment
    from merlin_experiments.phase1.corpus_inputs import require_reviewed_bundle

    from merlin.targetgen.target_experiment import load_target_experiment

    corpus = tmp_path / "corpus/isa"
    corpus.mkdir(parents=True)
    descriptor = tmp_path / "experiment/target_experiment.yaml"
    descriptor.parent.mkdir()
    descriptor.write_text(yaml.safe_dump({"target": "fixture-device", "capsule_corpus": str(corpus)}))
    manifest = descriptor.parent / "input_bundles/raw_baseline_public_v0/input_bundle_manifest.yaml"
    contract = descriptor.parent / "contracts/target_contract.yaml"
    selected = {"name": "fixture-device", "marker": "phase0"}
    monkeypatch.setattr(coverage_commitment, "read_inputs", lambda _: {"capability_contract": selected})
    bundle = {"bundle_id": "raw_baseline_public_v0", "allowed": [{"path": str(contract), "mode": "ro"}]}
    with pytest.raises(ValueError, match="freeze a new run"):
        require_reviewed_bundle(load_target_experiment(descriptor), manifest, bundle)
    contract.parent.mkdir()
    contract.write_text(yaml.safe_dump({"name": "fixture-device", "marker": "provider"}))
    document = yaml.safe_load(descriptor.read_bytes())
    document["hardware_spec"] = {"target_contract": str(contract)}
    descriptor.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="differs from selected Phase 0"):
        require_reviewed_bundle(load_target_experiment(descriptor), manifest, bundle)
    contract.write_text(yaml.safe_dump(selected))
    require_reviewed_bundle(load_target_experiment(descriptor), manifest, bundle)
    bundle["allowed"] = []
    with pytest.raises(ValueError, match="read-only"):
        require_reviewed_bundle(load_target_experiment(descriptor), manifest, bundle)


def test_prepared_release_materializes_declared_hardware_links(release_fixture, capsys):
    fixture = release_fixture
    root = fixture["root"]
    derivation = root / "derive.py"
    derivation.write_text(
        derivation.read_text()
        + "facts=output/'hardware/effective-views/loaded-facts.json'\n"
        + "facts.parent.mkdir(parents=True)\n"
        + 'facts.write_text(\'{"facts":{"target":"fixture-device"}}\')\n'
    )
    hardware = root / "public-hardware"
    hardware.mkdir()
    external = root / "selected-rtl"
    (external / "include").mkdir(parents=True)
    (external / "rtl").mkdir()
    (external / "include/device.h").write_text("#define DEVICE_DIM 16\n")
    (external / "rtl/device.scala").write_text("class Device\n")
    (hardware / "include").symlink_to(external / "include", target_is_directory=True)
    (hardware / "rtl").symlink_to(external / "rtl", target_is_directory=True)
    descriptor = root / "source-experiment/target_experiment.yaml"
    source_doc = yaml.safe_load(descriptor.read_text())
    source_doc["hardware_spec"] = {
        "hwbringup_set": "public-hardware",
        "isa_headers": ["public-hardware/include/device.h"],
    }
    descriptor.write_text(yaml.safe_dump(source_doc))

    report = _prepare(fixture, capsys)
    staged = fixture["release"] / "payload/experiment/hardware_spec/hwbringup"
    assert (staged / "include/device.h").read_text() == "#define DEVICE_DIM 16\n"
    assert (staged / "rtl/device.scala").read_text() == "class Device\n"
    assert not any(member.is_symlink() for member in staged.rglob("*"))
    promoted = yaml.safe_load((fixture["release"] / "payload/experiment/target_experiment.yaml").read_text())
    assert promoted["hardware_spec"]["hwbringup_set"] == str(staged)
    assert promoted["hardware_spec"]["isa_headers"] == [str(staged / "include/device.h")]
    preparation = json.loads((fixture["release"] / "private/preparation.json").read_text())
    links = preparation["scaffolding"]["hardware_spec/hwbringup"]["materialized_links"]
    assert [row["path"] for row in links] == ["include", "rtl"]
    bundle = yaml.safe_load(
        (
            fixture["release"]
            / "payload/experiment/input_bundles/merlin_assisted_rtlchecks_public_v0/input_bundle_manifest.yaml"
        ).read_text()
    )
    assert str(staged) in {row["path"] for row in bundle["allowed"]}
    assert bundle["selected_rtl_facts_file"] == str(fixture["release"] / "payload/experiment/rtl_facts/facts.json")
    sealed = _seal(fixture, report, capsys)
    definition = _phase1_definition(fixture, sealed)
    assert main(["preflight", str(definition), "--phase", "1"]) == 0


def test_legacy_selected_synthesis_can_prepare_but_cannot_seal(release_fixture, capsys, monkeypatch):
    """A pre-gate run stays inspectable; review cannot upgrade missing lineage."""
    from merlin_experiments import runner

    fixture = release_fixture
    root = fixture["root"]
    script = root / "derive.py"
    script.write_text(script.read_text().replace("a=p.parse_args()", "a=p.parse_known_args()[0]"))
    (root / "recipe.yaml").write_text("capsules: []\n")
    (root / "performance.yaml").write_text("sweeps: []\n")
    (root / "old.synth.yaml").write_text("provenance: {}\ncapsules: []\n")
    definition = yaml.safe_load(fixture["definition"].read_text())
    definition["phases"]["0"]["config"].update(
        recipe="recipe.yaml", performance_template="performance.yaml", synth_profile="old.synth.yaml"
    )
    fixture["definition"].write_text(yaml.safe_dump(definition))
    original = runner.preflight

    def old_preflight(plan):
        report = original(plan)
        report["errors"] = [error for error in report["errors"] if "unverified_legacy" not in error]
        report["configuration_ready"] = not report["errors"]
        return report

    with monkeypatch.context() as previous_runner:
        previous_runner.setattr(runner, "preflight", old_preflight)
        report = _prepare(fixture, capsys)
    assert report["review_digest"]
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
                "historical diagnostic only",
            ]
        )
        != 0
    )
    assert not (fixture["release"] / "private/seal.json").exists()


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


def test_public_coverage_reads_the_completed_phase0_run(release_fixture, capsys, monkeypatch):
    fixture = release_fixture
    generated = fixture["baseline"] / "isa/generated_member/capsule.yaml"
    capsule = yaml.safe_load(generated.read_text())
    capsule["semantic"] = {"semantic_family": "contraction"}
    capsule["inputs"] = [{"name": "A", "shape": [16, 16], "dtype": "i8"}]
    generated.write_text(yaml.safe_dump(capsule))
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    spec = fixture["root"] / "conformance.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "target": "fixture-device",
                "cells": [{"cell": "contraction/i8/aligned"}],
                "boundaries": {"tile_edge": 16},
            }
        )
    )
    from merlin.targetgen import conformance

    def ambient_coverage(*_args, **_kwargs):
        pytest.fail("public coverage reopened ambient target facts instead of frozen inputs")

    monkeypatch.setattr(conformance, "uncovered", ambient_coverage)
    assert main(["corpus", "coverage", str(fixture["run"]), "--spec", str(spec)]) == 0
    report = capsys.readouterr()
    result = json.loads(report.out)
    assert result["scope"] == "generated public source pool; not admitted, graded, or certified"
    assert result["coverage"]["n_required"] == 1
    assert result["coverage"]["n_covered"] == 1
    assert result["coverage"]["uncovered"] == []
    assert result["coverage"]["composition"]["phase"] == "phase0"
    assert "private_member_identity" not in report.out + report.err


@pytest.mark.parametrize("damage", ["undeclared", "nested"])
def test_public_coverage_rejects_silently_ignored_members(tmp_path, damage):
    _member(tmp_path, "isa", "declared", "public")
    (tmp_path / "MANIFEST.yaml").write_text(yaml.safe_dump({"generated": ["isa/declared"]}))
    if damage == "undeclared":
        _member(tmp_path, "layers", "extra", "public")
    else:
        nested = tmp_path / "isa/declared/extra"
        nested.mkdir()
        (nested / "capsule.yaml").write_text("name: extra\nlabel: public\n")
    with pytest.raises(SpecError, match="does not account|outside category/member layout"):
        _public_category_roots(tmp_path)


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
    assert main(["preflight", *args]) == 2
    assert "selected release's generated bundle manifest" in capsys.readouterr().out
    args[-1] = str(retained_bundle)
    assert main(["preflight", *args]) == 0
    assert json.loads(capsys.readouterr().out)["configuration_ready"] is True
    unsafe = yaml.safe_load(selected_bundle.read_text())
    unsafe["allowed"].append({"path": "merlin/contract/", "mode": "ro"})
    selected_bundle.write_text(yaml.safe_dump(unsafe))
    args[-1] = str(selected_bundle)
    assert main(["preflight", *args]) == 2
    assert "selected release's generated bundle manifest" in capsys.readouterr().out


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
    (selected / "env/p").mkdir(parents=True)
    (selected / "env/v").mkdir()
    (selected / "env/p/link.ld").write_text("SECTIONS {}\n")
    (selected / "env/v/link.ld").symlink_to("../p/link.ld")
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
    assert not (copied / "env/v/link.ld").is_symlink()
    assert (copied / "env/v/link.ld").read_bytes() == (selected / "env/p/link.ld").read_bytes()
    preparation = json.loads((fixture["release"] / "private/preparation.json").read_text())
    assert preparation["scaffolding"]["contracts/harness"]["materialized_file_links"] == ["env/v/link.ld"]
    sealed = _seal(fixture, report, capsys)
    (selected / "runtime.h").write_text("/* changed after preparation */\n")
    assert curated_harness_dir(load_target_experiment(prepared)) == str(copied)
    assert (copied / "runtime.h").read_text() == "/* selected public harness */\n"
    definition = _phase1_definition(fixture, sealed)
    assert main(["run", str(definition), "--phase", "1", "--run-dir", str(fixture["root"] / "out/runs/contracts")]) == 0


@pytest.mark.parametrize("kind", ["external_file", "directory", "broken"])
def test_curated_harness_link_must_remain_inside_declared_tree(release_fixture, capsys, kind):
    fixture = release_fixture
    source = fixture["root"] / "source-experiment"
    harness = source / "contracts/harness"
    harness.mkdir(parents=True)
    if kind == "external_file":
        (harness / "external.h").symlink_to(fixture["root"] / "merlin/contract/VERSION")
    elif kind == "directory":
        (harness / "headers").mkdir()
        (harness / "headers_alias").symlink_to("headers", target_is_directory=True)
    else:
        (harness / "missing.h").symlink_to("missing-target.h")
    descriptor = source / "target_experiment.yaml"
    document = yaml.safe_load(descriptor.read_text())
    document["hardware_spec"] = {"curated_harness": "contracts/harness"}
    descriptor.write_text(yaml.safe_dump(document))

    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    assert (
        main(["corpus", "prepare", "--historical-overlay", str(fixture["run"]), "--output", str(fixture["release"])])
        == 2
    )
    expected = "outside the declared harness" if kind == "external_file" else "directory or broken symlink"
    assert expected in (fixture["release"] / "private/failure.json").read_text()


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
    assert (
        main(["corpus", "prepare", "--historical-overlay", str(fixture["run"]), "--output", str(fixture["release"])])
        != 0
    )
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


def test_seal_records_member_digests_and_immutability_and_names_drift(release_fixture, capsys):
    """Verification says WHICH sealed member changed or vanished; host-private names stay withheld."""
    report = _prepare(release_fixture, capsys)
    sealed = _seal(release_fixture, report, capsys)
    seal_path = Path(sealed["seal"])
    document = json.loads(seal_path.read_text())
    assert "corpus/isa/generated_member/README.md" in document["members"]
    assert document["immutability"]["immutable"]["mode"] == "off"  # the suite's default; never a silent claim
    assert document["immutability"]["immutable"]["granted"] == 0
    assert sealed["immutability"]["records"]["read_only"]["files"] == 2
    assert not os.access(seal_path, os.W_OK) or os.geteuid() == 0
    public = release_fixture["release"] / "payload/corpus/isa/generated_member/README.md"
    hidden = release_fixture["release"] / "payload/corpus/hidden/private_member_identity/golden.yaml"
    for member in (public, hidden):
        member.chmod(0o600)  # deliberate operator tampering, not a normal writable release
        member.write_text("changed after review")
    with pytest.raises(SpecError) as refused:
        corpus_release.verify(seal_path, Path(sealed["descriptor"]))
    message = str(refused.value)
    assert "tampered 2" in message and "corpus/isa/generated_member/README.md" in message
    assert "private_member_identity" not in message and "a host-private member" in message


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
    assert (
        main(
            [
                "corpus",
                "prepare",
                "--historical-overlay",
                str(release_fixture["run"]),
                "--output",
                str(release_fixture["release"]),
            ]
        )
        == 2
    )
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
    prepare_entry = next(node for node in session.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_prepare"
        for node in ast.walk(prepare_entry)
    )
    admission = next(node for node in session.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare")
    calls = [node for node in ast.walk(admission) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    verification = [node.lineno for node in calls if node.func.id == "verify_snapshot_for_phase1"]
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
    assert (
        main(["corpus", "prepare", "--historical-overlay", str(fixture["run"]), "--output", str(fixture["release"])])
        == 2
    )
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
    assert (
        main(["corpus", "prepare", "--historical-overlay", str(fixture["run"]), "--output", str(fixture["release"])])
        == 2
    )
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
        Path(sealed["seal"]).chmod(0o600)  # the seal record is read-only; tampering must lift that first
        Path(sealed["seal"]).write_text(json.dumps(document))
        Path(sealed["seal"]).chmod(0o400)
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


def test_reviewed_private_instruction_model_survives_freeze_without_permission_widening(release_fixture, capsys):
    from merlin.targetgen.sandbox import bwrap as BW

    fixture = release_fixture
    _prepare(fixture, capsys)
    private = fixture["release"] / "private"
    model = private / "instruction-semantics.json"
    model.write_text('{"schema":"synthetic-test-only","instructions":[]}\n')
    model.chmod(0o400)
    preparation = private / "preparation.json"
    document = json.loads(preparation.read_text())
    document["instruction_semantics"] = {"sha256": fingerprint(model)}
    preparation.write_text(json.dumps(document))
    sealed = _seal(fixture, corpus_release.inspect_release(fixture["release"]), capsys)
    descriptor = Path(sealed["descriptor"])
    bundle = yaml.safe_load(
        (descriptor.parent / "input_bundles/raw_baseline_public_v0/input_bundle_manifest.yaml").read_text()
    )
    ws = fixture["root"] / "instruction-model-snapshot"
    ws.mkdir()
    BW.materialize_bundle_inputs(ws, bundle, repo=fixture["root"])
    try:
        verified = corpus_release.verify_snapshot_for_phase1(
            Path(sealed["seal"]), descriptor, ws, bundle, repo=fixture["root"]
        )
        frozen = verified.private_instruction_model
        assert frozen is not None and frozen.read_bytes() == model.read_bytes()
        assert frozen.stat().st_mode & 0o7777 == 0o400
        assert frozen.stat().st_nlink == 1 and not frozen.samefile(model)
        assert model.stat().st_mode & 0o7777 == 0o400
        # Correct bytes cannot authorize broader access to the private model.
        frozen.chmod(0o444)
        with pytest.raises(SpecError, match="frozen private instruction model"):
            corpus_release.verify_snapshot_for_phase1(
                Path(sealed["seal"]), descriptor, ws, bundle, repo=fixture["root"]
            )
    finally:
        BW.remove_bundle_snapshot(ws)


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
    argv = BW.full_argv(
        SimpleNamespace(target="fixture-device", capsule_corpus=None, corpus_siblings=lambda: []), ws, bundle
    )
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


def test_default_release_root_carries_the_target_axis(tmp_path, monkeypatch):
    import pytest
    from merlin_experiments.corpus.release import default_release_root
    from merlin_experiments.spec import SpecError

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    root = default_release_root("alpha", timestamp="20260929T000000Z", sha="abc1234")
    assert root == tmp_path / "out/artifacts/protocols/alpha/phase0-20260929T000000Z-abc1234"
    with pytest.raises(SpecError):
        default_release_root("../escape")


def test_a_derived_resource_policy_names_only_the_generated_admitted_models(release_fixture, capsys, monkeypatch):
    """The model policy is computed from the staged corpus, so it cannot name a model the corpus lacks."""
    from merlin.targetgen import eligibility

    fixture = release_fixture
    generated_member = fixture["baseline"] / "isa/generated_member/capsule.yaml"
    model = yaml.safe_load(generated_member.read_text())
    model["kind"] = "model"
    generated_member.write_text(yaml.safe_dump(model))
    descriptor = fixture["root"] / "source-experiment/target_experiment.yaml"
    authored = yaml.safe_load(descriptor.read_text())
    authored["grading"] = {
        "release_admission": "derive_from_corpus_v1",
        "resource_bound": {"policy": "fixture_review", "derive": "phase0_qualified_models_v1"},
    }
    descriptor.write_text(yaml.safe_dump(authored))
    monkeypatch.setattr(eligibility, "capability_map_for_target", lambda _target: {"fixture": object()})

    report = _prepare(fixture, capsys)
    promoted = yaml.safe_load(Path(report["descriptor"]).read_text())
    bound = promoted["grading"]["resource_bound"]
    assert bound == {"policy": "fixture_review", "required_admitted_models": ["generated_member"]}
    corpus_models = {
        yaml.safe_load(path.read_text())["name"]
        for path in Path(promoted["capsule_corpus"]).parent.rglob("capsule.yaml")
        if yaml.safe_load(path.read_text()).get("kind") == "model"
    }
    assert set(bound["required_admitted_models"]) <= corpus_models


def test_a_derived_policy_refuses_hand_maintained_names_and_an_unqualified_integer_model(tmp_path):
    from merlin_experiments.corpus import preparation
    from merlin_experiments.spec import SpecError

    corpus = tmp_path / "corpus"
    for category, name, extra in (
        ("model", "SY_qualified", {"model_qualification": {"status": "qualified"}}),
        ("model", "SY_unqualified", {"integer_partial_sum_bound": {"status": "unknown"}}),
        ("model", "M_retained", {}),
    ):
        directory = corpus / category / name
        directory.mkdir(parents=True)
        (directory / "capsule.yaml").write_text(
            yaml.safe_dump({"name": name, "kind": "model", "label": "public", **extra})
        )
    (corpus / "MANIFEST.yaml").write_text(yaml.safe_dump({"generated": ["model/SY_qualified", "model/SY_unqualified"]}))
    import merlin.targetgen.capsule_runner as capsule_runner

    def discover(root, labels=None):
        out = []
        for path in sorted(Path(root).rglob("capsule.yaml")):
            cap = yaml.safe_load(path.read_text())
            cap["__dir__"] = str(path.parent)
            out.append(cap)
        return out

    original = capsule_runner.discover_capsules
    capsule_runner.discover_capsules = discover
    try:
        derived = preparation.derive_resource_bound(corpus)
    finally:
        capsule_runner.discover_capsules = original
    assert derived["required_admitted_models"] == ["SY_qualified"]
    assert derived["exclude_capsules"] == ["M_retained", "SY_unqualified"]
    with pytest.raises(SpecError):
        empty = tmp_path / "empty"
        (empty / "model").mkdir(parents=True)
        (empty / "MANIFEST.yaml").write_text("generated: []\n")
        preparation.derive_resource_bound(empty)


def test_the_gemmini_descriptor_derives_its_model_policy():
    from merlin.common.paths import repo_root

    # Installed qualification copies this committed input beside the tests. A
    # source checkout uses the same authored file directly.
    archived = Path(__file__).with_name("source-inputs") / "examples/gemmini/target/descriptor.yaml"
    selected = archived if archived.is_file() else repo_root() / "examples/gemmini/target/descriptor.yaml"
    descriptor = yaml.safe_load(selected.read_text())
    bound = descriptor["grading"]["resource_bound"]
    assert bound["derive"] == "phase0_qualified_models_v1"
    assert "required_admitted_models" not in bound and "exclude_capsules" not in bound


def test_explicit_public_reference_release_grants(release_fixture, capsys):
    """References enter original bundle generation before admission and sealing."""
    import hashlib

    from merlin_experiments.corpus.release import prepare

    from merlin.targetgen.target_experiment import load_target_experiment

    fixture = release_fixture
    assert main(["run", str(fixture["definition"]), "--phase", "0", "--run-dir", str(fixture["run"])]) == 0
    capsys.readouterr()
    refs = fixture["root"] / "selected-references"
    refs.mkdir()
    leaf = refs / "example.c"
    leaf.write_text("int reference_example(void) { return 0; }\n")
    manifest = refs / "reference-manifest.json"
    manifest.write_text(json.dumps({"schema": "operator.allowed_reference_sources.v2", "review_id": "release-test",
        "origins": [{"name": "public_example", "kind": "public_repository", "container": ".",
            "identity": "1" * 40, "public_url": "https://example.org/compiler", "license": "LICENSE",
            "allowed_purpose": "reference_reading", "files": [{"member": "example.c", "destination": "example.c",
                "sha256": hashlib.sha256(leaf.read_bytes()).hexdigest()}]}]}))
    report = prepare(fixture["run"], fixture["release"], generated_only=False,
                     selected_variants=("public_v0",), public_reference_manifest=manifest)
    descriptor = load_target_experiment(report["descriptor"])
    grant = descriptor.information_set("public_v0")["allowed"][0]
    assert grant["as"] == "reference-kernels" and grant["mode"] == "ro"
    for path in (fixture["release"] / "payload/experiment/input_bundles").glob("*/input_bundle_manifest.yaml"):
        bundle = yaml.safe_load(path.read_text())
        assert grant in bundle["allowed"]
    assert (fixture["release"] / "payload/experiment/reference-kernels/public_example/example.c").read_bytes() == leaf.read_bytes()
    assert not (fixture["release"] / "private/seal.json").exists()
