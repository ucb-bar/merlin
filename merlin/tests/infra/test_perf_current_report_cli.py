"""Cold current-report replay of genuinely produced synthetic candidate evidence.

Only provider, simulator and whitelist-only bwrap transports are substituted.
No candidate, snapshot, certificate, admission, receipt or report verifier is mocked.
The hardware bytes and functional scores are synthetic; neither physical performance
nor OS isolation is qualified. Output, contract and functional input roots and the
descriptor target use the installed paired CLI's explicit interface.
The parent document is fixture-assembled and sealed by the real writer; this exercises
adoption/replay, not the coordinator's complete launch/checkpoint/resume lifecycle.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import candidate_verification as VERIFY
from merlin_experiments.phase2 import corpus as CORPUS
from merlin_experiments.phase2 import measurement_evidence as ME
from merlin_experiments.phase2 import paired_inputs as PI
from merlin_experiments.phase2 import paired_measurement as PME
from test_perf_candidate_report_integration import (
    PAS,
    _certificate,
    _sha,
    _simulator_execution,
    _yaml,
    admitted_inputs,  # noqa: F401 -- imported pytest fixture dependency
    qualified_stage,  # noqa: F401 -- imported real producer fixture
)

from merlin.common.paths import merlin_dir, python_import_roots


def _heldout(pair, root, target):
    """Independent synthetic reveal; real holdout parser and exact-tree admission."""
    directory = root / "heldout"
    rows = []
    for index, member in enumerate(CORPUS.discover_performance_corpus(target).capsules):
        descriptor = copy.deepcopy(member.descriptor)
        k = (24, 48, 96, 192)[index]
        name = f"PK_holdout_k{k}"
        descriptor["name"] = name
        descriptor["inputs"][0]["shape"][1] = k
        descriptor["inputs"][1]["shape"][0] = k
        relative = f"_perf/{name}"
        _yaml(directory / relative / "capsule.yaml", descriptor)
        rows.append(dict(name=name, path=relative, family="PK", cohort="PK_predictor", M=16, N=16, K=k))
    tree = PI.holdout_tree_record(directory)
    manifest = directory / "holdout_manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema_version=2,
                kind="generated_performance_holdout_reveal",
                domain={"target": target.target},
                cohorts={"PK_predictor": {"family": "PK", "member_count": len(rows)}},
                members=rows,
                corpus=tree,
            )
        )
    )
    for path in reversed([directory, *directory.rglob("*")]):
        path.chmod(0o555 if path.is_dir() else 0o444)
    return PI.load_holdout_corpus(
        directory,
        manifest,
        manifest_sha256=_sha(manifest),
        capsules_sha256=tree["sha256"],
        expected_target=target.target,
    )


def _engine(spec, *_args, **_kwargs):
    result = _simulator_execution(
        certificate=spec.gsim_certificate,
        decision=spec.gsim_decision,
        member=spec.member,
        arm=spec.arm,
    )
    k = spec.member.descriptor["inputs"][0]["shape"][1]
    assert result["measurement"]["gsim_qualification"]["admitted"]
    result["measurement"]["per_sim"]["gsim"]["provenance"] = dict(
        tier="L3",
        simulator="gsim",
        oracle_kind="rtl_gsim",
        derived_from_rtl=True,
        cycle_accurate=True,
        elf_sha256=hashlib.sha256(f"synthetic-elf-k{k}".encode()).hexdigest(),
        reused_measurement=False,
    )
    return {"execution": spec.as_dict(), **result}


@pytest.fixture
def report_evidence(qualified_stage, monkeypatch):  # noqa: F811 -- imported pytest fixture
    fixture = qualified_stage
    root, target = fixture["root"], fixture["target"]
    scripts = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    pair = importlib.import_module("merlin_experiments.phase2.paired_cli")
    coordinator = importlib.import_module("merlin_experiments.phase2.checkpoint_controller")
    statistics = importlib.import_module("merlin_experiments.phase2.statistics")
    measurements = root / "measurements"

    execute = PME.execute_schedule
    monkeypatch.setattr(PME, "execute_schedule", lambda *a, **kw: execute(*a, **kw, executor=_engine))
    heldout = _heldout(pair, root, target)
    qualification = root / "heldout-qualification"
    qualification.mkdir()
    heldout_certificate = _certificate(qualification, target, members=heldout.capsules)
    trials = [f"trial_{index:02d}" for index in range(3)]
    stage_roots = {trial: root / ("stage" if index == 0 else f"stage-{trial}") for index, trial in enumerate(trials)}
    declared_capsules = [
        dict(family=f"{phase}:{member.family}", capsule=member.capsule)
        for phase, members in (
            ("tuning", CORPUS.discover_performance_corpus(target).capsules),
            ("held_out", heldout.capsules),
        )
        for member in members
    ]
    # Declare the complete formal matrix before the first paired execution.
    declaration = statistics.predeclare(
        trials=[dict(trial=trial, agent_run_id=stage_roots[trial].name) for trial in trials],
        capsules=declared_capsules,
        replicates=ME.REPLICATES,
    )
    parent_root = root / "parent"
    parent_root.mkdir()
    stats_path = parent_root / "statistics_predeclaration.json"
    stats_path.write_text(statistics.canonical_json(declaration))
    stats_path.chmod(0o444)
    records, handoffs, receipts, children = {}, {}, [], []
    for index, trial in enumerate(trials):
        record = (
            fixture["record"]
            if index == 0
            else PAS.run_stage(**{**fixture["stage_kwargs"], "stage_root": stage_roots[trial]})
        )
        records[trial] = record
        handoff = VERIFY.verify_candidate_handoff(record, verify_authoring_tools=True, target_experiment=target)
        handoffs[trial] = handoff
        tuning = CORPUS.load_frozen_performance_corpus(
            handoff.corpus_root,
            manifest_sha256=handoff.corpus_manifest_sha256,
            capsules_sha256=handoff.corpus_sha256,
            expected_target=target.target,
        )
        for phase, corpus, certificate in (
            ("tuning", tuning, fixture["certificate"]),
            ("held_out", heldout, heldout_certificate),
        ):
            run_id = f"{trial}__{phase}"
            arguments = {
                "functional-run-id": fixture["run"].name,
                "functional-submission-sha256": fixture["digest"],
                "functional-runs-root": fixture["runs"],
                "source-root": fixture["stage_kwargs"]["source_root"],
                "measurement-root": measurements,
                "contract-root": fixture["stage_kwargs"]["contract_root"],
                "descriptor": target.path,
                "candidate-record": record,
                "corpus-root": corpus.root,
                "corpus-manifest": corpus.manifest_path,
                "corpus-manifest-sha256": corpus.manifest_sha256,
                "corpus-capsules-sha256": corpus.capsules_sha256,
                "phase": phase,
                "gsim-certificate": certificate,
                "gsim-certificate-sha256": _sha(certificate),
                "rtl-facts": fixture["stage_kwargs"]["rtl_facts"],
                "run-id": run_id,
            }
            assert pair.main([part for key, value in arguments.items() for part in (f"--{key}", str(value))]) == 0
            path = measurements / run_id / "campaign_manifest.json"
            binding = ME.MeasurementBinding(
                phase,
                fixture["run"].name,
                fixture["digest"],
                handoff.record_sha256,
                handoff.candidate_sha256,
                corpus.manifest_sha256,
                corpus.capsules_sha256,
                _sha(certificate),
            )
            verified = ME.verify_paired_measurement(path, expected=binding)
            adopted = coordinator._verify_measurement_manifest(
                path,
                phase=phase,
                functional_run_id=fixture["run"].name,
                functional_submission_sha256=fixture["digest"],
                handoff=handoff,
                corpus_manifest_sha256=corpus.manifest_sha256,
                corpus_capsules_sha256=corpus.capsules_sha256,
                certificate_sha256=_sha(certificate),
            )
            assert adopted == verified.receipt()
            receipts.append({"trial": trial, **adopted})
            children.append(verified)
    trial_evidence = [
        dict(trial=trial, agent_run_id=record.parent.name, agent_evidence_sha256=handoffs[trial].record_sha256)
        for trial, record in records.items()
    ]
    rows = [
        row
        for receipt, child in zip(receipts, children, strict=True)
        for row in ME.statistics_rows(child.rows, trial=receipt["trial"])
    ]
    result = statistics.evaluate(declaration, rows, trial_evidence=trial_evidence)
    assert result["status"] == "admitted"
    parent = dict(
        schema="merlin.agentic-performance-experiment.v1",
        status="GO",
        selection="all_three_trials_all_predeclared_cells_no_best_of_no_drop",
        declaration=dict(
            experiment_id="synthetic-real-candidate-fixture",
            trials=trials,
            measurement_phases=list(ME.PHASES),
            trial_contracts={trial: handoff.agent_contract for trial, handoff in handoffs.items()},
            gsim_certificate_sha256=_sha(fixture["certificate"]),
        ),
        trials=trial_evidence,
        measurement_manifests=receipts,
        holdout=dict(manifest_sha256=heldout.manifest_sha256, capsules_sha256=heldout.capsules_sha256),
        heldout_gsim_certificate={"sha256": _sha(heldout_certificate)},
        statistics_predeclaration_sha256=_sha(stats_path),
        statistics=result,
    )
    path = coordinator._seal_final(parent_root, parent)
    return SimpleNamespace(root=root, path=path, sha=_sha(path), records=records, children=children, scripts=scripts)


def _report_command(evidence, output):
    command = [
        sys.executable,
        str(evidence.scripts / "gen_perf_report.py"),
        "--experiment-manifest",
        str(evidence.path),
        "--manifest-sha256",
        evidence.sha,
        "--output",
        str(output),
    ]
    for trial, record in evidence.records.items():
        command += ["--candidate-record", f"{trial}={record}"]
    return command


def _cold_report(evidence, output):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("MERLIN_")}
    environment["PYTHONPATH"] = os.pathsep.join(map(str, python_import_roots()))
    # _pbcommon still obtains historical bench geometry at import time. Keep
    # that explicit synthetic input, not ambient checkout/output/model overrides.
    environment["MERLIN_RTL_FACTS"] = str(evidence.root / "rtl_facts.json")
    return subprocess.run(
        _report_command(evidence, output),
        cwd=evidence.root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cold_cli_admits_actual_candidate_and_paired_evidence(report_evidence):
    output = report_evidence.root / "report.md"
    result = _cold_report(report_evidence, output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "96 exact GSIM timing cells" in result.stdout
    report = output.read_text()
    assert "GSIM supplies RTL timing" in report
    assert "PK CLAIM ESTABLISHED" not in report


def test_cold_cli_refuses_each_corrupted_evidence_layer(report_evidence):
    evidence = report_evidence
    record = evidence.records["trial_00"]
    document = json.loads(record.read_bytes())
    child = evidence.children[0]
    layers = {
        "parent": (evidence.path, "parent experiment manifest"),
        "candidate_record": (record, "parent-pinned candidate record"),
        "candidate_bytes": (Path(document["candidate"]["path"]) / "tool.py", "sealed performance candidate bytes"),
        "transcript": (Path(document["agent"]["transcript"]), "transcript bytes"),
        "broker_receipt": (Path(document["broker"]["receipt_manifest"]), "receipt"),
        "feedback": (
            Path(document["broker"]["round_receipts"][0]["feedback_receipts"][0]["path"]),
            "feedback receipt bytes",
        ),
        "telemetry": (Path(document["telemetry"]["artifacts"]["preflight"]["path"]), "telemetry changed"),
        "certificate": (Path(document["development_feedback"]["certificate"]["path"]), "certificate bytes changed"),
        "measurement": (Path(child.manifest["raw_results"]["paired_cells"]), "result"),
    }
    for name, (path, error) in layers.items():
        original, mode = path.read_bytes(), path.stat().st_mode & 0o777
        try:
            path.chmod(0o644)
            path.write_bytes(original + b"\n ")
            path.chmod(0o444)
            output = evidence.root / f"refused-{name}.md"
            result = _cold_report(evidence, output)
            assert result.returncode != 0, name
            assert error in result.stderr, (name, result.stdout, result.stderr)
            assert not output.exists(), name
        finally:
            path.chmod(0o644)
            path.write_bytes(original)
            path.chmod(mode)
    # Each negative control restored the same original evidence, not a new seal.
    result = _cold_report(evidence, evidence.root / "restored.md")
    assert result.returncode == 0, result.stdout + result.stderr
